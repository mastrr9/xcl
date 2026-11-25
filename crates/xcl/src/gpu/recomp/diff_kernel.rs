//! `XCL_DXIL_DIFF=1`: differential kernel runs (the testing strategy of
//! docs/specs/2026-11-24-direct-dxil.md). `runner::run_code`, `run_code_with` and
//! `run_program` translate the program through both backends (HLSL + DXC and the direct DXIL
//! translator), run both kernels with the same inputs and compare everything a kernel test can
//! observe: SGPRs, SCC, the VGPRs of every lane, exports, flat memory and the UAV test textures
//! (raw). A difference is an error that names the field, the first `SHOWN` differing slots
//! (HLSL value / direct value) and the count; the run otherwise returns the HLSL result.
//! A program the direct translator refuses runs with the HLSL result alone (`refusals` counts
//! them, each distinct reason is logged once), so the whole suite runs in this mode.
//! Tests that compile the HLSL kernel themselves (`kernel_wave`, `direct_or` in
//! `recomp/tests.rs`) `pair` its container with the direct translation; every later
//! `runner::run*` (or, for a compute-mode program, `runner::dispatch`: flat memory only) of
//! that container on the same thread then runs both (a test that runs one kernel from many
//! states is compared for each).
//!
//! Each backend stores only the registers it uses and `res` starts zeroed, so a register that
//! one backend leaves unstored (0) while the other stores its unchanged input value is not a
//! difference: an SGPR on its own, a VGPR only when that holds in every lane that ran (one
//! side 0 in all of them, the other the input in all of them).
//!
//! The lane order of one atomic across a wave is unspecified, on the console and on the host:
//! the destinations of the program's returning atomics (`atomic_returns`) are compared as a
//! multiset of per-lane values when they differ lane by lane. A test whose values depend on
//! that order in another way (many lanes' cmpswap or umax on one address) names them with
//! `relax` and `Unordered`; its own check must then accept the direct run.
//!
//! A test may also compare the NaNs of some float VGPRs by class (`Nan`) instead of bit for
//! bit; nothing is relaxed unless a test asks.

use std::cell::RefCell;
use std::collections::BTreeSet;
use std::io::Write;
use std::ops::Range;
use std::sync::Mutex;

use super::diff_table::{self, Outcome};
use super::pso::StaticSampler;
use super::runner::{self, KernelResult, KernelState, TestSampler, TestTexture};
use crate::gpu::dxil::kernel::Kernel;
use super::{KERNEL_EXPORTS, KERNEL_LANES, KERNEL_SCC_SLOT, KERNEL_SPLIT_SGPRS, KERNEL_VGPRS};

/// Differing slots listed per field.
pub const SHOWN: usize = 8;

/// Programs the direct translator refused, and the distinct reasons already logged.
static REFUSED: Mutex<(usize, BTreeSet<String>)> = Mutex::new((0, BTreeSet::new()));

thread_local! {
    /// An HLSL kernel container `pair`ed with the lanes it runs, the program (for the
    /// `diff_table` line) and the direct translation of the same program.
    static PAIRED: RefCell<Option<(Vec<u8>, usize, Vec<u8>, Result<Kernel, String>)>> = const { RefCell::new(None) };
    /// The running test's `relax`ation of the comparison.
    static RELAX: RefCell<Relax> = RefCell::new(Relax::default());
}

/// How a test relaxes the comparison of its runs (`relax`); the default compares every slot
/// bit for bit.
#[derive(Default)]
pub struct Relax {
    /// Values that depend on the order in which the lanes of an atomic ran.
    pub unordered: Option<Unordered>,
    /// Float VGPRs whose NaNs compare by class.
    pub nan: Vec<Nan>,
}

/// VGPRs (every lane that ran) holding f32 values, or f64 values in register pairs from
/// `vgprs.start` (low dword first), whose NaNs compare by `eq`: a NaN on both sides with the
/// same class is equal whatever its sign and payload. Every other value in them stays bit for
/// bit, so a NaN against a number or an infinity is still a difference.
pub struct Nan {
    pub vgprs: Range<usize>,
    pub f64: bool,
    pub eq: NanEq,
}

/// When two NaNs are equal.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum NanEq {
    /// Both quiet or both signalling. Quiet: the top mantissa bit (f32 bit 22, f64 bit 51) set.
    Class,
    /// Any NaN equals any NaN.
    Any,
}

/// The class of the f32 (`f64`: false) or f64 value `bits`: None for a number or an infinity,
/// Some(quiet) for a NaN.
fn nan_class(bits: u64, f64: bool) -> Option<bool> {
    let (exp, mant, quiet) = if f64 { (bits >> 52 & 0x7FF, bits & ((1 << 52) - 1), 1 << 51) } else { (bits >> 23 & 0xFF, bits & 0x7F_FFFF, 1 << 22) };
    (exp == if f64 { 0x7FF } else { 0xFF } && mant != 0).then_some(mant & quiet != 0)
}

impl NanEq {
    /// Two values `eq` accepts although their bits differ: two NaNs of a class it equates.
    fn equal(self, a: u64, b: u64, f64: bool) -> bool {
        match (nan_class(a, f64), nan_class(b, f64)) {
            (Some(x), Some(y)) => self == NanEq::Any || x == y,
            _ => false,
        }
    }
}

/// VGPRs (every lane) and flat-memory dwords whose values depend on the order in which the
/// lanes of an atomic ran, such as many lanes' cmpswap or umax on one address: they are not
/// compared bit for bit, `check` (the test's own checks of them, which panic as its asserts
/// do) must accept the direct run instead (the test checks the HLSL result it gets back).
/// Kernel runs only: the memory `runner::dispatch` leaves is compared strictly.
pub struct Unordered {
    pub vgprs: Vec<Range<usize>>,
    pub mem: Vec<Range<usize>>,
    pub check: Box<dyn Fn(&KernelResult)>,
}

/// The comparison of this thread's runs is relaxed by `r` until the returned guard drops.
pub fn relax(r: Relax) -> Relaxed {
    RELAX.with(|t| *t.borrow_mut() = r);
    Relaxed(())
}

/// Ends a `relax`ation when dropped.
pub struct Relaxed(());

impl Drop for Relaxed {
    fn drop(&mut self) {
        RELAX.with(|t| *t.borrow_mut() = Relax::default());
    }
}

/// The VGPR ranges the returning atomics of `code` write (DS `_RTN`, GLC FLAT / GLOBAL /
/// MUBUF / MIMG atomics) and no other instruction of it writes; empty when `code` does not
/// decode or writes a VGPR M0 picks.
pub fn atomic_returns(code: &[u8]) -> Vec<Range<usize>> {
    use crate::gpu::rdna2;
    let (mut atomic, mut other) = (Vec::new(), Vec::new());
    let mut pc = 0;
    while pc < code.len() {
        let Ok(d) = rdna2::decode(&code[pc..], pc as u64) else { return Vec::new() };
        pc += d.len;
        let f = |n: &str| rdna2::field_value(d.enc, &d.words, n).unwrap_or(0) as usize;
        let name = d.inst.name;
        // v_movreld / v_movrelsd write a VGPR chosen by M0 at run time
        if name.starts_with("V_MOVRELD") || name.starts_with("V_MOVRELSD") {
            return Vec::new();
        }
        let ds = name.starts_with("DS_") && name.contains("_RTN");
        if ds || name.contains("_ATOMIC_") {
            // a FLAT / MUBUF / MIMG atomic without GLC writes no VGPR
            if !ds && f("GLC") == 0 {
                continue;
            }
            let at = if name.starts_with("BUFFER_") || name.starts_with("IMAGE_") { f("VDATA") } else { f("VDST") };
            let n = if name.starts_with("IMAGE_") {
                // DMASK: the data dwords, of which a cmpswap returns the first half
                let n = f("DMASK").count_ones() as usize;
                if name.contains("CMPSWAP") { n / 2 } else { n }
            } else if name.starts_with("BUFFER_") {
                1 + name.ends_with("_X2") as usize
            } else {
                d.ienc.operands.iter().find(|o| o.field == Some("VDST")).map_or(1, |o| (o.size as usize).div_ceil(32))
            };
            atomic.push(at..at + n.max(1));
            continue;
        }
        for o in d.ienc.operands.iter().filter(|o| o.output && !o.implicit && o.ty.starts_with("OPR_VGPR")) {
            if let Some(v) = o.field.and_then(|n| rdna2::field_value(d.enc, &d.words, n)) {
                other.push(v as usize..v as usize + (o.size as usize).div_ceil(32).max(1));
            }
        }
    }
    atomic.retain(|a| !other.iter().any(|o| o.start < a.end && a.start < o.end));
    atomic
}

/// `XCL_DXIL_DIFF=1`: kernel runs go through both backends and must agree.
pub fn enabled() -> bool {
    std::env::var("XCL_DXIL_DIFF").is_ok_and(|v| v == "1")
}

/// Programs the direct translator refused so far (their runs used the HLSL result alone).
pub fn refusals() -> usize {
    REFUSED.lock().unwrap_or_else(|p| p.into_inner()).0
}

fn refused(why: &str) {
    let mut r = REFUSED.lock().unwrap_or_else(|p| p.into_inner());
    r.0 += 1;
    let first = why.lines().next().unwrap_or_default().to_string();
    if r.1.insert(first.clone()) {
        let _ = writeln!(std::io::stderr(), "XCL_DXIL_DIFF: direct DXIL refused a program, HLSL result only: {first}");
    }
}

/// A failure of the direct translator itself (a panic, DXIL it built but could not finish or
/// validate), as opposed to a refusal of an unsupported construct.
fn translator_failed(why: &str) -> bool {
    use crate::gpu::dxil::coverage::{classify, Outcome};
    !matches!(classify(why), Outcome::Unsupported) || why.starts_with("direct DXIL builder error")
}

/// `runner::run_code_with` through both backends; the HLSL result when they agree.
pub fn run_code_with(code: &[u8], state: &KernelState, mem: &[u32], textures: &[TestTexture], samplers: &[TestSampler]) -> Result<KernelResult, String> {
    let t = super::translate(code, &runner::options(super::Mode::Kernel))?;
    if !t.complete() {
        return Err(format!("unsupported instructions: {:?}", t.unsupported));
    }
    let dxil = super::dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).map_err(|x| format!("{x}\n--- HLSL ---\n{}", t.hlsl))?;
    run_paired(&dxil, KERNEL_LANES, code, runner::direct_kernel_full(code, true), state, mem, textures, samplers, &t.static_samplers)
}

/// Under `XCL_DXIL_DIFF=1`: later runs of the HLSL kernel `hlsl` (a wave64 or wave32
/// translation of the console program `code`) on this thread also run the direct translation
/// of the same program (`direct`, called only in this mode) and compare.
pub fn pair(hlsl: &[u8], code: &[u8], wave64: bool, direct: impl FnOnce() -> Result<Kernel, String>) {
    if enabled() {
        let lanes = if wave64 { KERNEL_LANES } else { KERNEL_LANES / 2 };
        PAIRED.with(|p| *p.borrow_mut() = Some((hlsl.to_vec(), lanes, code.to_vec(), direct())));
    }
}

/// The lanes, program and direct translation `pair`ed with `dxil`, if any (taken out while
/// it runs).
fn take_paired(dxil: &[u8]) -> Option<(usize, Vec<u8>, Result<Kernel, String>)> {
    PAIRED.with(|p| {
        let mut p = p.borrow_mut();
        if p.as_ref().is_some_and(|(h, _, _, _)| h == dxil) {
            p.take().map(|(_, lanes, code, k)| (lanes, code, k))
        } else {
            None
        }
    })
}

/// `runner::run_with_statics` of a `pair`ed container: both backends, compared (the pair
/// stays for the next run). None when `dxil` is not paired.
pub fn run_if_paired(
    dxil: &[u8],
    state: &KernelState,
    mem: &[u32],
    textures: &[TestTexture],
    samplers: &[TestSampler],
    statics: &[StaticSampler],
) -> Option<Result<KernelResult, String>> {
    // taken out first, so the runs below do not come back here
    let (lanes, code, direct) = take_paired(dxil)?;
    let again = copy(&direct);
    let r = run_paired(dxil, lanes, &code, direct, state, mem, textures, samplers, statics);
    PAIRED.with(|p| *p.borrow_mut() = Some((dxil.to_vec(), lanes, code, again)));
    Some(r)
}

/// `Kernel` has no `Clone`.
fn copy(direct: &Result<Kernel, String>) -> Result<Kernel, String> {
    match direct {
        Ok(k) => Ok(Kernel { dxil: k.dxil.clone(), static_samplers: k.static_samplers.clone(), dispatch_chain: k.dispatch_chain.clone() }),
        Err(why) => Err(why.clone()),
    }
}

/// `runner::dispatch` of a `pair`ed compute-mode container: both backends, the flat memory
/// they leave compared (the pair stays for the next run). None when `dxil` is not paired.
pub fn dispatch_if_paired(dxil: &[u8], groups: [u32; 3], sgprs: &[u32], mem: &[u32]) -> Option<Result<Vec<u32>, String>> {
    let (lanes, code, direct) = take_paired(dxil)?;
    let again = copy(&direct);
    let r = runner::dispatch(dxil, groups, sgprs, mem).and_then(|h| {
        let k = match direct {
            Ok(k) if k.static_samplers.is_empty() => k,
            Ok(_) => {
                refused("needs static samplers (runner::dispatch binds none)");
                diff_table::record(&code, Outcome::Refused);
                return Ok(h);
            }
            Err(why) if !translator_failed(&why) => {
                refused(&why);
                diff_table::record(&code, Outcome::Refused);
                return Ok(h);
            }
            Err(why) => {
                diff_table::record(&code, Outcome::Failed);
                return Err(format!("XCL_DXIL_DIFF: {why}"));
            }
        };
        let d = runner::dispatch(&k.dxil, groups, sgprs, mem).map_err(|x| {
            diff_table::record(&code, Outcome::Failed);
            format!("XCL_DXIL_DIFF: the direct DXIL dispatch failed (the HLSL one ran): {x}")
        })?;
        let c = compare_mem(&h, &d);
        diff_table::record(&code, if c.is_ok() { Outcome::Matched } else { Outcome::Mismatched });
        c?;
        Ok(h)
    });
    PAIRED.with(|p| *p.borrow_mut() = Some((dxil.to_vec(), lanes, code, again)));
    Some(r)
}

/// The flat memory two dispatches left, compared; the error is the report.
pub fn compare_mem(hlsl: &[u32], direct: &[u32]) -> Result<(), String> {
    let at = |v: &[u32], i: usize| v.get(i).copied().unwrap_or(0);
    let mut fields = vec![field("mem", hlsl.len().max(direct.len()), |i| Some((at(hlsl, i), at(direct, i))), |i| format!("mem[{i}] (byte {:#x})", i * 4))];
    if hlsl.len() != direct.len() {
        fields.push(Field { name: "mem length", count: 1, shown: vec![format!("hlsl {} direct {} dwords", hlsl.len(), direct.len())] });
    }
    report(&fields)
}

/// Runs the HLSL kernel `hlsl` (`lanes` lanes) and, unless the direct translator refused,
/// `direct` with the same inputs; the HLSL result when they agree. `code` (the program) goes
/// into the `diff_table` line.
#[allow(clippy::too_many_arguments)]
fn run_paired(
    hlsl: &[u8],
    lanes: usize,
    code: &[u8],
    direct: Result<Kernel, String>,
    state: &KernelState,
    mem: &[u32],
    textures: &[TestTexture],
    samplers: &[TestSampler],
    statics: &[StaticSampler],
) -> Result<KernelResult, String> {
    let h = runner::run_with_statics(hlsl, state, mem, textures, samplers, statics)?;
    let k = match direct {
        Ok(k) => k,
        Err(why) if !translator_failed(&why) => {
            refused(&why);
            diff_table::record(code, Outcome::Refused);
            return Ok(h);
        }
        Err(why) => {
            diff_table::record(code, Outcome::Failed);
            return Err(format!("XCL_DXIL_DIFF: {why}"));
        }
    };
    let d = runner::run_with_statics(&k.dxil, state, mem, textures, samplers, &k.static_samplers).map_err(|x| {
        diff_table::record(code, Outcome::Failed);
        format!("XCL_DXIL_DIFF: the direct DXIL kernel failed (the HLSL one ran): {x}")
    })?;
    let atomic = atomic_returns(code);
    let c = RELAX.with(|r| compare_relaxed(state, lanes, &h, &d, &atomic, &r.borrow()));
    diff_table::record(code, if c.is_ok() { Outcome::Matched } else { Outcome::Mismatched });
    c?;
    Ok(h)
}

fn sgpr_name(i: usize) -> String {
    match i {
        106 => "vcc_lo".into(),
        107 => "vcc_hi".into(),
        124 => "m0".into(),
        126 => "exec_lo".into(),
        127 => "exec_hi".into(),
        _ => format!("s{i}"),
    }
}

/// One compared field: the number of differing slots and the first `SHOWN` of them.
struct Field {
    name: &'static str,
    count: usize,
    shown: Vec<String>,
}

/// Slots `0..n` where `pair` gives different values (`None`: the slot is not compared).
fn field(name: &'static str, n: usize, pair: impl Fn(usize) -> Option<(u32, u32)>, label: impl Fn(usize) -> String) -> Field {
    let mut f = Field { name, count: 0, shown: Vec::new() };
    for i in 0..n {
        if let Some((h, d)) = pair(i).filter(|(h, d)| h != d) {
            f.count += 1;
            if f.shown.len() < SHOWN {
                f.shown.push(format!("{}: hlsl {h:#010x} direct {d:#010x}", label(i)));
            }
        }
    }
    f
}

/// A register both backends agree on: equal, or one left it unstored (0) and the other
/// stored its unchanged input value.
fn reg(h: u32, d: u32, input: u32) -> Option<(u32, u32)> {
    if h == d || (h == 0 && d == input) || (d == 0 && h == input) {
        None
    } else {
        Some((h, d))
    }
}

fn dword(b: &[u8], i: usize) -> u32 {
    let mut w = [0u8; 4];
    for (k, x) in w.iter_mut().enumerate() {
        *x = b.get(i * 4 + k).copied().unwrap_or(0);
    }
    u32::from_le_bytes(w)
}

/// `compare` with the destinations of returning atomics `atomic` (`atomic_returns`) compared
/// as multisets of per-lane values and the test's `relax`ation `r`: the slots they accept take
/// the HLSL value in a copy of `direct`, which is compared strictly.
pub fn compare_relaxed(state: &KernelState, lanes: usize, hlsl: &KernelResult, direct: &KernelResult, atomic: &[Range<usize>], r: &Relax) -> Result<(), String> {
    let mut d = KernelResult {
        sgpr: direct.sgpr.clone(),
        vgpr: direct.vgpr.clone(),
        scc: direct.scc,
        raw: direct.raw.clone(),
        mem: direct.mem.clone(),
        textures: direct.textures.clone(),
    };
    let lanes = lanes.min(hlsl.vgpr.len()).min(d.vgpr.len());
    let regs = |x: &KernelResult, l: usize, a: &Range<usize>| a.clone().map(|v| x.vgpr[l].get(v).copied().unwrap_or(0)).collect::<Vec<u32>>();
    let take = |d: &mut KernelResult, a: &Range<usize>| {
        for l in 0..lanes {
            let n = d.vgpr[l].len();
            for v in a.clone().filter(|&v| v < n) {
                d.vgpr[l][v] = hlsl.vgpr[l].get(v).copied().unwrap_or(0);
            }
        }
    };
    for a in atomic {
        let (mut h, mut x): (Vec<_>, Vec<_>) = (0..lanes).map(|l| (regs(hlsl, l, a), regs(&d, l, a))).unzip();
        h.sort_unstable();
        x.sort_unstable();
        if h == x {
            take(&mut d, a);
        }
    }
    for n in &r.nan {
        let step = if n.f64 { 2 } else { 1 };
        for l in 0..lanes {
            let (h, x) = (&hlsl.vgpr[l], &mut d.vgpr[l]);
            let end = n.vgprs.end.min(h.len()).min(x.len());
            for v in n.vgprs.clone().step_by(step).filter(|&v| v + step <= end) {
                let get = |r: &[u32]| if n.f64 { r[v] as u64 | (r[v + 1] as u64) << 32 } else { r[v] as u64 };
                if n.eq.equal(get(h), get(x), n.f64) {
                    x[v..v + step].copy_from_slice(&h[v..v + step]);
                }
            }
        }
    }
    if let Some(u) = &r.unordered {
        let msg = |p: Box<dyn std::any::Any + Send>| p.downcast_ref::<String>().cloned().or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_default();
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| (u.check)(direct)))
            .map_err(|p| format!("XCL_DXIL_DIFF: the direct DXIL run fails the test's check of its unordered values: {}", msg(p)))?;
        for a in &u.vgprs {
            take(&mut d, a);
        }
        d.mem = relaxed_mem(&hlsl.mem, &d.mem, &u.mem);
    }
    compare(state, lanes, hlsl, &d)
}

/// `direct` flat memory with the dwords of `unordered` taken from `hlsl`.
fn relaxed_mem(hlsl: &[u32], direct: &[u32], unordered: &[Range<usize>]) -> Vec<u32> {
    let mut d = direct.to_vec();
    for i in unordered.iter().flat_map(|a| a.clone()).filter(|&i| i < direct.len()) {
        d[i] = hlsl.get(i).copied().unwrap_or(0);
    }
    d
}

/// Compares the observable state of an HLSL and a direct DXIL run of `lanes` lanes from
/// `state`; the error is the report.
pub fn compare(state: &KernelState, lanes: usize, hlsl: &KernelResult, direct: &KernelResult) -> Result<(), String> {
    let at = |v: &[u32], i: usize| v.get(i).copied().unwrap_or(0);
    let vin = |lane: usize, r: usize| state.vgpr.get(lane).and_then(|v| v.get(r)).copied().unwrap_or(0);
    let vg = |x: &KernelResult, l: usize, r: usize| x.vgpr.get(l).and_then(|v| v.get(r)).copied().unwrap_or(0);
    // VGPRs one side left unstored in every lane that ran while the other kept the input
    let unstored: Vec<bool> = (0..KERNEL_VGPRS)
        .map(|r| {
            let all = |a: &KernelResult, b: &KernelResult| (0..lanes).all(|l| vg(a, l, r) == 0 && vg(b, l, r) == vin(l, r));
            all(hlsl, direct) || all(direct, hlsl)
        })
        .collect();
    let mut fields = vec![
        field(
            "sgpr",
            hlsl.sgpr.len().max(direct.sgpr.len()),
            |i| if i == KERNEL_SCC_SLOT { None } else { reg(at(&hlsl.sgpr, i), at(&direct.sgpr, i), at(&state.sgpr, i)) },
            sgpr_name,
        ),
        field("scc", 1, |_| reg(hlsl.scc, direct.scc, state.scc), |_| "scc".into()),
        field(
            "vgpr",
            KERNEL_LANES * KERNEL_VGPRS,
            |i| {
                let (l, r) = (i / KERNEL_VGPRS, i % KERNEL_VGPRS);
                if unstored[r] { None } else { Some((vg(hlsl, l, r), vg(direct, l, r))) }
            },
            |i| format!("lane {} v{}", i / KERNEL_VGPRS, i % KERNEL_VGPRS),
        ),
        field(
            "exports",
            KERNEL_SPLIT_SGPRS - KERNEL_EXPORTS,
            |i| Some((at(&hlsl.raw, KERNEL_EXPORTS + i), at(&direct.raw, KERNEL_EXPORTS + i))),
            |i| format!("exp{} lane {}.{}", i / (KERNEL_LANES * 4), i / 4 % KERNEL_LANES, ['x', 'y', 'z', 'w'][i % 4]),
        ),
        field(
            "mem",
            hlsl.mem.len().max(direct.mem.len()),
            |i| Some((at(&hlsl.mem, i), at(&direct.mem, i))),
            |i| format!("mem[{i}] (byte {:#x})", i * 4),
        ),
    ];
    if hlsl.mem.len() != direct.mem.len() {
        fields.push(Field { name: "mem length", count: 1, shown: vec![format!("hlsl {} direct {} dwords", hlsl.mem.len(), direct.mem.len())] });
    }
    for t in 0..hlsl.textures.len().max(direct.textures.len()) {
        let (h, d) = (hlsl.textures.get(t).map_or(&[][..], |v| v), direct.textures.get(t).map_or(&[][..], |v| v));
        let mut f = field("texture", h.len().max(d.len()).div_ceil(4), |i| Some((dword(h, i), dword(d, i))), |i| format!("texture {t} dword {i} (byte {:#x})", i * 4));
        if h.len() != d.len() {
            f.count += 1;
            f.shown.push(format!("texture {t} length: hlsl {} direct {} bytes", h.len(), d.len()));
        }
        fields.push(f);
    }
    report(&fields)
}

/// The report of the compared `fields`: Ok when none differs.
fn report(fields: &[Field]) -> Result<(), String> {
    let total: usize = fields.iter().map(|f| f.count).sum();
    if total == 0 {
        return Ok(());
    }
    let mut s = format!("XCL_DXIL_DIFF: HLSL and direct DXIL runs differ in {total} values");
    for f in fields.iter().filter(|f| f.count > 0) {
        s.push_str(&format!("\n  {} ({} differ):", f.name, f.count));
        for line in &f.shown {
            s.push_str(&format!("\n    {line}"));
        }
        if f.count > f.shown.len() {
            s.push_str(&format!("\n    ... {} more", f.count - f.shown.len()));
        }
    }
    Err(s)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::rdna2::encode;
    use crate::gpu::recomp::{KERNEL_RES_DWORDS, KERNEL_SGPRS};

    fn blank() -> KernelResult {
        KernelResult {
            sgpr: vec![0; KERNEL_SGPRS],
            vgpr: vec![vec![0; KERNEL_VGPRS]; KERNEL_LANES],
            scc: 0,
            raw: vec![0; KERNEL_RES_DWORDS],
            mem: vec![0; 64],
            textures: Vec::new(),
        }
    }

    #[test]
    fn equal_results_compare_equal() {
        let st = KernelState::default();
        let mut a = blank();
        a.sgpr[3] = 7;
        a.vgpr[9][4] = 0xDEAD;
        a.mem[2] = 1;
        a.textures = vec![vec![1, 2, 3, 4, 5]];
        let mut b = blank();
        b.sgpr[3] = 7;
        b.vgpr[9][4] = 0xDEAD;
        b.mem[2] = 1;
        b.textures = vec![vec![1, 2, 3, 4, 5]];
        assert_eq!(compare(&st, KERNEL_LANES, &a, &b), Ok(()));
    }

    #[test]
    fn report_names_field_slot_values_and_count() {
        let st = KernelState::default();
        let (mut a, mut b) = (blank(), blank());
        a.vgpr[5][3] = 1;
        b.vgpr[5][3] = 2;
        a.sgpr[126] = 0xFFFF_0000;
        b.scc = 1;
        let e = compare(&st, KERNEL_LANES, &a, &b).unwrap_err();
        assert!(e.contains("differ in 3 values"), "{e}");
        assert!(e.contains("vgpr (1 differ):\n    lane 5 v3: hlsl 0x00000001 direct 0x00000002"), "{e}");
        assert!(e.contains("exec_lo: hlsl 0xffff0000 direct 0x00000000"), "{e}");
        assert!(e.contains("scc (1 differ):\n    scc: hlsl 0x00000000 direct 0x00000001"), "{e}");
        assert!(!e.contains("mem"), "{e}");
    }

    #[test]
    fn report_lists_the_first_shown_differences_only() {
        let st = KernelState::default();
        let (a, mut b) = (blank(), blank());
        for i in 0..20 {
            b.mem[i] = i as u32 + 1;
        }
        let e = compare(&st, KERNEL_LANES, &a, &b).unwrap_err();
        assert!(e.contains("mem (20 differ):"), "{e}");
        assert!(e.contains("mem[7] (byte 0x1c): hlsl 0x00000000 direct 0x00000008"), "{e}");
        assert!(!e.contains("mem[8]"), "{e}");
        assert!(e.contains("... 12 more"), "{e}");
    }

    #[test]
    fn unstored_unchanged_registers_are_not_differences() {
        let mut st = KernelState::default();
        st.sgpr[10] = 100;
        st.vgpr[2][7] = 0x55;
        let (mut a, mut b) = (blank(), blank());
        // HLSL stored the input values, direct DXIL left them unstored
        a.sgpr[10] = 100;
        a.vgpr[2][7] = 0x55;
        a.sgpr[126] = u32::MAX;
        a.sgpr[127] = u32::MAX;
        assert_eq!(compare(&st, KERNEL_LANES, &a, &b), Ok(()));
        // a value that is neither the input nor unstored is a difference
        b.sgpr[10] = 5;
        let e = compare(&st, KERNEL_LANES, &a, &b).unwrap_err();
        assert!(e.contains("s10: hlsl 0x00000064 direct 0x00000005"), "{e}");
    }

    #[test]
    fn exports_and_textures_are_decoded() {
        let st = KernelState::default();
        let (mut a, mut b) = (blank(), blank());
        a.raw[KERNEL_EXPORTS + (2 * KERNEL_LANES + 7) * 4 + 1] = 0x3F80_0000;
        a.textures = vec![vec![0; 8], vec![0; 8]];
        b.textures = vec![vec![0; 8], vec![0, 0, 0, 0, 0x78, 0x56, 0x34, 0x12]];
        let e = compare(&st, KERNEL_LANES, &a, &b).unwrap_err();
        assert!(e.contains("exp2 lane 7.y: hlsl 0x3f800000 direct 0x00000000"), "{e}");
        assert!(e.contains("texture 1 dword 1 (byte 0x4): hlsl 0x00000000 direct 0x12345678"), "{e}");
        assert!(!e.contains("texture 0"), "{e}");
        // a texture size mismatch is reported too
        b.textures[0].pop();
        assert!(compare(&st, KERNEL_LANES, &a, &b).unwrap_err().contains("texture 0 length: hlsl 8 direct 7 bytes"));
    }

    #[test]
    fn a_pair_is_found_by_its_own_container_only() {
        let k = || Ok(Kernel { dxil: vec![9], static_samplers: Vec::new(), dispatch_chain: None });
        PAIRED.with(|p| *p.borrow_mut() = Some((vec![1, 2, 3], 32, vec![7], k())));
        assert!(take_paired(&[1, 2]).is_none());
        let (lanes, code, k) = take_paired(&[1, 2, 3]).unwrap();
        assert_eq!((lanes, code, k.unwrap().dxil), (32, vec![7], vec![9]));
        // taken out while it runs (`run_if_paired` puts it back afterwards)
        assert!(take_paired(&[1, 2, 3]).is_none());
        assert!(run_if_paired(&[1, 2], &KernelState::default(), &[], &[], &[], &[]).is_none());
    }

    #[test]
    fn a_vgpr_zeroed_in_one_lane_is_a_difference() {
        let mut st = KernelState::default();
        for v in st.vgpr.iter_mut() {
            v[4] = 0x55;
        }
        let (mut a, mut b) = (blank(), blank());
        for l in 0..KERNEL_LANES {
            a.vgpr[l][4] = 0x55;
            b.vgpr[l][4] = 0x55;
        }
        assert_eq!(compare(&st, KERNEL_LANES, &a, &b), Ok(()));
        // direct stored the register but zeroed an EXEC-masked lane
        b.vgpr[3][4] = 0;
        let e = compare(&st, KERNEL_LANES, &a, &b).unwrap_err();
        assert!(e.contains("vgpr (1 differ):\n    lane 3 v4: hlsl 0x00000055 direct 0x00000000"), "{e}");
        // a wave32 run: lanes 32.. ran in neither, so one side unstored still matches
        let mut c = blank();
        for l in 0..32 {
            c.vgpr[l][4] = 0x55;
        }
        assert_eq!(compare(&st, 32, &c, &blank()), Ok(()));
        assert!(compare(&st, KERNEL_LANES, &c, &blank()).is_err());
    }

    #[test]
    fn dispatch_memory_is_compared() {
        assert_eq!(compare_mem(&[1, 2, 3], &[1, 2, 3]), Ok(()));
        let e = compare_mem(&[1, 2, 3], &[1, 7, 3]).unwrap_err();
        assert!(e.contains("mem (1 differ):\n    mem[1] (byte 0x4): hlsl 0x00000002 direct 0x00000007"), "{e}");
        assert!(compare_mem(&[1, 2], &[1, 2, 0]).unwrap_err().contains("hlsl 2 direct 3 dwords"));
    }

    #[test]
    fn refusals_are_not_translator_failures() {
        assert!(!translator_failed("+0x10: no direct lowering for some instruction"));
        assert!(translator_failed("direct DXIL failed validation: bad"));
        assert!(translator_failed("direct DXIL builder error: bad"));
        assert!(translator_failed("the direct DXIL translator panicked: overflow"));
    }

    /// A program with the returning atomics of the given (name, encoding, fields), then
    /// `S_ENDPGM`.
    fn program(list: &[(&str, &str, &[(&str, u64)])]) -> Vec<u8> {
        let mut code = Vec::new();
        for (n, e, f) in list {
            code.extend(encode(n, e, f, None).unwrap_or_else(|x| panic!("{n}: {x}")));
        }
        code.extend(encode("S_ENDPGM", "ENC_SOPP", &[], None).unwrap());
        code
    }

    #[test]
    fn atomic_returns_are_the_vgprs_returning_atomics_alone_write() {
        let code = program(&[
            ("DS_ADD_RTN_U32", "ENC_DS", &[("VDST", 26), ("ADDR", 27), ("DATA0", 28)]),
            ("DS_ADD_U32", "ENC_DS", &[("ADDR", 27), ("DATA0", 28)]),
            ("BUFFER_ATOMIC_ADD", "ENC_MUBUF", &[("VDATA", 15), ("VADDR", 0), ("SRSRC", 5)]),
            ("BUFFER_ATOMIC_UMAX", "ENC_MUBUF", &[("VDATA", 16), ("VADDR", 0), ("SRSRC", 5), ("GLC", 1)]),
            ("BUFFER_ATOMIC_CMPSWAP_X2", "ENC_MUBUF", &[("VDATA", 30), ("VADDR", 8), ("SRSRC", 3), ("GLC", 1)]),
            ("GLOBAL_ATOMIC_FCMPSWAP_X2", "ENC_FLAT_GLBL", &[("VDST", 44), ("ADDR", 9), ("DATA", 46), ("SADDR", 3), ("GLC", 1)]),
            ("IMAGE_ATOMIC_CMPSWAP", "ENC_MIMG", &[("VDATA", 50), ("VADDR", 4), ("SRSRC", 4), ("DMASK", 3), ("GLC", 1)]),
            // written by a VALU op too: compared lane by lane
            ("FLAT_ATOMIC_ADD", "ENC_FLAT", &[("VDST", 60), ("ADDR", 14), ("DATA", 16), ("GLC", 1)]),
            ("V_MOV_B32", "ENC_VOP1", &[("VDST", 60), ("SRC0", 128)]),
        ]);
        assert_eq!(atomic_returns(&code), vec![26..27, 16..17, 30..32, 44..46, 50..51]);
        // a program that does not decode is compared strictly, as is one that writes a VGPR
        // M0 picks
        assert_eq!(atomic_returns(&[0xFF; 4]), Vec::<Range<usize>>::new());
        let movrel = program(&[
            ("DS_ADD_RTN_U32", "ENC_DS", &[("VDST", 26), ("ADDR", 27), ("DATA0", 28)]),
            ("V_MOVRELD_B32", "ENC_VOP1", &[("VDST", 20), ("SRC0", 256 + 1)]),
        ]);
        assert_eq!(atomic_returns(&movrel), Vec::<Range<usize>>::new());
    }

    /// `ds_add_rtn_u32 v26` (every lane adds 1): the old values are 0..64 in any lane order.
    fn add_rtn() -> (Vec<u8>, KernelResult, KernelResult) {
        let code = program(&[("DS_ADD_RTN_U32", "ENC_DS", &[("VDST", 26), ("ADDR", 27), ("DATA0", 28)])]);
        let (mut h, mut d) = (blank(), blank());
        for l in 0..KERNEL_LANES {
            h.vgpr[l][26] = l as u32;
            d.vgpr[l][26] = (l as u32 * 37) % 64; // a permutation of 0..64
            h.vgpr[l][3] = l as u32;
            d.vgpr[l][3] = l as u32;
        }
        (code, h, d)
    }

    #[test]
    fn permuted_atomic_returns_compare_equal() {
        let (code, h, d) = add_rtn();
        let st = KernelState::default();
        assert!(compare(&st, KERNEL_LANES, &h, &d).is_err(), "strict: the lanes differ");
        assert_eq!(compare_relaxed(&st, KERNEL_LANES, &h, &d, &atomic_returns(&code), &Relax::default()), Ok(()));
    }

    #[test]
    fn a_wrong_atomic_return_is_still_a_difference() {
        let (code, h, mut d) = add_rtn();
        let st = KernelState::default();
        // one lane got a value no lane got on the HLSL side (returned the new value, say)
        d.vgpr[5][26] = 64;
        let e = compare_relaxed(&st, KERNEL_LANES, &h, &d, &atomic_returns(&code), &Relax::default()).unwrap_err();
        assert!(e.contains("vgpr (") && e.contains(" v26: hlsl"), "{e}");
        // the same values in another lane order, but another register differs as well
        let (code, h, mut d) = add_rtn();
        d.vgpr[9][3] = 7;
        let e = compare_relaxed(&st, KERNEL_LANES, &h, &d, &atomic_returns(&code), &Relax::default()).unwrap_err();
        assert!(e.contains("vgpr (1 differ):\n    lane 9 v3: hlsl 0x00000009 direct 0x00000007"), "{e}");
    }

    #[test]
    fn a_permutation_of_a_non_atomic_register_is_a_difference() {
        let (_, h, d) = add_rtn();
        // v26 written by a VALU op, not an atomic
        let code = program(&[("V_MOV_B32", "ENC_VOP1", &[("VDST", 26), ("SRC0", 128)])]);
        assert!(atomic_returns(&code).is_empty());
        let st = KernelState::default();
        assert!(compare_relaxed(&st, KERNEL_LANES, &h, &d, &atomic_returns(&code), &Relax::default()).is_err());
        // a non-atomic register of a program with atomics: two lanes swapped
        let (code, h, mut d) = add_rtn();
        d.vgpr[1][26] = h.vgpr[1][26];
        for l in 0..KERNEL_LANES {
            d.vgpr[l][26] = h.vgpr[l][26];
        }
        d.vgpr.swap(1, 2);
        let e = compare_relaxed(&st, KERNEL_LANES, &h, &d, &atomic_returns(&code), &Relax::default()).unwrap_err();
        assert!(e.contains("lane 1 v3: hlsl 0x00000001 direct 0x00000002"), "{e}");
    }

    #[test]
    fn a_wave32_multiset_holds_the_lanes_that_ran() {
        let (code, mut h, mut d) = add_rtn();
        let st = KernelState::default();
        for l in 0..KERNEL_LANES {
            h.vgpr[l][26] = if l < 32 { l as u32 } else { 0 };
            d.vgpr[l][26] = if l < 32 { 31 - l as u32 } else { 0 };
        }
        assert_eq!(compare_relaxed(&st, 32, &h, &d, &atomic_returns(&code), &Relax::default()), Ok(()));
        // a value swapped with a lane that did not run is a difference
        d.vgpr[40][26] = d.vgpr[2][26];
        d.vgpr[2][26] = 0;
        assert!(compare_relaxed(&st, 32, &h, &d, &atomic_returns(&code), &Relax::default()).is_err());
    }

    #[test]
    fn unordered_values_are_left_to_the_tests_check() {
        let st = KernelState::default();
        let (mut h, mut d) = (blank(), blank());
        // one shared cmpswap cell (mem[4..6]): lane 3 won on the HLSL side, lane 7 on the direct
        for l in 0..KERNEL_LANES {
            h.vgpr[l][40] = if l == 3 { 0x15 } else { 0x100 + 3 };
            d.vgpr[l][40] = if l == 7 { 0x15 } else { 0x100 + 7 };
        }
        h.mem[4] = 0x103;
        d.mem[4] = 0x107;
        assert!(compare(&st, KERNEL_LANES, &h, &d).is_err());
        // the test's check: one winner got the initial value, the others and the cell its value
        let check = |r: &KernelResult| {
            let w = (0..KERNEL_LANES).find(|&l| r.vgpr[l][40] == 0x15).expect("a winner");
            for l in (0..KERNEL_LANES).filter(|&l| l != w) {
                assert_eq!(r.vgpr[l][40], 0x100 + w as u32, "lane {l}");
            }
            assert_eq!(r.mem[4], 0x100 + w as u32, "cell");
        };
        let r = Relax { unordered: Some(Unordered { vgprs: vec![40..41], mem: vec![4..6], check: Box::new(check) }), ..Default::default() };
        assert_eq!(compare_relaxed(&st, KERNEL_LANES, &h, &d, &[], &r), Ok(()));
        // a direct run the check refuses is a difference
        d.mem[4] = 0x103;
        let e = compare_relaxed(&st, KERNEL_LANES, &h, &d, &[], &r).unwrap_err();
        assert!(e.contains("fails the test's check of its unordered values") && e.contains("cell"), "{e}");
        // and slots outside the named ones are compared strictly
        d.mem[4] = 0x107;
        d.mem[6] = 1;
        assert!(compare_relaxed(&st, KERNEL_LANES, &h, &d, &[], &r).unwrap_err().contains("mem[6] (byte 0x18)"));
    }

    #[test]
    fn a_relaxation_ends_with_its_guard() {
        let r = Relax { unordered: Some(Unordered { vgprs: vec![1..2], mem: Vec::new(), check: Box::new(|_| {}) }), ..Default::default() };
        {
            let _g = relax(r);
            assert!(RELAX.with(|t| t.borrow().unordered.is_some()));
        }
        assert!(RELAX.with(|t| t.borrow().unordered.is_none()));
    }

    /// `lanes` results with the f32 or f64 value `h` / `d` in v`at` (and v`at + 1`) of lane 0.
    fn nan_pair(at: usize, h: u64, d: u64) -> (KernelResult, KernelResult) {
        let (mut a, mut b) = (blank(), blank());
        (a.vgpr[0][at], a.vgpr[0][at + 1], b.vgpr[0][at], b.vgpr[0][at + 1]) = (h as u32, (h >> 32) as u32, d as u32, (d >> 32) as u32);
        (a, b)
    }

    fn nan_rule(vgprs: Range<usize>, f64: bool, eq: NanEq) -> Relax {
        Relax { nan: vec![Nan { vgprs, f64, eq }], ..Default::default() }
    }

    #[test]
    fn nan_classes() {
        assert_eq!(nan_class(0x7FC0_0000, false), Some(true));
        assert_eq!(nan_class(0xFFFF_FFFF, false), Some(true));
        assert_eq!(nan_class(0x7F80_0001, false), Some(false));
        assert_eq!(nan_class(0x7F80_0000, false), None, "+inf");
        assert_eq!(nan_class(0x3F80_0000, false), None);
        assert_eq!(nan_class(0x7FF8_0000_0000_0000, true), Some(true));
        assert_eq!(nan_class(0x7FF0_0000_0000_0001, true), Some(false));
        assert_eq!(nan_class(0xFFF0_0000_0000_0000, true), None, "-inf");
        // an f32 NaN pattern in the low dword of an f64 is a number
        assert_eq!(nan_class(0x7FC0_0000, true), None);
    }

    #[test]
    fn f32_nans_compare_by_class_where_a_test_asks() {
        let st = KernelState::default();
        let rel = |h, d, r: &Relax| {
            let (a, b) = nan_pair(6, h, d);
            compare_relaxed(&st, KERNEL_LANES, &a, &b, &[], r)
        };
        let (class, any) = (nan_rule(4..8, false, NanEq::Class), nan_rule(4..8, false, NanEq::Any));
        // two quiet NaNs with other signs and payloads: equal by class, not by default
        assert!(rel(0x7FC0_0000, 0xFFFF_FFFF, &Relax::default()).is_err());
        assert_eq!(rel(0x7FC0_0000, 0xFFFF_FFFF, &class), Ok(()));
        // signalling against quiet: a difference by class, equal for Any
        assert!(rel(0x7FA0_0000, 0x7FE0_0000, &class).is_err());
        assert_eq!(rel(0x7FA0_0000, 0x7FE0_0000, &any), Ok(()));
        // a NaN against an infinity or a number, and two numbers 1 ulp apart, still differ
        assert!(rel(0x7FC0_0000, 0x7F80_0000, &any).is_err());
        assert!(rel(0x3F80_0000, 0x3F80_0001, &any).is_err());
        // registers outside the rule stay bit for bit
        assert!(rel(0x7FC0_0000, 0xFFFF_FFFF, &nan_rule(0..6, false, NanEq::Any)).is_err());
    }

    #[test]
    fn f64_nans_compare_both_dwords_by_class() {
        let st = KernelState::default();
        let rel = |h, d, r: &Relax| {
            let (a, b) = nan_pair(14, h, d);
            compare_relaxed(&st, KERNEL_LANES, &a, &b, &[], r)
        };
        // a signalling NaN quieted on one side: the high dwords differ
        let (snan, qnan) = (0x7FF0_0000_0000_0001u64, 0x7FF8_0000_0000_0001u64);
        let e = rel(snan, qnan, &Relax::default()).unwrap_err();
        assert!(e.contains("lane 0 v15: hlsl 0x7ff00000 direct 0x7ff80000"), "{e}");
        assert!(rel(snan, qnan, &nan_rule(14..16, true, NanEq::Class)).is_err());
        assert_eq!(rel(snan, qnan, &nan_rule(14..16, true, NanEq::Any)), Ok(()));
        // quiet NaNs with other payloads in either dword: equal by class
        assert_eq!(rel(qnan, 0xFFF8_0000_DEAD_0000, &nan_rule(14..16, true, NanEq::Class)), Ok(()));
        // +inf (0x7ff00000:00000000) is not a NaN
        assert!(rel(0x7FF0_0000_0000_0000, qnan, &nan_rule(14..16, true, NanEq::Any)).is_err());
        // pairs run from the range's start: v13:v14 is not the pair v14:v15
        assert!(rel(snan, qnan, &nan_rule(13..17, true, NanEq::Any)).is_err());
    }

    /// A small program through both backends via the diff entry point (independent of
    /// `XCL_DXIL_DIFF`): the runs agree and give the expected registers.
    #[test]
    fn gpu_diff_run_of_a_small_program() {
        match (runner::adapter(), runner::wave64_programs_supported(), crate::gpu::recomp::dxc::location()) {
            (Ok(_), Ok(true), Ok(_)) => {}
            (a, w, d) => {
                runner::skip_test(if a.is_err() || w.is_err() { "no GPU" } else if d.is_err() { "no DXC" } else { "no wave64" });
                return;
            }
        }
        let mut code = Vec::new();
        code.extend(encode("S_ADD_U32", "ENC_SOP2", &[("SDST", 2), ("SSRC0", 0), ("SSRC1", 1)], None).unwrap());
        code.extend(encode("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 1), ("SRC0", 2), ("VSRC1", 0)], None).unwrap());
        code.extend(encode("S_ENDPGM", "ENC_SOPP", &[], None).unwrap());
        // the program must reach the direct backend, not just its refusal path
        runner::direct_kernel_full(&code, true).unwrap_or_else(|e| panic!("direct DXIL refused the test program: {e}"));
        let mut st = KernelState::default();
        st.sgpr[0] = 5;
        st.sgpr[1] = 7;
        for (lane, v) in st.vgpr.iter_mut().enumerate() {
            v[0] = lane as u32;
        }
        let r = run_code_with(&code, &st, &[1, 2, 3], &[], &[]).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!((r.sgpr[2], r.scc), (12, 0));
        for lane in 0..KERNEL_LANES {
            assert_eq!(r.vgpr[lane][1], 12 + lane as u32, "lane {lane}");
        }
        assert_eq!(&r.mem[..3], &[1, 2, 3]);
    }
}
