//! The covered set (`covered::TABLE`) checked against programs: a compute program for each
//! covered class, with the memory RDNA2 leaves after one group of 64 threads. The CPU tests
//! keep the programs inside the covered set, make every covered class appear in one of them
//! (a class marked covered needs a program here) and translate them directly. The GPU test
//! runs each through both backends, wave32 and wave64 (split where the adapter needs it),
//! compares the memory with the reference and prints the translation times; with
//! `XCL_COVERED_BENCH=1` it also times 4096 groups of each on the GPU. A console MS program
//! does the same for the mesh pass's own classes (`covered::MESH_TABLE`), through both
//! backends' compute pass and the pass-through mesh shader, and a console ES + GS program for
//! the geometry pass's (`covered::GEOM_TABLE`).
//!
//! Candidates are programs for the next classes to flip (transcendental, valu64, cross-lane,
//! barrier, the LDS, buffer and global atomics), each with its RDNA2 reference. The GPU test
//! runs them the same way and prints one verdict line per program and form, without failing:
//! they are the evidence a flip needs. Where the lanes' order decides a value (what a returning
//! atomic gives back), the reference is a property every order has, not one order's words.

use std::collections::{BTreeMap, BTreeSet};

use super::cfg::{Cfg, Term};
use super::covered;
use super::kernel::{translate_compute, translate_geometry, translate_mesh, KernelOptions};
use crate::gpu::rdna2::{self, encode};
use crate::gpu::recomp::{CsInterface, GsInterface, Mode, MsInterface, Options};

const V: u64 = 256; // VALU source code of v0
const BASE: u32 = 0x400; // s0: where a program's memory starts
const THREADS: u32 = 64;
const LOOPS: u32 = 100; // s2: program `int`'s loop count

/// s[0:1] = BASE (high half 0), s2 = loop count; v0 = thread id.
fn interface() -> CsInterface {
    CsInterface { threads: [THREADS, 1, 1], user_sgprs: 4, tgid: [false; 3], tg_size: false, tid_components: 1 }
}

/// A program under construction, with SOPP branches to labels.
struct Asm(Vec<u8>);

impl Asm {
    fn op(&mut self, name: &str, enc: &str, f: &[(&str, u64)], lit: Option<u32>) -> &mut Self {
        self.0.extend(encode(name, enc, f, lit).unwrap_or_else(|e| panic!("{name}: {e}")));
        self
    }

    fn here(&self) -> usize {
        self.0.len()
    }

    /// A branch to `to` (a label already placed), or, with None, one `land` places later.
    fn branch(&mut self, name: &str, to: Option<usize>) -> usize {
        let at = self.here();
        self.op(name, "ENC_SOPP", &[("SIMM16", 0)], None);
        if let Some(to) = to {
            self.aim(at, to);
        }
        at
    }

    /// The branch at `at` goes to the next instruction placed.
    fn land(&mut self, at: usize) {
        let to = self.here();
        self.aim(at, to);
    }

    fn aim(&mut self, at: usize, to: usize) {
        let words = (to as i64 - (at as i64 + 4)) / 4;
        self.0[at..at + 2].copy_from_slice(&(words as i16).to_le_bytes());
    }

    /// v6:v7 = BASE + thread id * 4 (v1 = thread id * 4), the address the stores use.
    fn lane_address(&mut self) -> &mut Self {
        self.op("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 1), ("SRC0", 130), ("VSRC1", 0)], None)
            .op("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 6), ("SRC0", 0), ("VSRC1", 1)], None)
            .op("V_MOV_B32", "ENC_VOP1", &[("VDST", 7), ("SRC0", 128)], None)
    }

    fn store(&mut self, data: u64, offset: u64) -> &mut Self {
        self.op("GLOBAL_STORE_DWORD", "ENC_FLAT_GLBL", &[("ADDR", 6), ("DATA", data), ("SADDR", 0x7D), ("OFFSET", offset)], None)
    }

    /// v12:v13 = BASE + 512 r + thread id * 8 (v9 = thread id * 8): lane t's 8 bytes of region r.
    fn at(&mut self, r: u32) -> &mut Self {
        self.op("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 12), ("SRC0", 255), ("VSRC1", 9)], Some(BASE + 512 * r))
            .op("V_MOV_B32", "ENC_VOP1", &[("VDST", 13), ("SRC0", 128)], None)
    }

    fn end(&mut self) -> Vec<u8> {
        self.op("S_ENDPGM", "ENC_SOPP", &[], None);
        std::mem::take(&mut self.0)
    }
}

/// A program, its memory before the run (BASE.. as words), the memory RDNA2 leaves, and the
/// words a backend that contracts a mul into the add reading it computes differently.
struct Case {
    name: &'static str,
    code: Vec<u8>,
    mem: Vec<u32>,
    want: Vec<u32>,
    contracted: Vec<usize>,
}

fn word(k: usize) -> usize {
    BASE as usize / 4 + k
}

fn input(t: u32) -> u32 {
    t.wrapping_mul(0x9E37_79B9) ^ 0x0123_4567
}

/// salu, flow, valu, global: an LCG loop counted per lane with v_cmp + v_cndmask, then a
/// divergent add on odd lanes (saveexec, execz skip). The wave size picks the EXEC width.
fn int(wave64: bool) -> Case {
    let mut a = Asm(Vec::new());
    a.op("S_MOV_B32", "ENC_SOP1", &[("SDST", 10), ("SSRC0", 128)], None)
        .op("V_MOV_B32", "ENC_VOP1", &[("VDST", 2), ("SRC0", V)], None)
        .op("V_MOV_B32", "ENC_VOP1", &[("VDST", 4), ("SRC0", 128)], None);
    let top = a.here();
    a.op("V_MUL_LO_U32", "ENC_VOP3", &[("VDST", 2), ("SRC0", V + 2), ("SRC1", 255)], Some(1_103_515_245))
        .op("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 2), ("SRC0", 255), ("VSRC1", 2)], Some(12345))
        .op("V_LSHRREV_B32", "ENC_VOP2", &[("VDST", 3), ("SRC0", 144), ("VSRC1", 2)], None)
        .op("V_CMP_GT_U32", "ENC_VOPC", &[("SRC0", 255), ("VSRC1", 3)], Some(0x4000))
        .op("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 5), ("SRC0", 129), ("VSRC1", 4)], None)
        .op("V_CNDMASK_B32", "ENC_VOP2", &[("VDST", 4), ("SRC0", V + 4), ("VSRC1", 5)], None)
        .op("S_ADD_U32", "ENC_SOP2", &[("SDST", 10), ("SSRC0", 10), ("SSRC1", 129)], None)
        .op("S_CMP_LG_U32", "ENC_SOPC", &[("SSRC0", 10), ("SSRC1", 2)], None);
    a.branch("S_CBRANCH_SCC1", Some(top));
    let (saveexec, mov) = if wave64 { ("S_AND_SAVEEXEC_B64", "S_MOV_B64") } else { ("S_AND_SAVEEXEC_B32", "S_MOV_B32") };
    a.op("V_AND_B32", "ENC_VOP2", &[("VDST", 8), ("SRC0", 129), ("VSRC1", 0)], None)
        .op("V_CMP_NE_U32", "ENC_VOPC", &[("SRC0", 128), ("VSRC1", 8)], None)
        .op(saveexec, "ENC_SOP1", &[("SDST", 20), ("SSRC0", 106)], None);
    let skip = a.branch("S_CBRANCH_EXECZ", None);
    a.op("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 4), ("SRC0", 255), ("VSRC1", 4)], Some(1000));
    a.land(skip);
    a.op(mov, "ENC_SOP1", &[("SDST", 126), ("SSRC0", 20)], None).lane_address().store(4, 0).store(2, 256);
    let code = a.end();
    let mem = vec![0u32; word(128)];
    let mut want = mem.clone();
    for t in 0..THREADS {
        let (mut x, mut n) = (t, 0u32);
        for _ in 0..LOOPS {
            x = x.wrapping_mul(1_103_515_245).wrapping_add(12345);
            n += (0x4000 > x >> 16) as u32;
        }
        want[word(t as usize)] = n + if t & 1 == 1 { 1000 } else { 0 };
        want[word(64 + t as usize)] = x;
    }
    Case { name: "int", code, mem, want, contracted: Vec::new() }
}

/// valu-f32: x = 1 + t / 4096; x * x, then x * x + (1 - 2x) (a mul read by an add: RDNA2
/// rounds each, a fused form keeps (t / 4096)^2 exactly), and u32(1000 x).
fn f32_ops() -> Case {
    let mut a = Asm(Vec::new());
    a.op("V_CVT_F32_U32", "ENC_VOP1", &[("VDST", 1), ("SRC0", V)], None)
        .op("V_MUL_F32", "ENC_VOP2", &[("VDST", 1), ("SRC0", 255), ("VSRC1", 1)], Some(0x3980_0000)) // 2^-12
        .op("V_ADD_F32", "ENC_VOP2", &[("VDST", 1), ("SRC0", 242), ("VSRC1", 1)], None) // 1.0 +
        .op("V_MUL_F32", "ENC_VOP2", &[("VDST", 2), ("SRC0", V + 1), ("VSRC1", 1)], None)
        .op("V_MUL_F32", "ENC_VOP2", &[("VDST", 4), ("SRC0", 245), ("VSRC1", 1)], None) // -2.0 *
        .op("V_ADD_F32", "ENC_VOP2", &[("VDST", 4), ("SRC0", 242), ("VSRC1", 4)], None)
        .op("V_ADD_F32", "ENC_VOP2", &[("VDST", 3), ("SRC0", V + 2), ("VSRC1", 4)], None)
        .op("V_MUL_F32", "ENC_VOP2", &[("VDST", 5), ("SRC0", 255), ("VSRC1", 1)], Some(1000f32.to_bits()))
        .op("V_CVT_U32_F32", "ENC_VOP1", &[("VDST", 5), ("SRC0", V + 5)], None)
        .lane_address()
        .store(2, 0)
        .store(3, 256)
        .store(5, 512);
    let code = a.end();
    let mem = vec![0u32; word(192)];
    let mut want = mem.clone();
    let mut contracted = Vec::new();
    for t in 0..THREADS as usize {
        let x = 1.0f32 + t as f32 * (1.0 / 4096.0);
        let (sq, c) = (x * x, 1.0f32 + -2.0 * x);
        want[word(t)] = sq.to_bits();
        want[word(64 + t)] = (sq + c).to_bits();
        want[word(128 + t)] = (1000.0 * x) as u32;
        if x.mul_add(x, c) != sq + c {
            contracted.push(word(64 + t));
        }
    }
    Case { name: "f32", code, mem, want, contracted }
}

/// global loads, lds (a write and a read of each lane's own words), hint (s_waitcnt).
fn lds() -> Case {
    let wait = ("S_WAITCNT", "ENC_SOPP", [("SIMM16", 0u64)]);
    let mut a = Asm(Vec::new());
    a.lane_address()
        .op("GLOBAL_LOAD_DWORD", "ENC_FLAT_GLBL", &[("VDST", 2), ("ADDR", 6), ("SADDR", 0x7D), ("OFFSET", 0)], None)
        .op(wait.0, wait.1, &wait.2, None)
        .op("V_XOR_B32", "ENC_VOP2", &[("VDST", 3), ("SRC0", 255), ("VSRC1", 2)], Some(0x5A5A_5A5A))
        .op("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 8), ("SRC0", 255), ("VSRC1", 1)], Some(256))
        .op("DS_WRITE_B32", "ENC_DS", &[("ADDR", 1), ("DATA0", 3)], None)
        .op("DS_WRITE_B32", "ENC_DS", &[("ADDR", 8), ("DATA0", 2)], None)
        .op(wait.0, wait.1, &wait.2, None)
        .op("DS_READ_B32", "ENC_DS", &[("VDST", 4), ("ADDR", 1)], None)
        .op("DS_READ_B32", "ENC_DS", &[("VDST", 5), ("ADDR", 8)], None)
        .op(wait.0, wait.1, &wait.2, None)
        .op("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 4), ("SRC0", V + 5), ("VSRC1", 4)], None)
        .store(4, 256);
    let code = a.end();
    let mut mem = vec![0u32; word(128)];
    for t in 0..THREADS {
        mem[word(t as usize)] = input(t);
    }
    let mut want = mem.clone();
    for t in 0..THREADS {
        want[word(64 + t as usize)] = (input(t) ^ 0x5A5A_5A5A).wrapping_add(input(t));
    }
    Case { name: "lds", code, mem, want, contracted: Vec::new() }
}

/// scalar-load (a word at BASE + 0x200) and buffer (a raw V# at BASE: a load of each lane's
/// word, a store 0x300 bytes on).
fn buffer() -> Case {
    let mut a = Asm(Vec::new());
    a.op("S_LOAD_DWORD", "ENC_SMEM", &[("SDATA", 8), ("SBASE", 0), ("OFFSET", 0x200), ("SOFFSET", 125)], None)
        .op("S_MOV_B32", "ENC_SOP1", &[("SDST", 12), ("SSRC0", 0)], None)
        .op("S_MOV_B32", "ENC_SOP1", &[("SDST", 13), ("SSRC0", 128)], None)
        .op("S_MOV_B32", "ENC_SOP1", &[("SDST", 14), ("SSRC0", 255)], Some(0x1000))
        .op("S_MOV_B32", "ENC_SOP1", &[("SDST", 15), ("SSRC0", 128)], None)
        .op("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 1), ("SRC0", 130), ("VSRC1", 0)], None)
        .op("BUFFER_LOAD_DWORD", "ENC_MUBUF", &[("VDATA", 2), ("VADDR", 1), ("SRSRC", 3), ("SOFFSET", 128), ("OFFEN", 1)], None)
        .op("S_WAITCNT", "ENC_SOPP", &[("SIMM16", 0)], None)
        .op("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 3), ("SRC0", 8), ("VSRC1", 2)], None)
        .op("BUFFER_STORE_DWORD", "ENC_MUBUF", &[("VDATA", 3), ("VADDR", 1), ("SRSRC", 3), ("SOFFSET", 128), ("OFFEN", 1), ("OFFSET", 0x300)], None);
    let code = a.end();
    let k = 0x0102_0304u32;
    let mut mem = vec![0u32; word(0x300 / 4 + 64)];
    for t in 0..THREADS {
        mem[word(t as usize)] = input(t);
    }
    mem[word(0x200 / 4)] = k;
    let mut want = mem.clone();
    for t in 0..THREADS {
        want[word(0x300 / 4 + t as usize)] = input(t).wrapping_add(k);
    }
    Case { name: "buffer", code, mem, want, contracted: Vec::new() }
}

fn cases(wave64: bool) -> Vec<Case> {
    vec![int(wave64), f32_ops(), lds(), buffer()]
}

/// A program for a class outside the covered set, run for evidence only. A `near` word is an
/// f32 result RDNA2 documents to 1 ulp: it passes within 1 ulp of its exact value (an f64), or
/// as any NaN where that value is NaN; the words of a `Race` pass its own check; every other
/// word must match `want` exactly. `names` label word ranges (start, end, name) for the verdict.
struct Candidate {
    class: &'static str,
    case: Case,
    near: Vec<(usize, f64)>,
    races: Vec<Race>,
    names: Vec<(usize, usize, String)>,
}

impl Candidate {
    fn passes(&self, k: usize, got: u32) -> bool {
        match self.near.iter().find(|n| n.0 == k) {
            Some(&(_, e)) => near(got, e),
            None => got == self.case.want[k],
        }
    }

    /// "OK", or how many words differ and the first, as its index from BASE (and its name),
    /// and each race no order of the lanes explains.
    fn verdict(&self, out: &[u32], hlsl: bool) -> String {
        let raced: BTreeSet<usize> = self.races.iter().flat_map(Race::words).collect();
        let mut fails: Vec<String> = self.races.iter().filter_map(|r| r.check(out).err()).collect();
        let bad: Vec<usize> = (0..self.case.want.len()).filter(|&k| !raced.contains(&k) && !self.passes(k, out[k])).collect();
        let Some(&k) = bad.first() else { return listed(fails) };
        if hlsl && fails.is_empty() && bad.iter().all(|b| self.case.contracted.contains(b)) {
            return format!("differs only by a fused mul + add ({} word(s))", bad.len());
        }
        let want = match self.near.iter().find(|n| n.0 == k) {
            Some(&(_, e)) if e.is_nan() => "any NaN".to_string(),
            Some(&(_, e)) => format!("{e:e} within 1 ulp"),
            None => format!("{:#010x}", self.case.want[k]),
        };
        let name = self.names.iter().find(|n| (n.0..n.1).contains(&k)).map_or(String::new(), |n| format!(" {}[{}]", n.2, k - n.0));
        fails.insert(0, format!("differs ({} word(s), first [{}]{name}: got {:#010x}, want {want})", bad.len(), k - word(0), out[k]));
        listed(fails)
    }
}

/// "OK", or the first three failures and how many more.
fn listed(fails: Vec<String>) -> String {
    match fails.len() {
        0 => "OK".into(),
        1..=3 => fails.join("; "),
        n => format!("{}; and {} more", fails[..3].join("; "), n - 3),
    }
}

/// `got` is within 1 ulp of `exact` (an f32 ulp at `exact`'s binade), or both are NaN.
fn near(got: u32, exact: f64) -> bool {
    let g = f32::from_bits(got) as f64;
    if exact.is_nan() {
        return g.is_nan();
    }
    let ulp = 2f64.powi((exact.abs().log2().floor() as i32).max(-126) - 23);
    (g - exact).abs() <= ulp
}

/// RDNA2's v_rcp / v_rsq / v_sqrt / v_exp / v_log f32 (`op` 0..5) of `x`: Ok(bits) where the
/// result is exact (zeros, infinities, representable values), else Err(the exact value) for the
/// 1-ulp compare. The ISA guide gives each 1 ulp with denormals flushed: a denormal input reads
/// as a zero of its sign, and a result below 2^-126 is a zero of its sign.
fn transcendental(op: usize, x: u32) -> Result<u32, f64> {
    let x = if x & 0x7F80_0000 == 0 { x & 0x8000_0000 } else { x };
    let v = f32::from_bits(x) as f64;
    let r = match op {
        0 => 1.0 / v,
        1 if v == 0.0 => 1.0 / v,
        1 => 1.0 / v.sqrt(),
        2 => v.sqrt(),
        3 => v.exp2(),
        _ => v.log2(),
    };
    let f = r as f32;
    if r.is_nan() {
        Err(r)
    } else if r.abs() < f32::MIN_POSITIVE as f64 {
        Ok(f.to_bits() & 0x8000_0000)
    } else if f as f64 == r || f.is_infinite() {
        Ok(f.to_bits())
    } else {
        Err(r)
    }
}

/// Lane t's input to the transcendental program: zeros, infinities, NaN, denormals, the edges
/// of exp's range and of rcp's, values whose results are exact, then a spread of others.
fn t_input(t: u32) -> u32 {
    const EDGE: [u32; 32] = [
        0, 0x8000_0000, 0x7F80_0000, 0xFF80_0000, 0x7FC0_0000, 1, 0x007F_FFFF, 0x8000_0001, 0x0080_0000, 0x8080_0000,
        0x7F7F_FFFF, 0x7F00_0000, 0x7180_0000, 0x3F80_0000, 0xBF80_0000, 0x4000_0000, 0xC000_0000, 0x4080_0000,
        0x3F00_0000, 0x4300_0000, 0xC2FC_0000, 0xC2FE_0000, 0xC315_0000, 0xC2FD_0000, 0x3F80_0001, 0x3F7F_FFFF,
        0x4040_0000, 0x3A83_126F, 0x7149_F2CA, 0xF149_F2CA, 0x2EDB_E6FF, 0xBF00_0000,
    ];
    match EDGE.get(t as usize) {
        Some(&x) => x,
        None => ((t as f32 - 48.0) * 3.25 + 0.125).to_bits(),
    }
}

/// transcendental: v_rcp / v_rsq / v_sqrt / v_exp / v_log f32 of each lane's input.
fn transcendental_ops() -> Candidate {
    const OPS: [&str; 5] = ["V_RCP_F32", "V_RSQ_F32", "V_SQRT_F32", "V_EXP_F32", "V_LOG_F32"];
    let mut a = Asm(Vec::new());
    a.lane_address()
        .op("GLOBAL_LOAD_DWORD", "ENC_FLAT_GLBL", &[("VDST", 2), ("ADDR", 6), ("SADDR", 0x7D), ("OFFSET", 0)], None)
        .op("S_WAITCNT", "ENC_SOPP", &[("SIMM16", 0)], None);
    for (k, op) in OPS.iter().enumerate() {
        a.op(op, "ENC_VOP1", &[("VDST", 10 + k as u64), ("SRC0", V + 2)], None);
    }
    for k in 0..5 {
        a.store(10 + k, 256 * (k + 1));
    }
    let code = a.end();
    let mut mem = vec![0u32; word(384)];
    for t in 0..THREADS {
        mem[word(t as usize)] = t_input(t);
    }
    let (mut want, mut near) = (mem.clone(), Vec::new());
    for op in 0..5 {
        for t in 0..THREADS as usize {
            let k = word(64 * (op + 1) + t);
            match transcendental(op, t_input(t as u32)) {
                Ok(bits) => want[k] = bits,
                Err(e) => {
                    want[k] = (e as f32).to_bits();
                    near.push((k, e));
                }
            }
        }
    }
    Candidate { class: "transcendental", case: Case { name: "transcendental", code, mem, want, contracted: Vec::new() }, near, races: Vec::new(), names: Vec::new() }
}

/// Lane t's 64-bit inputs: k 0..2 are normal f64 values (2^-30 to 2^30, both signs), k = 3 an
/// integer, all ones high on every third lane so v_mad_u64_u32 carries out.
fn wide_input(t: u32, k: u32) -> u64 {
    let m = (input(t + 64 * k) as u64) << 32 | input(t + 64 * k + 1000) as u64;
    match k {
        3 if t.is_multiple_of(3) => u64::MAX - t as u64,
        3 => m,
        _ => (m & 0x800F_FFFF_FFFF_FFFF) | (993 + (t * 7 + k * 13) as u64 % 61) << 52,
    }
}

/// valu64: f64 add, mul, fma and a mul read by an add (rounded each; a fused form differs on
/// the even lanes, where c = -(a * b)), 64-bit shifts by the thread id, v_mad_u64_u32 with its
/// carry out, and a signed 64-bit compare. Each lane's results are 8 bytes in a region of 512.
fn valu64() -> Candidate {
    let mut a = Asm(Vec::new());
    a.op("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 9), ("SRC0", 131), ("VSRC1", 0)], None);
    for (r, v) in [(0, 20), (1, 22), (2, 24), (3, 26)] {
        a.at(r).op("GLOBAL_LOAD_DWORDX2", "ENC_FLAT_GLBL", &[("VDST", v), ("ADDR", 12), ("SADDR", 0x7D), ("OFFSET", 0)], None);
    }
    a.op("S_WAITCNT", "ENC_SOPP", &[("SIMM16", 0)], None)
        .op("V_ADD_F64", "ENC_VOP3", &[("VDST", 30), ("SRC0", V + 20), ("SRC1", V + 22)], None)
        .op("V_MUL_F64", "ENC_VOP3", &[("VDST", 32), ("SRC0", V + 20), ("SRC1", V + 22)], None)
        .op("V_FMA_F64", "ENC_VOP3", &[("VDST", 34), ("SRC0", V + 20), ("SRC1", V + 22), ("SRC2", V + 24)], None)
        .op("V_ADD_F64", "ENC_VOP3", &[("VDST", 36), ("SRC0", V + 32), ("SRC1", V + 24)], None)
        .op("V_LSHLREV_B64", "ENC_VOP3", &[("VDST", 38), ("SRC0", V), ("SRC1", V + 20)], None)
        .op("V_ASHRREV_I64", "ENC_VOP3", &[("VDST", 40), ("SRC0", V), ("SRC1", V + 22)], None)
        .op("V_MAD_U64_U32", "VOP3_SDST_ENC", &[("VDST", 42), ("SDST", 106), ("SRC0", V + 20), ("SRC1", V + 22), ("SRC2", V + 26)], None)
        .op("V_MOV_B32", "ENC_VOP1", &[("VDST", 47), ("SRC0", 129)], None)
        .op("V_CNDMASK_B32", "ENC_VOP2", &[("VDST", 44), ("SRC0", 128), ("VSRC1", 47)], None)
        .op("V_CMP_LT_I64", "ENC_VOPC", &[("SRC0", V + 20), ("VSRC1", 22)], None)
        .op("V_CNDMASK_B32", "ENC_VOP2", &[("VDST", 45), ("SRC0", 128), ("VSRC1", 47)], None);
    for (r, v) in [(4, 30), (5, 32), (6, 34), (7, 36), (8, 38), (9, 40), (10, 42), (11, 44)] {
        a.at(r).op("GLOBAL_STORE_DWORDX2", "ENC_FLAT_GLBL", &[("ADDR", 12), ("DATA", v), ("SADDR", 0x7D), ("OFFSET", 0)], None);
    }
    let code = a.end();
    let mut mem = vec![0u32; word(12 * 128)];
    let at = |r: usize, t: u32| word(r * 128 + 2 * t as usize);
    let put = |mem: &mut [u32], r: usize, t: u32, v: u64| {
        mem[at(r, t)] = v as u32;
        mem[at(r, t) + 1] = (v >> 32) as u32;
    };
    let f = |t: u32, k: u32| f64::from_bits(wide_input(t, k));
    let c = |t: u32| if t.is_multiple_of(2) { -(f(t, 0) * f(t, 1)) } else { f(t, 2) };
    for t in 0..THREADS {
        put(&mut mem, 0, t, wide_input(t, 0));
        put(&mut mem, 1, t, wide_input(t, 1));
        put(&mut mem, 2, t, c(t).to_bits());
        put(&mut mem, 3, t, wide_input(t, 3));
    }
    let (mut want, mut contracted) = (mem.clone(), Vec::new());
    for t in 0..THREADS {
        let (x, y, z, d) = (f(t, 0), f(t, 1), c(t), wide_input(t, 3));
        let (xb, yb) = (x.to_bits(), y.to_bits());
        let mad = (xb as u32 as u128) * (yb as u32 as u128) + d as u128;
        put(&mut want, 4, t, (x + y).to_bits());
        put(&mut want, 5, t, (x * y).to_bits());
        put(&mut want, 6, t, x.mul_add(y, z).to_bits());
        put(&mut want, 7, t, (x * y + z).to_bits());
        put(&mut want, 8, t, xb << t);
        put(&mut want, 9, t, ((yb as i64) >> t) as u64);
        put(&mut want, 10, t, mad as u64);
        put(&mut want, 11, t, (mad >> 64) as u64 | (((xb as i64) < (yb as i64)) as u64) << 32);
        if x.mul_add(y, z) != x * y + z {
            contracted.extend([at(7, t), at(7, t) + 1]);
        }
    }
    Candidate { class: "valu64", case: Case { name: "valu64", code, mem, want, contracted }, near: Vec::new(), races: Vec::new(), names: Vec::new() }
}

/// cross-lane, for a wave of 32 or 64 lanes: v_readfirstlane (all lanes, then odd lanes only),
/// v_readlane of lanes 5 and the last, v_writelane to lane 3, ds_swizzle (lane ^ 0x11 in each
/// 32), v_permlane16 reversing each row of 16, and ds_bpermute from the next lane round the wave.
fn cross_lane(wave64: bool) -> Candidate {
    let w: u32 = if wave64 { 64 } else { 32 };
    let (saveexec, mov) = if wave64 { ("S_AND_SAVEEXEC_B64", "S_MOV_B64") } else { ("S_AND_SAVEEXEC_B32", "S_MOV_B32") };
    let wait = ("S_WAITCNT", "ENC_SOPP", [("SIMM16", 0u64)]);
    let last = 128 + w as u64 - 1;
    let mut a = Asm(Vec::new());
    a.lane_address()
        .op("GLOBAL_LOAD_DWORD", "ENC_FLAT_GLBL", &[("VDST", 2), ("ADDR", 6), ("SADDR", 0x7D), ("OFFSET", 0)], None)
        .op(wait.0, wait.1, &wait.2, None)
        .op("V_READFIRSTLANE_B32", "ENC_VOP1", &[("VDST", 20), ("SRC0", V + 2)], None)
        .op("V_READLANE_B32", "ENC_VOP3", &[("VDST", 21), ("SRC0", V + 2), ("SRC1", 133)], None)
        .op("V_READLANE_B32", "ENC_VOP3", &[("VDST", 22), ("SRC0", V + 2), ("SRC1", last)], None)
        .op("V_MOV_B32", "ENC_VOP1", &[("VDST", 3), ("SRC0", V)], None)
        .op("V_WRITELANE_B32", "ENC_VOP3", &[("VDST", 3), ("SRC0", 21), ("SRC1", 131)], None)
        .op("V_AND_B32", "ENC_VOP2", &[("VDST", 8), ("SRC0", 129), ("VSRC1", 0)], None)
        .op("V_CMP_NE_U32", "ENC_VOPC", &[("SRC0", 128), ("VSRC1", 8)], None)
        .op(saveexec, "ENC_SOP1", &[("SDST", 24), ("SSRC0", 106)], None)
        .op("V_READFIRSTLANE_B32", "ENC_VOP1", &[("VDST", 23), ("SRC0", V + 2)], None)
        .op(mov, "ENC_SOP1", &[("SDST", 126), ("SSRC0", 24)], None)
        .op("DS_SWIZZLE_B32", "ENC_DS", &[("VDST", 4), ("ADDR", 2), ("OFFSET0", 0x1F), ("OFFSET1", 0x44)], None)
        .op("S_MOV_B32", "ENC_SOP1", &[("SDST", 26), ("SSRC0", 255)], Some(0x89AB_CDEF))
        .op("S_MOV_B32", "ENC_SOP1", &[("SDST", 27), ("SSRC0", 255)], Some(0x0123_4567))
        .op("V_PERMLANE16_B32", "ENC_VOP3", &[("VDST", 5), ("SRC0", V + 2), ("SRC1", 26), ("SRC2", 27)], None)
        .op("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 10), ("SRC0", 129), ("VSRC1", 0)], None)
        .op("V_AND_B32", "ENC_VOP2", &[("VDST", 10), ("SRC0", last), ("VSRC1", 10)], None)
        .op("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 10), ("SRC0", 130), ("VSRC1", 10)], None)
        .op("DS_BPERMUTE_B32", "ENC_DS", &[("VDST", 11), ("ADDR", 10), ("DATA0", 2)], None)
        .op(wait.0, wait.1, &wait.2, None)
        .op("V_MOV_B32", "ENC_VOP1", &[("VDST", 12), ("SRC0", 20)], None)
        .op("V_MOV_B32", "ENC_VOP1", &[("VDST", 13), ("SRC0", 21)], None)
        .op("V_XOR_B32", "ENC_VOP2", &[("VDST", 13), ("SRC0", 22), ("VSRC1", 13)], None)
        .op("V_MOV_B32", "ENC_VOP1", &[("VDST", 14), ("SRC0", 23)], None);
    for (k, r) in [12u64, 13, 3, 14, 4, 5, 11].into_iter().enumerate() {
        a.store(r, 256 * (k as u64 + 1));
    }
    let code = a.end();
    let mut mem = vec![0u32; word(512)];
    for t in 0..THREADS {
        mem[word(t as usize)] = input(t);
    }
    let mut want = mem.clone();
    for t in 0..THREADS {
        let (base, l) = (t & !(w - 1), t & (w - 1));
        let at = |lane: u32| input(base + lane);
        let out = [
            at(0),
            at(5) ^ at(w - 1),
            if l == 3 { at(5) } else { t },
            at(1),
            at((l & 0x20) | ((l & 0x1F) ^ 0x11)),
            at((l & !15) | (15 - (l & 15))),
            at((l + 1) % w),
        ];
        for (k, v) in out.into_iter().enumerate() {
            want[word(64 * (k + 1) + t as usize)] = v;
        }
    }
    Candidate { class: "cross-lane", case: Case { name: "cross-lane", code, mem, want, contracted: Vec::new() }, near: Vec::new(), races: Vec::new(), names: Vec::new() }
}

/// barrier: each thread writes its word to LDS, s_barrier, reads thread 63 - t's (the other
/// wave's in wave32), writes that + t to a second area, s_barrier, reads thread t ^ 32's.
fn barrier() -> Candidate {
    let wait = ("S_WAITCNT", "ENC_SOPP", [("SIMM16", 0u64)]);
    let mut a = Asm(Vec::new());
    a.lane_address()
        .op("GLOBAL_LOAD_DWORD", "ENC_FLAT_GLBL", &[("VDST", 2), ("ADDR", 6), ("SADDR", 0x7D), ("OFFSET", 0)], None)
        .op(wait.0, wait.1, &wait.2, None)
        .op("DS_WRITE_B32", "ENC_DS", &[("ADDR", 1), ("DATA0", 2)], None)
        .op(wait.0, wait.1, &wait.2, None)
        .op("S_BARRIER", "ENC_SOPP", &[], None)
        .op("V_SUB_NC_U32", "ENC_VOP2", &[("VDST", 8), ("SRC0", 255), ("VSRC1", 1)], Some(252))
        .op("DS_READ_B32", "ENC_DS", &[("VDST", 3), ("ADDR", 8)], None)
        .op(wait.0, wait.1, &wait.2, None)
        .op("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 4), ("SRC0", V), ("VSRC1", 3)], None)
        .op("DS_WRITE_B32", "ENC_DS", &[("ADDR", 1), ("DATA0", 4), ("OFFSET1", 1)], None)
        .op(wait.0, wait.1, &wait.2, None)
        .op("S_BARRIER", "ENC_SOPP", &[], None)
        .op("V_XOR_B32", "ENC_VOP2", &[("VDST", 9), ("SRC0", 255), ("VSRC1", 1)], Some(128))
        .op("DS_READ_B32", "ENC_DS", &[("VDST", 5), ("ADDR", 9), ("OFFSET1", 1)], None)
        .op(wait.0, wait.1, &wait.2, None)
        .store(3, 256)
        .store(5, 512);
    let code = a.end();
    let mut mem = vec![0u32; word(192)];
    for t in 0..THREADS {
        mem[word(t as usize)] = input(t);
    }
    let mut want = mem.clone();
    for t in 0..THREADS {
        let u = t ^ 32;
        want[word(64 + t as usize)] = input(63 - t);
        want[word(128 + t as usize)] = input(63 - u).wrapping_add(u);
    }
    Candidate { class: "barrier", case: Case { name: "barrier", code, mem, want, contracted: Vec::new() }, near: Vec::new(), races: Vec::new(), names: Vec::new() }
}

/// The words of a returning atomic all lanes of a group ran on one word: each lane's (word it
/// got back, data, compare) and the words that hold the final value (one, or each lane's copy
/// for LDS). They pass when some order of the lanes explains them all (`serial`).
struct Race {
    what: String,
    op: &'static str,
    wide: bool,
    init: u64,
    lanes: Vec<(usize, u64, u64)>,
    finals: Vec<usize>,
}

impl Race {
    fn words(&self) -> Vec<usize> {
        self.lanes.iter().map(|l| l.0).chain(self.finals.iter().copied()).flat_map(|k| k..k + 1 + self.wide as usize).collect()
    }

    fn check(&self, out: &[u32]) -> Result<(), String> {
        let rd = |k: usize| if self.wide { out[k] as u64 | (out[k + 1] as u64) << 32 } else { out[k] as u64 };
        let last = rd(self.finals[0]);
        if let Some(&k) = self.finals.iter().find(|&&k| rd(k) != last) {
            return Err(format!("{}: lanes read different final words ({last:#x}, {:#x})", self.what, rd(k)));
        }
        let edges: Vec<(u64, u64)> = self.lanes.iter().map(|&(k, d, c)| (rd(k), rmw(self.op, self.wide, rd(k), d, c))).collect();
        serial(self.init, last, &edges).map_err(|e| format!("{}: {e}", self.what))
    }
}

/// Whether some order of the lanes explains what each got back and the final word `last`, from
/// `init`: lane i read `edges[i].0` and wrote `edges[i].1`. Such an order is a path from `init`
/// to `last` taking each lane once (an Euler path). It exists exactly when every value is read
/// (got back, or left as the final word) as often as it is written (by a lane, or as `init`),
/// and every lane's values connect to `init`.
fn serial(init: u64, last: u64, edges: &[(u64, u64)]) -> Result<(), String> {
    let all: Vec<(u64, u64)> = edges.iter().copied().chain([(last, init)]).collect();
    let mut count: BTreeMap<u64, (usize, usize)> = BTreeMap::new();
    for &(r, w) in &all {
        count.entry(r).or_default().0 += 1;
        count.entry(w).or_default().1 += 1;
    }
    if let Some((v, (r, w))) = count.iter().find(|c| c.1 .0 != c.1 .1) {
        return Err(format!("{v:#x} is read {r} time(s) but written {w} time(s)"));
    }
    let mut reach = BTreeSet::from([init]);
    while let Some(&(r, w)) = all.iter().find(|e| reach.contains(&e.0) != reach.contains(&e.1)) {
        reach.extend([r, w]);
    }
    match edges.iter().find(|e| !reach.contains(&e.0)) {
        Some(&(r, _)) => Err(format!("{r:#x} is read in a loop of lanes no order from {init:#x} reaches")),
        None => Ok(()),
    }
}

/// RDNA2's new value of an atomic `op` on `old`, with the lane's `data` (and `cmp` for
/// cmpswap), in 32 bits, or 64 with `wide`. inc and dec wrap at `data`.
fn rmw(op: &str, wide: bool, old: u64, data: u64, cmp: u64) -> u64 {
    let s = |x: u64| if wide { x as i64 } else { x as u32 as i32 as i64 };
    let v = match op {
        "add" => old.wrapping_add(data),
        "sub" => old.wrapping_sub(data),
        "and" => old & data,
        "or" => old | data,
        "xor" => old ^ data,
        "umin" => old.min(data),
        "umax" => old.max(data),
        "smin" if s(data) < s(old) => data,
        "smax" if s(data) > s(old) => data,
        "smin" | "smax" => old,
        "inc" if old >= data => 0,
        "inc" => old + 1,
        "dec" if old == 0 || old > data => data,
        "dec" => old - 1,
        "swap" => data,
        "cmpswap" if old == cmp => data,
        "cmpswap" => old,
        _ => unreachable!("{op}"),
    };
    if wide { v } else { v & 0xFFFF_FFFF }
}

/// Where an atomic candidate's atomics go.
#[derive(Clone, Copy, PartialEq)]
enum Where {
    Lds,
    Buffer,
    Global,
}

/// `Shared`: every lane of a group (the 64 threads, or each 32-lane half in LDS) on one word,
/// from `init`. `Own`: lane t on its own word, which holds `init` on every third lane and 0
/// elsewhere. `Oob` (buffers): every lane past num_records, the odd ones just past it, the even
/// ones by an offset whose sum with the instruction's wraps past 2^32 to the trap word.
#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Shared,
    Own,
    Oob,
}

/// One atomic of an atomic candidate: `op` as in buffer / global names, `ds` its DS name
/// without DS_ and _RTN, whether it returns the old value (GLC / _RTN), and lane t's operands.
#[derive(Clone, Copy)]
struct Slot {
    op: &'static str,
    ds: &'static str,
    kind: Kind,
    rtn: bool,
    init: u64,
    data: fn(u32) -> u64,
    cmp: fn(u32) -> u64,
}

/// The 32-bit atomics: each commutative op once with its result read back as the final word
/// (exact), the returning ones as races; inc / dec share one wrap value so lane order cannot
/// change them; swap and the cmpswap contest (one winner) are races too. The 64-bit ones: a
/// +1 counter across the carry into the high word, and umax deciding on both halves.
fn slots(wide: bool) -> Vec<Slot> {
    use Kind::*;
    let s = |op, ds, kind, rtn, init, data: fn(u32) -> u64, cmp: fn(u32) -> u64| Slot { op, ds, kind, rtn, init, data, cmp };
    let n: fn(u32) -> u64 = |_| 0;
    if wide {
        return vec![
            s("add", "ADD_U64", Shared, true, (1 << 32) - 20, |_| 1, n),
            s("umax", "MAX_U64", Shared, false, 0, |t| (input(t) as u64) << 32 | input(t + 64) as u64, n),
        ];
    }
    vec![
        s("add", "ADD_U32", Shared, true, 0xFFFF_FFF0, |_| 1, n),
        s("add", "ADD_U32", Shared, false, 0x1000, |t| input(t) as u64, n),
        s("sub", "SUB_U32", Shared, true, 0, |t| (input(t) >> 4) as u64, n),
        s("and", "AND_B32", Shared, false, 0xFFFF_FFFF, |t| !(1u32 << (t % 29)) as u64, n),
        s("or", "OR_B32", Shared, true, 0, |t| 1 << (t % 32), n),
        s("xor", "XOR_B32", Shared, false, 0x5A5A_5A5A, |t| input(t) as u64, n),
        s("umin", "MIN_U32", Shared, true, 0xFFFF_FFFF, |t| input(t) as u64, n),
        s("umax", "MAX_U32", Shared, false, 0, |t| input(t) as u64, n),
        s("smin", "MIN_I32", Shared, false, 0x7FFF_FFFF, |t| input(t) as u64, n),
        s("smax", "MAX_I32", Shared, true, 0x8000_0000, |t| input(t) as u64, n),
        s("inc", "INC_U32", Shared, true, 7, |_| 20, n),
        s("dec", "DEC_U32", Shared, false, 3, |_| 20, n),
        s("swap", "WRXCHG_B32", Shared, true, 0xFFFF_0000, |t| 0x100 + t as u64, n),
        s("cmpswap", "CMPST_B32", Shared, true, 5, |t| 0x200 + t as u64, |_| 5),
        s("cmpswap", "CMPST_B32", Own, true, 0x55, |t| t as u64 + 1, n),
        s("add", "ADD_U32", Oob, true, 0, |_| 1, n),
    ]
}

/// Words per slot from BASE: lane t's data at DATA + t (2 t for 64-bit), its compare at CMP + t
/// (an Oob slot's: its buffer offset), what it got back at RET, the final word it read (LDS) at
/// FIN. Then the shared words (4 apart), the own words, the trap word and the word past the end.
const BLOCK: usize = 448;
const DATA: usize = 0;
const CMP: usize = 128;
const RET: usize = 192;
const FIN: usize = 320;
const LDS_OWN: u64 = 512;

impl Asm {
    /// v12:v13 = BASE + 4 k + lane t's 4 t (or 8 t with `wide`, v9).
    fn lane_at(&mut self, k: usize, wide: bool) -> &mut Self {
        self.op("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 12), ("SRC0", 255), ("VSRC1", if wide { 9 } else { 1 })], Some(BASE + 4 * k as u32))
            .op("V_MOV_B32", "ENC_VOP1", &[("VDST", 13), ("SRC0", 128)], None)
    }

    fn global(&mut self, name: &str, v: u64, k: usize, wide: bool) -> &mut Self {
        let reg = if name.contains("LOAD") { "VDST" } else { "DATA" };
        self.lane_at(k, wide).op(name, "ENC_FLAT_GLBL", &[(reg, v), ("ADDR", 12), ("SADDR", 0x7D)], None)
    }
}

/// An atomic candidate: the 32-bit (or 64-bit, `wide`) slots on LDS, a raw buffer or global
/// memory, each lane's operands loaded from memory, what it got back stored. LDS words are set
/// first by the wave that uses them (a group per 32-lane half, so no barrier is needed in
/// wave32), and read back at the end. The reference applies the lanes in `order`; the races and
/// the exact words hold for every order.
fn atomics(at: Where, wide: bool, order: &[u32]) -> Candidate {
    let slots: Vec<Slot> = slots(wide).into_iter().filter(|s| s.kind != Kind::Oob || at == Where::Buffer).collect();
    let (w, n) = (1 + wide as usize, slots.len());
    let (cells, own, trap, end) = (BLOCK * n, BLOCK * n + 64, BLOCK * n + 128, BLOCK * n + 132);
    let wait = ("S_WAITCNT", "ENC_SOPP", [("SIMM16", 0u64)]);
    let (load, store) = if wide { ("GLOBAL_LOAD_DWORDX2", "GLOBAL_STORE_DWORDX2") } else { ("GLOBAL_LOAD_DWORD", "GLOBAL_STORE_DWORD") };
    let mut a = Asm(Vec::new());
    a.lane_address().op("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 9), ("SRC0", 131), ("VSRC1", 0)], None);
    if at == Where::Lds {
        // v28 = 8 (t >> 5): each half's words, 16 bytes per slot
        a.op("V_LSHRREV_B32", "ENC_VOP2", &[("VDST", 28), ("SRC0", 133), ("VSRC1", 0)], None)
            .op("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 28), ("SRC0", 131), ("VSRC1", 28)], None);
        for (j, s) in slots.iter().enumerate() {
            if s.kind == Kind::Own {
                a.global("GLOBAL_LOAD_DWORD", 26, own, false).op(wait.0, wait.1, &wait.2, None);
                a.op("DS_WRITE_B32", "ENC_DS", &[("ADDR", 1), ("DATA0", 26), ("OFFSET0", LDS_OWN & 0xFF), ("OFFSET1", LDS_OWN >> 8)], None);
                continue;
            }
            a.op("V_MOV_B32", "ENC_VOP1", &[("VDST", 26), ("SRC0", 255)], Some(s.init as u32));
            if wide {
                a.op("V_MOV_B32", "ENC_VOP1", &[("VDST", 27), ("SRC0", 255)], Some((s.init >> 32) as u32));
            }
            a.op(if wide { "DS_WRITE_B64" } else { "DS_WRITE_B32" }, "ENC_DS", &[("ADDR", 28), ("DATA0", 26), ("OFFSET0", 16 * j as u64)], None);
        }
        a.op(wait.0, wait.1, &wait.2, None);
    }
    // a buffer's offsets: from the shared words (a raw V# there, OOB_SELECT 3, num_records up
    // to the word past the end); global addresses: from 0
    let byte = |k: usize| if at == Where::Buffer { 4 * (k - cells) as u32 } else { BASE + 4 * k as u32 };
    let oob = byte(trap) as u64 + 8;
    if at == Where::Buffer {
        a.op("S_ADD_U32", "ENC_SOP2", &[("SDST", 12), ("SSRC0", 0), ("SSRC1", 255)], Some(4 * cells as u32))
            .op("S_MOV_B32", "ENC_SOP1", &[("SDST", 13), ("SSRC0", 128)], None)
            .op("S_MOV_B32", "ENC_SOP1", &[("SDST", 14), ("SSRC0", 255)], Some(byte(end)))
            .op("S_MOV_B32", "ENC_SOP1", &[("SDST", 15), ("SSRC0", 255)], Some(3 << 28));
    }
    for (j, s) in slots.iter().enumerate() {
        let b = BLOCK * j;
        a.global(load, 20, b + DATA, wide);
        // the compare value, or an Oob slot's offset
        for (on, v) in [(s.op == "cmpswap", 21), (s.kind == Kind::Oob, 24)] {
            if on {
                a.global("GLOBAL_LOAD_DWORD", v, b + CMP, false);
            }
        }
        a.op(wait.0, wait.1, &wait.2, None);
        let x2 = if wide { "_X2" } else { "" };
        let up = s.op.to_uppercase();
        match (at, s.kind) {
            (Where::Lds, _) => {
                let (name, ty) = s.ds.rsplit_once('_').unwrap();
                let name = format!("DS_{name}{}_{ty}", if s.rtn { "_RTN" } else { "" });
                let (addr, off) = if s.kind == Kind::Own { (1, LDS_OWN) } else { (28, 16 * j as u64) };
                let (d0, d1) = if s.op == "cmpswap" { (21, 20) } else { (20, 0) };
                a.op(&name, "ENC_DS", &[("ADDR", addr), ("DATA0", d0), ("DATA1", d1), ("VDST", 22), ("OFFSET0", off & 0xFF), ("OFFSET1", off >> 8)], None);
            }
            (_, Kind::Oob) => {}
            (_, Kind::Shared) => {
                a.op("V_MOV_B32", "ENC_VOP1", &[("VDST", 24), ("SRC0", 255)], Some(byte(cells + 4 * j)));
            }
            (_, Kind::Own) => {
                a.op("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 24), ("SRC0", 255), ("VSRC1", 1)], Some(byte(own)));
            }
        }
        if at == Where::Global {
            a.op("V_MOV_B32", "ENC_VOP1", &[("VDST", 25), ("SRC0", 128)], None)
                .op(&format!("GLOBAL_ATOMIC_{up}{x2}"), "ENC_FLAT_GLBL", &[("ADDR", 24), ("DATA", 20), ("VDST", 22), ("SADDR", 0x7D), ("GLC", s.rtn as u64)], None);
        }
        if at == Where::Buffer {
            let off = if s.kind == Kind::Oob { oob } else { 0 };
            a.op(&format!("BUFFER_ATOMIC_{up}{x2}"), "ENC_MUBUF", &[("VDATA", 20), ("VADDR", 24), ("SRSRC", 3), ("SOFFSET", 128), ("OFFEN", 1), ("OFFSET", off), ("GLC", s.rtn as u64)], None);
        }
        if s.rtn {
            // a buffer atomic returns into its data registers
            a.op(wait.0, wait.1, &wait.2, None).global(store, if at == Where::Buffer { 20 } else { 22 }, b + RET, wide);
        }
    }
    if at == Where::Lds {
        a.op(wait.0, wait.1, &wait.2, None);
        for (j, s) in slots.iter().enumerate() {
            let (addr, off, read) = match s.kind {
                Kind::Own => (1, LDS_OWN, "DS_READ_B32"),
                _ => (28, 16 * j as u64, if wide { "DS_READ_B64" } else { "DS_READ_B32" }),
            };
            a.op(read, "ENC_DS", &[("VDST", 22), ("ADDR", addr), ("OFFSET0", off & 0xFF), ("OFFSET1", off >> 8)], None)
                .op(wait.0, wait.1, &wait.2, None)
                .global(store, 22, BLOCK * j + FIN, wide);
        }
    }
    let code = a.end();

    let mut mem = vec![0u32; word(end + 1)];
    let put = |m: &mut [u32], k: usize, v: u64| {
        m[word(k)] = v as u32;
        if wide {
            m[word(k) + 1] = (v >> 32) as u32;
        }
    };
    let own_init = |s: &Slot, t: u32| if t.is_multiple_of(3) { s.init } else { 0 };
    for (j, s) in slots.iter().enumerate() {
        for t in 0..THREADS {
            put(&mut mem, BLOCK * j + DATA + w * t as usize, (s.data)(t));
            mem[word(BLOCK * j + CMP + t as usize)] = match s.kind {
                Kind::Oob if t.is_multiple_of(2) => byte(trap).wrapping_sub(oob as u32),
                Kind::Oob => byte(end) - oob as u32,
                _ => (s.cmp)(t) as u32,
            };
            if s.kind == Kind::Own {
                mem[word(own + t as usize)] = own_init(s, t) as u32;
            }
        }
        if s.kind == Kind::Shared && at != Where::Lds {
            put(&mut mem, cells + 4 * j, s.init);
        }
    }
    let mut want = mem.clone();
    let (mut races, mut names) = (Vec::new(), Vec::new());
    let groups: Vec<Vec<u32>> = match at {
        Where::Lds => vec![order.iter().copied().filter(|&t| t < 32).collect(), order.iter().copied().filter(|&t| t >= 32).collect()],
        _ => vec![order.to_vec()],
    };
    for (j, s) in slots.iter().enumerate() {
        let b = BLOCK * j;
        let label = format!("{j}:{}{}{}", s.op, if wide { "64" } else { "" }, match s.kind {
            Kind::Shared if s.rtn => " rtn",
            Kind::Shared => "",
            Kind::Own => " own",
            Kind::Oob => " oob",
        });
        names.push((word(b + RET), word(b + RET + 64 * w), format!("{label} ret")));
        names.push((word(b + FIN), word(b + FIN + 64 * w), format!("{label} fin")));
        names.push((word(cells + 4 * j), word(cells + 4 * j + w), format!("{label} word")));
        match s.kind {
            Kind::Shared => {
                for g in &groups {
                    let mut v = s.init;
                    for &t in g {
                        if s.rtn {
                            put(&mut want, b + RET + w * t as usize, v);
                        }
                        v = rmw(s.op, wide, v, (s.data)(t), (s.cmp)(t));
                    }
                    let finals: Vec<usize> = if at == Where::Lds { g.iter().map(|&t| b + FIN + w * t as usize).collect() } else { vec![cells + 4 * j] };
                    finals.iter().for_each(|&k| put(&mut want, k, v));
                    if s.rtn {
                        let half = if at == Where::Lds { format!(" (lanes {}-{})", g.iter().min().unwrap(), g.iter().max().unwrap()) } else { String::new() };
                        let lanes = g.iter().map(|&t| (word(b + RET + w * t as usize), (s.data)(t), (s.cmp)(t))).collect();
                        let finals = finals.into_iter().map(word).collect();
                        races.push(Race { what: format!("{label} returns{half}"), op: s.op, wide, init: s.init, lanes, finals });
                    }
                }
            }
            Kind::Own => {
                for t in 0..THREADS {
                    let old = own_init(s, t);
                    let new = rmw(s.op, false, old, (s.data)(t), (s.cmp)(t)) as u32;
                    want[word(b + RET + t as usize)] = old as u32;
                    want[word(if at == Where::Lds { b + FIN } else { own } + t as usize)] = new;
                }
            }
            // out of range: nothing written, 0 back
            Kind::Oob => {}
        }
    }
    names.extend([(word(own), word(own + 64), "own words".into()), (word(trap), word(trap + 1), "oob trap".into()), (word(end), word(end + 1), "oob past end".into())]);
    let class = ["lds-atomic", "buffer-atomic", "global-atomic"][at as usize];
    let name = ["lds-atomic", "buffer-atomic", "global-atomic", "lds-atomic64", "buffer-atomic64", "global-atomic64"][at as usize + 3 * w - 3];
    Candidate { class, case: Case { name, code, mem, want, contracted: Vec::new() }, near: Vec::new(), races, names }
}

fn candidates(wave64: bool) -> Vec<Candidate> {
    let mut all = vec![transcendental_ops(), valu64(), cross_lane(wave64), barrier()];
    let order: Vec<u32> = (0..THREADS).collect();
    for at in [Where::Lds, Where::Buffer, Where::Global] {
        all.extend([atomics(at, false, &order), atomics(at, true, &order)]);
    }
    all
}

/// A console MS of one group of 64 threads with a position and one param.
fn mesh_interface() -> MsInterface {
    let sig = |name: &str, sv: u32, reg: u32| crate::gpu::dxbc::SigElement {
        name: name.into(),
        index: 0,
        system_value: sv,
        component_type: 3,
        register: reg,
        mask: 0xF,
        rw_mask: 0xF,
        stream: 0,
    };
    MsInterface { outputs: vec![sig("SV_Position", 1, 0), sig("TEXCOORD", 0, 1)], threads: THREADS, max_vertices: THREADS, max_primitives: THREADS, ..Default::default() }
}

/// Primitive 0: vertices 0, 1, 2.
const PRIM: u32 = 1 << 10 | 2 << 20;

/// mesh-alloc and mesh-export in the console's style: the thread in the group from the wave
/// index (s3[27:24]) and v_mbcnt, the group from v5; GS_ALLOC_REQ of 3 vertices and 1
/// primitive, every thread's position (t / 2 - 1 + group, 2 (t & 1) - 1, 0, 1) and param
/// (group, t, 0.5, 1), and thread 0's primitive. Every f32 is exact, so no contraction changes
/// one; the groups' triangles do not overlap.
fn mesh_program(wave64: bool) -> Vec<u8> {
    let exp = |tgt: u64, en: u64, v: [u64; 4]| [("TGT", tgt), ("EN", en), ("VSRC0", v[0]), ("VSRC1", v[1]), ("VSRC2", v[2]), ("VSRC3", v[3])];
    let mut a = Asm(Vec::new());
    a.op("S_BFE_U32", "ENC_SOP2", &[("SDST", 0), ("SSRC0", 3), ("SSRC1", 255)], Some(0x40018))
        .op("V_MBCNT_LO_U32_B32", "ENC_VOP3", &[("VDST", 0), ("SRC0", 193), ("SRC1", 128)], None);
    if wave64 {
        a.op("V_MBCNT_HI_U32_B32", "ENC_VOP3", &[("VDST", 0), ("SRC0", 193), ("SRC1", V)], None);
    }
    a.op("V_LSHL_ADD_U32", "ENC_VOP3", &[("VDST", 0), ("SRC0", 0), ("SRC1", if wave64 { 134 } else { 133 }), ("SRC2", V)], None)
        .op("V_LSHRREV_B32", "ENC_VOP2", &[("VDST", 2), ("SRC0", 134), ("VSRC1", 5)], None) // group = v5 >> 6
        .op("V_CVT_F32_U32", "ENC_VOP1", &[("VDST", 3), ("SRC0", V + 2)], None)
        .op("V_CVT_F32_U32", "ENC_VOP1", &[("VDST", 1), ("SRC0", V)], None)
        .op("V_MUL_F32", "ENC_VOP2", &[("VDST", 10), ("SRC0", 240), ("VSRC1", 1)], None) // 0.5 *
        .op("V_ADD_F32", "ENC_VOP2", &[("VDST", 10), ("SRC0", 243), ("VSRC1", 10)], None) // -1.0 +
        .op("V_ADD_F32", "ENC_VOP2", &[("VDST", 10), ("SRC0", V + 3), ("VSRC1", 10)], None) // group +
        .op("V_AND_B32", "ENC_VOP2", &[("VDST", 6), ("SRC0", 129), ("VSRC1", 0)], None)
        .op("V_CVT_F32_U32", "ENC_VOP1", &[("VDST", 11), ("SRC0", V + 6)], None)
        .op("V_MUL_F32", "ENC_VOP2", &[("VDST", 11), ("SRC0", 244), ("VSRC1", 11)], None) // 2.0 *
        .op("V_ADD_F32", "ENC_VOP2", &[("VDST", 11), ("SRC0", 243), ("VSRC1", 11)], None)
        .op("V_MOV_B32", "ENC_VOP1", &[("VDST", 12), ("SRC0", 128)], None)
        .op("V_MOV_B32", "ENC_VOP1", &[("VDST", 13), ("SRC0", 242)], None)
        .op("V_MOV_B32", "ENC_VOP1", &[("VDST", 14), ("SRC0", 240)], None)
        .op("S_MOV_B32", "ENC_SOP1", &[("SDST", 124), ("SSRC0", 255)], Some(1 << 12 | 3))
        .op("S_SENDMSG", "ENC_SOPP", &[("SIMM16", 9)], None)
        .op("EXP", "ENC_EXP", &exp(12, 0xF, [10, 11, 12, 13]), None)
        .op("EXP", "ENC_EXP", &exp(32, 0xF, [3, 1, 14, 13]), None)
        .op("V_CMPX_EQ_U32", "ENC_VOPC", &[("SRC0", 128), ("VSRC1", 0)], None) // thread 0
        .op("V_MOV_B32", "ENC_VOP1", &[("VDST", 15), ("SRC0", 255)], Some(PRIM))
        .op("EXP", "ENC_EXP", &exp(20, 1, [15, 0, 0, 0]), None);
    a.end()
}

/// The words of group `g`'s record that `mesh_program` writes on RDNA2: (dword, value).
fn mesh_record(ms: &MsInterface, g: u32) -> Vec<(usize, u32)> {
    let header = crate::gpu::recomp::MESH_HEADER;
    let vd = ms.vert_dwords();
    let mut w = vec![(0, 3), (1, 1), (header + THREADS as usize * vd, PRIM)];
    for t in 0..THREADS {
        let v = header + t as usize * vd;
        let pos = [t as f32 * 0.5 - 1.0 + g as f32, (t & 1) as f32 * 2.0 - 1.0, 0.0, 1.0];
        let param = [g as f32, t as f32, 0.5, 1.0];
        w.extend(pos.iter().chain(&param).enumerate().map(|(c, x)| (v + c, x.to_bits())));
    }
    w
}

/// Primitives of the geometry pass's test: a full group of 21 and one of 4.
const GS_PRIMS: u32 = 25;
/// What the ES's last ESGS dword holds besides the vertex id (`v5 ^ GS_MARK`).
const GS_MARK: u32 = 0x5A5A;

/// The console ES + GS program of the geometry pass's test and its interface: one ES output
/// register (16-byte ESGS items), a float4 position and a uint4 TEXCOORD out, 4 vertices at
/// most per primitive. ES: each lane's item = (v5, 7 v5 + 1, lane, v5 ^ GS_MARK), written
/// with ds_write_b64 + ds_write2_b32, then s_barrier and the jump into the GS. GS: s_barrier,
/// EXEC = the group's primitives (s3[15:8]), the triangle's items read through v0 / v1,
/// four vertices per lane into its LDS output area (s2 + lane * 128) with ds_write_b128 +
/// ds_write2_b32, each emitted, a cut after the third on odd lanes, then GS_DONE. Vertex k:
/// position (2 (k >> 1) - 1, 2 (k & 1) - 1, 0, 1) (exact f32 moves), TEXCOORD (3 p,
/// k, 7 (3 p + 1) + 1, (3 lane + 2) + ((3 p + 2) ^ GS_MARK)) for primitive p.
fn geometry_program() -> (Vec<u8>, GsInterface) {
    let sig = |name: &str, sv: u32, reg: u32, ty: u32| crate::gpu::dxbc::SigElement { name: name.into(), index: 0, system_value: sv, component_type: ty, register: reg, mask: 0xF, rw_mask: 0xF, stream: 0 };
    let mut a = Asm(Vec::new());
    let lane = |a: &mut Asm| {
        a.op("V_MBCNT_LO_U32_B32", "ENC_VOP3", &[("VDST", 20), ("SRC0", 193), ("SRC1", 128)], None)
            .op("V_MBCNT_HI_U32_B32", "ENC_VOP3", &[("VDST", 20), ("SRC0", 193), ("SRC1", V + 20)], None);
    };
    // ES
    lane(&mut a);
    a.op("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 21), ("SRC0", 132), ("VSRC1", 20)], None) // lane * 16
        .op("V_MOV_B32", "ENC_VOP1", &[("VDST", 22), ("SRC0", V + 5)], None)
        .op("V_MAD_U32_U24", "ENC_VOP3", &[("VDST", 23), ("SRC0", V + 5), ("SRC1", 135), ("SRC2", 129)], None) // 7 v5 + 1
        .op("V_MOV_B32", "ENC_VOP1", &[("VDST", 24), ("SRC0", V + 20)], None)
        .op("V_XOR_B32", "ENC_VOP2", &[("VDST", 25), ("SRC0", 255), ("VSRC1", 5)], Some(GS_MARK))
        .op("DS_WRITE_B64", "ENC_DS", &[("ADDR", 21), ("DATA0", 22)], None)
        .op("DS_WRITE2_B32", "ENC_DS", &[("ADDR", 21), ("DATA0", 24), ("DATA1", 25), ("OFFSET0", 2), ("OFFSET1", 3)], None)
        .op("S_WAITCNT", "ENC_SOPP", &[], None)
        .op("S_BARRIER", "ENC_SOPP", &[], None)
        .op("S_SETPC_B64", "ENC_SOP1", &[("SSRC0", 6)], None);
    let gs_start = a.here() as u64;
    // GS
    a.op("S_BARRIER", "ENC_SOPP", &[], None)
        .op("S_BFE_U32", "ENC_SOP2", &[("SDST", 0), ("SSRC0", 3), ("SSRC1", 255)], Some(0x80008)) // primitives
        .op("S_BFM_B64", "ENC_SOP2", &[("SDST", 126), ("SSRC0", 0), ("SSRC1", 128)], None);
    lane(&mut a);
    a.op("V_AND_B32", "ENC_VOP2", &[("VDST", 30), ("SRC0", 255), ("VSRC1", 0)], Some(0xFFFF))
        .op("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 30), ("SRC0", 130), ("VSRC1", 30)], None)
        .op("V_LSHRREV_B32", "ENC_VOP2", &[("VDST", 31), ("SRC0", 144), ("VSRC1", 0)], None)
        .op("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 31), ("SRC0", 130), ("VSRC1", 31)], None)
        .op("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 32), ("SRC0", 130), ("VSRC1", 1)], None)
        .op("DS_READ_B32", "ENC_DS", &[("ADDR", 30), ("VDST", 40)], None)
        .op("DS_READ_B32", "ENC_DS", &[("ADDR", 31), ("VDST", 41), ("OFFSET0", 4)], None)
        .op("DS_READ2_B32", "ENC_DS", &[("ADDR", 32), ("VDST", 42), ("OFFSET0", 2), ("OFFSET1", 3)], None)
        .op("S_WAITCNT", "ENC_SOPP", &[], None)
        .op("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 43), ("SRC0", V + 42), ("VSRC1", 43)], None)
        .op("V_MAD_U32_U24", "ENC_VOP3", &[("VDST", 50), ("SRC0", V + 20), ("SRC1", 255), ("SRC2", 2)], Some(128)) // s2 + lane * 128
        .op("V_MOV_B32", "ENC_VOP1", &[("VDST", 62), ("SRC0", 128)], None)
        .op("V_MOV_B32", "ENC_VOP1", &[("VDST", 63), ("SRC0", 242)], None)
        .op("V_MOV_B32", "ENC_VOP1", &[("VDST", 64), ("SRC0", V + 40)], None)
        .op("V_MOV_B32", "ENC_VOP1", &[("VDST", 66), ("SRC0", V + 41)], None)
        .op("V_MOV_B32", "ENC_VOP1", &[("VDST", 67), ("SRC0", V + 43)], None);
    for k in 0..4u64 {
        let one = |b: bool| if b { 242 } else { 243 }; // 1.0 / -1.0
        a.op("V_MOV_B32", "ENC_VOP1", &[("VDST", 60), ("SRC0", one(k >> 1 == 1))], None)
            .op("V_MOV_B32", "ENC_VOP1", &[("VDST", 61), ("SRC0", one(k & 1 == 1))], None)
            .op("V_MOV_B32", "ENC_VOP1", &[("VDST", 65), ("SRC0", 128 + k)], None)
            .op("DS_WRITE_B128", "ENC_DS", &[("ADDR", 50), ("DATA0", 60)], None)
            .op("DS_WRITE2_B32", "ENC_DS", &[("ADDR", 50), ("DATA0", 64), ("DATA1", 65), ("OFFSET0", 4), ("OFFSET1", 5)], None)
            .op("DS_WRITE2_B32", "ENC_DS", &[("ADDR", 50), ("DATA0", 66), ("DATA1", 67), ("OFFSET0", 6), ("OFFSET1", 7)], None)
            .op("S_SENDMSG", "ENC_SOPP", &[("SIMM16", 0x22)], None) // emit
            .op("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 50), ("SRC0", 160), ("VSRC1", 50)], None);
        if k == 2 {
            // odd lanes: cut
            a.op("V_AND_B32", "ENC_VOP2", &[("VDST", 70), ("SRC0", 129), ("VSRC1", 20)], None)
                .op("V_CMP_NE_U32", "ENC_VOPC", &[("SRC0", 128), ("VSRC1", 70)], None)
                .op("S_AND_SAVEEXEC_B64", "ENC_SOP1", &[("SDST", 8), ("SSRC0", 106)], None)
                .op("S_SENDMSG", "ENC_SOPP", &[("SIMM16", 0x12)], None)
                .op("S_MOV_B64", "ENC_SOP1", &[("SDST", 126), ("SSRC0", 8)], None);
        }
    }
    a.op("S_SENDMSG", "ENC_SOPP", &[("SIMM16", 0x03)], None); // GS_DONE
    let g = GsInterface {
        es_inputs: vec![],
        es_outputs: vec![sig("SV_Position", 1, 0, 3)],
        outputs: vec![sig("SV_Position", 1, 0, 3), sig("TEXCOORD", 0, 1, 1)],
        max_vertex_count: 4,
        gs_start: Some(gs_start),
    };
    (a.end(), g)
}

/// The words of group `grp`'s record that `geometry_program` writes on RDNA2 for `GS_PRIMS`
/// primitives: (dword, value).
fn geometry_record(g: &GsInterface, grp: u32) -> Vec<(usize, u32)> {
    let header = crate::gpu::recomp::MESH_HEADER;
    let rec = g.record();
    let (vd, maxv) = (rec.vert_dwords(), rec.max_vertices as usize);
    let np = (GS_PRIMS - 21 * grp).min(21);
    let mut w = Vec::new();
    let mut tris = 0usize;
    for l in 0..np {
        let p = 21 * grp + l;
        let vb = 4 * l;
        for k in 0..4u32 {
            let v = header + (vb + k) as usize * vd;
            let pos = [2.0 * (k >> 1) as f32 - 1.0, 2.0 * (k & 1) as f32 - 1.0, 0.0, 1.0f32].map(f32::to_bits);
            let tex = [3 * p, k, 7 * (3 * p + 1) + 1, (3 * l + 2) + ((3 * p + 2) ^ GS_MARK)];
            w.extend(pos.iter().chain(&tex).enumerate().map(|(c, &x)| (v + c, x)));
        }
        let prim = header + maxv * vd + tris;
        w.push((prim, vb | (vb + 1) << 10 | (vb + 2) << 20));
        tris += 1;
        if l % 2 == 0 {
            // the strip's second triangle, its first two vertices swapped; odd lanes cut before
            w.push((prim + 1, (vb + 2) | (vb + 1) << 10 | (vb + 3) << 20));
            tris += 1;
        }
    }
    w.extend([(0, 4 * np), (1, tris as u32)]);
    w
}

/// The classes `classify` gives the instructions `covered::check` looks at in `code`.
fn classes(code: &[u8], classify: fn(&rdna2::Decoded) -> &'static str) -> BTreeSet<&'static str> {
    classes_in(&Cfg::build(code).unwrap_or_else(|e| panic!("{e}")), classify)
}

/// `classes` over a CFG built already (the geometry pass's, `Cfg::build_geometry`).
fn classes_in(cfg: &Cfg, classify: fn(&rdna2::Decoded) -> &'static str) -> BTreeSet<&'static str> {
    let mut out = BTreeSet::new();
    for &b in &cfg.rpo {
        let block = &cfg.blocks[b];
        let branch = if let Term::Cond(i, _, _) = block.term { Some(i) } else { None };
        out.extend(block.insts.iter().chain(branch.iter()).map(|&i| classify(&cfg.insts[i])));
    }
    out
}

/// `Asm`'s branches: program `int` loops back to its first LCG step and skips one add when
/// no lane is odd.
#[test]
fn branches_land_on_their_labels() {
    for (wave64, restore) in [(false, "S_MOV_B32"), (true, "S_MOV_B64")] {
        let code = int(wave64).code;
        let mut targets = Vec::new();
        let mut pc = 0u64;
        while (pc as usize) < code.len() {
            let d = rdna2::decode(&code[pc as usize..], pc).unwrap();
            if d.inst.name.starts_with("S_CBRANCH") {
                let simm = rdna2::field_value(d.enc, &d.words, "SIMM16").unwrap() as u16 as i16;
                let to = (pc as i64 + 4 + 4 * simm as i64) as u64;
                targets.push((d.inst.name, rdna2::decode(&code[to as usize..], to).unwrap().inst.name));
            }
            pc += d.len as u64;
        }
        assert_eq!(targets, [("S_CBRANCH_SCC1", "V_MUL_LO_U32"), ("S_CBRANCH_EXECZ", restore)], "wave64 {wave64}");
    }
}

#[test]
fn every_program_is_in_the_covered_set() {
    for wave64 in [false, true] {
        for c in cases(wave64) {
            assert_eq!(covered::check(&c.code), Ok(()), "{} (wave64 {wave64})", c.name);
        }
    }
}

/// A class `TABLE` marks covered needs a program here that the GPU test runs.
#[test]
fn every_covered_class_has_a_program() {
    let mut seen = BTreeSet::new();
    for wave64 in [false, true] {
        for c in cases(wave64) {
            seen.extend(classes(&c.code, covered::class));
        }
    }
    let missing: Vec<&str> = covered::TABLE.iter().filter(|r| r.1 && !seen.contains(r.0)).map(|r| r.0).collect();
    assert!(missing.is_empty(), "covered classes without a program in covered_tests: {missing:?}");
}

#[test]
fn the_reference_has_the_cases_it_tests() {
    let all = cases(true);
    // the f32 case has lanes where a fused mul + add differs, so the GPU test can tell
    assert!(all.iter().any(|c| !c.contracted.is_empty()));
    // each case writes something
    assert!(all.iter().all(|c| c.want != c.mem), "a case leaves memory as it was");
}

/// A candidate holds its own class and covered ones only, and stays outside the covered set:
/// once `TABLE` covers its class, its program moves to `cases`.
#[test]
fn every_candidate_is_its_class_and_covered_ones() {
    for wave64 in [false, true] {
        for c in candidates(wave64) {
            let seen = classes(&c.case.code, covered::class);
            let other: Vec<_> = seen.iter().filter(|&&k| k != c.class && !covered::covered(k)).collect();
            assert!(seen.contains(c.class) && other.is_empty(), "{} (wave64 {wave64}): {seen:?}", c.class);
            assert!(!covered::covered(c.class), "{} is covered: move its program to `cases`", c.class);
            let why = covered::check(&c.case.code).unwrap_err();
            assert!(why.ends_with(&format!("(class {})", c.class)), "{why}");
        }
    }
}

#[test]
fn candidate_references_have_their_cases() {
    // the ISA guide's examples for v_rcp / v_rsq / v_sqrt / v_exp / v_log f32
    let documented = [
        (0, 0xFF80_0000, 0x8000_0000), (0, 0xC000_0000, 0xBF00_0000), (0, 0x8000_0000, 0xFF80_0000), (0, 0, 0x7F80_0000),
        (0, 0x7F80_0000, 0), (1, 0x8000_0000, 0xFF80_0000), (1, 0, 0x7F80_0000), (1, 0x4080_0000, 0x3F00_0000),
        (1, 0x7F80_0000, 0), (2, 0x8000_0000, 0x8000_0000), (2, 0, 0), (2, 0x4080_0000, 0x4000_0000),
        (2, 0x7F80_0000, 0x7F80_0000), (3, 0xFF80_0000, 0), (3, 0x8000_0000, 0x3F80_0000), (3, 0x7F80_0000, 0x7F80_0000),
        (4, 0x8000_0000, 0xFF80_0000), (4, 0, 0xFF80_0000), (4, 0x3F80_0000, 0), (4, 0x7F80_0000, 0x7F80_0000),
    ];
    // denormals flushed (in: rcp, log; out: exp, rcp) and exp's overflow
    let flushed = [(0, 1, 0x7F80_0000), (0, 0x8000_0001, 0xFF80_0000), (4, 0x007F_FFFF, 0xFF80_0000), (3, 0xC2FE_0000, 0),
        (3, 0xC2FD_0000, 0), (3, 0xC2FC_0000, 0x0080_0000), (0, 0x7F00_0000, 0), (3, 0x4300_0000, 0x7F80_0000)];
    let inputs: Vec<u32> = (0..THREADS).map(t_input).collect();
    for (op, x, r) in documented.into_iter().chain(flushed) {
        assert_eq!(transcendental(op, x), Ok(r), "op {op} of {x:#x}");
        assert!(inputs.contains(&x), "{x:#x} is not an input");
    }
    // NaN: rsq / sqrt / log of -inf, log of -1, anything of NaN
    for (op, x) in [(1, 0xFF80_0000), (2, 0xFF80_0000), (4, 0xFF80_0000), (4, 0xBF80_0000), (0, 0x7FC0_0000), (3, 0x7FC0_0000)] {
        assert!(transcendental(op, x).is_err_and(|e| e.is_nan()), "op {op} of {x:#x}");
    }
    assert!(near(0x3EAA_AAAB, 1.0 / 3.0) && near(0x3EAA_AAAA, 1.0 / 3.0) && !near(0x3EAA_AAAC, 1.0 / 3.0));
    let t = transcendental_ops();
    assert!(t.near.len() > 64 && t.near.iter().any(|n| n.1.is_nan()), "{}", t.near.len());
    // valu64: v_mad_u64_u32 carries out on some lanes and not others, the compare goes both
    // ways, and a fused mul + add differs on some lanes
    let v = valu64();
    for half in [0, 1] {
        let seen: BTreeSet<u32> = (0..THREADS as usize).map(|t| v.case.want[word(11 * 128 + 2 * t + half)]).collect();
        assert_eq!(seen, BTreeSet::from([0, 1]), "half {half}");
    }
    assert!(!v.case.contracted.is_empty());
    // cross-lane: lane 40's first lane is 32 in wave32 and 0 in wave64; odd lanes start at 1
    let (c32, c64) = (cross_lane(false), cross_lane(true));
    assert_eq!((c32.case.want[word(104)], c64.case.want[word(104)]), (input(32), input(0)));
    assert_eq!((c32.case.want[word(296)], c64.case.want[word(296)]), (input(33), input(1)));
    // barrier: in wave32 every read crosses to the other wave
    let b = barrier();
    assert!((0..THREADS).all(|t| b.case.want[word(64 + t as usize)] == input(63 - t) && (63 - t) / 32 != t / 32));
}

/// The atomic candidates' references hold for any lane order: what another order leaves passes
/// the verdict of the one built in lane order (other returns, the same verdict), and a broken
/// atomic fails with the property it breaks.
#[test]
fn atomic_references_hold_in_any_lane_order() {
    let id: Vec<u32> = (0..THREADS).collect();
    let orders: [Vec<u32>; 2] = [(0..THREADS).rev().collect(), (0..THREADS).map(|t| t * 37 % THREADS).collect()];
    for at in [Where::Lds, Where::Buffer, Where::Global] {
        for wide in [false, true] {
            let c = atomics(at, wide, &id);
            assert_eq!(c.verdict(&c.case.want, false), "OK", "{}", c.case.name);
            assert!(!c.races.is_empty() && c.case.want != c.case.mem, "{}", c.case.name);
            for o in &orders {
                let other = atomics(at, wide, o);
                assert_ne!(other.case.want, c.case.want, "{}: no word depends on the order", c.case.name);
                assert_eq!(c.verdict(&other.case.want, false), "OK", "{} in another order", c.case.name);
            }
        }
    }
    let c = atomics(Where::Buffer, false, &id);
    let cells = BLOCK * 16;
    let ret = |j: usize, t: usize| word(BLOCK * j + RET + t);
    let fail = |edit: &dyn Fn(&mut Vec<u32>)| {
        let mut out = c.case.want.clone();
        edit(&mut out);
        c.verdict(&out, false)
    };
    // the +1 counter gives one value back twice (a lost update)
    assert_eq!(fail(&|o| o[ret(0, 1)] = o[ret(0, 0)]), "0:add rtn returns: 0xfffffff0 is read 2 time(s) but written 1 time(s)");
    // two lanes win the cmpswap contest
    let two = fail(&|o| o[ret(13, 1)] = 5);
    assert!(two.starts_with("13:cmpswap rtn returns: 0x5 is read 2 time(s) but written 1 time(s)"), "{two}");
    // lane 3 gets its own swap value back, lane 4 the one lane 3 got: balanced, but a loop
    let swap = fail(&|o| (o[ret(12, 3)], o[ret(12, 4)]) = (0x103, 0x102));
    assert_eq!(swap, "12:swap rtn returns: 0x103 is read in a loop of lanes no order from 0xffff0000 reaches");
    // a final word one off, and the trap word a wrapped offset reaches
    let off = fail(&|o| o[word(cells + 4)] += 1);
    assert!(off.starts_with("differs (1 word(s), first [") && off.contains(" 1:add word[0]: got "), "{off}");
    let trap = fail(&|o| o[word(cells + 128)] = 32);
    assert!(trap.contains(" oob trap[0]: got 0x00000020, want 0x00000000"), "{trap}");
}

#[test]
fn every_program_translates_directly() {
    for (wave64, split) in [(false, false), (true, false), (true, true)] {
        let o = KernelOptions { wave64, split, mem_windows: 1, sampler_base: 0 };
        for c in cases(wave64) {
            translate_compute(&c.code, o, &interface()).unwrap_or_else(|e| panic!("{} (wave64 {wave64}, split {split}): {e}", c.name));
        }
    }
}

/// The MS program is in the mesh pass's covered set, uses each of its own classes, and is no
/// compute program of the covered set (its record writes are the mesh pass's only).
#[test]
fn the_mesh_program_is_in_the_covered_mesh_set() {
    for wave64 in [false, true] {
        let code = mesh_program(wave64);
        assert_eq!(covered::check_mesh(&code), Ok(()), "wave64 {wave64}");
        assert_eq!(covered::check(&code), Err("s_sendmsg: outside the covered set (class system)".into()));
        let seen = classes(&code, covered::mesh_class);
        let missing: Vec<&str> = covered::MESH_TABLE.iter().filter(|r| r.1 && !seen.contains(r.0)).map(|r| r.0).collect();
        assert!(missing.is_empty(), "mesh classes without a program: {missing:?}");
    }
    let ms = mesh_interface();
    assert_eq!(ms.vert_dwords(), 8);
    assert_eq!(mesh_record(&ms, 1)[3..7], [(4, 0), (5, (-1f32).to_bits()), (6, 0), (7, 1f32.to_bits())]);
}

#[test]
fn the_mesh_program_translates_directly() {
    for (wave64, split) in [(false, false), (true, false), (true, true)] {
        let o = KernelOptions { wave64, split, mem_windows: 1, sampler_base: 0 };
        translate_mesh(&mesh_program(wave64), o, &mesh_interface()).unwrap_or_else(|e| panic!("wave64 {wave64}, split {split}: {e}"));
    }
}

/// The compute pass of the MS program through HLSL + DXC and through the direct path, each
/// drawn by the HLSL pass-through mesh shader: the records RDNA2 leaves and the same pixels.
#[test]
fn gpu_covered_mesh_pass_matches_on_both_backends() {
    use crate::gpu::recomp::{dxc, mesh_passthrough, runner, translate};
    match (runner::adapter(), runner::wave64_programs_supported(), dxc::location(), runner::mesh_shaders_supported()) {
        (Ok(_), Ok(true), Ok(_), Ok(true)) => {}
        (a, w, d, m) => {
            println!("GPU/DXC/mesh shaders unavailable, skipping ({:?} / {:?} / {:?} / {:?})", a.as_ref().err(), w.as_ref().ok(), d.as_ref().err(), m);
            runner::skip_test(if a.is_err() || w.is_err() || m.is_err() {
                "no GPU"
            } else if d.is_err() {
                "no DXC"
            } else if w != Ok(true) {
                "no wave64"
            } else {
                "no mesh shaders"
            });
            return;
        }
    }
    println!("GPU: {}", runner::adapter().unwrap());
    let ms = mesh_interface();
    let mode = Mode::Mesh(ms.clone());
    let pass = dxc::compile(&mesh_passthrough(&ms), "main", "ms_6_6", &[]).unwrap_or_else(|e| panic!("{e}"));
    let ps_src = "struct V { float4 pos : SV_Position; float4 p0 : TEXCOORD0; };\nfloat4 main(V v) : SV_Target0 { return v.p0; }\n";
    let ps = dxc::compile(ps_src, "main", "ps_6_6", &[]).unwrap();
    let (groups, rd) = (2u32, ms.record_dwords());
    let mut wrong = Vec::new();
    for wave64 in [false, true] {
        let split = wave64 && runner::split_for(&mode);
        let form = if !wave64 { "wave32" } else if split { "wave64 split" } else { "wave64" };
        let code = mesh_program(wave64);
        let t = translate(&code, &Options { mode: mode.clone(), wave64, split_wave64: split, ..Default::default() }).unwrap();
        assert!(t.complete(), "{form}: {:?}", t.unsupported_why);
        let hlsl = dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap_or_else(|e| panic!("{e}\n{}", t.hlsl));
        let o = KernelOptions { wave64, split, mem_windows: 1, sampler_base: 0 };
        let direct = translate_mesh(&code, o, &ms).unwrap_or_else(|e| panic!("{form}: {e}")).dxil;
        let mut drawn = Vec::new();
        for (backend, cs) in [("HLSL", &hlsl), ("direct", &direct)] {
            let (px, rec) = runner::mesh_draw(cs, &pass, &ps, groups, rd, &[0; 128], &[0; 256], 32, [-1.0; 4]).unwrap_or_else(|e| panic!("{backend} ({form}): {e}"));
            for g in 0..groups {
                let bad: Vec<(usize, u32, u32)> = mesh_record(&ms, g).into_iter().filter(|&(k, v)| rec[g as usize * rd + k] != v).map(|(k, v)| (k, rec[g as usize * rd + k], v)).collect();
                if !bad.is_empty() {
                    wrong.push(format!("{backend} ({form}) group {g}: {} word(s), first (word, got, want) {:x?}", bad.len(), &bad[..bad.len().min(4)]));
                }
            }
            // group 0's triangle on the left, group 1's on the right (param.x = group)
            for (x, y, want) in [(8, 29, [0.0, 0.0, 0.5, 1.0]), (24, 29, [1.0, 0.0, 0.5, 1.0]), (16, 2, [-1.0; 4])] {
                let got = px[y * 32 + x];
                if got[0] != want[0] || got[3] != want[3] {
                    wrong.push(format!("{backend} ({form}) pixel ({x}, {y}): {got:?}, want x and w of {want:?}"));
                }
            }
            drawn.push((px, rec));
        }
        if drawn[0] != drawn[1] {
            wrong.push(format!("{form}: the two backends leave other records or pixels"));
        }
    }
    assert!(wrong.is_empty(), "outside RDNA2's results:\n{}", wrong.join("\n"));
}

/// The GS program is in the geometry pass's covered set, uses each of its own classes, and
/// is no compute or MS program of the covered set (its messages and barrier are that pass's
/// only).
#[test]
fn the_geometry_program_is_in_the_covered_geometry_set() {
    let (code, g) = geometry_program();
    assert_eq!(covered::check_geometry(&code, g.gs_start), Ok(()));
    assert_eq!(covered::check(&code), Err("s_barrier: outside the covered set (class barrier)".into()));
    assert_eq!(covered::check_mesh(&code), Err("s_barrier: outside the covered set (class barrier)".into()));
    let seen = classes_in(&Cfg::build_geometry(&code, g.gs_start).unwrap(), covered::geom_class);
    let missing: Vec<&str> = covered::GEOM_TABLE.iter().filter(|r| r.1 && !seen.contains(r.0)).map(|r| r.0).collect();
    assert!(missing.is_empty(), "geometry classes without a program: {missing:?}");
    assert!(seen.contains("lds"), "the ESGS ring and the output vertices are LDS");
    let rec = g.record();
    assert_eq!((g.es_item_bytes(), g.es_ring_bytes(), g.vertex_bytes(), rec.vert_dwords()), (16, 1024, 32, 8));
    // 11 even lanes with 2 triangles, 10 odd ones with 1; the second group's 4 primitives
    let header = |grp| geometry_record(&g, grp).into_iter().filter(|&(k, _)| k < 2).collect::<Vec<_>>();
    assert_eq!(header(0), [(0, 84), (1, 32)]);
    assert_eq!(header(1), [(0, 16), (1, 6)]);
    assert!(rec.max_vertices >= 84 && rec.max_primitives >= 32);
    // group 1, primitive 22 (lane 1): its first vertex's TEXCOORD
    let v = crate::gpu::recomp::MESH_HEADER + 4 * rec.vert_dwords() + 4;
    let w = geometry_record(&g, 1);
    assert_eq!(w.iter().filter(|e| (v..v + 4).contains(&e.0)).map(|e| e.1).collect::<Vec<_>>(), [66, 0, 470, 5 + (68 ^ GS_MARK)]);
}

#[test]
fn the_geometry_program_translates_directly() {
    let (code, g) = geometry_program();
    for split in [false, true] {
        let o = KernelOptions { wave64: true, split, mem_windows: 1, sampler_base: 0 };
        translate_geometry(&code, o, &g).unwrap_or_else(|e| panic!("split {split}: {e}"));
    }
}

/// The compute pass of the ES + GS program through HLSL + DXC and through the direct path, in
/// the split form and (64-lane hardware) the native one, each drawn by the HLSL pass-through
/// mesh shader: the records RDNA2 leaves (a full and a partial group) and the same pixels.
#[test]
fn gpu_covered_geometry_pass_matches_on_both_backends() {
    use crate::gpu::recomp::{dxc, mesh_passthrough, runner, translate_geometry as hlsl_geometry, IO_GS_PRIMS};
    let skip = [
        (runner::adapter().is_err(), "no GPU"),
        (dxc::location().is_err(), "no DXC"),
        (runner::wave64_programs_supported() != Ok(true), "no wave64"),
        (runner::mesh_shaders_supported() != Ok(true), "no mesh shaders"),
    ];
    if let Some(&(_, reason)) = skip.iter().find(|s| s.0) {
        runner::skip_test(reason);
        return;
    }
    println!("GPU: {}", runner::adapter().unwrap());
    let (code, g) = geometry_program();
    let rec = g.record();
    let rd = rec.record_dwords();
    let pass = dxc::compile(&mesh_passthrough(&rec), "main", "ms_6_6", &[]).unwrap_or_else(|e| panic!("{e}"));
    let ps_src = "struct V { float4 pos : SV_Position; nointerpolation uint4 p0 : TEXCOORD0; };\nfloat4 main(V v) : SV_Target0 { return float4(1.0, 0.0, 0.0, 1.0); }\n";
    let ps = dxc::compile(ps_src, "main", "ps_6_6", &[]).unwrap();
    let mut io = vec![0u32; 128];
    io[IO_GS_PRIMS] = GS_PRIMS;
    let groups = GS_PRIMS.div_ceil(21);
    let mut forms = vec![true];
    if runner::wave64_supported() == Ok(true) {
        forms.push(false);
    }
    let mut wrong = Vec::new();
    for split in forms {
        let form = if split { "split" } else { "native" };
        let t = hlsl_geometry(&code, g.clone(), split, 1).unwrap_or_else(|e| panic!("{form}: {e}"));
        assert!(t.complete(), "{form}: {:?}", t.unsupported_why);
        let hlsl = dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap_or_else(|e| panic!("{e}\n{}", t.hlsl));
        let o = KernelOptions { wave64: true, split, mem_windows: 1, sampler_base: 0 };
        let direct = translate_geometry(&code, o, &g).unwrap_or_else(|e| panic!("{form}: {e}")).dxil;
        let mut drawn = Vec::new();
        for (backend, cs) in [("HLSL", &hlsl), ("direct", &direct)] {
            let (px, got) = runner::mesh_draw(cs, &pass, &ps, groups, rd, &io, &[0; 64], 32, [0.0; 4]).unwrap_or_else(|e| panic!("{backend} ({form}): {e}"));
            for grp in 0..groups {
                let bad: Vec<(usize, u32, u32)> =
                    geometry_record(&g, grp).into_iter().filter(|&(k, v)| got[grp as usize * rd + k] != v).map(|(k, v)| (k, got[grp as usize * rd + k], v)).collect();
                if !bad.is_empty() {
                    wrong.push(format!("{backend} ({form}) group {grp}: {} word(s), first (word, got, want) {:x?}", bad.len(), &bad[..bad.len().min(4)]));
                }
            }
            // the even lanes' strips cover the target, the odd lanes' triangle its lower left
            for (x, y) in [(4, 27), (28, 4)] {
                let p = px[y * 32 + x];
                if p != [1.0, 0.0, 0.0, 1.0] {
                    wrong.push(format!("{backend} ({form}) pixel ({x}, {y}): {p:?}, want red"));
                }
            }
            drawn.push((px, got));
        }
        if drawn[0] != drawn[1] {
            wrong.push(format!("{form}: the two backends leave other records or pixels"));
        }
    }
    assert!(wrong.is_empty(), "outside RDNA2's results:\n{}", wrong.join("\n"));
}

#[test]
fn gpu_covered_programs_match_rdna2_on_both_backends() {
    use crate::gpu::recomp::{dxc, runner, translate};
    use std::time::Instant;
    match (runner::adapter(), runner::wave64_programs_supported(), dxc::location()) {
        (Ok(_), Ok(true), Ok(_)) => {}
        (a, w, d) => {
            println!("GPU/DXC unavailable, skipping ({:?} / {:?} / {:?})", a.as_ref().err(), w.as_ref().ok(), d.as_ref().err());
            runner::skip_test(if a.is_err() || w.is_err() { "no GPU" } else if d.is_err() { "no DXC" } else { "no wave64" });
            return;
        }
    }
    let bench = std::env::var("XCL_COVERED_BENCH").is_ok_and(|v| v == "1");
    println!("GPU: {}", runner::adapter().unwrap());
    let mut io = vec![0u32; 128];
    io[0] = BASE;
    io[2] = LOOPS;
    let mode = Mode::Compute(interface());
    let mut wrong = Vec::new();
    let mut warm = false;
    for wave64 in [false, true] {
        let split = wave64 && runner::split_for(&mode);
        let form = if !wave64 { "wave32" } else if split { "wave64 split" } else { "wave64" };
        for c in cases(wave64) {
            let t0 = Instant::now();
            let t = translate(&c.code, &Options { mode: mode.clone(), wave64, split_wave64: split, ..Default::default() }).unwrap();
            assert!(t.complete(), "{} ({form}): {:?}", c.name, t.unsupported_why);
            if !warm {
                // DXC's first compile loads it: not part of the time
                dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap_or_else(|e| panic!("{e}"));
                warm = true;
            }
            let t1 = Instant::now();
            let hlsl = dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap_or_else(|e| panic!("{}: {e}\n{}", c.name, t.hlsl));
            let t2 = Instant::now();
            let o = KernelOptions { wave64, split, mem_windows: 1, sampler_base: 0 };
            let direct = translate_compute(&c.code, o, &interface()).unwrap_or_else(|e| panic!("{} ({form}): {e}", c.name)).dxil;
            let t3 = Instant::now();
            let ms = |d: std::time::Duration| d.as_secs_f64() * 1e3;
            println!(
                "{:7} {form:13}: HLSL translate {:.2} ms + DXC {:.2} ms, direct {:.2} ms",
                c.name,
                ms(t1 - t0),
                ms(t2 - t1),
                ms(t3 - t2)
            );
            for (backend, dxil) in [("HLSL", &hlsl), ("direct", &direct)] {
                let out = runner::dispatch(dxil, [1, 1, 1], &io, &c.mem).unwrap_or_else(|e| panic!("{} {backend} ({form}): {e}", c.name));
                let bad: Vec<(usize, u32, u32)> = (0..c.want.len()).filter(|&k| out[k] != c.want[k]).map(|k| (k, out[k], c.want[k])).collect();
                // the HLSL path leaves f32 ops `fast` (DXC), which lets a driver fuse a mul
                // into the add that reads it: noted here, not the covered set's failure
                let (fused, other): (Vec<_>, Vec<_>) = bad.into_iter().partition(|b| backend == "HLSL" && c.contracted.contains(&b.0));
                if !fused.is_empty() {
                    println!("  note: HLSL {} ({form}): {} word(s) came back as a fused mul + add", c.name, fused.len());
                }
                if !other.is_empty() {
                    wrong.push(format!("{} {backend} ({form}): {} word(s), first (word, got, want) {:x?}", c.name, other.len(), &other[..other.len().min(4)]));
                }
            }
            if bench {
                let mut big = io.clone();
                big[2] = LOOPS * 10;
                let med = |dxil: &[u8]| {
                    let mut t = runner::dispatch_timed(dxil, [4096, 1, 1], &big, &c.mem, 9).unwrap();
                    t.sort_by(f64::total_cmp);
                    t[t.len() / 2]
                };
                let (h, d) = (med(&hlsl), med(&direct));
                println!("  GPU, 4096 groups, median of 9: HLSL {h:.3} ms, direct {d:.3} ms = {:.2}x", d / h);
            }
        }
        // candidates: a verdict per class and backend, never a failure
        for c in candidates(wave64) {
            let hlsl = match translate(&c.case.code, &Options { mode: mode.clone(), wave64, split_wave64: split, ..Default::default() }) {
                Ok(t) if t.complete() => dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).map_err(|e| format!("DXC: {e}")),
                Ok(t) => Err(format!("refused: {:?}", t.unsupported_why)),
                Err(e) => Err(format!("refused: {e}")),
            };
            let o = KernelOptions { wave64, split, mem_windows: 1, sampler_base: 0 };
            let direct = translate_compute(&c.case.code, o, &interface()).map(|k| k.dxil).map_err(|e| format!("refused: {e}"));
            let run = |dxil: Result<Vec<u8>, String>, hlsl: bool| match dxil.and_then(|d| runner::dispatch(&d, [1, 1, 1], &io, &c.case.mem)) {
                Ok(out) => c.verdict(&out, hlsl),
                Err(e) => e,
            };
            println!("candidate {} ({form}): direct {} ; hlsl {}", c.case.name, run(direct, false), run(hlsl, true));
        }
    }
    assert!(wrong.is_empty(), "outside RDNA2's results:\n{}", wrong.join("\n"));
}
