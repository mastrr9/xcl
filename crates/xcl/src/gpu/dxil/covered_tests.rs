//! The covered set (`covered::TABLE`) checked against programs: a compute program for each
//! covered class, with the memory RDNA2 leaves after one group of 64 threads. The CPU tests
//! keep the programs inside the covered set, make every covered class appear in one of them
//! (a class marked covered needs a program here) and translate them directly. The GPU test
//! runs each through both backends, wave32 and wave64 (split where the adapter needs it),
//! compares the memory with the reference and prints the translation times; with
//! `XCL_COVERED_BENCH=1` it also times 4096 groups of each on the GPU.
//!
//! Candidates are programs for the next classes to flip (transcendental, valu64, cross-lane,
//! barrier), each with its RDNA2 reference. The GPU test runs them the same way and prints one
//! verdict line per class and form, without failing: they are the evidence a flip needs.

use std::collections::BTreeSet;

use super::cfg::{Cfg, Term};
use super::covered;
use super::kernel::{translate_compute, KernelOptions};
use crate::gpu::rdna2::{self, encode};
use crate::gpu::recomp::{CsInterface, Mode, Options};

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
/// as any NaN where that value is NaN; every other word must match `want` exactly.
struct Candidate {
    class: &'static str,
    case: Case,
    near: Vec<(usize, f64)>,
}

impl Candidate {
    fn passes(&self, k: usize, got: u32) -> bool {
        match self.near.iter().find(|n| n.0 == k) {
            Some(&(_, e)) => near(got, e),
            None => got == self.case.want[k],
        }
    }

    /// "OK", or how many words differ and the first, as its index from BASE.
    fn verdict(&self, out: &[u32], hlsl: bool) -> String {
        let bad: Vec<usize> = (0..self.case.want.len()).filter(|&k| !self.passes(k, out[k])).collect();
        let Some(&k) = bad.first() else { return "OK".into() };
        if hlsl && bad.iter().all(|b| self.case.contracted.contains(b)) {
            return format!("differs only by a fused mul + add ({} word(s))", bad.len());
        }
        let want = match self.near.iter().find(|n| n.0 == k) {
            Some(&(_, e)) if e.is_nan() => "any NaN".to_string(),
            Some(&(_, e)) => format!("{e:e} within 1 ulp"),
            None => format!("{:#010x}", self.case.want[k]),
        };
        format!("differs ({} word(s), first [{}]: got {:#010x}, want {want})", bad.len(), k - word(0), out[k])
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
    Candidate { class: "transcendental", case: Case { name: "transcendental", code, mem, want, contracted: Vec::new() }, near }
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
    Candidate { class: "valu64", case: Case { name: "valu64", code, mem, want, contracted }, near: Vec::new() }
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
    Candidate { class: "cross-lane", case: Case { name: "cross-lane", code, mem, want, contracted: Vec::new() }, near: Vec::new() }
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
    Candidate { class: "barrier", case: Case { name: "barrier", code, mem, want, contracted: Vec::new() }, near: Vec::new() }
}

fn candidates(wave64: bool) -> Vec<Candidate> {
    vec![transcendental_ops(), valu64(), cross_lane(wave64), barrier()]
}

/// The classes `covered::check` looks at in `code`.
fn classes(code: &[u8]) -> BTreeSet<&'static str> {
    let cfg = Cfg::build(code).unwrap_or_else(|e| panic!("{e}"));
    let mut out = BTreeSet::new();
    for &b in &cfg.rpo {
        let block = &cfg.blocks[b];
        let branch = if let Term::Cond(i, _, _) = block.term { Some(i) } else { None };
        out.extend(block.insts.iter().chain(branch.iter()).map(|&i| covered::class(&cfg.insts[i])));
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
            seen.extend(classes(&c.code));
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
            let seen = classes(&c.case.code);
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

#[test]
fn every_program_translates_directly() {
    for (wave64, split) in [(false, false), (true, false), (true, true)] {
        let o = KernelOptions { wave64, split, mem_windows: 1, sampler_base: 0 };
        for c in cases(wave64) {
            translate_compute(&c.code, o, &interface()).unwrap_or_else(|e| panic!("{} (wave64 {wave64}, split {split}): {e}", c.name));
        }
    }
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
            println!("candidate {} ({form}): direct {} ; hlsl {}", c.class, run(direct, false), run(hlsl, true));
        }
    }
    assert!(wrong.is_empty(), "outside RDNA2's results:\n{}", wrong.join("\n"));
}
