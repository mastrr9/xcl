//! The covered set (`covered::TABLE`) checked against programs: a compute program for each
//! covered class, with the memory RDNA2 leaves after one group of 64 threads. The CPU tests
//! keep the programs inside the covered set, make every covered class appear in one of them
//! (a class marked covered needs a program here) and translate them directly. The GPU test
//! runs each through both backends, wave32 and wave64 (split where the adapter needs it),
//! compares the memory with the reference and prints the translation times; with
//! `XCL_COVERED_BENCH=1` it also times 4096 groups of each on the GPU.

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
    }
    assert!(wrong.is_empty(), "outside RDNA2's results:\n{}", wrong.join("\n"));
}
