//! Size of the direct translator's output for synthetic programs: phis, instructions,
//! basic blocks and selects of the entry function, and the container's bytes. A report
//! for comparing translator changes (`cargo test -p xcl --lib direct_dxil_size_report --
//! --nocapture`, with `XCL_DXIL_NO_OPT` to leave out optimizer steps); it checks only that
//! every program translates.

use super::kernel::{self, KernelOptions};
use crate::gpu::dxbc::SigElement as S;
use crate::gpu::rdna2::encode;
use crate::gpu::recomp::{CsInterface, PsInterface, VsInterface};

fn prog(list: &[(&str, &str, &[(&str, u64)], Option<u32>)]) -> Vec<u8> {
    let mut v = Vec::new();
    for (n, e, f, l) in list {
        v.extend(encode(n, e, f, *l).unwrap_or_else(|x| panic!("{n}: {x}")));
    }
    v.extend(encode("S_ENDPGM", "ENC_SOPP", &[], None).unwrap());
    v
}

fn el(name: &str, index: u32, sv: u32, ct: u32, reg: u32, mask: u8) -> S {
    S { name: name.into(), index, system_value: sv, component_type: ct, register: reg, mask, rw_mask: mask, stream: 0 }
}

#[derive(Default, Clone, Copy)]
struct Stats {
    phis: usize,
    insts: usize,
    blocks: u64,
    selects: usize,
    bytes: usize,
}

/// The statistics of a container's entry function.
fn stats(c: &[u8]) -> Stats {
    let n = u32::from_le_bytes(c[28..32].try_into().unwrap()) as usize;
    let part = (0..n)
        .map(|i| u32::from_le_bytes(c[32 + 4 * i..36 + 4 * i].try_into().unwrap()) as usize)
        .find(|&o| &c[o..o + 4] == b"DXIL")
        .expect("DXIL part");
    let len = u32::from_le_bytes(c[part + 4..part + 8].try_into().unwrap()) as usize;
    // part header (8 bytes), program and bitcode headers (24 bytes), bitcode
    let root = super::bitreader_tests::parse(&c[part + 8 + 24..part + 8 + len]);
    let f = root.find(12).expect("function block");
    // FUNC_CODE_DECLAREBLOCKS = 1, FUNC_CODE_INST_PHI = 16, FUNC_CODE_INST_VSELECT = 29
    Stats {
        phis: f.records_with(16).len(),
        insts: f.records.iter().filter(|r| r.0 != 1).count(),
        blocks: f.records_with(1)[0][0],
        selects: f.records_with(29).len(),
        bytes: c.len(),
    }
}

fn fail<T>(name: &str, e: String) -> T {
    panic!("{name}: {}", &e[..e.len().min(600)])
}

/// Synthetic programs (name, container), each in the wave modes it runs in.
fn programs() -> Vec<(String, Vec<u8>)> {
    const V: u64 = 256;
    // a scalar counting loop and a few vector instructions (core_tests)
    let counting = prog(&[
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 128)], None),
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 1), ("SSRC0", 138)], None),
        ("S_ADD_U32", "ENC_SOP2", &[("SDST", 0), ("SSRC0", 0), ("SSRC1", 1)], None),
        ("S_SUB_U32", "ENC_SOP2", &[("SDST", 1), ("SSRC0", 1), ("SSRC1", 129)], None),
        ("S_CMP_LG_U32", "ENC_SOPC", &[("SSRC0", 1), ("SSRC1", 128)], None),
        ("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", 0xFFFC)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 1), ("SRC0", 0)], None),
        ("V_CMP_GT_U32", "ENC_VOPC", &[("SRC0", 160), ("VSRC1", 0)], None),
        ("V_ADD_F32", "ENC_VOP2", &[("VDST", 10), ("SRC0", 257), ("VSRC1", 2)], None),
    ]);
    // a loop, EXEC saved and changed, a guarded load, a lane read (robustness_tests)
    let branchy = prog(&[
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 128)], None),
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 1), ("SSRC0", 138)], None),
        ("S_ADD_U32", "ENC_SOP2", &[("SDST", 0), ("SSRC0", 0), ("SSRC1", 1)], None),
        ("S_SUB_U32", "ENC_SOP2", &[("SDST", 1), ("SSRC0", 1), ("SSRC1", 129)], None),
        ("S_CMP_LG_U32", "ENC_SOPC", &[("SSRC0", 1), ("SSRC1", 128)], None),
        ("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", 0xFFFC)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 1), ("SRC0", 0)], None),
        ("V_CMP_GT_U32", "ENC_VOPC", &[("SRC0", 160), ("VSRC1", 0)], None),
        ("S_AND_SAVEEXEC_B64", "ENC_SOP1", &[("SDST", 20), ("SSRC0", 106)], None),
        ("GLOBAL_LOAD_DWORD", "ENC_FLAT_GLBL", &[("VDST", 5), ("ADDR", 0), ("SADDR", 0x7D), ("OFFSET", 8)], None),
        ("V_ADD_F32", "ENC_VOP2", &[("VDST", 10), ("SRC0", V + 5), ("VSRC1", 2)], None),
        ("V_READLANE_B32", "ENC_VOP3", &[("VDST", 30), ("SRC0", V + 10), ("SRC1", 129)], None),
        ("S_MOV_B64", "ENC_SOP1", &[("SDST", 126), ("SSRC0", 20)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 12), ("SRC0", 30)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 13), ("SRC0", 31)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 14), ("SRC0", 32)], None),
    ]);
    // a loop whose body sets s2 and v3 back to their values from before the loop on one
    // path (loop-carried registers the loop never changes), then stores both
    let reset_loop = prog(&[
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 2), ("SSRC0", 128)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 3), ("SRC0", 242)], None),
        // loop:
        ("S_ADD_U32", "ENC_SOP2", &[("SDST", 0), ("SSRC0", 0), ("SSRC1", 129)], None),
        ("S_CMP_EQ_U32", "ENC_SOPC", &[("SSRC0", 0), ("SSRC1", 1)], None),
        ("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", 2)], None),
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 2), ("SSRC0", 128)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 3), ("SRC0", 242)], None),
        // skip:
        ("S_CMP_LG_U32", "ENC_SOPC", &[("SSRC0", 0), ("SSRC1", 3)], None),
        ("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", 0xFFF9)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 4), ("SRC0", 2)], None),
        ("GLOBAL_STORE_DWORD", "ENC_FLAT_GLBL", &[("DATA", 3), ("ADDR", 0), ("SADDR", 0x7D), ("OFFSET", 0)], None),
        ("GLOBAL_STORE_DWORD", "ENC_FLAT_GLBL", &[("DATA", 4), ("ADDR", 0), ("SADDR", 0x7D), ("OFFSET", 4)], None),
    ]);
    // vector math and a store that never touch EXEC
    let vector = prog(&[
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 1), ("SRC0", 242)], None),
        ("V_ADD_F32", "ENC_VOP2", &[("VDST", 2), ("SRC0", V), ("VSRC1", 1)], None),
        ("V_MUL_F32", "ENC_VOP2", &[("VDST", 3), ("SRC0", V + 2), ("VSRC1", 2)], None),
        ("GLOBAL_STORE_DWORD", "ENC_FLAT_GLBL", &[("DATA", 3), ("ADDR", 0), ("SADDR", 0x7D), ("OFFSET", 0)], None),
    ]);
    // guarded memory accesses between a register's write and its use (core_tests, shorter)
    let mut long: Vec<(&str, &str, Vec<(&str, u64)>, Option<u32>)> = vec![("V_MOV_B32", "ENC_VOP1", vec![("VDST", 7), ("SRC0", 129)], None)];
    for i in 0..300u64 {
        long.push(("GLOBAL_LOAD_DWORD", "ENC_FLAT_GLBL", vec![("VDST", 2 + (i % 4)), ("ADDR", 0), ("SADDR", 0x7D), ("OFFSET", 4 * (i % 64))], None));
        long.push(("V_ADD_NC_U32", "ENC_VOP2", vec![("VDST", 1), ("SRC0", V + 2 + (i % 4)), ("VSRC1", 1)], None));
    }
    let long_refs: Vec<(&str, &str, &[(&str, u64)], Option<u32>)> = long.iter().map(|(a, b, c, d)| (*a, *b, c.as_slice(), *d)).collect();
    let long = prog(&long_refs);
    // VS / PS that branch before their exports (core_tests)
    let vs_code = prog(&[
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 0), ("SRC0", V + 4)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 1), ("SRC0", V + 5)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 2), ("SRC0", 128)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 3), ("SRC0", 242)], None),
        ("S_CMP_LG_U32", "ENC_SOPC", &[("SSRC0", 0), ("SSRC1", 128)], None),
        ("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", 1)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 8), ("SRC0", 128)], None),
        ("EXP", "ENC_EXP", &[("TGT", 12), ("EN", 0xF), ("VSRC0", 0), ("VSRC1", 1), ("VSRC2", 2), ("VSRC3", 3), ("DONE", 1)], None),
        ("EXP", "ENC_EXP", &[("TGT", 32), ("EN", 0xF), ("VSRC0", 8), ("VSRC1", 9), ("VSRC2", 10), ("VSRC3", 11)], None),
        ("EXP", "ENC_EXP", &[("TGT", 33), ("EN", 0x3), ("VSRC0", 0), ("VSRC1", 1), ("VSRC2", 0), ("VSRC3", 0)], None),
    ]);
    let vsi = VsInterface {
        inputs: vec![el("POSITION", 0, 0, 3, 0, 3), el("COLOR", 0, 0, 3, 1, 0xF)],
        outputs: vec![el("SV_Position", 0, 1, 3, 0, 0xF), el("COLOR", 0, 0, 3, 1, 0xF), el("TEXCOORD", 3, 0, 3, 2, 0x3)],
        attr_base: 4,
        params: None,
    };
    let ps_code = prog(&[
        ("V_INTERP_P2_F32", "ENC_VINTRP", &[("VDST", 4), ("VSRC", 1), ("ATTR", 0), ("ATTRCHAN", 0)], None),
        ("V_INTERP_P2_F32", "ENC_VINTRP", &[("VDST", 5), ("VSRC", 1), ("ATTR", 0), ("ATTRCHAN", 1)], None),
        ("V_INTERP_P2_F32", "ENC_VINTRP", &[("VDST", 6), ("VSRC", 1), ("ATTR", 1), ("ATTRCHAN", 0)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 7), ("SRC0", 242)], None),
        ("S_CMP_LG_U32", "ENC_SOPC", &[("SSRC0", 0), ("SSRC1", 128)], None),
        ("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", 1)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 5), ("SRC0", 128)], None),
        ("EXP", "ENC_EXP", &[("TGT", 0), ("EN", 0xF), ("VSRC0", 4), ("VSRC1", 5), ("VSRC2", 6), ("VSRC3", 7), ("DONE", 1), ("VM", 1)], None),
    ]);
    let psi = PsInterface {
        inputs: vec![el("SV_Position", 0, 1, 3, 0, 0xF), el("COLOR", 0, 0, 3, 1, 0xF), el("TEXCOORD", 3, 0, 3, 2, 0x3)],
        outputs: vec![el("SV_Target", 0, 64, 3, 0, 0xF)],
        input_ena: 0x302,
        attrs: None,
    };
    let cs = |threads: u32| CsInterface { threads: [threads, 1, 1], user_sgprs: 4, tgid: [true, false, false], tg_size: false, tid_components: 1 };
    let o = |wave64, split| KernelOptions { wave64, split, mem_windows: 1, sampler_base: 0 };
    let tag = |wave64, split| match (wave64, split) {
        (false, _) => "w32",
        (true, false) => "w64",
        (true, true) => "split",
    };
    let mut out = Vec::new();
    for (w, s) in [(false, false), (true, false), (true, true)] {
        let t = tag(w, s);
        for (name, code) in [("counting", &counting), ("branchy", &branchy), ("reset_loop", &reset_loop), ("vector", &vector)] {
            let n = format!("{name} kernel {t}");
            out.push((n.clone(), kernel::translate(code, o(w, s)).unwrap_or_else(|e| fail(&n, e))));
            for threads in [64, 48] {
                let n = format!("{name} CS{threads} {t}");
                out.push((n.clone(), kernel::translate_compute(code, o(w, s), &cs(threads)).unwrap_or_else(|e| fail(&n, e)).dxil));
            }
        }
        let n = format!("long CS64 {t}");
        out.push((n.clone(), kernel::translate_compute(&long, o(w, s), &cs(64)).unwrap_or_else(|e| fail(&n, e)).dxil));
    }
    for w in [false, true] {
        let t = tag(w, false);
        let n = format!("VS {t}");
        out.push((n.clone(), kernel::translate_vertex(&vs_code, o(w, false), &vsi).unwrap_or_else(|e| fail(&n, e)).dxil));
        let n = format!("PS {t}");
        out.push((n.clone(), kernel::translate_pixel(&ps_code, o(w, false), &psi).unwrap_or_else(|e| fail(&n, e)).dxil));
    }
    out
}

#[test]
fn direct_dxil_size_report() {
    let off = std::env::var("XCL_DXIL_NO_OPT").unwrap_or_default();
    println!("direct DXIL size report (XCL_DXIL_NO_OPT={off:?})");
    println!("{:<24} {:>6} {:>7} {:>7} {:>8} {:>8}", "program", "phis", "insts", "blocks", "selects", "bytes");
    let mut total = Stats::default();
    for (name, c) in programs() {
        let s = stats(&c);
        assert!(s.insts > 0 && s.blocks > 0, "{name}: empty entry function");
        println!("{:<24} {:>6} {:>7} {:>7} {:>8} {:>8}", name, s.phis, s.insts, s.blocks, s.selects, s.bytes);
        total.phis += s.phis;
        total.insts += s.insts;
        total.blocks += s.blocks;
        total.selects += s.selects;
        total.bytes += s.bytes;
    }
    println!("{:<24} {:>6} {:>7} {:>7} {:>8} {:>8}", "total", total.phis, total.insts, total.blocks, total.selects, total.bytes);
}
