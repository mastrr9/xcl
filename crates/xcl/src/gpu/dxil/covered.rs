//! The covered set (`XCL_DIRECT_DXIL=covered`): the instruction classes a compute program may
//! use to take the direct translator while the rest of the live programs stay on the HLSL path
//! (phase 7 of `docs/specs/2026-11-24-direct-dxil.md`). A class is covered when the differential
//! runs (both backends, native and split wave64) show no difference on it. The first covered
//! classes are the compute core most programs need; those runs confirm them or take them out,
//! and the reason of every class still outside says what they have to show first.

use super::cfg::{Cfg, Term};
use crate::gpu::rdna2::{self, Variant};

/// Every class: (class, covered, what it holds and why it is in or out). A class moves in by
/// flipping its flag once its differential runs are clean; the commit says which runs.
pub const TABLE: &[(&str, bool, &str)] = &[
    ("salu", true, "scalar ALU and EXEC masks: one value per wave, no rounding"),
    ("flow", true, "branches, s_getpc / s_setpc / s_call, s_endpgm: the CFG both paths share"),
    ("hint", true, "s_nop, s_waitcnt*, s_clause, priority and prefetch hints: no effect on results"),
    ("scalar-load", true, "s_load / s_buffer_load"),
    ("valu", true, "32-bit integer and bitwise VALU, moves, selects, compares"),
    ("valu-f32", true, "f32 arithmetic and conversions: a mul then an add must not fuse on either path"),
    ("lds", true, "LDS reads and writes"),
    ("buffer", true, "untyped MUBUF loads and stores"),
    ("global", true, "global / flat loads and stores"),
    ("hwreg", false, "s_getreg / s_setreg, round and denorm modes: dynamic modes lower differently"),
    ("indexed", false, "s_movrel / v_movrel / v_swaprel: bounds of the indexed range"),
    ("system", false, "messages, traps, clocks, debug branches, subvector loops"),
    ("cache", false, "cache invalidation: needs the coherence of memory across groups shown first"),
    ("barrier", false, "s_barrier: needs multi-wave groups with early-exited waves shown first"),
    ("transcendental", false, "f32 rcp / rsq / sqrt / exp / log / sin / cos: precision against DXC's"),
    ("valu16", false, "f16 / 16-bit integer and packed VALU (VOP3P)"),
    ("valu64", false, "f64 and 64-bit integer VALU"),
    ("cross-lane", false, "readlane / writelane / permlane, ds_swizzle / ds_permute: wave64 split halves"),
    ("dpp", false, "DPP16 / DPP8: lane exchange, wave64 split halves"),
    ("sdwa", false, "SDWA operand selects"),
    ("interp", false, "attribute interpolation: pixel shaders only"),
    ("lds-atomic", false, "LDS atomics: the differential compare needs an order-free check"),
    ("gds", false, "GDS, GWS, append / consume, ordered count"),
    ("buffer-format", false, "typed and format buffer access: format conversions"),
    ("buffer-atomic", false, "buffer atomics: the differential compare needs an order-free check"),
    ("global-atomic", false, "global / flat atomics: the differential compare needs an order-free check"),
    ("scratch", false, "scratch memory"),
    ("mem-d16", false, "D16 half-register loads and stores"),
    ("image-sample", false, "image_sample* / image_gather*: LOD, offset and clamp handling"),
    ("image-load-store", false, "image_load* / image_store*, MSAA loads"),
    ("image-atomic", false, "image atomics"),
    ("image-query", false, "image_get_resinfo / image_get_lod"),
    ("bvh", false, "ray intersection"),
    ("export", false, "exp: graphics stages only"),
];

/// The classes of the compute pass of a console MS only (`check_mesh`): what it writes into
/// its record, which both paths lower to the same stores (`lower_mem.rs` `mesh_export` /
/// `mesh_sendmsg`, `emit.rs` `export_graphics` / `s_sendmsg`). Elsewhere these are `export` and
/// `system`.
pub const MESH_TABLE: &[(&str, bool, &str)] = &[
    ("mesh-export", true, "exp of position 0, params and the primitive into the record (pos 1-3 and null write nothing)"),
    ("mesh-alloc", true, "s_sendmsg GS_ALLOC_REQ: the vertex and primitive counts into the record header"),
];

/// The classes of the compute pass of a console GS (ES + GS, `check_geometry`): what that pass
/// does with the GS's own messages and barrier, which both paths lower alike (`lower_geom.rs`
/// `geom_inst`, `lower.rs` `sopp`; `emit.rs` `sopp`). Elsewhere these are `system` and
/// `barrier`. The ESGS ring and the GS output vertices are LDS reads and writes (`lds`): in the
/// GS code both paths also keep the lowest address each lane wrote, where its vertices are.
pub const GEOM_TABLE: &[(&str, bool, &str)] = &[
    ("gs-emit", true, "s_sendmsg MSG_GS of stream 0: emit (a strip vertex and triangle) and cut; the record has stream 0 only"),
    ("gs-done", true, "s_sendmsg MSG_GS_DONE: nothing to do, the record is written at the end"),
    ("gs-barrier", true, "s_barrier: the group is one wave64 (in the split form the GS code runs in one hardware wave and skips it)"),
];

/// Whether `class` is in the covered set (a class missing from `TABLE` is not).
pub fn covered(class: &str) -> bool {
    TABLE.iter().any(|&(c, on, _)| c == class && on)
}

/// `covered` in the compute pass of a console MS: `MESH_TABLE`'s classes too.
pub fn covered_mesh(class: &str) -> bool {
    covered(class) || MESH_TABLE.iter().any(|&(c, on, _)| c == class && on)
}

/// `covered` in the compute pass of a console GS: `GEOM_TABLE`'s classes too.
pub fn covered_geometry(class: &str) -> bool {
    covered(class) || GEOM_TABLE.iter().any(|&(c, on, _)| c == class && on)
}

/// The class of an instruction: its spec name (`Inst::name`), its encoding (`Encoding::name`)
/// and its VOP1 / VOP2 / VOPC form. Anything not recognised is `system` (not covered).
pub fn class_of(name: &str, enc: &str, variant: Variant) -> &'static str {
    match variant {
        Variant::Dpp16 | Variant::Dpp8 => return "dpp",
        Variant::Sdwa => return "sdwa",
        Variant::Plain => {}
    }
    let has = |s: &str| name.contains(s);
    if enc == "ENC_EXP" {
        return "export";
    }
    if matches!(name, "BUFFER_GL0_INV" | "BUFFER_GL1_INV" | "S_DCACHE_INV" | "S_GL1_INV" | "S_ICACHE_INV") {
        return "cache";
    }
    if enc == "ENC_MIMG" {
        return if has("_ATOMIC_") {
            "image-atomic"
        } else if has("_SAMPLE") || has("_GATHER") {
            "image-sample"
        } else if has("_BVH") {
            "bvh"
        } else if has("_GET_") {
            "image-query"
        } else if has("_LOAD") || has("_STORE") {
            "image-load-store"
        } else {
            "system"
        };
    }
    if matches!(enc, "ENC_MUBUF" | "ENC_MTBUF") {
        return if has("_ATOMIC_") {
            "buffer-atomic"
        } else if has("_FORMAT_") {
            "buffer-format"
        } else if has("_D16") {
            "mem-d16"
        } else if has("_LOAD_") || has("_STORE_") {
            "buffer"
        } else {
            "system"
        };
    }
    if matches!(enc, "ENC_FLAT" | "ENC_FLAT_GLBL" | "ENC_FLAT_SCRATCH") {
        return if has("_ATOMIC_") {
            "global-atomic"
        } else if has("_D16") {
            "mem-d16"
        } else if enc == "ENC_FLAT_SCRATCH" {
            "scratch"
        } else if has("_LOAD_") || has("_STORE_") {
            "global"
        } else {
            "system"
        };
    }
    if enc == "ENC_DS" {
        return if has("_GWS_") || matches!(name, "DS_APPEND" | "DS_CONSUME" | "DS_ORDERED_COUNT") {
            "gds"
        } else if matches!(name, "DS_SWIZZLE_B32" | "DS_PERMUTE_B32" | "DS_BPERMUTE_B32") {
            "cross-lane"
        } else if name == "DS_NOP" {
            "hint"
        } else if has("_D16") {
            "mem-d16"
        } else if name.starts_with("DS_READ") || name.starts_with("DS_WRITE") {
            "lds"
        } else {
            "lds-atomic"
        };
    }
    if enc == "ENC_SMEM" {
        return if name.starts_with("S_LOAD_") || name.starts_with("S_BUFFER_LOAD_") { "scalar-load" } else { "system" };
    }
    if has("MOVREL") || has("SWAPREL") {
        return "indexed";
    }
    if let Some(s) = name.strip_prefix("S_") {
        let c = match s {
            "BARRIER" => "barrier",
            "GETREG_B32" | "SETREG_B32" | "SETREG_IMM32_B32" | "DENORM_MODE" | "ROUND_MODE" => "hwreg",
            "BRANCH" | "CBRANCH_SCC0" | "CBRANCH_SCC1" | "CBRANCH_VCCZ" | "CBRANCH_VCCNZ" | "CBRANCH_EXECZ" | "CBRANCH_EXECNZ" | "ENDPGM" | "GETPC_B64"
            | "SETPC_B64" | "SWAPPC_B64" | "CALL_B64" => "flow",
            "NOP" | "CLAUSE" | "INST_PREFETCH" | "CODE_END" | "SETPRIO" | "SLEEP" | "INCPERFLEVEL" | "DECPERFLEVEL" | "VERSION" | "WAITCNT"
            | "WAITCNT_DEPCTR" | "WAITCNT_VMCNT" | "WAITCNT_VSCNT" | "WAITCNT_EXPCNT" | "WAITCNT_LGKMCNT" => "hint",
            _ if matches!(enc, "ENC_SOP1" | "ENC_SOP2" | "ENC_SOPC" | "ENC_SOPK") && !matches!(s, "RFE_B64" | "SUBVECTOR_LOOP_BEGIN" | "SUBVECTOR_LOOP_END") => "salu",
            _ => "system",
        };
        return c;
    }
    if !name.starts_with("V_") {
        return "system";
    }
    if enc == "ENC_VINTRP" || name.starts_with("V_INTERP_") {
        "interp"
    } else if name.starts_with("V_READLANE") || name.starts_with("V_WRITELANE") || name.starts_with("V_READFIRSTLANE") || name.starts_with("V_PERMLANE") {
        "cross-lane"
    } else if enc == "ENC_VOP3P" || name.starts_with("V_PK_") {
        "valu16"
    } else if has("F64") || has("I64") || has("U64") || has("B64") {
        "valu64"
    } else if has("F16") || has("I16") || has("U16") || has("B16") {
        "valu16"
    } else if matches!(name, "V_RCP_F32" | "V_RCP_IFLAG_F32" | "V_RSQ_F32" | "V_SQRT_F32" | "V_EXP_F32" | "V_LOG_F32" | "V_SIN_F32" | "V_COS_F32") {
        "transcendental"
    } else if has("F32") && !name.starts_with("V_CMP") {
        "valu-f32"
    } else {
        "valu"
    }
}

/// The class of a decoded instruction.
pub fn class(d: &rdna2::Decoded) -> &'static str {
    class_of(d.inst.name, d.enc.name, d.variant)
}

/// The class of a decoded instruction in the compute pass of a console MS: an export to a
/// target that pass writes is `mesh-export`, GS_ALLOC_REQ is `mesh-alloc`, the rest as `class`.
pub fn mesh_class(d: &rdna2::Decoded) -> &'static str {
    let f = |n: &str| rdna2::field_value(d.enc, &d.words, n).unwrap_or(0);
    match (d.inst.name, d.enc.name) {
        ("EXP", "ENC_EXP") if matches!(f("TGT"), 9 | 12..=15 | 20 | 32..=63) => "mesh-export",
        ("S_SENDMSG", "ENC_SOPP") if f("SIMM16") & 0xF == 9 => "mesh-alloc",
        _ => class(d),
    }
}

/// The class of a decoded instruction in the compute pass of a console GS: MSG_GS of stream
/// 0 is `gs-emit`, MSG_GS_DONE `gs-done`, `s_barrier` `gs-barrier`, the rest as `class`.
pub fn geom_class(d: &rdna2::Decoded) -> &'static str {
    let simm = rdna2::field_value(d.enc, &d.words, "SIMM16").unwrap_or(0);
    match (d.inst.name, d.enc.name) {
        ("S_SENDMSG", "ENC_SOPP") if simm & 0xF == 2 && (simm >> 8) & 3 == 0 => "gs-emit",
        ("S_SENDMSG", "ENC_SOPP") if simm & 0xF == 3 => "gs-done",
        ("S_BARRIER", "ENC_SOPP") => "gs-barrier",
        _ => class(d),
    }
}

/// `Ok` when every instruction the direct translator would lower (those of the blocks reachable
/// from the entry, as `kernel` builds its CFG) is in the covered set; else the first one that is
/// not, as `"<mnemonic>: outside the covered set (class <class>)"` (`coverage::reason_key` keeps
/// the mnemonic and the class), or the CFG's own error.
pub fn check(code: &[u8]) -> Result<(), String> {
    check_with(&Cfg::build(code)?, class, covered)
}

/// `check` for the compute pass of a console MS (`mesh_class`, `covered_mesh`).
pub fn check_mesh(code: &[u8]) -> Result<(), String> {
    check_with(&Cfg::build(code)?, mesh_class, covered_mesh)
}

/// `check` for the compute pass of a console GS (`geom_class`, `covered_geometry`): the ES
/// code, then the GS code from `gs_start` on, as `kernel::translate_geometry` builds its CFG.
pub fn check_geometry(code: &[u8], gs_start: Option<u64>) -> Result<(), String> {
    check_with(&Cfg::build_geometry(code, gs_start)?, geom_class, covered_geometry)
}

fn check_with(cfg: &Cfg, classify: fn(&rdna2::Decoded) -> &'static str, in_set: fn(&str) -> bool) -> Result<(), String> {
    for &b in &cfg.rpo {
        let block = &cfg.blocks[b];
        let branch = match block.term {
            Term::Cond(i, _, _) => Some(i),
            _ => None,
        };
        for &i in block.insts.iter().chain(branch.iter()) {
            let d = &cfg.insts[i];
            let c = classify(d);
            if !in_set(c) {
                return Err(format!("{}: outside the covered set (class {})", rdna2::mnemonic(d), c));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::rdna2::{encode, encoding};

    /// Every instruction the decoder knows, in every encoding it has, falls in a class of
    /// `TABLE`, and `TABLE` names each class once.
    #[test]
    fn every_instruction_has_a_tabled_class() {
        let mut seen = std::collections::BTreeSet::new();
        for &(c, _, why) in TABLE {
            assert!(seen.insert(c), "{} twice in TABLE", c);
            assert!(!why.is_empty());
        }
        let encs: Vec<&str> = crate::gpu::rdna2_spec::ENCODINGS.iter().map(|e| e.name).collect();
        for inst in rdna2::all_instructions() {
            for ie in inst.encs {
                let c = class_of(inst.name, encs[ie.encoding as usize], Variant::Plain);
                assert!(seen.contains(&c), "{} ({}): class {} not in TABLE", inst.name, encs[ie.encoding as usize], c);
            }
        }
    }

    #[test]
    fn classes_of_instructions() {
        let p = Variant::Plain;
        for (name, enc, want) in [
            ("S_ADD_U32", "ENC_SOP2", "salu"), ("S_AND_SAVEEXEC_B64", "ENC_SOP1", "salu"), ("S_CMPK_EQ_U32", "ENC_SOPK", "salu"),
            ("S_CBRANCH_EXECZ", "ENC_SOPP", "flow"), ("S_SETPC_B64", "ENC_SOP1", "flow"), ("S_ENDPGM", "ENC_SOPP", "flow"),
            ("S_WAITCNT", "ENC_SOPP", "hint"), ("S_WAITCNT_VSCNT", "ENC_SOPK", "hint"), ("S_SETREG_B32", "ENC_SOPK", "hwreg"),
            ("S_DENORM_MODE", "ENC_SOPP", "hwreg"), ("S_SENDMSG", "ENC_SOPP", "system"), ("S_ENDPGM_SAVED", "ENC_SOPP", "system"),
            ("S_SUBVECTOR_LOOP_BEGIN", "ENC_SOPK", "system"), ("S_RFE_B64", "ENC_SOP1", "system"), ("S_MEMTIME", "ENC_SMEM", "system"),
            ("S_BARRIER", "ENC_SOPP", "barrier"), ("S_MOVRELS_B32", "ENC_SOP1", "indexed"), ("V_MOVRELD_B32", "ENC_VOP1", "indexed"),
            ("S_BUFFER_LOAD_DWORDX4", "ENC_SMEM", "scalar-load"), ("S_DCACHE_INV", "ENC_SMEM", "cache"), ("BUFFER_GL1_INV", "ENC_MUBUF", "cache"),
            ("V_ADD_NC_U32", "ENC_VOP2", "valu"), ("V_CNDMASK_B32", "ENC_VOP2", "valu"), ("V_MBCNT_LO_U32_B32", "ENC_VOP3", "valu"),
            ("V_CMP_LT_F32", "ENC_VOPC", "valu"), ("V_CMPX_CLASS_F32", "ENC_VOPC", "valu"), ("V_ADD_F32", "ENC_VOP2", "valu-f32"),
            ("V_MAD_F32", "ENC_VOP3", "valu-f32"), ("V_CVT_I32_F32", "ENC_VOP1", "valu-f32"), ("V_RCP_F32", "ENC_VOP1", "transcendental"),
            ("V_SQRT_F32", "ENC_VOP3", "transcendental"), ("V_RCP_F16", "ENC_VOP1", "valu16"), ("V_RCP_F64", "ENC_VOP1", "valu64"),
            ("V_CVT_F32_F16", "ENC_VOP1", "valu16"), ("V_CVT_F64_F32", "ENC_VOP1", "valu64"), ("V_LSHLREV_B64", "ENC_VOP3", "valu64"),
            ("V_MAD_U64_U32", "VOP3_SDST_ENC", "valu64"), ("V_PK_ADD_F16", "ENC_VOP3P", "valu16"), ("V_FMA_MIX_F32", "ENC_VOP3P", "valu16"),
            ("V_READFIRSTLANE_B32", "ENC_VOP1", "cross-lane"), ("V_PERMLANEX16_B32", "ENC_VOP3", "cross-lane"), ("V_INTERP_P1_F32", "ENC_VINTRP", "interp"),
            ("DS_READ_B32", "ENC_DS", "lds"), ("DS_WRITE2ST64_B64", "ENC_DS", "lds"), ("DS_READ_U16_D16", "ENC_DS", "mem-d16"),
            ("DS_ADD_U32", "ENC_DS", "lds-atomic"), ("DS_WRXCHG_RTN_B32", "ENC_DS", "lds-atomic"), ("DS_BPERMUTE_B32", "ENC_DS", "cross-lane"),
            ("DS_APPEND", "ENC_DS", "gds"), ("DS_GWS_INIT", "ENC_DS", "gds"), ("BUFFER_LOAD_DWORDX4", "ENC_MUBUF", "buffer"),
            ("BUFFER_STORE_BYTE", "ENC_MUBUF", "buffer"), ("BUFFER_LOAD_SHORT_D16_HI", "ENC_MUBUF", "mem-d16"), ("BUFFER_LOAD_FORMAT_XYZW", "ENC_MUBUF", "buffer-format"),
            ("TBUFFER_STORE_FORMAT_D16_X", "ENC_MTBUF", "buffer-format"), ("BUFFER_ATOMIC_CMPSWAP", "ENC_MUBUF", "buffer-atomic"), ("GLOBAL_LOAD_DWORD", "ENC_FLAT_GLBL", "global"),
            ("FLAT_STORE_DWORDX2", "ENC_FLAT", "global"), ("GLOBAL_ATOMIC_ADD", "ENC_FLAT_GLBL", "global-atomic"), ("GLOBAL_LOAD_UBYTE_D16", "ENC_FLAT_GLBL", "mem-d16"),
            ("SCRATCH_LOAD_DWORD", "ENC_FLAT_SCRATCH", "scratch"), ("IMAGE_SAMPLE_LZ", "ENC_MIMG", "image-sample"), ("IMAGE_GATHER4_C_LZ_O", "ENC_MIMG", "image-sample"),
            ("IMAGE_LOAD_MIP", "ENC_MIMG", "image-load-store"), ("IMAGE_MSAA_LOAD", "ENC_MIMG", "image-load-store"), ("IMAGE_ATOMIC_ADD", "ENC_MIMG", "image-atomic"),
            ("IMAGE_GET_RESINFO", "ENC_MIMG", "image-query"), ("IMAGE_BVH64_INTERSECT_RAY", "ENC_MIMG", "bvh"), ("EXP", "ENC_EXP", "export"),
        ] {
            assert_eq!(class_of(name, enc, p), want, "{name}");
        }
        assert_eq!(class_of("V_ADD_F32", "ENC_VOP2", Variant::Dpp16), "dpp");
        assert_eq!(class_of("V_MOV_B32", "ENC_VOP1", Variant::Dpp8), "dpp");
        assert_eq!(class_of("V_ADD_F32", "ENC_VOP2", Variant::Sdwa), "sdwa");
        assert!(covered("valu") && covered("valu-f32") && covered("salu") && covered("flow"));
        assert!(!covered("transcendental") && !covered("barrier") && !covered("buffer-atomic"));
    }

    fn i(name: &str, enc: &str, fields: &[(&str, u64)]) -> Vec<u8> {
        encode(name, enc, fields, None).unwrap_or_else(|e| panic!("{name}: {e}"))
    }

    /// Whole programs through the decoder and the CFG: the first uncovered instruction on a
    /// reachable path is the refusal (and its fallback reason); code no path reaches is not.
    #[test]
    fn programs_in_and_out_of_the_covered_set() {
        const V: u64 = 256;
        let end = i("S_ENDPGM", "ENC_SOPP", &[]);
        let mov = i("S_MOV_B32", "ENC_SOP1", &[("SDST", 4), ("SSRC0", 0)]);
        let add = i("V_ADD_F32", "ENC_VOP2", &[("VDST", 1), ("SRC0", V + 1), ("VSRC1", 2)]);
        let rcp = i("V_RCP_F32", "ENC_VOP1", &[("VDST", 3), ("SRC0", V + 1)]);
        let wait = i("S_WAITCNT", "ENC_SOPP", &[]);
        let ok: Vec<u8> = [&mov, &add, &wait, &end].into_iter().flatten().copied().collect();
        assert_eq!(check(&ok), Ok(()));

        let out: Vec<u8> = [&mov, &add, &rcp, &end].into_iter().flatten().copied().collect();
        let e = check(&out).unwrap_err();
        assert_eq!(e, "v_rcp_f32: outside the covered set (class transcendental)");
        assert_eq!(super::super::coverage::reason_key(&e), e, "the fallback reason keeps mnemonic and class");

        // after s_endpgm, unreachable: the translator does not lower it either
        let dead: Vec<u8> = [&mov, &end, &rcp].into_iter().flatten().copied().collect();
        assert_eq!(check(&dead), Ok(()));

        // the taken side of a branch over it is not enough: the fall-through reaches it
        let skip = i("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", 1)]);
        let branchy: Vec<u8> = [&mov, &skip, &rcp, &end].into_iter().flatten().copied().collect();
        assert_eq!(check(&branchy).unwrap_err(), "v_rcp_f32: outside the covered set (class transcendental)");
        let jump = i("S_BRANCH", "ENC_SOPP", &[("SIMM16", 1)]);
        let jumped: Vec<u8> = [&mov, &jump, &rcp, &end].into_iter().flatten().copied().collect();
        assert_eq!(check(&jumped), Ok(()), "an s_branch over it leaves it unreachable");

        // the variant decides before the opcode
        assert!(encoding("VOP2_VOP_DPP16").is_some());
        let dpp = i("V_ADD_F32", "VOP2_VOP_DPP16", &[("VDST", 1), ("SRC0", 0xFA), ("VSRC0", 1), ("VSRC1", 2), ("DPP_CTRL", 0x140), ("ROW_MASK", 0xF), ("BANK_MASK", 0xF)]);
        let d = rdna2::decode(&dpp, 0).unwrap();
        assert_eq!(d.variant, Variant::Dpp16);
        assert_eq!(class(&d), "dpp");
    }

    /// The compute pass of a console MS: the exports into its record and GS_ALLOC_REQ are its
    /// own classes, covered there only; another target, message or class keeps its class.
    #[test]
    fn mesh_pass_classes() {
        let exp = |tgt: u64| i("EXP", "ENC_EXP", &[("TGT", tgt), ("EN", 0xF), ("VSRC0", 1), ("VSRC1", 2), ("VSRC2", 3), ("VSRC3", 4)]);
        let msg = |id: u64| i("S_SENDMSG", "ENC_SOPP", &[("SIMM16", id)]);
        let d = |b: &[u8]| rdna2::decode(b, 0).unwrap();
        for t in [9, 12, 13, 15, 20, 32, 63] {
            assert_eq!(mesh_class(&d(&exp(t))), "mesh-export", "target {t}");
            assert_eq!(class(&d(&exp(t))), "export", "target {t} outside the mesh pass");
        }
        for t in [0, 7, 8, 16, 21, 31] {
            assert_eq!(mesh_class(&d(&exp(t))), "export", "target {t}");
        }
        assert_eq!(mesh_class(&d(&msg(9))), "mesh-alloc");
        assert_eq!(class(&d(&msg(9))), "system");
        assert_eq!(mesh_class(&d(&msg(0x22))), "system", "a GS emit is no record write");
        assert!(MESH_TABLE.iter().all(|&(c, _, _)| !TABLE.iter().any(|r| r.0 == c)), "the mesh classes are their own");
        assert!(covered_mesh("mesh-export") && covered_mesh("valu") && !covered("mesh-export") && !covered_mesh("barrier"));

        let end = i("S_ENDPGM", "ENC_SOPP", &[]);
        // M0 = 1 primitive << 12 | 3 vertices
        let m0 = encode("S_MOV_B32", "ENC_SOP1", &[("SDST", 124), ("SSRC0", 255)], Some(1 << 12 | 3)).unwrap();
        let ms: Vec<u8> = [&m0, &msg(9), &exp(12), &exp(32), &exp(20), &end].into_iter().flatten().copied().collect();
        assert_eq!(check_mesh(&ms), Ok(()));
        assert_eq!(check(&ms).unwrap_err(), "s_sendmsg: outside the covered set (class system)", "not a compute program");
        let bar = i("S_BARRIER", "ENC_SOPP", &[]);
        let waits: Vec<u8> = [&m0, &msg(9), &bar, &exp(12), &end].into_iter().flatten().copied().collect();
        assert_eq!(check_mesh(&waits).unwrap_err(), "s_barrier: outside the covered set (class barrier)");
        let mrt: Vec<u8> = [&m0, &exp(0), &end].into_iter().flatten().copied().collect();
        assert_eq!(check_mesh(&mrt).unwrap_err(), "exp: outside the covered set (class export)");
    }

    /// The compute pass of a console GS: MSG_GS of stream 0, MSG_GS_DONE and `s_barrier` are
    /// its own classes, covered there only; the GS code counts from `gs_start` on, as the
    /// translator reaches it through the ES's `s_setpc_b64`.
    #[test]
    fn geometry_pass_classes() {
        let msg = |simm: u64| i("S_SENDMSG", "ENC_SOPP", &[("SIMM16", simm)]);
        let d = |b: &[u8]| rdna2::decode(b, 0).unwrap();
        // emit, cut, emit + cut, nop of stream 0
        for simm in [0x22, 0x12, 0x32, 0x02] {
            assert_eq!(geom_class(&d(&msg(simm))), "gs-emit", "{simm:#x}");
            assert_eq!(class(&d(&msg(simm))), "system", "{simm:#x} outside the GS pass");
        }
        assert_eq!(geom_class(&d(&msg(0x03))), "gs-done");
        assert_eq!(geom_class(&d(&msg(0x122))), "system", "an emit to stream 1: no record of it");
        assert_eq!(geom_class(&d(&msg(9))), "system", "GS_ALLOC_REQ is the mesh pass's");
        assert_eq!(mesh_class(&d(&msg(0x03))), "system");
        let bar = i("S_BARRIER", "ENC_SOPP", &[]);
        assert_eq!(geom_class(&d(&bar)), "gs-barrier");
        assert_eq!(class(&d(&bar)), "barrier");
        assert!(GEOM_TABLE.iter().all(|&(c, _, _)| !TABLE.iter().chain(MESH_TABLE).any(|r| r.0 == c)), "the GS classes are their own");
        assert!(covered_geometry("gs-emit") && covered_geometry("lds") && !covered("gs-barrier") && !covered_mesh("gs-emit"));
        assert!(!covered_geometry("mesh-export") && !covered_geometry("barrier"));

        // ES: its vertex to LDS, the barrier, the jump into the GS; GS: a vertex from LDS, emit, done
        const V: u64 = 256;
        let end = i("S_ENDPGM", "ENC_SOPP", &[]);
        let es: Vec<u8> = [
            i("DS_WRITE_B32", "ENC_DS", &[("ADDR", 1), ("DATA0", 5)]),
            i("S_WAITCNT", "ENC_SOPP", &[]),
            bar.clone(),
            i("S_SETPC_B64", "ENC_SOP1", &[("SSRC0", 6)]),
        ]
        .concat();
        let gs_start = Some(es.len() as u64);
        let read = i("DS_READ_B32", "ENC_DS", &[("ADDR", 0), ("VDST", 2)]);
        let write = i("DS_WRITE_B32", "ENC_DS", &[("ADDR", 3), ("DATA0", 2)]);
        let rcp = i("V_RCP_F32", "ENC_VOP1", &[("VDST", 2), ("SRC0", V + 2)]);
        let gs = |middle: &[u8], emit: u64| [&read[..], middle, &write, &msg(emit), &msg(0x03), &end].concat();
        let ok = [es.clone(), gs(&[], 0x22)].concat();
        assert_eq!(check_geometry(&ok, gs_start), Ok(()));
        assert_eq!(check(&ok).unwrap_err(), "s_barrier: outside the covered set (class barrier)", "not a compute program");
        assert_eq!(check_mesh(&ok).unwrap_err(), "s_barrier: outside the covered set (class barrier)", "not an MS program");
        let slow = [es.clone(), gs(&rcp, 0x22)].concat();
        assert_eq!(check_geometry(&slow, gs_start).unwrap_err(), "v_rcp_f32: outside the covered set (class transcendental)");
        // ES alone: its s_setpc_b64 ends the program, the GS code is not reached
        assert_eq!(check_geometry(&slow, None), Ok(()));
        let streamed = [es.clone(), gs(&[], 0x122)].concat();
        assert_eq!(check_geometry(&streamed, gs_start).unwrap_err(), "s_sendmsg: outside the covered set (class system)");
        let exp = i("EXP", "ENC_EXP", &[("TGT", 12), ("EN", 0xF), ("VSRC0", 1), ("VSRC1", 1), ("VSRC2", 1), ("VSRC3", 1)]);
        let exported = [es, gs(&exp, 0x22)].concat();
        assert_eq!(check_geometry(&exported, gs_start).unwrap_err(), "exp: outside the covered set (class export)");
    }
}
