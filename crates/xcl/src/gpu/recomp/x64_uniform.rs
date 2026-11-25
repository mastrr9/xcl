//! Split wave64 (docs/wave64-on-wave32.md): eliminating `v_readfirstlane_b32` exchanges for
//! wave-uniform VGPRs (e.g. scalar float math and scalar `udiv` on VALU) when `EXEC != 0` is
//! statically guaranteed. Producers in the chain are captured into immutable scalar temporaries
//! evaluated from uniform operands so both 32-lane halves compute the exact same value even
//! when one half has no active lanes.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use crate::gpu::rdna2::{self, Decoded, Variant, F_CONDITIONAL, F_INDIRECT, F_TERMINATOR};

type Mask = u128;
const EXEC_BIT: Mask = 1u128 << 126;

#[derive(Debug, Default, Clone)]
pub struct X64Uniform {
    /// Producer VALU PCs whose scalar uniform result is captured into a temporary.
    pub capture_pcs: BTreeSet<u64>,
    /// `v_readfirstlane_b32` PCs that read the captured uniform temporary directly.
    pub rfl_pcs: BTreeSet<u64>,
}

/// Pure, 32-bit, full-overwrite, lane-independent VALU ops (all take the single `vwrite` path
/// in `emit::Em::valu` and never read `cur`, lane masks, carry-in, or per-lane indices).
fn is_pure_uniform_valu32(name: &str) -> bool {
    matches!(
        name,
        "V_MOV_B32" | "V_MOV_FED_B32" | "V_NOT_B32" | "V_BFREV_B32" | "V_FFBH_U32" | "V_FFBL_B32" | "V_FFBH_I32"
            | "V_CVT_F32_I32" | "V_CVT_F32_U32" | "V_CVT_U32_F32" | "V_CVT_I32_F32" | "V_CVT_RPI_I32_F32" | "V_CVT_FLR_I32_F32"
            | "V_CVT_F32_UBYTE0" | "V_CVT_F32_UBYTE1" | "V_CVT_F32_UBYTE2" | "V_CVT_F32_UBYTE3" | "V_CVT_F32_F16"
            | "V_CVT_PKRTZ_F16_F32" | "V_PACK_B32_F16" | "V_FMA_MIX_F32"
            | "V_ADD_F32" | "V_SUB_F32" | "V_SUBREV_F32" | "V_MUL_F32" | "V_MUL_LEGACY_F32" | "V_FMA_F32" | "V_FMA_LEGACY_F32"
            | "V_MAD_F32" | "V_MAD_LEGACY_F32" | "V_FMAAK_F32" | "V_FMAMK_F32" | "V_MADAK_F32" | "V_MADMK_F32"
            | "V_MIN_F32" | "V_MAX_F32" | "V_MIN3_F32" | "V_MAX3_F32" | "V_MED3_F32"
            | "V_RCP_F32" | "V_RCP_IFLAG_F32" | "V_RSQ_F32" | "V_SQRT_F32" | "V_LOG_F32" | "V_EXP_F32" | "V_SIN_F32" | "V_COS_F32"
            | "V_FLOOR_F32" | "V_CEIL_F32" | "V_TRUNC_F32" | "V_RNDNE_F32" | "V_FRACT_F32"
            | "V_LDEXP_F32" | "V_FREXP_MANT_F32" | "V_FREXP_EXP_I32_F32" | "V_DIV_FIXUP_F32"
            | "V_CUBEID_F32" | "V_CUBESC_F32" | "V_CUBETC_F32" | "V_CUBEMA_F32"
            | "V_AND_B32" | "V_OR_B32" | "V_XOR_B32" | "V_AND_OR_B32" | "V_OR3_B32" | "V_XNOR_B32"
            | "V_BFI_B32" | "V_BFE_U32" | "V_BFE_I32" | "V_ALIGNBIT_B32" | "V_ALIGNBYTE_B32"
            | "V_LSHLREV_B32" | "V_LSHRREV_B32" | "V_ASHRREV_I32" | "V_LSHL_ADD_U32" | "V_LSHL_OR_B32"
            | "V_ADD_NC_U32" | "V_SUB_NC_U32" | "V_SUBREV_NC_U32" | "V_ADD3_U32" | "V_ADD_LSHL_U32"
            | "V_MUL_LO_U32" | "V_MUL_HI_U32" | "V_MUL_HI_I32" | "V_MUL_U32_U24" | "V_MUL_I32_I24"
            | "V_MAD_U32_U24" | "V_MAD_I32_I24" | "V_MIN_U32" | "V_MAX_U32" | "V_MIN_I32" | "V_MAX_I32"
            | "V_MIN3_U32" | "V_MAX3_U32" | "V_MIN3_I32" | "V_MAX3_I32" | "V_MED3_U32" | "V_MED3_I32"
            | "V_PERM_B32" | "V_LERP_U8" | "V_SAD_U8" | "V_SAD_HI_U8" | "V_SAD_U16" | "V_SAD_U32"
    )
}

fn pair_bit(code: u64) -> Mask {
    if code <= 126 && code != 125 && code & 1 == 0 { 1u128 << code } else { 0 }
}

fn written_pairs(d: &Decoded) -> Mask {
    let n = d.inst.name;
    if n.starts_with("S_MOVRELD") || n.starts_with("S_MOVRELSD") {
        return !EXEC_BIT;
    }
    let mut m: Mask = if n.starts_with("V_CMPX_") || n.contains("SAVEEXEC") || n.contains("WREXEC") { EXEC_BIT } else { 0 };
    if d.variant == Variant::Sdwa {
        return if d.enc.name == "ENC_VOPC" && !n.starts_with("V_CMPX_") { !EXEC_BIT } else { m };
    }
    for o in d.ienc.operands {
        if !o.output { continue; }
        match (o.field, o.ty) {
            (_, "OPR_VCC") => m |= pair_bit(106),
            (_, "OPR_EXEC" | "OPR_SDST_EXEC") => m |= EXEC_BIT,
            (Some(f), ty) if !o.implicit && !ty.starts_with("OPR_VGPR") && f != "LITERAL" => {
                let base = rdna2::field_value(d.enc, &d.words, f).unwrap_or(0) * rdna2::register_scale(f);
                for k in 0..(o.size as u64).div_ceil(32).max(1) {
                    if base + k < 128 { m |= pair_bit((base + k) & !1); }
                }
            }
            _ => {}
        }
    }
    m
}

fn is_nonzero_src(nz: Mask, code: u64, lit: Option<u32>) -> bool {
    matches!(code, 129..=208) || (code == 255 && lit.is_some_and(|v| v != 0)) || (nz & pair_bit(code)) != 0
}

/// Forward must-analysis: whether `EXEC != 0` is guaranteed on entry of each instruction.
/// Returns `None` if an unresolved indirect jump is present.
pub fn exec_nonzero(insts: &[Decoded], entry_nonzero: bool) -> Option<Vec<bool>> {
    let n = insts.len();
    if n == 0 { return Some(Vec::new()); }
    let pcs: BTreeMap<u64, usize> = insts.iter().enumerate().map(|(i, d)| (d.pc, i)).collect();
    // edge kind: 0 = normal, 1 = forces EXEC_BIT, 2 = forces EXEC_BIT if pred's scc == (exec != 0)
    let mut succs: Vec<Vec<(usize, u8)>> = vec![Vec::new(); n];
    for (i, d) in insts.iter().enumerate() {
        let next = insts.get(i + 1).and_then(|x| pcs.get(&x.pc).copied());
        if let Some(t) = d.far_target {
            if let Some(&j) = pcs.get(&t) { succs[i].push((j, 0)); }
            continue;
        }
        if d.inst.flags & F_INDIRECT != 0 || matches!(d.inst.name, "S_SETPC_B64" | "S_SWAPPC_B64") {
            return None;
        }
        if let Some(t) = rdna2::branch_target(d) {
            if let Some(&j) = pcs.get(&t) {
                let k = match d.inst.name { "S_CBRANCH_EXECNZ" => 1, "S_CBRANCH_SCC1" => 2, _ => 0 };
                succs[i].push((j, k));
            }
            if d.inst.flags & F_CONDITIONAL != 0 {
                if let Some(j) = next {
                    let k = match d.inst.name { "S_CBRANCH_EXECZ" => 1, "S_CBRANCH_SCC0" => 2, _ => 0 };
                    succs[i].push((j, k));
                }
            }
        } else if d.inst.flags & F_TERMINATOR == 0 {
            if let Some(j) = next { succs[i].push((j, 0)); }
        }
    }
    // Model `LOOP_GUARD` early exits from any loop instruction to the post-loop instruction
    for (a, b) in super::loop_ranges(insts) {
        if let Some((_, &exit_j)) = pcs.range(b..).next() {
            for (i, d) in insts.iter().enumerate() {
                if a <= d.pc && d.pc < b { succs[i].push((exit_j, 0)); }
            }
        }
    }
    let mut reachable = vec![false; n];
    let mut q = VecDeque::from([0usize]);
    reachable[0] = true;
    while let Some(u) = q.pop_front() {
        for &(v, _) in &succs[u] {
            if !reachable[v] { reachable[v] = true; q.push_back(v); }
        }
    }
    let mut preds: Vec<Vec<(usize, u8)>> = vec![Vec::new(); n];
    for (u, edges) in succs.iter().enumerate().filter(|&(u, _)| reachable[u]) {
        for &(v, k) in edges { preds[v].push((u, k)); }
    }
    let kills: Vec<Mask> = insts.iter().map(written_pairs).collect();
    let mut in_nz: Vec<Mask> = reachable.iter().map(|&r| if r { !0 } else { 0 }).collect();
    let mut out_nz = in_nz.clone();
    let mut in_scc = reachable.clone();
    let mut out_scc = reachable.clone();
    loop {
        let mut changed = false;
        for i in 0..n {
            if !reachable[i] { continue; }
            let (inn, iscc) = if i == 0 {
                (if entry_nonzero { EXEC_BIT } else { 0 }, false)
            } else {
                preds[i].iter().fold((!0 as Mask, true), |(nz, sc), &(p, k)| {
                    let edge = out_nz[p] | if k == 1 || (k == 2 && out_scc[p]) { EXEC_BIT } else { 0 };
                    (nz & edge, sc && out_scc[p])
                })
            };
            let d = &insts[i];
            let nm = d.inst.name;
            let fv = |f: &str| rdna2::field_value(d.enc, &d.words, f).unwrap_or(125);
            let mut out = inn & !kills[i];
            if nm == "S_MOV_B64" && is_nonzero_src(inn, fv("SSRC0"), d.literal) {
                out |= pair_bit(fv("SDST"));
            } else if nm == "S_OR_B64" && (is_nonzero_src(inn, fv("SSRC0"), d.literal) || is_nonzero_src(inn, fv("SSRC1"), d.literal)) {
                out |= pair_bit(fv("SDST"));
            } else if nm.contains("SAVEEXEC") {
                let dst = fv("SDST");
                if dst != 126 && (inn & EXEC_BIT) != 0 { out |= pair_bit(dst); }
                if nm == "S_OR_SAVEEXEC_B64" && ((inn & EXEC_BIT) != 0 || is_nonzero_src(inn, fv("SSRC0"), d.literal)) {
                    out |= EXEC_BIT;
                }
            } else if nm == "S_OR_WREXEC_B64" && ((inn & EXEC_BIT) != 0 || is_nonzero_src(inn, fv("SSRC0"), d.literal)) {
                out |= EXEC_BIT | pair_bit(fv("SDST"));
            }
            let sets_exec_scc = nm.contains("SAVEEXEC") || nm.contains("WREXEC")
                || (matches!(nm, "S_AND_B64" | "S_OR_B64" | "S_XOR_B64" | "S_ANDN2_B64" | "S_ORN2_B64" | "S_NAND_B64" | "S_NOR_B64" | "S_XNOR_B64" | "S_NOT_B64")
                    && fv("SDST") == 126);
            let touches_scc = matches!(d.enc.name, "ENC_SOP1" | "ENC_SOP2" | "ENC_SOPC" | "ENC_SOPK")
                && !matches!(nm, "S_MOV_B32" | "S_MOV_B64" | "S_MOVK_I32");
            let oscc = sets_exec_scc || (iscc && (kills[i] & EXEC_BIT) == 0 && !touches_scc);
            if (inn, out, iscc, oscc) != (in_nz[i], out_nz[i], in_scc[i], out_scc[i]) {
                (in_nz[i], out_nz[i], in_scc[i], out_scc[i]) = (inn, out, iscc, oscc);
                changed = true;
            }
        }
        if !changed { return Some(in_nz.into_iter().map(|m| (m & EXEC_BIT) != 0).collect()); }
    }
}

pub fn analyze(insts: &[Decoded], leaders: &BTreeSet<u64>, entry_nonzero: bool) -> X64Uniform {
    let Some(exec_nz) = exec_nonzero(insts, entry_nonzero) else { return X64Uniform::default() };
    let mut out = X64Uniform::default();
    let mut def: [Option<usize>; 256] = [None; 256];
    let mut deps: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (i, d) in insts.iter().enumerate() {
        if leaders.contains(&d.pc) {
            def = [None; 256];
            deps.clear();
        }
        let name = d.inst.name;
        if name == "V_READFIRSTLANE_B32" && d.variant == Variant::Plain {
            let src0 = rdna2::field_value(d.enc, &d.words, "SRC0").unwrap_or(0);
            if exec_nz[i] && (256..512).contains(&src0) {
                if let Some(prod) = def[(src0 - 256) as usize] {
                    out.rfl_pcs.insert(d.pc);
                    let mut stack = vec![prod];
                    while let Some(k) = stack.pop() {
                        if out.capture_pcs.insert(insts[k].pc) {
                            if let Some(ds) = deps.get(&k) { stack.extend(ds.iter().copied()); }
                        }
                    }
                }
            }
            continue;
        }
        if rdna2::branch_target(d).is_some() || d.far_target.is_some() || d.inst.flags & F_TERMINATOR != 0
            || name == "S_BARRIER" || (written_pairs(d) & EXEC_BIT) != 0
        {
            def = [None; 256];
            deps.clear();
            continue;
        }
        if d.variant == Variant::Plain && is_pure_uniform_valu32(name) {
            let (mut dst_v, mut src_deps, mut all_uniform) = (None, Vec::new(), true);
            for o in d.ienc.operands.iter().filter(|o| !o.implicit) {
                if o.output {
                    let v = o.field.and_then(|f| rdna2::field_value(d.enc, &d.words, f)).unwrap_or(256) as usize;
                    if o.ty.starts_with("OPR_VGPR") && o.size <= 32 && v < 256 { dst_v = Some(v); } else { all_uniform = false; }
                    continue;
                }
                if !o.input { continue; }
                let Some(f) = o.field else { all_uniform = false; continue; };
                if f == "LITERAL" { continue; }
                let val = rdna2::field_value(d.enc, &d.words, f).unwrap_or(u64::MAX);
                let is_vgpr = (o.ty.starts_with("OPR_VGPR") && d.enc.field_width(f) <= 8) || (256..512).contains(&val);
                if is_vgpr {
                    let sv = if val >= 256 { (val - 256) as usize } else { val as usize };
                    match def.get(sv).copied().flatten() { Some(p) => src_deps.push(p), None => all_uniform = false }
                } else if !matches!(val, 0..=208 | 235..=248 | 251..=253 | 255) {
                    all_uniform = false;
                }
            }
            if let Some(v) = dst_v {
                def[v] = all_uniform.then_some(i);
                if all_uniform { deps.insert(i, src_deps); }
                continue;
            }
        }
        if name.starts_with("V_MOVREL") || name == "V_SWAPREL_B32" {
            def = [None; 256];
            deps.clear();
            continue;
        }
        for v in rdna2::vgpr_uses(d).1 {
            if let Some(slot) = def.get_mut(v as usize) { *slot = None; }
        }
        if matches!(d.enc.name, "ENC_MUBUF" | "ENC_MTBUF" | "ENC_MIMG" | "ENC_FLAT" | "ENC_FLAT_GLBL" | "ENC_FLAT_SCRATCH" | "ENC_DS") {
            for f in ["VDATA", "VDST", "DATA0", "DATA1"] {
                if let Some(base) = rdna2::field_value(d.enc, &d.words, f) {
                    for k in 0..4usize {
                        if let Some(slot) = def.get_mut(base as usize + k) { *slot = None; }
                    }
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const V: u64 = 256;
    type I = (&'static str, &'static str, &'static [(&'static str, u64)], Option<u32>);
    const ENDPGM: I = ("S_ENDPGM", "ENC_SOPP", &[("SIMM16", 0)], None);

    fn asm(lines: &[I]) -> (Vec<Decoded>, BTreeSet<u64>) {
        let bytes: Vec<u8> = lines.iter().flat_map(|(n, e, f, l)| rdna2::encode(n, e, f, *l).unwrap()).collect();
        let (mut out, mut off) = (Vec::new(), 0usize);
        while off < bytes.len() {
            let d = rdna2::decode(&bytes[off..], off as u64).unwrap();
            off += d.len;
            out.push(d);
        }
        let mut leaders = BTreeSet::from([0u64]);
        for d in &out {
            if let Some(t) = rdna2::branch_target(d) {
                leaders.extend([t, d.pc + d.len as u64]);
            } else if d.inst.flags & F_TERMINATOR != 0 {
                leaders.insert(d.pc + d.len as u64);
            }
        }
        (out, leaders)
    }

    #[test]
    fn scalar_div_chain_with_nonzero_exec_is_captured() {
        let (insts, leaders) = asm(&[
            ("V_CVT_F32_U32", "ENC_VOP1", &[("VDST", 1), ("SRC0", 5)], None),
            ("V_RCP_IFLAG_F32", "ENC_VOP1", &[("VDST", 1), ("SRC0", V + 1)], None),
            ("V_MUL_LEGACY_F32", "ENC_VOP2", &[("VDST", 1), ("SRC0", 255), ("VSRC1", 1)], Some(0x4f7f_fffe)),
            ("V_CVT_U32_F32", "ENC_VOP1", &[("VDST", 1), ("SRC0", V + 1)], None),
            ("V_READFIRSTLANE_B32", "ENC_VOP1", &[("VDST", 6), ("SRC0", V + 1)], None),
            ENDPGM,
        ]);
        let u = analyze(&insts, &leaders, true);
        assert_eq!(u.rfl_pcs, BTreeSet::from([insts[4].pc]));
        assert_eq!(u.capture_pcs, BTreeSet::from([insts[0].pc, insts[1].pc, insts[2].pc, insts[3].pc]));
        assert!(analyze(&insts, &leaders, false).rfl_pcs.is_empty());
    }

    #[test]
    fn cbranch_execz_and_saved_exec_restore_prove_exec_nonzero() {
        let (insts, leaders) = asm(&[
            ("S_AND_SAVEEXEC_B64", "ENC_SOP1", &[("SDST", 10), ("SSRC0", 2)], None),
            ("S_CBRANCH_EXECZ", "ENC_SOPP", &[("SIMM16", 2)], None),
            ("V_CVT_U32_F32", "ENC_VOP1", &[("VDST", 3), ("SRC0", 4)], None),
            ("V_READFIRSTLANE_B32", "ENC_VOP1", &[("VDST", 5), ("SRC0", V + 3)], None),
            ("S_OR_B64", "ENC_SOP2", &[("SDST", 126), ("SSRC0", 126), ("SSRC1", 10)], None),
            ("V_ADD_F32", "ENC_VOP2", &[("VDST", 4), ("SRC0", 6), ("VSRC1", 3)], None),
            ("V_CVT_U32_F32", "ENC_VOP1", &[("VDST", 5), ("SRC0", 6)], None),
            ("V_READFIRSTLANE_B32", "ENC_VOP1", &[("VDST", 7), ("SRC0", V + 5)], None),
            ENDPGM,
        ]);
        let u = analyze(&insts, &leaders, true);
        assert!(u.rfl_pcs.contains(&insts[3].pc) && u.rfl_pcs.contains(&insts[7].pc));
        assert!(u.capture_pcs.contains(&insts[2].pc) && u.capture_pcs.contains(&insts[6].pc));
        assert!(!u.capture_pcs.contains(&insts[5].pc));
    }

    #[test]
    fn non_uniform_exec_maybe_zero_and_loop_exit_stay_conservative() {
        let cases: [&[I]; 4] = [
            &[("S_AND_SAVEEXEC_B64", "ENC_SOP1", &[("SDST", 10), ("SSRC0", 2)], None), ("V_CVT_U32_F32", "ENC_VOP1", &[("VDST", 1), ("SRC0", 5)], None), ("V_READFIRSTLANE_B32", "ENC_VOP1", &[("VDST", 6), ("SRC0", V + 1)], None), ENDPGM],
            &[("V_CNDMASK_B32", "ENC_VOP2", &[("VDST", 1), ("SRC0", 128), ("VSRC1", 2)], None), ("V_READFIRSTLANE_B32", "ENC_VOP1", &[("VDST", 6), ("SRC0", V + 1)], None), ENDPGM],
            &[("V_CVT_U32_F32", "ENC_VOP1", &[("VDST", 1), ("SRC0", 5)], None), ("V_WRITELANE_B32", "ENC_VOP3", &[("VDST", 1), ("SRC0", 7), ("SRC1", 128)], None), ("V_READFIRSTLANE_B32", "ENC_VOP1", &[("VDST", 6), ("SRC0", V + 1)], None), ENDPGM],
            &[("V_CMPX_EQ_U32", "ENC_VOPC", &[("SRC0", 129), ("VSRC1", 0)], None), ("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", 0xFFFF)], None), ("V_CVT_U32_F32", "ENC_VOP1", &[("VDST", 1), ("SRC0", 5)], None), ("V_READFIRSTLANE_B32", "ENC_VOP1", &[("VDST", 6), ("SRC0", V + 1)], None), ENDPGM],
        ];
        for c in cases {
            let (insts, leaders) = asm(c);
            assert!(analyze(&insts, &leaders, true).rfl_pcs.is_empty());
        }
    }
}
