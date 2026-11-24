//! The console's GFX10 (RB+) texture tiling layouts through `gpu::tiling`'s public API: every
//! mode, element size and several pipe configurations, odd and 1-texel sizes. Offsets are checked
//! against addrlib's swizzle equations (gfx10SwizzlePattern.h) and a GF(2)-linear model up to
//! the 16384 maximum coordinate; tile / detile, the mip chains and the detile fast paths against
//! element-by-element readers written here.

use std::panic::{catch_unwind, AssertUnwindSafe};

use xcl::gpu::desc::{Image, ImageType};
use xcl::gpu::tiling::{element, image_chain, image_layout, Chain, Layout, PipeConfig};

/// every SW_MODE `Layout::new` accepts (0 linear, 1 / 2 256 B, 5 / 6 4 KB, 9 / 10 64 KB, 16+ XOR)
const MODES: [u32; 15] = [0, 1, 2, 5, 6, 9, 10, 17, 18, 21, 22, 24, 25, 26, 27];
/// the thick 3D modes (`Layout::new_3d` with `block_d` > 1)
const THICK: [u32; 6] = [5, 9, 17, 21, 25, 26];

const fn cfg(pipes_log2: u32, pkrs_log2: u32) -> PipeConfig {
    PipeConfig { pipes_log2, pkrs_log2 }
}
/// none (XG's own layouts), Series S, Series X (gpu::texture) and two between
const CFGS: [PipeConfig; 5] = [cfg(0, 0), cfg(2, 0), cfg(3, 2), cfg(4, 2), cfg(5, 3)];

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0 >> 33
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n.max(1) as u64) as usize
    }
    fn bytes(&mut self, n: usize) -> Vec<u8> {
        let mut v = Vec::with_capacity(n + 4);
        while v.len() < n {
            v.extend_from_slice(&(self.next() as u32).to_le_bytes());
        }
        v.truncate(n);
        v
    }
}

/// Odd, non-power-of-two and 1-texel surface sizes around a layout's block, area-capped.
fn sizes(l: &Layout, r: &mut Lcg) -> Vec<(usize, usize)> {
    let (bw, bh) = (l.block_w, l.block_h);
    let w = 1 + r.below(2 * bw + 3);
    vec![(1, 1), (bw + 1, 1), (1, bh + 1), ((bw - 1).max(1), 3), (2 * bw + 3, 2), (w, (1 + r.below(2 * bh + 1)).min(3000 / w).max(1))]
}

/// Reads `w` × `h` × `d` elements from `src` at `at(x, y, z)`; bytes past the end read as 0.
fn read_at(src: &[u8], e: usize, (w, h, d): (usize, usize, usize), at: impl Fn(usize, usize, usize) -> usize) -> Vec<u8> {
    let mut out = vec![0u8; w * h * d * e];
    for z in 0..d {
        for y in 0..h {
            for x in 0..w {
                let o = at(x, y, z);
                if let Some(s) = src.get(o..o + e) {
                    out[((z * h + y) * w + x) * e..][..e].copy_from_slice(s);
                }
            }
        }
    }
    out
}

/// Every texel of every mode, element size and pipe configuration (with and without a base
/// swizzle) gets its own element-aligned offset inside `size(w, h)`, for odd and 1-texel sizes.
#[test]
fn offsets_are_a_bijection_into_the_surface() {
    let mut r = Lcg(1);
    for mode in MODES {
        for e in 0..5u32 {
            for c in CFGS {
                let mut l = Layout::new(mode, e, c).unwrap();
                for (w, h) in sizes(&l, &mut r) {
                    for bx in [0, r.below(l.block_bytes) & !((1 << e) - 1)] {
                        l.base_xor = bx;
                        let mut offs: Vec<usize> = (0..h).flat_map(|y| (0..w).map(move |x| (x, y))).map(|(x, y)| l.offset(x, y, w)).collect();
                        let what = format!("mode {mode} e {e} {c:?} {w}x{h} base_xor {bx:#x}");
                        assert!(offs.iter().all(|&o| o % (1 << e) == 0 && o + (1 << e) <= l.size(w, h)), "{what}: misaligned or outside");
                        offs.sort_unstable();
                        offs.dedup();
                        assert_eq!(offs.len(), w * h, "{what}: two texels share an offset");
                    }
                }
            }
        }
    }
}

/// Thick 3D layouts: one layer (`block_d` z-slices) of an odd-sized volume maps one to one onto
/// element-aligned offsets inside `size(w, h)`; the thin 3D modes are the 2D layout.
#[test]
fn thick_offsets_are_a_bijection_into_a_layer() {
    let mut r = Lcg(2);
    for mode in THICK {
        for e in 0..5u32 {
            for c in CFGS {
                let l = Layout::new_3d(mode, e, c).unwrap();
                let (w, h) = (1 + r.below(2 * l.block_w), 1 + r.below(l.block_h + 2));
                let mut offs = Vec::new();
                for z in 0..l.block_d {
                    for y in 0..h {
                        for x in 0..w {
                            offs.push(l.offset_z(x, y, z, w));
                        }
                    }
                }
                let what = format!("mode {mode} e {e} {c:?} {w}x{h}x{}", l.block_d);
                assert!(offs.iter().all(|&o| o % (1 << e) == 0 && o < l.size(w, h)), "{what}: misaligned or outside");
                offs.sort_unstable();
                offs.dedup();
                assert_eq!(offs.len(), w * h * l.block_d, "{what}: two texels share an offset");
            }
        }
        for thin in [0u32, 24, 27] {
            let l = Layout::new_3d(thin, 2, cfg(5, 3)).unwrap();
            assert_eq!((l.block_d, l.block_bytes), (1, Layout::new(thin, 2, cfg(5, 3)).unwrap().block_bytes), "mode {thin}");
        }
    }
}

/// addrlib's swizzle equations (gfx10SwizzlePattern.h, GFX10 RB+ rows, as tiling_tables.rs
/// transcribes them): per address bit, the x (bits 0-15), y (16-31) and z (32-47) coordinate bits
/// XORed into it. Rows with pipe bits from coordinates above the block (the XOR modes on 8 to
/// 32 pipes), whose high-coordinate tables #435 shrank to one period.
/// (2D?, mode, elem_log2, pipe config, block w × h × d, equations)
#[rustfmt::skip]
const ADDRLIB: [(bool, u32, u32, PipeConfig, (usize, usize, usize), &[u64]); 8] = [
    // SW_64KB_R_X, 4 B, Series X
    (true, 27, 2, cfg(5, 3), (128, 128, 1), &[0x0, 0x0, 0x1, 0x2, 0x10000, 0x20000, 0x4, 0x40000, 0x900080, 0x400100010, 0x200200040, 0x100400020, 0x1000140, 0x8, 0x80000, 0x400080]),
    // SW_64KB_Z_X, 1 B, Series S
    (true, 24, 0, cfg(3, 2), (256, 256, 1), &[0x1, 0x10000, 0x2, 0x20000, 0x4, 0x40000, 0x8, 0x80000, 0x500040, 0x200100010, 0x100200020, 0x20, 0x40, 0x400200, 0x1000080, 0x800100]),
    // SW_64KB_S_X, 16 B, Series X
    (true, 25, 4, cfg(5, 3), (64, 64, 1), &[0x0, 0x0, 0x0, 0x0, 0x10000, 0x20000, 0x1, 0x2, 0x40040, 0x400004, 0x400080020, 0x200200008, 0x100100010, 0x10, 0x200000, 0x20]),
    // SW_64KB_D_X, 8 B, 16 pipes 4 packers
    (true, 26, 3, cfg(4, 2), (128, 64, 1), &[0x0, 0x0, 0x0, 0x1, 0x10000, 0x2, 0x4, 0x20000, 0x40040, 0x200008, 0x200080020, 0x100100010, 0x100000, 0x20, 0x200000, 0x40]),
    // SW_4KB_S_X, 2 B, Series X
    (true, 21, 1, cfg(5, 3), (64, 32, 1), &[0x0, 0x1, 0x2, 0x4, 0x10000, 0x20000, 0x40000, 0x8, 0x80080, 0x400010, 0x400100040, 0x200200020]),
    // SW_64KB_S_T, 4 B, Series X
    (true, 17, 2, cfg(5, 3), (128, 128, 1), &[0x0, 0x0, 0x1, 0x2, 0x10000, 0x20000, 0x40000, 0x4, 0x80000, 0x8, 0x100040, 0x400010, 0x200020, 0x20, 0x400000, 0x40]),
    // SW_64KB_D3_X (thick), 4 B, Series X
    (false, 26, 2, cfg(5, 3), (32, 32, 16), &[0x0, 0x0, 0x1, 0x10000, 0x2, 0x100000000, 0x20000, 0x200000000, 0x900080, 0x100010, 0x240040, 0x480020, 0x400000080, 0x4, 0x800000000, 0x8]),
    // SW_4KB_S3_X (thick), 1 B, Series S
    (false, 21, 0, cfg(3, 2), (16, 16, 16), &[0x1, 0x2, 0x100000000, 0x10000, 0x200000000, 0x20000, 0x4, 0x400000000, 0x2000040020, 0x1000100008, 0x800080010, 0x80000]),
];

/// `offset_z` equals addrlib's equations evaluated directly (blocks row-major, `z` placed by the
/// caller) for coordinates across the whole 16384 × 16384 (× 2048 slices) range, so the
/// one-period high-coordinate tables (#435) give what the full 65536-entry ones did.
#[test]
fn xor_offsets_follow_addrlib_equations_up_to_16384() {
    let mut r = Lcg(3);
    for (two_d, mode, e, c, (bw, bh, bd), eq) in ADDRLIB {
        let l = if two_d { Layout::new(mode, e, c) } else { Layout::new_3d(mode, e, c) }.unwrap();
        assert_eq!((l.block_w, l.block_h, l.block_d, l.block_bytes), (bw, bh, bd, 1 << eq.len()), "mode {mode} e {e}");
        let want = |x: usize, y: usize, z: usize, w: usize| {
            let coord = (x as u64 & 0xFFFF) | (y as u64 & 0xFFFF) << 16 | (z as u64 & 0xFFFF) << 32;
            let inblock = eq.iter().enumerate().fold(0, |o, (i, &m)| o | ((m & coord).count_ones() as usize & 1) << i);
            ((y / bh) * w.div_ceil(bw) + x / bw) * l.block_bytes + inblock
        };
        let mut points: Vec<(usize, usize, usize)> = (0..15).flat_map(|i| [((1 << i) - 1, 1 << i, (1 << i) % 2048), (1 << i, (1 << i) - 1, ((1 << i) - 1) % 2048)]).collect();
        points.extend([(16383, 16383, 2047), (16383, 0, 0), (0, 16383, 1), (8191, 12287, 1023)]);
        points.extend((0..4000).map(|_| (r.below(16384), r.below(16384), r.below(2048))));
        for (x, y, z) in points {
            let w = (x + 1 + r.below(16384 - x)).max(x + 1);
            assert_eq!(l.offset_z(x, y, z, w), want(x, y, z, w), "mode {mode} e {e} {c:?} ({x}, {y}, {z}) w {w}");
        }
    }
}

/// Every layout (2D and thick, every pipe configuration) is GF(2)-linear in its coordinates, as
/// addrlib's equations are: the in-block offset of (x, y, z) is the XOR of what each set bit gives
/// alone, up to x, y < 16384 and z < 2048, and repeats every 65536 (the masks are 16 bits).
/// Blocks are row-major; linear rows are pitched to 256 bytes (addrlib HwlComputeSurfaceInfoLinear).
#[test]
fn offsets_are_linear_over_gf2_with_row_major_blocks() {
    let mut r = Lcg(4);
    let mut layouts = Vec::new();
    for c in CFGS {
        for e in 0..5u32 {
            layouts.extend(MODES.iter().map(|&m| (Layout::new(m, e, c).unwrap(), c)));
            layouts.extend(THICK.iter().map(|&m| (Layout::new_3d(m, e, c).unwrap(), c)));
        }
    }
    for (l, c) in layouts {
        let what = format!("mode {} e {} {c:?} block_d {}", l.mode, l.elem_log2, l.block_d);
        let inblock = |x, y, z| l.offset_z(x, y, z, 1 << 16) % l.block_bytes;
        let basis = |f: &dyn Fn(usize) -> usize| -> Vec<usize> { (0..16).map(|i| f(1 << i)).collect() };
        let (bx, by, bz) = (basis(&|v| inblock(v, 0, 0)), basis(&|v| inblock(0, v, 0)), basis(&|v| inblock(0, 0, v)));
        let lin = |b: &[usize], v: usize| (0..16).filter(|i| v >> i & 1 == 1).fold(0, |o, i| o ^ b[i]);
        for _ in 0..150 {
            let (x, y, z, w) = (r.below(16384), r.below(16384), r.below(2048), 1 + r.below(16384));
            let o = l.offset_z(x, y, z, w);
            assert_eq!(o % l.block_bytes, lin(&bx, x) ^ lin(&by, y) ^ lin(&bz, z), "{what}: ({x}, {y}, {z})");
            assert_eq!(o / l.block_bytes, (y / l.block_h) * w.div_ceil(l.block_w) + x / l.block_w, "{what}: block of ({x}, {y}) w {w}");
            assert_eq!(inblock(x + (1 << 16), y + (3 << 16), z + (1 << 16)), o % l.block_bytes, "{what}: ({x}, {y}, {z}) + 65536");
            if l.mode == 0 {
                assert_eq!(o, y * (w << l.elem_log2).next_multiple_of(256) + (x << l.elem_log2), "{what}: linear ({x}, {y}) w {w}");
            }
        }
    }
}

/// `tile` and `detile` are inverses on the bytes that hold texels (and `tile` zeroes the
/// padding); `detile` reads what `offset` names; `Chain::detile_level`'s fast paths (#451:
/// whole linear rows, per-row / per-column XOR split; #602 runs it on the title's bytes in place)
/// give the same bytes for widths and heights that are not a multiple of the block, every slice
/// and a source cut short (inside an element too).
#[test]
fn tile_detile_and_the_fast_paths_agree_with_offsets() {
    let mut r = Lcg(5);
    // random tiled surfaces are cut from one pool: filling each byte by byte is most of the time
    let pool = r.bytes(1 << 21);
    for mode in MODES {
        for e in 0..5u32 {
            for c in [cfg(0, 0), cfg(5, 3)] {
                let es = 1usize << e;
                let base = Layout::new(mode, e, c).unwrap();
                for (k, (w, h)) in sizes(&base, &mut r).into_iter().enumerate() {
                    let what = format!("mode {mode} e {e} {c:?} {w}x{h}");
                    let l = &base;
                    let at = r.below(pool.len() - l.size(w, h));
                    let tiled = pool[at..at + l.size(w, h)].to_vec();
                    let lin = l.detile(&tiled, w, h);
                    assert_eq!(lin, read_at(&tiled, es, (w, h, 1), |x, y, _| l.offset(x, y, w)), "{what}: detile");
                    let mut back = l.tile(&lin, w, h);
                    for (x, y) in (0..h).flat_map(|y| (0..w).map(move |x| (x, y))) {
                        let o = l.offset(x, y, w);
                        assert_eq!(back[o..o + es], tiled[o..o + es], "{what}: tile(detile) at ({x}, {y})");
                        back[o..o + es].fill(0);
                    }
                    assert!(back == vec![0u8; back.len()], "{what}: tile wrote padding");
                    let lin2 = r.bytes(w * h * es);
                    assert_eq!(l.detile(&l.tile(&lin2, w, h), w, h), lin2, "{what}: detile(tile)");
                    // one level of any slice, with no base swizzle (linear: the row-copy path) or one
                    let mut l = base.clone();
                    l.base_xor = if k % 2 == 0 { 0 } else { (r.below(l.block_bytes) & !(es - 1)).max(es) };
                    let chain = Chain::new(l.clone(), w as u32, h as u32, 1, 1);
                    assert_eq!(chain.slice_bytes, l.size(w, h), "{what}");
                    let slice = [0, 1, 7, 1 + r.below(2047)][r.below(4)];
                    for cut in [tiled.len(), tiled.len() - 1, r.below(tiled.len()), (w * es).min(tiled.len()) + 1] {
                        let src = &tiled[..cut.min(tiled.len())];
                        let want = read_at(src, es, (w, h, 1), |x, y, _| l.offset_z(x, y, slice, w));
                        assert_eq!(chain.detile_level(src, 0, slice), want, "{what}: base_xor {:#x} slice {slice} src {cut} B", l.base_xor);
                    }
                }
            }
        }
    }
}

/// Mip chains (full and partial, plain and BCn sizes, 6 array slices / cube faces): levels are
/// D3D12's sizes, every texel of every level and slice has its own aligned offset inside its
/// slice, levels outside the tail start on a block, tail levels lie in the first block, and
/// `tile_level` / `detile_level` round-trip each level of each slice.
#[test]
fn mip_chain_levels_and_slices_never_overlap() {
    let mut r = Lcg(6);
    const SLICES: usize = 6;
    for mode in MODES {
        for e in 0..5u32 {
            for c in [cfg(0, 0), cfg(5, 3)] {
                let side = if e >= 3 && r.below(2) == 0 { 4 } else { 1 };
                let (w, h) = (1 + r.below(100) as u32, 1 + r.below(64) as u32);
                let full = 32 - (w.max(h)).leading_zeros();
                let mips = [full, 1 + r.below(full as usize) as u32][r.below(2)];
                let chain = Chain::new(Layout::new(mode, e, c).unwrap(), w, h, side, mips);
                let l = &chain.layout;
                let (es, sb) = (1usize << e, chain.slice_bytes);
                let what = format!("mode {mode} e {e} {c:?} {w}x{h} side {side} mips {mips}");
                assert_eq!(chain.levels.len(), mips as usize, "{what}");
                let mut offs = Vec::new();
                for (i, lv) in chain.levels.iter().enumerate() {
                    assert_eq!((lv.width, lv.height, lv.depth), (((w >> i).max(1).div_ceil(side)) as usize, ((h >> i).max(1).div_ceil(side)) as usize, 1), "{what}: level {i} size");
                    assert!(lv.origin.0 + lv.width <= lv.pitch && lv.pitch % l.block_w == 0, "{what}: level {i} pitch {}", lv.pitch);
                    match lv.tail {
                        None => assert!(lv.offset % l.block_bytes == 0, "{what}: level {i} at {:#x}", lv.offset),
                        Some(at) => assert!(l.block_bytes > 256 && at < l.block_bytes && lv.offset == 0, "{what}: tail level {i} at {at:#x}"),
                    }
                    for s in 0..SLICES {
                        for y in 0..lv.height {
                            for x in 0..lv.width {
                                let o = chain.element_at_z(lv, x, y, s);
                                assert!(o % es == 0 && o / sb == s, "{what}: level {i} slice {s} ({x}, {y}) at {o:#x}");
                                assert!(lv.tail.is_none() || o % sb < l.block_bytes, "{what}: tail level {i} ({x}, {y}) past the tail block");
                                offs.push(o);
                            }
                        }
                    }
                }
                let n = offs.len();
                offs.sort_unstable();
                offs.dedup();
                assert_eq!(offs.len(), n, "{what}: two texels share an offset");
                // tail levels come last
                assert!(chain.levels.windows(2).all(|p| p[0].tail.is_none() || p[1].tail.is_some()), "{what}: tail order");
                let mut mem = vec![vec![0u8; sb]; SLICES];
                let data: Vec<Vec<Vec<u8>>> = (0..SLICES).map(|_| chain.levels.iter().map(|lv| r.bytes(lv.width * lv.height * es)).collect()).collect();
                for (s, levels) in data.iter().enumerate() {
                    levels.iter().enumerate().for_each(|(i, lin)| chain.tile_level(&mut mem[s], i, s, lin));
                }
                for (s, levels) in data.iter().enumerate() {
                    for (i, lin) in levels.iter().enumerate() {
                        assert_eq!(&chain.detile_level(&mem[s], i, s), lin, "{what}: level {i} slice {s}");
                    }
                }
            }
        }
    }
}

/// Mipped volumes (thick, thin and linear; odd and 1-slice depths; high pipe counts): every
/// texel of every level its own offset inside `slice_bytes * layers()`, `tile_volume` /
/// `detile_volume` round-trip, and a texture cut short reads as zeros past its end.
#[test]
fn volume_chains_round_trip_and_never_overlap() {
    let mut r = Lcg(7);
    for mode in THICK.iter().copied().chain([0, 24, 27]) {
        for e in 0..5u32 {
            let c = [cfg(3, 2), cfg(5, 3)][r.below(2)];
            let (w, h, d) = (1 + r.below(40) as u32, 1 + r.below(40) as u32, [1, 5, 17, 1 + r.below(33) as u32][r.below(4)]);
            let full = 32 - w.max(h).max(d).leading_zeros();
            let chain = Chain::new_3d(Layout::new_3d(mode, e, c).unwrap(), w, h, d, 1, 1 + r.below(full as usize) as u32);
            let es = 1usize << e;
            let total = chain.slice_bytes * chain.layers();
            let what = format!("mode {mode} e {e} {c:?} {w}x{h}x{d} mips {}", chain.levels.len());
            assert_eq!(chain.layers(), (d as usize).div_ceil(chain.layout.block_d), "{what}");
            let mut offs = Vec::new();
            for (i, lv) in chain.levels.iter().enumerate() {
                assert_eq!(lv.depth, (d >> i).max(1) as usize, "{what}: level {i} depth");
                for z in 0..lv.depth {
                    for y in 0..lv.height {
                        for x in 0..lv.width {
                            let o = chain.element_at_z(lv, x, y, z);
                            assert!(o % es == 0 && o + es <= total, "{what}: level {i} ({x}, {y}, {z}) at {o:#x}");
                            offs.push(o);
                        }
                    }
                }
            }
            let n = offs.len();
            offs.sort_unstable();
            offs.dedup();
            assert_eq!(offs.len(), n, "{what}: two texels share an offset");
            let mut mem = vec![0u8; total];
            let data: Vec<Vec<u8>> = chain.levels.iter().map(|lv| r.bytes(lv.width * lv.height * lv.depth * es)).collect();
            data.iter().enumerate().for_each(|(i, lin)| chain.tile_volume(&mut mem, i, lin));
            let cut = r.below(total);
            for (i, lin) in data.iter().enumerate() {
                assert_eq!(&chain.detile_volume(&mem, i), lin, "{what}: level {i}");
                let lv = chain.levels[i];
                let want = read_at(&mem[..cut], es, (lv.width, lv.height, lv.depth), |x, y, z| chain.element_at_z(&lv, x, y, z));
                assert_eq!(chain.detile_volume(&mem[..cut], i), want, "{what}: level {i} cut at {cut}");
            }
        }
    }
}

/// Unknown or reserved modes, elements over 16 B, 2D-only modes for a volume and pipe
/// configurations past the tables give `None`, never a panic (GB_ADDR_CONFIG's fields are 3 bits).
#[test]
fn invalid_layouts_are_none_not_a_panic() {
    for mode in (0..40u32).chain([63, 255, u32::MAX]) {
        for e in (0..9u32).chain([31, u32::MAX]) {
            for p in 0..8 {
                for k in 0..8 {
                    let c = cfg(p, k);
                    let got = catch_unwind(|| (Layout::new(mode, e, c), Layout::new_3d(mode, e, c)));
                    let (two, three) = got.unwrap_or_else(|_| panic!("mode {mode} e {e} {c:?} panicked"));
                    let known = MODES.contains(&mode) && e <= 4;
                    assert!(known || two.is_none(), "mode {mode} e {e} {c:?}: a 2D layout");
                    assert!((known && ![1, 2, 6, 10, 18, 22].contains(&mode)) || three.is_none(), "mode {mode} e {e} {c:?}: a 3D layout");
                    if known && (mode < 16 || CFGS.contains(&c)) {
                        assert!(two.is_some(), "mode {mode} e {e} {c:?}: no 2D layout");
                    }
                    for l in two.iter().chain(three.iter()) {
                        assert_eq!((l.block_w * l.block_h * l.block_d) << e, l.block_bytes, "mode {mode} e {e} {c:?}: block shape");
                    }
                }
            }
        }
    }
}

/// `element`, `image_layout` and `image_chain` on any T#-sized field values: `None` for 96-bit
/// and unknown formats or a mode the image type cannot have, never a panic; what they return
/// is consistent (elements of 1 to 16 B, 4 × 4 only for BCn, base swizzle from the address).
#[test]
fn t_sharp_layouts_never_panic() {
    for f in 0..512u32 {
        if let Some((e, side)) = element(f) {
            assert!(e <= 4 && (side == 1 || (side == 4 && e >= 3)), "format {f}: ({e}, {side})");
        }
    }
    assert!([72u32, 73, 74].iter().all(|&f| element(f).is_none()), "32_32_32 formats are not tiled");
    let mut r = Lcg(8);
    let tys = [ImageType::D1, ImageType::D2, ImageType::D3, ImageType::Cube, ImageType::D1Array, ImageType::D2Array, ImageType::D2Msaa, ImageType::D2MsaaArray];
    for n in 0..3000 {
        let t = Image {
            address: (r.next() << 16 | r.next()) & 0xFFFF_FFFF_FF00,
            format: if n % 2 == 0 { [181, 56, 13, 3, 71, 77, 169][r.below(7)] } else { r.below(512) as u32 },
            width: 1 + r.below(16384) as u32,
            height: 1 + r.below(16384) as u32,
            depth: r.below(2049) as u32,
            ty: tys[r.below(tys.len())],
            base_level: 0,
            last_level: 0,
            max_mip: r.below(16) as u32,
            base_array: 0,
            dst_sel: [4, 5, 6, 7],
            swizzle_mode: if n % 3 == 0 { MODES[r.below(MODES.len())] } else { r.below(32) as u32 },
            min_lod: 0.0,
        };
        let pc = CFGS[r.below(CFGS.len())];
        let (lay, chain) = catch_unwind(AssertUnwindSafe(|| (image_layout(&t, pc), image_chain(&t, pc)))).unwrap_or_else(|_| panic!("{t:?} {pc:?} panicked"));
        let el = element(t.format);
        assert_eq!(lay.is_some(), el.is_some() && Layout::new(t.swizzle_mode, el.unwrap().0, pc).is_some(), "{t:?}");
        if let Some((l, w, h)) = &lay {
            let side = el.unwrap().1;
            assert_eq!((*w, *h, l.base_xor), (t.width.div_ceil(side) as usize, t.height.div_ceil(side) as usize, t.address as usize & (l.block_bytes - 1)), "{t:?}");
        }
        if let Some(c) = chain {
            assert_eq!(c.levels.len(), t.max_mip as usize + 1, "{t:?}");
            assert_eq!(c.layout.base_xor, t.address as usize & (c.layout.block_bytes - 1), "{t:?}");
            assert!(c.levels.iter().all(|l| l.tail.is_some() || l.offset % c.layout.block_bytes == 0), "{t:?}");
        } else if t.ty != ImageType::D3 {
            assert!(lay.is_none(), "{t:?}: a 2D layout without a chain");
        }
    }
}
