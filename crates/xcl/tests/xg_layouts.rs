//! Properties of XCL's console texture layouts (`hle::xg`, its stand-in for `xg_xs.dll`) that
//! hold without the console's own library (`xg_oracle.rs` compares with that one): planes that
//! do not overlap and fit the reported size, chains that agree with `gpu::tiling` built
//! directly, texels with addresses of their own, and descriptors that fail without a panic.

use std::collections::HashSet;
use std::panic::{catch_unwind, AssertUnwindSafe};

use xcl::gpu::texture::{console_pipes, CONSOLE_PIPES, SERIES_S_PIPES};
use xcl::gpu::tiling::{Chain, Layout, PipeConfig};
use xcl::hle::xg::{element, optimal_mode, PlaneKind, Surface, Texture, LAYOUT_BYTES};

const E_NOTIMPL: i32 = 0x8000_4001u32 as i32;
const E_INVALIDARG: i32 = 0x8007_0057u32 as i32;

/// D3D12_RESOURCE_DESC as the console's (0x38 bytes), one sample
#[allow(clippy::too_many_arguments)]
fn desc(dim: u32, width: u64, height: u32, depth: u16, mips: u16, format: u32, layout: u32, flags: u32) -> [u8; 0x38] {
    let mut d = [0u8; 0x38];
    d[0..4].copy_from_slice(&dim.to_le_bytes());
    d[0x10..0x18].copy_from_slice(&width.to_le_bytes());
    d[0x18..0x1C].copy_from_slice(&height.to_le_bytes());
    d[0x1C..0x1E].copy_from_slice(&depth.to_le_bytes());
    d[0x1E..0x20].copy_from_slice(&mips.to_le_bytes());
    d[0x20..0x24].copy_from_slice(&format.to_le_bytes());
    d[0x24..0x28].copy_from_slice(&1u32.to_le_bytes());
    d[0x2C..0x30].copy_from_slice(&layout.to_le_bytes());
    d[0x30..0x34].copy_from_slice(&flags.to_le_bytes());
    d
}

fn tex(d: &[u8; 0x38]) -> Result<Texture, i32> {
    Texture::with_pipes(d, CONSOLE_PIPES)
}

/// levels of a full chain: until the largest side is 1 (D3D12 resource spec, mip levels)
fn full_chain(mut n: u64) -> u32 {
    let mut mips = 1;
    while n > 1 {
        n >>= 1;
        mips += 1;
    }
    mips
}

/// (bytes, alignment) a plane takes
fn extent(k: &PlaneKind) -> (u64, u64) {
    match k {
        PlaneKind::Surface(s) => (s.size(), s.alignment()),
        PlaneKind::Meta { bytes, align } => (*bytes, *align),
        PlaneKind::Blank => (0, 0),
    }
}

/// A surface's chain against one built directly from its mode, element and size, and the mip
/// order `tiling::Chain` documents: the tail block first, then the levels smallest to largest.
fn check_surface(s: &Surface, mips: u32, pipes: PipeConfig) -> Result<(), String> {
    let e = s.bpe.trailing_zeros();
    let c = &s.chain;
    let rebuilt = if s.depth > 0 {
        Chain::new_3d(Layout::new_3d(s.mode, e, pipes).ok_or("no 3D layout")?, s.width, s.height, s.depth, s.side, mips)
    } else {
        Chain::new(Layout::new(s.mode, e, pipes).ok_or("no layout")?, s.width, s.height, s.side, mips)
    };
    if rebuilt.levels != c.levels || rebuilt.slice_bytes != c.slice_bytes || rebuilt.layout.block_bytes != c.layout.block_bytes {
        return Err(format!("chain differs from tiling::Chain (mode {} e {e})", s.mode));
    }
    let l = &c.layout;
    if c.levels.len() != mips as usize || s.bpe != 1 << l.elem_log2 || s.alignment() != l.block_bytes as u64 {
        return Err("levels / element / alignment".into());
    }
    if s.first_tail != c.levels.iter().position(|m| m.tail.is_some()).unwrap_or(c.levels.len()) || c.levels[s.first_tail..].iter().any(|m| m.tail.is_none()) {
        return Err(format!("first_tail {} is not where the tail starts", s.first_tail));
    }
    let slices = if s.depth > 0 { c.layers() } else { s.array as usize };
    if s.size() != (c.slice_bytes * slices) as u64 || !s.size().is_multiple_of(s.alignment()) {
        return Err(format!("size {:#x} != slice {:#x} x {slices}", s.size(), c.slice_bytes));
    }
    let (w0, h0) = (s.width.div_ceil(s.side) as usize, s.height.div_ceil(s.side) as usize);
    let mut end = if s.first_tail < c.levels.len() { l.block_bytes } else { 0 };
    for (i, m) in c.levels.iter().enumerate().rev() {
        let size = ((s.width >> i).max(1).div_ceil(s.side) as usize, (s.height >> i).max(1).div_ceil(s.side) as usize);
        if (m.width, m.height) != size {
            return Err(format!("level {i} is {}x{} elements, not {size:?}", m.width, m.height));
        }
        match m.tail {
            Some(at) if at >= l.block_bytes => return Err(format!("level {i} at {at:#x} past the tail block")),
            Some(_) => {}
            None => {
                if m.offset != end {
                    return Err(format!("level {i} at {:#x}, expected {end:#x}", m.offset));
                }
                end += l.size(w0.div_ceil(1 << i), h0.div_ceil(1 << i));
            }
        }
    }
    if end != c.slice_bytes {
        return Err(format!("levels end at {end:#x}, slice is {:#x}", c.slice_bytes));
    }
    Ok(())
}

/// Everything a texture reports about itself that must agree, for descriptor `d`.
fn check(t: &Texture, d: &[u8; 0x38], pipes: PipeConfig) -> Result<(), String> {
    let u = |o: usize| u32::from_le_bytes(d[o..o + 4].try_into().unwrap());
    let w16 = |o: usize| u16::from_le_bytes(d[o..o + 2].try_into().unwrap()) as u32;
    let (dim, width, height, depth, asked) = (u(0), u64::from_le_bytes(d[0x10..0x18].try_into().unwrap()), u(0x18), w16(0x1C), w16(0x1E));
    let largest = if dim == 4 { width.max(height as u64).max(depth as u64) } else { width.max(height as u64) };
    if t.mips != if asked == 0 { full_chain(largest) } else { asked } {
        return Err(format!("{} mips for {asked} asked of a {largest} texture", t.mips));
    }
    // (a texture of blank planes only reports 0: `alignment_without_planes_is_a_power_of_two`)
    let blank = t.planes.iter().all(|p| matches!(p.kind, PlaneKind::Blank));
    if !(t.alignment().is_power_of_two() || blank) || t.alignment() != t.planes.iter().map(|p| extent(&p.kind).1).max().unwrap() {
        return Err(format!("alignment {:#x}", t.alignment()));
    }
    let end = t.planes.iter().map(|p| p.offset + extent(&p.kind).0).max().unwrap();
    if t.size() != if t.empty { 0 } else { end } {
        return Err(format!("size {:#x}, planes end at {end:#x}", t.size()));
    }
    let mut taken: Vec<(u64, u64)> = Vec::new();
    for (k, p) in t.planes.iter().enumerate() {
        let (bytes, align) = extent(&p.kind);
        // a plane the library leaves out is at 0 and takes no room
        if matches!(p.kind, PlaneKind::Blank) && p.offset != 0 {
            return Err(format!("blank plane {k} at {:#x}", p.offset));
        }
        if align > 0 && p.offset % align != 0 {
            return Err(format!("plane {k} at {:#x}, alignment {align:#x}", p.offset));
        }
        if let Some(&(a, b)) = taken.iter().find(|&&(a, b)| bytes > 0 && p.offset < b && a < p.offset + bytes) {
            return Err(format!("plane {k} [{:#x}, {:#x}) overlaps [{a:#x}, {b:#x})", p.offset, p.offset + bytes));
        }
        taken.push((p.offset, p.offset + bytes));
        match (&p.kind, t.plane_surface(k as u32)) {
            (PlaneKind::Surface(s), Some((q, at))) if std::ptr::eq(s, q) && at == p.offset => {
                check_surface(s, t.mips, pipes).map_err(|e| format!("plane {k}: {e}"))?;
                let (w, h) = (s.width as u64, s.height);
                let (a, z) = if dim == 4 { (1, depth) } else { (depth, 0) };
                if (w, h, s.array, s.depth) != (width, height, a, z) {
                    return Err(format!("plane {k} is {w}x{h} x{} depth {}", s.array, s.depth));
                }
            }
            (PlaneKind::Meta { .. } | PlaneKind::Blank, None) => {}
            _ => return Err(format!("plane_surface({k}) disagrees with planes[{k}]")),
        }
    }
    if t.plane_surface(t.planes.len() as u32).is_some() || t.main >= t.planes.len() {
        return Err("plane past the end / main".into());
    }
    if t.texels().map(|s| s as *const Surface) != t.plane_surface(t.main as u32).map(|(s, _)| s as *const Surface) {
        return Err("texels() is not the main plane".into());
    }
    // the header of XG_RESOURCE_LAYOUT says the same
    let mut b = [0u8; LAYOUT_BYTES];
    t.write(&mut b);
    let q = |o: usize| u64::from_le_bytes(b[o..o + 8].try_into().unwrap());
    let w = |o: usize| u32::from_le_bytes(b[o..o + 4].try_into().unwrap());
    if (q(0), q(8), w(0x10), w(0x14), w(0x1B38)) != (t.size(), t.alignment(), t.mips, t.planes.len() as u32, dim) {
        return Err("XG_RESOURCE_LAYOUT header".into());
    }
    Ok(())
}

/// The corners of every level, first and last slice, lie inside their plane, on an element.
fn check_corners(t: &Texture) -> Result<(), String> {
    for k in 0..t.planes.len() as u32 {
        let Some((s, at)) = t.plane_surface(k) else { continue };
        for (i, m) in s.chain.levels.iter().enumerate() {
            let last = if s.depth > 0 { m.depth } else { s.array as usize } - 1;
            for (x, y, z) in [(0, 0, 0), (m.width - 1, m.height - 1, last), (m.width - 1, 0, last), (0, m.height - 1, 0)] {
                let o = s.element_offset(i, x, y, z);
                if o % s.bpe as u64 != 0 || o + s.bpe as u64 > s.size() {
                    return Err(format!("plane {k} level {i} ({x}, {y}, {z}) at {o:#x} of {:#x}", s.size()));
                }
                if k == t.main as u32 && t.element_offset(i, x, y, z) != at + o {
                    return Err("Texture::element_offset is not the main plane's".into());
                }
            }
            if k == t.main as u32 && t.mip_offset(i) != at + s.mip_offset(i) {
                return Err(format!("mip_offset({i})"));
            }
        }
    }
    Ok(())
}

/// 2D formats of every element size: 128 / 64 / 32 / 16 / 8 bits, BC1 / BC3 / BC4 / BC7.
const FORMATS: [u32; 12] = [2, 10, 16, 28, 24, 41, 87, 49, 54, 61, 71, 98];

/// Colour textures, arrays and cubes in every 2D mode the library takes, with and without
/// render target / UAV flags: every check above holds.
#[test]
fn colour_layouts_are_consistent() {
    let sizes = [(1u64, 1u32), (7, 3), (64, 64), (300, 17), (1024, 1), (1, 513), (1920, 1080)];
    let layouts = [0u32, 1, 2, 0x101, 0x102, 0x105, 0x106, 0x109, 0x10A, 0x111, 0x115, 0x116, 0x119, 0x11A, 0x11B];
    let mut ok = 0;
    for f in FORMATS {
        for (w, h) in sizes {
            for (mips, array) in [(1u16, 1u16), (3, 6), (0, 1), (0, 6)] {
                for layout in layouts {
                    for flags in [0u32, 1, 4] {
                        let d = desc(3, w, h, array, mips, f, layout, flags);
                        let Ok(t) = tex(&d) else { continue };
                        check(&t, &d, CONSOLE_PIPES).and_then(|_| check_corners(&t)).unwrap_or_else(|e| panic!("{d:02x?}: {e}"));
                        ok += 1;
                    }
                }
            }
        }
    }
    assert!(ok > 3000, "only {ok} layouts accepted");
}

/// No two texels of a texture share memory: every element of every level and slice of small
/// textures (2D arrays and 3D volumes, tiled and linear) has its own, element-aligned address
/// inside the texture.
#[test]
fn every_texel_has_its_own_address() {
    let cases = [
        desc(3, 40, 24, 3, 0, 28, 0x109, 0),
        desc(3, 33, 17, 2, 0, 71, 0x105, 0),
        desc(3, 16, 16, 6, 0, 61, 0x11B, 0),
        desc(3, 70, 9, 1, 0, 10, 1, 0),
        desc(3, 128, 4, 1, 0, 2, 0x10A, 0),
        desc(4, 20, 12, 9, 0, 28, 0x105, 0),
        desc(4, 16, 16, 16, 0, 49, 0x115, 0),
        desc(4, 9, 7, 5, 0, 41, 1, 0),
        desc(4, 32, 8, 4, 0, 98, 0x11B, 0),
    ];
    for d in cases {
        let t = tex(&d).unwrap_or_else(|e| panic!("{d:02x?}: {e:#x}"));
        let s = t.texels().unwrap();
        let (at, bpe) = (t.planes[t.main].offset, s.bpe as u64);
        let mut seen = HashSet::new();
        for (i, m) in s.chain.levels.iter().enumerate() {
            let slices = if s.depth > 0 { m.depth } else { s.array as usize };
            for z in 0..slices {
                for y in 0..m.height {
                    for x in 0..m.width {
                        let o = t.element_offset(i, x, y, z);
                        assert!(o >= at && o + bpe <= at + s.size() && (o - at).is_multiple_of(bpe), "{d:02x?}: level {i} ({x}, {y}, {z}) at {o:#x}");
                        assert!(seen.insert(o), "{d:02x?}: level {i} ({x}, {y}, {z}) shares {o:#x}");
                    }
                }
            }
        }
    }
}

/// A linear texture (`Layout` 1) is rows padded to 256 bytes, one after the other (addrlib
/// Gfx10Lib::HwlComputeSurfaceInfoLinear), and slices of the whole padded level.
#[test]
fn linear_is_rows_padded_to_256_bytes() {
    for (w, h, f, array) in [(100u64, 60u32, 28u32, 1u16), (3, 5, 2, 4), (1000, 3, 61, 2), (64, 64, 49, 1)] {
        let t = tex(&desc(3, w, h, array, 1, f, 1, 0)).unwrap();
        let s = t.texels().unwrap();
        let pitch = (w * s.bpe as u64).next_multiple_of(256);
        assert_eq!(s.chain.slice_bytes as u64, pitch * h as u64, "{w}x{h} fmt {f}");
        assert_eq!(t.size(), pitch * h as u64 * array as u64);
        for (x, y, z) in [(0, 0, 0), (w - 1, 0, 0), (1, h - 1, array - 1), (w / 2, h / 2, 0)] {
            assert_eq!(t.element_offset(0, x as usize, y as usize, z as usize), z as u64 * pitch * h as u64 + y as u64 * pitch + x * s.bpe as u64);
        }
    }
}

/// Depth buffers: the HTILE (usage 4, 64 KiB aligned, a byte per 16 pixels of the size padded
/// to 1024 x 1024, per slice) first, then the depth plane (7) and for formats with stencil a
/// 1-byte stencil plane (8), both in 64KB_Z_X. Without the depth flag: depth (5) and stencil
/// (6) planes only.
#[test]
fn depth_stencil_planes() {
    // (format, depth bytes, has stencil): D32_S8X24, D32, D24S8, D16, the console's D16S8
    for (f, depth_bytes, stencil) in [(20u32, 4u32, true), (40, 4, false), (45, 4, true), (55, 2, false), (118, 2, true)] {
        for (w, h, array) in [(1920u64, 1080u32, 1u16), (100, 2000, 2)] {
            let d = desc(3, w, h, array, 1, f, 0, 2);
            let t = tex(&d).unwrap();
            check(&t, &d, CONSOLE_PIPES).unwrap();
            let usages: Vec<u32> = t.planes.iter().map(|p| p.usage).collect();
            assert_eq!(usages, if stencil { vec![4, 7, 8] } else { vec![4, 7] }, "fmt {f}");
            let htile = w.div_ceil(1024) * 1024 * (h.div_ceil(1024) * 1024) as u64 / 16 * array as u64;
            assert!(matches!(t.planes[0].kind, PlaneKind::Meta { bytes, align: 0x1_0000 } if bytes == htile));
            assert_eq!((t.planes[0].offset, t.main), (0, 1));
            let s = t.texels().unwrap();
            assert_eq!((s.bpe, s.mode), (depth_bytes, 24), "fmt {f}");
            if stencil {
                let (st, _) = t.plane_surface(2).unwrap();
                assert_eq!((st.bpe, st.mode), (1, 24));
            }
            // a placed swizzle mode other than 64KB_Z_X is refused for a depth buffer
            assert_eq!(tex(&desc(3, w, h, array, 1, f, 0x109, 2)).err(), Some(E_INVALIDARG));
        }
        if stencil {
            let t = tex(&desc(3, 256, 256, 1, 1, f, 0, 0)).unwrap();
            assert_eq!(t.planes.iter().map(|p| p.usage).collect::<Vec<_>>(), [5, 6], "fmt {f}");
        }
    }
    // a depth buffer in a colour format: the library leaves it empty
    let t = tex(&desc(3, 512, 512, 1, 1, 28, 0, 2)).unwrap();
    assert!(t.empty && t.size() == 0 && t.texels().is_none());
}

/// Render targets of one level get a CMASK after the texels (usage 2, 8 KiB aligned, a byte
/// per 128 pixels of the size padded to 1024 x 1024, per slice); with mips or the 64KB_D choice
/// they do not. A BCn render target keeps only the CMASK.
#[test]
fn render_target_cmask() {
    for (w, h, array) in [(1920u64, 1080u32, 1u16), (1u64, 1u32, 6u16), (4097, 33, 3)] {
        let t = tex(&desc(3, w, h, array, 1, 28, 0, 1)).unwrap();
        assert_eq!(t.planes.iter().map(|p| p.usage).collect::<Vec<_>>(), [1, 2]);
        assert_eq!(t.texels().unwrap().mode, 27); // 64KB_R_X
        let cmask = w.div_ceil(1024) * 1024 * (h.div_ceil(1024) * 1024) as u64 / 128 * array as u64;
        assert!(matches!(t.planes[1].kind, PlaneKind::Meta { bytes, align: 0x2000 } if bytes == cmask));
        assert_eq!(t.planes[1].offset, t.texels().unwrap().size().next_multiple_of(0x2000));
        assert_eq!(t.size(), t.planes[1].offset + cmask);
    }
    for d in [desc(3, 256, 256, 1, 3, 28, 0, 1), desc(3, 256, 256, 1, 1, 28, 2, 1)] {
        assert_eq!(tex(&d).unwrap().planes.len(), 1);
    }
    let t = tex(&desc(3, 256, 256, 1, 1, 71, 0, 1)).unwrap();
    assert!(t.texels().is_none() && matches!(t.planes[1].kind, PlaneKind::Meta { .. }));
    assert_eq!(t.size(), t.planes[1].offset + extent(&t.planes[1].kind).0);
}

/// 3D textures: one plane, `array` 1 and `depth` slices; 4KB_S unless placed, UAVs by the 2D
/// choice over all depth slices' texels; render targets / depth and the 64KB_D choice are not
/// modelled (E_NOTIMPL) and sides over 2048 are refused.
#[test]
fn volumes() {
    for f in FORMATS {
        for (w, h, z) in [(1u64, 1u32, 1u16), (64, 32, 16), (5, 300, 77), (2048, 1, 2048)] {
            for layout in [0u32, 1, 0x105, 0x109, 0x111, 0x115, 0x118, 0x119, 0x11A, 0x11B] {
                for (mips, flags) in [(0u16, 0u32), (1, 4), (2, 0)] {
                    let d = desc(4, w, h, z, mips, f, layout, flags);
                    if let Ok(t) = tex(&d) {
                        check(&t, &d, CONSOLE_PIPES).and_then(|_| check_corners(&t)).unwrap_or_else(|e| panic!("{d:02x?}: {e}"));
                        assert_eq!(t.planes.len(), 1);
                    }
                }
            }
        }
    }
    let mode = |w, h, z, flags| tex(&desc(4, w, h, z, 1, 28, 0, flags)).unwrap().texels().unwrap().mode;
    assert_eq!((mode(256, 256, 64, 0), mode(16, 16, 8, 4), mode(64, 64, 64, 4)), (5, 21, 27));
    for (d, hr) in [
        (desc(4, 64, 64, 8, 1, 28, 0, 1), E_NOTIMPL),
        (desc(4, 64, 64, 8, 1, 40, 0, 2), E_NOTIMPL),
        (desc(4, 64, 64, 8, 1, 28, 2, 0), E_NOTIMPL),
        (desc(4, 2049, 64, 8, 1, 28, 0, 0), E_INVALIDARG),
        (desc(4, 64, 64, 2049, 1, 28, 0, 0), E_INVALIDARG),
        (desc(4, 64, 64, 0, 1, 28, 0, 0), E_INVALIDARG),
        (desc(4, 64, 64, 8, 1, 28, 0x101, 0), E_INVALIDARG), // 256B_S: no 3D pattern
    ] {
        assert_eq!(tex(&d).err(), Some(hr), "{d:02x?}");
    }
}

/// 1D textures: linear by default, only the linear / 64KB_R_X placements, height 1 and no render
/// target or depth flags.
#[test]
fn lines() {
    for layout in [0u32, 1, 0x100, 0x11B] {
        let d = desc(2, 4000, 1, 3, 0, 28, layout, 4);
        let t = tex(&d).unwrap();
        check(&t, &d, CONSOLE_PIPES).unwrap();
        assert_eq!(t.texels().unwrap().mode, if layout == 0x11B { 27 } else { 0 });
    }
    for d in [desc(2, 64, 2, 1, 1, 28, 0, 0), desc(2, 64, 1, 1, 1, 28, 0x109, 0), desc(2, 64, 1, 1, 1, 28, 0, 1), desc(2, 64, 1, 1, 1, 40, 0, 2)] {
        assert_eq!(tex(&d).err(), Some(E_INVALIDARG), "{d:02x?}");
    }
}

/// Descriptors the library cannot lay out give an HRESULT: buffers and unknown dimensions,
/// MSAA, empty sizes, sides over 16384, more than 15 levels, unknown / video / 96-bit formats,
/// unknown layouts and swizzle modes with no 2D pattern (E_INVALIDARG) or not modelled (VAR).
#[test]
fn bad_descriptors_are_refused() {
    let ok = desc(3, 64, 64, 1, 1, 28, 0, 0);
    assert!(tex(&ok).is_ok());
    let mut cases = vec![];
    for dim in [0u32, 1, 5, 0xFFFF_FFFF] {
        cases.push((desc(dim, 64, 64, 1, 1, 28, 0, 0), E_NOTIMPL));
    }
    for dim in [2u32, 3, 4] {
        for samples in [2u32, 4, 8] {
            let mut d = desc(dim, 64, if dim == 2 { 1 } else { 64 }, 1, 1, 28, 0, 0);
            d[0x24..0x28].copy_from_slice(&samples.to_le_bytes());
            cases.push((d, E_NOTIMPL));
        }
    }
    for d in [desc(3, 0, 64, 1, 1, 28, 0, 0), desc(3, 64, 0, 1, 1, 28, 0, 0), desc(3, 64, 64, 0, 1, 28, 0, 0), desc(3, 16385, 1, 1, 1, 28, 0, 0), desc(3, 1, 16385, 1, 1, 28, 0, 0), desc(3, 1 << 40, 1, 1, 1, 28, 0, 0), desc(3, 64, 64, 1, 16, 28, 0, 0)] {
        cases.push((d, E_INVALIDARG));
    }
    // unknown, R32G32B32 (96-bit), R8G8_B8G8 (packed 4:2:2), AYUV / NV12 / P010 / YUY2
    for f in [0u32, 6, 68, 100, 103, 104, 107, 200] {
        cases.push((desc(3, 64, 64, 1, 1, f, 0, 0), E_INVALIDARG));
    }
    for layout in [3u32, 0xFF, 0x103, 0x104, 0x10C, 0x110, 0x113, 0x117, 0x120] {
        cases.push((desc(3, 64, 64, 1, 1, 28, layout, 0), E_INVALIDARG));
    }
    // 64KB_Z_X only for elements up to 8 bytes; VAR_Z_X / VAR_R_X are not modelled
    cases.extend([(desc(3, 64, 64, 1, 1, 2, 0x118, 0), E_INVALIDARG), (desc(3, 64, 64, 1, 1, 71, 0x118, 0), E_INVALIDARG)]);
    cases.extend([(desc(3, 64, 64, 1, 1, 28, 0x11C, 0), E_NOTIMPL), (desc(3, 64, 64, 1, 1, 28, 0x11F, 0), E_NOTIMPL)]);
    // a stencil format with the 64KB_D choice
    cases.push((desc(3, 64, 64, 1, 1, 45, 2, 0), E_INVALIDARG));
    for (d, hr) in cases {
        assert_eq!(tex(&d).err(), Some(hr), "{d:02x?}");
    }
    // the most levels there are: a 16384 side, 15
    assert_eq!(tex(&desc(3, 16384, 16384, 1, 0, 28, 0, 0)).unwrap().mips, 15);
}

/// Every DXGI format (and the console's own) a 2D texture takes is one `element` knows or a
/// console depth + stencil one (118 – 120), and `element` gives D3D's sizes (DXGI_FORMAT
/// documentation: bits per pixel, and BC1 / BC4 8 bytes and BC2 / 3 / 5 / 6H / 7 16 bytes per
/// 4 x 4 block).
#[test]
fn formats_follow_element() {
    for f in 0..=300u32 {
        let r = tex(&desc(3, 64, 64, 1, 1, f, 0x105, 0));
        assert_eq!(r.is_ok(), element(f).is_some() || (118..=120).contains(&f), "format {f}: {:?}", r.err());
    }
    let bits = [(2u32, 128u32), (10, 64), (16, 64), (20, 64), (24, 32), (28, 32), (41, 32), (45, 32), (67, 32), (87, 32), (116, 32), (49, 16), (54, 16), (85, 16), (115, 16), (61, 8), (65, 8)];
    for (f, b) in bits {
        assert_eq!(element(f), Some((b.trailing_zeros() - 3, 1)), "format {f}");
    }
    for (f, block) in [(71u32, 8u32), (80, 8), (74, 16), (77, 16), (83, 16), (95, 16), (98, 16)] {
        assert_eq!(element(f), Some((block.trailing_zeros(), 4)), "format {f}");
    }
}

/// `optimal_mode`: depth 64KB_Z_X first, then 64KB_D for `Layout` 2, render targets 64KB_R_X,
/// UAVs 4KB_S_X up to 64 KiB of texels (bits x pixels <= 0x80000) and 64KB_R_X above, else
/// 4KB_S; the same answer every time, and a mode `tiling::Layout` lays out for every element.
#[test]
fn optimal_mode_choices() {
    assert_eq!(optimal_mode(0, 2 | 1 | 4, 32, 64, 64), 24);
    assert_eq!(optimal_mode(2, 1 | 4, 32, 64, 64), 10);
    assert_eq!(optimal_mode(0, 1 | 4, 32, 64, 64), 27);
    assert_eq!(optimal_mode(0, 0, 32, 16384, 16384), 5);
    assert_eq!((optimal_mode(0, 4, 32, 128, 128), optimal_mode(0, 4, 32, 128, 129)), (21, 27));
    assert_eq!((optimal_mode(0, 4, 8, 256, 256), optimal_mode(0, 4, 128, 64, 64), optimal_mode(0, 4, 128, 65, 64)), (21, 21, 27));
    let mut seen = HashSet::new();
    for layout in [0u32, 1, 2] {
        for flags in 0..8u32 {
            for bits in [4u64, 8, 16, 32, 64, 128] {
                for (w, h) in [(1u64, 1u32), (128, 128), (256, 256), (16384, 16384)] {
                    let m = optimal_mode(layout, flags, bits, w, h);
                    assert_eq!(m, optimal_mode(layout, flags, bits, w, h));
                    seen.insert(m);
                    for e in 0..=4 {
                        assert!(Layout::new(m, e, CONSOLE_PIPES).is_some(), "mode {m} element {e}");
                    }
                }
            }
        }
    }
    assert_eq!(seen, HashSet::from([5, 10, 21, 24, 27]));
}

/// Only the XOR modes depend on the pipe configuration: sizes and chains are the same with the
/// Series S one, and texel addresses too outside the XOR modes; `new` is the default console's.
#[test]
fn pipes_change_only_xor_addresses() {
    assert_eq!(console_pipes(), CONSOLE_PIPES);
    for layout in [1u32, 0x101, 0x105, 0x109, 0x10A, 0x111, 0x115, 0x119, 0x11B] {
        let d = desc(3, 300, 200, 2, 0, 28, layout, 0);
        let (x, s) = (tex(&d).unwrap(), Texture::with_pipes(&d, SERIES_S_PIPES).unwrap());
        check(&s, &d, SERIES_S_PIPES).unwrap();
        assert_eq!(Texture::new(&d).unwrap().size(), x.size());
        assert_eq!((x.size(), x.alignment()), (s.size(), s.alignment()));
        let (a, b) = (x.texels().unwrap(), s.texels().unwrap());
        assert_eq!((&a.chain.levels, a.first_tail), (&b.chain.levels, b.first_tail));
        let differs = (0..300).step_by(7).any(|i| x.element_offset(0, i, i % 200, 1) != s.element_offset(0, i, i % 200, 1));
        assert_eq!(differs, layout & 0x1F >= 16, "layout {layout:#x}");
    }
}

/// Random descriptors (seeded LCG, biased to the fields' edges) never panic, and every one that
/// is laid out passes every check above.
#[test]
fn random_descriptors_never_panic() {
    let mut x = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = move || {
        x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        x >> 16
    };
    let (mut laid_out, mut refused) = (0, 0);
    for n in 0..20000 {
        let pick = |r: u64, v: &[u64]| v[r as usize % v.len()];
        let dim = pick(next(), &[2, 3, 3, 3, 4, 4, 0, 1, 5]) as u32;
        let side = |r: u64, big: u64| match r % 6 {
            0 => 1 + r % 64,
            1 => 1 << (r % 15),
            2 => big - 2 + r % 4,
            3 => r % 2,
            4 => 1 + r % 3000,
            _ => r,
        };
        let width = side(next(), 16384);
        let height = if dim == 2 && next() % 4 != 0 { 1 } else { side(next(), 2048) as u32 };
        let depth = pick(next(), &[1, 1, 2, 6, 7, 0, 2048, 2049, 65535, next() % 40]) as u16;
        let mips = if next() % 8 == 0 { next() as u16 } else { (next() % 18) as u16 };
        let format = match next() % 10 {
            0 => next() as u32,
            1..=4 => (next() % 125) as u32,
            _ => pick(next(), &[2, 10, 16, 28, 24, 41, 87, 49, 54, 61, 71, 98, 20, 40, 45, 55, 118, 119, 120]) as u32,
        };
        let samples = pick(next(), &[1, 1, 1, 1, 1, 1, 0, 2, 4, 8]) as u32;
        let layout = pick(next(), &[0, 0, 0, 1, 2, 3, 0x105, 0x109, 0x11B, 0x100 + next() % 0x20, 0x100 + next() % 0x20, 0x120, next() & 0xFFFF]) as u32;
        let flags = pick(next(), &[0, 0, 0, 1, 4, 2, 3, next() % 8, next()]) as u32;
        let mut d = desc(dim, width, height, depth, mips, format, layout, flags);
        d[0x24..0x28].copy_from_slice(&samples.to_le_bytes());
        // the alignment, sample quality and the console's last field: anything
        for o in (4..0x10).chain(0x28..0x2C).chain(0x34..0x38) {
            d[o] = next() as u8;
        }
        let pipes = if n % 2 == 0 { CONSOLE_PIPES } else { SERIES_S_PIPES };
        let r = catch_unwind(AssertUnwindSafe(|| match Texture::with_pipes(&d, pipes) {
            Ok(t) => check(&t, &d, pipes).and_then(|_| check_corners(&t)).map(|_| true),
            Err(hr) if hr == E_INVALIDARG || hr == E_NOTIMPL => Ok(false),
            Err(hr) => Err(format!("HRESULT {hr:#x}")),
        }));
        match r {
            Ok(Ok(true)) => laid_out += 1,
            Ok(Ok(false)) => refused += 1,
            Ok(Err(e)) => panic!("case {n}, descriptor {d:02x?}: {e}"),
            Err(_) => panic!("case {n}: panic on descriptor {d:02x?}"),
        }
    }
    assert!(laid_out > 1000 && refused > 1000, "{laid_out} laid out, {refused} refused");
}
