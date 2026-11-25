//! Small pure helpers that whole subsystems sit on: the console's VirtualAlloc / VirtualProtect
//! flags filtered for the PC (`hle::kernelx`), the `__security_check_cookie` finder (`eh`), the
//! keyboard / mouse gamepad mapping (`hle::kbm`), the BVH prebuild size and header bit
//! (`gpu::bvh`) and the preflight checks' verdicts (`preflight::checks`).

use xcl::eh::find_gs_check;
use xcl::gpu::bvh;
use xcl::gpu::recomp::dxc::{Library, Toolchain, OUTDATED_FIX, RECOMMENDED};
use xcl::hle::gameinput::{LEFT_SHOULDER, RIGHT_SHOULDER};
use xcl::hle::kbm::{stick_from_keys, stick_from_mouse, Velocity, Wheel};
use xcl::hle::kernelx::{sanitize_alloc_type, sanitize_protect, MEM_GRAPHICS, MEM_TITLE};
use xcl::preflight::{checks, Check, Found, Gpu, Level};

/// Fixed-seed LCG (Numerical Recipes constants).
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    fn below(&mut self, n: u32) -> u32 {
        self.next() % n
    }
    fn f(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * (self.next() as f32 / (1u64 << 31) as f32)
    }
}

// ---------------------------------------------------------------------------------------
// kernelx: VirtualAlloc flAllocationType / flProtect (Win32 docs: VirtualAlloc, "Memory
// Protection Constants")
// ---------------------------------------------------------------------------------------

const MEM_COMMIT: u32 = 0x1000;
const MEM_RESERVE: u32 = 0x2000;
const MEM_REPLACE_PLACEHOLDER: u32 = 0x4000;
const MEM_RESERVE_PLACEHOLDER: u32 = 0x4_0000;
const MEM_RESET: u32 = 0x8_0000;
const MEM_TOP_DOWN: u32 = 0x10_0000;
const MEM_WRITE_WATCH: u32 = 0x20_0000;
const MEM_PHYSICAL: u32 = 0x40_0000;
const MEM_RESET_UNDO: u32 = 0x100_0000;
const MEM_LARGE_PAGES: u32 = 0x2000_0000;
/// the flags VirtualAlloc on the PC takes from a title as they are
const PC_ALLOC: u32 = MEM_COMMIT | MEM_RESERVE | MEM_RESET | MEM_TOP_DOWN | MEM_WRITE_WATCH | MEM_RESET_UNDO;

const PAGE_NOACCESS: u32 = 0x01;
const PAGE_READONLY: u32 = 0x02;
const PAGE_READWRITE: u32 = 0x04;
const PAGE_EXECUTE_READ: u32 = 0x20;
const PAGE_EXECUTE_READWRITE: u32 = 0x40;
const PAGE_GUARD: u32 = 0x100;
const PAGE_NOCACHE: u32 = 0x200;
const PAGE_WRITECOMBINE: u32 = 0x400;
const PAGE_TARGETS_INVALID: u32 = 0x4000_0000;

fn alloc_model(t: u32) -> u32 {
    match t & PC_ALLOC {
        0 => MEM_COMMIT | MEM_RESERVE,
        s => s,
    }
}

fn protect_model(p: u32) -> u32 {
    let s = p & (0xFF | PAGE_GUARD | PAGE_NOCACHE | PAGE_WRITECOMBINE);
    if s & 0xFF == 0 {
        s | PAGE_READWRITE
    } else {
        s
    }
}

/// Every allocation flag bit alone and in random combinations: the PC's own flags pass,
/// everything else (console graphics / title bits, large and physical pages, placeholders) is
/// dropped, and nothing left means commit + reserve.
#[test]
fn alloc_type_keeps_pc_flags_and_defaults_to_commit_reserve() {
    for bit in 0..32 {
        let t = 1u32 << bit;
        let want = if PC_ALLOC & t != 0 { t } else { MEM_COMMIT | MEM_RESERVE };
        assert_eq!(sanitize_alloc_type(t), want, "bit {bit}");
    }
    assert_eq!(sanitize_alloc_type(0), MEM_COMMIT | MEM_RESERVE);
    assert_eq!(sanitize_alloc_type(MEM_RESET), MEM_RESET);
    assert_eq!(sanitize_alloc_type(MEM_RESERVE | MEM_TOP_DOWN | MEM_GRAPHICS), MEM_RESERVE | MEM_TOP_DOWN);
    assert_eq!(sanitize_alloc_type(MEM_RESERVE | MEM_PHYSICAL), MEM_RESERVE);
    assert_eq!(sanitize_alloc_type(MEM_COMMIT | MEM_RESERVE | MEM_LARGE_PAGES), MEM_COMMIT | MEM_RESERVE);
    assert_eq!(sanitize_alloc_type(MEM_LARGE_PAGES | MEM_PHYSICAL | MEM_COMMIT), MEM_COMMIT); // the console's 64K pages
    assert_eq!(sanitize_alloc_type(MEM_RESERVE_PLACEHOLDER | MEM_REPLACE_PLACEHOLDER | MEM_TITLE), MEM_COMMIT | MEM_RESERVE);
    let mut r = Lcg(10);
    for _ in 0..5000 {
        let t = r.next() ^ (r.next() << 16);
        let s = sanitize_alloc_type(t);
        assert_eq!(s, alloc_model(t), "{t:#x}");
        assert_eq!(sanitize_alloc_type(s), s, "{t:#x}: not idempotent");
    }
}

/// Every protection byte with every modifier set, plus bits above them: the protection byte and
/// GUARD / NOCACHE / WRITECOMBINE pass, the rest is dropped, and no protection left means
/// read-write (with the modifiers kept).
#[test]
fn protect_keeps_the_protection_byte_and_modifiers() {
    for low in 0..=0xFFu32 {
        for m in 0..8u32 {
            for high in [0, 0x800, 0x7_F800, PAGE_TARGETS_INVALID, 0x2000_0000, 0xFFFF_F800] {
                let p = low | m << 8 | high;
                assert_eq!(sanitize_protect(p), protect_model(p), "{p:#x}");
            }
        }
    }
    assert_eq!(sanitize_protect(0), PAGE_READWRITE);
    assert_eq!(sanitize_protect(PAGE_GUARD), PAGE_GUARD | PAGE_READWRITE);
    assert_eq!(sanitize_protect(PAGE_NOCACHE | 0x7_F800), PAGE_NOCACHE | PAGE_READWRITE);
    assert_eq!(sanitize_protect(PAGE_NOACCESS | MEM_GRAPHICS), PAGE_NOACCESS);
    assert_eq!(sanitize_protect(PAGE_EXECUTE_READWRITE | PAGE_TARGETS_INVALID), PAGE_EXECUTE_READWRITE);
    assert_eq!(sanitize_protect(PAGE_READONLY | PAGE_WRITECOMBINE), PAGE_READONLY | PAGE_WRITECOMBINE);
    assert_eq!(sanitize_protect(PAGE_EXECUTE_READ | MEM_TITLE), PAGE_EXECUTE_READ);
}

// ---------------------------------------------------------------------------------------
// eh: MSVC x64 vcruntime `__security_check_cookie`
// ---------------------------------------------------------------------------------------

/// The PC's `__security_check_cookie`, written from its disassembly: `cmp rcx, [rip+cookie]`
/// (48 3B 0D disp32), `jne fail` (75 10), `rol rcx, 16` (48 C1 C1 10), `test cx, 0FFFFh`
/// (66 F7 C1 FF FF), `jne restore` (75 01), `ret` (C3), `ror rcx, 16` (48 C1 C9 10),
/// `jmp __report_gsfailure` (E9 rel32). None: the cookie's address.
fn pc_check() -> Vec<Option<u8>> {
    let mut p = vec![Some(0x48), Some(0x3B), Some(0x0D), None, None, None, None];
    p.extend([0x75, 0x10, 0x48, 0xC1, 0xC1, 0x10, 0x66, 0xF7, 0xC1, 0xFF, 0xFF, 0x75, 0x01, 0xC3, 0x48, 0xC1, 0xC9, 0x10, 0xE9].map(Some));
    p
}

/// The console's: the same with `bnd` (F2) on both branches and the return, so the branch
/// distances grow by one and two.
fn console_check() -> Vec<Option<u8>> {
    let mut p = vec![Some(0x48), Some(0x3B), Some(0x0D), None, None, None, None];
    p.extend([0xF2, 0x75, 0x12, 0x48, 0xC1, 0xC1, 0x10, 0x66, 0xF7, 0xC1, 0xFF, 0xFF, 0xF2, 0x75, 0x02, 0xF2, 0xC3, 0x48, 0xC1, 0xC9, 0x10, 0xE9].map(Some));
    p
}

/// `prefix`, the check (wildcards from `r`), a rel32, `suffix` bytes of int3.
fn code_with(p: &[Option<u8>], prefix: &[u8], suffix: usize, r: &mut Lcg) -> Vec<u8> {
    let mut c = prefix.to_vec();
    c.extend(p.iter().map(|b| b.unwrap_or(r.next() as u8)));
    c.extend((r.next() as i32).to_le_bytes());
    c.extend(std::iter::repeat_n(0xCC, suffix));
    c
}

/// The jump's offset per the comment on `find_gs_check`: the first check (PC form searched
/// first) whose E9 and its whole rel32 lie in `code`.
fn gs_model(code: &[u8]) -> Option<usize> {
    for p in [pc_check(), console_check()] {
        let hit = (0..code.len()).find(|&i| i + p.len() + 4 <= code.len() && p.iter().enumerate().all(|(k, b)| b.is_none_or(|b| code[i + k] == b)));
        if let Some(i) = hit {
            return Some(i + p.len() - 1);
        }
    }
    None
}

/// Both forms are found at the start, in the middle, after any cookie address and rel32, and
/// the offset is the E9; the first of two is the one returned.
#[test]
fn gs_check_found_anywhere_with_any_cookie_address() {
    let mut r = Lcg(20);
    for p in [pc_check(), console_check()] {
        for prefix_len in [0usize, 1, 7, 64] {
            for _ in 0..20 {
                let prefix: Vec<u8> = (0..prefix_len).map(|_| [0xCC, 0x90, 0x48, 0xC3][r.below(4) as usize]).collect();
                let c = code_with(&p, &prefix, 1 + r.below(8) as usize, &mut r);
                let at = find_gs_check(&c);
                assert_eq!(at, Some(prefix_len + p.len() - 1), "{c:02X?}");
                assert_eq!(c[at.unwrap()], 0xE9);
            }
        }
        // two checks: the first
        let mut c = code_with(&p, &[0x90], 3, &mut r);
        let second = code_with(&p, &[], 1, &mut r);
        c.extend(&second);
        assert_eq!(find_gs_check(&c), Some(p.len()));
    }
}

/// Any fixed byte changed (`jmp` turned into `call`, a missing `bnd`, another branch distance)
/// is no check; nor is anything too short to hold one and its rel32, and no slice panics.
#[test]
fn gs_check_near_misses_and_short_slices() {
    let mut r = Lcg(21);
    for p in [pc_check(), console_check()] {
        let good = code_with(&p, &[0xCC; 3], 2, &mut r);
        for k in (0..p.len()).filter(|&k| p[k].is_some()) {
            for flip in [0x01u8, 0x80, 0xFF] {
                let mut c = good.clone();
                c[3 + k] ^= flip;
                assert_eq!(find_gs_check(&c), None, "byte {k} ^ {flip:#x}");
            }
        }
        for n in 0..p.len() + 4 {
            assert_eq!(find_gs_check(&good[3..3 + n]), None, "{n} bytes");
        }
    }
    // the PC form with the console's bnd branch (or the reverse) is neither
    let mut mixed = pc_check();
    mixed.insert(7, Some(0xF2));
    assert_eq!(find_gs_check(&code_with(&mixed, &[], 2, &mut r)), None);
    assert_eq!(find_gs_check(&[]), None);
    assert_eq!(find_gs_check(&[0xE9; 40]), None);
}

/// Mutated checks against the reference: same answer, never a panic, and a hit is an E9 with
/// its rel32 inside the slice.
#[test]
fn gs_check_matches_a_reference_on_mutated_code() {
    let mut r = Lcg(22);
    for round in 0..3000 {
        let p = if round % 2 == 0 { pc_check() } else { console_check() };
        let prefix: Vec<u8> = (0..r.below(12)).map(|_| r.next() as u8).collect();
        let mut c = code_with(&p, &prefix, 1 + r.below(6) as usize, &mut r);
        for _ in 0..r.below(3) {
            let i = r.below(c.len() as u32) as usize;
            c[i] = [0x48, 0xF2, 0x75, 0xE9, 0xC3, r.next() as u8][r.below(6) as usize];
        }
        let c = &c[..];
        let got = std::panic::catch_unwind(|| find_gs_check(c)).unwrap_or_else(|_| panic!("panicked on {c:02X?}"));
        assert_eq!(got, gs_model(c), "{c:02X?}");
        if let Some(j) = got {
            assert!(c[j] == 0xE9 && j + 5 <= c.len());
        }
    }
}

// ---------------------------------------------------------------------------------------
// kbm: keyboard and mouse as the gamepad
// ---------------------------------------------------------------------------------------

/// All 16 key combinations: right - left, up - down, scaled onto the unit circle.
#[test]
fn movement_keys_all_combinations() {
    for m in 0..16u32 {
        let (up, left, down, right) = (m & 1 != 0, m & 2 != 0, m & 4 != 0, m & 8 != 0);
        let (x, y) = ((right as i32 - left as i32) as f32, (up as i32 - down as i32) as f32);
        let len = (x * x + y * y).sqrt();
        let want = if len == 0.0 { (0.0, 0.0) } else { (x / len, y / len) };
        let got = stick_from_keys(up, left, down, right);
        assert!((got.0 - want.0).abs() < 1e-6 && (got.1 - want.1).abs() < 1e-6, "{m:04b}: {got:?} want {want:?}");
    }
}

/// Mouse speed → right stick: points where the mouse moves (screen y flipped), rests at zero
/// speed or sensitivity, grows with speed up to full deflection, and depends on speed ×
/// sensitivity only.
#[test]
fn mouse_deflection_properties() {
    assert_eq!(stick_from_mouse(0.0, 0.0, 1.0), (0.0, 0.0));
    let mut r = Lcg(30);
    for _ in 0..2000 {
        let (a, speed) = (r.f(0.0, std::f32::consts::TAU), r.f(100.0, 20_000.0));
        let (vx, vy) = (a.cos() * speed, a.sin() * speed);
        assert_eq!(stick_from_mouse(vx, vy, 0.0), (0.0, 0.0));
        assert_eq!(stick_from_mouse(vx, vy, -1.0), (0.0, 0.0));
        let s = r.f(0.1, 4.0);
        let (x, y) = stick_from_mouse(vx, vy, s);
        let m = (x * x + y * y).sqrt();
        assert!(m > 0.0 && m <= 1.0 + 1e-6, "{vx} {vy} {s}: {m}");
        // the direction (vx, -vy)
        assert!((x * -vy - y * vx).abs() <= 1e-3 * speed && x * vx - y * vy > 0.0, "{vx} {vy}: {x} {y}");
        // mirrored velocity, mirrored stick
        let (nx, ny) = stick_from_mouse(-vx, -vy, s);
        assert!((nx + x).abs() < 1e-5 && (ny + y).abs() < 1e-5);
        // faster is never less
        let (fx, fy) = stick_from_mouse(vx * 1.5, vy * 1.5, s);
        assert!((fx * fx + fy * fy).sqrt() >= m - 1e-6);
        // twice the speed = twice the sensitivity
        let (kx, ky) = stick_from_mouse(vx * 2.0, vy * 2.0, s);
        let (qx, qy) = stick_from_mouse(vx, vy, s * 2.0);
        assert!((kx - qx).abs() < 1e-4 && (ky - qy).abs() < 1e-4, "{vx} {vy} {s}");
    }
    let (x, y) = stick_from_mouse(-1e7, 0.0, 1.0);
    assert!((x + 1.0).abs() < 1e-6 && y == 0.0);
}

/// The smoothed velocity: an update step outside the clamp acts as the clamp's bound (no
/// division by zero for dt 0), a steady input is approached from one side without overshoot,
/// and the axes do not mix.
#[test]
fn velocity_clamps_dt_and_never_overshoots() {
    let step = |dx, dy, dt| {
        let mut v = Velocity { x: 300.0, y: -50.0 };
        v.update(dx, dy, dt);
        v
    };
    assert_eq!(step(7, -3, 0.0), step(7, -3, -1.0));
    assert!(step(7, -3, 0.0).x.is_finite() && step(i32::MAX, i32::MIN, 0.0).y.is_finite());
    assert_eq!(step(7, -3, 10.0), step(7, -3, 1000.0));
    let mut v = Velocity::default();
    let (tx, ty) = (12.0 / 0.008, -7.0 / 0.008);
    let mut last = v;
    for _ in 0..200 {
        v.update(12, -7, 0.008);
        assert!(v.x >= last.x && v.x <= tx + 1e-2 && v.y <= last.y && v.y >= ty - 1e-2, "{v:?}");
        last = v;
    }
    assert!((v.x - tx).abs() < 1.0 && (v.y - ty).abs() < 1.0, "{v:?}");
    let mut h = Velocity::default();
    for _ in 0..10 {
        h.update(5, 0, 0.004);
    }
    assert_eq!(h.y, 0.0);
}

/// Count the presses (rising edges) of `w` stepped `n` times by `dt`; also checks that only one
/// bumper (or none) is ever held.
fn presses(w: &mut Wheel, n: usize, dt: f32) -> Vec<u32> {
    let (mut out, mut last) = (Vec::new(), 0);
    for _ in 0..n {
        let b = w.step(dt);
        assert!(b == 0 || b == LEFT_SHOULDER || b == RIGHT_SHOULDER, "{b:#x}");
        if b != 0 && b != last {
            out.push(b);
        }
        last = b;
    }
    out
}

/// Wheel notches: at most 8 waiting either way, opposite notches cancel while waiting but not
/// a press under way, a press does not end while no time passes, and at frame rates from 60 to
/// 15 fps each notch is its own press.
#[test]
fn wheel_notches_queue_clamp_and_separate() {
    let mut w = Wheel::default();
    w.notches(100);
    assert_eq!(presses(&mut w, 400, 0.01), vec![RIGHT_SHOULDER; 8]);
    w.notches(5);
    w.notches(5);
    w.notches(-100);
    assert_eq!(presses(&mut w, 400, 0.01), vec![LEFT_SHOULDER; 8]);
    w.notches(3);
    w.notches(-3);
    assert!(presses(&mut w, 100, 0.01).is_empty());
    // up, then down while the up press is held
    w.notches(1);
    assert_eq!(w.step(0.01), RIGHT_SHOULDER);
    w.notches(-1);
    assert_eq!(presses(&mut w, 100, 0.01), vec![RIGHT_SHOULDER, LEFT_SHOULDER]);
    // no time, or time going backwards: the press stays
    w.notches(1);
    assert_eq!(w.step(0.0), RIGHT_SHOULDER);
    assert!((0..1000).all(|_| w.step(0.0) == RIGHT_SHOULDER && w.step(-1.0) == RIGHT_SHOULDER));
    let _ = presses(&mut w, 100, 0.01);
    for fps in [144.0, 60.0, 30.0, 15.0] {
        let mut w = Wheel::default();
        w.notches(-4);
        assert_eq!(presses(&mut w, 200, 1.0 / fps), vec![LEFT_SHOULDER; 4], "{fps} fps");
    }
}

// ---------------------------------------------------------------------------------------
// bvh: prebuild size and the compressed header bit
// ---------------------------------------------------------------------------------------

/// `prebuild_size` is the formula on its doc comment (XboxBvhBuilder.dll's), a whole number of
/// KiB, never shrinking with more leaves, and always room for what `build` / `build_top` make.
#[test]
fn prebuild_size_formula_and_room_for_the_build() {
    for n in 0..3000u64 {
        for (top, leaf) in [(false, 64), (true, 128)] {
            let nodes = n * leaf + (n.max(2) - 1) * 128;
            let want = (256 + nodes + nodes / 64 * 4).div_ceil(1024) * 1024;
            let got = bvh::prebuild_size(n, top);
            assert_eq!(got, want, "{n} {top}");
            assert!(got % 1024 == 0 && got >= 1024);
            if n > 0 {
                assert!(got >= bvh::prebuild_size(n - 1, top));
            }
        }
        assert!(bvh::prebuild_size(n, true) >= bvh::prebuild_size(n, false));
    }
    let empty = bvh::build(&[], 0);
    assert_eq!((empty.root, empty.data.len()), (bvh::INVALID, bvh::HEADER_BYTES));
    assert_eq!(empty.data[bvh::ROOT_AT..bvh::ROOT_AT + 4], bvh::INVALID.to_le_bytes());
    let mut r = Lcg(40);
    for n in 1..=96usize {
        let tris: Vec<bvh::Triangle> = (0..n).map(|_| [0; 3].map(|_| [0; 3].map(|_| r.f(-10.0, 10.0)))).collect();
        let same = vec![[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]; n];
        for t in [&tris, &same] {
            let b = bvh::build(t, 0);
            assert!(b.data.len() as u64 <= bvh::prebuild_size(n as u64, false), "{n} tris: {}", b.data.len());
            assert_eq!(b.data[bvh::ROOT_AT..bvh::ROOT_AT + 4], b.root.to_le_bytes());
        }
        let inst: Vec<bvh::Instance> = (0..n)
            .map(|k| bvh::Instance {
                transform: [[1.0, 0.0, 0.0, r.f(-50.0, 50.0)], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, k as f32]],
                id: k as u32,
                mask: 0xFF,
                contribution: 0,
                flags: 0,
                blas_address: 0x10_0000 + 0x100 * k as u64,
                blas_root: 0x25,
                blas_bounds: ([-1.0; 3], [1.0; 3]),
            })
            .collect();
        let t = bvh::build_top(&inst);
        assert!(t.data.len() as u64 <= bvh::prebuild_size(n as u64, true), "{n} instances: {}", t.data.len());
    }
}

/// Only bit 24 of the little-endian dword 0 counts, and fewer than 4 bytes is no header.
#[test]
fn compressed_header_is_bit_24_of_dword_0() {
    for n in 0..4 {
        assert!(!bvh::is_compressed_header(&[0xFF; 4][..n]), "{n} bytes");
    }
    assert!(bvh::is_compressed_header(&0x0100_0000u32.to_le_bytes()));
    assert!(bvh::is_compressed_header(&[0, 0, 0, 1, 0, 0, 0, 0]));
    assert!(!bvh::is_compressed_header(&0xFEFF_FFFFu32.to_le_bytes()));
    assert!(!bvh::is_compressed_header(&[1, 0, 0, 0])); // big-endian reading would say yes
    assert!(!bvh::is_compressed_header(&[0, 0, 0, 0, 0, 0, 0, 1])); // dword 1
    for bit in 0..32 {
        assert_eq!(bvh::is_compressed_header(&(1u32 << bit).to_le_bytes()), bit == 24, "bit {bit}");
    }
}

// ---------------------------------------------------------------------------------------
// preflight: the verdicts
// ---------------------------------------------------------------------------------------

/// The one place a `Found` is made (PR #775 adds a field to it).
fn found(gpu: Result<Gpu, String>, dxc: Result<String, String>, dxc_toolchain: Toolchain, xgameruntime: bool) -> Found {
    Found { dxc, dxc_toolchain, xgameruntime, gpu }
}

fn toolchain(file_version: &str, major: u32, minor: u32) -> Toolchain {
    Toolchain { dir: r"C:\dxc".into(), dxcompiler: Library { file_version: file_version.into(), major, minor, ..Default::default() }, ..Default::default() }
}

/// What `checks` should report, from the module's rules: level, whether a fix comes with it
/// (None: either way), and a piece of the text.
fn preflight_model(f: &Found, outdated: bool) -> Vec<(Level, Option<bool>, String)> {
    let mut v = Vec::new();
    match &f.gpu {
        Err(e) => v.push((Level::Error, Some(true), e.clone())),
        Ok(g) => {
            v.push(if g.warp { (Level::Warning, Some(true), "WARP".into()) } else { (Level::Ok, Some(false), g.name.clone()) });
            if g.shader_model < (6, 6) {
                v.push((Level::Error, Some(true), format!("has {}.{}", g.shader_model.0, g.shader_model.1)));
            }
            if !g.mesh {
                v.push((Level::Warning, Some(true), "mesh".into()));
            }
            let (w64, w32) = ((g.wave_min..=g.wave_max).contains(&64), (g.wave_min..=g.wave_max).contains(&32));
            if !w64 && w32 {
                v.push((Level::Note, None, g.wave_min.to_string()));
            } else if !w64 && g.wave_max > 0 {
                v.push((Level::Error, None, g.wave_max.to_string()));
            }
        }
    }
    match &f.dxc {
        Ok(dir) => {
            v.push((Level::Ok, Some(false), dir.clone()));
            if outdated {
                v.push((Level::Note, Some(true), format!("{}.{}", RECOMMENDED.0, RECOMMENDED.1)));
            }
        }
        Err(_) => v.push((Level::Error, Some(true), "DXC".into())),
    }
    if !f.xgameruntime {
        v.push((Level::Warning, Some(true), "xgameruntime.dll".into()));
    }
    v
}

/// Every combination of GPU (none, WARP, shader model, mesh shaders, wave widths), DXC (none,
/// old, new, unknown version) and XGameRuntime against the rules: same checks in the same
/// order, fixes where something can be done, and the numbers in the text.
#[test]
fn checks_over_all_input_combinations() {
    // (toolchain, older than RECOMMENDED)
    let dxcs = [(toolchain("1.7.2308.16", 1, 9), true), (toolchain("1.8.2502.8", 1, 8), false), (toolchain("", 1, 7), true), (toolchain("", 0, 0), false)];
    let mut gpus = vec![Err("DXGI: 0x887A0004".to_string())];
    for warp in [false, true] {
        for shader_model in [(5, 1), (6, 5), (6, 6), (6, 7)] {
            for mesh in [false, true] {
                for (wave_min, wave_max) in [(32, 64), (64, 64), (32, 32), (16, 128), (4, 16), (128, 128), (64, 32), (0, 0)] {
                    let name = "Test GPU (8192 MiB VRAM)".to_string();
                    gpus.push(Ok(Gpu { name, warp, shader_model, no_sm66: "no SM 6.6 here".into(), mesh, wave_min, wave_max }));
                }
            }
        }
    }
    let mut seen = 0;
    for gpu in &gpus {
        for dxc in [Ok(r"D:\tools\dxc".to_string()), Err("not found".to_string())] {
            for (t, outdated) in &dxcs {
                for xgr in [true, false] {
                    let f = found(gpu.clone(), dxc.clone(), t.clone(), xgr);
                    let got = checks(&f);
                    let want = preflight_model(&f, *outdated);
                    let ctx = || format!("{f:?}\n{got:#?}");
                    assert_eq!(got.iter().map(|c| c.level).collect::<Vec<_>>(), want.iter().map(|w| w.0).collect::<Vec<_>>(), "{}", ctx());
                    for (c, (_, fix, text)) in got.iter().zip(&want) {
                        if let Some(fix) = fix {
                            assert_eq!(c.fix.is_some(), *fix, "{c:?}\n{}", ctx());
                        }
                        assert!(c.what.contains(text.as_str()), "{c:?} lacks {text:?}");
                    }
                    if let (Ok(g), Some(c)) = (gpu, got.iter().find(|c| c.what.contains("has "))) {
                        assert!(c.what.contains(&g.no_sm66), "{c:?}");
                    }
                    if *outdated && dxc.is_ok() {
                        assert_eq!(got.iter().find(|c| c.level == Level::Note && c.what.contains("DXC")).and_then(|c| c.fix.as_deref()), Some(OUTDATED_FIX));
                    }
                    seen += 1;
                }
            }
        }
    }
    assert_eq!(seen, 129 * 2 * 4 * 2);
}

/// report.json: every level as its lower-case name, no `fix` key on a check without one, and a
/// check written without `fix` reads back.
#[test]
fn checks_serialize_for_the_report() {
    let all = [Level::Ok, Level::Note, Level::Warning, Level::Error];
    let names: Vec<String> = all.iter().map(|l| serde_json::to_string(l).unwrap()).collect();
    assert_eq!(names, ["\"ok\"", "\"note\"", "\"warning\"", "\"error\""]);
    let gpu = Gpu { name: "G".into(), shader_model: (6, 6), mesh: true, wave_min: 32, wave_max: 32, ..Default::default() };
    let c = checks(&found(Ok(gpu), Ok("C:\\dxc".into()), toolchain("1.9.2602.17", 1, 9), false));
    let json = serde_json::to_value(&c).unwrap();
    for (j, c) in json.as_array().unwrap().iter().zip(&c) {
        assert_eq!(j.get("fix").is_some(), c.fix.is_some(), "{j}");
    }
    let back: Check = serde_json::from_str(r#"{"level":"note","what":"x"}"#).unwrap();
    assert_eq!(back, Check { level: Level::Note, what: "x".into(), fix: None });
    assert!(serde_json::from_str::<Check>(r#"{"level":"fatal","what":"x"}"#).is_err());
}
