//! The string and HRESULT helpers of `util.rs` that the HLE exports build on: GDK-style
//! "string into the caller's buffer" outputs (`write_cstr_out`, `write_wide_out`), the C / UTF-16
//! string readers, the stub DB's return specs, DLL keys and globs. Pure parts only (no code
//! pages, no Wine probe).

use xcl::util::*;

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
}

/// HRESULT_FROM_WIN32 (winerror.h): 0 stays 0, any other Win32 code becomes
/// 0x8007xxxx (severity 1, FACILITY_WIN32 = 7, the code's low 16 bits).
#[test]
fn hresult_from_win32_matches_the_winerror_macro() {
    assert_eq!(hresult_from_win32(0), S_OK);
    // the codes behind util's own E_* constants (winerror.h)
    assert_eq!(hresult_from_win32(14), E_OUTOFMEMORY); // ERROR_OUTOFMEMORY
    assert_eq!(hresult_from_win32(50), E_NOT_SUPPORTED); // ERROR_NOT_SUPPORTED
    assert_eq!(hresult_from_win32(87), E_INVALIDARG); // ERROR_INVALID_PARAMETER
    assert_eq!(hresult_from_win32(122), E_NOT_SUFFICIENT_BUFFER); // ERROR_INSUFFICIENT_BUFFER
    assert_eq!(hresult_from_win32(1168), E_NOT_FOUND); // ERROR_NOT_FOUND
    assert_eq!(hresult_from_win32(5) as u32, 0x8007_0005); // ERROR_ACCESS_DENIED
    assert_eq!(hresult_from_win32(0xFFFF) as u32, 0x8007_FFFF);
    // only the low 16 bits are the code
    assert_eq!(hresult_from_win32(0x0001_0002) as u32, 0x8007_0002);
    let mut r = Lcg(1);
    for _ in 0..1000 {
        let c = r.below(0x1_0000).max(1);
        let h = hresult_from_win32(c);
        assert!(h < 0, "{c:#x}: a failure HRESULT");
        assert_eq!(h as u32 & 0xFFFF, c, "{c:#x}");
        assert_eq!((h as u32 >> 16) & 0x7FF, 7, "{c:#x}: FACILITY_WIN32");
    }
}

/// Return specs from the stub DB and `xcl.toml`: names in any case and with spaces around,
/// hex with either prefix, decimal, and negatives as two's complement of 64 bits.
#[test]
fn parse_ret_accepts_names_hex_decimal_and_negatives() {
    for (s, v) in [
        ("s_ok", 0u64),
        (" S_OK ", 0),
        ("false", 0),
        ("NULL", 0),
        ("True", 1),
        ("S_FALSE", 1),
        ("e_fail", 0x8000_4005),
        ("E_ABORT", 0x8000_4004),
        ("E_NOINTERFACE", 0x8000_4002),
        ("E_INVALIDARG", 0x8007_0057),
        ("E_NOT_SUPPORTED", 0x8007_0032),
        ("0X1f", 31),
        ("0xffffffffffffffff", u64::MAX),
        ("\t42\n", 42),
        ("-0", 0),
        ("-2", u64::MAX - 1),
        ("-9223372036854775808", 1 << 63),
        ("18446744073709551615", u64::MAX),
    ] {
        assert_eq!(parse_ret(s), Some(v), "{s:?}");
    }
    // the named HRESULTs come back as their 32-bit pattern, not sign-extended
    assert_eq!(parse_ret("E_NOTIMPL"), Some(E_NOTIMPL as u32 as u64));
    for s in ["", " ", "0x", "0xg", "-", "--1", "-0x1", "1.5", "18446744073709551616", "0x1_0000", "0x10000000000000000", "S_OKAY"] {
        assert_eq!(parse_ret(s), None, "{s:?}");
    }
    // negatives agree with i64 arithmetic for every value i64 holds
    let mut r = Lcg(2);
    for _ in 0..500 {
        let v = ((r.next() as u64) << 31 | r.next() as u64) as i64;
        assert_eq!(parse_ret(&v.to_string()), Some(v as u64), "{v}");
    }
}

/// `wide` encodes UTF-16 with one terminator; `wcslen` / `from_wide_ptr` read it back up to the
/// first NUL, surrogate pairs intact, a lone surrogate as U+FFFD.
#[test]
fn wide_strings_round_trip() {
    assert_eq!(wide(""), [0]);
    assert_eq!(wide("a😀"), [0x61, 0xD83D, 0xDE00, 0]);
    for s in ["", "kernelx.dll", "C:\\Spiele\\Größe\\game.exe", "漢字", "a😀b𝄞"] {
        let w = wide(s);
        assert_eq!(w.iter().filter(|&&c| c == 0).count(), 1, "{s:?}");
        unsafe {
            assert_eq!(wcslen(w.as_ptr()), w.len() - 1, "{s:?}");
            assert_eq!(from_wide_ptr(w.as_ptr()).as_deref(), Some(s));
        }
    }
    unsafe {
        assert_eq!(from_wide_ptr(std::ptr::null()), None);
        // an embedded NUL ends the string
        let w = wide("ab\0cd");
        assert_eq!(wcslen(w.as_ptr()), 2);
        assert_eq!(from_wide_ptr(w.as_ptr()).as_deref(), Some("ab"));
        let lone = [0x61u16, 0xD800, 0x62, 0];
        assert_eq!(from_wide_ptr(lone.as_ptr()).as_deref(), Some("a\u{FFFD}b"));
    }
}

/// `from_cstr_ptr`: null is None, the bytes up to the NUL, invalid UTF-8 replaced.
#[test]
fn c_strings_read_to_the_nul() {
    unsafe {
        assert_eq!(from_cstr_ptr(std::ptr::null()), None);
        assert_eq!(from_cstr_ptr(b"\0".as_ptr()).as_deref(), Some(""));
        assert_eq!(from_cstr_ptr(b"ab\0cd\0".as_ptr()).as_deref(), Some("ab"));
        assert_eq!(from_cstr_ptr("Größe\0".as_ptr()).as_deref(), Some("Größe"));
        assert_eq!(from_cstr_ptr(b"a\xFFb\0".as_ptr()).as_deref(), Some("a\u{FFFD}b"));
    }
}

/// DLL keys: the file name of a path with either separator, trimmed and lower-cased (ASCII),
/// ".dll" added only when there is no extension; a key is its own key.
#[test]
fn dll_keys_normalise_paths_and_case() {
    assert_eq!(dll_key("C:/Windows/System32/KERNEL32"), "kernel32.dll");
    assert_eq!(dll_key("G:\\Bin/sub\\XGameRuntime.DLL"), "xgameruntime.dll");
    assert_eq!(dll_key("  D3D12_X "), "d3d12_x.dll");
    assert_eq!(dll_key("xgameruntime.thunks"), "xgameruntime.thunks");
    assert_eq!(dll_key("api-ms-win-core-synch-l1-2-0"), "api-ms-win-core-synch-l1-2-0.dll");
    let mut r = Lcg(3);
    let alphabet = ['a', 'B', '.', '\\', '/', ' ', '1', 'Z', '_', '-'];
    for _ in 0..2000 {
        let s: String = (0..r.below(12)).map(|_| alphabet[r.below(alphabet.len() as u32) as usize]).collect();
        let k = dll_key(&s);
        assert_eq!(dll_key(&k), k, "{s:?}");
        assert!(!k.contains(['\\', '/']) && k.contains('.') && !k.bytes().any(|b| b.is_ascii_uppercase()), "{s:?} -> {k:?}");
    }
}

/// Globs with one '*' against a reference: some split of `s` is prefix + anything + suffix.
#[test]
fn glob_match_with_one_star_matches_a_reference() {
    assert!(glob_match("*", ""));
    assert!(glob_match("*", "anything.dll"));
    assert!(glob_match("", ""));
    assert!(!glob_match("", "a"));
    // prefix and suffix must not overlap
    assert!(!glob_match("ab*ba", "aba"));
    assert!(glob_match("ab*ba", "abba"));
    assert!(glob_match("*.dll", "kernelx.dll") && !glob_match("*.dll", "kernelx.dll.bak"));
    fn reference(pattern: &str, s: &str) -> bool {
        let (pre, post) = pattern.split_once('*').unwrap();
        (0..=s.len()).any(|a| (a..=s.len()).any(|b| &s[..a] == pre && &s[b..] == post))
    }
    let mut r = Lcg(4);
    let word = |r: &mut Lcg, n: u32| -> String { (0..r.below(n)).map(|_| if r.below(2) == 0 { 'a' } else { 'b' }).collect() };
    for _ in 0..5000 {
        let (pre, post, s) = (word(&mut r, 4), word(&mut r, 4), word(&mut r, 7));
        let p = format!("{pre}*{post}");
        assert_eq!(glob_match(&p, &s), reference(&p, &s), "{p:?} {s:?}");
        // without a star: equality
        assert_eq!(glob_match(&pre, &s), pre == s, "{pre:?} {s:?}");
    }
}

/// GDK `XUserGetGamertag`-style output: `used` is always the size with the terminator; S_OK
/// and the string NUL-terminated when it fits (exactly the size too), else
/// HRESULT_FROM_WIN32(ERROR_INSUFFICIENT_BUFFER) with the buffer untouched; a null buffer is
/// the size query.
#[test]
fn write_cstr_out_follows_the_gdk_buffer_convention() {
    assert_eq!(E_NOT_SUFFICIENT_BUFFER, hresult_from_win32(122));
    for s in ["", "a", "Player One", "Größe 漢字 😀"] {
        let need = s.len() + 1;
        for size in [0, need.saturating_sub(2), need - 1, need, need + 1, need + 8] {
            let mut buf = vec![0xAAu8; need + 9];
            let mut used = 0x5555usize;
            let hr = unsafe { write_cstr_out(s, buf.as_mut_ptr(), size, &mut used) };
            assert_eq!(used, need, "{s:?} size {size}");
            if size >= need {
                assert_eq!(hr, S_OK, "{s:?} size {size}");
                assert_eq!(&buf[..s.len()], s.as_bytes());
                assert_eq!(buf[s.len()], 0);
                assert!(buf[need..].iter().all(|&b| b == 0xAA), "{s:?} size {size}: wrote past the terminator");
            } else {
                assert_eq!(hr, E_NOT_SUFFICIENT_BUFFER, "{s:?} size {size}");
                assert!(buf.iter().all(|&b| b == 0xAA), "{s:?} size {size}: wrote on failure");
            }
            // no `used`: same result
            let mut buf2 = vec![0xAAu8; need + 9];
            assert_eq!(unsafe { write_cstr_out(s, buf2.as_mut_ptr(), size, std::ptr::null_mut()) }, hr);
            assert_eq!(buf2, buf);
        }
        // the size query: a null buffer of any claimed size
        let mut used = 0usize;
        assert_eq!(unsafe { write_cstr_out(s, std::ptr::null_mut(), 1 << 20, &mut used) }, E_NOT_SUFFICIENT_BUFFER);
        assert_eq!(used, need);
        assert_eq!(unsafe { write_cstr_out(s, std::ptr::null_mut(), 0, std::ptr::null_mut()) }, E_NOT_SUFFICIENT_BUFFER);
    }
}

/// GetModuleFileNameW (MSDN): fits → the length without the terminator; too small → truncated
/// to `size` characters including the NUL, and `size` returned; nothing past `size` written;
/// a null buffer or size 0 writes nothing and returns 0.
#[test]
fn write_wide_out_follows_get_module_file_name_w() {
    for s in ["", "C:\\game.exe", "D:\\Jeux\\Été\\game.exe", "E:\\😀\\𝄞.exe"] {
        let w: Vec<u16> = s.encode_utf16().collect();
        let len = w.len();
        for size in 0..=len as u32 + 3 {
            let mut buf = vec![0xAAAAu16; len + 8];
            let n = unsafe { write_wide_out(s, buf.as_mut_ptr(), size) };
            let size = size as usize;
            if size == 0 {
                assert_eq!(n, 0);
                assert!(buf.iter().all(|&c| c == 0xAAAA), "{s:?}: size 0 wrote");
                continue;
            }
            if size > len {
                assert_eq!(n as usize, len, "{s:?} size {size}");
                assert_eq!(&buf[..len], &w[..]);
                assert_eq!(buf[len], 0);
            } else {
                assert_eq!(n as usize, size, "{s:?} size {size}: truncated");
                assert_eq!(&buf[..size - 1], &w[..size - 1]);
                assert_eq!(buf[size - 1], 0, "{s:?} size {size}: terminated");
            }
            assert!(buf[size..].iter().all(|&c| c == 0xAAAA), "{s:?} size {size}: wrote past the buffer");
        }
        assert_eq!(unsafe { write_wide_out(s, std::ptr::null_mut(), 260) }, 0);
    }
}
