//! MicrosoftGame.config as XCL reads it: the manifest settings (`manifest::parse` / `load`),
//! the launcher's executables, identity and display name, and how `resolve_target` picks the
//! package root, the game exe and the title (the data folder's name) from a folder or an exe.

use std::fs;
use std::panic::catch_unwind;
use std::path::{Path, PathBuf};
use xcl::launcher::{display_name, read_game_config, resolve_target};
use xcl::manifest::{self, Manifest};

/// A fresh folder of this test's own under the temp folder, removed on drop.
struct TempDir(PathBuf);

impl TempDir {
    fn new(test: &str) -> TempDir {
        let p = std::env::temp_dir().join(format!("xcl_game_config_{test}_{}", std::process::id()));
        let _ = fs::remove_dir_all(&p);
        fs::create_dir_all(&p).unwrap();
        TempDir(p)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn utf16(bom: [u8; 2], text: &str, little_endian: bool) -> Vec<u8> {
    let mut raw = bom.to_vec();
    raw.extend(text.encode_utf16().flat_map(|u| if little_endian { u.to_le_bytes() } else { u.to_be_bytes() }));
    raw
}

fn canon(p: &Path) -> PathBuf {
    fs::canonicalize(p).unwrap()
}

/// Element lookup: a longer name with the same start, a self-closing element and attributes
/// on the opening tag do not throw it off; values are trimmed, and a size that is not a plain
/// number (or an element that is never closed) is no value.
#[test]
fn manifest_elements_are_found_by_their_exact_name() {
    let xml = "<Game>\n  <TitleIdentifier>FFFFFFFF</TitleIdentifier>\n  <TitleId/>\n  <TitleId kind=\"hex\">\n    1A2B3C4D\n  </TitleId>\n\
               <PersistentLocalStorage Mode=\"x\"><SizeMB>\t 2048 \n</SizeMB></PersistentLocalStorage></Game>";
    let m = manifest::parse(xml);
    assert_eq!((m.title_id.as_deref(), m.pls_mb), (Some("1A2B3C4D"), Some(2048)));
    for size in ["2048MB", "-1", "0x800", "1.5", "", "18446744073709551616"] {
        let xml = format!("<PersistentLocalStorage><SizeMB>{size}</SizeMB></PersistentLocalStorage>");
        assert_eq!(manifest::parse(&xml).pls_mb, None, "{size}");
    }
    assert_eq!(manifest::parse("<PersistentLocalStorage><SizeMB>18446744073709551615</SizeMB></PersistentLocalStorage>").pls_mb, Some(u64::MAX));
    assert_eq!(manifest::parse("<Game><TitleId>1A2B3C4D</Game>").title_id, None);
    assert_eq!(manifest::parse("<XboxAnacondaTitleMemory>"), Manifest::default());
}

/// `manifest::load` finds the file in any case of its name and reads UTF-8 (with or without a
/// BOM) and BOM-marked UTF-16; no folder, no file, an odd UTF-16 byte count and bytes that are
/// not UTF-8 are no manifest, and BOM-less UTF-16 (not XML, XML 1.0 §4.3.3) reads nothing.
#[test]
fn manifest_load_encodings_and_file_names() {
    let dir = TempDir::new("manifest_load");
    let xml = "<Game><TitleId>1A2B3C4D</TitleId><Resources><Resource Language=\"de-DE\"/></Resources></Game>";
    let want = Manifest { title_id: Some("1A2B3C4D".into()), languages: vec!["de-DE".into()], ..Default::default() };
    assert_eq!(manifest::load(&dir.0.join("missing")), None);
    assert_eq!(manifest::load(&dir.0), None);
    let file = dir.0.join("microsoftgame.CONFIG");
    let mut bom8 = b"\xEF\xBB\xBF".to_vec();
    bom8.extend(xml.as_bytes());
    for raw in [xml.as_bytes().to_vec(), bom8, utf16([0xFF, 0xFE], xml, true), utf16([0xFE, 0xFF], xml, false)] {
        fs::write(&file, &raw).unwrap();
        assert_eq!(manifest::load(&dir.0).as_ref(), Some(&want), "{:02X?}", &raw[..4]);
    }
    let mut odd = utf16([0xFF, 0xFE], xml, true);
    odd.pop();
    for raw in [odd, b"<Game><TitleId>\xC0\xAF</TitleId></Game>".to_vec(), vec![0xFF, 0xFE, 0x00, 0xDC]] {
        fs::write(&file, &raw).unwrap();
        assert_eq!(manifest::load(&dir.0), None, "{raw:02X?}");
    }
    fs::write(&file, xml.encode_utf16().flat_map(u16::to_le_bytes).collect::<Vec<_>>()).unwrap();
    assert_eq!(manifest::load(&dir.0).unwrap_or_default(), Manifest::default());
}

/// Comments hide what is in them in the manifest too: an unterminated one hides the rest of
/// the file, an empty one hides nothing.
#[test]
fn manifest_comments_hide_settings() {
    let dir = TempDir::new("manifest_comments");
    let file = dir.0.join("MicrosoftGame.config");
    fs::write(&file, "<Game><!----><TitleId>1A2B3C4D</TitleId><!-- -- <XboxAnacondaTitleMemory>Advanced</XboxAnacondaTitleMemory> --></Game>").unwrap();
    let m = manifest::load(&dir.0).unwrap();
    assert_eq!((m.title_id.as_deref(), m.anaconda_memory), (Some("1A2B3C4D"), None));
    fs::write(&file, "<Game><TitleId>1A2B3C4D</TitleId><!-- <PersistentLocalStorage><SizeMB>1</SizeMB></PersistentLocalStorage>").unwrap();
    assert_eq!(manifest::load(&dir.0).unwrap(), Manifest { title_id: Some("1A2B3C4D".into()), ..Default::default() });
}

/// The display name of a real-world multi-line `<ShellVisuals>`: attributes in any order, a
/// longer attribute name ending in the same word or a longer tag name are not it, references
/// are replaced; blank or missing is None. UTF-16 BE is read too.
#[test]
fn display_name_edges() {
    let dir = TempDir::new("display_name");
    let file = dir.0.join("MicrosoftGame.config");
    let shell = |attrs: &str| {
        format!(
            "<Game>\n  <ShellVisualsExtra DefaultDisplayName=\"wrong tag\"/>\n  <ShellVisuals\n\tSquare150x150Logo=\"a.png\"\n\
             \tXDefaultDisplayName=\"wrong attr\"\n\t{attrs}\n  />\n</Game>"
        )
    };
    fs::write(&file, shell("DefaultDisplayName=\"Caf&#233; &#x2603; &lt;3\"")).unwrap();
    assert_eq!(display_name(&dir.0).as_deref(), Some("Café \u{2603} <3"));
    fs::write(&file, utf16([0xFE, 0xFF], &shell("DefaultDisplayName=\"雪 Game\""), false)).unwrap();
    assert_eq!(display_name(&dir.0).as_deref(), Some("雪 Game"));
    for attrs in ["DefaultDisplayName=\" \t\"", "DefaultDisplayName=\"ms-resource:Resources/AppName\"", "Description=\"no name\""] {
        fs::write(&file, shell(attrs)).unwrap();
        assert_eq!(display_name(&dir.0), None, "{attrs}");
    }
    fs::write(&file, "<Game><Identity Name=\"Contoso.Game\"/></Game>").unwrap();
    assert_eq!(display_name(&dir.0), None);
}

/// Executables in manifest order, only from `<Executable>` elements that name one (not from
/// `<ExecutableList>` or comments); the first `<Identity>` that is not a longer tag name.
#[test]
fn read_game_config_executables_and_identity() {
    let dir = TempDir::new("read_config");
    assert_eq!(read_game_config(&dir.0), (Vec::new(), None));
    let xml = "<Game>\n<IdentityResource Name=\"wrong\"/>\n<Identity\n  Publisher=\"CN=Contoso\"\n  Name=\"Contoso.Game\"/>\n<Identity Name=\"second\"/>\n\
               <ExecutableList Name=\"list\">\n  <Executable Id=\"NoName\"/>\n  <Executable Name=\"b.exe\" Id=\"B\"></Executable>\n\
               <!-- <Executable Name=\"old.exe\"/> -->\n  <Executable\tId=\"A\"\tName=\"Q&amp;A.exe\"/>\n</ExecutableList></Game>";
    fs::write(dir.0.join("MicrosoftGame.config"), xml).unwrap();
    assert_eq!(read_game_config(&dir.0), (vec!["b.exe".to_string(), "Q&A.exe".to_string()], Some("Contoso.Game".to_string())));
}

/// Random markup and truncated files (UTF-8 and UTF-16 with odd byte counts) never panic the
/// manifest or launcher parsers.
#[test]
fn malformed_game_configs_never_panic() {
    const TOKENS: &[&str] = &[
        "<", ">", "/", "\"", "'", "=", " ", "\n", "<!--", "-->", "&", "&#", "&#x", ";", "Executable", "Name", "Identity", "ShellVisuals",
        "DefaultDisplayName", "TitleId", "SizeMB", "PersistentLocalStorage", "Resources", "Resource", "Language", "é", "雪", "1",
    ];
    let dir = TempDir::new("fuzz");
    let file = dir.0.join("MicrosoftGame.config");
    let check = |what: &str, raw: &[u8]| {
        fs::write(&file, raw).unwrap();
        let root = dir.0.clone();
        if catch_unwind(move || (read_game_config(&root), display_name(&root), manifest::load(&root))).is_err() {
            panic!("{what}: {:?}", String::from_utf8_lossy(raw));
        }
    };
    let mut seed: u64 = 0x5EED_0C0F_1600_0001;
    let mut next = move |n: usize| {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((seed >> 33) as usize) % n
    };
    for i in 0..300 {
        let len = next(40);
        let xml: String = (0..len).map(|_| TOKENS[next(TOKENS.len())]).collect();
        let x = xml.clone();
        assert!(catch_unwind(move || manifest::parse(&x)).is_ok(), "parse: {xml:?}");
        if i % 3 == 0 {
            check("utf-8", xml.as_bytes());
            check("utf-16", &utf16([0xFF, 0xFE], &xml, true));
        }
    }
    let full = "<Game><Identity Name=\"A&amp;B\"/><ExecutableList><Executable Name=\"雪.exe\"/></ExecutableList>\
                <ShellVisuals DefaultDisplayName=\"&#x2603;\"/><!-- c --><TitleId>1</TitleId></Game>";
    for (at, _) in full.char_indices() {
        let x = full[..at].to_string();
        assert!(catch_unwind(move || manifest::parse(&x)).is_ok(), "parse: {:?}", &full[..at]);
    }
    let wide = utf16([0xFE, 0xFF], full, false);
    for end in 0..=wide.len() {
        check("utf-16 prefix", &wide[..end]);
    }
}

/// A folder without a manifest: one exe at its top wins over subfolders; with none there, up to
/// two folder levels down are searched (not three); several candidates, a folder named `.exe`
/// and a missing target are errors that say what they found. The title is the folder's name
/// with everything but letters, digits, '.', '_' and '-' made '_'.
#[test]
fn resolve_target_picks_the_exe_of_a_folder() {
    let dir = TempDir::new("folder");
    let root = dir.0.join("My Game (2024)");
    fs::create_dir_all(root.join("a").join("b").join("c")).unwrap();
    fs::create_dir_all(root.join("not.exe")).unwrap();
    fs::write(root.join("a").join("b").join("c").join("deep.exe"), b"MZ").unwrap();
    let e = resolve_target(&root).unwrap_err();
    assert!(e.contains("no game .exe"), "{e}");
    fs::write(root.join("a").join("b").join("GAME.EXE"), b"MZ").unwrap();
    let (r, exe, title) = resolve_target(&root).unwrap();
    assert_eq!((canon(&r), canon(&exe)), (canon(&root), canon(&root.join("a").join("b").join("GAME.EXE"))));
    assert_eq!(title, "My_Game__2024_");
    fs::write(root.join("a").join("other.exe"), b"MZ").unwrap();
    let e = resolve_target(&root).unwrap_err();
    assert!(e.contains("GAME.EXE") && e.contains("other.exe"), "{e}");
    fs::write(root.join("top.exe"), b"MZ").unwrap();
    assert_eq!(canon(&resolve_target(&root).unwrap().1), canon(&root.join("top.exe")));
    let e = resolve_target(&dir.0.join("missing_folder")).unwrap_err();
    assert!(e.contains("missing_folder"), "{e}");
}

/// With a manifest: the first named exe that exists is the game (other exes in the folder do
/// not matter) and the title is the sanitized `<Identity Name>`; given an exe directly, the
/// package root is the nearest folder above whose manifest names that exe (Unreal's
/// `Binaries\XSX`), else the exe's own folder.
#[test]
fn resolve_target_follows_the_manifest() {
    let dir = TempDir::new("manifest_root");
    let root = dir.0.join("package");
    let bin = root.join("Game").join("Binaries").join("XSX");
    fs::create_dir_all(&bin).unwrap();
    fs::create_dir_all(root.join("dir.exe")).unwrap();
    fs::write(bin.join("game.exe"), b"MZ").unwrap();
    fs::write(bin.join("tool.exe"), b"MZ").unwrap();
    fs::write(root.join("launcher.exe"), b"MZ").unwrap();
    // forward slashes, so the name resolves on any host; real manifests use '\'
    fs::write(
        root.join("MicrosoftGame.config"),
        "<Game><Identity Name=\"Contoso Game/Ü\"/><ExecutableList><Executable Name=\"missing.exe\"/><Executable Name=\"dir.exe\"/>\
         <Executable Name=\"Game/Binaries/XSX/game.exe\"/></ExecutableList></Game>",
    )
    .unwrap();
    let (r, exe, title) = resolve_target(&root).unwrap();
    assert_eq!((canon(&r), canon(&exe)), (canon(&root), canon(&bin.join("game.exe"))));
    assert_eq!(title, "Contoso_Game_Ü");
    let (r, exe, title) = resolve_target(&bin.join("game.exe")).unwrap();
    assert_eq!((canon(&r), canon(&exe), title.as_str()), (canon(&root), canon(&bin.join("game.exe")), "Contoso_Game_Ü"));
    // an exe the manifest above does not name: its own folder is the root
    let (r, exe, title) = resolve_target(&bin.join("tool.exe")).unwrap();
    assert_eq!((canon(&r), canon(&exe), title.as_str()), (canon(&bin), canon(&bin.join("tool.exe")), "XSX"));
}
