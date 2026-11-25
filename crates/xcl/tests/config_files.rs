//! xcl.toml as the launcher reads and writes it (`Config::load_or_create`), `Config::resolve`
//! against a MicrosoftGame.Config, and the boot record (`Boot`) the host process gets.
//! Pure file and string handling: no title, no Windows API.

use std::fs;
use std::path::{Path, PathBuf};
use xcl::config::{parse_raytracing_tier, Boot, Config, CONSOLE_ID, PLS_MOUNT_TARGET};
use xcl::manifest::Manifest;

/// A fresh folder of this test's own under the temp folder, removed on drop.
struct TempDir(PathBuf);

impl TempDir {
    fn new(test: &str) -> TempDir {
        let p = std::env::temp_dir().join(format!("xcl_config_files_{test}_{}", std::process::id()));
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

/// Every field, so new fields are compared without naming them here.
fn json(c: &Config) -> serde_json::Value {
    serde_json::to_value(c).unwrap()
}

/// A missing xcl.toml is created (with its folders) from the defaults, under a comment header,
/// and reads back as the same config without being written again.
#[test]
fn a_missing_file_is_created_with_the_defaults_and_reads_back_unchanged() {
    let dir = TempDir::new("create");
    let path = dir.0.join("Contoso.Game").join("nested").join("xcl.toml");
    let created = Config::load_or_create(&path).unwrap();
    assert_eq!(json(&created), json(&Config::default()));
    let text = fs::read_to_string(&path).unwrap();
    assert!(text.starts_with('#'), "the file starts with its comment header:\n{text}");
    let parsed: Config = toml::from_str(&text).unwrap();
    assert_eq!(json(&parsed), json(&Config::default()));
    let again = Config::load_or_create(&path).unwrap();
    assert_eq!(json(&again), json(&created));
    assert_eq!(fs::read_to_string(&path).unwrap(), text, "an existing file is not rewritten");
}

/// The user's own settings, mounts, aliases and stub returns survive a load; default mounts the
/// file lacks are added back, and a default mount the user retargeted keeps the user's target.
#[test]
fn user_settings_survive_and_missing_default_mounts_are_added() {
    let dir = TempDir::new("user");
    let path = dir.0.join("xcl.toml");
    let text = r#"device_type = "lockhart"
gamertag = "Ada"
title_memory = "advanced"
pls_mb = 1024
some_setting_of_a_newer_xcl = 7

[mounts]
"G:" = 'C:\Games\Elsewhere'
"CACHE:" = '{data}\cache2'

[aliases]
"xmem*.dll" = "@stub"

[stub_returns]
"*!XFooBar" = "E_NOTIMPL"
"#;
    fs::write(&path, text).unwrap();
    let c = Config::load_or_create(&path).unwrap();
    assert!(c.is_series_s());
    assert_eq!((c.gamertag.as_str(), c.title_memory.as_str(), c.pls_mb), ("Ada", "advanced", 1024));
    assert_eq!(c.mounts.get("G:").map(String::as_str), Some(r"C:\Games\Elsewhere"));
    assert_eq!(c.mounts.get("CACHE:").map(String::as_str), Some(r"{data}\cache2"));
    for (k, v) in Config::default().mounts {
        if k != "G:" {
            assert_eq!(c.mounts.get(&k), Some(&v), "default mount {k} added back");
        }
    }
    assert_eq!(c.aliases.get("xmem*.dll").map(String::as_str), Some("@stub"));
    assert_eq!(c.stub_returns.get("*!XFooBar").map(String::as_str), Some("E_NOTIMPL"));
    // the settings the file does not name keep their defaults
    assert_eq!(c.console_id, CONSOLE_ID);
    assert_eq!(c.window_width, Config::default().window_width);
    assert_eq!(fs::read_to_string(&path).unwrap(), text, "loading never rewrites the user's file");
}

/// An empty xcl.toml (all lines deleted) is every default, mounts included.
#[test]
fn an_empty_file_is_the_defaults() {
    let dir = TempDir::new("empty");
    let path = dir.0.join("xcl.toml");
    fs::write(&path, "").unwrap();
    assert_eq!(json(&Config::load_or_create(&path).unwrap()), json(&Config::default()));
}

/// An xcl.toml edited in Notepad (UTF-8 with a BOM, CRLF line ends) reads as written.
#[test]
fn a_file_saved_with_a_bom_and_crlf_is_read() {
    let dir = TempDir::new("bom");
    let path = dir.0.join("xcl.toml");
    fs::write(&path, b"\xEF\xBB\xBFgamertag = \"Ada\"\r\n[mounts]\r\n\"X:\" = '{data}\\x'\r\n").unwrap();
    let c = Config::load_or_create(&path).unwrap();
    assert_eq!(c.gamertag, "Ada");
    assert_eq!(c.mounts.get("X:").map(String::as_str), Some("{data}\\x"));
}

/// A file that cannot be used is an error naming the file, never silent defaults, and it is
/// left as it is (issue #106): bad TOML, a wrong type, text that is not UTF-8, a folder.
#[test]
fn an_unusable_file_is_an_error_and_is_kept() {
    let dir = TempDir::new("bad");
    let path = dir.0.join("xcl.toml");
    let cases: [&[u8]; 4] = [
        b"device_type = \"lockhart\n",
        b"window_width = \"wide\"\n",
        b"[mounts]\n\"G:\" = 5\n",
        b"gamertag = \"\xFF\xFE\"\n",
    ];
    for raw in cases {
        fs::write(&path, raw).unwrap();
        let e = Config::load_or_create(&path).unwrap_err();
        assert!(e.contains("xcl.toml"), "{e}");
        assert_eq!(fs::read(&path).unwrap(), raw, "{e}");
    }
    let folder = dir.0.join("folder.toml");
    fs::create_dir(&folder).unwrap();
    assert!(Config::load_or_create(&folder).is_err());
    assert!(folder.is_dir());
}

/// `resolve` without a manifest: the standard memory budget of the device, 4096 MiB of PLS,
/// 2048 MiB of UMA, no title id, nothing reported as from MicrosoftGame.Config.
#[test]
fn resolve_without_a_manifest_gives_the_standard_values() {
    for (device, budget) in [("anaconda", 13_824), ("lockhart", 8_192), ("SERIES S", 8_192), ("xbox one", 13_824)] {
        let mut c = Config { device_type: device.into(), ..Config::default() };
        assert!(c.resolve(None).is_empty(), "{device}");
        assert_eq!((c.title_memory.as_str(), c.title_memory_mb), ("standard", budget), "{device}");
        assert_eq!((c.pls_mb, c.uma_mb), (4096, 2048), "{device}");
        assert_eq!(c.title_id, "", "{device}");
    }
}

/// `resolve` takes PLS size, title id and the device's memory mode from the manifest and says
/// so; a second `resolve` changes nothing and reports nothing.
#[test]
fn resolve_takes_the_manifest_values_once() {
    let m = Manifest {
        pls_mb: Some(2048),
        anaconda_memory: Some("Standard".into()),
        lockhart_memory: Some("Advanced".into()),
        title_id: Some("1A2B3C4D".into()),
        ..Default::default()
    };
    let mut c = Config { device_type: "Lockhart".into(), ..Config::default() };
    let from = c.resolve(Some(&m));
    assert_eq!(from.len(), 3, "{from:?}");
    assert!(from.iter().any(|f| f.contains("2048")), "{from:?}");
    assert!(from.iter().any(|f| f.contains("1A2B3C4D")), "{from:?}");
    assert_eq!((c.title_memory.as_str(), c.title_memory_mb, c.uma_mb), ("advanced", 8_192, 4095));
    assert_eq!((c.pls_mb, c.title_id.as_str()), (2048, "1A2B3C4D"));
    let first = json(&c);
    assert!(c.resolve(Some(&m)).is_empty());
    assert_eq!(json(&c), first);
}

/// What xcl.toml sets wins over the manifest: a fixed memory mode and budget, PLS size, title
/// id and UMA size are kept, and none of them is reported as from the manifest.
#[test]
fn resolve_keeps_what_the_user_set() {
    let m = Manifest {
        pls_mb: Some(2048),
        anaconda_memory: Some("Advanced".into()),
        lockhart_memory: Some("Advanced".into()),
        title_id: Some("1A2B3C4D".into()),
        ..Default::default()
    };
    let mut c = Config {
        title_memory: " Standard ".into(),
        title_memory_mb: 16_000,
        pls_mb: 512,
        title_id: "ABCDEF01".into(),
        uma_mb: 3000,
        ..Config::default()
    };
    assert!(c.resolve(Some(&m)).is_empty());
    assert_eq!((c.title_memory.as_str(), c.title_memory_mb), ("standard", 16_000));
    assert_eq!((c.pls_mb, c.title_id.as_str(), c.uma_mb), (512, "ABCDEF01", 3000));
    // a manifest that only gives the other device's mode: the standard one
    let only_x = Manifest { anaconda_memory: Some("Advanced".into()), ..Default::default() };
    let mut s = Config { device_type: "lockhart".into(), ..Config::default() };
    s.resolve(Some(&only_x));
    assert_eq!((s.title_memory.as_str(), s.uma_mb), ("standard", 2048));
}

/// `title_memory_budget_mb`: 0 is the device's standard budget, anything else is taken as
/// written, up to `u64::MAX`.
#[test]
fn title_memory_budget_edges() {
    let budget = |device: &str, mb: u64| Config { device_type: device.into(), title_memory_mb: mb, ..Config::default() }.title_memory_budget_mb();
    assert_eq!(budget("anaconda", 0), 13_824);
    assert_eq!(budget("LOCKHART", 0), 8_192);
    assert_eq!(budget("series s", 0), 8_192);
    assert_eq!(budget("", 0), 13_824);
    assert_eq!(budget("lockhart", 1), 1);
    assert_eq!(budget("anaconda", u64::MAX), u64::MAX);
}

/// `pls_drive` only takes a drive-letter mount onto `{data}\pls` (either separator, trailing
/// separators, any case), never a subfolder of it, another root or a non-letter "drive".
#[test]
fn pls_drive_edges() {
    let with = |prefix: &str, target: &str| {
        let mut c = Config::default();
        c.mounts.remove("S:");
        c.mounts.insert(prefix.into(), target.into());
        c.pls_drive()
    };
    assert_eq!(with("q:", "{DATA}/pls//").as_deref(), Some("Q:\\"));
    assert_eq!(with("Z:", "{data}\\PLS\\\\").as_deref(), Some("Z:\\"));
    assert_eq!(with("Q:", "{data}\\pls\\sub"), None);
    assert_eq!(with("Q:", "{game}\\pls"), None);
    assert_eq!(with("Q:", "{data}\\pls2"), None);
    assert_eq!(with("1:", PLS_MOUNT_TARGET), None);
    assert_eq!(with("é:", PLS_MOUNT_TARGET), None);
    assert_eq!(with("QQ:", PLS_MOUNT_TARGET), None);
    // a file without S: gets it back on load, so the title has its drive again
    let dir = TempDir::new("pls");
    let path = dir.0.join("xcl.toml");
    fs::write(&path, "[mounts]\n\"G:\" = \"{game}\"\n").unwrap();
    assert_eq!(Config::load_or_create(&path).unwrap().pls_drive().as_deref(), Some("S:\\"));
}

fn boot(root: &Path) -> Boot {
    Boot {
        title: "Contoso.Game".into(),
        exe_path: root.join("game").join("game.exe"),
        game_root: root.join("game"),
        data_dir: root.join("data ü"),
        config: Config::default(),
    }
}

/// `expand` replaces every `{game}` and `{data}` (repeated, adjacent) and leaves anything else,
/// unknown placeholders included, as written; the data subfolders hang off `data_dir`.
#[test]
fn boot_expands_placeholders_and_names_its_folders() {
    let b = boot(Path::new("base"));
    let game = b.game_root.to_string_lossy().into_owned();
    let data = b.data_dir.to_string_lossy().into_owned();
    assert_eq!(b.expand("{game}"), game);
    assert_eq!(b.expand("{data}\\temp"), format!("{data}\\temp"));
    assert_eq!(b.expand("{game}{data}{game}"), format!("{game}{data}{game}"));
    assert_eq!(b.expand("{save}\\{game\\{}x"), "{save}\\{game\\{}x");
    assert_eq!(b.expand(""), "");
    assert_eq!(b.expand(PLS_MOUNT_TARGET), format!("{data}\\pls"));
    assert_eq!(b.logs_dir(), b.data_dir.join("logs"));
    assert_eq!(b.cache_dir(), b.data_dir.join("cache"));
    assert_eq!(b.report_path(), b.data_dir.join("logs").join("report.json"));
}

/// boot.json: what the launcher writes is what the host reads, the user's mounts and
/// non-ASCII paths included.
#[test]
fn boot_record_roundtrips_through_json() {
    let mut b = boot(Path::new("C:\\Spiele\\Größe"));
    b.config.mounts.insert("CACHE:".into(), "{data}\\cache2".into());
    b.config.stub_returns.insert("kernelx.dll!XFoo".into(), "0x1".into());
    b.config.resolve(None);
    let text = serde_json::to_string_pretty(&b).unwrap();
    let back: Boot = serde_json::from_str(&text).unwrap();
    assert_eq!((back.title.as_str(), &back.exe_path, &back.game_root, &back.data_dir), (b.title.as_str(), &b.exe_path, &b.game_root, &b.data_dir));
    assert_eq!(json(&back.config), json(&b.config));
}

/// `parse_raytracing_tier` reads its words in any case; anything it does not know is "auto".
#[test]
fn raytracing_tier_words_in_any_case() {
    for (mode, want) in [("OFF", Some(0)), ("None", Some(0)), ("Tier1_0", Some(10)), ("TIER1_1", Some(11))] {
        assert_eq!(parse_raytracing_tier(mode), want, "{mode}");
    }
    for mode in ["AUTO", "on", "tier1_2", "12", "1", "1.2", "tier1", "-1", "0x0"] {
        assert_eq!(parse_raytracing_tier(mode), None, "{mode}");
    }
}
