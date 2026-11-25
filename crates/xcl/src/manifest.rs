//! The few `MicrosoftGame.Config` settings XCL uses (game manifest in the game folder):
//! the persistent local storage quota and the title memory mode (the title sees the PC's
//! processors, so SMT does not matter). Plain text search –
//! the file is small XML and only these elements matter.

use std::path::Path;

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Manifest {
    /// `<PersistentLocalStorage><SizeMB>`
    pub pls_mb: Option<u64>,
    /// `<XboxAnacondaTitleMemory>` / `<XboxLockhartTitleMemory>`: "Standard" or "Advanced"
    pub anaconda_memory: Option<String>,
    pub lockhart_memory: Option<String>,
    /// `<TitleId>` (hex)
    pub title_id: Option<String>,
    /// `<Resources><Resource Language="..."/>`: declared package locales in manifest order
    pub languages: Vec<String>,
    /// `<Identity Version="a.b.c.d">`
    pub version: Option<String>,
    /// `<StoreId>`
    pub store_id: Option<String>,
    /// `<ShellVisuals DefaultDisplayName="...">`
    pub display_name: Option<String>,
    /// `<ShellVisuals PublisherDisplayName="...">`
    pub publisher_display_name: Option<String>,
    /// `<ShellVisuals Description="...">`
    pub description: Option<String>,
}

fn element<'a>(xml: &'a str, name: &str) -> Option<&'a str> {
    let open = format!("<{name}");
    let mut from = 0;
    while let Some(i) = xml[from..].find(&open).map(|i| from + i) {
        let after = i + open.len();
        let Some(next) = xml[after..].chars().next() else { break };
        if next == '>' || next.is_whitespace() {
            let gt = xml[after..].find('>')?;
            if !xml[after..after + gt].trim_end().ends_with('/') {
                let body = after + gt + 1;
                let end = xml[body..].find(&format!("</{name}>"))?;
                return Some(xml[body..body + end].trim());
            }
        }
        from = after;
    }
    None
}

/// The first `<tag attr="...">` value, trimmed; None when missing or empty.
fn attr(xml: &str, tag: &str, name: &str) -> Option<String> {
    crate::launcher::xml_attrs(xml, tag, name).into_iter().next().map(|s| s.trim().to_string()).filter(|s| !s.is_empty())
}

pub fn parse(xml: &str) -> Manifest {
    let pls = element(xml, "PersistentLocalStorage").and_then(|section| element(section, "SizeMB"));
    let languages = element(xml, "Resources")
        .map(|section| {
            crate::launcher::xml_attrs(section, "Resource", "Language")
                .into_iter()
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect()
        })
        .unwrap_or_default();
    Manifest {
        pls_mb: pls.and_then(|v| v.parse().ok()),
        anaconda_memory: element(xml, "XboxAnacondaTitleMemory").map(str::to_string),
        lockhart_memory: element(xml, "XboxLockhartTitleMemory").map(str::to_string),
        title_id: element(xml, "TitleId").map(str::to_string),
        languages,
        version: attr(xml, "Identity", "Version"),
        store_id: element(xml, "StoreId").map(str::to_string).filter(|s| !s.is_empty()),
        display_name: attr(xml, "ShellVisuals", "DefaultDisplayName"),
        publisher_display_name: attr(xml, "ShellVisuals", "PublisherDisplayName"),
        description: attr(xml, "ShellVisuals", "Description"),
    }
}

/// `MicrosoftGame.Config` of a game folder (None: no manifest).
pub fn load(game_root: &Path) -> Option<Manifest> {
    let p = std::fs::read_dir(game_root).ok()?.flatten().map(|e| e.path()).find(|p| p.file_name().is_some_and(|n| n.eq_ignore_ascii_case("MicrosoftGame.Config")))?;
    let raw = std::fs::read(p).ok()?;
    // the launcher's decoding: UTF-8, or UTF-16 with its BOM; comments removed
    let text = crate::launcher::decode_game_config(&raw)?;
    Some(parse(&text))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_the_settings_xcl_uses() {
        let xml = r#"<Game configVersion="1">
  <PersistentLocalStorage>
    <SizeMB>2048</SizeMB>
  </PersistentLocalStorage>
  <VirtualMachine>
    <SMTEnabled>true</SMTEnabled>
    <XboxLockhartTitleMemory>Advanced</XboxLockhartTitleMemory>
    <XboxAnacondaTitleMemory>Advanced</XboxAnacondaTitleMemory>
  </VirtualMachine>
</Game>"#;
        let m = parse(xml);
        assert_eq!(m.pls_mb, Some(2048));
        assert_eq!((m.anaconda_memory.as_deref(), m.lockhart_memory.as_deref()), (Some("Advanced"), Some("Advanced")));
        assert_eq!(parse("<Game/>"), Manifest::default());
    }

    #[test]
    fn reads_the_package_identity_and_shell_visuals() {
        let xml = r#"<Game configVersion="1">
  <Identity Name="41336PublisherName.ExampleGame" Publisher="CN=A4954634-DF4B-47C7-AB70-D3215D246AF1" Version="1.6.0.0"/>
  <StoreId>9NZTKMP36L8J</StoreId>
  <ShellVisuals DefaultDisplayName="Example Game &amp; Co"
      PublisherDisplayName="Example Publisher"
      StoreLogo="StoreLogoFile.png"
      Description="ms-resource:ApplicationDescription"/>
</Game>"#;
        let m = parse(xml);
        assert_eq!(m.version.as_deref(), Some("1.6.0.0"));
        assert_eq!(m.store_id.as_deref(), Some("9NZTKMP36L8J"));
        assert_eq!(m.display_name.as_deref(), Some("Example Game & Co"));
        assert_eq!(m.publisher_display_name.as_deref(), Some("Example Publisher"));
        assert_eq!(m.description.as_deref(), Some("ms-resource:ApplicationDescription"));
        let bare = parse(r#"<Game><Identity Name="A.B" Publisher="CN=X"/><ShellVisuals DefaultDisplayName=""/></Game>"#);
        assert_eq!((bare.version, bare.display_name, bare.store_id), (None, None, None));
    }

    #[test]
    fn commented_out_settings_are_not_read() {
        let dir = std::env::temp_dir().join(format!("xcl_manifest_comments_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let xml = r#"<Game configVersion="1">
  <!-- <PersistentLocalStorage><SizeMB>4096</SizeMB></PersistentLocalStorage> -->
  <PersistentLocalStorage><SizeMB>1024</SizeMB></PersistentLocalStorage>
  <VirtualMachine>
    <!--<XboxAnacondaTitleMemory>Advanced</XboxAnacondaTitleMemory>-->
  </VirtualMachine>
</Game>"#;
        let want = Manifest { pls_mb: Some(1024), ..Default::default() };
        std::fs::write(dir.join("MicrosoftGame.config"), xml).unwrap();
        assert_eq!(load(&dir), Some(want.clone()));
        // UTF-16 big-endian, as the launcher reads it too
        let mut raw = vec![0xFE, 0xFF];
        raw.extend(xml.encode_utf16().flat_map(u16::to_be_bytes));
        std::fs::write(dir.join("MicrosoftGame.config"), raw).unwrap();
        assert_eq!(load(&dir), Some(want));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn does_not_take_storage_size_from_another_section() {
        let xml = r#"<Game>
  <PersistentLocalStorage></PersistentLocalStorage>
  <Other><SizeMB>512</SizeMB></Other>
</Game>"#;
        assert_eq!(parse(xml).pls_mb, None);
    }

    #[test]
    fn parses_declared_resources_languages_with_comments_and_utf16() {
        let xml = r#"<Game configVersion="1">
  <Other><Resource Language="xx-XX"/></Other>
  <Resources>
    <!-- <Resource Language="de-DE"/> -->
    <Resource Language="en-US"/>
    <Resource Scale="100"/>
    <Resource Language="fr-FR" />
    <Resource Language="   "/>
    <Resource Language="ja-JP"></Resource>
  </Resources>
</Game>"#;
        assert_eq!(parse(xml).languages, vec!["de-DE", "en-US", "fr-FR", "ja-JP"]);

        let dir = std::env::temp_dir().join(format!("xcl_manifest_langs_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("MicrosoftGame.Config"), xml).unwrap();
        assert_eq!(load(&dir).unwrap().languages, vec!["en-US", "fr-FR", "ja-JP"]);

        // UTF-16 LE with BOM
        let mut raw = vec![0xFF, 0xFE];
        raw.extend(xml.encode_utf16().flat_map(u16::to_le_bytes));
        std::fs::write(dir.join("MicrosoftGame.Config"), raw).unwrap();
        assert_eq!(load(&dir).unwrap().languages, vec!["en-US", "fr-FR", "ja-JP"]);

        let _ = std::fs::remove_dir_all(&dir);
    }
}
