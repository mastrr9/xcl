//! `crashes.log` in the title's logs folder: every unhandled exception, every abnormal end of the
//! process and every panic of XCL's own is appended there, with the title's last lines of debug
//! output, which often say why. `xcl.log` is recreated on each run (one earlier run is kept as
//! `xcl.prev.log`); this file keeps them all, so a crash can still be read after the game was
//! started again.
//!
//! Panics: XCL's HLE functions are `extern "system"`, so a panic in one aborts the process
//! (Rust does not unwind out of them). The default hook prints the message to stderr only, which
//! the game's window does not show; the hook installed here writes it to `xcl.log` as well, with
//! the thread and the calls that led there, and flushes the log before the abort.
//!
//! The idea (a persistent crash log with the title's last debug output, panics included) comes
//! from VIctory (github.com/Marvisak/VIctory, `victory-loader/src/crash.rs`); this file is
//! written for XCL, no code is taken from it.

use std::collections::VecDeque;
use std::io::Write;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

/// How many lines of the title's debug output a report carries.
const TITLE_LINES: usize = 8;
/// A debug line longer than this is cut (a title can print whole buffers).
const LINE_MAX: usize = 400;

static PATH: OnceLock<PathBuf> = OnceLock::new();
static TITLE_SAID: Mutex<VecDeque<String>> = Mutex::new(VecDeque::new());

/// Keeps `line` of the title's debug output (`OutputDebugString`) for the next report.
pub fn note_title_output(line: &str) {
    let line = line.trim();
    if line.is_empty() {
        return;
    }
    let line = match line.char_indices().nth(LINE_MAX) {
        Some((i, _)) => format!("{}…", &line[..i]),
        None => line.to_owned(),
    };
    // the lock is held for a push only; the report side never waits for it (`title_said`)
    let mut said = TITLE_SAID.lock().unwrap_or_else(|e| e.into_inner());
    if said.len() == TITLE_LINES {
        said.pop_front();
    }
    said.push_back(line);
}

/// The kept lines, one report line each (nothing if another thread holds them: it may be the
/// thread that crashed).
fn title_said() -> String {
    match TITLE_SAID.try_lock() {
        Ok(said) => said.iter().map(|l| format!("  title said: {l}\n")).collect(),
        Err(_) => String::new(),
    }
}

/// The entry `append` writes: a header line, `what`, then the title's last debug lines.
pub fn entry(unix_secs: u64, pid: u32, what: &str, title_said: &str) -> String {
    format!(
        "--- unix time {unix_secs}, process {pid}, XCL {}\n{}\n{title_said}",
        env!("CARGO_PKG_VERSION"),
        what.trim_end()
    )
}

/// Appends `what` to `crashes.log` (nothing before `install`). Opens the file for each entry:
/// a crash is rare, and an open handle could be lost with the thread that held it.
pub fn append(what: &str) {
    let Some(path) = PATH.get() else { return };
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs());
    let text = entry(now, std::process::id(), what, &title_said());
    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(path) {
        let _ = f.write_all(text.as_bytes());
        let _ = f.flush();
    }
}

/// Where crashes go (`<logs>/crashes.log`), and the panic hook. Call once, after the log is open.
pub fn install(logs_dir: &std::path::Path) {
    let _ = std::fs::create_dir_all(logs_dir);
    if PATH.set(logs_dir.join("crashes.log")).is_err() {
        return;
    }
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let thread = std::thread::current();
        let name = thread.name().unwrap_or("title or unnamed").to_owned();
        let tid = crate::eh::current_thread_id();
        let stack = crate::eh::callers_here(12);
        let what = format!("XCL panic on thread {tid} ({name}): {info}; on the stack: {stack}");
        crate::xlog_error!("{what}");
        append(&what);
        crate::log::flush();
        previous(info);
    }));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entry_has_header_what_and_title_lines() {
        let e = entry(1_700_000_000, 42, "unhandled exception 0xc0000005 in game.exe+0x1234\n", "  title said: a\n");
        let lines: Vec<&str> = e.lines().collect();
        assert!(lines[0].starts_with("--- unix time 1700000000, process 42, XCL "));
        assert_eq!(lines[1], "unhandled exception 0xc0000005 in game.exe+0x1234");
        assert_eq!(lines[2], "  title said: a");
        assert_eq!(lines.len(), 3);
    }

    #[test]
    fn title_output_keeps_the_last_lines_and_cuts_long_ones() {
        // the only test that touches the ring, so tests running in parallel do not disturb it
        for i in 0..TITLE_LINES + 3 {
            note_title_output(&format!("line {i}\n"));
        }
        note_title_output("   ");
        note_title_output(&"x".repeat(LINE_MAX + 50));
        let said = title_said();
        let lines: Vec<&str> = said.lines().collect();
        assert_eq!(lines.len(), TITLE_LINES);
        assert_eq!(lines[0], format!("  title said: line {}", 4));
        assert!(lines[TITLE_LINES - 1].ends_with('…'));
        assert_eq!(lines[TITLE_LINES - 1].chars().count(), "  title said: ".len() + LINE_MAX + 1);
    }
}
