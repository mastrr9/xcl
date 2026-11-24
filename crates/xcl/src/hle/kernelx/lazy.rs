//! Physical pages mapped into the title's own reservations, committed on first use.
//!
//! A title maps its physical pages wherever it reserved address space and places resources
//! there – render targets and depth buffers too, which on the PC live in video memory
//! (`d3d12x::vram`); the CPU never touches those pages. Committed at map time they doubled
//! that memory (a retail game: 5.4 GB of mapped pages next to its video-memory targets, past
//! the system's commit). So a page whose contents are still zero is mapped as reserved address
//! space only: the CPU's first access faults, the page is committed (zeros, as the page holds)
//! and the access goes on. Kernel calls do not fault in the title – they fail – so the buffers
//! of the file calls XCL routes, and DirectStorage's destinations, are committed first
//! (`ensure`).

use std::collections::BTreeSet;
use std::ffi::c_void;
use std::sync::{Once, RwLock};

use windows_sys::Win32::Foundation::{EXCEPTION_ACCESS_VIOLATION, NTSTATUS};
use windows_sys::Win32::System::Diagnostics::Debug::{AddVectoredExceptionHandler, EXCEPTION_POINTERS};
use windows_sys::Win32::System::Memory::{
    VirtualAlloc, VirtualQuery, MEMORY_BASIC_INFORMATION, MEM_COMMIT, PAGE_EXECUTE_READ, PAGE_EXECUTE_READWRITE,
    PAGE_EXECUTE_WRITECOPY, PAGE_GUARD, PAGE_READONLY, PAGE_READWRITE, PAGE_WRITECOPY,
};

const PAGE: usize = 0x1_0000;
const EXCEPTION_CONTINUE_EXECUTION: i32 = -1;
const EXCEPTION_CONTINUE_SEARCH: i32 = 0;

/// pages (64 KiB) mapped but not committed yet
static LAZY: RwLock<Pages> = RwLock::new(Pages(BTreeSet::new()));

/// The lowest and highest lazy page (usize::MAX and 0 with none), published with every change
/// so `ensure` – every ReadFile a title makes – can tell without the lock that a buffer holds
/// none: the lock's shared count is a cache line all streaming threads write. Only changed
/// under the write lock; each value read is a bound of every page added before the read.
static LOWEST: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(usize::MAX);
static HIGHEST: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
const PUBLISH: std::sync::atomic::Ordering = std::sync::atomic::Ordering::Release;
const READ: std::sync::atomic::Ordering = std::sync::atomic::Ordering::Acquire;

/// The lazy page set, keeping LOWEST and HIGHEST.
struct Pages(BTreeSet<usize>);

impl Pages {
    fn insert(&mut self, page: usize) -> bool {
        let new = self.0.insert(page);
        if new {
            self.publish();
        }
        new
    }

    fn remove(&mut self, page: &usize) -> bool {
        let had = self.0.remove(page);
        if had {
            self.publish();
        }
        had
    }

    fn publish(&self) {
        LOWEST.store(self.0.first().copied().unwrap_or(usize::MAX), PUBLISH);
        HIGHEST.store(self.0.last().copied().unwrap_or(0), PUBLISH);
    }

    fn contains(&self, page: &usize) -> bool {
        self.0.contains(page)
    }

    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    fn range(&self, r: std::ops::Range<usize>) -> std::collections::btree_set::Range<'_, usize> {
        self.0.range(r)
    }

    /// Whether a lazy page can start in `[p & !(PAGE - 1), p + len)` (`len` > 0), read without
    /// the lock: false when the range lies wholly below the lowest or above the highest.
    fn may_hold(p: usize, len: usize) -> bool {
        overlaps(p, len, LOWEST.load(READ), HIGHEST.load(READ))
    }
}

/// Whether a page in `lowest..=highest` can start in `[p & !(PAGE - 1), p + len)` (`len` > 0).
fn overlaps(p: usize, len: usize, lowest: usize, highest: usize) -> bool {
    p & !(PAGE - 1) <= highest && p.saturating_add(len - 1) >= lowest
}

fn install() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| unsafe {
        AddVectoredExceptionHandler(1, Some(handler));
    });
}

/// Commit `page` if it is a lazy one (true: it was).
unsafe fn commit(page: usize) -> bool {
    let mut g = LAZY.write().unwrap_or_else(|e| e.into_inner());
    if !g.remove(&page) {
        return false;
    }
    !VirtualAlloc(page as *const c_void, PAGE, MEM_COMMIT, PAGE_READWRITE).is_null()
}

unsafe extern "system" fn handler(info: *mut EXCEPTION_POINTERS) -> i32 {
    let r = &*(*info).ExceptionRecord;
    if r.ExceptionCode != EXCEPTION_ACCESS_VIOLATION as NTSTATUS || r.NumberParameters < 2 {
        return EXCEPTION_CONTINUE_SEARCH;
    }
    let page = r.ExceptionInformation[1] & !(PAGE - 1);
    let lazy = LAZY.read().unwrap_or_else(|e| e.into_inner()).contains(&page);
    if lazy && commit(page) || allows(r.ExceptionInformation[0], r.ExceptionInformation[1]) {
        EXCEPTION_CONTINUE_EXECUTION
    } else {
        EXCEPTION_CONTINUE_SEARCH
    }
}

unsafe fn allows(access: usize, at: usize) -> bool {
    let mut m: MEMORY_BASIC_INFORMATION = std::mem::zeroed();
    if VirtualQuery(at as *const c_void, &mut m, std::mem::size_of_val(&m)) == 0 || m.State != MEM_COMMIT || m.Protect & PAGE_GUARD != 0 {
        return false;
    }
    let writes = PAGE_READWRITE | PAGE_WRITECOPY | PAGE_EXECUTE_READWRITE | PAGE_EXECUTE_WRITECOPY;
    let reads = writes | PAGE_READONLY | PAGE_EXECUTE_READ;
    match access {
        0 => m.Protect & reads != 0,
        1 => m.Protect & writes != 0,
        _ => false,
    }
}

/// `page` (reserved, 64 KiB aligned) is mapped: committed on first use.
pub(crate) fn mark(page: usize) {
    install();
    LAZY.write().unwrap_or_else(|e| e.into_inner()).insert(page);
}

/// `page` is unmapped: true if it was never used (nothing to keep of it).
pub(super) fn unmark(page: usize) -> bool {
    LAZY.write().unwrap_or_else(|e| e.into_inner()).remove(&page)
}

/// The lazy pages of `[p, p + len)` committed – memory a kernel call or XCL writes.
pub fn ensure(p: usize, len: usize) {
    if len == 0 || !Pages::may_hold(p, len) {
        return;
    }
    let pages: Vec<usize> = {
        let g = LAZY.read().unwrap_or_else(|e| e.into_inner());
        if g.is_empty() {
            return;
        }
        g.range(p & !(PAGE - 1)..p.saturating_add(len)).copied().collect()
    };
    for page in pages {
        unsafe { commit(page) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows_sys::Win32::System::Memory::{VirtualFree, MEM_RELEASE, MEM_RESERVE};

    /// A lazy page reads as zeros on first touch, keeps what is written; `ensure` commits it
    /// without a fault.
    #[test]
    fn a_lazy_page_is_committed_on_first_use() {
        unsafe {
            let r = VirtualAlloc(std::ptr::null(), 2 * PAGE, MEM_RESERVE, PAGE_READWRITE) as usize;
            mark(r);
            mark(r + PAGE);
            assert_eq!(std::ptr::read_volatile((r + 8) as *const u32), 0);
            std::ptr::write_volatile((r + 8) as *mut u32, 7);
            assert_eq!(std::ptr::read_volatile((r + 8) as *const u32), 7);
            assert!(!unmark(r), "used: no longer lazy");
            ensure(r + PAGE + 4, 4);
            assert!(!LAZY.read().unwrap().contains(&(r + PAGE)));
            assert!(crate::gpu::shader::readable(r + PAGE, 4));
            VirtualFree(r as *mut c_void, 0, MEM_RELEASE);
        }
    }

    /// `ensure` skips the lock only for a buffer wholly outside the lazy pages' span, and the
    /// published span is the set's.
    #[test]
    fn ensure_skips_the_lock_only_outside_the_lazy_span() {
        let (a, b) = (0x7_0000_0000usize, 0x7_0004_0000usize);
        assert!(!overlaps(0x1_0000, 4, usize::MAX, 0), "no lazy page: never the lock");
        assert!(overlaps(a + 8, 4, a, b));
        assert!(overlaps(a - 4, 8, a, b), "a buffer reaching the lowest page");
        assert!(overlaps(b + PAGE - 1, 1, a, b), "the last byte of the highest page");
        assert!(overlaps(a + PAGE, PAGE, a, b), "between the two: the lock decides");
        assert!(overlaps(0, usize::MAX, a, b));
        assert!(!overlaps(a - 8, 8, a, b), "just below the lowest");
        assert!(!overlaps(b + PAGE, 16, a, b), "above the highest page");
        assert!(!overlaps(usize::MAX - 4, 100, a, b), "a range past the top");

        let span = || {
            let g = LAZY.read().unwrap_or_else(|e| e.into_inner());
            let set = (g.0.first().copied().unwrap_or(usize::MAX), g.0.last().copied().unwrap_or(0));
            (set, (LOWEST.load(READ), HIGHEST.load(READ)))
        };
        unsafe {
            let r = VirtualAlloc(std::ptr::null(), 2 * PAGE, MEM_RESERVE, PAGE_READWRITE) as usize;
            assert_ne!(r, 0);
            mark(r);
            mark(r + PAGE);
            let (set, published) = span();
            assert_eq!(set, published);
            assert!(published.0 <= r && r + PAGE <= published.1);
            assert!(Pages::may_hold(r + PAGE + 8, 4));
            ensure(r + PAGE + 8, 4);
            assert!(!LAZY.read().unwrap().contains(&(r + PAGE)), "a lazy page in the buffer is committed");
            assert!(unmark(r));
            let (set, published) = span();
            assert_eq!(set, published);
            VirtualFree(r as *mut c_void, 0, MEM_RELEASE);
        }
    }

    #[test]
    fn threads_first_touching_one_lazy_page_together_all_go_on() {
        const THREADS: usize = 8;
        const PAGES: usize = 256;
        unsafe {
            let r = VirtualAlloc(std::ptr::null(), PAGES * PAGE, MEM_RESERVE, PAGE_READWRITE) as usize;
            for i in 0..PAGES {
                mark(r + i * PAGE);
            }
            let barrier = std::sync::Arc::new(std::sync::Barrier::new(THREADS));
            let handles: Vec<_> = (0..THREADS)
                .map(|t| {
                    let barrier = barrier.clone();
                    std::thread::spawn(move || {
                        for i in 0..PAGES {
                            barrier.wait();
                            std::ptr::write_volatile((r + i * PAGE + t * 64) as *mut u64, (i * THREADS + t) as u64);
                        }
                    })
                })
                .collect();
            for h in handles {
                h.join().unwrap();
            }
            for i in 0..PAGES {
                for t in 0..THREADS {
                    assert_eq!(std::ptr::read_volatile((r + i * PAGE + t * 64) as *const u64), (i * THREADS + t) as u64);
                }
            }
            VirtualFree(r as *mut c_void, 0, MEM_RELEASE);
        }
    }
}
