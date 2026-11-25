//! Deferred looks for the CPU's writes into the sparse windows (`XCL_DEFERRED_LOOKS=1`).
//!
//! Before each of the title's ExecuteCommandLists XCL looks for what the CPU wrote in the sparse
//! windows' tiles (write-watch) and copies it to video memory (`title_textures::refresh`,
//! `sparse::before_execute`). The look walks every page with video memory whatever was written:
//! on a retail title's menu ~2 GB, ~0.8 ms, then ~0.3 ms of copies, on the submitting thread
//! and under one lock for all of them – at ~800 submits a second, most of every second.
//!
//! With deferred looks the submitting thread asks for a look (`Looks::request`), makes its queue
//! GPU-wait for that look's fence value and returns; a worker thread does the look and the
//! copies, then has the sparse queue signal the value after them. A look serves every request
//! made before it began, so the submits made while one look runs share the next one, and the
//! title's lists still run after a look that began after their submit: they see every CPU write
//! made before it. The pages of the texture look (on the submitting thread) that lie in the
//! sparse windows go with the request, and the look's pages go to the texture look in turn
//! (`title_textures`), so whichever of the two finds a written page, both see it.

use std::sync::{Condvar, Mutex, OnceLock};

/// Whether looks are deferred (`XCL_DEFERRED_LOOKS=1`; the worker could be started).
pub(super) fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("XCL_DEFERRED_LOOKS").is_ok_and(|v| v.trim() == "1"))
}

struct Sched {
    /// looks begun (the last one's number)
    started: u64,
    /// looks whose fence signal is issued
    issued: u64,
    /// the highest look number asked for
    wanted: u64,
    /// pages written (4 KiB, found by the submitting threads) for the next look
    carry: Vec<usize>,
}

pub(super) struct Looks {
    s: Mutex<Sched>,
    work: Condvar,
    done: Condvar,
}

impl Looks {
    pub(super) const fn new() -> Self {
        Looks { s: Mutex::new(Sched { started: 0, issued: 0, wanted: 0, carry: Vec::new() }), work: Condvar::new(), done: Condvar::new() }
    }

    /// Ask for a look that begins after this call, `pages` going with it: its number, the fence
    /// value the caller's queue waits for.
    pub(super) fn request(&self, pages: &[usize]) -> u64 {
        let mut s = self.s.lock().unwrap();
        let n = s.started + 1;
        s.wanted = s.wanted.max(n);
        s.carry.extend_from_slice(pages);
        self.work.notify_one();
        n
    }

    /// The worker: once a look is asked for, it begins (its number, and the pages that go with it).
    pub(super) fn begin(&self) -> (u64, Vec<usize>) {
        let mut s = self.s.lock().unwrap();
        while s.wanted <= s.started {
            s = self.work.wait(s).unwrap();
        }
        s.started += 1;
        (s.started, std::mem::take(&mut s.carry))
    }

    /// Look `n`'s fence signal is issued.
    pub(super) fn issued(&self, n: u64) {
        let mut s = self.s.lock().unwrap();
        s.issued = s.issued.max(n);
        self.done.notify_all();
    }

    /// Return once every look asked for so far has its signal issued. A queue that then waits for
    /// title work (a readback behind a title queue) has those signals ahead of it: the title's
    /// lists it waits for are not left waiting for a look behind that wait.
    pub(super) fn wait_issued(&self) {
        let mut s = self.s.lock().unwrap();
        let want = s.wanted;
        while s.issued < want {
            s = self.done.wait(s).unwrap();
        }
    }
}

pub(super) static LOOKS: Looks = Looks::new();

/// Start the worker once: `look(n, pages)` does look `n` and issues its signal. False: there is
/// no worker (looks are not deferred).
pub(super) fn start(look: fn(u64, Vec<usize>)) -> bool {
    static STARTED: OnceLock<bool> = OnceLock::new();
    *STARTED.get_or_init(|| {
        let spawned = std::thread::Builder::new().name("xcl-looks".into()).spawn(move || loop {
            let (n, pages) = LOOKS.begin();
            if std::panic::catch_unwind(move || look(n, pages)).is_err() {
                // the queues waiting for it go on (what it was to copy is not copied)
                if let Some(f) = super::sparse::look_fence() {
                    let _ = unsafe { f.Signal(n) };
                }
            }
            LOOKS.issued(n);
        });
        match spawned {
            Ok(_) => {
                crate::xlog_info!("D3D12.X: CPU writes into the sparse windows are looked for on a worker thread (XCL_DEFERRED_LOOKS=1)");
                true
            }
            Err(e) => {
                crate::xlog_warn!("D3D12.X: no thread for deferred looks ({e}): looking before each submit");
                false
            }
        }
    })
}

/// `a` without `b` (both sorted, disjoint `(base, size)` ranges): the sorted pieces of `a` that
/// no range of `b` covers.
pub(super) fn subtract(a: &[(usize, usize)], b: &[(usize, usize)]) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut j = 0;
    for &(base, size) in a {
        let end = base + size;
        let mut at = base;
        while j < b.len() && b[j].0 + b[j].1 <= at {
            j += 1;
        }
        let mut k = j;
        while at < end && k < b.len() && b[k].0 < end {
            let (lo, hi) = (b[k].0, b[k].0 + b[k].1);
            if lo > at {
                out.push((at, lo - at));
            }
            at = at.max(hi);
            k += 1;
        }
        if at < end {
            out.push((at, end - at));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::time::Duration;

    #[test]
    fn subtract_leaves_the_uncovered_pieces() {
        let a = [(0x1000, 0x9000), (0x20000, 0x1000), (0x30000, 0x4000)];
        // a hole in the middle, one over the start of a range and its whole neighbour, one past all
        let b = [(0x3000, 0x2000), (0x8000, 0x19000), (0x40000, 0x1000)];
        assert_eq!(subtract(&a, &b), vec![(0x1000, 0x2000), (0x5000, 0x3000), (0x30000, 0x4000)]);
        assert_eq!(subtract(&a, &[]), a.to_vec());
        assert_eq!(subtract(&[], &b), vec![]);
        // one range of b over several of a, and one ending exactly where a range starts
        assert_eq!(subtract(&[(0x1000, 0x1000), (0x3000, 0x1000), (0x5000, 0x2000)], &[(0x0, 0x1000), (0x800, 0x5000)]), vec![(0x5800, 0x1800)]);
        // pieces of b inside one range of a, touching each other
        assert_eq!(subtract(&[(0x0, 0x10000)], &[(0x1000, 0x1000), (0x2000, 0x1000), (0x8000, 0x8000)]), vec![(0x0, 0x1000), (0x3000, 0x5000)]);
    }

    /// Looks asked for while one runs are served by one more look, which begins after them, with
    /// every page that came with them; `wait_issued` returns once that look is issued.
    #[test]
    fn requests_during_a_look_share_the_next_one() {
        let l = Arc::new(Looks::new());
        assert_eq!(l.request(&[0x1000]), 1);
        assert_eq!(l.request(&[0x2000]), 1, "look 1 has not begun: both wait for it");
        assert_eq!(l.begin(), (1, vec![0x1000, 0x2000]));
        // look 1 runs: what is asked now needs a look that begins later
        assert_eq!(l.request(&[0x3000]), 2);
        assert_eq!(l.request(&[]), 2);
        assert_eq!(l.request(&[0x4000]), 2);
        l.issued(1);
        assert_eq!(l.begin(), (2, vec![0x3000, 0x4000]));
        // nothing asked for: the worker waits
        let worker = {
            let l = Arc::clone(&l);
            std::thread::spawn(move || l.begin())
        };
        std::thread::sleep(Duration::from_millis(50));
        assert!(!worker.is_finished(), "a look began that nobody asked for");
        // wait_issued waits for look 2 (asked for), not for look 3 (asked for after it)
        let waiter = {
            let l = Arc::clone(&l);
            std::thread::spawn(move || l.wait_issued())
        };
        std::thread::sleep(Duration::from_millis(50));
        assert!(!waiter.is_finished(), "returned before look 2 was issued");
        assert_eq!(l.request(&[0x5000]), 3);
        assert_eq!(worker.join().unwrap(), (3, vec![0x5000]));
        l.issued(2);
        waiter.join().unwrap();
        l.issued(3);
        l.wait_issued();
    }
}
