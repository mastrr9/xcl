//! XTaskQueue through its exported functions, against libHttpClient's XTaskQueue.h (MIT), the
//! contract the module cites. Tests share one process and its thread pool: each keeps its own
//! statics, never replaces the process queue, and waits with long deadlines so a slow machine
//! fails a test instead of hanging it.

use std::ffi::c_void;
use std::ptr::null_mut;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering::SeqCst};
use std::sync::{mpsc, Mutex};
use std::thread::ThreadId;
use std::time::{Duration, Instant};

use xcl::hle::xgameruntime::taskqueue::*;
use xcl::util::{E_ABORT, E_INVALIDARG, S_OK};

const DEADLINE: Duration = Duration::from_secs(20);

fn cb(f: XTaskQueueCallback) -> usize {
    f as usize
}

fn term_cb(f: XTaskQueueTerminatedCallback) -> usize {
    f as usize
}

unsafe fn create(work: u32, completion: u32) -> *mut c_void {
    let mut h = null_mut();
    assert_eq!(XTaskQueueCreate(work, completion, &mut h), S_OK);
    assert!(!h.is_null());
    h
}

unsafe fn shut(h: *mut c_void) {
    assert_eq!(XTaskQueueTerminate(h, true, null_mut(), 0), S_OK);
    XTaskQueueCloseHandle(h);
}

/// Poll `done` until it holds; fail (not hang) after `DEADLINE`.
fn wait_until(what: &str, done: impl Fn() -> bool) {
    let end = Instant::now() + DEADLINE;
    while !done() {
        assert!(Instant::now() < end, "timed out waiting for {what}");
        std::thread::sleep(Duration::from_millis(2));
    }
}

/// Manual ports run nothing until dispatched; each XTaskQueueDispatch runs one callback, oldest
/// first, from the port it names only, and returns false once that port is empty (XTaskQueue.h:
/// XTaskQueueDispatch "processes an item ... returns true if an item was processed").
#[test]
fn manual_ports_dispatch_one_item_at_a_time_in_fifo_order() {
    static LOG: Mutex<Vec<(usize, bool)>> = Mutex::new(Vec::new());
    unsafe extern "system" fn record(ctx: *mut c_void, canceled: bool) {
        LOG.lock().unwrap().push((ctx as usize, canceled));
    }
    unsafe {
        let h = create(MODE_MANUAL, MODE_MANUAL);
        for i in 0..10usize {
            assert_eq!(XTaskQueueSubmitCallback(h, PORT_WORK, i as *mut c_void, cb(record)), S_OK);
            assert_eq!(XTaskQueueSubmitCallback(h, PORT_COMPLETION, (100 + i) as *mut c_void, cb(record)), S_OK);
        }
        assert!(LOG.lock().unwrap().is_empty(), "a manual port ran work without a dispatch");
        for n in 1..=10 {
            assert!(XTaskQueueDispatch(h, PORT_COMPLETION, 0));
            assert_eq!(LOG.lock().unwrap().len(), n, "one dispatch must run exactly one callback");
        }
        assert!(!XTaskQueueDispatch(h, PORT_COMPLETION, 0));
        assert!(XTaskQueueIsEmpty(h, PORT_COMPLETION));
        assert!(!XTaskQueueIsEmpty(h, PORT_WORK), "draining completion must leave work queued");
        while XTaskQueueDispatch(h, PORT_WORK, 0) {}
        assert!(XTaskQueueIsEmpty(h, PORT_WORK));
        let want: Vec<(usize, bool)> = (100..110).chain(0..10).map(|i| (i, false)).collect();
        assert_eq!(*LOG.lock().unwrap(), want);
        shut(h);
    }
}

/// XTaskQueueDispatch's timeout: with nothing queued it waits the whole timeout and returns false;
/// work submitted meanwhile from another thread ends the wait with true. A port that is not
/// Manual, or a port number other than Work/Completion, returns false without waiting.
#[test]
fn dispatch_timeout_waits_for_work_and_returns_false_when_none_comes() {
    static RAN: AtomicU32 = AtomicU32::new(0);
    unsafe extern "system" fn count(_ctx: *mut c_void, _canceled: bool) {
        RAN.fetch_add(1, SeqCst);
    }
    unsafe {
        let h = create(MODE_MANUAL, MODE_MANUAL);
        let t0 = Instant::now();
        assert!(!XTaskQueueDispatch(h, PORT_WORK, 100));
        assert!(t0.elapsed() >= Duration::from_millis(100), "returned after {:?}, before the timeout", t0.elapsed());

        let addr = h as usize;
        let submitter = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(50));
            XTaskQueueSubmitCallback(addr as *mut c_void, PORT_WORK, null_mut(), cb(count))
        });
        assert!(XTaskQueueDispatch(h, PORT_WORK, DEADLINE.as_millis() as u32), "work submitted during the wait was not dispatched");
        assert_eq!(submitter.join().unwrap(), S_OK);
        assert_eq!(RAN.load(SeqCst), 1);

        let t0 = Instant::now();
        assert!(!XTaskQueueDispatch(h, 2, 60_000), "port 2");
        assert!(!XTaskQueueDispatch(h, u32::MAX, 60_000), "port u32::MAX");
        let pool = create(MODE_THREAD_POOL, MODE_IMMEDIATE);
        assert!(!XTaskQueueDispatch(pool, PORT_WORK, 60_000), "a thread-pool port");
        assert!(!XTaskQueueDispatch(pool, PORT_COMPLETION, 60_000), "an immediate port");
        assert!(t0.elapsed() < Duration::from_secs(30), "a non-dispatchable port waited for its timeout");
        shut(pool);
        shut(h);
    }
}

/// Immediate runs the callback on the submitting thread before XTaskQueueSubmitCallback returns;
/// ThreadPool runs it on another thread (XTaskQueue.h, XTaskQueueDispatchMode).
#[test]
fn immediate_runs_on_the_submitter_and_thread_pool_does_not() {
    static SEEN: Mutex<Vec<(usize, ThreadId)>> = Mutex::new(Vec::new());
    unsafe extern "system" fn record(ctx: *mut c_void, _canceled: bool) {
        SEEN.lock().unwrap().push((ctx as usize, std::thread::current().id()));
    }
    let me = std::thread::current().id();
    unsafe {
        let h = create(MODE_THREAD_POOL, MODE_IMMEDIATE);
        assert_eq!(XTaskQueueSubmitCallback(h, PORT_COMPLETION, 1 as *mut c_void, cb(record)), S_OK);
        assert_eq!(*SEEN.lock().unwrap(), [(1, me)], "immediate callback did not run inline");
        assert_eq!(XTaskQueueSubmitCallback(h, PORT_WORK, 2 as *mut c_void, cb(record)), S_OK);
        wait_until("the thread-pool callback", || SEEN.lock().unwrap().len() == 2);
        let (ctx, tid) = SEEN.lock().unwrap()[1];
        assert_eq!(ctx, 2);
        assert_ne!(tid, me, "a thread-pool callback ran on the submitting thread");
        shut(h);
    }
}

/// XTaskQueueSubmitDelayedCallback: a callback never runs before its delay has passed, and
/// callbacks become due in delay order whatever order they were submitted in; delay 0 is
/// queued at once. Order is checked on a Manual port, the delays also on a ThreadPool port.
#[test]
fn delayed_callbacks_never_fire_early_and_fire_in_due_order() {
    static RAN: Mutex<Vec<(usize, Instant)>> = Mutex::new(Vec::new());
    unsafe extern "system" fn record(ctx: *mut c_void, _canceled: bool) {
        RAN.lock().unwrap().push((ctx as usize, Instant::now()));
    }
    unsafe fn submit_all(h: *mut c_void, delays: &[u32]) -> Vec<Instant> {
        delays
            .iter()
            .map(|&d| {
                let t = Instant::now();
                assert_eq!(XTaskQueueSubmitDelayedCallback(h, PORT_WORK, d, d as usize as *mut c_void, cb(record)), S_OK);
                t
            })
            .collect()
    }
    fn check(delays: &[u32], submitted: &[Instant], want_order: Option<&[usize]>) {
        let ran = RAN.lock().unwrap().clone();
        if let Some(want) = want_order {
            let order: Vec<usize> = ran.iter().map(|r| r.0).collect();
            assert_eq!(order, want, "delayed callbacks out of due order");
        }
        for (i, &d) in delays.iter().enumerate() {
            let at = ran.iter().find(|r| r.0 == d as usize).unwrap().1;
            let waited = at - submitted[i];
            assert!(waited >= Duration::from_millis(d as u64), "delay {d} ms ran after {waited:?}");
        }
    }
    unsafe {
        let h = create(MODE_MANUAL, MODE_MANUAL);
        let delays = [300, 100, 200, 0];
        let submitted = submit_all(h, &delays);
        for _ in 0..delays.len() {
            assert!(XTaskQueueDispatch(h, PORT_WORK, DEADLINE.as_millis() as u32), "a delayed callback never became due");
        }
        check(&delays, &submitted, Some(&[0, 100, 200, 300]));
        shut(h);

        RAN.lock().unwrap().clear();
        let h = create(MODE_THREAD_POOL, MODE_THREAD_POOL);
        let delays = [250, 50, 150];
        let submitted = submit_all(h, &delays);
        wait_until("three delayed thread-pool callbacks", || RAN.lock().unwrap().len() == 3);
        // pool workers may record two due callbacks in either order: check only the delays
        check(&delays, &submitted, None);
        shut(h);
    }
}

/// SerializedThreadPool runs a port's callbacks on the pool one at a time, in submission order,
/// even when several threads submit at once (XTaskQueueDispatchMode::SerializedThreadPool).
#[test]
fn serialized_thread_pool_never_runs_two_callbacks_at_once() {
    static RUNNING: AtomicU32 = AtomicU32::new(0);
    static MAX: AtomicU32 = AtomicU32::new(0);
    static ORDER: Mutex<Vec<usize>> = Mutex::new(Vec::new());
    unsafe extern "system" fn serial(ctx: *mut c_void, _canceled: bool) {
        let n = RUNNING.fetch_add(1, SeqCst) + 1;
        MAX.fetch_max(n, SeqCst);
        if ctx as usize % 4 == 0 {
            std::thread::sleep(Duration::from_millis(1));
        } else {
            std::thread::yield_now();
        }
        ORDER.lock().unwrap().push(ctx as usize);
        RUNNING.fetch_sub(1, SeqCst);
    }
    unsafe {
        let h = create(MODE_SERIALIZED_THREAD_POOL, MODE_SERIALIZED_THREAD_POOL);
        for i in 0..40usize {
            assert_eq!(XTaskQueueSubmitCallback(h, PORT_WORK, i as *mut c_void, cb(serial)), S_OK);
        }
        wait_until("40 serialized callbacks", || ORDER.lock().unwrap().len() == 40);
        assert_eq!(*ORDER.lock().unwrap(), (0..40).collect::<Vec<_>>(), "serialized callbacks out of order");

        // four submitting threads: order across them is free, overlap is not
        let addr = h as usize;
        let threads: Vec<_> = (0..4usize)
            .map(|t| {
                std::thread::spawn(move || {
                    for i in 0..25usize {
                        let ctx = 1000 + t * 100 + i;
                        assert_eq!(XTaskQueueSubmitCallback(addr as *mut c_void, PORT_WORK, ctx as *mut c_void, cb(serial)), S_OK);
                    }
                })
            })
            .collect();
        for t in threads {
            t.join().unwrap();
        }
        wait_until("100 more serialized callbacks", || ORDER.lock().unwrap().len() == 140);
        assert_eq!(MAX.load(SeqCst), 1, "two serialized callbacks ran at once");
        shut(h);
    }
}

/// XTaskQueueDuplicateHandle gives a second handle to the same queue: work submitted through one
/// is dispatched through the other, closing one leaves the other working, and terminating through
/// either terminates the queue. A closed handle is refused with E_INVALIDARG.
#[test]
fn duplicate_handles_share_one_queue_and_close_independently() {
    static RAN: Mutex<Vec<usize>> = Mutex::new(Vec::new());
    unsafe extern "system" fn record(ctx: *mut c_void, _canceled: bool) {
        RAN.lock().unwrap().push(ctx as usize);
    }
    unsafe {
        let h = create(MODE_MANUAL, MODE_MANUAL);
        let mut d = null_mut();
        assert_eq!(XTaskQueueDuplicateHandle(h, &mut d), S_OK);
        assert!(!d.is_null() && d != h);
        assert_eq!(XTaskQueueSubmitCallback(h, PORT_WORK, 1 as *mut c_void, cb(record)), S_OK);
        assert!(!XTaskQueueIsEmpty(d, PORT_WORK), "the duplicate does not see the original's work");
        assert!(XTaskQueueDispatch(d, PORT_WORK, 0));
        assert_eq!(*RAN.lock().unwrap(), [1]);

        XTaskQueueCloseHandle(h);
        assert_eq!(XTaskQueueSubmitCallback(h, PORT_WORK, 2 as *mut c_void, cb(record)), E_INVALIDARG);
        let mut d2 = null_mut();
        assert_eq!(XTaskQueueDuplicateHandle(h, &mut d2), E_INVALIDARG);
        assert_eq!(XTaskQueueSubmitCallback(d, PORT_WORK, 3 as *mut c_void, cb(record)), S_OK);
        assert!(XTaskQueueDispatch(d, PORT_WORK, 0));
        assert_eq!(*RAN.lock().unwrap(), [1, 3], "closing one handle broke its duplicate");

        assert_eq!(XTaskQueueDuplicateHandle(d, &mut d2), S_OK);
        assert_eq!(XTaskQueueTerminate(d2, true, null_mut(), 0), S_OK);
        assert_eq!(XTaskQueueSubmitCallback(d, PORT_WORK, 4 as *mut c_void, cb(record)), E_ABORT, "terminate through a duplicate");
        XTaskQueueCloseHandle(d2);
        XTaskQueueCloseHandle(d);
        assert_eq!(XTaskQueueSubmitCallback(d, PORT_WORK, 5 as *mut c_void, cb(record)), E_INVALIDARG);
        assert_eq!(*RAN.lock().unwrap(), [1, 3]);
    }
}

/// XTaskQueueCreateComposite builds a queue from other queues' ports (XTaskQueue.h: "creates a
/// task queue composed of ports of other task queues"): its work is dispatched through the queue
/// that owns the port, each port keeps its own mode, and a composite's ports can seed another
/// composite. Bad ports and null outputs give E_INVALIDARG.
#[test]
fn composite_queues_share_their_parents_ports() {
    static RAN: Mutex<Vec<usize>> = Mutex::new(Vec::new());
    unsafe extern "system" fn record(ctx: *mut c_void, _canceled: bool) {
        RAN.lock().unwrap().push(ctx as usize);
    }
    unsafe {
        let a = create(MODE_MANUAL, MODE_MANUAL);
        let b = create(MODE_MANUAL, MODE_IMMEDIATE);
        let (mut aw, mut bc) = (null_mut(), null_mut());
        assert_eq!(XTaskQueueGetPort(a, PORT_WORK, &mut aw), S_OK);
        assert_eq!(XTaskQueueGetPort(b, PORT_COMPLETION, &mut bc), S_OK);
        assert!(!aw.is_null() && !bc.is_null());
        let mut c = null_mut();
        assert_eq!(XTaskQueueCreateComposite(aw, bc, &mut c), S_OK);

        assert_eq!(XTaskQueueSubmitCallback(c, PORT_WORK, 1 as *mut c_void, cb(record)), S_OK);
        assert!(RAN.lock().unwrap().is_empty(), "work on a's manual port ran undispatched");
        assert!(!XTaskQueueDispatch(b, PORT_WORK, 0), "b's work port got c's work");
        assert!(XTaskQueueDispatch(a, PORT_WORK, 0), "c's work is not on a's work port");
        assert_eq!(XTaskQueueSubmitCallback(c, PORT_COMPLETION, 2 as *mut c_void, cb(record)), S_OK);
        assert_eq!(*RAN.lock().unwrap(), [1, 2], "c's completion port is not b's immediate port");

        let mut cw = null_mut();
        assert_eq!(XTaskQueueGetPort(c, PORT_WORK, &mut cw), S_OK);
        let mut c2 = null_mut();
        assert_eq!(XTaskQueueCreateComposite(cw, cw, &mut c2), S_OK);
        assert_eq!(XTaskQueueSubmitCallback(c2, PORT_COMPLETION, 3 as *mut c_void, cb(record)), S_OK);
        assert!(XTaskQueueDispatch(a, PORT_WORK, 0), "a composite of a composite lost the port");
        assert_eq!(*RAN.lock().unwrap(), [1, 2, 3]);

        let mut out = null_mut();
        assert_eq!(XTaskQueueGetPort(a, 2, &mut out), E_INVALIDARG);
        assert_eq!(XTaskQueueGetPort(a, PORT_WORK, null_mut()), E_INVALIDARG);
        assert_eq!(XTaskQueueCreateComposite(null_mut(), bc, &mut out), E_INVALIDARG);
        assert_eq!(XTaskQueueCreateComposite(aw, null_mut(), &mut out), E_INVALIDARG);
        assert_eq!(XTaskQueueCreateComposite(aw, bc, null_mut()), E_INVALIDARG);
        let mut stray = 0u64;
        let stray = &mut stray as *mut u64 as *mut c_void;
        assert_eq!(XTaskQueueCreateComposite(stray, bc, &mut out), E_INVALIDARG);

        for h in [c2, c, b, a] {
            shut(h);
        }
    }
}

/// XTaskQueueTerminate with wait = true: every pending callback of both ports, delayed ones
/// included, runs with canceled = true, then the termination callback runs once with its context,
/// all before Terminate returns; afterwards the queue refuses new work with E_ABORT (XTaskQueue.h
/// and libHttpClient's VerifyNotTerminated).
#[test]
fn terminate_cancels_everything_pending_before_its_callback() {
    static LOG: Mutex<Vec<(usize, bool)>> = Mutex::new(Vec::new());
    unsafe extern "system" fn record(ctx: *mut c_void, canceled: bool) {
        LOG.lock().unwrap().push((ctx as usize, canceled));
    }
    unsafe extern "system" fn terminated(ctx: *mut c_void) {
        LOG.lock().unwrap().push((ctx as usize, false));
    }
    unsafe {
        let h = create(MODE_MANUAL, MODE_MANUAL);
        for i in [1usize, 2, 3] {
            assert_eq!(XTaskQueueSubmitCallback(h, PORT_WORK, i as *mut c_void, cb(record)), S_OK);
        }
        for i in [11usize, 12] {
            assert_eq!(XTaskQueueSubmitCallback(h, PORT_COMPLETION, i as *mut c_void, cb(record)), S_OK);
        }
        assert_eq!(XTaskQueueSubmitDelayedCallback(h, PORT_WORK, 60_000, 21 as *mut c_void, cb(record)), S_OK);
        assert_eq!(XTaskQueueSubmitDelayedCallback(h, PORT_COMPLETION, 60_000, 22 as *mut c_void, cb(record)), S_OK);
        assert_eq!(XTaskQueueTerminate(h, true, 99 as *mut c_void, term_cb(terminated)), S_OK);

        let log = LOG.lock().unwrap().clone();
        assert_eq!(log.len(), 8, "{log:?}");
        assert_eq!(log[7], (99, false), "the termination callback must run last, once");
        let mut canceled: Vec<usize> = log[..7]
            .iter()
            .map(|&(i, c)| {
                assert!(c, "pending callback {i} ran with canceled = false");
                i
            })
            .collect();
        canceled.sort();
        assert_eq!(canceled, [1, 2, 3, 11, 12, 21, 22]);

        assert_eq!(XTaskQueueSubmitCallback(h, PORT_WORK, 5 as *mut c_void, cb(record)), E_ABORT);
        assert_eq!(XTaskQueueSubmitCallback(h, PORT_COMPLETION, 5 as *mut c_void, cb(record)), E_ABORT);
        assert_eq!(XTaskQueueSubmitDelayedCallback(h, PORT_WORK, 1, 5 as *mut c_void, cb(record)), E_ABORT);
        assert!(!XTaskQueueDispatch(h, PORT_WORK, 0));
        assert!(XTaskQueueIsEmpty(h, PORT_WORK) && XTaskQueueIsEmpty(h, PORT_COMPLETION));
        assert_eq!(LOG.lock().unwrap().len(), 8, "work ran after Terminate");
        XTaskQueueCloseHandle(h);
    }
}

/// Terminate and a callback already running: the running callback keeps canceled = false and
/// finishes; the termination callback waits for it. With wait = false Terminate returns at once
/// and the termination callback comes later; with wait = true Terminate returns only after it
/// (XTaskQueue.h: XTaskQueueTerminate's wait parameter).
#[test]
fn terminate_waits_for_running_callbacks_only_when_asked() {
    struct Gate {
        started: AtomicBool,
        release: AtomicBool,
        canceled: AtomicBool,
        timed_out: AtomicBool,
        terminated: AtomicU32,
    }
    static NO_WAIT: Gate = Gate::new();
    static WAIT: Gate = Gate::new();
    impl Gate {
        const fn new() -> Gate {
            Gate {
                started: AtomicBool::new(false),
                release: AtomicBool::new(false),
                canceled: AtomicBool::new(false),
                timed_out: AtomicBool::new(false),
                terminated: AtomicU32::new(0),
            }
        }
    }
    unsafe extern "system" fn blocker(ctx: *mut c_void, canceled: bool) {
        let g = &*(ctx as *const Gate);
        g.canceled.store(canceled, SeqCst);
        g.started.store(true, SeqCst);
        let end = Instant::now() + DEADLINE;
        while !g.release.load(SeqCst) {
            if Instant::now() >= end {
                g.timed_out.store(true, SeqCst);
                return;
            }
            std::thread::sleep(Duration::from_millis(1));
        }
    }
    unsafe extern "system" fn terminated(ctx: *mut c_void) {
        (*(ctx as *const Gate)).terminated.fetch_add(1, SeqCst);
    }
    unsafe {
        let g = &NO_WAIT;
        let ctx = g as *const Gate as *mut c_void;
        let h = create(MODE_THREAD_POOL, MODE_THREAD_POOL);
        assert_eq!(XTaskQueueSubmitCallback(h, PORT_WORK, ctx, cb(blocker)), S_OK);
        wait_until("the blocking callback to start", || g.started.load(SeqCst));
        assert_eq!(XTaskQueueTerminate(h, false, ctx, term_cb(terminated)), S_OK);
        assert_eq!(g.terminated.load(SeqCst), 0, "termination callback ran while a callback was still running");
        assert_eq!(XTaskQueueSubmitCallback(h, PORT_WORK, ctx, cb(blocker)), E_ABORT);
        g.release.store(true, SeqCst);
        wait_until("the termination callback", || g.terminated.load(SeqCst) == 1);
        assert!(!g.timed_out.load(SeqCst), "wait = false blocked until the running callback gave up");
        assert!(!g.canceled.load(SeqCst), "a callback that started before Terminate saw canceled = true");
        XTaskQueueCloseHandle(h);

        let g = &WAIT;
        let ctx = g as *const Gate as usize;
        let h = create(MODE_THREAD_POOL, MODE_THREAD_POOL);
        assert_eq!(XTaskQueueSubmitCallback(h, PORT_WORK, ctx as *mut c_void, cb(blocker)), S_OK);
        wait_until("the blocking callback to start", || g.started.load(SeqCst));
        let addr = h as usize;
        let (tx, rx) = mpsc::channel();
        let terminator = std::thread::spawn(move || {
            let hr = XTaskQueueTerminate(addr as *mut c_void, true, ctx as *mut c_void, term_cb(terminated));
            let _ = tx.send((hr, (*(ctx as *const Gate)).terminated.load(SeqCst)));
        });
        assert!(rx.recv_timeout(Duration::from_millis(200)).is_err(), "wait = true returned while a callback was running");
        g.release.store(true, SeqCst);
        let (hr, seen) = rx.recv_timeout(DEADLINE).expect("Terminate(wait = true) never returned");
        terminator.join().unwrap();
        assert_eq!(hr, S_OK);
        assert_eq!(seen, 1, "Terminate(wait = true) returned before its callback ran");
        assert!(!g.canceled.load(SeqCst) && !g.timed_out.load(SeqCst));
        XTaskQueueCloseHandle(h);
    }
}

/// Argument checks (libHttpClient returns E_INVALIDARG for each): unknown modes and ports, null
/// outputs and callbacks, and handles that are not live queues; Dispatch on such a handle is
/// false, and closing one (or null) is harmless.
#[test]
fn bad_arguments_and_handles_are_refused() {
    unsafe extern "system" fn noop(_ctx: *mut c_void, _canceled: bool) {}
    unsafe extern "system" fn monitor(_ctx: *mut c_void, _q: *mut c_void, _port: u32) {}
    unsafe {
        let mut h = null_mut();
        assert_eq!(XTaskQueueCreate(4, MODE_MANUAL, &mut h), E_INVALIDARG);
        assert_eq!(XTaskQueueCreate(MODE_MANUAL, 9, &mut h), E_INVALIDARG);
        assert_eq!(XTaskQueueCreate(MODE_MANUAL, MODE_MANUAL, null_mut()), E_INVALIDARG);
        assert!(h.is_null(), "a refused Create wrote a handle");

        let mut word = 0u64;
        let stray = &mut word as *mut u64 as *mut c_void; // never a queue handle
        let mut out = null_mut();
        assert_eq!(XTaskQueueSubmitCallback(stray, PORT_WORK, null_mut(), cb(noop)), E_INVALIDARG);
        assert_eq!(XTaskQueueSubmitDelayedCallback(stray, PORT_WORK, 5, null_mut(), cb(noop)), E_INVALIDARG);
        assert_eq!(XTaskQueueTerminate(stray, true, null_mut(), 0), E_INVALIDARG);
        assert_eq!(XTaskQueueDuplicateHandle(stray, &mut out), E_INVALIDARG);
        assert_eq!(XTaskQueueGetPort(stray, PORT_WORK, &mut out), E_INVALIDARG);
        assert!(!XTaskQueueDispatch(stray, PORT_WORK, 0));
        XTaskQueueCloseHandle(stray);
        XTaskQueueCloseHandle(null_mut());
        assert_eq!(word, 0, "a stray handle was written through");

        let q = create(MODE_MANUAL, MODE_MANUAL);
        assert_eq!(XTaskQueueSubmitCallback(q, 2, null_mut(), cb(noop)), E_INVALIDARG);
        assert_eq!(XTaskQueueSubmitCallback(q, PORT_WORK, null_mut(), 0), E_INVALIDARG);
        assert_eq!(XTaskQueueSubmitDelayedCallback(q, 2, 5, null_mut(), cb(noop)), E_INVALIDARG);
        assert_eq!(XTaskQueueSubmitDelayedCallback(q, PORT_WORK, 5, null_mut(), 0), E_INVALIDARG);
        assert_eq!(XTaskQueueDuplicateHandle(q, null_mut()), E_INVALIDARG);
        let mut tok = RegistrationToken { token: 0 };
        assert_eq!(XTaskQueueRegisterMonitor(q, null_mut(), 0, &mut tok), E_INVALIDARG);
        assert_eq!(XTaskQueueRegisterMonitor(q, null_mut(), monitor as XTaskQueueMonitorCallback as usize, null_mut()), E_INVALIDARG);
        assert!(XTaskQueueIsEmpty(q, PORT_WORK), "a refused submit queued work");
        shut(q);
    }
}

/// A null queue handle means the process task queue, which by default exists and runs on the
/// thread pool; XTaskQueueGetCurrentProcessTaskQueue hands out a handle to it that the caller
/// closes without affecting the process queue (XTaskQueue.h).
#[test]
fn null_handle_and_get_current_process_queue_reach_the_process_queue() {
    static SEEN: Mutex<Vec<(usize, ThreadId)>> = Mutex::new(Vec::new());
    unsafe extern "system" fn record(ctx: *mut c_void, _canceled: bool) {
        SEEN.lock().unwrap().push((ctx as usize, std::thread::current().id()));
    }
    let me = std::thread::current().id();
    unsafe {
        let mut p = null_mut();
        assert!(XTaskQueueGetCurrentProcessTaskQueue(&mut p));
        assert!(!p.is_null());
        assert!(!XTaskQueueGetCurrentProcessTaskQueue(null_mut()));
        assert_eq!(XTaskQueueSubmitCallback(p, PORT_WORK, 1 as *mut c_void, cb(record)), S_OK);
        assert_eq!(XTaskQueueSubmitCallback(null_mut(), PORT_COMPLETION, 2 as *mut c_void, cb(record)), S_OK);
        wait_until("two process-queue callbacks", || SEEN.lock().unwrap().len() == 2);
        XTaskQueueCloseHandle(p);
        assert_eq!(XTaskQueueSubmitDelayedCallback(null_mut(), PORT_WORK, 10, 3 as *mut c_void, cb(record)), S_OK);
        wait_until("a process-queue callback after closing a handle to it", || SEEN.lock().unwrap().len() == 3);
        let seen = SEEN.lock().unwrap().clone();
        let mut ctxs: Vec<usize> = seen.iter().map(|s| s.0).collect();
        ctxs.sort();
        assert_eq!(ctxs, [1, 2, 3]);
        assert!(seen.iter().all(|s| s.1 != me), "the default process queue ran a callback inline");
    }
}

