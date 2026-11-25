//! XAsync and the XAsyncProvider protocol through the exported functions, against libHttpClient's
//! XAsync.h / XAsyncProvider.h (MIT), the contract the module cites. Blocks that queued work may
//! still touch are leaked, so a failing test cannot leave a callback writing to a dead stack
//! frame; every wait has a long deadline so a slow machine fails a test instead of hanging it.

use std::ffi::c_void;
use std::ptr::{null, null_mut};
use std::sync::atomic::{AtomicU32, Ordering::SeqCst};
use std::sync::{mpsc, Mutex};
use std::thread::ThreadId;
use std::time::{Duration, Instant};

use xcl::hle::xgameruntime::taskqueue::{self, MODE_IMMEDIATE, MODE_MANUAL, MODE_THREAD_POOL, PORT_COMPLETION, PORT_WORK};
use xcl::hle::xgameruntime::xasync::*;
use xcl::util::{E_ABORT, E_FAIL, E_INVALIDARG, E_NOT_SUFFICIENT_BUFFER, E_PENDING, HRESULT, S_FALSE, S_OK};

const DEADLINE: Duration = Duration::from_secs(20);
const E_CUSTOM: HRESULT = 0x8000_1234u32 as i32;

/// What a test provider or work function saw, one per call: the ops in order, the buffer sizes
/// GetResult got, the completion callbacks (status, thread) and the work threads.
struct Rec {
    tag: usize,
    block: usize,
    ops: Mutex<Vec<u32>>,
    sizes: Mutex<Vec<usize>>,
    wrong_block: AtomicU32,
    callbacks: Mutex<Vec<(HRESULT, ThreadId)>>,
    work_threads: Mutex<Vec<ThreadId>>,
    stamps: Mutex<Vec<Instant>>,
}

impl Rec {
    fn ops(&self) -> Vec<u32> {
        self.ops.lock().unwrap().clone()
    }
    fn callbacks(&self) -> Vec<(HRESULT, ThreadId)> {
        self.callbacks.lock().unwrap().clone()
    }
}

/// A leaked zeroed block on `queue` whose context is a fresh leaked `Rec` (also handed to the
/// provider as its context).
fn new_op(queue: *mut c_void, with_callback: bool, tag: usize) -> (*mut XAsyncBlock, &'static Rec) {
    let b: *mut XAsyncBlock = Box::leak(Box::new(unsafe { std::mem::zeroed::<XAsyncBlock>() }));
    let rec: &'static Rec = Box::leak(Box::new(Rec {
        tag,
        block: b as usize,
        ops: Mutex::new(Vec::new()),
        sizes: Mutex::new(Vec::new()),
        wrong_block: AtomicU32::new(0),
        callbacks: Mutex::new(Vec::new()),
        work_threads: Mutex::new(Vec::new()),
        stamps: Mutex::new(Vec::new()),
    }));
    unsafe {
        (*b).queue = queue;
        (*b).context = rec as *const Rec as *mut c_void;
        if with_callback {
            (*b).callback = on_complete as unsafe extern "system" fn(*mut XAsyncBlock) as usize;
        }
    }
    (b, rec)
}

unsafe fn begin(b: *mut XAsyncBlock, rec: &Rec, identity: usize, p: XAsyncProvider) -> HRESULT {
    XAsyncBegin(b, rec as *const Rec as *mut c_void, identity as *const c_void, null(), p as usize)
}

unsafe fn run(b: *mut XAsyncBlock, w: XAsyncWork) -> HRESULT {
    XAsyncRun(b, w as usize)
}

unsafe extern "system" fn on_complete(b: *mut XAsyncBlock) {
    let rec = &*((*b).context as *const Rec);
    rec.callbacks.lock().unwrap().push((XAsyncGetStatus(b, false), std::thread::current().id()));
}

/// Log `op` in the provider's `Rec` and return it.
unsafe fn log(op: u32, data: *const XAsyncProviderData) -> &'static Rec {
    let rec = &*((*data).context as *const Rec);
    if (*data).async_ as usize != rec.block {
        rec.wrong_block.fetch_add(1, SeqCst);
    }
    rec.ops.lock().unwrap().push(op);
    rec
}

/// Begin schedules DoWork; DoWork completes with a 6-byte result; GetResult writes it.
unsafe extern "system" fn p_schedules(op: u32, data: *const XAsyncProviderData) -> HRESULT {
    let rec = log(op, data);
    match op {
        OP_BEGIN => XAsyncSchedule((*data).async_, 0),
        OP_DO_WORK => {
            XAsyncComplete((*data).async_, S_OK, 6);
            S_OK
        }
        OP_GET_RESULT => {
            rec.sizes.lock().unwrap().push((*data).buffer_size);
            std::ptr::copy_nonoverlapping(b"result".as_ptr(), (*data).buffer as *mut u8, 6);
            S_OK
        }
        _ => S_OK,
    }
}

/// Does nothing on its own: the test completes the call. GetResult writes 16 bytes 0..16.
unsafe extern "system" fn p_passive(op: u32, data: *const XAsyncProviderData) -> HRESULT {
    let rec = log(op, data);
    if op == OP_GET_RESULT {
        rec.sizes.lock().unwrap().push((*data).buffer_size);
        let bytes: Vec<u8> = (0..16).collect();
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), (*data).buffer as *mut u8, 16);
    }
    S_OK
}

unsafe fn create(work: u32, completion: u32) -> *mut c_void {
    let mut h = null_mut();
    assert_eq!(taskqueue::XTaskQueueCreate(work, completion, &mut h), S_OK);
    h
}

unsafe fn shut(h: *mut c_void) {
    assert_eq!(taskqueue::XTaskQueueTerminate(h, true, null_mut(), 0), S_OK);
    taskqueue::XTaskQueueCloseHandle(h);
}

fn wait_until(what: &str, done: impl Fn() -> bool) {
    let end = Instant::now() + DEADLINE;
    while !done() {
        assert!(Instant::now() < end, "timed out waiting for {what}");
        std::thread::sleep(Duration::from_millis(2));
    }
}

/// The provider protocol (XAsyncProvider.h, XAsyncOp): Begin runs inside XAsyncBegin, DoWork runs
/// from the block queue's work port, the completion callback from its completion port, and
/// GetResult then Cleanup run inside XAsyncGetResult with the caller's buffer size. Each op gets
/// the caller's block and the provider context given to XAsyncBegin.
#[test]
fn provider_ops_run_in_protocol_order_on_the_blocks_queue() {
    unsafe {
        let q = create(MODE_MANUAL, MODE_MANUAL);
        let (b, rec) = new_op(q, true, 0);
        assert_eq!(begin(b, rec, 0xABC, p_schedules), S_OK);
        assert_eq!(rec.ops(), [OP_BEGIN]);
        assert_eq!(XAsyncGetStatus(b, false), E_PENDING);
        assert!(!taskqueue::XTaskQueueDispatch(q, PORT_COMPLETION, 0), "a completion was queued before the work ran");

        assert!(taskqueue::XTaskQueueDispatch(q, PORT_WORK, 0), "Begin's XAsyncSchedule queued nothing on the work port");
        assert_eq!(rec.ops(), [OP_BEGIN, OP_DO_WORK]);
        assert_eq!(XAsyncGetStatus(b, false), S_OK);
        assert!(rec.callbacks().is_empty(), "the completion callback ran before its port was dispatched");
        let mut size = 0usize;
        assert_eq!(XAsyncGetResultSize(b, &mut size), S_OK);
        assert_eq!(size, 6);

        assert!(taskqueue::XTaskQueueDispatch(q, PORT_COMPLETION, 0), "no completion callback on the completion port");
        let cbs = rec.callbacks();
        assert_eq!(cbs.len(), 1);
        assert_eq!(cbs[0], (S_OK, std::thread::current().id()));

        let mut buf = [0u8; 16];
        let mut used = 0usize;
        assert_eq!(XAsyncGetResult(b, 0xABC as *const c_void, 16, buf.as_mut_ptr() as *mut c_void, &mut used), S_OK);
        assert_eq!((used, &buf[..6]), (6, &b"result"[..]));
        assert_eq!(rec.ops(), [OP_BEGIN, OP_DO_WORK, OP_GET_RESULT, OP_CLEANUP]);
        assert_eq!(*rec.sizes.lock().unwrap(), [16], "GetResult must see the caller's buffer size");
        assert_eq!(XAsyncGetResult(b, 0xABC as *const c_void, 16, buf.as_mut_ptr() as *mut c_void, &mut used), E_ILLEGAL_METHOD_CALL);
        assert_eq!(XAsyncGetStatus(b, false), S_OK, "the status outlives the consumed result");
        assert_eq!(rec.wrong_block.load(SeqCst), 0, "a provider op got another block");
        assert!(!taskqueue::XTaskQueueDispatch(q, PORT_WORK, 0) && !taskqueue::XTaskQueueDispatch(q, PORT_COMPLETION, 0));
        shut(q);
    }
}

/// XAsyncGetStatus(wait = false) is E_PENDING until XAsyncComplete; wait = true blocks until the
/// call completes and returns its result, a success code other than S_OK included (XAsync.h).
#[test]
fn get_status_waits_for_completion_and_keeps_the_result_code() {
    unsafe {
        let (b, rec) = new_op(null_mut(), false, 0);
        assert_eq!(begin(b, rec, 0, p_passive), S_OK);
        assert_eq!(XAsyncGetStatus(b, false), E_PENDING);
        assert_eq!(XAsyncGetResultSize(b, &mut 0usize), E_PENDING);
        assert_eq!(XAsyncGetResult(b, null(), 0, null_mut(), null_mut()), E_PENDING);

        let addr = b as usize;
        let (tx, rx) = mpsc::channel();
        let waiter = std::thread::spawn(move || {
            let _ = tx.send(XAsyncGetStatus(addr as *mut XAsyncBlock, true));
        });
        assert!(rx.recv_timeout(Duration::from_millis(100)).is_err(), "wait = true returned before completion");
        XAsyncComplete(b, S_FALSE, 0);
        assert_eq!(rx.recv_timeout(DEADLINE).expect("XAsyncGetStatus(wait = true) never returned"), S_FALSE);
        waiter.join().unwrap();
        assert_eq!(XAsyncGetStatus(b, false), S_FALSE);
        assert_eq!(XAsyncGetStatus(b, true), S_FALSE);
        let mut size = 0xAAAAusize;
        assert_eq!(XAsyncGetResultSize(b, &mut size), S_FALSE);
        assert_eq!(size, 0);
        assert_eq!(rec.ops(), [OP_BEGIN, OP_CLEANUP]);
    }
}

/// XAsyncGetResult (XAsync.h): a buffer smaller than XAsyncGetResultSize, or none, is
/// E_NOT_SUFFICIENT_BUFFER and leaves the result to read; an identity other than XAsyncBegin's is
/// E_INVALIDARG; a big enough buffer gets the result, its used size, and GetResult then Cleanup.
#[test]
fn get_result_needs_a_big_enough_buffer_and_the_begin_identity() {
    unsafe {
        let (b, rec) = new_op(null_mut(), false, 0);
        assert_eq!(begin(b, rec, 0x51, p_passive), S_OK);
        XAsyncComplete(b, S_OK, 16);
        let mut size = 0usize;
        assert_eq!(XAsyncGetResultSize(b, &mut size), S_OK);
        assert_eq!(size, 16);

        let mut buf = [0xEEu8; 32];
        let mut used = 0usize;
        let id = 0x51 as *const c_void;
        assert_eq!(XAsyncGetResult(b, id, 15, buf.as_mut_ptr() as *mut c_void, &mut used), E_NOT_SUFFICIENT_BUFFER);
        assert_eq!(XAsyncGetResult(b, id, 0, buf.as_mut_ptr() as *mut c_void, &mut used), E_NOT_SUFFICIENT_BUFFER);
        assert_eq!(XAsyncGetResult(b, id, 32, null_mut(), &mut used), E_NOT_SUFFICIENT_BUFFER);
        assert_eq!(XAsyncGetResult(b, 0x52 as *const c_void, 32, buf.as_mut_ptr() as *mut c_void, &mut used), E_INVALIDARG);
        assert_eq!(rec.ops(), [OP_BEGIN], "a refused XAsyncGetResult reached the provider");
        assert!(buf.iter().all(|&x| x == 0xEE), "a refused XAsyncGetResult wrote the buffer");
        assert_eq!(XAsyncGetResultSize(b, &mut size), S_OK);
        assert_eq!(size, 16, "a refused XAsyncGetResult consumed the result");

        assert_eq!(XAsyncGetResult(b, id, 32, buf.as_mut_ptr() as *mut c_void, &mut used), S_OK);
        assert_eq!(used, 16);
        assert_eq!(buf[..16], (0..16).collect::<Vec<u8>>()[..]);
        assert_eq!(rec.ops(), [OP_BEGIN, OP_GET_RESULT, OP_CLEANUP]);
        assert_eq!(*rec.sizes.lock().unwrap(), [32]);
    }
}

/// A call that completes with an error has no result: XAsyncGetResult returns the error without
/// a GetResult op, Cleanup runs once, and a second XAsyncComplete changes nothing (XAsync.h,
/// XAsyncProvider.h: "the required buffer size is ignored on failure").
#[test]
fn a_failed_call_has_no_result_and_completes_once() {
    unsafe {
        let (b, rec) = new_op(null_mut(), false, 0);
        assert_eq!(begin(b, rec, 0, p_passive), S_OK);
        XAsyncComplete(b, E_FAIL, 8);
        assert_eq!(XAsyncGetStatus(b, false), E_FAIL);
        assert_eq!(XAsyncGetResultSize(b, &mut 0usize), E_FAIL);
        let mut buf = [0u8; 8];
        assert_eq!(XAsyncGetResult(b, null(), 8, buf.as_mut_ptr() as *mut c_void, null_mut()), E_FAIL);
        XAsyncComplete(b, S_OK, 0);
        assert_eq!(XAsyncGetStatus(b, false), E_FAIL, "a second XAsyncComplete changed the result");
        assert_eq!(rec.ops(), [OP_BEGIN, OP_CLEANUP]);
    }
}

/// Begin defers DoWork 60 s; Cancel completes the call with E_ABORT.
unsafe extern "system" fn p_cancelable(op: u32, data: *const XAsyncProviderData) -> HRESULT {
    log(op, data);
    match op {
        OP_BEGIN => XAsyncSchedule((*data).async_, 60_000),
        OP_CANCEL => {
            XAsyncComplete((*data).async_, E_ABORT, 0);
            S_OK
        }
        _ => S_OK,
    }
}

/// XAsyncCancel hands the provider a Cancel op; one that completes with E_ABORT ends the call:
/// status E_ABORT, one completion callback, Cleanup, and the DoWork it scheduled never runs, even
/// when the queue is later terminated. Cancel on a finished call does not reach the provider.
#[test]
fn cancel_asks_the_provider_and_its_e_abort_ends_the_call() {
    unsafe {
        let q = create(MODE_MANUAL, MODE_IMMEDIATE);
        let (b, rec) = new_op(q, true, 0);
        assert_eq!(begin(b, rec, 0, p_cancelable), S_OK);
        assert_eq!(XAsyncGetStatus(b, false), E_PENDING);
        XAsyncCancel(b);
        assert_eq!(XAsyncGetStatus(b, false), E_ABORT);
        assert_eq!(rec.ops(), [OP_BEGIN, OP_CANCEL, OP_CLEANUP]);
        let cbs = rec.callbacks();
        assert_eq!(cbs.len(), 1, "one completion callback");
        assert_eq!(cbs[0].0, E_ABORT);
        XAsyncCancel(b);
        shut(q); // cancels the 60 s DoWork item
        assert_eq!(rec.ops(), [OP_BEGIN, OP_CANCEL, OP_CLEANUP], "a canceled call got more provider ops");
        assert_eq!(rec.callbacks().len(), 1);
        assert_eq!(XAsyncGetStatus(b, false), E_ABORT);
    }
}

/// Begin stamps the time and schedules DoWork 200 ms out; DoWork stamps the time and returns.
unsafe extern "system" fn p_delayed(op: u32, data: *const XAsyncProviderData) -> HRESULT {
    let rec = log(op, data);
    match op {
        OP_BEGIN => {
            rec.stamps.lock().unwrap().push(Instant::now());
            XAsyncSchedule((*data).async_, 200)
        }
        OP_DO_WORK => {
            rec.stamps.lock().unwrap().push(Instant::now());
            S_OK
        }
        _ => S_OK,
    }
}

/// XAsyncSchedule's delay holds DoWork back at least that long; a DoWork that returns a result
/// other than E_PENDING without calling XAsyncComplete completes the call with it (libHttpClient
/// XAsync.cpp, the worker callback).
#[test]
fn schedule_delay_holds_do_work_back() {
    unsafe {
        let q = create(MODE_MANUAL, MODE_MANUAL);
        let (b, rec) = new_op(q, false, 0);
        assert_eq!(begin(b, rec, 0, p_delayed), S_OK);
        assert!(taskqueue::XTaskQueueDispatch(q, PORT_WORK, DEADLINE.as_millis() as u32), "the delayed DoWork never came");
        let stamps = rec.stamps.lock().unwrap().clone();
        assert_eq!(stamps.len(), 2);
        assert!(stamps[1] - stamps[0] >= Duration::from_millis(200), "DoWork ran {:?} after a 200 ms schedule", stamps[1] - stamps[0]);
        assert_eq!(XAsyncGetStatus(b, false), S_OK);
        assert_eq!(rec.ops(), [OP_BEGIN, OP_DO_WORK, OP_CLEANUP]);
        shut(q);
    }
}

/// First DoWork reschedules itself and returns E_PENDING; the second returns E_CUSTOM.
unsafe extern "system" fn p_two_steps(op: u32, data: *const XAsyncProviderData) -> HRESULT {
    let rec = log(op, data);
    match op {
        OP_BEGIN => XAsyncSchedule((*data).async_, 0),
        OP_DO_WORK if rec.ops().iter().filter(|&&o| o == OP_DO_WORK).count() == 1 => {
            let hr = XAsyncSchedule((*data).async_, 0);
            if hr < 0 {
                hr
            } else {
                E_PENDING
            }
        }
        OP_DO_WORK => E_CUSTOM,
        _ => S_OK,
    }
}

/// A DoWork that returns E_PENDING keeps the call open, and may schedule itself again, as
/// multi-step providers do (XAsyncProvider.h, XAsyncOp::DoWork); the step that returns a result
/// ends the call with it.
#[test]
fn do_work_returning_e_pending_keeps_the_call_open() {
    unsafe {
        let q = create(MODE_MANUAL, MODE_MANUAL);
        let (b, rec) = new_op(q, false, 0);
        assert_eq!(begin(b, rec, 0, p_two_steps), S_OK);
        assert!(taskqueue::XTaskQueueDispatch(q, PORT_WORK, 0));
        assert_eq!(XAsyncGetStatus(b, false), E_PENDING, "E_PENDING from DoWork completed the call");
        assert!(taskqueue::XTaskQueueDispatch(q, PORT_WORK, 0), "the rescheduled DoWork was not queued");
        assert_eq!(XAsyncGetStatus(b, false), E_CUSTOM);
        assert_eq!(rec.ops(), [OP_BEGIN, OP_DO_WORK, OP_DO_WORK, OP_CLEANUP]);
        assert!(!taskqueue::XTaskQueueDispatch(q, PORT_WORK, 0));
        shut(q);
    }
}

/// XAsyncRun's work function: records its thread and returns the call's tag as its result.
unsafe extern "system" fn tagged_work(b: *mut XAsyncBlock) -> HRESULT {
    let rec = &*((*b).context as *const Rec);
    rec.work_threads.lock().unwrap().push(std::thread::current().id());
    rec.tag as HRESULT
}

/// XAsyncRun (XAsync.h): the work function runs on the thread that dispatches the block queue's
/// work port, its return value is the call's result, and the completion callback runs on the
/// thread that dispatches the completion port.
#[test]
fn xasync_run_work_and_callback_run_where_their_ports_are_dispatched() {
    unsafe {
        let q = create(MODE_MANUAL, MODE_MANUAL);
        let (b, rec) = new_op(q, true, 3);
        assert_eq!(run(b, tagged_work), S_OK);
        assert_eq!(XAsyncGetStatus(b, false), E_PENDING);
        assert!(rec.work_threads.lock().unwrap().is_empty(), "XAsyncRun ran the work before a dispatch");
        let qa = q as usize;
        let dispatch_on_a_thread = move |port: u32| {
            std::thread::spawn(move || (taskqueue::XTaskQueueDispatch(qa as *mut c_void, port, 0), std::thread::current().id()))
                .join()
                .unwrap()
        };
        let (ran, worker) = dispatch_on_a_thread(PORT_WORK);
        assert!(ran);
        assert_eq!(*rec.work_threads.lock().unwrap(), [worker]);
        assert_eq!(XAsyncGetStatus(b, false), 3);
        assert!(rec.callbacks().is_empty());
        let (ran, completer) = dispatch_on_a_thread(PORT_COMPLETION);
        assert!(ran);
        assert_eq!(rec.callbacks(), [(3, completer)]);
        shut(q);
    }
}

/// Many XAsyncRun calls on one ThreadPool queue: each gets its own result and exactly one
/// completion callback.
#[test]
fn concurrent_runs_on_a_thread_pool_queue_keep_their_own_results() {
    unsafe {
        let q = create(MODE_THREAD_POOL, MODE_THREAD_POOL);
        let ops: Vec<_> = (0..32usize).map(|i| new_op(q, true, 100 + i)).collect();
        for &(b, _) in &ops {
            assert_eq!(run(b, tagged_work), S_OK);
        }
        wait_until("32 completion callbacks", || ops.iter().all(|(_, r)| !r.callbacks().is_empty()));
        shut(q);
        for &(b, rec) in &ops {
            let want = rec.tag as HRESULT;
            assert_eq!(XAsyncGetStatus(b, false), want);
            let cbs = rec.callbacks();
            assert_eq!(cbs.len(), 1, "call {} got {} completion callbacks", rec.tag, cbs.len());
            assert_eq!(cbs[0].0, want, "call {}'s callback saw another status", rec.tag);
            assert_eq!(rec.work_threads.lock().unwrap().len(), 1);
        }
    }
}

/// A block with a null queue uses the process task queue (XAsync.h, XAsyncBlock::queue), which by
/// default runs work and completion callbacks on the thread pool.
#[test]
fn a_null_queue_runs_on_the_process_queue() {
    let me = std::thread::current().id();
    unsafe {
        let (b, rec) = new_op(null_mut(), true, 5);
        assert_eq!(run(b, tagged_work), S_OK);
        wait_until("the process queue's completion callback", || !rec.callbacks().is_empty());
        assert_eq!(XAsyncGetStatus(b, false), 5);
        assert_ne!(rec.work_threads.lock().unwrap()[0], me, "process-queue work ran inline");
        assert_ne!(rec.callbacks()[0].1, me, "process-queue completion ran inline");

        let (b, rec) = new_op(null_mut(), true, 0);
        assert_eq!(begin(b, rec, 7, p_schedules), S_OK);
        wait_until("the provider's call on the process queue", || !rec.callbacks().is_empty());
        let mut buf = [0u8; 6];
        let mut used = 0usize;
        assert_eq!(XAsyncGetResult(b, 7 as *const c_void, 6, buf.as_mut_ptr() as *mut c_void, &mut used), S_OK);
        assert_eq!((&buf, used), (b"result", 6));
        assert_eq!(rec.ops(), [OP_BEGIN, OP_DO_WORK, OP_GET_RESULT, OP_CLEANUP]);
    }
}

/// Argument checks: a null block or provider or work function is E_INVALIDARG, and a block that
/// was never started reads as E_INVALIDARG; Cancel and Complete on it are harmless no-ops.
#[test]
fn bad_arguments_and_unstarted_blocks_are_refused() {
    unsafe {
        let (b, rec) = new_op(null_mut(), false, 0);
        assert_eq!(XAsyncBegin(null_mut(), null_mut(), null(), null(), p_passive as XAsyncProvider as usize), E_INVALIDARG);
        assert_eq!(XAsyncBegin(b, null_mut(), null(), null(), 0), E_INVALIDARG);
        assert_eq!(run(null_mut(), tagged_work), E_INVALIDARG);
        assert_eq!(XAsyncRun(b, 0), E_INVALIDARG);
        assert_eq!(XAsyncSchedule(null_mut(), 0), E_INVALIDARG);
        assert_eq!(XAsyncGetStatus(null_mut(), false), E_INVALIDARG);

        assert_eq!(XAsyncGetStatus(b, false), E_INVALIDARG, "a never-started block");
        assert_eq!(XAsyncGetResultSize(b, &mut 0usize), E_INVALIDARG);
        assert_eq!(XAsyncGetResult(b, null(), 0, null_mut(), null_mut()), E_INVALIDARG);
        XAsyncCancel(b);
        XAsyncComplete(b, S_OK, 0);
        assert_eq!(XAsyncGetStatus(b, false), E_INVALIDARG, "Complete on a never-started block started it");
        assert!(rec.ops().is_empty() && rec.work_threads.lock().unwrap().is_empty());
    }
}
