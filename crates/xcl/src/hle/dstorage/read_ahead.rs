//! Read-ahead for a queue's file requests: while the worker inflates or copies one request,
//! reader threads read the source bytes of the next few from their files, so the disk and the
//! CPU work at the same time instead of in turn, and an SSD gets several reads at once.
//!
//! Nothing a title can see changes. The worker still runs every entry in queue order and makes
//! the same checks at the same point (`execute`): a request's destination is checked first, a
//! file the title closed meanwhile still fails with E_FILE_HANDLE, and a read that failed
//! gives the same HRESULT as when the worker reads it. A request read ahead and then cancelled,
//! or reached after Close, is skipped as before; only its bytes are dropped. The readers write
//! only into their own buffers, never into the title's memory.
//!
//! Parallel reads: the files are synchronous handles, on which Windows runs one read at a time,
//! so each reader reads with a handle of its own: reader 0 with the title's, readers 1 to 3 with
//! handles `ReOpenFile` makes from it (read-only, sharing everything; the same file object the
//! VFS opened, so no path goes through its redirection again), kept with the file and closed
//! when the title closes it. Only on a drive that reports no seek penalty: on a hard disk,
//! reads out of order would seek, so its files keep one reader that reads in queue order, as
//! does a file `ReOpenFile` fails on. A request with nothing to overlap but other reads is read
//! ahead only while reads come from the device: reads from the file cache take microseconds,
//! less than handing them to another thread.

use std::collections::VecDeque;
use std::os::windows::io::{AsRawHandle, FromRawHandle};
use std::time::Instant;

use windows_sys::Win32::Foundation::{GENERIC_READ, INVALID_HANDLE_VALUE};
use windows_sys::Win32::Storage::FileSystem::{
    CreateFileW, GetFinalPathNameByHandleW, ReOpenFile, FILE_SHARE_DELETE, FILE_SHARE_READ, FILE_SHARE_WRITE, OPEN_EXISTING, VOLUME_NAME_GUID,
};
use windows_sys::Win32::System::IO::DeviceIoControl;

use super::*;

/// At most this many requests read ahead with one reader ...
const MAX_REQUESTS: usize = 4;
/// ... and with several readers, on a file whose reads run in parallel ...
const PARALLEL_REQUESTS: usize = 8;
/// ... holding at most this many bytes; a bigger request is read by the worker.
const MAX_BYTES: usize = 32 << 20;
/// A plain copy this large is work a read can overlap (a zlib request always is). Behind small
/// plain copies a read would overlap nothing and only add a hand-over between threads.
const HEAVY_COPY: u32 = 512 << 10;
/// Readers of a queue, and so handles of a file: the title's and up to 3 from `ReOpenFile`.
/// `XCL_DSTORAGE_READERS` (1 to 4) lowers it; 1 is one reader in queue order on every drive.
const MAX_READERS: usize = 4;
/// A read this slow came from the device, not the file cache.
const SLOW_READ_US: u32 = 50;
/// While reads are fast, one in this many requests with nothing to overlap is still read
/// ahead, to see whether they still are.
const PROBE_EVERY: u32 = 32;

pub(super) type ReadResult = Result<Vec<u8>, HRESULT>;

fn max_readers() -> usize {
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| std::env::var("XCL_DSTORAGE_READERS").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(MAX_READERS).clamp(1, MAX_READERS))
}

/// A file's handles for parallel reads, kept with the file.
#[derive(Default)]
pub(super) struct FileReaders {
    /// reader k (1 to 3) reads with `more[k - 1]`
    more: Mutex<[Option<Arc<std::fs::File>>; MAX_READERS - 1]>,
    /// its drive has no seek penalty (asked once)
    no_seek_penalty: OnceLock<bool>,
    reopen_failed: AtomicBool,
}

impl FileReaders {
    /// The title closed the file (after taking its handle): the extra handles are closed too;
    /// a read still running keeps its handle until it ends.
    pub(super) fn close(&self) {
        *self.more.lock().unwrap() = Default::default();
    }

    /// Whether the file's reads may run in parallel.
    fn parallel(&self, main: &std::fs::File) -> bool {
        !self.reopen_failed.load(Ordering::Relaxed) && *self.no_seek_penalty.get_or_init(|| unsafe { no_seek_penalty(main.as_raw_handle() as HANDLE) })
    }

    /// The handle reader `k` reads `owner` with: the title's (`main`, taken when the read was
    /// started) for reader 0, after the title closed the file, and where `ReOpenFile` fails.
    fn handle(owner: &FileInner, k: usize, main: &Arc<std::fs::File>) -> Arc<std::fs::File> {
        let r = &owner.readers;
        if k == 0 || k >= MAX_READERS || r.reopen_failed.load(Ordering::Relaxed) {
            return Arc::clone(main);
        }
        let mut more = r.more.lock().unwrap();
        if let Some(h) = &more[k - 1] {
            return Arc::clone(h);
        }
        // a handle is made only while the title has the file open: its Close takes its own
        // handle first, then clears these (under this lock)
        let Some(open) = owner.handle() else { return Arc::clone(main) };
        match unsafe { reopen(&open) } {
            Some(h) => {
                let h = Arc::new(h);
                more[k - 1] = Some(Arc::clone(&h));
                h
            }
            None => {
                r.reopen_failed.store(true, Ordering::Relaxed);
                Arc::clone(main)
            }
        }
    }
}

/// A second handle of the same file object, read-only, sharing everything, synchronous.
unsafe fn reopen(f: &std::fs::File) -> Option<std::fs::File> {
    let h = ReOpenFile(f.as_raw_handle() as HANDLE, GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, 0);
    (h != INVALID_HANDLE_VALUE && !h.is_null()).then(|| std::fs::File::from_raw_handle(h as _))
}

/// Whether the volume of the file `h` reports no seek penalty (an SSD). Unknown (a network
/// share, a volume over several disks, a query refused) counts as a seek penalty.
unsafe fn no_seek_penalty(h: HANDLE) -> bool {
    // "\\?\Volume{GUID}\path" -> "\\.\Volume{GUID}" (a device path the VFS passes through)
    let mut buf = [0u16; 1024];
    let n = GetFinalPathNameByHandleW(h, buf.as_mut_ptr(), buf.len() as u32, VOLUME_NAME_GUID) as usize;
    if n == 0 || n >= buf.len() {
        return false;
    }
    let path = String::from_utf16_lossy(&buf[..n]);
    let Some(volume) = path.strip_prefix(r"\\?\").and_then(|r| r.split('\\').next()) else { return false };
    let name: Vec<u16> = format!(r"\\.\{volume}").encode_utf16().chain([0]).collect();
    let v = CreateFileW(name.as_ptr(), 0, FILE_SHARE_READ | FILE_SHARE_WRITE, std::ptr::null(), OPEN_EXISTING, 0, std::ptr::null_mut());
    if v == INVALID_HANDLE_VALUE || v.is_null() {
        return false;
    }
    // STORAGE_PROPERTY_QUERY { StorageDeviceSeekPenaltyProperty, PropertyStandardQuery } ->
    // DEVICE_SEEK_PENALTY_DESCRIPTOR { Version, Size, IncursSeekPenalty: BOOLEAN }
    const IOCTL_STORAGE_QUERY_PROPERTY: u32 = 0x002D_1400;
    let query: [u32; 3] = [7, 0, 0];
    let mut out = [0u8; 12];
    let mut got = 0u32;
    let ok = DeviceIoControl(
        v,
        IOCTL_STORAGE_QUERY_PROPERTY,
        query.as_ptr() as *const c_void,
        std::mem::size_of_val(&query) as u32,
        out.as_mut_ptr() as *mut c_void,
        out.len() as u32,
        &mut got,
        std::ptr::null_mut(),
    ) != 0;
    CloseHandle(v);
    ok && got >= 9 && out[8] == 0
}

struct Job {
    /// the title's handle when the read was started
    file: Arc<std::fs::File>,
    owner: Arc<FileInner>,
    offset: u64,
    len: usize,
    /// the bytes or the failure, and how long the read took (µs)
    done: mpsc::SyncSender<(ReadResult, u32)>,
}

pub(super) struct ReadAhead {
    readers: Vec<(mpsc::Sender<Job>, std::thread::JoinHandle<()>)>,
    max_readers: usize,
    stop: Arc<AtomicBool>,
    /// reads started, in queue order: (seq, bytes, reader, result)
    started: VecDeque<(u64, usize, usize, mpsc::Receiver<(ReadResult, u32)>)>,
    bytes: usize,
    /// requests up to this seq were looked at
    considered: u64,
    /// recent read time (µs), None before the first
    read_us: Option<u32>,
    probes: u32,
}

fn reader(k: usize, jobs: mpsc::Receiver<Job>, stop: Arc<AtomicBool>) {
    for j in jobs {
        if stop.load(Ordering::Acquire) {
            continue;
        }
        let f = FileReaders::handle(&j.owner, k, &j.file);
        let t = Instant::now();
        let mut b = vec![0u8; j.len];
        let r = read_exact_at(&f, j.offset, &mut b).map(|()| b);
        let _ = j.done.send((r, t.elapsed().as_micros().min(u32::MAX as u128) as u32));
    }
}

/// Whether `CancelRequestsWithTag` already covers the request, without pruning the rules
/// (`QueueInner::is_cancelled` prunes them in queue order, so only the worker may call it).
fn would_cancel(q: &QueueInner, seq: u64, tag: u64) -> bool {
    q.cancels.lock().unwrap().iter().any(|r| r.up_to_seq >= seq && tag & r.mask == r.value)
}

impl ReadAhead {
    pub(super) fn new() -> Self {
        Self {
            readers: Vec::new(),
            max_readers: max_readers(),
            stop: Arc::new(AtomicBool::new(false)),
            started: VecDeque::new(),
            bytes: 0,
            considered: 0,
            read_us: None,
            probes: 0,
        }
    }

    /// Reads come from the device (or none was timed yet), or it is time for a probe.
    fn device_reads(&mut self) -> bool {
        self.probes = self.probes.wrapping_add(1);
        self.read_us.is_none_or(|us| us >= SLOW_READ_US) || self.probes % PROBE_EVERY == 0
    }

    /// Start reads for the file requests in `ahead` (the worker runs `ahead[0]` next): those
    /// after an entry with CPU work to overlap and, on a file whose reads run in parallel,
    /// the others while reads come from the device; within the limits.
    pub(super) fn top_up(&mut self, q: &QueueInner, ahead: &VecDeque<Entry>) {
        // reads of requests the worker skipped (cancelled) are dropped, so they hold no place
        let next = ahead.iter().find_map(|e| if let Entry::Request { seq, .. } = e { Some(*seq) } else { None });
        while self.started.front().is_some_and(|s| next.is_some_and(|n| s.0 < n)) {
            let (_, len, _, _) = self.started.pop_front().unwrap();
            self.bytes -= len;
        }
        let mut work_before = false;
        for e in ahead {
            let Entry::Request { seq, req, file, .. } = e else { continue };
            if *seq > self.considered && (work_before || self.max_readers > 1) {
                let main = file.as_ref().filter(|_| !req.memory_source()).and_then(|f| f.handle().map(|h| (f, h)));
                let parallel = self.max_readers > 1 && main.as_ref().is_some_and(|(f, h)| f.readers.parallel(h));
                // a request no entry before it gives work to overlap stays the worker's: the
                // entries before it only leave, so that does not change
                if !(work_before || parallel && self.device_reads()) {
                    self.considered = *seq;
                } else {
                    if self.started.len() >= if parallel { PARALLEL_REQUESTS } else { MAX_REQUESTS } {
                        return;
                    }
                    let len = req.source_size as usize;
                    if len <= MAX_BYTES && !self.started.is_empty() && self.bytes + len > MAX_BYTES {
                        return;
                    }
                    self.considered = *seq;
                    if let Some((f, h)) = main.filter(|_| len <= MAX_BYTES && !would_cancel(q, *seq, req.tag)) {
                        if let Some((k, rx)) = self.start(h, f, parallel, req.offset, len) {
                            self.started.push_back((*seq, len, k, rx));
                            self.bytes += len;
                        }
                    }
                }
            }
            work_before |= req.zlib() || req.destination_size >= HEAVY_COPY;
        }
    }

    /// Hand a read to a reader: the least busy one for a file whose reads run in parallel
    /// (started when first needed), else reader 0, which reads in queue order.
    fn start(&mut self, file: Arc<std::fs::File>, owner: &Arc<FileInner>, parallel: bool, offset: u64, len: usize) -> Option<(usize, mpsc::Receiver<(ReadResult, u32)>)> {
        let mut k = 0;
        if parallel {
            let busy = |k: usize| self.started.iter().filter(|s| s.2 == k).count();
            k = (0..self.max_readers).min_by_key(|&k| (busy(k), k)).unwrap_or(0);
        }
        while self.readers.len() <= k {
            let (tx, rx) = mpsc::channel::<Job>();
            let (stop, n) = (Arc::clone(&self.stop), self.readers.len());
            match std::thread::Builder::new().name("xcl-dstorage-read".into()).spawn(move || reader(n, rx, stop)) {
                Ok(h) => self.readers.push((tx, h)),
                Err(_) => {
                    // no more readers: the ones there are (none: the worker reads every request)
                    self.max_readers = self.readers.len().max(1);
                    if self.readers.is_empty() {
                        self.max_readers = 0;
                        return None;
                    }
                    k = k.min(self.readers.len() - 1);
                    break;
                }
            }
        }
        let (done, rx) = mpsc::sync_channel(1);
        self.readers[k].0.send(Job { file, owner: Arc::clone(owner), offset, len, done }).ok()?;
        Some((k, rx))
    }

    /// The source bytes of request `seq` if they were read ahead (waiting for the read), else
    /// None and the worker reads them. Reads of requests before it that were skipped are dropped.
    pub(super) fn take(&mut self, seq: u64) -> Option<ReadResult> {
        while self.started.front().is_some_and(|s| s.0 <= seq) {
            let (s, len, _, rx) = self.started.pop_front().unwrap();
            self.bytes -= len;
            if s == seq {
                let (r, us) = rx.recv().ok()?;
                self.read_us = Some(self.read_us.map_or(us, |a| ((a as u64 * 7 + us as u64) / 8) as u32));
                return Some(r);
            }
        }
        None
    }

    /// The queue was closed: reads not yet begun are not made.
    pub(super) fn stop(&self) {
        self.stop.store(true, Ordering::Release);
    }
}

impl Drop for ReadAhead {
    /// Wait for the readers, so no read of the queue is still running once the worker has
    /// exited (`Close` waits for the worker).
    fn drop(&mut self) {
        self.stop();
        self.started.clear();
        let threads: Vec<_> = self.readers.drain(..).map(|(tx, h)| {
            drop(tx);
            h
        }).collect();
        for h in threads {
            let _ = h.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_file(name: &str, bytes: &[u8]) -> (std::path::PathBuf, Arc<FileInner>) {
        let path = std::env::temp_dir().join(format!("xcl-read-ahead-{}-{name}", std::process::id()));
        std::fs::write(&path, bytes).unwrap();
        let f = std::fs::File::open(&path).unwrap();
        (path, Arc::new(FileInner { id: 0, path: Vec::new(), file: Mutex::new(Some(Arc::new(f))), readers: Default::default() }))
    }

    fn file_request(seq: u64, file: &Arc<FileInner>, zlib: bool, offset: u64, src: u32, dst: u32, tag: u64) -> Entry {
        let req = Request { options: if zlib { ZLIB_BIT } else { 0 }, destination: 0x1000, destination_size: dst, offset, source_size: src, tag, ..Default::default() };
        Entry::Request { seq, req, file: Some(Arc::clone(file)), dest_pages: None, src_pages: None }
    }

    unsafe fn new_queue(source_memory: bool) -> *mut c_void {
        let mut f = std::ptr::null_mut();
        assert_eq!(DStorageGetFactory(&IID_FACTORY, &mut f), S_OK);
        let mut desc = [0u8; 0x18];
        desc[0..4].copy_from_slice(&(source_memory as u32).to_le_bytes());
        desc[8..10].copy_from_slice(&64u16.to_le_bytes());
        desc[10] = if source_memory { 2 } else { 0 };
        let mut q = std::ptr::null_mut();
        assert_eq!(factory_create_queue(f, desc.as_ptr(), &IID_QUEUE, &mut q), S_OK);
        q
    }

    /// Only reads behind CPU work start, in order and within the limits; take hands them over.
    #[test]
    fn reads_start_behind_cpu_work_within_limits() {
        unsafe {
            let q = new_queue(false);
            let data: Vec<u8> = (0..4096u32).map(|i| i as u8).collect();
            let (path, f) = temp_file("limits", &data);
            let _ = f.readers.no_seek_penalty.set(false); // one reader, as on a hard disk
            let mut ra = ReadAhead::new();

            // small plain copies: nothing to overlap, nothing read ahead
            let plain: VecDeque<Entry> = (1..=3).map(|s| file_request(s, &f, false, 0, 64, 64, 0)).collect();
            ra.top_up(queue(q), &plain);
            assert!(ra.started.is_empty());

            // behind a zlib request: the next file requests, up to MAX_REQUESTS, but not one
            // CancelRequestsWithTag already covers
            queue(q).cancels.lock().unwrap().push(CancelRule { up_to_seq: 100, mask: 0xFF, value: 0x77 });
            let mut ahead: VecDeque<Entry> = VecDeque::new();
            ahead.push_back(file_request(10, &f, true, 0, 16, 64, 0));
            ahead.push_back(file_request(11, &f, false, 100, 32, 32, 0x77)); // cancelled
            for s in 12..=17 {
                ahead.push_back(file_request(s, &f, false, s * 10, 8, 8, 0));
            }
            ra.top_up(queue(q), &ahead);
            assert_eq!(ra.started.iter().map(|s| s.0).collect::<Vec<_>>(), vec![12, 13, 14, 15]);
            assert_eq!(ra.take(10), None, "the front is read by the worker");
            assert_eq!(ra.take(13), Some(Ok(data[130..138].to_vec())), "12 is dropped on the way");
            assert_eq!(ra.started.iter().map(|s| s.0).collect::<Vec<_>>(), vec![14, 15]);
            // looked-at requests are not read twice; the window moves on
            ra.top_up(queue(q), &ahead);
            assert_eq!(ra.started.iter().map(|s| s.0).collect::<Vec<_>>(), vec![14, 15, 16, 17]);
            // 14 and 15 skipped by the worker (cancelled after their reads began): dropped when
            // the window moves past them, and 16 and 17 stay
            ahead.retain(|e| !matches!(e, Entry::Request { seq, .. } if *seq < 16));
            ra.top_up(queue(q), &ahead);
            assert_eq!((ra.started.iter().map(|s| s.0).collect::<Vec<_>>(), ra.bytes), (vec![16, 17], 16));

            // a request above MAX_BYTES is left to the worker
            let mut big: VecDeque<Entry> = VecDeque::new();
            big.push_back(file_request(20, &f, true, 0, 16, 64, 0));
            big.push_back(file_request(21, &f, false, 0, (MAX_BYTES + 1) as u32, (MAX_BYTES + 1) as u32, 0));
            let mut ra2 = ReadAhead::new();
            ra2.top_up(queue(q), &big);
            assert!(ra2.started.is_empty());
            assert_eq!(ra2.take(21), None);

            drop(ra);
            drop(ra2);
            queue_close(q);
            queue_release(q);
            let _ = std::fs::remove_file(path);
        }
    }

    /// A read made ahead, then the title closes the file: the request still fails with
    /// E_FILE_HANDLE; and a read past the end gives E_END_OF_FILE as an inline read does.
    #[test]
    fn a_read_ahead_keeps_the_requests_result() {
        unsafe {
            let q = new_queue(false);
            let data = vec![5u8; 256];
            let (path, f) = temp_file("result", &data);
            let _ = f.readers.no_seek_penalty.set(false);
            let mut ra = ReadAhead::new();
            let mut ahead: VecDeque<Entry> = VecDeque::new();
            ahead.push_back(file_request(1, &f, true, 0, 16, 64, 0));
            ahead.push_back(file_request(2, &f, false, 0, 64, 64, 0));
            ahead.push_back(file_request(3, &f, false, 250, 64, 64, 0)); // past the end
            ra.top_up(queue(q), &ahead);
            assert_eq!(ra.started.len(), 2);
            let mut dst = [0u8; 64];
            let dp = dst.as_mut_ptr() as u64;
            let req = |off: u64| Request { destination: dp, destination_size: 64, offset: off, source_size: 64, ..Default::default() };
            let r2 = ra.take(2);
            assert_eq!(r2, Some(Ok(vec![5u8; 64])));
            assert_eq!(ra.take(3), Some(Err(E_END_OF_FILE)));
            assert_eq!(execute(&req(250), Some(&*f), None, None, Some(Err(E_END_OF_FILE))), execute(&req(250), Some(&*f), None, None, None));
            file_close_inner(&f);
            assert_eq!(execute(&req(0), Some(&*f), None, None, r2), Err(E_FILE_HANDLE));
            assert_eq!(dst, [0u8; 64], "the destination is untouched");
            drop(ra);
            queue_close(q);
            queue_release(q);
            let _ = std::fs::remove_file(path);
        }
    }

    fn file_close_inner(f: &FileInner) {
        f.file.lock().unwrap().take();
        f.readers.close();
    }

    /// End to end on a file queue: zlib and plain requests read ahead complete in queue order,
    /// each status gets the first failure since the previous one, and the error record names
    /// the first failed request.
    #[test]
    fn a_file_queue_with_read_ahead_completes_in_order() {
        file_queue_completes_in_order(false);
        file_queue_completes_in_order(true);
    }

    fn file_queue_completes_in_order(parallel: bool) {
        unsafe {
            let q = new_queue(false);
            let mut fac = std::ptr::null_mut();
            assert_eq!(DStorageGetFactory(&IID_FACTORY, &mut fac), S_OK);
            let mut s = std::ptr::null_mut();
            assert_eq!(factory_create_status_array(fac, 8, std::ptr::null(), &IID_STATUS_ARRAY, &mut s), S_OK);
            let plains: Vec<Vec<u8>> = (0..6u8).map(|k| (0..20_000u32).map(|i| (i / 7) as u8 ^ k).collect()).collect();
            let mut file = Vec::new();
            let mut at = Vec::new();
            for p in &plains {
                let z = miniz_oxide::deflate::compress_to_vec_zlib(p, 6);
                at.push((file.len() as u64, z.len() as u32));
                file.extend_from_slice(&z);
            }
            let raw_at = file.len() as u64;
            file.extend_from_slice(&plains[0]);
            let (path, f) = temp_file(&format!("queue-{parallel}"), &file);
            let _ = f.readers.no_seek_penalty.set(parallel);
            let fo = new_object(file_vtbl(), Arc::clone(&f));
            let mut dst: Vec<Vec<u8>> = (0..8).map(|_| vec![0u8; 20_000]).collect();
            let req = |opt: u64, d: &mut Vec<u8>, off: u64, src: u32| {
                let mut b = [0u8; REQUEST_BYTES];
                b[0..8].copy_from_slice(&opt.to_le_bytes());
                b[8..16].copy_from_slice(&(d.as_mut_ptr() as u64).to_le_bytes());
                b[0x18..0x1C].copy_from_slice(&(d.len() as u32).to_le_bytes());
                b[0x20..0x28].copy_from_slice(&(fo as u64).to_le_bytes());
                b[0x28..0x30].copy_from_slice(&off.to_le_bytes());
                b[0x30..0x34].copy_from_slice(&src.to_le_bytes());
                queue_enqueue_request(q, b.as_ptr());
            };
            let (head, tail) = dst.split_at_mut(6);
            for (k, d) in head.iter_mut().enumerate().take(3) {
                req(ZLIB_BIT, d, at[k].0, at[k].1);
            }
            queue_enqueue_status(q, s, 0);
            // a plain read past the end, then a good zlib, a plain and two zlib requests
            req(0, &mut tail[0], file.len() as u64 - 100, 20_000);
            for (k, d) in head.iter_mut().enumerate().skip(3).take(1) {
                req(ZLIB_BIT, d, at[k].0, at[k].1);
            }
            req(0, &mut tail[1], raw_at, 20_000);
            queue_enqueue_status(q, s, 1);
            for (k, d) in head.iter_mut().enumerate().skip(4) {
                req(ZLIB_BIT, d, at[k].0, at[k].1);
            }
            queue_enqueue_status(q, s, 2);
            queue_submit(q);
            let t = std::time::Instant::now();
            while !status_is_complete(s, 2) && t.elapsed().as_secs() < 10 {
                std::thread::yield_now();
            }
            assert_eq!((status_get_hresult(s, 0), status_get_hresult(s, 1), status_get_hresult(s, 2)), (S_OK, E_END_OF_FILE, S_OK));
            for (k, p) in plains.iter().enumerate() {
                assert_eq!(&dst[k], p, "zlib request {k}");
            }
            assert_eq!(dst[6], vec![0u8; 20_000], "the failed read left its destination");
            assert_eq!(&dst[7], &plains[0]);
            let mut rec = [0u8; ERROR_RECORD_BYTES];
            queue_retrieve_error_record(q, rec.as_mut_ptr());
            assert_eq!((u32::from_le_bytes(rec[0..4].try_into().unwrap()), i32::from_le_bytes(rec[8..12].try_into().unwrap())), (1, E_END_OF_FILE));
            queue_close(q);
            queue_release(q);
            file_release(fo);
            status_release(s);
            let _ = std::fs::remove_file(path);
        }
    }

    /// A drive without seek penalty: reads spread over the readers, each with a handle of its
    /// own made by ReOpenFile, come back in queue order with the file's bytes; the extra handles
    /// are closed with the file and not made again after it.
    #[test]
    fn parallel_reads_use_reopened_handles_and_close_with_the_file() {
        unsafe {
            let q = new_queue(false);
            let data: Vec<u8> = (0..65536u32).map(|i| (i * 7 / 3) as u8).collect();
            let (path, f) = temp_file("parallel", &data);
            let _ = f.readers.no_seek_penalty.set(true);
            let mut ra = ReadAhead::new();
            // small plain copies: nothing to overlap but other reads, read ahead in parallel
            let mut ahead: VecDeque<Entry> = (1..=10).map(|s| file_request(s, &f, false, s * 1000, 900, 900, 0)).collect();
            ra.top_up(queue(q), &ahead);
            let n = ra.started.len();
            assert_eq!(n, if max_readers() > 1 { PARALLEL_REQUESTS } else { 0 });
            if max_readers() > 1 {
                let readers: std::collections::BTreeSet<usize> = ra.started.iter().map(|s| s.2).collect();
                assert_eq!(readers.len(), max_readers(), "every reader gets reads");
            }
            for s in 1..=10u64 {
                let want = data[s as usize * 1000..s as usize * 1000 + 900].to_vec();
                match ra.take(s) {
                    Some(got) => assert_eq!(got, Ok(want), "request {s}"),
                    None => assert!(s as usize > n, "request {s} was read ahead"),
                }
                ahead.pop_front();
                ra.top_up(queue(q), &ahead);
            }
            if max_readers() == MAX_READERS {
                assert!(f.readers.more.lock().unwrap().iter().all(|h| h.is_some()), "readers 1 to 3 have their handles");
            }
            let main = f.handle().unwrap();
            file_close_inner(&f);
            assert!(f.readers.more.lock().unwrap().iter().all(|h| h.is_none()), "closed with the file");
            assert!(Arc::ptr_eq(&FileReaders::handle(&f, 1, &main), &main), "none made after Close");
            assert!(f.readers.more.lock().unwrap().iter().all(|h| h.is_none()));
            drop(ra);
            queue_close(q);
            queue_release(q);
            let _ = std::fs::remove_file(path);
        }
    }

    /// The title closes the file while its requests are read ahead in parallel: each still
    /// fails with E_FILE_HANDLE, in order, and no destination is written.
    #[test]
    fn closing_the_file_during_parallel_reads_keeps_e_file_handle() {
        unsafe {
            let q = new_queue(false);
            let mut fac = std::ptr::null_mut();
            assert_eq!(DStorageGetFactory(&IID_FACTORY, &mut fac), S_OK);
            let mut s = std::ptr::null_mut();
            assert_eq!(factory_create_status_array(fac, 1, std::ptr::null(), &IID_STATUS_ARRAY, &mut s), S_OK);
            let data = vec![9u8; 64 * 1024];
            let (path, f) = temp_file("close", &data);
            let _ = f.readers.no_seek_penalty.set(true);
            let fo = new_object(file_vtbl(), Arc::clone(&f));
            // the worker waits at the gate after starting the reads of the requests behind it
            let gate = Arc::new(std::sync::Barrier::new(2));
            queue(q).enqueue(Entry::Gate(Arc::clone(&gate)));
            let mut dst: Vec<Vec<u8>> = (0..12).map(|_| vec![0u8; 4096]).collect();
            for (k, d) in dst.iter_mut().enumerate() {
                let mut b = [0u8; REQUEST_BYTES];
                b[8..16].copy_from_slice(&(d.as_mut_ptr() as u64).to_le_bytes());
                b[0x18..0x1C].copy_from_slice(&4096u32.to_le_bytes());
                b[0x20..0x28].copy_from_slice(&(fo as u64).to_le_bytes());
                b[0x28..0x30].copy_from_slice(&(k as u64 * 4096).to_le_bytes());
                b[0x30..0x34].copy_from_slice(&4096u32.to_le_bytes());
                queue_enqueue_request(q, b.as_ptr());
            }
            queue_enqueue_status(q, s, 0);
            queue_submit(q);
            file_close(fo);
            gate.wait();
            let t = std::time::Instant::now();
            while !status_is_complete(s, 0) && t.elapsed().as_secs() < 10 {
                std::thread::yield_now();
            }
            assert_eq!(status_get_hresult(s, 0), E_FILE_HANDLE);
            assert!(dst.iter().all(|d| d.iter().all(|&b| b == 0)), "no destination written");
            let mut rec = [0u8; ERROR_RECORD_BYTES];
            queue_retrieve_error_record(q, rec.as_mut_ptr());
            assert_eq!((u32::from_le_bytes(rec[0..4].try_into().unwrap()), i32::from_le_bytes(rec[8..12].try_into().unwrap())), (12, E_FILE_HANDLE));
            assert!(f.readers.more.lock().unwrap().iter().all(|h| h.is_none()), "the extra handles went with the file");
            queue_close(q);
            queue_release(q);
            file_release(fo);
            status_release(s);
            let _ = std::fs::remove_file(path);
        }
    }
}
