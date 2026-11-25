//! The console device is also an IDXGIDevice: titles reach the display output through it
//! (device → IDXGIDevice1 → GetAdapter → EnumOutputs, the output SetFrameIntervalX takes –
//! as Microsoft's public GDK samples do). A PC D3D12 device is not, so a small object answers
//! IDXGIDevice / 1 / 2 with the device's real adapter; every other interface is the device's.
//! Console code only (`real::dev_qi_x`): DXGI asks the device the same and needs the PC answer.

use std::ffi::c_void;
use std::sync::atomic::{AtomicI32, AtomicU32, AtomicU64, Ordering};
use std::sync::Mutex;

use windows::core::{Interface, GUID};
use windows::Win32::Foundation::HANDLE;
use windows::Win32::Graphics::Direct3D12::{ID3D12CommandQueue, ID3D12Device, ID3D12Device1, ID3D12Fence, D3D12_FENCE_FLAG_NONE, D3D12_MULTIPLE_FENCE_WAIT_FLAG_ALL};
use windows::Win32::Graphics::Dxgi::*;

use crate::util::{HRESULT, S_OK};

const E_NOTIMPL: HRESULT = 0x8000_4001u32 as i32;
const E_POINTER: HRESULT = 0x8000_4003u32 as i32;

/// Whether `iid` is one of the DXGI interfaces this object answers.
pub fn answers(iid: &GUID) -> bool {
    [IDXGIObject::IID, IDXGIDevice::IID, IDXGIDevice1::IID, IDXGIDevice2::IID].contains(iid)
}

#[repr(C)]
struct DxgiDevice {
    vtbl: *const [usize; 17],
    refs: AtomicU32,
    /// the D3D12 device (one reference held)
    device: *mut c_void,
    priority: AtomicI32,
    latency: AtomicU32,
}

fn vtbl() -> &'static [usize; 17] {
    static V: std::sync::OnceLock<[usize; 17]> = std::sync::OnceLock::new();
    V.get_or_init(|| [
        qi as usize,
        add_ref as usize,
        release as usize,
        set_private_data as usize,
        set_private_data_interface as usize,
        get_private_data as usize,
        get_parent as usize,
        get_adapter as usize,
        create_surface as usize,
        query_resource_residency as usize,
        set_gpu_thread_priority as usize,
        get_gpu_thread_priority as usize,
        set_maximum_frame_latency as usize,
        get_maximum_frame_latency as usize,
        offer_resources as usize,
        reclaim_resources as usize,
        enqueue_set_event as usize,
    ])
}

/// A new IDXGIDevice2 for `device` (one reference, the caller's).
pub(super) unsafe fn create(device: *mut c_void) -> *mut c_void {
    if let Some(d) = ID3D12Device::from_raw_borrowed(&device) {
        std::mem::forget(d.clone()); // the reference the object holds
    }
    let o = Box::new(DxgiDevice { vtbl: vtbl(), refs: AtomicU32::new(1), device, priority: AtomicI32::new(0), latency: AtomicU32::new(3) });
    Box::into_raw(o) as *mut c_void
}

unsafe fn this<'a>(p: *mut c_void) -> &'a DxgiDevice {
    &*(p as *const DxgiDevice)
}

unsafe fn device<'a>(p: *mut c_void) -> &'a ID3D12Device {
    ID3D12Device::from_raw_borrowed(&this(p).device).unwrap()
}

unsafe extern "system" fn qi(p: *mut c_void, riid: *const GUID, out: *mut *mut c_void) -> HRESULT {
    if riid.is_null() || out.is_null() {
        return E_POINTER;
    }
    if answers(&*riid) {
        add_ref(p);
        *out = p;
        return S_OK;
    }
    // IUnknown and the D3D12 interfaces: the device's (one identity)
    device(p).query(&*riid, out).0
}

unsafe extern "system" fn add_ref(p: *mut c_void) -> u32 {
    this(p).refs.fetch_add(1, Ordering::Relaxed) + 1
}

unsafe extern "system" fn release(p: *mut c_void) -> u32 {
    let n = this(p).refs.fetch_sub(1, Ordering::AcqRel) - 1;
    if n == 0 {
        let o = Box::from_raw(p as *mut DxgiDevice);
        drop(ID3D12Device::from_raw(o.device));
    }
    n
}

unsafe extern "system" fn set_private_data(p: *mut c_void, name: *const GUID, size: u32, data: *const c_void) -> HRESULT {
    device(p).SetPrivateData(name, size, Some(data)).map_or_else(|e| e.code().0, |_| S_OK)
}

unsafe extern "system" fn set_private_data_interface(p: *mut c_void, name: *const GUID, unknown: *mut c_void) -> HRESULT {
    let u = windows::core::IUnknown::from_raw_borrowed(&unknown);
    device(p).SetPrivateDataInterface(name, u).map_or_else(|e| e.code().0, |_| S_OK)
}

unsafe extern "system" fn get_private_data(p: *mut c_void, name: *const GUID, size: *mut u32, data: *mut c_void) -> HRESULT {
    device(p).GetPrivateData(name, size, Some(data)).map_or_else(|e| e.code().0, |_| S_OK)
}

/// the device's adapter (as on PC: the DXGI adapter with its LUID)
unsafe fn adapter(p: *mut c_void) -> Result<IDXGIAdapter, HRESULT> {
    let luid = device(p).GetAdapterLuid();
    let factory: IDXGIFactory4 = CreateDXGIFactory1().map_err(|e| e.code().0)?;
    factory.EnumAdapterByLuid(luid).map_err(|e| e.code().0)
}

unsafe extern "system" fn get_parent(p: *mut c_void, riid: *const GUID, out: *mut *mut c_void) -> HRESULT {
    if riid.is_null() || out.is_null() {
        return E_POINTER;
    }
    match adapter(p) {
        Ok(a) => a.query(&*riid, out).0,
        Err(hr) => hr,
    }
}

unsafe extern "system" fn get_adapter(p: *mut c_void, out: *mut *mut c_void) -> HRESULT {
    if out.is_null() {
        return E_POINTER;
    }
    match adapter(p) {
        Ok(a) => {
            *out = a.into_raw();
            S_OK
        }
        Err(hr) => {
            *out = std::ptr::null_mut();
            hr
        }
    }
}

unsafe extern "system" fn create_surface(_p: *mut c_void, _desc: *const c_void, _n: u32, _usage: u32, _shared: *const c_void, out: *mut *mut c_void) -> HRESULT {
    if !out.is_null() {
        *out = std::ptr::null_mut();
    }
    E_NOTIMPL
}

/// everything is resident (DXGI_RESIDENCY_FULLY_RESIDENT)
unsafe extern "system" fn query_resource_residency(_p: *mut c_void, _resources: *const *mut c_void, status: *mut u32, n: u32) -> HRESULT {
    if status.is_null() {
        return E_POINTER;
    }
    std::slice::from_raw_parts_mut(status, n as usize).fill(1);
    S_OK
}

unsafe extern "system" fn set_gpu_thread_priority(p: *mut c_void, priority: i32) -> HRESULT {
    if !(-7..=7).contains(&priority) {
        return windows::Win32::Foundation::E_INVALIDARG.0;
    }
    this(p).priority.store(priority, Ordering::Relaxed);
    S_OK
}

unsafe extern "system" fn get_gpu_thread_priority(p: *mut c_void, out: *mut i32) -> HRESULT {
    if out.is_null() {
        return E_POINTER;
    }
    *out = this(p).priority.load(Ordering::Relaxed);
    S_OK
}

unsafe extern "system" fn set_maximum_frame_latency(p: *mut c_void, n: u32) -> HRESULT {
    this(p).latency.store(if n == 0 { 3 } else { n.min(16) }, Ordering::Relaxed);
    S_OK
}

unsafe extern "system" fn get_maximum_frame_latency(p: *mut c_void, out: *mut u32) -> HRESULT {
    if out.is_null() {
        return E_POINTER;
    }
    *out = this(p).latency.load(Ordering::Relaxed);
    S_OK
}

unsafe extern "system" fn offer_resources(_p: *mut c_void, _n: u32, _resources: *const *mut c_void, _priority: i32) -> HRESULT {
    S_OK
}

unsafe extern "system" fn reclaim_resources(_p: *mut c_void, n: u32, _resources: *const *mut c_void, discarded: *mut i32) -> HRESULT {
    if !discarded.is_null() {
        std::slice::from_raw_parts_mut(discarded, n as usize).fill(0);
    }
    S_OK
}

/// Sets `event` once the GPU has done the work submitted so far, on every queue the title has
/// given work to on this device. A title waits on it to keep the CPU a frame or so ahead of the
/// GPU; refused, it waited on an event never set, or skipped the wait and ran ahead without
/// bound.
unsafe extern "system" fn enqueue_set_event(p: *mut c_void, event: *mut c_void) -> HRESULT {
    if event.is_null() {
        return windows::Win32::Foundation::E_INVALIDARG.0;
    }
    match enqueue(device(p), HANDLE(event)) {
        Ok(()) => S_OK,
        Err(e) => {
            static N: AtomicU64 = AtomicU64::new(0);
            if N.fetch_add(1, Ordering::Relaxed) < 4 {
                crate::xlog_warn!("D3D12.X: IDXGIDevice2::EnqueueSetEvent: {e}");
            }
            e.code().0
        }
    }
}

/// queue → a fence of ours on it and the last value it was told to signal
static QUEUE_FENCES: Mutex<Vec<(usize, ID3D12Fence, u64)>> = Mutex::new(Vec::new());

/// Each of `dev`'s queues signals its fence of ours after what it has been given so far; `event`
/// is set when all of them have.
unsafe fn enqueue(dev: &ID3D12Device, event: HANDLE) -> windows::core::Result<()> {
    let mut fences = Vec::new();
    let mut values = Vec::new();
    {
        let mut g = QUEUE_FENCES.lock().unwrap_or_else(|e| e.into_inner());
        for q in super::present::known_queues() {
            let raw = q as *mut c_void;
            let Some(queue) = ID3D12CommandQueue::from_raw_borrowed(&raw) else { continue };
            let mut owner: Option<ID3D12Device> = None;
            if queue.GetDevice(&mut owner).is_err() || owner.as_ref().map(|d| d.as_raw()) != Some(dev.as_raw()) {
                continue;
            }
            let i = match g.iter().position(|e| e.0 == q) {
                Some(i) => i,
                None => {
                    g.push((q, dev.CreateFence(0, D3D12_FENCE_FLAG_NONE)?, 0));
                    g.len() - 1
                }
            };
            let (_, fence, value) = &mut g[i];
            queue.Signal(&*fence, *value + 1)?;
            *value += 1;
            fences.push(Some(fence.clone()));
            values.push(*value);
        }
    }
    if fences.is_empty() {
        // nothing submitted: nothing to wait for
        windows::Win32::System::Threading::SetEvent(event)?;
        return Ok(());
    }
    dev.cast::<ID3D12Device1>()?.SetEventOnMultipleFenceCompletion(fences.as_ptr(), values.as_ptr(), fences.len() as u32, D3D12_MULTIPLE_FENCE_WAIT_FLAG_ALL, event)
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
    use windows::Win32::Graphics::Direct3D12::*;
    use windows::Win32::System::Threading::{CreateEventW, WaitForSingleObject};
    use windows::Win32::Foundation::{CloseHandle, WAIT_OBJECT_0, WAIT_TIMEOUT};

    /// The event is set when the queue has done what it was given before the call, not sooner.
    #[test]
    fn enqueue_set_event_waits_for_the_queues() {
        unsafe {
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                let Ok(factory) = CreateDXGIFactory1::<IDXGIFactory4>() else { return crate::gpu::recomp::runner::skip_test("no D3D12 device") };
                let Ok(warp) = factory.EnumWarpAdapter::<IDXGIAdapter>() else { return crate::gpu::recomp::runner::skip_test("no D3D12 device") };
                if D3D12CreateDevice(&warp, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                    return crate::gpu::recomp::runner::skip_test("no D3D12 device");
                }
            }
            let dev = dev.unwrap();
            let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() }).unwrap();
            super::super::present::note_execute(queue.as_raw(), &[]);
            let dxgi = create(dev.as_raw());
            let enqueue_fn = vtbl()[16];
            let enqueue_set_event: unsafe extern "system" fn(*mut c_void, *mut c_void) -> HRESULT = std::mem::transmute(enqueue_fn);
            assert_eq!(enqueue_set_event(dxgi, std::ptr::null_mut()), windows::Win32::Foundation::E_INVALIDARG.0);

            // the queue is held at a wait until the CPU signals `gate`
            let gate: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            queue.Wait(&gate, 1).unwrap();
            let event = CreateEventW(None, false, false, None).unwrap();
            assert_eq!(enqueue_set_event(dxgi, event.0), S_OK);
            assert_eq!(WaitForSingleObject(event, 200), WAIT_TIMEOUT, "set before the queue got there");
            gate.Signal(1).unwrap();
            assert_eq!(WaitForSingleObject(event, 10_000), WAIT_OBJECT_0, "never set");

            // again, with the queue idle: set at once
            assert_eq!(enqueue_set_event(dxgi, event.0), S_OK);
            assert_eq!(WaitForSingleObject(event, 10_000), WAIT_OBJECT_0);
            CloseHandle(event).unwrap();
            release(dxgi);
        }
    }
}
