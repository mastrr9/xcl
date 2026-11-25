//! Console mesh pipeline streams that `translated_mesh_stream_once` refuses before it needs a
//! device: one not read to its end, and a console AS without an MS.

use super::*;
use crate::gpu::xbox::tests::{fssm, sample, CODE};

/// What `translated_mesh_stream_once` makes of `stream` before it would need a device.
unsafe fn translated(stream: &[u8]) -> Option<HRESULT> {
    let desc = [stream.len(), stream.as_ptr() as usize];
    let mut out = std::ptr::null_mut();
    translated_mesh_stream_once(std::ptr::null_mut(), desc.as_ptr().cast(), &pc_iid(Kind::PipelineState), &mut out)
}

/// A console MS, then a subobject XCL cannot read (a type the PC's headers do not have, or
/// one SizeInBytes cuts): refused, not left for the caller to hand to the driver as it is.
#[test]
fn a_console_mesh_stream_not_read_to_its_end_is_refused() {
    let ms = sample(13, &[fssm(&CODE, &[(7, 0)])]);
    assert!(crate::gpu::recomp::pso::is_console(&ms));
    let mut unknown = StreamOut::default();
    unknown.shader(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS, &ms);
    unknown.raw(23, &[0; 16], true);
    assert!(stream_subobjects(&unknown.0).is_none());
    assert_eq!(unsafe { translated(&unknown.0) }, Some(E_INVALIDARG));
    let e = unsafe { unread_console_mesh(&unknown.0) }.expect("refused");
    assert!(e.contains("not read past 0x18 of 0x30 (type 23)") && e.contains("console MS"), "{e}");

    // RENDER_TARGET_FORMATS (36 B) cut after its type
    let mut cut = StreamOut::default();
    cut.shader(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS, &ms);
    let mut cut = cut.0;
    cut.extend_from_slice(&15u32.to_le_bytes());
    cut.extend_from_slice(&[0; 4]);
    assert_eq!(unsafe { translated(&cut) }, Some(E_INVALIDARG));
    assert!(unsafe { unread_console_mesh(&cut) }.is_some_and(|e| e.contains("(type 15)")));

    // a PC MS there: not XCL's to refuse
    let pc = [0x44u8, 0x58, 0x42, 0x43, 1, 2, 3, 4];
    let mut s = StreamOut::default();
    s.shader(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS, &pc);
    s.raw(23, &[0; 16], true);
    assert_eq!(unsafe { translated(&s.0) }, None);
    assert_eq!(unsafe { unread_console_mesh(&s.0) }, None);
}

/// A console AS and no MS: refused (a console MS in that slot already was, on its own path).
#[test]
fn a_console_as_without_an_ms_is_refused() {
    let a = sample(14, &[fssm(&CODE, &[(7, 0)])]);
    assert!(crate::gpu::recomp::pso::is_console(&a));
    let mut s = StreamOut::default();
    s.shader(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_AS, &a);
    s.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_MASK, &u32::MAX);
    assert_eq!(unsafe { translated(&s.0) }, Some(E_INVALIDARG));
    // no AS / MS at all: not a mesh pipeline
    let mut none = StreamOut::default();
    none.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_MASK, &u32::MAX);
    assert_eq!(unsafe { translated(&none.0) }, None);
}
