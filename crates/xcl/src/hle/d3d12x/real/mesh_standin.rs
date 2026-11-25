//! The stand-in pipeline of a stream with a mesh shader (`mesh_standin_stream`, `fallback_stream`,
//! `standin_stream`): synthetic streams only.
use super::*;
use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT, DXGI_FORMAT_R32_UINT, DXGI_SAMPLE_DESC};

/// A console mesh stream as a title may write it: a null VS slot, line topology, a strip cut,
/// console render target and depth formats, blending on an integer target, depth bounds,
/// rasteriser, input layout and stream output slots. `rs`: the root signature's pointer; `ms`,
/// `ps`: the shaders' bytes (`ps` None: no PS subobject).
fn console_mesh_stream(rs: usize, ms: &[u8], ps: Option<&[u8]>) -> StreamOut {
    let mut s = StreamOut::default();
    s.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE, &rs);
    s.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS, &D3D12_SHADER_BYTECODE::default());
    s.shader(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS, ms);
    if let Some(ps) = ps {
        s.shader(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS, ps);
    }
    s.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_INPUT_LAYOUT, &D3D12_INPUT_LAYOUT_DESC::default());
    s.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_STREAM_OUTPUT, &D3D12_STREAM_OUTPUT_DESC::default());
    s.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_IB_STRIP_CUT_VALUE, &D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_0xFFFF);
    s.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PRIMITIVE_TOPOLOGY, &D3D12_PRIMITIVE_TOPOLOGY_TYPE_LINE);
    let mut blend = D3D12_BLEND_DESC { IndependentBlendEnable: true.into(), ..Default::default() };
    blend.RenderTarget[1].BlendEnable = true.into();
    blend.RenderTarget[1].RenderTargetWriteMask = 15;
    s.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_BLEND, &blend);
    s.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER, &D3D12_RASTERIZER_DESC { FillMode: D3D12_FILL_MODE_SOLID, CullMode: D3D12_CULL_MODE_NONE, ..Default::default() });
    s.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL1, &D3D12_DEPTH_STENCIL_DESC1 { DepthEnable: true.into(), DepthBoundsTestEnable: true.into(), ..Default::default() });
    let mut rt = D3D12_RT_FORMAT_ARRAY { NumRenderTargets: 2, ..Default::default() };
    rt.RTFormats[0] = DXGI_FORMAT(116); // a console-only render target format
    rt.RTFormats[1] = DXGI_FORMAT_R32_UINT;
    s.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS, &rt);
    s.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL_FORMAT, &DXGI_FORMAT(118)); // the console's D16S8
    s.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_DESC, &DXGI_SAMPLE_DESC { Count: 1, Quality: 0 });
    s.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_MASK, &u32::MAX);
    s.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_NODE_MASK, &0u32);
    s.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_FLAGS, &D3D12_PIPELINE_STATE_FLAG_NONE);
    s
}

fn types(subs: &[(u32, &[u8])]) -> Vec<u32> {
    let mut t: Vec<u32> = subs.iter().map(|s| s.0).collect();
    t.sort_unstable();
    t
}

fn find<'a>(subs: &[(u32, &'a [u8])], ty: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE) -> &'a [u8] {
    subs.iter().find(|s| s.0 == ty.0 as u32).unwrap_or_else(|| panic!("no subobject {}", ty.0)).1
}

fn word(p: &[u8], k: usize) -> u64 {
    u64::from_le_bytes(p[8 * k..8 * k + 8].try_into().unwrap())
}

fn dword(p: &[u8], k: usize) -> u32 {
    u32::from_le_bytes(p[4 * k..4 * k + 4].try_into().unwrap())
}

/// Only what D3D12 takes in a mesh pipeline: the title's root signature, the stand-in MS and PS,
/// triangles, the PC's formats, sample desc, mask and node mask. No VS, input layout, stream
/// output, strip cut, or state that can make the driver refuse it.
fn assert_a_mesh_standin(subs: &[(u32, &[u8])], rs: usize, ms: &[u8], ps: Option<&[u8]>) {
    let mut want = vec![0, 9, 14, 15, 16, 17, 18, 25];
    if ps.is_some() {
        want.push(2);
        want.sort_unstable();
    }
    assert_eq!(types(subs), want, "the stand-in's subobjects");
    assert_eq!(word(find(subs, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE), 0), rs as u64, "the title's root signature");
    let m = find(subs, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS);
    assert_eq!((word(m, 0), word(m, 1)), (ms.as_ptr() as u64, ms.len() as u64), "the stand-in MS");
    if let Some(ps) = ps {
        let p = find(subs, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS);
        assert_eq!((word(p, 0), word(p, 1)), (ps.as_ptr() as u64, ps.len() as u64), "the stand-in PS");
    }
    assert_eq!(dword(find(subs, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PRIMITIVE_TOPOLOGY), 0), D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE.0 as u32);
    let rt = find(subs, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS);
    assert_eq!(dword(rt, 0), super::super::device_x::pc_target_format(116));
    assert_ne!(dword(rt, 0), 116, "a PC format");
    assert_eq!((dword(rt, 1), dword(rt, 8)), (DXGI_FORMAT_R32_UINT.0 as u32, 2), "PC formats kept, the count too");
    let ds = dword(find(subs, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL_FORMAT), 0);
    assert_eq!(ds, super::super::device_x::pc_target_format(118));
    assert_ne!(ds, 118, "a PC format");
}

/// The stand-in for a console mesh stream is a mesh pipeline D3D12 accepts: on 568e7db
/// `fallback_stream` gave it a VS (the null VS slot filled with the stand-in VS), the line
/// topology, the strip cut and the console's formats, and the driver could refuse it.
#[test]
fn a_mesh_standin_keeps_only_what_a_mesh_pipeline_takes() {
    let (ms, ps, title_ms, title_ps) = ([0x4du8; 24], [0x50u8; 12], [1u8; 40], [2u8; 40]);
    let rs = 0x7E57_0000_5000usize;
    for title_has_ps in [true, false] {
        let title = console_mesh_stream(rs, &title_ms, title_has_ps.then_some(&title_ps[..]));
        let subs = stream_subobjects(&title.0).expect("a stream XCL reads to its end");
        let out = mesh_standin_stream(&subs, &ms, &ps);
        let standin = stream_subobjects(&out.0).expect("a well-formed stream");
        assert_a_mesh_standin(&standin, rs, &ms, title_has_ps.then_some(&ps[..]));
        assert_eq!(out.shader_bytes(), ms.len() + if title_has_ps { ps.len() } else { 0 });
    }
}

/// A null PS subobject is no pixel shader; of a subobject the stream holds twice, the first
/// counts (D3D12 refuses a stream with two of one type).
#[test]
fn a_mesh_standin_takes_no_null_ps_and_one_of_each_subobject() {
    let (ms, ps, title_ms) = ([0x4du8; 24], [0x50u8; 12], [1u8; 40]);
    let rs = 0x7E57_0000_6000usize;
    let mut title = console_mesh_stream(rs, &title_ms, None);
    title.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS, &D3D12_SHADER_BYTECODE::default());
    title.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE, &0x7E57_0000_7000usize);
    title.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL_FORMAT, &DXGI_FORMAT(40));
    let subs = stream_subobjects(&title.0).unwrap();
    assert!(has_shader(&subs, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS) && !has_shader(&subs, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS));
    assert!(!has_shader(&subs, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS) && !has_shader(&subs, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_AS), "a null slot, no slot");
    let standin = mesh_standin_stream(&subs, &ms, &ps);
    assert_a_mesh_standin(&stream_subobjects(&standin.0).unwrap(), rs, &ms, None);
}

/// `fallback_stream` (the stand-in `standin_stream` hands the title while the pipeline is built
/// on a worker, and the last resort of `dev_create_pipeline_state`) makes a console mesh stream
/// a mesh stand-in. Needs DXC for the stand-in MS (no GPU).
#[test]
fn fallback_stream_of_a_console_mesh_stream_is_a_mesh_pipeline() {
    use crate::gpu::recomp::{dxc, runner};
    if dxc::location().is_err() {
        return runner::skip_test("no DXC");
    }
    let (ms, ps) = dummy_mesh().expect("the stand-in mesh shaders");
    let (title_ms, title_ps) = ([1u8; 40], [2u8; 40]);
    let rs = 0x7E57_0000_8000usize;
    let title = console_mesh_stream(rs, &title_ms, Some(&title_ps));
    let desc: [usize; 2] = [title.0.len(), title.0.as_ptr() as usize];
    let (buf, _) = unsafe { fallback_stream(desc.as_ptr() as *const u8) }.expect("a stand-in");
    let subs = stream_subobjects(&buf).expect("a well-formed stream");
    assert!(!subs.iter().any(|s| s.0 == D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS.0 as u32), "no VS in a mesh pipeline");
    assert_a_mesh_standin(&subs, rs, ms, Some(ps));
    // a graphics stream with a null MS slot is not made a mesh pipeline (rewritten as before)
    let mut vs_stream = StreamOut::default();
    vs_stream.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE, &rs);
    vs_stream.shader(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS, &title_ms);
    vs_stream.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS, &D3D12_SHADER_BYTECODE::default());
    let desc: [usize; 2] = [vs_stream.0.len(), vs_stream.0.as_ptr() as usize];
    let (buf, _) = unsafe { fallback_stream(desc.as_ptr() as *const u8) }.expect("a stand-in");
    assert_eq!(types(&stream_subobjects(&buf).unwrap()), vec![0, 1, 25], "the stream's own subobjects, rewritten in place");
}

/// The driver builds the stand-in of a synthetic console mesh stream (`standin_stream`), so the
/// title gets it at once and the real pipeline is built on a worker, not in the title's call.
#[test]
fn gpu_the_driver_builds_the_standin_of_a_console_mesh_stream() {
    use crate::gpu::recomp::{dxc, runner};
    if dxc::location().is_err() {
        return runner::skip_test("no DXC");
    }
    if super::super::backend() != super::super::Backend::Real {
        return runner::skip_test("no GPU");
    }
    let raw = device_raw();
    let dev = unsafe { ID3D12Device::from_raw_borrowed(&raw) }.unwrap();
    let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS7::default();
    if unsafe { dev.CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS7, (&mut options as *mut D3D12_FEATURE_DATA_D3D12_OPTIONS7).cast(), std::mem::size_of_val(&options) as u32) }.is_err()
        || options.MeshShaderTier.0 < D3D12_MESH_SHADER_TIER_1.0
    {
        return runner::skip_test("no mesh shaders");
    }
    unsafe {
        let mut sig = None;
        D3D12SerializeRootSignature(&D3D12_ROOT_SIGNATURE_DESC::default(), D3D_ROOT_SIGNATURE_VERSION_1, &mut sig, None).unwrap();
        let sig = sig.unwrap();
        let rs: ID3D12RootSignature = dev.CreateRootSignature(0, std::slice::from_raw_parts(sig.GetBufferPointer() as *const u8, sig.GetBufferSize())).unwrap();
        // console bytes the stand-in never reads
        let (title_ms, title_ps) = ([1u8; 40], [2u8; 40]);
        let title = console_mesh_stream(rs.as_raw() as usize, &title_ms, Some(&title_ps));
        let desc: [usize; 2] = [title.0.len(), title.0.as_ptr() as usize];
        let create: PsoFn = std::mem::transmute(orig_fn(raw, slot_of(raw, "CreatePipelineState")));
        let mut pso = std::ptr::null_mut();
        let hr = standin_stream(raw, create, desc.as_ptr() as *const c_void, &pc_iid(Kind::PipelineState), &mut pso);
        assert!(hr >= 0 && !pso.is_null(), "the driver refused the mesh stand-in ({hr:#x})");
        drop(ID3D12PipelineState::from_raw(pso));
    }
}
