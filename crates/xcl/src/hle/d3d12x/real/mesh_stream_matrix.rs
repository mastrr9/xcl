//! The mesh pipelines XCL makes of console mesh streams, with each kind of title state a stream
//! can carry: one synthetic console MS and a PC PS through `translated_mesh_stream`, the path
//! of a title's CreatePipelineState (translation, XCL's RS and pass-through MS, the title's
//! state copied, the driver). The GPU test writes each variant's HRESULT to stderr; with
//! `XCL_D3D12_DEBUG=1` the debug layer's reason for a refusal is logged just before it
//! ("D3D12 debug layer (error, id N)"). It fails when a pipeline D3D12's mesh rules allow is
//! refused; the others are reported only.

use super::*;
use windows::Win32::Graphics::Dxgi::Common::*;

/// What D3D12's rules say of a variant's pipeline.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Rule {
    /// valid on any device with mesh shaders: must be created
    Valid,
    /// valid where the device has this feature (`Caps`): must be created there
    Needs(Feature),
    /// up to the D3D12 runtime's version, or to a rule this test does not settle: reported
    Report,
    /// not a mesh pipeline D3D12 takes as the title gives it (XCL has to leave the subobject
    /// out, or change it): reported
    TitleOnly,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Feature {
    DepthBounds,
    ConservativeRaster,
}

struct Variant {
    name: &'static str,
    rule: Rule,
    /// the title's subobject types `add` puts in the stream, in order
    types: &'static [u32],
    add: fn(&mut StreamOut),
}

fn opaque() -> D3D12_BLEND_DESC {
    let mut b = D3D12_BLEND_DESC::default();
    for rt in &mut b.RenderTarget {
        rt.SrcBlend = D3D12_BLEND_ONE;
        rt.DestBlend = D3D12_BLEND_ZERO;
        rt.BlendOp = D3D12_BLEND_OP_ADD;
        rt.SrcBlendAlpha = D3D12_BLEND_ONE;
        rt.DestBlendAlpha = D3D12_BLEND_ZERO;
        rt.BlendOpAlpha = D3D12_BLEND_OP_ADD;
        rt.LogicOp = D3D12_LOGIC_OP_NOOP;
        rt.RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8;
    }
    b
}

fn raster() -> D3D12_RASTERIZER_DESC {
    D3D12_RASTERIZER_DESC { FillMode: D3D12_FILL_MODE_SOLID, CullMode: D3D12_CULL_MODE_NONE, DepthClipEnable: true.into(), ..Default::default() }
}

fn depth() -> D3D12_DEPTH_STENCIL_DESC {
    let keep = D3D12_DEPTH_STENCILOP_DESC { StencilFailOp: D3D12_STENCIL_OP_KEEP, StencilDepthFailOp: D3D12_STENCIL_OP_KEEP, StencilPassOp: D3D12_STENCIL_OP_KEEP, StencilFunc: D3D12_COMPARISON_FUNC_ALWAYS };
    D3D12_DEPTH_STENCIL_DESC { DepthEnable: true.into(), DepthWriteMask: D3D12_DEPTH_WRITE_MASK_ALL, DepthFunc: D3D12_COMPARISON_FUNC_LESS, StencilReadMask: 0xFF, StencilWriteMask: 0xFF, FrontFace: keep, BackFace: keep, ..Default::default() }
}

/// Blend, sample mask, one target of `rt`, depth format `dsv`, one sample.
fn targets(s: &mut StreamOut, rt: u32, dsv: u32) {
    blend_targets(s, &opaque(), rt, dsv)
}

/// `targets` with the blend state `b`.
fn blend_targets(s: &mut StreamOut, b: &D3D12_BLEND_DESC, rt: u32, dsv: u32) {
    s.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_BLEND, b);
    s.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_MASK, &u32::MAX);
    let mut f = D3D12_RT_FORMAT_ARRAY { NumRenderTargets: 1, ..Default::default() };
    f.RTFormats[0] = DXGI_FORMAT(rt as i32);
    s.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS, &f);
    s.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL_FORMAT, &DXGI_FORMAT(dsv as i32));
    s.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_DESC, &DXGI_SAMPLE_DESC { Count: 1, Quality: 0 });
}

/// `targets`' types
const TARGETS: [u32; 5] = [8, 9, 15, 16, 17];

fn variants() -> Vec<Variant> {
    use Rule::*;
    const R8: u32 = 28; // R8G8B8A8_UNORM
    const D32: u32 = 40; // D32_FLOAT
    vec![
        Variant { name: "no title state", rule: Valid, types: &[], add: |_| {} },
        Variant { name: "blend, targets, depth format", rule: Valid, types: &TARGETS, add: |s| targets(s, R8, D32) },
        Variant {
            name: "+ rasterizer, depth stencil",
            rule: Valid,
            types: &[8, 9, 15, 16, 17, 10, 11],
            add: |s| {
                targets(s, R8, D32);
                s.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER, &raster());
                s.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL, &depth());
            },
        },
        // the console's formats, which XCL makes the PC's
        Variant { name: "console target 116, depth 118", rule: Valid, types: &TARGETS, add: |s| targets(s, 116, 118) },
        Variant { name: "console B5G5R5A1 target", rule: Valid, types: &TARGETS, add: |s| targets(s, 86, 0) },
        Variant {
            name: "a format past NumRenderTargets",
            rule: Report,
            types: &[15],
            add: |s| {
                let mut f = D3D12_RT_FORMAT_ARRAY { NumRenderTargets: 1, ..Default::default() };
                f.RTFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
                f.RTFormats[1] = DXGI_FORMAT_R8G8B8A8_UNORM;
                s.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS, &f);
            },
        },
        Variant {
            name: "DEPTH_STENCIL1, depth bounds",
            rule: Needs(Feature::DepthBounds),
            types: &[8, 9, 15, 16, 17, 21],
            add: |s| {
                targets(s, R8, D32);
                let d = depth();
                let d1 = D3D12_DEPTH_STENCIL_DESC1 {
                    DepthEnable: d.DepthEnable,
                    DepthWriteMask: d.DepthWriteMask,
                    DepthFunc: d.DepthFunc,
                    StencilEnable: d.StencilEnable,
                    StencilReadMask: d.StencilReadMask,
                    StencilWriteMask: d.StencilWriteMask,
                    FrontFace: d.FrontFace,
                    BackFace: d.BackFace,
                    DepthBoundsTestEnable: true.into(),
                };
                s.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL1, &d1);
            },
        },
        Variant {
            name: "DEPTH_STENCIL2",
            rule: Report,
            types: &[8, 9, 15, 16, 17, 26],
            add: |s| {
                targets(s, R8, D32);
                let face = D3D12_DEPTH_STENCILOP_DESC1 { StencilFailOp: D3D12_STENCIL_OP_KEEP, StencilDepthFailOp: D3D12_STENCIL_OP_KEEP, StencilPassOp: D3D12_STENCIL_OP_KEEP, StencilFunc: D3D12_COMPARISON_FUNC_ALWAYS, StencilReadMask: 0xFF, StencilWriteMask: 0xFF };
                let d2 = D3D12_DEPTH_STENCIL_DESC2 { DepthEnable: true.into(), DepthWriteMask: D3D12_DEPTH_WRITE_MASK_ALL, DepthFunc: D3D12_COMPARISON_FUNC_LESS, FrontFace: face, BackFace: face, ..Default::default() };
                s.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL2, &d2);
            },
        },
        Variant {
            name: "RASTERIZER1",
            rule: Report,
            types: &[8, 9, 15, 16, 17, 27],
            add: |s| {
                targets(s, R8, D32);
                let r = D3D12_RASTERIZER_DESC1 { FillMode: D3D12_FILL_MODE_SOLID, CullMode: D3D12_CULL_MODE_NONE, DepthBias: 0.5, DepthClipEnable: true.into(), ..Default::default() };
                s.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER1, &r);
            },
        },
        Variant {
            name: "RASTERIZER2",
            rule: Report,
            types: &[8, 9, 15, 16, 17, 28],
            add: |s| {
                targets(s, R8, D32);
                let r = D3D12_RASTERIZER_DESC2 { FillMode: D3D12_FILL_MODE_SOLID, CullMode: D3D12_CULL_MODE_NONE, DepthClipEnable: true.into(), ..Default::default() };
                s.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER2, &r);
            },
        },
        Variant {
            name: "VIEW_INSTANCING, no views",
            rule: Valid,
            types: &[8, 9, 15, 16, 17, 22],
            add: |s| {
                targets(s, R8, D32);
                s.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VIEW_INSTANCING, &D3D12_VIEW_INSTANCING_DESC::default());
            },
        },
        Variant {
            name: "FLAGS none, NODE_MASK 1",
            rule: Valid,
            types: &[8, 9, 15, 16, 17, 20, 18],
            add: |s| {
                targets(s, R8, D32);
                s.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_FLAGS, &D3D12_PIPELINE_STATE_FLAG_NONE);
                s.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_NODE_MASK, &1u32);
            },
        },
        Variant {
            name: "4x MSAA",
            rule: Valid,
            types: &TARGETS,
            add: |s| {
                s.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_BLEND, &opaque());
                s.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_MASK, &u32::MAX);
                let mut f = D3D12_RT_FORMAT_ARRAY { NumRenderTargets: 1, ..Default::default() };
                f.RTFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
                s.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS, &f);
                s.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL_FORMAT, &DXGI_FORMAT_D32_FLOAT);
                s.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_DESC, &DXGI_SAMPLE_DESC { Count: 4, Quality: 0 });
            },
        },
        Variant {
            name: "conservative rasterization",
            rule: Needs(Feature::ConservativeRaster),
            types: &[8, 9, 15, 16, 17, 10],
            add: |s| {
                targets(s, R8, D32);
                s.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER, &D3D12_RASTERIZER_DESC { ConservativeRaster: D3D12_CONSERVATIVE_RASTERIZATION_MODE_ON, ..raster() });
            },
        },
        Variant {
            name: "wireframe",
            rule: Valid,
            types: &[8, 9, 15, 16, 17, 10],
            add: |s| {
                targets(s, R8, D32);
                s.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER, &D3D12_RASTERIZER_DESC { FillMode: D3D12_FILL_MODE_WIREFRAME, ..raster() });
            },
        },
        // what XCL replaces or leaves out on main
        Variant {
            name: "line topology type",
            rule: Valid,
            types: &[8, 9, 15, 16, 17, 14],
            add: |s| {
                targets(s, R8, D32);
                s.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PRIMITIVE_TOPOLOGY, &D3D12_PRIMITIVE_TOPOLOGY_TYPE_LINE);
            },
        },
        Variant {
            name: "a title RS and cached PSO",
            rule: Valid,
            types: &[8, 9, 15, 16, 17, 0, 19],
            add: |s| {
                targets(s, R8, D32);
                // never read: XCL puts its own RS and no cached PSO
                static BLOB: [u8; 16] = [7; 16];
                s.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE, &(0x10usize as *const c_void));
                s.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_CACHED_PSO, &D3D12_CACHED_PIPELINE_STATE { pCachedBlob: BLOB.as_ptr().cast(), CachedBlobSizeInBytes: 16 });
            },
        },
        // integer targets
        Variant {
            name: "logic op on an R32_UINT target",
            // (the PS writes floats there: a warning at most, but not settled here)
            rule: Report,
            types: &TARGETS,
            add: |s| {
                let mut b = opaque();
                b.RenderTarget[0].LogicOpEnable = true.into();
                b.RenderTarget[0].LogicOp = D3D12_LOGIC_OP_OR;
                blend_targets(s, &b, 42, D32);
            },
        },
        Variant {
            name: "blending on an R32_UINT target",
            rule: TitleOnly,
            types: &TARGETS,
            add: |s| {
                let mut b = opaque();
                b.RenderTarget[0].BlendEnable = true.into();
                blend_targets(s, &b, 42, D32);
            },
        },
        // another stage or the input assembler
        Variant { name: "a null VS", rule: Report, types: &[1], add: |s| s.shader(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS, &[]) },
        Variant {
            name: "an input layout",
            rule: TitleOnly,
            types: &[12],
            add: |s| {
                let e: &'static D3D12_INPUT_ELEMENT_DESC = Box::leak(Box::new(D3D12_INPUT_ELEMENT_DESC { SemanticName: PCSTR(b"POSITION\0".as_ptr()), Format: DXGI_FORMAT_R32G32B32A32_FLOAT, ..Default::default() }));
                s.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_INPUT_LAYOUT, &D3D12_INPUT_LAYOUT_DESC { pInputElementDescs: e, NumElements: 1 });
            },
        },
        Variant { name: "a strip cut value", rule: TitleOnly, types: &[13], add: |s| s.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_IB_STRIP_CUT_VALUE, &D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_0xFFFF) },
        Variant { name: "an empty stream output", rule: Report, types: &[7], add: |s| s.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_STREAM_OUTPUT, &D3D12_STREAM_OUTPUT_DESC::default()) },
    ]
}

/// Each variant's stream has the subobjects it is named for, in a layout XCL reads (sizes and
/// alignment as `null::subobject_payload` has them).
#[test]
fn each_mesh_stream_variant_holds_the_subobjects_it_names() {
    for v in variants() {
        let mut s = StreamOut::default();
        (v.add)(&mut s);
        let subs = stream_subobjects(&s.0).unwrap_or_else(|| panic!("{}: stream not read", v.name));
        let got: Vec<u32> = subs.iter().map(|x| x.0).collect();
        assert_eq!(got, v.types, "{}", v.name);
    }
}

/// GPU: every variant through `translated_mesh_stream` on XCL's device. Run with
/// `XCL_D3D12_DEBUG=1` for the debug layer's reasons.
#[test]
fn gpu_mesh_pipelines_from_console_streams_with_each_kind_of_title_state() {
    use crate::gpu::recomp::{dxc, runner};
    use std::io::Write;
    if dxc::location().is_err() {
        return runner::skip_test("no DXC");
    }
    if super::super::backend() != super::super::Backend::Real {
        return runner::skip_test("no GPU");
    }
    let raw = device_raw();
    let dev = unsafe { ID3D12Device::from_raw_borrowed(&raw) }.unwrap();
    let feature = |f: D3D12_FEATURE, data: *mut c_void, size: usize| unsafe { dev.CheckFeatureSupport(f, data, size as u32) }.is_ok();
    let mut o7 = D3D12_FEATURE_DATA_D3D12_OPTIONS7::default();
    if !feature(D3D12_FEATURE_D3D12_OPTIONS7, (&mut o7 as *mut D3D12_FEATURE_DATA_D3D12_OPTIONS7).cast(), std::mem::size_of_val(&o7)) || o7.MeshShaderTier.0 < D3D12_MESH_SHADER_TIER_1.0 {
        return runner::skip_test("no mesh shaders");
    }
    let mut o = D3D12_FEATURE_DATA_D3D12_OPTIONS::default();
    let mut o2 = D3D12_FEATURE_DATA_D3D12_OPTIONS2::default();
    let has_o = feature(D3D12_FEATURE_D3D12_OPTIONS, (&mut o as *mut D3D12_FEATURE_DATA_D3D12_OPTIONS).cast(), std::mem::size_of_val(&o));
    let has_o2 = feature(D3D12_FEATURE_D3D12_OPTIONS2, (&mut o2 as *mut D3D12_FEATURE_DATA_D3D12_OPTIONS2).cast(), std::mem::size_of_val(&o2));
    let has = |f: Feature| match f {
        Feature::DepthBounds => has_o2 && o2.DepthBoundsTestSupported.as_bool(),
        Feature::ConservativeRaster => has_o && o.ConservativeRasterizationTier.0 >= D3D12_CONSERVATIVE_RASTERIZATION_TIER_1.0,
    };
    let ms = crate::gpu::recomp::pso::shader_cache_tests::synthetic_mesh_container();
    let ps = dxc::compile("float4 main(float4 p : SV_Position) : SV_Target0 { return p; }\n", "main", "ps_6_6", &[]).expect("PS");
    let iid = pc_iid(Kind::PipelineState);
    let mut err = std::io::stderr();
    let mut refused = Vec::new();
    for v in variants() {
        let mut s = StreamOut::default();
        s.shader(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS, &ms);
        s.shader(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS, &ps);
        (v.add)(&mut s);
        let desc = [s.0.len(), s.0.as_ptr() as usize];
        let mut pso: *mut c_void = std::ptr::null_mut();
        let _ = writeln!(err, "mesh stream: {} ...", v.name);
        let hr = unsafe { translated_mesh_stream(raw, desc.as_ptr().cast(), &iid, &mut pso) };
        let made = matches!(hr, Some(h) if h >= 0) && !pso.is_null();
        let must = match v.rule {
            Rule::Valid => true,
            Rule::Needs(f) => has(f),
            Rule::Report | Rule::TitleOnly => false,
        };
        let _ = writeln!(err, "mesh stream: {:<34} {:<10} {:?}{}", v.name, if made { "created" } else { "REFUSED" }, v.rule, hr.map_or(" (not translated)".into(), |h| format!(" hr {:#010x}", h as u32)));
        if made {
            drop(unsafe { ID3D12PipelineState::from_raw(pso) });
        } else if must {
            refused.push(v.name);
        }
    }
    assert!(refused.is_empty(), "mesh pipelines D3D12 allows, refused: {refused:?} (rerun with XCL_D3D12_DEBUG=1 and --nocapture for the debug layer's reasons)");
}
