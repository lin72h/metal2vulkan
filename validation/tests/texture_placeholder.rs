//! M-D4 proof: a compute kernel that reads a texture the manifest does NOT provide runs against a
//! synthesized zero placeholder (`append_texture_placeholders` in `runner_linux.rs`) instead of
//! aborting the runner (`descriptor set expects texture binding 32`).
//!
//! A synth-override can make a sampler/texture kernel runnable while its `textures[]` is empty (or
//! short of the reflected bindings). Apple's oracle binds a nil texture for the unbound slot, which
//! reads as zero at every coordinate. This test drives the same shape locally: `read_rgba8_uint`
//! fetches `tex[(tid&7, tid>>3)]` and packs the four channels into `out[tid]`, run with `textures:
//! &[]`. The placeholder is a 1x1 zero image; on MoltenVK an out-of-bounds `read` maps to a Metal
//! texture read, which is defined to return zero — so every fetch yields 0 and the whole output is
//! zero, matching what Apple's nil texture produces. Before the M-D4 fix this panicked in
//! `descriptor_set`; the assertion here is that it now runs and reads zero.
//!
//! macOS-only and opt-in via `METAL2VULKAN_MOLTENVK` (needs the MoltenVK ICD env, and the
//! out-of-bounds-read-returns-zero guarantee is Metal's — a conformant Linux driver without
//! robustImageAccess would not promise it, so the zero assertion is pinned to the Apple GPU path).
#![cfg(target_os = "macos")]

use metal2vulkan_validation::{
    runner_linux, BufferInput, BufferRole, DataFormat, Dispatch, Inputs, Output, Render, Seed,
    Stage,
};

// `read_rgba8_uint`: out[tid] = pack(tex.read(tid&7, tid>>3)). Same AIR as
// `compute_texture::compute_read_rgba8_uint_texture`, exercised here with no texture bound.
const READ_RGBA8_UINT_LL: &str = r#"source_filename = "case.metal"

define void @read_rgba8_uint(ptr addrspace(1) readonly captures(none) %0, ptr addrspace(1) noundef writeonly captures(none) "air-buffer-no-alias" %1, i32 noundef %2) local_unnamed_addr #0 {
  %4 = and i32 %2, 7
  %5 = lshr i32 %2, 3
  %6 = insertelement <2 x i32> undef, i32 %4, i64 0
  %7 = insertelement <2 x i32> %6, i32 %5, i64 1
  %8 = tail call ptr addrspace(2) @air.get_read_sampler() #3
  %9 = tail call { <4 x i32>, i8 } @air.read_texture_2d.u.v4i32(ptr addrspace(1) readonly captures(none) %0, ptr addrspace(2) %8, <2 x i32> %7, <2 x i32> zeroinitializer, i32 0, i32 1) #4
  %10 = extractvalue { <4 x i32>, i8 } %9, 0
  %11 = extractelement <4 x i32> %10, i64 0
  %12 = extractelement <4 x i32> %10, i64 1
  %13 = shl i32 %12, 8
  %14 = or i32 %13, %11
  %15 = extractelement <4 x i32> %10, i64 2
  %16 = shl i32 %15, 16
  %17 = or i32 %14, %16
  %18 = extractelement <4 x i32> %10, i64 3
  %19 = shl i32 %18, 24
  %20 = or i32 %17, %19
  %21 = zext i32 %2 to i64
  %22 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 %21
  store i32 %20, ptr addrspace(1) %22, align 4, !tbaa !22, !alias.scope !26, !noalias !29
  ret void
}

declare ptr addrspace(2) @air.get_read_sampler() local_unnamed_addr #1

declare { <4 x i32>, i8 } @air.read_texture_2d.u.v4i32(ptr addrspace(1) readonly captures(none), ptr addrspace(2), <2 x i32>, <2 x i32>, i32, i32) local_unnamed_addr #2

attributes #0 = { mustprogress nofree nounwind willreturn "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="128" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }
attributes #1 = { mustprogress nofree nounwind willreturn memory(inaccessiblemem: read) }
attributes #2 = { mustprogress nofree nounwind willreturn memory(argmem: read) }
attributes #3 = { nounwind willreturn memory(inaccessiblemem: read) }
attributes #4 = { nounwind willreturn memory(argmem: read) }

!llvm.module.flags = !{!0, !1, !2, !3, !4, !5, !6, !7, !8}
!air.kernel = !{!9}
!air.compile_options = !{!15, !16, !17}
!llvm.ident = !{!18}
!air.version = !{!19}
!air.language_version = !{!20}
!air.source_file_name = !{!21}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 26, i32 2]}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 7, !"frame-pointer", i32 2}
!3 = !{i32 7, !"air.max_device_buffers", i32 31}
!4 = !{i32 7, !"air.max_constant_buffers", i32 31}
!5 = !{i32 7, !"air.max_threadgroup_buffers", i32 31}
!6 = !{i32 7, !"air.max_textures", i32 128}
!7 = !{i32 7, !"air.max_read_write_textures", i32 8}
!8 = !{i32 7, !"air.max_samplers", i32 16}
!9 = !{ptr @read_rgba8_uint, !10, !11}
!10 = !{}
!11 = !{!12, !13, !14}
!12 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.read", !"air.arg_type_name", !"texture2d<uint, read>", !"air.arg_name", !"tex"}
!13 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!14 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid"}
!15 = !{!"air.compile.denorms_disable"}
!16 = !{!"air.compile.fast_math_enable"}
!17 = !{!"air.compile.framebuffer_fetch_enable"}
!18 = !{!"Apple metal version 32023.864 (metalfe-32023.864)"}
!19 = !{i32 2, i32 8, i32 0}
!20 = !{!"Metal", i32 4, i32 0, i32 0}
!21 = !{!"case.metal"}
!22 = !{!23, !23, i64 0}
!23 = !{!"int", !24, i64 0}
!24 = !{!"omnipotent char", !25, i64 0}
!25 = !{!"Simple C++ TBAA"}
!26 = !{!27}
!27 = distinct !{!27, !28, !"air-alias-scope-arg(1)"}
!28 = distinct !{!28, !"air-alias-scopes(read_rgba8_uint)"}
!29 = !{!30}
!30 = distinct !{!30, !28, !"air-alias-scope-textures"}"#;

static BUFFERS: &[BufferInput] = &[BufferInput {
    index: 0,
    len: 256,
    role: BufferRole::Output,
    seed: Seed::Deterministic { tag: 22 },
}];

#[test]
fn unbound_texture_binding_reads_zero_via_placeholder() {
    // Opt-in: this needs the MoltenVK ICD (VK_ICD_FILENAMES + loader) that the conformance script
    // wires. Without it there is no Vulkan device to run on, so skip rather than fail spuriously.
    if !std::env::var("METAL2VULKAN_MOLTENVK").is_ok_and(|v| v != "0") {
        eprintln!("skipping: set METAL2VULKAN_MOLTENVK=1 (with the MoltenVK ICD env) to run");
        return;
    }

    let tmp = std::env::temp_dir().join("metal2vulkan-texture-placeholder-proof");
    std::fs::create_dir_all(&tmp).expect("create scratch dir");

    let spv =
        metal2vulkan::translate_sanitized_native(READ_RGBA8_UINT_LL, Stage::Kernel.into(), &tmp)
            .unwrap_or_else(|e| panic!("metal2vulkan FALLBACK translating read_rgba8_uint: {e}"));

    // The kernel declares `texture2d<uint, read> tex [[texture(0)]]`, but no texture is provided.
    // Before M-D4 this aborted the runner at `descriptor set expects texture binding 32`.
    let inputs = Inputs {
        buffers: BUFFERS,
        textures: &[],
        output: Output::Buffer {
            index: 0,
            format: DataFormat::U32,
            len: 256,
        },
        dispatch: Dispatch::default_1d(64),
        render: Render::fullscreen_triangle(1, 1),
        embedded_textures: &[],
    };

    let out = runner_linux::execute(Stage::Kernel, READ_RGBA8_UINT_LL, &spv, &inputs, &tmp);

    // Every fetch reads the zero placeholder (in-bounds texel 0, out-of-bounds → Metal returns 0),
    // so each packed `out[tid]` is 0 — matching Apple's nil texture.
    assert_eq!(
        out,
        vec![0u8; 256],
        "unbound-texture kernel did not read all-zero through the placeholder"
    );
}
