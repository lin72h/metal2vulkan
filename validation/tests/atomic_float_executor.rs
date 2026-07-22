//! Synthetic native-Vulkan executor proof for shared `VK_EXT_shader_atomic_float`.
//!
//! Workstream F2 needs proof that the Linux runner can actually execute the advertised
//! `shaderSharedFloat32AtomicAdd` path before workloads depending on it rely on it. This test is
//! deliberately tiny and analytic: 64 invocations atomically add `1.0f` to the same Workgroup slot,
//! synchronize, then lane 0 writes the sum to a storage buffer. The final word must be exactly
//! `64.0f`.

#![cfg(target_os = "linux")]

use metal2vulkan_validation::{
    runner_linux, BufferInput, BufferRole, DataFormat, Dispatch, Inputs, Output, Render, Seed,
    Stage,
};

const ATOMIC_F32_ADD_LL: &str = r#"target triple = "spirv-unknown-vulkan1.3"

@scratch = internal unnamed_addr addrspace(3) global float undef, align 4

define void @atomic_f32_add(ptr addrspace(1) %out, i32 %tid) {
entry:
  %old = tail call fast float @air.atomic.global.add.f32(ptr addrspace(3) @scratch, float 1.000000e+00, i32 0, i32 2, i1 true)
  tail call void @air.wg.barrier(i32 2, i32 1)
  %is_lane0 = icmp eq i32 %tid, 0
  br i1 %is_lane0, label %write, label %done

write:
  %sum = load float, ptr addrspace(3) @scratch, align 4
  store float %sum, ptr addrspace(1) %out, align 4
  br label %done

done:
  ret void
}

declare float @air.atomic.global.add.f32(ptr addrspace(3), float, i32, i32, i1)
declare void @air.wg.barrier(i32, i32)

!air.kernel = !{!0}
!0 = !{ptr @atomic_f32_add, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"out"}
!4 = !{i32 1, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid"}
"#;

#[test]
fn shader_shared_f32_atomic_add_executes_known_result() {
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan-atomic-float-executor-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&tmp).expect("create scratch dir");

    let spv =
        metal2vulkan::translate_sanitized_native(ATOMIC_F32_ADD_LL, Stage::Kernel.into(), &tmp)
            .unwrap_or_else(|error| panic!("translate atomic-float proof: {error}"));
    let asm = metal2vulkan::disassemble(&spv)
        .unwrap_or_else(|error| panic!("disassemble atomic-float proof: {error}"));
    assert!(
        asm.contains("OpExtension \"SPV_EXT_shader_atomic_float_add\""),
        "{asm}"
    );
    assert!(asm.contains("OpCapability AtomicFloat32AddEXT"), "{asm}");
    assert!(asm.contains("OpAtomicFAddEXT"), "{asm}");
    metal2vulkan::tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");

    let initial = 0.0f32.to_le_bytes().to_vec();
    let buffers = leak_buffers(vec![BufferInput {
        index: 0,
        len: initial.len(),
        role: BufferRole::InOut,
        seed: exact(initial),
    }]);
    let inputs = Inputs {
        buffers,
        textures: &[],
        output: Output::Buffer {
            index: 0,
            format: DataFormat::F32,
            len: 4,
        },
        dispatch: Dispatch::default_1d(64),
        render: Render::fullscreen_triangle(1, 1),
        embedded_textures: &[],
    };

    let out = runner_linux::execute(Stage::Kernel, ATOMIC_F32_ADD_LL, &spv, &inputs, &tmp);
    assert_eq!(
        out,
        64.0f32.to_le_bytes().to_vec(),
        "64 lanes of atomic f32 add did not produce exactly 64.0"
    );
}

fn exact(bytes: Vec<u8>) -> Seed {
    Seed::ExactBytes {
        bytes: leak_bytes(bytes),
        reason: "atomic-float executor proof: deterministic analytic input",
    }
}

fn leak_bytes(bytes: Vec<u8>) -> &'static [u8] {
    Box::leak(bytes.into_boxed_slice())
}

fn leak_buffers(buffers: Vec<BufferInput>) -> &'static [BufferInput] {
    Box::leak(buffers.into_boxed_slice())
}
