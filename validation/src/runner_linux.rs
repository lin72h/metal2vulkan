#[cfg(test)]
use crate::texture::texture_kind_from_type_name;
use crate::texture::{texture_kind, texture_seed_bytes, TextureKind};
use crate::{
    seeded_buffer_bytes, seeded_render_target_bytes, BlendMode, DataFormat, Extent3d, Inputs,
    Output, Stage, TextureRole,
};
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::Arc;
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferToImageInfo, CopyImageToBufferInfo,
    PrimaryAutoCommandBuffer, RenderingAttachmentInfo, RenderingInfo,
};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::layout::DescriptorType;
use vulkano::descriptor_set::{DescriptorImageViewInfo, DescriptorSet, WriteDescriptorSet};
use vulkano::device::physical::PhysicalDeviceType;
use vulkano::device::{
    Device, DeviceCreateInfo, DeviceExtensions, DeviceFeatures, Queue, QueueCreateInfo, QueueFlags,
};
use vulkano::format::{Format, NumericType};
use vulkano::image::sampler::{Filter, Sampler, SamplerAddressMode, SamplerCreateInfo};
use vulkano::image::view::{ImageView, ImageViewCreateInfo, ImageViewType};
use vulkano::image::{
    Image, ImageCreateFlags, ImageCreateInfo, ImageLayout, ImageType, ImageUsage,
};
use vulkano::instance::{Instance, InstanceCreateFlags, InstanceCreateInfo, InstanceExtensions};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
use vulkano::pipeline::compute::ComputePipelineCreateInfo;
use vulkano::pipeline::graphics::color_blend::{
    AttachmentBlend, BlendFactor, BlendOp, ColorBlendAttachmentState, ColorBlendState,
};
use vulkano::pipeline::graphics::depth_stencil::DepthStencilState;
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::multisample::MultisampleState;
use vulkano::pipeline::graphics::rasterization::RasterizationState;
use vulkano::pipeline::graphics::subpass::PipelineRenderingCreateInfo;
use vulkano::pipeline::graphics::vertex_input::VertexInputState;
use vulkano::pipeline::graphics::viewport::{Viewport, ViewportState};
use vulkano::pipeline::graphics::{GraphicsPipeline, GraphicsPipelineCreateInfo};
use vulkano::pipeline::layout::{PipelineDescriptorSetLayoutCreateInfo, PipelineLayout};
use vulkano::pipeline::{
    ComputePipeline, Pipeline, PipelineBindPoint, PipelineShaderStageCreateInfo,
};
use vulkano::render_pass::{AttachmentLoadOp, AttachmentStoreOp};
use vulkano::shader::spirv::bytes_to_words;
use vulkano::shader::{ShaderModule, ShaderModuleCreateInfo};
use vulkano::sync::{now, GpuFuture};
use vulkano::Version;
use vulkano::VulkanLibrary;

// The descriptor ABI is metal2vulkan's public reflection contract (R5 dogfood): consume the
// constants the translator decorates with, never redeclare them here.
pub use metal2vulkan::reflect::{
    COLOR_INPUT_BINDING_BASE, RESOURCE_DESCRIPTOR_SET as DESCRIPTOR_SET, SAMPLER_BINDING_BASE,
    TEXTURE_BINDING_BASE,
};

pub fn execute(
    stage: Stage,
    sanitized_ll: &str,
    spv: &[u8],
    inputs: &Inputs,
    tmp: &Path,
) -> Vec<u8> {
    match stage {
        Stage::Kernel => execute_compute(sanitized_ll, spv, inputs),
        Stage::Fragment => execute_render_fragment(sanitized_ll, spv, inputs, tmp),
        Stage::Vertex => execute_vertex(sanitized_ll, spv, inputs),
    }
}

fn submit_and_wait(
    device: Arc<Device>,
    queue: Arc<Queue>,
    command_buffer: Arc<PrimaryAutoCommandBuffer>,
    label: &str,
) {
    let future = now(device)
        .then_execute(queue.clone(), command_buffer)
        .unwrap_or_else(|e| panic!("submit {label} command buffer: {e}"))
        .boxed();
    future
        .flush()
        .unwrap_or_else(|e| panic!("flush {label} command buffer: {e}"));
    queue.with(|mut queue| {
        queue
            .wait_idle()
            .unwrap_or_else(|e| panic!("wait for {label} completion: {e}"));
    });
    // queue_wait_idle proves this submission has finished, so Vulkano may release tracked resources.
    unsafe {
        future.signal_finished();
    }
}

fn execute_compute(sanitized_ll: &str, spv: &[u8], inputs: &Inputs) -> Vec<u8> {
    assert!(
        spv.len().is_multiple_of(4),
        "SPIR-V byte stream length must be word-aligned"
    );
    let (device, queue) = device_and_queue(QueueFlags::COMPUTE, false);
    let pipeline = compute_pipeline(device.clone(), spv);
    let required_views = required_texture_view_types(device.clone(), spv);
    let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
    let buffers = make_buffers(memory_allocator.clone(), inputs);
    let mut textures = make_textures(
        memory_allocator.clone(),
        inputs,
        sanitized_ll,
        &required_views,
    );
    append_texture_placeholders(&memory_allocator, device.clone(), spv, &mut textures);
    let pipeline_layout = pipeline.layout().clone();
    let descriptor_set = descriptor_set(
        device.clone(),
        pipeline_layout.clone(),
        &buffers,
        &textures,
        sanitized_ll,
    );
    let texture_readback = make_texture_readback(memory_allocator, inputs.output, sanitized_ll);

    let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
        device.clone(),
        Default::default(),
    ));
    let mut builder = AutoCommandBufferBuilder::primary(
        command_buffer_allocator,
        queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )
    .expect("create primary command buffer");

    for texture in &textures {
        builder
            .copy_buffer_to_image(CopyBufferToImageInfo::buffer_image(
                texture.staging.clone(),
                texture.image.clone(),
            ))
            .unwrap_or_else(|e| panic!("upload texture {}: {e}", texture.index));
    }

    builder
        .bind_pipeline_compute(pipeline.clone())
        .expect("bind compute pipeline");
    if let Some(descriptor_set) = descriptor_set {
        builder
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                pipeline.layout().clone(),
                0,
                descriptor_set,
            )
            .expect("bind compute descriptor set");
    }
    unsafe {
        builder
            .dispatch(workgroup_counts(inputs))
            .expect("dispatch compute pipeline");
    }
    if let Some((texture, readback)) = output_texture_and_readback(&textures, &texture_readback) {
        let mut copy = CopyImageToBufferInfo::image_buffer(texture.image.clone(), readback.clone());
        if texture.kind == TextureKind::Cube {
            copy.regions[0].image_subresource.array_layers = 0..1;
        }
        builder
            .copy_image_to_buffer(copy)
            .unwrap_or_else(|e| panic!("read back texture {}: {e}", texture.index));
    }
    let command_buffer = builder.build().expect("build command buffer");
    submit_and_wait(device, queue, command_buffer, "compute");

    match inputs.output {
        Output::Buffer { index, len, .. } => {
            let buffer = buffers
                .iter()
                .find_map(|(buffer_index, buffer)| (*buffer_index == index).then_some(buffer))
                .unwrap_or_else(|| panic!("output buffer index {index} was not bound"));
            assert!(
                buffer.size() >= len as u64,
                "output buffer index {index} has length {}, expected at least {len}",
                buffer.size()
            );
            let read = buffer.read().expect("read output buffer");
            read[..len].to_vec()
        }
        Output::Texture { .. } => texture_readback
            .expect("texture output readback was not allocated")
            .read()
            .expect("read output texture")
            .to_vec(),
        Output::RenderTarget { .. } => {
            panic!("vulkano runner currently supports compute buffer/texture outputs only")
        }
    }
}

fn execute_render_fragment(
    sanitized_ll: &str,
    fragment_spv: &[u8],
    inputs: &Inputs,
    tmp: &Path,
) -> Vec<u8> {
    assert!(
        fragment_spv.len().is_multiple_of(4),
        "SPIR-V byte stream length must be word-aligned"
    );
    let (format, extent) = match inputs.output {
        Output::RenderTarget { format, extent } => (format, extent),
        other => panic!("fragment render cases must use RenderTarget output, got {other:?}"),
    };
    assert_eq!(
        extent.depth, 1,
        "vulkano runner currently supports 2D render targets only"
    );
    assert_eq!(
        inputs.render.target, extent,
        "render target extent must match render pass target extent"
    );

    let fragment_ll = tmp.join("fragment.ll");
    fs::write(&fragment_ll, sanitized_ll)
        .unwrap_or_else(|e| panic!("write {}: {e}", fragment_ll.display()));
    let vertex_spv = metal2vulkan::translate_passthrough(
        fragment_ll
            .to_str()
            .expect("fragment ll scratch path is not UTF-8"),
        tmp,
    )
    .unwrap_or_else(|e| panic!("translate passthrough vertex shader: {e}"));

    let (device, queue) = device_and_queue(QueueFlags::GRAPHICS, true);
    let pipeline = graphics_pipeline(
        device.clone(),
        &vertex_spv,
        fragment_spv,
        format,
        extent,
        inputs.render.blend,
    );
    let required_views = required_texture_view_types(device.clone(), fragment_spv);
    let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
    let buffers = make_buffers(memory_allocator.clone(), inputs);
    let mut textures = make_textures(
        memory_allocator.clone(),
        inputs,
        sanitized_ll,
        &required_views,
    );
    append_texture_placeholders(
        &memory_allocator,
        device.clone(),
        fragment_spv,
        &mut textures,
    );
    let pipeline_layout = pipeline.layout().clone();
    let descriptor_set = descriptor_set(
        device.clone(),
        pipeline_layout.clone(),
        &buffers,
        &textures,
        sanitized_ll,
    );
    let image = Image::new(
        memory_allocator.clone(),
        ImageCreateInfo {
            image_type: ImageType::Dim2d,
            format: vulkan_format(format),
            extent: [extent.width, extent.height, 1],
            usage: ImageUsage::COLOR_ATTACHMENT
                | ImageUsage::TRANSFER_DST
                | ImageUsage::TRANSFER_SRC,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
            ..Default::default()
        },
    )
    .expect("create render target image");
    let view = ImageView::new_default(image.clone()).expect("create render target view");
    let target_seed = seeded_render_target_bytes(format, extent);
    let target_staging = Buffer::from_iter(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::TRANSFER_SRC,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_HOST
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        target_seed,
    )
    .expect("create render target seed staging buffer");
    let readback = Buffer::from_iter(
        memory_allocator,
        BufferCreateInfo {
            usage: BufferUsage::TRANSFER_DST,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_HOST
                | MemoryTypeFilter::HOST_RANDOM_ACCESS,
            ..Default::default()
        },
        vec![0u8; texture_byte_len(format, extent)],
    )
    .expect("create render target readback buffer");

    let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
        device.clone(),
        Default::default(),
    ));
    let mut builder = AutoCommandBufferBuilder::primary(
        command_buffer_allocator,
        queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )
    .expect("create primary command buffer");

    builder
        .copy_buffer_to_image(CopyBufferToImageInfo::buffer_image(
            target_staging,
            image.clone(),
        ))
        .expect("upload render target seed");
    for texture in &textures {
        builder
            .copy_buffer_to_image(CopyBufferToImageInfo::buffer_image(
                texture.staging.clone(),
                texture.image.clone(),
            ))
            .unwrap_or_else(|e| panic!("upload texture {}: {e}", texture.index));
    }

    let color_attachment = RenderingAttachmentInfo {
        load_op: AttachmentLoadOp::Load,
        store_op: AttachmentStoreOp::Store,
        ..RenderingAttachmentInfo::image_view(view)
    };
    builder
        .begin_rendering(RenderingInfo {
            render_area_extent: [extent.width, extent.height],
            layer_count: 1,
            color_attachments: vec![Some(color_attachment)],
            ..Default::default()
        })
        .expect("begin dynamic rendering")
        .bind_pipeline_graphics(pipeline)
        .expect("bind graphics pipeline");
    if let Some(descriptor_set) = descriptor_set {
        builder
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                pipeline_layout,
                0,
                descriptor_set,
            )
            .expect("bind graphics descriptor set");
    }
    unsafe {
        builder
            .draw(inputs.render.vertex_count, 1, 0, 0)
            .expect("draw fullscreen triangle");
    }
    builder
        .end_rendering()
        .expect("end dynamic rendering")
        .copy_image_to_buffer(CopyImageToBufferInfo::image_buffer(image, readback.clone()))
        .expect("read back render target");

    let command_buffer = builder.build().expect("build command buffer");
    submit_and_wait(device, queue, command_buffer, "render");

    let read = readback.read().expect("read render target");
    read.to_vec()
}

fn execute_vertex(sanitized_ll: &str, vertex_spv: &[u8], inputs: &Inputs) -> Vec<u8> {
    assert!(
        vertex_spv.len().is_multiple_of(4),
        "SPIR-V byte stream length must be word-aligned"
    );
    let (output_index, output_len) = match inputs.output {
        Output::Buffer { index, len, .. } => (index, len),
        other => panic!("standalone vertex cases must use Buffer output, got {other:?}"),
    };
    assert_eq!(
        inputs.render.target.depth, 1,
        "vulkano runner currently supports 2D vertex validation targets only"
    );

    let (device, queue) = device_and_queue(QueueFlags::GRAPHICS, true);
    let pipeline = vertex_pipeline(device.clone(), vertex_spv);
    let required_views = required_texture_view_types(device.clone(), vertex_spv);
    let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
    let buffers = make_buffers(memory_allocator.clone(), inputs);
    let mut textures = make_textures(
        memory_allocator.clone(),
        inputs,
        sanitized_ll,
        &required_views,
    );
    append_texture_placeholders(&memory_allocator, device.clone(), vertex_spv, &mut textures);
    let pipeline_layout = pipeline.layout().clone();
    let descriptor_set = descriptor_set(
        device.clone(),
        pipeline_layout.clone(),
        &buffers,
        &textures,
        sanitized_ll,
    );

    let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
        device.clone(),
        Default::default(),
    ));
    let mut builder = AutoCommandBufferBuilder::primary(
        command_buffer_allocator,
        queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )
    .expect("create primary command buffer");

    for texture in &textures {
        builder
            .copy_buffer_to_image(CopyBufferToImageInfo::buffer_image(
                texture.staging.clone(),
                texture.image.clone(),
            ))
            .unwrap_or_else(|e| panic!("upload texture {}: {e}", texture.index));
    }

    builder
        .begin_rendering(RenderingInfo {
            render_area_extent: [inputs.render.target.width, inputs.render.target.height],
            layer_count: 1,
            color_attachments: vec![],
            ..Default::default()
        })
        .expect("begin vertex validation dynamic rendering")
        .bind_pipeline_graphics(pipeline)
        .expect("bind vertex validation graphics pipeline");
    if let Some(descriptor_set) = descriptor_set {
        builder
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                pipeline_layout,
                0,
                descriptor_set,
            )
            .expect("bind vertex validation descriptor set");
    }
    unsafe {
        builder
            .draw(inputs.render.vertex_count, 1, 0, 0)
            .expect("draw vertex validation primitives");
    }
    builder
        .end_rendering()
        .expect("end vertex validation rendering");

    let command_buffer = builder.build().expect("build command buffer");
    submit_and_wait(device, queue, command_buffer, "vertex validation");

    let buffer = buffers
        .iter()
        .find_map(|(index, buffer)| (*index == output_index).then_some(buffer))
        .unwrap_or_else(|| panic!("output buffer index {output_index} was not bound"));
    assert!(
        buffer.size() >= output_len as u64,
        "output buffer index {output_index} has length {}, expected at least {output_len}",
        buffer.size()
    );
    let read = buffer.read().expect("read vertex validation output buffer");
    read[..output_len].to_vec()
}

fn device_and_queue(
    required_queue_flags: QueueFlags,
    need_dynamic_rendering: bool,
) -> (Arc<Device>, Arc<Queue>) {
    let library = VulkanLibrary::new().expect("load Vulkan library");
    // MoltenVK (macOS) is a NON-CONFORMANT portability driver: the Vulkan loader hides it unless the
    // instance opts in with the ENUMERATE_PORTABILITY flag + the khr_portability_enumeration extension.
    // On Linux (a conformant ICD) neither is needed — set them only when the loader actually exposes the
    // extension, so the same code path is a no-op there.
    let portability_ext = InstanceExtensions {
        khr_portability_enumeration: true,
        ..InstanceExtensions::empty()
    };
    let want_portability = library.supported_extensions().contains(&portability_ext);
    // Cap at 1.3 to match metal2vulkan's spirv-val / emit contract. Vulkano's default is
    // HEADER_VERSION (often newer); that is fine with a modern loader+ICD, but explicit is clearer.
    let instance = Instance::new(
        library,
        InstanceCreateInfo {
            flags: if want_portability {
                InstanceCreateFlags::ENUMERATE_PORTABILITY
            } else {
                InstanceCreateFlags::empty()
            },
            enabled_extensions: if want_portability {
                portability_ext
            } else {
                InstanceExtensions::empty()
            },
            max_api_version: Some(Version::V1_3),
            ..Default::default()
        },
    )
    .unwrap_or_else(|error| {
        panic!(
            "create Vulkan instance: {error}\n\
             VK_ERROR_INCOMPATIBLE_DRIVER almost always means no conformant ICD is installed \
             (or portability devices are hidden). On Linux install mesa-vulkan-drivers \
             (lavapipe) + libvulkan1; on macOS load MoltenVK with ENUMERATE_PORTABILITY."
        )
    });
    let enabled_features = DeviceFeatures {
        dynamic_rendering: need_dynamic_rendering,
        shader_demote_to_helper_invocation: true,
        shader_float16: true,
        shader_int8: true,
        shader_int16: true,
        shader_int64: true,
        shader_buffer_float32_atomic_add: true,
        shader_shared_float32_atomic_add: true,
        shader_subgroup_extended_types: true,
        variable_pointers_storage_buffer: true,
        variable_pointers: true,
        // PhysicalStorageBuffer64 executor (the PSB cross-binding pointer-merge lowering, `native/
        // psb.rs`): the rewritten module reads each merged buffer's 64-bit device address from a
        // synthesized address table and `OpConvertUToPtr`s it. `buffer_device_address` (Vulkan 1.2
        // core, no extension) lets us query `vkGetBufferDeviceAddress` per bound buffer to fill that
        // table — without it a PSB module is byte-WRONG (garbage addresses), i.e. fake conformance.
        buffer_device_address: true,
        ..DeviceFeatures::empty()
    };
    // Extensions REQUIRED of any candidate device (both platforms' drivers advertise these).
    let required_extensions = DeviceExtensions {
        ext_shader_atomic_float: true,
        ..DeviceExtensions::empty()
    };
    // Optional device-name substring override (e.g. METAL2VULKAN_VK_DEVICE=llvmpipe) — used to pin a
    // reference executor when the default DiscreteGpu pick has a driver bug on a specific module
    // class (the NVIDIA driver SIGSEGVs compiling PhysicalStorageBuffer64 compute modules).
    let device_filter = std::env::var("METAL2VULKAN_VK_DEVICE").ok();
    let (physical_device, queue_family_index) = instance
        .enumerate_physical_devices()
        .expect("enumerate Vulkan physical devices")
        .filter(|device| device.supported_extensions().contains(&required_extensions))
        .filter(|device| device.supported_features().contains(&enabled_features))
        .filter(|device| match &device_filter {
            Some(name) => device
                .properties()
                .device_name
                .to_lowercase()
                .contains(&name.to_lowercase()),
            None => true,
        })
        .filter_map(|device| {
            let family = device
                .queue_family_properties()
                .iter()
                .position(|family| family.queue_flags.contains(required_queue_flags))?;
            Some((device, family as u32))
        })
        .min_by_key(|(device, _)| match device.properties().device_type {
            PhysicalDeviceType::DiscreteGpu => 0,
            PhysicalDeviceType::IntegratedGpu => 1,
            PhysicalDeviceType::VirtualGpu => 2,
            PhysicalDeviceType::Cpu => 3,
            PhysicalDeviceType::Other => 4,
            _ => 5,
        })
        .expect("no Vulkan device with a compute queue");

    if std::env::var("METAL2VULKAN_PSB_DEBUG").is_ok() {
        eprintln!(
            "[psb-executor] selected device: {} ({:?})",
            physical_device.properties().device_name,
            physical_device.properties().device_type
        );
    }

    // The Vulkan portability spec MANDATES enabling VK_KHR_portability_subset whenever the device
    // advertises it (MoltenVK does; a conformant Linux ICD does not). Enable it exactly when supported so
    // the same code creates a valid device on both.
    let enabled_extensions = DeviceExtensions {
        ext_shader_atomic_float: true,
        khr_portability_subset: physical_device
            .supported_extensions()
            .khr_portability_subset,
        ..DeviceExtensions::empty()
    };

    let (device, mut queues) = Device::new(
        physical_device,
        DeviceCreateInfo {
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index,
                ..Default::default()
            }],
            enabled_features,
            enabled_extensions,
            ..Default::default()
        },
    )
    .expect("create Vulkan logical device");
    (
        device,
        queues.next().expect("logical device returned no queue"),
    )
}

/// The image-view type each set-0 texture binding's SPIR-V image type requires, keyed by the
/// runner's texture index (binding - TEXTURE_BINDING_BASE). The translator may legally declare a
/// binding with a different dimensionality than the Metal argument type implies (e.g. a texel-read
/// `texturecube` binds as a 2D ARRAY image because Vulkan has no cube fetch), so the view must
/// follow the MODULE's declaration, not the AIR type name.
fn required_texture_view_types(device: Arc<Device>, spv: &[u8]) -> HashMap<u32, ImageViewType> {
    let words = match bytes_to_words(spv) {
        Ok(words) => words,
        Err(_) => return HashMap::new(),
    };
    let module = match unsafe { ShaderModule::new(device, ShaderModuleCreateInfo::new(&words)) } {
        Ok(module) => module,
        Err(_) => return HashMap::new(),
    };
    let Some(entry) = module.entry_point("main") else {
        return HashMap::new();
    };
    entry
        .info()
        .descriptor_binding_requirements
        .iter()
        .filter(|((set, binding), _)| *set == DESCRIPTOR_SET && *binding >= TEXTURE_BINDING_BASE)
        .filter_map(|((_, binding), reqs)| {
            reqs.image_view_type
                .map(|view_type| (*binding - TEXTURE_BINDING_BASE, view_type))
        })
        .collect()
}

fn compute_pipeline(device: Arc<Device>, spv: &[u8]) -> Arc<ComputePipeline> {
    let words = bytes_to_words(spv).expect("SPIR-V bytes must decode to words");
    let module = unsafe { ShaderModule::new(device.clone(), ShaderModuleCreateInfo::new(&words)) }
        .expect("create shader module");
    let entry = module.entry_point("main").expect("SPIR-V entry point main");
    let stage = PipelineShaderStageCreateInfo::new(entry);
    let layout = PipelineLayout::new(
        device.clone(),
        PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
            .into_pipeline_layout_create_info(device.clone())
            .expect("reflect compute pipeline layout"),
    )
    .expect("create compute pipeline layout");
    ComputePipeline::new(
        device,
        None,
        ComputePipelineCreateInfo::stage_layout(stage, layout),
    )
    .expect("create compute pipeline")
}

fn graphics_pipeline(
    device: Arc<Device>,
    vertex_spv: &[u8],
    fragment_spv: &[u8],
    format: DataFormat,
    extent: Extent3d,
    blend: BlendMode,
) -> Arc<GraphicsPipeline> {
    let vertex_stage = shader_stage(device.clone(), vertex_spv);
    let fragment_stage = shader_stage(device.clone(), fragment_spv);
    let layout = PipelineLayout::new(
        device.clone(),
        PipelineDescriptorSetLayoutCreateInfo::from_stages([&vertex_stage, &fragment_stage])
            .into_pipeline_layout_create_info(device.clone())
            .expect("reflect graphics pipeline layout"),
    )
    .expect("create graphics pipeline layout");
    let mut create_info = GraphicsPipelineCreateInfo::layout(layout);
    create_info.stages = [vertex_stage, fragment_stage].into_iter().collect();
    create_info.vertex_input_state = Some(VertexInputState::new());
    create_info.input_assembly_state = Some(InputAssemblyState::default());
    let mut viewport_state = ViewportState::default();
    viewport_state.viewports[0] = Viewport {
        offset: [0.0, 0.0],
        extent: [extent.width as f32, extent.height as f32],
        depth_range: 0.0..=1.0,
    };
    create_info.viewport_state = Some(viewport_state);
    create_info.rasterization_state = Some(RasterizationState::default());
    create_info.multisample_state = Some(MultisampleState::default());
    create_info.color_blend_state = Some(ColorBlendState::with_attachment_states(
        1,
        color_blend_attachment(blend),
    ));
    create_info.subpass = Some(
        PipelineRenderingCreateInfo {
            color_attachment_formats: vec![Some(vulkan_format(format))],
            ..Default::default()
        }
        .into(),
    );
    GraphicsPipeline::new(device, None, create_info).expect("create graphics pipeline")
}

fn vertex_pipeline(device: Arc<Device>, vertex_spv: &[u8]) -> Arc<GraphicsPipeline> {
    let vertex_stage = shader_stage(device.clone(), vertex_spv);
    let layout = PipelineLayout::new(
        device.clone(),
        PipelineDescriptorSetLayoutCreateInfo::from_stages([&vertex_stage])
            .into_pipeline_layout_create_info(device.clone())
            .expect("reflect vertex validation pipeline layout"),
    )
    .expect("create vertex validation pipeline layout");
    let mut create_info = GraphicsPipelineCreateInfo::layout(layout);
    create_info.stages = [vertex_stage].into_iter().collect();
    create_info.vertex_input_state = Some(VertexInputState::new());
    create_info.input_assembly_state = Some(InputAssemblyState::default());
    create_info.rasterization_state = Some(RasterizationState {
        rasterizer_discard_enable: true,
        ..RasterizationState::default()
    });
    create_info.depth_stencil_state = Some(DepthStencilState::default());
    create_info.subpass = Some(PipelineRenderingCreateInfo::default().into());
    GraphicsPipeline::new(device, None, create_info).expect("create vertex validation pipeline")
}

fn color_blend_attachment(blend: BlendMode) -> ColorBlendAttachmentState {
    match blend {
        BlendMode::Replace => ColorBlendAttachmentState::default(),
        BlendMode::SourceOver => ColorBlendAttachmentState {
            blend: Some(AttachmentBlend {
                src_color_blend_factor: BlendFactor::SrcAlpha,
                dst_color_blend_factor: BlendFactor::OneMinusSrcAlpha,
                color_blend_op: BlendOp::Add,
                src_alpha_blend_factor: BlendFactor::One,
                dst_alpha_blend_factor: BlendFactor::OneMinusSrcAlpha,
                alpha_blend_op: BlendOp::Add,
            }),
            ..ColorBlendAttachmentState::default()
        },
    }
}

fn shader_stage(device: Arc<Device>, spv: &[u8]) -> PipelineShaderStageCreateInfo {
    let words = bytes_to_words(spv).expect("SPIR-V bytes must decode to words");
    let module = unsafe { ShaderModule::new(device, ShaderModuleCreateInfo::new(&words)) }
        .expect("create shader module");
    let entry = module.entry_point("main").expect("SPIR-V entry point main");
    PipelineShaderStageCreateInfo::new(entry)
}

fn make_buffers(
    memory_allocator: Arc<StandardMemoryAllocator>,
    inputs: &Inputs,
) -> Vec<(u32, Subbuffer<[u8]>)> {
    inputs
        .buffers
        .iter()
        .map(|input| {
            let bytes = seeded_buffer_bytes(input);
            assert!(
                !bytes.is_empty(),
                "vulkano runner does not support zero-length buffers yet"
            );
            let buffer = Buffer::from_iter(
                memory_allocator.clone(),
                BufferCreateInfo {
                    // SHADER_DEVICE_ADDRESS so a PSB-lowered module can read this buffer's
                    // `vkGetBufferDeviceAddress` into its address table (see `descriptor_set`).
                    // Harmless for non-PSB cases (the address is simply never queried).
                    usage: BufferUsage::STORAGE_BUFFER | BufferUsage::SHADER_DEVICE_ADDRESS,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                        | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                    ..Default::default()
                },
                bytes,
            )
            .unwrap_or_else(|e| panic!("create storage buffer {}: {e}", input.index));
            (input.index, buffer)
        })
        .collect()
}

struct TextureResource {
    index: u32,
    kind: TextureKind,
    is_output: bool,
    image: Arc<Image>,
    view: Arc<ImageView>,
    staging: Subbuffer<[u8]>,
}

fn make_textures(
    memory_allocator: Arc<StandardMemoryAllocator>,
    inputs: &Inputs,
    sanitized_ll: &str,
    required_views: &HashMap<u32, ImageViewType>,
) -> Vec<TextureResource> {
    inputs
        .textures
        .iter()
        .map(|input| {
            let kind = texture_kind(Some(sanitized_ll), input.index);
            let mut shape = texture_shape(input.extent, kind);
            // The module's declared image type wins over the AIR arg-type-derived view: the same
            // underlying image (e.g. a 6-layer cube-compatible allocation) can legally back either
            // view, but the DESCRIPTOR must match what the SPIR-V declares.
            if let Some(required) = required_views.get(&input.index) {
                if shape.view_type != Some(*required) {
                    shape.view_type = Some(*required);
                }
            }
            let bytes = texture_seed_bytes(input, kind, shape.seed_extent);
            assert!(
                !bytes.is_empty(),
                "vulkano runner does not support zero-length textures"
            );
            let staging = Buffer::from_iter(
                memory_allocator.clone(),
                BufferCreateInfo {
                    usage: BufferUsage::TRANSFER_SRC,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_HOST
                        | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
                bytes,
            )
            .unwrap_or_else(|e| panic!("create texture staging buffer {}: {e}", input.index));
            let image = Image::new(
                memory_allocator.clone(),
                ImageCreateInfo {
                    flags: shape.flags,
                    image_type: shape.image_type,
                    format: vulkan_format(input.format),
                    extent: shape.extent,
                    array_layers: shape.array_layers,
                    usage: vulkan_image_usage(input.role),
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                    ..Default::default()
                },
            )
            .unwrap_or_else(|e| panic!("create sampled image {}: {e}", input.index));
            let mut view_info = ImageViewCreateInfo::from_image(&image);
            if let Some(view_type) = shape.view_type {
                view_info.view_type = view_type;
            }
            let view = ImageView::new(image.clone(), view_info)
                .unwrap_or_else(|e| panic!("create image view {}: {e}", input.index));
            TextureResource {
                index: input.index,
                kind,
                is_output: is_output_texture(inputs.output, input.index),
                image,
                view,
                staging,
            }
        })
        .collect()
}

/// A set-0 texture binding the translated SPIR-V declares, reflected structurally from the module
/// (never keyed on any shader/case name). `is_storage` is true for a `StorageImage` binding, false
/// for a `SampledImage`; the format/scalar-type/view-type carry vulkano's recovered `OpTypeImage`
/// constraints, used to synthesize a placeholder image that Vulkan accepts for that binding.
struct TextureBindingReq {
    index: u32,
    is_storage: bool,
    image_format: Option<Format>,
    image_scalar_type: Option<NumericType>,
    image_view_type: Option<ImageViewType>,
}

/// Reflect every set-0 sampled/storage-image binding the SPIR-V declares (bindings in
/// `[TEXTURE_BINDING_BASE, SAMPLER_BINDING_BASE)`), keyed by the runner's texture index
/// (`binding - TEXTURE_BINDING_BASE`). Mirrors `required_texture_view_types` but also recovers the
/// descriptor type, image format, and scalar type so a missing binding can be filled with a
/// correctly-typed placeholder.
fn reflect_texture_bindings(device: Arc<Device>, spv: &[u8]) -> Vec<TextureBindingReq> {
    let Ok(words) = bytes_to_words(spv) else {
        return Vec::new();
    };
    let Ok(module) = (unsafe { ShaderModule::new(device, ShaderModuleCreateInfo::new(&words)) })
    else {
        return Vec::new();
    };
    let Some(entry) = module.entry_point("main") else {
        return Vec::new();
    };
    entry
        .info()
        .descriptor_binding_requirements
        .iter()
        .filter(|((set, binding), _)| {
            *set == DESCRIPTOR_SET
                && *binding >= TEXTURE_BINDING_BASE
                && *binding < SAMPLER_BINDING_BASE
        })
        .filter_map(|((_, binding), reqs)| {
            let is_sampled = reqs
                .descriptor_types
                .contains(&DescriptorType::SampledImage);
            let is_storage = reqs
                .descriptor_types
                .contains(&DescriptorType::StorageImage);
            if !is_sampled && !is_storage {
                return None;
            }
            Some(TextureBindingReq {
                index: *binding - TEXTURE_BINDING_BASE,
                // A binding advertised as both binds as a storage image (the stronger usage).
                is_storage,
                image_format: reqs.image_format,
                image_scalar_type: reqs.image_scalar_type,
                image_view_type: reqs.image_view_type,
            })
        })
        .collect()
}

/// The `(image_type, array_layers, flags)` a placeholder image needs to back a view of the given
/// type — the minimal shape `texture_shape` would produce for a 1x1 texture of that dimensionality.
/// A cube view needs six array layers and the cube-compatible flag.
fn placeholder_image_shape(view_type: ImageViewType) -> (ImageType, u32, ImageCreateFlags) {
    match view_type {
        ImageViewType::Dim1d | ImageViewType::Dim1dArray => {
            (ImageType::Dim1d, 1, ImageCreateFlags::empty())
        }
        ImageViewType::Dim3d => (ImageType::Dim3d, 1, ImageCreateFlags::empty()),
        ImageViewType::Cube | ImageViewType::CubeArray => {
            (ImageType::Dim2d, 6, ImageCreateFlags::CUBE_COMPATIBLE)
        }
        // Dim2d and Dim2dArray both back a single-layer 2D image here.
        _ => (ImageType::Dim2d, 1, ImageCreateFlags::empty()),
    }
}

/// The format for a placeholder image. A storage image binding declares an explicit `OpTypeImage`
/// format (recovered as `image_format`), which the view MUST match; a sampled image is
/// format-flexible, so pick a standard format whose scalar class matches what the shader samples
/// (float/uint/sint), defaulting to float.
fn placeholder_format(req: &TextureBindingReq) -> Format {
    if let Some(format) = req.image_format {
        return format;
    }
    match req.image_scalar_type {
        Some(NumericType::Int) => Format::R8G8B8A8_SINT,
        Some(NumericType::Uint) => Format::R8G8B8A8_UINT,
        // Float or unknown scalar class.
        _ => Format::R32G32B32A32_SFLOAT,
    }
}

/// A `TextureKind` for a placeholder, derived purely from its view type. Placeholders are never the
/// output texture, so `kind` is not load-bearing for readback — it only keeps the resource
/// self-consistent.
fn placeholder_kind(view_type: ImageViewType) -> TextureKind {
    match view_type {
        ImageViewType::Dim1d | ImageViewType::Dim1dArray => TextureKind::Dim1d,
        ImageViewType::Dim3d => TextureKind::Dim3d,
        ImageViewType::Cube | ImageViewType::CubeArray => TextureKind::Cube,
        ImageViewType::Dim2dArray => TextureKind::Dim2dArray,
        _ => TextureKind::Plain,
    }
}

/// Append a zero-filled 1x1 placeholder `TextureResource` for every set-0 texture binding the
/// translated SPIR-V declares but `inputs.textures` did not provide (a synth-override can make a
/// sampler kernel runnable while its texture manifest is short of the reflected bindings). Without
/// this the descriptor loop aborts the whole runner (`descriptor set expects texture binding`); the
/// placeholder reads as zero, matching what Apple's oracle sees for an unbound (nil) texture, so the
/// case runs and can be byte-gated. Placeholders ride the same zero-staging + `copy_buffer_to_image`
/// upload path as real textures (the caller uploads every `TextureResource` before dispatch), giving
/// deterministic zeros. Entirely structural — the binding set, types, and formats come from module
/// reflection, never from a shader/case name.
fn append_texture_placeholders(
    memory_allocator: &Arc<StandardMemoryAllocator>,
    device: Arc<Device>,
    spv: &[u8],
    textures: &mut Vec<TextureResource>,
) {
    for req in reflect_texture_bindings(device, spv) {
        if textures.iter().any(|texture| texture.index == req.index) {
            continue;
        }
        let view_type = req.image_view_type.unwrap_or(ImageViewType::Dim2d);
        let (image_type, array_layers, flags) = placeholder_image_shape(view_type);
        let format = placeholder_format(&req);
        let extent = [1u32, 1, 1];
        let usage = if req.is_storage {
            ImageUsage::STORAGE | ImageUsage::TRANSFER_DST | ImageUsage::TRANSFER_SRC
        } else {
            ImageUsage::SAMPLED | ImageUsage::TRANSFER_DST | ImageUsage::TRANSFER_SRC
        };
        let texel_count = (extent[0] * extent[1] * extent[2] * array_layers) as usize;
        let byte_len = format.block_size() as usize * texel_count;
        let staging = Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            vec![0u8; byte_len],
        )
        .unwrap_or_else(|e| panic!("create placeholder texture staging {}: {e}", req.index));
        let image = Image::new(
            memory_allocator.clone(),
            ImageCreateInfo {
                flags,
                image_type,
                format,
                extent,
                array_layers,
                usage,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
        )
        .unwrap_or_else(|e| panic!("create placeholder texture image {}: {e}", req.index));
        let mut view_info = ImageViewCreateInfo::from_image(&image);
        view_info.view_type = view_type;
        let view = ImageView::new(image.clone(), view_info)
            .unwrap_or_else(|e| panic!("create placeholder texture view {}: {e}", req.index));
        textures.push(TextureResource {
            index: req.index,
            kind: placeholder_kind(view_type),
            is_output: false,
            image,
            view,
            staging,
        });
    }
}

fn is_output_texture(output: Output, index: u32) -> bool {
    matches!(output, Output::Texture { index: output_index, .. } if output_index == index)
}

fn make_texture_readback(
    memory_allocator: Arc<StandardMemoryAllocator>,
    output: Output,
    sanitized_ll: &str,
) -> Option<Subbuffer<[u8]>> {
    match output {
        Output::Texture {
            index,
            format,
            extent,
        } => {
            let kind = texture_kind(Some(sanitized_ll), index);
            let len = texture_byte_len(format, texture_output_extent(extent, kind));
            Some(
                Buffer::from_iter(
                    memory_allocator,
                    BufferCreateInfo {
                        usage: BufferUsage::TRANSFER_DST,
                        ..Default::default()
                    },
                    AllocationCreateInfo {
                        memory_type_filter: MemoryTypeFilter::PREFER_HOST
                            | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                        ..Default::default()
                    },
                    vec![0u8; len],
                )
                .expect("create texture readback buffer"),
            )
        }
        Output::Buffer { .. } | Output::RenderTarget { .. } => None,
    }
}

fn output_texture_and_readback<'a>(
    textures: &'a [TextureResource],
    texture_readback: &'a Option<Subbuffer<[u8]>>,
) -> Option<(&'a TextureResource, &'a Subbuffer<[u8]>)> {
    let readback = texture_readback.as_ref()?;
    let texture = textures
        .iter()
        .find(|texture| texture.is_output)
        .expect("output texture was not bound");
    Some((texture, readback))
}

fn texture_byte_len(format: DataFormat, extent: Extent3d) -> usize {
    let stride = format
        .bytes_per_pixel()
        .unwrap_or_else(|| panic!("texture format {format:?} has no pixel stride"));
    extent.texel_count() * stride
}

#[derive(Clone, Copy, Debug)]
struct TextureShape {
    flags: ImageCreateFlags,
    image_type: ImageType,
    extent: [u32; 3],
    array_layers: u32,
    view_type: Option<ImageViewType>,
    seed_extent: Extent3d,
}

fn texture_shape(extent: Extent3d, kind: TextureKind) -> TextureShape {
    match kind {
        TextureKind::Dim1d => TextureShape {
            flags: ImageCreateFlags::empty(),
            image_type: ImageType::Dim1d,
            extent: [extent.width, 1, 1],
            array_layers: 1,
            view_type: Some(ImageViewType::Dim1d),
            seed_extent: Extent3d::new(extent.width, 1, 1),
        },
        TextureKind::Dim2dArray => TextureShape {
            flags: ImageCreateFlags::empty(),
            image_type: ImageType::Dim2d,
            extent: [extent.width, extent.height, 1],
            array_layers: extent.depth.max(1),
            view_type: Some(ImageViewType::Dim2dArray),
            seed_extent: extent,
        },
        TextureKind::Dim3d => TextureShape {
            flags: ImageCreateFlags::empty(),
            image_type: ImageType::Dim3d,
            extent: [extent.width, extent.height, extent.depth.max(1)],
            array_layers: 1,
            view_type: Some(ImageViewType::Dim3d),
            seed_extent: extent,
        },
        TextureKind::Cube => TextureShape {
            flags: ImageCreateFlags::CUBE_COMPATIBLE,
            image_type: ImageType::Dim2d,
            extent: [extent.width, extent.height, 1],
            array_layers: 6,
            view_type: Some(ImageViewType::Cube),
            seed_extent: Extent3d::new(extent.width, extent.height, 6),
        },
        TextureKind::Plain => TextureShape {
            flags: ImageCreateFlags::empty(),
            image_type: vulkan_image_type(extent),
            extent: [extent.width, extent.height, extent.depth],
            array_layers: 1,
            view_type: None,
            seed_extent: extent,
        },
    }
}

/// The extent an output texture's readback actually covers, derived from the texture's declared
/// kind rather than the caller's (2D-shaped) contract extent. A 1D texture holds one row of
/// `width` texels regardless of the contract's `h`; a cube readback covers the single face the
/// harness copies (`array_layers = 0..1`). Reading a larger region than the texture stores would
/// silently return zero padding — never real texel data — so the contract length must follow the
/// texture's real shape. Mirrors the macOS oracle's readback extent so goldens stay comparable.
fn texture_output_extent(extent: Extent3d, kind: TextureKind) -> Extent3d {
    match kind {
        TextureKind::Dim1d => Extent3d::new(extent.width, 1, 1),
        TextureKind::Cube => Extent3d::new(extent.width, extent.height, 1),
        TextureKind::Plain | TextureKind::Dim2dArray | TextureKind::Dim3d => extent,
    }
}

fn vulkan_image_usage(role: TextureRole) -> ImageUsage {
    let usage = ImageUsage::TRANSFER_DST | ImageUsage::TRANSFER_SRC;
    match role {
        TextureRole::Sampled | TextureRole::StorageRead | TextureRole::InputAttachment => {
            usage | ImageUsage::SAMPLED
        }
        TextureRole::StorageWrite => usage | ImageUsage::STORAGE,
        TextureRole::StorageReadWrite => usage | ImageUsage::SAMPLED | ImageUsage::STORAGE,
        TextureRole::ColorTarget => usage | ImageUsage::COLOR_ATTACHMENT,
    }
}

fn vulkan_format(format: DataFormat) -> Format {
    match format {
        DataFormat::Rgba8Unorm => Format::R8G8B8A8_UNORM,
        DataFormat::Rgba8Uint => Format::R8G8B8A8_UINT,
        DataFormat::Rgba8Sint => Format::R8G8B8A8_SINT,
        DataFormat::Rgba16Uint => Format::R16G16B16A16_UINT,
        DataFormat::Rgba16Float => Format::R16G16B16A16_SFLOAT,
        DataFormat::Rgba32Float => Format::R32G32B32A32_SFLOAT,
        DataFormat::R32Float => Format::R32_SFLOAT,
        _ => panic!("unsupported Vulkan texture format {format:?}"),
    }
}

fn vulkan_image_type(extent: Extent3d) -> ImageType {
    if extent.depth > 1 {
        ImageType::Dim3d
    } else {
        ImageType::Dim2d
    }
}

fn descriptor_set(
    device: Arc<Device>,
    layout: Arc<PipelineLayout>,
    buffers: &[(u32, Subbuffer<[u8]>)],
    textures: &[TextureResource],
    sanitized_ll: &str,
) -> Option<Arc<DescriptorSet>> {
    let set_layout = layout.set_layouts().first()?.clone();
    if set_layout.bindings().is_empty() {
        return None;
    }
    let allocator = Arc::new(StandardDescriptorSetAllocator::new(
        device.clone(),
        Default::default(),
    ));
    // PSB address-table binding. The `native/psb.rs` lowering synthesizes ONE extra StorageBuffer
    // (`{ runtimearray u64 }`) at a fresh binding holding each merged buffer's device address indexed
    // by its OWN descriptor binding. That binding has no corresponding seeded input, so it is exactly
    // the StorageBuffer binding in the reflected layout with no matching input buffer. Detect it (there
    // is at most one — a non-PSB module has none) and fill it with `vkGetBufferDeviceAddress` per bound
    // buffer; a second unmatched storage binding would be a real wiring bug, so we panic on it.
    let unmatched_storage: Vec<u32> = set_layout
        .bindings()
        .iter()
        .filter(|(_, info)| info.descriptor_type == DescriptorType::StorageBuffer)
        .map(|(binding, _)| *binding)
        .filter(|binding| !buffers.iter().any(|(index, _)| index == binding))
        .collect();
    let address_table_binding: Option<u32> = match unmatched_storage.as_slice() {
        [] => None,
        [binding] => Some(*binding),
        many => panic!(
            "multiple storage-buffer bindings have no input buffer: {many:?} (expected at most one \
             synthesized PSB address table)"
        ),
    };
    let address_table_buffer = address_table_binding.map(|_| {
        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
        // table[binding] = device address of the buffer bound at that binding, 0 where none. Sized to
        // cover every real buffer binding; the shader only indexes table[binding] for actual buffers.
        let max_binding = buffers.iter().map(|(index, _)| *index).max().unwrap_or(0);
        let mut table = vec![0u64; max_binding as usize + 1];
        for (index, buffer) in buffers {
            table[*index as usize] = buffer
                .device_address()
                .unwrap_or_else(|e| panic!("query device address of buffer {index}: {e}"))
                .get();
        }
        if std::env::var("METAL2VULKAN_PSB_DEBUG").is_ok() {
            eprintln!(
                "[psb-executor] table binding={:?} entries={:?}",
                address_table_binding, table
            );
        }
        Buffer::from_iter(
            memory_allocator,
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            table,
        )
        .expect("create PSB address-table buffer")
    });
    let default_sampler =
        Sampler::new(device.clone(), SamplerCreateInfo::default()).expect("create sampler");
    let static_samplers = static_sampler_states(sanitized_ll)
        .into_iter()
        .map(|state| {
            Sampler::new(device.clone(), state.create_info()).unwrap_or_else(|e| {
                panic!("create static sampler for AIR word {:#x}: {e}", state.word)
            })
        })
        .collect::<Vec<_>>();
    let mut static_sampler_index = 0usize;
    let writes = set_layout
        .bindings()
        .iter()
        .map(|(binding, info)| match info.descriptor_type {
            DescriptorType::StorageBuffer => {
                if address_table_binding == Some(*binding) {
                    let table = address_table_buffer
                        .clone()
                        .expect("PSB address-table buffer must exist for its binding");
                    return WriteDescriptorSet::buffer(*binding, table);
                }
                let buffer = buffers
                    .iter()
                    .find_map(|(index, buffer)| (*index == *binding).then_some(buffer))
                    .unwrap_or_else(|| {
                        panic!("descriptor set expects storage buffer binding {binding}")
                    });
                WriteDescriptorSet::buffer(*binding, buffer.clone())
            }
            DescriptorType::SampledImage => {
                let texture_index =
                    binding
                        .checked_sub(TEXTURE_BINDING_BASE)
                        .unwrap_or_else(|| {
                            panic!("sampled image binding {binding} is below texture base")
                        });
                let texture = textures
                    .iter()
                    .find(|texture| texture.index == texture_index)
                    .unwrap_or_else(|| panic!("descriptor set expects texture binding {binding}"));
                WriteDescriptorSet::image_view(*binding, texture.view.clone())
            }
            DescriptorType::StorageImage => {
                let texture_index =
                    binding
                        .checked_sub(TEXTURE_BINDING_BASE)
                        .unwrap_or_else(|| {
                            panic!("storage image binding {binding} is below texture base")
                        });
                let texture = textures
                    .iter()
                    .find(|texture| texture.index == texture_index)
                    .unwrap_or_else(|| panic!("descriptor set expects texture binding {binding}"));
                WriteDescriptorSet::image_view_with_layout(
                    *binding,
                    DescriptorImageViewInfo {
                        image_view: texture.view.clone(),
                        image_layout: ImageLayout::General,
                    },
                )
            }
            DescriptorType::Sampler => {
                let sampler = if *binding < SAMPLER_BINDING_BASE {
                    let sampler = static_samplers
                        .get(static_sampler_index)
                        .cloned()
                        .unwrap_or_else(|| default_sampler.clone());
                    static_sampler_index += 1;
                    sampler
                } else {
                    default_sampler.clone()
                };
                WriteDescriptorSet::sampler(*binding, sampler)
            }
            other => panic!("unsupported descriptor binding {binding}: {other:?}"),
        })
        .collect::<Vec<_>>();
    Some(DescriptorSet::new(allocator, set_layout, writes, []).expect("create descriptor set"))
}

#[derive(Clone, Copy, Debug)]
struct StaticSamplerState {
    word: u64,
    address_mode: [SamplerAddressMode; 3],
    filter: Filter,
    unnormalized_coordinates: bool,
}

impl StaticSamplerState {
    fn from_air_word(word: i64) -> Self {
        let word = word as u64;
        Self {
            word,
            // Verified against macOS-seeded AIR constants for clamp-to-zero, clamp-to-edge, repeat,
            // mirrored-repeat, and coord::pixel samplers. Keep unverified sampler dimensions on
            // Vulkano defaults.
            address_mode: [
                decode_air_sampler_address(word & 0x7),
                decode_air_sampler_address((word >> 3) & 0x7),
                decode_air_sampler_address((word >> 6) & 0x7),
            ],
            filter: if word & 0x0a00 == 0x0a00 {
                Filter::Linear
            } else {
                Filter::Nearest
            },
            unnormalized_coordinates: word & 0x8000 != 0,
        }
    }

    fn create_info(self) -> SamplerCreateInfo {
        SamplerCreateInfo {
            mag_filter: self.filter,
            min_filter: self.filter,
            address_mode: self.address_mode,
            unnormalized_coordinates: self.unnormalized_coordinates,
            ..Default::default()
        }
    }
}

fn decode_air_sampler_address(code: u64) -> SamplerAddressMode {
    match code {
        0 => SamplerAddressMode::ClampToBorder,
        1 => SamplerAddressMode::ClampToEdge,
        2 => SamplerAddressMode::Repeat,
        3 => SamplerAddressMode::MirroredRepeat,
        4 => SamplerAddressMode::ClampToBorder,
        _ => SamplerAddressMode::ClampToEdge,
    }
}

fn static_sampler_states(sanitized_ll: &str) -> Vec<StaticSamplerState> {
    sanitized_ll
        .lines()
        .filter(|line| line.contains("@__air_sampler_state") && line.contains("constant"))
        .filter_map(first_i64_literal)
        .map(StaticSamplerState::from_air_word)
        .collect()
}

fn first_i64_literal(line: &str) -> Option<i64> {
    let (_, after_i64) = line.split_once("i64 ")?;
    let token = after_i64
        .trim_start()
        .chars()
        .take_while(|c| c.is_ascii_digit() || *c == '-')
        .collect::<String>();
    (!token.is_empty()).then(|| token.parse().ok()).flatten()
}

fn workgroup_counts(inputs: &Inputs) -> [u32; 3] {
    [
        div_ceil(
            inputs.dispatch.threads_per_grid[0],
            inputs.dispatch.threads_per_threadgroup[0],
        ),
        div_ceil(
            inputs.dispatch.threads_per_grid[1],
            inputs.dispatch.threads_per_threadgroup[1],
        ),
        div_ceil(
            inputs.dispatch.threads_per_grid[2],
            inputs.dispatch.threads_per_threadgroup[2],
        ),
    ]
}

fn div_ceil(numer: u32, denom: u32) -> u32 {
    assert!(denom != 0, "dispatch threadgroup dimension cannot be zero");
    numer.saturating_add(denom - 1) / denom
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DataFormat, Dispatch, Render};

    #[test]
    fn workgroup_counts_round_up_partial_groups() {
        let inputs = Inputs::new(
            &[],
            &[],
            Output::Buffer {
                index: 0,
                format: DataFormat::RawBytes,
                len: 1,
            },
            Dispatch {
                threads_per_grid: [65, 1, 130],
                threads_per_threadgroup: [64, 1, 64],
            },
            Render::fullscreen_triangle(1, 1),
        );
        assert_eq!(workgroup_counts(&inputs), [2, 1, 3]);
    }

    #[test]
    fn texture1d_type_name_uses_1d_shape() {
        let kind = texture_kind_from_type_name(Some("texture1d<float, sample>"));
        let shape = texture_shape(Extent3d::new(8, 8, 1), kind);

        assert_eq!(kind, TextureKind::Dim1d);
        assert_eq!(shape.image_type, ImageType::Dim1d);
        assert_eq!(shape.extent, [8, 1, 1]);
        assert_eq!(shape.view_type, Some(ImageViewType::Dim1d));
        assert_eq!(shape.seed_extent, Extent3d::new(8, 1, 1));
    }

    #[test]
    fn texture3d_type_name_uses_3d_shape_even_for_depth_one() {
        let kind = texture_kind_from_type_name(Some("texture3d<float, write>"));
        let shape = texture_shape(Extent3d::new(8, 8, 1), kind);

        assert_eq!(kind, TextureKind::Dim3d);
        assert_eq!(shape.image_type, ImageType::Dim3d);
        assert_eq!(shape.extent, [8, 8, 1]);
        assert_eq!(shape.view_type, Some(ImageViewType::Dim3d));
        assert_eq!(shape.seed_extent, Extent3d::new(8, 8, 1));
    }

    #[test]
    fn static_sampler_state_decodes_repeat_addressing() {
        let states = static_sampler_states(
            r#"@__air_sampler_state = internal addrspace(2) constant [2 x i64] [i64 34901797601018002, i64 0], align 8"#,
        );

        assert_eq!(states.len(), 1);
        assert_eq!(states[0].address_mode, [SamplerAddressMode::Repeat; 3]);
        assert!(!states[0].unnormalized_coordinates);
    }

    #[test]
    fn static_sampler_state_decodes_pixel_coordinates() {
        let states = static_sampler_states(
            r#"@__air_sampler_state = internal addrspace(2) constant [2 x i64] [i64 34901797601050697, i64 0], align 8"#,
        );

        assert_eq!(states.len(), 1);
        assert_eq!(states[0].address_mode, [SamplerAddressMode::ClampToEdge; 3]);
        assert_eq!(states[0].filter, Filter::Nearest);
        assert!(states[0].unnormalized_coordinates);
    }

    #[test]
    fn static_sampler_state_decodes_linear_filter_bits() {
        let states = static_sampler_states(
            r#"@__air_sampler_state.6 = internal addrspace(2) constant i64 -9188470239253755319, align 8"#,
        );

        assert_eq!(states.len(), 1);
        assert_eq!(states[0].address_mode, [SamplerAddressMode::ClampToEdge; 3]);
        assert_eq!(states[0].filter, Filter::Linear);
        assert!(!states[0].unnormalized_coordinates);
    }

    #[test]
    fn static_sampler_state_decodes_clamp_to_zero_addressing() {
        let states = static_sampler_states(
            r#"@__air_sampler_state = internal addrspace(2) constant [2 x i64] [i64 34901797601017856, i64 0], align 8"#,
        );

        assert_eq!(states.len(), 1);
        assert_eq!(
            states[0].address_mode,
            [SamplerAddressMode::ClampToBorder; 3]
        );
        assert_eq!(states[0].filter, Filter::Nearest);
        assert!(!states[0].unnormalized_coordinates);
    }

    #[test]
    fn static_sampler_state_decodes_mirrored_repeat_addressing() {
        let states = static_sampler_states(
            r#"@__air_sampler_state = internal addrspace(2) constant [2 x i64] [i64 34901797601018075, i64 0], align 8"#,
        );

        assert_eq!(states.len(), 1);
        assert_eq!(
            states[0].address_mode,
            [SamplerAddressMode::MirroredRepeat; 3]
        );
        assert!(!states[0].unnormalized_coordinates);
    }
}
