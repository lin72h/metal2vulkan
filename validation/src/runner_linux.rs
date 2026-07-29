#[cfg(test)]
use crate::texture::texture_kind_from_type_name;
use crate::texture::{
    texture_kind, texture_output_extent, texture_seed_bytes, texture_seed_extent, TextureKind,
};
use crate::{
    seeded_buffer_bytes, seeded_render_target_bytes, BlendMode, DataFormat, Extent3d, Inputs,
    Output, Stage, TextureInput, TextureRole,
};
use std::collections::HashMap;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use vulkano::buffer::view::{BufferView, BufferViewCreateInfo};
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
use vulkano::pipeline::graphics::depth_stencil::{CompareOp, DepthState, DepthStencilState};
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::multisample::MultisampleState;
use vulkano::pipeline::graphics::rasterization::{FrontFace, RasterizationState};
use vulkano::pipeline::graphics::subpass::PipelineRenderingCreateInfo;
use vulkano::pipeline::graphics::vertex_input::{
    VertexInputAttributeDescription, VertexInputBindingDescription, VertexInputState,
};
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
    execute_result(stage, sanitized_ll, spv, inputs, tmp).unwrap_or_else(|error| panic!("{error}"))
}

pub fn execute_result(
    stage: Stage,
    sanitized_ll: &str,
    spv: &[u8],
    inputs: &Inputs,
    tmp: &Path,
) -> Result<Vec<u8>, String> {
    preflight_shader_device_support(stage, spv)?;
    preflight_texture_binding_view_conflicts(spv)?;
    match stage {
        Stage::Kernel => execute_compute(sanitized_ll, spv, inputs, tmp),
        Stage::Fragment => execute_render_fragment(sanitized_ll, spv, inputs, tmp),
        Stage::Vertex => execute_vertex(sanitized_ll, spv, inputs, tmp),
    }
}

fn preflight_shader_device_support(stage: Stage, spv: &[u8]) -> Result<(), String> {
    let required_queue_flags = match stage {
        Stage::Kernel => QueueFlags::COMPUTE,
        Stage::Fragment | Stage::Vertex => QueueFlags::GRAPHICS,
    };
    let need_dynamic_rendering = matches!(stage, Stage::Fragment | Stage::Vertex);
    let (device, _) = device_and_queue_result(required_queue_flags, need_dynamic_rendering)?;
    let features = device.enabled_features();
    let extensions = device.enabled_extensions();
    for capability in spirv_capabilities(spv)? {
        match capability {
            2 if !features.geometry_shader => {
                return Err("Vulkan device does not support SPIR-V Geometry capability".into());
            }
            57 if !features.multi_viewport => {
                return Err(
                    "Vulkan device does not support SPIR-V MultiViewport capability".into(),
                );
            }
            70 if !features.shader_output_viewport_index => {
                return Err(
                    "Vulkan device does not support SPIR-V ShaderViewportIndex capability".into(),
                );
            }
            69 if !features.shader_output_layer => {
                return Err("Vulkan device does not support SPIR-V ShaderLayer capability".into());
            }
            32 if !features.shader_clip_distance => {
                return Err("Vulkan device does not support SPIR-V ClipDistance capability".into());
            }
            35 if !features.sample_rate_shading => {
                return Err(
                    "Vulkan device does not support SPIR-V SampleRateShading capability".into(),
                );
            }
            5013 if !extensions.ext_shader_stencil_export => {
                return Err(
                    "Vulkan device does not support SPIR-V StencilExportEXT capability".into(),
                );
            }
            _ => {}
        }
    }
    Ok(())
}

fn spirv_capabilities(spv: &[u8]) -> Result<Vec<u32>, String> {
    let words = bytes_to_words(spv).map_err(|error| format!("SPIR-V bytes decode: {error}"))?;
    let mut capabilities = Vec::new();
    let mut index = 5usize;
    while index < words.len() {
        let word = words[index];
        let word_count = (word >> 16) as usize;
        let opcode = word & 0xffff;
        if word_count == 0 || index + word_count > words.len() {
            return Err("SPIR-V instruction stream is malformed".into());
        }
        if opcode == 17 && word_count >= 2 {
            capabilities.push(words[index + 1]);
        }
        index += word_count;
    }
    Ok(capabilities)
}

fn preflight_texture_binding_view_conflicts(spv: &[u8]) -> Result<(), String> {
    let words = bytes_to_words(spv).map_err(|error| format!("SPIR-V bytes decode: {error}"))?;
    let reqs = texture_image_binding_reqs_from_words(&words);
    let mut view_bindings = reqs
        .iter()
        .filter_map(|(index, req)| req.image_view_type.is_none().then_some(index))
        .map(|index| index + TEXTURE_BINDING_BASE)
        .collect::<Vec<_>>();
    let mut scalar_bindings = reqs
        .iter()
        .filter_map(|(index, req)| req.image_scalar_type_conflict.then_some(index))
        .map(|index| index + TEXTURE_BINDING_BASE)
        .collect::<Vec<_>>();
    view_bindings.sort_unstable();
    scalar_bindings.sort_unstable();
    let mut errors = Vec::new();
    if !view_bindings.is_empty() {
        errors.push(format!(
            "incompatible SPIR-V image view types: {view_bindings:?}"
        ));
    }
    if !scalar_bindings.is_empty() {
        errors.push(format!(
            "incompatible SPIR-V image scalar types: {scalar_bindings:?}"
        ));
    }
    if errors.is_empty() {
        return Ok(());
    }
    Err(format!(
        "Vulkan validation runner does not support texture bindings with {}",
        errors.join("; ")
    ))
}

pub(crate) fn fragment_writes_color_location(spv: &[u8], location: u32) -> bool {
    fragment_color_output_locations(spv).contains(&location)
}

fn fragment_color_output_locations(spv: &[u8]) -> Vec<u32> {
    const OP_TYPE_POINTER: u32 = 32;
    const OP_VARIABLE: u32 = 59;
    const OP_DECORATE: u32 = 71;
    const OP_MEMBER_DECORATE: u32 = 72;
    const DECORATION_LOCATION: u32 = 30;
    const STORAGE_CLASS_OUTPUT: u32 = 3;

    let Ok(words) = bytes_to_words(spv) else {
        return Vec::new();
    };
    let mut variable_locations = HashMap::new();
    let mut member_locations: HashMap<u32, Vec<u32>> = HashMap::new();
    let mut pointer_pointees = HashMap::new();
    let mut output_variables = Vec::new();
    let mut index = 5usize;
    while index < words.len() {
        let word = words[index];
        let word_count = (word >> 16) as usize;
        let opcode = word & 0xffff;
        if word_count == 0 || index + word_count > words.len() {
            return Vec::new();
        }
        match opcode {
            OP_TYPE_POINTER if word_count >= 4 => {
                pointer_pointees.insert(words[index + 1], words[index + 3]);
            }
            OP_VARIABLE if word_count >= 4 && words[index + 3] == STORAGE_CLASS_OUTPUT => {
                output_variables.push((words[index + 2], words[index + 1]));
            }
            OP_DECORATE if word_count >= 4 && words[index + 2] == DECORATION_LOCATION => {
                variable_locations.insert(words[index + 1], words[index + 3]);
            }
            OP_MEMBER_DECORATE if word_count >= 5 && words[index + 3] == DECORATION_LOCATION => {
                member_locations
                    .entry(words[index + 1])
                    .or_default()
                    .push(words[index + 4]);
            }
            _ => {}
        }
        index += word_count;
    }

    let mut out = output_variables
        .into_iter()
        .flat_map(|(variable_id, pointer_type_id)| {
            let direct = variable_locations.get(&variable_id).copied().into_iter();
            let members = pointer_pointees
                .get(&pointer_type_id)
                .and_then(|pointee_id| member_locations.get(pointee_id))
                .into_iter()
                .flatten()
                .copied();
            direct.chain(members).collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    out.sort_unstable();
    out.dedup();
    out
}

fn render_target_writes_color0(format: DataFormat, fragment_spv: &[u8]) -> bool {
    !is_depth_format(format) && fragment_writes_color_location(fragment_spv, 0)
}

fn fragment_render_target_attachment_formats(
    sanitized_ll: &str,
    output_format: DataFormat,
    fragment_spv: &[u8],
) -> Vec<Option<DataFormat>> {
    let mut declared_formats = HashMap::new();
    for line in sanitized_ll
        .lines()
        .filter(|line| line.contains(r#""air.render_target""#))
    {
        let Some(index) = metadata_i32_after(line, "air.render_target") else {
            continue;
        };
        let Some(type_name) = metadata_string_after(line, "air.arg_type_name") else {
            continue;
        };
        let Some(format) = fragment_color_format_from_air_type(&type_name) else {
            continue;
        };
        declared_formats.entry(index).or_insert(format);
    }

    let written_locations = fragment_color_output_locations(fragment_spv);
    let mut formats = HashMap::new();
    for location in written_locations {
        let format = declared_formats
            .get(&location)
            .copied()
            .unwrap_or(output_format);
        formats.insert(location, format);
    }

    if formats.is_empty() && render_target_writes_color0(output_format, fragment_spv) {
        formats.insert(0, output_format);
    }

    if formats.is_empty() {
        return Vec::new();
    }

    let max_location = formats.keys().copied().max().unwrap_or(0);
    let mut out = vec![None; max_location as usize + 1];
    for (location, format) in formats {
        let format = if location == 0 { output_format } else { format };
        out[location as usize] = Some(format);
    }
    out
}

fn make_render_color_attachment_views(
    memory_allocator: Arc<StandardMemoryAllocator>,
    formats: &[Option<DataFormat>],
    output_format: DataFormat,
    extent: Extent3d,
    color0_view: Arc<ImageView>,
) -> Vec<Option<Arc<ImageView>>> {
    formats
        .iter()
        .enumerate()
        .map(|(index, format)| {
            let format = (*format)?;
            if index == 0 && format == output_format {
                return Some(color0_view.clone());
            }
            let image = Image::new(
                memory_allocator.clone(),
                ImageCreateInfo {
                    image_type: ImageType::Dim2d,
                    format: vulkan_format(format),
                    extent: [extent.width, extent.height, 1],
                    usage: render_target_usage(format),
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                    ..Default::default()
                },
            )
            .unwrap_or_else(|e| panic!("create extra render target image {index}: {e}"));
            Some(
                ImageView::new_default(image)
                    .unwrap_or_else(|e| panic!("create extra render target view {index}: {e}")),
            )
        })
        .collect()
}

fn fragment_has_flat_integer_input(spv: &[u8]) -> bool {
    const OP_TYPE_INT: u32 = 21;
    const OP_TYPE_VECTOR: u32 = 23;
    const OP_TYPE_POINTER: u32 = 32;
    const OP_VARIABLE: u32 = 59;
    const OP_DECORATE: u32 = 71;
    const DECORATION_FLAT: u32 = 14;
    const STORAGE_CLASS_INPUT: u32 = 1;

    let Ok(words) = bytes_to_words(spv) else {
        return false;
    };
    let mut flat_ids = std::collections::HashSet::new();
    let mut integer_types = std::collections::HashSet::new();
    let mut vector_components = HashMap::new();
    let mut pointer_pointees = HashMap::new();
    let mut input_variables = Vec::new();
    let mut index = 5usize;
    while index < words.len() {
        let word = words[index];
        let word_count = (word >> 16) as usize;
        let opcode = word & 0xffff;
        if word_count == 0 || index + word_count > words.len() {
            return false;
        }
        match opcode {
            OP_TYPE_INT if word_count >= 4 => {
                integer_types.insert(words[index + 1]);
            }
            OP_TYPE_VECTOR if word_count >= 4 => {
                vector_components.insert(words[index + 1], words[index + 2]);
            }
            OP_TYPE_POINTER if word_count >= 4 => {
                pointer_pointees.insert(words[index + 1], words[index + 3]);
            }
            OP_VARIABLE if word_count >= 4 && words[index + 3] == STORAGE_CLASS_INPUT => {
                input_variables.push((words[index + 2], words[index + 1]));
            }
            OP_DECORATE if word_count >= 3 && words[index + 2] == DECORATION_FLAT => {
                flat_ids.insert(words[index + 1]);
            }
            _ => {}
        }
        index += word_count;
    }

    input_variables
        .into_iter()
        .filter(|(variable_id, _)| flat_ids.contains(variable_id))
        .filter_map(|(_, pointer_type_id)| pointer_pointees.get(&pointer_type_id).copied())
        .any(|pointee| {
            integer_types.contains(&pointee)
                || vector_components
                    .get(&pointee)
                    .is_some_and(|component| integer_types.contains(component))
        })
}

fn preflight_nvidia_fragment_graphics_pipeline(
    device: &Arc<Device>,
    vertex_spv: &[u8],
    fragment_spv: &[u8],
    target: RenderPipelineTarget,
    blend: BlendMode,
    tmp: &Path,
) -> Result<(), String> {
    if device.physical_device().properties().vendor_id != 0x10de {
        return Ok(());
    }
    let crash_message = if fragment_has_flat_integer_input(fragment_spv) {
        "Vulkan validation runner skipped NVIDIA graphics pipeline compiler crash for flat integer \
         fragment input"
    } else {
        "Vulkan validation runner skipped NVIDIA graphics pipeline compiler crash"
    };
    run_graphics_pipeline_probe(vertex_spv, fragment_spv, target, blend, tmp, crash_message)
}

fn preflight_nvidia_vertex_validation_graphics_pipeline(
    device: &Arc<Device>,
    vertex_spv: &[u8],
    sanitized_ll: &str,
    tmp: &Path,
) -> Result<(), String> {
    if device.physical_device().properties().vendor_id != 0x10de {
        return Ok(());
    }
    run_vertex_pipeline_probe(vertex_spv, sanitized_ll, tmp)
}

fn preflight_nvidia_compute_pipeline(
    device: &Arc<Device>,
    compute_spv: &[u8],
    tmp: &Path,
) -> Result<(), String> {
    if device.physical_device().properties().vendor_id != 0x10de {
        return Ok(());
    }
    run_compute_pipeline_probe(compute_spv, tmp)
}

fn submit_and_wait(
    device: Arc<Device>,
    queue: Arc<Queue>,
    command_buffer: Arc<PrimaryAutoCommandBuffer>,
    label: &str,
) -> Result<(), String> {
    let future = now(device)
        .then_execute(queue.clone(), command_buffer)
        .map_err(|error| format!("submit {label} command buffer: {error}"))?
        .boxed();
    if let Err(error) = future.flush() {
        // Vulkano's submitted-future Drop path unwraps fence status; after DeviceLost that can
        // panic while we are already trying to return an executor error. This validation worker is
        // about to end the case, so mark the future finished to keep classification honest.
        unsafe {
            future.signal_finished();
        }
        return Err(format!("flush {label} command buffer: {error}"));
    }
    if let Err(error) = queue.with(|mut queue| queue.wait_idle()) {
        // Same rationale as the flush error path above: preserve the executor error row instead
        // of replacing it with a Vulkano Drop panic.
        unsafe {
            future.signal_finished();
        }
        return Err(format!("wait for {label} completion: {error}"));
    }
    // queue_wait_idle proves this submission has finished, so Vulkano may release tracked resources.
    unsafe {
        future.signal_finished();
    }
    Ok(())
}

fn execute_compute(
    sanitized_ll: &str,
    spv: &[u8],
    inputs: &Inputs,
    tmp: &Path,
) -> Result<Vec<u8>, String> {
    assert!(
        spv.len().is_multiple_of(4),
        "SPIR-V byte stream length must be word-aligned"
    );
    let (device, queue) = device_and_queue(QueueFlags::COMPUTE, false);
    preflight_nvidia_compute_pipeline(&device, spv, tmp)?;
    let pipeline = compute_pipeline(device.clone(), spv)?;
    let texture_reqs = required_texture_bindings(device.clone(), spv);
    let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
    let mut buffers = make_buffers(memory_allocator.clone(), inputs);
    buffers.extend(make_compute_stage_input_buffers(
        memory_allocator.clone(),
        &buffers,
        sanitized_ll,
        inputs,
    ));
    let mut textures = make_textures(
        memory_allocator.clone(),
        inputs,
        sanitized_ll,
        &texture_reqs,
    );
    append_texture_placeholders(&memory_allocator, device.clone(), spv, &mut textures);
    let pipeline_layout = pipeline.layout().clone();
    let descriptor_set = descriptor_set(
        device.clone(),
        pipeline_layout.clone(),
        &buffers,
        &textures,
        &[],
        sanitized_ll,
        &texture_reqs,
    );
    let texture_readback =
        make_texture_readback(memory_allocator, inputs.output, sanitized_ll, &texture_reqs);

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
        if texture.texel_buffer.is_none() {
            let mut copy =
                CopyImageToBufferInfo::image_buffer(texture.image.clone(), readback.clone());
            if texture.kind == TextureKind::Cube {
                copy.regions[0].image_subresource.array_layers = 0..1;
            }
            builder
                .copy_image_to_buffer(copy)
                .unwrap_or_else(|e| panic!("read back texture {}: {e}", texture.index));
        }
    }
    let command_buffer = builder.build().expect("build command buffer");
    submit_and_wait(device, queue, command_buffer, "compute")?;

    let output = match inputs.output {
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
        Output::Texture {
            index,
            format,
            extent,
        } => {
            let texture = textures
                .iter()
                .find(|texture| texture.index == index)
                .unwrap_or_else(|| panic!("output texture index {index} was not bound"));
            if let Some(texel_buffer) = &texture.texel_buffer {
                let len = texture_byte_len(format, texture_output_extent(extent, texture.kind));
                assert!(
                    texel_buffer.size() >= len as u64,
                    "output texture buffer index {index} has length {}, expected at least {len}",
                    texel_buffer.size()
                );
                let read = texel_buffer.read().expect("read output texture buffer");
                read[..len].to_vec()
            } else {
                texture_readback
                    .expect("texture output readback was not allocated")
                    .read()
                    .expect("read output texture")
                    .to_vec()
            }
        }
        Output::RenderTarget { .. } => {
            panic!("vulkano runner currently supports compute buffer/texture outputs only")
        }
    };
    Ok(output)
}

fn execute_render_fragment(
    sanitized_ll: &str,
    fragment_spv: &[u8],
    inputs: &Inputs,
    tmp: &Path,
) -> Result<Vec<u8>, String> {
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
    let depth_output = is_depth_format(format);
    let color_attachment_formats = if depth_output {
        Vec::new()
    } else {
        fragment_render_target_attachment_formats(sanitized_ll, format, fragment_spv)
    };
    let target = RenderPipelineTarget {
        format,
        extent,
        color_attachment_formats,
        depth_output,
    };
    preflight_nvidia_fragment_graphics_pipeline(
        &device,
        &vertex_spv,
        fragment_spv,
        target.clone(),
        inputs.render.blend,
        tmp,
    )?;
    let pipeline = graphics_pipeline(
        device.clone(),
        &vertex_spv,
        fragment_spv,
        target.clone(),
        inputs.render.blend,
    );
    let texture_reqs = required_texture_bindings(device.clone(), fragment_spv);
    let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
    let buffers = make_buffers(memory_allocator.clone(), inputs);
    let mut textures = make_textures(
        memory_allocator.clone(),
        inputs,
        sanitized_ll,
        &texture_reqs,
    );
    append_texture_placeholders(
        &memory_allocator,
        device.clone(),
        fragment_spv,
        &mut textures,
    );
    let pipeline_layout = pipeline.layout().clone();
    let target_seed = seeded_render_target_bytes(format, extent);
    let color_inputs = make_color_input_attachments(
        memory_allocator.clone(),
        &pipeline_layout,
        format,
        extent,
        sanitized_ll,
    );
    let descriptor_set = descriptor_set(
        device.clone(),
        pipeline_layout.clone(),
        &buffers,
        &textures,
        &color_inputs,
        sanitized_ll,
        &texture_reqs,
    );
    let image = Image::new(
        memory_allocator.clone(),
        ImageCreateInfo {
            image_type: ImageType::Dim2d,
            format: vulkan_format(format),
            extent: [extent.width, extent.height, 1],
            usage: render_target_usage(format)
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
    let color_attachment_views = make_render_color_attachment_views(
        memory_allocator.clone(),
        &target.color_attachment_formats,
        format,
        extent,
        view.clone(),
    );
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
        target_seed.iter().copied(),
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
    for color_input in &color_inputs {
        builder
            .copy_buffer_to_image(CopyBufferToImageInfo::buffer_image(
                color_input.staging.clone(),
                color_input.image.clone(),
            ))
            .unwrap_or_else(|e| panic!("upload color input {}: {e}", color_input.index));
    }

    let color_attachments = color_attachment_views
        .iter()
        .enumerate()
        .map(|(index, attachment_view)| {
            attachment_view
                .as_ref()
                .map(|attachment_view| RenderingAttachmentInfo {
                    load_op: if index == 0 {
                        AttachmentLoadOp::Load
                    } else {
                        AttachmentLoadOp::DontCare
                    },
                    store_op: AttachmentStoreOp::Store,
                    ..RenderingAttachmentInfo::image_view(attachment_view.clone())
                })
        })
        .collect();
    let depth_attachment = if depth_output {
        Some(RenderingAttachmentInfo {
            load_op: AttachmentLoadOp::Load,
            store_op: AttachmentStoreOp::Store,
            ..RenderingAttachmentInfo::image_view(view.clone())
        })
    } else {
        None
    };
    builder
        .begin_rendering(RenderingInfo {
            render_area_extent: [extent.width, extent.height],
            layer_count: 1,
            color_attachments,
            depth_attachment,
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
    submit_and_wait(device, queue, command_buffer, "render")?;

    let read = readback.read().expect("read render target");
    Ok(read.to_vec())
}

fn execute_vertex(
    sanitized_ll: &str,
    vertex_spv: &[u8],
    inputs: &Inputs,
    tmp: &Path,
) -> Result<Vec<u8>, String> {
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
    let vertex_inputs = vertex_inputs(sanitized_ll);
    preflight_nvidia_vertex_validation_graphics_pipeline(&device, vertex_spv, sanitized_ll, tmp)?;
    let pipeline = vertex_pipeline(device.clone(), vertex_spv, &vertex_inputs)?;
    let texture_reqs = required_texture_bindings(device.clone(), vertex_spv);
    let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
    let buffers = make_buffers(memory_allocator.clone(), inputs);
    let vertex_input_buffer = make_vertex_input_buffer(memory_allocator.clone(), &vertex_inputs);
    let mut textures = make_textures(
        memory_allocator.clone(),
        inputs,
        sanitized_ll,
        &texture_reqs,
    );
    append_texture_placeholders(&memory_allocator, device.clone(), vertex_spv, &mut textures);
    let pipeline_layout = pipeline.layout().clone();
    let descriptor_set = descriptor_set(
        device.clone(),
        pipeline_layout.clone(),
        &buffers,
        &textures,
        &[],
        sanitized_ll,
        &texture_reqs,
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
    if let Some(vertex_input_buffer) = vertex_input_buffer {
        builder
            .bind_vertex_buffers(0, vertex_input_buffer)
            .expect("bind vertex validation vertex input buffer");
    }
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
    submit_and_wait(device, queue, command_buffer, "vertex validation")?;

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
    Ok(read[..output_len].to_vec())
}

/// Load the Vulkan loader dylib/so, including Homebrew Apple Silicon paths that stock vulkano
/// does not probe.
///
/// Vulkano's default macOS search ends at bare names + `/usr/local/lib/libvulkan.dylib` (Intel
/// Homebrew / LunarG SDK layout). On arm64 Homebrew the loader lives at
/// `/opt/homebrew/lib/libvulkan.dylib`, which is outside dyld's fallback list for bare names —
/// every `VulkanLibrary::new()` call then fails and the corpus runner banks
/// `vulkan execute panicked`.
///
/// Override order:
/// 1. `METAL2VULKAN_LIBVULKAN` absolute path (validation tooling only)
/// 2. `$VULKAN_SDK/lib/libvulkan.{dylib,1.dylib}` and `$VULKAN_SDK/macOS/lib/...`
/// 3. Vulkano default (`VulkanLibrary::new`)
/// 4. Known absolute locations (Homebrew prefixes, `/usr/local/lib`)
pub fn load_vulkan_library() -> Result<Arc<VulkanLibrary>, String> {
    if let Ok(explicit) = std::env::var("METAL2VULKAN_LIBVULKAN") {
        let path = explicit.trim();
        if !path.is_empty() {
            return load_vulkan_library_from(path)
                .map_err(|e| format!("METAL2VULKAN_LIBVULKAN={path}: {e}"));
        }
    }

    if let Ok(sdk) = std::env::var("VULKAN_SDK") {
        let sdk = sdk.trim();
        if !sdk.is_empty() {
            for rel in [
                "lib/libvulkan.dylib",
                "lib/libvulkan.1.dylib",
                "macOS/lib/libvulkan.dylib",
                "macOS/lib/libvulkan.1.dylib",
                "lib/libvulkan.so.1",
                "lib/libvulkan.so",
            ] {
                let candidate = format!("{sdk}/{rel}");
                if Path::new(&candidate).is_file() {
                    if let Ok(lib) = load_vulkan_library_from(&candidate) {
                        return Ok(lib);
                    }
                }
            }
        }
    }

    if let Ok(lib) = VulkanLibrary::new() {
        return Ok(lib);
    }

    // Fall through to absolute path probes after the stock search fails.
    let mut last = "VulkanLibrary::new failed".to_string();
    for path in vulkan_library_fallback_paths() {
        match load_vulkan_library_from(&path) {
            Ok(lib) => return Ok(lib),
            Err(e) => last = format!("{path}: {e}"),
        }
    }
    Err(format!(
        "load Vulkan library failed ({last}). \
         Install vulkan-loader (and MoltenVK on macOS), or set METAL2VULKAN_LIBVULKAN to \
         the absolute path of libvulkan.dylib / libvulkan.so.1"
    ))
}

fn load_vulkan_library_from(path: &str) -> Result<Arc<VulkanLibrary>, String> {
    use vulkano::library::DynamicLibraryLoader;
    let loader = unsafe { DynamicLibraryLoader::new(path) }.map_err(|e| e.to_string())?;
    VulkanLibrary::with_loader(loader).map_err(|e| e.to_string())
}

fn vulkan_library_fallback_paths() -> Vec<String> {
    let mut paths = Vec::new();

    // Homebrew: prefer the active prefix, then the two common install roots.
    if let Ok(prefix) = std::env::var("HOMEBREW_PREFIX") {
        let p = prefix.trim();
        if !p.is_empty() {
            paths.push(format!("{p}/lib/libvulkan.dylib"));
            paths.push(format!("{p}/lib/libvulkan.1.dylib"));
            paths.push(format!("{p}/lib/libvulkan.so.1"));
        }
    }
    for root in ["/opt/homebrew", "/usr/local"] {
        paths.push(format!("{root}/lib/libvulkan.dylib"));
        paths.push(format!("{root}/lib/libvulkan.1.dylib"));
    }

    // Dedup while preserving order (HOMEBREW_PREFIX may equal one of the roots).
    let mut seen = std::collections::HashSet::new();
    paths.retain(|p| seen.insert(p.clone()));
    paths
}

fn device_and_queue(
    required_queue_flags: QueueFlags,
    need_dynamic_rendering: bool,
) -> (Arc<Device>, Arc<Queue>) {
    device_and_queue_result(required_queue_flags, need_dynamic_rendering)
        .unwrap_or_else(|error| panic!("{error}"))
}

pub fn device_and_queue_result(
    required_queue_flags: QueueFlags,
    need_dynamic_rendering: bool,
) -> Result<(Arc<Device>, Arc<Queue>), String> {
    let library = load_vulkan_library()?;
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
    .map_err(|error| {
        format!(
            "create Vulkan instance: {error}: \
             VK_ERROR_INCOMPATIBLE_DRIVER almost always means no conformant ICD is installed \
             (or portability devices are hidden). On Linux install mesa-vulkan-drivers \
             (lavapipe) + libvulkan1; on macOS load MoltenVK with ENUMERATE_PORTABILITY."
        )
    })?;
    let required_features = DeviceFeatures {
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
        .map_err(|error| format!("enumerate Vulkan physical devices: {error}"))?
        .filter(|device| device.supported_extensions().contains(&required_extensions))
        .filter(|device| device.supported_features().contains(&required_features))
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
        .ok_or_else(|| {
            format!(
                "no Vulkan device with required features and queue flags {:?}",
                required_queue_flags
            )
        })?;

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
    let mut enabled_features = required_features;
    if physical_device.supported_features().multi_viewport {
        enabled_features.multi_viewport = true;
    }
    if physical_device
        .supported_features()
        .shader_output_viewport_index
    {
        enabled_features.shader_output_viewport_index = true;
    }
    if physical_device.supported_features().shader_output_layer {
        enabled_features.shader_output_layer = true;
    }
    if physical_device.supported_features().shader_clip_distance {
        enabled_features.shader_clip_distance = true;
    }
    if physical_device.supported_features().geometry_shader {
        enabled_features.geometry_shader = true;
    }
    if physical_device.supported_features().image_cube_array {
        enabled_features.image_cube_array = true;
    }
    if physical_device.supported_features().sample_rate_shading {
        enabled_features.sample_rate_shading = true;
    }
    let enabled_extensions = DeviceExtensions {
        ext_shader_atomic_float: true,
        ext_shader_stencil_export: physical_device
            .supported_extensions()
            .ext_shader_stencil_export,
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
    .map_err(|error| format!("create Vulkan logical device: {error}"))?;
    let queue = queues
        .next()
        .ok_or_else(|| "logical device returned no queue".to_string())?;
    Ok((device, queue))
}

fn compute_pipeline(device: Arc<Device>, spv: &[u8]) -> Result<Arc<ComputePipeline>, String> {
    let words = bytes_to_words(spv).expect("SPIR-V bytes must decode to words");
    let module = unsafe { ShaderModule::new(device.clone(), ShaderModuleCreateInfo::new(&words)) }
        .map_err(|error| format!("create shader module: {error}"))?;
    let entry = module
        .entry_point("main")
        .ok_or_else(|| "SPIR-V entry point main not found".to_string())?;
    let stage = PipelineShaderStageCreateInfo::new(entry);
    let layout = PipelineLayout::new(
        device.clone(),
        PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
            .into_pipeline_layout_create_info(device.clone())
            .map_err(|error| format!("reflect compute pipeline layout: {error}"))?,
    )
    .map_err(|error| format!("create compute pipeline layout: {error}"))?;
    ComputePipeline::new(
        device,
        None,
        ComputePipelineCreateInfo::stage_layout(stage, layout),
    )
    .map_err(|error| format!("create compute pipeline: {error}"))
}

#[derive(Clone, Debug)]
struct RenderPipelineTarget {
    format: DataFormat,
    extent: Extent3d,
    color_attachment_formats: Vec<Option<DataFormat>>,
    depth_output: bool,
}

fn graphics_pipeline(
    device: Arc<Device>,
    vertex_spv: &[u8],
    fragment_spv: &[u8],
    target: RenderPipelineTarget,
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
        extent: [target.extent.width as f32, target.extent.height as f32],
        depth_range: 0.0..=1.0,
    };
    create_info.viewport_state = Some(viewport_state);
    create_info.rasterization_state = Some(RasterizationState {
        front_face: FrontFace::Clockwise,
        ..RasterizationState::default()
    });
    create_info.multisample_state = Some(MultisampleState::default());
    create_info.depth_stencil_state = target.depth_output.then(|| DepthStencilState {
        depth: Some(DepthState {
            write_enable: true,
            compare_op: CompareOp::Always,
        }),
        ..Default::default()
    });
    create_info.color_blend_state = if target.color_attachment_formats.is_empty() {
        None
    } else {
        Some(ColorBlendState::with_attachment_states(
            target.color_attachment_formats.len() as u32,
            color_blend_attachment(blend),
        ))
    };
    let color_attachment_formats = target
        .color_attachment_formats
        .iter()
        .map(|format| format.map(vulkan_format))
        .collect();
    let depth_attachment_format = target.depth_output.then(|| vulkan_format(target.format));
    create_info.subpass = Some(
        PipelineRenderingCreateInfo {
            color_attachment_formats,
            depth_attachment_format,
            ..Default::default()
        }
        .into(),
    );
    GraphicsPipeline::new(device, None, create_info).expect("create graphics pipeline")
}

fn run_graphics_pipeline_probe(
    vertex_spv: &[u8],
    fragment_spv: &[u8],
    target: RenderPipelineTarget,
    blend: BlendMode,
    tmp: &Path,
    crash_message: &'static str,
) -> Result<(), String> {
    if !pipeline_probe_subcommands_available() {
        return Ok(());
    }
    let vertex_path = tmp.join("graphics-pipeline-probe.vert.spv");
    let fragment_path = tmp.join("graphics-pipeline-probe.frag.spv");
    fs::write(&vertex_path, vertex_spv)
        .map_err(|error| format!("write {}: {error}", vertex_path.display()))?;
    fs::write(&fragment_path, fragment_spv)
        .map_err(|error| format!("write {}: {error}", fragment_path.display()))?;

    let result = run_graphics_pipeline_probe_child(
        &vertex_path,
        &fragment_path,
        target,
        blend,
        crash_message,
    );
    let _ = fs::remove_file(&vertex_path);
    let _ = fs::remove_file(&fragment_path);
    result
}

fn run_graphics_pipeline_probe_child(
    vertex_path: &Path,
    fragment_path: &Path,
    target: RenderPipelineTarget,
    blend: BlendMode,
    crash_message: &'static str,
) -> Result<(), String> {
    let exe =
        std::env::current_exe().map_err(|error| format!("locate graphics probe exe: {error}"))?;
    let mut cmd = std::process::Command::new(exe);
    cmd.arg("--graphics-pipeline-probe")
        .arg(vertex_path)
        .arg(fragment_path)
        .arg(format!("{:?}", target.format))
        .arg(target.extent.width.to_string())
        .arg(target.extent.height.to_string())
        .arg(probe_color_attachment_formats_arg(
            &target.color_attachment_formats,
        ))
        .arg(if target.depth_output { "1" } else { "0" })
        .arg(format!("{:?}", blend))
        .env("RUST_BACKTRACE", "0")
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped());
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        cmd.process_group(0);
    }

    let mut child = cmd
        .spawn()
        .map_err(|error| format!("spawn graphics pipeline probe: {error}"))?;
    let start = std::time::Instant::now();
    let timeout = std::time::Duration::from_secs(15);
    loop {
        match child.try_wait() {
            Ok(Some(status)) if status.success() => return Ok(()),
            Ok(Some(status)) => {
                #[cfg(unix)]
                {
                    use std::os::unix::process::ExitStatusExt;
                    if status.signal() == Some(11) {
                        return Err(crash_message.into());
                    }
                }
                let stderr = read_child_stderr(&mut child);
                return Err(format_probe_failure(
                    "graphics pipeline probe",
                    status,
                    &stderr,
                ));
            }
            Ok(None) if start.elapsed() < timeout => {
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
            Ok(None) => {
                kill_probe_worker(&mut child);
                let _ = child.wait();
                return Err("graphics pipeline probe timed out after 15s".into());
            }
            Err(error) => {
                kill_probe_worker(&mut child);
                let _ = child.wait();
                return Err(format!("wait graphics pipeline probe: {error}"));
            }
        }
    }
}

fn run_vertex_pipeline_probe(
    vertex_spv: &[u8],
    sanitized_ll: &str,
    tmp: &Path,
) -> Result<(), String> {
    if !pipeline_probe_subcommands_available() {
        return Ok(());
    }
    let vertex_path = tmp.join("vertex-pipeline-probe.vert.spv");
    let ll_path = tmp.join("vertex-pipeline-probe.ll");
    fs::write(&vertex_path, vertex_spv)
        .map_err(|error| format!("write {}: {error}", vertex_path.display()))?;
    fs::write(&ll_path, sanitized_ll)
        .map_err(|error| format!("write {}: {error}", ll_path.display()))?;

    let result = run_vertex_pipeline_probe_child(&vertex_path, &ll_path);
    let _ = fs::remove_file(&vertex_path);
    let _ = fs::remove_file(&ll_path);
    result
}

fn run_vertex_pipeline_probe_child(vertex_path: &Path, ll_path: &Path) -> Result<(), String> {
    let exe =
        std::env::current_exe().map_err(|error| format!("locate vertex probe exe: {error}"))?;
    let mut cmd = std::process::Command::new(exe);
    cmd.arg("--vertex-pipeline-probe")
        .arg(vertex_path)
        .arg(ll_path)
        .env("RUST_BACKTRACE", "0")
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped());
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        cmd.process_group(0);
    }

    let mut child = cmd
        .spawn()
        .map_err(|error| format!("spawn vertex pipeline probe: {error}"))?;
    let start = std::time::Instant::now();
    let timeout = std::time::Duration::from_secs(15);
    loop {
        match child.try_wait() {
            Ok(Some(status)) if status.success() => return Ok(()),
            Ok(Some(status)) => {
                #[cfg(unix)]
                {
                    use std::os::unix::process::ExitStatusExt;
                    if status.signal() == Some(11) {
                        return Err(
                            "Vulkan validation runner skipped NVIDIA graphics pipeline compiler \
                             crash for vertex validation pipeline"
                                .into(),
                        );
                    }
                }
                let stderr = read_child_stderr(&mut child);
                return Err(format_probe_failure(
                    "vertex pipeline probe",
                    status,
                    &stderr,
                ));
            }
            Ok(None) if start.elapsed() < timeout => {
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
            Ok(None) => {
                kill_probe_worker(&mut child);
                let _ = child.wait();
                return Err("vertex pipeline probe timed out after 15s".into());
            }
            Err(error) => {
                kill_probe_worker(&mut child);
                let _ = child.wait();
                return Err(format!("wait vertex pipeline probe: {error}"));
            }
        }
    }
}

fn run_compute_pipeline_probe(compute_spv: &[u8], tmp: &Path) -> Result<(), String> {
    if !pipeline_probe_subcommands_available() {
        return Ok(());
    }
    let compute_path = tmp.join("compute-pipeline-probe.comp.spv");
    fs::write(&compute_path, compute_spv)
        .map_err(|error| format!("write {}: {error}", compute_path.display()))?;

    let result = run_compute_pipeline_probe_child(&compute_path);
    let _ = fs::remove_file(&compute_path);
    result
}

fn run_compute_pipeline_probe_child(compute_path: &Path) -> Result<(), String> {
    let exe =
        std::env::current_exe().map_err(|error| format!("locate compute probe exe: {error}"))?;
    let mut cmd = std::process::Command::new(exe);
    cmd.arg("--compute-pipeline-probe")
        .arg(compute_path)
        .env("RUST_BACKTRACE", "0")
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped());
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        cmd.process_group(0);
    }

    let mut child = cmd
        .spawn()
        .map_err(|error| format!("spawn compute pipeline probe: {error}"))?;
    let start = std::time::Instant::now();
    let timeout = std::time::Duration::from_secs(15);
    loop {
        match child.try_wait() {
            Ok(Some(status)) if status.success() => return Ok(()),
            Ok(Some(status)) => {
                #[cfg(unix)]
                {
                    use std::os::unix::process::ExitStatusExt;
                    if status.signal() == Some(11) {
                        return Err(
                            "Vulkan validation runner skipped NVIDIA compute pipeline compiler \
                             crash"
                                .into(),
                        );
                    }
                }
                let stderr = read_child_stderr(&mut child);
                return Err(format_probe_failure(
                    "compute pipeline probe",
                    status,
                    &stderr,
                ));
            }
            Ok(None) if start.elapsed() < timeout => {
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
            Ok(None) => {
                kill_probe_worker(&mut child);
                let _ = child.wait();
                return Err("compute pipeline probe timed out after 15s".into());
            }
            Err(error) => {
                kill_probe_worker(&mut child);
                let _ = child.wait();
                return Err(format!("wait compute pipeline probe: {error}"));
            }
        }
    }
}

fn pipeline_probe_subcommands_available() -> bool {
    std::env::current_exe()
        .ok()
        .and_then(|path| path.file_name().map(|name| name.to_owned()))
        .and_then(|name| name.into_string().ok())
        .is_some_and(|name| {
            name.starts_with("corpus-run-vulkan") || name.starts_with("corpus-run-moltenvk")
        })
}

fn read_child_stderr(child: &mut std::process::Child) -> String {
    let mut stderr = String::new();
    if let Some(mut pipe) = child.stderr.take() {
        let _ = pipe.read_to_string(&mut stderr);
    }
    stderr
}

fn format_probe_failure(label: &str, status: std::process::ExitStatus, stderr: &str) -> String {
    let stderr = stderr.trim();
    if stderr.is_empty() {
        format!("{label} failed: {status}")
    } else {
        format!("{label} failed: {status}: {stderr}")
    }
}

fn catch_probe_unwind<F, R>(f: F) -> std::thread::Result<R>
where
    F: FnOnce() -> R + std::panic::UnwindSafe,
{
    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let result = std::panic::catch_unwind(f);
    std::panic::set_hook(hook);
    result
}

fn kill_probe_worker(child: &mut std::process::Child) {
    #[cfg(unix)]
    {
        let pid = child.id() as i32;
        // SAFETY: the probe is spawned into its own process group above.
        unsafe {
            libc::kill(-pid, libc::SIGKILL);
        }
    }
    let _ = child.kill();
}

pub fn run_graphics_pipeline_probe_args<I>(args: I) -> i32
where
    I: IntoIterator<Item = std::ffi::OsString>,
{
    let args = args.into_iter().collect::<Vec<_>>();
    if args.len() != 8 {
        eprintln!(
            "usage: --graphics-pipeline-probe <vertex.spv> <fragment.spv> <format> \
             <width> <height> <color-attachments> <depth-output:0|1> <blend>"
        );
        return 2;
    }
    let run = || -> Result<(), String> {
        let vertex_path = PathBuf::from(&args[0]);
        let fragment_path = PathBuf::from(&args[1]);
        let vertex_spv = fs::read(&vertex_path)
            .map_err(|error| format!("read {}: {error}", vertex_path.display()))?;
        let fragment_spv = fs::read(&fragment_path)
            .map_err(|error| format!("read {}: {error}", fragment_path.display()))?;
        let format = args[2]
            .to_str()
            .and_then(parse_probe_format)
            .ok_or_else(|| "unknown graphics probe format".to_string())?;
        let width = parse_probe_u32(&args[3], "width")?;
        let height = parse_probe_u32(&args[4], "height")?;
        let color_attachment_formats =
            parse_probe_color_attachment_formats(&args[5], "color-attachments")?;
        let depth_output = parse_probe_bool(&args[6], "depth-output")?;
        let blend = args[7]
            .to_str()
            .and_then(parse_probe_blend)
            .ok_or_else(|| "unknown graphics probe blend mode".to_string())?;
        let (device, _) = device_and_queue_result(QueueFlags::GRAPHICS, true)?;
        let target = RenderPipelineTarget {
            format,
            extent: Extent3d::new(width, height, 1),
            color_attachment_formats,
            depth_output,
        };
        let result = catch_probe_unwind(std::panic::AssertUnwindSafe(|| {
            let _pipeline = graphics_pipeline(device, &vertex_spv, &fragment_spv, target, blend);
        }));
        result.map_err(crate::corpus_run::panic_payload_message)?;
        Ok(())
    };

    match run() {
        Ok(()) => 0,
        Err(error) => {
            eprintln!("graphics-pipeline-probe: {error}");
            2
        }
    }
}

pub fn run_vertex_pipeline_probe_args<I>(args: I) -> i32
where
    I: IntoIterator<Item = std::ffi::OsString>,
{
    let args = args.into_iter().collect::<Vec<_>>();
    if args.len() != 2 {
        eprintln!("usage: --vertex-pipeline-probe <vertex.spv> <sanitized.ll>");
        return 2;
    }
    let run = || -> Result<(), String> {
        let vertex_path = PathBuf::from(&args[0]);
        let ll_path = PathBuf::from(&args[1]);
        let vertex_spv = fs::read(&vertex_path)
            .map_err(|error| format!("read {}: {error}", vertex_path.display()))?;
        let sanitized_ll = fs::read_to_string(&ll_path)
            .map_err(|error| format!("read {}: {error}", ll_path.display()))?;
        let (device, _) = device_and_queue_result(QueueFlags::GRAPHICS, true)?;
        let vertex_inputs = vertex_inputs(&sanitized_ll);
        let result = catch_probe_unwind(std::panic::AssertUnwindSafe(|| {
            vertex_pipeline(device, &vertex_spv, &vertex_inputs).map(|_| ())
        }));
        result.map_err(crate::corpus_run::panic_payload_message)??;
        Ok(())
    };

    match run() {
        Ok(()) => 0,
        Err(error) => {
            eprintln!("vertex-pipeline-probe: {error}");
            2
        }
    }
}

pub fn run_compute_pipeline_probe_args<I>(args: I) -> i32
where
    I: IntoIterator<Item = std::ffi::OsString>,
{
    let args = args.into_iter().collect::<Vec<_>>();
    if args.len() != 1 {
        eprintln!("usage: --compute-pipeline-probe <compute.spv>");
        return 2;
    }
    let run = || -> Result<(), String> {
        let compute_path = PathBuf::from(&args[0]);
        let compute_spv = fs::read(&compute_path)
            .map_err(|error| format!("read {}: {error}", compute_path.display()))?;
        let (device, _) = device_and_queue_result(QueueFlags::COMPUTE, false)?;
        let result = catch_probe_unwind(std::panic::AssertUnwindSafe(|| {
            compute_pipeline(device, &compute_spv).map(|_| ())
        }));
        result.map_err(crate::corpus_run::panic_payload_message)??;
        Ok(())
    };

    match run() {
        Ok(()) => 0,
        Err(error) => {
            eprintln!("compute-pipeline-probe: {error}");
            2
        }
    }
}

fn parse_probe_format(value: &str) -> Option<DataFormat> {
    Some(match value {
        "RawBytes" => DataFormat::RawBytes,
        "U32" => DataFormat::U32,
        "I32" => DataFormat::I32,
        "F32" => DataFormat::F32,
        "Rgba8Unorm" => DataFormat::Rgba8Unorm,
        "Rgba8Uint" => DataFormat::Rgba8Uint,
        "Rgba8Sint" => DataFormat::Rgba8Sint,
        "R16Uint" => DataFormat::R16Uint,
        "Rg16Uint" => DataFormat::Rg16Uint,
        "Rgba16Uint" => DataFormat::Rgba16Uint,
        "R32Uint" => DataFormat::R32Uint,
        "Rg32Uint" => DataFormat::Rg32Uint,
        "Rgba32Uint" => DataFormat::Rgba32Uint,
        "R16Sint" => DataFormat::R16Sint,
        "Rg16Sint" => DataFormat::Rg16Sint,
        "Rgba16Sint" => DataFormat::Rgba16Sint,
        "R32Sint" => DataFormat::R32Sint,
        "Rg32Sint" => DataFormat::Rg32Sint,
        "Rgba32Sint" => DataFormat::Rgba32Sint,
        "R16Float" => DataFormat::R16Float,
        "Rg16Float" => DataFormat::Rg16Float,
        "Rgba16Float" => DataFormat::Rgba16Float,
        "Rg32Float" => DataFormat::Rg32Float,
        "Rgba32Float" => DataFormat::Rgba32Float,
        "R32Float" => DataFormat::R32Float,
        "Depth32Float" => DataFormat::Depth32Float,
        "Depth24Stencil8" => DataFormat::Depth24Stencil8,
        _ => return None,
    })
}

fn parse_probe_blend(value: &str) -> Option<BlendMode> {
    match value {
        "Replace" => Some(BlendMode::Replace),
        "SourceOver" => Some(BlendMode::SourceOver),
        _ => None,
    }
}

fn probe_color_attachment_formats_arg(formats: &[Option<DataFormat>]) -> String {
    if formats.is_empty() {
        return "-".to_string();
    }
    formats
        .iter()
        .map(|format| {
            format
                .map(|format| format!("{format:?}"))
                .unwrap_or_else(|| "none".to_string())
        })
        .collect::<Vec<_>>()
        .join(",")
}

fn parse_probe_color_attachment_formats(
    value: &std::ffi::OsStr,
    label: &str,
) -> Result<Vec<Option<DataFormat>>, String> {
    let value = value
        .to_str()
        .ok_or_else(|| format!("{label} is not UTF-8"))?;
    if value == "-" {
        return Ok(Vec::new());
    }
    value
        .split(',')
        .map(|part| {
            if part == "none" {
                Ok(None)
            } else {
                parse_probe_format(part)
                    .map(Some)
                    .ok_or_else(|| format!("{label}: unknown format {part}"))
            }
        })
        .collect()
}

fn parse_probe_bool(value: &std::ffi::OsStr, label: &str) -> Result<bool, String> {
    match value.to_str() {
        Some("0") => Ok(false),
        Some("1") => Ok(true),
        _ => Err(format!("{label} must be 0 or 1")),
    }
}

fn parse_probe_u32(value: &std::ffi::OsStr, label: &str) -> Result<u32, String> {
    value
        .to_str()
        .ok_or_else(|| format!("{label} is not UTF-8"))?
        .parse::<u32>()
        .map_err(|error| format!("{label}: {error}"))
}

fn vertex_pipeline(
    device: Arc<Device>,
    vertex_spv: &[u8],
    vertex_inputs: &[VertexInput],
) -> Result<Arc<GraphicsPipeline>, String> {
    let vertex_stage = shader_stage(device.clone(), vertex_spv);
    let layout_info = PipelineDescriptorSetLayoutCreateInfo::from_stages([&vertex_stage])
        .into_pipeline_layout_create_info(device.clone())
        .map_err(|error| format!("reflect vertex validation pipeline layout: {error}"))?;
    let layout = PipelineLayout::new(device.clone(), layout_info)
        .map_err(|error| format!("create vertex validation pipeline layout: {error}"))?;
    let create_info = vertex_validation_pipeline_create_info(layout, vertex_stage, vertex_inputs);
    if spirv_capabilities(vertex_spv).is_ok_and(|caps| caps.contains(&32)) {
        // Vulkano 0.35's shader-stage validation unwraps ClipDistance decorations as if the
        // decoration target were the array type rather than the output variable. The SPIR-V has
        // already passed spirv-val and Vulkan still validates pipeline creation below.
        unsafe { GraphicsPipeline::new_unchecked(device, None, create_info) }
            .map_err(|error| format!("create vertex validation pipeline: {error:?}"))
    } else {
        GraphicsPipeline::new(device, None, create_info)
            .map_err(|error| format!("create vertex validation pipeline: {error:?}"))
    }
}

fn vertex_validation_pipeline_create_info(
    layout: Arc<PipelineLayout>,
    vertex_stage: PipelineShaderStageCreateInfo,
    vertex_inputs: &[VertexInput],
) -> GraphicsPipelineCreateInfo {
    let mut create_info = GraphicsPipelineCreateInfo::layout(layout);
    create_info.stages = [vertex_stage].into_iter().collect();
    create_info.vertex_input_state = Some(vertex_input_state(vertex_inputs));
    create_info.input_assembly_state = Some(InputAssemblyState::default());
    create_info.viewport_state = vertex_validation_viewport_state();
    create_info.rasterization_state = Some(RasterizationState {
        rasterizer_discard_enable: true,
        ..RasterizationState::default()
    });
    create_info.depth_stencil_state = Some(DepthStencilState::default());
    create_info.subpass = Some(PipelineRenderingCreateInfo::default().into());
    create_info
}

fn vertex_validation_viewport_state() -> Option<ViewportState> {
    None
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

fn make_synthetic_storage_buffers(
    memory_allocator: Arc<StandardMemoryAllocator>,
    existing_buffers: &[(u32, Subbuffer<[u8]>)],
    sanitized_ll: &str,
) -> Vec<(u32, Subbuffer<[u8]>)> {
    synthetic_storage_buffer_sizes(sanitized_ll)
        .into_iter()
        .filter(|(index, _)| {
            !existing_buffers
                .iter()
                .any(|(existing_index, _)| existing_index == index)
        })
        .map(|(index, len)| {
            let buffer = Buffer::from_iter(
                memory_allocator.clone(),
                BufferCreateInfo {
                    usage: BufferUsage::STORAGE_BUFFER | BufferUsage::SHADER_DEVICE_ADDRESS,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                        | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                    ..Default::default()
                },
                vec![0u8; len.max(1)],
            )
            .unwrap_or_else(|e| panic!("create synthetic storage buffer {index}: {e}"));
            (index, buffer)
        })
        .collect()
}

fn make_compute_stage_input_buffers(
    memory_allocator: Arc<StandardMemoryAllocator>,
    existing_buffers: &[(u32, Subbuffer<[u8]>)],
    sanitized_ll: &str,
    inputs: &Inputs,
) -> Vec<(u32, Subbuffer<[u8]>)> {
    let bindings = compute_stage_input_binding_candidates(sanitized_ll);
    if bindings.is_empty() {
        return Vec::new();
    }
    let threads = inputs.dispatch.threads_per_grid[0].max(1) as usize;
    let mut buffers = Vec::new();
    for (param_idx, binding) in bindings {
        if existing_buffers
            .iter()
            .chain(buffers.iter())
            .any(|(existing_index, _)| *existing_index == binding)
        {
            continue;
        }
        let Some(type_name) = stage_input_type_name(sanitized_ll, param_idx) else {
            continue;
        };
        let Some((value_size, stride)) = compute_stage_input_size_stride(&type_name) else {
            panic!("vulkano runner does not support compute stage input type {type_name:?}");
        };
        let mut bytes = Vec::with_capacity(stride * threads);
        for thread in 0..threads {
            let start = bytes.len();
            append_compute_stage_input_value(&mut bytes, &type_name, thread);
            assert_eq!(
                bytes.len() - start,
                value_size,
                "compute stage input writer size mismatch for {type_name:?}"
            );
            bytes.resize(start + stride, 0);
        }
        let buffer = Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
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
        .unwrap_or_else(|e| panic!("create compute stage input storage buffer {binding}: {e}"));
        buffers.push((binding, buffer));
    }
    buffers
}

fn compute_stage_input_binding_candidates(sanitized_ll: &str) -> Vec<(u32, u32)> {
    let mut out = Vec::new();
    for promote_fc_buffers in [false, true] {
        let Some(meta) =
            metal2vulkan::meta::parse_air_kernel_meta_with(sanitized_ll, promote_fc_buffers)
        else {
            continue;
        };
        for binding in kernel_stage_input_bindings(&meta) {
            if !out.contains(&binding) {
                out.push(binding);
            }
        }
    }
    out
}

fn kernel_stage_input_bindings(meta: &metal2vulkan::meta::KernMeta) -> Vec<(u32, u32)> {
    let mut occupied = meta
        .roles
        .iter()
        .filter_map(|(_, role)| match role {
            metal2vulkan::meta::KernRole::Buffer(binding)
            | metal2vulkan::meta::KernRole::AccelerationStructureShadow(binding) => Some(*binding),
            _ => None,
        })
        .collect::<std::collections::HashSet<_>>();
    let mut next = 0u32;
    let mut out = Vec::new();
    for (idx, role) in &meta.roles {
        if !matches!(role, metal2vulkan::meta::KernRole::StageInput(_)) {
            continue;
        }
        while occupied.contains(&next) {
            next = next.saturating_add(1);
        }
        occupied.insert(next);
        out.push((*idx, next));
    }
    out
}

fn stage_input_type_name(sanitized_ll: &str, param_idx: u32) -> Option<String> {
    sanitized_ll.lines().find_map(|line| {
        (metadata_i32_at_start(line)? == param_idx && line.contains(r#""air.stage_in""#))
            .then(|| metadata_string_after(line, "air.arg_type_name"))
            .flatten()
    })
}

fn compute_stage_input_size_stride(type_name: &str) -> Option<(usize, usize)> {
    let (scalar_size, lanes) = scalar_size_and_lanes(type_name)?;
    let value_size = scalar_size * lanes;
    let stride_lanes = if lanes == 3 { 4 } else { lanes };
    Some((value_size, scalar_size * stride_lanes))
}

fn scalar_size_and_lanes(type_name: &str) -> Option<(usize, usize)> {
    let (base, size) = [
        ("float", 4),
        ("half", 2),
        ("int", 4),
        ("uint", 4),
        ("short", 2),
        ("ushort", 2),
        ("char", 1),
        ("uchar", 1),
    ]
    .into_iter()
    .find(|(base, _)| type_name.starts_with(base))?;
    let lanes = type_name
        .strip_prefix(base)
        .filter(|suffix| !suffix.is_empty())
        .and_then(|suffix| suffix.parse::<usize>().ok())
        .unwrap_or(1);
    (1..=4).contains(&lanes).then_some((size, lanes))
}

fn append_compute_stage_input_value(out: &mut Vec<u8>, type_name: &str, thread: usize) {
    let floats = vertex_float_values(thread);
    match type_name {
        "float" => push_f32s(out, &floats[0..1]),
        "float2" => push_f32s(out, &floats[0..2]),
        "float3" => push_f32s(out, &floats[0..3]),
        "float4" => push_f32s(out, &floats),
        "half" => push_half_zeros(out, 1),
        "half2" => push_half_zeros(out, 2),
        "half3" => push_half_zeros(out, 3),
        "half4" => push_half_zeros(out, 4),
        "int" => push_i32s(out, &[1 + thread as i32]),
        "int2" => push_i32s(out, &[1 + thread as i32, 2]),
        "int3" => push_i32s(out, &[1 + thread as i32, 2, 3]),
        "int4" => push_i32s(out, &[1 + thread as i32, 2, 3, 4]),
        "uint" => push_u32s(out, &[1 + thread as u32]),
        "uint2" => push_u32s(out, &[1 + thread as u32, 2]),
        "uint3" => push_u32s(out, &[1 + thread as u32, 2, 3]),
        "uint4" => push_u32s(out, &[1 + thread as u32, 2, 3, 4]),
        "short" => push_i16s(out, &[1 + thread as i16]),
        "short2" => push_i16s(out, &[1 + thread as i16, 2]),
        "short3" => push_i16s(out, &[1 + thread as i16, 2, 3]),
        "short4" => push_i16s(out, &[1 + thread as i16, 2, 3, 4]),
        "ushort" => push_u16s(out, &[1 + thread as u16]),
        "ushort2" => push_u16s(out, &[1 + thread as u16, 2]),
        "ushort3" => push_u16s(out, &[1 + thread as u16, 2, 3]),
        "ushort4" => push_u16s(out, &[1 + thread as u16, 2, 3, 4]),
        "char" => out.push(1 + thread as u8),
        "char2" => out.extend_from_slice(&[1 + thread as u8, 2]),
        "char3" => out.extend_from_slice(&[1 + thread as u8, 2, 3]),
        "char4" => out.extend_from_slice(&[1 + thread as u8, 2, 3, 4]),
        "uchar" => out.push(1 + thread as u8),
        "uchar2" => out.extend_from_slice(&[1 + thread as u8, 2]),
        "uchar3" => out.extend_from_slice(&[1 + thread as u8, 2, 3]),
        "uchar4" => out.extend_from_slice(&[1 + thread as u8, 2, 3, 4]),
        _ => panic!("unsupported compute stage input type {type_name:?}"),
    }
}

fn synthetic_storage_buffer_sizes(sanitized_ll: &str) -> Vec<(u32, usize)> {
    let mut sizes = HashMap::new();
    for line in sanitized_ll.lines() {
        let Some(index) = metadata_i32_after(line, "air.location_index") else {
            continue;
        };
        let size = if line.contains(r#""air.instance_acceleration_structure""#)
            || line.contains(r#""air.primitive_acceleration_structure""#)
        {
            metal2vulkan::as_shadow::CHILD_REFERENCES_BYTE_OFFSET as usize
                + 64 * metal2vulkan::as_shadow::CHILD_REFERENCE_BYTE_STRIDE as usize
        } else if line.contains(r#""air.indirect_buffer""#) {
            metadata_i32_after(line, "air.arg_type_size")
                .or_else(|| metadata_i32_after(line, "air.buffer_size"))
                .unwrap_or(8) as usize
        } else {
            continue;
        };
        sizes
            .entry(index)
            .and_modify(|existing: &mut usize| *existing = (*existing).max(size))
            .or_insert(size);
    }
    let mut sizes = sizes.into_iter().collect::<Vec<_>>();
    sizes.sort_by_key(|(index, _)| *index);
    sizes
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct VertexInput {
    location: u32,
    type_name: String,
}

fn vertex_inputs(sanitized_ll: &str) -> Vec<VertexInput> {
    let metadata_defs: Vec<_> = sanitized_ll
        .lines()
        .filter_map(|line| Some((metadata_definition_id(line)?, line)))
        .collect();
    let mut inputs = Vec::new();

    for line in sanitized_ll.lines() {
        if line.contains(r#""air.vertex_input""#) || line.contains(r#""air.stage_in""#) {
            push_vertex_input_metadata(&mut inputs, line);
        }
        if line.contains(r#""air.patch_control_point_input""#) {
            for ref_id in metadata_refs(line) {
                if let Some((_, field_line)) =
                    metadata_defs.iter().find(|(def_id, _)| *def_id == ref_id)
                {
                    push_vertex_input_metadata(&mut inputs, field_line);
                }
            }
        }
    }

    inputs
}

fn push_vertex_input_metadata(inputs: &mut Vec<VertexInput>, line: &str) {
    let Some(location) = metadata_i32_after(line, "air.location_index") else {
        return;
    };
    let Some(type_name) = metadata_string_after(line, "air.arg_type_name") else {
        return;
    };
    inputs.push(VertexInput {
        location,
        type_name,
    });
}

fn metadata_definition_id(line: &str) -> Option<u32> {
    let rest = line.trim_start().strip_prefix('!')?;
    let digits_len = rest
        .bytes()
        .take_while(|byte| byte.is_ascii_digit())
        .count();
    if digits_len == 0 || !rest[digits_len..].starts_with(" = !{") {
        return None;
    }
    rest[..digits_len].parse().ok()
}

fn metadata_refs(line: &str) -> Vec<u32> {
    let Some(body) = line.split_once("!{").map(|(_, body)| body) else {
        return Vec::new();
    };
    let bytes = body.as_bytes();
    let mut refs = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] != b'!' {
            i += 1;
            continue;
        }
        i += 1;
        let start = i;
        while i < bytes.len() && bytes[i].is_ascii_digit() {
            i += 1;
        }
        if i > start {
            if let Ok(value) = body[start..i].parse() {
                refs.push(value);
            }
        }
    }
    refs
}

fn vertex_input_state(inputs: &[VertexInput]) -> VertexInputState {
    if inputs.is_empty() {
        return VertexInputState::new();
    }

    let stride = attribute_stride(inputs, "vertex input") as u32;
    let mut state = VertexInputState::new().binding(
        0,
        VertexInputBindingDescription {
            stride: stride.max(1),
            ..Default::default()
        },
    );
    let mut offset = 0u32;
    for input in inputs {
        let (format, size) = vertex_format_and_size(&input.type_name).unwrap_or_else(|| {
            panic!(
                "vulkano runner does not support vertex input type {:?}",
                input.type_name
            )
        });
        state = state.attribute(
            input.location,
            VertexInputAttributeDescription {
                binding: 0,
                format,
                offset,
                ..Default::default()
            },
        );
        offset += size as u32;
    }
    state
}

fn make_vertex_input_buffer(
    memory_allocator: Arc<StandardMemoryAllocator>,
    inputs: &[VertexInput],
) -> Option<Subbuffer<[u8]>> {
    if inputs.is_empty() {
        return None;
    }

    let stride = attribute_stride(inputs, "vertex input");
    let mut bytes = Vec::with_capacity(stride * 3);
    for vertex in 0..3 {
        for input in inputs {
            append_vertex_attribute_value(&mut bytes, &input.type_name, vertex);
        }
    }
    Some(
        Buffer::from_iter(
            memory_allocator,
            BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                ..Default::default()
            },
            bytes,
        )
        .expect("create vertex validation vertex input buffer"),
    )
}

fn attribute_stride(inputs: &[VertexInput], label: &str) -> usize {
    inputs
        .iter()
        .map(|input| {
            vertex_format_and_size(&input.type_name)
                .unwrap_or_else(|| panic!("unsupported {label} type {:?}", input.type_name))
                .1
        })
        .sum()
}

fn vertex_format_and_size(type_name: &str) -> Option<(Format, usize)> {
    Some(match type_name {
        "float" => (Format::R32_SFLOAT, 4),
        "float2" => (Format::R32G32_SFLOAT, 8),
        "float3" | "float4" => (Format::R32G32B32A32_SFLOAT, 16),
        "half" => (Format::R16_SFLOAT, 2),
        "half2" => (Format::R16G16_SFLOAT, 4),
        "half3" | "half4" => (Format::R16G16B16A16_SFLOAT, 8),
        "int" => (Format::R32_SINT, 4),
        "int2" => (Format::R32G32_SINT, 8),
        "int3" | "int4" => (Format::R32G32B32A32_SINT, 16),
        "uint" => (Format::R32_UINT, 4),
        "uint2" => (Format::R32G32_UINT, 8),
        "uint3" | "uint4" => (Format::R32G32B32A32_UINT, 16),
        "short" => (Format::R16_SINT, 2),
        "short2" => (Format::R16G16_SINT, 4),
        "short3" | "short4" => (Format::R16G16B16A16_SINT, 8),
        "ushort" => (Format::R16_UINT, 2),
        "ushort2" => (Format::R16G16_UINT, 4),
        "ushort3" | "ushort4" => (Format::R16G16B16A16_UINT, 8),
        "char" => (Format::R8_SINT, 1),
        "char2" => (Format::R8G8_SINT, 2),
        "char3" | "char4" => (Format::R8G8B8A8_SINT, 4),
        "uchar" => (Format::R8_UINT, 1),
        "uchar2" => (Format::R8G8_UINT, 2),
        "uchar3" | "uchar4" => (Format::R8G8B8A8_UINT, 4),
        _ => return None,
    })
}

fn append_vertex_attribute_value(out: &mut Vec<u8>, type_name: &str, vertex: usize) {
    let floats = vertex_float_values(vertex);
    match type_name {
        "float" => push_f32s(out, &floats[0..1]),
        "float2" => push_f32s(out, &floats[0..2]),
        "float3" => push_f32s(out, &floats),
        "float4" => push_f32s(out, &floats),
        "half" => push_half_zeros(out, 1),
        "half2" => push_half_zeros(out, 2),
        "half3" => push_half_zeros(out, 4),
        "half4" => push_half_zeros(out, 4),
        "int" => push_i32s(out, &[1 + vertex as i32]),
        "int2" => push_i32s(out, &[1 + vertex as i32, 2]),
        "int3" => push_i32s(out, &[1 + vertex as i32, 2, 3, 4]),
        "int4" => push_i32s(out, &[1 + vertex as i32, 2, 3, 4]),
        "uint" => push_u32s(out, &[1 + vertex as u32]),
        "uint2" => push_u32s(out, &[1 + vertex as u32, 2]),
        "uint3" => push_u32s(out, &[1 + vertex as u32, 2, 3, 4]),
        "uint4" => push_u32s(out, &[1 + vertex as u32, 2, 3, 4]),
        "short" => push_i16s(out, &[1 + vertex as i16]),
        "short2" => push_i16s(out, &[1 + vertex as i16, 2]),
        "short3" => push_i16s(out, &[1 + vertex as i16, 2, 3, 4]),
        "short4" => push_i16s(out, &[1 + vertex as i16, 2, 3, 4]),
        "ushort" => push_u16s(out, &[1 + vertex as u16]),
        "ushort2" => push_u16s(out, &[1 + vertex as u16, 2]),
        "ushort3" => push_u16s(out, &[1 + vertex as u16, 2, 3, 4]),
        "ushort4" => push_u16s(out, &[1 + vertex as u16, 2, 3, 4]),
        "char" => out.push(1 + vertex as u8),
        "char2" => out.extend_from_slice(&[1 + vertex as u8, 2]),
        "char3" => out.extend_from_slice(&[1 + vertex as u8, 2, 3, 4]),
        "char4" => out.extend_from_slice(&[1 + vertex as u8, 2, 3, 4]),
        "uchar" => out.push(1 + vertex as u8),
        "uchar2" => out.extend_from_slice(&[1 + vertex as u8, 2]),
        "uchar3" => out.extend_from_slice(&[1 + vertex as u8, 2, 3, 4]),
        "uchar4" => out.extend_from_slice(&[1 + vertex as u8, 2, 3, 4]),
        _ => panic!("unsupported vertex input type {type_name:?}"),
    }
}

fn vertex_float_values(vertex: usize) -> [f32; 4] {
    match vertex {
        0 => [-1.0, -1.0, 0.0, 1.0],
        1 => [3.0, -1.0, 0.0, 1.0],
        _ => [-1.0, 3.0, 0.0, 1.0],
    }
}

fn push_f32s(out: &mut Vec<u8>, values: &[f32]) {
    for value in values {
        out.extend_from_slice(&value.to_le_bytes());
    }
}

fn push_i32s(out: &mut Vec<u8>, values: &[i32]) {
    for value in values {
        out.extend_from_slice(&value.to_le_bytes());
    }
}

fn push_u32s(out: &mut Vec<u8>, values: &[u32]) {
    for value in values {
        out.extend_from_slice(&value.to_le_bytes());
    }
}

fn push_i16s(out: &mut Vec<u8>, values: &[i16]) {
    for value in values {
        out.extend_from_slice(&value.to_le_bytes());
    }
}

fn push_u16s(out: &mut Vec<u8>, values: &[u16]) {
    for value in values {
        out.extend_from_slice(&value.to_le_bytes());
    }
}

fn push_half_zeros(out: &mut Vec<u8>, components: usize) {
    out.extend(std::iter::repeat_n(0, components * 2));
}

fn metadata_i32_after(body: &str, marker: &str) -> Option<u32> {
    let marker = format!("!\"{marker}\"");
    let tail = body.get(body.find(&marker)? + marker.len()..)?;
    let mut tokens = tail.split_whitespace().peekable();
    while let Some(token) = tokens.next() {
        if token == "i32" {
            let value = tokens.peek()?.trim_end_matches(',');
            if let Ok(parsed) = value.parse() {
                return Some(parsed);
            }
        }
    }
    None
}

fn metadata_i32_at_start(body: &str) -> Option<u32> {
    let tail = body.get(body.find("!{i32")? + "!{i32".len()..)?;
    let value = tail.trim_start().split(',').next()?.trim();
    value.parse().ok()
}

fn metadata_string_after(line: &str, marker: &str) -> Option<String> {
    let marker = format!("!\"{marker}\", !\"");
    let tail = line.get(line.find(&marker)? + marker.len()..)?;
    let end = tail.find('"')?;
    Some(tail[..end].to_string())
}

struct TextureResource {
    index: u32,
    kind: TextureKind,
    is_output: bool,
    image: Arc<Image>,
    view: Arc<ImageView>,
    staging: Subbuffer<[u8]>,
    texel_buffer: Option<Subbuffer<[u8]>>,
    texel_view: Option<Arc<BufferView>>,
}

struct ColorInputAttachment {
    index: u32,
    image: Arc<Image>,
    view: Arc<ImageView>,
    staging: Subbuffer<[u8]>,
}

fn make_color_input_attachments(
    memory_allocator: Arc<StandardMemoryAllocator>,
    layout: &PipelineLayout,
    fallback_format: DataFormat,
    extent: Extent3d,
    sanitized_ll: &str,
) -> Vec<ColorInputAttachment> {
    let Some(set_layout) = layout.set_layouts().first() else {
        return Vec::new();
    };
    let mut bindings = set_layout
        .bindings()
        .iter()
        .filter_map(|(binding, info)| {
            (info.descriptor_type == DescriptorType::InputAttachment).then_some(*binding)
        })
        .collect::<Vec<_>>();
    bindings.sort_unstable();

    bindings
        .into_iter()
        .map(|binding| {
            let index = binding
                .checked_sub(COLOR_INPUT_BINDING_BASE)
                .unwrap_or_else(|| {
                    panic!("color input binding {binding} is below color input base")
                });
            let format =
                color_input_attachment_format(sanitized_ll, index).unwrap_or(fallback_format);
            let seed = seeded_render_target_bytes(format, extent);
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
                seed.iter().copied(),
            )
            .unwrap_or_else(|e| panic!("create color input staging buffer {index}: {e}"));
            let image = Image::new(
                memory_allocator.clone(),
                ImageCreateInfo {
                    image_type: ImageType::Dim2d,
                    format: vulkan_format(format),
                    extent: [extent.width, extent.height, 1],
                    usage: ImageUsage::INPUT_ATTACHMENT
                        | ImageUsage::TRANSFER_DST
                        | ImageUsage::TRANSFER_SRC,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                    ..Default::default()
                },
            )
            .unwrap_or_else(|e| panic!("create color input image {index}: {e}"));
            let view = ImageView::new_default(image.clone())
                .unwrap_or_else(|e| panic!("create color input view {index}: {e}"));
            ColorInputAttachment {
                index,
                image,
                view,
                staging,
            }
        })
        .collect()
}

fn color_input_attachment_format(sanitized_ll: &str, index: u32) -> Option<DataFormat> {
    let meta = metal2vulkan::meta::parse_air_fragment_meta(sanitized_ll)?;
    meta.color_input_type_name(index)
        .and_then(fragment_color_format_from_air_type)
}

fn fragment_color_format_from_air_type(type_name: &str) -> Option<DataFormat> {
    Some(match type_name {
        "half" => DataFormat::R16Float,
        "half2" => DataFormat::Rg16Float,
        "half3" | "half4" => DataFormat::Rgba16Float,
        "float" => DataFormat::R32Float,
        "float2" => DataFormat::Rg32Float,
        "float3" | "float4" => DataFormat::Rgba32Float,
        "ushort" => DataFormat::R16Uint,
        "ushort2" => DataFormat::Rg16Uint,
        "ushort3" | "ushort4" => DataFormat::Rgba16Uint,
        "short" => DataFormat::R16Sint,
        "short2" => DataFormat::Rg16Sint,
        "short3" | "short4" => DataFormat::Rgba16Sint,
        "uint" => DataFormat::R32Uint,
        "uint2" => DataFormat::Rg32Uint,
        "uint3" | "uint4" => DataFormat::Rgba32Uint,
        "int" => DataFormat::R32Sint,
        "int2" => DataFormat::Rg32Sint,
        "int3" | "int4" => DataFormat::Rgba32Sint,
        _ => return None,
    })
}

fn make_textures(
    memory_allocator: Arc<StandardMemoryAllocator>,
    inputs: &Inputs,
    sanitized_ll: &str,
    texture_reqs: &HashMap<u32, TextureBindingReq>,
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
            let required = texture_binding_req_for_index(texture_reqs, input.index);
            let reflected_usage = texture_binding_image_usage_for_index(texture_reqs, input.index);
            let image_format = reflected_texture_format(input.format, required);
            if let Some(required) = required {
                if let Some(view_type) = required.image_view_type {
                    if shape.view_type != Some(view_type) {
                        shape.view_type = Some(view_type);
                    }
                }
            }
            let seed_input = TextureInput {
                format: image_format,
                ..*input
            };
            let bytes = texture_seed_bytes(&seed_input, kind, shape.seed_extent);
            assert!(
                !bytes.is_empty(),
                "vulkano runner does not support zero-length textures"
            );
            let texel_resource = texture_buffer_resource(sanitized_ll, input.index).then(|| {
                make_texture_buffer_resource(
                    memory_allocator.clone(),
                    input.index,
                    input.role,
                    image_format,
                    &bytes,
                )
            });
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
                    flags: shape.flags
                        | texture_binding_mutable_format_flag(
                            texture_reqs,
                            input.index,
                            image_format,
                        ),
                    image_type: shape.image_type,
                    format: vulkan_format(image_format),
                    extent: shape.extent,
                    array_layers: shape.array_layers,
                    usage: vulkan_image_usage(input.role) | reflected_usage,
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
                texel_buffer: texel_resource
                    .as_ref()
                    .map(|(texel_buffer, _)| texel_buffer.clone()),
                texel_view: texel_resource.map(|(_, texel_view)| texel_view),
            }
        })
        .collect()
}

fn texture_buffer_resource(sanitized_ll: &str, texture_location: u32) -> bool {
    if let Some(meta) = metal2vulkan::meta::parse_air_kernel_meta(sanitized_ll) {
        for (param_idx, role) in &meta.roles {
            if matches!(role, metal2vulkan::meta::KernRole::Texture(location) if *location == texture_location)
                && meta
                    .texture_type_name(*param_idx)
                    .is_some_and(|name| name.starts_with("texture_buffer<"))
            {
                return true;
            }
        }
    }
    if let Some(meta) = metal2vulkan::meta::parse_air_fragment_meta(sanitized_ll) {
        for (param_idx, role) in &meta.roles {
            if matches!(role, metal2vulkan::meta::FragRole::Texture(location) if *location == texture_location)
                && meta
                    .texture_type_name(*param_idx)
                    .is_some_and(|name| name.starts_with("texture_buffer<"))
            {
                return true;
            }
        }
    }
    if let Some(meta) = metal2vulkan::meta::parse_air_vertex_meta(sanitized_ll) {
        for (param_idx, role) in &meta.roles {
            if matches!(role, metal2vulkan::meta::VertRole::Texture(location) if *location == texture_location)
                && meta
                    .texture_type_name(*param_idx)
                    .is_some_and(|name| name.starts_with("texture_buffer<"))
            {
                return true;
            }
        }
    }
    false
}

fn make_texture_buffer_resource(
    memory_allocator: Arc<StandardMemoryAllocator>,
    index: u32,
    role: TextureRole,
    format: DataFormat,
    bytes: &[u8],
) -> (Subbuffer<[u8]>, Arc<BufferView>) {
    let buffer = Buffer::from_iter(
        memory_allocator,
        BufferCreateInfo {
            usage: texel_buffer_usage(role),
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_RANDOM_ACCESS,
            ..Default::default()
        },
        bytes.iter().copied(),
    )
    .unwrap_or_else(|e| panic!("create texture buffer {index}: {e}"));
    let view = BufferView::new(
        buffer.clone(),
        BufferViewCreateInfo {
            format: vulkan_format(format),
            ..Default::default()
        },
    )
    .unwrap_or_else(|e| panic!("create texture buffer view {index}: {e}"));
    (buffer, view)
}

fn texel_buffer_usage(role: TextureRole) -> BufferUsage {
    match role {
        TextureRole::StorageWrite | TextureRole::StorageReadWrite => {
            BufferUsage::STORAGE_TEXEL_BUFFER
        }
        _ => BufferUsage::UNIFORM_TEXEL_BUFFER,
    }
}

/// A set-0 texture binding the translated SPIR-V declares, reflected structurally from the module
/// (never keyed on any shader/case name). The descriptor-type booleans carry the image usage the
/// allocation must support; the format/scalar-type/view-type carry vulkano's recovered `OpTypeImage`
/// constraints, used to synthesize a placeholder image that Vulkan accepts for that binding.
#[derive(Clone, Copy, Debug)]
struct TextureBindingReq {
    index: u32,
    descriptor_count: u32,
    needs_sampled: bool,
    needs_storage: bool,
    image_format: Option<Format>,
    image_scalar_type: Option<NumericType>,
    image_view_type: Option<ImageViewType>,
}

#[derive(Clone, Copy, Debug)]
struct SpirvImageBindingReq {
    descriptor_count: u32,
    image_view_type: Option<ImageViewType>,
    image_scalar_type: Option<NumericType>,
    image_scalar_type_conflict: bool,
    needs_sampled: bool,
    needs_storage: bool,
}

fn texture_binding_image_usage(req: Option<&TextureBindingReq>) -> ImageUsage {
    let Some(req) = req else {
        return ImageUsage::empty();
    };
    let mut usage = ImageUsage::empty();
    if req.needs_sampled {
        usage |= ImageUsage::SAMPLED;
    }
    if req.needs_storage {
        usage |= ImageUsage::STORAGE;
    }
    usage
}

fn texture_binding_image_usage_for_index(
    reqs: &HashMap<u32, TextureBindingReq>,
    index: u32,
) -> ImageUsage {
    reqs.values()
        .filter(|req| {
            index >= req.index && index < req.index.saturating_add(req.descriptor_count.max(1))
        })
        .fold(ImageUsage::empty(), |usage, req| {
            usage | texture_binding_image_usage(Some(req))
        })
}

fn texture_binding_req_for_index(
    reqs: &HashMap<u32, TextureBindingReq>,
    index: u32,
) -> Option<&TextureBindingReq> {
    reqs.get(&index).or_else(|| {
        reqs.values()
            .filter(|req| {
                index >= req.index && index < req.index.saturating_add(req.descriptor_count.max(1))
            })
            .min_by_key(|req| req.index)
    })
}

fn texture_binding_mutable_format_flag(
    reqs: &HashMap<u32, TextureBindingReq>,
    index: u32,
    image_format: DataFormat,
) -> ImageCreateFlags {
    let image_format_vk = vulkan_format(image_format);
    let needs_mutable = reqs.values().any(|req| {
        index >= req.index
            && index < req.index.saturating_add(req.descriptor_count.max(1))
            && vulkan_format(reflected_texture_format(image_format, Some(req))) != image_format_vk
    });
    if needs_mutable {
        ImageCreateFlags::MUTABLE_FORMAT
    } else {
        ImageCreateFlags::empty()
    }
}

fn reflected_texture_format(
    input_format: DataFormat,
    req: Option<&TextureBindingReq>,
) -> DataFormat {
    let Some(req) = req else {
        return input_format;
    };
    if let Some(format) = req.image_format.and_then(data_format_for_vulkan_format) {
        return format;
    }
    let Some(target_type) = req.image_scalar_type else {
        return input_format;
    };
    if data_format_numeric_type(input_format) == Some(target_type) {
        return input_format;
    }
    data_format_with_numeric_type(input_format, target_type).unwrap_or(input_format)
}

fn merge_spirv_image_binding_req(req: &mut TextureBindingReq, spirv_req: SpirvImageBindingReq) {
    req.descriptor_count = req.descriptor_count.max(spirv_req.descriptor_count);
    req.needs_sampled |= spirv_req.needs_sampled;
    req.needs_storage |= spirv_req.needs_storage;
    if req.image_scalar_type.is_none() {
        req.image_scalar_type = spirv_req.image_scalar_type;
    }
    match spirv_req.image_view_type {
        Some(view_type) => {
            req.image_view_type.get_or_insert(view_type);
        }
        None => {
            req.image_view_type = None;
        }
    }
}

/// Reflect every set-0 sampled/storage-image binding the SPIR-V declares (bindings in
/// `[TEXTURE_BINDING_BASE, SAMPLER_BINDING_BASE)`), keyed by the runner's texture index
/// (`binding - TEXTURE_BINDING_BASE`). Mirrors `required_texture_view_types` but also recovers the
/// descriptor type, image format, and scalar type so a missing binding can be filled with a
/// correctly-typed placeholder.
fn required_texture_bindings(device: Arc<Device>, spv: &[u8]) -> HashMap<u32, TextureBindingReq> {
    let Ok(words) = bytes_to_words(spv) else {
        return HashMap::new();
    };
    let spirv_reqs = texture_image_binding_reqs_from_words(&words);
    let Ok(module) = (unsafe { ShaderModule::new(device, ShaderModuleCreateInfo::new(&words)) })
    else {
        return spirv_reqs
            .into_iter()
            .map(|(index, req)| {
                (
                    index,
                    TextureBindingReq {
                        index,
                        descriptor_count: req.descriptor_count,
                        needs_sampled: req.needs_sampled,
                        needs_storage: req.needs_storage,
                        image_format: None,
                        image_scalar_type: req.image_scalar_type,
                        image_view_type: req.image_view_type,
                    },
                )
            })
            .collect();
    };
    let Some(entry) = module.entry_point("main") else {
        return spirv_reqs
            .into_iter()
            .map(|(index, req)| {
                (
                    index,
                    TextureBindingReq {
                        index,
                        descriptor_count: req.descriptor_count,
                        needs_sampled: req.needs_sampled,
                        needs_storage: req.needs_storage,
                        image_format: None,
                        image_scalar_type: req.image_scalar_type,
                        image_view_type: req.image_view_type,
                    },
                )
            })
            .collect();
    };
    let mut out = entry
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
            let index = *binding - TEXTURE_BINDING_BASE;
            Some((
                index,
                TextureBindingReq {
                    index,
                    descriptor_count: reqs.descriptor_count.unwrap_or(1),
                    needs_sampled: is_sampled,
                    needs_storage: is_storage,
                    image_format: reqs.image_format,
                    image_scalar_type: reqs.image_scalar_type,
                    image_view_type: reqs.image_view_type,
                },
            ))
        })
        .collect::<HashMap<_, _>>();
    for (index, spirv_req) in spirv_reqs {
        out.entry(index)
            .and_modify(|req| merge_spirv_image_binding_req(req, spirv_req))
            .or_insert(TextureBindingReq {
                index,
                descriptor_count: spirv_req.descriptor_count,
                needs_sampled: spirv_req.needs_sampled,
                needs_storage: spirv_req.needs_storage,
                image_format: None,
                image_scalar_type: spirv_req.image_scalar_type,
                image_view_type: spirv_req.image_view_type,
            });
    }
    out
}

fn texture_image_binding_reqs_from_words(words: &[u32]) -> HashMap<u32, SpirvImageBindingReq> {
    const OP_DECORATE: u16 = 71;
    const OP_TYPE_FLOAT: u16 = 22;
    const OP_TYPE_IMAGE: u16 = 25;
    const OP_TYPE_INT: u16 = 21;
    const OP_CONSTANT: u16 = 43;
    const OP_TYPE_POINTER: u16 = 32;
    const OP_TYPE_ARRAY: u16 = 28;
    const OP_TYPE_RUNTIME_ARRAY: u16 = 29;
    const OP_VARIABLE: u16 = 59;
    const DECORATION_BINDING: u32 = 33;
    const DECORATION_DESCRIPTOR_SET: u32 = 34;
    const STORAGE_CLASS_UNIFORM_CONSTANT: u32 = 0;

    let mut bindings = HashMap::new();
    let mut descriptor_sets = HashMap::new();
    let mut image_types = HashMap::new();
    let mut scalar_types = HashMap::new();
    let mut constants = HashMap::new();
    let mut array_element_types = HashMap::new();
    let mut pointer_pointees = HashMap::new();
    let mut variables = Vec::new();

    let mut i = 5usize;
    while i < words.len() {
        let word = words[i];
        let opcode = (word & 0xffff) as u16;
        let word_count = (word >> 16) as usize;
        if word_count == 0 || i + word_count > words.len() {
            break;
        }
        let operands = &words[i + 1..i + word_count];
        match opcode {
            OP_DECORATE if operands.len() >= 3 => match operands[1] {
                DECORATION_BINDING => {
                    bindings.insert(operands[0], operands[2]);
                }
                DECORATION_DESCRIPTOR_SET => {
                    descriptor_sets.insert(operands[0], operands[2]);
                }
                _ => {}
            },
            OP_TYPE_INT if operands.len() >= 3 => {
                let numeric_type = match operands[2] {
                    0 => Some(NumericType::Uint),
                    1 => Some(NumericType::Int),
                    _ => None,
                };
                if let Some(numeric_type) = numeric_type {
                    scalar_types.insert(operands[0], numeric_type);
                }
            }
            OP_TYPE_FLOAT if operands.len() >= 2 => {
                scalar_types.insert(operands[0], NumericType::Float);
            }
            OP_TYPE_IMAGE if operands.len() >= 8 => {
                let result_id = operands[0];
                let sampled_type = operands[1];
                let dim = operands[2];
                let arrayed = operands[4] != 0;
                let sampled = operands[6];
                if let Some(view_type) = image_view_type_for_spirv_image(dim, arrayed) {
                    let (needs_sampled, needs_storage) = match sampled {
                        1 => (true, false),
                        2 => (false, true),
                        _ => (true, true),
                    };
                    image_types.insert(
                        result_id,
                        SpirvImageBindingReq {
                            descriptor_count: 1,
                            image_view_type: Some(view_type),
                            image_scalar_type: scalar_types.get(&sampled_type).copied(),
                            image_scalar_type_conflict: false,
                            needs_sampled,
                            needs_storage,
                        },
                    );
                }
            }
            OP_CONSTANT if operands.len() >= 3 => {
                constants.insert(operands[1], operands[2]);
            }
            OP_TYPE_ARRAY | OP_TYPE_RUNTIME_ARRAY if operands.len() >= 2 => {
                let descriptor_count = if opcode == OP_TYPE_ARRAY && operands.len() >= 3 {
                    constants.get(&operands[2]).copied().unwrap_or(1).max(1)
                } else {
                    1
                };
                array_element_types.insert(operands[0], (operands[1], descriptor_count));
            }
            OP_TYPE_POINTER if operands.len() >= 3 => {
                pointer_pointees.insert(operands[0], (operands[1], operands[2]));
            }
            OP_VARIABLE if operands.len() >= 3 => {
                variables.push((operands[0], operands[1], operands[2]));
            }
            _ => {}
        }
        i += word_count;
    }

    let mut out = HashMap::new();
    for (pointer_type, variable_id, storage_class) in variables {
        if storage_class != STORAGE_CLASS_UNIFORM_CONSTANT {
            continue;
        }
        if descriptor_sets.get(&variable_id).copied().unwrap_or(0) != DESCRIPTOR_SET {
            continue;
        }
        let Some(binding) = bindings.get(&variable_id).copied() else {
            continue;
        };
        if !(TEXTURE_BINDING_BASE..SAMPLER_BINDING_BASE).contains(&binding) {
            continue;
        }
        let Some((_, pointee)) = pointer_pointees.get(&pointer_type).copied() else {
            continue;
        };
        if let Some(req) = resolve_image_binding_req(pointee, &image_types, &array_element_types) {
            let index = binding - TEXTURE_BINDING_BASE;
            out.entry(index)
                .and_modify(|existing: &mut SpirvImageBindingReq| {
                    existing.needs_sampled |= req.needs_sampled;
                    existing.needs_storage |= req.needs_storage;
                    existing.descriptor_count = existing.descriptor_count.max(req.descriptor_count);
                    if existing.image_view_type != req.image_view_type {
                        existing.image_view_type = None;
                    }
                    match (existing.image_scalar_type, req.image_scalar_type) {
                        (Some(existing_type), Some(req_type)) if existing_type != req_type => {
                            existing.image_scalar_type = None;
                            existing.image_scalar_type_conflict = true;
                        }
                        (None, Some(req_type)) if !existing.image_scalar_type_conflict => {
                            existing.image_scalar_type = Some(req_type);
                        }
                        _ => {}
                    }
                })
                .or_insert(req);
        }
    }
    out
}

fn resolve_image_binding_req(
    ty: u32,
    image_types: &HashMap<u32, SpirvImageBindingReq>,
    array_element_types: &HashMap<u32, (u32, u32)>,
) -> Option<SpirvImageBindingReq> {
    if let Some(req) = image_types.get(&ty).copied() {
        return Some(req);
    }
    let (elem, descriptor_count) = array_element_types.get(&ty).copied()?;
    let mut req = resolve_image_binding_req(elem, image_types, array_element_types)?;
    req.descriptor_count = req.descriptor_count.saturating_mul(descriptor_count).max(1);
    Some(req)
}

fn image_view_type_for_spirv_image(dim: u32, arrayed: bool) -> Option<ImageViewType> {
    match (dim, arrayed) {
        (0, false) => Some(ImageViewType::Dim1d),
        (0, true) => Some(ImageViewType::Dim1dArray),
        (1, false) => Some(ImageViewType::Dim2d),
        (1, true) => Some(ImageViewType::Dim2dArray),
        (2, false) => Some(ImageViewType::Dim3d),
        (3, false) => Some(ImageViewType::Cube),
        (3, true) => Some(ImageViewType::CubeArray),
        _ => None,
    }
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
        ImageViewType::Dim1d => TextureKind::Dim1d,
        ImageViewType::Dim1dArray => TextureKind::Dim1dArray,
        ImageViewType::Dim3d => TextureKind::Dim3d,
        ImageViewType::Cube => TextureKind::Cube,
        ImageViewType::CubeArray => TextureKind::CubeArray,
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
    for req in required_texture_bindings(device, spv).into_values() {
        let view_type = req.image_view_type.unwrap_or(ImageViewType::Dim2d);
        let (image_type, array_layers, flags) = placeholder_image_shape(view_type);
        let format = placeholder_format(&req);
        let extent = [1u32, 1, 1];
        let usage = ImageUsage::TRANSFER_DST
            | ImageUsage::TRANSFER_SRC
            | texture_binding_image_usage(Some(&req));
        for offset in 0..req.descriptor_count {
            let index = req.index + offset;
            if textures.iter().any(|texture| texture.index == index) {
                continue;
            }
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
            .unwrap_or_else(|e| panic!("create placeholder texture staging {index}: {e}"));
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
            .unwrap_or_else(|e| panic!("create placeholder texture image {index}: {e}"));
            let mut view_info = ImageViewCreateInfo::from_image(&image);
            view_info.view_type = view_type;
            let view = ImageView::new(image.clone(), view_info)
                .unwrap_or_else(|e| panic!("create placeholder texture view {index}: {e}"));
            textures.push(TextureResource {
                index,
                kind: placeholder_kind(view_type),
                is_output: false,
                image,
                view,
                staging,
                texel_buffer: None,
                texel_view: None,
            });
        }
    }
}

fn texture_view_array_for_binding(
    textures: &[TextureResource],
    texture_index: u32,
    descriptor_count: u32,
    req: Option<&TextureBindingReq>,
) -> Vec<Arc<ImageView>> {
    (0..descriptor_count)
        .map(|offset| {
            let index = texture_index + offset;
            let texture = textures
                .iter()
                .find(|texture| texture.index == index)
                .unwrap_or_else(|| panic!("descriptor set expects texture binding {}", index));
            texture_view_for_req(texture, req)
        })
        .collect()
}

fn texture_view_for_req(
    texture: &TextureResource,
    req: Option<&TextureBindingReq>,
) -> Arc<ImageView> {
    let Some(view_type) = req.and_then(|req| req.image_view_type) else {
        return texture.view.clone();
    };
    if texture.view.view_type() == view_type
        && descriptor_view_format(texture.image.format(), req) == texture.image.format()
    {
        return texture.view.clone();
    }
    let mut view_info = ImageViewCreateInfo::from_image(&texture.image);
    view_info.view_type = view_type;
    view_info.format = descriptor_view_format(texture.image.format(), req);
    match view_type {
        ImageViewType::Dim1d | ImageViewType::Dim2d | ImageViewType::Dim3d => {
            view_info.subresource_range.array_layers = 0..1;
        }
        ImageViewType::Cube => {
            view_info.subresource_range.array_layers = 0..6;
        }
        ImageViewType::CubeArray => {
            let layers = texture.image.array_layers();
            view_info.subresource_range.array_layers = 0..((layers / 6).max(1) * 6);
        }
        _ => {}
    }
    ImageView::new(texture.image.clone(), view_info)
        .unwrap_or_else(|e| panic!("create descriptor texture view {}: {e}", texture.index))
}

fn descriptor_view_format(parent_format: Format, req: Option<&TextureBindingReq>) -> Format {
    let Some(parent_data_format) = data_format_for_vulkan_format(parent_format) else {
        return parent_format;
    };
    let descriptor_format = reflected_texture_format(parent_data_format, req);
    vulkan_format(descriptor_format)
}

fn storage_texture_view_array_for_binding(
    textures: &[TextureResource],
    texture_index: u32,
    descriptor_count: u32,
    req: Option<&TextureBindingReq>,
) -> Vec<DescriptorImageViewInfo> {
    texture_view_array_for_binding(textures, texture_index, descriptor_count, req)
        .into_iter()
        .map(|image_view| DescriptorImageViewInfo {
            image_view,
            image_layout: ImageLayout::General,
        })
        .collect()
}

fn is_output_texture(output: Output, index: u32) -> bool {
    matches!(output, Output::Texture { index: output_index, .. } if output_index == index)
}

fn make_texture_readback(
    memory_allocator: Arc<StandardMemoryAllocator>,
    output: Output,
    sanitized_ll: &str,
    texture_reqs: &HashMap<u32, TextureBindingReq>,
) -> Option<Subbuffer<[u8]>> {
    match output {
        Output::Texture {
            index,
            format,
            extent,
        } => {
            let kind = texture_kind(Some(sanitized_ll), index);
            let readback_format = reflected_texture_format(
                format,
                texture_binding_req_for_index(texture_reqs, index),
            );
            let len = texture_byte_len(readback_format, texture_output_extent(extent, kind));
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
            seed_extent: texture_seed_extent(extent, kind),
        },
        TextureKind::Dim1dArray => TextureShape {
            flags: ImageCreateFlags::empty(),
            image_type: ImageType::Dim1d,
            extent: [extent.width, 1, 1],
            array_layers: extent.depth.max(1),
            view_type: Some(ImageViewType::Dim1dArray),
            seed_extent: texture_seed_extent(extent, kind),
        },
        TextureKind::Dim2dArray => TextureShape {
            flags: ImageCreateFlags::empty(),
            image_type: ImageType::Dim2d,
            extent: [extent.width, extent.height, 1],
            array_layers: extent.depth.max(1),
            view_type: Some(ImageViewType::Dim2dArray),
            seed_extent: texture_seed_extent(extent, kind),
        },
        TextureKind::Dim3d => TextureShape {
            flags: ImageCreateFlags::empty(),
            image_type: ImageType::Dim3d,
            extent: [extent.width, extent.height, extent.depth.max(1)],
            array_layers: 1,
            view_type: Some(ImageViewType::Dim3d),
            seed_extent: texture_seed_extent(extent, kind),
        },
        TextureKind::Cube => TextureShape {
            flags: ImageCreateFlags::CUBE_COMPATIBLE,
            image_type: ImageType::Dim2d,
            extent: [extent.width, extent.height, 1],
            array_layers: 6,
            view_type: Some(ImageViewType::Cube),
            seed_extent: texture_seed_extent(extent, kind),
        },
        TextureKind::CubeArray => TextureShape {
            flags: ImageCreateFlags::CUBE_COMPATIBLE,
            image_type: ImageType::Dim2d,
            extent: [extent.width, extent.height, 1],
            array_layers: 6 * extent.depth.max(1),
            view_type: Some(ImageViewType::CubeArray),
            seed_extent: texture_seed_extent(extent, kind),
        },
        TextureKind::Plain => TextureShape {
            flags: ImageCreateFlags::empty(),
            image_type: vulkan_image_type(extent),
            extent: [extent.width, extent.height, extent.depth],
            array_layers: 1,
            view_type: None,
            seed_extent: texture_seed_extent(extent, kind),
        },
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

fn render_target_usage(format: DataFormat) -> ImageUsage {
    if is_depth_format(format) {
        ImageUsage::DEPTH_STENCIL_ATTACHMENT
    } else {
        ImageUsage::COLOR_ATTACHMENT
    }
}

fn is_depth_format(format: DataFormat) -> bool {
    matches!(format, DataFormat::Depth32Float)
}

fn vulkan_format(format: DataFormat) -> Format {
    match format {
        DataFormat::Rgba8Unorm => Format::R8G8B8A8_UNORM,
        DataFormat::Rgba8Uint => Format::R8G8B8A8_UINT,
        DataFormat::Rgba8Sint => Format::R8G8B8A8_SINT,
        DataFormat::R16Uint => Format::R16_UINT,
        DataFormat::Rg16Uint => Format::R16G16_UINT,
        DataFormat::Rgba16Uint => Format::R16G16B16A16_UINT,
        DataFormat::R32Uint => Format::R32_UINT,
        DataFormat::Rg32Uint => Format::R32G32_UINT,
        DataFormat::Rgba32Uint => Format::R32G32B32A32_UINT,
        DataFormat::R16Sint => Format::R16_SINT,
        DataFormat::Rg16Sint => Format::R16G16_SINT,
        DataFormat::Rgba16Sint => Format::R16G16B16A16_SINT,
        DataFormat::R32Sint => Format::R32_SINT,
        DataFormat::Rg32Sint => Format::R32G32_SINT,
        DataFormat::Rgba32Sint => Format::R32G32B32A32_SINT,
        DataFormat::R16Float => Format::R16_SFLOAT,
        DataFormat::Rg16Float => Format::R16G16_SFLOAT,
        DataFormat::Rgba16Float => Format::R16G16B16A16_SFLOAT,
        DataFormat::Rg32Float => Format::R32G32_SFLOAT,
        DataFormat::Rgba32Float => Format::R32G32B32A32_SFLOAT,
        DataFormat::R32Float => Format::R32_SFLOAT,
        DataFormat::Depth32Float => Format::D32_SFLOAT,
        _ => panic!("unsupported Vulkan texture format {format:?}"),
    }
}

fn data_format_for_vulkan_format(format: Format) -> Option<DataFormat> {
    match format {
        Format::R8G8B8A8_UNORM => Some(DataFormat::Rgba8Unorm),
        Format::R8G8B8A8_UINT => Some(DataFormat::Rgba8Uint),
        Format::R8G8B8A8_SINT => Some(DataFormat::Rgba8Sint),
        Format::R16_UINT => Some(DataFormat::R16Uint),
        Format::R16G16_UINT => Some(DataFormat::Rg16Uint),
        Format::R16G16B16A16_UINT => Some(DataFormat::Rgba16Uint),
        Format::R32_UINT => Some(DataFormat::R32Uint),
        Format::R32G32_UINT => Some(DataFormat::Rg32Uint),
        Format::R32G32B32A32_UINT => Some(DataFormat::Rgba32Uint),
        Format::R16_SINT => Some(DataFormat::R16Sint),
        Format::R16G16_SINT => Some(DataFormat::Rg16Sint),
        Format::R16G16B16A16_SINT => Some(DataFormat::Rgba16Sint),
        Format::R32_SINT => Some(DataFormat::R32Sint),
        Format::R32G32_SINT => Some(DataFormat::Rg32Sint),
        Format::R32G32B32A32_SINT => Some(DataFormat::Rgba32Sint),
        Format::R16_SFLOAT => Some(DataFormat::R16Float),
        Format::R16G16_SFLOAT => Some(DataFormat::Rg16Float),
        Format::R16G16B16A16_SFLOAT => Some(DataFormat::Rgba16Float),
        Format::R32G32_SFLOAT => Some(DataFormat::Rg32Float),
        Format::R32G32B32A32_SFLOAT => Some(DataFormat::Rgba32Float),
        Format::R32_SFLOAT => Some(DataFormat::R32Float),
        Format::D32_SFLOAT => Some(DataFormat::Depth32Float),
        _ => None,
    }
}

fn data_format_numeric_type(format: DataFormat) -> Option<NumericType> {
    match format {
        DataFormat::Rgba8Uint
        | DataFormat::R16Uint
        | DataFormat::Rg16Uint
        | DataFormat::Rgba16Uint
        | DataFormat::R32Uint
        | DataFormat::Rg32Uint
        | DataFormat::Rgba32Uint => Some(NumericType::Uint),
        DataFormat::Rgba8Sint
        | DataFormat::R16Sint
        | DataFormat::Rg16Sint
        | DataFormat::Rgba16Sint
        | DataFormat::R32Sint
        | DataFormat::Rg32Sint
        | DataFormat::Rgba32Sint => Some(NumericType::Int),
        DataFormat::Rgba8Unorm
        | DataFormat::R16Float
        | DataFormat::Rg16Float
        | DataFormat::Rgba16Float
        | DataFormat::Rg32Float
        | DataFormat::Rgba32Float
        | DataFormat::R32Float
        | DataFormat::Depth32Float => Some(NumericType::Float),
        _ => None,
    }
}

fn data_format_with_numeric_type(
    format: DataFormat,
    numeric_type: NumericType,
) -> Option<DataFormat> {
    match numeric_type {
        NumericType::Uint => match format {
            DataFormat::Rgba8Unorm | DataFormat::Rgba8Uint | DataFormat::Rgba8Sint => {
                Some(DataFormat::Rgba8Uint)
            }
            DataFormat::R16Uint | DataFormat::R16Sint | DataFormat::R16Float => {
                Some(DataFormat::R16Uint)
            }
            DataFormat::Rg16Uint | DataFormat::Rg16Sint | DataFormat::Rg16Float => {
                Some(DataFormat::Rg16Uint)
            }
            DataFormat::Rgba16Uint | DataFormat::Rgba16Sint | DataFormat::Rgba16Float => {
                Some(DataFormat::Rgba16Uint)
            }
            DataFormat::R32Uint | DataFormat::R32Sint | DataFormat::R32Float => {
                Some(DataFormat::R32Uint)
            }
            DataFormat::Rg32Uint | DataFormat::Rg32Sint | DataFormat::Rg32Float => {
                Some(DataFormat::Rg32Uint)
            }
            DataFormat::Rgba32Uint | DataFormat::Rgba32Sint | DataFormat::Rgba32Float => {
                Some(DataFormat::Rgba32Uint)
            }
            _ => None,
        },
        NumericType::Int => match format {
            DataFormat::Rgba8Unorm | DataFormat::Rgba8Uint | DataFormat::Rgba8Sint => {
                Some(DataFormat::Rgba8Sint)
            }
            DataFormat::R16Uint | DataFormat::R16Sint | DataFormat::R16Float => {
                Some(DataFormat::R16Sint)
            }
            DataFormat::Rg16Uint | DataFormat::Rg16Sint | DataFormat::Rg16Float => {
                Some(DataFormat::Rg16Sint)
            }
            DataFormat::Rgba16Uint | DataFormat::Rgba16Sint | DataFormat::Rgba16Float => {
                Some(DataFormat::Rgba16Sint)
            }
            DataFormat::R32Uint | DataFormat::R32Sint | DataFormat::R32Float => {
                Some(DataFormat::R32Sint)
            }
            DataFormat::Rg32Uint | DataFormat::Rg32Sint | DataFormat::Rg32Float => {
                Some(DataFormat::Rg32Sint)
            }
            DataFormat::Rgba32Uint | DataFormat::Rgba32Sint | DataFormat::Rgba32Float => {
                Some(DataFormat::Rgba32Sint)
            }
            _ => None,
        },
        NumericType::Float => match format {
            DataFormat::Rgba8Unorm | DataFormat::Rgba8Uint | DataFormat::Rgba8Sint => {
                Some(DataFormat::Rgba8Unorm)
            }
            DataFormat::R16Uint | DataFormat::R16Sint | DataFormat::R16Float => {
                Some(DataFormat::R16Float)
            }
            DataFormat::Rg16Uint | DataFormat::Rg16Sint | DataFormat::Rg16Float => {
                Some(DataFormat::Rg16Float)
            }
            DataFormat::Rgba16Uint | DataFormat::Rgba16Sint | DataFormat::Rgba16Float => {
                Some(DataFormat::Rgba16Float)
            }
            DataFormat::R32Uint | DataFormat::R32Sint | DataFormat::R32Float => {
                Some(DataFormat::R32Float)
            }
            DataFormat::Rg32Uint | DataFormat::Rg32Sint | DataFormat::Rg32Float => {
                Some(DataFormat::Rg32Float)
            }
            DataFormat::Rgba32Uint | DataFormat::Rgba32Sint | DataFormat::Rgba32Float => {
                Some(DataFormat::Rgba32Float)
            }
            _ => None,
        },
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
    color_inputs: &[ColorInputAttachment],
    sanitized_ll: &str,
    texture_reqs: &HashMap<u32, TextureBindingReq>,
) -> Option<Arc<DescriptorSet>> {
    let set_layout = layout.set_layouts().first()?.clone();
    if set_layout.bindings().is_empty() {
        return None;
    }
    let allocator = Arc::new(StandardDescriptorSetAllocator::new(
        device.clone(),
        Default::default(),
    ));
    let storage_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
    let mut bound_buffers = buffers.to_vec();
    bound_buffers.extend(make_synthetic_storage_buffers(
        storage_allocator.clone(),
        &bound_buffers,
        sanitized_ll,
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
        .filter(|binding| !bound_buffers.iter().any(|(index, _)| index == binding))
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
        // table[binding] = device address of the buffer bound at that binding, 0 where none. Sized to
        // cover every real buffer binding; the shader only indexes table[binding] for actual buffers.
        let max_binding = bound_buffers
            .iter()
            .map(|(index, _)| *index)
            .max()
            .unwrap_or(0);
        let mut table = vec![0u64; max_binding as usize + 1];
        for (index, buffer) in &bound_buffers {
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
            storage_allocator,
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
    let static_samplers = static_sampler_binding_states(sanitized_ll)
        .into_iter()
        .map(|(binding, state)| {
            let sampler = Sampler::new(device.clone(), state.create_info()).unwrap_or_else(|e| {
                panic!(
                    "create static sampler for AIR word {:#x} binding {binding}: {e}",
                    state.word
                )
            });
            (binding, sampler)
        })
        .collect::<HashMap<_, _>>();
    let writes = set_layout
        .bindings()
        .iter()
        .map(|(binding, info)| match info.descriptor_type {
            DescriptorType::StorageBuffer => {
                if address_table_binding == Some(*binding) {
                    let table = address_table_buffer
                        .clone()
                        .expect("PSB address-table buffer must exist for its binding");
                    return WriteDescriptorSet::buffer_array(
                        *binding,
                        0,
                        std::iter::repeat_n(table, info.descriptor_count as usize),
                    );
                }
                let buffer = bound_buffers
                    .iter()
                    .find_map(|(index, buffer)| (*index == *binding).then_some(buffer))
                    .unwrap_or_else(|| {
                        panic!("descriptor set expects storage buffer binding {binding}")
                    });
                WriteDescriptorSet::buffer_array(
                    *binding,
                    0,
                    std::iter::repeat_n(buffer.clone(), info.descriptor_count as usize),
                )
            }
            DescriptorType::SampledImage => {
                let texture_index =
                    binding
                        .checked_sub(TEXTURE_BINDING_BASE)
                        .unwrap_or_else(|| {
                            panic!("sampled image binding {binding} is below texture base")
                        });
                WriteDescriptorSet::image_view_array(
                    *binding,
                    0,
                    texture_view_array_for_binding(
                        textures,
                        texture_index,
                        info.descriptor_count,
                        texture_binding_req_for_index(texture_reqs, texture_index),
                    ),
                )
            }
            DescriptorType::StorageImage => {
                let texture_index =
                    binding
                        .checked_sub(TEXTURE_BINDING_BASE)
                        .unwrap_or_else(|| {
                            panic!("storage image binding {binding} is below texture base")
                        });
                WriteDescriptorSet::image_view_with_layout_array(
                    *binding,
                    0,
                    storage_texture_view_array_for_binding(
                        textures,
                        texture_index,
                        info.descriptor_count,
                        texture_binding_req_for_index(texture_reqs, texture_index),
                    ),
                )
            }
            DescriptorType::StorageTexelBuffer | DescriptorType::UniformTexelBuffer => {
                let texture_index =
                    binding
                        .checked_sub(TEXTURE_BINDING_BASE)
                        .unwrap_or_else(|| {
                            panic!("texel buffer binding {binding} is below texture base")
                        });
                let texture = textures
                    .iter()
                    .find(|texture| texture.index == texture_index)
                    .unwrap_or_else(|| {
                        panic!("descriptor set expects texture buffer binding {binding}")
                    });
                let texel_view = texture.texel_view.clone().unwrap_or_else(|| {
                    panic!("descriptor set expects texel buffer view for binding {binding}")
                });
                WriteDescriptorSet::buffer_view_array(
                    *binding,
                    0,
                    std::iter::repeat_n(texel_view, info.descriptor_count as usize),
                )
            }
            DescriptorType::InputAttachment => {
                let input_index = binding
                    .checked_sub(COLOR_INPUT_BINDING_BASE)
                    .unwrap_or_else(|| {
                        panic!("input attachment binding {binding} is below color input base")
                    });
                let color_input = color_inputs
                    .iter()
                    .find(|color_input| color_input.index == input_index)
                    .unwrap_or_else(|| {
                        panic!("descriptor set expects color input binding {binding}")
                    });
                WriteDescriptorSet::image_view_with_layout_array(
                    *binding,
                    0,
                    std::iter::repeat_n(
                        DescriptorImageViewInfo {
                            image_view: color_input.view.clone(),
                            image_layout: ImageLayout::ShaderReadOnlyOptimal,
                        },
                        info.descriptor_count as usize,
                    ),
                )
            }
            DescriptorType::Sampler => {
                let static_sampler = static_samplers.get(binding).cloned();
                let samplers = (0..info.descriptor_count)
                    .map(|_| {
                        static_sampler
                            .clone()
                            .unwrap_or_else(|| default_sampler.clone())
                    })
                    .collect::<Vec<_>>();
                WriteDescriptorSet::sampler_array(*binding, 0, samplers)
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

fn static_sampler_binding_states(sanitized_ll: &str) -> HashMap<u32, StaticSamplerState> {
    let mut occupied = runtime_sampler_bindings(sanitized_ll);
    let mut bindings = HashMap::new();
    for state in static_sampler_states(sanitized_ll) {
        let Some(binding) = (SAMPLER_BINDING_BASE..COLOR_INPUT_BINDING_BASE)
            .find(|binding| !occupied.contains(binding))
        else {
            break;
        };
        occupied.insert(binding);
        bindings.insert(binding, state);
    }
    bindings
}

fn runtime_sampler_bindings(sanitized_ll: &str) -> std::collections::HashSet<u32> {
    sanitized_ll
        .lines()
        .filter(|line| line.contains("air.sampler") && line.contains("air.location_index"))
        .filter_map(|line| {
            extract_i32_after(line, "air.location_index")
                .and_then(|loc| u32::try_from(loc).ok())
                .map(|loc| SAMPLER_BINDING_BASE.saturating_add(loc))
        })
        .collect()
}

fn extract_i32_after(line: &str, key: &str) -> Option<i32> {
    let (_, tail) = line.split_once(key)?;
    let (_, after_i32) = tail.split_once("i32 ")?;
    let token = after_i32
        .trim_start()
        .chars()
        .take_while(|c| c.is_ascii_digit() || *c == '-')
        .collect::<String>();
    token.parse().ok()
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
    fn compute_stage_input_values_follow_grid_element() {
        let mut thread_two = Vec::new();
        append_compute_stage_input_value(&mut thread_two, "float3", 2);

        let mut thread_three = Vec::new();
        append_compute_stage_input_value(&mut thread_three, "float3", 3);

        let mut expected = Vec::new();
        push_f32s(&mut expected, &[-1.0, 3.0, 0.0]);

        assert_eq!(thread_two, expected);
        assert_eq!(thread_three, expected);
    }

    #[test]
    fn compute_stage_input_bindings_include_fc_promoted_layout() {
        let ll = r#"
@flag = internal addrspace(2) global i8 0, align 1
!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5, !6}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"uint"}
!4 = !{i32 1, !"air.stage_in", !"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"float3"}
!5 = !{i32 2, !"air.function_constant", !7, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.arg_type_name", !"float"}
!6 = !{i32 3, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.arg_type_name", !"float"}
!7 = !{ptr addrspace(2) @flag, !"bool", !"enabled"}
"#;

        assert_eq!(
            compute_stage_input_binding_candidates(ll),
            vec![(1, 1), (1, 3)]
        );
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
    fn texturecube_array_type_name_uses_cube_array_shape() {
        let kind = texture_kind_from_type_name(Some("texturecube_array<half, sample>"));
        let shape = texture_shape(Extent3d::new(8, 8, 2), kind);

        assert_eq!(kind, TextureKind::CubeArray);
        assert_eq!(shape.image_type, ImageType::Dim2d);
        assert!(shape.flags.intersects(ImageCreateFlags::CUBE_COMPATIBLE));
        assert_eq!(shape.extent, [8, 8, 1]);
        assert_eq!(shape.array_layers, 12);
        assert_eq!(shape.view_type, Some(ImageViewType::CubeArray));
        assert_eq!(shape.seed_extent, Extent3d::new(8, 8, 12));
    }

    fn spirv_inst(opcode: u16, operands: &[u32]) -> Vec<u32> {
        let word_count = operands.len() as u32 + 1;
        let mut words = vec![(word_count << 16) | opcode as u32];
        words.extend_from_slice(operands);
        words
    }

    fn spirv_bytes(words: &[u32]) -> Vec<u8> {
        words.iter().flat_map(|word| word.to_le_bytes()).collect()
    }

    #[test]
    fn fragment_render_target_attachment_formats_include_all_mrt_locations() {
        let ll = r#"
!air.fragment = !{!15}
!15 = !{ptr @frag, !16, !25}
!16 = !{!17, !18, !19, !20, !21, !22, !23, !24}
!17 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"int4"}
!18 = !{!"air.render_target", i32 1, i32 0, !"air.arg_type_name", !"int4"}
!19 = !{!"air.render_target", i32 2, i32 0, !"air.arg_type_name", !"int4"}
!20 = !{!"air.render_target", i32 3, i32 0, !"air.arg_type_name", !"int4"}
!21 = !{!"air.render_target", i32 4, i32 0, !"air.arg_type_name", !"int4"}
!22 = !{!"air.render_target", i32 5, i32 0, !"air.arg_type_name", !"int4"}
!23 = !{!"air.render_target", i32 6, i32 0, !"air.arg_type_name", !"int4"}
!24 = !{!"air.render_target", i32 7, i32 0, !"air.arg_type_name", !"int4"}
"#;
        let mut words = vec![0x0723_0203, 0x0001_0300, 0, 32, 0];
        words.extend(spirv_inst(21, &[1, 32, 1])); // sint
        words.extend(spirv_inst(23, &[2, 1, 4])); // v4sint
        words.extend(spirv_inst(32, &[3, 3, 2])); // Output pointer to v4sint
        for location in 0..8 {
            let var = 10 + location;
            words.extend(spirv_inst(59, &[3, var, 3])); // Output variable
            words.extend(spirv_inst(71, &[var, 30, location])); // Location
        }

        let formats = fragment_render_target_attachment_formats(
            ll,
            DataFormat::Rgba32Sint,
            &spirv_bytes(&words),
        );

        assert_eq!(formats.len(), 8);
        assert!(formats
            .iter()
            .all(|format| *format == Some(DataFormat::Rgba32Sint)));
    }

    #[test]
    fn fragment_render_target_attachment_formats_ignore_unwritten_metadata_locations() {
        let ll = r#"
!air.fragment = !{!15}
!15 = !{ptr @frag, !16, !25}
!16 = !{!17, !18}
!17 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!18 = !{!"air.render_target", i32 4, i32 0, !"air.arg_type_name", !"int4"}
"#;
        let mut words = vec![0x0723_0203, 0x0001_0300, 0, 16, 0];
        words.extend(spirv_inst(22, &[1, 32])); // float
        words.extend(spirv_inst(23, &[2, 1, 4])); // v4float
        words.extend(spirv_inst(32, &[3, 3, 2])); // Output pointer to v4float
        words.extend(spirv_inst(59, &[3, 10, 3])); // Output variable
        words.extend(spirv_inst(71, &[10, 30, 0])); // Location 0

        let formats = fragment_render_target_attachment_formats(
            ll,
            DataFormat::Rgba32Float,
            &spirv_bytes(&words),
        );

        assert_eq!(formats, vec![Some(DataFormat::Rgba32Float)]);
    }

    #[test]
    fn fragment_flat_integer_input_scan_detects_scalar_and_vector_inputs() {
        let mut scalar = vec![0x0723_0203, 0x0001_0300, 0, 16, 0];
        scalar.extend(spirv_inst(21, &[1, 32, 0])); // uint
        scalar.extend(spirv_inst(32, &[2, 1, 1])); // Input pointer to uint
        scalar.extend(spirv_inst(59, &[2, 3, 1])); // Input variable
        scalar.extend(spirv_inst(71, &[3, 14])); // Flat

        assert!(fragment_has_flat_integer_input(&spirv_bytes(&scalar)));

        let mut vector = vec![0x0723_0203, 0x0001_0300, 0, 16, 0];
        vector.extend(spirv_inst(21, &[1, 32, 1])); // int
        vector.extend(spirv_inst(23, &[2, 1, 2])); // int2
        vector.extend(spirv_inst(32, &[3, 1, 2])); // Input pointer to int2
        vector.extend(spirv_inst(59, &[3, 4, 1])); // Input variable
        vector.extend(spirv_inst(71, &[4, 14])); // Flat

        assert!(fragment_has_flat_integer_input(&spirv_bytes(&vector)));

        let mut non_flat = scalar;
        non_flat.truncate(5 + 4 + 4 + 4);

        assert!(!fragment_has_flat_integer_input(&spirv_bytes(&non_flat)));
    }

    #[test]
    fn depth_render_target_does_not_request_color_attachment() {
        assert!(!render_target_writes_color0(DataFormat::Depth32Float, &[]));
    }

    #[test]
    fn vertex_inputs_resolve_patch_control_point_fields() {
        let ll = r#"
!50 = !{i32 0, !"air.patch_control_point_input", !51, !52, !53, !55}
!51 = !{!"air.patch_control_point_function", ptr @_Z12scn_vertex_t.MTL_CONTROL_POINT_FN}
!52 = !{!"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"float3", !"air.arg_name", !"position"}
!53 = !{!"air.function_constant", !54, !"air.location_index", i32 1, i32 1, !"air.arg_type_name", !"float3", !"air.arg_name", !"normal"}
!55 = !{!"air.function_constant", !54, !"air.location_index", i32 6, i32 1, !"air.arg_type_name", !"float2", !"air.arg_name", !"texcoord0"}
"#;
        assert_eq!(
            vertex_inputs(ll),
            vec![
                VertexInput {
                    location: 0,
                    type_name: "float3".to_string(),
                },
                VertexInput {
                    location: 1,
                    type_name: "float3".to_string(),
                },
                VertexInput {
                    location: 6,
                    type_name: "float2".to_string(),
                },
            ]
        );
    }

    #[test]
    fn texture_binding_scan_recovers_arrayed_image_view_type() {
        let mut words = vec![0x0723_0203, 0x0001_0300, 0, 16, 0];
        words.extend(spirv_inst(21, &[1, 32])); // %1 = OpTypeInt 32 0
        words.extend(spirv_inst(25, &[2, 1, 1, 0, 1, 0, 1, 0])); // image2D arrayed
        words.extend(spirv_inst(32, &[3, 0, 2])); // UniformConstant pointer to image
        words.extend(spirv_inst(59, &[3, 4, 0])); // variable
        words.extend(spirv_inst(71, &[4, 34, 0])); // DescriptorSet 0
        words.extend(spirv_inst(71, &[4, 33, TEXTURE_BINDING_BASE])); // Binding 32

        let reqs = texture_image_binding_reqs_from_words(&words);
        let req = reqs.get(&0).expect("texture binding 0");

        assert_eq!(req.image_view_type, Some(ImageViewType::Dim2dArray));
        assert!(req.needs_sampled);
        assert!(!req.needs_storage);
    }

    #[test]
    fn texture_binding_scan_recovers_descriptor_array_count() {
        let mut words = vec![0x0723_0203, 0x0001_0300, 0, 16, 0];
        words.extend(spirv_inst(21, &[1, 32, 0])); // %1 = OpTypeInt 32 0
        words.extend(spirv_inst(43, &[1, 2, 4])); // %2 = OpConstant %1 4
        words.extend(spirv_inst(25, &[3, 1, 1, 0, 0, 0, 1, 0])); // sampled image2D
        words.extend(spirv_inst(28, &[4, 3, 2])); // %4 = OpTypeArray %3 %2
        words.extend(spirv_inst(32, &[5, 0, 4])); // UniformConstant pointer to image array
        words.extend(spirv_inst(59, &[5, 6, 0])); // variable
        words.extend(spirv_inst(71, &[6, 34, 0])); // DescriptorSet 0
        words.extend(spirv_inst(71, &[6, 33, TEXTURE_BINDING_BASE])); // Binding 32

        let reqs = texture_image_binding_reqs_from_words(&words);
        let req = reqs.get(&0).expect("texture binding 0");

        assert_eq!(req.descriptor_count, 4);
        assert_eq!(req.image_view_type, Some(ImageViewType::Dim2d));
        assert!(req.needs_sampled);
        assert!(!req.needs_storage);
    }

    #[test]
    fn texture_binding_scan_clears_conflicting_view_type_aliases() {
        let mut words = vec![0x0723_0203, 0x0001_0300, 0, 16, 0];
        words.extend(spirv_inst(21, &[1, 32])); // %1 = OpTypeInt 32 0
        words.extend(spirv_inst(25, &[2, 1, 1, 0, 0, 0, 1, 0])); // image2D
        words.extend(spirv_inst(25, &[3, 1, 1, 0, 1, 0, 1, 0])); // image2D arrayed
        words.extend(spirv_inst(32, &[4, 0, 2])); // UniformConstant pointer to image2D
        words.extend(spirv_inst(32, &[5, 0, 3])); // UniformConstant pointer to image2DArray
        words.extend(spirv_inst(59, &[4, 6, 0])); // variable
        words.extend(spirv_inst(59, &[5, 7, 0])); // variable
        words.extend(spirv_inst(71, &[6, 34, 0])); // DescriptorSet 0
        words.extend(spirv_inst(71, &[6, 33, TEXTURE_BINDING_BASE])); // Binding 32
        words.extend(spirv_inst(71, &[7, 34, 0])); // DescriptorSet 0
        words.extend(spirv_inst(71, &[7, 33, TEXTURE_BINDING_BASE])); // Binding 32

        let reqs = texture_image_binding_reqs_from_words(&words);
        let req = reqs.get(&0).expect("texture binding 0");

        assert_eq!(req.image_view_type, None);
        assert!(req.needs_sampled);
        assert!(!req.needs_storage);
    }

    #[test]
    fn preflight_rejects_conflicting_texture_view_type_aliases() {
        let mut words = vec![0x0723_0203, 0x0001_0300, 0, 16, 0];
        words.extend(spirv_inst(21, &[1, 32])); // %1 = OpTypeInt 32 0
        words.extend(spirv_inst(25, &[2, 1, 1, 0, 0, 0, 1, 0])); // image2D
        words.extend(spirv_inst(25, &[3, 1, 1, 0, 1, 0, 1, 0])); // image2D arrayed
        words.extend(spirv_inst(32, &[4, 0, 2])); // UniformConstant pointer to image2D
        words.extend(spirv_inst(32, &[5, 0, 3])); // UniformConstant pointer to image2DArray
        words.extend(spirv_inst(59, &[4, 6, 0])); // variable
        words.extend(spirv_inst(59, &[5, 7, 0])); // variable
        words.extend(spirv_inst(71, &[6, 34, 0])); // DescriptorSet 0
        words.extend(spirv_inst(71, &[6, 33, TEXTURE_BINDING_BASE])); // Binding 32
        words.extend(spirv_inst(71, &[7, 34, 0])); // DescriptorSet 0
        words.extend(spirv_inst(71, &[7, 33, TEXTURE_BINDING_BASE])); // Binding 32

        let err = preflight_texture_binding_view_conflicts(&spirv_bytes(&words))
            .expect_err("conflicting view types should fail preflight");

        assert!(err.contains("incompatible SPIR-V image view types"));
        assert!(err.contains("32"));
    }

    #[test]
    fn preflight_rejects_conflicting_texture_scalar_type_aliases() {
        let mut words = vec![0x0723_0203, 0x0001_0300, 0, 16, 0];
        words.extend(spirv_inst(22, &[1, 32])); // %1 = OpTypeFloat 32
        words.extend(spirv_inst(21, &[2, 32, 0])); // %2 = OpTypeInt 32 0
        words.extend(spirv_inst(25, &[3, 1, 1, 0, 0, 0, 1, 0])); // float image2D
        words.extend(spirv_inst(25, &[4, 2, 1, 0, 0, 0, 1, 0])); // uint image2D
        words.extend(spirv_inst(32, &[5, 0, 3])); // UniformConstant pointer to float image
        words.extend(spirv_inst(32, &[6, 0, 4])); // UniformConstant pointer to uint image
        words.extend(spirv_inst(59, &[5, 7, 0])); // variable
        words.extend(spirv_inst(59, &[6, 8, 0])); // variable
        words.extend(spirv_inst(71, &[7, 34, 0])); // DescriptorSet 0
        words.extend(spirv_inst(71, &[7, 33, TEXTURE_BINDING_BASE + 2])); // Binding 34
        words.extend(spirv_inst(71, &[8, 34, 0])); // DescriptorSet 0
        words.extend(spirv_inst(71, &[8, 33, TEXTURE_BINDING_BASE + 2])); // Binding 34

        let reqs = texture_image_binding_reqs_from_words(&words);
        let req = reqs.get(&2).expect("texture binding 2");

        assert_eq!(req.image_scalar_type, None);
        assert!(req.image_scalar_type_conflict);

        let err = preflight_texture_binding_view_conflicts(&spirv_bytes(&words))
            .expect_err("conflicting scalar types should fail preflight");

        assert!(err.contains("incompatible SPIR-V image scalar types"));
        assert!(err.contains("34"));
    }

    #[test]
    fn conflicting_spirv_view_type_clears_reflected_view_hint() {
        let mut req = TextureBindingReq {
            index: 0,
            descriptor_count: 1,
            needs_sampled: true,
            needs_storage: false,
            image_format: None,
            image_scalar_type: None,
            image_view_type: Some(ImageViewType::Dim2dArray),
        };

        merge_spirv_image_binding_req(
            &mut req,
            SpirvImageBindingReq {
                descriptor_count: 1,
                image_view_type: None,
                image_scalar_type: None,
                image_scalar_type_conflict: false,
                needs_sampled: true,
                needs_storage: false,
            },
        );

        assert_eq!(req.image_view_type, None);
    }

    #[test]
    fn reflected_sampled_image_requirement_adds_sampled_usage() {
        let req = TextureBindingReq {
            index: 0,
            descriptor_count: 1,
            needs_sampled: true,
            needs_storage: false,
            image_format: None,
            image_scalar_type: None,
            image_view_type: Some(ImageViewType::Dim2d),
        };

        let usage =
            vulkan_image_usage(TextureRole::StorageWrite) | texture_binding_image_usage(Some(&req));

        assert!(usage.intersects(ImageUsage::STORAGE));
        assert!(usage.intersects(ImageUsage::SAMPLED));
    }

    #[test]
    fn color_input_attachment_format_uses_air_render_target_type() {
        let ll = r#"
!air.fragment = !{!0}
!0 = !{ptr @frag, !1, !3}
!1 = !{!2}
!2 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"half4"}
!3 = !{!4}
!4 = !{i32 2, !"air.render_target", i32 2, !"air.arg_type_name", !"float", !"air.arg_name", !"d0"}
"#;

        assert_eq!(
            color_input_attachment_format(ll, 2),
            Some(DataFormat::R32Float)
        );
        assert_eq!(color_input_attachment_format(ll, 0), None);
    }

    #[test]
    fn reflected_sampled_image_array_requirement_reaches_later_texture_index() {
        let req = TextureBindingReq {
            index: 0,
            descriptor_count: 5,
            needs_sampled: true,
            needs_storage: false,
            image_format: None,
            image_scalar_type: None,
            image_view_type: Some(ImageViewType::Dim2d),
        };
        let reqs = HashMap::from([(0, req)]);

        let usage = vulkan_image_usage(TextureRole::StorageWrite)
            | texture_binding_image_usage_for_index(&reqs, 4);

        assert!(usage.intersects(ImageUsage::STORAGE));
        assert!(usage.intersects(ImageUsage::SAMPLED));
        assert!(!texture_binding_image_usage_for_index(&reqs, 5).intersects(ImageUsage::SAMPLED));
    }

    #[test]
    fn reflected_sampled_image_array_requirement_overrides_later_texture_format() {
        let req = TextureBindingReq {
            index: 0,
            descriptor_count: 3,
            needs_sampled: true,
            needs_storage: false,
            image_format: None,
            image_scalar_type: Some(NumericType::Float),
            image_view_type: Some(ImageViewType::Dim2d),
        };
        let reqs = HashMap::from([(0, req)]);

        let format = reflected_texture_format(
            DataFormat::Rgba16Uint,
            texture_binding_req_for_index(&reqs, 2),
        );

        assert_eq!(format, DataFormat::Rgba16Float);
    }

    #[test]
    fn descriptor_array_numeric_alias_requires_mutable_format_view() {
        let req = TextureBindingReq {
            index: 0,
            descriptor_count: 3,
            needs_sampled: true,
            needs_storage: false,
            image_format: None,
            image_scalar_type: Some(NumericType::Float),
            image_view_type: Some(ImageViewType::Dim2d),
        };
        let reqs = HashMap::from([(0, req)]);

        let flags = texture_binding_mutable_format_flag(&reqs, 2, DataFormat::Rgba16Uint);
        let view_format = descriptor_view_format(Format::R16G16B16A16_UINT, Some(&req));

        assert!(flags.intersects(ImageCreateFlags::MUTABLE_FORMAT));
        assert_eq!(view_format, Format::R16G16B16A16_SFLOAT);
    }

    #[test]
    fn reflected_storage_image_format_overrides_plan_format() {
        let req = TextureBindingReq {
            index: 0,
            descriptor_count: 1,
            needs_sampled: false,
            needs_storage: true,
            image_format: Some(Format::R16G16B16A16_SFLOAT),
            image_scalar_type: None,
            image_view_type: Some(ImageViewType::Dim2d),
        };

        assert_eq!(
            reflected_texture_format(DataFormat::Rgba32Float, Some(&req)),
            DataFormat::Rgba16Float
        );
        assert_eq!(
            reflected_texture_format(DataFormat::Rgba32Float, None),
            DataFormat::Rgba32Float
        );
    }

    #[test]
    fn reflected_sampled_image_scalar_type_overrides_plan_numeric_class() {
        let req = TextureBindingReq {
            index: 0,
            descriptor_count: 1,
            needs_sampled: true,
            needs_storage: false,
            image_format: None,
            image_scalar_type: Some(NumericType::Uint),
            image_view_type: Some(ImageViewType::Dim2d),
        };

        assert_eq!(
            reflected_texture_format(DataFormat::Rgba32Float, Some(&req)),
            DataFormat::Rgba32Uint
        );
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
    fn static_sampler_binding_skips_runtime_sampler_slots() {
        let states = static_sampler_binding_states(
            r#"
@__air_sampler_state = internal addrspace(2) constant [2 x i64] [i64 34901797601020489, i64 0], align 8
!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.sampler", !"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"sampler"}
"#,
        );

        assert!(!states.contains_key(&SAMPLER_BINDING_BASE));
        let state = states
            .get(&(SAMPLER_BINDING_BASE + 1))
            .expect("static sampler should use first unoccupied sampler-band binding");
        assert_eq!(state.filter, Filter::Linear);
        assert_eq!(state.address_mode, [SamplerAddressMode::ClampToEdge; 3]);
    }

    #[test]
    fn vertex_validation_pipeline_omits_viewport_state_under_discard() {
        assert!(vertex_validation_viewport_state().is_none());
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
