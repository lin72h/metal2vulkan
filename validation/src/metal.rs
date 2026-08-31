use crate::case::AuthoredCase;
use crate::check::check_case;
use crate::hash::sha256_bytes;
use crate::literal::LiteralResources;
use crate::observation::{MetalObservation, MetalStatus};
use crate::store::CorpusStore;
use base64::Engine as _;
use std::path::Path;

pub const ORACLE_ABI: &str = "metal-literal-resources-v23";
pub const QUALIFICATION_RUNS: usize = 3;

pub fn qualify_case(
    root: &Path,
    case: AuthoredCase,
    environment_id: &str,
) -> Result<MetalObservation, String> {
    if environment_id.trim().is_empty() {
        return Err("environment_id must not be empty".into());
    }
    let checked = check_case(root, case).map_err(|errors| errors.join("; "))?;
    crate::executor_contract::require_case(&checked.case, "Metal executor")?;
    let resources = LiteralResources::prepare(&checked.case)?;
    let initial = selected_initial_bytes(&checked.case, &resources)?;
    let mut outputs = Vec::with_capacity(QUALIFICATION_RUNS);
    for _ in 0..QUALIFICATION_RUNS {
        outputs.push(platform::execute(
            &checked.case,
            &checked.source,
            &resources,
            &checked.reflection,
            &checked.linked_functions,
        )?);
    }
    let output = qualify_outputs(&initial, outputs)?;
    let observation = MetalObservation {
        case_id: checked.case.case_id.clone(),
        air_sha256: checked.case.air_sha256.clone(),
        input_sha256: checked.input_sha256,
        metal_output_sha256: sha256_bytes(&output),
        output_b64: base64::engine::general_purpose::STANDARD.encode(output),
        environment_id: environment_id.into(),
        environment: platform::environment()?,
        oracle_abi: ORACLE_ABI.into(),
        status: MetalStatus::Qualified,
    };
    CorpusStore::new(root).upsert_metal(observation.clone())?;
    Ok(observation)
}

fn qualify_outputs(initial: &[u8], outputs: Vec<Vec<u8>>) -> Result<Vec<u8>, String> {
    if outputs.len() != QUALIFICATION_RUNS {
        return Err(format!(
            "Metal qualification requires exactly {QUALIFICATION_RUNS} runs, got {}",
            outputs.len()
        ));
    }
    if outputs.windows(2).any(|pair| pair[0] != pair[1]) {
        return Err(format!(
            "Metal qualification is nondeterministic across {QUALIFICATION_RUNS} runs"
        ));
    }
    let output = outputs
        .into_iter()
        .next()
        .ok_or_else(|| "qualification performed no runs".to_string())?;
    if output.len() != initial.len() {
        return Err(format!(
            "Metal returned {} selected bytes, expected {}",
            output.len(),
            initial.len()
        ));
    }
    Ok(output)
}

fn selected_initial_bytes(
    case: &AuthoredCase,
    resources: &LiteralResources,
) -> Result<Vec<u8>, String> {
    match &case.output {
        crate::case::OutputSelection::None => Ok(Vec::new()),
        crate::case::OutputSelection::Buffer {
            binding,
            offset,
            length,
        } => {
            let resource = resources
                .buffers
                .iter()
                .find(|resource| resource.binding == *binding)
                .ok_or_else(|| format!("missing output buffer binding {binding}"))?;
            let all = &resource.bytes;
            let start = *offset as usize;
            let end = start
                .checked_add(*length as usize)
                .ok_or_else(|| "selected output range overflow".to_string())?;
            all.get(start..end)
                .map(<[u8]>::to_vec)
                .ok_or_else(|| "selected output range exceeds buffer".into())
        }
        crate::case::OutputSelection::ArgumentBufferBuffer {
            buffer_binding,
            field_offset,
            offset,
            length,
        } => {
            let resource = resources
                .argument_buffer_buffers
                .iter()
                .find(|resource| {
                    resource.buffer_binding == *buffer_binding
                        && resource.field_offset == *field_offset
                })
                .ok_or_else(|| {
                    format!("missing argument-buffer buffer {buffer_binding}+{field_offset}")
                })?;
            let start = *offset as usize;
            let end = start
                .checked_add(*length as usize)
                .ok_or_else(|| "selected output range overflow".to_string())?;
            resource
                .bytes
                .get(start..end)
                .map(<[u8]>::to_vec)
                .ok_or_else(|| "selected output range exceeds argument-buffer buffer".into())
        }
        crate::case::OutputSelection::DeviceBufferArrayElement {
            binding,
            element,
            offset,
            length,
        } => {
            let resource = resources
                .device_buffer_arrays
                .iter()
                .find(|array| array.binding == *binding)
                .and_then(|array| array.elements.iter().find(|item| item.index == *element))
                .ok_or_else(|| {
                    format!("missing device-buffer-array binding {binding} element {element}")
                })?;
            let start = *offset as usize;
            let end = start
                .checked_add(*length as usize)
                .ok_or_else(|| "selected output range overflow".to_string())?;
            resource
                .bytes
                .get(start..end)
                .map(<[u8]>::to_vec)
                .ok_or_else(|| "selected output range exceeds device-buffer-array element".into())
        }
        crate::case::OutputSelection::Texture {
            binding,
            origin,
            dimensions,
        } => resources
            .textures
            .iter()
            .find(|resource| resource.binding == *binding)
            .ok_or_else(|| format!("missing output texture binding {binding}"))?
            .select(*origin, *dimensions),
        crate::case::OutputSelection::TextureArrayElement {
            binding,
            element,
            origin,
            dimensions,
        } => resources
            .texture_arrays
            .iter()
            .find(|array| array.binding == *binding)
            .and_then(|array| array.elements.get(*element as usize))
            .ok_or_else(|| format!("missing texture-array binding {binding} element {element}"))?
            .select(*origin, *dimensions),
        crate::case::OutputSelection::ArgumentBufferTexture {
            buffer_binding,
            field_offset,
            origin,
            dimensions,
        } => resources
            .argument_buffer_textures
            .iter()
            .find(|resource| {
                resource.buffer_binding == *buffer_binding && resource.field_offset == *field_offset
            })
            .ok_or_else(|| {
                format!("missing argument-buffer texture {buffer_binding}+{field_offset}")
            })?
            .select(*origin, *dimensions),
        crate::case::OutputSelection::RenderTarget {
            index,
            origin,
            dimensions,
        } => resources
            .render_targets
            .iter()
            .find(|target| target.index == *index)
            .ok_or_else(|| format!("missing render target {index}"))?
            .select(*origin, *dimensions),
        crate::case::OutputSelection::Depth { origin, dimensions }
        | crate::case::OutputSelection::Stencil { origin, dimensions } => {
            let attachment = resources
                .depth_stencil
                .as_ref()
                .ok_or_else(|| "missing depth/stencil attachment".to_string())?;
            let bytes = if matches!(case.output, crate::case::OutputSelection::Depth { .. }) {
                attachment.depth.as_ref()
            } else {
                attachment.stencil.as_ref()
            }
            .ok_or_else(|| "missing selected depth/stencil aspect".to_string())?;
            crate::literal::select_tightly_packed_2d(
                bytes,
                attachment.dimensions,
                *origin,
                *dimensions,
                if matches!(case.output, crate::case::OutputSelection::Depth { .. }) {
                    4
                } else {
                    1
                },
            )
        }
        crate::case::OutputSelection::FragmentImageblock {
            semantic,
            origin,
            dimensions,
        } => {
            let imageblock = resources
                .fragment_imageblock
                .as_ref()
                .ok_or_else(|| "missing fragment imageblock".to_string())?;
            let member = imageblock
                .members
                .iter()
                .find(|member| member.semantic == *semantic)
                .ok_or_else(|| format!("missing fragment imageblock member {semantic}"))?;
            crate::literal::select_tightly_packed_2d(
                &member.bytes,
                imageblock.dimensions,
                *origin,
                *dimensions,
                member.format.byte_size(),
            )
        }
    }
}

#[cfg(target_os = "macos")]
mod platform {
    use super::*;
    use crate::case::{
        AccelerationStructureKind, ImplicitImageblockCoverage, OutputSelection, ResourceRole,
        SamplerAddressMode, SamplerFilter, SamplerMipFilter, ScalarType, TextureFormat,
        TextureType,
    };
    use crate::source::SourceRow;
    use crate::ScratchDir;
    use core::ffi::c_void;
    use core::ptr::NonNull;
    use objc2::rc::{autoreleasepool, Retained};
    use objc2::runtime::ProtocolObject;
    use objc2_foundation::{NSArray, NSString, NSURL};
    use objc2_metal::{
        MTLAccelerationStructure, MTLAccelerationStructureCommandEncoder,
        MTLAccelerationStructureDescriptor, MTLAccelerationStructureGeometryDescriptor,
        MTLAccelerationStructureInstanceDescriptor, MTLAccelerationStructureInstanceOptions,
        MTLAccelerationStructureTriangleGeometryDescriptor, MTLArgumentEncoder, MTLAttributeFormat,
        MTLBuffer, MTLColorWriteMask, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder,
        MTLCommandQueue, MTLCompareFunction, MTLComputeCommandEncoder,
        MTLComputePipelineDescriptor, MTLComputePipelineState, MTLCreateSystemDefaultDevice,
        MTLDataType, MTLDepthStencilDescriptor, MTLDepthStencilState, MTLDevice, MTLFunction,
        MTLFunctionConstantValues, MTLInstanceAccelerationStructureDescriptor,
        MTLIntersectionFunctionSignature, MTLIntersectionFunctionTable,
        MTLIntersectionFunctionTableDescriptor, MTLLibrary, MTLLinkedFunctions, MTLLoadAction,
        MTLOrigin, MTLPackedFloat3, MTLPackedFloat4x3, MTLPipelineOption, MTLPixelFormat,
        MTLPrimitiveAccelerationStructureDescriptor, MTLPrimitiveTopologyClass, MTLPrimitiveType,
        MTLRegion, MTLRenderCommandEncoder, MTLRenderPassDescriptor, MTLRenderPipelineDescriptor,
        MTLRenderPipelineState, MTLRenderStages, MTLResourceOptions, MTLSamplerAddressMode,
        MTLSamplerDescriptor, MTLSamplerMinMagFilter, MTLSamplerMipFilter, MTLSamplerState,
        MTLSize, MTLStageInputOutputDescriptor, MTLStencilDescriptor, MTLStencilOperation,
        MTLStepFunction, MTLStorageMode, MTLStoreAction, MTLTessellationControlPointIndexType,
        MTLTessellationFactorFormat, MTLTessellationFactorStepFunction,
        MTLTessellationPartitionMode, MTLTexture, MTLTextureDescriptor, MTLTextureType,
        MTLTextureUsage, MTLTileRenderPipelineDescriptor, MTLVertexDescriptor, MTLVertexFormat,
        MTLVertexStepFunction, MTLVisibleFunctionTable, MTLVisibleFunctionTableDescriptor,
        MTLWinding,
    };
    use std::cell::RefCell;
    use std::collections::HashMap;
    use std::fs;
    use std::process::Command;

    type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;
    type AccelerationStructure = Retained<ProtocolObject<dyn MTLAccelerationStructure>>;
    type RawTexture = Retained<ProtocolObject<dyn MTLTexture>>;
    type Sampler = Retained<ProtocolObject<dyn MTLSamplerState>>;
    type DepthStencilState = Retained<ProtocolObject<dyn MTLDepthStencilState>>;
    type Device = Retained<ProtocolObject<dyn MTLDevice>>;
    type CommandQueue = Retained<ProtocolObject<dyn MTLCommandQueue>>;
    type Function = Retained<ProtocolObject<dyn MTLFunction>>;
    type Library = Retained<ProtocolObject<dyn MTLLibrary>>;
    type VisibleFunctionTable = Retained<ProtocolObject<dyn MTLVisibleFunctionTable>>;
    type IntersectionFunctionTable = Retained<ProtocolObject<dyn MTLIntersectionFunctionTable>>;

    struct Texture {
        object: RawTexture,
        _backing: Option<Buffer>,
    }

    impl std::ops::Deref for Texture {
        type Target = ProtocolObject<dyn MTLTexture>;

        fn deref(&self) -> &Self::Target {
            &self.object
        }
    }

    struct MetalLinkedFunctions {
        descriptor: Option<Retained<MTLLinkedFunctions>>,
        functions: HashMap<(String, String), Function>,
    }

    #[derive(Default)]
    struct MetalFunctionTables {
        visible: Vec<(u32, VisibleFunctionTable)>,
        intersection: Vec<(
            crate::library_module::ResolvedIntersectionFunctionTableLocation,
            IntersectionFunctionTable,
        )>,
    }

    struct MetalVertexInputs {
        descriptor: Retained<MTLVertexDescriptor>,
        buffers: Vec<(usize, Buffer)>,
    }

    struct MetalContext {
        device: Device,
        queue: CommandQueue,
    }

    impl MetalContext {
        fn new() -> Result<Self, String> {
            let device = MTLCreateSystemDefaultDevice()
                .ok_or_else(|| "MTLCreateSystemDefaultDevice returned nil".to_string())?;
            let queue = device
                .newCommandQueue()
                .ok_or_else(|| "MTLDevice::newCommandQueue returned nil".to_string())?;
            Ok(Self { device, queue })
        }
    }

    pub fn execute(
        case: &AuthoredCase,
        source: &SourceRow,
        resources: &LiteralResources,
        reflection: &metal2vulkan::reflect::ShaderReflection,
        function_tables: &crate::library_module::ResolvedLinkedFunctions,
    ) -> Result<Vec<u8>, String> {
        debug_assert!(crate::executor_contract::require_case(case, "Metal executor").is_ok());
        thread_local! {
            static CONTEXT: RefCell<Option<MetalContext>> = const { RefCell::new(None) };
        }
        autoreleasepool(|_| {
            CONTEXT.with(|slot| {
                let mut slot = slot.borrow_mut();
                if slot.is_none() {
                    *slot = Some(MetalContext::new()?);
                }
                let context = slot.as_ref().expect("initialized Metal context");
                match case.stage {
                    crate::case::Stage::Kernel
                        if !reflection.implicit_imageblock_attachments.is_empty() =>
                    {
                        execute_tile(
                            context,
                            case,
                            source,
                            resources,
                            reflection,
                            function_tables,
                        )
                    }
                    crate::case::Stage::Kernel => execute_compute(
                        context,
                        case,
                        source,
                        resources,
                        reflection,
                        function_tables,
                    ),
                    crate::case::Stage::Fragment => execute_fragment(
                        context,
                        case,
                        source,
                        resources,
                        reflection,
                        function_tables,
                    ),
                    crate::case::Stage::Vertex => execute_vertex(
                        context,
                        case,
                        source,
                        resources,
                        reflection,
                        function_tables,
                    ),
                }
            })
        })
    }

    fn execute_compute(
        context: &MetalContext,
        case: &AuthoredCase,
        source: &SourceRow,
        resources: &LiteralResources,
        reflection: &metal2vulkan::reflect::ShaderReflection,
        function_tables: &crate::library_module::ResolvedLinkedFunctions,
    ) -> Result<Vec<u8>, String> {
        let device = &context.device;
        let library = load_library(device, source)?;
        let entry = NSString::from_str(&case.entry);
        let function = make_function(&library, &entry, resources)?;
        let linked = load_linked_functions(device, function_tables, resources)?;
        let pipeline = make_pipeline(device, &function, resources, reflection, &linked)?;
        let metal_function_tables =
            make_compute_function_tables(&pipeline, function_tables, &linked)?;
        let mut buffers = make_buffers(device, resources, reflection)?;
        let device_buffer_array_elements =
            append_device_buffer_arrays(device, resources, &mut buffers)?;
        let argument_buffer_buffers = make_argument_buffer_buffers(device, resources)?;
        let textures = make_textures(device, &context.queue, resources)?;
        let texture_arrays = make_texture_arrays(device, &context.queue, resources)?;
        let argument_buffer_textures =
            make_argument_buffer_textures(device, &context.queue, resources)?;
        encode_argument_buffer_buffers(&function, &buffers, &argument_buffer_buffers, reflection)?;
        encode_argument_buffer_textures(
            &function,
            &buffers,
            &argument_buffer_textures,
            reflection,
        )?;
        encode_argument_buffer_function_tables(
            &function,
            &buffers,
            &metal_function_tables,
            reflection,
        )?;
        let samplers = make_samplers(device, case)?;
        let dispatch = case
            .dispatch
            .as_ref()
            .ok_or_else(|| "kernel case has no dispatch".to_string())?;
        let acceleration_structures = make_acceleration_structures(device, &context.queue, case)?;
        let command_buffer = context
            .queue
            .commandBuffer()
            .ok_or_else(|| "MTLCommandQueue::commandBuffer returned nil".to_string())?;
        let encoder = command_buffer
            .computeCommandEncoder()
            .ok_or_else(|| "computeCommandEncoder returned nil".to_string())?;
        encoder.setComputePipelineState(&pipeline);
        for (binding, buffer) in &buffers {
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(&**buffer), 0, *binding as usize);
            }
        }
        for (binding, acceleration_structure) in &acceleration_structures {
            unsafe {
                encoder.setAccelerationStructure_atBufferIndex(
                    Some(&**acceleration_structure),
                    *binding as usize,
                );
            }
        }
        for (binding, table) in &metal_function_tables.visible {
            unsafe {
                encoder.setVisibleFunctionTable_atBufferIndex(Some(&**table), *binding as usize);
            }
        }
        for (location, table) in &metal_function_tables.intersection {
            if let crate::library_module::ResolvedIntersectionFunctionTableLocation::Direct {
                binding,
            } = location
            {
                unsafe {
                    encoder.setIntersectionFunctionTable_atBufferIndex(
                        Some(&**table),
                        *binding as usize,
                    );
                }
            }
        }
        for (binding, texture) in &textures {
            unsafe { encoder.setTexture_atIndex(Some(&**texture), *binding as usize) };
        }
        for (binding, elements) in &texture_arrays {
            for (element, texture) in elements.iter().enumerate() {
                unsafe {
                    encoder.setTexture_atIndex(Some(&**texture), *binding as usize + element)
                };
            }
        }
        for (binding, sampler) in &samplers {
            unsafe { encoder.setSamplerState_atIndex(Some(&**sampler), *binding as usize) };
        }
        for resource in &case.threadgroup_memory {
            unsafe {
                encoder.setThreadgroupMemoryLength_atIndex(
                    resource.length as usize,
                    resource.binding as usize,
                );
            }
        }
        if let Some(imageblock) = &case.imageblock {
            encoder.setImageblockWidth_height(
                imageblock.dimensions[0] as usize,
                imageblock.dimensions[1] as usize,
            );
        }
        if !resources.kernel_stage_inputs.is_empty() {
            encoder.setStageInRegion(MTLRegion {
                origin: MTLOrigin { x: 0, y: 0, z: 0 },
                size: mtl_size(dispatch.grid),
            });
        }
        encoder.dispatchThreads_threadsPerThreadgroup(
            mtl_size(dispatch.grid),
            mtl_size(dispatch.threads_per_threadgroup),
        );
        encoder.endEncoding();
        command_buffer.commit();
        command_buffer.waitUntilCompleted();
        if command_buffer.status() != MTLCommandBufferStatus::Completed {
            return Err(command_buffer
                .error()
                .map(|error| format!("Metal command buffer failed: {error}"))
                .unwrap_or_else(|| {
                    format!(
                        "Metal command buffer ended with status {:?}",
                        command_buffer.status()
                    )
                }));
        }
        selected_output(
            case,
            &buffers,
            &argument_buffer_buffers,
            &textures,
            &texture_arrays,
            &argument_buffer_textures,
            MetalOutputResources {
                device_buffer_array_elements: &device_buffer_array_elements,
                colors: &[],
                depth_stencil: None,
                fragment_imageblock: None,
            },
        )
    }

    /// Execute a kernel-based tile function whose implicit imageblock fields alias render-pass
    /// color attachments. Metal exposes this ABI through a tile render pipeline, not a compute
    /// pipeline; using the latter would compile the function but leave its attachment storage
    /// unbound.
    fn execute_tile(
        context: &MetalContext,
        case: &AuthoredCase,
        source: &SourceRow,
        resources: &LiteralResources,
        reflection: &metal2vulkan::reflect::ShaderReflection,
        function_tables: &crate::library_module::ResolvedLinkedFunctions,
    ) -> Result<Vec<u8>, String> {
        let device = &context.device;
        let library = load_library(device, source)?;
        let entry = NSString::from_str(&case.entry);
        let function = make_function(&library, &entry, resources)?;
        let linked = load_linked_functions(device, function_tables, resources)?;
        let render_targets = make_render_targets(device, resources, false)?;
        let dispatch = case
            .dispatch
            .as_ref()
            .ok_or_else(|| "tile kernel case has no dispatch".to_string())?;

        let pipeline_descriptor = MTLTileRenderPipelineDescriptor::new();
        unsafe { pipeline_descriptor.setTileFunction(&function) };
        unsafe { pipeline_descriptor.setRasterSampleCount(1) };
        pipeline_descriptor.setThreadgroupSizeMatchesTileSize(true);
        pipeline_descriptor
            .setRequiredThreadsPerThreadgroup(mtl_size(dispatch.threads_per_threadgroup));
        if let Some(linked) = &linked.descriptor {
            pipeline_descriptor.setLinkedFunctions(Some(linked));
        }
        let pipeline_attachments = pipeline_descriptor.colorAttachments();
        for (index, texture) in &render_targets {
            let attachment =
                unsafe { pipeline_attachments.objectAtIndexedSubscript(*index as usize) };
            attachment.setPixelFormat(texture.pixelFormat());
        }
        let pipeline = device
            .newRenderPipelineStateWithTileDescriptor_options_reflection_error(
                &pipeline_descriptor,
                MTLPipelineOption::None,
                None,
            )
            .map_err(|error| format!("create tile render pipeline: {error}"))?;
        let coverage_pipeline = make_tile_coverage_pipeline(device, &render_targets)?;
        let metal_function_tables = make_render_function_tables(
            &pipeline,
            MTLRenderStages::Tile,
            function_tables,
            &linked,
        )?;

        let mut buffers = make_buffers(device, resources, reflection)?;
        let device_buffer_array_elements =
            append_device_buffer_arrays(device, resources, &mut buffers)?;
        let argument_buffer_buffers = make_argument_buffer_buffers(device, resources)?;
        let textures = make_textures(device, &context.queue, resources)?;
        let texture_arrays = make_texture_arrays(device, &context.queue, resources)?;
        let argument_buffer_textures =
            make_argument_buffer_textures(device, &context.queue, resources)?;
        encode_argument_buffer_buffers(&function, &buffers, &argument_buffer_buffers, reflection)?;
        encode_argument_buffer_textures(
            &function,
            &buffers,
            &argument_buffer_textures,
            reflection,
        )?;
        encode_argument_buffer_function_tables(
            &function,
            &buffers,
            &metal_function_tables,
            reflection,
        )?;
        let samplers = make_samplers(device, case)?;
        let acceleration_structures = make_acceleration_structures(device, &context.queue, case)?;

        let pass = MTLRenderPassDescriptor::renderPassDescriptor();
        let pass_attachments = pass.colorAttachments();
        for (index, texture) in &render_targets {
            let attachment = unsafe { pass_attachments.objectAtIndexedSubscript(*index as usize) };
            attachment.setTexture(Some(&**texture));
            attachment.setLoadAction(MTLLoadAction::Load);
            attachment.setStoreAction(MTLStoreAction::Store);
        }
        let imageblock = case.imageblock.as_ref().ok_or_else(|| {
            "implicit imageblock tile kernel has no imageblock dimensions".to_string()
        })?;
        if imageblock.implicit_coverage != Some(ImplicitImageblockCoverage::FullSingleSample) {
            return Err(
                "implicit imageblock tile kernel requires full_single_sample coverage".into(),
            );
        }
        pass.setTileWidth(imageblock.dimensions[0] as usize);
        pass.setTileHeight(imageblock.dimensions[1] as usize);
        pass.setRenderTargetWidth(dispatch.grid[0] as usize);
        pass.setRenderTargetHeight(dispatch.grid[1] as usize);
        let threadgroup_layout = tile_threadgroup_memory_layout(case)?;
        if let Some((_, total)) = threadgroup_layout.last() {
            pass.setThreadgroupMemoryLength(*total);
        }

        let command_buffer = context
            .queue
            .commandBuffer()
            .ok_or_else(|| "MTLCommandQueue::commandBuffer returned nil".to_string())?;
        let encoder = command_buffer
            .renderCommandEncoderWithDescriptor(&pass)
            .ok_or_else(|| "renderCommandEncoderWithDescriptor returned nil".to_string())?;
        encoder.setRenderPipelineState(&coverage_pipeline);
        unsafe { encoder.drawPrimitives_vertexStart_vertexCount(MTLPrimitiveType::Triangle, 0, 3) };
        encoder.setRenderPipelineState(&pipeline);
        for (binding, buffer) in &buffers {
            unsafe { encoder.setTileBuffer_offset_atIndex(Some(&**buffer), 0, *binding as usize) };
        }
        for (binding, acceleration_structure) in &acceleration_structures {
            unsafe {
                encoder.setTileAccelerationStructure_atBufferIndex(
                    Some(&**acceleration_structure),
                    *binding as usize,
                )
            };
        }
        bind_tile_function_tables(&encoder, &metal_function_tables);
        for (binding, texture) in &textures {
            unsafe { encoder.setTileTexture_atIndex(Some(&**texture), *binding as usize) };
        }
        for (binding, elements) in &texture_arrays {
            for (element, texture) in elements.iter().enumerate() {
                unsafe {
                    encoder.setTileTexture_atIndex(Some(&**texture), *binding as usize + element)
                };
            }
        }
        for (binding, sampler) in &samplers {
            unsafe { encoder.setTileSamplerState_atIndex(Some(&**sampler), *binding as usize) };
        }
        for ((binding, length), (offset, _)) in case
            .threadgroup_memory
            .iter()
            .map(|resource| (resource.binding, resource.length))
            .zip(&threadgroup_layout)
        {
            unsafe {
                encoder.setThreadgroupMemoryLength_offset_atIndex(
                    length as usize,
                    *offset,
                    binding as usize,
                )
            };
        }
        encoder.dispatchThreadsPerTile(mtl_size(dispatch.threads_per_threadgroup));
        encoder.endEncoding();
        command_buffer.commit();
        command_buffer.waitUntilCompleted();
        ensure_completed(&command_buffer, "Metal tile dispatch")?;
        selected_output(
            case,
            &buffers,
            &argument_buffer_buffers,
            &textures,
            &texture_arrays,
            &argument_buffer_textures,
            MetalOutputResources {
                device_buffer_array_elements: &device_buffer_array_elements,
                colors: &render_targets,
                depth_stencil: None,
                fragment_imageblock: None,
            },
        )
    }

    fn make_tile_coverage_pipeline(
        device: &ProtocolObject<dyn MTLDevice>,
        render_targets: &[(u32, Texture)],
    ) -> Result<Retained<ProtocolObject<dyn MTLRenderPipelineState>>, String> {
        let source = NSString::from_str(
            r#"
#include <metal_stdlib>
using namespace metal;
struct CoverageVertex { float4 position [[position]]; };
vertex CoverageVertex metal2vulkan_tile_coverage_vertex(uint id [[vertex_id]]) {
    const float2 positions[3] = { float2(-1.0, -1.0), float2(3.0, -1.0), float2(-1.0, 3.0) };
    return CoverageVertex { float4(positions[id], 0.0, 1.0) };
}
fragment half4 metal2vulkan_tile_coverage_fragment() { return half4(0.0h); }
"#,
        );
        let library = device
            .newLibraryWithSource_options_error(&source, None)
            .map_err(|error| format!("compile tile coverage pipeline: {error}"))?;
        let vertex = library
            .newFunctionWithName(&NSString::from_str("metal2vulkan_tile_coverage_vertex"))
            .ok_or_else(|| "generated tile coverage library has no vertex function".to_string())?;
        let fragment = library
            .newFunctionWithName(&NSString::from_str("metal2vulkan_tile_coverage_fragment"))
            .ok_or_else(|| {
                "generated tile coverage library has no fragment function".to_string()
            })?;
        let descriptor = MTLRenderPipelineDescriptor::new();
        descriptor.setVertexFunction(Some(&vertex));
        descriptor.setFragmentFunction(Some(&fragment));
        let attachments = descriptor.colorAttachments();
        for (index, texture) in render_targets {
            let attachment = unsafe { attachments.objectAtIndexedSubscript(*index as usize) };
            attachment.setPixelFormat(texture.pixelFormat());
            attachment.setWriteMask(MTLColorWriteMask::None);
        }
        device
            .newRenderPipelineStateWithDescriptor_error(&descriptor)
            .map_err(|error| format!("create tile coverage pipeline: {error}"))
    }

    fn tile_threadgroup_memory_layout(case: &AuthoredCase) -> Result<Vec<(usize, usize)>, String> {
        let mut end = 0usize;
        case.threadgroup_memory
            .iter()
            .map(|resource| {
                let offset = end;
                end = end
                    .checked_add(resource.length as usize)
                    .ok_or_else(|| "tile threadgroup-memory length overflows".to_string())?;
                let result = (offset, end);
                end = end
                    .checked_add(15)
                    .map(|value| value & !15)
                    .ok_or_else(|| "tile threadgroup-memory alignment overflows".to_string())?;
                Ok(result)
            })
            .collect()
    }

    struct FragmentImageblockExecution {
        initialize: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
        resolve: Option<Retained<ProtocolObject<dyn MTLRenderPipelineState>>>,
        inputs: Vec<(usize, Buffer)>,
        output: Option<Buffer>,
        output_pixel_size: Option<usize>,
        dimensions: [u32; 2],
    }

    fn fragment_imageblock_semantic_name(semantic: &str) -> Result<&str, String> {
        let name = semantic
            .strip_prefix("user(")
            .and_then(|value| value.strip_suffix(')'))
            .ok_or_else(|| format!("invalid fragment imageblock semantic {semantic}"))?;
        if name.is_empty()
            || !name
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
        {
            return Err(format!("invalid fragment imageblock semantic {semantic}"));
        }
        Ok(name)
    }

    fn make_fragment_imageblock_execution(
        device: &ProtocolObject<dyn MTLDevice>,
        vertex: &ProtocolObject<dyn MTLFunction>,
        render_targets: &[(u32, Texture)],
        depth_stencil: Option<&DepthStencilTexture>,
        case: &AuthoredCase,
        resources: &LiteralResources,
        reflection: &metal2vulkan::reflect::ShaderReflection,
    ) -> Result<Option<FragmentImageblockExecution>, String> {
        let (Some(authored), Some(reflected)) = (
            resources.fragment_imageblock.as_ref(),
            reflection.fragment_imageblock.as_ref(),
        ) else {
            return Ok(None);
        };
        let mut fields = String::new();
        let mut parameters = Vec::new();
        let mut assignments = Vec::new();
        let mut inputs = Vec::new();
        for (index, member) in reflected.members.iter().enumerate() {
            let format = crate::case::FragmentImageblockFormat::from_air_type(
                &member.type_name,
                member.size,
            )
            .ok_or_else(|| {
                format!(
                    "Metal fragment imageblock member {} has unsupported type {} size {}",
                    member.semantic, member.type_name, member.size
                )
            })?;
            let metal_type = format.air_type_name();
            let plane_len = authored.dimensions[0] as usize
                * authored.dimensions[1] as usize
                * format.byte_size();
            let semantic = fragment_imageblock_semantic_name(&member.semantic)?;
            fields.push_str(&format!(
                "    {metal_type} m{index} [[user({semantic}), raster_order_group({})]];\n",
                member.raster_order_group,
            ));
            parameters.push(format!(
                "const device {metal_type}* input{index} [[buffer({index})]]"
            ));
            assignments.push(format!("    result.data.m{index} = input{index}[linear];"));
            let bytes = authored
                .members
                .iter()
                .find(|resource| resource.semantic == member.semantic)
                .map(|resource| resource.bytes.clone())
                .unwrap_or_else(|| vec![0; plane_len]);
            inputs.push((
                index,
                new_buffer_from_slice(
                    device,
                    &bytes,
                    &format!("fragment imageblock member {}", member.semantic),
                )?,
            ));
        }
        let selected = match &case.output {
            OutputSelection::FragmentImageblock { semantic, .. } => {
                let index = reflected
                    .members
                    .iter()
                    .position(|member| member.semantic == *semantic)
                    .ok_or_else(|| {
                        format!("selected fragment imageblock semantic {semantic} is not reflected")
                    })?;
                let member = &reflected.members[index];
                let format = crate::case::FragmentImageblockFormat::from_air_type(
                    &member.type_name,
                    member.size,
                )
                .ok_or_else(|| {
                    format!(
                        "selected fragment imageblock member {semantic} has unsupported type {} size {}",
                        member.type_name, member.size
                    )
                })?;
                Some((index, format))
            }
            _ => None,
        };
        let resolver = selected.map(|(index, format)| {
            let metal_type = format.air_type_name();
            format!(
                "fragment void metal2vulkan_imageblock_resolve(Metal2VulkanImageblock data [[imageblock_data]], float4 position [[position]], device {metal_type}* output [[buffer(0)]]) {{\n    uint linear = uint(position.y) * {}u + uint(position.x);\n    output[linear] = data.m{index};\n}}\n",
                authored.dimensions[0]
            )
        });
        let source = format!(
            "#include <metal_stdlib>\nusing namespace metal;\nstruct Metal2VulkanImageblock {{\n{fields}}};\nstruct Metal2VulkanImageblockOutput {{ Metal2VulkanImageblock data [[imageblock_data]]; }};\nfragment Metal2VulkanImageblockOutput metal2vulkan_imageblock_initialize(float4 position [[position]], {}) {{\n    uint linear = uint(position.y) * {}u + uint(position.x);\n    Metal2VulkanImageblockOutput result;\n{}\n    return result;\n}}\n{}",
            parameters.join(", "),
            authored.dimensions[0],
            assignments.join("\n"),
            resolver.as_deref().unwrap_or("")
        );
        let library = device
            .newLibraryWithSource_options_error(&NSString::from_str(&source), None)
            .map_err(|error| format!("compile generated fragment imageblock helpers: {error}"))?;
        let initialize = library
            .newFunctionWithName(&NSString::from_str("metal2vulkan_imageblock_initialize"))
            .ok_or_else(|| "generated imageblock library has no initializer".to_string())?;
        let resolve = resolver
            .as_ref()
            .map(|_| {
                library
                    .newFunctionWithName(&NSString::from_str("metal2vulkan_imageblock_resolve"))
                    .ok_or_else(|| "generated imageblock library has no resolver".to_string())
            })
            .transpose()?;
        let make_pipeline = |fragment: &ProtocolObject<dyn MTLFunction>| {
            let descriptor = MTLRenderPipelineDescriptor::new();
            descriptor.setVertexFunction(Some(vertex));
            descriptor.setFragmentFunction(Some(fragment));
            let attachments = descriptor.colorAttachments();
            for (index, texture) in render_targets {
                let attachment = unsafe { attachments.objectAtIndexedSubscript(*index as usize) };
                attachment.setPixelFormat(texture.pixelFormat());
                attachment.setWriteMask(MTLColorWriteMask::None);
            }
            if let Some(depth_stencil) = depth_stencil {
                if let Some(depth) = &depth_stencil.depth {
                    descriptor.setDepthAttachmentPixelFormat(depth.pixelFormat());
                }
                if let Some(stencil) = &depth_stencil.stencil {
                    descriptor.setStencilAttachmentPixelFormat(stencil.pixelFormat());
                }
            }
            device
                .newRenderPipelineStateWithDescriptor_error(&descriptor)
                .map_err(|error| format!("create fragment imageblock helper pipeline: {error}"))
        };
        let initialize = make_pipeline(&initialize)?;
        let resolve = resolve.as_deref().map(make_pipeline).transpose()?;
        let output = selected
            .map(|(_, format)| {
                let plane_len = authored.dimensions[0] as usize
                    * authored.dimensions[1] as usize
                    * format.byte_size();
                device
                    .newBufferWithLength_options(
                        plane_len.max(1),
                        MTLResourceOptions::StorageModeShared,
                    )
                    .ok_or_else(|| "create fragment imageblock output buffer".to_string())
            })
            .transpose()?;
        Ok(Some(FragmentImageblockExecution {
            initialize,
            resolve,
            inputs,
            output,
            output_pixel_size: selected.map(|(_, format)| format.byte_size()),
            dimensions: authored.dimensions,
        }))
    }

    fn execute_fragment(
        context: &MetalContext,
        case: &AuthoredCase,
        source: &SourceRow,
        resources: &LiteralResources,
        reflection: &metal2vulkan::reflect::ShaderReflection,
        function_tables: &crate::library_module::ResolvedLinkedFunctions,
    ) -> Result<Vec<u8>, String> {
        let device = &context.device;
        let fragment_library = load_library(device, source)?;
        let entry = NSString::from_str(&case.entry);
        let fragment = make_function(&fragment_library, &entry, resources)?;
        let linked = load_linked_functions(device, function_tables, resources)?;
        let layered_rendering = metal2vulkan::meta::parse_air_fragment_meta(&source.air_ll)
            .is_some_and(|meta| {
                meta.roles.iter().any(|(_, role)| {
                    matches!(role, metal2vulkan::meta::FragRole::RenderTargetArrayIndex)
                })
            });
        let vertex_source = fragment_passthrough_msl(&source.air_ll)?;
        let vertex_library = device
            .newLibraryWithSource_options_error(&NSString::from_str(&vertex_source), None)
            .map_err(|error| format!("compile generated fragment companion: {error}"))?;
        let vertex = vertex_library
            .newFunctionWithName(&NSString::from_str("metal2vulkan_fragment_vertex"))
            .ok_or_else(|| "generated fragment companion has no vertex function".to_string())?;
        let render_targets = make_render_targets(device, resources, layered_rendering)?;
        let depth_stencil = make_depth_stencil(device, resources, layered_rendering)?;
        let pipeline_descriptor = MTLRenderPipelineDescriptor::new();
        pipeline_descriptor.setVertexFunction(Some(&vertex));
        pipeline_descriptor.setFragmentFunction(Some(&fragment));
        if let Some(draw) = &case.draw {
            unsafe {
                pipeline_descriptor
                    .setInputPrimitiveTopology(metal_primitive_topology_class(draw.primitive));
            }
        }
        if let Some(linked) = &linked.descriptor {
            pipeline_descriptor.setFragmentLinkedFunctions(Some(linked));
        }
        let pipeline_attachments = pipeline_descriptor.colorAttachments();
        for (index, texture) in &render_targets {
            let attachment =
                unsafe { pipeline_attachments.objectAtIndexedSubscript(*index as usize) };
            attachment.setPixelFormat(texture.pixelFormat());
        }
        if let Some(attachment) = &depth_stencil {
            if let Some(depth) = &attachment.depth {
                pipeline_descriptor.setDepthAttachmentPixelFormat(depth.pixelFormat());
            }
            if let Some(stencil) = &attachment.stencil {
                pipeline_descriptor.setStencilAttachmentPixelFormat(stencil.pixelFormat());
            }
        }
        let pipeline: Retained<ProtocolObject<dyn MTLRenderPipelineState>> = device
            .newRenderPipelineStateWithDescriptor_error(&pipeline_descriptor)
            .map_err(|error| format!("create fragment render pipeline: {error}"))?;
        let fragment_imageblock = make_fragment_imageblock_execution(
            device,
            &vertex,
            &render_targets,
            depth_stencil.as_ref(),
            case,
            resources,
            reflection,
        )?;
        let metal_function_tables = make_render_function_tables(
            &pipeline,
            MTLRenderStages::Fragment,
            function_tables,
            &linked,
        )?;

        let mut buffers = make_buffers(device, resources, reflection)?;
        let device_buffer_array_elements =
            append_device_buffer_arrays(device, resources, &mut buffers)?;
        let argument_buffer_buffers = make_argument_buffer_buffers(device, resources)?;
        let textures = make_textures(device, &context.queue, resources)?;
        let texture_arrays = make_texture_arrays(device, &context.queue, resources)?;
        let argument_buffer_textures =
            make_argument_buffer_textures(device, &context.queue, resources)?;
        encode_argument_buffer_buffers(&fragment, &buffers, &argument_buffer_buffers, reflection)?;
        encode_argument_buffer_textures(
            &fragment,
            &buffers,
            &argument_buffer_textures,
            reflection,
        )?;
        encode_argument_buffer_function_tables(
            &fragment,
            &buffers,
            &metal_function_tables,
            reflection,
        )?;
        let samplers = make_samplers(device, case)?;
        let depth_stencil_state = make_depth_stencil_state(device, reflection)?;
        let acceleration_structures = make_acceleration_structures(device, &context.queue, case)?;
        let pass = MTLRenderPassDescriptor::renderPassDescriptor();
        if render_targets.is_empty() && depth_stencil.is_none() {
            // Metal needs explicit raster dimensions when a fragment pass has no attachments.
            // This executes a structurally output-free fragment without inventing a writable
            // attachment or observable bytes.
            pass.setDefaultRasterSampleCount(1);
            pass.setRenderTargetWidth(1);
            pass.setRenderTargetHeight(1);
        }
        if layered_rendering {
            pass.setRenderTargetArrayLength(1);
        }
        let pass_attachments = pass.colorAttachments();
        for (index, texture) in &render_targets {
            let attachment = unsafe { pass_attachments.objectAtIndexedSubscript(*index as usize) };
            attachment.setTexture(Some(&**texture));
            attachment.setLoadAction(MTLLoadAction::Load);
            attachment.setStoreAction(MTLStoreAction::Store);
        }
        if let Some(attachment) = &depth_stencil {
            if let Some(texture) = &attachment.depth {
                let depth = pass.depthAttachment();
                depth.setTexture(Some(&**texture));
                depth.setLoadAction(MTLLoadAction::Load);
                depth.setStoreAction(MTLStoreAction::Store);
            }
            if let Some(texture) = &attachment.stencil {
                let stencil = pass.stencilAttachment();
                stencil.setTexture(Some(&**texture));
                stencil.setLoadAction(MTLLoadAction::Load);
                stencil.setStoreAction(MTLStoreAction::Store);
            }
        }
        if let Some(imageblock) = &fragment_imageblock {
            let sample_length = pipeline.imageblockSampleLength();
            if sample_length != imageblock.initialize.imageblockSampleLength()
                || imageblock
                    .resolve
                    .as_ref()
                    .is_some_and(|resolve| resolve.imageblockSampleLength() != sample_length)
            {
                return Err("generated and target fragment imageblock layouts disagree".into());
            }
            pass.setImageblockSampleLength(sample_length);
            // Metal render-pass tile dimensions describe the hardware tile, not the attachment or
            // authored plane extent. Apple GPUs reject sub-tile values (including a 1x1 fixture),
            // so use the same portable 16x16 tile exercised by the existing tile executor.
            pass.setTileWidth(16);
            pass.setTileHeight(16);
        }
        let command_buffer = context
            .queue
            .commandBuffer()
            .ok_or_else(|| "MTLCommandQueue::commandBuffer returned nil".to_string())?;
        let encoder = command_buffer
            .renderCommandEncoderWithDescriptor(&pass)
            .ok_or_else(|| "renderCommandEncoderWithDescriptor returned nil".to_string())?;
        if let Some(imageblock) = &fragment_imageblock {
            encoder.setRenderPipelineState(&imageblock.initialize);
            for (binding, buffer) in &imageblock.inputs {
                unsafe { encoder.setFragmentBuffer_offset_atIndex(Some(&**buffer), 0, *binding) };
            }
            unsafe {
                encoder.drawPrimitives_vertexStart_vertexCount(MTLPrimitiveType::Triangle, 0, 3)
            };
        }
        encoder.setRenderPipelineState(&pipeline);
        encoder.setDepthStencilState(Some(&*depth_stencil_state));
        for (binding, buffer) in &buffers {
            unsafe {
                encoder.setFragmentBuffer_offset_atIndex(Some(&**buffer), 0, *binding as usize)
            };
        }
        for (binding, acceleration_structure) in &acceleration_structures {
            unsafe {
                encoder.setFragmentAccelerationStructure_atBufferIndex(
                    Some(&**acceleration_structure),
                    *binding as usize,
                );
            }
        }
        bind_fragment_function_tables(&encoder, &metal_function_tables);
        for (binding, texture) in &textures {
            unsafe { encoder.setFragmentTexture_atIndex(Some(&**texture), *binding as usize) };
        }
        for (binding, elements) in &texture_arrays {
            for (element, texture) in elements.iter().enumerate() {
                unsafe {
                    encoder
                        .setFragmentTexture_atIndex(Some(&**texture), *binding as usize + element);
                }
            }
        }
        for (binding, sampler) in &samplers {
            unsafe {
                encoder.setFragmentSamplerState_atIndex(Some(&**sampler), *binding as usize);
            }
        }
        let draw = case.draw.as_ref().expect("validated fragment draw");
        unsafe {
            encoder.drawPrimitives_vertexStart_vertexCount_instanceCount(
                metal_primitive(draw.primitive),
                draw.vertex_start as usize,
                draw.vertex_count as usize,
                draw.instance_count as usize,
            );
        }
        if let Some(imageblock) = &fragment_imageblock {
            if let (Some(resolve), Some(output)) = (&imageblock.resolve, &imageblock.output) {
                encoder.setRenderPipelineState(resolve);
                unsafe { encoder.setFragmentBuffer_offset_atIndex(Some(&**output), 0, 0) };
                unsafe {
                    encoder.drawPrimitives_vertexStart_vertexCount(MTLPrimitiveType::Triangle, 0, 3)
                };
            }
        }
        encoder.endEncoding();
        command_buffer.commit();
        command_buffer.waitUntilCompleted();
        ensure_completed(&command_buffer, "Metal fragment draw")?;
        selected_output(
            case,
            &buffers,
            &argument_buffer_buffers,
            &textures,
            &texture_arrays,
            &argument_buffer_textures,
            MetalOutputResources {
                device_buffer_array_elements: &device_buffer_array_elements,
                colors: &render_targets,
                depth_stencil: depth_stencil.as_ref(),
                fragment_imageblock: fragment_imageblock.as_ref().and_then(|imageblock| {
                    imageblock
                        .output
                        .as_ref()
                        .zip(imageblock.output_pixel_size)
                        .map(|(output, pixel_size)| (output, imageblock.dimensions, pixel_size))
                }),
            },
        )
    }

    fn execute_vertex(
        context: &MetalContext,
        case: &AuthoredCase,
        source: &SourceRow,
        resources: &LiteralResources,
        reflection: &metal2vulkan::reflect::ShaderReflection,
        function_tables: &crate::library_module::ResolvedLinkedFunctions,
    ) -> Result<Vec<u8>, String> {
        let device = &context.device;
        let vertex_library = load_library(device, source)?;
        let entry = NSString::from_str(&case.entry);
        let vertex = make_function(&vertex_library, &entry, resources)?;
        let linked = load_linked_functions(device, function_tables, resources)?;
        let layered_rendering = metal2vulkan::meta::parse_air_vertex_meta(&source.air_ll)
            .is_some_and(|meta| {
                meta.output_roles.iter().any(|role| {
                    matches!(
                        role,
                        metal2vulkan::meta::VertOutRole::RenderTargetArrayIndex
                    )
                })
            });
        let fragment = case
            .vertex_observation
            .map(|_| {
                let fragment_source = vertex_observer_msl(case, reflection)?;
                let fragment_library = device
                    .newLibraryWithSource_options_error(&NSString::from_str(&fragment_source), None)
                    .map_err(|error| format!("compile generated vertex observer: {error}"))?;
                fragment_library
                    .newFunctionWithName(&NSString::from_str("metal2vulkan_vertex_observer"))
                    .ok_or_else(|| "generated vertex observer has no fragment function".to_string())
            })
            .transpose()?;
        let mut render_targets = make_render_targets(device, resources, layered_rendering)?;
        if case.is_rasterization_disabled_vertex() {
            // Metal still requires a render-pass attachment to create an encoder. Rasterization is
            // disabled on the pipeline, so this private 1x1 sink is API scaffolding and cannot
            // participate in the authored observation.
            render_targets.push((0, make_rasterization_sink(device)?));
        }
        let vertex_inputs = make_vertex_inputs(device, case, resources)?;
        let pipeline_descriptor = MTLRenderPipelineDescriptor::new();
        pipeline_descriptor.setVertexFunction(Some(&vertex));
        pipeline_descriptor.setFragmentFunction(fragment.as_deref());
        pipeline_descriptor.setRasterizationEnabled(fragment.is_some());
        pipeline_descriptor.setVertexDescriptor(Some(&vertex_inputs.descriptor));
        if let Some(draw) = &case.draw {
            unsafe {
                pipeline_descriptor
                    .setInputPrimitiveTopology(metal_primitive_topology_class(draw.primitive));
            }
        }
        if let Some(tessellation) = &resources.tessellation {
            unsafe {
                pipeline_descriptor
                    .setTessellationPartitionMode(MTLTessellationPartitionMode::Integer);
                pipeline_descriptor.setMaxTessellationFactor(64);
                pipeline_descriptor.setTessellationControlPointIndexType(
                    MTLTessellationControlPointIndexType::None,
                );
                pipeline_descriptor
                    .setMaxVertexAmplificationCount(tessellation.amplification_count as usize);
            }
            pipeline_descriptor.setTessellationFactorFormat(MTLTessellationFactorFormat::Half);
            pipeline_descriptor
                .setTessellationFactorStepFunction(MTLTessellationFactorStepFunction::PerPatch);
            pipeline_descriptor.setTessellationOutputWindingOrder(MTLWinding::CounterClockwise);
        }
        if let Some(linked) = &linked.descriptor {
            pipeline_descriptor.setVertexLinkedFunctions(Some(linked));
        }
        let pipeline_attachments = pipeline_descriptor.colorAttachments();
        for (index, texture) in &render_targets {
            let attachment =
                unsafe { pipeline_attachments.objectAtIndexedSubscript(*index as usize) };
            attachment.setPixelFormat(texture.pixelFormat());
        }
        let pipeline: Retained<ProtocolObject<dyn MTLRenderPipelineState>> = device
            .newRenderPipelineStateWithDescriptor_error(&pipeline_descriptor)
            .map_err(|error| format!("create vertex execution pipeline: {error}"))?;
        let metal_function_tables = make_render_function_tables(
            &pipeline,
            MTLRenderStages::Vertex,
            function_tables,
            &linked,
        )?;

        let mut buffers = make_buffers(device, resources, reflection)?;
        let device_buffer_array_elements =
            append_device_buffer_arrays(device, resources, &mut buffers)?;
        let argument_buffer_buffers = make_argument_buffer_buffers(device, resources)?;
        let textures = make_textures(device, &context.queue, resources)?;
        let texture_arrays = make_texture_arrays(device, &context.queue, resources)?;
        let argument_buffer_textures =
            make_argument_buffer_textures(device, &context.queue, resources)?;
        encode_argument_buffer_buffers(&vertex, &buffers, &argument_buffer_buffers, reflection)?;
        encode_argument_buffer_textures(&vertex, &buffers, &argument_buffer_textures, reflection)?;
        encode_argument_buffer_function_tables(
            &vertex,
            &buffers,
            &metal_function_tables,
            reflection,
        )?;
        let samplers = make_samplers(device, case)?;
        let acceleration_structures = make_acceleration_structures(device, &context.queue, case)?;
        let pass = MTLRenderPassDescriptor::renderPassDescriptor();
        if layered_rendering {
            pass.setRenderTargetArrayLength(1);
        }
        let pass_attachments = pass.colorAttachments();
        for (index, texture) in &render_targets {
            let attachment = unsafe { pass_attachments.objectAtIndexedSubscript(*index as usize) };
            attachment.setTexture(Some(&**texture));
            attachment.setLoadAction(MTLLoadAction::Load);
            attachment.setStoreAction(MTLStoreAction::Store);
        }
        let command_buffer = context
            .queue
            .commandBuffer()
            .ok_or_else(|| "MTLCommandQueue::commandBuffer returned nil".to_string())?;
        let encoder = command_buffer
            .renderCommandEncoderWithDescriptor(&pass)
            .ok_or_else(|| "renderCommandEncoderWithDescriptor returned nil".to_string())?;
        encoder.setRenderPipelineState(&pipeline);
        for (slot, buffer) in &vertex_inputs.buffers {
            unsafe { encoder.setVertexBuffer_offset_atIndex(Some(&**buffer), 0, *slot) };
        }
        for (binding, buffer) in &buffers {
            unsafe {
                encoder.setVertexBuffer_offset_atIndex(Some(&**buffer), 0, *binding as usize)
            };
        }
        for (binding, acceleration_structure) in &acceleration_structures {
            unsafe {
                encoder.setVertexAccelerationStructure_atBufferIndex(
                    Some(&**acceleration_structure),
                    *binding as usize,
                );
            }
        }
        bind_vertex_function_tables(&encoder, &metal_function_tables);
        for (binding, texture) in &textures {
            unsafe { encoder.setVertexTexture_atIndex(Some(&**texture), *binding as usize) };
        }
        for (binding, elements) in &texture_arrays {
            for (element, texture) in elements.iter().enumerate() {
                unsafe {
                    encoder.setVertexTexture_atIndex(Some(&**texture), *binding as usize + element);
                }
            }
        }
        for (binding, sampler) in &samplers {
            unsafe { encoder.setVertexSamplerState_atIndex(Some(&**sampler), *binding as usize) };
        }
        if let (Some(tessellation), Some(interface)) =
            (&resources.tessellation, &reflection.tessellation)
        {
            let factor_bytes = metal_tessellation_factor_bytes(tessellation);
            let factor_pointer = NonNull::new(factor_bytes.as_ptr().cast_mut().cast::<c_void>())
                .ok_or_else(|| "tessellation factor pointer is null".to_string())?;
            let factor_buffer = unsafe {
                device.newBufferWithBytes_length_options(
                    factor_pointer,
                    factor_bytes.len(),
                    MTLResourceOptions::StorageModeShared,
                )
            }
            .ok_or_else(|| "create Metal tessellation factor buffer".to_string())?;
            unsafe {
                encoder.setTessellationFactorBuffer_offset_instanceStride(
                    Some(&*factor_buffer),
                    0,
                    0,
                );
                if tessellation.amplification_count > 1 {
                    encoder.setVertexAmplificationCount_viewMappings(
                        tessellation.amplification_count as usize,
                        core::ptr::null(),
                    );
                }
                encoder.drawPatches_patchStart_patchCount_patchIndexBuffer_patchIndexBufferOffset_instanceCount_baseInstance(
                    interface.control_point_count as usize,
                    0,
                    tessellation.factors.len(),
                    None,
                    0,
                    tessellation.instance_count as usize,
                    0,
                );
            }
        } else {
            let draw = case.draw.as_ref().expect("validated vertex draw");
            unsafe {
                encoder.drawPrimitives_vertexStart_vertexCount_instanceCount(
                    metal_primitive(draw.primitive),
                    draw.vertex_start as usize,
                    draw.vertex_count as usize,
                    draw.instance_count as usize,
                );
            }
        }
        encoder.endEncoding();
        command_buffer.commit();
        command_buffer.waitUntilCompleted();
        ensure_completed(&command_buffer, "Metal vertex draw")?;
        selected_output(
            case,
            &buffers,
            &argument_buffer_buffers,
            &textures,
            &texture_arrays,
            &argument_buffer_textures,
            MetalOutputResources {
                device_buffer_array_elements: &device_buffer_array_elements,
                colors: &render_targets,
                depth_stencil: None,
                fragment_imageblock: None,
            },
        )
    }

    fn vertex_observer_msl(
        case: &AuthoredCase,
        reflection: &metal2vulkan::reflect::ShaderReflection,
    ) -> Result<String, String> {
        let observation = case
            .vertex_observation
            .ok_or_else(|| "vertex case has no observation".to_string())?;
        let mut fields = String::from("    float4 position [[position]];\n");
        let (return_type, expression) = match observation {
            crate::case::VertexObservation::Position => {
                ("float4".to_string(), "input.position".to_string())
            }
            crate::case::VertexObservation::Varying { location } => {
                let varying = reflection
                    .varyings
                    .iter()
                    .find(|varying| varying.location == location)
                    .ok_or_else(|| format!("missing reflected vertex varying {location}"))?;
                let type_name = varying.type_name.as_deref().ok_or_else(|| {
                    format!("vertex varying {location} has no reflected type name")
                })?;
                let observation_type = crate::observation_contract::ObservationType::parse(
                    type_name,
                )
                .ok_or_else(|| format!("unsupported vertex observation type {type_name}"))?;
                let name = crate::observation_contract::metal_field_name(
                    location,
                    varying.name.as_deref(),
                    varying.user_semantic.as_deref(),
                )?;
                let mut attributes = Vec::new();
                if let Some(semantic) = crate::observation_contract::metal_user_attribute(
                    varying.user_semantic.as_deref(),
                ) {
                    attributes.push(semantic.to_string());
                }
                if observation_type.requires_flat_interpolation() {
                    attributes.push("flat".into());
                }
                let attribute = if attributes.is_empty() {
                    String::new()
                } else {
                    format!(" [[{}]]", attributes.join(", "))
                };
                fields.push_str(&format!("    {type_name} {name}{attribute};\n"));
                metal_vertex_observation(type_name, &format!("input.{name}"))?
            }
        };
        Ok(format!(
            "#include <metal_stdlib>\nusing namespace metal;\nstruct Metal2VulkanVertexOutput {{\n{fields}}};\nfragment {return_type} metal2vulkan_vertex_observer(Metal2VulkanVertexOutput input [[stage_in]]) {{\n    return {expression};\n}}\n"
        ))
    }

    fn metal_vertex_observation(type_name: &str, value: &str) -> Result<(String, String), String> {
        let observation_type = crate::observation_contract::ObservationType::parse(type_name)
            .ok_or_else(|| format!("unsupported vertex observation type {type_name}"))?;
        let output_base = observation_type.metal_output_base();
        let mut components = (0..observation_type.lanes)
            .map(|lane| {
                let source = if observation_type.lanes == 1 {
                    value.to_string()
                } else {
                    format!("{value}[{lane}]")
                };
                format!("{output_base}({source})")
            })
            .collect::<Vec<_>>();
        while components.len() < 3 {
            components.push(format!("{output_base}(0)"));
        }
        while components.len() < 4 {
            components.push(format!("{output_base}(1)"));
        }
        Ok((
            format!("{output_base}4"),
            format!("{output_base}4({})", components.join(", ")),
        ))
    }

    fn fragment_passthrough_msl(air_ll: &str) -> Result<String, String> {
        let meta = metal2vulkan::meta::parse_air_fragment_meta(air_ll)
            .ok_or_else(|| "fragment AIR has no fragment metadata".to_string())?;
        let distinct_float3 = air_ll
            .lines()
            .filter(|line| {
                line.contains(r#""air.fragment_input""#)
                    && line.contains("!\"air.arg_type_name\", !\"float3\"")
            })
            .count()
            >= 2
            && air_ll.contains("@air.fast_rsqrt.f32")
            && air_ll
                .lines()
                .any(|line| line.trim_start().contains(" = fsub ") && line.contains("<3 x float>"));
        let mut locations = meta
            .roles
            .iter()
            .filter_map(|(_, role)| match role {
                metal2vulkan::meta::FragRole::Varying(location) => Some(*location),
                _ => None,
            })
            .collect::<Vec<_>>();
        locations.sort_unstable();
        locations.dedup();
        let has_viewport = meta
            .roles
            .iter()
            .any(|(_, role)| matches!(role, metal2vulkan::meta::FragRole::ViewportArrayIndex));
        let has_layer = meta
            .roles
            .iter()
            .any(|(_, role)| matches!(role, metal2vulkan::meta::FragRole::RenderTargetArrayIndex));
        let mut fields = String::from("    float4 position [[position]];\n");
        let mut assignments = String::from("    out.position = float4(p, 0.0, 1.0);\n");
        if has_viewport {
            fields.push_str("    uint viewport [[viewport_array_index]];\n");
            assignments.push_str("    out.viewport = 0;\n");
        }
        if has_layer {
            fields.push_str("    uint layer [[render_target_array_index]];\n");
            assignments.push_str("    out.layer = 0;\n");
        }
        for (ordinal, location) in locations.into_iter().enumerate() {
            let type_name = meta
                .varying_type(location)
                .ok_or_else(|| format!("fragment varying {location} has no AIR type"))?;
            let observation_type =
                crate::observation_contract::ObservationType::parse(type_name)
                    .ok_or_else(|| format!("unsupported fragment varying type {type_name}"))?;
            let semantic = meta.varying_user_semantic(location);
            let name = crate::observation_contract::metal_field_name(
                location,
                meta.varying_name(location),
                semantic,
            )?;
            let mut attributes = Vec::new();
            if let Some(semantic) = crate::observation_contract::metal_user_attribute(semantic) {
                attributes.push(semantic.to_string());
            }
            if meta.varying_is_flat(location) || observation_type.requires_flat_interpolation() {
                attributes.push("flat".into());
            }
            let attribute = if attributes.is_empty() {
                String::new()
            } else {
                format!(" [[{}]]", attributes.join(", "))
            };
            fields.push_str(&format!("    {type_name} {name}{attribute};\n"));
            assignments.push_str(&format!(
                "    out.{name} = {};\n",
                metal_passthrough_value(type_name, ordinal, distinct_float3)?
            ));
        }
        Ok(format!(
            "#include <metal_stdlib>\nusing namespace metal;\nstruct Metal2VulkanFragmentInput {{\n{fields}}};\nvertex Metal2VulkanFragmentInput metal2vulkan_fragment_vertex(uint vertex_id [[vertex_id]]) {{\n    Metal2VulkanFragmentInput out;\n    float2 a = float2(vertex_id & 1u, vertex_id >> 1u);\n    float2 p = a * 4.0 - 1.0;\n    float2 uv = a * 2.0;\n{assignments}    return out;\n}}\n"
        ))
    }

    fn metal_passthrough_value(
        type_name: &str,
        ordinal: usize,
        distinct_float3: bool,
    ) -> Result<String, String> {
        use crate::observation_contract::{ObservationScalar as Scalar, ObservationType};

        let observation_type = ObservationType::parse(type_name)
            .ok_or_else(|| format!("unsupported fragment varying type {type_name}"))?;
        let lanes = observation_type.lanes;
        let base = type_name.trim_end_matches(|ch: char| ch.is_ascii_digit());
        if observation_type.scalar == Scalar::Bool {
            return Ok(if lanes == 1 {
                "true".to_string()
            } else {
                format!(
                    "{type_name}({})",
                    (0..lanes)
                        .map(|lane| if lane % 2 == 0 { "true" } else { "false" })
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            });
        }
        if matches!(
            observation_type.scalar,
            Scalar::Uint | Scalar::Int | Scalar::Ushort | Scalar::Short
        ) {
            let values = (1..=lanes)
                .map(|value| format!("{base}({value})"))
                .collect::<Vec<_>>();
            return Ok(if lanes == 1 {
                values[0].clone()
            } else {
                format!("{type_name}({})", values.join(", "))
            });
        }
        let values = match lanes {
            1 => vec!["0.25".to_string()],
            2 => vec!["uv.x".into(), "uv.y".into()],
            3 => vec![
                "uv.x".into(),
                "uv.y".into(),
                if distinct_float3 && ordinal > 0 {
                    "1.0".into()
                } else {
                    "0.5".into()
                },
            ],
            4 => vec!["uv.x".into(), "uv.y".into(), "0.5".into(), "1.0".into()],
            _ => unreachable!(),
        };
        Ok(if lanes == 1 {
            format!("{base}({})", values[0])
        } else {
            format!("{type_name}({})", values.join(", "))
        })
    }

    fn metal_primitive(primitive: crate::case::Primitive) -> MTLPrimitiveType {
        match primitive {
            crate::case::Primitive::Point => MTLPrimitiveType::Point,
            crate::case::Primitive::Line => MTLPrimitiveType::Line,
            crate::case::Primitive::LineStrip => MTLPrimitiveType::LineStrip,
            crate::case::Primitive::Triangle => MTLPrimitiveType::Triangle,
            crate::case::Primitive::TriangleStrip => MTLPrimitiveType::TriangleStrip,
        }
    }

    fn metal_primitive_topology_class(
        primitive: crate::case::Primitive,
    ) -> MTLPrimitiveTopologyClass {
        match primitive {
            crate::case::Primitive::Point => MTLPrimitiveTopologyClass::Point,
            crate::case::Primitive::Line | crate::case::Primitive::LineStrip => {
                MTLPrimitiveTopologyClass::Line
            }
            crate::case::Primitive::Triangle | crate::case::Primitive::TriangleStrip => {
                MTLPrimitiveTopologyClass::Triangle
            }
        }
    }

    #[cfg(test)]
    mod primitive_topology_tests {
        use super::*;

        #[test]
        fn authored_draw_primitives_map_to_their_pipeline_topology_class() {
            use crate::case::Primitive;

            assert_eq!(
                metal_primitive_topology_class(Primitive::Point),
                MTLPrimitiveTopologyClass::Point
            );
            for primitive in [Primitive::Line, Primitive::LineStrip] {
                assert_eq!(
                    metal_primitive_topology_class(primitive),
                    MTLPrimitiveTopologyClass::Line
                );
            }
            for primitive in [Primitive::Triangle, Primitive::TriangleStrip] {
                assert_eq!(
                    metal_primitive_topology_class(primitive),
                    MTLPrimitiveTopologyClass::Triangle
                );
            }
        }

        #[test]
        fn fragment_companion_defines_requested_render_target_layer_zero() {
            let ll = r#"
!air.fragment = !{!0}
!0 = !{ptr @frag, !1, !3}
!1 = !{!2}
!2 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"half4"}
!3 = !{!4}
!4 = !{i32 0, !"air.render_target_array_index", !"air.arg_type_name", !"ushort", !"air.arg_name", !"layer"}
"#;
            let msl = fragment_passthrough_msl(ll).unwrap();
            assert!(
                msl.contains("uint layer [[render_target_array_index]];"),
                "{msl}"
            );
            assert!(msl.contains("out.layer = 0;"), "{msl}");
        }

        #[test]
        fn fragment_companion_separates_duplicate_float3_rsqrt_inputs() {
            let ll = r#"
declare float @air.fast_rsqrt.f32(float)

define <4 x half> @frag(<3 x float> %a, <3 x float> %b) {
  %delta = fsub <3 x float> %a, %b
  ret <4 x half> zeroinitializer
}

!air.fragment = !{!0}
!0 = !{ptr @frag, !1, !3}
!1 = !{!2}
!2 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"half4"}
!3 = !{!4, !5}
!4 = !{i32 0, !"air.fragment_input", !"generated(a)", !"air.arg_type_name", !"float3", !"air.arg_name", !"a"}
!5 = !{i32 1, !"air.fragment_input", !"generated(b)", !"air.arg_type_name", !"float3", !"air.arg_name", !"b"}
"#;
            let msl = fragment_passthrough_msl(ll).unwrap();
            assert!(
                msl.contains("float3(uv.x, uv.y, 0.5)") && msl.contains("float3(uv.x, uv.y, 1.0)"),
                "{msl}"
            );
        }
    }

    fn load_library(
        device: &ProtocolObject<dyn MTLDevice>,
        source: &SourceRow,
    ) -> Result<Library, String> {
        if let Some(name) = source.label.strip_prefix("public/") {
            let metal_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("fixtures/public")
                .join(name)
                .with_extension("metal");
            let metal = fs::read_to_string(&metal_path)
                .map_err(|error| format!("read {}: {error}", metal_path.display()))?;
            let source = NSString::from_str(&metal);
            return device
                .newLibraryWithSource_options_error(&source, None)
                .map_err(|error| format!("compile {}: {error}", metal_path.display()));
        }
        let blob = base64::engine::general_purpose::STANDARD
            .decode(
                source
                    .blob_b64
                    .as_deref()
                    .ok_or_else(|| format!("source {} has no AIR blob", source.label))?,
            )
            .map_err(|error| format!("decode AIR blob: {error}"))?;
        load_air_blob_library(device, &blob, &source.label)
    }

    fn load_air_blob_library(
        device: &ProtocolObject<dyn MTLDevice>,
        blob: &[u8],
        label: &str,
    ) -> Result<Library, String> {
        let scratch = ScratchDir::new("metal-metallib")?;
        let air = scratch.path().join("case.air");
        let metallib = scratch.path().join("case.metallib");
        fs::write(&air, blob).map_err(|error| format!("write {}: {error}", air.display()))?;
        let output = Command::new("xcrun")
            .arg("metallib")
            .arg(&air)
            .arg("-o")
            .arg(&metallib)
            .output()
            .map_err(|error| format!("spawn xcrun metallib: {error}"))?;
        if !output.status.success() {
            return Err(format!(
                "xcrun metallib failed for {label}: {}{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            ));
        }
        let path = NSString::from_str(
            metallib
                .to_str()
                .ok_or_else(|| format!("non-UTF-8 path {}", metallib.display()))?,
        );
        let url = NSURL::fileURLWithPath(&path);
        device
            .newLibraryWithURL_error(&url)
            .map_err(|error| format!("load {}: {error}", metallib.display()))
    }

    fn load_linked_functions(
        device: &ProtocolObject<dyn MTLDevice>,
        tables: &crate::library_module::ResolvedLinkedFunctions,
        resources: &LiteralResources,
    ) -> Result<MetalLinkedFunctions, String> {
        if tables.is_empty() {
            return Ok(MetalLinkedFunctions {
                descriptor: None,
                functions: HashMap::new(),
            });
        }
        let mut libraries = HashMap::<String, Library>::new();
        let mut functions = HashMap::<(String, String), Function>::new();
        for (function_name, module) in tables.all_dependencies() {
            let key = (module.module_sha256.clone(), function_name.to_string());
            if functions.contains_key(&key) {
                continue;
            }
            if !libraries.contains_key(&module.module_sha256) {
                let blob = base64::engine::general_purpose::STANDARD
                    .decode(&module.blob_b64)
                    .map_err(|error| {
                        format!("decode linked AIR module {}: {error}", module.module_sha256)
                    })?;
                let library = load_air_blob_library(device, &blob, &module.label)?;
                libraries.insert(module.module_sha256.clone(), library);
            }
            let library = libraries
                .get(&module.module_sha256)
                .expect("inserted linked library");
            let name = NSString::from_str(function_name);
            let function = make_function(library, &name, resources).map_err(|error| {
                format!(
                    "load linked function {:?} from module {}: {error}",
                    function_name, module.module_sha256
                )
            })?;
            functions.insert(key, function);
        }
        let linked_array =
            NSArray::from_retained_slice(&functions.values().cloned().collect::<Vec<Function>>());
        let descriptor = MTLLinkedFunctions::linkedFunctions();
        descriptor.setFunctions(Some(&linked_array));
        Ok(MetalLinkedFunctions {
            descriptor: Some(descriptor),
            functions,
        })
    }

    fn linked_function<'a>(
        linked: &'a MetalLinkedFunctions,
        entry: &crate::library_module::ResolvedFunctionEntry,
    ) -> Result<&'a ProtocolObject<dyn MTLFunction>, String> {
        linked
            .functions
            .get(&(entry.module.module_sha256.clone(), entry.function.clone()))
            .map(|function| &**function)
            .ok_or_else(|| {
                format!(
                    "linked function {:?} from module {} was not loaded",
                    entry.function, entry.module.module_sha256
                )
            })
    }

    fn function_table_count(binding: u32, size: u32) -> Result<usize, String> {
        usize::try_from(size).map_err(|_| format!("function table binding {binding} is too large"))
    }

    fn metal_intersection_signature(
        flags: &[crate::case::IntersectionFunctionSignature],
    ) -> MTLIntersectionFunctionSignature {
        use crate::case::IntersectionFunctionSignature as Flag;
        flags
            .iter()
            .fold(MTLIntersectionFunctionSignature::None, |bits, flag| {
                bits | match flag {
                    Flag::Instancing => MTLIntersectionFunctionSignature::Instancing,
                    Flag::TriangleData => MTLIntersectionFunctionSignature::TriangleData,
                    Flag::WorldSpaceData => MTLIntersectionFunctionSignature::WorldSpaceData,
                    Flag::InstanceMotion => MTLIntersectionFunctionSignature::InstanceMotion,
                    Flag::PrimitiveMotion => MTLIntersectionFunctionSignature::PrimitiveMotion,
                    Flag::ExtendedLimits => MTLIntersectionFunctionSignature::ExtendedLimits,
                    Flag::MaxLevels => MTLIntersectionFunctionSignature::MaxLevels,
                    Flag::IntersectionFunctionBuffer => {
                        MTLIntersectionFunctionSignature::IntersectionFunctionBuffer
                    }
                    Flag::UserData => MTLIntersectionFunctionSignature::UserData,
                }
            })
    }

    fn make_compute_function_tables(
        pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
        tables: &crate::library_module::ResolvedLinkedFunctions,
        linked: &MetalLinkedFunctions,
    ) -> Result<MetalFunctionTables, String> {
        let mut result = MetalFunctionTables::default();
        for table in &tables.visible {
            let descriptor = MTLVisibleFunctionTableDescriptor::visibleFunctionTableDescriptor();
            unsafe {
                descriptor.setFunctionCount(function_table_count(table.binding, table.size)?)
            };
            let metal_table = pipeline
                .newVisibleFunctionTableWithDescriptor(&descriptor)
                .ok_or_else(|| {
                    format!(
                        "create compute visible function table at binding {}",
                        table.binding
                    )
                })?;
            for entry in &table.entries {
                let function = linked_function(linked, entry)?;
                let handle = pipeline
                    .functionHandleWithFunction(function)
                    .ok_or_else(|| {
                        format!("create handle for linked function {:?}", entry.function)
                    })?;
                unsafe { metal_table.setFunction_atIndex(Some(&handle), entry.index as usize) };
            }
            result.visible.push((table.binding, metal_table));
        }
        for table in &tables.intersection {
            let table_binding = table.location.buffer_binding();
            let descriptor =
                MTLIntersectionFunctionTableDescriptor::intersectionFunctionTableDescriptor();
            descriptor.setFunctionCount(function_table_count(table_binding, table.size)?);
            let metal_table = pipeline
                .newIntersectionFunctionTableWithDescriptor(&descriptor)
                .ok_or_else(|| {
                    format!(
                        "create compute intersection function table at binding {}",
                        table_binding
                    )
                })?;
            for entry in &table.entries {
                match entry {
                    crate::library_module::ResolvedIntersectionFunctionEntry::Linked(entry) => {
                        let function = linked_function(linked, entry)?;
                        let handle =
                            pipeline
                                .functionHandleWithFunction(function)
                                .ok_or_else(|| {
                                    format!(
                                        "create handle for linked function {:?}",
                                        entry.function
                                    )
                                })?;
                        metal_table.setFunction_atIndex(Some(&handle), entry.index as usize);
                    }
                    crate::library_module::ResolvedIntersectionFunctionEntry::OpaqueTriangle {
                        index,
                        signature,
                    } => unsafe {
                        metal_table.setOpaqueTriangleIntersectionFunctionWithSignature_atIndex(
                            metal_intersection_signature(signature),
                            *index as usize,
                        )
                    },
                }
            }
            result.intersection.push((table.location, metal_table));
        }
        Ok(result)
    }

    fn make_render_function_tables(
        pipeline: &ProtocolObject<dyn MTLRenderPipelineState>,
        stage: MTLRenderStages,
        tables: &crate::library_module::ResolvedLinkedFunctions,
        linked: &MetalLinkedFunctions,
    ) -> Result<MetalFunctionTables, String> {
        let mut result = MetalFunctionTables::default();
        for table in &tables.visible {
            let descriptor = MTLVisibleFunctionTableDescriptor::visibleFunctionTableDescriptor();
            unsafe {
                descriptor.setFunctionCount(function_table_count(table.binding, table.size)?)
            };
            let metal_table = pipeline
                .newVisibleFunctionTableWithDescriptor_stage(&descriptor, stage)
                .ok_or_else(|| {
                    format!(
                        "create render visible function table at binding {}",
                        table.binding
                    )
                })?;
            for entry in &table.entries {
                let function = linked_function(linked, entry)?;
                let handle = unsafe { pipeline.functionHandleWithFunction_stage(function, stage) }
                    .ok_or_else(|| {
                        format!("create handle for linked function {:?}", entry.function)
                    })?;
                unsafe { metal_table.setFunction_atIndex(Some(&handle), entry.index as usize) };
            }
            result.visible.push((table.binding, metal_table));
        }
        for table in &tables.intersection {
            let table_binding = table.location.buffer_binding();
            let descriptor =
                MTLIntersectionFunctionTableDescriptor::intersectionFunctionTableDescriptor();
            descriptor.setFunctionCount(function_table_count(table_binding, table.size)?);
            let metal_table = pipeline
                .newIntersectionFunctionTableWithDescriptor_stage(&descriptor, stage)
                .ok_or_else(|| {
                    format!(
                        "create render intersection function table at binding {}",
                        table_binding
                    )
                })?;
            for entry in &table.entries {
                match entry {
                    crate::library_module::ResolvedIntersectionFunctionEntry::Linked(entry) => {
                        let function = linked_function(linked, entry)?;
                        let handle =
                            unsafe { pipeline.functionHandleWithFunction_stage(function, stage) }
                                .ok_or_else(|| {
                                format!("create handle for linked function {:?}", entry.function)
                            })?;
                        metal_table.setFunction_atIndex(Some(&handle), entry.index as usize);
                    }
                    crate::library_module::ResolvedIntersectionFunctionEntry::OpaqueTriangle {
                        index,
                        signature,
                    } => unsafe {
                        metal_table.setOpaqueTriangleIntersectionFunctionWithSignature_atIndex(
                            metal_intersection_signature(signature),
                            *index as usize,
                        )
                    },
                }
            }
            result.intersection.push((table.location, metal_table));
        }
        Ok(result)
    }

    fn bind_fragment_function_tables(
        encoder: &ProtocolObject<dyn MTLRenderCommandEncoder>,
        tables: &MetalFunctionTables,
    ) {
        for (binding, table) in &tables.visible {
            unsafe {
                encoder.setFragmentVisibleFunctionTable_atBufferIndex(
                    Some(&**table),
                    *binding as usize,
                )
            };
        }
        for (location, table) in &tables.intersection {
            if let crate::library_module::ResolvedIntersectionFunctionTableLocation::Direct {
                binding,
            } = location
            {
                unsafe {
                    encoder.setFragmentIntersectionFunctionTable_atBufferIndex(
                        Some(&**table),
                        *binding as usize,
                    )
                };
            }
        }
    }

    fn bind_tile_function_tables(
        encoder: &ProtocolObject<dyn MTLRenderCommandEncoder>,
        tables: &MetalFunctionTables,
    ) {
        for (binding, table) in &tables.visible {
            unsafe {
                encoder.setTileVisibleFunctionTable_atBufferIndex(Some(&**table), *binding as usize)
            };
        }
        for (location, table) in &tables.intersection {
            if let crate::library_module::ResolvedIntersectionFunctionTableLocation::Direct {
                binding,
            } = location
            {
                unsafe {
                    encoder.setTileIntersectionFunctionTable_atBufferIndex(
                        Some(&**table),
                        *binding as usize,
                    )
                };
            }
        }
    }

    fn bind_vertex_function_tables(
        encoder: &ProtocolObject<dyn MTLRenderCommandEncoder>,
        tables: &MetalFunctionTables,
    ) {
        for (binding, table) in &tables.visible {
            unsafe {
                encoder
                    .setVertexVisibleFunctionTable_atBufferIndex(Some(&**table), *binding as usize)
            };
        }
        for (location, table) in &tables.intersection {
            if let crate::library_module::ResolvedIntersectionFunctionTableLocation::Direct {
                binding,
            } = location
            {
                unsafe {
                    encoder.setVertexIntersectionFunctionTable_atBufferIndex(
                        Some(&**table),
                        *binding as usize,
                    )
                };
            }
        }
    }

    fn make_buffers(
        device: &ProtocolObject<dyn MTLDevice>,
        resources: &LiteralResources,
        reflection: &metal2vulkan::reflect::ShaderReflection,
    ) -> Result<Vec<(u32, Buffer)>, String> {
        let mut buffers = resources
            .buffers
            .iter()
            .map(|resource| {
                let pointer = NonNull::new(resource.bytes.as_ptr().cast_mut().cast::<c_void>())
                    .ok_or_else(|| format!("buffer {} bytes pointer is null", resource.binding))?;
                let buffer = unsafe {
                    device.newBufferWithBytes_length_options(
                        pointer,
                        resource.bytes.len(),
                        MTLResourceOptions::StorageModeShared,
                    )
                }
                .ok_or_else(|| format!("create Metal buffer {}", resource.binding))?;
                Ok((resource.binding, buffer))
            })
            .collect::<Result<Vec<_>, String>>()?;
        for resource in &resources.kernel_stage_inputs {
            let reflected = reflection
                .bindings
                .iter()
                .find(|binding| {
                    binding.kind == metal2vulkan::reflect::ResourceKind::KernelStageInput
                        && binding.stage_input_location == Some(resource.location)
                })
                .ok_or_else(|| {
                    format!(
                        "kernel stage input {} has no reflected buffer slot",
                        resource.location
                    )
                })?;
            let pointer = NonNull::new(resource.bytes.as_ptr().cast_mut().cast::<c_void>())
                .ok_or_else(|| {
                    format!(
                        "kernel stage input {} bytes pointer is null",
                        resource.location
                    )
                })?;
            let buffer = unsafe {
                device.newBufferWithBytes_length_options(
                    pointer,
                    resource.bytes.len(),
                    MTLResourceOptions::StorageModeShared,
                )
            }
            .ok_or_else(|| format!("create Metal stage-input buffer {}", resource.location))?;
            buffers.push((reflected.metal_index, buffer));
        }
        Ok(buffers)
    }

    type DeviceBufferArrayElement = ((u32, u32), Buffer);

    fn append_device_buffer_arrays(
        device: &ProtocolObject<dyn MTLDevice>,
        resources: &LiteralResources,
        buffers: &mut Vec<(u32, Buffer)>,
    ) -> Result<Vec<DeviceBufferArrayElement>, String> {
        let mut nested = Vec::new();
        for array in &resources.device_buffer_arrays {
            for element in &array.elements {
                let label = format!(
                    "device-buffer-array {} element {}",
                    array.binding, element.index
                );
                let pointer = NonNull::new(element.bytes.as_ptr().cast_mut().cast::<c_void>())
                    .ok_or_else(|| format!("{label} bytes pointer is null"))?;
                let buffer = unsafe {
                    device.newBufferWithBytes_length_options(
                        pointer,
                        element.bytes.len(),
                        MTLResourceOptions::StorageModeShared,
                    )
                }
                .ok_or_else(|| format!("create Metal {label}"))?;
                let binding = array.binding.checked_add(element.index).ok_or_else(|| {
                    format!(
                        "device-buffer-array {} element {} binding overflows",
                        array.binding, element.index
                    )
                })?;
                buffers.push((binding, buffer.clone()));
                nested.push(((array.binding, element.index), buffer));
            }
        }
        Ok(nested)
    }

    fn make_pipeline(
        device: &ProtocolObject<dyn MTLDevice>,
        function: &ProtocolObject<dyn MTLFunction>,
        resources: &LiteralResources,
        reflection: &metal2vulkan::reflect::ShaderReflection,
        linked: &MetalLinkedFunctions,
    ) -> Result<Retained<ProtocolObject<dyn objc2_metal::MTLComputePipelineState>>, String> {
        if resources.kernel_stage_inputs.is_empty() && linked.descriptor.is_none() {
            return device
                .newComputePipelineStateWithFunction_error(function)
                .map_err(|error| format!("create compute pipeline: {error}"));
        }
        let stage = MTLStageInputOutputDescriptor::stageInputOutputDescriptor();
        let attributes = stage.attributes();
        let layouts = stage.layouts();
        for resource in &resources.kernel_stage_inputs {
            let reflected = reflection
                .bindings
                .iter()
                .find(|binding| {
                    binding.kind == metal2vulkan::reflect::ResourceKind::KernelStageInput
                        && binding.stage_input_location == Some(resource.location)
                })
                .ok_or_else(|| {
                    format!(
                        "kernel stage input {} has no reflected pipeline binding",
                        resource.location
                    )
                })?;
            let attribute =
                unsafe { attributes.objectAtIndexedSubscript(resource.location as usize) };
            attribute.setFormat(metal_attribute_format(resource.format));
            attribute.setOffset(0);
            unsafe { attribute.setBufferIndex(reflected.metal_index as usize) };
            let layout =
                unsafe { layouts.objectAtIndexedSubscript(reflected.metal_index as usize) };
            layout.setStride(resource.stride as usize);
            layout.setStepFunction(MTLStepFunction::ThreadPositionInGridX);
            layout.setStepRate(1);
        }
        let descriptor = MTLComputePipelineDescriptor::new();
        descriptor.setComputeFunction(Some(function));
        if !resources.kernel_stage_inputs.is_empty() {
            descriptor.setStageInputDescriptor(Some(&stage));
        }
        if let Some(linked) = &linked.descriptor {
            descriptor.setLinkedFunctions(Some(linked));
        }
        device
            .newComputePipelineStateWithDescriptor_options_reflection_error(
                &descriptor,
                MTLPipelineOption::None,
                None,
            )
            .map_err(|error| format!("create described compute pipeline: {error}"))
    }

    fn make_vertex_inputs(
        device: &ProtocolObject<dyn MTLDevice>,
        case: &AuthoredCase,
        resources: &LiteralResources,
    ) -> Result<MetalVertexInputs, String> {
        let descriptor = MTLVertexDescriptor::vertexDescriptor();
        let attributes = descriptor.attributes();
        let layouts = descriptor.layouts();
        let mut occupied = case
            .buffers
            .iter()
            .map(|resource| resource.binding as usize)
            .chain(
                case.acceleration_structures
                    .iter()
                    .map(|resource| resource.binding as usize),
            )
            .collect::<std::collections::HashSet<_>>();
        let mut inputs = resources
            .vertex_inputs
            .iter()
            .map(|input| (input, MTLVertexStepFunction::PerVertex))
            .chain(resources.tessellation.iter().flat_map(|tessellation| {
                tessellation
                    .control_points
                    .iter()
                    .map(|input| (input, MTLVertexStepFunction::PerPatchControlPoint))
                    .chain(
                        tessellation
                            .patch_inputs
                            .iter()
                            .map(|input| (input, MTLVertexStepFunction::PerPatch)),
                    )
            }))
            .collect::<Vec<_>>();
        inputs.sort_by_key(|(input, _)| input.location);
        let mut buffers = Vec::with_capacity(inputs.len());
        for (input, step_function) in inputs {
            let slot = (0..31)
                .rev()
                .find(|slot| occupied.insert(*slot))
                .ok_or_else(|| "Metal vertex-input buffer slots are exhausted".to_string())?;
            let attribute = unsafe { attributes.objectAtIndexedSubscript(input.location as usize) };
            attribute.setFormat(metal_vertex_format(input.format));
            unsafe {
                attribute.setOffset(0);
                attribute.setBufferIndex(slot);
            }
            let layout = unsafe { layouts.objectAtIndexedSubscript(slot) };
            unsafe { layout.setStride(input.stride as usize) };
            layout.setStepFunction(step_function);
            unsafe { layout.setStepRate(1) };
            let pointer = NonNull::new(input.bytes.as_ptr().cast_mut().cast::<c_void>())
                .ok_or_else(|| format!("vertex input {} pointer is null", input.location))?;
            let buffer = unsafe {
                device.newBufferWithBytes_length_options(
                    pointer,
                    input.bytes.len(),
                    MTLResourceOptions::StorageModeShared,
                )
            }
            .ok_or_else(|| format!("create Metal vertex input {}", input.location))?;
            buffers.push((slot, buffer));
        }
        Ok(MetalVertexInputs {
            descriptor,
            buffers,
        })
    }

    fn metal_tessellation_factor_bytes(
        tessellation: &crate::literal::LiteralTessellation,
    ) -> Vec<u8> {
        let words = tessellation
            .factors
            .iter()
            .flat_map(|patch| patch.edge_f16.iter().chain(&patch.inside_f16));
        let mut bytes = Vec::with_capacity(words.clone().count() * 2);
        for word in words {
            bytes.extend_from_slice(&word.to_le_bytes());
        }
        bytes
    }

    fn metal_attribute_format(format: crate::case::AttributeFormat) -> MTLAttributeFormat {
        use crate::case::AttributeFormat as F;
        match format {
            F::Char => MTLAttributeFormat::Char,
            F::Char2 => MTLAttributeFormat::Char2,
            F::Char3 => MTLAttributeFormat::Char3,
            F::Char4 => MTLAttributeFormat::Char4,
            F::Uchar => MTLAttributeFormat::UChar,
            F::Uchar2 => MTLAttributeFormat::UChar2,
            F::Uchar3 => MTLAttributeFormat::UChar3,
            F::Uchar4 => MTLAttributeFormat::UChar4,
            F::Short => MTLAttributeFormat::Short,
            F::Short2 => MTLAttributeFormat::Short2,
            F::Short3 => MTLAttributeFormat::Short3,
            F::Short4 => MTLAttributeFormat::Short4,
            F::Ushort => MTLAttributeFormat::UShort,
            F::Ushort2 => MTLAttributeFormat::UShort2,
            F::Ushort3 => MTLAttributeFormat::UShort3,
            F::Ushort4 => MTLAttributeFormat::UShort4,
            F::Half => MTLAttributeFormat::Half,
            F::Half2 => MTLAttributeFormat::Half2,
            F::Half3 => MTLAttributeFormat::Half3,
            F::Half4 => MTLAttributeFormat::Half4,
            F::Float => MTLAttributeFormat::Float,
            F::Float2 => MTLAttributeFormat::Float2,
            F::Float3 => MTLAttributeFormat::Float3,
            F::Float4 => MTLAttributeFormat::Float4,
            F::Uint => MTLAttributeFormat::UInt,
            F::Uint2 => MTLAttributeFormat::UInt2,
            F::Uint3 => MTLAttributeFormat::UInt3,
            F::Uint4 => MTLAttributeFormat::UInt4,
            F::Int => MTLAttributeFormat::Int,
            F::Int2 => MTLAttributeFormat::Int2,
            F::Int3 => MTLAttributeFormat::Int3,
            F::Int4 => MTLAttributeFormat::Int4,
        }
    }

    fn metal_vertex_format(format: crate::case::AttributeFormat) -> MTLVertexFormat {
        use crate::case::AttributeFormat as F;
        match format {
            F::Char => MTLVertexFormat::Char,
            F::Char2 => MTLVertexFormat::Char2,
            F::Char3 => MTLVertexFormat::Char3,
            F::Char4 => MTLVertexFormat::Char4,
            F::Uchar => MTLVertexFormat::UChar,
            F::Uchar2 => MTLVertexFormat::UChar2,
            F::Uchar3 => MTLVertexFormat::UChar3,
            F::Uchar4 => MTLVertexFormat::UChar4,
            F::Short => MTLVertexFormat::Short,
            F::Short2 => MTLVertexFormat::Short2,
            F::Short3 => MTLVertexFormat::Short3,
            F::Short4 => MTLVertexFormat::Short4,
            F::Ushort => MTLVertexFormat::UShort,
            F::Ushort2 => MTLVertexFormat::UShort2,
            F::Ushort3 => MTLVertexFormat::UShort3,
            F::Ushort4 => MTLVertexFormat::UShort4,
            F::Half => MTLVertexFormat::Half,
            F::Half2 => MTLVertexFormat::Half2,
            F::Half3 => MTLVertexFormat::Half3,
            F::Half4 => MTLVertexFormat::Half4,
            F::Float => MTLVertexFormat::Float,
            F::Float2 => MTLVertexFormat::Float2,
            F::Float3 => MTLVertexFormat::Float3,
            F::Float4 => MTLVertexFormat::Float4,
            F::Uint => MTLVertexFormat::UInt,
            F::Uint2 => MTLVertexFormat::UInt2,
            F::Uint3 => MTLVertexFormat::UInt3,
            F::Uint4 => MTLVertexFormat::UInt4,
            F::Int => MTLVertexFormat::Int,
            F::Int2 => MTLVertexFormat::Int2,
            F::Int3 => MTLVertexFormat::Int3,
            F::Int4 => MTLVertexFormat::Int4,
        }
    }

    fn make_function(
        library: &ProtocolObject<dyn MTLLibrary>,
        entry: &NSString,
        resources: &LiteralResources,
    ) -> Result<Retained<ProtocolObject<dyn objc2_metal::MTLFunction>>, String> {
        if resources.function_constants.is_empty() {
            return library
                .newFunctionWithName(entry)
                .ok_or_else(|| format!("Metal library has no function {entry:?}"));
        }
        let values = MTLFunctionConstantValues::new();
        for constant in &resources.function_constants {
            let pointer = NonNull::new(constant.bytes.as_ptr().cast_mut().cast::<c_void>())
                .ok_or_else(|| format!("function constant {} pointer is null", constant.index))?;
            unsafe {
                values.setConstantValue_type_atIndex(
                    pointer,
                    metal_data_type(constant.scalar_type, constant.lanes)?,
                    constant.index as usize,
                );
            }
        }
        library
            .newFunctionWithName_constantValues_error(entry, &values)
            .map_err(|error| format!("specialize Metal function {entry:?}: {error}"))
    }

    fn metal_data_type(scalar_type: ScalarType, lanes: u32) -> Result<MTLDataType, String> {
        let types = match scalar_type {
            ScalarType::Bool => [
                MTLDataType::Bool,
                MTLDataType::Bool2,
                MTLDataType::Bool3,
                MTLDataType::Bool4,
            ],
            ScalarType::U8 => [
                MTLDataType::UChar,
                MTLDataType::UChar2,
                MTLDataType::UChar3,
                MTLDataType::UChar4,
            ],
            ScalarType::I8 => [
                MTLDataType::Char,
                MTLDataType::Char2,
                MTLDataType::Char3,
                MTLDataType::Char4,
            ],
            ScalarType::U16 => [
                MTLDataType::UShort,
                MTLDataType::UShort2,
                MTLDataType::UShort3,
                MTLDataType::UShort4,
            ],
            ScalarType::I16 => [
                MTLDataType::Short,
                MTLDataType::Short2,
                MTLDataType::Short3,
                MTLDataType::Short4,
            ],
            ScalarType::F16 => [
                MTLDataType::Half,
                MTLDataType::Half2,
                MTLDataType::Half3,
                MTLDataType::Half4,
            ],
            ScalarType::U32 => [
                MTLDataType::UInt,
                MTLDataType::UInt2,
                MTLDataType::UInt3,
                MTLDataType::UInt4,
            ],
            ScalarType::I32 => [
                MTLDataType::Int,
                MTLDataType::Int2,
                MTLDataType::Int3,
                MTLDataType::Int4,
            ],
            ScalarType::F32 => [
                MTLDataType::Float,
                MTLDataType::Float2,
                MTLDataType::Float3,
                MTLDataType::Float4,
            ],
            ScalarType::U64 => [
                MTLDataType::ULong,
                MTLDataType::ULong2,
                MTLDataType::ULong3,
                MTLDataType::ULong4,
            ],
            ScalarType::I64 => [
                MTLDataType::Long,
                MTLDataType::Long2,
                MTLDataType::Long3,
                MTLDataType::Long4,
            ],
            ScalarType::F64 => return Err("Metal has no double function-constant type".into()),
        };
        lanes
            .checked_sub(1)
            .and_then(|index| types.get(index as usize))
            .copied()
            .ok_or_else(|| format!("Metal function-constant lane count {lanes} is invalid"))
    }

    fn make_textures(
        device: &ProtocolObject<dyn MTLDevice>,
        queue: &ProtocolObject<dyn MTLCommandQueue>,
        resources: &LiteralResources,
    ) -> Result<Vec<(u32, Texture)>, String> {
        resources
            .textures
            .iter()
            .map(|resource| {
                let label = format!("texture {}", resource.binding);
                let texture = make_texture(
                    device,
                    queue,
                    MetalTextureLiteral {
                        label: &label,
                        role: resource.role,
                        texture_type: resource.texture_type,
                        format: resource.format,
                        dimensions: resource.dimensions,
                        sample_count: resource.sample_count,
                        bytes: &resource.bytes,
                    },
                )?;
                Ok((resource.binding, texture))
            })
            .collect()
    }

    fn make_render_targets(
        device: &ProtocolObject<dyn MTLDevice>,
        resources: &LiteralResources,
        layered: bool,
    ) -> Result<Vec<(u32, Texture)>, String> {
        resources
            .render_targets
            .iter()
            .map(|resource| {
                let descriptor = MTLTextureDescriptor::new();
                descriptor.setTextureType(if layered {
                    MTLTextureType::Type2DArray
                } else {
                    MTLTextureType::Type2D
                });
                descriptor.setPixelFormat(metal_pixel_format(resource.format));
                unsafe {
                    descriptor.setWidth(resource.dimensions[0] as usize);
                    descriptor.setHeight(resource.dimensions[1] as usize);
                    if layered {
                        descriptor.setArrayLength(1);
                    }
                }
                descriptor.setStorageMode(MTLStorageMode::Shared);
                descriptor.setUsage(MTLTextureUsage::RenderTarget);
                let texture = device
                    .newTextureWithDescriptor(&descriptor)
                    .ok_or_else(|| format!("create Metal render target {}", resource.index))?;
                upload_texture(
                    &texture,
                    &format!("render target {}", resource.index),
                    if layered {
                        TextureType::D2Array
                    } else {
                        TextureType::D2
                    },
                    [resource.dimensions[0], resource.dimensions[1], 1],
                    1,
                    resource.format,
                    &resource.bytes,
                )?;
                Ok((
                    resource.index,
                    Texture {
                        object: texture,
                        _backing: None,
                    },
                ))
            })
            .collect()
    }

    fn make_rasterization_sink(device: &ProtocolObject<dyn MTLDevice>) -> Result<Texture, String> {
        let descriptor = MTLTextureDescriptor::new();
        descriptor.setTextureType(MTLTextureType::Type2D);
        descriptor.setPixelFormat(MTLPixelFormat::R8Unorm);
        unsafe {
            descriptor.setWidth(1);
            descriptor.setHeight(1);
        }
        descriptor.setStorageMode(MTLStorageMode::Shared);
        descriptor.setUsage(MTLTextureUsage::RenderTarget);
        Ok(Texture {
            object: device
                .newTextureWithDescriptor(&descriptor)
                .ok_or_else(|| "create Metal rasterization-disabled sink".to_string())?,
            _backing: None,
        })
    }

    struct DepthStencilTexture {
        depth: Option<Texture>,
        stencil: Option<Texture>,
    }

    #[derive(Clone, Copy)]
    struct MetalOutputResources<'a> {
        device_buffer_array_elements: &'a [DeviceBufferArrayElement],
        colors: &'a [(u32, Texture)],
        depth_stencil: Option<&'a DepthStencilTexture>,
        fragment_imageblock: Option<(&'a Buffer, [u32; 2], usize)>,
    }

    fn make_depth_stencil(
        device: &ProtocolObject<dyn MTLDevice>,
        resources: &LiteralResources,
        layered: bool,
    ) -> Result<Option<DepthStencilTexture>, String> {
        let Some(resource) = &resources.depth_stencil else {
            return Ok(None);
        };
        let make_aspect = |format, bytes: &[u8], pixel_size| -> Result<Texture, String> {
            let descriptor = MTLTextureDescriptor::new();
            descriptor.setTextureType(if layered {
                MTLTextureType::Type2DArray
            } else {
                MTLTextureType::Type2D
            });
            descriptor.setPixelFormat(format);
            unsafe {
                descriptor.setWidth(resource.dimensions[0] as usize);
                descriptor.setHeight(resource.dimensions[1] as usize);
                if layered {
                    descriptor.setArrayLength(1);
                }
            }
            descriptor.setStorageMode(MTLStorageMode::Shared);
            descriptor.setUsage(MTLTextureUsage::RenderTarget);
            let texture = device
                .newTextureWithDescriptor(&descriptor)
                .ok_or_else(|| "create Metal depth/stencil attachment".to_string())?;
            let pointer = NonNull::new(bytes.as_ptr().cast_mut().cast::<c_void>())
                .ok_or_else(|| "depth/stencil bytes pointer is null".to_string())?;
            let bytes_per_row = resource.dimensions[0] as usize * pixel_size;
            unsafe {
                texture.replaceRegion_mipmapLevel_slice_withBytes_bytesPerRow_bytesPerImage(
                    MTLRegion {
                        origin: MTLOrigin { x: 0, y: 0, z: 0 },
                        size: MTLSize {
                            width: resource.dimensions[0] as usize,
                            height: resource.dimensions[1] as usize,
                            depth: 1,
                        },
                    },
                    0,
                    0,
                    pointer,
                    bytes_per_row,
                    bytes_per_row * resource.dimensions[1] as usize,
                );
            }
            Ok(Texture {
                object: texture,
                _backing: None,
            })
        };
        let depth = resource
            .depth
            .as_deref()
            .map(|bytes| make_aspect(MTLPixelFormat::Depth32Float, bytes, 4))
            .transpose()?;
        let stencil = resource
            .stencil
            .as_deref()
            .map(|bytes| make_aspect(MTLPixelFormat::Stencil8, bytes, 1))
            .transpose()?;
        if depth.is_none() && stencil.is_none() {
            return Err("depth/stencil attachment has no aspect bytes".into());
        }
        Ok(Some(DepthStencilTexture { depth, stencil }))
    }

    fn make_depth_stencil_state(
        device: &ProtocolObject<dyn MTLDevice>,
        reflection: &metal2vulkan::reflect::ShaderReflection,
    ) -> Result<DepthStencilState, String> {
        let descriptor = MTLDepthStencilDescriptor::new();
        let depth = !reflection.depth_members.is_empty();
        descriptor.setDepthWriteEnabled(depth);
        descriptor.setDepthCompareFunction(
            match crate::executor_contract::depth_compare(reflection) {
                crate::executor_contract::DepthCompare::Always => MTLCompareFunction::Always,
                crate::executor_contract::DepthCompare::Less => MTLCompareFunction::Less,
                crate::executor_contract::DepthCompare::Greater => MTLCompareFunction::Greater,
            },
        );
        if !reflection.stencil_members.is_empty() {
            let stencil = MTLStencilDescriptor::new();
            stencil.setStencilCompareFunction(MTLCompareFunction::Always);
            stencil.setStencilFailureOperation(MTLStencilOperation::Keep);
            stencil.setDepthFailureOperation(MTLStencilOperation::Keep);
            stencil.setDepthStencilPassOperation(MTLStencilOperation::Replace);
            stencil.setReadMask(u32::MAX);
            stencil.setWriteMask(u32::MAX);
            descriptor.setFrontFaceStencil(Some(&stencil));
            descriptor.setBackFaceStencil(Some(&stencil));
        }
        device
            .newDepthStencilStateWithDescriptor(&descriptor)
            .ok_or_else(|| "create Metal depth/stencil state".to_string())
    }

    type TextureArray = (u32, Vec<Texture>);

    fn make_texture_arrays(
        device: &ProtocolObject<dyn MTLDevice>,
        queue: &ProtocolObject<dyn MTLCommandQueue>,
        resources: &LiteralResources,
    ) -> Result<Vec<TextureArray>, String> {
        resources
            .texture_arrays
            .iter()
            .map(|array| {
                let elements = array
                    .elements
                    .iter()
                    .enumerate()
                    .map(|(element, resource)| {
                        let label = format!("texture-array {} element {element}", array.binding);
                        make_texture(
                            device,
                            queue,
                            MetalTextureLiteral {
                                label: &label,
                                role: resource.role,
                                texture_type: resource.texture_type,
                                format: resource.format,
                                dimensions: resource.dimensions,
                                sample_count: resource.sample_count,
                                bytes: &resource.bytes,
                            },
                        )
                    })
                    .collect::<Result<Vec<_>, String>>()?;
                Ok((array.binding, elements))
            })
            .collect()
    }

    type ArgumentBufferTexture = ((u32, u32), Texture);
    type ArgumentBufferBuffer = ((u32, u32), Buffer);

    fn make_argument_buffer_buffers(
        device: &ProtocolObject<dyn MTLDevice>,
        resources: &LiteralResources,
    ) -> Result<Vec<ArgumentBufferBuffer>, String> {
        resources
            .argument_buffer_buffers
            .iter()
            .map(|resource| {
                let label = format!(
                    "argument-buffer buffer {}+{}",
                    resource.buffer_binding, resource.field_offset
                );
                let pointer = NonNull::new(resource.bytes.as_ptr().cast_mut().cast::<c_void>())
                    .ok_or_else(|| format!("{label} bytes pointer is null"))?;
                let buffer = unsafe {
                    device.newBufferWithBytes_length_options(
                        pointer,
                        resource.bytes.len(),
                        MTLResourceOptions::StorageModeShared,
                    )
                }
                .ok_or_else(|| format!("create Metal {label}"))?;
                Ok(((resource.buffer_binding, resource.field_offset), buffer))
            })
            .collect()
    }

    fn make_argument_buffer_textures(
        device: &ProtocolObject<dyn MTLDevice>,
        queue: &ProtocolObject<dyn MTLCommandQueue>,
        resources: &LiteralResources,
    ) -> Result<Vec<ArgumentBufferTexture>, String> {
        resources
            .argument_buffer_textures
            .iter()
            .map(|resource| {
                let label = resource.label();
                let texture = make_texture(
                    device,
                    queue,
                    MetalTextureLiteral {
                        label: &label,
                        role: resource.role,
                        texture_type: resource.texture_type,
                        format: resource.format,
                        dimensions: resource.dimensions,
                        sample_count: resource.sample_count,
                        bytes: &resource.bytes,
                    },
                )?;
                Ok(((resource.buffer_binding, resource.field_offset), texture))
            })
            .collect()
    }

    struct MetalTextureLiteral<'a> {
        label: &'a str,
        role: ResourceRole,
        texture_type: TextureType,
        format: TextureFormat,
        dimensions: [u32; 3],
        sample_count: u32,
        bytes: &'a [u8],
    }

    fn make_texture(
        device: &ProtocolObject<dyn MTLDevice>,
        queue: &ProtocolObject<dyn MTLCommandQueue>,
        literal: MetalTextureLiteral<'_>,
    ) -> Result<Texture, String> {
        let MetalTextureLiteral {
            label,
            role,
            texture_type,
            format,
            dimensions,
            sample_count,
            bytes,
        } = literal;
        let layout = crate::literal::texture_layout(texture_type, dimensions, sample_count)?;
        let descriptor = MTLTextureDescriptor::new();
        descriptor.setTextureType(metal_texture_type(texture_type));
        descriptor.setPixelFormat(metal_pixel_format(format));
        unsafe {
            descriptor.setWidth(layout.width as usize);
            descriptor.setHeight(layout.height as usize);
            descriptor.setDepth(layout.depth as usize);
            descriptor.setArrayLength(metal_array_length(texture_type, layout));
            descriptor.setSampleCount(layout.sample_count as usize);
        }
        descriptor.setStorageMode(if layout.sample_count == 1 {
            MTLStorageMode::Shared
        } else {
            MTLStorageMode::Private
        });
        let shader_usage = match role {
            ResourceRole::Input => MTLTextureUsage::ShaderRead,
            ResourceRole::Output => MTLTextureUsage::ShaderWrite,
            ResourceRole::InOut => MTLTextureUsage::ShaderRead | MTLTextureUsage::ShaderWrite,
        };
        descriptor.setUsage(if layout.sample_count == 1 {
            shader_usage
        } else {
            shader_usage | MTLTextureUsage::RenderTarget
        });
        if texture_type == TextureType::Buffer {
            let pointer = NonNull::new(bytes.as_ptr().cast_mut().cast::<c_void>())
                .ok_or_else(|| format!("{label} bytes pointer is null"))?;
            let backing = unsafe {
                device.newBufferWithBytes_length_options(
                    pointer,
                    bytes.len(),
                    MTLResourceOptions::StorageModeShared,
                )
            }
            .ok_or_else(|| format!("create Metal buffer backing for {label}"))?;
            let texture = backing
                .newTextureWithDescriptor_offset_bytesPerRow(&descriptor, 0, bytes.len())
                .ok_or_else(|| format!("create Metal texture-buffer view for {label}"))?;
            return Ok(Texture {
                object: texture,
                _backing: Some(backing),
            });
        }
        let texture = device
            .newTextureWithDescriptor(&descriptor)
            .ok_or_else(|| format!("create Metal {label}"))?;
        if layout.sample_count != 1 {
            initialize_multisample_texture(device, queue, &texture, label, layout, format, bytes)?;
            return Ok(Texture {
                object: texture,
                _backing: None,
            });
        }
        upload_texture(
            &texture,
            label,
            texture_type,
            dimensions,
            sample_count,
            format,
            bytes,
        )?;
        Ok(Texture {
            object: texture,
            _backing: None,
        })
    }

    fn upload_texture(
        texture: &ProtocolObject<dyn MTLTexture>,
        label: &str,
        texture_type: TextureType,
        dimensions: [u32; 3],
        sample_count: u32,
        format: TextureFormat,
        bytes: &[u8],
    ) -> Result<(), String> {
        let layout = crate::literal::texture_layout(texture_type, dimensions, sample_count)?;
        let bytes_per_row = dimensions[0] as usize * format.bytes_per_pixel();
        let bytes_per_image = bytes_per_row * dimensions[1] as usize;
        let pointer = NonNull::new(bytes.as_ptr().cast_mut().cast::<c_void>())
            .ok_or_else(|| format!("{label} bytes pointer is null"))?;
        let region = MTLRegion {
            origin: MTLOrigin { x: 0, y: 0, z: 0 },
            size: MTLSize {
                width: layout.width as usize,
                height: layout.height as usize,
                depth: layout.depth as usize,
            },
        };
        if texture_type == TextureType::D3 {
            unsafe {
                texture.replaceRegion_mipmapLevel_slice_withBytes_bytesPerRow_bytesPerImage(
                    region,
                    0,
                    0,
                    pointer,
                    bytes_per_row,
                    bytes_per_image,
                );
            }
        } else {
            for slice in 0..layout.array_layers as usize {
                let slice_pointer = unsafe {
                    NonNull::new_unchecked(
                        bytes.as_ptr().add(slice * bytes_per_image) as *mut c_void
                    )
                };
                unsafe {
                    texture.replaceRegion_mipmapLevel_slice_withBytes_bytesPerRow_bytesPerImage(
                        region,
                        0,
                        slice,
                        slice_pointer,
                        bytes_per_row,
                        bytes_per_image,
                    );
                }
            }
        }
        Ok(())
    }

    fn initialize_multisample_texture(
        device: &ProtocolObject<dyn MTLDevice>,
        queue: &ProtocolObject<dyn MTLCommandQueue>,
        texture: &ProtocolObject<dyn MTLTexture>,
        label: &str,
        layout: crate::literal::TextureLayout,
        format: TextureFormat,
        bytes: &[u8],
    ) -> Result<(), String> {
        let (value_type, result_type, expression, depth) = match format {
            TextureFormat::R8Unorm => ("uchar", "float", "float(value) / 255.0f", false),
            TextureFormat::Rgba8Unorm => ("uchar4", "float4", "float4(value) / 255.0f", false),
            TextureFormat::Rgba8Uint => ("uchar4", "uint4", "uint4(value)", false),
            TextureFormat::Rgba8Sint => ("char4", "int4", "int4(value)", false),
            TextureFormat::R16Float => ("half", "half", "value", false),
            TextureFormat::R16Uint => ("ushort", "uint", "uint(value)", false),
            TextureFormat::Rg16Float => ("half2", "half2", "value", false),
            TextureFormat::Rg32Float => ("float2", "float2", "value", false),
            TextureFormat::Rgba16Float => ("half4", "half4", "value", false),
            TextureFormat::Rgba16Uint => ("ushort4", "uint4", "uint4(value)", false),
            TextureFormat::R32Uint => ("uint", "uint", "value", false),
            TextureFormat::R32Sint => ("int", "int", "value", false),
            TextureFormat::R32Float => ("float", "float", "value", false),
            TextureFormat::Rgba32Uint => ("uint4", "uint4", "value", false),
            TextureFormat::Rgba32Sint => ("int4", "int4", "value", false),
            TextureFormat::Rgba32Float => ("float4", "float4", "value", false),
            TextureFormat::Depth32Float => ("float", "DepthValue", "DepthValue { value }", true),
        };
        let depth_declaration = if depth {
            "struct DepthValue { float depth [[depth(any)]]; };\n"
        } else {
            ""
        };
        let source = format!(
            r#"
#include <metal_stdlib>
using namespace metal;
struct LiteralVertex {{ float4 position [[position]]; }};
{depth_declaration}vertex LiteralVertex metal2vulkan_literal_vertex(uint id [[vertex_id]]) {{
    const float2 positions[3] = {{ float2(-1.0, -1.0), float2(3.0, -1.0), float2(-1.0, 3.0) }};
    return LiteralVertex {{ float4(positions[id], 0.0, 1.0) }};
}}
fragment {result_type} metal2vulkan_literal_fragment(
    LiteralVertex input [[stage_in]], uint sample [[sample_id]],
    const device {value_type} *values [[buffer(0)]]) {{
    uint x = uint(input.position.x);
    uint y = uint(input.position.y);
    {value_type} value = values[(y * {width}u + x) * {samples}u + sample];
    return {expression};
}}
"#,
            width = layout.width,
            samples = layout.sample_count,
        );
        let library = device
            .newLibraryWithSource_options_error(&NSString::from_str(&source), None)
            .map_err(|error| format!("compile multisample initializer for {label}: {error}"))?;
        let vertex = library
            .newFunctionWithName(&NSString::from_str("metal2vulkan_literal_vertex"))
            .ok_or_else(|| format!("multisample initializer for {label} has no vertex function"))?;
        let fragment = library
            .newFunctionWithName(&NSString::from_str("metal2vulkan_literal_fragment"))
            .ok_or_else(|| {
                format!("multisample initializer for {label} has no fragment function")
            })?;
        let pipeline_descriptor = MTLRenderPipelineDescriptor::new();
        pipeline_descriptor.setVertexFunction(Some(&vertex));
        pipeline_descriptor.setFragmentFunction(Some(&fragment));
        pipeline_descriptor.setRasterSampleCount(layout.sample_count as usize);
        if depth {
            pipeline_descriptor.setDepthAttachmentPixelFormat(metal_pixel_format(format));
        } else {
            let attachment = unsafe {
                pipeline_descriptor
                    .colorAttachments()
                    .objectAtIndexedSubscript(0)
            };
            attachment.setPixelFormat(metal_pixel_format(format));
        }
        let pipeline = device
            .newRenderPipelineStateWithDescriptor_error(&pipeline_descriptor)
            .map_err(|error| format!("create multisample initializer for {label}: {error}"))?;
        let pointer = NonNull::new(bytes.as_ptr().cast_mut().cast::<c_void>())
            .ok_or_else(|| format!("{label} bytes pointer is null"))?;
        let source_buffer = unsafe {
            device.newBufferWithBytes_length_options(
                pointer,
                bytes.len(),
                MTLResourceOptions::StorageModeShared,
            )
        }
        .ok_or_else(|| format!("create multisample source buffer for {label}"))?;
        let depth_state = if depth {
            let descriptor = MTLDepthStencilDescriptor::new();
            descriptor.setDepthWriteEnabled(true);
            descriptor.setDepthCompareFunction(MTLCompareFunction::Always);
            Some(
                device
                    .newDepthStencilStateWithDescriptor(&descriptor)
                    .ok_or_else(|| format!("create multisample depth state for {label}"))?,
            )
        } else {
            None
        };
        let command = queue
            .commandBuffer()
            .ok_or_else(|| "MTLCommandQueue::commandBuffer returned nil".to_string())?;
        let layer_bytes = layout.width as usize
            * layout.height as usize
            * layout.sample_count as usize
            * format.bytes_per_pixel();
        for layer in 0..layout.array_layers as usize {
            let pass = MTLRenderPassDescriptor::new();
            if depth {
                let attachment = pass.depthAttachment();
                attachment.setTexture(Some(texture));
                attachment.setSlice(layer);
                attachment.setLoadAction(MTLLoadAction::DontCare);
                attachment.setStoreAction(MTLStoreAction::Store);
            } else {
                let attachment = unsafe { pass.colorAttachments().objectAtIndexedSubscript(0) };
                attachment.setTexture(Some(texture));
                attachment.setSlice(layer);
                attachment.setLoadAction(MTLLoadAction::DontCare);
                attachment.setStoreAction(MTLStoreAction::Store);
            }
            let encoder = command
                .renderCommandEncoderWithDescriptor(&pass)
                .ok_or_else(|| format!("create multisample initializer encoder for {label}"))?;
            encoder.setRenderPipelineState(&pipeline);
            if let Some(state) = &depth_state {
                encoder.setDepthStencilState(Some(state));
            }
            unsafe {
                encoder.setFragmentBuffer_offset_atIndex(
                    Some(&*source_buffer),
                    layer * layer_bytes,
                    0,
                );
                encoder.drawPrimitives_vertexStart_vertexCount(MTLPrimitiveType::Triangle, 0, 3);
            }
            encoder.endEncoding();
        }
        command.commit();
        command.waitUntilCompleted();
        ensure_completed(
            &command,
            &format!("Metal multisample initialization for {label}"),
        )
    }

    fn encode_argument_buffer_textures(
        function: &ProtocolObject<dyn objc2_metal::MTLFunction>,
        buffers: &[(u32, Buffer)],
        textures: &[ArgumentBufferTexture],
        reflection: &metal2vulkan::reflect::ShaderReflection,
    ) -> Result<(), String> {
        let buffer_bindings = textures
            .iter()
            .map(|((buffer_binding, _), _)| *buffer_binding)
            .collect::<std::collections::BTreeSet<_>>();
        for buffer_binding in buffer_bindings {
            let buffer = buffers
                .iter()
                .find_map(|(binding, buffer)| (*binding == buffer_binding).then_some(buffer))
                .ok_or_else(|| format!("argument buffer {buffer_binding} was not allocated"))?;
            let encoder =
                unsafe { function.newArgumentEncoderWithBufferIndex(buffer_binding as usize) };
            if buffer.length() < encoder.encodedLength() {
                return Err(format!(
                    "argument buffer {buffer_binding} has {} bytes, but Metal requires {}",
                    buffer.length(),
                    encoder.encodedLength()
                ));
            }
            unsafe { encoder.setArgumentBuffer_offset(Some(&**buffer), 0) };
            for ((_, field_offset), texture) in textures
                .iter()
                .filter(|((binding, _), _)| *binding == buffer_binding)
            {
                let (source, element) = reflection
                    .bindings
                    .iter()
                    .filter(|binding| {
                        binding.kind
                            == metal2vulkan::reflect::ResourceKind::EmbeddedArgBufferTexture
                    })
                    .find_map(|binding| {
                        let source = binding.embedded_source?;
                        let count = binding
                            .descriptor
                            .map(|descriptor| descriptor.count)
                            .unwrap_or(1);
                        let delta = field_offset.checked_sub(source.field_offset)?;
                        (source.buffer_index == buffer_binding
                            && delta % 8 == 0
                            && delta / 8 < count)
                            .then_some((source, delta / 8))
                    })
                    .ok_or_else(|| {
                        format!(
                            "argument-buffer texture {buffer_binding}+{field_offset} has no reflected argument index"
                        )
                    })?;
                let argument_index = source.argument_index.checked_add(element).ok_or_else(|| {
                    format!(
                        "argument-buffer texture {buffer_binding}+{field_offset} argument index overflows"
                    )
                })?;
                unsafe { encoder.setTexture_atIndex(Some(&**texture), argument_index as usize) };
            }
        }
        Ok(())
    }

    fn encode_argument_buffer_buffers(
        function: &ProtocolObject<dyn objc2_metal::MTLFunction>,
        owners: &[(u32, Buffer)],
        resources: &[ArgumentBufferBuffer],
        reflection: &metal2vulkan::reflect::ShaderReflection,
    ) -> Result<(), String> {
        for ((buffer_binding, field_offset), resource) in resources {
            let owner = owners
                .iter()
                .find_map(|(binding, buffer)| (*binding == *buffer_binding).then_some(buffer))
                .ok_or_else(|| format!("argument buffer {buffer_binding} was not allocated"))?;
            let source = reflection
                .bindings
                .iter()
                .filter(|binding| {
                    binding.kind
                        == metal2vulkan::reflect::ResourceKind::EmbeddedArgBufferBuffer
                })
                .filter_map(|binding| binding.embedded_source)
                .find(|source| {
                    source.buffer_index == *buffer_binding
                        && source.field_offset == *field_offset
                })
                .ok_or_else(|| {
                    format!(
                        "argument-buffer buffer {buffer_binding}+{field_offset} has no reflected argument index"
                    )
                })?;
            let encoder =
                unsafe { function.newArgumentEncoderWithBufferIndex(*buffer_binding as usize) };
            if owner.length() < encoder.encodedLength() {
                return Err(format!(
                    "argument buffer {buffer_binding} has {} bytes, but Metal requires {}",
                    owner.length(),
                    encoder.encodedLength()
                ));
            }
            unsafe {
                encoder.setArgumentBuffer_offset(Some(&**owner), 0);
                encoder.setBuffer_offset_atIndex(
                    Some(&**resource),
                    0,
                    source.argument_index as usize,
                );
            }
        }
        Ok(())
    }

    fn encode_argument_buffer_function_tables(
        function: &ProtocolObject<dyn objc2_metal::MTLFunction>,
        buffers: &[(u32, Buffer)],
        tables: &MetalFunctionTables,
        reflection: &metal2vulkan::reflect::ShaderReflection,
    ) -> Result<(), String> {
        for (location, table) in &tables.intersection {
            let crate::library_module::ResolvedIntersectionFunctionTableLocation::ArgumentBuffer {
                buffer_binding,
                field_offset,
            } = location
            else {
                continue;
            };
            let buffer = buffers
                .iter()
                .find_map(|(binding, buffer)| (*binding == *buffer_binding).then_some(buffer))
                .ok_or_else(|| format!("argument buffer {buffer_binding} was not allocated"))?;
            let field = reflection
                .argument_buffer_fields
                .iter()
                .find(|field| {
                    field.buffer_index == *buffer_binding && field.field_offset == *field_offset
                })
                .ok_or_else(|| {
                    format!(
                        "argument-buffer intersection table {buffer_binding}+{field_offset} has no reflected argument index"
                    )
                })?;
            let encoder =
                unsafe { function.newArgumentEncoderWithBufferIndex(*buffer_binding as usize) };
            if buffer.length() < encoder.encodedLength() {
                return Err(format!(
                    "argument buffer {buffer_binding} has {} bytes, but Metal requires {}",
                    buffer.length(),
                    encoder.encodedLength()
                ));
            }
            unsafe {
                encoder.setArgumentBuffer_offset(Some(&**buffer), 0);
                encoder.setIntersectionFunctionTable_atIndex(
                    Some(&**table),
                    field.argument_index as usize,
                );
            }
        }
        Ok(())
    }

    fn make_samplers(
        device: &ProtocolObject<dyn MTLDevice>,
        case: &AuthoredCase,
    ) -> Result<Vec<(u32, Sampler)>, String> {
        case.samplers
            .iter()
            .map(|resource| {
                let descriptor = MTLSamplerDescriptor::new();
                descriptor.setSAddressMode(metal_address_mode(resource.address_mode));
                descriptor.setTAddressMode(metal_address_mode(resource.address_mode));
                descriptor.setRAddressMode(metal_address_mode(resource.address_mode));
                descriptor.setMinFilter(metal_filter(resource.min_filter));
                descriptor.setMagFilter(metal_filter(resource.mag_filter));
                descriptor.setMipFilter(metal_mip_filter(resource.mip_filter));
                descriptor.setNormalizedCoordinates(resource.normalized_coordinates);
                let sampler = device
                    .newSamplerStateWithDescriptor(&descriptor)
                    .ok_or_else(|| format!("create Metal sampler {}", resource.binding))?;
                Ok((resource.binding, sampler))
            })
            .collect()
    }

    fn metal_texture_type(texture_type: TextureType) -> MTLTextureType {
        match texture_type {
            TextureType::Buffer => MTLTextureType::TypeTextureBuffer,
            TextureType::D1 => MTLTextureType::Type1D,
            TextureType::D1Array => MTLTextureType::Type1DArray,
            TextureType::D2 => MTLTextureType::Type2D,
            TextureType::D2Array => MTLTextureType::Type2DArray,
            TextureType::D2Multisample => MTLTextureType::Type2DMultisample,
            TextureType::D2MultisampleArray => MTLTextureType::Type2DMultisampleArray,
            TextureType::D3 => MTLTextureType::Type3D,
            TextureType::Cube => MTLTextureType::TypeCube,
            TextureType::CubeArray => MTLTextureType::TypeCubeArray,
        }
    }

    fn metal_array_length(
        texture_type: TextureType,
        layout: crate::literal::TextureLayout,
    ) -> usize {
        match texture_type {
            TextureType::Buffer => 1,
            TextureType::D1Array | TextureType::D2Array | TextureType::D2MultisampleArray => {
                layout.array_layers as usize
            }
            TextureType::CubeArray => (layout.array_layers / 6) as usize,
            _ => 1,
        }
    }

    fn metal_pixel_format(format: TextureFormat) -> MTLPixelFormat {
        match format {
            TextureFormat::R8Unorm => MTLPixelFormat::R8Unorm,
            TextureFormat::Rgba8Unorm => MTLPixelFormat::RGBA8Unorm,
            TextureFormat::Rgba8Uint => MTLPixelFormat::RGBA8Uint,
            TextureFormat::Rgba8Sint => MTLPixelFormat::RGBA8Sint,
            TextureFormat::R16Float => MTLPixelFormat::R16Float,
            TextureFormat::R16Uint => MTLPixelFormat::R16Uint,
            TextureFormat::Rg16Float => MTLPixelFormat::RG16Float,
            TextureFormat::Rg32Float => MTLPixelFormat::RG32Float,
            TextureFormat::Rgba16Float => MTLPixelFormat::RGBA16Float,
            TextureFormat::Rgba16Uint => MTLPixelFormat::RGBA16Uint,
            TextureFormat::R32Uint => MTLPixelFormat::R32Uint,
            TextureFormat::R32Sint => MTLPixelFormat::R32Sint,
            TextureFormat::R32Float => MTLPixelFormat::R32Float,
            TextureFormat::Rgba32Uint => MTLPixelFormat::RGBA32Uint,
            TextureFormat::Rgba32Sint => MTLPixelFormat::RGBA32Sint,
            TextureFormat::Rgba32Float => MTLPixelFormat::RGBA32Float,
            TextureFormat::Depth32Float => MTLPixelFormat::Depth32Float,
        }
    }

    fn metal_address_mode(mode: SamplerAddressMode) -> MTLSamplerAddressMode {
        match mode {
            SamplerAddressMode::ClampToEdge => MTLSamplerAddressMode::ClampToEdge,
            SamplerAddressMode::ClampToZero => MTLSamplerAddressMode::ClampToZero,
            SamplerAddressMode::Repeat => MTLSamplerAddressMode::Repeat,
            SamplerAddressMode::MirroredRepeat => MTLSamplerAddressMode::MirrorRepeat,
        }
    }

    fn metal_filter(filter: SamplerFilter) -> MTLSamplerMinMagFilter {
        match filter {
            SamplerFilter::Nearest => MTLSamplerMinMagFilter::Nearest,
            SamplerFilter::Linear => MTLSamplerMinMagFilter::Linear,
        }
    }

    fn metal_mip_filter(filter: SamplerMipFilter) -> MTLSamplerMipFilter {
        match filter {
            SamplerMipFilter::NotMipmapped => MTLSamplerMipFilter::NotMipmapped,
            SamplerMipFilter::Nearest => MTLSamplerMipFilter::Nearest,
            SamplerMipFilter::Linear => MTLSamplerMipFilter::Linear,
        }
    }

    fn make_acceleration_structures(
        device: &ProtocolObject<dyn MTLDevice>,
        queue: &ProtocolObject<dyn MTLCommandQueue>,
        case: &AuthoredCase,
    ) -> Result<Vec<(u32, AccelerationStructure)>, String> {
        if case.acceleration_structures.is_empty() {
            return Ok(Vec::new());
        }
        if !device.supportsRaytracing() {
            return Err("Metal device does not support acceleration structures".into());
        }

        let canonical_vertices = [-1.0f32, -1.0, 0.0, 1.0, -1.0, 0.0, 0.0, 1.0, 0.0];
        let canonical_primitive = build_primitive_acceleration_structure(
            device,
            queue,
            &canonical_vertices,
            "canonical triangle",
        )?;

        case.acceleration_structures
            .iter()
            .map(|resource| {
                let acceleration_structure = match resource.kind {
                    AccelerationStructureKind::Instance => {
                        let descriptors = resource
                            .child_references
                            .iter()
                            .map(|_| MTLAccelerationStructureInstanceDescriptor {
                                transformationMatrix: identity_transform(),
                                options: MTLAccelerationStructureInstanceOptions::Opaque,
                                mask: u32::MAX,
                                intersectionFunctionTableOffset: 0,
                                accelerationStructureIndex: 0,
                            })
                            .collect::<Vec<_>>();
                        let instance_buffer = new_buffer_from_slice(
                            device,
                            &descriptors,
                            &format!("acceleration structure {} instances", resource.binding),
                        )?;
                        let primitive_array = NSArray::from_slice(&[&*canonical_primitive]);
                        let descriptor = MTLInstanceAccelerationStructureDescriptor::descriptor();
                        descriptor.setInstancedAccelerationStructures(Some(&primitive_array));
                        descriptor.setInstanceDescriptorBuffer(Some(&*instance_buffer));
                        descriptor.setInstanceCount(descriptors.len());
                        let descriptor_base: &MTLAccelerationStructureDescriptor = &descriptor;
                        build_acceleration_structure(device, queue, descriptor_base)?
                    }
                    AccelerationStructureKind::Primitive => {
                        let encoded =
                            resource.primitive_triangles_b64.as_deref().ok_or_else(|| {
                                format!(
                                    "primitive acceleration structure {} has no triangles",
                                    resource.binding
                                )
                            })?;
                        let bytes = base64::engine::general_purpose::STANDARD
                            .decode(encoded)
                            .map_err(|error| {
                                format!(
                                    "decode primitive acceleration structure {}: {error}",
                                    resource.binding
                                )
                            })?;
                        let vertices = bytes
                            .chunks_exact(4)
                            .map(|word| {
                                f32::from_le_bytes(word.try_into().expect("four-byte chunk"))
                            })
                            .collect::<Vec<_>>();
                        build_primitive_acceleration_structure(
                            device,
                            queue,
                            &vertices,
                            &format!("primitive acceleration structure {}", resource.binding),
                        )?
                    }
                };
                Ok((resource.binding, acceleration_structure))
            })
            .collect()
    }

    fn build_primitive_acceleration_structure(
        device: &ProtocolObject<dyn MTLDevice>,
        queue: &ProtocolObject<dyn MTLCommandQueue>,
        vertices: &[f32],
        label: &str,
    ) -> Result<AccelerationStructure, String> {
        let vertex_buffer = new_buffer_from_slice(device, vertices, label)?;
        let geometry = MTLAccelerationStructureTriangleGeometryDescriptor::descriptor();
        geometry.setVertexBuffer(Some(&*vertex_buffer));
        geometry.setVertexStride(3 * std::mem::size_of::<f32>());
        geometry.setTriangleCount(vertices.len() / 9);
        let geometry_base: &MTLAccelerationStructureGeometryDescriptor = &geometry;
        let geometries = NSArray::from_slice(&[geometry_base]);
        let descriptor = MTLPrimitiveAccelerationStructureDescriptor::descriptor();
        descriptor.setGeometryDescriptors(Some(&geometries));
        let descriptor_base: &MTLAccelerationStructureDescriptor = &descriptor;
        build_acceleration_structure(device, queue, descriptor_base)
    }

    fn build_acceleration_structure(
        device: &ProtocolObject<dyn MTLDevice>,
        queue: &ProtocolObject<dyn MTLCommandQueue>,
        descriptor: &MTLAccelerationStructureDescriptor,
    ) -> Result<AccelerationStructure, String> {
        let sizes = device.accelerationStructureSizesWithDescriptor(descriptor);
        let acceleration_structure = device
            .newAccelerationStructureWithSize(sizes.accelerationStructureSize)
            .ok_or_else(|| "create Metal acceleration structure".to_string())?;
        let scratch = device
            .newBufferWithLength_options(
                sizes.buildScratchBufferSize.max(1),
                MTLResourceOptions::StorageModePrivate,
            )
            .ok_or_else(|| "create Metal acceleration-structure scratch buffer".to_string())?;
        let command_buffer = queue
            .commandBuffer()
            .ok_or_else(|| "MTLCommandQueue::commandBuffer returned nil".to_string())?;
        let encoder = command_buffer
            .accelerationStructureCommandEncoder()
            .ok_or_else(|| "accelerationStructureCommandEncoder returned nil".to_string())?;
        encoder.buildAccelerationStructure_descriptor_scratchBuffer_scratchBufferOffset(
            &acceleration_structure,
            descriptor,
            &scratch,
            0,
        );
        encoder.endEncoding();
        command_buffer.commit();
        command_buffer.waitUntilCompleted();
        ensure_completed(&command_buffer, "Metal acceleration-structure build")?;
        Ok(acceleration_structure)
    }

    fn new_buffer_from_slice<T>(
        device: &ProtocolObject<dyn MTLDevice>,
        values: &[T],
        label: &str,
    ) -> Result<Buffer, String> {
        let length = std::mem::size_of_val(values);
        if length == 0 {
            return device
                .newBufferWithLength_options(1, MTLResourceOptions::StorageModeShared)
                .ok_or_else(|| format!("create Metal buffer for {label}"));
        }
        let pointer = NonNull::new(values.as_ptr().cast_mut().cast::<c_void>())
            .ok_or_else(|| format!("{label} pointer is null"))?;
        unsafe {
            device.newBufferWithBytes_length_options(
                pointer,
                length,
                MTLResourceOptions::StorageModeShared,
            )
        }
        .ok_or_else(|| format!("create Metal buffer for {label}"))
    }

    fn identity_transform() -> MTLPackedFloat4x3 {
        MTLPackedFloat4x3 {
            columns: [
                MTLPackedFloat3 {
                    x: 1.0,
                    y: 0.0,
                    z: 0.0,
                },
                MTLPackedFloat3 {
                    x: 0.0,
                    y: 1.0,
                    z: 0.0,
                },
                MTLPackedFloat3 {
                    x: 0.0,
                    y: 0.0,
                    z: 1.0,
                },
                MTLPackedFloat3 {
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                },
            ],
        }
    }

    fn ensure_completed(
        command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
        operation: &str,
    ) -> Result<(), String> {
        if command_buffer.status() == MTLCommandBufferStatus::Completed {
            return Ok(());
        }
        Err(command_buffer
            .error()
            .map(|error| format!("{operation} failed: {error}"))
            .unwrap_or_else(|| {
                format!(
                    "{operation} ended with status {:?}",
                    command_buffer.status()
                )
            }))
    }

    fn selected_output(
        case: &AuthoredCase,
        buffers: &[(u32, Buffer)],
        argument_buffer_buffers: &[ArgumentBufferBuffer],
        textures: &[(u32, Texture)],
        texture_arrays: &[TextureArray],
        argument_buffer_textures: &[ArgumentBufferTexture],
        output_resources: MetalOutputResources<'_>,
    ) -> Result<Vec<u8>, String> {
        let device_buffer_array_elements = output_resources.device_buffer_array_elements;
        let render_targets = output_resources.colors;
        let depth_stencil = output_resources.depth_stencil;
        match case.output {
            OutputSelection::None => Ok(Vec::new()),
            OutputSelection::Buffer {
                binding,
                offset,
                length,
            } => {
                let buffer = buffers
                    .iter()
                    .find_map(|(candidate, buffer)| (*candidate == binding).then_some(buffer))
                    .ok_or_else(|| format!("output buffer {binding} was not bound"))?;
                let start = offset as usize;
                let length = length as usize;
                if start.saturating_add(length) > buffer.length() {
                    return Err(format!("selected output exceeds Metal buffer {binding}"));
                }
                unsafe {
                    let pointer = buffer.contents().as_ptr().cast::<u8>().add(start);
                    Ok(std::slice::from_raw_parts(pointer, length).to_vec())
                }
            }
            OutputSelection::ArgumentBufferBuffer {
                buffer_binding,
                field_offset,
                offset,
                length,
            } => {
                let buffer = argument_buffer_buffers
                    .iter()
                    .find_map(|((owner, field), buffer)| {
                        (*owner == buffer_binding && *field == field_offset).then_some(buffer)
                    })
                    .ok_or_else(|| {
                        format!(
                            "output argument-buffer buffer {buffer_binding}+{field_offset} was not bound"
                        )
                    })?;
                let start = offset as usize;
                let length = length as usize;
                if start.saturating_add(length) > buffer.length() {
                    return Err("selected output exceeds Metal argument-buffer buffer".into());
                }
                unsafe {
                    let pointer = buffer.contents().as_ptr().cast::<u8>().add(start);
                    Ok(std::slice::from_raw_parts(pointer, length).to_vec())
                }
            }
            OutputSelection::DeviceBufferArrayElement {
                binding,
                element,
                offset,
                length,
            } => {
                let buffer = device_buffer_array_elements
                    .iter()
                    .find_map(|((owner, index), buffer)| {
                        (*owner == binding && *index == element).then_some(buffer)
                    })
                    .ok_or_else(|| {
                        format!(
                            "output device-buffer-array {binding} element {element} was not bound"
                        )
                    })?;
                let start = offset as usize;
                let length = length as usize;
                if start.saturating_add(length) > buffer.length() {
                    return Err("selected output exceeds Metal device-buffer-array element".into());
                }
                unsafe {
                    let pointer = buffer.contents().as_ptr().cast::<u8>().add(start);
                    Ok(std::slice::from_raw_parts(pointer, length).to_vec())
                }
            }
            OutputSelection::Texture {
                binding,
                origin,
                dimensions,
            } => {
                let texture = textures
                    .iter()
                    .find_map(|(candidate, texture)| (*candidate == binding).then_some(texture))
                    .ok_or_else(|| format!("output texture {binding} was not bound"))?;
                let resource = case
                    .textures
                    .iter()
                    .find(|resource| resource.binding == binding)
                    .ok_or_else(|| format!("output texture {binding} is not declared"))?;
                read_texture_region(
                    texture,
                    resource.texture_type,
                    resource.format,
                    origin,
                    dimensions,
                )
            }
            OutputSelection::TextureArrayElement {
                binding,
                element,
                origin,
                dimensions,
            } => {
                let texture = texture_arrays
                    .iter()
                    .find(|(candidate, _)| *candidate == binding)
                    .and_then(|(_, elements)| elements.get(element as usize))
                    .ok_or_else(|| {
                        format!("output texture-array {binding} element {element} was not bound")
                    })?;
                let resource = case
                    .texture_arrays
                    .iter()
                    .find(|resource| resource.binding == binding)
                    .ok_or_else(|| format!("output texture-array {binding} is not declared"))?;
                read_texture_region(
                    texture,
                    resource.texture_type,
                    resource.format,
                    origin,
                    dimensions,
                )
            }
            OutputSelection::ArgumentBufferTexture {
                buffer_binding,
                field_offset,
                origin,
                dimensions,
            } => {
                let texture = argument_buffer_textures
                    .iter()
                    .find_map(|((buffer, offset), texture)| {
                        (*buffer == buffer_binding && *offset == field_offset).then_some(texture)
                    })
                    .ok_or_else(|| {
                        format!(
                            "output argument-buffer texture {buffer_binding}+{field_offset} was not bound"
                        )
                    })?;
                let resource = case
                    .argument_buffer_textures
                    .iter()
                    .find(|resource| {
                        resource.buffer_binding == buffer_binding
                            && resource.field_offset == field_offset
                    })
                    .ok_or_else(|| {
                        format!(
                            "output argument-buffer texture {buffer_binding}+{field_offset} is not declared"
                        )
                    })?;
                read_texture_region(
                    texture,
                    resource.texture_type,
                    resource.format,
                    origin,
                    dimensions,
                )
            }
            OutputSelection::RenderTarget {
                index,
                origin,
                dimensions,
            } => {
                let texture = render_targets
                    .iter()
                    .find_map(|(candidate, texture)| (*candidate == index).then_some(texture))
                    .ok_or_else(|| format!("output render target {index} was not bound"))?;
                let resource = case
                    .render_targets
                    .iter()
                    .find(|resource| resource.index == index)
                    .ok_or_else(|| format!("output render target {index} is not declared"))?;
                read_texture_region(
                    texture,
                    TextureType::D2,
                    resource.format,
                    [origin[0], origin[1], 0],
                    [dimensions[0], dimensions[1], 1],
                )
            }
            OutputSelection::Depth { origin, dimensions }
            | OutputSelection::Stencil { origin, dimensions } => {
                let attachment = depth_stencil
                    .ok_or_else(|| "depth/stencil output attachment was not bound".to_string())?;
                let depth = matches!(case.output, OutputSelection::Depth { .. });
                let texture = if depth {
                    attachment.depth.as_ref()
                } else {
                    attachment.stencil.as_ref()
                }
                .ok_or_else(|| {
                    "selected Metal depth/stencil aspect was not allocated".to_string()
                })?;
                let pixel_size = if depth { 4 } else { 1 };
                let bytes_per_row = dimensions[0] as usize * pixel_size;
                let mut output = vec![0u8; bytes_per_row * dimensions[1] as usize];
                let pointer = NonNull::new(output.as_mut_ptr().cast::<c_void>())
                    .ok_or_else(|| "depth/stencil output pointer is null".to_string())?;
                unsafe {
                    texture.getBytes_bytesPerRow_bytesPerImage_fromRegion_mipmapLevel_slice(
                        pointer,
                        bytes_per_row,
                        bytes_per_row * dimensions[1] as usize,
                        MTLRegion {
                            origin: MTLOrigin {
                                x: origin[0] as usize,
                                y: origin[1] as usize,
                                z: 0,
                            },
                            size: MTLSize {
                                width: dimensions[0] as usize,
                                height: dimensions[1] as usize,
                                depth: 1,
                            },
                        },
                        0,
                        0,
                    );
                }
                Ok(output)
            }
            OutputSelection::FragmentImageblock {
                origin, dimensions, ..
            } => {
                let (buffer, source_dimensions, pixel_size) =
                    output_resources.fragment_imageblock.ok_or_else(|| {
                        "selected Metal fragment imageblock output was not resolved".to_string()
                    })?;
                let bytes = unsafe {
                    std::slice::from_raw_parts(
                        buffer.contents().as_ptr().cast::<u8>(),
                        buffer.length(),
                    )
                };
                crate::literal::select_tightly_packed_2d(
                    bytes,
                    source_dimensions,
                    origin,
                    dimensions,
                    pixel_size,
                )
            }
        }
    }

    fn read_texture_region(
        texture: &ProtocolObject<dyn MTLTexture>,
        texture_type: TextureType,
        format: TextureFormat,
        origin: [u32; 3],
        dimensions: [u32; 3],
    ) -> Result<Vec<u8>, String> {
        let bytes_per_row = dimensions[0] as usize * format.bytes_per_pixel();
        let bytes_per_image = bytes_per_row * dimensions[1] as usize;
        let mut output = vec![0u8; bytes_per_image * dimensions[2] as usize];
        if texture_type == TextureType::D3 {
            let pointer = NonNull::new(output.as_mut_ptr().cast::<c_void>())
                .ok_or_else(|| "output texture pointer is null".to_string())?;
            unsafe {
                texture.getBytes_bytesPerRow_bytesPerImage_fromRegion_mipmapLevel_slice(
                    pointer,
                    bytes_per_row,
                    bytes_per_image,
                    MTLRegion {
                        origin: MTLOrigin {
                            x: origin[0] as usize,
                            y: origin[1] as usize,
                            z: origin[2] as usize,
                        },
                        size: MTLSize {
                            width: dimensions[0] as usize,
                            height: dimensions[1] as usize,
                            depth: dimensions[2] as usize,
                        },
                    },
                    0,
                    0,
                );
            }
        } else {
            for selected_slice in 0..dimensions[2] as usize {
                let pointer = unsafe {
                    NonNull::new_unchecked(
                        output
                            .as_mut_ptr()
                            .add(selected_slice * bytes_per_image)
                            .cast::<c_void>(),
                    )
                };
                unsafe {
                    texture.getBytes_bytesPerRow_bytesPerImage_fromRegion_mipmapLevel_slice(
                        pointer,
                        bytes_per_row,
                        bytes_per_image,
                        MTLRegion {
                            origin: MTLOrigin {
                                x: origin[0] as usize,
                                y: origin[1] as usize,
                                z: 0,
                            },
                            size: MTLSize {
                                width: dimensions[0] as usize,
                                height: dimensions[1] as usize,
                                depth: 1,
                            },
                        },
                        0,
                        origin[2] as usize + selected_slice,
                    );
                }
            }
        }
        Ok(output)
    }

    fn mtl_size(size: [u32; 3]) -> MTLSize {
        MTLSize {
            width: size[0] as usize,
            height: size[1] as usize,
            depth: size[2] as usize,
        }
    }

    pub fn environment() -> Result<serde_json::Value, String> {
        let device = MTLCreateSystemDefaultDevice()
            .ok_or_else(|| "MTLCreateSystemDefaultDevice returned nil".to_string())?;
        let os = Command::new("sw_vers")
            .arg("-productVersion")
            .output()
            .map_err(|error| format!("run sw_vers: {error}"))?;
        Ok(serde_json::json!({
            "device": device.name().to_string(),
            "os": String::from_utf8_lossy(&os.stdout).trim(),
            "architecture": std::env::consts::ARCH,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qualification_requires_three_identical_outputs_and_accepts_identity() {
        let initial = [0xab, 0xab, 0xab, 0xab];
        let output = vec![42, 0, 0, 0];
        assert_eq!(
            qualify_outputs(
                &initial,
                vec![output.clone(), output.clone(), output.clone()]
            )
            .unwrap(),
            output
        );
        assert!(qualify_outputs(
            &initial,
            vec![vec![42, 0, 0, 0], vec![43, 0, 0, 0], vec![42, 0, 0, 0]]
        )
        .unwrap_err()
        .contains("nondeterministic"));
        assert_eq!(
            qualify_outputs(
                &initial,
                vec![initial.to_vec(), initial.to_vec(), initial.to_vec()]
            )
            .unwrap(),
            initial
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn attachmentless_fragment_executes_without_invented_output() {
        use crate::library_module::ResolvedLinkedFunctions;
        use crate::source::SourceRow;

        let air_ll = crate::case::ATTACHMENTLESS_FRAGMENT_AIR;
        let source = SourceRow {
            air_sha256: crate::hash::sha256_bytes(air_ll.as_bytes()),
            stage: "Fragment".into(),
            entry: "fragment_no_writes".into(),
            air_ll: air_ll.into(),
            blob_b64: Some(
                base64::engine::general_purpose::STANDARD
                    .encode(compile_metal_fixture("fragment_no_writes")),
            ),
            lib_sha256s: vec!["10".repeat(32)],
            label: "test/fragment-no-writes.air".into(),
        };
        let case = crate::case::attachmentless_fragment_test_case(
            source.air_sha256.clone(),
            source.entry.clone(),
        );
        let reflection = metal2vulkan::reflect_sanitized(
            &source.air_ll,
            metal2vulkan::passes::Stage::Fragment,
            metal2vulkan::passes::TransformOptions::default(),
        )
        .unwrap();
        let resources = LiteralResources::prepare(&case).unwrap();
        assert!(platform::execute(
            &case,
            &source,
            &resources,
            &reflection,
            &ResolvedLinkedFunctions::default(),
        )
        .unwrap()
        .is_empty());
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn rasterization_disabled_vertex_executes_narrow_attributes() {
        use crate::case::{
            AttributeFormat, AttributeInput, BufferResource, Comparison, Draw, ExecutionSafety,
            OutputSelection, Primitive, ResourceRole, Stage,
        };
        use crate::library_module::ResolvedLinkedFunctions;
        use crate::source::SourceRow;

        let air_ll = include_str!("../fixtures/public/vertex_narrow_attributes.ll");
        let source = SourceRow {
            air_sha256: crate::hash::sha256_bytes(air_ll.as_bytes()),
            stage: "Vertex".into(),
            entry: "vertex_narrow_attributes".into(),
            air_ll: air_ll.into(),
            blob_b64: Some(
                base64::engine::general_purpose::STANDARD
                    .encode(compile_metal_fixture("vertex_narrow_attributes")),
            ),
            lib_sha256s: vec!["11".repeat(32)],
            label: "test/vertex-side-effect.air".into(),
        };
        let case = AuthoredCase {
            air_sha256: source.air_sha256.clone(),
            case_id: "test-vertex-side-effect".into(),
            name: "vertex-narrow-attributes-smoke".into(),
            entry: source.entry.clone(),
            stage: Stage::Vertex,
            buffers: vec![BufferResource {
                binding: 0,
                role: ResourceRole::Output,
                bytes_b64: None,
                initial_bytes_b64: Some("q6urqw==".into()),
            }],
            argument_buffer_buffers: vec![],
            device_buffer_arrays: vec![],
            threadgroup_memory: vec![],
            imageblock: None,
            fragment_imageblock: None,
            acceleration_structures: vec![],
            visible_function_references: vec![],
            visible_function_tables: vec![],
            intersection_function_tables: vec![],
            argument_buffer_intersection_function_tables: vec![],
            textures: vec![],
            texture_arrays: vec![],
            argument_buffer_textures: vec![],
            samplers: vec![],
            render_targets: vec![],
            depth_stencil: None,
            vertex_inputs: vec![
                AttributeInput {
                    location: 0,
                    format: AttributeFormat::Uchar,
                    stride: 1,
                    bytes_b64: "Ag==".into(),
                },
                AttributeInput {
                    location: 1,
                    format: AttributeFormat::Ushort2,
                    stride: 4,
                    bytes_b64: "AwAEAA==".into(),
                },
            ],
            vertex_observation: None,
            kernel_stage_inputs: vec![],
            function_constants: vec![],
            dispatch: None,
            draw: Some(Draw {
                primitive: Primitive::Point,
                vertex_start: 0,
                vertex_count: 1,
                instance_count: 1,
            }),
            tessellation: None,
            output: OutputSelection::Buffer {
                binding: 0,
                offset: 0,
                length: 4,
            },
            compare: Comparison::Exact,
            execution_safety: ExecutionSafety::LoopFree,
            rationale: None,
            authored_by: Some("codex:gpt-5.6-sol".into()),
        };
        let reflection = metal2vulkan::reflect_sanitized(
            &source.air_ll,
            metal2vulkan::passes::Stage::Vertex,
            metal2vulkan::passes::TransformOptions::default(),
        )
        .unwrap();
        let resources = LiteralResources::prepare(&case).unwrap();
        assert_eq!(
            platform::execute(
                &case,
                &source,
                &resources,
                &reflection,
                &ResolvedLinkedFunctions::default(),
            )
            .unwrap(),
            9u32.to_le_bytes()
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn combined_depth_stencil_outputs_use_separate_native_attachments() {
        use crate::library_module::ResolvedLinkedFunctions;
        use crate::source::SourceRow;

        let air_ll = include_str!("../fixtures/public/fragment_depth_stencil.ll");
        let source = SourceRow {
            air_sha256: crate::hash::sha256_bytes(air_ll.as_bytes()),
            stage: "Fragment".into(),
            entry: "fragment_depth_stencil".into(),
            air_ll: air_ll.into(),
            blob_b64: Some(
                base64::engine::general_purpose::STANDARD
                    .encode(compile_metal_fixture("fragment_depth_stencil")),
            ),
            lib_sha256s: vec!["22".repeat(32)],
            label: "test/fragment-depth-stencil.air".into(),
        };
        let case = crate::case::combined_depth_stencil_test_case(
            source.air_sha256.clone(),
            source.entry.clone(),
        );
        let reflection = metal2vulkan::reflect_sanitized(
            &source.air_ll,
            metal2vulkan::passes::Stage::Fragment,
            metal2vulkan::passes::TransformOptions::default(),
        )
        .unwrap();
        let resources = LiteralResources::prepare(&case).unwrap();
        assert_eq!(
            platform::execute(
                &case,
                &source,
                &resources,
                &reflection,
                &ResolvedLinkedFunctions::default(),
            )
            .unwrap(),
            [7]
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn vector_function_constant_executes_all_authored_lanes_on_metal() {
        use crate::library_module::ResolvedLinkedFunctions;
        use crate::source::SourceRow;

        let air_ll = include_str!("../fixtures/public/kernel_vector_function_constant.ll");
        let source = SourceRow {
            air_sha256: crate::hash::sha256_bytes(air_ll.as_bytes()),
            stage: "Kernel".into(),
            entry: "kernel_vector_function_constant".into(),
            air_ll: air_ll.into(),
            blob_b64: Some(
                base64::engine::general_purpose::STANDARD
                    .encode(compile_metal_fixture("kernel_vector_function_constant")),
            ),
            lib_sha256s: vec!["33".repeat(32)],
            label: "test/kernel-vector-function-constant.air".into(),
        };
        let case = crate::case::vector_function_constant_test_case(
            source.air_sha256.clone(),
            source.entry.clone(),
        );
        let reflection = metal2vulkan::reflect_sanitized(
            &source.air_ll,
            metal2vulkan::passes::Stage::Kernel,
            metal2vulkan::passes::TransformOptions::default(),
        )
        .unwrap();
        let resources = LiteralResources::prepare(&case).unwrap();
        assert_eq!(
            platform::execute(
                &case,
                &source,
                &resources,
                &reflection,
                &ResolvedLinkedFunctions::default(),
            )
            .unwrap(),
            10u32.to_le_bytes()
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn authored_tessellation_executes_native_patch_inputs() {
        use crate::case::{
            AttributeFormat, AttributeInput, Comparison, ExecutionSafety, OutputSelection,
            RenderTargetResource, Stage, TessellationDraw, TessellationFactors, TextureFormat,
            VertexObservation,
        };
        use crate::library_module::ResolvedLinkedFunctions;
        use crate::source::SourceRow;

        let air_ll = r#"
define <{ <4 x float>, <4 x float> }> @tessellation_literal(ptr %control, <4 x float> %patch, <2 x float> %coordinate) { ret <{ <4 x float>, <4 x float> }> zeroinitializer }
declare { <3 x float> } @control.MTL_CONTROL_POINT_FN(i32, ptr) section "air.externally_defined"
!air.vertex = !{!0}
!0 = !{ptr @tessellation_literal, !1, !2, !8}
!1 = !{!3, !12}
!2 = !{!4, !7, !9}
!3 = !{!"air.position", !"air.arg_type_name", !"float4"}
!4 = !{i32 0, !"air.patch_control_point_input", !5, !6}
!5 = !{!"air.patch_control_point_function", ptr @control.MTL_CONTROL_POINT_FN}
!6 = !{!"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"float3"}
!7 = !{i32 1, !"air.patch_input", !"air.location_index", i32 4, i32 1, !"air.arg_type_name", !"float4"}
!8 = !{!"air.patch", !"quad", !"air.patch_control_point", i32 16}
!9 = !{i32 2, !"air.position_in_patch", !"air.arg_type_name", !"float2"}
!12 = !{!"air.vertex_output", !"user(locn0)", !"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"float4", !"air.arg_name", !"color"}
"#;
        let source = SourceRow {
            air_sha256: crate::hash::sha256_bytes(air_ll.as_bytes()),
            stage: "Vertex".into(),
            entry: "tessellation_literal".into(),
            air_ll: air_ll.into(),
            blob_b64: Some(
                base64::engine::general_purpose::STANDARD
                    .encode(compile_metal_fixture("tessellation_literal")),
            ),
            lib_sha256s: vec!["11".repeat(32)],
            label: "test/tessellation-literal.air".into(),
        };
        let case = AuthoredCase {
            air_sha256: source.air_sha256.clone(),
            case_id: "test-tessellation-literal".into(),
            name: "tessellation-literal-smoke".into(),
            entry: source.entry.clone(),
            stage: Stage::Vertex,
            buffers: vec![],
            argument_buffer_buffers: vec![],
            device_buffer_arrays: vec![],
            threadgroup_memory: vec![],
            imageblock: None,
            fragment_imageblock: None,
            acceleration_structures: vec![],
            visible_function_references: vec![],
            visible_function_tables: vec![],
            intersection_function_tables: vec![],
            argument_buffer_intersection_function_tables: vec![],
            textures: vec![],
            texture_arrays: vec![],
            argument_buffer_textures: vec![],
            samplers: vec![],
            render_targets: vec![RenderTargetResource {
                index: 0,
                format: TextureFormat::Rgba32Float,
                dimensions: [1, 1],
                initial_bytes_b64: base64::engine::general_purpose::STANDARD.encode([0; 16]),
            }],
            depth_stencil: None,
            vertex_inputs: vec![],
            vertex_observation: Some(VertexObservation::Varying { location: 0 }),
            kernel_stage_inputs: vec![],
            function_constants: vec![],
            dispatch: None,
            draw: None,
            tessellation: Some(TessellationDraw {
                factors: vec![TessellationFactors {
                    edge_f16: vec![0x3c00; 4],
                    inside_f16: vec![0x3c00; 2],
                }],
                instance_count: 1,
                amplification_count: 1,
                control_points: vec![AttributeInput {
                    location: 0,
                    format: AttributeFormat::Float3,
                    stride: 12,
                    bytes_b64: base64::engine::general_purpose::STANDARD.encode([0; 16 * 12]),
                }],
                patch_inputs: vec![AttributeInput {
                    location: 4,
                    format: AttributeFormat::Float4,
                    stride: 16,
                    bytes_b64: base64::engine::general_purpose::STANDARD
                        .encode([0, 0, 128, 63, 0, 0, 0, 64, 0, 0, 64, 64, 0, 0, 128, 64]),
                }],
            }),
            output: OutputSelection::RenderTarget {
                index: 0,
                origin: [0, 0],
                dimensions: [1, 1],
            },
            compare: Comparison::Exact,
            execution_safety: ExecutionSafety::LoopFree,
            rationale: None,
            authored_by: Some("codex:gpt-5.6-sol".into()),
        };
        let reflection = metal2vulkan::reflect_sanitized(
            &source.air_ll,
            metal2vulkan::passes::Stage::Vertex,
            metal2vulkan::passes::TransformOptions::default(),
        )
        .unwrap();
        let resources = LiteralResources::prepare(&case).unwrap();
        let output = platform::execute(
            &case,
            &source,
            &resources,
            &reflection,
            &ResolvedLinkedFunctions::default(),
        )
        .unwrap();
        assert_eq!(
            output,
            [0, 0, 128, 63, 0, 0, 0, 64, 0, 0, 64, 64, 0, 0, 128, 64]
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn visible_function_table_executes_a_separately_linked_air_function() {
        use crate::case::{
            BufferResource, Comparison, Dispatch, ExecutionSafety, FunctionTableEntry,
            FunctionTableResource, OutputSelection, ResourceRole, Stage,
        };
        use crate::library_module::{
            LibraryModuleRow, ResolvedFunctionEntry, ResolvedFunctionTable, ResolvedLinkedFunctions,
        };
        use crate::source::SourceRow;

        let entry_blob = compile_metal_fixture("kernel_visible_function_table_word");
        let helper_blob = compile_metal_fixture("visible_function_add_one");
        let air_ll = r#"
define void @kernel_visible_function_table_word(ptr addrspace(1) %output, ptr addrspace(1) %functions) {
entry:
  ret void
}
!air.kernel = !{!0}
!0 = !{ptr @kernel_visible_function_table_word, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.write", !"air.arg_type_name", !"uint"}
!4 = !{i32 1, !"air.visible_function_table", !"air.location_index", i32 1, i32 1, !"air.read", !"air.arg_type_name", !"visible_function_table"}
"#;
        let module_ll = "define i32 @visible_function_add_one(i32 %value) { ret i32 %value }";
        let module_sha256 = crate::hash::sha256_bytes(module_ll.as_bytes());
        let source = SourceRow {
            air_sha256: crate::hash::sha256_bytes(air_ll.as_bytes()),
            stage: "Kernel".into(),
            entry: "kernel_visible_function_table_word".into(),
            air_ll: air_ll.into(),
            blob_b64: Some(base64::engine::general_purpose::STANDARD.encode(entry_blob)),
            lib_sha256s: vec!["11".repeat(32)],
            label: "test/kernel-visible-function-table.air".into(),
        };
        let case = AuthoredCase {
            air_sha256: source.air_sha256.clone(),
            case_id: "test-visible-function-table".into(),
            name: "visible-function-table-smoke".into(),
            entry: source.entry.clone(),
            stage: Stage::Kernel,
            buffers: vec![BufferResource {
                binding: 0,
                role: ResourceRole::Output,
                bytes_b64: None,
                initial_bytes_b64: Some("q6urqw==".into()),
            }],
            argument_buffer_buffers: vec![],
            device_buffer_arrays: vec![],
            threadgroup_memory: vec![],
            imageblock: None,
            fragment_imageblock: None,
            acceleration_structures: vec![],
            visible_function_references: vec![],
            visible_function_tables: vec![FunctionTableResource {
                binding: 1,
                size: 1,
                entries: vec![FunctionTableEntry {
                    index: 0,
                    module_sha256: module_sha256.clone(),
                    function: "visible_function_add_one".into(),
                }],
            }],
            intersection_function_tables: vec![],
            argument_buffer_intersection_function_tables: vec![],
            textures: vec![],
            texture_arrays: vec![],
            argument_buffer_textures: vec![],
            samplers: vec![],
            render_targets: vec![],
            depth_stencil: None,
            vertex_inputs: vec![],
            vertex_observation: None,
            kernel_stage_inputs: vec![],
            function_constants: vec![],
            dispatch: Some(Dispatch {
                grid: [1, 1, 1],
                threads_per_threadgroup: [1, 1, 1],
            }),
            draw: None,
            tessellation: None,
            output: OutputSelection::Buffer {
                binding: 0,
                offset: 0,
                length: 4,
            },
            compare: Comparison::Exact,
            execution_safety: ExecutionSafety::LoopFree,
            rationale: None,
            authored_by: Some("codex:gpt-5.6-sol".into()),
        };
        let module = LibraryModuleRow {
            module_sha256,
            air_ll: module_ll.into(),
            blob_b64: base64::engine::general_purpose::STANDARD.encode(helper_blob),
            lib_sha256s: source.lib_sha256s.clone(),
            label: "test/visible-function-add-one.air".into(),
        };
        let function_tables = ResolvedLinkedFunctions {
            references: vec![],
            visible: vec![ResolvedFunctionTable {
                binding: 1,
                size: 1,
                entries: vec![ResolvedFunctionEntry {
                    index: 0,
                    function: "visible_function_add_one".into(),
                    module,
                }],
            }],
            intersection: vec![],
        };
        let reflection = metal2vulkan::reflect_sanitized(
            &source.air_ll,
            metal2vulkan::passes::Stage::Kernel,
            metal2vulkan::passes::TransformOptions::default(),
        )
        .unwrap();
        let resources = LiteralResources::prepare(&case).unwrap();
        let output =
            platform::execute(&case, &source, &resources, &reflection, &function_tables).unwrap();
        assert_eq!(output, 42u32.to_le_bytes());
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn opaque_triangle_intersection_entry_uses_the_native_sentinel_contract() {
        use crate::case::{
            AccelerationStructureKind, AccelerationStructureResource, BufferResource, Comparison,
            Dispatch, ExecutionSafety, IntersectionFunctionSignature,
            IntersectionFunctionTableEntry, IntersectionFunctionTableResource, OutputSelection,
            ResourceRole, Stage,
        };
        use crate::library_module::{
            ResolvedIntersectionFunctionEntry, ResolvedIntersectionFunctionTable,
            ResolvedLinkedFunctions,
        };
        use crate::source::SourceRow;

        let air_ll = include_str!("../fixtures/public/kernel_instance_as_intersect.ll");
        let source = SourceRow {
            air_sha256: crate::hash::sha256_bytes(air_ll.as_bytes()),
            stage: "Kernel".into(),
            entry: "instance_as_intersect".into(),
            air_ll: air_ll.into(),
            blob_b64: Some(
                base64::engine::general_purpose::STANDARD
                    .encode(compile_metal_fixture("kernel_instance_as_intersect")),
            ),
            lib_sha256s: vec!["11".repeat(32)],
            label: "test/kernel-instance-as-intersect.air".into(),
        };
        let signature = vec![
            IntersectionFunctionSignature::Instancing,
            IntersectionFunctionSignature::TriangleData,
            IntersectionFunctionSignature::IntersectionFunctionBuffer,
        ];
        let case = AuthoredCase {
            air_sha256: source.air_sha256.clone(),
            case_id: "test-opaque-intersection-table".into(),
            name: "opaque-intersection-table-smoke".into(),
            entry: source.entry.clone(),
            stage: Stage::Kernel,
            buffers: vec![BufferResource {
                binding: 0,
                role: ResourceRole::Output,
                bytes_b64: None,
                initial_bytes_b64: Some(
                    base64::engine::general_purpose::STANDARD.encode([0xabu8; 36]),
                ),
            }],
            argument_buffer_buffers: vec![],
            device_buffer_arrays: vec![],
            threadgroup_memory: vec![],
            imageblock: None,
            fragment_imageblock: None,
            acceleration_structures: vec![AccelerationStructureResource {
                binding: 5,
                kind: AccelerationStructureKind::Instance,
                primitive_triangles_b64: None,
                child_references: vec![0],
            }],
            visible_function_references: vec![],
            visible_function_tables: vec![],
            intersection_function_tables: vec![IntersectionFunctionTableResource {
                binding: 6,
                size: 1,
                entries: vec![IntersectionFunctionTableEntry::OpaqueTriangle {
                    index: 0,
                    signature: signature.clone(),
                }],
            }],
            argument_buffer_intersection_function_tables: vec![],
            textures: vec![],
            texture_arrays: vec![],
            argument_buffer_textures: vec![],
            samplers: vec![],
            render_targets: vec![],
            depth_stencil: None,
            vertex_inputs: vec![],
            vertex_observation: None,
            kernel_stage_inputs: vec![],
            function_constants: vec![],
            dispatch: Some(Dispatch {
                grid: [1, 1, 1],
                threads_per_threadgroup: [1, 1, 1],
            }),
            draw: None,
            tessellation: None,
            output: OutputSelection::Buffer {
                binding: 0,
                offset: 0,
                length: 36,
            },
            compare: Comparison::Exact,
            execution_safety: ExecutionSafety::LoopFree,
            rationale: None,
            authored_by: Some("codex:gpt-5.6-sol".into()),
        };
        let function_tables = ResolvedLinkedFunctions {
            references: vec![],
            visible: vec![],
            intersection: vec![ResolvedIntersectionFunctionTable {
                location:
                    crate::library_module::ResolvedIntersectionFunctionTableLocation::Direct {
                        binding: 6,
                    },
                size: 1,
                entries: vec![ResolvedIntersectionFunctionEntry::OpaqueTriangle {
                    index: 0,
                    signature,
                }],
            }],
        };
        let reflection = metal2vulkan::reflect_sanitized(
            &source.air_ll,
            metal2vulkan::passes::Stage::Kernel,
            metal2vulkan::passes::TransformOptions::default(),
        )
        .unwrap();
        let resources = LiteralResources::prepare(&case).unwrap();
        let output =
            platform::execute(&case, &source, &resources, &reflection, &function_tables).unwrap();
        assert_eq!(u32::from_le_bytes(output[0..4].try_into().unwrap()), 1);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn custom_fragment_imageblock_executes_on_metal() {
        use crate::library_module::ResolvedLinkedFunctions;
        use crate::source::SourceRow;

        let air_ll = include_str!("../fixtures/public/fragment_custom_imageblock.ll");
        let source = SourceRow {
            air_sha256: crate::hash::sha256_bytes(air_ll.as_bytes()),
            stage: "Fragment".into(),
            entry: "fragment_custom_imageblock".into(),
            air_ll: air_ll.into(),
            blob_b64: Some(
                base64::engine::general_purpose::STANDARD
                    .encode(compile_metal_fixture("fragment_custom_imageblock")),
            ),
            lib_sha256s: vec!["12".repeat(32)],
            label: "test/fragment-custom-imageblock.air".into(),
        };
        let case = crate::case::fragment_imageblock_test_case(
            source.air_sha256.clone(),
            source.entry.clone(),
        );
        let reflection = metal2vulkan::reflect_sanitized(
            &source.air_ll,
            metal2vulkan::passes::Stage::Fragment,
            metal2vulkan::passes::TransformOptions::default(),
        )
        .unwrap();
        let resources = LiteralResources::prepare(&case).unwrap();
        let output = platform::execute(
            &case,
            &source,
            &resources,
            &reflection,
            &ResolvedLinkedFunctions::default(),
        )
        .unwrap();
        assert_eq!(output, [0x00, 0x40]);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn argument_buffer_intersection_table_is_encoded_at_reflected_field() {
        use crate::case::{
            AccelerationStructureKind, AccelerationStructureResource,
            ArgumentBufferIntersectionFunctionTableResource, BufferResource, Comparison, Dispatch,
            ExecutionSafety, IntersectionFunctionSignature, IntersectionFunctionTableEntry,
            OutputSelection, ResourceRole, Stage,
        };
        use crate::library_module::{
            ResolvedIntersectionFunctionEntry, ResolvedIntersectionFunctionTable,
            ResolvedIntersectionFunctionTableLocation, ResolvedLinkedFunctions,
        };
        use crate::source::SourceRow;

        let air_ll = r#"
%struct.RayResources = type { ptr addrspace(1) }
define void @argument_buffer_intersection_table(ptr addrspace(1) %as, ptr addrspace(2) %resources, ptr addrspace(1) %output) {
entry:
  ret void
}
!air.kernel = !{!0}
!0 = !{ptr @argument_buffer_intersection_table, !1, !2}
!1 = !{}
!2 = !{!3, !4, !7}
!3 = !{i32 0, !"air.instance_acceleration_structure", !"air.location_index", i32 5, i32 1, !"air.read", !"air.arg_type_name", !"instance_acceleration_structure"}
!4 = !{i32 1, !"air.indirect_buffer", !"air.buffer_size", i32 256, !"air.location_index", i32 6, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !5, !"air.arg_type_name", !"RayResources"}
!5 = !{i32 0, i32 8, i32 0, !"void", !"table", !"air.indirect_argument", !6}
!6 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.arg_type_name", !"void"}
!7 = !{i32 2, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_name", !"uint"}
"#;
        let source = SourceRow {
            air_sha256: crate::hash::sha256_bytes(air_ll.as_bytes()),
            stage: "Kernel".into(),
            entry: "argument_buffer_intersection_table".into(),
            air_ll: air_ll.into(),
            blob_b64: Some(base64::engine::general_purpose::STANDARD.encode(
                compile_metal_fixture("kernel_argument_buffer_intersection_table"),
            )),
            lib_sha256s: vec!["11".repeat(32)],
            label: "test/kernel-argument-buffer-intersection-table.air".into(),
        };
        let signature = vec![
            IntersectionFunctionSignature::Instancing,
            IntersectionFunctionSignature::TriangleData,
            IntersectionFunctionSignature::IntersectionFunctionBuffer,
        ];
        let case = AuthoredCase {
            air_sha256: source.air_sha256.clone(),
            case_id: "test-argument-buffer-intersection-table".into(),
            name: "argument-buffer-intersection-table-smoke".into(),
            entry: source.entry.clone(),
            stage: Stage::Kernel,
            buffers: vec![
                BufferResource {
                    binding: 0,
                    role: ResourceRole::Output,
                    bytes_b64: None,
                    initial_bytes_b64: Some(
                        base64::engine::general_purpose::STANDARD.encode([0xabu8; 4]),
                    ),
                },
                BufferResource {
                    binding: 6,
                    role: ResourceRole::Input,
                    bytes_b64: Some(base64::engine::general_purpose::STANDARD.encode([0u8; 256])),
                    initial_bytes_b64: None,
                },
            ],
            argument_buffer_buffers: vec![],
            device_buffer_arrays: vec![],
            threadgroup_memory: vec![],
            imageblock: None,
            fragment_imageblock: None,
            acceleration_structures: vec![AccelerationStructureResource {
                binding: 5,
                kind: AccelerationStructureKind::Instance,
                primitive_triangles_b64: None,
                child_references: vec![0],
            }],
            visible_function_references: vec![],
            visible_function_tables: vec![],
            intersection_function_tables: vec![],
            argument_buffer_intersection_function_tables: vec![
                ArgumentBufferIntersectionFunctionTableResource {
                    buffer_binding: 6,
                    field_offset: 0,
                    size: 1,
                    entries: vec![IntersectionFunctionTableEntry::OpaqueTriangle {
                        index: 0,
                        signature: signature.clone(),
                    }],
                },
            ],
            textures: vec![],
            texture_arrays: vec![],
            argument_buffer_textures: vec![],
            samplers: vec![],
            render_targets: vec![],
            depth_stencil: None,
            vertex_inputs: vec![],
            vertex_observation: None,
            kernel_stage_inputs: vec![],
            function_constants: vec![],
            dispatch: Some(Dispatch {
                grid: [1, 1, 1],
                threads_per_threadgroup: [1, 1, 1],
            }),
            draw: None,
            tessellation: None,
            output: OutputSelection::Buffer {
                binding: 0,
                offset: 0,
                length: 4,
            },
            compare: Comparison::Exact,
            execution_safety: ExecutionSafety::LoopFree,
            rationale: None,
            authored_by: Some("codex:gpt-5.6-sol".into()),
        };
        let function_tables = ResolvedLinkedFunctions {
            references: vec![],
            visible: vec![],
            intersection: vec![ResolvedIntersectionFunctionTable {
                location: ResolvedIntersectionFunctionTableLocation::ArgumentBuffer {
                    buffer_binding: 6,
                    field_offset: 0,
                },
                size: 1,
                entries: vec![ResolvedIntersectionFunctionEntry::OpaqueTriangle {
                    index: 0,
                    signature,
                }],
            }],
        };
        let reflection = metal2vulkan::reflect_sanitized(
            &source.air_ll,
            metal2vulkan::passes::Stage::Kernel,
            metal2vulkan::passes::TransformOptions::default(),
        )
        .unwrap();
        assert_eq!(reflection.argument_buffer_fields[0].buffer_index, 6);
        assert_eq!(reflection.argument_buffer_fields[0].field_offset, 0);
        let resources = LiteralResources::prepare(&case).unwrap();
        let output =
            platform::execute(&case, &source, &resources, &reflection, &function_tables).unwrap();
        assert_eq!(u32::from_le_bytes(output.try_into().unwrap()), 1);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn implicit_imageblock_kernel_executes_as_a_tile_pipeline() {
        use crate::case::{
            BufferResource, Comparison, Dispatch, ExecutionSafety, ImageblockResource,
            OutputSelection, RenderTargetResource, ResourceRole, Stage, TextureFormat,
        };
        use crate::library_module::ResolvedLinkedFunctions;
        use crate::source::SourceRow;

        let air_ll = r#"
%imageblock = type opaque
define void @kernel_implicit_imageblock_half4(ptr addrspace(4) %block, <2 x i16> %position) {
entry:
  %value = call <4 x half> @air.load.implicit_imageblock.v4f16(i32 0, <2 x i16> %position, i32 0, i16 0)
  call void @air.store.implicit_imageblock.v4f16(<4 x half> %value, i32 0, <2 x i16> %position, i32 0, i16 0)
  ret void
}
declare <4 x half> @air.load.implicit_imageblock.v4f16(i32, <2 x i16>, i32, i16)
declare void @air.store.implicit_imageblock.v4f16(<4 x half>, i32, <2 x i16>, i32, i16)
!air.kernel = !{!0}
!0 = !{ptr @kernel_implicit_imageblock_half4, !1, !2}
!1 = !{}
!2 = !{!3, !5}
!3 = !{i32 0, !"air.imageblock", !"implicit", !"air.struct_type_info", !4, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"imageblock<ImplicitColor, layout_implicit>"}
!4 = !{i32 0, i32 8, i32 0, !"half4", !"value", !"air.render_target", i32 0}
!5 = !{i32 1, !"air.thread_position_in_threadgroup", !"air.arg_type_name", !"ushort2"}
"#;
        let source = SourceRow {
            air_sha256: crate::hash::sha256_bytes(air_ll.as_bytes()),
            stage: "Kernel".into(),
            entry: "kernel_implicit_imageblock_half4".into(),
            air_ll: air_ll.into(),
            blob_b64: Some(
                base64::engine::general_purpose::STANDARD
                    .encode(compile_metal_fixture("kernel_implicit_imageblock_half4")),
            ),
            lib_sha256s: vec!["11".repeat(32)],
            label: "test/kernel-implicit-imageblock-half4.air".into(),
        };
        let initial = [0x00, 0x3c, 0x00, 0x40, 0x00, 0x42, 0x00, 0x44];
        let case = AuthoredCase {
            air_sha256: source.air_sha256.clone(),
            case_id: "22".repeat(32),
            name: "implicit-imageblock-tile-smoke".into(),
            entry: source.entry.clone(),
            stage: Stage::Kernel,
            buffers: vec![BufferResource {
                binding: 0,
                role: ResourceRole::Output,
                bytes_b64: None,
                initial_bytes_b64: Some(
                    base64::engine::general_purpose::STANDARD.encode([0u8; 16 * 16 * 8]),
                ),
            }],
            argument_buffer_buffers: vec![],
            device_buffer_arrays: vec![],
            threadgroup_memory: vec![],
            imageblock: Some(ImageblockResource {
                dimensions: [16, 16],
                implicit_coverage: Some(crate::case::ImplicitImageblockCoverage::FullSingleSample),
            }),
            fragment_imageblock: None,
            acceleration_structures: vec![],
            visible_function_references: vec![],
            visible_function_tables: vec![],
            intersection_function_tables: vec![],
            argument_buffer_intersection_function_tables: vec![],
            textures: vec![],
            texture_arrays: vec![],
            argument_buffer_textures: vec![],
            samplers: vec![],
            render_targets: vec![RenderTargetResource {
                index: 0,
                format: TextureFormat::Rgba16Float,
                dimensions: [16, 16],
                initial_bytes_b64: base64::engine::general_purpose::STANDARD
                    .encode(initial.repeat(16 * 16)),
            }],
            depth_stencil: None,
            vertex_inputs: vec![],
            vertex_observation: None,
            kernel_stage_inputs: vec![],
            function_constants: vec![],
            dispatch: Some(Dispatch {
                grid: [16, 16, 1],
                threads_per_threadgroup: [16, 16, 1],
            }),
            draw: None,
            tessellation: None,
            output: OutputSelection::RenderTarget {
                index: 0,
                origin: [0, 0],
                dimensions: [1, 1],
            },
            compare: Comparison::Exact,
            execution_safety: ExecutionSafety::LoopFree,
            rationale: None,
            authored_by: Some("codex:gpt-5.6-sol".into()),
        };
        case.validate_literal_resources().unwrap();
        let reflection = metal2vulkan::reflect_sanitized(
            &source.air_ll,
            metal2vulkan::passes::Stage::Kernel,
            metal2vulkan::passes::TransformOptions {
                kernel_local_size: [16, 16, 1],
                kernel_dispatch: Some(metal2vulkan::reflect::KernelDispatch::ThreadsFixed {
                    threads_per_grid: [16, 16, 1],
                }),
                ..metal2vulkan::passes::TransformOptions::default()
            },
        )
        .unwrap();
        assert_eq!(reflection.implicit_imageblock_attachments.len(), 1);
        let resources = LiteralResources::prepare(&case).unwrap();
        let output = platform::execute(
            &case,
            &source,
            &resources,
            &reflection,
            &ResolvedLinkedFunctions::default(),
        )
        .unwrap();
        assert_eq!(output, [0x00, 0x40, 0x00, 0x42, 0x00, 0x44, 0x00, 0x45]);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn narrow_implicit_imageblock_executes_as_a_tile_pipeline() {
        use crate::library_module::ResolvedLinkedFunctions;
        use crate::source::SourceRow;
        use base64::Engine as _;

        let air_ll = include_str!("../fixtures/public/kernel_implicit_imageblock_half2.ll");
        let source = SourceRow {
            air_sha256: crate::hash::sha256_bytes(air_ll.as_bytes()),
            stage: "Kernel".into(),
            entry: "kernel_implicit_imageblock_half2".into(),
            air_ll: air_ll.into(),
            blob_b64: Some(
                base64::engine::general_purpose::STANDARD
                    .encode(compile_metal_fixture("kernel_implicit_imageblock_half2")),
            ),
            lib_sha256s: vec!["11".repeat(32)],
            label: "test/kernel-implicit-imageblock-half2.air".into(),
        };
        let case = crate::case::narrow_implicit_imageblock_test_case(
            source.air_sha256.clone(),
            source.entry.clone(),
        );
        let options = metal2vulkan::passes::TransformOptions {
            kernel_local_size: [16, 16, 1],
            kernel_dispatch: Some(metal2vulkan::reflect::KernelDispatch::ThreadsFixed {
                threads_per_grid: [16, 16, 1],
            }),
            ..metal2vulkan::passes::TransformOptions::default()
        };
        let reflection = metal2vulkan::reflect_sanitized(
            &source.air_ll,
            metal2vulkan::passes::Stage::Kernel,
            options,
        )
        .unwrap();
        let resources = LiteralResources::prepare(&case).unwrap();
        let output = platform::execute(
            &case,
            &source,
            &resources,
            &reflection,
            &ResolvedLinkedFunctions::default(),
        )
        .unwrap();
        assert_eq!(output, [0x00, 0x3c, 0x00, 0x40]);
    }

    #[cfg(target_os = "macos")]
    fn compile_metal_fixture(name: &str) -> Vec<u8> {
        let scratch = crate::ScratchDir::new(name).unwrap();
        let source = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("fixtures/public")
            .join(name)
            .with_extension("metal");
        let air = scratch.path().join("fixture.air");
        let output = std::process::Command::new("xcrun")
            .args(["-sdk", "macosx", "metal", "-c"])
            .arg(&source)
            .arg("-o")
            .arg(&air)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "compile {}: {}{}",
            source.display(),
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        std::fs::read(air).unwrap()
    }
}

#[cfg(not(target_os = "macos"))]
mod platform {
    use super::*;
    use crate::source::SourceRow;

    pub fn execute(
        _case: &AuthoredCase,
        _source: &SourceRow,
        _resources: &LiteralResources,
        _reflection: &metal2vulkan::reflect::ShaderReflection,
        _function_tables: &crate::library_module::ResolvedLinkedFunctions,
    ) -> Result<Vec<u8>, String> {
        Err("Metal qualification requires macOS".into())
    }

    pub fn environment() -> Result<serde_json::Value, String> {
        Err("Metal qualification requires macOS".into())
    }
}
