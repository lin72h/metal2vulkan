//! Minimal Vulkan compute-pipeline probe for an already-emitted SPIR-V module.
//!
//! Usage: `spirv_pipeline_probe [--module-only|--layout-only|--pipeline] <module.spv>`.

use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

use vulkano::device::physical::PhysicalDeviceType;
use vulkano::device::{
    Device, DeviceCreateInfo, DeviceExtensions, DeviceFeatures, Queue, QueueCreateInfo, QueueFlags,
};
use vulkano::instance::{Instance, InstanceCreateFlags, InstanceCreateInfo, InstanceExtensions};
use vulkano::pipeline::compute::ComputePipelineCreateInfo;
use vulkano::pipeline::layout::{PipelineDescriptorSetLayoutCreateInfo, PipelineLayout};
use vulkano::pipeline::{ComputePipeline, PipelineShaderStageCreateInfo};
use vulkano::shader::spirv::bytes_to_words;
use vulkano::shader::{ShaderModule, ShaderModuleCreateInfo};
use vulkano::{Version, VulkanLibrary};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ProbeMode {
    ModuleOnly,
    LayoutOnly,
    Pipeline,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("spirv-pipeline-probe: {error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), String> {
    let mut mode = ProbeMode::Pipeline;
    let mut spv_path = None;
    for arg in std::env::args_os().skip(1) {
        match arg.to_str() {
            Some("--module-only") => mode = ProbeMode::ModuleOnly,
            Some("--layout-only") => mode = ProbeMode::LayoutOnly,
            Some("--pipeline") => mode = ProbeMode::Pipeline,
            Some("-h" | "--help") => {
                println!(
                    "usage: spirv_pipeline_probe [--module-only|--layout-only|--pipeline] <module.spv>"
                );
                return Ok(());
            }
            _ if spv_path.is_none() => spv_path = Some(PathBuf::from(arg)),
            _ => {
                return Err(
                    "usage: spirv_pipeline_probe [--module-only|--layout-only|--pipeline] <module.spv>"
                        .into(),
                );
            }
        }
    }
    let spv_path = spv_path.ok_or_else(|| {
        "usage: spirv_pipeline_probe [--module-only|--layout-only|--pipeline] <module.spv>"
            .to_string()
    })?;
    let spv =
        fs::read(&spv_path).map_err(|error| format!("read {}: {error}", spv_path.display()))?;
    let words = bytes_to_words(&spv).map_err(|error| format!("decode SPIR-V words: {error}"))?;
    let device = device()?;
    let module = unsafe { ShaderModule::new(device.clone(), ShaderModuleCreateInfo::new(&words)) }
        .map_err(|error| format!("create shader module: {error}"))?;
    if mode == ProbeMode::ModuleOnly {
        println!("MODULE_OK");
        return Ok(());
    }
    let entry = module
        .entry_point("main")
        .ok_or_else(|| "SPIR-V entry point 'main' is missing".to_string())?;
    let stage = PipelineShaderStageCreateInfo::new(entry);
    let layout = PipelineLayout::new(
        device.clone(),
        PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
            .into_pipeline_layout_create_info(device.clone())
            .map_err(|error| format!("reflect compute pipeline layout: {error}"))?,
    )
    .map_err(|error| format!("create compute pipeline layout: {error}"))?;
    if mode == ProbeMode::LayoutOnly {
        println!("LAYOUT_OK");
        return Ok(());
    }
    let _pipeline = ComputePipeline::new(
        device,
        None,
        ComputePipelineCreateInfo::stage_layout(stage, layout),
    )
    .map_err(|error| format!("create compute pipeline: {error}"))?;
    println!("PIPELINE_OK");
    Ok(())
}

fn device() -> Result<Arc<Device>, String> {
    let library = VulkanLibrary::new().map_err(|error| format!("load Vulkan library: {error}"))?;
    let portability_ext = InstanceExtensions {
        khr_portability_enumeration: true,
        ..InstanceExtensions::empty()
    };
    let want_portability = library.supported_extensions().contains(&portability_ext);
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
            "create Vulkan instance: {error} \
             (VK_ERROR_INCOMPATIBLE_DRIVER usually means no conformant ICD — install \
             mesa-vulkan-drivers/libvulkan1 on Linux, or MoltenVK with portability on macOS)"
        )
    })?;
    let enabled_features = DeviceFeatures {
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
        buffer_device_address: true,
        ..DeviceFeatures::empty()
    };
    let required_extensions = DeviceExtensions {
        ext_shader_atomic_float: true,
        ..DeviceExtensions::empty()
    };
    let device_filter = std::env::var("METAL2VULKAN_VK_DEVICE").ok();
    let (physical_device, queue_family_index) = instance
        .enumerate_physical_devices()
        .map_err(|error| format!("enumerate Vulkan physical devices: {error}"))?
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
            let queue_family_index = device
                .queue_family_properties()
                .iter()
                .position(|family| family.queue_flags.contains(QueueFlags::COMPUTE))?;
            Some((device, queue_family_index as u32))
        })
        .min_by_key(|(device, _)| match device.properties().device_type {
            PhysicalDeviceType::DiscreteGpu => 0,
            PhysicalDeviceType::IntegratedGpu => 1,
            PhysicalDeviceType::VirtualGpu => 2,
            PhysicalDeviceType::Cpu => 3,
            PhysicalDeviceType::Other => 4,
            _ => 5,
        })
        .ok_or_else(|| "no Vulkan device with the required compute features".to_string())?;
    eprintln!(
        "spirv-pipeline-probe: selected device {} ({:?})",
        physical_device.properties().device_name,
        physical_device.properties().device_type
    );
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
            ..DeviceCreateInfo::default()
        },
    )
    .map_err(|error| format!("create Vulkan logical device: {error}"))?;
    let _: Arc<Queue> = queues
        .next()
        .ok_or_else(|| "logical device returned no queue".to_string())?;
    Ok(device)
}
