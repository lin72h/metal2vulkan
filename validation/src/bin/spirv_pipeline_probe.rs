//! Minimal Vulkan compute-pipeline probe for an already-emitted SPIR-V module.
//!
//! Usage: `spirv_pipeline_probe [--module-only|--layout-only|--pipeline] <module.spv>`.

use std::fs;
use std::path::PathBuf;

use vulkano::device::QueueFlags;
use vulkano::pipeline::compute::ComputePipelineCreateInfo;
use vulkano::pipeline::layout::{PipelineDescriptorSetLayoutCreateInfo, PipelineLayout};
use vulkano::pipeline::{ComputePipeline, PipelineShaderStageCreateInfo};
use vulkano::shader::spirv::bytes_to_words;
use vulkano::shader::{ShaderModule, ShaderModuleCreateInfo};

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
    let (device, _queue) =
        metal2vulkan_validation::runner_linux::device_and_queue_result(QueueFlags::COMPUTE, false)?;
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
