use serde::{Deserialize, Serialize};
use std::{fmt, str::FromStr};

/// A concrete gap between harvested AIR structure and the authored execution contract.
///
/// This is the shared vocabulary for classification, schema/executor capability checks, cached
/// census rows, and focused audits. Keeping it typed prevents one component from silently
/// inventing a requirement name that another component cannot select or resolve.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolingRequirement {
    FunctionConstantLiteral,
    VertexAttributeLiteral,
    TessellationAttributeLiteral,
    TessellationSystemInputLiteral,
    FragmentImageblockMemberLiteral,
    KernelStageInputLiteral,
    TextureBufferLiteral,
    StorageTextureFormatLiteral,
    StaticSamplerReduction,
    StaticSamplerPixelFilter,
    StaticSamplerBicubic,
    StaticSamplerPixelAddress,
    ImplicitImageblockLiteral,
    RayIntersectionLowering,
    IndirectCommandBuffer,
    VisibleFunctionTable,
    VertexSideEffectObservation,
    FragmentVaryingObservationType,
    FragmentVaryingLinkage,
    FragmentOutputObservationType,
    SynthesizedPlaceholderDescriptor,
}

impl ToolingRequirement {
    pub const ALL: [Self; 21] = [
        Self::FunctionConstantLiteral,
        Self::VertexAttributeLiteral,
        Self::TessellationAttributeLiteral,
        Self::TessellationSystemInputLiteral,
        Self::FragmentImageblockMemberLiteral,
        Self::KernelStageInputLiteral,
        Self::TextureBufferLiteral,
        Self::StorageTextureFormatLiteral,
        Self::StaticSamplerReduction,
        Self::StaticSamplerPixelFilter,
        Self::StaticSamplerBicubic,
        Self::StaticSamplerPixelAddress,
        Self::ImplicitImageblockLiteral,
        Self::RayIntersectionLowering,
        Self::IndirectCommandBuffer,
        Self::VisibleFunctionTable,
        Self::VertexSideEffectObservation,
        Self::FragmentVaryingObservationType,
        Self::FragmentVaryingLinkage,
        Self::FragmentOutputObservationType,
        Self::SynthesizedPlaceholderDescriptor,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::FunctionConstantLiteral => "function_constant_literal",
            Self::VertexAttributeLiteral => "vertex_attribute_literal",
            Self::TessellationAttributeLiteral => "tessellation_attribute_literal",
            Self::TessellationSystemInputLiteral => "tessellation_system_input_literal",
            Self::FragmentImageblockMemberLiteral => "fragment_imageblock_member_literal",
            Self::KernelStageInputLiteral => "kernel_stage_input_literal",
            Self::TextureBufferLiteral => "texture_buffer_literal",
            Self::StorageTextureFormatLiteral => "storage_texture_format_literal",
            Self::StaticSamplerReduction => "static_sampler_reduction",
            Self::StaticSamplerPixelFilter => "static_sampler_pixel_filter",
            Self::StaticSamplerBicubic => "static_sampler_bicubic",
            Self::StaticSamplerPixelAddress => "static_sampler_pixel_address",
            Self::ImplicitImageblockLiteral => "implicit_imageblock_literal",
            Self::RayIntersectionLowering => "ray_intersection_lowering",
            Self::IndirectCommandBuffer => "indirect_command_buffer",
            Self::VisibleFunctionTable => "visible_function_table",
            Self::VertexSideEffectObservation => "vertex_side_effect_observation",
            Self::FragmentVaryingObservationType => "fragment_varying_observation_type",
            Self::FragmentVaryingLinkage => "fragment_varying_linkage",
            Self::FragmentOutputObservationType => "fragment_output_observation_type",
            Self::SynthesizedPlaceholderDescriptor => "synthesized_placeholder_descriptor",
        }
    }
}

impl FromStr for ToolingRequirement {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Self::ALL
            .into_iter()
            .find(|requirement| requirement.as_str() == value)
            .ok_or_else(|| format!("unknown tooling requirement {value:?}"))
    }
}

impl fmt::Display for ToolingRequirement {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_and_json_use_the_same_stable_requirement_name() {
        for requirement in ToolingRequirement::ALL {
            assert_eq!(
                serde_json::to_string(&requirement).unwrap(),
                format!("\"{requirement}\"")
            );
            assert_eq!(requirement.to_string().parse(), Ok(requirement));
        }
    }
}
