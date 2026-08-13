//! Canonical preparation of authored literal resources.
//!
//! Schema validation establishes shape and ownership. This module performs the one execution-time
//! interpretation shared by every backend: choose the bytes implied by a resource role, decode
//! base64, serialize the acceleration-structure shadow ABI, and preserve function-constant bits.

use crate::case::{
    AccelerationStructureKind, AttributeFormat, AuthoredCase, FragmentImageblockFormat,
    ResourceRole, ScalarType, TextureFormat, TextureType,
};
use base64::Engine as _;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LiteralBuffer {
    pub binding: u32,
    pub bytes: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LiteralArgumentBufferBuffer {
    pub buffer_binding: u32,
    pub field_offset: u32,
    pub bytes: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LiteralAccelerationStructure {
    pub binding: u32,
    pub kind: AccelerationStructureKind,
    pub bytes: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LiteralStageInput {
    pub location: u32,
    pub format: AttributeFormat,
    pub stride: u32,
    pub bytes: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LiteralTexture {
    pub binding: u32,
    pub role: ResourceRole,
    pub texture_type: TextureType,
    pub format: TextureFormat,
    pub dimensions: [u32; 3],
    pub sample_count: u32,
    pub bytes: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LiteralArgumentBufferTexture {
    pub buffer_binding: u32,
    pub field_offset: u32,
    pub role: ResourceRole,
    pub texture_type: TextureType,
    pub format: TextureFormat,
    pub dimensions: [u32; 3],
    pub sample_count: u32,
    pub bytes: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LiteralTextureArray {
    pub binding: u32,
    pub elements: Vec<LiteralTexture>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LiteralRenderTarget {
    pub index: u32,
    pub format: TextureFormat,
    pub dimensions: [u32; 2],
    pub bytes: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LiteralDepthStencil {
    pub dimensions: [u32; 2],
    pub depth: Option<Vec<u8>>,
    pub stencil: Option<Vec<u8>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LiteralFragmentImageblockMember {
    pub semantic: String,
    pub format: FragmentImageblockFormat,
    pub role: ResourceRole,
    pub bytes: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LiteralFragmentImageblock {
    pub dimensions: [u32; 2],
    pub members: Vec<LiteralFragmentImageblockMember>,
}

impl LiteralRenderTarget {
    pub fn bytes_per_row(&self) -> Result<usize, String> {
        (self.dimensions[0] as usize)
            .checked_mul(self.format.bytes_per_pixel())
            .ok_or_else(|| format!("render target {} row byte size overflows", self.index))
    }

    pub fn select(&self, origin: [u32; 2], dimensions: [u32; 2]) -> Result<Vec<u8>, String> {
        let pixel_size = self.format.bytes_per_pixel();
        let row_stride = self.bytes_per_row()?;
        let selected_row = dimensions[0] as usize * pixel_size;
        let mut output = Vec::with_capacity(selected_row * dimensions[1] as usize);
        for y in origin[1]..origin[1] + dimensions[1] {
            let start = y as usize * row_stride + origin[0] as usize * pixel_size;
            let end = start + selected_row;
            let row = self.bytes.get(start..end).ok_or_else(|| {
                format!("selected region exceeds render target {} bytes", self.index)
            })?;
            output.extend_from_slice(row);
        }
        Ok(output)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TextureLayout {
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub array_layers: u32,
    pub sample_count: u32,
}

impl LiteralTexture {
    pub fn layout(&self) -> Result<TextureLayout, String> {
        texture_layout(self.texture_type, self.dimensions, self.sample_count)
    }

    pub fn bytes_per_row(&self) -> Result<usize, String> {
        (self.dimensions[0] as usize)
            .checked_mul(self.sample_count as usize)
            .and_then(|width| width.checked_mul(self.format.bytes_per_pixel()))
            .ok_or_else(|| format!("texture {} row byte size overflows", self.binding))
    }

    pub fn bytes_per_image(&self) -> Result<usize, String> {
        self.bytes_per_row()?
            .checked_mul(self.dimensions[1] as usize)
            .ok_or_else(|| format!("texture {} image byte size overflows", self.binding))
    }

    pub fn select(&self, origin: [u32; 3], dimensions: [u32; 3]) -> Result<Vec<u8>, String> {
        let pixel_size = self.format.bytes_per_pixel();
        let row_stride = self.bytes_per_row()?;
        let image_stride = self.bytes_per_image()?;
        let texel_size = pixel_size * self.sample_count as usize;
        let selected_row = dimensions[0] as usize * texel_size;
        let mut output =
            Vec::with_capacity(selected_row * dimensions[1] as usize * dimensions[2] as usize);
        for z in origin[2]..origin[2] + dimensions[2] {
            for y in origin[1]..origin[1] + dimensions[1] {
                let start = z as usize * image_stride
                    + y as usize * row_stride
                    + origin[0] as usize * texel_size;
                let end = start + selected_row;
                let row = self.bytes.get(start..end).ok_or_else(|| {
                    format!("selected region exceeds texture {} bytes", self.binding)
                })?;
                output.extend_from_slice(row);
            }
        }
        Ok(output)
    }
}

impl LiteralArgumentBufferTexture {
    pub fn layout(&self) -> Result<TextureLayout, String> {
        texture_layout(self.texture_type, self.dimensions, self.sample_count)
    }

    pub fn bytes_per_row(&self) -> Result<usize, String> {
        (self.dimensions[0] as usize)
            .checked_mul(self.sample_count as usize)
            .and_then(|width| width.checked_mul(self.format.bytes_per_pixel()))
            .ok_or_else(|| format!("{} row byte size overflows", self.label()))
    }

    pub fn bytes_per_image(&self) -> Result<usize, String> {
        self.bytes_per_row()?
            .checked_mul(self.dimensions[1] as usize)
            .ok_or_else(|| format!("{} image byte size overflows", self.label()))
    }

    pub fn select(&self, origin: [u32; 3], dimensions: [u32; 3]) -> Result<Vec<u8>, String> {
        let pixel_size = self.format.bytes_per_pixel();
        let row_stride = self.bytes_per_row()?;
        let image_stride = self.bytes_per_image()?;
        let texel_size = pixel_size * self.sample_count as usize;
        let selected_row = dimensions[0] as usize * texel_size;
        let mut output =
            Vec::with_capacity(selected_row * dimensions[1] as usize * dimensions[2] as usize);
        for z in origin[2]..origin[2] + dimensions[2] {
            for y in origin[1]..origin[1] + dimensions[1] {
                let start = z as usize * image_stride
                    + y as usize * row_stride
                    + origin[0] as usize * texel_size;
                let end = start + selected_row;
                let row = self
                    .bytes
                    .get(start..end)
                    .ok_or_else(|| format!("selected region exceeds {} bytes", self.label()))?;
                output.extend_from_slice(row);
            }
        }
        Ok(output)
    }

    pub fn label(&self) -> String {
        format!(
            "argument-buffer texture {}+{}",
            self.buffer_binding, self.field_offset
        )
    }
}

pub fn texture_layout(
    texture_type: TextureType,
    dimensions: [u32; 3],
    sample_count: u32,
) -> Result<TextureLayout, String> {
    let [width, second, third] = dimensions;
    let invalid = || {
        format!(
            "invalid {texture_type:?} dimensions {dimensions:?} and sample count {sample_count}"
        )
    };
    let layout = match texture_type {
        TextureType::Buffer if second == 1 && third == 1 && sample_count == 1 => TextureLayout {
            width,
            height: 1,
            depth: 1,
            array_layers: 1,
            sample_count: 1,
        },
        TextureType::D1 if second == 1 && third == 1 && sample_count == 1 => TextureLayout {
            width,
            height: 1,
            depth: 1,
            array_layers: 1,
            sample_count: 1,
        },
        TextureType::D1Array if third == 1 && sample_count == 1 => TextureLayout {
            width,
            height: 1,
            depth: 1,
            array_layers: second,
            sample_count: 1,
        },
        TextureType::D2 if third == 1 && sample_count == 1 => TextureLayout {
            width,
            height: second,
            depth: 1,
            array_layers: 1,
            sample_count: 1,
        },
        TextureType::D2Array if sample_count == 1 => TextureLayout {
            width,
            height: second,
            depth: 1,
            array_layers: third,
            sample_count: 1,
        },
        TextureType::D2Multisample if third == 1 && matches!(sample_count, 2 | 4 | 8) => {
            TextureLayout {
                width,
                height: second,
                depth: 1,
                array_layers: 1,
                sample_count,
            }
        }
        TextureType::D2MultisampleArray if matches!(sample_count, 2 | 4 | 8) => TextureLayout {
            width,
            height: second,
            depth: 1,
            array_layers: third,
            sample_count,
        },
        TextureType::D3 if sample_count == 1 => TextureLayout {
            width,
            height: second,
            depth: third,
            array_layers: 1,
            sample_count: 1,
        },
        TextureType::Cube if width == second && third == 6 && sample_count == 1 => TextureLayout {
            width,
            height: width,
            depth: 1,
            array_layers: 6,
            sample_count: 1,
        },
        TextureType::CubeArray
            if width == second && third.is_multiple_of(6) && sample_count == 1 =>
        {
            TextureLayout {
                width,
                height: width,
                depth: 1,
                array_layers: third,
                sample_count: 1,
            }
        }
        _ => return Err(invalid()),
    };
    if width == 0
        || layout.height == 0
        || layout.depth == 0
        || layout.array_layers == 0
        || layout.sample_count == 0
    {
        return Err(invalid());
    }
    Ok(layout)
}

pub(crate) fn select_tightly_packed_2d(
    bytes: &[u8],
    extent: [u32; 2],
    origin: [u32; 2],
    dimensions: [u32; 2],
    pixel_size: usize,
) -> Result<Vec<u8>, String> {
    let source_row = extent[0] as usize * pixel_size;
    let selected_row = dimensions[0] as usize * pixel_size;
    let mut output = Vec::with_capacity(selected_row * dimensions[1] as usize);
    for y in origin[1]..origin[1] + dimensions[1] {
        let start = y as usize * source_row + origin[0] as usize * pixel_size;
        let end = start + selected_row;
        output.extend_from_slice(
            bytes
                .get(start..end)
                .ok_or("selected attachment range exceeds bytes")?,
        );
    }
    Ok(output)
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LiteralFunctionConstant {
    pub index: u32,
    pub scalar_type: ScalarType,
    pub lanes: u32,
    pub bytes: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LiteralTessellation {
    pub factors: Vec<crate::case::TessellationFactors>,
    pub instance_count: u32,
    pub amplification_count: u32,
    pub control_points: Vec<LiteralStageInput>,
    pub patch_inputs: Vec<LiteralStageInput>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LiteralResources {
    pub buffers: Vec<LiteralBuffer>,
    pub argument_buffer_buffers: Vec<LiteralArgumentBufferBuffer>,
    pub vertex_inputs: Vec<LiteralStageInput>,
    pub kernel_stage_inputs: Vec<LiteralStageInput>,
    pub acceleration_structure_shadows: Vec<LiteralAccelerationStructure>,
    pub textures: Vec<LiteralTexture>,
    pub texture_arrays: Vec<LiteralTextureArray>,
    pub argument_buffer_textures: Vec<LiteralArgumentBufferTexture>,
    pub render_targets: Vec<LiteralRenderTarget>,
    pub depth_stencil: Option<LiteralDepthStencil>,
    pub fragment_imageblock: Option<LiteralFragmentImageblock>,
    pub function_constants: Vec<LiteralFunctionConstant>,
    pub tessellation: Option<LiteralTessellation>,
}

impl LiteralResources {
    pub fn prepare(case: &AuthoredCase) -> Result<Self, String> {
        let buffers = case
            .buffers
            .iter()
            .map(|resource| {
                let encoded = role_bytes(
                    resource.role,
                    resource.bytes_b64.as_deref(),
                    resource.initial_bytes_b64.as_deref(),
                )
                .ok_or_else(|| format!("buffer {} has no literal bytes", resource.binding))?;
                Ok(LiteralBuffer {
                    binding: resource.binding,
                    bytes: decode(encoded, &format!("buffer {}", resource.binding))?,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        let argument_buffer_buffers = case
            .argument_buffer_buffers
            .iter()
            .map(|resource| {
                let label = format!(
                    "argument-buffer buffer {}+{}",
                    resource.buffer_binding, resource.field_offset
                );
                let encoded = role_bytes(
                    resource.role,
                    resource.bytes_b64.as_deref(),
                    resource.initial_bytes_b64.as_deref(),
                )
                .ok_or_else(|| format!("{label} has no literal bytes"))?;
                Ok(LiteralArgumentBufferBuffer {
                    buffer_binding: resource.buffer_binding,
                    field_offset: resource.field_offset,
                    bytes: decode(encoded, &label)?,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        let acceleration_structure_shadows = case
            .acceleration_structures
            .iter()
            .map(acceleration_structure_shadow)
            .collect::<Result<Vec<_>, String>>()?;
        let kernel_stage_inputs = case
            .kernel_stage_inputs
            .iter()
            .map(|resource| {
                Ok(LiteralStageInput {
                    location: resource.location,
                    format: resource.format,
                    stride: resource.stride,
                    bytes: decode(
                        &resource.bytes_b64,
                        &format!("kernel stage input {}", resource.location),
                    )?,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        let vertex_inputs = case
            .vertex_inputs
            .iter()
            .map(|resource| {
                Ok(LiteralStageInput {
                    location: resource.location,
                    format: resource.format,
                    stride: resource.stride,
                    bytes: decode(
                        &resource.bytes_b64,
                        &format!("vertex input {}", resource.location),
                    )?,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        let textures = case
            .textures
            .iter()
            .map(|resource| {
                let encoded = role_bytes(
                    resource.role,
                    resource.bytes_b64.as_deref(),
                    resource.initial_bytes_b64.as_deref(),
                )
                .ok_or_else(|| format!("texture {} has no literal bytes", resource.binding))?;
                Ok(LiteralTexture {
                    binding: resource.binding,
                    role: resource.role,
                    texture_type: resource.texture_type,
                    format: resource.format,
                    dimensions: resource.dimensions,
                    sample_count: resource.sample_count,
                    bytes: decode(encoded, &format!("texture {}", resource.binding))?,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        let argument_buffer_textures = case
            .argument_buffer_textures
            .iter()
            .map(|resource| {
                let label = format!(
                    "argument-buffer texture {}+{}",
                    resource.buffer_binding, resource.field_offset
                );
                let encoded = role_bytes(
                    resource.role,
                    resource.bytes_b64.as_deref(),
                    resource.initial_bytes_b64.as_deref(),
                )
                .ok_or_else(|| format!("{label} has no literal bytes"))?;
                Ok(LiteralArgumentBufferTexture {
                    buffer_binding: resource.buffer_binding,
                    field_offset: resource.field_offset,
                    role: resource.role,
                    texture_type: resource.texture_type,
                    format: resource.format,
                    dimensions: resource.dimensions,
                    sample_count: resource.sample_count,
                    bytes: decode(encoded, &label)?,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        let texture_arrays = case
            .texture_arrays
            .iter()
            .map(|array| {
                let elements = array
                    .elements
                    .iter()
                    .enumerate()
                    .map(|(index, element)| {
                        let label = format!("texture-array {} element {index}", array.binding);
                        let encoded = role_bytes(
                            array.role,
                            element.bytes_b64.as_deref(),
                            element.initial_bytes_b64.as_deref(),
                        )
                        .ok_or_else(|| format!("{label} has no literal bytes"))?;
                        Ok(LiteralTexture {
                            binding: array.binding,
                            role: array.role,
                            texture_type: array.texture_type,
                            format: array.format,
                            dimensions: element.dimensions,
                            sample_count: array.sample_count,
                            bytes: decode(encoded, &label)?,
                        })
                    })
                    .collect::<Result<Vec<_>, String>>()?;
                Ok(LiteralTextureArray {
                    binding: array.binding,
                    elements,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        let function_constants = case
            .function_constants
            .iter()
            .map(|constant| {
                Ok(LiteralFunctionConstant {
                    index: constant.index,
                    scalar_type: constant.scalar_type,
                    lanes: constant.lanes,
                    bytes: decode(
                        &constant.bytes_b64,
                        &format!("function constant {}", constant.index),
                    )?,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        let render_targets = case
            .render_targets
            .iter()
            .map(|target| {
                Ok(LiteralRenderTarget {
                    index: target.index,
                    format: target.format,
                    dimensions: target.dimensions,
                    bytes: decode(
                        &target.initial_bytes_b64,
                        &format!("render target {}", target.index),
                    )?,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        let fragment_imageblock = case
            .fragment_imageblock
            .as_ref()
            .map(|imageblock| {
                let members = imageblock
                    .members
                    .iter()
                    .map(|member| {
                        let encoded = role_bytes(
                            member.role,
                            member.bytes_b64.as_deref(),
                            member.initial_bytes_b64.as_deref(),
                        )
                        .ok_or_else(|| {
                            format!(
                                "fragment imageblock member {} has no literal bytes",
                                member.semantic
                            )
                        })?;
                        Ok(LiteralFragmentImageblockMember {
                            semantic: member.semantic.clone(),
                            format: member.format,
                            role: member.role,
                            bytes: decode(
                                encoded,
                                &format!("fragment imageblock member {}", member.semantic),
                            )?,
                        })
                    })
                    .collect::<Result<Vec<_>, String>>()?;
                Ok::<_, String>(LiteralFragmentImageblock {
                    dimensions: imageblock.dimensions,
                    members,
                })
            })
            .transpose()?;
        let depth_stencil = case
            .depth_stencil
            .as_ref()
            .map(|attachment| {
                Ok::<_, String>(LiteralDepthStencil {
                    dimensions: attachment.dimensions,
                    depth: attachment
                        .initial_depth_b64
                        .as_deref()
                        .map(|bytes| decode(bytes, "depth attachment"))
                        .transpose()?,
                    stencil: attachment
                        .initial_stencil_b64
                        .as_deref()
                        .map(|bytes| decode(bytes, "stencil attachment"))
                        .transpose()?,
                })
            })
            .transpose()?;
        let tessellation = case
            .tessellation
            .as_ref()
            .map(|tessellation| {
                Ok::<_, String>(LiteralTessellation {
                    factors: tessellation.factors.clone(),
                    instance_count: tessellation.instance_count,
                    amplification_count: tessellation.amplification_count,
                    control_points: decode_stage_inputs(
                        "tessellation control point",
                        &tessellation.control_points,
                    )?,
                    patch_inputs: decode_stage_inputs(
                        "tessellation patch input",
                        &tessellation.patch_inputs,
                    )?,
                })
            })
            .transpose()?;
        Ok(Self {
            buffers,
            argument_buffer_buffers,
            vertex_inputs,
            kernel_stage_inputs,
            acceleration_structure_shadows,
            textures,
            texture_arrays,
            argument_buffer_textures,
            render_targets,
            depth_stencil,
            fragment_imageblock,
            function_constants,
            tessellation,
        })
    }

    pub fn function_constant_values(&self) -> Vec<(u32, Vec<u8>)> {
        self.function_constants
            .iter()
            .map(|constant| (constant.index, constant.bytes.clone()))
            .collect()
    }
}

fn decode_stage_inputs(
    label: &str,
    inputs: &[crate::case::AttributeInput],
) -> Result<Vec<LiteralStageInput>, String> {
    inputs
        .iter()
        .map(|input| {
            Ok(LiteralStageInput {
                location: input.location,
                format: input.format,
                stride: input.stride,
                bytes: decode(&input.bytes_b64, &format!("{label} {}", input.location))?,
            })
        })
        .collect()
}

fn acceleration_structure_shadow(
    resource: &crate::case::AccelerationStructureResource,
) -> Result<LiteralAccelerationStructure, String> {
    let bytes = match resource.kind {
        AccelerationStructureKind::Instance => {
            let instance_count = u32::try_from(resource.child_references.len()).map_err(|_| {
                format!(
                    "acceleration-structure binding {} has too many instances",
                    resource.binding
                )
            })?;
            let mut bytes = Vec::with_capacity(
                metal2vulkan::as_shadow::CHILD_REFERENCES_BYTE_OFFSET as usize
                    + resource.child_references.len()
                        * metal2vulkan::as_shadow::CHILD_REFERENCE_BYTE_STRIDE as usize,
            );
            bytes.extend_from_slice(&instance_count.to_le_bytes());
            bytes.extend_from_slice(&0u32.to_le_bytes());
            for reference in &resource.child_references {
                bytes.extend_from_slice(&reference.to_le_bytes());
            }
            bytes
        }
        AccelerationStructureKind::Primitive => {
            let triangles = decode(
                resource.primitive_triangles_b64.as_deref().ok_or_else(|| {
                    format!(
                        "primitive acceleration structure {} has no triangles",
                        resource.binding
                    )
                })?,
                &format!("primitive acceleration structure {}", resource.binding),
            )?;
            let triangle_count = u32::try_from(
                triangles.len() / metal2vulkan::as_shadow::PRIMITIVE_TRIANGLE_BYTE_STRIDE as usize,
            )
            .map_err(|_| {
                format!(
                    "primitive acceleration structure {} has too many triangles",
                    resource.binding
                )
            })?;
            let mut bytes = Vec::with_capacity(
                metal2vulkan::as_shadow::PRIMITIVE_TRIANGLES_BYTE_OFFSET as usize + triangles.len(),
            );
            bytes.extend_from_slice(&triangle_count.to_le_bytes());
            bytes.extend_from_slice(&0u32.to_le_bytes());
            bytes.extend_from_slice(&triangles);
            bytes
        }
    };
    Ok(LiteralAccelerationStructure {
        binding: resource.binding,
        kind: resource.kind,
        bytes,
    })
}

fn role_bytes<'a>(
    role: ResourceRole,
    input: Option<&'a str>,
    initial: Option<&'a str>,
) -> Option<&'a str> {
    match role {
        ResourceRole::Input => input,
        ResourceRole::Output | ResourceRole::InOut => initial,
    }
}

fn decode(encoded: &str, label: &str) -> Result<Vec<u8>, String> {
    base64::engine::general_purpose::STANDARD
        .decode(encoded)
        .map_err(|error| format!("decode {label}: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::case::AccelerationStructureResource;

    #[test]
    fn acceleration_structure_shadow_matches_product_abi() {
        let resource = AccelerationStructureResource {
            binding: 8,
            kind: AccelerationStructureKind::Instance,
            primitive_triangles_b64: None,
            child_references: vec![0x1122_3344_5566_7788, 0xaabb_ccdd_eeff_0011],
        };
        assert_eq!(
            acceleration_structure_shadow(&resource).unwrap().bytes,
            [
                2, 0, 0, 0, 0, 0, 0, 0, 0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11, 0x11, 0x00,
                0xff, 0xee, 0xdd, 0xcc, 0xbb, 0xaa,
            ]
        );
    }

    #[test]
    fn primitive_acceleration_structure_shadow_preserves_authored_triangles() {
        let vertices = [-1.0f32, -1.0, 0.0, 1.0, -1.0, 0.0, 0.0, 1.0, 0.0];
        let triangle_bytes = vertices
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        let resource = AccelerationStructureResource {
            binding: 5,
            kind: AccelerationStructureKind::Primitive,
            primitive_triangles_b64: Some(
                base64::engine::general_purpose::STANDARD.encode(&triangle_bytes),
            ),
            child_references: vec![],
        };
        let shadow = acceleration_structure_shadow(&resource).unwrap();
        assert_eq!(shadow.kind, AccelerationStructureKind::Primitive);
        assert_eq!(&shadow.bytes[..8], &[1, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(&shadow.bytes[8..], triangle_bytes);
    }

    #[test]
    fn texture_layout_distinguishes_depth_layers_faces_and_samples() {
        assert_eq!(
            texture_layout(TextureType::Buffer, [8, 1, 1], 1).unwrap(),
            TextureLayout {
                width: 8,
                height: 1,
                depth: 1,
                array_layers: 1,
                sample_count: 1,
            }
        );
        assert_eq!(
            texture_layout(TextureType::D3, [2, 3, 4], 1).unwrap(),
            TextureLayout {
                width: 2,
                height: 3,
                depth: 4,
                array_layers: 1,
                sample_count: 1,
            }
        );
        assert_eq!(
            texture_layout(TextureType::CubeArray, [2, 2, 12], 1).unwrap(),
            TextureLayout {
                width: 2,
                height: 2,
                depth: 1,
                array_layers: 12,
                sample_count: 1,
            }
        );
        assert_eq!(
            texture_layout(TextureType::D2Multisample, [2, 3, 1], 4).unwrap(),
            TextureLayout {
                width: 2,
                height: 3,
                depth: 1,
                array_layers: 1,
                sample_count: 4,
            }
        );
    }

    #[test]
    fn texture_region_selection_is_tightly_packed_across_slices() {
        let texture = LiteralTexture {
            binding: 7,
            role: ResourceRole::Output,
            texture_type: TextureType::D2Array,
            format: TextureFormat::R8Unorm,
            dimensions: [3, 2, 2],
            sample_count: 1,
            bytes: (0..12).collect(),
        };
        assert_eq!(
            texture.select([1, 0, 0], [2, 2, 2]).unwrap(),
            vec![1, 2, 4, 5, 7, 8, 10, 11]
        );
    }

    #[test]
    fn render_target_selection_is_tightly_packed() {
        let target = LiteralRenderTarget {
            index: 2,
            format: TextureFormat::R8Unorm,
            dimensions: [3, 2],
            bytes: (0..6).collect(),
        };
        assert_eq!(target.select([1, 0], [2, 2]).unwrap(), vec![1, 2, 4, 5]);
    }

    #[test]
    fn depth_stencil_selection_is_tightly_packed() {
        let bytes = (0u8..16).collect::<Vec<_>>();
        assert_eq!(
            select_tightly_packed_2d(&bytes, [4, 4], [1, 1], [2, 2], 1).unwrap(),
            [5, 6, 9, 10]
        );
    }
}
