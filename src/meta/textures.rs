use crate::passes::ImageComp;
use spirv::{Dim, ImageFormat};

/// Descriptor capacity used for AIR texture-handle arrays. Fixed arrays occupy their prefix;
/// runtime `array_ref` cases author the logical prefix they may access.
pub const TEXTURE_HANDLE_ARRAY_DESCRIPTOR_COUNT: u32 = 128;

/// Texture dimensionality decoded from a Metal texture type name, independent of the SPIR-V emit
/// enum (`spirv::Dim`) so it can appear in the serializable reflection ABI.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum TextureDimension {
    D1,
    D2,
    D3,
    Cube,
    Buffer,
}

impl TextureDimension {
    /// The SPIR-V `Dim` the emitter decorates the `OpTypeImage` with.
    pub fn to_spirv_dim(self) -> Dim {
        match self {
            TextureDimension::D1 => Dim::Dim1D,
            TextureDimension::D2 => Dim::Dim2D,
            TextureDimension::D3 => Dim::Dim3D,
            TextureDimension::Cube => Dim::DimCube,
            TextureDimension::Buffer => Dim::DimBuffer,
        }
    }

    /// The neutral dimension for an emit-side `Dim` (the five Metal texture dims; any other maps to
    /// `D2`, the Metal default).
    pub fn from_spirv_dim(dim: Dim) -> Self {
        match dim {
            Dim::Dim1D => TextureDimension::D1,
            Dim::Dim3D => TextureDimension::D3,
            Dim::DimCube => TextureDimension::Cube,
            Dim::DimBuffer => TextureDimension::Buffer,
            _ => TextureDimension::D2,
        }
    }
}

/// Sampled component class decoded from a texture type name's scalar (`float`/`int`/`uint` family),
/// independent of the emit enum (`ImageComp`) so it can appear in the reflection ABI.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum TextureComponent {
    Float,
    Sint,
    Uint,
}

impl TextureComponent {
    /// The emitter's `ImageComp` for this component class.
    pub fn to_image_comp(self) -> ImageComp {
        match self {
            TextureComponent::Float => ImageComp::Float,
            TextureComponent::Sint => ImageComp::Sint,
            TextureComponent::Uint => ImageComp::Uint,
        }
    }

    /// The neutral component class for an emit-side `ImageComp`.
    pub fn from_image_comp(comp: ImageComp) -> Self {
        match comp {
            ImageComp::Float => TextureComponent::Float,
            ImageComp::Sint => TextureComponent::Sint,
            ImageComp::Uint => TextureComponent::Uint,
        }
    }
}

/// The SPIR-V `OpTypeImage` format a write-capable Metal texture lowers to, decoded from its type
/// name's scalar. Neutral of the emit enum (`spirv::ImageFormat`) so it can appear in the reflection
/// ABI; the emitter maps it back via [`TextureFormat::to_spirv_format`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum TextureFormat {
    R8,
    Rgba8,
    R16f,
    R16ui,
    Rg16f,
    R32f,
    R32i,
    R32ui,
    Rgba32i,
    Rgba32ui,
    Rgba32f,
    Rgba16f,
    Rgba8ui,
    Rgba16ui,
    Rgba8i,
}

impl TextureFormat {
    /// The SPIR-V `ImageFormat` the emitter decorates the storage image with.
    pub fn to_spirv_format(self) -> ImageFormat {
        match self {
            TextureFormat::R8 => ImageFormat::R8,
            TextureFormat::Rgba8 => ImageFormat::Rgba8,
            TextureFormat::R16f => ImageFormat::R16f,
            TextureFormat::R16ui => ImageFormat::R16ui,
            TextureFormat::Rg16f => ImageFormat::Rg16f,
            TextureFormat::R32f => ImageFormat::R32f,
            TextureFormat::R32i => ImageFormat::R32i,
            TextureFormat::R32ui => ImageFormat::R32ui,
            TextureFormat::Rgba32i => ImageFormat::Rgba32i,
            TextureFormat::Rgba32ui => ImageFormat::Rgba32ui,
            TextureFormat::Rgba32f => ImageFormat::Rgba32f,
            TextureFormat::Rgba16f => ImageFormat::Rgba16f,
            TextureFormat::Rgba8ui => ImageFormat::Rgba8ui,
            TextureFormat::Rgba16ui => ImageFormat::Rgba16ui,
            TextureFormat::Rgba8i => ImageFormat::Rgba8i,
        }
    }
}

/// The full shape a Metal texture type name (`texture2d_array<half, sample>`, `texture_buffer<uint,
/// read>`, `array_ref<texture2d<float, sample>>`, …) implies. This is THE one decoder of the texture
/// type-name grammar — the interface pass (emit-time image types + storage classification), the
/// embedded-argument-buffer scan, and the public reflection facade all derive from it, so the
/// grammar lives in exactly one place. See [`texture_shape_from_name`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TextureShape {
    pub dimension: TextureDimension,
    pub arrayed: bool,
    pub multisampled: bool,
    pub component: TextureComponent,
    /// The access qualifier declares `write`/`read_write` — a storage image (vs `sample`/`read`).
    pub writable: bool,
    /// The argument is a runtime-indexed descriptor array of texture handles.
    pub array_ref: bool,
    /// Fixed texture-handle array length (`array<texture..., N>`), or `None` for a runtime
    /// `array_ref` and for a single texture.
    pub array_length: Option<u32>,
    /// For a `writable` texture, the storage-image texel format the emitter decorates the
    /// `OpTypeImage` with; `None` for a sampled texture.
    pub storage_format: Option<TextureFormat>,
}

/// Decode a Metal texture argument/type name into its [`TextureShape`]. The dimensionality/arrayed
/// classification is substring-order-sensitive (`1d_array` before `1d`, `cube_array` before `cube`)
/// and matches what the interface pass uses to construct the emitted image type, including its
/// multisample operand.
pub fn texture_shape_from_name(name: &str) -> TextureShape {
    let (writable, array_ref) = texture_access_from_name(name);
    let array_length = fixed_texture_array_length(name);
    let shape_name = if array_ref {
        name.find("texture")
            .or_else(|| name.find("depth"))
            .and_then(|start| name.get(start..))
            .unwrap_or(name)
    } else {
        name
    };
    let head = shape_name
        .split_once('<')
        .map(|(h, _)| h)
        .unwrap_or(shape_name);
    let (dimension, arrayed) = if head.contains("texture_buffer") {
        (TextureDimension::Buffer, false)
    } else if head.contains("1d_array") {
        (TextureDimension::D1, true)
    } else if head.contains("1d") {
        (TextureDimension::D1, false)
    } else if head.contains("3d") {
        (TextureDimension::D3, false)
    } else if head.contains("cube_array") {
        (TextureDimension::Cube, true)
    } else if head.contains("cube") {
        (TextureDimension::Cube, false)
    } else if head.contains("2d_array") {
        (TextureDimension::D2, true)
    } else {
        (TextureDimension::D2, false)
    };
    let multisampled = head.contains("_ms");
    let component = texture_component_from_name(shape_name);
    let storage_format = if writable {
        Some(storage_format_from_name(name, component))
    } else {
        None
    };
    TextureShape {
        dimension,
        arrayed,
        multisampled,
        component,
        writable,
        array_ref,
        array_length,
        storage_format,
    }
}

fn fixed_texture_array_length(name: &str) -> Option<u32> {
    let compact = name
        .chars()
        .filter(|ch| !ch.is_whitespace())
        .collect::<String>();
    if !compact.starts_with("array<texture") && !compact.starts_with("array<depth") {
        return None;
    }
    let end = compact.rfind('>')?;
    let before_end = &compact[..end];
    before_end.rsplit_once(',')?.1.parse().ok()
}

/// The storage-image texel format a write-capable texture lowers to, from its scalar precision. A
/// `half`/`ushort` scalar narrows the format; the default is the full-width form.
fn storage_format_from_name(name: &str, component: TextureComponent) -> TextureFormat {
    match component {
        TextureComponent::Float => {
            if name.contains("<half") {
                TextureFormat::Rgba16f
            } else if name.contains("<float") {
                TextureFormat::R32f
            } else {
                TextureFormat::Rgba32f
            }
        }
        TextureComponent::Uint => {
            if name.contains("<ushort") {
                TextureFormat::Rgba16ui
            } else {
                TextureFormat::Rgba8ui
            }
        }
        TextureComponent::Sint => TextureFormat::Rgba8i,
    }
}

fn texture_component_from_name(name: &str) -> TextureComponent {
    let Some((_, rest)) = name.split_once('<') else {
        return TextureComponent::Float;
    };
    let scalar = rest
        .split(|c: char| c == ',' || c == '>' || c.is_whitespace())
        .find(|part| !part.is_empty())
        .unwrap_or("");
    // A nested `array<texture2d<uint, ...` puts the real scalar after the inner `<`.
    let scalar = scalar.rsplit('<').next().unwrap_or(scalar);
    match scalar {
        "uint" | "ushort" | "uchar" => TextureComponent::Uint,
        "int" | "short" | "char" => TextureComponent::Sint,
        _ => TextureComponent::Float,
    }
}

/// `(writable, array_ref)` decoded from the access qualifier — the second template field
/// (`texture2d<float, write>`) and texture-handle array wrappers.
fn texture_access_from_name(name: &str) -> (bool, bool) {
    let compact = name
        .chars()
        .filter(|ch| !ch.is_ascii_whitespace())
        .collect::<String>();
    let array_ref = compact.contains("array_ref<texture")
        || compact.contains("array<texture")
        || compact.contains("array_ref<depth")
        || compact.contains("array<depth");
    let Some((_, rest)) = name.split_once('<') else {
        return (false, array_ref);
    };
    let Some(inner) = rest.split('>').next() else {
        return (false, array_ref);
    };
    let mut fields = inner.split(',').map(str::trim);
    let _scalar = fields.next();
    let writable = matches!(fields.next(), Some("write") | Some("read_write"));
    (writable, array_ref)
}
