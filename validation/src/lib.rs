pub mod air;
pub mod corpus_run;
pub mod corpus_shards;
pub mod corpus_source;
pub mod corpus_triage;
pub mod hash;
pub mod jsonl;
pub mod loop_budget;
pub mod spirv_delta;
mod texture;
pub mod translate_ledger;

#[cfg(target_os = "macos")]
pub mod oracle_macos;

// The vulkano byte-run executor. Built on Linux (native ICD) and macOS (same executor targeting the
// Apple GPU via MoltenVK, so byte-verification can run locally).
#[cfg(any(target_os = "linux", target_os = "macos"))]
pub mod runner_linux;

use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

static SCRATCH_COUNTER: AtomicU64 = AtomicU64::new(0);
pub const RENDER_TARGET_SEED_TAG: u32 = 197;
const RENDER_TARGET_SEED: Seed = Seed::Deterministic {
    tag: RENDER_TARGET_SEED_TAG,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    Kernel,
    Vertex,
    Fragment,
}

impl From<Stage> for metal2vulkan::passes::Stage {
    fn from(stage: Stage) -> Self {
        match stage {
            Stage::Kernel => metal2vulkan::passes::Stage::Kernel,
            Stage::Vertex => metal2vulkan::passes::Stage::Vertex,
            Stage::Fragment => metal2vulkan::passes::Stage::Fragment,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DataFormat {
    RawBytes,
    U32,
    I32,
    F32,
    Rgba8Unorm,
    Rgba8Uint,
    Rgba8Sint,
    R16Uint,
    Rg16Uint,
    Rgba16Uint,
    R32Uint,
    Rg32Uint,
    Rgba32Uint,
    R16Sint,
    Rg16Sint,
    Rgba16Sint,
    R32Sint,
    Rg32Sint,
    Rgba32Sint,
    R16Float,
    Rg16Float,
    Rgba16Float,
    Rg32Float,
    Rgba32Float,
    R32Float,
    Depth32Float,
    Depth24Stencil8,
}

impl DataFormat {
    pub const fn is_float_like(self) -> bool {
        matches!(
            self,
            DataFormat::F32
                | DataFormat::Rgba8Unorm
                | DataFormat::R16Float
                | DataFormat::Rg16Float
                | DataFormat::Rgba16Float
                | DataFormat::Rg32Float
                | DataFormat::Rgba32Float
                | DataFormat::R32Float
                | DataFormat::Depth32Float
        )
    }

    pub const fn bytes_per_pixel(self) -> Option<usize> {
        match self {
            DataFormat::Rgba8Unorm | DataFormat::Rgba8Uint | DataFormat::Rgba8Sint => Some(4),
            DataFormat::R16Uint | DataFormat::R16Sint => Some(2),
            DataFormat::Rg16Uint | DataFormat::Rg16Sint => Some(4),
            DataFormat::Rgba16Uint | DataFormat::Rgba16Sint => Some(8),
            DataFormat::R32Uint | DataFormat::R32Sint => Some(4),
            DataFormat::Rg32Uint | DataFormat::Rg32Sint => Some(8),
            DataFormat::Rgba32Uint | DataFormat::Rgba32Sint => Some(16),
            DataFormat::R16Float => Some(2),
            DataFormat::Rg16Float => Some(4),
            DataFormat::Rgba16Float => Some(8),
            DataFormat::Rg32Float => Some(8),
            DataFormat::Rgba32Float => Some(16),
            DataFormat::R32Float => Some(4),
            DataFormat::Depth32Float => Some(4),
            DataFormat::Depth24Stencil8 => Some(4),
            DataFormat::RawBytes | DataFormat::U32 | DataFormat::I32 | DataFormat::F32 => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Seed {
    Deterministic {
        tag: u32,
    },
    /// Like [`Seed::Deterministic`], but for float-typed image formats every element's top
    /// exponent bit is cleared so no texel is NaN/Inf. Synthetic float textures should carry
    /// valid finite data — arbitrary bytes produce ~1/64 NaN/Inf half lanes, whose backend-
    /// specific propagation is the whole reason many texture-sampling cases byte-diverge. This is
    /// opt-in (via a texture override's `finite` flag) so it never disturbs existing goldens.
    DeterministicFinite {
        tag: u32,
    },
    /// Like [`Seed::DeterministicFinite`], but every texel of an image is IDENTICAL (no x/y/z
    /// variation), and finite for float formats. This is the well-defined seed for a texture the
    /// shader reads through a resource-shape-ambiguous path — chiefly `texture2d_ms` read via
    /// `air.read_texture_2d_ms` while the harness can only bind a single-sample texture: the two
    /// backends disagree only on WHICH texel/sample the ambiguous read returns, so a uniform texture
    /// makes that read unambiguous and the (faithful) translation byte-matches. Opt-in via a texture
    /// override's `uniform` flag; buffers ignore it (no element type — behaves as Deterministic).
    DeterministicUniform {
        tag: u32,
    },
    ScalarOneForHarness {
        reason: &'static str,
    },
    ZeroForTest {
        reason: &'static str,
    },
    ExactBytes {
        bytes: &'static [u8],
        reason: &'static str,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BufferRole {
    Input,
    Output,
    InOut,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BufferInput {
    pub index: u32,
    pub len: usize,
    pub role: BufferRole,
    pub seed: Seed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TextureRole {
    Sampled,
    StorageRead,
    StorageWrite,
    StorageReadWrite,
    ColorTarget,
    InputAttachment,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Extent3d {
    pub width: u32,
    pub height: u32,
    pub depth: u32,
}

impl Extent3d {
    pub const fn new(width: u32, height: u32, depth: u32) -> Self {
        Self {
            width,
            height,
            depth,
        }
    }

    pub const fn texel_count(self) -> usize {
        self.width as usize * self.height as usize * self.depth as usize
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TextureInput {
    pub index: u32,
    pub format: DataFormat,
    pub extent: Extent3d,
    pub role: TextureRole,
    pub seed: Seed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Output {
    Buffer {
        index: u32,
        format: DataFormat,
        len: usize,
    },
    Texture {
        index: u32,
        format: DataFormat,
        extent: Extent3d,
    },
    RenderTarget {
        format: DataFormat,
        extent: Extent3d,
    },
}

impl Output {
    pub const fn format(self) -> DataFormat {
        match self {
            Output::Buffer { format, .. }
            | Output::Texture { format, .. }
            | Output::RenderTarget { format, .. } => format,
        }
    }

    pub const fn byte_len(self) -> usize {
        match self {
            Output::Buffer { len, .. } => len,
            Output::Texture { format, extent, .. } | Output::RenderTarget { format, extent } => {
                match format.bytes_per_pixel() {
                    Some(stride) => extent.texel_count() * stride,
                    None => 0,
                }
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Dispatch {
    pub threads_per_grid: [u32; 3],
    pub threads_per_threadgroup: [u32; 3],
}

impl Dispatch {
    pub const fn default_1d(threads: u32) -> Self {
        Self {
            threads_per_grid: [threads, 1, 1],
            threads_per_threadgroup: [64, 1, 1],
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlendMode {
    Replace,
    SourceOver,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Render {
    pub target: Extent3d,
    pub vertex_count: u32,
    pub blend: BlendMode,
}

impl Render {
    pub const fn fullscreen_triangle(width: u32, height: u32) -> Self {
        Self {
            target: Extent3d::new(width, height, 1),
            vertex_count: 3,
            blend: BlendMode::Replace,
        }
    }

    pub const fn fullscreen_triangle_source_over(width: u32, height: u32) -> Self {
        Self {
            target: Extent3d::new(width, height, 1),
            vertex_count: 3,
            blend: BlendMode::SourceOver,
        }
    }
}

/// Binds a texture that lives INSIDE an argument buffer (an `air.indirect_buffer` whose
/// `air.struct_type_info` embeds an `air.indirect_argument` → `air.texture`, read by the kernel via
/// an integer-coord `air.read_texture`). The translator materializes a standalone sampled image for
/// it at `TEXTURE_BINDING_BASE + texture_index` (see `meta::embedded_synthetic_texture_index`); this
/// record tells the harness which top-level `TextureInput` supplies the seeded pixels
/// (`texture_index`) and where the texture handle sits inside the owning argument buffer
/// (`buffer_index` + `field_offset` of element 0). The Apple oracle writes the seeded texture's
/// `gpuResourceID` into that handle slot (+ `useResource`); the Vulkan runner reads the texture
/// through the standalone descriptor and only needs the handle non-zero for the `is_null_texture`
/// guard. Both sides seed the SAME `TextureInput` pixels, so the read is byte-identical.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EmbeddedTextureBinding {
    /// Kernel parameter index of the owning `air.indirect_buffer` argument.
    pub buffer_index: u32,
    /// Byte offset of the texture handle within the argument-buffer struct's element 0.
    pub field_offset: u32,
    /// Index of the `TextureInput` (synthetic index K) that supplies the seeded pixels.
    pub texture_index: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Inputs {
    pub buffers: &'static [BufferInput],
    pub textures: &'static [TextureInput],
    pub output: Output,
    pub dispatch: Dispatch,
    pub render: Render,
    /// Argument-buffer-embedded textures (usually empty). See [`EmbeddedTextureBinding`].
    pub embedded_textures: &'static [EmbeddedTextureBinding],
}

impl Inputs {
    pub const fn new(
        buffers: &'static [BufferInput],
        textures: &'static [TextureInput],
        output: Output,
        dispatch: Dispatch,
        render: Render,
    ) -> Self {
        Self {
            buffers,
            textures,
            output,
            dispatch,
            render,
            embedded_textures: &[],
        }
    }

    /// Attach argument-buffer-embedded texture bindings (builder; keeps `new` callers unchanged).
    pub const fn with_embedded_textures(
        mut self,
        embedded_textures: &'static [EmbeddedTextureBinding],
    ) -> Self {
        self.embedded_textures = embedded_textures;
        self
    }

    pub const fn output_format(&self) -> DataFormat {
        self.output.format()
    }
}

pub fn seeded_buffer_bytes(input: &BufferInput) -> Vec<u8> {
    seeded_linear_bytes(input.len, input.index, input.seed)
}

pub fn seeded_texture_bytes(input: &TextureInput) -> Vec<u8> {
    seeded_image_bytes(input.format, input.extent, input.seed)
}

pub fn seeded_texture_bytes_for_extent(input: &TextureInput, extent: Extent3d) -> Vec<u8> {
    seeded_image_bytes(input.format, extent, input.seed)
}

pub fn seeded_unit_rgba32_float_texture_bytes(extent: Extent3d) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(extent.texel_count() * 16);
    for _ in 0..extent.texel_count() * 4 {
        bytes.extend_from_slice(&1.0f32.to_le_bytes());
    }
    bytes
}

pub fn seeded_render_target_bytes(format: DataFormat, extent: Extent3d) -> Vec<u8> {
    let seed = if float_element_size(format).is_some() {
        match RENDER_TARGET_SEED {
            Seed::Deterministic { tag } => Seed::DeterministicFinite { tag },
            other => other,
        }
    } else {
        RENDER_TARGET_SEED
    };
    seeded_image_bytes(format, extent, seed)
}

/// Element byte size of a float image format, or `None` if the format is not float-typed (so no
/// NaN/Inf sanitization applies — integer/unorm formats are finite by construction).
fn float_element_size(format: DataFormat) -> Option<usize> {
    match format {
        DataFormat::R16Float | DataFormat::Rg16Float | DataFormat::Rgba16Float => Some(2),
        DataFormat::Rg32Float
        | DataFormat::Rgba32Float
        | DataFormat::R32Float
        | DataFormat::Depth32Float
        | DataFormat::F32 => Some(4),
        // Rgba8Unorm is normalized on read (byte/255) — already finite.
        _ => None,
    }
}

/// Clear the most-significant exponent bit of every float element in `bytes`, guaranteeing each is
/// finite (exponent field can no longer be all-ones ⇒ no NaN/Inf) while keeping values bounded
/// (magnitude < 2) and still varied. `bytes` is little-endian, so the element's high byte is at
/// `elem_base + elem_size - 1`, and 0x40 there is the exponent MSB for both half and f32.
fn sanitize_float_finite(bytes: &mut [u8], elem_size: usize) {
    let mut hi = elem_size - 1;
    while hi < bytes.len() {
        bytes[hi] &= !0x40;
        hi += elem_size;
    }
}

fn seeded_image_bytes(format: DataFormat, extent: Extent3d, seed: Seed) -> Vec<u8> {
    let stride = format.bytes_per_pixel().unwrap_or(4);
    let len = extent.texel_count() * stride;
    match seed {
        Seed::ZeroForTest { .. } => vec![0; len],
        Seed::ScalarOneForHarness { .. } => seeded_scalar_one_bytes(len),
        Seed::ExactBytes { bytes, .. } => {
            assert_eq!(bytes.len(), len, "exact image seed length mismatch");
            bytes.to_vec()
        }
        Seed::DeterministicFinite { tag } => {
            let mut bytes = seeded_image_bytes(format, extent, Seed::Deterministic { tag });
            if let Some(elem_size) = float_element_size(format) {
                sanitize_float_finite(&mut bytes, elem_size);
            }
            bytes
        }
        Seed::DeterministicUniform { tag } => {
            // One texel's finite bytes, replicated across the whole image so every texel is equal.
            let mut cell = seeded_image_bytes(
                format,
                Extent3d {
                    width: 1,
                    height: 1,
                    depth: 1,
                },
                Seed::DeterministicFinite { tag },
            );
            cell.truncate(stride);
            (0..len).map(|i| cell[i % stride.max(1)]).collect()
        }
        Seed::Deterministic { tag } => {
            let mut bytes = vec![0; len];
            let width = extent.width.max(1);
            let height = extent.height.max(1);
            let depth = extent.depth.max(1);
            for z in 0..depth {
                for y in 0..height {
                    for x in 0..width {
                        let texel = ((z * height + y) * width + x) as usize;
                        let base = texel * stride;
                        let in_center_triangle =
                            width >= 4 && height >= 4 && y >= height / 4 && x >= y / 2;
                        for lane in 0..stride {
                            let gradient = x
                                .wrapping_mul(17)
                                .wrapping_add(y.wrapping_mul(29))
                                .wrapping_add(z.wrapping_mul(43))
                                .wrapping_add(lane as u32 * 61)
                                .wrapping_add(tag);
                            let value = if in_center_triangle {
                                0xffu8.wrapping_sub((gradient & 0xff) as u8)
                            } else {
                                (gradient & 0xff) as u8
                            };
                            bytes[base + lane] = value.max(1);
                        }
                    }
                }
            }
            bytes
        }
    }
}

fn seeded_linear_bytes(len: usize, index: u32, seed: Seed) -> Vec<u8> {
    match seed {
        Seed::ZeroForTest { .. } => vec![0; len],
        Seed::ScalarOneForHarness { .. } => seeded_scalar_one_bytes(len),
        Seed::ExactBytes { bytes, .. } => {
            assert_eq!(bytes.len(), len, "exact linear seed length mismatch");
            bytes.to_vec()
        }
        // A raw buffer carries no element type here, so finiteness/uniformity can't be applied
        // structurally; callers that need those for buffers use ExactBytes. Behave as Deterministic.
        Seed::DeterministicFinite { tag } | Seed::DeterministicUniform { tag } => {
            seeded_linear_bytes(len, index, Seed::Deterministic { tag })
        }
        Seed::Deterministic { tag } => (0..len)
            .map(|i| {
                let mixed = ((i as u64 + 1)
                    .wrapping_mul(2_654_435_761)
                    .wrapping_add((index as u64) << 32)
                    .wrapping_add(tag as u64))
                    >> ((i & 3) * 8);
                (mixed as u8).max(1)
            })
            .collect(),
    }
}

fn seeded_scalar_one_bytes(len: usize) -> Vec<u8> {
    (0..len).map(|i| if i % 4 == 0 { 1 } else { 0 }).collect()
}

pub(crate) fn scratch_dir_for(id: &str) -> PathBuf {
    let serial = SCRATCH_COUNTER.fetch_add(1, Ordering::Relaxed);
    let safe_id: String = id
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
                ch
            } else {
                '_'
            }
        })
        .collect();
    let dir = std::env::temp_dir().join(format!(
        "metal2vulkan-validation-{}-{}-{safe_id}",
        std::process::id(),
        serial
    ));
    fs::create_dir_all(&dir).unwrap_or_else(|e| panic!("create scratch {}: {e}", dir.display()));
    dir
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_buffer_seed_is_nonzero_and_stable() {
        let input = BufferInput {
            index: 2,
            len: 64,
            role: BufferRole::InOut,
            seed: Seed::Deterministic { tag: 7 },
        };
        let a = seeded_buffer_bytes(&input);
        let b = seeded_buffer_bytes(&input);
        assert_eq!(a, b);
        assert_eq!(a.len(), 64);
        assert!(a.iter().any(|byte| *byte != 0));
        assert!(a.iter().all(|byte| *byte != 0));
    }

    #[test]
    fn deterministic_render_target_seed_is_nonzero_and_stable() {
        let extent = Extent3d::new(2, 2, 1);
        let a = seeded_render_target_bytes(DataFormat::Rgba8Unorm, extent);
        let b = seeded_render_target_bytes(DataFormat::Rgba8Unorm, extent);
        assert_eq!(a, b);
        assert_eq!(a.len(), 16);
        assert!(a.iter().any(|byte| *byte != 0));
        assert!(a.iter().all(|byte| *byte != 0));
    }

    #[test]
    fn float_render_target_seed_is_finite() {
        let extent = Extent3d::new(8, 8, 1);
        for &(fmt, elem) in &[
            (DataFormat::Rgba16Float, 2usize),
            (DataFormat::Rgba32Float, 4usize),
            (DataFormat::Depth32Float, 4usize),
        ] {
            let bytes = seeded_render_target_bytes(fmt, extent);
            for chunk in bytes.chunks_exact(elem) {
                let finite = if elem == 2 {
                    let h = u16::from_le_bytes([chunk[0], chunk[1]]);
                    ((h >> 10) & 0x1f) != 0x1f
                } else {
                    let f = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                    f.is_finite()
                };
                assert!(
                    finite,
                    "{fmt:?} render target element not finite: {chunk:02x?}"
                );
            }
        }
    }

    #[test]
    fn exact_texture_seed_bytes_are_used_verbatim() {
        let input = TextureInput {
            index: 0,
            format: DataFormat::Rgba8Unorm,
            extent: Extent3d::new(1, 1, 1),
            role: TextureRole::StorageRead,
            seed: Seed::ExactBytes {
                bytes: &[0x12, 0x34, 0x56, 0x78],
                reason: "unit test",
            },
        };
        assert_eq!(seeded_texture_bytes(&input), vec![0x12, 0x34, 0x56, 0x78]);
    }

    #[test]
    fn unit_rgba32_float_texture_seed_is_finite() {
        assert_eq!(
            seeded_unit_rgba32_float_texture_bytes(Extent3d::new(1, 1, 1)),
            [
                0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x80, 0x3f, 0x00, 0x00,
                0x80, 0x3f,
            ]
        );
    }

    #[test]
    fn deterministic_finite_texture_seed_has_no_nan_or_inf() {
        // Every half lane of a Deterministic Rgba16Float texture can be NaN/Inf (exponent all-ones);
        // DeterministicFinite must clear that for every element while staying varied and bounded.
        let extent = Extent3d::new(8, 8, 1);
        for &(fmt, elem) in &[
            (DataFormat::Rgba16Float, 2usize),
            (DataFormat::Rgba32Float, 4usize),
        ] {
            let bytes = seeded_image_bytes(fmt, extent, Seed::DeterministicFinite { tag: 100 });
            let mut saw_nonzero = false;
            for chunk in bytes.chunks_exact(elem) {
                let (finite, nonzero) = if elem == 2 {
                    let h = u16::from_le_bytes([chunk[0], chunk[1]]);
                    (((h >> 10) & 0x1f) != 0x1f, h != 0)
                } else {
                    let f = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                    (f.is_finite(), f != 0.0)
                };
                assert!(finite, "{fmt:?} element not finite: {chunk:02x?}");
                saw_nonzero |= nonzero;
            }
            assert!(
                saw_nonzero,
                "{fmt:?} finite seed is entirely zero — not varied"
            );
        }
        // Non-float formats are untouched: DeterministicFinite == Deterministic bytes.
        assert_eq!(
            seeded_image_bytes(
                DataFormat::Rgba8Unorm,
                extent,
                Seed::DeterministicFinite { tag: 7 }
            ),
            seeded_image_bytes(
                DataFormat::Rgba8Unorm,
                extent,
                Seed::Deterministic { tag: 7 }
            ),
        );
    }

    #[test]
    fn deterministic_uniform_texture_seed_is_uniform_and_finite() {
        // Every texel identical (so a resource-shape-ambiguous read returns the same value on both
        // backends) and finite for float formats.
        let extent = Extent3d::new(8, 8, 1);
        for &(fmt, stride) in &[
            (DataFormat::Rgba16Float, 8usize),
            (DataFormat::Rgba32Float, 16usize),
            (DataFormat::Rgba8Unorm, 4usize),
        ] {
            let bytes = seeded_image_bytes(fmt, extent, Seed::DeterministicUniform { tag: 100 });
            let first = &bytes[..stride];
            assert!(
                first.iter().any(|&b| b != 0),
                "{fmt:?} uniform seed all-zero"
            );
            for texel in bytes.chunks_exact(stride) {
                assert_eq!(texel, first, "{fmt:?} texel differs — not uniform");
            }
            if let Some(elem) = float_element_size(fmt) {
                for chunk in bytes.chunks_exact(elem) {
                    let finite = if elem == 2 {
                        ((u16::from_le_bytes([chunk[0], chunk[1]]) >> 10) & 0x1f) != 0x1f
                    } else {
                        f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]).is_finite()
                    };
                    assert!(finite, "{fmt:?} uniform element not finite: {chunk:02x?}");
                }
            }
        }
    }

    #[test]
    fn scalar_one_seed_bounds_control_words() {
        let input = BufferInput {
            index: 0,
            len: 10,
            role: BufferRole::Input,
            seed: Seed::ScalarOneForHarness {
                reason: "unit test",
            },
        };
        assert_eq!(
            seeded_buffer_bytes(&input),
            vec![1, 0, 0, 0, 1, 0, 0, 0, 1, 0]
        );
    }
}
