# Shader reflection for consumers

When you translate a Metal AIR / sanitized LLVM IR module with metal2vulkan, the crate already
knows how the **stage interface** was mapped into Vulkan: descriptor set/binding numbers, Metal
resource indices, vertex attributes, varyings, render targets, and more. That knowledge is exposed
as [`ShaderReflection`](../src/reflect/mod.rs) so a host (engine, executor, cache) does **not** have
to re-walk the emitted SPIR-V or re-encode the ABI.

Reflection is a pure re-shaping of AIR metadata the translator already parsed. It is **byte-neutral**:
`translate_reflected` produces the same SPIR-V as `translate` for the same inputs.

## Getting reflection

### Library

```rust
use metal2vulkan::passes::Stage;
use metal2vulkan::reflect::ShaderReflection;
use std::path::Path;

fn convert(air_or_ll_path: &str) -> Result<(Vec<u8>, ShaderReflection), String> {
    let tmp = std::env::temp_dir().join("m2v-scratch");
    std::fs::create_dir_all(&tmp).ok();
    metal2vulkan::translate_reflected(air_or_ll_path, Stage::Kernel, Path::new(&tmp))
}
```

Other entry points:

| API | Use when |
|---|---|
| `translate_reflected` / `translate_reflected_with_options` | Path to `.air` or `.ll` + stage |
| `translate_sanitized_native_reflected` | You already have sanitized LLVM IR text |
| `ShaderReflection::from_{fragment,vertex,kernel}` | You already have `meta::{Frag,Vert,Kern}Meta` |

Optional transforms (e.g. `--simd-cluster32` / `TransformOptions`) go through `*_with_options`.

### CLI JSON dump

Build with the `serde` feature and pass `--emit-meta`:

```sh
cargo install metal2vulkan --features serde
metal2vulkan in.ll out.spv --stage kernel --emit-meta out.json
```

`--emit-meta` is not supported for `--stage passthrough` (no Metal interface metadata).

`out.json` is a pretty-printed `ShaderReflection` (schema version in `reflection_version`).

### Persist with `serde`

```toml
metal2vulkan = { version = "0.1", features = ["serde"] }
```

`ShaderReflection` and its nested types derive `Serialize`/`Deserialize` under that feature.
Bump-aware field: `reflection_version` (`REFLECTION_VERSION`, currently `2`). Invalidate any
on-disk cache when that constant changes.

## Descriptor ABI (binding map)

Every Metal-facing resource is decorated in **descriptor set 0** (`RESOURCE_DESCRIPTOR_SET`).
Bindings are a fixed base plus the Metal resource index `n`:

| Metal resource | Kind | SPIR-V binding |
|---|---|---|
| `[[buffer(n)]]` (device / constant) | `Buffer` | `BUFFER_BINDING_BASE + n` → **`n`** (`0..32`) |
| `[[buffer(n)]]` (threadgroup) | `ThreadgroupBuffer` | **no descriptor** (`descriptor: None`) |
| `[[texture(n)]]` | `Texture` / `StorageImage` / `TextureArray` | `TEXTURE_BINDING_BASE + n` → **`32 + n`** |
| `[[sampler(n)]]` | `Sampler` | `SAMPLER_BINDING_BASE + n` → **`64 + n`** |
| `[[color(n)]]` (framebuffer fetch) | `ColorInput` | `COLOR_INPUT_BINDING_BASE + n` → **`96 + n`** |
| Acceleration-structure shadow buffer | `AccelerationStructureShadow` | Metal buffer index `n` (set 0) |
| Texture embedded in argument buffer | `EmbeddedArgBufferTexture` | `32 + synthetic_index` |

Constants live in `metal2vulkan::reflect`:

```text
RESOURCE_DESCRIPTOR_SET = 0
BUFFER_BINDING_BASE     = 0
TEXTURE_BINDING_BASE    = 32
SAMPLER_BINDING_BASE    = 64
COLOR_INPUT_BINDING_BASE = 96
```

The stage-input / stage-output passes decorate the module with **exactly these** numbers. Use
reflection to allocate descriptor sets / write descriptor updates without disassembling SPIR-V.

AIR address spaces (on buffer bindings when present):

| Value | Meaning |
|---|---|
| `ADDRESS_SPACE_CONSTANT` (2) | Constant / `const device` — typically read-only |
| `ADDRESS_SPACE_THREADGROUP` (3) | Threadgroup — Workgroup variable, no descriptor |

## Reading `ShaderReflection`

Top-level fields:

| Field | Meaning |
|---|---|
| `reflection_version` | Schema version for cache invalidation |
| `stage` | `Vertex` / `Fragment` / `Kernel` |
| `entry_point` | **Original Metal entry name** (SPIR-V `OpEntryPoint` is always `"main"`) |
| `bindings` | Resources in entry-parameter order (synthesized arg-buffer textures last) |
| `vertex_attributes` | Vertex `[[attribute(n)]]` / stage-in locations |
| `varyings` | Fragment stage-in or vertex user varyings (location, type, semantic) |
| `render_targets` | Fragment color attachments (member index + location + type name) |
| `depth_members` / `stencil_members` | Fragment return members tagged depth/stencil |
| `local_size` | Kernel GLCompute local size `[x,y,z]` when known |
| `vertex_builtins` | Whether vertex uses `VertexIndex` / `InstanceIndex` / writes `Position` |
| `imageblock_layouts` | Kernel `[[imageblock]]` tiles (param index + AIR struct layout; no descriptor) |
| `datalayout` | Source LLVM `target datalayout` string when translate started from unsanitized IR |
| `function_constants` | `[[function_constant(N)]]` inventory (index / name / LLVM type) |

### Per-binding: `ResourceBinding`

| Field | Use for |
|---|---|
| `kind` | Choose descriptor type (UBO/SSBO/sampled image/storage image/sampler/input attachment) |
| `metal_index` | Guest Metal slot `n`, or synthetic index for embedded textures |
| `descriptor` | `{ set, binding }` or `None` if no descriptor (threadgroup / some locals) |
| `param_index` | SPIR-V `OpFunctionParameter` order, if any |
| `address_space` / `declared_size` / `type_layout` | Buffer layout and sizing |
| `type_name` | AIR type string when metadata carried it |
| `texture_shape` | Dim / arrayed / MS / component / writable / storage format (decoded) |
| `embedded_source` | For arg-buffer textures: owning buffer index + field byte offset |
| `access` | When known: `ReadOnly` / `ReadWrite` / `Sampled` / `Storage` |

**Gaps consumers should expect:**

- **Device buffer R/W:** precise read-vs-write often needs SPIR-V dataflow; many buffers have
  `access: None`. Prefer IR analysis or treat as read-write unless `address_space` is constant.
- **Function constants from meta-only builders:** `from_*` constructors leave
  `function_constants` empty; populate via the reflected translate paths (they scan sanitized IR).
- **Datalayout:** only filled when translating from unsanitized `.air`/`.ll` via
  `translate_reflected*` (sanitization strips the line; the reflected path captures it first).

## Typical consumer flow

1. Call `translate_reflected` (or CLI with `--emit-meta` + `serde`).
2. Cache `(spv_bytes, reflection)` keyed by input hash + `reflection_version` + translator version.
3. Create a Vulkan pipeline with entry point `"main"`.
4. For each `bindings` entry with `Some(descriptor)`:
   - Map `kind` → descriptor type.
   - Write `set` / `binding` from `descriptor`.
   - Use `metal_index` to pick the host resource that was bound as Metal slot `n`.
5. For `ThreadgroupBuffer` / `imageblock_layouts`, allocate Workgroup / tile storage from
   `declared_size` / `type_layout` (no descriptor write).
6. For vertex: bind attributes from `vertex_attributes` and respect `vertex_builtins`.
7. For fragment: attach color targets from `render_targets`; handle depth/stencil member lists.
8. For kernels: set local size from `local_size` when present; specialize function constants if
   the host supplies values (see `specialize_function_constants` / AIR FC ABI).

## Related APIs

| Module | Role |
|---|---|
| `metal2vulkan::reflect` | Public facade (`ShaderReflection`, ABI constants) |
| `metal2vulkan::meta` | Lower-level AIR metadata parsers (`FragMeta` / `VertMeta` / `KernMeta`) |
| `metal2vulkan::passes::Stage` | Stage enum for translate |
| `metal2vulkan::specialize_function_constants` | Specialize FC values on sanitized IR (when needed) |

Unit coverage for binding numbers lives in `src/reflect/tests.rs` (ABI contract: set 0, bases 0/32/64/96).

## What reflection is *not*

- Not a full SPIR-V reflector (no walk of every `OpDecorate` / CFG).
- Not a substitute for `spirv-val` or runtime pipeline creation.
- Not populated for passthrough vertex generation (`translate_passthrough`).
