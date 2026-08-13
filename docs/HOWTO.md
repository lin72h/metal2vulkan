# How to translate and integrate a shader

This guide takes one Metal AIR or sanitized LLVM-IR module from input bytes to a validated Vulkan
shader and the host state needed to use it. For the complete field-by-field reflection contract, see
[Shader reflection for consumers](REFLECTION.md).

## 1. Install the tools

Product translation uses two external executables:

- `llvm-dis` converts AIR bitcode to LLVM IR. It is not needed when the input is already textual
  `.ll`.
- `spirv-val` validates every candidate under the Vulkan 1.3 environment. A retry candidate is
  adopted only after it validates.

Put both tools on `PATH`, or set an absolute per-tool override:

```sh
export METAL2VULKAN_LLVM_DIS=/path/to/llvm-dis
export METAL2VULKAN_SPIRV_VAL=/path/to/spirv-val
```

Install the CLI with reflection JSON enabled:

```sh
cargo install metal2vulkan --features serde
```

## 2. Translate from the command line

The default `--stage auto` reads `!air.vertex`, `!air.fragment`, or `!air.kernel` metadata from the
module:

```sh
metal2vulkan input.air output.spv --emit-meta reflection.json
```

Successful output contains `PASS`; the process exits `0` after writing validated SPIR-V and schema-
versioned reflection. Use an explicit stage only when appropriate:

```sh
metal2vulkan input.ll output.spv --stage vertex
metal2vulkan input.ll output.spv --stage fragment
metal2vulkan input.ll output.spv --stage kernel
metal2vulkan input.ll passthrough.spv --stage passthrough
```

Passthrough generation has no Metal interface metadata, so it cannot be combined with
`--emit-meta`. The accepted alias `--stage compute` is equivalent to `kernel`.

Two options affect pipeline-dependent lowering:

| Option | Supply it when |
|---|---|
| `--raster-samples 1|2|4|8|16|32|64` | Fragment AIR calls `air.get_num_samples.i32`; use the exact graphics-pipeline sample count |
| `--simd-cluster32` | A caller explicitly needs Metal's 32-lane simdgroup reduction partition on a wider Vulkan subgroup |

The translator automatically preserves the 32-lane contract for recognized `air.simd_*` modules.
Do not use either option as a workaround for an unrelated translation failure.

## 3. Translate from Rust

Use `translate_reflected` for an `.air`/`.ll` path and
`translate_sanitized_native_reflected` when sanitized LLVM IR is already in memory. All translation
entry points require caller-owned scratch space. Give concurrent calls different directories and
remove each directory on success or failure.

The repository includes a complete, compiled example that does this cleanup and writes both output
files:

```sh
cargo run --features serde --example translate_reflected -- \
  input.air output.spv reflection.json auto
```

The core call is:

```rust
use metal2vulkan::passes::{Stage, TransformOptions};
use std::path::Path;

fn translate_ir(
    sanitized_ll: &str,
    scratch: &Path,
) -> Result<(Vec<u8>, metal2vulkan::reflect::ShaderReflection), String> {
    metal2vulkan::translate_sanitized_native_reflected(
        sanitized_ll,
        Stage::Kernel,
        scratch,
        TransformOptions::default(),
    )
}
```

For path input, call `detect_stage(path, scratch)` and pass the resulting `Stage` to
`translate_reflected`. Translation already validates the final module; calling `spirv-val` again is
useful only as an independent deployment check or when bytes have changed after translation.

## 4. Create the descriptor-set layout

All reflected descriptors use set `0`, but descriptors come from three top-level places. A complete
consumer must inspect all three:

1. `bindings[*].descriptor`
2. `implicit_imageblock_attachments[*].binding`
3. `fragment_imageblock.members[*].binding` when non-null

The latter two are single storage-image descriptors in set `0`. For an entry in `bindings`, use
`descriptor.set`, `descriptor.binding`, and `descriptor.count` exactly as reported:

| `ResourceBinding.kind` | Vulkan descriptor type |
|---|---|
| `Buffer`, `KernelStageInput`, `AccelerationStructureShadow`, `PrimitiveAccelerationStructure`, `BufferAddressTable` | Storage buffer |
| `Texture`, `EmbeddedArgBufferTexture` | Sampled image |
| `TextureArray` with `access: Sampled` | Sampled-image array |
| `TextureArray` with `access: Storage` | Storage-image array |
| `StorageImage` | Storage image |
| `Sampler`, `StaticSampler` | Sampler |
| `ColorInput` | Input attachment |

`ThreadgroupBuffer`, `EmbeddedArgBufferBuffer`, visible/intersection function tables, and any other
entry with `descriptor: null` do not consume a Vulkan descriptor. Do not derive binding numbers
from list positions or Metal indices; synthesized resources are deliberately assigned by the
translator and reflection is authoritative.

Static samplers are ordinary Vulkan sampler descriptors whose creation state comes from
`static_sampler`. Embedded argument-buffer textures use `embedded_source` to identify their owner,
field offset, and Metal argument-encoder index. Embedded buffers use matching entries in
`argument_buffer_fields`; write the Vulkan device address into the reflected owner field rather than
allocating a separate descriptor.

## 5. Stage only the buffer bytes the shader can reach

Successful reflected translation attaches `footprint` to descriptor-backed `Buffer`,
`KernelStageInput`, and `AccelerationStructureShadow` resources. It describes accesses in the final
SPIR-V module selected by the retry cascade.

Apply this decision in order:

1. If `footprint` is null, there is no final-module proof. Retain the complete caller-provided
   window.
2. If `has_unbounded_access` is true, retain the complete window. Static or strided entries remain
   diagnostic only.
3. Otherwise, stage the union of `static_ranges` and every draw/dispatch-bounded
   `strided_accesses` range. An empty union means the final module does not access the binding.

Each static item `{ offset, size }` is the half-open interval `[offset, offset + size)`. Each strided
item describes accesses of `access_size` bytes at:

```text
base_offset + sum(actual_index_value(source) * stride)
```

Use the exact values generated by the current command, not counts alone. In particular,
`VertexIndex`, `InstanceIndex`, and `WorkgroupId` may include nonzero base values. Local invocation
IDs range over the pipeline's local size; global invocation IDs range over the dispatched grid. A
consumer can enumerate and coalesce the exact accesses, or conservatively copy the interval from
the minimum reachable address through the maximum reachable address plus `access_size`.

All offset, multiply, and end calculations must use checked arithmetic. Overflow means the
consumer cannot prove a bound and must retain the complete window. Clip computed ranges to neither
`declared_size` nor `extent`: `declared_size` may be only one pointee's size, while an unbounded AIR
pointer can legally index beyond it.

`Object { bytes }` is an independent AIR-metadata guarantee. If a supposedly bounded object's final
footprint extends beyond `bytes`, do not hide the inconsistency by clipping it: retain the complete
caller window and report the translator/reflection mismatch.

`access` answers read/write/unused classification when it can be proven. It does not replace the
footprint soundness gate. Treat `access: null` conservatively as read-write.

## 6. Configure the stage interface

- Create the Vulkan pipeline with entry point `"main"`; `entry_point` contains the original Metal
  function name for identity and diagnostics.
- Vertex stages use `vertex_attributes`, `varyings`, and `vertex_builtins`.
- A reflected `TessellationEvaluation` stage also uses `tessellation`; its control-point locations
  are arrays and its other listed locations carry patch data.
- Fragment stages use `varyings`, `render_targets`, `depth_members`, `stencil_members`, and
  `depth_qualifier`.
- Kernel stages use `local_size`. `imageblock_layouts` and threadgroup bindings describe Workgroup
  storage rather than descriptors.
- Use `function_constants` to discover exact indices and Metal ABI type encodings. Specialize IR
  with `specialize_function_constant_bytes` when exact scalar or vector payloads are required.

## 7. Cache outputs safely

Cache SPIR-V and reflection as one atomic result. A practical cache key includes:

- the exact input bytes;
- stage and every `TransformOptions` value;
- the metal2vulkan crate/binary version; and
- `reflection_version`.

Invalidate both artifacts together when any key changes. Reflection is byte-neutral, but it is tied
to the exact final module returned by the same call. Metadata-only `reflect_sanitized` deliberately
has no final-module footprint and is not a substitute for reflected translation in a staging path.

## 8. Handle failures

The Rust API returns `Err(String)`. The CLI prints `FALLBACK`, exits nonzero, and writes a self-
contained repro bundle under `$TMPDIR/metal2vulkan-repros` unless `METAL2VULKAN_REPRO_DIR` overrides
the base directory. Unsupported input must remain a fallback; do not continue with partial metadata
or a rejected retry candidate.

For local diagnosis:

```sh
METAL2VULKAN_RETRY_DEBUG=1 metal2vulkan input.air output.spv
METAL2VULKAN_WHY=1 metal2vulkan input.air output.spv
```

The first command shows retry-tier attempts. The second reports structured-CFG admission decisions.
These diagnostics do not authorize a different product translation path.

## 9. Verify the integration

At minimum:

```sh
spirv-val --target-env vulkan1.3 output.spv
cargo test -p metal2vulkan -- --test-threads=1
```

`spirv-val` checks structural validity, not Metal equivalence. For a semantic claim, follow the
[validation playbook](VALIDATION.md): use an owned synthetic regression, exact byte A/B where
appropriate, and an authored Metal/Vulkan case when output behavior changes.
