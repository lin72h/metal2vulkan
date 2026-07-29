# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added

- Public `specialize_function_constants_zero` helper for baking discovered Metal function
  constants to their zero/default values, including branch pruning and removal of now-dead entry
  interface globals.
- Reflection schema v3: AIR `constexpr sampler` globals are reported as `StaticSampler` bindings
  with decoded sampler state. Texture bindings also expose declared access through
  `ResourceBinding::access` where the AIR metadata carries it.
- Additional stage-interface support for fragment `[[point_coord]]`, `[[primitive_id]]`, and
  `[[sample_id]]`, flat varyings, framebuffer-fetch color inputs, vertex builtins, and fragment
  outputs with nonzero render-target locations.
- Broader native translation support for texture arrays, storage-image arrays, texture
  gather/sample/read/write variants, half/integer render-target formats, scalar 64-bit integer
  arithmetic, and Workgroup memory patterns used by shared-memory reductions.

### Changed

- Floating-point lowering is closer to AIR for the covered cases, including f32-to-f16 clamping,
  bf16 narrowing and NaN handling, fast `sin`/`cos`, `pow` zero edges, and exact `mix` endpoints.
- Buffer, pointer, control-flow, and access-chain lowering handles more structural cases, reducing
  fallbacks and invalid SPIR-V for shaders that use dynamic indices, pointer selects, aggregate
  copies, raw subword loads/stores, and local pointer tables.
- Structured control-flow retries are bounded more tightly, improving behavior on large shaders and
  avoiding unnecessary fallback to slower emission paths.
- Reflection documentation now describes the schema v3 descriptor contract.

### Fixed

- Unsupported Metal visible function references now fail fast with an explicit fallback instead of
  being treated as ordinary functions.
- Multiple `OpReturnValue` sites are rewritten consistently to stage outputs, and undefined
  fragment output stores are skipped rather than materialized.
- Kernel local-size options are validated as nonzero and are propagated into AIR local-size queries
  and imageblock lowering.
- SPIR-V generation avoids several invalid Logical-addressing forms by normalizing pointer phis,
  pointer selects, reinterpret loads/stores, access-chain index widths, and cross-binding pointer
  merges before final validation.
- Avoided a large-shader performance and size regression by measuring structured-CFG synthetic
  growth as added blocks rather than total module blocks, keeping large already-structured shaders
  on the primary emit path instead of falling back to the relooper tier.
- Removed inline denorm flush-to-zero emulation and avoided emitting `DenormFlushToZero`
  capability/execution modes on Vulkan devices that do not advertise float-controls support,
  restoring expected output size and compile time for affected shaders.
- Corrected imageblock explicit-slice stores so out-of-bounds oversized writes are discarded while
  zero-area writes still materialize transparent zero.
- Corrected integer `air.calculate_unclamped_lod_texture_2d` lowering so `OpImageQueryLod` uses a
  translator-owned default nearest sampler when the source AIR does not provide one.
- Fixed selected static sampler operands for AIR depth sampling and compare-depth lowering so
  `OpSampledImage` receives a valid SPIR-V sampler value even when AIR selects between sampler
  state globals.
- Fixed private indexed-container GEP lowering so dynamic accesses through palette-array pointer
  phis use ordinary access chains instead of invalid `OpPtrAccessChain` pointer-stride forms.
- Fixed function-constant-gated fragment outputs so the default-zero AIR predicate model does not
  advertise mutually exclusive render-target formats at the same Vulkan location.
- Fixed fragment `[[sample_id]]` lowering to use `BuiltIn SampleId` with the required
  `SampleRateShading` capability, and lowered AIR `texture2d_ms` reads to MS `OpTypeImage` fetches
  with a `Sample` operand instead of treating the sample id as a mip LOD.

## v0.1.0

### Added

- First public release of the `metal2vulkan` crate and CLI: native Metal AIR / sanitized
  LLVM IR → Vulkan SPIR-V (no LLVM `llc` on the product path).
- Library entry points for stage-aware translate, optional reflection metadata, and
  function-constant specialization.
- CLI: `metal2vulkan <in.air|.ll> <out.spv> --stage …`, optional `--emit-meta` JSON with the
  `serde` feature, `PASS` / `FALLBACK` reporting, and FALLBACK repro bundles under
  `$TMPDIR/metal2vulkan-repros` (override with `METAL2VULKAN_REPRO_DIR`).
- Optional `serde` feature for serializable `ShaderReflection` / metadata dumps.
- Consumer docs: architecture overview and reflection binding layout (`docs/`).

### Notes

- The crate is **alpha** (`0.x`): public API, CLI flags, and SPIR-V output may still change.
- This package does not ship third-party captured shaders; coverage is synthetic fixtures and
  unit tests only.
