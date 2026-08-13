# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added

- Public `specialize_function_constants_zero` helper for baking discovered Metal function
  constants to their zero/default values, including branch pruning and removal of now-dead entry
  interface globals.
- Public byte-exact function-constant specialization, reflection-only AIR inspection, authored
  linked-function/table specialization, and vertex-observer generation APIs.
- Reflection schema v21: consumer metadata now covers decoded static samplers, texture shape and
  access, argument-buffer resources, kernel stage inputs, tessellation, imageblocks, exact function-
  constant ABI types, buffer extent/access classification, and conservative final-module static and
  invocation-strided buffer footprints.
- A task-oriented translation and reflection integration guide plus a compiled serde reflection
  example.
- Additional stage-interface support for fragment `[[point_coord]]`, `[[primitive_id]]`, and
  `[[sample_id]]`, flat varyings, framebuffer-fetch color inputs, vertex builtins, and fragment
  outputs with nonzero render-target locations.
- Broader native translation support for texture arrays, storage-image arrays, texture
  gather/sample/read/write variants, half/integer render-target formats, scalar 64-bit integer
  arithmetic, and Workgroup memory patterns used by shared-memory reductions.
- Native lowering and reflection for linked functions, tessellation patch inputs, ray/intersection
  queries and result fields, argument-buffer resources, and implicit, custom, and direct-layout
  imageblocks.
- Native distributed `simdgroup_matrix` 16x16x16 multiply-accumulate lowering for the observed
  f32, f16, bf16, float8, and signed/unsigned i8 AIR element combinations, including dynamic
  transpose operands and 32-lane tile ownership.
- Authored validation contracts and executable Metal/Vulkan cases for tessellation, depth/stencil
  attachments, framebuffer fetch, multisample and buffer textures, narrow vertex attributes,
  vertex side effects, function constants, argument buffers, imageblocks, and ray intersections.
- A sharded validation workflow with dependency-exact observations, an incremental SQLite source
  index, focused hash/shard selection, explicit full reclassification, capability audits, native
  Metal and Vulkan/MoltenVK A/B execution, and optional OpenRouter-authored case proposals.

### Changed

- Native emitter wrapper APIs under `tools` now use `emit_vulkan_spirv*` names that match their
  implementation.
- The CLI accepts `--raster-samples` for AIR sample-count queries and derives a default `.vk.spv`
  output path when the output argument is omitted.
- Floating-point lowering is closer to AIR for the covered cases, including f32-to-f16 clamping,
  bf16 narrowing and NaN handling, fast `sin`/`cos`, `pow` zero edges, and exact `mix` endpoints.
- Buffer, pointer, control-flow, and access-chain lowering handles more structural cases, reducing
  fallbacks and invalid SPIR-V for shaders that use dynamic indices, pointer selects, aggregate
  copies, raw subword loads/stores, and local pointer tables.
- Structured control-flow retries are bounded more tightly, improving behavior on large shaders and
  avoiding unnecessary fallback to slower emission paths.
- Corpus translation and classification use bounded parallel workers, a 30-second per-item
  watchdog, a 512 MiB per-item memory ceiling, size-aware scheduling, and incremental index/cache
  reuse. Warm audits avoid reopening unchanged source shards; forced audits remain explicit.
- Validation capability checks, authored schema validation, backend execution gates, and cache
  identities now share one typed contract so a clean audit cannot hide a later executor rejection.
- Corpus capability audits now use the product's canonical AIR-call inventory and report every
  called intrinsic outside a recognized lowering or static-linkage family, including exact symbols
  and counts, instead of interpreting an unrelated clean authoring contract as complete support.
- Full AIR-intrinsic reclassification now updates every cached source in bounded keyset batches,
  independent of `--limit`, without reopening source shards; matrix capability recognition and
  lowering share one exact ABI parser.
- Reflection documentation now describes the complete schema v21 descriptor, argument-buffer,
  stage-interface, and conservative buffer-staging contracts.

### Removed

- Retired the monolithic validation ledgers and superseded mint/remint/run/why utilities in favor of
  sharded authored cases, dependency-exact observations, the source index, and unified corpus
  commands.
- Removed obsolete emitter naming and compatibility terminology; the product and tooling describe
  only the native AIR-to-SPIR-V pipeline.

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
- Corrected native lowering for tessellation patch inputs; ray-intersection result typing and
  setters; custom, direct, narrow, and integer imageblocks; embedded texture arrays; array gather
  operands; array depth comparison sampling; and integer storage-image atomics.
- Corrected Metal SIMD/quad operations for vector `u16` prefix scans, integer extrema, active
  masks, and exact votes, and preserved signedness for same-width integer conversions and atomic
  subtraction.
- Repaired additional raw byte/word buffer, opaque-pointer, aggregate-pointee, record-layout,
  cross-storage select, and late pointer-typing cases without name-keyed workload exceptions.
- Reduced worst-case translation time and memory growth by caching retry verdicts, pruning dead CFG
  before source re-emission, using linear CFG ordering and candidate scans, bounding generated CFG
  growth, and applying resource limits from worker startup.
- Validation now uploads sampled 3D images through bounded transfer buffers instead of relying on
  non-portable linear-tiling 3D images, restoring exact issue-4 and issue-5 Metal/MoltenVK checks.
- Opaque private-tensor descriptor intrinsics now have an explicit exact static-linkage contract,
  keeping their Apple-defined layout paired with the externally defined tensor-operation helper
  that consumes it instead of inventing a partial native representation.

## v0.1.0

### Added

- First public release of the `metal2vulkan` crate and CLI: native Metal AIR / sanitized LLVM IR →
  Vulkan SPIR-V.
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
