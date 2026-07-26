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
  with decoded filter, address, coordinate, compare, LOD, border-color, reduction, and raw-word
  state. Texture arrays also report sampled vs storage access through `ResourceBinding::access`.
- Additional stage-interface coverage for fragment builtins such as `[[point_coord]]` and
  `[[primitive_id]]`, flat fragment varyings, framebuffer-fetch color inputs, vertex builtins, and
  fragment outputs with nonzero render-target locations.
- Native lowering support for fixed and runtime-indexed texture arrays, storage-image array access,
  more texture gather/sample/read/write shapes, half/integer render-target formats, and scalar
  64-bit integer arithmetic emulation.
- Workgroup-memory lowering for deterministic zero initialization, small Workgroup atomic-loop
  unrolling, and additional atomic reinterpretation patterns used by shared-memory reductions.
- Rust validation/corpus tooling: `corpus-harvest`, `corpus-mint`, `corpus-remint`, `corpus-why`,
  `corpus-run-metal`, `corpus-run-vulkan`, `corpus-run-moltenvk`, and `corpus-triage`, backed by
  translate and execution JSONL ledgers.

### Changed

- Translation now derives AIR metadata once per entry and shares it across emission, reflection,
  lowering passes, and retry tiers, including the function-constant-buffer-promoted kernel metadata
  used by retry paths.
- The retry cascade and structured-CFG repair path were bounded and hardened: planner clone/search
  growth is capped, fallback classification is more specific, graph-walk retry inputs are cached,
  and timeout-prone retry rows are handled with per-case limits in validation tooling.
- Floating-point behavior is closer to AIR for the covered cases, including f32-to-f16 clamping,
  bf16 narrowing/NaN handling, fast `sin`/`cos` large-argument behavior, `pow` zero edges, and
  exact `mix` endpoints.
- Buffer, pointer, and access-chain lowering handles more structural cases, including dynamic
  struct/word indices, local pointer tables, pointer-select loads, aggregate memcpy forms, raw
  subword loads/stores, unaligned metadata byte fields, and direct call pointer results.
- Validation now uses local shard JSONL as the private corpus source of truth and committed
  `metal2vulkan-ledger*.jsonl` files for hash-identified translate/execution evidence. The ledger
  files are tracked with Git LFS.
- Vulkan validation reruns now account stale or non-comparable Metal goldens as `missing`/skip
  diagnostic rows instead of failed candidate regressions, so `--status ok --force` is usable as a
  regression gate.
- Developer documentation now describes the ledger-based validation ladder and the updated
  reflection v3 descriptor contract.

### Fixed

- Visible Metal function references now fail fast with an explicit fallback instead of being treated
  as ordinary functions.
- Multiple `OpReturnValue` sites are rewritten consistently to stage outputs, and undefined
  fragment output stores are skipped rather than materialized.
- Kernel local-size options are validated as nonzero and are propagated into AIR local-size queries,
  imageblock lowering, and validation execution plans.
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
- Hardened validation seed planning for bounded control data: byte-addressed dynamic GEP stride
  controls and typed `bool` fields now receive valid bounded values, and stale goldens with older
  layouts are marked for rebanking instead of producing backend-dependent mismatches.
- Expanded finite-struct float validation seeding to cover repeated scalar/vector float fields and
  repeated nested structs.

### Removed

- Removed the old `scripts/metal2vulkan-drift/` and `scripts/metal2vulkan-harvest/` workflows in
  favor of the validation crate's Rust corpus/ledger binaries.
- Removed the generated local corpus test workflow and obsolete corpus files
  (`drift-ledger.jsonl`, `broken.jsonl`, `tolerances.jsonl`, and `validation/TOOLCHAIN.lock`).

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
