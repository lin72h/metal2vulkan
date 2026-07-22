# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

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
