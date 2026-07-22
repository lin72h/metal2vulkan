# metal2vulkan-validation

Optional helpers for exercising `metal2vulkan` against a local Metal oracle (macOS) and/or a local
Vulkan ICD (Linux or MoltenVK on macOS).

This package is a **workspace member** and is **not published** to crates.io. It does not ship
third-party captured shaders.

## Build / test

From the repository root:

```sh
cargo test -p metal2vulkan-validation -- --test-threads=1
```

Or from this directory:

```sh
cargo test -- --test-threads=1
```

Linux **executor** tests (notably `atomic_float_executor`) need a Vulkan ICD. On a machine with
no GPU, install lavapipe:

```sh
# Debian/Ubuntu
sudo apt-get install -y mesa-vulkan-drivers libvulkan1
```

Missing ICDs surface as `create Vulkan instance: … IncompatibleDriver` /
`VK_ERROR_INCOMPATIBLE_DRIVER`, not as old `spirv-tools` / LLVM. The test file is
`#![cfg(target_os = "linux")]`, so macOS CI skips it rather than exercising MoltenVK for that
proof.

## Tools

| Binary | Purpose |
|---|---|
| `spirv_delta` | Classify SPIR-V byte/ID/order deltas |
| `spirv_pipeline_probe` | Load a module and build a compute pipeline |
| `spirv_pipeline_crash_predicate` | Interestingness predicate for pipeline crash reduction |
| `metallib_gen_tests` | Generate gitignored `tests/corpus_NN.rs` stubs from local JSONL shards |

macOS oracle entry points live in the library (`oracle_macos`); the Vulkan byte-run executor is
`runner_linux` (built on both Linux and macOS).

## Optional private corpus

See [`corpus/README.md`](corpus/README.md). Drop metallibs/AIR/JSONL under `corpus/local/`
(gitignored), generate stubs with `metallib_gen_tests`, and run `run_corpus_case` translate smokes.
Cross-version **hash** pins (no shader bodies) live in `corpus/drift-ledger.jsonl` and are checked
with [`scripts/metal2vulkan-drift/`](../scripts/metal2vulkan-drift/). Per-AIR tolerances and
not-applicable skips are committed as JSONL: `corpus/tolerances.jsonl`, `corpus/broken.jsonl`.

**Developer playbook** (when to harvest, bank hashes, A/B, what not to commit):
[`docs/VALIDATION.md`](../docs/VALIDATION.md).
