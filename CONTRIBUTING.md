# Contributing

Thanks for interest in `metal2vulkan`. This document covers the day-to-day developer loop for the
standalone crate.

## Prerequisites

- Rust stable (see `rust-version` in `Cargo.toml`)
- External tools used by some paths:
  - `llvm-dis` (LLVM)
  - `spirv-val` (SPIRV-Tools)
  - `spirv-diff` / `spirv-as` (validation suite / some tools)
- Optional but required for Linux executor tests (e.g.
  `shader_shared_f32_atomic_add_executes_known_result`): a Vulkan ICD. On headless Linux that
  means `mesa-vulkan-drivers` (lavapipe) + `libvulkan1`. Without an ICD, `Instance::new` fails
  with `VK_ERROR_INCOMPATIBLE_DRIVER`. On macOS, MoltenVK for the same path.

## Repository layout

```text
.
├── src/                  # library + CLI
├── tests/                # integration tests
├── examples/             # cargo examples
├── validation/           # optional oracle / Vulkan helpers (workspace, not published)
├── docs/                 # architecture + reflection notes
├── scripts/              # developer utilities
├── Cargo.toml            # workspace root + metal2vulkan package
├── LICENSE               # LGPL-3.0-or-later
└── README.md
```

| Path | Role |
|---|---|
| `src/` | Library + CLI (`metal2vulkan`) |
| `tests/` | Integration tests |
| `examples/` | Small runnable examples |
| `validation/` | Optional oracle / Vulkan helpers (not published) |
| `docs/` | Design notes (`ARCHITECTURE.md`, `REFLECTION.md`) |
| `scripts/` | Developer utilities (A/B harness, mtlb-extract, grammar regen, …) |

## Development

```sh
# format
cargo fmt --all

# clippy (CI denies warnings)
cargo clippy --workspace --all-targets -- -D warnings

# unit + integration tests (serial)
cargo test -p metal2vulkan -- --test-threads=1

# optional validation package
cargo test -p metal2vulkan-validation -- --test-threads=1
```

External tools used by some paths: `llvm-dis`, `spirv-val` (and friends). On macOS with Homebrew:

```sh
PATH=/opt/homebrew/opt/llvm/bin:$PATH cargo test -p metal2vulkan -- --test-threads=1
```

Byte-level A/B of two translator binaries: [`scripts/metal2vulkan-ab/`](scripts/metal2vulkan-ab/).

## Build

```sh
cargo build --release
cargo build --release --features serde
```

## Design rules

1. **Structure and semantics over names.** Translate from AIR/LLVM structure (types, storage classes,
   access chains, AIR metadata ABI). Do **not** special-case individual shader, function, or type
   names observed in a particular workload.
2. **Stable ABI symbols are allowed.** Dispatching on documented `air.*` / `llvm.*` intrinsics is
   fine; prefer structural tests when possible.
3. **No third-party captured shaders in tree.** Reduce regressions to synthetic `.ll` tests.
   Optional **private** harvest writes gitignored shard JSONL under
   `validation/corpus/local/shards/` — see
   [`validation/corpus/README.md`](validation/corpus/README.md). Translate ledgers store hashes;
   execution ledgers may also store deterministic input/output digests and output payloads for
   reproducibility, but never AIR source bodies.
4. **Honest FALLBACK.** Unsupported inputs must return `Err` / CLI `FALLBACK`, never emit
   wrong-but-valid SPIR-V.

### Validation while refactoring

The full developer workflow (synthetic tests → binary A/B → hash ledger → private system
harvest → optional oracle) is in **[`docs/VALIDATION.md`](docs/VALIDATION.md)**.

Quick anchors:

```sh
# always
cargo test -p metal2vulkan -- --test-threads=1

# before/after a byte-stable refactor
cp target/release/metal2vulkan ./m2v-old
# … edit …
scripts/metal2vulkan-ab/metal2vulkan-ab.sh --old ./m2v-old
# optional: translate ledger + execution goldens (see validation/README.md, plan.md)
# cargo run -p metal2vulkan-validation --release --bin corpus-mint
# cargo run -p metal2vulkan-validation --release --bin corpus-remint -- --failed-only
# cargo run -p metal2vulkan-validation --release --bin corpus-why -- <air_sha256>
# cargo run -p metal2vulkan-validation --release --bin corpus-run-metal   # macOS
# cargo run -p metal2vulkan-validation --release --bin corpus-run-vulkan  # Linux
# cargo run -p metal2vulkan-validation --release --bin corpus-run-moltenvk

# optional private real-AIR bank (macOS; shard JSONL is gitignored)
cargo run -p metal2vulkan-validation --release --bin corpus-harvest
```

See also [AGENTS.md](AGENTS.md) for the full agent operating guide.

## Pull requests

- One focused change per PR when practical
- Include tests for new behavior or bug fixes
- Update `docs/ARCHITECTURE.md` when the pipeline shape changes
- Update `docs/REFLECTION.md` when the consumer binding contract changes
- Update `CHANGELOG.md` under `[Unreleased]`
