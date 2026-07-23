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

### SPIR-V / pipeline utilities

| Binary | Purpose |
|---|---|
| `spirv_delta` | Classify SPIR-V byte/ID/order deltas |
| `spirv_pipeline_probe` | Load a module and build a compute pipeline |
| `spirv_pipeline_crash_predicate` | Interestingness predicate for pipeline crash reduction |
| `corpus-harvest` | Harvest `.metallib` AIR into JSONL shards |

### Corpus ledger CLIs (`corpus-*`)

Design: repo-root [`plan.md`](../plan.md). Ledgers are **JSONL**: translate ledger is hashes only;
execution ledgers (`-metal` / `-vulkan` / `-moltenvk`) store hash-identified plans, deterministic
input/output digests, and full run `output_b64` payloads (no AIR/source bodies).

| Binary | Writes / role |
|---|---|
| `corpus-mint` | Append **new** `air_sha256` rows → `corpus/metal2vulkan-ledger.jsonl` (translate pins) |
| `corpus-remint` | Re-translate existing ledger rows; rewrite those rows (`--failed-only` = `status != ok`) |
| `corpus-why` | One `air_sha256` → real translate FALLBACK error (no ledger write) |
| `corpus-run-metal` | macOS Metal oracle → `corpus/metal2vulkan-ledger-metal.jsonl` (plan + golden `output_b64`) |
| `corpus-run-vulkan` | Linux Vulkan vs metal golden → `corpus/metal2vulkan-ledger-vulkan.jsonl` |
| `corpus-run-moltenvk` | MoltenVK vs metal golden → `corpus/metal2vulkan-ledger-moltenvk.jsonl` |
| `corpus-triage` | Summarize ledgers, bucket failures, print single-case rerun commands |

#### Translate pins

```sh
# Additive: public fixtures + local shards; only hashes not already in the ledger
cargo run -p metal2vulkan-validation --release --bin corpus-mint -- --dry-run
cargo run -p metal2vulkan-validation --release --bin corpus-mint

# Remint existing rows (all with source in public fixtures/shards, or failures only)
cargo run -p metal2vulkan-validation --release --bin corpus-remint -- --dry-run
cargo run -p metal2vulkan-validation --release --bin corpus-remint
cargo run -p metal2vulkan-validation --release --bin corpus-remint -- --failed-only

# Diagnose one hash (surfaces the translator error mint banks as status=fallback)
cargo run -p metal2vulkan-validation --release --bin corpus-why -- <air_sha256>
```

`corpus-mint`: `--jobs N`, `--quiet`, `--ledger PATH`, `--dry-run`.
`corpus-remint`: same flags plus `--failed-only`.

#### Execution goldens (Metal → Vulkan / MoltenVK)

For each eligible translate-ledger row **missing** from the tech ledger (or an existing row selected
by `--force`, `--failed-only`, `--status`, `--bucket`, or `--contains`): resolve or infer a harness
plan, seed non-zero inputs, run, append to a per-run delta, then merge that delta into the backend
ledger once with `air_sha256` dedupe.

- **`corpus-run-metal`:** translate `status=ok` **or** `fallback` (Metal runs AIR; translator
  FALLBACK does not block the oracle).
- **`corpus-run-vulkan` / `corpus-run-moltenvk`:** translate `status=ok` only (need SPIR-V).

```sh
# macOS: Metal oracle (banks plan + full output_b64 + digests)
cargo run -p metal2vulkan-validation --release --bin corpus-run-metal -- --dry-run
cargo run -p metal2vulkan-validation --release --bin corpus-run-metal

# Linux: Vulkan candidate vs metal golden (needs metal rows)
cargo run -p metal2vulkan-validation --release --bin corpus-run-vulkan -- --dry-run
cargo run -p metal2vulkan-validation --release --bin corpus-run-vulkan

# macOS: MoltenVK candidate (separate ledger; same plan as metal)
cargo run -p metal2vulkan-validation --release --bin corpus-run-moltenvk
```

Runners: `--force`, `--failed-only`, `--status STATUS`, `--bucket TEXT`, `--contains TEXT`,
`--quiet`, `--jobs N` (default **min(CPU cores, 4)** parallel workers; raise carefully on GPU
hosts), `--air-sha256 HEX`, `--ledger-dir DIR`, `--dry-run`.
Per-case worker timeout defaults to **60s** and can be overridden with
`METAL2VULKAN_CORPUS_TIMEOUT_SECS=N`; there is no timeout CLI flag.
Tech JSONL files are **deduped by `air_sha256` at delta merge** (last row for a hash wins).
For candidate ledgers, `status=ok` and `status=tolerance` are accepted success states;
`--failed-only` reruns existing non-success rows. Use `--status` / `--bucket` / `--contains` for a
targeted batch over one triage group.

**Plan source of truth:** metal JSONL row when present; otherwise lazy `infer_plan` from `.ll`.
Candidates **reuse** the metal plan. Tolerances for candidate outcomes live **on the candidate
JSONL line** (`status` / `tolerance` / `observed`), not a separate global file.

Eligibility for `corpus-run-*`:

```text
source in public fixtures or local shards  AND  air_sha256 not yet in tech ledger
  AND  (metal: translate status ok|fallback
        vulkan/moltenvk: translate status ok)
```

`--force` ignores the tech-ledger presence check. `--failed-only`, `--status`, `--bucket`, and
`--contains` select existing tech rows instead of new missing rows.

macOS oracle entry points live in the library (`oracle_macos`); the Vulkan byte-run executor is
`runner_linux` (built on both Linux and macOS for MoltenVK).

#### Triage / agent loop

```sh
# Summarize all ledgers and list the first non-success rows
cargo run -p metal2vulkan-validation --release --bin corpus-triage

# Pick MoltenVK pipeline failures and print exact reproduction commands
cargo run -p metal2vulkan-validation --release --bin corpus-triage -- \
  --backend moltenvk --status fallback --contains pipeline --commands

# Rerun one hash, then the matching bucket
cargo run -p metal2vulkan-validation --release --bin corpus-run-moltenvk -- \
  --air-sha256 <air_sha256> --force --jobs 1
cargo run -p metal2vulkan-validation --release --bin corpus-run-moltenvk -- \
  --status fallback --bucket "create vertex validation pipeline" --jobs 1

# Full non-success sweep; useful late, but broad.
cargo run -p metal2vulkan-validation --release --bin corpus-run-moltenvk -- \
  --failed-only --dry-run
```

## Optional private corpus

See [`corpus/README.md`](corpus/README.md). `corpus-harvest` writes shard JSONL under
`corpus/local/shards/` (gitignored). Corpus ledger tools resolve those shards directly.

Cross-version **translate** pins: `corpus/metal2vulkan-ledger.jsonl`.  
**Execution** pins: `metal2vulkan-ledger-{metal,vulkan,moltenvk}.jsonl` (optional tracked
reproducibility artifacts; no AIR/source bodies).

For live before/after SPIR-V compares without execution goldens, use
[`scripts/metal2vulkan-ab/`](../scripts/metal2vulkan-ab/).

**Developer playbook:** [`docs/VALIDATION.md`](../docs/VALIDATION.md). Full execution design:
[`plan.md`](../plan.md).
