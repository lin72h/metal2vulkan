# AGENTS.md

Operating guide for any AI agent working in this repository. Follow it unless the user
explicitly overrides a rule for the current task.

## What this project is

**metal2vulkan** is a standalone Rust crate and CLI that translates **Metal AIR** (LLVM bitcode
or sanitized `.ll`) to **Vulkan SPIR-V** with a native Rust emitter.

- **License:** LGPL-3.0-or-later (`LICENSE`)
- **Publishable crate:** `metal2vulkan` (library + `metal2vulkan` binary)
- **Unpublished workspace member:** `validation/` (`metal2vulkan-validation`, `publish = false`) —
  authored semantic cases, dependency-exact observations, deterministic harvest, and GPU-free A/B

This tree is **not** the paravirtual GPU monorepo. Do not reintroduce monorepo paths (`host/`,
`vm/`, `kb/`, `journal/`), device/protocol code, or private capture corpora.

**Orientation for humans and agents:**

| Doc | Role |
|---|---|
| `README.md` | Install, CLI, quick library use |
| `docs/HOWTO.md` | End-to-end translation and consumer integration |
| `docs/ARCHITECTURE.md` | Pipeline, structurizer/relooper, retry cascade |
| `docs/REFLECTION.md` | Consumer binding metadata (`ShaderReflection`) |
| `docs/VALIDATION.md` | Developer validation ladder (harvest, drift, A/B) |
| `CONTRIBUTING.md` | Day-to-day build/test loop |
| `src/env_vars.rs` | Authoritative `METAL2VULKAN_*` registry (`--help`) |

## Components (start here)

| Path | Role |
|---|---|
| `src/` | Product library + CLI (`translate*`, native emitter, passes, reflect) |
| `tests/` | Integration tests for the product crate |
| `examples/` | Small runnable examples |
| `validation/` | Authored-case validation tooling (not published to crates.io) |
| `docs/` | Architecture and consumer guides |
| `scripts/` | Dev utilities (`mtlb-extract`, SPIR-V grammar regeneration) |
| `.github/workflows/ci.yml` | Format, warnings-denied clippy/Rustdoc, and parallel tests on `ubuntu-26.04` + `macos-26`; public Vulkan qualification on Linux |

Crown-jewel code lives under `src/native/`, `src/passes/`, and `src/reflect/`. Prefer structural
fixes over one-off workarounds.

## Ground rules

### Translation honesty

- **Structure and semantics over names.** Decide emit/lowering from IR structure (types, storage
  classes, access chains, AIR metadata ABI) — **never** from a hardcoded function, type, variable,
  or corpus-style identifier. A name-keyed branch that green-lights one shader while failing
  identically shaped others is a defect.
  - **Allowed:** dispatch on **stable ABI symbols** that are part of the AIR/LLVM contract
    (`air.*`, `llvm.*` families). Prefer a structural test when one exists.
  - **Honest FALLBACK:** unsupported inputs return `Err` / CLI `FALLBACK`. Do not emit
    wrong-but-valid SPIR-V to silence a gate.
- **No env-gated product paths.** Product translation behavior must not branch on env vars. Keep
  `METAL2VULKAN_*` for operational settings, tool path overrides, and default-off diagnostics /
  measurement only (`src/env_vars.rs`).
- **Unknown stays unknown.** Do not invent semantics for unsupported opcodes, types, or layouts.
  Fail visibly and leave a test or doc note if the gap is real.
- **Fix causes, not observations.** Temporary probes may explain a failure; committed behavior
  needs a structural/ABI justification, not matching one workload’s bytes by special-case.

### Claims and evidence

- **A claim is only as broad as its evidence.** Verifying one path does not license “all / never /
  zero” statements. Scope replies, commits, and docs to what you actually ran.
- Say **verified** only for commands you executed and results you saw; use **expected** /
  **unaudited** for the rest.

### Measurement before polish

- Prefer the smallest change that can be **tested** (unit/integration, local A/B, or a synthetic
  `.ll` fixture). Do not ship third-party captured shaders or mined corpora.
- When fixing a bug class, add or update a **test** (or a clear measurable check) so it cannot
  regress silently.

### Validation coverage is a closed contract

- **Chase closure, not a better-looking queue.** The authoring-capability census covers every
  indexed AIR identity, including identities that already have authored cases. A fresh complete
  census is closed only when `classified == total`, `remaining == 0`, and `unresolved == 0`.
  Reviews, annotations, allowlists, previous observations, or a smaller selection do not resolve a
  tooling gap. A newly observed requirement is work to model honestly, not a row to suppress.
- **One typed capability vocabulary owns the truth.** `ToolingRequirement` and the shared
  structural checks in `validation/src/executor_contract.rs` are the contract among classification,
  the case checker, both Metal and Vulkan executors, persisted facts, and reporting. Do not grow
  parallel string lists or component-local support tables. Adding support for a structure means
  updating the literal schema, validation, both applicable execution paths, classification, and
  regression tests together; if one side cannot execute it exactly, it remains an unsupported
  requirement.
- **Describe supported families independently from missing tooling.** A focused `AuditTarget`
  selects current typed structural facts, not rows carrying an unsupported requirement. Fixing a
  capability must not make its regression audit select zero rows. Keep descriptive facts and
  blocking requirements separate even when they initially identify the same corpus family.
- **Prefer architectural fit over policy patches.** Put shared semantics at the narrowest common
  boundary and delete superseded blocker fields, adapters, commands, and duplicated mappings.
  Never post-filter a stale classifier result, special-case a hash/name, or add a review policy to
  make incompatible components appear consistent.
- **The disposable index is the acceleration structure, not another corpus.** It owns compact
  identities, exact source byte locations, dependency hashes, and versioned structural facts.
  Ordinary refresh and lookup paths must inspect only new/changed shards or directly selected byte
  ranges. Once warm, a no-change capability audit should select no rows and open/read zero source
  shards; exact-row and focused audits must not scan unrelated shards.
- **Cache validity follows semantics.** Analyzer, executor, oracle, product, and dependency
  fingerprints must include every behavior that can change classification, translation, or
  expected output. Bump the appropriate version when that behavior changes and recompute through
  the normal indexed path. Do not trust cached validation produced by different semantics, and do
  not rebuild or rescan source data merely to invalidate derived facts.
- **Tests own their inputs.** Index, cache, and audit tests must build isolated synthetic shards and
  observations. They may not depend on the developer's ambient private corpus, pre-existing index,
  filesystem ordering, or cache warmth.
- **Use the right proof for the claim.** After capability/classifier changes, run a full fresh
  `authoring-capabilities --reclassify-all` census and then a warm pass that proves zero source
  reads. After feature-specific changes, run the corresponding focused audit across its complete
  selected family in bounded, resumable batches. After product translation changes, run a fresh
  translation fingerprint sweep when the local corpus is available. Corpus-wide claims require
  corpus-wide results; otherwise state the measured subset.
- **Authored evidence is execution, not bookkeeping.** Qualify an authored case freshly on Metal,
  then execute the Vulkan candidate and compare the declared observations. Existing observation
  files, successful SPIR-V validation, or an authored manifest alone are not semantic proof.
- **`authored_linkage_required` is narrow and structural.** Use it only when exact AIR structure
  establishes that translation needs explicit authored linked inputs which a standalone harvested
  row cannot supply. Timeouts, resource-limit breaches, malformed input, unsupported lowering, and
  ordinary translator errors remain failures; never relabel them to complete a census.

### Translation performance is a correctness contract

- **30 seconds is a hard end-to-end ceiling per translation attempt.** Measure from handing the
  selected AIR input to an isolated worker through translation, SPIR-V validation, and reporting
  the result. Success or an honest unsupported-input `FALLBACK` must arrive within the ceiling;
  exceeding it, timing out, or killing a stuck worker is a failing result that must be diagnosed
  and fixed before the affected work is done. A timeout is a safety rail, not an acceptable way
  to classify supported input. Do not meet the ceiling by skipping required validation, reducing
  semantic coverage, or converting an input that was supported before into a premature fallback.
- **500 MiB is the hard memory budget per translation attempt.** Enforce it at the worker
  boundary, covering peak translation-attributable live allocations and resident growth. Kill
  the complete worker process group on a time or memory breach and clean up its scratch files.
  This explicit 500-MiB ceiling supersedes the former 300/350-MiB escalation contract. Do not raise
  it again unless a new explicit user instruction supersedes this rule; no automatic increase
  remains available.
- **Keep total resource use bounded.** Parallelize only independent rows or phases, cap worker and
  queue counts, and account for the aggregate worst case (`workers * 500 MiB`). No optimization
  may introduce unbounded threads, processes, queues, caches, retained IR, or candidate modules.
- **Use host parallelism for corpus runs.** Corpus tooling defaults `jobs` to the host's available
  logical CPU count (the equivalent of `nproc`). Use that default for full-corpus work; pass an
  explicit `--jobs N` only when the user requests an override.
- **Remove work instead of hiding it.** Prefer shard-local incremental indexing, parsing only the
  selected or changed source, caching immutable analysis, reusing still-valid CFG facts, skipping
  equivalent retry states, and releasing failed candidates promptly. Never reuse analysis across
  a semantics-affecting rewrite unless its validity is structurally guaranteed.
- **Warm indexes must prevent corpus-wide revisits.** Refresh only new or changed sources; an
  exact-row translation should read its hash-derived shard, not scan unrelated shards. If a
  compatibility migration needs a broader scan, make it explicit, resumable, one-time, and
  measured rather than silently placing it on the translation path.
- **Performance changes require release-mode evidence.** When touching translation, retry,
  planner, indexing, or validation hot paths, measure the end-to-end wall time and peak memory for
  the largest relevant locally available row plus a bounded representative batch. Report the
  slowest row and the scope measured. Debug-build timing does not verify the ceiling.
- When a slow or memory-heavy failure class is found, add a deterministic regression check for
  the underlying redundant-work or boundedness property. Avoid machine-fragile microbenchmarks;
  keep the 30-second and 500-MiB worker guards as hard integration backstops.
- **Do not normalize known breaches.** An observed over-budget translation is an active correctness
  bug, not acceptable technical debt or a reason to weaken the workload. Profile where its time
  and memory go, remove redundant or global work structurally, and remeasure the same input under
  the same release-mode boundary before claiming the regression fixed.

### Tooling and temp files

- External tools: **`llvm-dis`** for AIR bitcode input, **`spirv-val`** for product validation, and
  **`spirv-as`** for passthrough generation. Resolve via PATH or `METAL2VULKAN_<TOOL>` overrides.
- Scratch files under the OS temp dir (or a caller-supplied `tmp`) must be **removed as soon as
  the tool no longer needs them**. The CLI removes its work directory on success and before
  `process::exit` on FALLBACK. Do not reintroduce long-lived dumps under fixed `/tmp/...` paths.
- FALLBACK **repro bundles** under `$TMPDIR/metal2vulkan-repros` (or `METAL2VULKAN_REPRO_DIR`) are
  intentional and may be kept for debugging.

### What not to commit

- Apple-owned binaries, guest disk images, mined metallib/AIR corpora, or private golden sets
- Contents of `validation/corpus/local/`, raw `*.metallib` / `*.air` / `*.spv` dumps
  (all gitignored — see `validation/corpus/README.md`)
- `target/`, `Cargo.lock` (gitignored for this library-first tree), `.cache/`
- Name-keyed special cases “just to pass one case”

**Allowed (committed):** synthetic fixtures under `validation/fixtures/public/`, authored case
shards under `validation/corpus/cases/`, and dependency-exact observation shards under
`validation/corpus/observations/`. None may contain AIR source bodies.

## Layout & ownership

- **Product logic** stays in the `metal2vulkan` crate (`src/`).
- **Optional offline/oracle work** stays in `validation/` (may depend on MoltenVK or a Vulkan ICD).
  Do not pull that into the published crate’s default dependency graph.
- **Grammar tables** under `src/spirv_binary/*_generated.rs` are regenerated via
  `scripts/regen-spirv-grammar/` from Khronos SPIRV-Headers + rspirv-autogen; do not hand-edit.
- **Reflection** for consumers: `src/reflect/` + `docs/REFLECTION.md`. Binding numbers must stay
  aligned with the interface pass ABI (set 0; bases 0 / 32 / 64 / 96 / 128 / 160).

## Build, test, CI

Run Rust tests with Cargo's default available parallelism (the logical CPU count):

```sh
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
RUSTDOCFLAGS='-D warnings' cargo doc --workspace --all-features --no-deps
cargo test -p metal2vulkan
cargo test -p metal2vulkan-validation
```

- **MSRV:** see `rust-version` in `Cargo.toml` (currently 1.87).
- **Warnings are hard failures.** CI sets `RUSTFLAGS=-Dwarnings` and runs clippy with
  `-D warnings`. Treat any new rustc or clippy warning as a bug you must fix before committing —
  do not leave `#[allow(...)]` noise to paper over real issues, and do not ask reviewers (or a
  later CI run) to clean up after you.
- **Features:** `serde` enables `ShaderReflection` JSON and CLI `--emit-meta`.
- CI runners: **`ubuntu-26.04`** and **`macos-26`** (not `*-latest`).

When diagnosing a hang, prefer reading full `cargo test` output over piping through `head`/`grep`
(easy to hide a later failure).

## Code style

- **No empty-placeholder ownership tricks.** Do not use `std::mem::take`, `mem::replace` with an
  empty/default sentinel, `swap` with an empty value, or an equivalent helper to move IR, modules,
  functions, blocks, instructions, or analysis collections out of a borrowed owner. Express the
  real ownership contract instead: consume and return the owned value, borrow the existing
  allocation, or redesign the transformation around an explicit result. When touching an existing
  occurrence, remove it structurally rather than copying the pattern.

- **Format before every commit.** Run `cargo fmt --all` (repo `rustfmt.toml`) on the whole
  workspace so the diff you commit already matches what CI’s `fmt --check` expects. If a change
  is pure formatting, keep it in its own commit when mixed with logic would obscure the review.
- **Clippy must be clean.** Before you commit Rust edits, run
  `cargo clippy --workspace --all-targets -- -D warnings` and clear every finding. Match
  surrounding module style; prefer small, reviewable files and structural fixes over growing
  special-case tables.
- Public API and CLI changes should update `README.md` / `docs/` when consumer-visible.

## Git workflow

- Prefer **focused commits** with a clear subject and body (what / why / how verified).
- **Pre-commit gate (Rust):** `cargo fmt --all`, clippy with `-D warnings`, and warnings-denied
  all-features Rustdoc as above. Do not commit unformatted code or known clippy/rustc/Rustdoc
  warnings; CI will fail the same checks.
- Do not force-push shared history unless the user asks.
- Do not commit secrets, large binaries, or lockfiles that this repo intentionally ignores.

## The loop (unit of work)

1. **Scope** — smallest testable increment; name how you will verify it.
2. **Validate** — read the relevant code and `docs/`; do not assume monorepo facts.
3. **Change** — implement; keep product paths free of env gates and name keys.
4. **Test** — serial `cargo test` for the packages you touched; full workspace clippy with
   `-D warnings` when Rust sources changed; warnings-denied all-features Rustdoc when public APIs or
   documentation changed.
5. **Document** — update `docs/` or comments only when the durable contract changed.
6. **Commit** — `cargo fmt --all` first, then one concern per commit when practical. Only commit
   when fmt, clippy, and applicable Rustdoc checks are clean.

**Done** = it works under the checks above, claims match evidence, fmt/clippy/Rustdoc are green,
measured translations in the touched scope stay within 30 seconds and 500 MiB, relevant capability
audits remain closed without stale evidence or unrelated source reads, and the tree stays
publishable as a general AIR→SPIR-V translator.
