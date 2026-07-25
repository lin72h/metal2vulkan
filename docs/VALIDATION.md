# Validation workflow — how not to ship a broken translator

This is the developer playbook for **staying confident while you change the emitter**. It is
written for people who will refactor, not for people who only want a green CI badge.

CI on a clean clone is deliberately **synthetic-only**: unit tests, small owned fixtures, no Apple
system metallibs. That keeps the public tree legal and fast. Everything else below is optional
**local power** — private harvest, hash ledgers, A/B binaries, oracle/executor probes — that you
turn on when the change is large enough that synthetic tests alone would lie to you.

Related:

| Doc / tree | Role |
|---|---|
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | How the translator is built |
| [`REFLECTION.md`](REFLECTION.md) | Consumer binding metadata |
| [`../CONTRIBUTING.md`](../CONTRIBUTING.md) | Day-to-day build loop |
| [`../validation/`](../validation/) | Optional oracle / executor crate |
| [`../validation/corpus/`](../validation/corpus/) | Translate/execution ledgers + private corpus layout |

---

## The mental model: ladders, not a single gate

Think in **layers of evidence**. Climb as far as the risk of your change warrants.

```text
  ┌─────────────────────────────────────────────────────────────┐
  │  E  Execution (optional)                                    │
  │     Metal oracle / Vulkan runner — “bytes match a golden”   │
  ├─────────────────────────────────────────────────────────────┤
  │  D  Private real AIR (gitignored)                           │
  │     system harvest → translate smokes / local corpus tests  │
  ├─────────────────────────────────────────────────────────────┤
  │  C  Hash ledger (public or private fingerprints)            │
  │     sha256(AIR) → sha256(SPIR-V) — “this module still emits │
  │     the same bytes” without shipping shader bodies          │
  ├─────────────────────────────────────────────────────────────┤
  │  B  Byte A/B of two translator binaries                     │
  │     old binary vs new on the same sample set                │
  ├─────────────────────────────────────────────────────────────┤
  │  A  Synthetic tests (always)                                │
  │     cargo test — owned .ll / unit cases, CI default         │
  └─────────────────────────────────────────────────────────────┘
```

| Layer | Cost | What it proves | What it does *not* prove |
|---|---|---|---|
| **A** Synthetic | seconds | structural regressions you already encoded | real system-shader surface |
| **B** Binary A/B | minutes | your refactor did not move SPIR-V on a sample | that the sample is complete |
| **C** Hash ledger | minutes | banked AIR still maps to banked SPIR-V | semantic correctness vs Metal |
| **D** Harvest corpus | tens of min (first time) | stage-marked system shaders still translate / val | full binding/oracle fidelity |
| **E** Execution | machine-specific | candidate matches oracle (or known tolerance) | every metallib on every OS |

A pure formatting or clippy-only change can stop at **A**. A structurizer or pointer rewrite
should climb at least to **B**, and preferably **C**/**D** if you have a local harvest.

---

## What “good” looks like after a refactor

You want all of these to be true for the scope you claimed:

1. **Synthetic suite still green** — `cargo test -p metal2vulkan -- --test-threads=1`.
2. **No accidental SPIR-V churn** on banked cases — A/B clean, ledger remint clean, *or* an
   intentional re-mint with a clear commit message.
3. **No silent FALLBACK explosion** — cases that used to emit still emit (or are honestly marked
   broken / fallback in the ledger).
4. **No name-keyed “fixes”** — you did not special-case a function name from the harvest to green
   a test (see design rules). Failures get a structural fix or an honest skip.

If you only have (1), you have CI confidence. If you have (1)+(2)+(3) on a fat private sample,
you have **refactor confidence**.

---

## One-time setup (do this once per machine)

### Prerequisites

- Rust (see `rust-version` in `Cargo.toml`)
- `llvm-dis`, `spirv-val` on `PATH` (Homebrew LLVM on macOS is fine)
- macOS if you want **system** metallib harvest; Linux can still mint public fixtures and
  `--metallib` paths you already have

```sh
export PATH="/opt/homebrew/opt/llvm/bin:$PATH"   # if needed
cargo build --release --bin metal2vulkan
cargo test -p metal2vulkan -- --test-threads=1
```

### Optional: seed a private real-AIR bank (macOS)

This is the “I refuse to trust only toy shaders” path. Output is **gitignored** — never commit it.

```sh
# Scan prioritized system metallibs → validation/corpus/local/shards/
# No default --limit: processes the full prioritized list (long first run).
cargo run -p metal2vulkan-validation --release --bin corpus-harvest

# Optional: bank private hashes into the committed ledger as kind=private
# (fingerprints only — still no shader bodies in git)
cargo run -p metal2vulkan-validation --release --bin corpus-mint
```

What harvest keeps:

- Real **Kernel / Vertex / Fragment** entries with `!air.*` stage metadata → JSONL shard rows
- Drops metallib-embedded **helpers** without stage meta (stdlib / libcall / MPS stitching)
  so they never land in ledgers or A/B samples
- Skips AIR larger than the configured size cap
- Stores sanitized `air_ll`, optional `blob_b64`, source metallib path/hash, and shard name in JSONL

Details: [`validation/corpus/README.md`](../validation/corpus/README.md).

---

## The everyday refactor loop

### 0. Know what you are claiming

Before you start:

- **Byte-stable refactor** (“must not change SPIR-V on the sample”) → bank ledger / A/B *before*
  editing, demand clean after.
- **Intentional SPIR-V change** (“emitter is allowed to move”) → re-mint is expected; still run
  translate smokes and unit tests; explain the delta class if you can (`spirv_delta`).

### 1. Capture “before” (when you care about byte stability)

```sh
cargo build --release --bin metal2vulkan
cp target/release/metal2vulkan ./m2v-old    # gitignored name pattern; fine locally
```

Optional: ensure the public + local ledgers are current *before* the change:

```sh
cargo run -p metal2vulkan-validation --release --bin corpus-mint
```

### 2. Make the change

Work as usual. Prefer structural tests in `src/**/tests` for any new bug class you fix — those
are the only regressions CI will guard for strangers.

### 3. Cheap gates (every save cycle)

```sh
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
cargo test -p metal2vulkan -- --test-threads=1
```

### 4. Byte A/B against the saved binary

```sh
cargo build --release --bin metal2vulkan
scripts/metal2vulkan-ab/metal2vulkan-ab.sh --old ./m2v-old
```

A/B walks (when present):

- `tests/fixtures/**` (private, gitignored)
- `validation/fixtures/public/**` (committed synthetic)
- `validation/corpus/local/shards/*.jsonl` (private harvest; temp `.ll` materialized only while A/B runs)

Shard rows use the embedded sanitized `air_ll`; no retained `local/air` mirror is required.

Exit 0 means no `DIFFERS` / asymmetric produce-vs-fail on that sample. `BOTH-FAIL` is not a
regression (both sides FALLBACK).

### 5. Hash ledger mint / remint (no need to keep two binaries forever)

Bank AIR/SPIR-V content hashes in `validation/corpus/metal2vulkan-ledger.jsonl` (no shader bodies).
Use byte A/B (`metal2vulkan-ab`) when you need a before/after binary compare on a live sample.

After an intentional emit change (or to bank **new** hashes only):

```sh
# additive: only air_sha256 not already in the ledger (public + local)
cargo run -p metal2vulkan-validation --release --bin corpus-mint
# review the ledger diff; commit only hash rows you mean to bank
```

To re-bank existing ledger pins after a translator fix (or a full intentional re-emit):

```sh
# remint every banked hash still present in public fixtures or local shards
cargo run -p metal2vulkan-validation --release --bin corpus-remint -- --dry-run
cargo run -p metal2vulkan-validation --release --bin corpus-remint

# only status != ok (fallback / timeout / …)
cargo run -p metal2vulkan-validation --release --bin corpus-remint -- --failed-only

# bounded timeout-row cleanup; each row runs in a killable subprocess
cargo run -p metal2vulkan-validation --release --bin corpus-remint -- --status timeout --skip 50 --limit 50 --case-timeout-secs 60
```

Rows without a matching public fixture or shard row are left alone. Prefer `--dry-run` first.

### 6. Execution ledgers (when translate-green is not enough)

Execution builds on the translate ledger and up to three backend ledgers:

```text
metal2vulkan-ledger.jsonl          # translate pins: AIR hash -> SPIR-V hash/status
metal2vulkan-ledger-metal.jsonl    # Metal oracle: plan + golden output_b64
metal2vulkan-ledger-moltenvk.jsonl # MoltenVK candidate vs Metal golden
metal2vulkan-ledger-vulkan.jsonl   # Linux Vulkan candidate vs Metal golden, when produced
```

Metal is the source of truth. Candidate runners reuse the Metal row's plan and compare their bytes
against the Metal row's `output_b64` / `output_sha256`. A candidate cannot establish its own plan
or golden.

```sh
# macOS: establish or refresh the oracle row
cargo run -p metal2vulkan-validation --release --bin corpus-run-metal -- \
  --air-sha256 <hash> --force --jobs 1

# macOS: rerun one MoltenVK candidate row
cargo run -p metal2vulkan-validation --release --bin corpus-run-moltenvk -- \
  --air-sha256 <hash> --force --jobs 1

# Linux: rerun one native Vulkan candidate row
cargo run -p metal2vulkan-validation --release --bin corpus-run-vulkan -- \
  --air-sha256 <hash> --force --jobs 1
```

For a batch over existing bad execution rows, prefer a targeted filter from `corpus-triage`:

```sh
cargo run -p metal2vulkan-validation --release --bin corpus-run-moltenvk -- \
  --status fallback --bucket "create compute pipeline" --jobs 1

# Full non-success sweep; useful late, but broad.
cargo run -p metal2vulkan-validation --release --bin corpus-run-moltenvk -- \
  --failed-only --dry-run
```

Runners skip hashes already present in that backend ledger unless `--force` or an existing-row
filter is set (`--failed-only`, `--status`, `--bucket`, `--contains`). `--dry-run` lists what would
run. The parent writes a per-run delta under the OS temp dir and then rewrites the backend ledger
once with `air_sha256` dedupe; the last row for a hash wins.

Candidate statuses:

| Status | Meaning | Agent action |
|---|---|---|
| `ok` | exact byte match | leave it alone |
| `tolerance` | float-like output is within recorded tolerance | accepted; leave it alone unless the policy is wrong |
| `smoke` | candidate executed and captured bytes, but the Metal row is `compare=none` rather than a semantic golden | accepted as an execution smoke; rebank Metal for semantic comparison |
| `failure` | candidate ran and produced bytes, but they do not match / exceed tolerance | inspect output class; reduce to a synthetic test when possible |
| `fallback` | translate, bind, pipeline creation, dispatch, readback, or harness panic before comparable bytes | inspect `error`; usually a harness or executor gap |
| `missing` | no usable Metal golden for this hash | rerun/fix `corpus-run-metal` first |
| `quarantine` | Metal loop guard refused to dispatch | do not force GPU execution; improve loop-budget proof or accept quarantine |
| `timeout` | worker process exceeded `METAL2VULKAN_CORPUS_TIMEOUT_SECS` | rerun single-case with `--jobs 1`; treat persistent timeouts as harness/tool hangs |

Use `corpus-triage` to see the current queue:

```sh
# summarize all ledgers and list the first non-success rows
cargo run -p metal2vulkan-validation --release --bin corpus-triage

# group/list MoltenVK failures and print reproduction commands
cargo run -p metal2vulkan-validation --release --bin corpus-triage -- \
  --backend moltenvk --status fallback --contains pipeline --commands
```

On macOS, validation's Metal oracle and MoltenVK-backed runner are machine-specific. On Linux, use
a native Vulkan ICD. Treat execution as **layer E**, not the daily default.

### 7. Agent loop over ledger failures

An agent should not start by rerunning the whole corpus. The useful loop is:

1. **Pick a bucket, not a random row.**

   ```sh
   cargo run -p metal2vulkan-validation --release --bin corpus-triage -- \
     --backend moltenvk --limit 0
   ```

   Work the largest actionable bucket first: pipeline creation panics, missing Metal rows, output
   mismatches, timeouts, etc. Quarantines are a loop-budget project, not a candidate rerun project.

2. **Pull one reproducible hash with commands.**

   ```sh
   cargo run -p metal2vulkan-validation --release --bin corpus-triage -- \
     --backend moltenvk --status fallback --contains "create vertex validation pipeline" --commands
   ```

3. **Diagnose translate separately from execution.**

   ```sh
   cargo run -p metal2vulkan-validation --release --bin corpus-why -- <hash>
   cargo run -p metal2vulkan-validation --release --bin corpus-run-moltenvk -- \
     --air-sha256 <hash> --force --jobs 1
   ```

   If the Metal row is missing, stale, or not `status=ok`, fix/rerun Metal before touching the
   candidate path.

4. **Classify the bug before editing.**

   | Observation | Likely owner |
   |---|---|
   | `corpus-why` FALLBACKs | product translator / stage detect |
   | Metal `fallback` with oracle error | Metal harness / oracle limitation |
   | Candidate `fallback` with `vulkan execute panicked` | Vulkan/MoltenVK executor harness |
   | Candidate `failure` with output bytes | product semantics, plan mismatch, or tolerance policy |
   | Candidate `missing` | Metal ledger is absent or stale |
   | Persistent `timeout` | harness/tool hang; rerun single-case before broad changes |

5. **Make the smallest structural fix and add a synthetic test when it is a translator class.**
   Execution-ledger rows are evidence; CI protection still comes from owned fixtures/tests.

6. **Rerun the same hash, then the bucket.**

   ```sh
   cargo run -p metal2vulkan-validation --release --bin corpus-run-moltenvk -- \
     --air-sha256 <hash> --force --jobs 1
   cargo run -p metal2vulkan-validation --release --bin corpus-run-moltenvk -- \
     --status fallback --bucket "create vertex validation pipeline" --jobs 1
   ```

   `--failed-only` is the whole existing non-success queue; use it as a dry-run or late sweep, not
   as the first bucket loop.

7. **Commit only intentional ledger movement.**
   A single forced rerun rewrites the row for that `air_sha256` during delta merge. Review the JSONL
   diff: `fallback -> ok`, `failure -> ok`, or `failure -> tolerance` are meaningful; a changed
   `output_b64` without an explanation is not.

For risky reruns, work against copied ledgers:

```sh
mkdir -p /tmp/m2v-ledgers
cp validation/corpus/metal2vulkan-ledger.jsonl /tmp/m2v-ledgers/
cp validation/corpus/metal2vulkan-ledger-metal.jsonl /tmp/m2v-ledgers/

cargo run -p metal2vulkan-validation --release --bin corpus-run-moltenvk -- \
  --ledger-dir /tmp/m2v-ledgers --air-sha256 <hash> --force --jobs 1
```

---

## Recipes by change type

### “I’m restructuring the structurizer / CFG / emitter”

1. Synthetic tests (**A**).
2. Save `m2v-old`, implement, A/B (**B**).
3. `corpus-remint --dry-run` / `corpus-remint` on public + any local pins (**C**).
4. If you harvest: `corpus-mint` for new rows and `corpus-triage` for execution queues (**D/E**).
5. Re-mint only if you *meant* SPIR-V to move, and say so in the commit.

### “I fixed one FALLBACK class”

1. Add a **synthetic** regression that would have failed before (so CI owns the class).
2. Confirm the real metallib case (if you have it in a local shard) translates.
3. Optionally bank its hashes so the fix cannot silently reverse.

### “I only touched docs / scripts / validation helpers”

Stay on **A** for the product crate; run validation package tests if you edited that crate.

### “I need a new public sample everyone can run”

1. Author a **small owned** `.ll` under `validation/fixtures/public/` (no third-party capture).
2. Run `corpus-mint` and commit the fixture + the new `kind=synthetic` translate-ledger row.
3. Never paste system-metallib AIR into the public tree.

### “Harvest is huge / slow”

- Full harvest is expensive because it re-carves and rewrites shard JSONL.
- Use `--limit` / `--offset` only when batching deliberately.
- A/B and corpus ledger tools materialize temporary `.ll` files from shard `air_ll` rows.

---

## What lives in git vs on your disk vs crates.io

| Artifact | Git? | crates.io? | Notes |
|---|---|---|---|
| Unit / synthetic tests | yes | yes (product crate) | CI backbone |
| `src/`, `tests/` (product) | yes | yes | Published package |
| `validation/` workspace member | yes | **no** | `publish = false` + root `exclude` |
| `validation/fixtures/public/*.ll` | yes | **no** | Repo-only owned samples |
| `validation/corpus/metal2vulkan-ledger.jsonl` | yes | **no** | Translate hashes only |
| `validation/corpus/metal2vulkan-ledger-*.jsonl` | optional | **no** | Execution plans + deterministic input/output digests + `output_b64`; hash-identified, no AIR bodies |
| `validation/corpus/local/**` | **no** | **no** | Private shard JSONL (`air_ll` + optional AIR blob) |
| `scripts/`, `docs/` | yes | **no** | Root package `exclude` |
| `m2v-old`, `*.spv` dumps | **no** | **no** | Local experiment debris |

**Rule of thumb:** if it contains Apple or app shader bodies, it stays private. Hash-identified
translate and execution ledgers may be public **in the git repo** for reproducibility, but still are
**not** shipped inside the `metal2vulkan` crates.io tarball (validation is a separate, non-published
package).

Verify before a release:

```sh
cargo package --list
# must not list validation/, docs/, scripts/, private corpus rows, *.air, *.spv, etc.
cargo publish --dry-run
```

---

## Tool map (where to click)

| Tool | Path | One-liner |
|---|---|---|
| Unit tests | `cargo test -p metal2vulkan` | Synthetic regressions |
| Harvest | `corpus-harvest` | System metallib → shard JSONL |
| `corpus-mint` | validation bin | Append new translate `air_sha256` pins |
| `corpus-remint` | validation bin | Re-translate existing pins (`--failed-only`) |
| `corpus-why` | validation bin | One `air_sha256` → translate error string |
| `corpus-run-metal` | validation bin | Metal oracle → plan + golden `output_b64` + digests |
| `corpus-run-vulkan` | validation bin | Linux Vulkan vs metal golden |
| `corpus-run-moltenvk` | validation bin | MoltenVK vs metal golden |
| `corpus-triage` | validation bin | Summarize ledgers and print failure rerun commands |
| Byte A/B | `scripts/metal2vulkan-ab/` | Two binaries, one sample |
| AIR carve | `scripts/mtlb-extract/` | Low-level MTLB → `.air` |
| SPIR-V delta class | `spirv_delta` validation bin | Classify id/order vs semantic delta |
| Pipeline probe | `spirv_pipeline_probe` | Module + pipeline create |
| Oracle / runner | `validation` lib | Execution experiments |

---

## Interpreting outcomes

| Symptom | Likely story | Next move |
|---|---|---|
| Unit red | You broke a structural case | Fix or update the synthetic test honestly |
| A/B `DIFFERS` | SPIR-V bytes moved | Revert, or re-mint if intentional |
| A/B `NEW-ONLY` / `OLD-ONLY` | One side FALLBACKs | Investigate emit path / stage detect |
| Corpus mass FAIL | Broad emit regression | Bisect; do not name-key harvest cases |
| Corpus sparse FAIL | Narrow class | Reduce to synthetic `.ll`, land unit test |
| spirv-val fail after translate | Invalid module | Prefer honest FALLBACK over wrong SPIR-V |

---

## Design rules that keep the ladder honest

1. **Structure over names** — never branch the translator on a harvested function name.
2. **Synthetic first for CI** — private harvest is for *you*; public regression is for *everyone*.
3. **Honest FALLBACK** — unsupported is better than wrong-but-valid SPIR-V.
4. **Re-mint is a product decision** — changing thousands of hashes without explanation is how
   regressions hide.
5. **Match the ladder to the claim** — “no SPIR-V change on my machine” is only as strong as the
   sample you actually ran.

---

## Minimal cheat sheet

```sh
# Always
cargo test -p metal2vulkan -- --test-threads=1

# Before a scary refactor
cargo build --release --bin metal2vulkan && cp target/release/metal2vulkan ./m2v-old

# After
cargo build --release --bin metal2vulkan
scripts/metal2vulkan-ab/metal2vulkan-ab.sh --old ./m2v-old

# Once per machine (macOS): fat private sample
cargo run -p metal2vulkan-validation --release --bin corpus-harvest
cargo run -p metal2vulkan-validation --release --bin corpus-mint
# optional execution goldens (see plan.md / validation/README.md)
# cargo run -p metal2vulkan-validation --release --bin corpus-run-metal
# cargo run -p metal2vulkan-validation --release --bin corpus-run-vulkan
```

Corpus ledger CLIs (full flag list and pipeline): [`validation/README.md`](../validation/README.md),
[`validation/corpus/README.md`](../validation/corpus/README.md), [`plan.md`](../plan.md).

When the ladder is green at the height you chose, ship the refactor. When it is not, the failure
is a gift: it named a class before a user did.
