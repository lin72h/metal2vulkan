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
| [`../validation/corpus/`](../validation/corpus/) | Drift ledger + private corpus layout |

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
2. **No accidental SPIR-V churn** on banked cases — drift check clean, or A/B clean, *or* an
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
# Scan prioritized system metallibs → validation/corpus/local/{air,shards,ledger}/
# No default --limit: processes the full prioritized list (long first run).
scripts/metal2vulkan-harvest/metal2vulkan-harvest.py

# Optional: generate cargo tests that seek into the JSONL shards
cargo run -p metal2vulkan-validation --bin metallib_gen_tests

# Optional: bank private hashes into the committed ledger as kind=private
# (fingerprints only — still no shader bodies in git)
scripts/metal2vulkan-drift/metal2vulkan-drift.sh mint --local
```

What harvest keeps:

- Real **Kernel / Vertex / Fragment** entries with `!air.*` stage metadata → translate smokes
- Drops metallib-embedded **helpers** without stage meta (stdlib / libcall noise)
- Skips AIR larger than the configured size cap
- Content-addresses under `local/air/` so re-runs are additive for files

Details: [`scripts/metal2vulkan-harvest/`](../scripts/metal2vulkan-harvest/),
[`validation/corpus/README.md`](../validation/corpus/README.md).

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
scripts/metal2vulkan-drift/metal2vulkan-drift.sh mint --public
# if you maintain private pins:
scripts/metal2vulkan-drift/metal2vulkan-drift.sh mint --local
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
- `validation/corpus/local/air/**` (private harvest)

Paired harvest files use **only `.ll`** when both `stem.ll` and `stem.air` exist, so you do not
pay double translate cost.

Exit 0 means no `DIFFERS` / asymmetric produce-vs-fail on that sample. `BOTH-FAIL` is not a
regression (both sides FALLBACK).

### 5. Hash ledger check (no need to keep two binaries forever)

```sh
scripts/metal2vulkan-drift/metal2vulkan-drift.sh check
```

- **`kind: synthetic`** rows with missing sources → fail (public fixtures must stay present).
- **`kind: private`** rows with missing AIR → skip (other developers without your harvest are fine).
- Present sources whose SPIR-V hash moved → **DRIFT** (either fix the refactor or re-mint).

After an intentional emit change:

```sh
scripts/metal2vulkan-drift/metal2vulkan-drift.sh mint --public
# and if you use private pins:
scripts/metal2vulkan-drift/metal2vulkan-drift.sh mint --local
# review the ledger diff; commit only hash rows you mean to publish
```

### 6. Private corpus translate smokes (if harvested)

```sh
cargo run -p metal2vulkan-validation --bin metallib_gen_tests   # if stubs missing/stale
cargo test -p metal2vulkan-validation --test corpus_00 -- --test-threads=1
# more shards as needed: corpus_01 … corpus_15
```

Each case is a **translate smoke** (and optional `spirv-val`), not a Metal-vs-Vulkan byte oracle.
That is still enormously useful: “did I break half of QuartzCore?” shows up as a wall of red
without needing an ICD.

Reclassify without re-carving:

```sh
scripts/metal2vulkan-harvest/metal2vulkan-harvest.py --shards-only
```

### 7. Optional execution (when translate-green is not enough)

On macOS, validation’s Metal oracle and MoltenVK-backed runner can compare execution results for
hand-built or corpus-shaped cases. On Linux, use a native Vulkan ICD. This path is the richest and
the most environment-specific — treat it as **layer E**, not the daily default.

See [`validation/README.md`](../validation/README.md) and the `oracle_macos` / `runner_linux`
modules.

---

## Recipes by change type

### “I’m restructuring the structurizer / CFG / emitter”

1. Synthetic tests (**A**).
2. Save `m2v-old`, implement, A/B (**B**).
3. `drift check` on public + any local pins (**C**).
4. If you harvest: corpus smokes (**D**).
5. Re-mint only if you *meant* SPIR-V to move, and say so in the commit.

### “I fixed one FALLBACK class”

1. Add a **synthetic** regression that would have failed before (so CI owns the class).
2. Confirm the real metallib case (if you have it under `local/air`) translates.
3. Optionally bank its hashes so the fix cannot silently reverse.

### “I only touched docs / scripts / validation helpers”

Stay on **A** for the product crate; run validation package tests if you edited that crate.

### “I need a new public sample everyone can run”

1. Author a **small owned** `.ll` under `validation/fixtures/public/` (no third-party capture).
2. `mint --public` and commit the fixture + ledger row.
3. Never paste system-metallib AIR into the public tree.

### “Harvest is huge / slow”

- First full harvest is the expensive one; `air/` is content-addressed, re-runs skip existing
  blobs.
- Use `--limit` / `--offset` only when batching deliberately.
- Prefer `--shards-only` when only classification changed.
- Drift/A/B prefer `.ll` over paired `.air` — one translate per stem.

---

## What lives in git vs on your disk vs crates.io

| Artifact | Git? | crates.io? | Notes |
|---|---|---|---|
| Unit / synthetic tests | yes | yes (product crate) | CI backbone |
| `src/`, `tests/` (product) | yes | yes | Published package |
| `validation/` workspace member | yes | **no** | `publish = false` + root `exclude` |
| `validation/fixtures/public/*.ll` | yes | **no** | Repo-only owned samples |
| `validation/corpus/drift-ledger.jsonl` | yes | **no** | Hashes only, not in product crate |
| `validation/corpus/tolerances.jsonl` | yes | **no** | Repo-only |
| `validation/corpus/broken.jsonl` | yes | **no** | Repo-only |
| `validation/corpus/local/**` | **no** | **no** | Metallibs, AIR, shards |
| `validation/tests/corpus_*.rs` | **no** | **no** | Generated local stubs |
| `scripts/`, `docs/` | yes | **no** | Root package `exclude` |
| `m2v-old`, `*.spv` dumps | **no** | **no** | Local experiment debris |

**Rule of thumb:** if it contains Apple or app shader bodies, it stays private. If it is only a
SHA-256 and a reason string, it may be public **in the git repo**, but still is **not** shipped
inside the `metal2vulkan` crates.io tarball (validation is a separate, non-published package).

Verify before a release:

```sh
cargo package --list
# must not list validation/, corpus/, *.air, harvest scripts, etc.
cargo publish --dry-run
```

---

## Tool map (where to click)

| Tool | Path | One-liner |
|---|---|---|
| Unit tests | `cargo test -p metal2vulkan` | Synthetic regressions |
| Harvest | `scripts/metal2vulkan-harvest/` | System metallib → `local/` |
| Gen corpus tests | `metallib_gen_tests` | JSONL → gitignored `corpus_NN.rs` |
| Drift ledger | `scripts/metal2vulkan-drift/` | AIR/SPIR-V hash mint + check |
| Byte A/B | `scripts/metal2vulkan-ab/` | Two binaries, one sample |
| AIR carve | `scripts/mtlb-extract/` | Low-level MTLB → `.air` |
| SPIR-V delta class | `validation` `spirv_delta` | Classify id/order vs semantic delta |
| Pipeline probe | `spirv_pipeline_probe` | Module + pipeline create |
| Oracle / runner | `validation` lib | Execution experiments |

---

## Interpreting outcomes

| Symptom | Likely story | Next move |
|---|---|---|
| Unit red | You broke a structural case | Fix or update the synthetic test honestly |
| A/B `DIFFERS` | SPIR-V bytes moved | Revert, or re-mint if intentional |
| A/B `NEW-ONLY` / `OLD-ONLY` | One side FALLBACKs | Investigate emit path / stage detect |
| Drift `DRIFT` | Ledger out of date vs HEAD | Same as A/B |
| Drift `MISSING` synthetic | Public fixture deleted | Restore fixture or drop ledger row |
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
scripts/metal2vulkan-drift/metal2vulkan-drift.sh check

# Once per machine (macOS): fat private sample
scripts/metal2vulkan-harvest/metal2vulkan-harvest.py
cargo run -p metal2vulkan-validation --bin metallib_gen_tests
scripts/metal2vulkan-drift/metal2vulkan-drift.sh mint --local
```

When the ladder is green at the height you chose, ship the refactor. When it is not, the failure
is a gift: it named a class before a user did.
