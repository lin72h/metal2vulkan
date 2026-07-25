# Optional private metallib corpus + metal2vulkan ledgers

This directory holds **JSONL ledgers** (plans, digests, and execution `output_b64` payloads) and a
gitignored private tree for harvested AIR shard rows. Full execution design: repo-root
[`plan.md`](../../plan.md).

| Path | Tracked? | Purpose |
|---|---|---|
| `metal2vulkan-ledger.jsonl` | **yes** | Translate pins (`corpus-mint` / `corpus-remint`) — hashes only |
| `metal2vulkan-ledger-metal.jsonl` | optional | Metal oracle plan + deterministic digests + golden `output_b64` (`corpus-run-metal`) |
| `metal2vulkan-ledger-vulkan.jsonl` | optional | Linux Vulkan vs metal + deterministic digests + candidate `output_b64` (`corpus-run-vulkan`) |
| `metal2vulkan-ledger-moltenvk.jsonl` | optional | MoltenVK vs metal + deterministic digests + candidate `output_b64` (`corpus-run-moltenvk`) |
| `local/` | **gitignored** | Private shard JSONL rows (`air_ll` + optional AIR `blob_b64`) |
| `../fixtures/public/` | **yes** | Tiny owned synthetic `.ll` samples for CI / demos |

CI and a clean clone never need `local/`. Default `cargo test` stays synthetic-only.

## Layout for a private corpus

```text
validation/corpus/local/
  shards/
    shard_00.jsonl    # … shard_15.jsonl  (embedded air_ll + metadata)
```

`corpus-harvest` removes old `local/{air,metallib,ledger,tmp}` directories when it runs. JSONL rows
are the private source of truth.

JSONL row shape:

```json
{
  "id": "…",
  "hash": "0002abf7",
  "shard": "shard_03.jsonl",
  "label": "local/<air_sha256>.ll",
  "lib": "/System/Library/…/default.metallib",
  "lib_sha256": "…",
  "fn": "kernel_name",
  "stage": "Kernel",
  "air_ll": "… sanitized LLVM text …",
  "blob_b64": "… optional …"
}
```

Harvest keeps only AIR with real `!air.kernel` / `!air.vertex` / `!air.fragment` metadata.
Helpers without stage meta are dropped.

## Harvest Shards

```sh
cargo run -p metal2vulkan-validation --release --bin corpus-harvest
```

Set `METAL2VULKAN_CORPUS_DIR` if shards live outside `validation/corpus/local`. Ledger tools
(`corpus-mint`, `corpus-remint`, `corpus-run-*`, `corpus-why`) resolve harvested cases directly
from shard JSONL.

`corpus-harvest --metallib PATH` works with explicit metallibs. Without `--metallib`, system
enumeration is macOS-only. It keeps no `.air` / `.ll` mirror; `llvm-dis` temp files are removed as
soon as each row is built.

Do **not** commit metallibs, shard rows, or private AIR bodies.

---

## Corpus CLIs

All from the repo root:

```sh
cargo run -p metal2vulkan-validation --release --bin <name> -- [flags]
```

### `corpus-mint` — additive translate ledger

Scans **public fixtures** + **local shards**, unique by content `air_sha256`. Translates only hashes
**not** already in `metal2vulkan-ledger.jsonl` and **appends** rows.

```sh
cargo run -p metal2vulkan-validation --release --bin corpus-mint -- --dry-run
cargo run -p metal2vulkan-validation --release --bin corpus-mint
```

Flags: `--jobs N`, `--quiet`, `--ledger PATH`, `--dry-run`.

### `corpus-remint` — refresh existing translate rows

Re-translates rows already in the ledger (source in public fixtures or local shards) and rewrites
them.

```sh
cargo run -p metal2vulkan-validation --release --bin corpus-remint
cargo run -p metal2vulkan-validation --release --bin corpus-remint -- --failed-only
cargo run -p metal2vulkan-validation --release --bin corpus-remint -- --failed-only --status fallback --limit 50
cargo run -p metal2vulkan-validation --release --bin corpus-remint -- --status timeout --skip 50 --limit 50 --case-timeout-secs 60
```

Flags: `--jobs N`, `--quiet`, `--ledger PATH`, `--dry-run`, `--failed-only`, `--status STATUS`,
`--contains TEXT`, `--skip N`, `--limit N`, `--case-timeout-secs N` (`0` keeps the legacy
in-process path).

### `corpus-why` — diagnose one translate failure

```sh
cargo run -p metal2vulkan-validation --release --bin corpus-why -- <air_sha256>
```

Locates the source, runs stage auto + translate, prints `why: …` (or ok + `spv_sha256`). No write.

### `corpus-run-metal` — Metal oracle (macOS)

For each translate row with `status=ok` **or** `fallback` missing from
`metal2vulkan-ledger-metal.jsonl` (or selected by `--force`, `--failed-only`, `--status`,
`--bucket`, or `--contains`): infer or reuse plan, seed non-zero inputs, run Metal, append a
per-run delta row, then merge once into the ledger. The final row carries plan + full golden
`output_b64` + digests. Translator FALLBACK does **not** skip Metal (oracle is independent of
SPIR-V emit).

```sh
cargo run -p metal2vulkan-validation --release --bin corpus-run-metal -- --dry-run
cargo run -p metal2vulkan-validation --release --bin corpus-run-metal
cargo run -p metal2vulkan-validation --release --bin corpus-run-metal -- --air-sha256 <hex>
```

Needs real AIR bitcode for many kernels (`blob_b64` in harvested shard rows); hand-written public
`.ll` alone may fail assembly.

**Infinite-loop safety (structural).** A committed Metal command buffer cannot be cancelled — an
unbounded compute loop pins the GPU until the machine is rebooted (killing the CPU worker does not
stop it), and no seed choice can guarantee an arbitrary kernel halts (halting problem). So the
oracle bounds GPU work **before** submitting, classifying every case (`oracle_macos` →
[`loop_budget`](../src/loop_budget.rs)):

- **loop-free** → provably bounded; runs unchanged; `compare=full`.
- **has loops, instrumentable** → a per-thread back-edge budget is injected into the AIR so every
  loop is forced to exit after a fixed cap; the golden is byte-identical for any run that stays
  under budget. These rows are `compare=none`: the vulkan/moltenvk candidates apply the same
  loop-budget transform to the harvested LL before translating and executing SPIR-V.
- **cannot instrument + verify** (switch back-edges, unparseable control flow, a loop that calls a
  loopy callee, or IR `metal-as` rejects) → `status=quarantine`, never dispatched.

### `corpus-run-vulkan` — Linux Vulkan candidate

Only translate **`status=ok`** rows (need SPIR-V). Reuses the **metal** plan/golden when present.
Writes compare status (+ optional tolerance fields) on each `metal2vulkan-ledger-vulkan.jsonl` line.

```sh
cargo run -p metal2vulkan-validation --release --bin corpus-run-vulkan -- --dry-run
cargo run -p metal2vulkan-validation --release --bin corpus-run-vulkan
```

Without a metal golden for a hash → candidate status `missing`.
`status=tolerance` is an accepted candidate success; it records the policy and observed margins on
the candidate row.
`status=smoke` is also an accepted candidate success; it means the candidate executed and captured
bytes, but the Metal row was `compare=none` and is not a semantic golden.

### `corpus-run-moltenvk` — MoltenVK candidate (macOS)

Same as vulkan (`status=ok` only), separate ledger `metal2vulkan-ledger-moltenvk.jsonl`. Configure
the Vulkan loader / `VK_ICD_FILENAMES` for MoltenVK when needed.

```sh
cargo run -p metal2vulkan-validation --release --bin corpus-run-moltenvk
```

### Shared runner flags

`--force` (re-run even if tech row exists), `--failed-only` (re-run existing non-success tech rows),
`--status STATUS` (existing tech rows with that status), `--bucket TEXT` (existing tech rows whose
failure bucket contains that text), `--contains TEXT` (existing tech rows whose label, error,
status, bucket, or hash contains that text), `--quiet`, `--jobs N` (default **min(CPU cores, 4)**
parallel workers).
Per-case worker timeout defaults to **60s** and can be overridden with
`METAL2VULKAN_CORPUS_TIMEOUT_SECS=N`; there is no timeout CLI flag. On expiry, the runner SIGKILLs
the worker and banks `status=timeout`.
The timeout frees the CPU worker but **cannot** cancel an in-flight GPU kernel (reboot is the only
recovery) — infinite-loop safety comes from the pre-submission loop-budget guard above, not from
this timeout; with that guard bounded GPU work finishes well under a second, so a case still running
at the bound means a CPU-side tool hang, not a GPU loop.
`--air-sha256 HEX`, `--ledger-dir DIR`, `--dry-run`.
Writes one append-only delta per runner execution, then rewrites the target ledger once with
`air_sha256` dedupe (last row for a hash wins).

### `corpus-triage` — failure queue view

Read-only helper for agent-style iteration over existing ledgers:

```sh
cargo run -p metal2vulkan-validation --release --bin corpus-triage
cargo run -p metal2vulkan-validation --release --bin corpus-triage -- \
  --backend moltenvk --status fallback --contains pipeline --commands
```

It prints status counts, non-success buckets sorted by count, selected rows, and optional
`corpus-why` / `corpus-run-* --air-sha256 ... --force --jobs 1` commands. Candidate rows print
`metal-if-needed` unless the candidate status is `missing`. Use `--limit 0` for summary only.

### Pipeline

```text
corpus-harvest → corpus-mint → metal2vulkan-ledger.jsonl
       → corpus-run-metal → metal2vulkan-ledger-metal.jsonl
       → corpus-run-vulkan / corpus-run-moltenvk → compare ledgers
```

Run gate: source in public fixtures or local shards; metal accepts translate `ok|fallback`,
vulkan/moltenvk require translate `ok`. Eligibility is **not** driven by a separate skip file.
`--force` ignores existing backend rows; `--failed-only`, `--status`, `--bucket`, and `--contains`
select existing backend rows.

---

## Translate ledger schema (`metal2vulkan-ledger.jsonl`)

| Field | Meaning |
|---|---|
| `air_sha256` | SHA-256 of source file bytes (`.ll` / `.air`) |
| `shard` | Private shard file name when row came from `corpus/local/shards` |
| `spv_sha256` | SHA-256 of SPIR-V when `status=ok` |
| `status` | `ok` \| `fallback` \| `timeout` \| … |
| `stage` | CLI stage used (`auto` recommended) |
| `label` | Short non-proprietary tag |
| `kind` | `synthetic` (public) or `private` (local) |

### Execution ledgers (metal / vulkan / moltenvk)

See [`plan.md`](../../plan.md): metal row owns harness **plan** + golden `output_b64`; candidate rows
record `status` (`ok` / `tolerance` / `smoke` / `failure` / `missing` / `fallback` /
`quarantine`) and may embed `tolerance` / `observed` on the same line. A metal row may also be
`quarantine` (unbounded / uninstrumentable loop — not dispatched); candidates record
`status=quarantine` for that hash.

## Byte A/B

[`scripts/metal2vulkan-ab/`](../../scripts/metal2vulkan-ab/) walks public fixtures and
`local/shards` when present for binary-to-binary SPIR-V compares (no execution goldens required).
