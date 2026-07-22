# metal2vulkan-drift — AIR / SPIR-V hash ledger

Track **byte-level translator drift** across versions by publishing only hashes:

| Field | Meaning |
|---|---|
| `air_sha256` | SHA-256 of the source `.ll` / `.air` file bytes |
| `spv_sha256` | SHA-256 of emitted SPIR-V (when `status=ok`) |
| `status` | `ok`, `fallback`, or `timeout` (killed after wall limit) |
| `stage` | CLI stage used when banking (`auto` recommended) |
| `label` | Short non-proprietary tag |
| `kind` | `synthetic` (CI / in-tree) or `private` (local-only) |

No shader bodies are stored. This complements [`metal2vulkan-ab`](../metal2vulkan-ab/), which
diffs two binaries on a live sample; the ledger pins “what HEAD used to emit for this AIR hash.”

## Why hashes (and what they are not)

- **Good for:** catching accidental emitter churn; proving a refactor is byte-stable on a banked set;
  sharing a *fingerprint* of private captures without redistributing Apple/app AIR.
- **Not a semantic oracle:** same SPIR-V hash ⇒ same bytes, not “correct Metal.” Use the validation
  package / monorepo corpus for execution goldens.
- **Intentional changes** require re-`mint` and a reviewed ledger diff — same discipline as updating
  any golden file.
- **CI** can only verify rows whose source is present. Commit `kind=synthetic` rows with fixtures
  under `validation/fixtures/public/`. `kind=private` rows are skipped when the AIR is absent.

## Usage

```sh
# verify public fixtures (and any present local air) against the committed ledger
./metal2vulkan-drift.sh check

# re-bank public fixtures after an intentional SPIR-V change
./metal2vulkan-drift.sh mint --public

# bank gitignored local harvest under validation/corpus/local/air/
./metal2vulkan-drift.sh mint --local

# single file
./metal2vulkan-drift.sh mint --file path/to/case.ll --label my_case --kind synthetic
```

Flags: `--bin PATH`, `--ledger PATH`, `--stage auto|kernel|…`, `--quiet`,
`--jobs N` (parallel translate workers; default **CPU cores × 2**, or
`METAL2VULKAN_DRIFT_JOBS`), `--timeout SECS` (per-source kill; default **120**,
or `METAL2VULKAN_DRIFT_TIMEOUT`). Hung translates are SIGKILL'd and banked as
`status=timeout` (no `spv_sha256`).

Implementation: thin shell wrapper + [`metal2vulkan_drift.py`](metal2vulkan_drift.py)
(ThreadPoolExecutor over metal2vulkan subprocesses).

## Layout

- Ledger (committed): `validation/corpus/drift-ledger.jsonl`
- Public sources: `validation/fixtures/public/**/*.{ll,air}`
- Private sources: `validation/corpus/local/air/**/*.{ll,air}` (gitignored)

When a directory has both `stem.ll` and `stem.air` (as harvest does), only the **`.ll`** is
used so each module is translated once. Lone `.air` files are still included.

See [`validation/corpus/README.md`](../../validation/corpus/README.md).
