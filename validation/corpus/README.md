# Optional private metallib corpus + public drift ledger

This directory supports two complementary workflows:

| Path | Tracked? | Purpose |
|---|---|---|
| `drift-ledger.jsonl` | **yes** (hashes only) | Cross-version byte-drift checks without shipping shaders |
| `tolerances.jsonl` | **yes** (hashes + tolerances) | Per-AIR numeric compare overrides (execution gates) |
| `broken.jsonl` | **yes** (hashes + reasons) | Not-applicable / known-broken skips (no shader bodies) |
| `local/` | **gitignored** | Your metallibs, AIR, JSONL shards, working trees |
| `../tests/corpus_*.rs` | **gitignored** | Generated nextest stubs that seek into local shards |
| `../fixtures/public/` | **yes** | Tiny owned synthetic `.ll` samples for CI / demos |

CI and a clean clone never need `local/` or `corpus_*.rs`. Default `cargo test` stays
synthetic-only.

## Layout for a private corpus

```text
validation/corpus/local/
  metallib/           # optional raw .metallib inputs
  air/                # optional extracted .air / .ll files
  shards/
    shard_00.jsonl    # … shard_15.jsonl  (embedded air_ll + metadata)
```

JSONL row shape (compatible with the monorepo collect format, subset is fine):

```json
{
  "id": "…",
  "hash": "0002abf7",
  "fn": "kernel_name",
  "stage": "Kernel",
  "synth": true,
  "ignore_reason": null,
  "air_ll": "… sanitized LLVM text …",
  "blob_b64": "… optional …"
}
```

Only `synth: true` rows are executed by the slim runner. Non-synth rows should be
`#[ignore]` in generated stubs (see gen path below).

Harvest classification (after `--shards-only` fix): only AIR with real `!air.kernel` /
`!air.vertex` / `!air.fragment` metadata is emitted. Embedded stdlib/libcall blobs without
stage meta are dropped (they are not shader entries). Vertex and Fragment are synth=true
for **translate smokes** (not full monorepo oracle binding).

## Generate `corpus_NN.rs` stubs (optional)

With shards present:

```sh
cargo run -p metal2vulkan-validation --bin metallib_gen_tests
```

Writes `validation/tests/corpus_00.rs` … `corpus_15.rs` (gitignored). Each test is:

```rust
fn c_<hash>() {
    metal2vulkan_validation::run_corpus_case("00", byte_offset, byte_len);
}
```

Run:

```sh
cargo test -p metal2vulkan-validation --test corpus_00 -- --test-threads=1
# include ignored stubs only when intentional:
cargo test -p metal2vulkan-validation --test corpus_00 -- --test-threads=1 --ignored
```

`run_corpus_case` here is a **translate smoke** (plus optional `spirv-val` and drift-ledger
check). It is **not** the monorepo’s full Metal-oracle byte gate. Point
`METAL2VULKAN_CORPUS_DIR` at another root if your shards live outside the default
`validation/corpus/local`.

## Harvest inputs

**Preferred (macOS):** run the system harvest into the gitignored local tree:

```sh
scripts/metal2vulkan-harvest/metal2vulkan-harvest.py
# optional batching: --limit 25 / --offset 25 --limit 25
cargo run -p metal2vulkan-validation --bin metallib_gen_tests
```

See [`scripts/metal2vulkan-harvest/README.md`](../../scripts/metal2vulkan-harvest/README.md).

Manual path:

1. Drop `.metallib` files under `local/metallib/` (or keep them elsewhere).
2. Carve AIR with [`scripts/mtlb-extract/`](../../scripts/mtlb-extract/).
3. Disassemble with `llvm-dis` into `.ll` under `local/air/`, **or** re-run harvest
   (which emits `shards/shard_NN.jsonl`).
4. Generate stubs with `metallib_gen_tests`.

Do **not** commit metallibs, AIR, shards, goldens, or generated `corpus_*.rs`. Those are
Apple- or app-owned content and stay private.

## Committed sidecars (JSONL, hashes only)

Both files are one JSON object per line. Blank lines and `#` comments are ignored.
Keys are always **`air_sha256`** = SHA-256 of the AIR source bytes (`.ll` text or `.air` blob),
not the short monorepo corpus hash.

### `tolerances.jsonl`

Numeric compare overrides for execution / byte gates (parent analogue: `tolerances.jsonl`).

```json
{"air_sha256":"<64-hex>","label":"optional","tolerance":{"kind":"Ulp","max_ulp":1,"reason":"…"}}
```

`tolerance.kind` is one of: `Exact`, `Abs` (`max_abs`), `Ulp` (`max_ulp`), `RawF16Ulp`,
`RawU8Ulp`, `AbsAndUlp` (`max_abs` + `max_ulp`). Non-`Exact` kinds require a `reason`.

### `broken.jsonl`

Cases that must not run (parent analogue: `not_applicable.jsonl` / harness gaps).

```json
{"air_sha256":"<64-hex>","reason":"…","label":"optional","category":"translator_gap"}
```

`category` (optional, default `not_applicable`): `not_applicable`, `harness_gap`,
`translator_gap`, `oracle_hazard`, `other`.

Loaders: `metal2vulkan_validation::{load_tolerances,load_broken,tolerance_for_air_sha256,broken_for_air_sha256}`.

## Public drift ledger (hashes only)

`drift-ledger.jsonl` records:

- `air_sha256` — SHA-256 of the **source file bytes** (`.ll` / `.air`)
- `spv_sha256` — SHA-256 of translator output, or omitted when status is `fallback`
- `status` — `ok` | `fallback`
- `stage` — CLI stage used (`auto` recommended)
- `label` — short non-proprietary tag (e.g. fixture path basename)
- `kind` — `synthetic` (in-tree / public) or `private` (local-only verify)

Hashes are not shader bodies. Publishing `air_sha256` of a private capture is a
fingerprint, not redistribution of the AIR. Prefer `kind: synthetic` rows for anything
CI must verify; `kind: private` rows are skipped when the matching source is absent.

### Mint / check

```sh
# self-check public fixtures against the committed ledger
scripts/metal2vulkan-drift/metal2vulkan-drift.sh check

# re-bank after an intentional emitter change (rewrites matching labels)
scripts/metal2vulkan-drift/metal2vulkan-drift.sh mint --public

# bank hashes for every local air under corpus/local/air (does not commit)
scripts/metal2vulkan-drift/metal2vulkan-drift.sh mint --local
```

See [`scripts/metal2vulkan-drift/README.md`](../../scripts/metal2vulkan-drift/README.md).

## Byte A/B

[`scripts/metal2vulkan-ab/`](../../scripts/metal2vulkan-ab/) also walks
`validation/corpus/local/air/**/*.{ll,air}` and `validation/fixtures/public/**/*.{ll,air}`
when present, so a local harvest participates in binary A/B without being in git.
