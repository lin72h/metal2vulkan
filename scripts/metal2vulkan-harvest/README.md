# metal2vulkan-harvest — system metallib → local corpus

**Single Python entrypoint:** `metal2vulkan-harvest.py`  
(`metal2vulkan-harvest.sh` is a thin `exec` wrapper for the same CLI.)

Harvests macOS `.metallib` files into the **gitignored** private corpus tree:

```text
validation/corpus/local/
  metallib/<lib_sha256>/     # source-path.txt (+ optional library.metallib)
  air/<air_sha256>.air       # carved AIR bitcode
  air/<air_sha256>.ll        # llvm-dis text
  shards/shard_NN.jsonl      # rows for metallib_gen_tests / run_corpus_case
  ledger/                    # metallibs.tsv, summary, skips
```

Nothing under `local/` is committed. Output is Apple/app-owned content for **private**
validation only.

**Size cap:** any carved AIR blob **larger than 512 KiB** is skipped (logged as
`air_too_large` in `ledger/skipped.tsv`). Override with `--max-air-bytes` or
`METAL2VULKAN_HARVEST_MAX_AIR_BYTES`.

## Requirements

- **macOS** for system enumeration (`/System/Library`, `/Library`, `/Applications`)
- `python3`
- `llvm-dis` (Homebrew LLVM, or `METAL2VULKAN_LLVM_DIS` / `--llvm-dis`)
- Repo script [`scripts/mtlb-extract/mtlb_extract.py`](../mtlb-extract/) (loaded in-process)

On Linux you can still run with explicit paths: `--metallib /path/to/lib.metallib`.

## Usage

```sh
# All prioritized system metallibs (no default cap)
scripts/metal2vulkan-harvest/metal2vulkan-harvest.py

# Optional batching
scripts/metal2vulkan-harvest/metal2vulkan-harvest.py --limit 25
scripts/metal2vulkan-harvest/metal2vulkan-harvest.py --offset 25 --limit 25

# Applications only
scripts/metal2vulkan-harvest/metal2vulkan-harvest.py --start-set apps

# One library (any host that has the file + llvm-dis)
scripts/metal2vulkan-harvest/metal2vulkan-harvest.py \
  --metallib /path/to/Something.metallib

# Also hardlink/copy the metallib binary under local/metallib/<sha>/
scripts/metal2vulkan-harvest/metal2vulkan-harvest.py --copy-metallib --limit 5
```

### Generate and run tests

```sh
cargo run -p metal2vulkan-validation --bin metallib_gen_tests
cargo test -p metal2vulkan-validation --test corpus_00 -- --test-threads=1
```

### Flags

| Flag | Meaning |
|---|---|
| `--out DIR` | Output root (default `validation/corpus/local`) |
| `--limit N` | Optional max metallibs (default: **unlimited**) |
| `--offset N` | Skip first N after prioritization |
| `--start-set system\|apps\|all` | Scan roots |
| `--include-apps` | With `system`, also scan `/Applications` |
| `--copy-metallib` | Store the binary under `metallib/<sha>/library.metallib` |
| `--no-shards` | Skip JSONL emission |
| `--no-ll` | Skip `llvm-dis` (implies `--no-shards`) |
| `--metallib PATH` | Explicit input (repeatable; skips find) |
| `--max-air-bytes N` | Size cap (default 524288) |
| `--llvm-dis PATH` | llvm-dis binary |

## What gets into shards

Metallibs embed many bitcode wrappers that are **not** shader entries (`air.*` libcalls,
AGX helpers, etc.). Those have **no** `!air.kernel` / `!air.vertex` / `!air.fragment`
metadata. By default they are **dropped** from shards (`dropped_helpers` in the summary),
not emitted as `#[ignore]` stubs.

Each remaining unique `.ll` (deduped by SHA-256 of the text) becomes one JSONL row:

- `synth: true` — real stage metadata (Kernel / Vertex / Fragment) → translate smoke
- `synth: false` + `ignore_reason` — rare cases (e.g. module ctor / stitching helper)

`--keep-helpers` re-includes non-shader blobs as ignored stubs. `--shards-only` reclassifies
existing `air/*.ll` without re-scanning metallibs.

Binding synthesis is **minimal** (empty buffers/textures, dummy output). The local runner is a
**translate smoke**, not the monorepo Metal-oracle byte gate. Re-harvest is additive for
content-addressed `air/`; shards are **rewritten** from the current `air/*.ll` set each emit.

## Relation to other tools

| Tool | Role |
|---|---|
| This harvest | System → private `local/` (bodies) |
| [`metal2vulkan-drift`](../metal2vulkan-drift/) | Hash-only public pins |
| [`metal2vulkan-ab`](../metal2vulkan-ab/) | Byte A/B on `local/air` + public fixtures |
| `mtlb-extract` | Low-level AIR carver (imported by harvest) |
