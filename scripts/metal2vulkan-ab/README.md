# metal2vulkan-ab — byte-A/B translation harness

Translates a pinned sample set with two `metal2vulkan` binaries and diffs the emitted SPIR-V
**byte-for-byte**, so a behavior-preserving refactor can be proven not to move any output on the
sample before more expensive checks.

## Why byte comparison is meaningful

The emitter's output is id-canonicalized by `passes::canonicalize_ids`, so equal semantics ⇒ equal
bytes for the vast majority of cases. If a case stays nondeterministic across runs, fall back to a
semantic or validator gate rather than trusting the byte diff alone.

The CLI writes `out.spv` *before* running spirv-val, so a case that FALLBACKs at the spirv-val step
still produces comparable bytes. A case whose `translate()` errors writes no file and is reported as
a translate failure, not a diff.

## Usage

```sh
# self-test: HEAD vs HEAD (exit 0 when every case is IDENTICAL or BOTH-FAIL)
./metal2vulkan-ab.sh

# real A/B: build the crate before your change and after, then diff
git stash    # or check out the parent commit
cargo build --release --bin metal2vulkan
cp target/release/metal2vulkan ./m2v-old
git stash pop
./metal2vulkan-ab.sh --old ./m2v-old

# optional extra shaders after --
./metal2vulkan-ab.sh --old ./m2v-old -- path/to/case.ll
```

Flags:

- `--old PATH` — "before" binary. Default: build `--release` from this checkout.
- `--new PATH` — "after" binary. Default: same as `--old` (a self-test).
- `--quiet` — print only the summary and any regressions.
- `-- file…` — additional `.ll` / `.air` inputs to include in the sample.

Because `--new` defaults to `--old`, an explicit A/B run is usually: pass the pre-change binary as
`--old` and let `--new` build the current tree, **or** pass both explicitly.

## Sample set

1. Every `*.air` / `*.ll` under `tests/fixtures/**` when that directory exists (gitignored).
2. Every `*.air` / `*.ll` under `validation/fixtures/public/**` (committed synthetic samples).
3. Every `air_ll` row under `validation/corpus/local/shards/shard_*.jsonl` (gitignored private harvest).
4. Any paths listed after `--`.

Shard rows materialize temporary `.ll` files under the script work directory and are removed when
the run exits.

This public tree does not ship third-party captured shaders. For cross-version hash pins (without
keeping two binaries around), bank with `corpus-mint` / `corpus-remint` (and optional
`corpus-run-*` execution ledgers) — see [`validation/corpus/README.md`](../../validation/corpus/README.md)
and [`plan.md`](../../plan.md).

## Output

Per-case verdicts (`IDENTICAL` / `DIFFERS` / `OLD-ONLY` / `NEW-ONLY` / `BOTH-FAIL`) plus a summary
line. Exit status is `0` only when there are **no** `DIFFERS` and no asymmetric cases; any byte
divergence exits `1`. `BOTH-FAIL` (both binaries FALLBACK identically) is not a regression.
