#!/usr/bin/env bash
#
# metal2vulkan byte-A/B harness.
#
# Translate a pinned sample set with two `metal2vulkan` binaries and diff the emitted SPIR-V
# byte-for-byte, reporting per-case identical / differs / asymmetric. This is a cheap gate for
# behavior-preserving refactors: build the crate before a change (old) and after (new), run this,
# and demand zero diffs on the sample.
#
# Sample set:
#   * every `*.air` / `*.ll` under `tests/fixtures/**`, when present
#   * every path passed after `--` on the command line
#
# The `metal2vulkan` CLI writes `out.spv` *before* running spirv-val, so a case that FALLBACKs on
# spirv-val still yields comparable bytes; a case whose translate() errors produces no file and is
# reported as a translate failure (not a diff).
#
# Exit status: 0 when no case DIFFERS and no case is asymmetric (one binary produced output, the
# other did not); nonzero otherwise. A HEAD-vs-HEAD self-test therefore exits 0.
#
# Usage:
#   metal2vulkan-ab.sh [--old PATH] [--new PATH] [--quiet] [-- file.ll ...]
#
#   --old PATH    "before" binary. Default: build `--release` from this checkout.
#   --new PATH    "after"  binary. Default: same as --old (a self-test: everything must be IDENTICAL).
#   --quiet       print only the summary, not per-case lines.
#
set -euo pipefail

CRATE_DIR="$(cd -- "$(dirname -- "$0")/../.." && pwd)"

OLD_BIN=""
NEW_BIN=""
QUIET=0
EXTRA_SRCS=()

while [ $# -gt 0 ]; do
  case "$1" in
    --old) OLD_BIN="$2"; shift 2 ;;
    --new) NEW_BIN="$2"; shift 2 ;;
    --quiet) QUIET=1; shift ;;
    -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
    --) shift; EXTRA_SRCS+=("$@"); break ;;
    *) echo "unknown arg: $1" >&2; exit 64 ;;
  esac
done

if [ -z "$OLD_BIN" ]; then
  echo "# building release metal2vulkan (old = new = HEAD self-test)..." >&2
  ( cd "$CRATE_DIR" && cargo build --release --bin metal2vulkan >&2 )
  OLD_BIN="$CRATE_DIR/target/release/metal2vulkan"
fi
[ -z "$NEW_BIN" ] && NEW_BIN="$OLD_BIN"

for b in "$OLD_BIN" "$NEW_BIN"; do
  [ -x "$b" ] || { echo "not an executable: $b" >&2; exit 64; }
done

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# translate SRC with BIN into DST.spv; echo "ok" if the file was produced, "fail" otherwise.
translate() {
  local bin="$1" src="$2" dst="$3"
  rm -f "$dst"
  # --stage auto is the CLI default; spirv-val failure is fine, the .spv is already on disk.
  "$bin" "$src" "$dst" --stage auto >/dev/null 2>&1 || true
  if [ -s "$dst" ]; then echo ok; else echo fail; fi
}

n_identical=0; n_differs=0; n_asym=0; n_bothfail=0
report() {
  local name="$1" verdict="$2"
  case "$verdict" in
    IDENTICAL) n_identical=$((n_identical+1)) ;;
    DIFFERS)   n_differs=$((n_differs+1)) ;;
    OLD-ONLY|NEW-ONLY) n_asym=$((n_asym+1)) ;;
    BOTH-FAIL) n_bothfail=$((n_bothfail+1)) ;;
  esac
  if [ "$QUIET" -eq 0 ] || [ "$verdict" = DIFFERS ] || [ "$verdict" = OLD-ONLY ] || [ "$verdict" = NEW-ONLY ]; then
    printf '  %-10s %s\n' "$verdict" "$name"
  fi
}

compare_case() {
  local name="$1" src="$2"
  local o="$WORK/old.spv" n="$WORK/new.spv"
  local ostat nstat
  ostat="$(translate "$OLD_BIN" "$src" "$o")"
  nstat="$(translate "$NEW_BIN" "$src" "$n")"
  if [ "$ostat" = ok ] && [ "$nstat" = ok ]; then
    if cmp -s "$o" "$n"; then report "$name" IDENTICAL; else report "$name" DIFFERS; fi
  elif [ "$ostat" = ok ]; then report "$name" OLD-ONLY
  elif [ "$nstat" = ok ]; then report "$name" NEW-ONLY
  else report "$name" BOTH-FAIL
  fi
}

echo "# old: $OLD_BIN"
echo "# new: $NEW_BIN"

# List .ll/.air under root, preferring stem.ll when stem.air is also present (harvest pairs).
list_sources() {
  local root="$1"
  # Print paths: for each (dir,stem) keep .ll over .air.
  find "$root" \( -name '*.air' -o -name '*.ll' \) -type f | sort | python3 -c '
import sys
from pathlib import Path
best = {}
for line in sys.stdin:
    p = Path(line.strip())
    if not p.is_file():
        continue
    key = (str(p.parent.resolve()), p.stem)
    prev = best.get(key)
    if prev is None or (p.suffix == ".ll" and Path(prev).suffix == ".air"):
        best[key] = str(p)
for path in sorted(best.values()):
    print(path)
'
}

scan_tree() {
  local root="$1" tag="$2"
  if [ -d "$root" ]; then
    echo "# $tag:"
    while IFS= read -r src; do
      [ -n "$src" ] || continue
      compare_case "${tag}:${src#"$root"/}" "$src"
    done < <(list_sources "$root")
  else
    echo "# $tag: skipped (not present)"
  fi
}

# Sample set (all optional except explicit -- paths):
#   1) tests/fixtures/**                 — private local fixtures
#   2) validation/fixtures/public/**     — committed synthetic samples
#   3) validation/corpus/local/air/**    — gitignored mined AIR
scan_tree "$CRATE_DIR/tests/fixtures" "fixture"
scan_tree "$CRATE_DIR/validation/fixtures/public" "public"
scan_tree "$CRATE_DIR/validation/corpus/local/air" "local-air"

if [ "${#EXTRA_SRCS[@]}" -gt 0 ]; then
  echo "# extra inputs:"
  for src in "${EXTRA_SRCS[@]}"; do
    compare_case "input:$(basename "$src")" "$src"
  done
fi

echo "# summary: identical=$n_identical differs=$n_differs asymmetric=$n_asym both-fail=$n_bothfail"
if [ "$n_differs" -gt 0 ] || [ "$n_asym" -gt 0 ]; then
  echo "# RESULT: REGRESSION (byte divergence detected)"
  exit 1
fi
echo "# RESULT: clean (no byte divergence)"
