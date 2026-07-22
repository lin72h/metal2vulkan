#!/usr/bin/env bash
#
# Public / local AIR→SPIR-V hash drift ledger.
#
# Records sha256(source) + sha256(spv) so you can detect translator byte-drift across versions
# without committing proprietary metallibs or AIR. Synthetic fixtures ship in-tree; private captures
# stay under validation/corpus/local/ (gitignored) and may be banked as kind=private rows.
#
# Implementation: metal2vulkan_drift.py (translates with workers = CPU count × 2 by default).
# Each translate is killed after 120s (override with --timeout / METAL2VULKAN_DRIFT_TIMEOUT)
# and banked as status=timeout.
#
# Usage:
#   metal2vulkan-drift.sh check [--ledger PATH] [--bin PATH]
#   metal2vulkan-drift.sh mint  --public [--ledger PATH] [--bin PATH]
#   metal2vulkan-drift.sh mint  --local  [--ledger PATH] [--bin PATH]
#   metal2vulkan-drift.sh mint  --file PATH [--label NAME] [--kind synthetic|private] ...
#
# Exit 0 when check finds no drift among present sources; 1 on drift / asymmetric status.
#
set -euo pipefail

DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
CRATE_DIR="$(cd -- "$DIR/../.." && pwd)"
PY="$DIR/metal2vulkan_drift.py"
LEDGER_DEFAULT="$CRATE_DIR/validation/corpus/drift-ledger.jsonl"
PUBLIC_DIR="$CRATE_DIR/validation/fixtures/public"
LOCAL_AIR_DIR="$CRATE_DIR/validation/corpus/local/air"

CMD=""
LEDGER="$LEDGER_DEFAULT"
BIN=""
KIND=""
LABEL=""
FILES=()
PUBLIC=0
LOCAL=0
QUIET=0
STAGE="auto"
JOBS=""
TIMEOUT=""

usage() {
  cat <<USAGE
usage: $(basename "$0") check|mint [options]

  check   re-translate present sources; compare to ledger
  mint    bank sha256(source) + sha256(spv) into the ledger

Options:
  --ledger PATH     ledger JSONL (default: validation/corpus/drift-ledger.jsonl)
  --bin PATH        metal2vulkan binary (default: build --release)
  --file PATH       extra source (repeatable)
  --label NAME      label for --file mint
  --kind KIND       synthetic|private (for --file mint)
  --public          include validation/fixtures/public
  --local           include validation/corpus/local/air
  --stage STAGE     CLI stage (default: auto)
  --jobs N          parallel translate workers (default: CPU cores × 2)
  --timeout SECS    kill a hung translate after SECS (default: 120)
  --quiet           less per-case output
  -h, --help

Env: METAL2VULKAN_DRIFT_JOBS, METAL2VULKAN_DRIFT_TIMEOUT.
USAGE
}

while [ $# -gt 0 ]; do
  case "$1" in
    check|mint) CMD="$1"; shift ;;
    --ledger) LEDGER="$2"; shift 2 ;;
    --bin) BIN="$2"; shift 2 ;;
    --file) FILES+=("$2"); shift 2 ;;
    --label) LABEL="$2"; shift 2 ;;
    --kind) KIND="$2"; shift 2 ;;
    --public) PUBLIC=1; shift ;;
    --local) LOCAL=1; shift ;;
    --stage) STAGE="$2"; shift 2 ;;
    --jobs) JOBS="$2"; shift 2 ;;
    --timeout) TIMEOUT="$2"; shift 2 ;;
    --quiet) QUIET=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown arg: $1" >&2; usage >&2; exit 64 ;;
  esac
done

[ -n "$CMD" ] || { echo "need check|mint" >&2; usage >&2; exit 64; }
[ -f "$PY" ] || { echo "missing $PY" >&2; exit 1; }

if [ -z "$BIN" ]; then
  echo "# building release metal2vulkan..." >&2
  ( cd "$CRATE_DIR" && cargo build --release --bin metal2vulkan >&2 )
  BIN="$CRATE_DIR/target/release/metal2vulkan"
fi
[ -x "$BIN" ] || { echo "not an executable: $BIN" >&2; exit 64; }

export METAL2VULKAN_DRIFT_BIN="$BIN"
export METAL2VULKAN_DRIFT_STAGE="$STAGE"
export METAL2VULKAN_DRIFT_LEDGER="$LEDGER"
export METAL2VULKAN_DRIFT_PUBLIC_DIR="$PUBLIC_DIR"
export METAL2VULKAN_DRIFT_LOCAL_AIR="$LOCAL_AIR_DIR"
export METAL2VULKAN_DRIFT_QUIET="$QUIET"
export METAL2VULKAN_DRIFT_KIND="$KIND"
export METAL2VULKAN_DRIFT_LABEL="$LABEL"
export METAL2VULKAN_DRIFT_PUBLIC="$PUBLIC"
export METAL2VULKAN_DRIFT_LOCAL="$LOCAL"
export METAL2VULKAN_DRIFT_CMD="$CMD"
if [ -n "$JOBS" ]; then
  export METAL2VULKAN_DRIFT_JOBS="$JOBS"
fi
if [ -n "$TIMEOUT" ]; then
  export METAL2VULKAN_DRIFT_TIMEOUT="$TIMEOUT"
fi

FILE_LIST="$(mktemp)"
trap 'rm -f "$FILE_LIST"' EXIT
if [ "${#FILES[@]}" -gt 0 ]; then
  printf '%s\n' "${FILES[@]}" >"$FILE_LIST"
fi
export METAL2VULKAN_DRIFT_FILE_LIST="$FILE_LIST"

exec python3 "$PY"
