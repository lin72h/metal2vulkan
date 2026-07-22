#!/usr/bin/env python3
"""Harvest system (and optional app) .metallib files into the gitignored local corpus.

  validation/corpus/local/{metallib,air,shards,ledger}/

Does NOT commit anything. Output is private Apple/app-owned content — keep it local.

Pipeline per metallib:
  1. enumerate / prioritize paths
  2. record source path under metallib/<lib_sha256>/
  3. carve AIR via scripts/mtlb-extract/mtlb_extract.py (in-process)
  4. drop any carved AIR larger than 512 KiB
  5. llvm-dis each blob → air/<air_sha256>.{air,ll}
  6. emit shard_NN.jsonl rows for metallib_gen_tests / run_corpus_case

Usage:
  scripts/metal2vulkan-harvest/metal2vulkan-harvest.py
  scripts/metal2vulkan-harvest/metal2vulkan-harvest.py --limit 10 --start-set system
  scripts/metal2vulkan-harvest/metal2vulkan-harvest.py --metallib /path/to/one.metallib
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import hashlib
import importlib.util
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
DEFAULT_OUT = ROOT / "validation" / "corpus" / "local"
EXTRACT_PY = ROOT / "scripts" / "mtlb-extract" / "mtlb_extract.py"

DEFAULT_MAX_AIR_BYTES = 1024 * 1024
SHARD_COUNT = 16

DEFINE_RE = re.compile(r"define\s+(?:[^@]*\s+)?@([^\s(]+)\s*\(")
# !N = !{ptr @entry_name, ...}  (stage root body after resolve)
META_PTR_NAME_RE = re.compile(r"ptr\s+@([A-Za-z0-9_$.]+)")
KERNEL_META = "!air.kernel ="
VERTEX_META = "!air.vertex ="
FRAGMENT_META = "!air.fragment ="


# --- hashing / tools ------------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_mtlb_extract():
    if not EXTRACT_PY.is_file():
        raise SystemExit(f"missing mtlb_extract: {EXTRACT_PY}")
    spec = importlib.util.spec_from_file_location("mtlb_extract", EXTRACT_PY)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load {EXTRACT_PY}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def resolve_llvm_dis(explicit: str | None) -> str:
    candidates: list[str] = []
    if explicit:
        candidates.append(explicit)
    env = os.environ.get("METAL2VULKAN_LLVM_DIS")
    if env:
        candidates.append(env)
    candidates.extend(
        [
            "/opt/homebrew/opt/llvm/bin/llvm-dis",
            "/usr/local/opt/llvm/bin/llvm-dis",
            "llvm-dis",
        ]
    )
    for c in candidates:
        if not c:
            continue
        p = Path(c)
        if p.is_file() and os.access(p, os.X_OK):
            return str(p)
        found = shutil.which(c)
        if found:
            return found
    raise SystemExit(
        "llvm-dis not found (install LLVM or pass --llvm-dis / set METAL2VULKAN_LLVM_DIS)"
    )


# --- enumeration ----------------------------------------------------------------------------------


def scan_roots(start_set: str, include_apps: bool) -> list[Path]:
    roots: list[Path] = []
    if start_set in ("system", "all"):
        roots.extend([Path("/System/Library"), Path("/Library")])
        if include_apps and start_set == "system":
            roots.append(Path("/Applications"))
    if start_set in ("apps", "all"):
        roots.append(Path("/Applications"))
    found: list[Path] = []
    for root in roots:
        if not root.is_dir():
            continue
        try:
            for dirpath, _dirnames, filenames in os.walk(root, followlinks=False):
                for name in filenames:
                    if name.endswith(".metallib"):
                        found.append(Path(dirpath) / name)
        except PermissionError:
            continue
    # unique by resolved path when possible
    uniq: dict[str, Path] = {}
    for p in found:
        try:
            key = str(p.resolve())
        except OSError:
            key = str(p)
        uniq[key] = p
    return list(uniq.values())


def priority_key(path: Path) -> tuple[str, str]:
    s = str(path)
    if any(
        x in s
        for x in (
            "WindowServer",
            "SkyLight",
            "CoreDisplay",
            "CoreAnimation",
            "QuartzCore",
        )
    ):
        band = "00"
    elif any(x in s for x in ("AppleMetalOpenGLRenderer", "PixelConverter", "OpenGL")):
        band = "01"
    elif "AGXMetal" in s:
        band = "02"
    elif s.startswith("/System/Library"):
        band = "03"
    elif s.startswith("/Library"):
        band = "04"
    elif s.startswith("/Applications"):
        band = "05"
    else:
        band = "99"
    return band, s


def select_libs(
    start_set: str,
    include_apps: bool,
    limit: int | None,
    offset: int,
    extra: list[Path],
) -> tuple[list[Path], list[Path]]:
    """Return (batch, all_sorted).

    `limit is None` means no cap (everything after `offset`).
    """
    if extra:
        for p in extra:
            if not p.is_file():
                raise SystemExit(f"not a file: {p}")
        return list(extra), list(extra)
    all_libs = sorted(scan_roots(start_set, include_apps), key=priority_key)
    if offset < 0:
        raise SystemExit(f"bad --offset: {offset}")
    if limit is None:
        batch = all_libs[offset:]
    else:
        if limit < 0:
            raise SystemExit(f"bad --limit: {limit}")
        batch = all_libs[offset : offset + limit]
    return batch, all_libs


# --- store helpers --------------------------------------------------------------------------------


def store_metallib_binary(src: Path, dest: Path) -> None:
    if dest.exists():
        return
    try:
        os.link(src, dest)
    except OSError:
        shutil.copy2(src, dest)


def cleanup_tmp(out: Path) -> None:
    """Remove scratch `out/tmp` after a successful harvest (extract dirs + llvm-dis logs)."""
    tmp = out / "tmp"
    if tmp.is_dir():
        shutil.rmtree(tmp, ignore_errors=True)
        print(f"# cleaned {tmp}")


# --- shard emit (inlined from former harvest_emit_shards.py) --------------------------------------


def first_define_name(ll: str) -> str:
    m = DEFINE_RE.search(ll)
    return m.group(1) if m else "unknown"


def stage_of(ll: str) -> str | None:
    if KERNEL_META in ll:
        return "Kernel"
    if VERTEX_META in ll:
        return "Vertex"
    if FRAGMENT_META in ll:
        return "Fragment"
    return None


def stage_root_id(ll: str, stage_key: str) -> int | None:
    """Parse `!air.<stage> = !{!N}` → N."""
    m = re.search(rf"!air\.{stage_key}\s*=\s*!\{{!(\d+)\}}", ll)
    return int(m.group(1)) if m else None


def meta_node_body(ll: str, node_id: int) -> str | None:
    m = re.search(rf"!{node_id}\s*=\s*!{{([^}}]*)}}", ll)
    return m.group(1) if m else None


def entry_name_from_stage_meta(ll: str, stage: str) -> str | None:
    """Entry name from `!air.<stage>` root `ptr @name` (matches metal2vulkan::meta)."""
    key = {"Kernel": "kernel", "Vertex": "vertex", "Fragment": "fragment"}.get(stage)
    if not key:
        return None
    root = stage_root_id(ll, key)
    if root is None:
        return None
    body = meta_node_body(ll, root)
    if not body:
        return None
    m = META_PTR_NAME_RE.search(body)
    return m.group(1) if m else None


def classify(ll: str) -> tuple[bool, str | None, str, str] | None:
    """Classify one AIR module for the local translate-smoke corpus.

    Returns:
      (synth, ignore_reason, stage, fn_name) — emit a shard row
      None — drop from shards (not a shader entry; e.g. carved stdlib/libcall blob)

    Local harvest only needs translate smokes: Kernel / Vertex / Fragment with real
    `!air.*` stage metadata are synth=true. Blobs without stage metadata are almost
    always metallib-embedded `air.*` / `agx.*` helpers, not dispatchable entries —
    they flood the suite if left as #[ignore] stubs, so they are dropped by default.
    """
    stage = stage_of(ll)
    if stage is None:
        # Not a Metal shader entry (stdlib intrinsic, AGX helper, partial blob, …).
        return None

    fn = entry_name_from_stage_meta(ll, stage) or first_define_name(ll)

    # Module ctors / C++ stitching helpers carry kernel meta but are not dispatchable.
    if fn.startswith("_GLOBAL__sub_I_") or "_stitching_traits_impl" in fn:
        return (
            False,
            "metallib: kernel entry is a module initializer or template helper, not dispatchable",
            stage,
            fn,
        )

    # Everything with real stage metadata is runnable as a translate smoke.
    # (Full monorepo oracle/binding synthesis is a separate, stricter path.)
    return True, None, stage, fn


def load_source_metallib(air_dir: Path, stem: str) -> str:
    p = air_dir / f"{stem}.source-metallib.txt"
    if p.is_file():
        return p.read_text(encoding="utf-8", errors="replace").strip()
    return "unknown"


def emit_shards(
    air_dir: Path,
    shards_dir: Path,
    ledger_dir: Path | None,
    max_air_bytes: int,
    *,
    keep_helpers: bool = False,
) -> int:
    ll_files = sorted(air_dir.glob("*.ll"))
    if not ll_files:
        print(f"emit_shards: no *.ll under {air_dir}", file=sys.stderr)
        return 1

    shards: list[list[str]] = [[] for _ in range(SHARD_COUNT)]
    seen: set[str] = set()
    n_synth = n_stub = n_dup = n_skip_large = n_drop_helper = 0
    stage_counts: dict[str, int] = {}

    for ll_path in ll_files:
        stem = ll_path.stem
        air_path = air_dir / f"{stem}.air"
        if air_path.is_file():
            air_size = air_path.stat().st_size
        else:
            air_size = ll_path.stat().st_size
        if air_size > max_air_bytes:
            n_skip_large += 1
            continue

        ll_text = ll_path.read_text(encoding="utf-8", errors="replace")
        full = sha256_bytes(ll_text.encode("utf-8"))
        if full in seen:
            n_dup += 1
            continue
        seen.add(full)

        classified = classify(ll_text)
        if classified is None:
            if keep_helpers:
                synth, reason, stage, fn = (
                    False,
                    "metallib: no recognized !air stage metadata (helper/libcall blob)",
                    "Kernel",
                    first_define_name(ll_text),
                )
            else:
                n_drop_helper += 1
                continue
        else:
            synth, reason, stage, fn = classified

        if synth:
            n_synth += 1
        else:
            n_stub += 1
        stage_counts[stage] = stage_counts.get(stage, 0) + 1

        if air_path.is_file():
            blob_b64 = base64.b64encode(air_path.read_bytes()).decode("ascii")
        else:
            blob_b64 = ""

        short = full[:8]
        lib = load_source_metallib(air_dir, stem)
        row = {
            "id": f"metallib/{fn}/{short}",
            "hash": short,
            "lib": lib,
            "fn": fn,
            "stage": stage,
            "synth": synth,
            "ignore_reason": reason,
            "buffers": [],
            "textures": [],
            "output": {"kind": "buffer", "index": 0, "format": "RawBytes", "len": 16},
            "dispatch_grid": [64, 1, 1],
            "dispatch_tg": [64, 1, 1],
            "blob_b64": blob_b64,
            "air_ll": ll_text,
            "air_sha256": full,
        }
        line = json.dumps(row, ensure_ascii=False, separators=(",", ":"))
        shards[int(full[:16], 16) % SHARD_COUNT].append(line)

    shards_dir.mkdir(parents=True, exist_ok=True)
    for i, lines in enumerate(shards):
        path = shards_dir / f"shard_{i:02d}.jsonl"
        path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        print(f"  shard_{i:02d}: {len(lines)} rows → {path}")

    if ledger_dir is not None:
        ledger_dir.mkdir(parents=True, exist_ok=True)
        (ledger_dir / "shards_summary.tsv").write_text(
            "emitted\tsynth\tstub\tdup_skipped\tskipped_large\tdropped_helpers\tmax_air_bytes\n"
            f"{n_synth + n_stub}\t{n_synth}\t{n_stub}\t{n_dup}\t{n_skip_large}\t"
            f"{n_drop_helper}\t{max_air_bytes}\n",
            encoding="utf-8",
        )

    stages_s = " ".join(f"{k}={v}" for k, v in sorted(stage_counts.items()))
    print(
        f"emit_shards: emitted={n_synth + n_stub} synth={n_synth} stub={n_stub} "
        f"dropped_helpers={n_drop_helper} dup_skipped={n_dup} "
        f"skipped_large={n_skip_large} max_air_bytes={max_air_bytes}"
    )
    if stages_s:
        print(f"emit_shards: stages {stages_s}")
    return 0


# --- harvest --------------------------------------------------------------------------------------


def harvest_one(
    lib: Path,
    out: Path,
    *,
    copy_metallib: bool,
    max_air_bytes: int,
    no_ll: bool,
    llvm_dis: str | None,
    extract_mod,
    skipped: list[str],
) -> tuple[bool, int, int, int, str, int]:
    """Process one metallib.

    Returns (ok, carved, kept, skipped_large, lib_sha, metallib_size).
    """
    lib_sha = sha256_file(lib)
    size = lib.stat().st_size
    dest_dir = out / "metallib" / lib_sha
    dest_dir.mkdir(parents=True, exist_ok=True)
    (dest_dir / "source-path.txt").write_text(f"{lib}\n", encoding="utf-8")
    (dest_dir / "sha256.txt").write_text(f"{lib_sha}\n", encoding="utf-8")

    if copy_metallib:
        store_metallib_binary(lib, dest_dir / "library.metallib")

    extract_dir = out / "tmp" / f"extract-{lib_sha}"
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True)

    with (dest_dir / "extract.log").open("w", encoding="utf-8") as log:
        try:
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                found = extract_mod.extract(str(lib), str(extract_dir))
            log.write(f"# extract returned {len(found)} records\n")
        except Exception as e:  # noqa: BLE001 — ledger extract_failed
            log.write(f"extract error: {e}\n")
            skipped.append(f"{lib}\textract_failed\t{e}")
            print(f"  FAIL extract {lib}", file=sys.stderr)
            shutil.rmtree(extract_dir, ignore_errors=True)
            return False, 0, 0, 0, lib_sha, size

    carved = kept = skip_large = 0
    for air in sorted(extract_dir.glob("*.air")):
        carved += 1
        air_size = air.stat().st_size
        if air_size > max_air_bytes:
            skip_large += 1
            skipped.append(f"{lib}\t{air_size}\tair_too_large\tmax={max_air_bytes}")
            continue
        air_sha = sha256_file(air)
        out_air = out / "air" / f"{air_sha}.air"
        if not out_air.is_file():
            shutil.copy2(air, out_air)
        (out / "air" / f"{air_sha}.source-metallib.txt").write_text(
            f"{lib}\n", encoding="utf-8"
        )
        (out / "air" / f"{air_sha}.lib-sha256.txt").write_text(
            f"{lib_sha}\n", encoding="utf-8"
        )

        if not no_ll:
            assert llvm_dis is not None
            out_ll = out / "air" / f"{air_sha}.ll"
            if not out_ll.is_file():
                log_path = out / "tmp" / f"llvm-dis-{air_sha}.log"
                proc = subprocess.run(
                    [llvm_dis, str(out_air), "-o", str(out_ll)],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                log_path.write_text(
                    (proc.stdout or "") + (proc.stderr or ""), encoding="utf-8"
                )
                if proc.returncode != 0:
                    print(f"  FAIL llvm-dis {air_sha} (from {lib})", file=sys.stderr)
                    skipped.append(f"{lib}\t{air_sha}\tllvm-dis_failed")
                    if out_ll.is_file():
                        out_ll.unlink()
                    continue
        kept += 1

    shutil.rmtree(extract_dir, ignore_errors=True)
    print(f"  ok  kept={kept} carved={carved}  {lib}")
    return True, carved, kept, skip_large, lib_sha, size


def main(argv: list[str] | None = None) -> int:
    env_limit = os.environ.get("METAL2VULKAN_HARVEST_LIMIT")
    env_offset = os.environ.get("METAL2VULKAN_HARVEST_OFFSET")
    env_set = os.environ.get("METAL2VULKAN_HARVEST_START_SET")
    env_max = os.environ.get("METAL2VULKAN_HARVEST_MAX_AIR_BYTES")

    ap = argparse.ArgumentParser(
        description="Harvest macOS .metallib files into gitignored validation/corpus/local/"
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"output root (default: {DEFAULT_OUT})",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="max metallibs this run (default: no limit; all after --offset)",
    )
    ap.add_argument(
        "--offset",
        type=int,
        default=int(env_offset) if env_offset else 0,
        help="skip first N metallibs after prioritization",
    )
    ap.add_argument(
        "--start-set",
        choices=("system", "apps", "all"),
        default=env_set if env_set in ("system", "apps", "all") else "system",
    )
    ap.add_argument(
        "--include-apps",
        action="store_true",
        help="with --start-set system, also scan /Applications",
    )
    ap.add_argument(
        "--copy-metallib",
        action="store_true",
        help="hardlink/copy the .metallib under metallib/<sha>/library.metallib",
    )
    ap.add_argument("--no-shards", action="store_true", help="skip JSONL shard emission")
    ap.add_argument(
        "--no-ll",
        action="store_true",
        help="skip llvm-dis (implies --no-shards)",
    )
    ap.add_argument(
        "--metallib",
        action="append",
        default=[],
        type=Path,
        help="explicit metallib (repeatable; skips system enumeration)",
    )
    ap.add_argument("--llvm-dis", default=None, help="llvm-dis binary path")
    ap.add_argument(
        "--max-air-bytes",
        type=int,
        default=int(env_max) if env_max else DEFAULT_MAX_AIR_BYTES,
        help=f"skip AIR larger than this (default {DEFAULT_MAX_AIR_BYTES})",
    )
    ap.add_argument(
        "--shards-only",
        action="store_true",
        help="only re-emit shards from existing air/*.ll (no metallib scan)",
    )
    ap.add_argument(
        "--keep-helpers",
        action="store_true",
        help="include non-shader AIR (no !air.* stage meta) as ignored stubs",
    )
    args = ap.parse_args(argv)

    # Env only applies when --limit was not passed on the CLI.
    if args.limit is None and env_limit is not None and env_limit != "":
        try:
            args.limit = int(env_limit)
        except ValueError as e:
            raise SystemExit(f"bad METAL2VULKAN_HARVEST_LIMIT={env_limit!r}: {e}") from e

    if args.no_ll:
        args.no_shards = True

    out: Path = args.out
    for sub in ("metallib", "air", "shards", "ledger", "tmp"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    ledger = out / "ledger"

    if args.shards_only:
        print(f"# shards-only from {out / 'air'} → {out / 'shards'}")
        rc = emit_shards(
            out / "air",
            out / "shards",
            ledger,
            args.max_air_bytes,
            keep_helpers=args.keep_helpers,
        )
        if rc == 0:
            print("# next: cargo run -p metal2vulkan-validation --bin metallib_gen_tests")
        return rc

    if platform.system() != "Darwin" and not args.metallib:
        print(
            "metal2vulkan-harvest: system enumeration requires macOS; "
            "pass --metallib PATH on other hosts",
            file=sys.stderr,
        )
        return 1

    batch, all_libs = select_libs(
        args.start_set,
        args.include_apps,
        args.limit,
        args.offset,
        list(args.metallib),
    )
    (ledger / "all-metallibs.txt").write_text(
        "\n".join(str(p) for p in all_libs) + ("\n" if all_libs else ""),
        encoding="utf-8",
    )
    (ledger / "input-metallibs.txt").write_text(
        "\n".join(str(p) for p in batch) + ("\n" if batch else ""),
        encoding="utf-8",
    )

    limit_s = "none" if args.limit is None else str(args.limit)
    print(
        f"# harvest out={out} start_set={args.start_set} "
        f"total_found={len(all_libs)} batch={len(batch)} "
        f"offset={args.offset} limit={limit_s}"
    )
    print(f"# max AIR size: {args.max_air_bytes} bytes (skip above this)")

    llvm_dis = None
    if not args.no_ll:
        llvm_dis = resolve_llvm_dis(args.llvm_dis)
        print(f"# llvm-dis: {llvm_dis}")

    extract_mod = load_mtlb_extract()
    skipped: list[str] = []
    n_ok = n_fail = n_air = n_skip_large = 0
    metallib_rows = ["path\tlib_sha256\tsize\tair_carved\tair_kept"]

    for lib in batch:
        ok, carved, kept, skip_large, lib_sha, size = harvest_one(
            lib,
            out,
            copy_metallib=args.copy_metallib,
            max_air_bytes=args.max_air_bytes,
            no_ll=args.no_ll,
            llvm_dis=llvm_dis,
            extract_mod=extract_mod,
            skipped=skipped,
        )
        if not ok:
            n_fail += 1
            continue
        n_ok += 1
        n_air += kept
        n_skip_large += skip_large
        metallib_rows.append(f"{lib}\t{lib_sha}\t{size}\t{carved}\t{kept}")

    (ledger / "skipped.tsv").write_text(
        "\n".join(skipped) + ("\n" if skipped else ""), encoding="utf-8"
    )
    (ledger / "metallibs.tsv").write_text(
        "\n".join(metallib_rows) + "\n", encoding="utf-8"
    )
    (ledger / "summary.tsv").write_text(
        "ok_libs\tfail_libs\tair_kept\tair_skipped_large\ttotal_found\tbatch\tmax_air_bytes\n"
        f"{n_ok}\t{n_fail}\t{n_air}\t{n_skip_large}\t{len(all_libs)}\t{len(batch)}\t{args.max_air_bytes}\n",
        encoding="utf-8",
    )
    print(
        f"# summary: ok_libs={n_ok} fail_libs={n_fail} "
        f"air_kept={n_air} air_skipped_large={n_skip_large}"
    )

    if not args.no_shards:
        print(f"# emitting shards from {out / 'air'} → {out / 'shards'}")
        rc = emit_shards(
            out / "air",
            out / "shards",
            ledger,
            args.max_air_bytes,
            keep_helpers=args.keep_helpers,
        )
        if rc != 0:
            return rc
        print("# next: cargo run -p metal2vulkan-validation --bin metallib_gen_tests")

    success = n_fail == 0 or n_ok > 0
    if success:
        cleanup_tmp(out)
    print(f"# RESULT: harvest complete → {out}")
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
