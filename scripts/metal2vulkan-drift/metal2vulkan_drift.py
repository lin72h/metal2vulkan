#!/usr/bin/env python3
"""AIR → SPIR-V hash drift ledger (mint / check).

Invoked by metal2vulkan-drift.sh. Configuration via env / CLI-equivalent env vars
set by the shell wrapper (METAL2VULKAN_DRIFT_*).

Translates many sources concurrently (default workers = CPU count × 2).
Ledger rows are unique on air_sha256 (last write wins).
"""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def env_bool(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v not in ("", "0", "false", "False", "no", "NO")


def default_workers() -> int:
    raw = os.environ.get("METAL2VULKAN_DRIFT_JOBS")
    if raw:
        n = int(raw)
        if n < 1:
            raise SystemExit(f"bad METAL2VULKAN_DRIFT_JOBS={raw!r}")
        return n
    cores = os.cpu_count() or 4
    return max(1, cores * 2)


def default_timeout_secs() -> float:
    """Per-source translate timeout; hung jobs become status=timeout."""
    raw = os.environ.get("METAL2VULKAN_DRIFT_TIMEOUT", "120")
    try:
        t = float(raw)
    except ValueError as e:
        raise SystemExit(f"bad METAL2VULKAN_DRIFT_TIMEOUT={raw!r}") from e
    if t <= 0:
        raise SystemExit(f"METAL2VULKAN_DRIFT_TIMEOUT must be > 0, got {t}")
    return t


cmd = os.environ["METAL2VULKAN_DRIFT_CMD"]
bin_path = Path(os.environ["METAL2VULKAN_DRIFT_BIN"])
stage = os.environ.get("METAL2VULKAN_DRIFT_STAGE", "auto")
ledger = Path(os.environ["METAL2VULKAN_DRIFT_LEDGER"])
public_dir = Path(os.environ["METAL2VULKAN_DRIFT_PUBLIC_DIR"])
local_air = Path(os.environ["METAL2VULKAN_DRIFT_LOCAL_AIR"])
quiet = env_bool("METAL2VULKAN_DRIFT_QUIET")
kind_override = os.environ.get("METAL2VULKAN_DRIFT_KIND") or None
label_override = os.environ.get("METAL2VULKAN_DRIFT_LABEL") or None
want_public = env_bool("METAL2VULKAN_DRIFT_PUBLIC")
want_local = env_bool("METAL2VULKAN_DRIFT_LOCAL")
file_list = Path(os.environ["METAL2VULKAN_DRIFT_FILE_LIST"])
workers = default_workers()
timeout_secs = default_timeout_secs()


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def translate_hash(src: Path) -> tuple[str, str | None]:
    """Run metal2vulkan on src; return (status, spv_sha256|None).

    Status is ``ok`` (SPIR-V written), ``fallback`` (failed/empty), or ``timeout``
    (subprocess exceeded ``timeout_secs`` and was killed).
    """
    with tempfile.TemporaryDirectory(prefix="m2v-drift-") as td:
        spv = Path(td) / "out.spv"
        try:
            # New session so a timeout can kill the whole process group if the
            # translator ever spawns helpers.
            proc = subprocess.Popen(
                [str(bin_path), str(src), str(spv), "--stage", stage],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            try:
                proc.communicate(timeout=timeout_secs)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                return "timeout", None
        except OSError:
            return "fallback", None
        if spv.is_file() and spv.stat().st_size > 0:
            return "ok", sha256_file(spv)
        return "fallback", None


def gather_ll_air(root: Path) -> list[Path]:
    """Collect shader sources under root.

    Harvest writes paired ``stem.air`` + ``stem.ll`` (same module). Prefer ``.ll`` when
    both exist so mint/check does not translate each module twice under two different
    content hashes. Lone ``.air`` files are still included.
    """
    if not root.is_dir():
        return []
    by_stem: dict[tuple[Path, str], Path] = {}
    for pattern in ("*.ll", "*.air"):
        for p in root.rglob(pattern):
            if not p.is_file():
                continue
            key = (p.parent.resolve(), p.stem)
            prev = by_stem.get(key)
            if prev is None:
                by_stem[key] = p
            elif p.suffix == ".ll" and prev.suffix == ".air":
                by_stem[key] = p
    return sorted(by_stem.values(), key=lambda p: str(p))


def collect_sources() -> list[Path]:
    out: list[Path] = []
    if file_list.is_file() and file_list.stat().st_size:
        for line in file_list.read_text().splitlines():
            line = line.strip()
            if line:
                out.append(Path(line))
        return _unique_paths(out)
    if want_public and public_dir.is_dir():
        out.extend(gather_ll_air(public_dir))
    if want_local and local_air.is_dir():
        out.extend(gather_ll_air(local_air))
    if cmd == "check" and not want_public and not want_local and not out:
        out.extend(gather_ll_air(public_dir))
        out.extend(gather_ll_air(local_air))
    return _unique_paths(out)


def _unique_paths(paths: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    uniq: list[Path] = []
    for p in paths:
        try:
            rp = p.resolve()
        except OSError:
            rp = p
        if rp in seen or not p.is_file():
            continue
        seen.add(rp)
        uniq.append(p)
    return uniq


def label_for(src: Path) -> str:
    if label_override:
        return label_override
    try:
        return f"public/{src.relative_to(public_dir)}"
    except ValueError:
        pass
    try:
        return f"local/{src.relative_to(local_air)}"
    except ValueError:
        pass
    return src.name


def kind_for(src: Path) -> str:
    if kind_override:
        return kind_override
    try:
        src.relative_to(public_dir)
        return "synthetic"
    except ValueError:
        return "private"


def emit_row(air_hash: str, status: str, spv_hash: str | None, label: str, kind: str) -> dict:
    row = {
        "air_sha256": air_hash,
        "status": status,
        "stage": stage,
        "label": label,
        "kind": kind,
    }
    if status == "ok" and spv_hash:
        row["spv_sha256"] = spv_hash
    return row


def load_ledger() -> dict[str, dict]:
    """Load ledger keyed by air_sha256. Later lines win; duplicates are collapsed."""
    if not ledger.is_file():
        return {}
    by_hash: dict[str, dict] = {}
    n_dup = 0
    for line in ledger.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        row = json.loads(line)
        air = row.get("air_sha256")
        if not air:
            continue
        if air in by_hash:
            n_dup += 1
        by_hash[air] = row
    if n_dup:
        log(f"# load: collapsed {n_dup} duplicate air_sha256 row(s)")
    return by_hash


def write_ledger(by_hash: dict[str, dict]) -> None:
    """Write ledger with exactly one row per air_sha256."""
    rows = sorted(
        by_hash.values(),
        key=lambda r: (r.get("kind", ""), r.get("label", ""), r.get("air_sha256", "")),
    )
    # Defensive: never write two rows for the same hash.
    seen: set[str] = set()
    unique: list[dict] = []
    for r in rows:
        air = r.get("air_sha256", "")
        if not air or air in seen:
            continue
        seen.add(air)
        unique.append(r)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("w") as f:
        f.write("# metal2vulkan drift ledger — hashes only, no shader bodies\n")
        f.write(
            "# schema: air_sha256 (unique), spv_sha256?, status=ok|fallback|timeout, "
            "stage, label, kind\n"
        )
        for r in unique:
            f.write(json.dumps(r, separators=(",", ":"), sort_keys=True) + "\n")
    log(f"# wrote {len(unique)} unique rows -> {ledger}")


def map_translate(
    sources: list[Path],
    *,
    air_hashes: dict[Path, str] | None = None,
    progress_prefix: str = "run",
) -> dict[Path, tuple[str, str | None]]:
    """Translate sources in parallel; print live progress as each finishes."""
    if not sources:
        return {}
    results: dict[Path, tuple[str, str | None]] = {}
    n = len(sources)
    jobs = min(workers, n)
    log(f"# translate workers={jobs} sources={n} timeout={timeout_secs:g}s")
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futs = {pool.submit(translate_hash, src): src for src in sources}
        done = 0
        for fut in as_completed(futs):
            src = futs[fut]
            done += 1
            try:
                status, spv_hash = fut.result()
            except Exception as e:  # noqa: BLE001 — surface as fallback-ish fail
                log(f"  [{done}/{n}] ERROR      {label_for(src)}: {e}")
                results[src] = ("fallback", None)
                continue
            results[src] = (status, spv_hash)
            if quiet:
                # Still show a compact progress ticker so long mints aren't silent.
                if done == n or done % 25 == 0 or done == 1:
                    log(f"  [{done}/{n}] …")
            else:
                air_short = ""
                if air_hashes and src in air_hashes:
                    air_short = f"  {air_hashes[src][:12]}"
                log(f"  [{done}/{n}] {progress_prefix} {status:<10} {label_for(src)}{air_short}")
    return results


def cmd_mint() -> int:
    if not want_public and not want_local and not (
        file_list.is_file() and file_list.stat().st_size
    ):
        log("mint requires --public and/or --local and/or --file")
        return 64
    sources = collect_sources()
    if not sources:
        log("mint: no source files found")
        return 1

    # Hash sources serially (cheap I/O); translate in parallel (process-bound).
    log(f"# hashing {len(sources)} source(s)…")
    air_hashes = {src: sha256_file(src) for src in sources}

    # Prefer one source per content hash so we don't translate identical AIR twice.
    by_content: dict[str, Path] = {}
    for src in sources:
        air = air_hashes[src]
        # Prefer the path that sorts first by label for stable labels.
        prev = by_content.get(air)
        if prev is None or label_for(src) < label_for(prev):
            by_content[air] = src
    if len(by_content) < len(sources):
        log(
            f"# dedupe: {len(sources)} files → {len(by_content)} unique air_sha256 "
            f"({len(sources) - len(by_content)} content duplicate(s) skipped)"
        )
    unique_sources = list(by_content.values())
    translated = map_translate(
        unique_sources, air_hashes=air_hashes, progress_prefix="mint"
    )

    existing = load_ledger()
    for src in unique_sources:
        air_hash = air_hashes[src]
        status, spv_hash = translated[src]
        existing[air_hash] = emit_row(
            air_hash, status, spv_hash, label_for(src), kind_for(src)
        )
    write_ledger(existing)
    return 0


def cmd_check() -> int:
    sources = collect_sources()
    by_hash = {sha256_file(s): s for s in sources}
    ledger_rows = load_ledger()
    if not ledger_rows:
        log(f"no ledger rows at {ledger}")
        return 1

    # Only translate sources that appear in the ledger and are present on disk.
    to_run: list[Path] = []
    seen: set[Path] = set()
    for air, row in ledger_rows.items():
        src = by_hash.get(air)
        if src is None:
            continue
        rp = src.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        to_run.append(src)

    translated = map_translate(to_run, progress_prefix="check")

    n_ok = n_drift = n_missing = n_skip = n_status = 0
    # Stable report order by label.
    for air, row in sorted(
        ledger_rows.items(), key=lambda kv: (kv[1].get("label", ""), kv[0])
    ):
        label = row.get("label", air[:12])
        kind = row.get("kind", "private")
        status = row.get("status", "")
        spv_exp = row.get("spv_sha256") or ""
        src = by_hash.get(air)
        if src is None:
            if kind == "synthetic":
                print(f"  MISSING   {label}  air={air} (synthetic source not found)", flush=True)
                n_missing += 1
            else:
                if not quiet:
                    print(f"  skip      {label}  air={air} (private source absent)", flush=True)
                n_skip += 1
            continue
        got_status, got_spv = translated[src]
        if got_status != status:
            print(f"  STATUS    {label}  ledger={status} got={got_status}", flush=True)
            n_status += 1
            continue
        if got_status == "ok" and (got_spv or "") != spv_exp:
            print(f"  DRIFT     {label}  air={air}", flush=True)
            print(f"            ledger_spv={spv_exp}", flush=True)
            print(f"            got_spv   ={got_spv}", flush=True)
            n_drift += 1
            continue
        if not quiet:
            print(f"  ok        {label}", flush=True)
        n_ok += 1
    print(
        f"# summary: ok={n_ok} drift={n_drift} status_mismatch={n_status} "
        f"missing_synthetic={n_missing} skipped_private={n_skip}",
        flush=True,
    )
    if n_drift or n_status or n_missing:
        print("# RESULT: DRIFT or missing synthetic source", flush=True)
        return 1
    print("# RESULT: clean", flush=True)
    return 0


def main() -> int:
    if cmd == "mint":
        return cmd_mint()
    if cmd == "check":
        return cmd_check()
    log(f"unknown cmd {cmd}")
    return 64


if __name__ == "__main__":
    sys.exit(main())
