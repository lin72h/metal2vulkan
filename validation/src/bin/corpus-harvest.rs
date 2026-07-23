//! Harvest macOS `.metallib` files directly into JSONL shards.
//!
//! The only durable private artifacts are:
//!
//! - `validation/corpus/local/shards/shard_NN.jsonl`

use base64::Engine as _;
use metal2vulkan_validation::air::{entry_name_from_ll, stage_label_from_ll};
use metal2vulkan_validation::corpus_shards::{
    self, shard_name_for_hash, sort_json, ShardRecord, SHARD_COUNT,
};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

const PROGRAM: &str = "corpus-harvest";
const DEFAULT_MAX_AIR_BYTES: usize = 1024 * 1024;
const AIR_WRAP: &[u8; 4] = b"\xde\xc0\x17\x0b";

#[derive(Debug)]
struct Options {
    out: PathBuf,
    limit: Option<usize>,
    offset: usize,
    start_set: StartSet,
    include_apps: bool,
    metallibs: Vec<PathBuf>,
    llvm_dis: Option<PathBuf>,
    max_air_bytes: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StartSet {
    System,
    Apps,
    All,
}

impl StartSet {
    fn parse(s: &str) -> Option<Self> {
        match s {
            "system" => Some(Self::System),
            "apps" => Some(Self::Apps),
            "all" => Some(Self::All),
            _ => None,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::System => "system",
            Self::Apps => "apps",
            Self::All => "all",
        }
    }
}

#[derive(Debug)]
struct AirBlob {
    offset: usize,
    bytes: Vec<u8>,
}

#[derive(Default)]
struct HarvestStats {
    libs_ok: usize,
    libs_failed: usize,
    blobs_carved: usize,
    rows_kept: usize,
    skipped_large: usize,
    dropped_helpers: usize,
    llvm_failed: usize,
    duplicates: usize,
}

fn main() {
    let Some(opts) = parse_args() else {
        return;
    };
    let code = run(opts);
    std::process::exit(code);
}

fn parse_args() -> Option<Options> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let env_limit = std::env::var("METAL2VULKAN_HARVEST_LIMIT").ok();
    let env_offset = std::env::var("METAL2VULKAN_HARVEST_OFFSET").ok();
    let env_set = std::env::var("METAL2VULKAN_HARVEST_START_SET").ok();
    let env_max = std::env::var("METAL2VULKAN_HARVEST_MAX_AIR_BYTES").ok();

    let mut opts = Options {
        out: manifest_dir.join("corpus/local"),
        limit: env_limit
            .as_deref()
            .filter(|s| !s.is_empty())
            .map(parse_usize_env)
            .transpose()
            .unwrap_or_else(|e| fatal(&e)),
        offset: env_offset
            .as_deref()
            .filter(|s| !s.is_empty())
            .map(parse_usize_env)
            .transpose()
            .unwrap_or_else(|e| fatal(&e))
            .unwrap_or(0),
        start_set: env_set
            .as_deref()
            .and_then(StartSet::parse)
            .unwrap_or(StartSet::System),
        include_apps: false,
        metallibs: Vec::new(),
        llvm_dis: None,
        max_air_bytes: env_max
            .as_deref()
            .filter(|s| !s.is_empty())
            .map(parse_usize_env)
            .transpose()
            .unwrap_or_else(|e| fatal(&e))
            .unwrap_or(DEFAULT_MAX_AIR_BYTES),
    };

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-h" | "--help" => {
                print_usage();
                return None;
            }
            "--out" => {
                opts.out =
                    PathBuf::from(args.next().unwrap_or_else(|| fatal("--out requires path")))
            }
            "--limit" => {
                let n = args.next().unwrap_or_else(|| fatal("--limit requires N"));
                opts.limit = Some(parse_usize_arg("--limit", &n));
            }
            "--offset" => {
                let n = args.next().unwrap_or_else(|| fatal("--offset requires N"));
                opts.offset = parse_usize_arg("--offset", &n);
            }
            "--start-set" => {
                let set = args
                    .next()
                    .unwrap_or_else(|| fatal("--start-set requires system|apps|all"));
                opts.start_set = StartSet::parse(&set)
                    .unwrap_or_else(|| fatal(&format!("bad --start-set {set:?}")));
            }
            "--include-apps" => opts.include_apps = true,
            "--metallib" => {
                opts.metallibs.push(PathBuf::from(
                    args.next()
                        .unwrap_or_else(|| fatal("--metallib requires path")),
                ));
            }
            "--llvm-dis" => {
                opts.llvm_dis = Some(PathBuf::from(
                    args.next()
                        .unwrap_or_else(|| fatal("--llvm-dis requires path")),
                ));
            }
            "--max-air-bytes" => {
                let n = args
                    .next()
                    .unwrap_or_else(|| fatal("--max-air-bytes requires N"));
                opts.max_air_bytes = parse_usize_arg("--max-air-bytes", &n);
            }
            other if other.starts_with("--out=") => {
                opts.out = PathBuf::from(other.trim_start_matches("--out="));
            }
            other if other.starts_with("--limit=") => {
                let n = other.trim_start_matches("--limit=");
                opts.limit = Some(parse_usize_arg("--limit", n));
            }
            other if other.starts_with("--offset=") => {
                let n = other.trim_start_matches("--offset=");
                opts.offset = parse_usize_arg("--offset", n);
            }
            other if other.starts_with("--start-set=") => {
                let set = other.trim_start_matches("--start-set=");
                opts.start_set = StartSet::parse(set)
                    .unwrap_or_else(|| fatal(&format!("bad --start-set {set:?}")));
            }
            other if other.starts_with("--metallib=") => {
                opts.metallibs
                    .push(PathBuf::from(other.trim_start_matches("--metallib=")));
            }
            other if other.starts_with("--llvm-dis=") => {
                opts.llvm_dis = Some(PathBuf::from(other.trim_start_matches("--llvm-dis=")));
            }
            other if other.starts_with("--max-air-bytes=") => {
                let n = other.trim_start_matches("--max-air-bytes=");
                opts.max_air_bytes = parse_usize_arg("--max-air-bytes", n);
            }
            other => fatal(&format!("unknown arg: {other}")),
        }
    }

    Some(opts)
}

fn print_usage() {
    eprintln!(
        "usage: {PROGRAM} [--out DIR] [--limit N] [--offset N]\n\
                \t\t[--start-set system|apps|all] [--include-apps]\n\
                \t\t[--metallib PATH ...] [--llvm-dis PATH]\n\
                \t\t[--max-air-bytes N]\n\
         \n\
         Harvests metallib AIR directly into local/shards/shard_NN.jsonl.\n\
         No local/air, local/metallib, local/ledger, or local/tmp\n\
         intermediates are retained. Environment defaults: METAL2VULKAN_HARVEST_LIMIT,\n\
         METAL2VULKAN_HARVEST_OFFSET, METAL2VULKAN_HARVEST_START_SET,\n\
         METAL2VULKAN_HARVEST_MAX_AIR_BYTES, METAL2VULKAN_LLVM_DIS."
    );
}

fn parse_usize_env(s: &str) -> Result<usize, String> {
    s.parse::<usize>()
        .map_err(|e| format!("bad harvest env integer {s:?}: {e}"))
}

fn parse_usize_arg(flag: &str, s: &str) -> usize {
    s.parse::<usize>()
        .unwrap_or_else(|e| fatal(&format!("bad {flag} {s:?}: {e}")))
}

fn fatal(msg: &str) -> ! {
    eprintln!("{PROGRAM}: {msg}");
    std::process::exit(64);
}

fn run(opts: Options) -> i32 {
    let t0 = Instant::now();
    eprintln!("# {PROGRAM}");
    eprintln!("# out       {}", opts.out.display());
    eprintln!("# max AIR   {} bytes", opts.max_air_bytes);

    if cfg!(not(target_os = "macos")) && opts.metallibs.is_empty() {
        eprintln!("{PROGRAM}: system metallib enumeration requires macOS; pass --metallib PATH");
        return 1;
    }

    let llvm_dis = match resolve_llvm_dis(opts.llvm_dis.as_deref()) {
        Ok(path) => path,
        Err(e) => {
            eprintln!("{PROGRAM}: {e}");
            return 1;
        }
    };
    eprintln!("# llvm-dis  {}", llvm_dis.display());

    let (batch, total_found) = match select_metallibs(&opts) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("{PROGRAM}: {e}");
            return 1;
        }
    };
    eprintln!(
        "# start-set={} total_found={} batch={} offset={} limit={}",
        opts.start_set.as_str(),
        total_found,
        batch.len(),
        opts.offset,
        opts.limit
            .map(|n| n.to_string())
            .unwrap_or_else(|| "none".into())
    );

    cleanup_legacy_intermediates(&opts.out);
    let _cleanup_intermediates_on_exit = HarvestIntermediateCleanup::new(opts.out.clone());

    let mut stats = HarvestStats::default();
    let mut by_hash: HashMap<String, ShardRecord> = HashMap::new();
    for lib in batch {
        harvest_one(&lib, &llvm_dis, &opts, &mut by_hash, &mut stats);
    }

    if by_hash.is_empty() {
        eprintln!(
            "# RESULT: no shard rows (libs_ok={} libs_failed={} carved={} llvm_failed={})",
            stats.libs_ok, stats.libs_failed, stats.blobs_carved, stats.llvm_failed
        );
        return if stats.libs_failed == 0 { 0 } else { 1 };
    }

    if let Err(e) = write_shards(&opts.out, by_hash.into_values().collect()) {
        eprintln!("{PROGRAM}: write shards: {e}");
        return 1;
    }

    eprintln!(
        "# RESULT: libs_ok={} libs_failed={} carved={} kept={} dup={} large={} dropped_helpers={} llvm_failed={} ({:.1}s)",
        stats.libs_ok,
        stats.libs_failed,
        stats.blobs_carved,
        stats.rows_kept,
        stats.duplicates,
        stats.skipped_large,
        stats.dropped_helpers,
        stats.llvm_failed,
        t0.elapsed().as_secs_f64()
    );
    if stats.libs_failed > 0 && stats.libs_ok == 0 {
        1
    } else {
        0
    }
}

fn select_metallibs(opts: &Options) -> Result<(Vec<PathBuf>, usize), String> {
    if !opts.metallibs.is_empty() {
        for path in &opts.metallibs {
            if !path.is_file() {
                return Err(format!("not a file: {}", path.display()));
            }
        }
        return Ok((opts.metallibs.clone(), opts.metallibs.len()));
    }

    let mut roots = Vec::new();
    match opts.start_set {
        StartSet::System | StartSet::All => {
            roots.push(PathBuf::from("/System/Library"));
            roots.push(PathBuf::from("/Library"));
            if opts.include_apps && opts.start_set == StartSet::System {
                roots.push(PathBuf::from("/Applications"));
            }
        }
        StartSet::Apps => {}
    }
    match opts.start_set {
        StartSet::Apps | StartSet::All => roots.push(PathBuf::from("/Applications")),
        StartSet::System => {}
    }

    let mut found = Vec::new();
    for root in roots {
        scan_metallibs(&root, &mut found);
    }
    let mut unique = HashMap::<String, PathBuf>::new();
    for path in found {
        let key = path
            .canonicalize()
            .unwrap_or_else(|_| path.clone())
            .display()
            .to_string();
        unique.insert(key, path);
    }
    let mut all: Vec<_> = unique.into_values().collect();
    all.sort_by_key(|a| priority_key(a));
    let total = all.len();
    let start = opts.offset.min(total);
    let end = opts
        .limit
        .map(|limit| start.saturating_add(limit).min(total))
        .unwrap_or(total);
    Ok((all[start..end].to_vec(), total))
}

fn scan_metallibs(root: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(root) else {
        return;
    };
    for ent in entries.flatten() {
        let path = ent.path();
        let Ok(ft) = ent.file_type() else {
            continue;
        };
        if ft.is_dir() {
            scan_metallibs(&path, out);
        } else if ft.is_file()
            && path
                .extension()
                .and_then(|e| e.to_str())
                .is_some_and(|e| e == "metallib")
        {
            out.push(path);
        }
    }
}

fn priority_key(path: &Path) -> (String, String) {
    let s = path.display().to_string();
    let band = if [
        "WindowServer",
        "SkyLight",
        "CoreDisplay",
        "CoreAnimation",
        "QuartzCore",
    ]
    .iter()
    .any(|needle| s.contains(needle))
    {
        "00"
    } else if ["AppleMetalOpenGLRenderer", "PixelConverter", "OpenGL"]
        .iter()
        .any(|needle| s.contains(needle))
    {
        "01"
    } else if s.contains("AGXMetal") {
        "02"
    } else if s.starts_with("/System/Library") {
        "03"
    } else if s.starts_with("/Library") {
        "04"
    } else if s.starts_with("/Applications") {
        "05"
    } else {
        "99"
    };
    (band.to_string(), s)
}

fn resolve_llvm_dis(explicit: Option<&Path>) -> Result<PathBuf, String> {
    let mut candidates = Vec::<PathBuf>::new();
    if let Some(path) = explicit {
        candidates.push(path.to_path_buf());
    }
    if let Some(path) = std::env::var_os("METAL2VULKAN_LLVM_DIS") {
        candidates.push(PathBuf::from(path));
    }
    candidates.push(PathBuf::from("/opt/homebrew/opt/llvm/bin/llvm-dis"));
    candidates.push(PathBuf::from("/usr/local/opt/llvm/bin/llvm-dis"));
    candidates.push(PathBuf::from("llvm-dis"));

    for candidate in candidates {
        if candidate.components().count() > 1 && candidate.is_file() {
            return Ok(candidate);
        }
        if candidate.components().count() == 1 {
            if let Some(found) = find_on_path(&candidate) {
                return Ok(found);
            }
        }
    }
    Err("llvm-dis not found (pass --llvm-dis or set METAL2VULKAN_LLVM_DIS)".into())
}

fn find_on_path(name: &Path) -> Option<PathBuf> {
    let name = name.to_str()?;
    let path = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path) {
        let candidate = dir.join(name);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

struct HarvestIntermediateCleanup {
    out: PathBuf,
}

impl HarvestIntermediateCleanup {
    fn new(out: PathBuf) -> Self {
        Self { out }
    }
}

impl Drop for HarvestIntermediateCleanup {
    fn drop(&mut self) {
        cleanup_intermediate_dirs(&self.out);
    }
}

fn cleanup_legacy_intermediates(out: &Path) {
    cleanup_intermediate_dirs(out);
    cleanup_legacy_root_shards(out);
}

fn cleanup_intermediate_dirs(out: &Path) {
    for sub in ["air", "metallib", "ledger", "tmp"] {
        let path = out.join(sub);
        if path.exists() {
            if let Err(e) = fs::remove_dir_all(&path) {
                eprintln!("# warn: remove intermediate {}: {e}", path.display());
            } else {
                eprintln!("# removed intermediate {}", path.display());
            }
        }
    }
}

fn cleanup_legacy_root_shards(out: &Path) {
    let Ok(entries) = fs::read_dir(out) else {
        return;
    };
    for ent in entries.flatten() {
        let path = ent.path();
        let Some(name) = path.file_name().and_then(|s| s.to_str()) else {
            continue;
        };
        if name.starts_with("shard_") && name.ends_with(".jsonl") && path.is_file() {
            if let Err(e) = fs::remove_file(&path) {
                eprintln!("# warn: remove legacy shard {}: {e}", path.display());
            } else {
                eprintln!("# removed legacy shard {}", path.display());
            }
        }
    }
}

fn harvest_one(
    lib: &Path,
    llvm_dis: &Path,
    opts: &Options,
    by_hash: &mut HashMap<String, ShardRecord>,
    stats: &mut HarvestStats,
) {
    let lib_bytes = match fs::read(lib) {
        Ok(bytes) => bytes,
        Err(e) => {
            stats.libs_failed += 1;
            eprintln!("  FAIL read {}: {e}", lib.display());
            return;
        }
    };
    let lib_sha = corpus_shards::sha256_bytes(&lib_bytes);
    let blobs = extract_air_blobs(&lib_bytes, opts.max_air_bytes, stats);
    let carved_for_lib = blobs.len();
    stats.blobs_carved += carved_for_lib;
    let mut kept_for_lib = 0usize;

    for blob in blobs {
        let ll_text = match disassemble_air(&blob.bytes, llvm_dis, &lib_sha, blob.offset) {
            Ok(ll) => ll,
            Err(e) => {
                stats.llvm_failed += 1;
                eprintln!(
                    "  FAIL llvm-dis {} off={} ({e})",
                    lib.display(),
                    blob.offset
                );
                continue;
            }
        };
        let Some((stage, entry)) = classify_module(&ll_text) else {
            stats.dropped_helpers += 1;
            continue;
        };
        let air_sha = corpus_shards::sha256_bytes(ll_text.as_bytes());
        let short = air_sha.get(..8).unwrap_or("unknown").to_string();
        let label = format!("local/{air_sha}.ll");
        let row = ShardRecord {
            id: format!("metallib/{entry}/{short}"),
            hash: short,
            shard: Some(shard_name_for_hash(&air_sha)),
            label: Some(label),
            lib: Some(lib.display().to_string()),
            lib_sha256: Some(lib_sha.clone()),
            fn_name: Some(entry),
            stage: Some(stage),
            blob_b64: Some(base64::engine::general_purpose::STANDARD.encode(&blob.bytes)),
            air_ll: ll_text,
            air_sha256: air_sha.clone(),
        };
        match by_hash.get(&air_sha) {
            None => {
                by_hash.insert(air_sha, row);
                kept_for_lib += 1;
                stats.rows_kept += 1;
            }
            Some(prev) if row.lib.as_deref().unwrap_or("") < prev.lib.as_deref().unwrap_or("") => {
                by_hash.insert(air_sha, row);
                stats.duplicates += 1;
            }
            Some(_) => {
                stats.duplicates += 1;
            }
        }
    }

    stats.libs_ok += 1;
    eprintln!(
        "  ok kept={} carved={} {}",
        kept_for_lib,
        carved_for_lib,
        lib.display()
    );
}

fn extract_air_blobs(data: &[u8], max_air_bytes: usize, stats: &mut HarvestStats) -> Vec<AirBlob> {
    let mut out = Vec::new();
    let mut i = 0usize;
    while let Some(rel) = data[i..]
        .windows(AIR_WRAP.len())
        .position(|w| w == AIR_WRAP)
    {
        let j = i + rel;
        if j + 0x14 <= data.len() {
            let bc_off = u32::from_le_bytes(data[j + 8..j + 12].try_into().unwrap()) as usize;
            let bc_size = u32::from_le_bytes(data[j + 12..j + 16].try_into().unwrap()) as usize;
            let blen = bc_off.saturating_add(bc_size);
            if 0x14 <= blen && blen <= data.len() - j {
                if blen <= max_air_bytes {
                    out.push(AirBlob {
                        offset: j,
                        bytes: data[j..j + blen].to_vec(),
                    });
                } else {
                    stats.skipped_large += 1;
                }
            }
        }
        i = j + 1;
    }
    out
}

fn disassemble_air(
    air: &[u8],
    llvm_dis: &Path,
    lib_sha: &str,
    offset: usize,
) -> Result<String, String> {
    let tmp = std::env::temp_dir().join(format!(
        "m2v-corpus-harvest-{}-{}-{offset}",
        std::process::id(),
        lib_sha.get(..12).unwrap_or(lib_sha)
    ));
    let _ = fs::remove_dir_all(&tmp);
    fs::create_dir_all(&tmp).map_err(|e| format!("create {}: {e}", tmp.display()))?;
    let in_air = tmp.join("case.air");
    let out_ll = tmp.join("case.ll");
    fs::write(&in_air, air).map_err(|e| format!("write {}: {e}", in_air.display()))?;
    let output = Command::new(llvm_dis)
        .arg(&in_air)
        .arg("-o")
        .arg(&out_ll)
        .output()
        .map_err(|e| format!("spawn {}: {e}", llvm_dis.display()));
    let result = match output {
        Ok(output) if output.status.success() && out_ll.is_file() => {
            fs::read_to_string(&out_ll).map_err(|e| format!("read {}: {e}", out_ll.display()))
        }
        Ok(output) => Err(format!(
            "llvm-dis exited {}: {}{}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        )),
        Err(e) => Err(e),
    };
    let _ = fs::remove_dir_all(&tmp);
    result
}

fn classify_module(ll: &str) -> Option<(String, String)> {
    let stage = stage_label_from_ll(ll)?.to_string();
    let entry = entry_name_from_ll(ll).unwrap_or_else(|| "unknown".into());

    if entry.starts_with("_GLOBAL__sub_I_") || entry.contains("_stitching_traits_impl") {
        return None;
    }
    Some((stage, entry))
}

fn write_shards(out: &Path, mut rows: Vec<ShardRecord>) -> std::io::Result<()> {
    let shards_dir = out.join("shards");
    if shards_dir.exists() {
        fs::remove_dir_all(&shards_dir)?;
    }
    fs::create_dir_all(&shards_dir)?;

    rows.sort_by(|a, b| {
        a.lib
            .cmp(&b.lib)
            .then_with(|| a.fn_name.cmp(&b.fn_name))
            .then_with(|| a.air_sha256.cmp(&b.air_sha256))
    });

    let mut shards: Vec<Vec<ShardRecord>> = vec![Vec::new(); SHARD_COUNT];
    for mut row in rows {
        let index = corpus_shards::shard_index_for_hash(&row.air_sha256);
        row.shard = Some(corpus_shards::shard_name_for_index(index));
        shards[index].push(row);
    }

    for (index, rows) in shards.iter().enumerate() {
        let path = shards_dir.join(corpus_shards::shard_name_for_index(index));
        let mut file = File::create(&path)?;
        for row in rows {
            let value = serde_json::to_value(row).map_err(std::io::Error::other)?;
            let line = serde_json::to_string(&sort_json(value)).map_err(std::io::Error::other)?;
            writeln!(file, "{line}")?;
        }
        eprintln!(
            "  shard_{index:02}: {} row(s) -> {}",
            rows.len(),
            path.display()
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn exit_cleanup_removes_air_and_metallib_dirs() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock")
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "m2v-corpus-harvest-cleanup-test-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir_all(root.join("air")).expect("create air dir");
        fs::create_dir_all(root.join("metallib")).expect("create metallib dir");
        fs::write(root.join("air/case.air"), b"air").expect("write air fixture");
        fs::write(root.join("metallib/case.metallib"), b"metallib")
            .expect("write metallib fixture");

        {
            let _cleanup = HarvestIntermediateCleanup::new(root.clone());
        }

        assert!(!root.join("air").exists());
        assert!(!root.join("metallib").exists());
        let _ = fs::remove_dir_all(root);
    }
}
