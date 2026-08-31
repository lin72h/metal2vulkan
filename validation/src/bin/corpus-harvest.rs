//! Harvest macOS `.metallib` files directly into JSONL shards.
//!
//! The only durable private artifacts are:
//!
//! - `validation/corpus/local/sources/shard_NNN.jsonl`
//! - `validation/corpus/local/library-modules/shard_NNN.jsonl`

use base64::Engine as _;
use metal2vulkan_validation::air::stage_entry_from_ll;
use metal2vulkan_validation::hash::sha256_bytes;
use metal2vulkan_validation::library_module::{self, LibraryModuleRow};
use metal2vulkan_validation::source::{self, SourceRow};
use metal2vulkan_validation::ScratchDir;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};
use wait_timeout::ChildExt as _;

const PROGRAM: &str = "corpus-harvest";
const DEFAULT_MAX_AIR_BYTES: usize = 1024 * 1024;
const DEFAULT_LLVM_DIS_TIMEOUT: Duration = Duration::from_secs(60);
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
    llvm_dis_timeout: Duration,
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
    library_modules_kept: usize,
    skipped_large: usize,
    dropped_nonfunctions: usize,
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
        out: default_output_dir(&manifest_dir),
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
        llvm_dis_timeout: DEFAULT_LLVM_DIS_TIMEOUT,
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
            "--llvm-dis-timeout-secs" => {
                let seconds = args
                    .next()
                    .unwrap_or_else(|| fatal("--llvm-dis-timeout-secs requires N"));
                opts.llvm_dis_timeout = parse_timeout_arg(&seconds);
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
            other if other.starts_with("--llvm-dis-timeout-secs=") => {
                opts.llvm_dis_timeout =
                    parse_timeout_arg(other.trim_start_matches("--llvm-dis-timeout-secs="));
            }
            other => fatal(&format!("unknown arg: {other}")),
        }
    }

    Some(opts)
}

fn default_output_dir(manifest_dir: &Path) -> PathBuf {
    manifest_dir.join(source::DEFAULT_CORPUS_REL)
}

fn print_usage() {
    eprintln!(
        "usage: {PROGRAM} [--out DIR] [--limit N] [--offset N]\n\
                \t\t[--start-set system|apps|all] [--include-apps]\n\
                \t\t[--metallib PATH ...] [--llvm-dis PATH]\n\
                \t\t[--max-air-bytes N] [--llvm-dis-timeout-secs N]\n\
         \n\
         DIR is the corpus root. Harvests metallib AIR directly into\n\
         DIR/local/sources/shard_NNN.jsonl.\n\
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

fn parse_timeout_arg(value: &str) -> Duration {
    let seconds = parse_usize_arg("--llvm-dis-timeout-secs", value);
    if seconds == 0 {
        fatal("--llvm-dis-timeout-secs must be greater than zero");
    }
    Duration::from_secs(seconds as u64)
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
    let mut by_hash = HashMap::new();
    let mut library_modules = HashMap::new();
    let mut indexed_entry_duplicates = 0usize;
    let indexed_source_memberships = match source::indexed_source_memberships(&opts.out) {
        Ok(memberships) => memberships,
        Err(error) => {
            eprintln!("{PROGRAM}: load source index: {error}");
            return 1;
        }
    };
    for lib in batch {
        harvest_one(
            &lib,
            &llvm_dis,
            &opts,
            &mut by_hash,
            &mut library_modules,
            &mut stats,
        );
        // Dependency modules are deliberately retained independently. A stage-entry row can be
        // discarded only when SQLite already knows every parent-library membership observed in
        // this batch. Apply that filter after each library so a broad reharvest cannot accumulate
        // gigabytes of already-complete entries.
        let before_filter = by_hash.len();
        by_hash.retain(|hash, row| {
            !source_memberships_already_indexed(&indexed_source_memberships, hash, row)
        });
        indexed_entry_duplicates += before_filter - by_hash.len();
    }

    if by_hash.is_empty() && library_modules.is_empty() {
        eprintln!(
            "# RESULT: no new AIR modules (libs_ok={} libs_failed={} carved={} llvm_failed={})",
            stats.libs_ok, stats.libs_failed, stats.blobs_carved, stats.llvm_failed
        );
        return if stats.libs_failed == 0 { 0 } else { 1 };
    }

    let mut merge = match source::merge_source_shards(&opts.out, by_hash.into_values()) {
        Ok(merge) => merge,
        Err(error) => {
            eprintln!("{PROGRAM}: merge source shards: {error}");
            return 1;
        }
    };
    merge.duplicates += indexed_entry_duplicates;
    let library_merge =
        match library_module::merge_library_module_shards(&opts.out, library_modules.into_values())
        {
            Ok(merge) => merge,
            Err(error) => {
                eprintln!("{PROGRAM}: merge library-module shards: {error}");
                return 1;
            }
        };

    eprintln!(
        "# RESULT: libs_ok={} libs_failed={} carved={} entry_batch_unique={} entry_inserted={} entry_memberships_added={} entry_replaced={} entry_dup={} entry_shards={} library_module_batch_unique={} library_module_inserted={} library_memberships_added={} library_module_dup={} library_module_shards={} large={} dropped_nonfunctions={} llvm_failed={} ({:.1}s)",
        stats.libs_ok,
        stats.libs_failed,
        stats.blobs_carved,
        stats.rows_kept,
        merge.inserted,
        merge.merged_memberships,
        merge.replaced,
        stats.duplicates + merge.duplicates,
        merge.affected_shards,
        stats.library_modules_kept,
        library_merge.inserted,
        library_merge.merged_memberships,
        library_merge.duplicates,
        library_merge.affected_shards,
        stats.skipped_large,
        stats.dropped_nonfunctions,
        stats.llvm_failed,
        t0.elapsed().as_secs_f64()
    );
    if stats.libs_failed > 0 && stats.libs_ok == 0 {
        1
    } else {
        0
    }
}

fn source_memberships_already_indexed(
    indexed: &HashMap<String, HashSet<String>>,
    hash: &str,
    row: &SourceRow,
) -> bool {
    indexed.get(hash).is_some_and(|memberships| {
        row.lib_sha256s
            .iter()
            .all(|library| memberships.contains(library))
    })
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
        cleanup_intermediate_dirs(&self.out.join("local"));
    }
}

fn cleanup_legacy_intermediates(out: &Path) {
    let local = out.join("local");
    cleanup_intermediate_dirs(&local);
    cleanup_legacy_root_shards(&local);
}

fn cleanup_intermediate_dirs(out: &Path) {
    for sub in ["air", "metallib", "ledger", "tmp", "shards"] {
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
    by_hash: &mut HashMap<String, SourceRow>,
    library_modules: &mut HashMap<String, LibraryModuleRow>,
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
    let lib_sha = sha256_bytes(&lib_bytes);
    let blobs = extract_air_blobs(&lib_bytes, opts.max_air_bytes, stats);
    let carved_for_lib = blobs.len();
    stats.blobs_carved += carved_for_lib;
    let mut kept_for_lib = 0usize;

    for blob in blobs {
        let raw_ll = match disassemble_air(
            &blob.bytes,
            llvm_dis,
            &lib_sha,
            blob.offset,
            opts.llvm_dis_timeout,
        ) {
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
        let ll_text = metal2vulkan::tools::sanitize_ll_text_with_datalayout(&raw_ll).0;
        let air_sha = sha256_bytes(ll_text.as_bytes());
        let blob_b64 = base64::engine::general_purpose::STANDARD.encode(&blob.bytes);
        let Some((stage, entry)) = classify_module(&ll_text) else {
            if !ll_text
                .lines()
                .any(|line| line.trim_start().starts_with("define "))
            {
                stats.dropped_nonfunctions += 1;
                continue;
            }
            let row = LibraryModuleRow {
                module_sha256: air_sha.clone(),
                air_ll: ll_text,
                blob_b64,
                lib_sha256s: vec![lib_sha.clone()],
                label: format!("local/library-module/{air_sha}.ll"),
            };
            merge_library_module_row(library_modules, row, stats);
            continue;
        };
        let label = format!("local/{air_sha}.ll");
        let row = SourceRow {
            air_sha256: air_sha.clone(),
            stage,
            entry,
            air_ll: ll_text,
            blob_b64: Some(blob_b64),
            lib_sha256s: vec![lib_sha.clone()],
            label,
        };
        if merge_source_row(by_hash, row, stats) {
            kept_for_lib += 1;
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

fn merge_library_module_row(
    by_hash: &mut HashMap<String, LibraryModuleRow>,
    row: LibraryModuleRow,
    stats: &mut HarvestStats,
) {
    let hash = row.module_sha256.clone();
    match by_hash.get_mut(&hash) {
        None => {
            by_hash.insert(hash, row);
            stats.library_modules_kept += 1;
        }
        Some(previous) => {
            previous.lib_sha256s.extend(row.lib_sha256s);
            previous.lib_sha256s.sort();
            previous.lib_sha256s.dedup();
            if row.blob_b64 < previous.blob_b64 {
                previous.blob_b64 = row.blob_b64;
            }
        }
    }
}

fn merge_source_row(
    by_hash: &mut HashMap<String, SourceRow>,
    row: SourceRow,
    stats: &mut HarvestStats,
) -> bool {
    let hash = row.air_sha256.clone();
    match by_hash.get_mut(&hash) {
        None => {
            by_hash.insert(hash, row);
            stats.rows_kept += 1;
            true
        }
        Some(previous) => {
            previous.lib_sha256s.extend(row.lib_sha256s);
            previous.lib_sha256s.sort();
            previous.lib_sha256s.dedup();
            if source::source_blob_is_preferred(
                previous.blob_b64.as_deref(),
                row.blob_b64.as_deref(),
            ) {
                previous.blob_b64 = row.blob_b64;
            }
            stats.duplicates += 1;
            false
        }
    }
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
    timeout: Duration,
) -> Result<String, String> {
    let scratch = ScratchDir::new(&format!(
        "harvest-disassemble-{}-{offset}",
        lib_sha.get(..12).unwrap_or(lib_sha)
    ))?;
    let in_air = scratch.path().join("case.air");
    let out_ll = scratch.path().join("case.ll");
    let stdout_path = scratch.path().join("stdout.txt");
    let stderr_path = scratch.path().join("stderr.txt");
    fs::write(&in_air, air).map_err(|e| format!("write {}: {e}", in_air.display()))?;
    let mut child = Command::new(llvm_dis)
        .arg(&in_air)
        .arg("-o")
        .arg(&out_ll)
        .stdout(Stdio::from(fs::File::create(&stdout_path).map_err(
            |error| format!("create {}: {error}", stdout_path.display()),
        )?))
        .stderr(Stdio::from(fs::File::create(&stderr_path).map_err(
            |error| format!("create {}: {error}", stderr_path.display()),
        )?))
        .spawn()
        .map_err(|e| format!("spawn {}: {e}", llvm_dis.display()))?;
    let status = child
        .wait_timeout(timeout)
        .map_err(|error| format!("wait for {}: {error}", llvm_dis.display()))?;
    let Some(status) = status else {
        let _ = child.kill();
        let _ = child.wait();
        return Err(format!(
            "llvm-dis timed out after {} seconds",
            timeout.as_secs_f64()
        ));
    };
    if status.success() && out_ll.is_file() {
        fs::read_to_string(&out_ll).map_err(|e| format!("read {}: {e}", out_ll.display()))
    } else {
        let stdout = fs::read(&stdout_path).unwrap_or_default();
        let stderr = fs::read(&stderr_path).unwrap_or_default();
        Err(format!(
            "llvm-dis exited {status}: {}{}",
            String::from_utf8_lossy(&stdout),
            String::from_utf8_lossy(&stderr)
        ))
    }
}

fn classify_module(ll: &str) -> Option<(String, String)> {
    let (stage, entry) = stage_entry_from_ll(ll)?;

    if entry.starts_with("_GLOBAL__sub_I_") || entry.contains("_stitching_traits_impl") {
        return None;
    }
    Some((stage.to_string(), entry))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn default_output_is_the_corpus_root() {
        let manifest_dir = Path::new("/validation");
        let output = default_output_dir(manifest_dir);
        assert_eq!(output, Path::new("/validation/corpus"));
        assert_eq!(
            source::source_shards_dir(&output),
            Path::new("/validation/corpus/local/sources")
        );
    }

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
        fs::create_dir_all(root.join("local/air")).expect("create air dir");
        fs::create_dir_all(root.join("local/metallib")).expect("create metallib dir");
        fs::write(root.join("local/air/case.air"), b"air").expect("write air fixture");
        fs::write(root.join("local/metallib/case.metallib"), b"metallib")
            .expect("write metallib fixture");

        {
            let _cleanup = HarvestIntermediateCleanup::new(root.clone());
        }

        assert!(!root.join("local/air").exists());
        assert!(!root.join("local/metallib").exists());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn extracts_valid_wrappers_and_enforces_size_limit() {
        let payload = b"ABCD";
        let mut wrapper = vec![0u8; 0x18];
        wrapper[..4].copy_from_slice(AIR_WRAP);
        wrapper[8..12].copy_from_slice(&0x14u32.to_le_bytes());
        wrapper[12..16].copy_from_slice(&(payload.len() as u32).to_le_bytes());
        wrapper[0x14..].copy_from_slice(payload);
        let mut stats = HarvestStats::default();
        let blobs = extract_air_blobs(&wrapper, wrapper.len(), &mut stats);
        assert_eq!(blobs.len(), 1);
        assert_eq!(blobs[0].bytes, wrapper);
        assert!(extract_air_blobs(&wrapper, wrapper.len() - 1, &mut stats).is_empty());
        assert_eq!(stats.skipped_large, 1);
    }

    #[test]
    fn classification_requires_stable_stage_metadata() {
        let ll = "define void @k() { ret void }\n!air.kernel = !{!0}\n!0 = !{ptr @k}";
        assert_eq!(classify_module(ll), Some(("Kernel".into(), "k".into())));
        assert_eq!(classify_module("define void @helper() { ret void }"), None);
        assert_eq!(
            classify_module("define void @k() { ret void }\n!air.kernel = !{!0}\n!0 = !{i32 1}"),
            None
        );
    }

    #[test]
    fn non_entry_function_modules_retain_every_parent_library() {
        let ll = "define void @visible() { ret void }";
        assert_eq!(classify_module(ll), None);
        let mut modules = HashMap::new();
        let mut stats = HarvestStats::default();
        for library in ["11".repeat(32), "22".repeat(32)] {
            merge_library_module_row(
                &mut modules,
                LibraryModuleRow {
                    module_sha256: sha256_bytes(ll.as_bytes()),
                    air_ll: ll.into(),
                    blob_b64: base64::engine::general_purpose::STANDARD.encode(b"bitcode"),
                    lib_sha256s: vec![library],
                    label: "local/library-module.ll".into(),
                },
                &mut stats,
            );
        }
        assert_eq!(stats.library_modules_kept, 1);
        assert_eq!(modules.values().next().unwrap().lib_sha256s.len(), 2);
    }

    #[test]
    fn deduplication_is_by_sanitized_air_hash_and_order_independent() {
        fn row(library: &str) -> SourceRow {
            SourceRow {
                air_sha256: "11".repeat(32),
                stage: "Kernel".into(),
                entry: "k".into(),
                air_ll: "ll".into(),
                blob_b64: Some("YmxvYg==".into()),
                lib_sha256s: vec![library.into()],
                label: "local/test.ll".into(),
            }
        }
        let mut forward = HashMap::new();
        let mut forward_stats = HarvestStats::default();
        merge_source_row(&mut forward, row("bb"), &mut forward_stats);
        merge_source_row(&mut forward, row("aa"), &mut forward_stats);

        let mut reverse = HashMap::new();
        let mut reverse_stats = HarvestStats::default();
        merge_source_row(&mut reverse, row("aa"), &mut reverse_stats);
        merge_source_row(&mut reverse, row("bb"), &mut reverse_stats);
        assert_eq!(
            forward[&"11".repeat(32)].lib_sha256s,
            ["aa".to_string(), "bb".to_string()]
        );
        assert_eq!(
            reverse[&"11".repeat(32)].lib_sha256s,
            ["aa".to_string(), "bb".to_string()]
        );
        assert_eq!(forward_stats.duplicates, 1);
        assert_eq!(reverse_stats.duplicates, 1);
    }

    #[test]
    fn indexed_entry_is_reopened_only_for_a_new_parent_library() {
        let hash = "11".repeat(32);
        let mut row = SourceRow {
            air_sha256: hash.clone(),
            stage: "Kernel".into(),
            entry: "k".into(),
            air_ll: "ll".into(),
            blob_b64: Some("YmxvYg==".into()),
            lib_sha256s: vec!["aa".into()],
            label: "local/test.ll".into(),
        };
        let mut indexed = HashMap::new();
        assert!(!source_memberships_already_indexed(&indexed, &hash, &row));

        indexed.insert(hash.clone(), HashSet::from(["aa".to_string()]));
        assert!(source_memberships_already_indexed(&indexed, &hash, &row));
        row.lib_sha256s.push("bb".into());
        assert!(!source_memberships_already_indexed(&indexed, &hash, &row));
    }

    #[cfg(unix)]
    #[test]
    fn disassembler_scratch_is_removed_on_success_failure_timeout_and_signal() {
        use std::collections::HashSet;
        use std::os::unix::fs::PermissionsExt as _;

        fn scratch_paths() -> HashSet<PathBuf> {
            fs::read_dir(std::env::temp_dir())
                .into_iter()
                .flatten()
                .filter_map(Result::ok)
                .map(|entry| entry.path())
                .filter(|path| {
                    path.file_name()
                        .and_then(|name| name.to_str())
                        .is_some_and(|name| {
                            name.starts_with("metal2vulkan-validation-")
                                && name.contains("-harvest-disassemble-")
                        })
                })
                .collect()
        }

        let outer = ScratchDir::new("harvest-disassembler-test").unwrap();
        let tool = outer.path().join("llvm-dis.sh");
        let baseline = scratch_paths();
        for (script, timeout, succeeds) in [
            (
                "#!/bin/sh\nprintf 'define void @k() { ret void }' > \"$3\"\n",
                Duration::from_secs(2),
                true,
            ),
            ("#!/bin/sh\nexit 1\n", Duration::from_secs(2), false),
            (
                "#!/bin/sh\nexec sleep 2\n",
                Duration::from_millis(30),
                false,
            ),
            ("#!/bin/sh\nkill -TERM $$\n", Duration::from_secs(2), false),
        ] {
            fs::write(&tool, script).unwrap();
            let mut permissions = fs::metadata(&tool).unwrap().permissions();
            permissions.set_mode(0o700);
            fs::set_permissions(&tool, permissions).unwrap();
            assert_eq!(
                disassemble_air(b"air", &tool, &"11".repeat(32), 0, timeout).is_ok(),
                succeeds
            );
            assert_eq!(scratch_paths(), baseline);
        }
    }
}
