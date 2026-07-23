//! Corpus execution ledgers and runners (see repo-root `plan.md`).
//!
//! Lazy plan inference + Metal golden / Vulkan / MoltenVK candidate JSONL writers.

use crate::air::{entry_name_from_ll, stage_from_ll};
use crate::corpus_shards;
#[cfg(target_os = "macos")]
use crate::corpus_source::source_metallib_for_air;
use crate::corpus_source::{air_blob_for_oracle, load_ll_text, resolve_source, SourceFile};
use crate::hash::sha256_bytes as sha256_hex;
use crate::jsonl::sort_json;
use crate::texture::fragment_writes_depth;
use crate::{
    seeded_buffer_bytes, seeded_texture_bytes, BufferInput, BufferRole, DataFormat, Dispatch,
    Extent3d, Inputs, Output, Render, Seed, Stage, TextureInput, TextureRole,
};
use base64::Engine as _;
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

/// Bumped when default seed construction changes (bounded control-param buffers).
pub const SEED_PROFILE: &str = "deterministic_v2_bounded_control";
pub const PLAN_VERSION: u32 = 1;
pub const DEFAULT_BUFFER_LEN: usize = 256;
pub const DEFAULT_TEXTURE_EXTENT: Extent3d = Extent3d::new(8, 8, 1);
pub const CASE_TIMEOUT_ENV: &str = "METAL2VULKAN_CORPUS_TIMEOUT_SECS";
/// Per-kernel wall timeout (seconds). On expiry the parent SIGKILLs the worker's process group.
///
/// NOTE: this frees the CPU worker (and `metal-as`/helper descendants) but does **not** cancel an
/// in-flight GPU kernel — a committed Metal command buffer cannot be cancelled; only a reboot
/// recovers a wedged GPU. Infinite-loop safety therefore comes from the pre-submission loop-budget
/// guard in `oracle_macos`, not from this timeout. With that guard, bounded GPU work finishes in
/// well under a second, so a case still running at this bound means a CPU-side tool hang, not a
/// GPU loop — hence the short default (was 300s, which merely prolonged a wedge).
pub const DEFAULT_CASE_TIMEOUT_SECS: u64 = 60;
/// Log `# SLOW <air_sha256> …` when a case is still running (or finished) past this wall time.
pub const SLOW_CASE_SECS: u64 = 30;

/// Seed mode written into [`PlanBuffer::seed_mode`].
pub const SEED_MODE_DETERMINISTIC: &str = "deterministic";
/// Fixed-size control/param buffers whose integers feed loop trip counts / grid checks.
/// Seeded with small dims so MPS-style GEMMs cannot pin the GPU for ~10^9 iterations.
pub const SEED_MODE_BOUNDED_CONTROL: &str = "bounded_control";
/// Upper bound written into integer fields of bounded-control buffers (and used as
/// repeating `u32` fill when no struct field list is available).
pub const BOUNDED_CONTROL_DIM: u32 = 16;
/// Max `air.buffer_size` (bytes) still treated as a control blob rather than payload.
pub const BOUNDED_CONTROL_MAX_BYTES: usize = 4096;

// --- JSON schemas --------------------------------------------------------------------------------

pub type TranslateRow = crate::translate_ledger::TranslateLedgerRow;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanBuffer {
    pub index: u32,
    pub len: usize,
    pub role: String,
    pub seed_tag: u32,
    /// `deterministic` (default) or `bounded_control` — see [`SEED_MODE_BOUNDED_CONTROL`].
    #[serde(default = "default_seed_mode")]
    pub seed_mode: String,
}

fn default_seed_mode() -> String {
    SEED_MODE_DETERMINISTIC.to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanTexture {
    pub index: u32,
    pub format: String,
    pub role: String,
    pub w: u32,
    pub h: u32,
    pub d: u32,
    pub seed_tag: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanOutput {
    pub kind: String,
    pub index: u32,
    pub format: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub len: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub w: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub h: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub d: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarnessPlan {
    pub buffers: Vec<PlanBuffer>,
    pub textures: Vec<PlanTexture>,
    pub output: PlanOutput,
    pub dispatch_grid: [u32; 3],
    pub dispatch_tg: [u32; 3],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetalRow {
    pub air_sha256: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shard: Option<String>,
    #[serde(default)]
    pub label: String,
    pub status: String,
    pub backend: String,
    pub seed_profile: String,
    pub plan_version: u32,
    pub plan: HarnessPlan,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_sha256: Option<String>,
    /// Full oracle output bytes (standard base64). Present on `status=ok` so candidates can
    /// apply numeric tolerance without re-running Metal.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_b64: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub spv_sha256: Option<String>,
    #[serde(default)]
    pub compare: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stage: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub entry: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToleranceSpecJson {
    pub kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_abs: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_ulp: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObservedMargins {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_abs: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_ulp: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandidateRow {
    pub air_sha256: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shard: Option<String>,
    #[serde(default)]
    pub label: String,
    pub status: String,
    pub backend: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_sha256: Option<String>,
    /// Full candidate output bytes (standard base64). Present when the runner produced bytes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_b64: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub golden_output_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub spv_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tolerance: Option<ToleranceSpecJson>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub observed: Option<ObservedMargins>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Standard base64 for execution-ledger output payloads.
pub fn encode_output_b64(bytes: &[u8]) -> String {
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

/// Decode a ledger `output_b64` field.
pub fn decode_output_b64(b64: &str) -> Result<Vec<u8>, String> {
    base64::engine::general_purpose::STANDARD
        .decode(b64)
        .map_err(|e| format!("decode output_b64: {e}"))
}

/// Default numeric policy for float-like outputs when candidate ≠ metal golden.
/// Matches the AbsAndUlp example in `plan.md` (max_abs=1e-3, max_ulp=8).
pub fn default_float_tolerance() -> ToleranceSpecJson {
    ToleranceSpecJson {
        kind: "AbsAndUlp".into(),
        max_abs: Some(1e-3),
        max_ulp: Some(8),
    }
}

// --- Paths / config ------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RunBackend {
    Metal,
    Vulkan,
    MoltenVk,
}

impl RunBackend {
    pub fn as_str(self) -> &'static str {
        match self {
            RunBackend::Metal => "metal",
            RunBackend::Vulkan => "vulkan",
            RunBackend::MoltenVk => "moltenvk",
        }
    }

    pub fn ledger_file_name(self) -> &'static str {
        match self {
            RunBackend::Metal => "metal2vulkan-ledger-metal.jsonl",
            RunBackend::Vulkan => "metal2vulkan-ledger-vulkan.jsonl",
            RunBackend::MoltenVk => "metal2vulkan-ledger-moltenvk.jsonl",
        }
    }

    pub fn program_name(self) -> &'static str {
        match self {
            RunBackend::Metal => "corpus-run-metal",
            RunBackend::Vulkan => "corpus-run-vulkan",
            RunBackend::MoltenVk => "corpus-run-moltenvk",
        }
    }
}

#[derive(Debug, Clone)]
pub struct RunConfig {
    pub backend: RunBackend,
    pub corpus_dir: PathBuf,
    pub public_dir: PathBuf,
    pub local_corpus: PathBuf,
    pub translate_ledger: PathBuf,
    pub tech_ledger: PathBuf,
    pub metal_ledger: PathBuf,
    pub delta_ledger: Option<PathBuf>,
    pub force: bool,
    pub failed_only: bool,
    pub only_status: Option<String>,
    pub only_bucket: Option<String>,
    pub contains: Option<String>,
    pub dry_run: bool,
    pub quiet: bool,
    pub only_air: Option<String>,
    pub jobs: usize,
    /// When true, process in-process (worker mode) and exit with outcome code.
    pub oneshot: bool,
    /// Per-case wall timeout for worker subprocesses (default 60s; env override in CLI parsing).
    pub timeout_secs: u64,
}

impl RunConfig {
    pub fn from_manifest(backend: RunBackend) -> Self {
        let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let corpus = manifest.join("corpus");
        Self {
            backend,
            public_dir: manifest.join("fixtures/public"),
            local_corpus: corpus_shards::corpus_root_from_env_or_manifest(),
            translate_ledger: corpus.join("metal2vulkan-ledger.jsonl"),
            tech_ledger: corpus.join(backend.ledger_file_name()),
            metal_ledger: corpus.join(RunBackend::Metal.ledger_file_name()),
            delta_ledger: None,
            corpus_dir: corpus,
            force: false,
            failed_only: false,
            only_status: None,
            only_bucket: None,
            contains: None,
            dry_run: false,
            quiet: false,
            only_air: None,
            jobs: default_workers(),
            oneshot: false,
            timeout_secs: DEFAULT_CASE_TIMEOUT_SECS,
        }
    }
}

/// Default parallel workers for corpus-run-*.
///
/// Each job may translate and submit GPU work; too many concurrent Metal/Vulkan
/// contexts thrash memory. Cap at 4 unless the user raises `--jobs`.
fn default_workers() -> usize {
    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    cores.clamp(1, 4)
}

// --- CLI -----------------------------------------------------------------------------------------

pub fn parse_run_args(backend: RunBackend) -> Option<RunConfig> {
    let mut cfg = RunConfig::from_manifest(backend);
    let program = backend.program_name();
    cfg.timeout_secs = case_timeout_secs_from_env(program);
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-h" | "--help" => {
                print_run_usage(program);
                return None;
            }
            "--force" => cfg.force = true,
            "--failed-only" => cfg.failed_only = true,
            "--status" => {
                cfg.only_status = Some(
                    args.next()
                        .unwrap_or_else(|| fatal(program, "--status requires a value")),
                );
            }
            "--bucket" => {
                cfg.only_bucket = Some(
                    args.next()
                        .unwrap_or_else(|| fatal(program, "--bucket requires text")),
                );
            }
            "--contains" => {
                cfg.contains = Some(
                    args.next()
                        .unwrap_or_else(|| fatal(program, "--contains requires text")),
                );
            }
            "--dry-run" => cfg.dry_run = true,
            "--quiet" => cfg.quiet = true,
            "--oneshot" => cfg.oneshot = true,
            "--delta-ledger" => {
                cfg.delta_ledger =
                    Some(PathBuf::from(args.next().unwrap_or_else(|| {
                        fatal(program, "--delta-ledger requires path")
                    })));
            }
            "--jobs" => {
                let n = args
                    .next()
                    .unwrap_or_else(|| fatal(program, "--jobs requires N"));
                cfg.jobs = n
                    .parse()
                    .unwrap_or_else(|_| fatal(program, &format!("bad --jobs {n}")));
            }
            "--air-sha256" | "--air" => {
                let h = args
                    .next()
                    .unwrap_or_else(|| fatal(program, "--air-sha256 requires a hash"));
                cfg.only_air = Some(normalize_hash(&h));
            }
            "--ledger-dir" => {
                let p = PathBuf::from(
                    args.next()
                        .unwrap_or_else(|| fatal(program, "--ledger-dir requires path")),
                );
                cfg.corpus_dir = p.clone();
                cfg.translate_ledger = p.join("metal2vulkan-ledger.jsonl");
                cfg.tech_ledger = p.join(backend.ledger_file_name());
                cfg.metal_ledger = p.join(RunBackend::Metal.ledger_file_name());
            }
            other if other.starts_with("--delta-ledger=") => {
                cfg.delta_ledger = Some(PathBuf::from(other.trim_start_matches("--delta-ledger=")));
            }
            other if other.starts_with("--jobs=") => {
                let n = other.trim_start_matches("--jobs=");
                cfg.jobs = n
                    .parse()
                    .unwrap_or_else(|_| fatal(program, &format!("bad --jobs {n}")));
            }
            other if other.starts_with("--status=") => {
                cfg.only_status = Some(other.trim_start_matches("--status=").to_string());
            }
            other if other.starts_with("--bucket=") => {
                cfg.only_bucket = Some(other.trim_start_matches("--bucket=").to_string());
            }
            other if other.starts_with("--contains=") => {
                cfg.contains = Some(other.trim_start_matches("--contains=").to_string());
            }
            other if !other.starts_with('-') && cfg.only_air.is_none() => {
                cfg.only_air = Some(normalize_hash(other));
            }
            other => fatal(program, &format!("unknown arg: {other}")),
        }
    }
    if cfg.oneshot && cfg.only_air.is_none() {
        fatal(program, "--oneshot requires --air-sha256 HEX");
    }
    if cfg.oneshot && cfg.delta_ledger.is_none() {
        fatal(program, "--oneshot requires --delta-ledger PATH");
    }
    Some(cfg)
}

fn case_timeout_secs_from_env(program: &str) -> u64 {
    let Some(raw) = std::env::var_os(CASE_TIMEOUT_ENV) else {
        return DEFAULT_CASE_TIMEOUT_SECS;
    };
    let raw = raw.to_string_lossy();
    let timeout = raw
        .parse::<u64>()
        .unwrap_or_else(|_| fatal(program, &format!("bad {CASE_TIMEOUT_ENV}={raw:?}")));
    if timeout == 0 {
        fatal(program, &format!("{CASE_TIMEOUT_ENV} must be >= 1"));
    }
    timeout
}

fn print_run_usage(program: &str) {
    eprintln!(
        "usage: {program} [--dry-run] [--force] [--failed-only] [--quiet] [--jobs N]\n\
                \t\t[--air-sha256 HEX] [--ledger-dir DIR]\n\
                \t\t[--status STATUS] [--bucket TEXT] [--contains TEXT]\n\
         \n\
         For each eligible translate-ledger row missing from this backend's JSONL\n\
         (or existing backend rows selected by --force / --failed-only / filters):\n\
         resolve/infer harness plan, seed non-zero inputs, run, append a delta result.\n\
         Metal: eligible if translate status is ok OR fallback (Metal is not the translator).\n\
         Vulkan/MoltenVK: eligible only if translate status is ok (need SPIR-V).\n\
         Metal banks plan+golden; vulkan/moltenvk compare to metal and record tolerance on the line.\n\
         --force          re-run rows even if a backend ledger row already exists\n\
         --failed-only    re-run existing non-success backend rows only\n\
         --status S       re-run existing backend rows with status S\n\
         --bucket TEXT    re-run existing backend rows whose failure bucket contains TEXT\n\
         --contains TEXT  re-run existing backend rows whose label/error/status/hash contains TEXT\n\
         --jobs N           parallel workers (default: min(CPU cores, 4))\n\
         {CASE_TIMEOUT_ENV}=N  kill a hung case after N seconds (default: {DEFAULT_CASE_TIMEOUT_SECS})\n\
         --oneshot          internal: run one --air-sha256 in-process (used by the parent)\n\
         --delta-ledger     internal: append-only per-run worker delta"
    );
}

fn fatal(program: &str, msg: &str) -> ! {
    eprintln!("{program}: {msg}");
    std::process::exit(64);
}

fn normalize_hash(s: &str) -> String {
    s.trim().trim_start_matches("0x").to_ascii_lowercase()
}

// --- ledger I/O ----------------------------------------------------------------------------------

/// Whether a translate-ledger row is eligible for this execution backend.
///
/// - **Metal** oracle runs AIR on Apple's runtime — translator FALLBACK does not block it.
/// - **Vulkan / MoltenVK** need emitted SPIR-V, so only translate `status=ok` is eligible.
pub fn translate_row_eligible(backend: RunBackend, status: &str) -> bool {
    match backend {
        RunBackend::Metal => status == "ok" || status == "fallback",
        RunBackend::Vulkan | RunBackend::MoltenVk => status == "ok",
    }
}

pub fn load_translate_rows(path: &Path, backend: RunBackend) -> Vec<TranslateRow> {
    let mut by_hash: HashMap<String, TranslateRow> = HashMap::new();
    let Ok(file) = File::open(path) else {
        return Vec::new();
    };
    for line in BufReader::new(file).lines().map_while(Result::ok) {
        let t = line.trim();
        if t.is_empty() || t.starts_with('#') {
            continue;
        }
        let Ok(row) = serde_json::from_str::<TranslateRow>(t) else {
            continue;
        };
        if translate_row_eligible(backend, &row.status) {
            by_hash.insert(row.air_sha256.clone(), row);
        }
    }
    let mut v: Vec<_> = by_hash.into_values().collect();
    v.sort_by(|a, b| a.label.cmp(&b.label).then(a.air_sha256.cmp(&b.air_sha256)));
    v
}

pub fn load_tech_keys(path: &Path) -> HashSet<String> {
    let mut keys = HashSet::new();
    let Ok(file) = File::open(path) else {
        return keys;
    };
    for line in BufReader::new(file).lines().map_while(Result::ok) {
        let t = line.trim();
        if t.is_empty() || t.starts_with('#') {
            continue;
        }
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(t) {
            if let Some(h) = v.get("air_sha256").and_then(|x| x.as_str()) {
                keys.insert(h.to_ascii_lowercase());
            }
        }
    }
    keys
}

#[derive(Debug, Clone)]
pub struct TechRowInfo {
    pub air_sha256: String,
    pub status: String,
    pub label: String,
    pub error: Option<String>,
    pub signature: String,
}

impl TechRowInfo {
    pub fn matches_text(&self, text: &str) -> bool {
        let text = text.to_ascii_lowercase();
        self.air_sha256.contains(&text)
            || self.status.to_ascii_lowercase().contains(&text)
            || self.label.to_ascii_lowercase().contains(&text)
            || self.signature.to_ascii_lowercase().contains(&text)
            || self
                .error
                .as_deref()
                .unwrap_or("")
                .to_ascii_lowercase()
                .contains(&text)
    }
}

pub fn load_tech_rows(path: &Path) -> HashMap<String, TechRowInfo> {
    let mut rows = HashMap::new();
    let Ok(file) = File::open(path) else {
        return rows;
    };
    for line in BufReader::new(file).lines().map_while(Result::ok) {
        let t = line.trim();
        if t.is_empty() || t.starts_with('#') {
            continue;
        }
        let Ok(v) = serde_json::from_str::<serde_json::Value>(t) else {
            continue;
        };
        let Some(h) = v.get("air_sha256").and_then(|x| x.as_str()) else {
            continue;
        };
        let air_sha256 = h.to_ascii_lowercase();
        let status = v
            .get("status")
            .and_then(|x| x.as_str())
            .unwrap_or("")
            .to_string();
        let label = v
            .get("label")
            .and_then(|x| x.as_str())
            .unwrap_or("")
            .to_string();
        let error = v.get("error").and_then(|x| x.as_str()).map(str::to_string);
        let has_tolerance = v.get("tolerance").is_some();
        let signature = execution_failure_signature(&status, error.as_deref(), has_tolerance);
        rows.insert(
            air_sha256.clone(),
            TechRowInfo {
                air_sha256,
                status,
                label,
                error,
                signature,
            },
        );
    }
    rows
}

pub fn execution_status_is_success(backend: RunBackend, status: &str) -> bool {
    match backend {
        RunBackend::Metal => status == "ok",
        RunBackend::Vulkan | RunBackend::MoltenVk => status == "ok" || status == "tolerance",
    }
}

pub fn execution_failure_signature(
    status: &str,
    error: Option<&str>,
    has_tolerance: bool,
) -> String {
    if let Some(error) = error {
        let first = error.lines().next().unwrap_or("").trim();
        if !first.is_empty() {
            return normalize_execution_error_signature(first);
        }
    }

    match status {
        "ok" => "ok".into(),
        "tolerance" => "within candidate tolerance".into(),
        "failure" if has_tolerance => "candidate output mismatch outside tolerance".into(),
        "failure" => "candidate output mismatch".into(),
        "missing" => "missing metal golden".into(),
        "quarantine" => "loop quarantine".into(),
        "timeout" => "worker timeout".into(),
        "fallback" => "fallback without error text".into(),
        "" => "missing status".into(),
        other => other.into(),
    }
}

fn normalize_execution_error_signature(first_line: &str) -> String {
    let mut s = first_line.trim();
    for prefix in [
        "vulkan execute panicked: ",
        "metal oracle panicked: ",
        "quarantined: ",
    ] {
        if let Some(rest) = s.strip_prefix(prefix) {
            s = rest.trim();
        }
    }
    for marker in [
        ": a validation error occurred",
        ": a non-validation error occurred",
        ": Validation Error:",
        ": VK_ERROR_",
        ": device lost",
    ] {
        if let Some((head, _)) = s.split_once(marker) {
            s = head.trim();
        }
    }
    s.to_string()
}

pub fn load_metal_rows(path: &Path) -> HashMap<String, MetalRow> {
    let mut map = HashMap::new();
    let Ok(file) = File::open(path) else {
        return map;
    };
    for line in BufReader::new(file).lines().map_while(Result::ok) {
        let t = line.trim();
        if t.is_empty() || t.starts_with('#') {
            continue;
        }
        if let Ok(row) = serde_json::from_str::<MetalRow>(t) {
            map.insert(row.air_sha256.to_ascii_lowercase(), row);
        }
    }
    map
}

/// Append one row to a per-run JSONL delta.
///
/// The expensive dedupe/rewrite happens once in the parent via [`merge_delta_into_ledger`].
pub fn append_jsonl_delta_row(path: &Path, value: &impl Serialize) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let v = serde_json::to_value(value).map_err(std::io::Error::other)?;
    let _air = v
        .get("air_sha256")
        .and_then(|x| x.as_str())
        .ok_or_else(|| std::io::Error::other("row missing air_sha256"))?
        .to_ascii_lowercase();

    let mut file = OpenOptions::new().create(true).append(true).open(path)?;
    let mut line = serde_json::to_string(&sort_json(v)).map_err(std::io::Error::other)?;
    line.push('\n');
    file.write_all(line.as_bytes())?;
    file.flush()?;
    Ok(())
}

fn append_result_row(cfg: &RunConfig, value: &impl Serialize) -> std::io::Result<()> {
    let path = cfg.delta_ledger.as_deref().unwrap_or(&cfg.tech_ledger);
    append_jsonl_delta_row(path, value)
}

pub fn merge_delta_into_ledger(ledger: &Path, delta: &Path) -> std::io::Result<usize> {
    let file = match File::open(delta) {
        Ok(file) => file,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(0),
        Err(e) => return Err(e),
    };
    if file.metadata()?.len() == 0 {
        return Ok(0);
    }

    if let Some(parent) = ledger.parent() {
        fs::create_dir_all(parent)?;
    }

    let mut by_hash = load_jsonl_objects_by_air_sha256(ledger)?;
    let mut n_delta = 0usize;
    for line in BufReader::new(file).lines() {
        let line = line?;
        let t = line.trim();
        if t.is_empty() || t.starts_with('#') {
            continue;
        }
        let Ok(v) = serde_json::from_str::<serde_json::Value>(t) else {
            continue;
        };
        let Some(air) = v.get("air_sha256").and_then(|x| x.as_str()) else {
            continue;
        };
        by_hash.insert(air.to_ascii_lowercase(), sort_json(v));
        n_delta += 1;
    }
    rewrite_jsonl_ledger(ledger, &by_hash)?;
    Ok(n_delta)
}

fn load_jsonl_objects_by_air_sha256(
    path: &Path,
) -> std::io::Result<HashMap<String, serde_json::Value>> {
    let mut by_hash = HashMap::new();
    let file = match File::open(path) {
        Ok(f) => f,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(by_hash),
        Err(e) => return Err(e),
    };
    for line in BufReader::new(file).lines() {
        let line = line?;
        let t = line.trim();
        if t.is_empty() || t.starts_with('#') {
            continue;
        }
        let Ok(v) = serde_json::from_str::<serde_json::Value>(t) else {
            continue;
        };
        let Some(air) = v.get("air_sha256").and_then(|x| x.as_str()) else {
            continue;
        };
        by_hash.insert(air.to_ascii_lowercase(), v);
    }
    Ok(by_hash)
}

fn rewrite_jsonl_ledger(
    path: &Path,
    by_hash: &HashMap<String, serde_json::Value>,
) -> std::io::Result<()> {
    let mut rows: Vec<&serde_json::Value> = by_hash.values().collect();
    rows.sort_by(|a, b| {
        let la = a.get("label").and_then(|x| x.as_str()).unwrap_or("");
        let lb = b.get("label").and_then(|x| x.as_str()).unwrap_or("");
        let ha = a.get("air_sha256").and_then(|x| x.as_str()).unwrap_or("");
        let hb = b.get("air_sha256").and_then(|x| x.as_str()).unwrap_or("");
        la.cmp(lb).then(ha.cmp(hb))
    });

    let tmp = path.with_extension("jsonl.tmp");
    {
        let mut f = File::create(&tmp)?;
        writeln!(
            f,
            "# metal2vulkan execution ledger — plan + digests + output_b64; no shader sources"
        )?;
        writeln!(
            f,
            "# unique by air_sha256 after per-run delta merge; last write wins for a given hash"
        )?;
        for row in rows {
            let line =
                serde_json::to_string(&sort_json(row.clone())).map_err(std::io::Error::other)?;
            writeln!(f, "{line}")?;
        }
        f.flush()?;
    }
    fs::rename(&tmp, path)?;
    Ok(())
}

fn metal_status_row(
    tr: &TranslateRow,
    src: &SourceFile,
    status: &str,
    compare: &str,
    error: Option<String>,
) -> MetalRow {
    MetalRow {
        air_sha256: tr.air_sha256.clone(),
        shard: src.shard.clone(),
        label: src.label.clone(),
        status: status.into(),
        backend: RunBackend::Metal.as_str().into(),
        seed_profile: SEED_PROFILE.into(),
        plan_version: PLAN_VERSION,
        plan: infer_plan(""),
        input_sha256: None,
        output_sha256: None,
        output_b64: None,
        spv_sha256: tr.spv_sha256.clone(),
        compare: compare.into(),
        stage: None,
        entry: None,
        error,
    }
}

fn candidate_status_row(
    cfg: &RunConfig,
    tr: &TranslateRow,
    src: &SourceFile,
    status: &str,
    golden_output_sha256: Option<String>,
    error: Option<String>,
) -> CandidateRow {
    CandidateRow {
        air_sha256: tr.air_sha256.clone(),
        shard: src.shard.clone(),
        label: src.label.clone(),
        status: status.into(),
        backend: cfg.backend.as_str().into(),
        output_sha256: None,
        output_b64: None,
        golden_output_sha256,
        spv_sha256: tr.spv_sha256.clone(),
        tolerance: None,
        observed: None,
        error,
    }
}

fn append_status_row(
    cfg: &RunConfig,
    tr: &TranslateRow,
    src: &SourceFile,
    status: &str,
    metal_compare: &str,
    golden_output_sha256: Option<String>,
    error: String,
) {
    match cfg.backend {
        RunBackend::Metal => {
            let row = metal_status_row(tr, src, status, metal_compare, Some(error));
            let _ = append_result_row(cfg, &row);
        }
        RunBackend::Vulkan | RunBackend::MoltenVk => {
            let row = candidate_status_row(cfg, tr, src, status, golden_output_sha256, Some(error));
            let _ = append_result_row(cfg, &row);
        }
    }
}

// --- plan inference ------------------------------------------------------------------------------

pub fn infer_plan(ll_or_meta_text: &str) -> HarnessPlan {
    let mut buffers = infer_buffers(ll_or_meta_text);
    let textures = infer_textures(ll_or_meta_text);
    if buffers.is_empty() {
        buffers.push(PlanBuffer {
            index: 0,
            len: DEFAULT_BUFFER_LEN,
            role: "InOut".into(),
            seed_tag: 1,
            seed_mode: SEED_MODE_DETERMINISTIC.into(),
        });
    }
    let stage = stage_from_ll(ll_or_meta_text);
    let output = if stage == Stage::Fragment {
        let format = fragment_render_target_format(ll_or_meta_text).unwrap_or("Rgba32Float");
        PlanOutput {
            kind: "render_target".into(),
            index: 0,
            format: format.into(),
            len: None,
            w: Some(DEFAULT_TEXTURE_EXTENT.width),
            h: Some(DEFAULT_TEXTURE_EXTENT.height),
            d: Some(1),
        }
    } else if let Some(texture) = textures
        .iter()
        .find(|t| t.role == "StorageWrite" || t.role == "StorageReadWrite")
    {
        PlanOutput {
            kind: "texture".into(),
            index: texture.index,
            format: texture.format.clone(),
            len: None,
            w: Some(texture.w),
            h: Some(texture.h),
            d: Some(texture.d),
        }
    } else {
        // Prefer a writable buffer as output; else first buffer.
        let out_idx = buffers
            .iter()
            .find(|b| b.role == "InOut" || b.role == "Output")
            .map(|b| b.index)
            .unwrap_or(buffers[0].index);
        let out_len = buffers
            .iter()
            .find(|b| b.index == out_idx)
            .map(|b| b.len)
            .unwrap_or(DEFAULT_BUFFER_LEN);
        PlanOutput {
            kind: "buffer".into(),
            index: out_idx,
            format: "RawBytes".into(),
            len: Some(out_len),
            w: None,
            h: None,
            d: None,
        }
    };

    HarnessPlan {
        buffers,
        textures,
        output,
        dispatch_grid: [64, 1, 1],
        dispatch_tg: [64, 1, 1],
    }
}

fn infer_buffers(ll: &str) -> Vec<PlanBuffer> {
    // Match AIR buffer metadata nodes: air.buffer ... air.buffer_size i32 N ... air.location_index i32 L
    let loop_bound_bufs = buffers_with_loads_used_as_loop_bounds(ll);
    let mut out = Vec::new();
    for line in ll.lines() {
        if !line.contains("air.buffer") || !line.contains("air.location_index") {
            continue;
        }
        if line.contains("air.texture") || line.contains("air.sampler") {
            continue;
        }
        // Threadgroup memory (addrspace 3) reuses location_index with device/constant buffers
        // (e.g. params @0 and shBlob @0). It is bound via setThreadgroupMemoryLength, not as a
        // seeded MTLBuffer — skip it so it cannot clobber control-param plan entries.
        if extract_i32_after(line, "air.address_space") == Some(3) {
            continue;
        }
        let fixed_size = extract_i32_after(line, "air.buffer_size");
        let size = fixed_size.unwrap_or(DEFAULT_BUFFER_LEN as i32);
        let loc = extract_i32_after(line, "air.location_index").unwrap_or(0);
        let role = if line.contains("air.read_write") {
            "InOut"
        } else if line.contains("!\"air.write\"") || line.contains("\"air.write\"") {
            "Output"
        } else {
            "Input"
        };
        let len = (size as usize).max(4);
        let loc_u = loc as u32;
        let seed_mode =
            if is_control_param_buffer_meta(line, fixed_size) || loop_bound_bufs.contains(&loc_u) {
                SEED_MODE_BOUNDED_CONTROL
            } else {
                SEED_MODE_DETERMINISTIC
            };
        out.push(PlanBuffer {
            index: loc_u,
            len,
            role: role.into(),
            seed_tag: loc_u.wrapping_add(1),
            seed_mode: seed_mode.into(),
        });
    }
    // Dedup by index. Prefer bounded_control / fixed-size control blobs over payload defaults
    // when two device/constant metas ever share a location (should be rare after skipping as3).
    let mut by_idx: HashMap<u32, PlanBuffer> = HashMap::new();
    for b in out {
        match by_idx.get(&b.index) {
            None => {
                by_idx.insert(b.index, b);
            }
            Some(prev) if prev.seed_mode == SEED_MODE_BOUNDED_CONTROL => {}
            Some(_) if b.seed_mode == SEED_MODE_BOUNDED_CONTROL => {
                by_idx.insert(b.index, b);
            }
            Some(prev) if prev.len < b.len && prev.len != DEFAULT_BUFFER_LEN => {}
            Some(_) => {
                by_idx.insert(b.index, b);
            }
        }
    }
    let mut v: Vec<_> = by_idx.into_values().collect();
    v.sort_by_key(|b| b.index);
    v
}

fn infer_textures(ll: &str) -> Vec<PlanTexture> {
    let mut out = Vec::new();
    for line in ll.lines() {
        if !line.contains("air.texture") || !line.contains("air.location_index") {
            continue;
        }
        let loc = extract_i32_after(line, "air.location_index").unwrap_or(0) as u32;
        let type_name = quoted_metadata_string_after(line, "air.arg_type_name");
        out.push(PlanTexture {
            index: loc,
            format: texture_format_from_air_type(type_name.as_deref()).into(),
            role: texture_role_from_air_meta(line).into(),
            w: DEFAULT_TEXTURE_EXTENT.width,
            h: DEFAULT_TEXTURE_EXTENT.height,
            d: texture_plan_depth(type_name.as_deref()),
            seed_tag: loc.wrapping_add(1),
        });
    }
    let mut by_idx: HashMap<u32, PlanTexture> = HashMap::new();
    for texture in out {
        match by_idx.get(&texture.index) {
            Some(prev) if texture_role_rank(&prev.role) >= texture_role_rank(&texture.role) => {}
            _ => {
                by_idx.insert(texture.index, texture);
            }
        }
    }
    let mut v: Vec<_> = by_idx.into_values().collect();
    v.sort_by_key(|t| t.index);
    v
}

fn texture_role_from_air_meta(line: &str) -> &'static str {
    if line.contains("air.read_write") {
        "StorageReadWrite"
    } else if line.contains("air.write") {
        "StorageWrite"
    } else if line.contains("air.read") {
        "StorageRead"
    } else {
        "Sampled"
    }
}

fn texture_role_rank(role: &str) -> u8 {
    match role {
        "StorageReadWrite" => 3,
        "StorageWrite" => 2,
        "StorageRead" => 1,
        _ => 0,
    }
}

fn texture_format_from_air_type(type_name: Option<&str>) -> &'static str {
    let Some(type_name) = type_name else {
        return "Rgba32Float";
    };
    if type_name.starts_with("depth") {
        return "R32Float";
    }
    let component = type_name
        .split_once('<')
        .and_then(|(_, rest)| rest.split([',', '>']).next())
        .map(str::trim)
        .unwrap_or("");
    match component {
        "half" => "Rgba16Float",
        "float" => "Rgba32Float",
        "ushort" => "Rgba16Uint",
        "uint" | "uchar" => "Rgba8Uint",
        "int" | "char" | "short" => "Rgba8Sint",
        _ => "Rgba32Float",
    }
}

fn texture_plan_depth(type_name: Option<&str>) -> u32 {
    match type_name {
        Some(name) if name.starts_with("texturecube<") => 6,
        _ => DEFAULT_TEXTURE_EXTENT.depth,
    }
}

fn fragment_render_target_format(ll: &str) -> Option<&'static str> {
    if let Some(meta) = metal2vulkan::meta::parse_air_fragment_meta(ll) {
        if let Some((_, member)) = meta
            .render_target_members
            .iter()
            .min_by_key(|(_, loc)| *loc)
        {
            return Some(fragment_output_format_from_air_type(
                meta.render_target_type_name(*member),
            ));
        }
    }
    if fragment_writes_depth(ll) {
        Some("Depth32Float")
    } else {
        None
    }
}

fn fragment_output_format_from_air_type(type_name: Option<&str>) -> &'static str {
    match type_name.unwrap_or("") {
        "half" => "R16Float",
        "half2" => "Rg16Float",
        "half3" | "half4" => "Rgba16Float",
        "float" => "R32Float",
        "float2" => "Rg32Float",
        "float3" | "float4" => "Rgba32Float",
        "uint" => "R32Uint",
        "uint2" => "Rg32Uint",
        "uint3" | "uint4" => "Rgba32Uint",
        "int" => "R32Sint",
        "int2" => "Rg32Sint",
        "int3" | "int4" => "Rgba32Sint",
        _ => "Rgba32Float",
    }
}

/// Structural: fixed-size, read-only buffer that looks like a control/params blob
/// (constant address space and/or AIR struct_type_info), not an unbounded payload array.
///
/// These commonly carry M/N/K / radius / iteration counts. Seeding them with
/// [`Seed::Deterministic`] produces multi-billion trip counts (GPU hang).
fn is_control_param_buffer_meta(line: &str, fixed_size: Option<i32>) -> bool {
    let Some(sz) = fixed_size else {
        return false;
    };
    if sz <= 0 || sz as usize > BOUNDED_CONTROL_MAX_BYTES {
        return false;
    }
    // Writable blobs are treated as payload even when small.
    if line.contains("air.read_write") {
        return false;
    }
    if line.contains("!\"air.write\"") || line.contains(", \"air.write\"") {
        return false;
    }
    let aspace = extract_i32_after(line, "air.address_space");
    let constant_as = aspace == Some(2);
    let has_struct = line.contains("air.struct_type_info");
    constant_as || has_struct
}

/// Lightweight IR scan: buffer locations whose `load i32` values appear in an `icmp` that
/// feeds a `br` (trip-count / early-out class). Used to catch device-space counters that
/// are not tagged as constant-param structs.
///
/// Not a full relooper — false positives only force small integer seeds on that buffer,
/// which is safe for execution harnesses (goldens re-derived under the new seed profile).
fn buffers_with_loads_used_as_loop_bounds(ll: &str) -> HashSet<u32> {
    let Some(body) = entry_function_body(ll) else {
        return HashSet::new();
    };
    // arg ordinal → buffer location_index from AIR kernel/vertex/fragment arg metadata.
    let arg_to_buf = arg_index_to_buffer_location(ll);
    if arg_to_buf.is_empty() {
        return HashSet::new();
    }

    // reg → buffer location, for values that are the arg pointer or GEP/bitcast of it.
    let mut ptr_buf: HashMap<&str, u32> = HashMap::new();
    // reg → buffer location for i32 loads from those pointers.
    let mut i32_from_buf: HashMap<&str, u32> = HashMap::new();
    // icmp reg → buffer if either side is an i32 loaded from that buffer.
    let mut icmp_from_buf: HashMap<&str, u32> = HashMap::new();
    let mut branched: HashSet<u32> = HashSet::new();

    for line in body.lines() {
        let line = line.trim();
        if let Some((reg, rhs)) = split_assign(line) {
            if rhs.starts_with("getelementptr") || rhs.starts_with("bitcast") {
                if let Some(buf) = first_buf_operand(rhs, &ptr_buf, &arg_to_buf) {
                    ptr_buf.insert(reg, buf);
                }
                continue;
            }
            // %N = load i32, ptr ... %p
            if let Some(rest) = rhs.strip_prefix("load i32") {
                if let Some(buf) = first_buf_operand(rest, &ptr_buf, &arg_to_buf) {
                    i32_from_buf.insert(reg, buf);
                }
                continue;
            }
            // Preserve provenance through integer casts into the cmp.
            if rhs.starts_with("zext ") || rhs.starts_with("sext ") || rhs.starts_with("trunc ") {
                if let Some(src) = first_percent_reg(rhs) {
                    if let Some(&buf) = i32_from_buf.get(src) {
                        i32_from_buf.insert(reg, buf);
                    }
                }
                continue;
            }
            if let Some(rest) = rhs.strip_prefix("icmp ") {
                let mut hit = None;
                for tok in rest.split([',', ' ']) {
                    let t = tok.trim();
                    if let Some(name) = t.strip_prefix('%') {
                        if let Some(&buf) = i32_from_buf.get(name) {
                            hit = Some(buf);
                            break;
                        }
                    }
                }
                if let Some(buf) = hit {
                    icmp_from_buf.insert(reg, buf);
                }
                continue;
            }
        }
        // br i1 %cmp — load used as branch condition (loop exit / early-out / grid check).
        if let Some(rest) = line.strip_prefix("br i1 ") {
            if let Some(cmp) = rest.split(',').next() {
                let cmp = cmp.trim().trim_start_matches('%');
                if let Some(&buf) = icmp_from_buf.get(cmp) {
                    branched.insert(buf);
                }
            }
        }
    }
    branched
}

fn entry_function_body(ll: &str) -> Option<&str> {
    let start = ll.find("\ndefine ")?;
    let after = &ll[start + 1..];
    let brace = after.find('{')?;
    let body_start = start + 1 + brace + 1;
    // Match the function's closing `}` at column 0 (AIR dumps use that convention).
    let rest = &ll[body_start..];
    let end = rest.find("\n}")?;
    Some(&rest[..end])
}

/// Map kernel argument ordinal → `air.location_index` for buffer args only.
fn arg_index_to_buffer_location(ll: &str) -> HashMap<usize, u32> {
    let mut map = HashMap::new();
    // Walk metadata lines that declare buffers; argument order is the first i32 in the node
    // (`!{i32 ARGORD, !"air.buffer", ...}`).
    for line in ll.lines() {
        if !line.contains("air.buffer") || !line.contains("air.location_index") {
            continue;
        }
        if line.contains("air.texture") || line.contains("air.sampler") {
            continue;
        }
        // !18 = !{i32 0, !"air.buffer", ...
        let Some(ord) = extract_meta_first_i32(line) else {
            continue;
        };
        let loc = extract_i32_after(line, "air.location_index").unwrap_or(ord) as u32;
        map.insert(ord as usize, loc);
    }
    map
}

fn extract_meta_first_i32(line: &str) -> Option<i32> {
    let idx = line.find("!{i32 ")?;
    let rest = &line[idx + "!{i32 ".len()..];
    let num: String = rest
        .chars()
        .take_while(|c| c.is_ascii_digit() || *c == '-')
        .collect();
    num.parse().ok()
}

fn split_assign(line: &str) -> Option<(&str, &str)> {
    let line = line.trim();
    if !line.starts_with('%') {
        return None;
    }
    let eq = line.find('=')?;
    let reg = line[1..eq].trim(); // without '%'
    let rhs = line[eq + 1..].trim();
    if reg.is_empty() {
        return None;
    }
    Some((reg, rhs))
}

fn first_percent_reg(s: &str) -> Option<&str> {
    let i = s.find('%')?;
    let rest = &s[i + 1..];
    let name: &str = rest
        .split(|c: char| !c.is_ascii_alphanumeric() && c != '.' && c != '_')
        .next()?;
    if name.is_empty() {
        None
    } else {
        Some(name)
    }
}

/// Resolve the first SSA operand that is a known buffer pointer (arg ordinal or tracked reg).
fn first_buf_operand(
    rhs: &str,
    ptr_buf: &HashMap<&str, u32>,
    arg_to_buf: &HashMap<usize, u32>,
) -> Option<u32> {
    // Prefer named regs that we already classified.
    let mut best: Option<u32> = None;
    for tok in rhs.split([',', ' ', '(', ')']) {
        let t = tok.trim();
        if let Some(name) = t.strip_prefix('%') {
            if let Some(&buf) = ptr_buf.get(name) {
                return Some(buf);
            }
            // Function args are `%0`, `%1`, ... (pure digits).
            if name.chars().all(|c| c.is_ascii_digit()) {
                if let Ok(ord) = name.parse::<usize>() {
                    if let Some(&buf) = arg_to_buf.get(&ord) {
                        best = Some(buf);
                    }
                }
            }
        }
    }
    best
}

/// Bytes for a bounded-control buffer: every 4-byte lane is `BOUNDED_CONTROL_DIM` as LE `u32`.
/// That makes M/N/K / row-bytes / counters small; float lanes become tiny subnormals (~0),
/// which is the safe beta=0 / disabled-scale path for MPS-style kernels.
pub fn bounded_control_buffer_bytes(len: usize) -> Vec<u8> {
    let mut out = vec![0u8; len];
    let dim = BOUNDED_CONTROL_DIM;
    let bytes = dim.to_le_bytes();
    let mut i = 0;
    while i + 4 <= len {
        out[i..i + 4].copy_from_slice(&bytes);
        i += 4;
    }
    // Tail: non-zero so we never leave a full-zero pad that some kernels treat as "unset".
    while i < len {
        out[i] = (dim as u8).max(1);
        i += 1;
    }
    out
}

fn extract_i32_after(line: &str, key: &str) -> Option<i32> {
    let idx = line.find(key)?;
    let rest = &line[idx + key.len()..];
    // patterns: ,"air.buffer_size", i32 4,  or  !"air.buffer_size", i32 4
    let mut found_i32 = false;
    for tok in rest.split([',', ' ', '!', '"']) {
        let t = tok.trim();
        if t.is_empty() {
            continue;
        }
        if t == "i32" {
            found_i32 = true;
            continue;
        }
        if found_i32 {
            if let Ok(n) = t.parse::<i32>() {
                return Some(n);
            }
            found_i32 = false;
        }
    }
    None
}

fn quoted_metadata_string_after(line: &str, key: &str) -> Option<String> {
    let idx = line.find(key)?;
    let rest = &line[idx + key.len()..];
    let marker = "!\"";
    let start = rest.find(marker)? + marker.len();
    let rest = &rest[start..];
    let end = rest.find('"')?;
    Some(rest[..end].to_string())
}

// --- plan → Inputs (leak for 'static) ------------------------------------------------------------

pub struct OwnedInputs {
    pub inputs: Inputs,
    // keep ownership so leak targets stay valid for the call duration via 'static on Inputs
    _buffers: &'static [BufferInput],
    _textures: &'static [TextureInput],
}

pub fn plan_to_owned_inputs(plan: &HarnessPlan) -> Result<OwnedInputs, String> {
    let buffers: Vec<BufferInput> = plan
        .buffers
        .iter()
        .map(|b| {
            let seed = if b.seed_mode == SEED_MODE_BOUNDED_CONTROL {
                let bytes = bounded_control_buffer_bytes(b.len);
                let leaked: &'static [u8] = Box::leak(bytes.into_boxed_slice());
                Seed::ExactBytes {
                    bytes: leaked,
                    reason: "bounded_control_param_buffer",
                }
            } else {
                Seed::Deterministic { tag: b.seed_tag }
            };
            BufferInput {
                index: b.index,
                len: b.len,
                role: parse_buffer_role(&b.role),
                seed,
            }
        })
        .collect();
    let textures: Vec<TextureInput> = plan
        .textures
        .iter()
        .map(|t| {
            Ok(TextureInput {
                index: t.index,
                format: parse_format(&t.format)?,
                extent: Extent3d::new(t.w.max(1), t.h.max(1), t.d.max(1)),
                role: parse_texture_role(&t.role),
                seed: Seed::Deterministic { tag: t.seed_tag },
            })
        })
        .collect::<Result<Vec<_>, String>>()?;

    let output = plan_output_to_output(&plan.output)?;
    let buffers = Box::leak(buffers.into_boxed_slice());
    let textures = Box::leak(textures.into_boxed_slice());
    let inputs = Inputs::new(
        buffers,
        textures,
        output,
        Dispatch {
            threads_per_grid: plan.dispatch_grid,
            threads_per_threadgroup: plan.dispatch_tg,
        },
        Render::fullscreen_triangle(8, 8),
    );
    Ok(OwnedInputs {
        inputs,
        _buffers: buffers,
        _textures: textures,
    })
}

fn plan_output_to_output(o: &PlanOutput) -> Result<Output, String> {
    let format = parse_format(&o.format)?;
    match o.kind.as_str() {
        "buffer" => Ok(Output::Buffer {
            index: o.index,
            format,
            len: o.len.unwrap_or(DEFAULT_BUFFER_LEN),
        }),
        "texture" => Ok(Output::Texture {
            index: o.index,
            format,
            extent: Extent3d::new(
                o.w.unwrap_or(8).max(1),
                o.h.unwrap_or(8).max(1),
                o.d.unwrap_or(1).max(1),
            ),
        }),
        "render_target" | "RenderTarget" => Ok(Output::RenderTarget {
            format,
            extent: Extent3d::new(
                o.w.unwrap_or(DEFAULT_TEXTURE_EXTENT.width).max(1),
                o.h.unwrap_or(DEFAULT_TEXTURE_EXTENT.height).max(1),
                o.d.unwrap_or(1).max(1),
            ),
        }),
        other => Err(format!("unsupported output kind {other}")),
    }
}

fn parse_buffer_role(s: &str) -> BufferRole {
    match s {
        "Input" | "input" => BufferRole::Input,
        "Output" | "output" => BufferRole::Output,
        _ => BufferRole::InOut,
    }
}

fn parse_texture_role(s: &str) -> TextureRole {
    match s {
        "StorageWrite" => TextureRole::StorageWrite,
        "StorageReadWrite" => TextureRole::StorageReadWrite,
        "StorageRead" => TextureRole::StorageRead,
        "ColorTarget" => TextureRole::ColorTarget,
        "InputAttachment" => TextureRole::InputAttachment,
        _ => TextureRole::Sampled,
    }
}

fn parse_format(s: &str) -> Result<DataFormat, String> {
    Ok(match s {
        "RawBytes" => DataFormat::RawBytes,
        "U32" => DataFormat::U32,
        "I32" => DataFormat::I32,
        "F32" => DataFormat::F32,
        "Rgba8Unorm" => DataFormat::Rgba8Unorm,
        "Rgba8Uint" => DataFormat::Rgba8Uint,
        "Rgba8Sint" => DataFormat::Rgba8Sint,
        "Rgba16Uint" => DataFormat::Rgba16Uint,
        "R32Uint" => DataFormat::R32Uint,
        "Rg32Uint" => DataFormat::Rg32Uint,
        "Rgba32Uint" => DataFormat::Rgba32Uint,
        "R32Sint" => DataFormat::R32Sint,
        "Rg32Sint" => DataFormat::Rg32Sint,
        "Rgba32Sint" => DataFormat::Rgba32Sint,
        "R16Float" => DataFormat::R16Float,
        "Rg16Float" => DataFormat::Rg16Float,
        "Rgba16Float" => DataFormat::Rgba16Float,
        "Rg32Float" => DataFormat::Rg32Float,
        "Rgba32Float" => DataFormat::Rgba32Float,
        "R32Float" => DataFormat::R32Float,
        "Depth32Float" => DataFormat::Depth32Float,
        "Depth24Stencil8" => DataFormat::Depth24Stencil8,
        other => return Err(format!("unknown DataFormat {other}")),
    })
}

pub fn input_digest(plan: &HarnessPlan) -> Result<String, String> {
    let owned = plan_to_owned_inputs(plan)?;
    let mut bytes = Vec::new();
    for buffer in owned.inputs.buffers {
        bytes.extend_from_slice(&buffer.index.to_le_bytes());
        bytes.extend_from_slice(&(buffer.len as u64).to_le_bytes());
        bytes.extend_from_slice(&seeded_buffer_bytes(buffer));
    }
    for texture in owned.inputs.textures {
        bytes.extend_from_slice(&texture.index.to_le_bytes());
        bytes.extend_from_slice(&seeded_texture_bytes(texture));
    }
    Ok(sha256_hex(&bytes))
}

// --- compare -------------------------------------------------------------------------------------

#[derive(Debug)]
pub struct CompareResult {
    pub status: String,
    pub observed: Option<ObservedMargins>,
    pub tolerance: Option<ToleranceSpecJson>,
}

/// Compare candidate bytes to a banked metal golden row.
///
/// Prefers full `output_b64` (enables float AbsAndUlp and observed margins). Falls back to
/// `output_sha256` equality when older metal rows lack payloads.
pub fn compare_candidate_to_metal(
    candidate: &[u8],
    metal: &MetalRow,
    out_hash: &str,
    golden_hash: &str,
    format: DataFormat,
) -> (String, Option<ObservedMargins>, Option<ToleranceSpecJson>) {
    if let Some(b64) = metal.output_b64.as_deref() {
        match decode_output_b64(b64) {
            Ok(golden) => {
                let policy = if format.is_float_like() {
                    Some(default_float_tolerance())
                } else {
                    None
                };
                let result = compare_to_golden(candidate, &golden, format, policy.as_ref());
                return (result.status, result.observed, result.tolerance);
            }
            Err(_) => {
                // Corrupt payload: fall through to hash compare.
            }
        }
    }
    if !golden_hash.is_empty() && out_hash == golden_hash {
        ("ok".to_string(), None, None)
    } else {
        // No usable golden bytes → cannot classify tolerance; hash mismatch is failure.
        ("failure".to_string(), None, None)
    }
}

/// Exact compare by default. Optional AbsAndUlp defaults unused unless `policy` is set.
pub fn compare_to_golden(
    candidate: &[u8],
    golden: &[u8],
    format: DataFormat,
    policy: Option<&ToleranceSpecJson>,
) -> CompareResult {
    if candidate == golden {
        return CompareResult {
            status: "ok".into(),
            observed: None,
            tolerance: None,
        };
    }
    if candidate.len() != golden.len() {
        return CompareResult {
            status: "failure".into(),
            observed: None,
            tolerance: policy.cloned(),
        };
    }
    let Some(pol) = policy else {
        return CompareResult {
            status: "failure".into(),
            observed: None,
            tolerance: None,
        };
    };
    let (max_abs, max_ulp) = simple_margins(candidate, golden, format);
    let within = match pol.kind.as_str() {
        "Abs" => max_abs <= pol.max_abs.unwrap_or(0.0),
        "Ulp" => max_ulp <= pol.max_ulp.unwrap_or(0),
        "AbsAndUlp" => max_abs <= pol.max_abs.unwrap_or(0.0) && max_ulp <= pol.max_ulp.unwrap_or(0),
        _ => false,
    };
    CompareResult {
        status: if within {
            "tolerance".into()
        } else {
            "failure".into()
        },
        observed: Some(ObservedMargins {
            max_abs: Some(max_abs),
            max_ulp: Some(max_ulp),
        }),
        tolerance: Some(pol.clone()),
    }
}

fn simple_margins(candidate: &[u8], golden: &[u8], format: DataFormat) -> (f32, u32) {
    if !format.is_float_like() || candidate.len() < 4 {
        let diffs = candidate
            .iter()
            .zip(golden.iter())
            .filter(|(a, b)| a != b)
            .count();
        return (diffs as f32, diffs as u32);
    }
    let mut max_abs = 0.0f32;
    let mut max_ulp = 0u32;
    for (c, g) in candidate.chunks_exact(4).zip(golden.chunks_exact(4)) {
        let cf = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
        let gf = f32::from_le_bytes([g[0], g[1], g[2], g[3]]);
        max_abs = max_abs.max((cf - gf).abs());
        let cu = u32::from_le_bytes([c[0], c[1], c[2], c[3]]);
        let gu = u32::from_le_bytes([g[0], g[1], g[2], g[3]]);
        max_ulp = max_ulp.max(cu.abs_diff(gu));
    }
    (max_abs, max_ulp)
}

// --- run one case --------------------------------------------------------------------------------

/// Run the driver. Returns process exit code (0 = ok / nothing; 1 = failures; oneshot: 0/1/2).
pub fn run_driver(cfg: &RunConfig) -> i32 {
    // Always capture panics with a full backtrace (parent + inherited by oneshot workers).
    // Do not override a more verbose value (e.g. `full`) if the caller already set one.
    match std::env::var_os("RUST_BACKTRACE") {
        None => std::env::set_var("RUST_BACKTRACE", "1"),
        Some(v) if v.is_empty() => std::env::set_var("RUST_BACKTRACE", "1"),
        Some(_) => {}
    }

    let program = cfg.backend.program_name();
    let jobs = cfg.jobs.max(1);
    eprintln!("# {program}");
    eprintln!("# translate {}", cfg.translate_ledger.display());
    eprintln!("# tech      {}", cfg.tech_ledger.display());
    eprintln!("# metal     {}", cfg.metal_ledger.display());
    eprintln!("# local     {}", cfg.local_corpus.display());
    eprintln!("# jobs      {jobs}");
    eprintln!("# timeout   {}s per case", cfg.timeout_secs);
    if cfg.oneshot {
        eprintln!("# oneshot   (in-process worker)");
    }
    if cfg.dry_run {
        eprintln!("# dry-run");
    }

    let translate = load_translate_rows(&cfg.translate_ledger, cfg.backend);
    let tech_row_filters = cfg.failed_only
        || cfg.only_status.is_some()
        || cfg.only_bucket.is_some()
        || cfg.contains.is_some();
    let existing_rows = if tech_row_filters {
        load_tech_rows(&cfg.tech_ledger)
    } else {
        HashMap::new()
    };
    let existing = if cfg.force || cfg.oneshot || tech_row_filters {
        // Oneshot: parent already decided eligibility; always attempt the case.
        HashSet::new()
    } else {
        load_tech_keys(&cfg.tech_ledger)
    };
    let metal_rows = load_metal_rows(&cfg.metal_ledger);

    let mut todo: Vec<TranslateRow> = translate
        .into_iter()
        .filter(|r| {
            if let Some(ref only) = cfg.only_air {
                &r.air_sha256 == only
            } else {
                true
            }
        })
        .filter(|r| {
            if tech_row_filters {
                return existing_rows
                    .get(&r.air_sha256)
                    .is_some_and(|row| tech_row_selected(cfg, row));
            }
            cfg.force || cfg.oneshot || !existing.contains(&r.air_sha256)
        })
        .collect();
    todo.sort_by(|a, b| a.label.cmp(&b.label));

    let eligible_hint = match cfg.backend {
        RunBackend::Metal => "translate status ok|fallback",
        RunBackend::Vulkan | RunBackend::MoltenVk => "translate status ok only",
    };
    eprintln!(
        "# eligible: {} ({eligible_hint}; force={} only={:?})",
        todo.len(),
        cfg.force,
        cfg.only_air
    );
    if cfg.failed_only {
        eprintln!("# failed-only: existing non-success rows only");
    }
    if let Some(status) = cfg.only_status.as_deref() {
        eprintln!("# filter-status: {status}");
    }
    if let Some(bucket) = cfg.only_bucket.as_deref() {
        eprintln!("# filter-bucket: {bucket}");
    }
    if let Some(text) = cfg.contains.as_deref() {
        eprintln!("# filter-contains: {text}");
    }
    if todo.is_empty() {
        eprintln!("# nothing to do");
        return 0;
    }
    if cfg.dry_run {
        for r in &todo {
            eprintln!("  would-run {} {}", r.air_sha256, r.label);
        }
        eprintln!("# RESULT: dry-run {}", todo.len());
        return 0;
    }

    // Oneshot worker: run in-process (parent applies the wall timeout around this process).
    if cfg.oneshot {
        let row = &todo[0];
        let outcome = process_one(cfg, row, &metal_rows);
        eprintln!(
            "# oneshot {:?} {} {}",
            outcome,
            row.air_sha256.get(..12).unwrap_or(&row.air_sha256),
            row.label
        );
        return match outcome {
            ProcessOutcome::Ok => 0,
            ProcessOutcome::Fail | ProcessOutcome::Timeout => 1,
            ProcessOutcome::Skip => 2,
        };
    }

    // Parent owns the worker process groups. Ctrl-C / SIGTERM must SIGKILL them — workers
    // are started with process_group(0), so the terminal does not forward SIGINT to them.
    install_worker_signal_handlers();

    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::thread;

    let delta_ledger = run_delta_path(cfg);
    let _ = fs::remove_file(&delta_ledger);
    let mut run_cfg = cfg.clone();
    run_cfg.delta_ledger = Some(delta_ledger.clone());
    eprintln!("# delta     {}", delta_ledger.display());

    let n_total = todo.len();
    let workers = jobs.min(n_total).max(1);
    let n_ok = AtomicUsize::new(0);
    let n_fail = AtomicUsize::new(0);
    let n_skip = AtomicUsize::new(0);
    let n_timeout = AtomicUsize::new(0);
    let done = AtomicUsize::new(0);
    let chunk_size = n_total.div_ceil(workers);

    eprintln!(
        "# workers={workers} cases={n_total} timeout={}s (subprocess per case)",
        cfg.timeout_secs
    );

    thread::scope(|scope| {
        for chunk in todo.chunks(chunk_size) {
            let run_cfg = &run_cfg;
            let n_ok = &n_ok;
            let n_fail = &n_fail;
            let n_skip = &n_skip;
            let n_timeout = &n_timeout;
            let done = &done;
            scope.spawn(move || {
                for row in chunk {
                    let outcome = run_case_subprocess(run_cfg, row);
                    match outcome {
                        ProcessOutcome::Ok => {
                            n_ok.fetch_add(1, Ordering::Relaxed);
                        }
                        ProcessOutcome::Fail => {
                            n_fail.fetch_add(1, Ordering::Relaxed);
                        }
                        ProcessOutcome::Skip => {
                            n_skip.fetch_add(1, Ordering::Relaxed);
                        }
                        ProcessOutcome::Timeout => {
                            n_timeout.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                    let i = done.fetch_add(1, Ordering::Relaxed) + 1;
                    if cfg.quiet {
                        if i == 1 || i == n_total || i.is_multiple_of(25) {
                            eprintln!("  [{i}/{n_total}] …");
                        }
                    } else {
                        eprintln!(
                            "  [{i}/{n_total}] {} {}  {:?}",
                            row.air_sha256.get(..12).unwrap_or(&row.air_sha256),
                            row.label,
                            outcome
                        );
                    }
                }
            });
        }
    });

    let timeouts = n_timeout.load(Ordering::Relaxed);
    let merged = match merge_delta_into_ledger(&cfg.tech_ledger, &delta_ledger) {
        Ok(n) => {
            let _ = fs::remove_file(&delta_ledger);
            n
        }
        Err(e) => {
            eprintln!(
                "# RESULT: failed to merge delta {} into {}: {e}",
                delta_ledger.display(),
                cfg.tech_ledger.display()
            );
            return 1;
        }
    };
    eprintln!(
        "# RESULT: ok={} fail={} skip={} timeout={timeouts} merged_delta={merged} → {}",
        n_ok.load(Ordering::Relaxed),
        n_fail.load(Ordering::Relaxed),
        n_skip.load(Ordering::Relaxed),
        cfg.tech_ledger.display()
    );
    if n_fail.load(Ordering::Relaxed) > 0 || timeouts > 0 {
        1
    } else {
        0
    }
}

fn run_delta_path(cfg: &RunConfig) -> PathBuf {
    let file = format!(
        "m2v-{}-{}-{}.delta.jsonl",
        cfg.backend.as_str(),
        std::process::id(),
        chrono_free_timestamp()
    );
    std::env::temp_dir().join(file)
}

fn chrono_free_timestamp() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0)
}

fn tech_row_selected(cfg: &RunConfig, row: &TechRowInfo) -> bool {
    if cfg.failed_only && execution_status_is_success(cfg.backend, &row.status) {
        return false;
    }
    if let Some(status) = cfg.only_status.as_deref() {
        if row.status != status {
            return false;
        }
    }
    if let Some(bucket) = cfg.only_bucket.as_deref() {
        if !row
            .signature
            .to_ascii_lowercase()
            .contains(&bucket.to_ascii_lowercase())
        {
            return false;
        }
    }
    if let Some(text) = cfg.contains.as_deref() {
        if !row.matches_text(text) {
            return false;
        }
    }
    true
}

/// Spawn this binary in `--oneshot` mode for one hash; kill after `timeout_secs`.
fn run_case_subprocess(cfg: &RunConfig, row: &TranslateRow) -> ProcessOutcome {
    use std::process::{Command, Stdio};
    use std::time::{Duration, Instant};

    let exe = match std::env::current_exe() {
        Ok(p) => p,
        Err(e) => {
            write_timeout_or_error_row(cfg, row, &format!("current_exe: {e}"), "fallback");
            return ProcessOutcome::Fail;
        }
    };

    let mut cmd = Command::new(&exe);
    cmd.arg("--oneshot")
        .arg("--air-sha256")
        .arg(&row.air_sha256)
        .arg("--ledger-dir")
        .arg(&cfg.corpus_dir);
    if let Some(delta) = cfg.delta_ledger.as_ref() {
        cmd.arg("--delta-ledger").arg(delta);
    }
    // Explicit so workers keep panics useful even if the parent env is scrubbed.
    let backtrace = std::env::var("RUST_BACKTRACE").unwrap_or_else(|_| "1".into());
    cmd.env("RUST_BACKTRACE", backtrace);
    if cfg.force {
        cmd.arg("--force");
    }
    if cfg.quiet {
        cmd.arg("--quiet");
    }
    // New process group so a timeout / Ctrl-C can kill metal-as / helper descendants.
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        cmd.process_group(0);
    }

    cmd.stdin(Stdio::null());
    if cfg.quiet {
        cmd.stdout(Stdio::null()).stderr(Stdio::null());
    } else {
        cmd.stdout(Stdio::inherit()).stderr(Stdio::inherit());
    }

    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => {
            write_timeout_or_error_row(cfg, row, &format!("spawn worker: {e}"), "fallback");
            return ProcessOutcome::Fail;
        }
    };

    // Track for Ctrl-C / SIGTERM (process groups are not in the terminal FG group).
    let _live = register_live_worker(child.id());

    let timeout = Duration::from_secs(cfg.timeout_secs);
    let slow_after = Duration::from_secs(SLOW_CASE_SECS);
    let start = Instant::now();
    let mut logged_slow = false;
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                let elapsed = start.elapsed();
                // Log on finish if slow and we did not already flag it while running.
                if elapsed >= slow_after && !logged_slow {
                    eprintln!(
                        "# SLOW {} finished in {:.1}s — slow",
                        row.air_sha256,
                        elapsed.as_secs_f64()
                    );
                } else if logged_slow {
                    eprintln!(
                        "# SLOW {} finished in {:.1}s",
                        row.air_sha256,
                        elapsed.as_secs_f64()
                    );
                }
                return if status.success() {
                    ProcessOutcome::Ok
                } else if status.code() == Some(2) {
                    ProcessOutcome::Skip
                } else {
                    ProcessOutcome::Fail
                };
            }
            Ok(None) if start.elapsed() >= timeout => {
                if !logged_slow {
                    eprintln!(
                        "# SLOW {} still running after {}s — slow",
                        row.air_sha256, SLOW_CASE_SECS
                    );
                }
                eprintln!(
                    "# SLOW {} timed out after {}s",
                    row.air_sha256, cfg.timeout_secs
                );
                // Honest warning: SIGKILL frees the CPU worker but cannot cancel an in-flight GPU
                // kernel. The pre-submission loop-budget guard should make this unreachable for a
                // GPU loop; reaching it means either a CPU-side tool hang or (if the machine is
                // unresponsive) a wedged GPU that only a reboot recovers.
                eprintln!(
                    "# WARN {} exceeded {}s — killing the CPU worker, but a committed Metal kernel \
                     cannot be cancelled. If the machine is unresponsive, the GPU is wedged and a \
                     reboot is the only recovery.",
                    row.air_sha256, cfg.timeout_secs
                );
                kill_worker(&mut child);
                let _ = child.wait();
                write_timeout_or_error_row(
                    cfg,
                    row,
                    &format!("timeout after {}s", cfg.timeout_secs),
                    "timeout",
                );
                return ProcessOutcome::Timeout;
            }
            Ok(None) => {
                if !logged_slow && start.elapsed() >= slow_after {
                    logged_slow = true;
                    eprintln!(
                        "# SLOW {} still running after {}s — slow",
                        row.air_sha256, SLOW_CASE_SECS
                    );
                }
                std::thread::sleep(Duration::from_millis(100));
            }
            Err(e) => {
                kill_worker(&mut child);
                let _ = child.wait();
                write_timeout_or_error_row(cfg, row, &format!("wait worker: {e}"), "fallback");
                return ProcessOutcome::Fail;
            }
        }
    }
}

fn kill_worker(child: &mut std::process::Child) {
    #[cfg(unix)]
    {
        let pid = child.id() as i32;
        // Kill the whole process group (negative pid) — metal-as / helpers included. This reclaims
        // the CPU worker but does NOT cancel an in-flight GPU command buffer (a committed Metal
        // kernel cannot be cancelled; only a reboot recovers a wedged GPU). Infinite-loop safety
        // is enforced before submission by the loop-budget guard, not here.
        // SAFETY: pid is this child's process-group id (we spawned with process_group(0)).
        unsafe {
            libc::kill(-pid, libc::SIGKILL);
        }
    }
    let _ = child.kill();
}

// --- Live worker registry (Ctrl-C / SIGTERM) ------------------------------------------------------
//
// Workers use `process_group(0)`, so they are *not* in the terminal foreground group and do not
// receive SIGINT when the user hits Ctrl-C. Without an explicit kill of each process group, the
// parent exits and oneshot children (and metal-as / helpers under them) keep running.

#[cfg(unix)]
const LIVE_WORKER_SLOTS: usize = 256;

#[cfg(unix)]
static LIVE_WORKER_PIDS: [std::sync::atomic::AtomicI32; LIVE_WORKER_SLOTS] =
    [const { std::sync::atomic::AtomicI32::new(0) }; LIVE_WORKER_SLOTS];

#[cfg(unix)]
static WORKER_HANDLERS_INSTALLED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// RAII slot in [`LIVE_WORKER_PIDS`]; cleared on drop so the signal handler never kills a reused pid.
struct LiveWorker {
    #[cfg(unix)]
    pid: i32,
}

impl Drop for LiveWorker {
    fn drop(&mut self) {
        #[cfg(unix)]
        {
            use std::sync::atomic::Ordering;
            for slot in &LIVE_WORKER_PIDS {
                let _ = slot.compare_exchange(self.pid, 0, Ordering::SeqCst, Ordering::SeqCst);
            }
        }
    }
}

fn register_live_worker(pid: u32) -> LiveWorker {
    #[cfg(unix)]
    {
        use std::sync::atomic::Ordering;
        let pid = pid as i32;
        if pid > 0 {
            for slot in &LIVE_WORKER_PIDS {
                if slot
                    .compare_exchange(0, pid, Ordering::SeqCst, Ordering::SeqCst)
                    .is_ok()
                {
                    return LiveWorker { pid };
                }
            }
            // Slot table full: still return a guard so Drop is a no-op for untracked pid;
            // the process group remains killable via the Child handle on timeout.
            eprintln!("# warn: live-worker table full; pid {pid} not tracked for Ctrl-C");
            return LiveWorker { pid: 0 };
        }
        LiveWorker { pid: 0 }
    }
    #[cfg(not(unix))]
    {
        let _ = pid;
        LiveWorker {}
    }
}

/// SIGKILL every registered worker process group. Async-signal-safe (atomics + kill only).
#[cfg(unix)]
fn kill_all_live_workers() {
    use std::sync::atomic::Ordering;
    for slot in &LIVE_WORKER_PIDS {
        let pid = slot.swap(0, Ordering::SeqCst);
        if pid > 0 {
            // SAFETY: pid is a process-group id we registered after process_group(0) spawn.
            unsafe {
                libc::kill(-pid, libc::SIGKILL);
            }
        }
    }
}

#[cfg(unix)]
extern "C" fn on_parent_interrupt(sig: libc::c_int) {
    kill_all_live_workers();
    // Restore default disposition and re-raise so the shell sees the usual 128+sig exit.
    unsafe {
        libc::signal(sig, libc::SIG_DFL);
        libc::raise(sig);
    }
}

fn install_worker_signal_handlers() {
    #[cfg(unix)]
    {
        use std::sync::atomic::Ordering;
        if WORKER_HANDLERS_INSTALLED.swap(true, Ordering::SeqCst) {
            return;
        }
        // SAFETY: handler only uses atomics + kill/signal/raise (async-signal-safe).
        unsafe {
            let handler = on_parent_interrupt as *const () as libc::sighandler_t;
            libc::signal(libc::SIGINT, handler);
            libc::signal(libc::SIGTERM, handler);
        }
    }
}

fn write_timeout_or_error_row(cfg: &RunConfig, tr: &TranslateRow, err: &str, status: &str) {
    // Prefer a real source label if still in shards; else ledger label.
    let src = resolve_source(
        &tr.air_sha256,
        &tr.label,
        &tr.kind,
        tr.shard.as_deref(),
        &cfg.public_dir,
        &cfg.local_corpus,
    );
    let label = src
        .as_ref()
        .map(|s| s.label.clone())
        .unwrap_or_else(|| tr.label.clone());
    let kind_label_src = SourceFile {
        label,
        kind: src
            .as_ref()
            .map(|s| s.kind.clone())
            .unwrap_or_else(|| tr.kind.clone()),
        air_sha256: tr.air_sha256.clone(),
        shard: src
            .as_ref()
            .and_then(|s| s.shard.clone())
            .or_else(|| tr.shard.clone()),
        air_ll: src.as_ref().map(|s| s.air_ll.clone()).unwrap_or_default(),
        blob_b64: src.as_ref().and_then(|s| s.blob_b64.clone()),
        lib: src.as_ref().and_then(|s| s.lib.clone()),
        lib_sha256: src.as_ref().and_then(|s| s.lib_sha256.clone()),
        public_path: src.as_ref().and_then(|s| s.public_path.clone()),
    };
    append_status_row(cfg, tr, &kind_label_src, status, "full", None, err.into());
}

#[derive(Debug)]
enum ProcessOutcome {
    Ok,
    Fail,
    Skip,
    Timeout,
}

fn process_one(
    cfg: &RunConfig,
    tr: &TranslateRow,
    metal_rows: &HashMap<String, MetalRow>,
) -> ProcessOutcome {
    let Some(src) = resolve_source(
        &tr.air_sha256,
        &tr.label,
        &tr.kind,
        tr.shard.as_deref(),
        &cfg.public_dir,
        &cfg.local_corpus,
    ) else {
        eprintln!("    skip: source not in shards for {}", tr.air_sha256);
        return ProcessOutcome::Skip;
    };

    let ll = match load_ll_text(&src) {
        Ok(t) => t,
        Err(e) => {
            write_failure_row(cfg, tr, &src, &format!("load ll: {e}"));
            return ProcessOutcome::Fail;
        }
    };
    let stage = stage_from_ll(&ll);
    let entry = entry_name_from_ll(&ll).unwrap_or_else(|| "unknown".into());

    // Metal oracle: always re-infer the harness plan from current AIR + seed rules.
    // Reusing a banked metal.plan freezes bugs (e.g. threadgroup meta clobbering constant
    // params → deterministic M/N/K / numTopK → multi-minute GPU work). The plan written into
    // the new metal row is still the one used for this run (and for candidate input matching).
    if cfg.backend == RunBackend::Metal {
        if let Some(m) = metal_rows.get(&tr.air_sha256) {
            if m.status == "ok" && m.compare == "none" {
                eprintln!("    skip: metal compare=none");
                return ProcessOutcome::Skip;
            }
        }
        let plan = infer_plan(&ll);
        return run_metal(cfg, tr, &src, &ll, stage, &entry, &plan);
    }

    // Vulkan / MoltenVK candidates: inputs must match the metal golden's banked plan.
    let Some(metal) = metal_rows.get(&tr.air_sha256) else {
        let row = candidate_status_row(
            cfg,
            tr,
            &src,
            "missing",
            None,
            Some("no metal golden row".into()),
        );
        let _ = append_result_row(cfg, &row);
        return ProcessOutcome::Fail;
    };
    if metal.status == "quarantine" {
        let row = candidate_status_row(
            cfg,
            tr,
            &src,
            "quarantine",
            metal.output_sha256.clone(),
            Some("metal status=quarantine".into()),
        );
        let _ = append_result_row(cfg, &row);
        return ProcessOutcome::Fail;
    }
    if metal.status != "ok" || metal.output_sha256.is_none() {
        let row = candidate_status_row(
            cfg,
            tr,
            &src,
            "missing",
            metal.output_sha256.clone(),
            Some(format!("metal status={}", metal.status)),
        );
        let _ = append_result_row(cfg, &row);
        return ProcessOutcome::Fail;
    }
    let plan = metal.plan.clone();
    run_candidate(cfg, tr, &src, &ll, stage, &entry, &plan, metal)
}

fn write_failure_row(cfg: &RunConfig, tr: &TranslateRow, src: &SourceFile, err: &str) {
    append_status_row(cfg, tr, src, "fallback", "full", None, err.into());
}

/// Record a case the oracle refused to submit because it could not prove the GPU work bounded
/// (unbounded/uninstrumentable loop). `status=quarantine`, `compare=none`; counted as a failure
/// outcome with a quarantine status. A committed Metal kernel cannot be cancelled, so quarantining
/// is the only safe outcome — the case is recorded for visibility, not dispatched.
fn write_quarantine_row(cfg: &RunConfig, tr: &TranslateRow, src: &SourceFile, reason: &str) {
    append_status_row(
        cfg,
        tr,
        src,
        "quarantine",
        "none",
        None,
        format!("quarantined: {reason}"),
    );
}

#[allow(clippy::too_many_arguments)]
fn run_metal(
    cfg: &RunConfig,
    tr: &TranslateRow,
    src: &SourceFile,
    ll: &str,
    stage: Stage,
    entry: &str,
    plan: &HarnessPlan,
) -> ProcessOutcome {
    #[cfg(not(target_os = "macos"))]
    {
        let _ = (cfg, tr, src, ll, stage, entry, plan);
        eprintln!("    fallback: corpus-run-metal requires macOS");
        return ProcessOutcome::Fail;
    }
    #[cfg(target_os = "macos")]
    {
        let owned = match plan_to_owned_inputs(plan) {
            Ok(o) => o,
            Err(e) => {
                write_failure_row(cfg, tr, src, &e);
                return ProcessOutcome::Fail;
            }
        };
        let input_sha = match input_digest(plan) {
            Ok(h) => h,
            Err(e) => {
                write_failure_row(cfg, tr, src, &e);
                return ProcessOutcome::Fail;
            }
        };
        let air = match air_blob_for_oracle(src) {
            Ok(a) => a,
            Err(e) => {
                write_failure_row(cfg, tr, src, &e);
                return ProcessOutcome::Fail;
            }
        };
        let source_metallib = source_metallib_for_air(src);
        let b64 = base64::engine::general_purpose::STANDARD.encode(&air);
        let result = catch_oracle_unwind(|| {
            crate::oracle_macos::execute_metallib_blob(
                &b64,
                entry,
                stage,
                &owned.inputs,
                ll,
                source_metallib.as_deref(),
            )
        });
        let bytes = match result {
            Ok(b) => b,
            Err(e) => {
                // The oracle refuses to submit work it cannot prove bounded (a committed Metal
                // kernel cannot be cancelled). Record it as quarantined; it is a failure status,
                // but not a dispatched GPU failure.
                if let Some(reason) = e.strip_prefix("m2v-quarantine:") {
                    write_quarantine_row(cfg, tr, src, reason.trim());
                    return ProcessOutcome::Fail;
                }
                write_failure_row(cfg, tr, src, &classify_metal_oracle_panic(&e));
                return ProcessOutcome::Fail;
            }
        };
        // Instrumented (loopy) kernels are marked compare=none so candidate runners know to
        // translate and execute the same bounded LL shape instead of the unguarded source text.
        let compare = match crate::oracle_macos::last_oracle_compare_mode() {
            crate::oracle_macos::OracleCompare::Full => "full",
            crate::oracle_macos::OracleCompare::MetalOnly => "none",
        };
        let row = MetalRow {
            air_sha256: tr.air_sha256.clone(),
            shard: src.shard.clone(),
            label: src.label.clone(),
            status: "ok".into(),
            backend: "metal".into(),
            seed_profile: SEED_PROFILE.into(),
            plan_version: PLAN_VERSION,
            plan: plan.clone(),
            input_sha256: Some(input_sha),
            output_sha256: Some(sha256_hex(&bytes)),
            output_b64: Some(encode_output_b64(&bytes)),
            spv_sha256: tr.spv_sha256.clone(),
            compare: compare.into(),
            stage: Some(format!("{stage:?}")),
            entry: Some(entry.into()),
            error: None,
        };
        if let Err(e) = append_result_row(cfg, &row) {
            eprintln!("    write ledger: {e}");
            return ProcessOutcome::Fail;
        }
        ProcessOutcome::Ok
    }
}

#[cfg(target_os = "macos")]
fn catch_oracle_unwind<F>(f: F) -> Result<Vec<u8>, String>
where
    F: FnOnce() -> Vec<u8> + std::panic::UnwindSafe,
{
    static HOOK_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let _guard = HOOK_LOCK.lock().expect("oracle panic hook mutex poisoned");
    let previous_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let result = std::panic::catch_unwind(f).map_err(panic_payload_message);
    std::panic::set_hook(previous_hook);
    result
}

fn panic_payload_message(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else if let Some(message) = payload.downcast_ref::<&'static str>() {
        (*message).to_string()
    } else {
        "non-string panic payload".to_string()
    }
}

fn classify_metal_oracle_panic(message: &str) -> String {
    let refs = unresolved_visible_refs_from_error(message);
    if refs.is_empty() {
        return format!("metal oracle panicked: {message}");
    }
    format!(
        "unsupported Metal visible function reference(s): {}",
        refs.join(", ")
    )
}

fn unresolved_visible_refs_from_error(message: &str) -> Vec<String> {
    let mut refs = Vec::new();
    for line in message.lines() {
        let Some(rest) = line.split("unresolved visible function reference:").nth(1) else {
            continue;
        };
        let name = rest.trim();
        if !name.is_empty() && !refs.iter().any(|seen| seen == name) {
            refs.push(name.to_string());
        }
    }
    refs
}

/// True when a MoltenVK ICD manifest is visible without `VK_ICD_FILENAMES` (Homebrew layouts).
fn moltenvk_icd_likely_present() -> bool {
    let mut roots = Vec::new();
    if let Ok(prefix) = std::env::var("HOMEBREW_PREFIX") {
        let p = prefix.trim();
        if !p.is_empty() {
            roots.push(PathBuf::from(p));
        }
    }
    roots.push(PathBuf::from("/opt/homebrew"));
    roots.push(PathBuf::from("/usr/local"));
    for root in roots {
        let icd = root.join("etc/vulkan/icd.d/MoltenVK_icd.json");
        if icd.is_file() {
            return true;
        }
    }
    false
}

fn candidate_ll_for_metal_compare<'a>(
    ll: &'a str,
    entry: &str,
    metal: &MetalRow,
) -> Result<Cow<'a, str>, String> {
    if metal.compare != "none" {
        return Ok(Cow::Borrowed(ll));
    }

    match crate::loop_budget::classify_and_instrument(ll, entry) {
        crate::loop_budget::GuardPlan::Instrumented(text) => Ok(Cow::Owned(text)),
        crate::loop_budget::GuardPlan::LoopFree => Ok(Cow::Borrowed(ll)),
        crate::loop_budget::GuardPlan::Quarantine(reason) => Err(reason),
    }
}

fn write_candidate_quarantine_row(
    cfg: &RunConfig,
    tr: &TranslateRow,
    src: &SourceFile,
    metal: &MetalRow,
    err: &str,
) {
    let row = candidate_status_row(
        cfg,
        tr,
        src,
        "quarantine",
        metal.output_sha256.clone(),
        Some(err.into()),
    );
    let _ = append_result_row(cfg, &row);
}

#[allow(clippy::too_many_arguments)]
fn run_candidate(
    cfg: &RunConfig,
    tr: &TranslateRow,
    src: &SourceFile,
    ll: &str,
    stage: Stage,
    entry: &str,
    plan: &HarnessPlan,
    metal: &MetalRow,
) -> ProcessOutcome {
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        let _ = (cfg, tr, src, ll, stage, entry, plan, metal);
        write_failure_row(cfg, tr, src, "vulkan runner unsupported on this OS");
        return ProcessOutcome::Fail;
    }
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    {
        if cfg.backend == RunBackend::MoltenVk {
            // Homebrew MoltenVK usually registers under $HOMEBREW_PREFIX/etc/vulkan/icd.d; only
            // nudge when nothing is configured and we cannot see a common ICD path either.
            if std::env::var_os("VK_ICD_FILENAMES").is_none()
                && std::env::var_os("VK_DRIVER_FILES").is_none()
                && !cfg.quiet
                && !moltenvk_icd_likely_present()
            {
                eprintln!(
                    "    note: MoltenVK ICD not found in common paths; set VK_ICD_FILENAMES if devices fail to enumerate"
                );
            }
        }
        let owned = match plan_to_owned_inputs(plan) {
            Ok(o) => o,
            Err(e) => {
                write_failure_row(cfg, tr, src, &e);
                return ProcessOutcome::Fail;
            }
        };
        let tmp = crate::scratch_dir_for(&format!("corpus-run-{}", &tr.air_sha256[..12]));
        let candidate_ll = match candidate_ll_for_metal_compare(ll, entry, metal) {
            Ok(candidate_ll) => candidate_ll,
            Err(reason) => {
                let _ = fs::remove_dir_all(&tmp);
                write_candidate_quarantine_row(
                    cfg,
                    tr,
                    src,
                    metal,
                    &format!("candidate loop guard: {reason}"),
                );
                return ProcessOutcome::Fail;
            }
        };
        let candidate_ll = candidate_ll.as_ref();
        let m2v_stage: metal2vulkan::passes::Stage = stage.into();
        let spv = match metal2vulkan::translate_sanitized_native(candidate_ll, m2v_stage, &tmp) {
            Ok(s) if !s.is_empty() => s,
            Ok(_) => {
                let _ = fs::remove_dir_all(&tmp);
                write_failure_row(cfg, tr, src, "translate produced empty SPIR-V");
                return ProcessOutcome::Fail;
            }
            Err(e) => {
                let _ = fs::remove_dir_all(&tmp);
                write_failure_row(cfg, tr, src, &format!("translate: {e}"));
                return ProcessOutcome::Fail;
            }
        };
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            crate::runner_linux::execute(stage, candidate_ll, &spv, &owned.inputs, &tmp)
        }));
        let _ = fs::remove_dir_all(&tmp);
        let candidate = match result {
            Ok(b) => b,
            Err(payload) => {
                let detail = panic_payload_message(payload);
                write_failure_row(cfg, tr, src, &format!("vulkan execute panicked: {detail}"));
                return ProcessOutcome::Fail;
            }
        };
        let golden_hash = metal.output_sha256.clone().unwrap_or_default();
        let out_hash = sha256_hex(&candidate);
        let format = parse_format(&metal.plan.output.format).unwrap_or(DataFormat::RawBytes);
        let (status, observed, tolerance) =
            compare_candidate_to_metal(&candidate, metal, &out_hash, &golden_hash, format);
        let row = CandidateRow {
            air_sha256: tr.air_sha256.clone(),
            shard: src.shard.clone(),
            label: src.label.clone(),
            status,
            backend: cfg.backend.as_str().into(),
            output_sha256: Some(out_hash),
            output_b64: Some(encode_output_b64(&candidate)),
            golden_output_sha256: Some(golden_hash),
            spv_sha256: Some(sha256_hex(&spv)),
            tolerance,
            observed,
            error: None,
        };
        if let Err(e) = append_result_row(cfg, &row) {
            eprintln!("    write ledger: {e}");
            return ProcessOutcome::Fail;
        }
        if execution_status_is_success(cfg.backend, &row.status) {
            ProcessOutcome::Ok
        } else {
            ProcessOutcome::Fail
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal AIR-shaped snippet of an MPS-style GEMM: constant-space params struct +
    /// K loaded into a loop `icmp`/`br`. Deterministic seeds of this blob are what wedged
    /// the GPU (K ≈ 1.8e9); bounded_control must fire on buffer 4.
    const MPS_LIKE_LL: &str = r#"
define void @f16MatrixMultiplyNN_aligned(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3, ptr addrspace(2) %4, <2 x i32> %5) {
  %8 = getelementptr inbounds i8, ptr addrspace(2) %4, i64 8
  %9 = load i32, ptr addrspace(2) %8, align 4
  %10 = getelementptr inbounds i8, ptr addrspace(2) %4, i64 0
  %11 = load i32, ptr addrspace(2) %10, align 4
  %14 = extractelement <2 x i32> %5, i64 0
  %15 = shl i32 %14, 3
  %16 = icmp ugt i32 %11, %15
  br i1 %16, label %17, label %372
17:
  br label %95
95:
  %97 = phi i32 [ 0, %17 ], [ %109, %95 ]
  %109 = add nuw i32 %97, 1
  %110 = icmp eq i32 %109, %9
  br i1 %110, label %372, label %95
372:
  ret void
}

!air.kernel = !{!15}
!15 = !{ptr @f16MatrixMultiplyNN_aligned, !16, !17}
!16 = !{}
!17 = !{!18, !19, !20, !21, !22, !24}
!18 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 2, !"air.arg_type_name", !"half", !"air.arg_name", !"A"}
!19 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 2, !"air.arg_type_name", !"half", !"air.arg_name", !"B"}
!20 = !{i32 2, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 2, !"air.arg_type_name", !"half", !"air.arg_name", !"C"}
!21 = !{i32 3, !"air.buffer", !"air.location_index", i32 3, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 2, !"air.arg_type_name", !"half", !"air.arg_name", !"D"}
!22 = !{i32 4, !"air.buffer", !"air.buffer_size", i32 36, !"air.location_index", i32 4, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !23, !"air.arg_type_size", i32 36, !"air.arg_type_name", !"MPSMatrixMulParameters", !"air.arg_name", !"p"}
!23 = !{i32 0, i32 4, i32 0, !"uint", !"M"}
!24 = !{i32 5, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint2", !"air.arg_name", !"gid"}
"#;

    /// Top-K shaped metadata: constant params at location 0 and threadgroup scratch also at
    /// location 0. The plan must keep the 48-byte constant params and ignore the threadgroup row.
    const TOPK_LIKE_LL: &str = r#"
define void @matrix_topk_like(ptr addrspace(2) %0, ptr addrspace(3) %1, <2 x i32> %2) {
  %dest_y_ptr = getelementptr inbounds i8, ptr addrspace(2) %0, i64 4
  %dest_y = load i32, ptr addrspace(2) %dest_y_ptr, align 4
  %gid_y = extractelement <2 x i32> %2, i64 1
  %in_bounds = icmp ult i32 %gid_y, %dest_y
  br i1 %in_bounds, label %work, label %exit
work:
  %num_topk_ptr = getelementptr inbounds i8, ptr addrspace(2) %0, i64 44
  %num_topk = load i32, ptr addrspace(2) %num_topk_ptr, align 4
  %done = icmp eq i32 %num_topk, 0
  br i1 %done, label %exit, label %work
exit:
  ret void
}

!air.kernel = !{!10}
!10 = !{ptr @matrix_topk_like, !11, !12}
!11 = !{}
!12 = !{!13, !15, !16}
!13 = !{i32 0, !"air.buffer", !"air.buffer_size", i32 48, !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !14, !"air.arg_type_size", i32 48, !"air.arg_type_name", !"MatrixTopKParams", !"air.arg_name", !"params"}
!14 = !{i32 0, i32 4, i32 0, !"uint", !"destination_size_y"}
!15 = !{i32 1, !"air.buffer", !"air.buffer_size", i32 4096, !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 3, !"air.arg_type_size", i32 4096, !"air.arg_type_name", !"threadgroup uchar*", !"air.arg_name", !"shBlob"}
!16 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint2", !"air.arg_name", !"gid"}
"#;

    #[test]
    fn infer_plan_marks_constant_param_struct_bounded() {
        let plan = infer_plan(MPS_LIKE_LL);
        let p = plan
            .buffers
            .iter()
            .find(|b| b.index == 4)
            .expect("param buffer 4");
        assert_eq!(p.len, 36);
        assert_eq!(p.seed_mode, SEED_MODE_BOUNDED_CONTROL);
        // Payload half buffers stay deterministic.
        for idx in [0u32, 1, 2, 3] {
            let b = plan.buffers.iter().find(|b| b.index == idx).unwrap();
            assert_eq!(
                b.seed_mode, SEED_MODE_DETERMINISTIC,
                "buffer {idx} should stay deterministic"
            );
        }
    }

    #[test]
    fn bounded_control_seed_keeps_m_n_k_small() {
        let plan = infer_plan(MPS_LIKE_LL);
        let owned = plan_to_owned_inputs(&plan).unwrap();
        let param = owned
            .inputs
            .buffers
            .iter()
            .find(|b| b.index == 4)
            .expect("param");
        let bytes = seeded_buffer_bytes(param);
        assert_eq!(bytes.len(), 36);
        let m = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
        let n = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        let k = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
        assert_eq!(m, BOUNDED_CONTROL_DIM);
        assert_eq!(n, BOUNDED_CONTROL_DIM);
        assert_eq!(k, BOUNDED_CONTROL_DIM);
        // Work estimate: threads * K must stay tiny vs the wedging ~1e11 FMULs.
        assert!(u64::from(k) * 64 < 100_000);
    }

    #[test]
    fn loop_bound_scan_sees_param_buffer() {
        let hit = buffers_with_loads_used_as_loop_bounds(MPS_LIKE_LL);
        assert!(
            hit.contains(&4),
            "expected buffer 4 (K/M loads → icmp → br), got {hit:?}"
        );
    }

    /// MPS Top-K: constant params @ location 0 plus threadgroup scratch also @ location 0.
    /// Plan must keep the 48-byte params (bounded), not let shBlob clobber them.
    #[test]
    fn topk_params_not_clobbered_by_threadgroup_meta() {
        let plan = infer_plan(TOPK_LIKE_LL);
        let p0 = plan
            .buffers
            .iter()
            .find(|b| b.index == 0)
            .expect("buffer 0");
        assert_eq!(
            p0.seed_mode, SEED_MODE_BOUNDED_CONTROL,
            "MatrixTopKParams must be bounded_control, not clobbered by threadgroup shBlob"
        );
        assert_eq!(p0.len, 48);
        let owned = plan_to_owned_inputs(&plan).unwrap();
        let bytes =
            seeded_buffer_bytes(owned.inputs.buffers.iter().find(|b| b.index == 0).unwrap());
        // numTopK is ushort at offset 44; LE u32 fill puts 16 in those bytes.
        let num_topk = u16::from_le_bytes(bytes[44..46].try_into().unwrap());
        assert_eq!(num_topk, BOUNDED_CONTROL_DIM as u16);
        // destination_size.y (i32 @ offset 4) gates the grid early-out.
        let dest_y = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        assert_eq!(dest_y, BOUNDED_CONTROL_DIM);
    }

    #[test]
    fn output_b64_roundtrip_and_sha() {
        let bytes = [0u8, 1, 2, 3, 0xfe, 0xff];
        let b64 = encode_output_b64(&bytes);
        assert_eq!(decode_output_b64(&b64).unwrap(), bytes);
        assert_eq!(
            sha256_hex(&bytes),
            sha256_hex(&decode_output_b64(&b64).unwrap())
        );
    }

    #[test]
    fn compare_with_output_b64_classifies_float_tolerance() {
        let golden = 1.0f32.to_le_bytes();
        // One ULP above 1.0 — well within max_ulp=8.
        let candidate = f32::from_bits(1.0f32.to_bits() + 1).to_le_bytes();
        let metal = MetalRow {
            air_sha256: "x".into(),
            shard: None,
            label: String::new(),
            status: "ok".into(),
            backend: "metal".into(),
            seed_profile: SEED_PROFILE.into(),
            plan_version: PLAN_VERSION,
            plan: infer_plan(""),
            input_sha256: None,
            output_sha256: Some(sha256_hex(&golden)),
            output_b64: Some(encode_output_b64(&golden)),
            spv_sha256: None,
            compare: "full".into(),
            stage: None,
            entry: None,
            error: None,
        };
        let out_hash = sha256_hex(&candidate);
        let golden_hash = metal.output_sha256.clone().unwrap();
        let (status, observed, tolerance) = compare_candidate_to_metal(
            &candidate,
            &metal,
            &out_hash,
            &golden_hash,
            DataFormat::F32,
        );
        assert_eq!(status, "tolerance");
        assert!(observed.is_some());
        assert_eq!(
            tolerance.as_ref().map(|t| t.kind.as_str()),
            Some("AbsAndUlp")
        );
    }

    #[test]
    fn compare_without_output_b64_is_hash_only() {
        let golden = [1u8, 2, 3, 4];
        let candidate = [1u8, 2, 3, 5];
        let metal = MetalRow {
            air_sha256: "x".into(),
            shard: None,
            label: String::new(),
            status: "ok".into(),
            backend: "metal".into(),
            seed_profile: SEED_PROFILE.into(),
            plan_version: PLAN_VERSION,
            plan: infer_plan(""),
            input_sha256: None,
            output_sha256: Some(sha256_hex(&golden)),
            output_b64: None,
            spv_sha256: None,
            compare: "full".into(),
            stage: None,
            entry: None,
            error: None,
        };
        let (status, observed, tolerance) = compare_candidate_to_metal(
            &candidate,
            &metal,
            &sha256_hex(&candidate),
            metal.output_sha256.as_deref().unwrap(),
            DataFormat::RawBytes,
        );
        assert_eq!(status, "failure");
        assert!(observed.is_none());
        assert!(tolerance.is_none());
    }

    #[test]
    fn execution_success_accepts_candidate_tolerance() {
        assert!(execution_status_is_success(RunBackend::Metal, "ok"));
        assert!(!execution_status_is_success(RunBackend::Metal, "tolerance"));
        assert!(execution_status_is_success(RunBackend::Vulkan, "ok"));
        assert!(execution_status_is_success(RunBackend::Vulkan, "tolerance"));
        assert!(execution_status_is_success(
            RunBackend::MoltenVk,
            "tolerance"
        ));
        assert!(!execution_status_is_success(
            RunBackend::MoltenVk,
            "failure"
        ));
    }

    #[test]
    fn execution_failure_signature_groups_common_noise() {
        assert_eq!(
            execution_failure_signature(
                "fallback",
                Some(
                    "vulkan execute panicked: create compute pipeline: a non-validation error occurred",
                ),
                false,
            ),
            "create compute pipeline"
        );
        assert_eq!(
            execution_failure_signature("missing", None, false),
            "missing metal golden"
        );
        assert_eq!(
            execution_failure_signature("failure", None, true),
            "candidate output mismatch outside tolerance"
        );
    }

    #[test]
    fn tech_row_filters_select_targeted_existing_rows() {
        let mut cfg = RunConfig::from_manifest(RunBackend::MoltenVk);
        cfg.failed_only = true;
        cfg.only_status = Some("fallback".into());
        cfg.only_bucket = Some("compute".into());
        cfg.contains = Some("local/foo".into());

        let row = TechRowInfo {
            air_sha256: "a".repeat(64),
            status: "fallback".into(),
            label: "local/foo.ll".into(),
            error: Some("vulkan execute panicked: create compute pipeline".into()),
            signature: "create compute pipeline".into(),
        };
        assert!(tech_row_selected(&cfg, &row));

        let mut success = row.clone();
        success.status = "tolerance".into();
        assert!(!tech_row_selected(&cfg, &success));

        let mut wrong_bucket = row.clone();
        wrong_bucket.signature = "create vertex validation pipeline".into();
        assert!(!tech_row_selected(&cfg, &wrong_bucket));

        let mut wrong_text = row;
        wrong_text.label = "local/bar.ll".into();
        wrong_text.error = None;
        assert!(!tech_row_selected(&cfg, &wrong_text));
    }

    #[test]
    fn candidate_compare_none_uses_guarded_ll() {
        let ll = "\
define void @spin(ptr addrspace(1) %0) {
  br label %1
1:
  br label %1
}
";
        let mut metal = MetalRow {
            air_sha256: "x".into(),
            shard: None,
            label: String::new(),
            status: "ok".into(),
            backend: "metal".into(),
            seed_profile: SEED_PROFILE.into(),
            plan_version: PLAN_VERSION,
            plan: infer_plan(""),
            input_sha256: None,
            output_sha256: Some(sha256_hex(&[])),
            output_b64: Some(encode_output_b64(&[])),
            spv_sha256: None,
            compare: "none".into(),
            stage: None,
            entry: None,
            error: None,
        };

        match candidate_ll_for_metal_compare(ll, "spin", &metal).unwrap() {
            Cow::Owned(text) => assert!(text.contains("m2v.g.0:"), "{text}"),
            Cow::Borrowed(_) => panic!("compare=none loop should be instrumented"),
        }

        metal.compare = "full".into();
        match candidate_ll_for_metal_compare(ll, "spin", &metal).unwrap() {
            Cow::Borrowed(text) => assert_eq!(text, ll),
            Cow::Owned(_) => panic!("compare=full should keep original LL"),
        }
    }

    #[test]
    fn entry_name_handles_quoted_air_symbols() {
        let ll = r#"
define void @"persona::ksDepthDilate"(ptr addrspace(2) %0) {
  ret void
}

!air.kernel = !{!15}
!15 = !{ptr @"persona::ksDepthDilate", !16, !17}
!16 = !{}
!17 = !{!18}
!18 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_name", !"Params"}
"#;

        assert_eq!(
            entry_name_from_ll(ll),
            Some("persona::ksDepthDilate".into())
        );
    }

    #[test]
    fn unresolved_visible_refs_are_classified_as_unsupported() {
        let err = "newComputePipelineStateWithFunction(topkv): \
error: unresolved visible function reference: postfixPrimary_f\n  Reason: visible function not loaded\n\
error: unresolved visible function reference: prefixPrimary_f\n  Reason: visible function not loaded\n\
error: unresolved visible function reference: postfixPrimary_f\n  Reason: visible function not loaded";

        assert_eq!(
            classify_metal_oracle_panic(err),
            "unsupported Metal visible function reference(s): postfixPrimary_f, prefixPrimary_f"
        );
    }

    #[test]
    fn fragment_plan_uses_render_target_output() {
        let ll = r#"
define <{ <4 x float> }> @frag() {
  ret <{ <4 x float> }> zeroinitializer
}

!air.fragment = !{!15}
!15 = !{ptr @frag, !16, !17}
!16 = !{!18}
!17 = !{}
!18 = !{!"air.render_target", !"air.arg_type_name", !"float4", !"air.arg_name", !"color"}
"#;

        let plan = infer_plan(ll);
        assert_eq!(plan.output.kind, "render_target");
        assert_eq!(plan.output.format, "Rgba32Float");
        assert_eq!(plan.output.w, Some(DEFAULT_TEXTURE_EXTENT.width));
        assert_eq!(plan.output.h, Some(DEFAULT_TEXTURE_EXTENT.height));
    }

    #[test]
    fn fragment_plan_uses_depth_output_for_depth_only_fragment() {
        let ll = r#"
define <{ float }> @frag() {
  ret <{ float }> zeroinitializer
}

!air.fragment = !{!15}
!15 = !{ptr @frag, !16, !18}
!16 = !{!17}
!17 = !{!"air.depth", !"air.depth_qualifier", !"air.any", !"air.arg_type_name", !"float", !"air.arg_name", !"depth"}
!18 = !{}
"#;

        let plan = infer_plan(ll);
        assert_eq!(plan.output.kind, "render_target");
        assert_eq!(plan.output.format, "Depth32Float");
    }

    #[test]
    fn texture_metadata_becomes_plan_texture() {
        let ll = r#"
define void @k(ptr addrspace(2) %0) {
  ret void
}

!air.kernel = !{!15}
!15 = !{ptr @k, !16, !17}
!16 = !{}
!17 = !{!18, !19}
!18 = !{i32 0, !"air.texture", !"air.location_index", i32 2, i32 1, !"air.read_write", !"air.arg_type_name", !"texture2d<half, read_write>", !"air.arg_name", !"tex"}
!19 = !{i32 1, !"air.sampler", !"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"sampler", !"air.arg_name", !"s"}
"#;

        let plan = infer_plan(ll);
        assert_eq!(plan.textures.len(), 1);
        assert_eq!(plan.textures[0].index, 2);
        assert_eq!(plan.textures[0].role, "StorageReadWrite");
        assert_eq!(plan.textures[0].format, "Rgba16Float");
        assert_eq!(plan.output.kind, "texture");
        assert_eq!(plan.output.index, 2);
    }
}
