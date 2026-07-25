//! Corpus execution ledgers and runners (see repo-root `plan.md`).
//!
//! Lazy plan inference + Metal golden / Vulkan / MoltenVK candidate JSONL writers.

use crate::air::{entry_name_from_ll, stage_from_ll};
use crate::corpus_shards;
#[cfg(target_os = "macos")]
use crate::corpus_source::{air_blob_for_oracle, source_metallib_for_air};
use crate::corpus_source::{load_ll_text, resolve_source, SourceFile};
use crate::hash::sha256_bytes as sha256_hex;
use crate::jsonl::sort_json;
use crate::texture::fragment_writes_depth;
use crate::{
    seeded_buffer_bytes, seeded_render_target_bytes, seeded_texture_bytes, BufferInput, BufferRole,
    DataFormat, Dispatch, Extent3d, Inputs, Output, Render, Seed, Stage, TextureInput, TextureRole,
    RENDER_TARGET_SEED_TAG,
};
use base64::Engine as _;
use metal2vulkan::meta::{FragRole, KernRole, VertRole};
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

/// Bumped when default seed construction changes.
pub const SEED_PROFILE: &str = "deterministic_v6_finite_float_render_targets";
pub const PLAN_VERSION: u32 = 1;
const POINT_COORD_TOPOLOGY_PLAN_VERSION: u32 = 2;
pub const DEFAULT_BUFFER_LEN: usize = 256;
const DEFAULT_DISPATCH_GRID_X: usize = 64;
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
const FC_SPECIALIZATION_ZERO: &str = "zero";
const FC_SPECIALIZATION_VALUES: &str = "values";

/// Seed mode written into [`PlanBuffer::seed_mode`].
pub const SEED_MODE_DETERMINISTIC: &str = "deterministic";
/// Fixed-size control/param buffers whose integers feed loop trip counts / grid checks.
/// Seeded with small dims so MPS-style GEMMs cannot pin the GPU for ~10^9 iterations.
pub const SEED_MODE_BOUNDED_CONTROL: &str = "bounded_control";
/// Deterministic f16/f32 payload buffers with non-finite bit patterns sanitized out.
pub const SEED_MODE_FINITE_FLOAT16: &str = "finite_float16";
pub const SEED_MODE_FINITE_FLOAT32: &str = "finite_float32";
pub const SEED_MODE_FINITE_BFLOAT16: &str = "finite_bfloat16";
pub const SEED_MODE_FINITE_STRUCT_FLOAT: &str = "finite_struct_float";
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
    /// See the `SEED_MODE_*` constants.
    #[serde(default = "default_seed_mode")]
    pub seed_mode: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub seed_layout: Vec<ControlSeedField>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed_stride: Option<usize>,
}

fn default_seed_mode() -> String {
    SEED_MODE_DETERMINISTIC.to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ControlSeedField {
    pub offset: usize,
    pub size: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub value: Option<u64>,
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
    /// See the `SEED_MODE_*` constants.
    #[serde(default = "default_seed_mode")]
    pub seed_mode: String,
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
    /// Function-constant mode used for this Metal golden. Older rows omit this; candidates cannot
    /// byte-compare those rows when the AIR declares FCs because Metal empty-specialization and the
    /// translator's disabled-zero model are not guaranteed to pick the same path.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fc_specialization: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fc_values: Option<Vec<FunctionConstantValueJson>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stage: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub entry: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FunctionConstantValueJson {
    pub index: u32,
    pub value: u64,
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

/// Default numeric policy for float-like outputs when candidate ≠ metal golden. Absolute distance
/// covers values near zero; ULP distance covers large finite values where a tiny representational
/// drift has a large absolute magnitude.
pub fn default_float_tolerance() -> ToleranceSpecJson {
    ToleranceSpecJson {
        kind: "AbsOrUlp".into(),
        max_abs: Some(1e-3),
        max_ulp: Some(32),
    }
}

/// Numeric policy for a specific output format.
///
/// Half render/storage outputs are finally quantized to 16-bit lanes, and fast-math/FTZ differences
/// between Metal and Vulkan commonly show up as a few half-code steps while still being tiny in
/// absolute value. Keep f32-like outputs on the generic policy, but allow a half-specific absolute
/// bound of 2^-9.
pub fn float_tolerance_for_format(format: DataFormat) -> ToleranceSpecJson {
    let mut tolerance = default_float_tolerance();
    if matches!(
        format,
        DataFormat::R16Float | DataFormat::Rg16Float | DataFormat::Rgba16Float
    ) {
        tolerance.max_abs = Some(0.001_953_125);
    }
    tolerance
}

fn float_tolerance_for_context(format: DataFormat, ll: Option<&str>) -> ToleranceSpecJson {
    let mut tolerance = float_tolerance_for_format(format);
    if matches!(
        format,
        DataFormat::R16Float | DataFormat::Rg16Float | DataFormat::Rgba16Float
    ) && ll.is_some_and(ll_has_fast_no_nans_float_semantics)
    {
        tolerance.max_abs = Some(0.003_906_25);
    }
    tolerance
}

fn tolerance_for_context(format: DataFormat, ll: Option<&str>) -> Option<ToleranceSpecJson> {
    if format.is_float_like() {
        return Some(float_tolerance_for_context(format, ll));
    }
    if packed_unorm_raw_byte_tolerance_applies(format, ll?) {
        return Some(ToleranceSpecJson {
            kind: "Abs".into(),
            max_abs: Some(1.0),
            max_ulp: None,
        });
    }
    if integer_render_target_quantization_tolerance_applies(format, ll?) {
        return Some(ToleranceSpecJson {
            kind: "AbsAndUlp".into(),
            max_abs: Some(2.0),
            max_ulp: Some(16),
        });
    }
    None
}

fn tolerance_for_metal_context(
    format: DataFormat,
    ll: Option<&str>,
    metal: &MetalRow,
) -> Option<ToleranceSpecJson> {
    let mut tolerance = tolerance_for_context(format, ll)?;
    if sampled_half_render_target_fast_math_tolerance_applies(format, ll, metal) {
        tolerance.max_abs = Some(tolerance.max_abs.unwrap_or(0.0).max(0.007_812_5));
    }
    Some(tolerance)
}

fn sampled_half_render_target_fast_math_tolerance_applies(
    format: DataFormat,
    ll: Option<&str>,
    metal: &MetalRow,
) -> bool {
    if !matches!(
        format,
        DataFormat::R16Float | DataFormat::Rg16Float | DataFormat::Rgba16Float
    ) || metal.plan.output.kind != "render_target"
    {
        return false;
    }
    let Some(ll) = ll else {
        return false;
    };
    ll_has_fast_no_nans_float_semantics(ll)
        && ll.contains("@air.sample_texture")
        && ll.contains(".v4f16")
}

fn packed_unorm_raw_byte_tolerance_applies(format: DataFormat, ll: &str) -> bool {
    format == DataFormat::RawBytes && ll.contains("@air.pack.unorm4x8.")
}

fn integer_render_target_quantization_tolerance_applies(format: DataFormat, ll: &str) -> bool {
    if !matches!(
        format,
        DataFormat::Rgba8Uint
            | DataFormat::Rgba8Sint
            | DataFormat::R16Uint
            | DataFormat::Rg16Uint
            | DataFormat::Rgba16Uint
            | DataFormat::R32Uint
            | DataFormat::Rg32Uint
            | DataFormat::Rgba32Uint
            | DataFormat::R16Sint
            | DataFormat::Rg16Sint
            | DataFormat::Rgba16Sint
            | DataFormat::R32Sint
            | DataFormat::Rg32Sint
            | DataFormat::Rgba32Sint
            | DataFormat::U32
            | DataFormat::I32
    ) {
        return false;
    }
    ll_has_fast_no_nans_float_semantics(ll)
        && ll.lines().any(|line| {
            line.contains("\"air.render_target\"")
                && line
                    .split("\"air.arg_type_name\"")
                    .nth(1)
                    .is_some_and(line_has_integer_air_type_name)
        })
        && ll.lines().any(|line| {
            (line.contains("@air.convert.u.") || line.contains("@air.convert.s."))
                && line.contains(".f.")
                || line.contains(" fptoui ")
                || line.contains(" fptosi ")
        })
}

fn line_has_integer_air_type_name(line: &str) -> bool {
    [
        "\"uchar", "\"char", "\"ushort", "\"short", "\"uint", "\"int",
    ]
    .iter()
    .any(|needle| line.contains(needle))
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
    pub skip: usize,
    pub limit: Option<usize>,
    pub dry_run: bool,
    pub quiet: bool,
    pub only_air: Option<String>,
    pub only_air_list: Option<PathBuf>,
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
            skip: 0,
            limit: None,
            dry_run: false,
            quiet: false,
            only_air: None,
            only_air_list: None,
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
            "--limit" => {
                let n = args
                    .next()
                    .unwrap_or_else(|| fatal(program, "--limit requires N"));
                cfg.limit = Some(
                    n.parse()
                        .unwrap_or_else(|_| fatal(program, &format!("bad --limit {n}"))),
                );
            }
            "--skip" => {
                let n = args
                    .next()
                    .unwrap_or_else(|| fatal(program, "--skip requires N"));
                cfg.skip = n
                    .parse()
                    .unwrap_or_else(|_| fatal(program, &format!("bad --skip {n}")));
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
            "--air-list" | "--air-sha256-list" => {
                cfg.only_air_list =
                    Some(PathBuf::from(args.next().unwrap_or_else(|| {
                        fatal(program, "--air-list requires path")
                    })));
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
            other if other.starts_with("--limit=") => {
                let n = other.trim_start_matches("--limit=");
                cfg.limit = Some(
                    n.parse()
                        .unwrap_or_else(|_| fatal(program, &format!("bad --limit {n}"))),
                );
            }
            other if other.starts_with("--skip=") => {
                let n = other.trim_start_matches("--skip=");
                cfg.skip = n
                    .parse()
                    .unwrap_or_else(|_| fatal(program, &format!("bad --skip {n}")));
            }
            other if other.starts_with("--air-list=") => {
                cfg.only_air_list = Some(PathBuf::from(other.trim_start_matches("--air-list=")));
            }
            other if other.starts_with("--air-sha256-list=") => {
                cfg.only_air_list = Some(PathBuf::from(
                    other.trim_start_matches("--air-sha256-list="),
                ));
            }
            other if !other.starts_with('-') && cfg.only_air.is_none() => {
                cfg.only_air = Some(normalize_hash(other));
            }
            other => fatal(program, &format!("unknown arg: {other}")),
        }
    }
    if cfg.only_air.is_some() && cfg.only_air_list.is_some() {
        fatal(
            program,
            "--air-sha256 and --air-list are mutually exclusive",
        );
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
                \t\t[--air-sha256 HEX] [--air-list FILE] [--ledger-dir DIR]\n\
                \t\t[--status STATUS] [--bucket TEXT] [--contains TEXT] [--skip N] [--limit N]\n\
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
         --skip N         skip N eligible rows after filters and stable sorting, before --limit\n\
         --limit N        run at most N eligible rows after filters and stable sorting\n\
         --air-list FILE  run AIR SHA-256 hashes listed one per line, after other filters\n\
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

fn load_hash_list(program: &str, path: &Path) -> HashSet<String> {
    let file = File::open(path)
        .unwrap_or_else(|e| fatal(program, &format!("open --air-list {}: {e}", path.display())));
    let mut hashes = HashSet::new();
    for (i, line) in BufReader::new(file).lines().enumerate() {
        let line = line.unwrap_or_else(|e| {
            fatal(
                program,
                &format!("read --air-list {} line {}: {e}", path.display(), i + 1),
            )
        });
        let raw = line.split('#').next().unwrap_or("").trim();
        if raw.is_empty() {
            continue;
        }
        let hash = normalize_hash(raw);
        if hash.len() != 64 || !hash.bytes().all(|b| b.is_ascii_hexdigit()) {
            fatal(
                program,
                &format!("bad --air-list hash at {}:{}: {raw}", path.display(), i + 1),
            );
        }
        hashes.insert(hash);
    }
    hashes
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
        RunBackend::Vulkan | RunBackend::MoltenVk => {
            status == "ok" || status == "tolerance" || status == "smoke"
        }
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
        fc_specialization: None,
        fc_values: None,
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
            seed_layout: Vec::new(),
            seed_stride: None,
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
        let writeonly_buffers = writeonly_entry_buffer_locations(ll_or_meta_text);
        let out_idx = buffers
            .iter()
            .find(|b| writeonly_buffers.contains(&b.index))
            .or_else(|| buffers.iter().find(|b| b.role == "Output"))
            .or_else(|| buffers.iter().find(|b| b.role == "InOut"))
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
            format: buffer_output_format(ll_or_meta_text, out_idx)
                .unwrap_or("RawBytes")
                .into(),
            len: Some(out_len),
            w: None,
            h: None,
            d: None,
        }
    };

    let (dispatch_grid, dispatch_tg) = dispatch_plan_for_output(ll_or_meta_text, &output);

    HarnessPlan {
        buffers,
        textures,
        output,
        dispatch_grid,
        dispatch_tg,
    }
}

fn dispatch_plan_for_output(ll: &str, output: &PlanOutput) -> ([u32; 3], [u32; 3]) {
    let default = [DEFAULT_DISPATCH_GRID_X as u32, 1, 1];
    if output.kind != "texture" || thread_position_in_grid_lanes(ll).unwrap_or(1) < 2 {
        return (default, default);
    }
    let grid = [
        output.w.unwrap_or(DEFAULT_TEXTURE_EXTENT.width).max(1),
        output.h.unwrap_or(DEFAULT_TEXTURE_EXTENT.height).max(1),
        if thread_position_in_grid_lanes(ll).unwrap_or(1) >= 3 {
            output.d.unwrap_or(1).max(1)
        } else {
            1
        },
    ];
    let tg = [grid[0].clamp(1, 8), grid[1].clamp(1, 8), 1];
    (grid, tg)
}

fn thread_position_in_grid_lanes(ll: &str) -> Option<u32> {
    ll.lines()
        .find(|line| line.contains(r#""air.thread_position_in_grid""#))
        .and_then(|line| quoted_metadata_string_after(line, "air.arg_type_name"))
        .and_then(|name| vector_lane_count_from_air_type(&name))
}

fn vector_lane_count_from_air_type(name: &str) -> Option<u32> {
    let suffix = name
        .trim()
        .trim_end_matches('*')
        .chars()
        .last()
        .filter(|ch| matches!(ch, '2' | '3' | '4'))?;
    Some(suffix as u32 - '0' as u32)
}

fn infer_buffers(ll: &str) -> Vec<PlanBuffer> {
    // Match AIR buffer metadata nodes: air.buffer ... air.buffer_size i32 N ... air.location_index i32 L
    let loop_bound_bufs = buffers_with_loads_used_as_loop_bounds(ll);
    let stride_control_bufs = stride_control_buffer_locations(ll, &loop_bound_bufs);
    let atomic_i32_load_bufs = buffers_with_atomic_i32_loads(ll);
    let fdiv_denominator_control_bufs = buffers_with_float_loads_used_as_fdiv_denominators(ll);
    let bounded_control_module = module_uses_bounded_control_buffers(ll, &loop_bound_bufs);
    let locations = stage_resource_locations(ll);
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
        let loc = metadata_param_index(line)
            .and_then(|idx| locations.buffers.get(&idx).copied())
            .or_else(|| extract_i32_after(line, "air.location_index").map(|loc| loc as u32))
            .unwrap_or(0);
        let role = if line.contains("air.read_write") {
            "InOut"
        } else if line.contains("!\"air.write\"") || line.contains("\"air.write\"") {
            "Output"
        } else {
            "Input"
        };
        let loc_u = loc;
        let type_name = quoted_metadata_string_after(line, "air.arg_type_name");
        let seed_mode = if is_control_param_buffer_meta(line, fixed_size)
            || loop_bound_bufs.contains(&loc_u)
            || (role == "Input" && atomic_i32_load_bufs.contains(&loc_u))
        {
            SEED_MODE_BOUNDED_CONTROL
        } else if let Some(seed_mode) = finite_float_buffer_seed_mode(type_name.as_deref()) {
            seed_mode
        } else {
            SEED_MODE_DETERMINISTIC
        };
        let mut len = (size as usize).max(4);
        if fixed_size.is_none() && role == "Input" && bounded_control_module {
            if let Some(payload_len) = bounded_control_float_payload_len(type_name.as_deref()) {
                len = len.max(payload_len);
            }
        }
        let finite_struct_seed = finite_struct_float_seed_layout(ll, line, len);
        let seed_mode = if seed_mode == SEED_MODE_DETERMINISTIC
            && role != "Output"
            && finite_struct_seed.is_some()
        {
            SEED_MODE_FINITE_STRUCT_FLOAT
        } else {
            seed_mode
        };
        let seed_layout = if seed_mode == SEED_MODE_BOUNDED_CONTROL {
            bounded_control_seed_layout(
                ll,
                line,
                len,
                &stride_control_bufs,
                &fdiv_denominator_control_bufs,
            )
        } else if seed_mode == SEED_MODE_FINITE_STRUCT_FLOAT {
            finite_struct_seed
                .as_ref()
                .map(|(_, layout)| layout.clone())
                .unwrap_or_default()
        } else {
            Vec::new()
        };
        let seed_stride = if seed_mode == SEED_MODE_FINITE_STRUCT_FLOAT {
            finite_struct_seed.map(|(stride, _)| stride)
        } else {
            None
        };
        if fixed_size.is_none() && role == "Input" && seed_mode == SEED_MODE_FINITE_STRUCT_FLOAT {
            if let Some(stride) = seed_stride {
                len = len.max(stride.saturating_mul(DEFAULT_DISPATCH_GRID_X));
            }
        }
        out.push(PlanBuffer {
            index: loc_u,
            len,
            role: role.into(),
            seed_tag: loc_u.wrapping_add(1),
            seed_mode: seed_mode.into(),
            seed_layout,
            seed_stride,
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
            Some(prev)
                if prev.seed_mode == SEED_MODE_BOUNDED_CONTROL
                    && b.seed_mode == SEED_MODE_BOUNDED_CONTROL
                    && prev.seed_layout.is_empty()
                    && !b.seed_layout.is_empty() =>
            {
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

fn writeonly_entry_buffer_locations(ll: &str) -> HashSet<u32> {
    let Some(args) = primary_entry_function_args(ll) else {
        return HashSet::new();
    };
    let arg_to_buf = arg_index_to_buffer_location(ll);
    args.split(',')
        .enumerate()
        .filter(|(_, arg)| arg.contains("writeonly"))
        .filter_map(|(ord, _)| arg_to_buf.get(&ord).copied())
        .collect()
}

fn module_uses_bounded_control_buffers(ll: &str, loop_bound_bufs: &HashSet<u32>) -> bool {
    if !loop_bound_bufs.is_empty() {
        return true;
    }
    ll.lines().any(|line| {
        line.contains("air.buffer")
            && line.contains("air.location_index")
            && !line.contains("air.texture")
            && !line.contains("air.sampler")
            && extract_i32_after(line, "air.address_space") != Some(3)
            && is_control_param_buffer_meta(line, extract_i32_after(line, "air.buffer_size"))
    })
}

fn stride_control_buffer_locations(ll: &str, loop_bound_bufs: &HashSet<u32>) -> HashSet<u32> {
    let arg_to_buf = arg_index_to_buffer_location(ll);
    let arg_name_to_buf = arg_name_to_buffer_location(ll, &arg_to_buf);
    let mut ptr_buf: HashMap<&str, u32> = HashMap::new();
    for arg in entry_function_args(ll)
        .into_iter()
        .flat_map(|args| args.split(',').enumerate())
    {
        if let Some(&buf) = arg_to_buf.get(&arg.0) {
            if let Some(name) = arg.1.rsplit_once('%').map(|(_, name)| name.trim()) {
                if !name.is_empty() {
                    ptr_buf.insert(name, buf);
                }
            }
        }
    }

    let mut reg_sources: HashMap<String, HashSet<u32>> = HashMap::new();
    let mut stride_controls = HashSet::new();
    let mut branch_controls = HashSet::new();
    for line in ll.lines() {
        let Some((reg, rhs)) = split_assign(line) else {
            let trimmed = line.trim_start();
            if trimmed.starts_with("br i1") {
                if let Some(cond) = first_percent_reg(trimmed) {
                    if let Some(sources) = reg_sources.get(cond) {
                        branch_controls.extend(sources.iter().copied());
                    }
                }
            }
            continue;
        };
        let mut sources = HashSet::new();
        if rhs.starts_with("load ") {
            if let Some(buf) = first_buf_operand(rhs, &ptr_buf, &arg_to_buf, &arg_name_to_buf) {
                sources.insert(buf);
            }
        }
        for operand in percent_operands(rhs) {
            if let Some(prev) = reg_sources.get(operand) {
                sources.extend(prev.iter().copied());
            }
        }
        if rhs.contains("getelementptr") && rhs.contains("ptr addrspace(1)") {
            stride_controls.extend(
                sources
                    .iter()
                    .copied()
                    .filter(|buf| !loop_bound_bufs.contains(buf)),
            );
        }
        if !sources.is_empty() {
            reg_sources.insert(reg.to_string(), sources);
        }
    }
    stride_controls.retain(|buf| !branch_controls.contains(buf));
    stride_controls
}

fn infer_textures(ll: &str) -> Vec<PlanTexture> {
    let locations = stage_resource_locations(ll);
    let mut out = Vec::new();
    for line in ll.lines() {
        if !line.contains("air.texture") || !line.contains("air.location_index") {
            continue;
        }
        let loc = metadata_param_index(line)
            .and_then(|idx| locations.textures.get(&idx).copied())
            .or_else(|| extract_i32_after(line, "air.location_index").map(|loc| loc as u32))
            .unwrap_or(0);
        let count = literal_location_index_count(line).unwrap_or(1).max(1);
        let type_name = quoted_metadata_string_after(line, "air.arg_type_name");
        let format = texture_format_from_air_type(type_name.as_deref());
        for offset in 0..count {
            let index = loc.wrapping_add(offset);
            out.push(PlanTexture {
                index,
                format: format.into(),
                role: texture_role_from_air_meta(line).into(),
                w: DEFAULT_TEXTURE_EXTENT.width,
                h: DEFAULT_TEXTURE_EXTENT.height,
                d: texture_plan_depth(ll, type_name.as_deref()),
                seed_tag: index.wrapping_add(1),
                seed_mode: finite_float_texture_seed_mode(format)
                    .unwrap_or(SEED_MODE_DETERMINISTIC)
                    .into(),
            });
        }
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

#[derive(Default)]
struct StageResourceLocations {
    buffers: HashMap<u32, u32>,
    textures: HashMap<u32, u32>,
}

fn stage_resource_locations(ll: &str) -> StageResourceLocations {
    match stage_from_ll(ll) {
        Stage::Fragment => {
            let Some(meta) = metal2vulkan::meta::parse_air_fragment_meta(ll) else {
                return StageResourceLocations::default();
            };
            let mut locations = StageResourceLocations::default();
            for (idx, role) in meta.roles {
                match role {
                    FragRole::Buffer(loc) => {
                        locations.buffers.insert(idx, loc);
                    }
                    FragRole::Texture(loc) => {
                        locations.textures.insert(idx, loc);
                    }
                    _ => {}
                }
            }
            locations
        }
        Stage::Vertex => {
            let Some(meta) = metal2vulkan::meta::parse_air_vertex_meta(ll) else {
                return StageResourceLocations::default();
            };
            let mut locations = StageResourceLocations::default();
            for (idx, role) in meta.roles {
                match role {
                    VertRole::Buffer(loc) => {
                        locations.buffers.insert(idx, loc);
                    }
                    VertRole::Texture(loc) => {
                        locations.textures.insert(idx, loc);
                    }
                    _ => {}
                }
            }
            locations
        }
        Stage::Kernel => {
            let Some(meta) = metal2vulkan::meta::parse_air_kernel_meta(ll) else {
                return StageResourceLocations::default();
            };
            let mut locations = StageResourceLocations::default();
            for (idx, role) in meta.roles {
                match role {
                    KernRole::Buffer(loc) => {
                        locations.buffers.insert(idx, loc);
                    }
                    KernRole::Texture(loc) => {
                        locations.textures.insert(idx, loc);
                    }
                    _ => {}
                }
            }
            locations
        }
    }
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
        "float" if texture_type_is_writable(type_name) => "R32Float",
        "float" => "Rgba32Float",
        "ushort" => "Rgba16Uint",
        "uint" | "uchar" => "Rgba8Uint",
        "int" | "char" | "short" => "Rgba8Sint",
        _ => "Rgba32Float",
    }
}

fn texture_type_is_writable(type_name: &str) -> bool {
    type_name.contains(", write") || type_name.contains(", read_write")
}

fn finite_float_texture_seed_mode(format: &str) -> Option<&'static str> {
    match format {
        "Rgba16Float" | "R16Float" | "Rg16Float" => Some(SEED_MODE_FINITE_FLOAT16),
        "Rgba32Float" | "R32Float" | "Rg32Float" | "Depth32Float" => Some(SEED_MODE_FINITE_FLOAT32),
        _ => None,
    }
}

fn texture_plan_depth(ll: &str, type_name: Option<&str>) -> u32 {
    match type_name {
        Some(name) if name.starts_with("texturecube<") => 6,
        Some(name) if name.starts_with("texture2d_array<") => {
            max_literal_sample_texture_array_layer(ll)
                .and_then(|layer| layer.checked_add(1))
                .unwrap_or(DEFAULT_TEXTURE_EXTENT.depth)
                .max(DEFAULT_TEXTURE_EXTENT.depth)
        }
        Some(name) if name.starts_with("texturecube_array<") => 1,
        _ => DEFAULT_TEXTURE_EXTENT.depth,
    }
}

fn max_literal_sample_texture_array_layer(ll: &str) -> Option<u32> {
    ll.lines()
        .filter(|line| line.contains("@air.sample_texture_2d_array."))
        .filter_map(literal_sample_texture_array_layer)
        .max()
}

fn literal_sample_texture_array_layer(line: &str) -> Option<u32> {
    let (_, tail) = line.split_once(", i32 ")?;
    let token = tail
        .trim_start()
        .chars()
        .take_while(|ch| ch.is_ascii_digit())
        .collect::<String>();
    token.parse().ok()
}

fn fragment_render_target_format(ll: &str) -> Option<&'static str> {
    if let Some(meta) = metal2vulkan::meta::parse_air_fragment_meta(ll) {
        if let Some((_, member)) = meta
            .render_target_members
            .iter()
            .min_by_key(|(_, loc)| *loc)
        {
            if let Some(format) = meta
                .render_target_type_name(*member)
                .and_then(fragment_output_format_from_air_type)
            {
                return Some(format);
            }
        }
    }
    if fragment_writes_depth(ll) {
        Some("Depth32Float")
    } else {
        None
    }
}

fn fragment_output_format_from_air_type(type_name: &str) -> Option<&'static str> {
    Some(match type_name {
        "half" => "R16Float",
        "half2" => "Rg16Float",
        "half3" | "half4" => "Rgba16Float",
        "float" => "R32Float",
        "float2" => "Rg32Float",
        "float3" | "float4" => "Rgba32Float",
        "ushort" => "R16Uint",
        "ushort2" => "Rg16Uint",
        "ushort3" | "ushort4" => "Rgba16Uint",
        "short" => "R16Sint",
        "short2" => "Rg16Sint",
        "short3" | "short4" => "Rgba16Sint",
        "uint" => "R32Uint",
        "uint2" => "Rg32Uint",
        "uint3" | "uint4" => "Rgba32Uint",
        "int" => "R32Sint",
        "int2" => "Rg32Sint",
        "int3" | "int4" => "Rgba32Sint",
        _ => return None,
    })
}

fn unsupported_fragment_color_output_arity(ll: &str) -> Option<String> {
    let meta = metal2vulkan::meta::parse_air_fragment_meta(ll)?;
    let mut render_targets: Vec<_> = meta.render_target_members.iter().collect();
    render_targets.sort_by_key(|(_, location)| *location);

    for (member, location) in render_targets {
        let Some(type_name) = meta.render_target_type_name(*member) else {
            continue;
        };
        if fragment_output_type_has_three_components(type_name) {
            return Some(format!(
                "unsupported Metal fragment color output attachment arity: render target location \
                 {location} uses AIR type {type_name:?}, but Metal has no renderable RGB color \
                 attachment format for a full golden"
            ));
        }
    }
    None
}

fn fragment_output_type_has_three_components(type_name: &str) -> bool {
    matches!(
        type_name,
        "half3" | "float3" | "ushort3" | "short3" | "uint3" | "int3"
    )
}

fn buffer_output_format(ll: &str, output_index: u32) -> Option<&'static str> {
    for line in ll.lines() {
        if !line.contains("air.buffer") || !line.contains("air.location_index") {
            continue;
        }
        let loc = extract_i32_after(line, "air.location_index")? as u32;
        if loc != output_index {
            continue;
        }
        if !(line.contains("air.read_write")
            || line.contains("!\"air.write\"")
            || line.contains("\"air.write\""))
        {
            continue;
        }
        let type_name = quoted_metadata_string_after(line, "air.arg_type_name");
        if let Some(format) = buffer_output_format_from_air_type(type_name.as_deref()) {
            return Some(format);
        }
        return buffer_output_format_from_struct_type_info(ll, line);
    }
    None
}

fn buffer_output_format_from_struct_type_info(
    ll: &str,
    buffer_meta_line: &str,
) -> Option<&'static str> {
    let node = metadata_ref_after(buffer_meta_line, "air.struct_type_info")?;
    let line = metadata_node_line(ll, node)?;
    let fields = quoted_metadata_strings(line);
    if fields.len() != 2 {
        return buffer_output_format_from_multi_field_struct_type_info(ll, line);
    }
    buffer_output_format_from_air_type(fields.first().map(String::as_str))
        .or_else(|| buffer_output_format_from_multi_field_struct_type_info(ll, line))
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum FloatStructScalarKind {
    F16,
    F32,
}

fn buffer_output_format_from_multi_field_struct_type_info(
    ll: &str,
    line: &str,
) -> Option<&'static str> {
    let kind = float_struct_scalar_kind(ll, line)?;
    Some(match kind {
        FloatStructScalarKind::F16 => "R16Float",
        FloatStructScalarKind::F32 => "F32",
    })
}

fn float_struct_scalar_kind(ll: &str, line: &str) -> Option<FloatStructScalarKind> {
    let payload = metadata_payload(line)?;
    let tokens = metadata_tokens(payload);
    let mut pending_nested = Vec::new();
    let mut kind = None;
    let mut i = 0;
    while i < tokens.len() {
        if let Some(node) = metadata_ref_token(tokens[i]) {
            pending_nested.push(node);
            i += 1;
            continue;
        }
        if i + 4 >= tokens.len()
            || metadata_i32_token(tokens[i]).is_none()
            || metadata_i32_token(tokens[i + 1]).is_none()
            || metadata_i32_token(tokens[i + 2]).is_none()
        {
            i += 1;
            continue;
        }
        let type_name = metadata_quoted_token(tokens[i + 3])?;
        let field_kind = float_scalar_kind_from_air_type(type_name).or_else(|| {
            let node = pending_nested.pop()?;
            let nested = metadata_node_line(ll, node)?;
            float_struct_scalar_kind(ll, nested)
        })?;
        if kind.is_some_and(|kind| kind != field_kind) {
            return None;
        }
        kind = Some(field_kind);
        i += 5;
    }
    kind
}

fn float_scalar_kind_from_air_type(type_name: &str) -> Option<FloatStructScalarKind> {
    let name = type_name.trim().trim_end_matches('*').trim();
    let scalar = name
        .strip_prefix("packed_")
        .unwrap_or(name)
        .trim_end_matches(|c: char| c.is_ascii_digit())
        .trim_end_matches(|c: char| c == 'x' || c.is_ascii_digit());
    match scalar {
        "half" => Some(FloatStructScalarKind::F16),
        "float" => Some(FloatStructScalarKind::F32),
        _ => None,
    }
}

fn buffer_output_format_from_air_type(type_name: Option<&str>) -> Option<&'static str> {
    let name = type_name.unwrap_or("").trim().trim_end_matches('*').trim();
    match name {
        "half" => Some("R16Float"),
        "half2" => Some("Rg16Float"),
        "half3" | "half4" => Some("Rgba16Float"),
        "float" => Some("F32"),
        "float2" => Some("Rg32Float"),
        "float3" | "float4" => Some("Rgba32Float"),
        "packed_float2" => Some("Rg32Float"),
        "packed_float3" | "packed_float4" => Some("Rgba32Float"),
        _ => matrix_buffer_output_format_from_air_type(name),
    }
}

fn finite_float_buffer_seed_mode(type_name: Option<&str>) -> Option<&'static str> {
    let name = type_name?.trim().trim_end_matches('*').trim();
    let scalar = name
        .strip_prefix("packed_")
        .unwrap_or(name)
        .trim_end_matches(|c: char| c.is_ascii_digit());
    match scalar {
        "half" => Some(SEED_MODE_FINITE_FLOAT16),
        "bfloat" => Some(SEED_MODE_FINITE_BFLOAT16),
        "float" => Some(SEED_MODE_FINITE_FLOAT32),
        _ => float_scalar_kind_from_air_type(name).map(|kind| match kind {
            FloatStructScalarKind::F16 => SEED_MODE_FINITE_FLOAT16,
            FloatStructScalarKind::F32 => SEED_MODE_FINITE_FLOAT32,
        }),
    }
}

fn matrix_buffer_output_format_from_air_type(name: &str) -> Option<&'static str> {
    let scalar = float_scalar_kind_from_air_type(name)?;
    let columns = matrix_or_vector_column_count(name)?;
    Some(match (scalar, columns) {
        (FloatStructScalarKind::F16, 1) => "R16Float",
        (FloatStructScalarKind::F16, 2) => "Rg16Float",
        (FloatStructScalarKind::F16, _) => "Rgba16Float",
        (FloatStructScalarKind::F32, 1) => "F32",
        (FloatStructScalarKind::F32, 2) => "Rg32Float",
        (FloatStructScalarKind::F32, _) => "Rgba32Float",
    })
}

fn matrix_or_vector_column_count(name: &str) -> Option<u8> {
    let suffix = name
        .strip_prefix("packed_")
        .unwrap_or(name)
        .strip_prefix("half")
        .or_else(|| {
            name.strip_prefix("packed_")
                .unwrap_or(name)
                .strip_prefix("float")
        })?;
    let first = suffix.chars().next().filter(|ch| matches!(ch, '1'..='4'))?;
    let first = first as u8 - b'0';
    let second = suffix
        .split_once('x')
        .and_then(|(_, rows)| rows.chars().next())
        .filter(|ch| matches!(ch, '1'..='4'))
        .map(|ch| ch as u8 - b'0')
        .unwrap_or(first);
    Some(first.max(second))
}

fn finite_struct_float_seed_layout(
    ll: &str,
    buffer_meta_line: &str,
    len: usize,
) -> Option<(usize, Vec<ControlSeedField>)> {
    let stride = extract_i32_after(buffer_meta_line, "air.arg_type_size")? as usize;
    if stride == 0 || stride > len.max(1) {
        return None;
    }
    let node = metadata_ref_after(buffer_meta_line, "air.struct_type_info")?;
    let line = metadata_node_line(ll, node)?;
    let payload = metadata_payload(line)?;
    let tokens = metadata_tokens(payload);
    let mut fields = Vec::new();
    let mut i = 0;
    while i + 3 < tokens.len() {
        let Some(offset) = metadata_i32_token(tokens[i]) else {
            i += 1;
            continue;
        };
        let Some(byte_size) = metadata_i32_token(tokens[i + 1]) else {
            i += 1;
            continue;
        };
        let Some(repeat_count) = metadata_i32_token(tokens[i + 2]) else {
            i += 1;
            continue;
        };
        let Some(type_name) = metadata_quoted_token(tokens[i + 3]) else {
            i += 1;
            continue;
        };
        if let Some((elem_size, lanes)) = finite_float_field_shape(type_name, byte_size) {
            let offset = offset as usize;
            let total_lanes = if lanes == 1 && repeat_count > 0 {
                repeat_count as usize
            } else {
                lanes
            };
            for lane in 0..total_lanes {
                let lane_offset = offset + lane * elem_size;
                if lane_offset.saturating_add(elem_size) <= stride {
                    fields.push(ControlSeedField {
                        offset: lane_offset,
                        size: elem_size,
                        value: None,
                    });
                }
            }
        }
        i += 5;
    }
    (!fields.is_empty()).then_some((stride, fields))
}

fn finite_float_field_shape(type_name: &str, byte_size: i32) -> Option<(usize, usize)> {
    let name = type_name.trim().trim_end_matches('*').trim();
    let name = name.strip_prefix("packed_").unwrap_or(name);
    let scalar = name.trim_end_matches(|c: char| c.is_ascii_digit());
    let elem_size = match scalar {
        "half" | "bfloat" => 2usize,
        "float" => 4usize,
        _ => return None,
    };
    let lane_suffix = &name[scalar.len()..];
    let lanes = if lane_suffix.is_empty() {
        1usize
    } else {
        lane_suffix.parse::<usize>().ok()?
    };
    let byte_size = usize::try_from(byte_size).ok()?;
    (lanes > 0 && lanes.saturating_mul(elem_size) <= byte_size).then_some((elem_size, lanes))
}

fn bounded_control_float_payload_len(type_name: Option<&str>) -> Option<usize> {
    let elem_size = match finite_float_buffer_seed_mode(type_name)? {
        SEED_MODE_FINITE_FLOAT16 => 2,
        SEED_MODE_FINITE_BFLOAT16 => 2,
        SEED_MODE_FINITE_FLOAT32 => 4,
        _ => return None,
    };
    let dim = BOUNDED_CONTROL_DIM as usize;
    Some(dim * dim * elem_size)
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

/// Lightweight IR scan: buffer locations whose scalar integer loads appear in an `icmp` that
/// feeds a `br` (trip-count / early-out class). Used to catch device-space counters that
/// are not tagged as constant-param structs.
///
/// Not a full relooper — false positives only force small integer seeds on that buffer,
/// which is safe for execution harnesses (goldens re-derived under the new seed profile).
fn buffers_with_loads_used_as_loop_bounds(ll: &str) -> HashSet<u32> {
    let Some(body) = entry_function_body(ll) else {
        return HashSet::new();
    };
    let cyclic_blocks = cyclic_cfg_blocks(body);
    // arg ordinal/name → buffer location_index from AIR kernel/vertex/fragment arg metadata.
    let arg_to_buf = arg_index_to_buffer_location(ll);
    let arg_name_to_buf = arg_name_to_buffer_location(ll, &arg_to_buf);
    if arg_to_buf.is_empty() {
        return HashSet::new();
    }

    // reg → buffer location, for values that are the arg pointer or GEP/bitcast of it.
    let mut ptr_buf: HashMap<&str, u32> = HashMap::new();
    // reg → buffer location for scalar integer loads from those pointers.
    let mut int_from_buf: HashMap<&str, u32> = HashMap::new();
    // icmp reg → buffer if either side is an integer loaded from that buffer.
    let mut icmp_from_buf: HashMap<&str, u32> = HashMap::new();
    let mut branched: HashSet<u32> = HashSet::new();
    let mut current_block: Option<&str> = None;

    for line in body.lines() {
        let line = line.trim();
        if let Some(label) = block_label(line) {
            current_block = Some(label);
            continue;
        }
        if let Some((reg, rhs)) = split_assign(line) {
            if rhs.starts_with("getelementptr") || rhs.starts_with("bitcast") {
                if let Some(buf) = first_buf_operand(rhs, &ptr_buf, &arg_to_buf, &arg_name_to_buf) {
                    ptr_buf.insert(reg, buf);
                }
                continue;
            }
            // %N = load i32/i64/etc, ptr ... %p
            if is_scalar_integer_load_rhs(rhs) {
                if let Some(buf) = first_buf_operand(rhs, &ptr_buf, &arg_to_buf, &arg_name_to_buf) {
                    int_from_buf.insert(reg, buf);
                }
                continue;
            }
            // Preserve provenance through integer casts into the cmp.
            if rhs.starts_with("zext ") || rhs.starts_with("sext ") || rhs.starts_with("trunc ") {
                if let Some(src) = first_percent_reg(rhs) {
                    if let Some(&buf) = int_from_buf.get(src) {
                        int_from_buf.insert(reg, buf);
                    }
                }
                continue;
            }
            if let Some(rest) = rhs.strip_prefix("icmp ") {
                let mut hit = None;
                for tok in rest.split([',', ' ']) {
                    let t = tok.trim();
                    if let Some(name) = t.strip_prefix('%') {
                        if let Some(&buf) = int_from_buf.get(name) {
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
            if !current_block.is_some_and(|block| cyclic_blocks.contains(block)) {
                continue;
            }
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

fn is_scalar_integer_load_rhs(rhs: &str) -> bool {
    let Some(rest) = rhs.strip_prefix("load ") else {
        return false;
    };
    let Some((ty, _)) = rest.split_once(',') else {
        return false;
    };
    let ty = ty.trim();
    ty.strip_prefix('i').is_some_and(|bits| {
        bits.parse::<u32>()
            .is_ok_and(|n| matches!(n, 1 | 8 | 16 | 32 | 64))
    })
}

fn cyclic_cfg_blocks(body: &str) -> HashSet<&str> {
    let mut current = None;
    let mut succs: HashMap<&str, Vec<&str>> = HashMap::new();
    for line in body.lines().map(str::trim) {
        if let Some(label) = block_label(line) {
            current = Some(label);
            succs.entry(label).or_default();
            continue;
        }
        let Some(block) = current else {
            continue;
        };
        if line.starts_with("br ") {
            succs.entry(block).or_default().extend(branch_labels(line));
        }
    }

    let mut out = HashSet::new();
    for &start in succs.keys() {
        let mut stack = succs.get(start).cloned().unwrap_or_default();
        let mut seen = HashSet::new();
        while let Some(next) = stack.pop() {
            if next == start {
                out.insert(start);
                break;
            }
            if !seen.insert(next) {
                continue;
            }
            if let Some(next_succs) = succs.get(next) {
                stack.extend(next_succs.iter().copied());
            }
        }
    }
    out
}

fn block_label(line: &str) -> Option<&str> {
    let label = line.strip_suffix(':')?;
    (!label.is_empty() && !label.starts_with(';') && !label.contains(' ') && !label.contains('\t'))
        .then_some(label)
}

fn branch_labels(line: &str) -> Vec<&str> {
    line.split(',')
        .filter_map(|part| {
            let part = part.trim();
            let label = part.strip_prefix("label %")?;
            Some(
                label
                    .split(|c: char| c == ',' || c.is_whitespace())
                    .next()
                    .unwrap_or(label),
            )
        })
        .collect()
}

/// Buffer locations read through AIR global i32 atomics.
///
/// These are counter/control sources even when the AIR metadata describes the root buffer as a
/// read-only raw payload (`uchar*`). Arbitrary deterministic bytes make those counters enormous or
/// backend-specific; bounded-control seeding keeps the validation input in the same small-counter
/// family as constant params.
fn buffers_with_atomic_i32_loads(ll: &str) -> HashSet<u32> {
    let Some(body) = entry_function_body(ll) else {
        return HashSet::new();
    };
    let arg_to_buf = arg_index_to_buffer_location(ll);
    let arg_name_to_buf = arg_name_to_buffer_location(ll, &arg_to_buf);
    if arg_to_buf.is_empty() {
        return HashSet::new();
    }

    let mut ptr_buf: HashMap<&str, u32> = HashMap::new();
    let mut out = HashSet::new();
    for line in body.lines() {
        let line = line.trim();
        let Some((reg, rhs)) = split_assign(line) else {
            continue;
        };
        if rhs.starts_with("getelementptr") || rhs.starts_with("bitcast") {
            if let Some(buf) = first_buf_operand(rhs, &ptr_buf, &arg_to_buf, &arg_name_to_buf) {
                ptr_buf.insert(reg, buf);
            }
            continue;
        }
        if rhs.contains("@air.atomic.global.load.i32") {
            if let Some(buf) = first_buf_operand(rhs, &ptr_buf, &arg_to_buf, &arg_name_to_buf) {
                out.insert(buf);
            }
        }
    }
    out
}

/// Constant/control buffers whose floating-point fields flow to an `fdiv` denominator.
///
/// Bounded-control seeds normally write the integer value 16 into every lane. For `float` fields that
/// bit pattern is a tiny subnormal; under AIR denorm-flush semantics it becomes zero, and denominator
/// expressions can turn into `0/0` before a later clamp. Marking the whole control buffer keeps this a
/// structural dataflow rule rather than relying on field names.
fn buffers_with_float_loads_used_as_fdiv_denominators(ll: &str) -> HashSet<u32> {
    let Some(body) = entry_function_body(ll) else {
        return HashSet::new();
    };
    let arg_to_buf = arg_index_to_buffer_location(ll);
    let arg_name_to_buf = arg_name_to_buffer_location(ll, &arg_to_buf);
    if arg_to_buf.is_empty() {
        return HashSet::new();
    }

    let mut ptr_buf: HashMap<&str, u32> = HashMap::new();
    let mut float_from_buf: HashMap<&str, u32> = HashMap::new();
    let mut out = HashSet::new();
    for line in body.lines() {
        let line = line.trim();
        let Some((reg, rhs)) = split_assign(line) else {
            continue;
        };
        if rhs.starts_with("getelementptr") || rhs.starts_with("bitcast") {
            if let Some(buf) = first_buf_operand(rhs, &ptr_buf, &arg_to_buf, &arg_name_to_buf) {
                ptr_buf.insert(reg, buf);
            }
            continue;
        }
        if is_float_load_rhs(rhs) {
            if let Some(buf) = first_buf_operand(rhs, &ptr_buf, &arg_to_buf, &arg_name_to_buf) {
                float_from_buf.insert(reg, buf);
            }
            continue;
        }
        if rhs.starts_with("fdiv ") {
            if let Some((_, denom)) = rhs.split_once(',') {
                for name in percent_operands(denom) {
                    if let Some(&buf) = float_from_buf.get(name) {
                        out.insert(buf);
                    }
                }
            }
        }
        if rhs.starts_with("fadd ")
            || rhs.starts_with("fsub ")
            || rhs.starts_with("fmul ")
            || rhs.starts_with("fdiv ")
            || rhs.starts_with("fpext ")
            || rhs.starts_with("fptrunc ")
        {
            if let Some(buf) = percent_operands(rhs)
                .into_iter()
                .find_map(|name| float_from_buf.get(name).copied())
            {
                float_from_buf.insert(reg, buf);
            }
        }
    }
    out
}

fn is_float_load_rhs(rhs: &str) -> bool {
    let Some(rest) = rhs.strip_prefix("load ") else {
        return false;
    };
    let Some((ty, _)) = rest.split_once(',') else {
        return false;
    };
    let ty = ty.trim();
    matches!(ty, "half" | "float" | "double")
        || (ty.starts_with('<') && (ty.contains(" x half>") || ty.contains(" x float>")))
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

fn entry_function_args(ll: &str) -> Option<&str> {
    let start = ll.find("\ndefine ")?;
    let after = &ll[start + 1..];
    let open = after.find('(')?;
    let args_start = start + 1 + open + 1;
    let rest = &ll[args_start..];
    let close = rest.find(") {")?;
    Some(&rest[..close])
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

fn arg_name_to_buffer_location(ll: &str, arg_to_buf: &HashMap<usize, u32>) -> HashMap<String, u32> {
    let Some(args) = entry_function_args(ll) else {
        return HashMap::new();
    };
    let mut map = HashMap::new();
    for (ord, arg) in args.split(',').enumerate() {
        let Some(&buf) = arg_to_buf.get(&ord) else {
            continue;
        };
        let Some(name) = arg.rsplit_once('%').map(|(_, name)| name) else {
            continue;
        };
        let name = name.trim();
        if !name.is_empty() {
            map.insert(name.to_string(), buf);
        }
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

fn percent_operands(s: &str) -> Vec<&str> {
    s.split('%')
        .skip(1)
        .filter_map(|rest| {
            let name = rest
                .split(|c: char| !c.is_ascii_alphanumeric() && c != '.' && c != '_')
                .next()?;
            (!name.is_empty()).then_some(name)
        })
        .collect()
}

/// Resolve the first SSA operand that is a known buffer pointer (arg ordinal or tracked reg).
fn first_buf_operand(
    rhs: &str,
    ptr_buf: &HashMap<&str, u32>,
    arg_to_buf: &HashMap<usize, u32>,
    arg_name_to_buf: &HashMap<String, u32>,
) -> Option<u32> {
    // Prefer named regs that we already classified.
    let mut best: Option<u32> = None;
    for tok in rhs.split([',', ' ', '(', ')']) {
        let t = tok.trim();
        if let Some(name) = t.strip_prefix('%') {
            if let Some(&buf) = ptr_buf.get(name) {
                return Some(buf);
            }
            if let Some(&buf) = arg_name_to_buf.get(name) {
                best = Some(buf);
                continue;
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
    bounded_control_buffer_bytes_with_layout(len, &[])
}

fn bounded_control_buffer_bytes_with_layout(
    len: usize,
    seed_layout: &[ControlSeedField],
) -> Vec<u8> {
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
    for field in seed_layout {
        let end = field.offset.saturating_add(field.size);
        if end > out.len() {
            continue;
        }
        let value = field.value.unwrap_or(u64::from(dim));
        let bytes = value.to_le_bytes();
        match field.size {
            1 | 2 | 4 | 8 => out[field.offset..end].copy_from_slice(&bytes[..field.size]),
            _ => {}
        }
    }
    out
}

fn bounded_control_seed_layout(
    ll: &str,
    buffer_meta_line: &str,
    len: usize,
    stride_control_bufs: &HashSet<u32>,
    fdiv_denominator_control_bufs: &HashSet<u32>,
) -> Vec<ControlSeedField> {
    let Some(node) = metadata_ref_after(buffer_meta_line, "air.struct_type_info") else {
        return scalar_bounded_control_seed_layout(
            buffer_meta_line,
            len,
            stride_control_bufs,
            fdiv_denominator_control_bufs,
        );
    };
    let loc = extract_i32_after(buffer_meta_line, "air.location_index").map(|v| v as u32);
    let seed_float_one = loc.is_some_and(|loc| fdiv_denominator_control_bufs.contains(&loc));
    let Some(line) = metadata_node_line(ll, node) else {
        return Vec::new();
    };
    let Some(payload) = metadata_payload(line) else {
        return Vec::new();
    };
    let tokens = metadata_tokens(payload);
    let mut fields = Vec::new();
    let mut i = 0;
    while i + 3 < tokens.len() {
        let Some(offset) = metadata_i32_token(tokens[i]) else {
            i += 1;
            continue;
        };
        let Some(byte_size) = metadata_i32_token(tokens[i + 1]) else {
            i += 1;
            continue;
        };
        if metadata_i32_token(tokens[i + 2]).is_none() {
            i += 1;
            continue;
        }
        let Some(type_name) = metadata_quoted_token(tokens[i + 3]) else {
            i += 1;
            continue;
        };
        if let Some(size) = bounded_control_field_seed_size(type_name, byte_size) {
            let offset = offset as usize;
            if offset < len && offset.saturating_add(size) <= len {
                fields.push(ControlSeedField {
                    offset,
                    size,
                    value: bounded_control_field_seed_value(type_name, seed_float_one),
                });
            }
        }
        i += 5;
    }
    fields
}

fn scalar_bounded_control_seed_layout(
    buffer_meta_line: &str,
    len: usize,
    stride_control_bufs: &HashSet<u32>,
    fdiv_denominator_control_bufs: &HashSet<u32>,
) -> Vec<ControlSeedField> {
    let Some(type_name) = quoted_metadata_string_after(buffer_meta_line, "air.arg_type_name")
    else {
        return Vec::new();
    };
    let Some(byte_size) = extract_i32_after(buffer_meta_line, "air.arg_type_size") else {
        return Vec::new();
    };
    let Some(size) = bounded_control_field_seed_size(&type_name, byte_size) else {
        return Vec::new();
    };
    if size <= len {
        let loc = extract_i32_after(buffer_meta_line, "air.location_index").map(|v| v as u32);
        let value = loc
            .filter(|loc| stride_control_bufs.contains(loc))
            .map(|_| 1)
            .or_else(|| {
                let seed_float_one =
                    loc.is_some_and(|loc| fdiv_denominator_control_bufs.contains(&loc));
                bounded_control_field_seed_value(&type_name, seed_float_one)
            });
        vec![ControlSeedField {
            offset: 0,
            size,
            value,
        }]
    } else {
        Vec::new()
    }
}

fn metadata_payload(line: &str) -> Option<&str> {
    let start = line.find("!{")? + 2;
    let end = line.rfind('}')?;
    (start <= end).then_some(&line[start..end])
}

fn metadata_tokens(payload: &str) -> Vec<&str> {
    payload
        .split(',')
        .map(str::trim)
        .filter(|tok| !tok.is_empty())
        .collect()
}

fn metadata_i32_token(token: &str) -> Option<i32> {
    token.trim().strip_prefix("i32 ")?.trim().parse().ok()
}

fn metadata_quoted_token(token: &str) -> Option<&str> {
    token.trim().strip_prefix("!\"")?.strip_suffix('"')
}

fn metadata_ref_token(token: &str) -> Option<u32> {
    let token = token.trim();
    if !token.starts_with('!') || token.starts_with("!\"") {
        return None;
    }
    token.strip_prefix('!')?.parse().ok()
}

fn bounded_control_field_seed_size(type_name: &str, byte_size: i32) -> Option<usize> {
    let name = type_name.trim().trim_end_matches('*').trim();
    let size = match name {
        "uchar" | "char" | "bool" => 1,
        "ushort" | "short" | "half" => 2,
        "uint" | "int" | "float" => 4,
        "ulong" | "long" | "double" => 8,
        _ => return None,
    };
    (byte_size as usize == size).then_some(size)
}

fn bounded_control_field_seed_value(type_name: &str, _seed_float_one: bool) -> Option<u64> {
    let name = type_name.trim().trim_end_matches('*').trim();
    match name {
        "half" => Some(0x3c00),
        "float" => Some(0x3f80_0000),
        "double" => Some(0x3ff0_0000_0000_0000),
        _ => None,
    }
}

fn finite_float_buffer_bytes(len: usize, index: u32, tag: u32, elem_size: usize) -> Vec<u8> {
    debug_assert!(matches!(elem_size, 2 | 4));
    let input = BufferInput {
        index,
        len,
        role: BufferRole::Input,
        seed: Seed::Deterministic { tag },
    };
    let mut bytes = seeded_buffer_bytes(&input);
    sanitize_float_buffer_finite(&mut bytes, elem_size);
    bytes
}

fn finite_struct_float_buffer_bytes(
    len: usize,
    index: u32,
    tag: u32,
    stride: usize,
    layout: &[ControlSeedField],
) -> Vec<u8> {
    let input = BufferInput {
        index,
        len,
        role: BufferRole::Input,
        seed: Seed::Deterministic { tag },
    };
    let mut bytes = seeded_buffer_bytes(&input);
    if stride == 0 {
        return bytes;
    }
    let mut base = 0usize;
    while base < bytes.len() {
        for field in layout {
            let start = base.saturating_add(field.offset);
            let end = start.saturating_add(field.size);
            if end <= bytes.len() && matches!(field.size, 2 | 4) {
                sanitize_float_buffer_finite(&mut bytes[start..end], field.size);
            }
        }
        base = base.saturating_add(stride);
    }
    bytes
}

fn sanitize_float_buffer_finite(bytes: &mut [u8], elem_size: usize) {
    let mut hi = elem_size - 1;
    while hi < bytes.len() {
        bytes[hi] &= !0x40;
        hi += elem_size;
    }
}

fn contains_nonfinite_float_lane(bytes: &[u8], elem_size: usize) -> bool {
    match elem_size {
        2 => bytes.chunks_exact(2).any(|lane| {
            let bits = u16::from_le_bytes([lane[0], lane[1]]);
            ((bits >> 10) & 0x1f) == 0x1f
        }),
        4 => bytes.chunks_exact(4).any(|lane| {
            let bits = u32::from_le_bytes([lane[0], lane[1], lane[2], lane[3]]);
            ((bits >> 23) & 0xff) == 0xff
        }),
        _ => false,
    }
}

#[cfg(test)]
fn contains_nonfinite_bfloat_lane(bytes: &[u8]) -> bool {
    bytes.chunks_exact(2).any(|lane| {
        let bits = u16::from_le_bytes([lane[0], lane[1]]);
        ((bits >> 7) & 0xff) == 0xff
    })
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

fn metadata_param_index(line: &str) -> Option<u32> {
    extract_i32_after(line, "!{").and_then(|idx| u32::try_from(idx).ok())
}

fn literal_location_index_count(line: &str) -> Option<u32> {
    let payload = metadata_payload(line)?;
    let tokens = metadata_tokens(payload);
    for (i, token) in tokens.iter().enumerate() {
        if metadata_quoted_token(token) != Some("air.location_index") {
            continue;
        }
        let loc = tokens.get(i + 1).and_then(|tok| metadata_i32_token(tok))?;
        let count = tokens.get(i + 2).and_then(|tok| metadata_i32_token(tok))?;
        if loc < 0 || count <= 0 {
            return None;
        }
        return u32::try_from(count).ok();
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

fn quoted_metadata_strings(line: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut rest = line;
    while let Some(start) = rest.find("!\"") {
        rest = &rest[start + 2..];
        let Some(end) = rest.find('"') else {
            break;
        };
        out.push(rest[..end].to_string());
        rest = &rest[end + 1..];
    }
    out
}

fn metadata_ref_after(line: &str, key: &str) -> Option<u32> {
    let idx = line.find(key)?;
    let rest = &line[idx + key.len()..];
    for tok in rest.split([',', ' ', '}']) {
        let Some(num) = tok.trim().strip_prefix('!') else {
            continue;
        };
        if !num.is_empty() && num.bytes().all(|b| b.is_ascii_digit()) {
            return num.parse().ok();
        }
    }
    None
}

fn metadata_node_line(ll: &str, node: u32) -> Option<&str> {
    let prefix = format!("!{node} = ");
    ll.lines()
        .find(|line| line.trim_start().starts_with(&prefix))
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
        .map(|b| -> Result<BufferInput, String> {
            let seed = match b.seed_mode.as_str() {
                SEED_MODE_BOUNDED_CONTROL => {
                    let bytes = bounded_control_buffer_bytes_with_layout(b.len, &b.seed_layout);
                    let leaked: &'static [u8] = Box::leak(bytes.into_boxed_slice());
                    Seed::ExactBytes {
                        bytes: leaked,
                        reason: "bounded_control_param_buffer",
                    }
                }
                SEED_MODE_FINITE_FLOAT16 => {
                    let bytes = finite_float_buffer_bytes(b.len, b.index, b.seed_tag, 2);
                    let leaked: &'static [u8] = Box::leak(bytes.into_boxed_slice());
                    Seed::ExactBytes {
                        bytes: leaked,
                        reason: "finite_float16_buffer",
                    }
                }
                SEED_MODE_FINITE_BFLOAT16 => {
                    let bytes = finite_float_buffer_bytes(b.len, b.index, b.seed_tag, 2);
                    let leaked: &'static [u8] = Box::leak(bytes.into_boxed_slice());
                    Seed::ExactBytes {
                        bytes: leaked,
                        reason: "finite_bfloat16_buffer",
                    }
                }
                SEED_MODE_FINITE_FLOAT32 => {
                    let bytes = finite_float_buffer_bytes(b.len, b.index, b.seed_tag, 4);
                    let leaked: &'static [u8] = Box::leak(bytes.into_boxed_slice());
                    Seed::ExactBytes {
                        bytes: leaked,
                        reason: "finite_float32_buffer",
                    }
                }
                SEED_MODE_FINITE_STRUCT_FLOAT => {
                    let Some(stride) = b.seed_stride else {
                        return Err(format!(
                            "buffer {} finite_struct_float seed missing stride",
                            b.index
                        ));
                    };
                    let bytes = finite_struct_float_buffer_bytes(
                        b.len,
                        b.index,
                        b.seed_tag,
                        stride,
                        &b.seed_layout,
                    );
                    let leaked: &'static [u8] = Box::leak(bytes.into_boxed_slice());
                    Seed::ExactBytes {
                        bytes: leaked,
                        reason: "finite_struct_float_buffer",
                    }
                }
                _ => Seed::Deterministic { tag: b.seed_tag },
            };
            Ok(BufferInput {
                index: b.index,
                len: b.len,
                role: parse_buffer_role(&b.role),
                seed,
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let textures: Vec<TextureInput> = plan
        .textures
        .iter()
        .map(|t| {
            let seed = match t.seed_mode.as_str() {
                SEED_MODE_FINITE_FLOAT16 | SEED_MODE_FINITE_FLOAT32 => {
                    Seed::DeterministicFinite { tag: t.seed_tag }
                }
                _ => Seed::Deterministic { tag: t.seed_tag },
            };
            Ok(TextureInput {
                index: t.index,
                format: parse_format(&t.format)?,
                extent: Extent3d::new(t.w.max(1), t.h.max(1), t.d.max(1)),
                role: parse_texture_role(&t.role),
                seed,
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
        "R16Uint" => DataFormat::R16Uint,
        "Rg16Uint" => DataFormat::Rg16Uint,
        "Rgba16Uint" => DataFormat::Rgba16Uint,
        "R32Uint" => DataFormat::R32Uint,
        "Rg32Uint" => DataFormat::Rg32Uint,
        "Rgba32Uint" => DataFormat::Rgba32Uint,
        "R16Sint" => DataFormat::R16Sint,
        "Rg16Sint" => DataFormat::Rg16Sint,
        "Rgba16Sint" => DataFormat::Rgba16Sint,
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

fn candidate_compare_format(ll: &str, plan: &HarnessPlan, metal: &MetalRow) -> DataFormat {
    current_output_format_for_plan(ll, &plan.output)
        .and_then(|format| parse_format(format).ok())
        .or_else(|| parse_format(&plan.output.format).ok())
        .or_else(|| parse_format(&metal.plan.output.format).ok())
        .unwrap_or(DataFormat::RawBytes)
}

fn current_output_format_for_plan(ll: &str, output: &PlanOutput) -> Option<&'static str> {
    match output.kind.as_str() {
        "buffer" => buffer_output_format(ll, output.index),
        "texture" => infer_textures(ll)
            .into_iter()
            .find(|texture| texture.index == output.index)
            .map(|texture| match texture.format.as_str() {
                "Rgba16Float" => "Rgba16Float",
                "Rgba32Float" => "Rgba32Float",
                "Rgba16Uint" => "Rgba16Uint",
                "Rgba8Uint" => "Rgba8Uint",
                "Rgba8Sint" => "Rgba8Sint",
                "R32Float" => "R32Float",
                _ => "Rgba32Float",
            }),
        "render_target" => fragment_render_target_format(ll),
        _ => None,
    }
}

fn incompatible_function_constant_golden(ll: &str, metal: &MetalRow) -> Option<String> {
    if !declares_air_function_constants(ll) {
        return None;
    }
    match metal.fc_specialization.as_deref() {
        Some(FC_SPECIALIZATION_ZERO) => None,
        Some(FC_SPECIALIZATION_VALUES) if metal.fc_values.as_deref().is_some_and(|v| !v.is_empty()) => None,
        Some(FC_SPECIALIZATION_VALUES) => Some(
            "metal golden declares function-constant value mode without fc_values; rebank Metal row"
                .into(),
        ),
        Some(other) => Some(format!(
            "metal golden has unsupported function-constant specialization {other:?}; rebank Metal row"
        )),
        None => Some(
            "metal golden lacks explicit function-constant specialization metadata; rebank Metal row"
                .into(),
        ),
    }
}

fn incompatible_zero_function_constant_divisor_golden(
    ll: &str,
    metal: &MetalRow,
) -> Option<String> {
    if metal.fc_specialization.as_deref() != Some(FC_SPECIALIZATION_ZERO) {
        return None;
    }
    if !zero_function_constant_feeds_integer_divisor(ll) {
        return None;
    }
    Some(
        "metal golden zero-specializes an AIR function constant that feeds an integer div/rem \
         denominator; the Metal result observes undefined divide-by-zero behavior, so rebank with \
         nonzero function-constant values or drop the row"
            .into(),
    )
}

fn incompatible_texture_array_plan_golden(ll: &str, metal: &MetalRow) -> Option<String> {
    if !declares_fixed_texture_array_metadata(ll) {
        return None;
    }
    let current_textures = texture_plan_signature(&infer_textures(ll));
    let banked_textures = texture_plan_signature(&metal.plan.textures);
    if current_textures == banked_textures {
        return None;
    }
    Some(format!(
        "metal golden input texture plan [{}] differs from current AIR texture-array plan [{}]; \
         rebank Metal row",
        banked_textures.join(","),
        current_textures.join(","),
    ))
}

fn incompatible_function_constant_definedness_golden(ll: &str, metal: &MetalRow) -> Option<String> {
    if !matches!(
        metal.fc_specialization.as_deref(),
        Some(FC_SPECIALIZATION_ZERO | FC_SPECIALIZATION_VALUES)
    ) || !ll.contains("@air.is_function_constant_defined")
    {
        return None;
    }
    Some(
        "metal golden explicitly specializes an AIR function constant used by \
         air.is_function_constant_defined; the current candidate path specializes initializer values \
         but still lowers definedness to the unspecialized false default, so this is not a \
         comparable validation oracle yet; rebank after definedness-aware specialization or drop the \
         row"
            .into(),
    )
}

fn zero_function_constant_feeds_integer_divisor(ll: &str) -> bool {
    let fc_globals = zero_function_constant_working_globals(ll);
    if fc_globals.is_empty() {
        return false;
    }

    let mut fc_values = std::collections::HashSet::new();
    for line in ll.lines() {
        if let Some((result, global)) = load_result_and_global(line) {
            if fc_globals.contains(&global) {
                fc_values.insert(result);
                continue;
            }
        }

        if let Some(result) = cast_result_from_fc_value(line, &fc_values) {
            fc_values.insert(result);
            continue;
        }

        if integer_div_rem_denominator(line).is_some_and(|value| fc_values.contains(&value)) {
            return true;
        }
    }
    false
}

fn zero_function_constant_working_globals(ll: &str) -> std::collections::HashSet<String> {
    let mut fc_loaded_values = std::collections::HashSet::new();
    let mut globals = std::collections::HashSet::new();
    for line in ll.lines() {
        if line.contains(".MTL_FC_INIT_") {
            if let Some((result, _)) = load_result_and_global(line) {
                fc_loaded_values.insert(result);
            }
            continue;
        }
        let Some((stored_value, dest_global)) = store_value_and_dest_global(line) else {
            continue;
        };
        if fc_loaded_values.contains(&stored_value) {
            globals.insert(dest_global);
        }
    }
    globals
}

fn load_result_and_global(line: &str) -> Option<(String, String)> {
    if !line.contains(" = load ") {
        return None;
    }
    let result = instruction_result_name(line)?;
    let global = global_name_after(line, "ptr addrspace(")?;
    Some((result, global))
}

fn store_value_and_dest_global(line: &str) -> Option<(String, String)> {
    let store = line.trim_start().strip_prefix("store ")?;
    let (value_part, dest_part) = store.split_once(',')?;
    let stored_value = value_part.split_whitespace().last()?.to_string();
    let dest_global = global_name_after(dest_part, "ptr addrspace(")?;
    Some((stored_value, dest_global))
}

fn instruction_result_name(line: &str) -> Option<String> {
    let (result, _) = line.trim_start().split_once(" = ")?;
    result.starts_with('%').then(|| result.to_string())
}

fn global_name_after(line: &str, marker: &str) -> Option<String> {
    let (_, tail) = line.split_once(marker)?;
    let at = tail.find('@')?;
    let global = &tail[at..];
    let end = global
        .find(|ch: char| ch == ',' || ch.is_whitespace())
        .unwrap_or(global.len());
    Some(global[..end].to_string())
}

fn cast_result_from_fc_value(
    line: &str,
    fc_values: &std::collections::HashSet<String>,
) -> Option<String> {
    if ![" zext ", " sext ", " trunc "]
        .iter()
        .any(|opcode| line.contains(opcode))
    {
        return None;
    }
    let result = instruction_result_name(line)?;
    fc_values
        .iter()
        .any(|value| llvm_value_token_occurs(line, value))
        .then_some(result)
}

fn llvm_value_token_occurs(line: &str, value: &str) -> bool {
    line.split(|ch: char| !(ch.is_ascii_alphanumeric() || ch == '_' || ch == '%' || ch == '.'))
        .any(|token| token == value)
}

fn integer_div_rem_denominator(line: &str) -> Option<String> {
    if ![" sdiv ", " udiv ", " srem ", " urem "]
        .iter()
        .any(|opcode| line.contains(opcode))
    {
        return None;
    }
    line.rsplit_once(',')?
        .1
        .split_whitespace()
        .next()
        .map(str::to_string)
}

fn declares_air_function_constants(ll: &str) -> bool {
    ll.lines().any(|line| {
        line.contains(".MTL_FC_INIT_")
            && line.contains("section \"air.fc_initializer\"")
            && line.contains("externally_initialized")
    })
}

fn incompatible_output_plan_golden(ll: &str, metal: &MetalRow) -> Option<String> {
    let current_plan = infer_plan(ll);
    let current = &current_plan.output;
    let banked = &metal.plan.output;
    if !output_plans_match(current, banked) {
        return Some(format!(
            "metal golden output plan {} differs from current AIR output plan {}; rebank Metal row",
            plan_output_summary(banked),
            plan_output_summary(current),
        ));
    }
    if metal.plan.dispatch_grid != current_plan.dispatch_grid
        || metal.plan.dispatch_tg != current_plan.dispatch_tg
    {
        return Some(format!(
            "metal golden dispatch plan grid={:?} tg={:?} differs from current AIR dispatch plan \
             grid={:?} tg={:?}; rebank Metal row",
            metal.plan.dispatch_grid,
            metal.plan.dispatch_tg,
            current_plan.dispatch_grid,
            current_plan.dispatch_tg,
        ));
    }
    None
}

fn incompatible_static_resource_plan_golden(ll: &str, metal: &MetalRow) -> Option<String> {
    if metal.fc_specialization.as_deref() != Some(FC_SPECIALIZATION_ZERO) {
        return None;
    }
    if !ll.lines().any(|line| {
        (line.contains(r#""air.texture""#) || line.contains(r#""air.buffer""#))
            && line.contains(r#""air.location_index", ptr addrspace(2)"#)
    }) {
        return None;
    }
    let current_plan = infer_plan(ll);
    let current_buffers = buffer_plan_signature(&current_plan.buffers);
    let banked_buffers = buffer_plan_signature(&metal.plan.buffers);
    let current_textures = texture_plan_signature(&current_plan.textures);
    let banked_textures = texture_plan_signature(&metal.plan.textures);
    if current_buffers == banked_buffers && current_textures == banked_textures {
        return None;
    }
    Some(format!(
        "metal golden input resource plan buffers=[{}] textures=[{}] differs from current AIR \
         static-location plan buffers=[{}] textures=[{}]; rebank Metal row",
        banked_buffers.join(","),
        banked_textures.join(","),
        current_buffers.join(","),
        current_textures.join(","),
    ))
}

fn buffer_plan_signature(buffers: &[PlanBuffer]) -> Vec<String> {
    let mut out = buffers
        .iter()
        .map(|buffer| format!("{}:{}:{}", buffer.index, buffer.len, buffer.role))
        .collect::<Vec<_>>();
    out.sort();
    out
}

fn texture_plan_signature(textures: &[PlanTexture]) -> Vec<String> {
    let mut out = textures
        .iter()
        .map(|texture| {
            format!(
                "{}:{}:{}:{}x{}x{}",
                texture.index, texture.format, texture.role, texture.w, texture.h, texture.d
            )
        })
        .collect::<Vec<_>>();
    out.sort();
    out
}

fn declares_fixed_texture_array_metadata(ll: &str) -> bool {
    ll.lines().any(|line| {
        line.contains("air.texture")
            && literal_location_index_count(line).is_some_and(|count| count > 1)
    }) || max_literal_sample_texture_array_layer(ll).is_some_and(|layer| layer > 0)
}

fn declares_fragment_render_target(ll: &str) -> bool {
    metal2vulkan::meta::parse_air_fragment_meta(ll)
        .map(|meta| !meta.render_target_members.is_empty())
        .unwrap_or(false)
}

fn incompatible_undefined_fragment_color_output_golden(
    ll: &str,
    plan: &HarnessPlan,
    spv: &[u8],
) -> Option<String> {
    if plan.output.kind != "render_target" || plan.output.format == "Depth32Float" {
        return None;
    }
    if !declares_fragment_render_target(ll) {
        return None;
    }
    if crate::runner_linux::fragment_writes_color_location(spv, plan.output.index) {
        return None;
    }
    Some(
        "metal golden compares a fragment render target member that lowers to no Vulkan color output \
         because the AIR return value is undefined; rebank or drop the Metal row"
            .into(),
    )
}

fn incompatible_compare_none_loop_guard_golden(
    ll: &str,
    entry: &str,
    metal: &MetalRow,
) -> Option<String> {
    if metal.compare != "none" {
        return None;
    }
    match crate::loop_budget::classify_and_instrument(ll, entry) {
        crate::loop_budget::GuardPlan::Quarantine(reason) => Some(format!(
            "metal golden compare=none cannot be reproduced by current loop guard: {reason}; \
             rebank Metal row"
        )),
        crate::loop_budget::GuardPlan::Instrumented(_)
        | crate::loop_budget::GuardPlan::LoopFree => None,
    }
}

fn incompatible_nonportable_ptrtoint_golden(ll: &str, metal: &MetalRow) -> Option<String> {
    if metal.compare == "none" || !declares_nonportable_pointer_address_materialization(ll) {
        return None;
    }
    Some(
        "metal golden observes backend-specific AIR ptrtoint device/constant pointer address bits; \
         rebank or drop Metal row"
            .into(),
    )
}

fn incompatible_parallel_dynamic_buffer_scatter_golden(
    ll: &str,
    metal: &MetalRow,
) -> Option<String> {
    if metal.compare == "none"
        || metal.plan.output.kind != "buffer"
        || metal.plan.output.format != "RawBytes"
        || !metal.plan.dispatch_grid.iter().any(|&n| n > 1)
        || !ll.contains("\"air.thread_position_in_grid\"")
        || !ll.contains("@air.atomic.global.")
    {
        return None;
    }
    let output = metal
        .plan
        .buffers
        .iter()
        .find(|buffer| buffer.index == metal.plan.output.index)?;
    if output.role != "InOut" {
        return None;
    }
    if !declares_data_dependent_non_atomic_store_to_buffer(ll, metal.plan.output.index) {
        return None;
    }
    Some(
        "metal golden compares parallel non-atomic dynamic buffer scatter output whose write order \
         is backend-schedule-dependent; rebank Metal row with a serial/smoke plan"
            .into(),
    )
}

fn incompatible_undefined_threadgroup_memory_golden(ll: &str, metal: &MetalRow) -> Option<String> {
    if metal.compare == "none" || !declares_undefined_threadgroup_memory_read(ll) {
        return None;
    }
    Some(
        "metal golden observes AIR threadgroup memory loaded through thread_index_in_simdgroup \
         after only simdgroup_index_in_threadgroup-indexed stores; unwritten threadgroup lanes are \
         undefined, so rebank or drop the Metal row"
            .into(),
    )
}

#[derive(Clone, Copy, Default)]
struct ThreadgroupIndexOrigin {
    simdgroup_index: bool,
    simd_lane_index: bool,
}

impl ThreadgroupIndexOrigin {
    fn simdgroup_index() -> Self {
        Self {
            simdgroup_index: true,
            simd_lane_index: false,
        }
    }

    fn simd_lane_index() -> Self {
        Self {
            simdgroup_index: false,
            simd_lane_index: true,
        }
    }

    fn merge(&mut self, other: Self) {
        self.simdgroup_index |= other.simdgroup_index;
        self.simd_lane_index |= other.simd_lane_index;
    }
}

fn declares_undefined_threadgroup_memory_read(ll: &str) -> bool {
    if !ll.contains("addrspace(3)") || !ll.contains("@air.wg.barrier") {
        return false;
    }
    let Some(body) = primary_entry_function_body(ll) else {
        return false;
    };
    let Some(args) = primary_entry_function_args(ll) else {
        return false;
    };
    let mut value_origin: HashMap<&str, ThreadgroupIndexOrigin> = HashMap::new();
    for arg in arg_names_for_metadata_key(args, ll, "air.simdgroup_index_in_threadgroup") {
        value_origin.insert(arg, ThreadgroupIndexOrigin::simdgroup_index());
    }
    for arg in arg_names_for_metadata_key(args, ll, "air.thread_index_in_simdgroup") {
        value_origin.insert(arg, ThreadgroupIndexOrigin::simd_lane_index());
    }
    if value_origin.is_empty() {
        return false;
    }

    let mut ptr_origin: HashMap<&str, ThreadgroupIndexOrigin> = HashMap::new();
    let mut saw_simdgroup_indexed_store = false;
    let mut after_barrier = false;
    for line in body.lines().map(str::trim) {
        if line.contains("@air.wg.barrier") {
            after_barrier = true;
        }
        if line.starts_with("store ") && line.contains("ptr addrspace(3)") {
            if percent_operands(line).into_iter().any(|name| {
                ptr_origin
                    .get(name)
                    .is_some_and(|origin| origin.simdgroup_index)
            }) {
                saw_simdgroup_indexed_store = true;
            }
            continue;
        }
        if line.starts_with("%")
            && line.contains("= load ")
            && line.contains("ptr addrspace(3)")
            && after_barrier
            && saw_simdgroup_indexed_store
            && percent_operands(line).into_iter().any(|name| {
                ptr_origin
                    .get(name)
                    .is_some_and(|origin| origin.simd_lane_index)
            })
        {
            return true;
        }

        let Some((reg, rhs)) = split_assign(line) else {
            continue;
        };
        let mut origin = ThreadgroupIndexOrigin::default();
        for operand in percent_operands(rhs) {
            if let Some(prev) = value_origin.get(operand) {
                origin.merge(*prev);
            }
        }
        if origin.simdgroup_index || origin.simd_lane_index {
            value_origin.insert(reg, origin);
        }
        if rhs.starts_with("getelementptr") && rhs.contains("addrspace(3)") {
            let mut ptr = ThreadgroupIndexOrigin::default();
            for operand in percent_operands(rhs) {
                if let Some(prev) = value_origin.get(operand) {
                    ptr.merge(*prev);
                }
            }
            if ptr.simdgroup_index || ptr.simd_lane_index {
                ptr_origin.insert(reg, ptr);
            }
        }
    }
    false
}

#[derive(Clone, Copy, Default)]
struct LlValueOrigin {
    thread_position: bool,
    buffer_or_atomic: bool,
}

impl LlValueOrigin {
    fn thread_position() -> Self {
        Self {
            thread_position: true,
            buffer_or_atomic: false,
        }
    }

    fn buffer_or_atomic() -> Self {
        Self {
            thread_position: false,
            buffer_or_atomic: true,
        }
    }

    fn merge(&mut self, other: Self) {
        self.thread_position |= other.thread_position;
        self.buffer_or_atomic |= other.buffer_or_atomic;
    }
}

fn declares_data_dependent_non_atomic_store_to_buffer(ll: &str, output_buffer: u32) -> bool {
    let Some(body) = primary_entry_function_body(ll) else {
        return false;
    };
    let Some(args) = primary_entry_function_args(ll) else {
        return false;
    };
    let arg_to_buf = arg_index_to_buffer_location(ll);
    let arg_name_to_buf = arg_name_to_buffer_location_from_args(args, &arg_to_buf);
    if arg_to_buf.is_empty() {
        return false;
    }

    let mut ptr_buf: HashMap<&str, u32> = HashMap::new();
    let mut value_origin: HashMap<&str, LlValueOrigin> = HashMap::new();
    for arg in arg_names_for_metadata_key(args, ll, "air.thread_position_in_grid") {
        value_origin.insert(arg, LlValueOrigin::thread_position());
    }
    let mut data_dependent_output_ptrs: HashSet<&str> = HashSet::new();

    for line in body.lines().map(str::trim) {
        if line.starts_with("store ")
            && line.contains("ptr addrspace(1)")
            && first_buf_operand(line, &ptr_buf, &arg_to_buf, &arg_name_to_buf)
                == Some(output_buffer)
            && percent_operands(line)
                .into_iter()
                .any(|name| data_dependent_output_ptrs.contains(name))
        {
            return true;
        }

        let Some((reg, rhs)) = split_assign(line) else {
            continue;
        };
        if rhs.starts_with("getelementptr") || rhs.starts_with("bitcast") {
            let buf = first_buf_operand(rhs, &ptr_buf, &arg_to_buf, &arg_name_to_buf);
            if let Some(buf) = buf {
                ptr_buf.insert(reg, buf);
            }
            if buf == Some(output_buffer)
                && (data_dependent_output_ptr(rhs, &data_dependent_output_ptrs)
                    || rhs
                        .split(',')
                        .skip(2)
                        .flat_map(percent_operands)
                        .any(|name| {
                            value_origin
                                .get(name)
                                .is_some_and(|origin| origin.buffer_or_atomic)
                        }))
            {
                data_dependent_output_ptrs.insert(reg);
            }
            continue;
        }

        let mut origin = LlValueOrigin::default();
        for operand in percent_operands(rhs) {
            if let Some(prev) = value_origin.get(operand) {
                origin.merge(*prev);
            }
        }
        if rhs.starts_with("load ") {
            if first_buf_operand(rhs, &ptr_buf, &arg_to_buf, &arg_name_to_buf).is_some() {
                origin.merge(LlValueOrigin::buffer_or_atomic());
            }
        } else if rhs.contains("@air.atomic.global.")
            && first_buf_operand(rhs, &ptr_buf, &arg_to_buf, &arg_name_to_buf).is_some()
        {
            origin.merge(LlValueOrigin::buffer_or_atomic());
        }
        if origin.thread_position || origin.buffer_or_atomic {
            value_origin.insert(reg, origin);
        }
    }
    false
}

fn primary_entry_function_body(ll: &str) -> Option<&str> {
    entry_name_from_ll(ll)
        .and_then(|entry| function_body_for_entry(ll, &entry))
        .or_else(|| entry_function_body(ll))
}

fn primary_entry_function_args(ll: &str) -> Option<&str> {
    entry_name_from_ll(ll)
        .and_then(|entry| function_args_for_entry(ll, &entry))
        .or_else(|| entry_function_args(ll))
}

fn function_body_for_entry<'a>(ll: &'a str, entry: &str) -> Option<&'a str> {
    let (start, brace) = function_header_for_entry(ll, entry)?;
    let body_start = start + 1 + brace + 1;
    let rest = &ll[body_start..];
    let end = rest.find("\n}")?;
    Some(&rest[..end])
}

fn function_args_for_entry<'a>(ll: &'a str, entry: &str) -> Option<&'a str> {
    let (start, brace) = function_header_for_entry(ll, entry)?;
    let after = &ll[start + 1..];
    let header = &after[..brace];
    let unquoted = format!("@{entry}(");
    let quoted = format!("@\"{entry}\"(");
    let open = header
        .find(&unquoted)
        .map(|pos| pos + unquoted.len() - 1)
        .or_else(|| header.find(&quoted).map(|pos| pos + quoted.len() - 1))?;
    let close = header.rfind(')')?;
    (open < close).then_some(&header[open + 1..close])
}

fn function_header_for_entry(ll: &str, entry: &str) -> Option<(usize, usize)> {
    let unquoted = format!("@{entry}(");
    let quoted = format!("@\"{entry}\"(");
    for (start, _) in ll.match_indices("\ndefine ") {
        let after = &ll[start + 1..];
        let Some(brace) = after.find('{') else {
            continue;
        };
        let header = &after[..brace];
        if header.contains(&unquoted) || header.contains(&quoted) {
            return Some((start, brace));
        }
    }
    None
}

fn data_dependent_output_ptr(rhs: &str, data_dependent_output_ptrs: &HashSet<&str>) -> bool {
    percent_operands(rhs)
        .into_iter()
        .any(|name| data_dependent_output_ptrs.contains(name))
}

fn arg_name_to_buffer_location_from_args(
    args: &str,
    arg_to_buf: &HashMap<usize, u32>,
) -> HashMap<String, u32> {
    let mut map = HashMap::new();
    for (ord, arg) in args.split(',').enumerate() {
        let Some(&buf) = arg_to_buf.get(&ord) else {
            continue;
        };
        let Some(name) = arg.rsplit_once('%').map(|(_, name)| name) else {
            continue;
        };
        let name = name.trim();
        if !name.is_empty() {
            map.insert(name.to_string(), buf);
        }
    }
    map
}

fn arg_names_for_metadata_key<'a>(args: &'a str, ll: &str, key: &str) -> Vec<&'a str> {
    let ords: HashSet<usize> = ll
        .lines()
        .filter(|line| line.contains(key))
        .filter_map(extract_meta_first_i32)
        .filter_map(|ord| usize::try_from(ord).ok())
        .collect();
    args.split(',')
        .enumerate()
        .filter_map(|(ord, arg)| {
            if !ords.contains(&ord) {
                return None;
            }
            arg.rsplit_once('%')
                .map(|(_, name)| name.trim())
                .filter(|name| !name.is_empty())
        })
        .collect()
}

fn declares_nonportable_pointer_address_materialization(ll: &str) -> bool {
    let modeled_payloads = modeled_pointer_payload_values(ll);
    ll.lines().any(|line| {
        let Some(source) = ptrtoint_buffer_pointer_source(line) else {
            return false;
        };
        !modeled_payloads.contains(source)
    })
}

fn modeled_pointer_payload_values(ll: &str) -> HashSet<String> {
    ll.lines()
        .filter(|line| {
            line.contains(
                "@air.get_primitive_acceleration_structure_instance_acceleration_structure",
            ) || line.contains("@air.get_data_pointer_instance_acceleration_structure")
        })
        .filter_map(llvm_assignment_value)
        .map(str::to_string)
        .collect()
}

fn ptrtoint_buffer_pointer_source(line: &str) -> Option<&str> {
    let (_, rhs) = line.split_once("ptrtoint ")?;
    let (from, _) = rhs.split_once(" to ")?;
    if !from.contains("addrspace(1)") && !from.contains("addrspace(2)") {
        return None;
    }
    from.split_whitespace().last().filter(|token| {
        token
            .strip_prefix('%')
            .or_else(|| token.strip_prefix('@'))
            .is_some_and(|name| {
                !name.is_empty()
                    && name
                        .chars()
                        .all(|c| c == '.' || c == '_' || c == '-' || c.is_ascii_alphanumeric())
            })
    })
}

fn incompatible_point_coord_golden(ll: &str, metal: &MetalRow) -> Option<String> {
    if !declares_fragment_point_coord(ll) || metal.plan_version >= POINT_COORD_TOPOLOGY_PLAN_VERSION
    {
        return None;
    }
    Some(
        "metal golden uses legacy triangle-only validation plan for AIR point_coord; rebank Metal row with topology-aware plan"
            .into(),
    )
}

fn declares_fragment_point_coord(ll: &str) -> bool {
    ll.lines().any(|line| line.contains("\"air.point_coord\""))
}

fn incompatible_undefined_texture_write_lanes_golden(ll: &str, metal: &MetalRow) -> Option<String> {
    if metal.plan.output.kind != "texture" || !metal.plan.output.format.starts_with("Rgba") {
        return None;
    }
    let undef_texels = undef_lane_texture_write_values(ll);
    if undef_texels.is_empty() {
        return None;
    }
    if ll.lines().any(|line| {
        line.contains("@air.write_texture")
            && undef_texels
                .iter()
                .any(|value| llvm_line_uses_value(line, value))
    }) {
        return Some(
            "metal golden observes undefined lanes from an AIR texture write texel backed by an \
             RGBA validation texture; rebank Metal row"
                .into(),
        );
    }
    None
}

fn undef_lane_texture_write_values(ll: &str) -> HashSet<String> {
    let mut values: HashSet<String> = HashSet::new();
    for line in ll
        .lines()
        .filter(|line| line.contains("insertelement <4 x"))
    {
        let Some(value) = llvm_assignment_value(line) else {
            continue;
        };
        let undef_base = line.contains(" undef,") || line.contains(" poison,");
        let undef_chain = values.iter().any(|known| llvm_line_uses_value(line, known));
        if undef_base || undef_chain {
            values.insert(value.to_string());
        }
    }
    values
}

fn llvm_assignment_value(line: &str) -> Option<&str> {
    let (lhs, _) = line.split_once('=')?;
    let value = lhs.trim();
    value.starts_with('%').then_some(value)
}

fn llvm_line_uses_value(line: &str, value: &str) -> bool {
    line.split(|ch: char| {
        ch.is_whitespace() || matches!(ch, ',' | '(' | ')' | '[' | ']' | '{' | '}')
    })
    .any(|token| token == value)
}

fn output_plans_match(current: &PlanOutput, banked: &PlanOutput) -> bool {
    current.kind == banked.kind
        && current.index == banked.index
        && current.format == banked.format
        && current.len == banked.len
        && current.w == banked.w
        && current.h == banked.h
        && current.d == banked.d
}

fn plan_output_summary(output: &PlanOutput) -> String {
    match output.kind.as_str() {
        "buffer" => format!(
            "buffer(index={}, format={}, len={})",
            output.index,
            output.format,
            output
                .len
                .map(|len| len.to_string())
                .unwrap_or_else(|| "none".into())
        ),
        _ => format!(
            "{}(index={}, format={}, extent={}x{}x{})",
            output.kind,
            output.index,
            output.format,
            output.w.unwrap_or(0),
            output.h.unwrap_or(0),
            output.d.unwrap_or(0)
        ),
    }
}

fn incompatible_bounded_control_seed_golden(ll: &str, metal: &MetalRow) -> Option<String> {
    let current = infer_buffers(ll);
    for banked in metal
        .plan
        .buffers
        .iter()
        .filter(|b| b.seed_mode == SEED_MODE_BOUNDED_CONTROL)
    {
        let Some(current) = current.iter().find(|b| b.index == banked.index) else {
            continue;
        };
        if current.seed_mode != banked.seed_mode {
            return Some(format!(
                "metal golden uses bounded_control buffer {} now seeded {}; rebank Metal row",
                current.index, current.seed_mode
            ));
        }
    }
    for current in current
        .iter()
        .filter(|b| b.seed_mode == SEED_MODE_BOUNDED_CONTROL)
    {
        let Some(banked) = metal.plan.buffers.iter().find(|b| b.index == current.index) else {
            continue;
        };
        if banked.seed_mode != current.seed_mode {
            return Some(format!(
                "metal golden uses {} buffer {} for AIR control/atomic counter input now seeded {}; rebank Metal row",
                banked.seed_mode, current.index, current.seed_mode
            ));
        }
        if banked.seed_layout != current.seed_layout
            && bounded_control_buffer_bytes_with_layout(banked.len, &banked.seed_layout)
                != bounded_control_buffer_bytes_with_layout(current.len, &current.seed_layout)
        {
            return Some(format!(
                "metal golden uses legacy bounded_control layout for buffer {} now seeded from typed AIR control metadata; rebank Metal row",
                current.index
            ));
        }
    }
    None
}

fn incompatible_oob_vector_input_golden(ll: &str, metal: &MetalRow) -> Option<String> {
    if metal.compare == "none" || metal.stage.as_deref() != Some("Kernel") {
        return None;
    }
    let grid_x = u64::from(metal.plan.dispatch_grid[0].max(1));
    for buffer in &metal.plan.buffers {
        if buffer.role != "Input" {
            continue;
        }
        let Some(type_name) = buffer_type_name_for_location(ll, buffer.index) else {
            continue;
        };
        let Some((llvm_ty, elem_bytes)) = llvm_thread_indexed_input_type_and_size(&type_name)
        else {
            continue;
        };
        let required_elems =
            dynamic_device_gep_required_elements(ll, llvm_ty, grid_x).unwrap_or(grid_x);
        if required_elems.saturating_mul(elem_bytes as u64) <= buffer.len as u64 {
            continue;
        }
        if module_has_dynamic_device_gep(ll, llvm_ty) {
            return Some(format!(
                "metal golden uses input buffer {} length {} for a thread-indexed {type_name} \
                 load over dispatch_grid.x={}; rebank with an input buffer large enough \
                 to avoid undefined out-of-bounds reads",
                buffer.index, buffer.len, metal.plan.dispatch_grid[0]
            ));
        }
    }
    None
}

fn buffer_type_name_for_location(ll: &str, location: u32) -> Option<String> {
    ll.lines().find_map(|line| {
        if !line.contains(r#""air.buffer""#) || !line.contains(r#""air.location_index""#) {
            return None;
        }
        let loc = extract_i32_after(line, "air.location_index")?;
        (loc as u32 == location).then(|| quoted_metadata_string_after(line, "air.arg_type_name"))?
    })
}

fn llvm_thread_indexed_input_type_and_size(type_name: &str) -> Option<(&'static str, usize)> {
    Some(match type_name {
        "char" | "uchar" => ("i8", 1),
        "char2" => ("<2 x i8>", 2),
        "char3" => ("<3 x i8>", 3),
        "char4" => ("<4 x i8>", 4),
        "uchar2" => ("<2 x i8>", 2),
        "uchar3" => ("<3 x i8>", 3),
        "uchar4" => ("<4 x i8>", 4),
        "short" | "ushort" => ("i16", 2),
        "short2" => ("<2 x i16>", 4),
        "short3" => ("<3 x i16>", 6),
        "short4" => ("<4 x i16>", 8),
        "ushort2" => ("<2 x i16>", 4),
        "ushort3" => ("<3 x i16>", 6),
        "ushort4" => ("<4 x i16>", 8),
        "int" | "uint" => ("i32", 4),
        "int2" => ("<2 x i32>", 8),
        "int3" => ("<3 x i32>", 12),
        "int4" => ("<4 x i32>", 16),
        "uint2" => ("<2 x i32>", 8),
        "uint3" => ("<3 x i32>", 12),
        "uint4" => ("<4 x i32>", 16),
        "long" | "ulong" => ("i64", 8),
        "half" => ("half", 2),
        "float" => ("float", 4),
        "float2" => ("<2 x float>", 8),
        "float3" => ("<3 x float>", 12),
        "float4" => ("<4 x float>", 16),
        "double" => ("double", 8),
        _ => return None,
    })
}

#[derive(Clone, Copy)]
struct LinearThreadIndex {
    coeff: u64,
    offset: u64,
}

fn dynamic_device_gep_required_elements(ll: &str, llvm_ty: &str, grid_x: u64) -> Option<u64> {
    let needle = format!("getelementptr inbounds {llvm_ty}, ptr addrspace(1)");
    let mut exprs: HashMap<String, LinearThreadIndex> = HashMap::new();
    let mut required = None;
    for line in ll.lines() {
        let line = line.trim();
        if let Some((result, src, factor)) = parse_mul_i32_const(line) {
            let src_expr = exprs.get(src).copied().unwrap_or(LinearThreadIndex {
                coeff: 1,
                offset: 0,
            });
            exprs.insert(
                result.to_string(),
                LinearThreadIndex {
                    coeff: src_expr.coeff.saturating_mul(u64::from(factor)),
                    offset: src_expr.offset.saturating_mul(u64::from(factor)),
                },
            );
            continue;
        }
        if let Some((result, src, addend)) = parse_add_i32_const(line) {
            let src_expr = exprs.get(src).copied().unwrap_or(LinearThreadIndex {
                coeff: 1,
                offset: 0,
            });
            exprs.insert(
                result.to_string(),
                LinearThreadIndex {
                    coeff: src_expr.coeff,
                    offset: src_expr.offset.saturating_add(u64::from(addend)),
                },
            );
            continue;
        }
        if let Some((result, src)) = parse_zext_i32_to_i64(line) {
            let src_expr = exprs.get(src).copied().unwrap_or(LinearThreadIndex {
                coeff: 1,
                offset: 0,
            });
            exprs.insert(result.to_string(), src_expr);
            continue;
        }
        if !line.contains(&needle) {
            continue;
        }
        let Some(index) = trailing_dynamic_gep_index(line) else {
            continue;
        };
        let expr = exprs.get(index).copied().unwrap_or(LinearThreadIndex {
            coeff: 1,
            offset: 0,
        });
        let elems = expr
            .coeff
            .saturating_mul(grid_x.saturating_sub(1))
            .saturating_add(expr.offset)
            .saturating_add(1);
        required = Some(required.unwrap_or(0).max(elems));
    }
    required
}

fn parse_mul_i32_const(line: &str) -> Option<(&str, &str, u32)> {
    let (result, rhs) = line.split_once(" = mul i32 ")?;
    let (src, factor) = rhs.split_once(", ")?;
    Some((result.trim(), src.trim(), factor.trim().parse().ok()?))
}

fn parse_add_i32_const(line: &str) -> Option<(&str, &str, u32)> {
    let (result, rhs) = line.split_once(" = add i32 ")?;
    let (src, addend) = rhs.split_once(", ")?;
    Some((result.trim(), src.trim(), addend.trim().parse().ok()?))
}

fn parse_zext_i32_to_i64(line: &str) -> Option<(&str, &str)> {
    let (result, rhs) = line.split_once(" = zext i32 ")?;
    let src = rhs.strip_suffix(" to i64")?;
    Some((result.trim(), src.trim()))
}

fn trailing_dynamic_gep_index(line: &str) -> Option<&str> {
    let idx = line.rfind(", i64 %").or_else(|| line.rfind(", i32 %"))?;
    let rest = line[idx + 2..].trim();
    rest.split_whitespace().nth(1)
}

fn module_has_dynamic_device_gep(ll: &str, llvm_ty: &str) -> bool {
    let needle = format!("getelementptr inbounds {llvm_ty}, ptr addrspace(1)");
    ll.lines().any(|line| {
        line.contains(&needle) && (line.contains(", i64 %") || line.contains(", i32 %"))
    })
}

fn incompatible_float_seed_golden(ll: &str, metal: &MetalRow) -> Option<String> {
    if !ll_has_fast_no_nans_float_semantics(ll) {
        return None;
    }
    let current_buffers = infer_buffers(ll);
    for current in current_buffers {
        if current.role == "Output" {
            continue;
        }
        if current.seed_mode == SEED_MODE_FINITE_STRUCT_FLOAT {
            let Some(banked) = metal.plan.buffers.iter().find(|b| b.index == current.index) else {
                continue;
            };
            match banked.seed_mode.as_str() {
                SEED_MODE_DETERMINISTIC => {
                    let raw = {
                        let input = BufferInput {
                            index: banked.index,
                            len: banked.len,
                            role: BufferRole::Input,
                            seed: Seed::Deterministic {
                                tag: banked.seed_tag,
                            },
                        };
                        seeded_buffer_bytes(&input)
                    };
                    let finite = finite_struct_float_buffer_bytes(
                        banked.len,
                        banked.index,
                        banked.seed_tag,
                        current.seed_stride.unwrap_or(0),
                        &current.seed_layout,
                    );
                    if raw != finite {
                        return Some(format!(
                            "metal golden uses deterministic float-struct buffer {} now sanitized for AIR fast/no-nans math; rebank Metal row",
                            current.index
                        ));
                    }
                }
                SEED_MODE_FINITE_STRUCT_FLOAT => {
                    let banked_bytes = finite_struct_float_buffer_bytes(
                        banked.len,
                        banked.index,
                        banked.seed_tag,
                        banked.seed_stride.unwrap_or(0),
                        &banked.seed_layout,
                    );
                    let current_bytes = finite_struct_float_buffer_bytes(
                        current.len,
                        current.index,
                        current.seed_tag,
                        current.seed_stride.unwrap_or(0),
                        &current.seed_layout,
                    );
                    if banked_bytes != current_bytes {
                        return Some(format!(
                            "metal golden uses legacy finite_struct_float seed for buffer {} now sized from AIR struct stride and dispatch; rebank Metal row",
                            current.index
                        ));
                    }
                }
                _ => {}
            }
            continue;
        }
        let Some(elem_size) = finite_seed_elem_size(&current.seed_mode) else {
            continue;
        };
        let Some(banked) = metal.plan.buffers.iter().find(|b| b.index == current.index) else {
            continue;
        };
        if banked.seed_mode != SEED_MODE_DETERMINISTIC {
            continue;
        }
        let finite =
            finite_float_buffer_bytes(banked.len, banked.index, banked.seed_tag, elem_size);
        if finite.len() != banked.len {
            continue;
        }
        let raw = {
            let input = BufferInput {
                index: banked.index,
                len: banked.len,
                role: BufferRole::Input,
                seed: Seed::Deterministic {
                    tag: banked.seed_tag,
                },
            };
            seeded_buffer_bytes(&input)
        };
        if raw != finite {
            return Some(format!(
                "metal golden uses deterministic {} buffer {} now sanitized for AIR fast/no-nans math; rebank Metal row",
                finite_seed_label(&current.seed_mode),
                current.index
            ));
        }
    }
    if let Some(reason) = incompatible_float_render_target_seed_golden(ll, metal) {
        return Some(reason);
    }
    if let Some(reason) = incompatible_float_texture_seed_golden(ll, metal) {
        return Some(reason);
    }
    None
}

fn incompatible_float_output_golden(ll: &str, metal: &MetalRow) -> Option<String> {
    if !ll_has_fast_no_nans_float_semantics(ll) {
        return None;
    }
    let format = current_output_format_for_plan(ll, &metal.plan.output)
        .and_then(|format| parse_format(format).ok())
        .or_else(|| parse_format(&metal.plan.output.format).ok())?;
    let elem_size = float_lane_size_for_compare(format)?;
    let output_b64 = metal.output_b64.as_deref()?;
    let golden = decode_output_b64(output_b64).ok()?;
    if !contains_nonfinite_float_lane(&golden, elem_size) {
        return None;
    }
    Some(format!(
        "metal golden output {} {} contains {} NaN/Inf lanes under AIR fast/no-nans math; rebank Metal row",
        metal.plan.output.kind,
        metal.plan.output.index,
        if elem_size == 2 { "f16" } else { "f32" }
    ))
}

fn incompatible_sampled_fast_pow_texture_golden(ll: &str, metal: &MetalRow) -> Option<String> {
    if !ll_has_fast_no_nans_float_semantics(ll)
        || !ll.contains("@air.sample_texture")
        || !ll.contains("@air.fast_pow.")
        || metal.plan.output.kind != "render_target"
    {
        return None;
    }
    let has_signed_sampled_f32_texture = infer_textures(ll).into_iter().any(|texture| {
        texture.role == "Sampled"
            && texture.format == "Rgba32Float"
            && texture.seed_mode == SEED_MODE_FINITE_FLOAT32
    });
    if !has_signed_sampled_f32_texture {
        return None;
    }
    Some(
        "metal golden samples signed finite f32 texture data through AIR fast_pow under fast/no-nans \
         math; Metal/Vulkan linear sampling and sign-sensitive pow behavior are not a portable \
         validation oracle for this synthetic seed; rebank or drop Metal row"
            .into(),
    )
}

fn incompatible_dependent_sampled_lookup_golden(ll: &str, metal: &MetalRow) -> Option<String> {
    if !ll_has_fast_no_nans_float_semantics(ll)
        || metal.plan.output.kind != "render_target"
        || !ll.contains("@air.sample_texture")
        || !ll.contains("@air.fast_ceil.")
        || !ll.contains("@air.fast_floor.")
    {
        return None;
    }
    let sampled_f32_count = infer_textures(ll)
        .into_iter()
        .filter(|texture| {
            texture.role == "Sampled"
                && texture.format == "Rgba32Float"
                && texture.seed_mode == SEED_MODE_FINITE_FLOAT32
        })
        .count();
    if sampled_f32_count < 2 {
        return None;
    }
    Some(
        "metal golden derives a dependent texture lookup through AIR fast_ceil/fast_floor from \
         synthetic finite f32 sampled texture data; small Metal/Vulkan sampling differences can \
         cross lookup-cell boundaries, so this seed is not a portable validation oracle; rebank or \
         drop Metal row"
            .into(),
    )
}

fn incompatible_dependent_sampled_half_lookup_golden(ll: &str, metal: &MetalRow) -> Option<String> {
    if !ll_has_fast_no_nans_float_semantics(ll)
        || metal.plan.output.kind != "render_target"
        || !ll.contains("@air.sample_texture")
    {
        return None;
    }
    let textures = infer_textures(ll);
    let sampled_half_count = textures
        .iter()
        .filter(|texture| {
            texture.role == "Sampled"
                && texture.format == "Rgba16Float"
                && texture.seed_mode == SEED_MODE_FINITE_FLOAT16
        })
        .count();
    if sampled_half_count == 0 {
        return None;
    }
    let has_half_lookup_quantizer =
        sampled_half_count >= 2 && ll.contains("@air.fast_floor.") && ll.contains(".v4f16");
    let has_sampled_half_to_3d_lookup = ll.contains("@air.sample_texture_3d.v4f32")
        && ll.contains("@air.convert.f.v3f32.f.v3f16")
        && textures.iter().any(|texture| {
            texture.role == "Sampled"
                && texture.format == "Rgba32Float"
                && texture.seed_mode == SEED_MODE_FINITE_FLOAT32
        });
    if !(has_half_lookup_quantizer || has_sampled_half_to_3d_lookup) {
        return None;
    }
    Some(
        "metal golden derives dependent texture lookup coordinates from synthetic finite f16 sampled \
         texture data under AIR fast/no-nans math; small Metal/Vulkan sampling or half-rounding \
         differences can cross lookup/branch boundaries, so this seed is not a portable validation \
         oracle; rebank or drop Metal row"
            .into(),
    )
}

fn incompatible_sampled_half_fast_sqrt_render_target_golden(
    ll: &str,
    metal: &MetalRow,
) -> Option<String> {
    if !ll_has_fast_no_nans_float_semantics(ll)
        || metal.plan.output.kind != "render_target"
        || !ll.contains("@air.sample_texture_2d.v4f16")
        || !ll.contains("@air.fast_sqrt.")
    {
        return None;
    }
    let format = current_output_format_for_plan(ll, &metal.plan.output)
        .and_then(|format| parse_format(format).ok())
        .or_else(|| parse_format(&metal.plan.output.format).ok())?;
    if !matches!(format, DataFormat::Rgba16Float | DataFormat::Rgba32Float) {
        return None;
    }
    let has_sampled_half = infer_textures(ll).into_iter().any(|texture| {
        texture.role == "Sampled"
            && texture.format == "Rgba16Float"
            && texture.seed_mode == SEED_MODE_FINITE_FLOAT16
    });
    if !has_sampled_half {
        return None;
    }
    Some(
        "metal golden samples synthetic finite f16 texture data through AIR fast_sqrt-derived half \
         render-target math; Metal/Vulkan texture sampling and approximate sqrt/half-rounding are \
         not a portable validation oracle for this synthetic seed; rebank or drop Metal row"
            .into(),
    )
}

fn incompatible_sampled_half_exact_control_flow_golden(
    ll: &str,
    metal: &MetalRow,
) -> Option<String> {
    if !ll_has_fast_no_nans_float_semantics(ll)
        || metal.plan.output.kind != "render_target"
        || !ll.contains("@air.sample_texture_2d.v4f16")
        || !ll.contains("br i1")
        || !(ll.contains("fcmp fast oeq half") || ll.contains("fcmp fast une half"))
    {
        return None;
    }
    let has_sampled_half = infer_textures(ll).into_iter().any(|texture| {
        texture.role == "Sampled"
            && texture.format == "Rgba16Float"
            && texture.seed_mode == SEED_MODE_FINITE_FLOAT16
    });
    if !has_sampled_half {
        return None;
    }
    Some(
        "metal golden branches on exact predicates over synthetic finite f16 sampled texture data \
         under AIR fast/no-nans math; small Metal/Vulkan sampling or half-rounding differences can \
         choose different control flow, so this seed is not a portable validation oracle; rebank or \
         drop Metal row"
            .into(),
    )
}

fn incompatible_sampled_half_cube_fast_math_golden(ll: &str, metal: &MetalRow) -> Option<String> {
    if !ll_has_fast_no_nans_float_semantics(ll)
        || metal.plan.output.kind != "render_target"
        || !ll.contains("@air.sample_texture_cube.v4f16")
        || !ll.contains("@air.fast_rsqrt.")
    {
        return None;
    }
    let has_sampled_half_cube = infer_textures(ll).into_iter().any(|texture| {
        texture.role == "Sampled"
            && texture.format == "Rgba16Float"
            && texture.seed_mode == SEED_MODE_FINITE_FLOAT16
            && texture.d >= 6
    });
    if !has_sampled_half_cube {
        return None;
    }
    Some(
        "metal golden samples synthetic finite f16 cube texture data through AIR fast_rsqrt-derived \
         coordinates before half render-target output; Metal/Vulkan cube filtering and approximate \
         math are not a portable validation oracle for this synthetic seed; rebank or drop Metal row"
            .into(),
    )
}

fn incompatible_sampled_half_buffer_fast_math_golden(ll: &str, metal: &MetalRow) -> Option<String> {
    if !ll_has_fast_no_nans_float_semantics(ll)
        || metal.plan.output.kind != "buffer"
        || !ll.contains("@air.sample_texture")
        || !ll.contains("@air.read_texture")
        || !ll.contains("@air.fast_pow.")
        || !ll.contains("@air.fast_rsqrt.")
    {
        return None;
    }
    let format = current_output_format_for_plan(ll, &metal.plan.output)
        .and_then(|format| parse_format(format).ok())
        .or_else(|| parse_format(&metal.plan.output.format).ok())?;
    if !matches!(
        format,
        DataFormat::R16Float | DataFormat::Rg16Float | DataFormat::Rgba16Float
    ) {
        return None;
    }
    let textures = infer_textures(ll);
    let has_sampled_f32 = textures.iter().any(|texture| {
        texture.role == "Sampled"
            && texture.format == "Rgba32Float"
            && texture.seed_mode == SEED_MODE_FINITE_FLOAT32
    });
    let has_sampled_half = textures.iter().any(|texture| {
        texture.role == "Sampled"
            && texture.format == "Rgba16Float"
            && texture.seed_mode == SEED_MODE_FINITE_FLOAT16
    });
    let has_read_half = textures.iter().any(|texture| {
        texture.role == "StorageRead"
            && texture.format == "Rgba16Float"
            && texture.seed_mode == SEED_MODE_FINITE_FLOAT16
    });
    if !(has_sampled_f32 && has_sampled_half && has_read_half) {
        return None;
    }
    Some(
        "metal golden samples synthetic finite f32/half texture data and half texture reads through \
         AIR fast_pow/fast_rsqrt before half buffer output; Metal/Vulkan linear sampling and \
         approximate math are not a portable validation oracle for this synthetic seed; rebank or \
         drop Metal row"
            .into(),
    )
}

fn incompatible_float_texture_seed_golden(ll: &str, metal: &MetalRow) -> Option<String> {
    let current_textures = infer_textures(ll);
    for current in current_textures {
        let Some(elem_size) = finite_seed_elem_size(&current.seed_mode) else {
            continue;
        };
        let Some(banked) = metal
            .plan
            .textures
            .iter()
            .find(|t| t.index == current.index)
        else {
            continue;
        };
        if banked.seed_mode != SEED_MODE_DETERMINISTIC {
            continue;
        }
        let Ok(format) = parse_format(&banked.format) else {
            continue;
        };
        let input = TextureInput {
            index: banked.index,
            format,
            extent: Extent3d::new(banked.w.max(1), banked.h.max(1), banked.d.max(1)),
            role: TextureRole::Sampled,
            seed: Seed::Deterministic {
                tag: banked.seed_tag,
            },
        };
        let raw = seeded_texture_bytes(&input);
        let finite = seeded_texture_bytes(&TextureInput {
            seed: Seed::DeterministicFinite {
                tag: banked.seed_tag,
            },
            ..input
        });
        if raw != finite {
            return Some(format!(
                "metal golden uses deterministic {} texture {} now sanitized for AIR fast/no-nans math; rebank Metal row",
                if elem_size == 2 { "f16" } else { "f32" },
                current.index
            ));
        }
    }
    None
}

fn incompatible_float_render_target_seed_golden(ll: &str, metal: &MetalRow) -> Option<String> {
    if metal.seed_profile == SEED_PROFILE || metal.plan.output.kind != "render_target" {
        return None;
    }
    let format = current_output_format_for_plan(ll, &metal.plan.output)
        .and_then(|format| parse_format(format).ok())
        .or_else(|| parse_format(&metal.plan.output.format).ok())?;
    let elem_size = float_lane_size_for_compare(format)?;
    let extent = Extent3d::new(
        metal
            .plan
            .output
            .w
            .unwrap_or(DEFAULT_TEXTURE_EXTENT.width)
            .max(1),
        metal
            .plan
            .output
            .h
            .unwrap_or(DEFAULT_TEXTURE_EXTENT.height)
            .max(1),
        metal
            .plan
            .output
            .d
            .unwrap_or(DEFAULT_TEXTURE_EXTENT.depth)
            .max(1),
    );
    let raw = seeded_texture_bytes(&TextureInput {
        index: metal.plan.output.index,
        format,
        extent,
        role: TextureRole::ColorTarget,
        seed: Seed::Deterministic {
            tag: RENDER_TARGET_SEED_TAG,
        },
    });
    let finite = seeded_render_target_bytes(format, extent);
    if raw == finite {
        return None;
    }
    let has_color_input = metal2vulkan::meta::parse_air_fragment_meta(ll).is_some_and(|meta| {
        meta.roles
            .iter()
            .any(|(_, role)| matches!(role, metal2vulkan::meta::FragRole::ColorInput(_)))
    });
    let stale_output_seed_lanes = metal
        .output_b64
        .as_deref()
        .and_then(|b64| decode_output_b64(b64).ok())
        .map(|golden| stale_render_target_seed_lane_matches(&golden, &raw, &finite, elem_size))
        .unwrap_or(0);
    if !has_color_input && stale_output_seed_lanes < 4 {
        return None;
    }
    let mode = if stale_output_seed_lanes >= 4 {
        "render target seed bytes preserved in golden output"
    } else {
        "portable AIR/Vulkan float image reads"
    };
    Some(format!(
        "metal golden uses deterministic {} render target seed now sanitized for {mode}; rebank Metal row",
        if elem_size == 2 { "f16" } else { "f32" },
    ))
}

fn stale_render_target_seed_lane_matches(
    golden: &[u8],
    raw: &[u8],
    finite: &[u8],
    elem_size: usize,
) -> usize {
    golden
        .chunks_exact(elem_size)
        .zip(raw.chunks_exact(elem_size))
        .zip(finite.chunks_exact(elem_size))
        .filter(|((golden, raw), finite)| raw != finite && golden == raw)
        .count()
}

fn ll_has_fast_no_nans_float_semantics(ll: &str) -> bool {
    ll.contains("\"no-nans-fp-math\"=\"true\"")
        || ll.contains("!\"air.compile.fast_math_enable\"")
        || ll.lines().any(|line| {
            line.contains(" fast ")
                && (line.contains(" fadd ")
                    || line.contains(" fmul ")
                    || line.contains(" fsub ")
                    || line.contains(" fdiv "))
        })
}

fn finite_seed_elem_size(seed_mode: &str) -> Option<usize> {
    match seed_mode {
        SEED_MODE_FINITE_FLOAT16 => Some(2),
        SEED_MODE_FINITE_BFLOAT16 => Some(2),
        SEED_MODE_FINITE_FLOAT32 => Some(4),
        _ => None,
    }
}

fn finite_seed_label(seed_mode: &str) -> &'static str {
    match seed_mode {
        SEED_MODE_FINITE_FLOAT16 => "f16",
        SEED_MODE_FINITE_BFLOAT16 => "bf16",
        SEED_MODE_FINITE_FLOAT32 => "f32",
        _ => "float",
    }
}

fn candidate_spv_for_metal_function_constants(
    spv: Vec<u8>,
    metal: &MetalRow,
) -> Result<Vec<u8>, String> {
    match metal.fc_specialization.as_deref() {
        Some(FC_SPECIALIZATION_ZERO) => {
            return metal2vulkan::specialize_function_constants_zero(&spv)
                .map_err(|e| format!("function-constant zero specialize: {e}"));
        }
        Some(FC_SPECIALIZATION_VALUES) => {}
        _ => return Ok(spv),
    }
    let values = metal
        .fc_values
        .as_deref()
        .ok_or_else(|| "function-constant value mode missing fc_values".to_string())?;
    let values: Vec<_> = values.iter().map(|v| (v.index, v.value)).collect();
    metal2vulkan::specialize_function_constants(&spv, &values)
        .map_err(|e| format!("function-constant specialize: {e}"))
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
/// Prefers full `output_b64` (enables float tolerance and observed margins). Falls back to
/// `output_sha256` equality when older metal rows lack payloads.
pub fn compare_candidate_to_metal(
    candidate: &[u8],
    metal: &MetalRow,
    out_hash: &str,
    golden_hash: &str,
    format: DataFormat,
    ll: Option<&str>,
) -> (String, Option<ObservedMargins>, Option<ToleranceSpecJson>) {
    if let Some(b64) = metal.output_b64.as_deref() {
        match decode_output_b64(b64) {
            Ok(golden) => {
                if let Some(result) =
                    compare_finite_struct_float_raw_bytes(candidate, &golden, metal, format, ll)
                {
                    if metal.compare == "none" && result.status != "ok" {
                        return ("smoke".to_string(), result.observed, result.tolerance);
                    }
                    return (result.status, result.observed, result.tolerance);
                }
                let policy = tolerance_for_metal_context(format, ll, metal);
                let mut result = compare_to_golden(candidate, &golden, format, policy.as_ref());
                if fast_math_nonfinite_domain_mismatch(format, ll, &result) {
                    result.status = "missing".into();
                    result.tolerance = Some(ToleranceSpecJson {
                        kind: "FastMathNonFiniteDomain".into(),
                        max_abs: None,
                        max_ulp: None,
                    });
                }
                if metal.compare == "none" && result.status != "ok" {
                    return ("smoke".to_string(), result.observed, result.tolerance);
                }
                return (result.status, result.observed, result.tolerance);
            }
            Err(_) => {
                // Corrupt payload: fall through to hash compare.
            }
        }
    }
    if !golden_hash.is_empty() && out_hash == golden_hash {
        ("ok".to_string(), None, None)
    } else if metal.compare == "none" {
        ("smoke".to_string(), None, None)
    } else if format.is_float_like() {
        // Hash-only float goldens cannot distinguish real numeric drift from benign NaN payload or
        // signed-zero differences. Rebank with output_b64 before treating them as product failures.
        ("missing".to_string(), None, None)
    } else {
        // No usable golden bytes → cannot classify tolerance; hash mismatch is failure.
        ("failure".to_string(), None, None)
    }
}

fn candidate_compare_error(
    status: &str,
    metal: &MetalRow,
    tolerance: Option<&ToleranceSpecJson>,
) -> Option<String> {
    if status == "smoke" && metal.compare == "none" {
        Some("metal golden compare=none is not a full semantic golden; rebank Metal row".into())
    } else if status == "missing" && metal.output_b64.is_none() {
        Some(
            "metal golden lacks output_b64 for float/tolerance comparison; rebank Metal row".into(),
        )
    } else if status == "missing" && tolerance.is_some_and(|t| t.kind == "FastMathNonFiniteDomain")
    {
        Some(
            "metal golden compares a non-finite result from AIR fast/no-nans domain-sensitive math; \
             rebank validation inputs away from undefined fast-math domains"
                .into(),
        )
    } else {
        None
    }
}

fn fast_math_nonfinite_domain_mismatch(
    format: DataFormat,
    ll: Option<&str>,
    result: &CompareResult,
) -> bool {
    if result.status != "failure" || !format.is_float_like() {
        return false;
    }
    let Some(ll) = ll else {
        return false;
    };
    if !ll_has_fast_no_nans_float_semantics(ll) || !ll_has_domain_sensitive_float_math(ll) {
        return false;
    }
    result.observed.as_ref().is_some_and(|observed| {
        observed.max_abs.is_some_and(f32::is_infinite) || observed.max_ulp == Some(u32::MAX)
    })
}

fn ll_has_domain_sensitive_float_math(ll: &str) -> bool {
    ll.contains("@air.fast_rsqrt.")
        || ll.contains("@air.rsqrt.")
        || ll.contains("@air.fast_sqrt.")
        || ll.contains("@air.sqrt.")
        || ll.contains("@air.fast_pow.")
        || ll.contains("@air.pow.")
        || ll.contains("@air.sincos.")
        || ll.contains("@air.sin.")
        || ll.contains("@air.cos.")
        || ll.contains("@air.tan.")
        || ll
            .lines()
            .any(|line| line.trim_start().starts_with("%") && line.contains(" fdiv fast "))
}

fn compare_finite_struct_float_raw_bytes(
    candidate: &[u8],
    golden: &[u8],
    metal: &MetalRow,
    format: DataFormat,
    ll: Option<&str>,
) -> Option<CompareResult> {
    if format != DataFormat::RawBytes || !ll.is_some_and(ll_has_fast_no_nans_float_semantics) {
        return None;
    }
    let output = metal.plan.buffers.iter().find(|buffer| {
        buffer.index == metal.plan.output.index
            && buffer.seed_mode == SEED_MODE_FINITE_STRUCT_FLOAT
            && buffer.seed_stride.is_some()
            && !buffer.seed_layout.is_empty()
    })?;
    Some(compare_masked_struct_float_raw_bytes(
        candidate,
        golden,
        output.seed_stride.unwrap_or(0),
        &output.seed_layout,
    ))
}

fn compare_masked_struct_float_raw_bytes(
    candidate: &[u8],
    golden: &[u8],
    stride: usize,
    layout: &[ControlSeedField],
) -> CompareResult {
    let tolerance = ToleranceSpecJson {
        kind: "MaskedStructFloatAbsOrUlp".into(),
        max_abs: Some(0.003_906_25),
        max_ulp: Some(32),
    };
    if candidate == golden {
        return CompareResult {
            status: "ok".into(),
            observed: None,
            tolerance: None,
        };
    }
    if candidate.len() != golden.len() || stride == 0 {
        return CompareResult {
            status: "failure".into(),
            observed: None,
            tolerance: Some(tolerance),
        };
    }

    let mut float_mask = vec![false; candidate.len()];
    let mut max_abs = 0.0f32;
    let mut max_ulp = 0u32;
    let mut within = true;
    let mut base = 0usize;
    while base < candidate.len() {
        for field in layout.iter().filter(|field| matches!(field.size, 2 | 4)) {
            let start = base.saturating_add(field.offset);
            let end = start.saturating_add(field.size);
            if end > candidate.len() {
                continue;
            }
            float_mask[start..end].fill(true);
            let policy = if field.size == 2 {
                float_tolerance_for_context(DataFormat::R16Float, Some(""))
            } else {
                default_float_tolerance()
            };
            let (abs, ulp, field_within) = simple_margins(
                &candidate[start..end],
                &golden[start..end],
                field_format(field),
                &policy,
            );
            max_abs = max_abs.max(abs);
            max_ulp = max_ulp.max(ulp);
            within &= field_within;
        }
        base = base.saturating_add(stride);
    }
    let exact_diffs = candidate
        .iter()
        .zip(golden.iter())
        .zip(float_mask.iter())
        .filter(|((a, b), is_float)| !**is_float && a != b)
        .count() as u32;
    if exact_diffs != 0 {
        within = false;
        max_abs = max_abs.max(exact_diffs as f32);
        max_ulp = max_ulp.max(exact_diffs);
    }
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
        tolerance: Some(tolerance),
    }
}

fn field_format(field: &ControlSeedField) -> DataFormat {
    match field.size {
        2 => DataFormat::R16Float,
        4 => DataFormat::F32,
        _ => DataFormat::RawBytes,
    }
}

/// Exact compare by default. Optional float tolerance is unused unless `policy` is set.
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
    let (max_abs, max_ulp, within) = simple_margins(candidate, golden, format, pol);
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

fn simple_margins(
    candidate: &[u8],
    golden: &[u8],
    format: DataFormat,
    policy: &ToleranceSpecJson,
) -> (f32, u32, bool) {
    let Some(lane_size) = float_lane_size_for_compare(format) else {
        let mut max_abs = 0.0f32;
        let diffs = candidate
            .iter()
            .zip(golden.iter())
            .filter(|(a, b)| a != b)
            .map(|(a, b)| {
                let abs = a.abs_diff(*b) as f32;
                max_abs = max_abs.max(abs);
                abs
            })
            .count();
        return (
            max_abs,
            diffs as u32,
            tolerance_policy_accepts(policy, max_abs, diffs as u32),
        );
    };
    if candidate.len() < lane_size {
        let diff = (candidate != golden) as u32;
        return (
            diff as f32,
            diff,
            tolerance_policy_accepts(policy, diff as f32, diff),
        );
    }
    let mut max_abs = 0.0f32;
    let mut max_ulp = 0u32;
    let mut within = true;
    let mut compared = 0usize;
    for (c, g) in candidate
        .chunks_exact(lane_size)
        .zip(golden.chunks_exact(lane_size))
    {
        let (cf, gf, ulp) = match lane_size {
            2 => {
                let cu = u16::from_le_bytes([c[0], c[1]]);
                let gu = u16::from_le_bytes([g[0], g[1]]);
                (
                    half_bits_to_f32(cu),
                    half_bits_to_f32(gu),
                    ordered_half_ulp_key(cu).abs_diff(ordered_half_ulp_key(gu)),
                )
            }
            4 => {
                let cu = u32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                let gu = u32::from_le_bytes([g[0], g[1], g[2], g[3]]);
                (
                    f32::from_bits(cu),
                    f32::from_bits(gu),
                    ordered_float_ulp_key(cu).abs_diff(ordered_float_ulp_key(gu)),
                )
            }
            _ => unreachable!("unsupported float lane size"),
        };
        compared += lane_size;
        if cf.is_nan() && gf.is_nan() {
            continue;
        }
        if cf.is_nan() || gf.is_nan() {
            max_abs = f32::INFINITY;
            max_ulp = u32::MAX;
            within = false;
            continue;
        }
        let abs = (cf - gf).abs();
        max_abs = max_abs.max(abs);
        max_ulp = max_ulp.max(ulp);
        within &= tolerance_policy_accepts(policy, abs, ulp);
    }
    let trailing_diffs = candidate[compared..]
        .iter()
        .zip(golden[compared..].iter())
        .filter(|(a, b)| a != b)
        .count() as u32;
    max_abs = max_abs.max(trailing_diffs as f32);
    max_ulp = max_ulp.max(trailing_diffs);
    if trailing_diffs != 0 {
        within &= tolerance_policy_accepts(policy, trailing_diffs as f32, trailing_diffs);
    }
    (max_abs, max_ulp, within)
}

fn tolerance_policy_accepts(policy: &ToleranceSpecJson, abs: f32, ulp: u32) -> bool {
    match policy.kind.as_str() {
        "Abs" => abs <= policy.max_abs.unwrap_or(0.0),
        "Ulp" => ulp <= policy.max_ulp.unwrap_or(0),
        "AbsAndUlp" => abs <= policy.max_abs.unwrap_or(0.0) && ulp <= policy.max_ulp.unwrap_or(0),
        "AbsOrUlp" => abs <= policy.max_abs.unwrap_or(0.0) || ulp <= policy.max_ulp.unwrap_or(0),
        _ => false,
    }
}

fn ordered_half_ulp_key(bits: u16) -> u32 {
    ordered_float_bits(bits as u32, 0x8000)
}

fn ordered_float_ulp_key(bits: u32) -> u32 {
    ordered_float_bits(bits, 0x8000_0000)
}

fn ordered_float_bits(bits: u32, sign_bit: u32) -> u32 {
    if bits & sign_bit == 0 {
        bits | sign_bit
    } else {
        (!bits).wrapping_add(1) & ((sign_bit << 1).wrapping_sub(1))
    }
}

fn float_lane_size_for_compare(format: DataFormat) -> Option<usize> {
    match format {
        DataFormat::R16Float | DataFormat::Rg16Float | DataFormat::Rgba16Float => Some(2),
        DataFormat::F32
        | DataFormat::R32Float
        | DataFormat::Rg32Float
        | DataFormat::Rgba32Float
        | DataFormat::Depth32Float => Some(4),
        _ => None,
    }
}

fn half_bits_to_f32(bits: u16) -> f32 {
    let sign = if bits & 0x8000 == 0 { 1.0 } else { -1.0 };
    let exp = (bits >> 10) & 0x1f;
    let frac = bits & 0x03ff;
    match (exp, frac) {
        (0, 0) => sign * 0.0,
        (0, _) => sign * (frac as f32 / 1024.0) * 2.0f32.powi(-14),
        (0x1f, 0) => sign * f32::INFINITY,
        (0x1f, _) => f32::NAN,
        _ => sign * (1.0 + frac as f32 / 1024.0) * 2.0f32.powi(exp as i32 - 15),
    }
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
    let only_air_set = cfg
        .only_air_list
        .as_deref()
        .map(|path| load_hash_list(program, path));
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
            } else if let Some(ref set) = only_air_set {
                set.contains(&r.air_sha256)
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
    let unbounded_todo_len = todo.len();
    if cfg.skip > 0 {
        if cfg.skip >= todo.len() {
            todo.clear();
        } else {
            todo.drain(..cfg.skip);
        }
    }
    if let Some(limit) = cfg.limit {
        todo.truncate(limit);
    }

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
    if let Some(path) = cfg.only_air_list.as_deref() {
        eprintln!("# air-list: {}", path.display());
    }
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
    if cfg.skip > 0 {
        eprintln!("# skip: {}", cfg.skip);
    }
    if let Some(limit) = cfg.limit {
        eprintln!(
            "# limit: {limit} (selected {}/{unbounded_todo_len})",
            todo.len()
        );
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
                    if let Some(error) = worker_exit_error(status) {
                        write_timeout_or_error_row(cfg, row, &error, "fallback");
                    }
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

fn worker_exit_error(status: std::process::ExitStatus) -> Option<String> {
    if status.code() == Some(1) {
        return None;
    }
    Some(describe_worker_exit(status))
}

fn describe_worker_exit(status: std::process::ExitStatus) -> String {
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;
        if let Some(signal) = status.signal() {
            return format!("worker terminated by signal {signal}");
        }
    }
    match status.code() {
        Some(code) => format!("worker exited with status {code}"),
        None => "worker exited without status".into(),
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
    if let Some(reason) = incompatible_function_constant_golden(&ll, metal) {
        let row = candidate_status_row(
            cfg,
            tr,
            &src,
            "missing",
            metal.output_sha256.clone(),
            Some(reason),
        );
        let _ = append_result_row(cfg, &row);
        return ProcessOutcome::Fail;
    }
    if let Some(reason) = incompatible_function_constant_definedness_golden(&ll, metal) {
        let row = candidate_status_row(
            cfg,
            tr,
            &src,
            "missing",
            metal.output_sha256.clone(),
            Some(reason),
        );
        let _ = append_result_row(cfg, &row);
        return ProcessOutcome::Fail;
    }
    if let Some(reason) = incompatible_zero_function_constant_divisor_golden(&ll, metal) {
        let row = candidate_status_row(
            cfg,
            tr,
            &src,
            "missing",
            metal.output_sha256.clone(),
            Some(reason),
        );
        let _ = append_result_row(cfg, &row);
        return ProcessOutcome::Fail;
    }
    if let Some(reason) = incompatible_output_plan_golden(&ll, metal) {
        let row = candidate_status_row(
            cfg,
            tr,
            &src,
            "missing",
            metal.output_sha256.clone(),
            Some(reason),
        );
        let _ = append_result_row(cfg, &row);
        return ProcessOutcome::Fail;
    }
    if let Some(reason) = incompatible_texture_array_plan_golden(&ll, metal) {
        let row = candidate_status_row(
            cfg,
            tr,
            &src,
            "missing",
            metal.output_sha256.clone(),
            Some(reason),
        );
        let _ = append_result_row(cfg, &row);
        return ProcessOutcome::Fail;
    }
    if let Some(reason) = incompatible_static_resource_plan_golden(&ll, metal) {
        let row = candidate_status_row(
            cfg,
            tr,
            &src,
            "missing",
            metal.output_sha256.clone(),
            Some(reason),
        );
        let _ = append_result_row(cfg, &row);
        return ProcessOutcome::Fail;
    }
    if let Some(reason) = incompatible_point_coord_golden(&ll, metal) {
        let row = candidate_status_row(
            cfg,
            tr,
            &src,
            "missing",
            metal.output_sha256.clone(),
            Some(reason),
        );
        let _ = append_result_row(cfg, &row);
        return ProcessOutcome::Fail;
    }
    if let Some(reason) = incompatible_undefined_texture_write_lanes_golden(&ll, metal) {
        let row = candidate_status_row(
            cfg,
            tr,
            &src,
            "missing",
            metal.output_sha256.clone(),
            Some(reason),
        );
        let _ = append_result_row(cfg, &row);
        return ProcessOutcome::Fail;
    }
    if let Some(reason) = incompatible_bounded_control_seed_golden(&ll, metal) {
        let row = candidate_status_row(
            cfg,
            tr,
            &src,
            "missing",
            metal.output_sha256.clone(),
            Some(reason),
        );
        let _ = append_result_row(cfg, &row);
        return ProcessOutcome::Fail;
    }
    if let Some(reason) = incompatible_oob_vector_input_golden(&ll, metal) {
        let row = candidate_status_row(
            cfg,
            tr,
            &src,
            "missing",
            metal.output_sha256.clone(),
            Some(reason),
        );
        let _ = append_result_row(cfg, &row);
        return ProcessOutcome::Fail;
    }
    if let Some(reason) = incompatible_float_seed_golden(&ll, metal) {
        let row = candidate_status_row(
            cfg,
            tr,
            &src,
            "missing",
            metal.output_sha256.clone(),
            Some(reason),
        );
        let _ = append_result_row(cfg, &row);
        return ProcessOutcome::Fail;
    }
    if let Some(reason) = incompatible_float_output_golden(&ll, metal) {
        let row = candidate_status_row(
            cfg,
            tr,
            &src,
            "missing",
            metal.output_sha256.clone(),
            Some(reason),
        );
        let _ = append_result_row(cfg, &row);
        return ProcessOutcome::Fail;
    }
    if let Some(reason) = incompatible_sampled_fast_pow_texture_golden(&ll, metal) {
        let row = candidate_status_row(
            cfg,
            tr,
            &src,
            "missing",
            metal.output_sha256.clone(),
            Some(reason),
        );
        let _ = append_result_row(cfg, &row);
        return ProcessOutcome::Fail;
    }
    if let Some(reason) = incompatible_dependent_sampled_lookup_golden(&ll, metal) {
        let row = candidate_status_row(
            cfg,
            tr,
            &src,
            "missing",
            metal.output_sha256.clone(),
            Some(reason),
        );
        let _ = append_result_row(cfg, &row);
        return ProcessOutcome::Fail;
    }
    if let Some(reason) = incompatible_dependent_sampled_half_lookup_golden(&ll, metal) {
        let row = candidate_status_row(
            cfg,
            tr,
            &src,
            "missing",
            metal.output_sha256.clone(),
            Some(reason),
        );
        let _ = append_result_row(cfg, &row);
        return ProcessOutcome::Fail;
    }
    if let Some(reason) = incompatible_sampled_half_fast_sqrt_render_target_golden(&ll, metal) {
        let row = candidate_status_row(
            cfg,
            tr,
            &src,
            "missing",
            metal.output_sha256.clone(),
            Some(reason),
        );
        let _ = append_result_row(cfg, &row);
        return ProcessOutcome::Fail;
    }
    if let Some(reason) = incompatible_sampled_half_exact_control_flow_golden(&ll, metal) {
        let row = candidate_status_row(
            cfg,
            tr,
            &src,
            "missing",
            metal.output_sha256.clone(),
            Some(reason),
        );
        let _ = append_result_row(cfg, &row);
        return ProcessOutcome::Fail;
    }
    if let Some(reason) = incompatible_sampled_half_cube_fast_math_golden(&ll, metal) {
        let row = candidate_status_row(
            cfg,
            tr,
            &src,
            "missing",
            metal.output_sha256.clone(),
            Some(reason),
        );
        let _ = append_result_row(cfg, &row);
        return ProcessOutcome::Fail;
    }
    if let Some(reason) = incompatible_sampled_half_buffer_fast_math_golden(&ll, metal) {
        let row = candidate_status_row(
            cfg,
            tr,
            &src,
            "missing",
            metal.output_sha256.clone(),
            Some(reason),
        );
        let _ = append_result_row(cfg, &row);
        return ProcessOutcome::Fail;
    }
    if let Some(reason) = incompatible_compare_none_loop_guard_golden(&ll, &entry, metal) {
        let row = candidate_status_row(
            cfg,
            tr,
            &src,
            "missing",
            metal.output_sha256.clone(),
            Some(reason),
        );
        let _ = append_result_row(cfg, &row);
        return ProcessOutcome::Fail;
    }
    if let Some(reason) = incompatible_nonportable_ptrtoint_golden(&ll, metal) {
        let row = candidate_status_row(
            cfg,
            tr,
            &src,
            "missing",
            metal.output_sha256.clone(),
            Some(reason),
        );
        let _ = append_result_row(cfg, &row);
        return ProcessOutcome::Fail;
    }
    if let Some(reason) = incompatible_parallel_dynamic_buffer_scatter_golden(&ll, metal) {
        let row = candidate_status_row(
            cfg,
            tr,
            &src,
            "missing",
            metal.output_sha256.clone(),
            Some(reason),
        );
        let _ = append_result_row(cfg, &row);
        return ProcessOutcome::Fail;
    }
    if let Some(reason) = incompatible_undefined_threadgroup_memory_golden(&ll, metal) {
        let row = candidate_status_row(
            cfg,
            tr,
            &src,
            "missing",
            metal.output_sha256.clone(),
            Some(reason),
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
#[cfg(target_os = "macos")]
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
    if stage == Stage::Fragment {
        if let Some(reason) = unsupported_fragment_color_output_arity(ll) {
            write_failure_row(cfg, tr, src, &reason);
            return ProcessOutcome::Fail;
        }
    }

    #[cfg(not(target_os = "macos"))]
    {
        let _ = (cfg, tr, src, ll, stage, entry, plan);
        eprintln!("    fallback: corpus-run-metal requires macOS");
        ProcessOutcome::Fail
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
        let (fc_specialization, fc_values) = oracle_function_constant_row_fields();
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
            fc_specialization,
            fc_values,
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
fn oracle_function_constant_row_fields() -> (Option<String>, Option<Vec<FunctionConstantValueJson>>)
{
    match crate::oracle_macos::last_oracle_function_constants() {
        crate::oracle_macos::OracleFunctionConstants::None => (None, None),
        crate::oracle_macos::OracleFunctionConstants::Zero => {
            (Some(FC_SPECIALIZATION_ZERO.into()), None)
        }
        crate::oracle_macos::OracleFunctionConstants::Values(values) => (
            Some(FC_SPECIALIZATION_VALUES.into()),
            Some(
                values
                    .into_iter()
                    .map(|(index, value)| FunctionConstantValueJson {
                        index: index as u32,
                        value,
                    })
                    .collect(),
            ),
        ),
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

pub(crate) fn panic_payload_message(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else if let Some(message) = payload.downcast_ref::<&'static str>() {
        (*message).to_string()
    } else {
        "non-string panic payload".to_string()
    }
}

#[cfg(any(target_os = "macos", test))]
fn classify_metal_oracle_panic(message: &str) -> String {
    let refs = unresolved_visible_refs_from_error(message);
    if !refs.is_empty() {
        return format!(
            "unsupported Metal visible function reference(s): {}",
            refs.join(", ")
        );
    }
    if message.contains("fragment shader color output does not have enough components") {
        return format!("unsupported Metal fragment color output attachment arity: {message}");
    }
    format!("metal oracle panicked: {message}")
}

#[cfg(any(target_os = "macos", test))]
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

fn candidate_execution_error_status(error: &str) -> &'static str {
    if error.contains("Vulkan validation runner skipped NVIDIA")
        || error.contains("pipeline probe timed out after")
        || error.contains("the logical or physical device has been lost")
    {
        "quarantine"
    } else {
        "fallback"
    }
}

fn write_candidate_execution_error_row(
    cfg: &RunConfig,
    tr: &TranslateRow,
    src: &SourceFile,
    metal: &MetalRow,
    error: &str,
) {
    let status = candidate_execution_error_status(error);
    let golden_output_sha256 = (status == "quarantine")
        .then(|| metal.output_sha256.clone())
        .flatten();
    let row = candidate_status_row(
        cfg,
        tr,
        src,
        status,
        golden_output_sha256,
        Some(format!("vulkan execute: {error}")),
    );
    let _ = append_result_row(cfg, &row);
}

fn translate_candidate_spv_for_plan(
    candidate_ll: &str,
    stage: Stage,
    plan: &HarnessPlan,
    tmp: &Path,
) -> Result<Vec<u8>, String> {
    let m2v_stage: metal2vulkan::passes::Stage = stage.into();
    let options = if stage == Stage::Kernel {
        metal2vulkan::passes::TransformOptions {
            kernel_local_size: plan.dispatch_tg,
            kernel_threads_per_grid: Some(plan.dispatch_grid),
            ..Default::default()
        }
    } else {
        Default::default()
    };
    metal2vulkan::translate_sanitized_native_with_options(candidate_ll, m2v_stage, tmp, options)
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
        let spv = match translate_candidate_spv_for_plan(candidate_ll, stage, plan, &tmp) {
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
        let spv = match candidate_spv_for_metal_function_constants(spv, metal) {
            Ok(spv) => spv,
            Err(e) => {
                let _ = fs::remove_dir_all(&tmp);
                write_failure_row(cfg, tr, src, &e);
                return ProcessOutcome::Fail;
            }
        };
        if let Err(e) = metal2vulkan::tools::spirv_val_bytes(&spv, &tmp) {
            let _ = fs::remove_dir_all(&tmp);
            write_failure_row(
                cfg,
                tr,
                src,
                &format!("translate produced invalid SPIR-V: {e}"),
            );
            return ProcessOutcome::Fail;
        }
        if let Some(reason) =
            incompatible_undefined_fragment_color_output_golden(candidate_ll, plan, &spv)
        {
            let _ = fs::remove_dir_all(&tmp);
            let row = candidate_status_row(
                cfg,
                tr,
                src,
                "missing",
                metal.output_sha256.clone(),
                Some(reason),
            );
            let _ = append_result_row(cfg, &row);
            return ProcessOutcome::Fail;
        }
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            crate::runner_linux::execute_result(stage, candidate_ll, &spv, &owned.inputs, &tmp)
        }));
        let _ = fs::remove_dir_all(&tmp);
        let candidate = match result {
            Ok(Ok(b)) => b,
            Ok(Err(error)) => {
                write_candidate_execution_error_row(cfg, tr, src, metal, &error);
                return ProcessOutcome::Fail;
            }
            Err(payload) => {
                let detail = panic_payload_message(payload);
                write_failure_row(cfg, tr, src, &format!("vulkan execute panicked: {detail}"));
                return ProcessOutcome::Fail;
            }
        };
        let golden_hash = metal.output_sha256.clone().unwrap_or_default();
        let out_hash = sha256_hex(&candidate);
        let format = candidate_compare_format(candidate_ll, plan, metal);
        let (status, observed, tolerance) = compare_candidate_to_metal(
            &candidate,
            metal,
            &out_hash,
            &golden_hash,
            format,
            Some(candidate_ll),
        );
        let error = candidate_compare_error(&status, metal, tolerance.as_ref());
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
            error,
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

    fn metal_row_for_compare(golden: &[u8], plan: HarnessPlan, stage: Option<&str>) -> MetalRow {
        MetalRow {
            air_sha256: "x".into(),
            shard: None,
            label: String::new(),
            status: "ok".into(),
            backend: "metal".into(),
            seed_profile: SEED_PROFILE.into(),
            plan_version: PLAN_VERSION,
            plan,
            input_sha256: None,
            output_sha256: Some(sha256_hex(golden)),
            output_b64: Some(encode_output_b64(golden)),
            spv_sha256: None,
            compare: "full".into(),
            fc_specialization: None,
            fc_values: None,
            stage: stage.map(str::to_string),
            entry: None,
            error: None,
        }
    }

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

    /// Raw device payload whose i32 counter is read through an AIR atomic load. Even though the
    /// root buffer is metadata-readonly `uchar*`, the atomic load makes it a control/counter source
    /// for validation seeding.
    const ATOMIC_COUNTER_SOURCE_LL: &str = r#"
define void @atomic_counter_source(ptr addrspace(2) %header, ptr addrspace(1) %data, ptr addrspace(1) %out) {
  %offset_ptr = getelementptr inbounds i8, ptr addrspace(2) %header, i64 0
  %offset = load i32, ptr addrspace(2) %offset_ptr, align 4
  %offset64 = zext i32 %offset to i64
  %base = getelementptr inbounds i8, ptr addrspace(1) %data, i64 %offset64
  %slot = getelementptr inbounds i8, ptr addrspace(1) %base, i64 4
  %value = tail call i32 @air.atomic.global.load.i32(ptr addrspace(1) %slot, i32 0, i32 2, i1 true)
  store i32 %value, ptr addrspace(1) %out, align 4
  ret void
}

declare i32 @air.atomic.global.load.i32(ptr addrspace(1), i32, i32, i1)

!air.kernel = !{!0}
!0 = !{ptr @atomic_counter_source, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.buffer", !"air.buffer_size", i32 8, !"air.location_index", i32 4, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !6, !"air.arg_type_size", i32 8, !"air.arg_type_name", !"Header", !"air.arg_name", !"header"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 5, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 1, !"air.arg_type_name", !"uchar", !"air.arg_name", !"data"}
!5 = !{i32 2, !"air.buffer", !"air.buffer_size", i32 4, !"air.location_index", i32 6, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!6 = !{i32 0, i32 4, i32 0, !"uint", !"offset"}
"#;

    const COPY_ARGS_U64_LL: &str = r#"
%struct.MTLCopyArgs = type { i64, i64, i64 }

define void @copyKernel(i32 %tid, ptr addrspace(1) writeonly %out, ptr addrspace(1) readonly %in, ptr addrspace(2) readonly %args) {
entry:
  %idx = zext i32 %tid to i64
  %len_ptr = getelementptr %struct.MTLCopyArgs, ptr addrspace(2) %args, i64 0, i32 2
  %len = load i64, ptr addrspace(2) %len_ptr
  %in_bounds = icmp ugt i64 %len, %idx
  br i1 %in_bounds, label %copy, label %exit
copy:
  %src_ptr = getelementptr %struct.MTLCopyArgs, ptr addrspace(2) %args, i64 0, i32 0
  %src = load i64, ptr addrspace(2) %src_ptr
  %src_idx = add i64 %src, %idx
  %read_ptr = getelementptr i32, ptr addrspace(1) %in, i64 %src_idx
  %value = load i32, ptr addrspace(1) %read_ptr
  %dst_ptr = getelementptr %struct.MTLCopyArgs, ptr addrspace(2) %args, i64 0, i32 1
  %dst = load i64, ptr addrspace(2) %dst_ptr
  %dst_idx = add i64 %dst, %idx
  %write_ptr = getelementptr i32, ptr addrspace(1) %out, i64 %dst_idx
  store i32 %value, ptr addrspace(1) %write_ptr
  br label %exit
exit:
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @copyKernel, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5, !6}
!3 = !{i32 0, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 28, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_name", !"uint*", !"air.arg_name", !"out"}
!5 = !{i32 2, !"air.buffer", !"air.location_index", i32 29, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_name", !"uint*", !"air.arg_name", !"in"}
!6 = !{i32 3, !"air.buffer", !"air.buffer_size", i32 24, !"air.location_index", i32 30, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !7, !"air.arg_type_size", i32 24, !"air.arg_type_name", !"MTLCopyArgs", !"air.arg_name", !"args"}
!7 = !{i32 0, i32 8, i32 0, !"ulong", !"srcOffset", i32 8, i32 8, i32 0, !"ulong", !"dstOffset", i32 16, i32 8, i32 0, !"ulong", !"length"}
"#;

    const SCALAR_ULONG_CONTROL_LL: &str = r#"
define void @scalar_ulong_control(ptr addrspace(2) %stride, ptr addrspace(1) %out, i32 %tid) {
entry:
  %value = load i64, ptr addrspace(2) %stride, align 8
  %narrow = trunc i64 %value to i32
  store i32 %narrow, ptr addrspace(1) %out, align 4
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @scalar_ulong_control, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.buffer", !"air.buffer_size", i32 8, !"air.location_index", i32 7, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"ulong", !"air.arg_name", !"stride"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 8, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!5 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid"}
"#;

    const VECTOR_CONTROL_STRUCT_LL: &str = r#"
define <4 x float> @fragment(ptr addrspace(2) %uniforms) {
entry:
  ret <4 x float> zeroinitializer
}

!air.fragment = !{!0}
!0 = !{ptr @fragment, !1, !2}
!1 = !{!3}
!2 = !{!4}
!3 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!4 = !{i32 0, !"air.buffer", !"air.buffer_size", i32 112, !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !5, !"air.arg_type_size", i32 112, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"Uniforms", !"air.arg_name", !"u"}
!5 = !{i32 0, i32 8, i32 8, !"float2", !"offset", i32 64, i32 16, i32 2, !"float4", !"weight", i32 96, i32 4, i32 0, !"float", !"divide"}
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

    const LOOP_GUARDED_BDA_AS_POINTER_LL: &str = r#"
define void @bda_loop_guard(ptr addrspace(1) %out, ptr addrspace(1) %in) {
entry:
  %p = load ptr addrspace(1), ptr addrspace(1) %in, align 8
  %d = call ptr addrspace(1) @air.get_data_pointer_instance_acceleration_structure(ptr addrspace(1) %p)
  store ptr addrspace(1) %d, ptr addrspace(1) %out, align 8
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %next, %loop ]
  %field = getelementptr inbounds i8, ptr addrspace(1) %d, i64 136
  %f = load float, ptr addrspace(1) %field, align 4
  store float %f, ptr addrspace(1) %out, align 4
  %next = add i32 %i, 1
  %done = icmp eq i32 %next, 1
  br i1 %done, label %exit, label %loop, !llvm.loop !5

exit:
  ret void
}

declare ptr addrspace(1) @air.get_data_pointer_instance_acceleration_structure(ptr addrspace(1))

!air.kernel = !{!0}
!0 = !{ptr @bda_loop_guard, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.buffer", !"air.buffer_size", i32 8, !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"ulong", !"air.arg_name", !"out"}
!4 = !{i32 1, !"air.buffer", !"air.buffer_size", i32 8, !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"ulong", !"air.arg_name", !"in"}
!5 = distinct !{!5}
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
        // Payload half buffers use finite synthetic values so backend-specific NaN handling does
        // not dominate validation.
        for idx in [0u32, 1, 2, 3] {
            let b = plan.buffers.iter().find(|b| b.index == idx).unwrap();
            assert_eq!(
                b.seed_mode, SEED_MODE_FINITE_FLOAT16,
                "buffer {idx} should use finite f16 seeds"
            );
        }
    }

    #[test]
    fn finite_float_buffer_seed_has_no_nan_or_inf() {
        let mut plan = infer_plan(
            r#"
define void @kernel(ptr addrspace(1) %in, ptr addrspace(1) %out) {
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @kernel, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 3, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"in"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 4, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 2, !"air.arg_type_name", !"half", !"air.arg_name", !"out"}
!5 = !{i32 2, !"air.buffer", !"air.location_index", i32 5, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 2, !"air.arg_type_name", !"bfloat", !"air.arg_name", !"bf"}
"#,
        );
        plan.buffers.iter_mut().for_each(|b| b.len = 256);

        let f32_buf = plan.buffers.iter().find(|b| b.index == 3).unwrap();
        assert_eq!(f32_buf.seed_mode, SEED_MODE_FINITE_FLOAT32);
        let f16_buf = plan.buffers.iter().find(|b| b.index == 4).unwrap();
        assert_eq!(f16_buf.seed_mode, SEED_MODE_FINITE_FLOAT16);
        let bf16_buf = plan.buffers.iter().find(|b| b.index == 5).unwrap();
        assert_eq!(bf16_buf.seed_mode, SEED_MODE_FINITE_BFLOAT16);

        let owned = plan_to_owned_inputs(&plan).unwrap();
        for input in owned.inputs.buffers {
            let bytes = seeded_buffer_bytes(input);
            match input.index {
                3 => assert!(
                    !contains_nonfinite_float_lane(&bytes, 4),
                    "buffer {} contains non-finite f32 lanes",
                    input.index
                ),
                4 => assert!(
                    !contains_nonfinite_float_lane(&bytes, 2),
                    "buffer {} contains non-finite f16 lanes",
                    input.index
                ),
                5 => assert!(
                    !contains_nonfinite_bfloat_lane(&bytes),
                    "buffer {} contains non-finite bf16 lanes",
                    input.index
                ),
                _ => {}
            }
        }
    }

    #[test]
    fn finite_struct_float_buffer_seed_sanitizes_only_float_fields() {
        let mut plan = infer_plan(
            r#"
%struct.Payload = type { float, i32, <2 x float> }

define void @kernel(ptr addrspace(1) %in, ptr addrspace(1) %out) {
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @kernel, !1, !2}
!1 = !{}
!2 = !{!3, !5}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 7, i32 1, !"air.read", !"air.address_space", i32 1, !"air.struct_type_info", !4, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"Payload", !"air.arg_name", !"in"}
!4 = !{i32 0, i32 4, i32 0, !"float", !"a", i32 4, i32 4, i32 0, !"int", !"b", i32 8, i32 8, i32 0, !"float2", !"c"}
!5 = !{i32 1, !"air.buffer", !"air.location_index", i32 8, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 1, !"air.arg_type_name", !"bool", !"air.arg_name", !"out"}
"#,
        );
        let buf = plan.buffers.iter_mut().find(|b| b.index == 7).unwrap();
        buf.len = 256;
        assert_eq!(buf.seed_mode, SEED_MODE_FINITE_STRUCT_FLOAT);
        assert_eq!(buf.seed_stride, Some(16));
        assert_eq!(
            buf.seed_layout,
            vec![
                ControlSeedField {
                    offset: 0,
                    size: 4,
                    value: None,
                },
                ControlSeedField {
                    offset: 8,
                    size: 4,
                    value: None,
                },
                ControlSeedField {
                    offset: 12,
                    size: 4,
                    value: None,
                },
            ]
        );

        let raw = seeded_buffer_bytes(&BufferInput {
            index: 7,
            len: 256,
            role: BufferRole::Input,
            seed: Seed::Deterministic { tag: 8 },
        });
        let owned = plan_to_owned_inputs(&plan).unwrap();
        let input = owned.inputs.buffers.iter().find(|b| b.index == 7).unwrap();
        let seeded = seeded_buffer_bytes(input);
        assert_ne!(seeded, raw);
        for base in (0..256).step_by(16) {
            for offset in [0usize, 8, 12] {
                assert_eq!(seeded[base + offset + 3] & 0x40, 0);
            }
            assert_eq!(&seeded[base + 4..base + 8], &raw[base + 4..base + 8]);
        }
    }

    #[test]
    fn finite_struct_float_input_buffer_covers_default_dispatch() {
        let ll = r#"
%struct.Payload = type { float, i32, <2 x float> }

define void @kernel(i32 %tid, ptr addrspace(1) readonly %in, ptr addrspace(1) writeonly %out) #0 {
  %idx = zext i32 %tid to i64
  %ptr = getelementptr inbounds %struct.Payload, ptr addrspace(1) %in, i64 %idx, i32 0
  %v = load float, ptr addrspace(1) %ptr, align 4
  %p = fcmp fast olt float %v, 0.000000e+00
  %byte = zext i1 %p to i8
  %outp = getelementptr inbounds i8, ptr addrspace(1) %out, i64 %idx
  store i8 %byte, ptr addrspace(1) %outp, align 1
  ret void
}

attributes #0 = { "no-nans-fp-math"="true" }

!air.kernel = !{!0}
!0 = !{ptr @kernel, !1, !2}
!1 = !{}
!2 = !{!3, !4, !6}
!3 = !{i32 0, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 7, i32 1, !"air.read", !"air.address_space", i32 1, !"air.struct_type_info", !5, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"Payload", !"air.arg_name", !"in"}
!5 = !{i32 0, i32 4, i32 0, !"float", !"a", i32 4, i32 4, i32 0, !"int", !"b", i32 8, i32 8, i32 0, !"float2", !"c"}
!6 = !{i32 2, !"air.buffer", !"air.location_index", i32 8, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_size", i32 1, !"air.arg_type_align_size", i32 1, !"air.arg_type_name", !"bool", !"air.arg_name", !"out"}
"#;
        let plan = infer_plan(ll);
        let input = plan.buffers.iter().find(|b| b.index == 7).unwrap();
        assert_eq!(input.seed_mode, SEED_MODE_FINITE_STRUCT_FLOAT);
        assert_eq!(input.seed_stride, Some(16));
        assert_eq!(input.len, 16 * DEFAULT_DISPATCH_GRID_X);

        let mut old_plan = plan.clone();
        old_plan
            .buffers
            .iter_mut()
            .find(|b| b.index == 7)
            .unwrap()
            .len = 256;
        let metal = MetalRow {
            air_sha256: "x".into(),
            shard: None,
            label: String::new(),
            status: "ok".into(),
            backend: "metal".into(),
            seed_profile: SEED_PROFILE.into(),
            plan_version: PLAN_VERSION,
            plan: old_plan,
            input_sha256: None,
            output_sha256: Some(sha256_hex(&[])),
            output_b64: Some(encode_output_b64(&[])),
            spv_sha256: None,
            compare: "full".into(),
            fc_specialization: None,
            fc_values: None,
            stage: None,
            entry: None,
            error: None,
        };
        let reason = incompatible_float_seed_golden(ll, &metal).unwrap();
        assert!(
            reason.contains("legacy finite_struct_float seed for buffer 7"),
            "{reason}"
        );
    }

    #[test]
    fn finite_struct_float_buffer_seed_expands_scalar_array_fields() {
        let plan = infer_plan(
            r#"
define void @kernel(ptr addrspace(1) %in) {
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @kernel, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 9, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.struct_type_info", !4, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"Payload", !"air.arg_name", !"in"}
!4 = !{i32 0, i32 4, i32 4, !"float", !"values"}
"#,
        );
        let buf = plan.buffers.iter().find(|b| b.index == 9).unwrap();
        assert_eq!(buf.seed_mode, SEED_MODE_FINITE_STRUCT_FLOAT);
        assert_eq!(buf.seed_stride, Some(16));
        assert_eq!(
            buf.seed_layout,
            vec![
                ControlSeedField {
                    offset: 0,
                    size: 4,
                    value: None,
                },
                ControlSeedField {
                    offset: 4,
                    size: 4,
                    value: None,
                },
                ControlSeedField {
                    offset: 8,
                    size: 4,
                    value: None,
                },
                ControlSeedField {
                    offset: 12,
                    size: 4,
                    value: None,
                },
            ]
        );
    }

    #[test]
    fn bounded_control_float_payload_inputs_cover_square_tile() {
        let plan = infer_plan(
            r#"
define void @gemv_like(ptr addrspace(1) %mat, ptr addrspace(1) %in_vec, ptr addrspace(1) %bias, ptr addrspace(1) %out_vec, ptr addrspace(2) %in_vec_size, ptr addrspace(2) %out_vec_size, ptr addrspace(2) %matrix_ld, i32 %simd_gid, i32 %simd_lid) {
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @gemv_like, !1, !2, !12}
!1 = !{}
!2 = !{!3, !4, !5, !6, !7, !8, !9, !10, !11}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"mat"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"in_vec"}
!5 = !{i32 2, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"bias", !"air.arg_unused"}
!6 = !{i32 3, !"air.buffer", !"air.location_index", i32 3, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"out_vec"}
!7 = !{i32 4, !"air.buffer", !"air.buffer_size", i32 4, !"air.location_index", i32 4, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_size", i32 4, !"air.arg_type_name", !"int", !"air.arg_name", !"in_vec_size"}
!8 = !{i32 5, !"air.buffer", !"air.buffer_size", i32 4, !"air.location_index", i32 5, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_size", i32 4, !"air.arg_type_name", !"int", !"air.arg_name", !"out_vec_size"}
!9 = !{i32 6, !"air.buffer", !"air.buffer_size", i32 4, !"air.location_index", i32 6, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_size", i32 4, !"air.arg_type_name", !"int", !"air.arg_name", !"matrix_ld"}
!10 = !{i32 7, !"air.simdgroup_index_in_threadgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"simd_gid"}
!11 = !{i32 8, !"air.thread_index_in_simdgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"simd_lid"}
!12 = !{!"air.max_work_group_size", i32 128}
"#,
        );
        let bounded_square_f32 = BOUNDED_CONTROL_DIM as usize * BOUNDED_CONTROL_DIM as usize * 4;
        for idx in [0u32, 1, 2] {
            let b = plan.buffers.iter().find(|b| b.index == idx).unwrap();
            assert_eq!(b.seed_mode, SEED_MODE_FINITE_FLOAT32);
            assert_eq!(b.len, bounded_square_f32, "buffer {idx}");
        }
        let out = plan.buffers.iter().find(|b| b.index == 3).unwrap();
        assert_eq!(out.len, DEFAULT_BUFFER_LEN);
        for idx in [4u32, 5, 6] {
            let b = plan.buffers.iter().find(|b| b.index == idx).unwrap();
            assert_eq!(b.seed_mode, SEED_MODE_BOUNDED_CONTROL);
            assert_eq!(b.len, 4);
        }
    }

    #[test]
    fn bounded_control_stride_scalar_seeds_to_one() {
        let plan = infer_plan(
            r#"
define void @stride_control(ptr addrspace(1) %data, ptr addrspace(2) %limit, ptr addrspace(2) %stride, i32 %tid) {
entry:
  %n = load i32, ptr addrspace(2) %limit, align 4
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %next, %loop ]
  %s = load i32, ptr addrspace(2) %stride, align 4
  %scaled = mul i32 %i, %s
  %idx = zext i32 %scaled to i64
  %ptr = getelementptr inbounds float, ptr addrspace(1) %data, i64 %idx
  %v = load float, ptr addrspace(1) %ptr, align 4
  %next = add i32 %i, 1
  %keep = icmp ult i32 %next, %n
  br i1 %keep, label %loop, label %done
done:
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @stride_control, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5, !6}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_name", !"float", !"air.arg_name", !"data"}
!4 = !{i32 1, !"air.buffer", !"air.buffer_size", i32 4, !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"limit"}
!5 = !{i32 2, !"air.buffer", !"air.buffer_size", i32 4, !"air.location_index", i32 2, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"stride"}
!6 = !{i32 3, !"air.thread_position_in_threadgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid"}
"#,
        );
        let limit = plan.buffers.iter().find(|b| b.index == 1).unwrap();
        let stride = plan.buffers.iter().find(|b| b.index == 2).unwrap();
        assert_eq!(
            limit.seed_layout,
            vec![ControlSeedField {
                offset: 0,
                size: 4,
                value: None
            }]
        );
        assert_eq!(
            stride.seed_layout,
            vec![ControlSeedField {
                offset: 0,
                size: 4,
                value: Some(1)
            }]
        );

        let owned = plan_to_owned_inputs(&plan).unwrap();
        let limit_bytes =
            seeded_buffer_bytes(owned.inputs.buffers.iter().find(|b| b.index == 1).unwrap());
        let stride_bytes =
            seeded_buffer_bytes(owned.inputs.buffers.iter().find(|b| b.index == 2).unwrap());
        assert_eq!(u32::from_le_bytes(limit_bytes[..4].try_into().unwrap()), 16);
        assert_eq!(u32::from_le_bytes(stride_bytes[..4].try_into().unwrap()), 1);
    }

    #[test]
    fn bounded_control_float_denominator_fields_seed_to_normal_one() {
        let plan = infer_plan(
            r#"
%struct.Params = type { float, float, i32 }

define float @fragment(ptr addrspace(2) %params) {
  %a_ptr = getelementptr inbounds %struct.Params, ptr addrspace(2) %params, i64 0, i32 0
  %a = load float, ptr addrspace(2) %a_ptr, align 4
  %b_ptr = getelementptr inbounds %struct.Params, ptr addrspace(2) %params, i64 0, i32 1
  %b = load float, ptr addrspace(2) %b_ptr, align 4
  %den = fsub float %b, %a
  %q = fdiv float %a, %den
  ret float %q
}

!air.fragment = !{!0}
!0 = !{ptr @fragment, !1, !2}
!1 = !{!3}
!2 = !{!4}
!3 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float"}
!4 = !{i32 0, !"air.buffer", !"air.buffer_size", i32 12, !"air.location_index", i32 5, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !5, !"air.arg_type_size", i32 12, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"Params", !"air.arg_name", !"params"}
!5 = !{i32 0, i32 4, i32 0, !"float", !"a", i32 4, i32 4, i32 0, !"float", !"b", i32 8, i32 4, i32 0, !"int", !"mode"}
"#,
        );
        let params = plan.buffers.iter().find(|b| b.index == 5).unwrap();
        assert_eq!(params.seed_mode, SEED_MODE_BOUNDED_CONTROL);
        assert_eq!(
            params.seed_layout,
            vec![
                ControlSeedField {
                    offset: 0,
                    size: 4,
                    value: Some(0x3f80_0000),
                },
                ControlSeedField {
                    offset: 4,
                    size: 4,
                    value: Some(0x3f80_0000),
                },
                ControlSeedField {
                    offset: 8,
                    size: 4,
                    value: None,
                },
            ]
        );

        let owned = plan_to_owned_inputs(&plan).unwrap();
        let bytes =
            seeded_buffer_bytes(owned.inputs.buffers.iter().find(|b| b.index == 5).unwrap());
        assert_eq!(f32::from_le_bytes(bytes[0..4].try_into().unwrap()), 1.0);
        assert_eq!(f32::from_le_bytes(bytes[4..8].try_into().unwrap()), 1.0);
        assert_eq!(u32::from_le_bytes(bytes[8..12].try_into().unwrap()), 16);
    }

    #[test]
    fn bounded_control_float_fields_seed_to_normal_one() {
        let plan = infer_plan(
            r#"
%struct.Params = type { float, i32, half }

define <4 x float> @fragment(ptr addrspace(2) %params) {
  ret <4 x float> zeroinitializer
}

!air.fragment = !{!0}
!0 = !{ptr @fragment, !1, !2}
!1 = !{!3}
!2 = !{!4}
!3 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!4 = !{i32 0, !"air.buffer", !"air.buffer_size", i32 12, !"air.location_index", i32 5, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !5, !"air.arg_type_size", i32 12, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"Params", !"air.arg_name", !"params"}
!5 = !{i32 0, i32 4, i32 0, !"float", !"gain", i32 4, i32 4, i32 0, !"int", !"mode", i32 8, i32 2, i32 0, !"half", !"opacity"}
"#,
        );
        let params = plan.buffers.iter().find(|b| b.index == 5).unwrap();
        assert_eq!(params.seed_mode, SEED_MODE_BOUNDED_CONTROL);
        assert_eq!(
            params.seed_layout,
            vec![
                ControlSeedField {
                    offset: 0,
                    size: 4,
                    value: Some(0x3f80_0000),
                },
                ControlSeedField {
                    offset: 4,
                    size: 4,
                    value: None,
                },
                ControlSeedField {
                    offset: 8,
                    size: 2,
                    value: Some(0x3c00),
                },
            ]
        );

        let owned = plan_to_owned_inputs(&plan).unwrap();
        let bytes =
            seeded_buffer_bytes(owned.inputs.buffers.iter().find(|b| b.index == 5).unwrap());
        assert_eq!(f32::from_le_bytes(bytes[0..4].try_into().unwrap()), 1.0);
        assert_eq!(u32::from_le_bytes(bytes[4..8].try_into().unwrap()), 16);
        assert_eq!(u16::from_le_bytes(bytes[8..10].try_into().unwrap()), 0x3c00);
    }

    #[test]
    fn finite_float_texture_seed_has_no_nan_or_inf() {
        let plan = infer_plan(
            r#"
define <4 x half> @fragment(ptr addrspace(1) %tex, ptr addrspace(2) %sampler, <2 x float> %uv) {
  %s = call { <4 x half>, i8 } @air.sample_texture_2d.v4f16(ptr addrspace(1) %tex, ptr addrspace(2) %sampler, <2 x float> %uv, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.0, float 0.0, i32 0)
  %c = extractvalue { <4 x half>, i8 } %s, 0
  ret <4 x half> %c
}

declare { <4 x half>, i8 } @air.sample_texture_2d.v4f16(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32)

!air.fragment = !{!0}
!0 = !{ptr @fragment, !1, !2}
!1 = !{!3}
!2 = !{!4, !5, !6}
!3 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"half4"}
!4 = !{i32 0, !"air.texture", !"air.location_index", i32 2, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<half, sample>", !"air.arg_name", !"tex"}
!5 = !{i32 1, !"air.sampler", !"air.location_index", i32 2, i32 1, !"air.arg_type_name", !"sampler", !"air.arg_name", !"sampler"}
!6 = !{i32 2, !"air.fragment_input", !"air.arg_type_name", !"float2", !"air.arg_name", !"uv"}
"#,
        );
        let tex = plan.textures.iter().find(|t| t.index == 2).unwrap();
        assert_eq!(tex.format, "Rgba16Float");
        assert_eq!(tex.seed_mode, SEED_MODE_FINITE_FLOAT16);

        let owned = plan_to_owned_inputs(&plan).unwrap();
        let input = owned.inputs.textures.iter().find(|t| t.index == 2).unwrap();
        let bytes = seeded_texture_bytes(input);
        assert!(!contains_nonfinite_float_lane(&bytes, 2));
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
    fn bounded_control_struct_seed_writes_ulong_fields_as_u64() {
        let plan = infer_plan(COPY_ARGS_U64_LL);
        let args = plan
            .buffers
            .iter()
            .find(|b| b.index == 30)
            .expect("copy args buffer");
        assert_eq!(args.seed_mode, SEED_MODE_BOUNDED_CONTROL);
        assert_eq!(
            args.seed_layout,
            vec![
                ControlSeedField {
                    offset: 0,
                    size: 8,
                    value: None
                },
                ControlSeedField {
                    offset: 8,
                    size: 8,
                    value: None
                },
                ControlSeedField {
                    offset: 16,
                    size: 8,
                    value: None
                },
            ]
        );

        let owned = plan_to_owned_inputs(&plan).unwrap();
        let bytes =
            seeded_buffer_bytes(owned.inputs.buffers.iter().find(|b| b.index == 30).unwrap());
        for offset in [0usize, 8, 16] {
            let value = u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap());
            assert_eq!(value, u64::from(BOUNDED_CONTROL_DIM), "offset {offset}");
        }
    }

    #[test]
    fn bounded_control_scalar_seed_writes_ulong_as_u64() {
        let plan = infer_plan(SCALAR_ULONG_CONTROL_LL);
        let stride = plan
            .buffers
            .iter()
            .find(|b| b.index == 7)
            .expect("stride buffer");
        assert_eq!(stride.seed_mode, SEED_MODE_BOUNDED_CONTROL);
        assert_eq!(
            stride.seed_layout,
            vec![ControlSeedField {
                offset: 0,
                size: 8,
                value: None
            }]
        );

        let owned = plan_to_owned_inputs(&plan).unwrap();
        let bytes =
            seeded_buffer_bytes(owned.inputs.buffers.iter().find(|b| b.index == 7).unwrap());
        let value = u64::from_le_bytes(bytes[0..8].try_into().unwrap());
        assert_eq!(value, u64::from(BOUNDED_CONTROL_DIM));
    }

    #[test]
    fn loop_bound_scan_sees_param_buffer() {
        let hit = buffers_with_loads_used_as_loop_bounds(MPS_LIKE_LL);
        assert!(
            hit.contains(&4),
            "expected buffer 4 (K/M loads → icmp → br), got {hit:?}"
        );
    }

    #[test]
    fn loop_bound_scan_sees_i64_shape_buffer() {
        let ll = r#"
define void @argmax_like(ptr addrspace(1) %input, ptr addrspace(1) %output, ptr addrspace(1) %shape) {
entry:
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %next, %loop ]
  %shape_value = load i64, ptr addrspace(1) %shape, align 8
  %next = add nuw i64 %i, 1
  %keep = icmp ult i64 %next, %shape_value
  br i1 %keep, label %loop, label %done
done:
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @argmax_like, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"input"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"output"}
!5 = !{i32 2, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 8, !"air.arg_type_name", !"ulong", !"air.arg_name", !"shape"}
"#;
        let hit = buffers_with_loads_used_as_loop_bounds(ll);
        assert!(
            hit.contains(&2),
            "expected buffer 2 (i64 shape load -> icmp -> loop br), got {hit:?}"
        );
        let plan = infer_plan(ll);
        let shape = plan.buffers.iter().find(|b| b.index == 2).unwrap();
        assert_eq!(shape.seed_mode, SEED_MODE_BOUNDED_CONTROL);
    }

    #[test]
    fn texture_bounds_guard_does_not_make_coordinate_buffer_bounded_control() {
        let ll = r#"
define void @asvMetalTraversalKernel(i32 %tid, ptr addrspace(1) %tex, ptr addrspace(1) %rayBuckets, ptr addrspace(1) %rayData, ptr addrspace(2) %targetRayCount) {
entry:
  %idx = zext i32 %tid to i64
  %bucket_ptr = getelementptr inbounds i32, ptr addrspace(1) %rayBuckets, i64 %idx
  %bucket = load i32, ptr addrspace(1) %bucket_ptr, align 4
  %w = tail call i32 @air.get_width_texture_2d(ptr addrspace(1) %tex, i32 0)
  %h = tail call i32 @air.get_height_texture_2d(ptr addrspace(1) %tex, i32 0)
  %area = mul i32 %h, %w
  %outside = icmp ugt i32 %bucket, %area
  br i1 %outside, label %done, label %write
write:
  ret void
done:
  ret void
}

declare i32 @air.get_width_texture_2d(ptr addrspace(1), i32)
declare i32 @air.get_height_texture_2d(ptr addrspace(1), i32)

!air.kernel = !{!0}
!0 = !{ptr @asvMetalTraversalKernel, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5, !6, !7}
!3 = !{i32 0, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid"}
!4 = !{i32 1, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.arg_type_name", !"texture2d<uint, read_write>", !"air.arg_name", !"traversalTexture"}
!5 = !{i32 2, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"rayBuckets"}
!6 = !{i32 3, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 12, !"air.arg_type_name", !"RayData", !"air.arg_name", !"allocateUSCValues"}
!7 = !{i32 4, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_size", i32 8, !"air.arg_type_name", !"ulong", !"air.arg_name", !"targetRayCount"}
"#;
        let hit = buffers_with_loads_used_as_loop_bounds(ll);
        assert!(
            !hit.contains(&0),
            "texture bounds guard must not make rayBuckets bounded_control: {hit:?}"
        );
        let plan = infer_plan(ll);
        let ray_buckets = plan.buffers.iter().find(|b| b.index == 0).unwrap();
        assert_eq!(ray_buckets.seed_mode, SEED_MODE_DETERMINISTIC);

        let mut old_plan = plan.clone();
        old_plan
            .buffers
            .iter_mut()
            .find(|b| b.index == 0)
            .unwrap()
            .seed_mode = SEED_MODE_BOUNDED_CONTROL.into();
        let metal = MetalRow {
            air_sha256: "x".into(),
            shard: None,
            label: "x".into(),
            status: "ok".into(),
            backend: "metal".into(),
            stage: Some("Kernel".into()),
            entry: Some("asvMetalTraversalKernel".into()),
            input_sha256: None,
            output_sha256: Some("gold".into()),
            output_b64: None,
            spv_sha256: None,
            compare: "full".into(),
            seed_profile: "deterministic_v2_bounded_control".into(),
            plan_version: PLAN_VERSION,
            plan: old_plan,
            fc_specialization: None,
            fc_values: None,
            error: None,
        };
        let reason = incompatible_bounded_control_seed_golden(ll, &metal)
            .expect("old bounded_control rayBuckets golden should need rebank");
        assert!(reason.contains("rebank Metal row"), "{reason}");
    }

    #[test]
    fn atomic_load_source_buffer_is_bounded_control() {
        let plan = infer_plan(ATOMIC_COUNTER_SOURCE_LL);
        let header = plan
            .buffers
            .iter()
            .find(|b| b.index == 4)
            .expect("header buffer");
        let data = plan
            .buffers
            .iter()
            .find(|b| b.index == 5)
            .expect("data buffer");
        let out = plan
            .buffers
            .iter()
            .find(|b| b.index == 6)
            .expect("output buffer");

        assert_eq!(header.seed_mode, SEED_MODE_BOUNDED_CONTROL);
        assert_eq!(data.seed_mode, SEED_MODE_BOUNDED_CONTROL);
        assert_eq!(out.seed_mode, SEED_MODE_DETERMINISTIC);
    }

    #[test]
    fn stale_bounded_control_seed_golden_is_missing() {
        let mut old_plan = infer_plan(ATOMIC_COUNTER_SOURCE_LL);
        let data = old_plan
            .buffers
            .iter_mut()
            .find(|b| b.index == 5)
            .expect("data buffer");
        data.seed_mode = SEED_MODE_DETERMINISTIC.into();
        let metal = MetalRow {
            air_sha256: "x".into(),
            shard: None,
            label: String::new(),
            status: "ok".into(),
            backend: "metal".into(),
            seed_profile: "deterministic_v4_finite_float_inputs".into(),
            plan_version: PLAN_VERSION,
            plan: old_plan,
            input_sha256: None,
            output_sha256: Some(sha256_hex(&16u32.to_le_bytes())),
            output_b64: Some(encode_output_b64(&16u32.to_le_bytes())),
            spv_sha256: None,
            compare: "none".into(),
            fc_specialization: None,
            fc_values: None,
            stage: None,
            entry: None,
            error: None,
        };

        let reason = incompatible_bounded_control_seed_golden(ATOMIC_COUNTER_SOURCE_LL, &metal)
            .expect("stale bounded-control seed");
        assert!(reason.contains("buffer 5"), "{reason}");
        assert!(reason.contains("rebank Metal row"), "{reason}");
    }

    #[test]
    fn thread_indexed_vector_input_oob_golden_is_missing() {
        let ll = r#"
define void @copy(ptr addrspace(1) %src, ptr addrspace(1) %dst, i32 %tid) {
entry:
  %idx = zext i32 %tid to i64
  %p = getelementptr inbounds <4 x i32>, ptr addrspace(1) %src, i64 %idx
  %v = load <4 x i32>, ptr addrspace(1) %p, align 16
  %x = extractelement <4 x i32> %v, i64 0
  %f = sitofp i32 %x to float
  %q = getelementptr inbounds float, ptr addrspace(1) %dst, i64 %idx
  store float %f, ptr addrspace(1) %q, align 4
  ret void
}
!air.kernel = !{!0}
!0 = !{ptr @copy, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_name", !"int4", !"air.arg_name", !"src"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"dst"}
!5 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid"}
"#;
        let mut metal = MetalRow {
            air_sha256: "x".into(),
            shard: None,
            label: "x".into(),
            status: "ok".into(),
            backend: "metal".into(),
            seed_profile: SEED_PROFILE.into(),
            plan_version: PLAN_VERSION,
            plan: HarnessPlan {
                buffers: vec![
                    PlanBuffer {
                        index: 0,
                        len: 16,
                        role: "Input".into(),
                        seed_tag: 1,
                        seed_mode: SEED_MODE_DETERMINISTIC.into(),
                        seed_layout: Vec::new(),
                        seed_stride: None,
                    },
                    PlanBuffer {
                        index: 1,
                        len: 64,
                        role: "InOut".into(),
                        seed_tag: 2,
                        seed_mode: SEED_MODE_FINITE_FLOAT32.into(),
                        seed_layout: Vec::new(),
                        seed_stride: None,
                    },
                ],
                textures: Vec::new(),
                output: PlanOutput {
                    kind: "buffer".into(),
                    index: 1,
                    format: "F32".into(),
                    len: Some(64),
                    w: None,
                    h: None,
                    d: None,
                },
                dispatch_grid: [4, 1, 1],
                dispatch_tg: [4, 1, 1],
            },
            input_sha256: None,
            output_sha256: Some("gold".into()),
            output_b64: None,
            spv_sha256: None,
            compare: "full".into(),
            fc_specialization: None,
            fc_values: None,
            stage: Some("Kernel".into()),
            entry: Some("copy".into()),
            error: None,
        };

        let reason = incompatible_oob_vector_input_golden(ll, &metal)
            .expect("thread-indexed vector load exceeds seeded input buffer");
        assert!(reason.contains("out-of-bounds"), "{reason}");

        metal.plan.buffers[0].len = 64;
        assert!(incompatible_oob_vector_input_golden(ll, &metal).is_none());
    }

    #[test]
    fn thread_indexed_scalar_input_stride_oob_golden_is_missing() {
        let ll = r#"
define void @norm(ptr addrspace(1) %src, ptr addrspace(1) %dst, i32 %tid) {
entry:
  %base = mul i32 %tid, 3
  %idx0 = zext i32 %base to i64
  %p0 = getelementptr inbounds float, ptr addrspace(1) %src, i64 %idx0
  %x = load float, ptr addrspace(1) %p0, align 4
  %plus = add i32 %base, 1
  %idx1 = zext i32 %plus to i64
  %p1 = getelementptr inbounds float, ptr addrspace(1) %src, i64 %idx1
  %y = load float, ptr addrspace(1) %p1, align 4
  %xx = fmul fast float %x, %x
  %yy = fmul fast float %y, %y
  %sum = fadd fast float %xx, %yy
  %out = tail call fast float @air.fast_sqrt.f32(float %sum)
  %dstp = getelementptr inbounds float, ptr addrspace(1) %dst, i64 %idx0
  store float %out, ptr addrspace(1) %dstp, align 4
  ret void
}

declare float @air.fast_sqrt.f32(float)

!air.kernel = !{!0}
!0 = !{ptr @norm, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"src"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"dst"}
!5 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid"}
"#;
        let mut metal = metal_row_for_compare(&[], infer_plan(ll), Some("Kernel"));
        metal.plan.dispatch_grid = [64, 1, 1];
        metal.plan.buffers.iter_mut().for_each(|buffer| {
            if buffer.index == 0 {
                buffer.len = 256;
            }
        });

        let reason = incompatible_oob_vector_input_golden(ll, &metal)
            .expect("strided scalar load exceeds seeded input buffer");
        assert!(reason.contains("out-of-bounds"), "{reason}");

        metal.plan.buffers.iter_mut().for_each(|buffer| {
            if buffer.index == 0 {
                buffer.len = 764;
            }
        });
        assert!(incompatible_oob_vector_input_golden(ll, &metal).is_none());
    }

    #[test]
    fn stale_typed_bounded_control_layout_golden_is_missing() {
        let mut old_plan = infer_plan(COPY_ARGS_U64_LL);
        let args = old_plan
            .buffers
            .iter_mut()
            .find(|b| b.index == 30)
            .expect("copy args buffer");
        args.seed_layout.clear();
        let metal = MetalRow {
            air_sha256: "x".into(),
            shard: None,
            label: String::new(),
            status: "ok".into(),
            backend: "metal".into(),
            seed_profile: SEED_PROFILE.into(),
            plan_version: PLAN_VERSION,
            plan: old_plan,
            input_sha256: None,
            output_sha256: Some(sha256_hex(&[0u8; 4])),
            output_b64: Some(encode_output_b64(&[0u8; 4])),
            spv_sha256: None,
            compare: "full".into(),
            fc_specialization: None,
            fc_values: None,
            stage: None,
            entry: None,
            error: None,
        };

        let reason = incompatible_bounded_control_seed_golden(COPY_ARGS_U64_LL, &metal)
            .expect("stale typed bounded-control layout");
        assert!(reason.contains("buffer 30"), "{reason}");
        assert!(reason.contains("typed AIR control metadata"), "{reason}");
        assert!(reason.contains("rebank Metal row"), "{reason}");
    }

    #[test]
    fn device_ptrtoint_golden_requires_rebank() {
        let ll = r#"
define void @kernel(ptr addrspace(1) %in, ptr addrspace(1) %out) {
entry:
  %p = getelementptr inbounds i32, ptr addrspace(1) %in, i64 1
  %bits = ptrtoint ptr addrspace(1) %p to i64
  store i64 %bits, ptr addrspace(1) %out, align 8
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @kernel, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_name", !"uint", !"air.arg_name", !"in"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_name", !"ulong", !"air.arg_name", !"out"}
"#;
        let metal = MetalRow {
            air_sha256: "x".into(),
            shard: None,
            label: String::new(),
            status: "ok".into(),
            backend: "metal".into(),
            seed_profile: SEED_PROFILE.into(),
            plan_version: PLAN_VERSION,
            plan: infer_plan(ll),
            input_sha256: None,
            output_sha256: Some(sha256_hex(&[])),
            output_b64: Some(encode_output_b64(&[])),
            spv_sha256: None,
            compare: "full".into(),
            fc_specialization: None,
            fc_values: None,
            stage: None,
            entry: None,
            error: None,
        };

        let reason = incompatible_nonportable_ptrtoint_golden(ll, &metal)
            .expect("ordinary device pointer address is backend-specific");
        assert!(
            reason.contains("ptrtoint device/constant pointer"),
            "{reason}"
        );
    }

    #[test]
    fn modeled_acceleration_structure_payload_ptrtoint_stays_comparable() {
        let ll = r#"
define void @kernel(ptr addrspace(1) %as, ptr addrspace(1) %out, i32 %idx) {
entry:
  %child = call ptr addrspace(1) @air.get_primitive_acceleration_structure_instance_acceleration_structure(ptr addrspace(1) %as, i32 %idx)
  %bits = ptrtoint ptr addrspace(1) %child to i64
  store i64 %bits, ptr addrspace(1) %out, align 8
  ret void
}

declare ptr addrspace(1) @air.get_primitive_acceleration_structure_instance_acceleration_structure(ptr addrspace(1), i32)

!air.kernel = !{!0}
!0 = !{ptr @kernel, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.instance_acceleration_structure", !"air.location_index", i32 8, i32 1, !"air.read", !"air.arg_type_name", !"acceleration_structure<instancing>", !"air.arg_name", !"as"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_name", !"ulong", !"air.arg_name", !"out"}
!5 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"idx"}
"#;
        let metal = MetalRow {
            air_sha256: "x".into(),
            shard: None,
            label: String::new(),
            status: "ok".into(),
            backend: "metal".into(),
            seed_profile: SEED_PROFILE.into(),
            plan_version: PLAN_VERSION,
            plan: infer_plan(ll),
            input_sha256: None,
            output_sha256: Some(sha256_hex(&[])),
            output_b64: Some(encode_output_b64(&[])),
            spv_sha256: None,
            compare: "full".into(),
            fc_specialization: None,
            fc_values: None,
            stage: None,
            entry: None,
            error: None,
        };

        assert!(
            incompatible_nonportable_ptrtoint_golden(ll, &metal).is_none(),
            "modeled AS payload bits are part of the validation ABI"
        );
    }

    #[test]
    fn dynamic_buffer_scatter_rawbytes_golden_requires_serial_rebank() {
        let ll = r#"
define void @scatter(i32 %tid, ptr addrspace(1) %out, ptr addrspace(1) %indices, ptr addrspace(1) %counter) {
entry:
  %tid64 = zext i32 %tid to i64
  %slotp = getelementptr i32, ptr addrspace(1) %indices, i64 %tid64
  %slot = load i32, ptr addrspace(1) %slotp, align 4
  %slot64 = zext i32 %slot to i64
  %dst = getelementptr i32, ptr addrspace(1) %out, i64 %slot64
  store i32 %tid, ptr addrspace(1) %dst, align 4
  %counterp = getelementptr i32, ptr addrspace(1) %counter, i64 0
  %old = call i32 @air.atomic.global.add.u.i32(ptr addrspace(1) captures(none) %counterp, i32 1, i32 0, i32 2, i1 true)
  ret void
}

declare i32 @air.atomic.global.add.u.i32(ptr addrspace(1) captures(none), i32, i32, i32, i1)

!air.kernel = !{!0}
!0 = !{ptr @scatter, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5, !6}
!3 = !{i32 0, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 256, !"air.arg_type_name", !"uchar", !"air.arg_name", !"out"}
!5 = !{i32 2, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 256, !"air.arg_type_name", !"uint", !"air.arg_name", !"indices"}
!6 = !{i32 3, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"counter"}
"#;
        let mut plan = infer_plan(ll);
        plan.output.kind = "buffer".into();
        plan.output.index = 0;
        plan.output.format = "RawBytes".into();
        plan.output.len = Some(256);
        plan.dispatch_grid = [64, 1, 1];
        let metal = MetalRow {
            air_sha256: "x".into(),
            shard: None,
            label: String::new(),
            status: "ok".into(),
            backend: "metal".into(),
            seed_profile: SEED_PROFILE.into(),
            plan_version: PLAN_VERSION,
            plan,
            input_sha256: None,
            output_sha256: Some(sha256_hex(&[])),
            output_b64: Some(encode_output_b64(&[])),
            spv_sha256: None,
            compare: "full".into(),
            fc_specialization: None,
            fc_values: None,
            stage: None,
            entry: None,
            error: None,
        };

        let reason = incompatible_parallel_dynamic_buffer_scatter_golden(ll, &metal)
            .expect("buffer-loaded scatter indices are schedule-dependent");
        assert!(reason.contains("dynamic buffer scatter"), "{reason}");
        assert!(reason.contains("serial/smoke plan"), "{reason}");
    }

    #[test]
    fn thread_indexed_rawbytes_output_store_stays_comparable() {
        let ll = r#"
define void @scatter(i32 %tid, ptr addrspace(1) %out, ptr addrspace(1) %input, ptr addrspace(1) %counter) {
entry:
  %tid64 = zext i32 %tid to i64
  %src = getelementptr i32, ptr addrspace(1) %input, i64 %tid64
  %value = load i32, ptr addrspace(1) %src, align 4
  %dst = getelementptr i32, ptr addrspace(1) %out, i64 %tid64
  store i32 %value, ptr addrspace(1) %dst, align 4
  %counterp = getelementptr i32, ptr addrspace(1) %counter, i64 0
  %old = call i32 @air.atomic.global.add.u.i32(ptr addrspace(1) captures(none) %counterp, i32 1, i32 0, i32 2, i1 true)
  ret void
}

declare i32 @air.atomic.global.add.u.i32(ptr addrspace(1) captures(none), i32, i32, i32, i1)

!air.kernel = !{!0}
!0 = !{ptr @scatter, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5, !6}
!3 = !{i32 0, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 256, !"air.arg_type_name", !"uchar", !"air.arg_name", !"out"}
!5 = !{i32 2, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 256, !"air.arg_type_name", !"uint", !"air.arg_name", !"input"}
!6 = !{i32 3, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"counter"}
"#;
        let mut plan = infer_plan(ll);
        plan.output.kind = "buffer".into();
        plan.output.index = 0;
        plan.output.format = "RawBytes".into();
        plan.output.len = Some(256);
        plan.dispatch_grid = [64, 1, 1];
        let metal = MetalRow {
            air_sha256: "x".into(),
            shard: None,
            label: String::new(),
            status: "ok".into(),
            backend: "metal".into(),
            seed_profile: SEED_PROFILE.into(),
            plan_version: PLAN_VERSION,
            plan,
            input_sha256: None,
            output_sha256: Some(sha256_hex(&[])),
            output_b64: Some(encode_output_b64(&[])),
            spv_sha256: None,
            compare: "full".into(),
            fc_specialization: None,
            fc_values: None,
            stage: None,
            entry: None,
            error: None,
        };

        assert!(
            incompatible_parallel_dynamic_buffer_scatter_golden(ll, &metal).is_none(),
            "per-thread output indices should remain byte-comparable"
        );
    }

    #[test]
    fn infer_plan_prefers_entry_writeonly_buffer_when_metadata_is_ambiguous() {
        let ll = r#"
define void @copy(ptr addrspace(1) noundef readonly "air-buffer-no-alias" %in, ptr addrspace(1) noundef writeonly "air-buffer-no-alias" %out, i32 %tid) {
entry:
  %idx = zext i32 %tid to i64
  %src = getelementptr i32, ptr addrspace(1) %in, i64 %idx
  %value = load i32, ptr addrspace(1) %src, align 4
  %dst = getelementptr i32, ptr addrspace(1) %out, i64 %idx
  store i32 %value, ptr addrspace(1) %dst, align 4
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @copy, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.buffer", !"air.buffer_size", i32 256, !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_name", !"uint", !"air.arg_name", !"in"}
!4 = !{i32 1, !"air.buffer", !"air.buffer_size", i32 256, !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!5 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid"}
"#;

        let plan = infer_plan(ll);
        assert_eq!(plan.output.kind, "buffer");
        assert_eq!(plan.output.index, 1);
    }

    #[test]
    fn simd_lane_threadgroup_read_after_simdgroup_store_requires_rebank() {
        let ll = r#"
@tg = internal addrspace(3) global [8 x i32] undef, align 4

define void @kernel(i16 %sg, i16 %lane, ptr addrspace(1) %out) {
entry:
  %sgi = zext i16 %sg to i64
  %p = getelementptr inbounds [8 x i32], ptr addrspace(3) @tg, i64 0, i64 %sgi
  store i32 7, ptr addrspace(3) %p, align 4
  call void @air.wg.barrier(i32 2, i32 1)
  %li = zext i16 %lane to i64
  %q = getelementptr inbounds [8 x i32], ptr addrspace(3) @tg, i64 0, i64 %li
  %v = load i32, ptr addrspace(3) %q, align 4
  store i32 %v, ptr addrspace(1) %out, align 4
  ret void
}

declare void @air.wg.barrier(i32, i32)

!air.kernel = !{!0}
!0 = !{ptr @kernel, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.simdgroup_index_in_threadgroup", !"air.arg_type_name", !"ushort", !"air.arg_name", !"sg"}
!4 = !{i32 1, !"air.thread_index_in_simdgroup", !"air.arg_type_name", !"ushort", !"air.arg_name", !"lane"}
!5 = !{i32 2, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
"#;
        let metal = MetalRow {
            air_sha256: "x".into(),
            shard: None,
            label: String::new(),
            status: "ok".into(),
            backend: "metal".into(),
            seed_profile: SEED_PROFILE.into(),
            plan_version: PLAN_VERSION,
            plan: infer_plan(ll),
            input_sha256: None,
            output_sha256: Some(sha256_hex(&[])),
            output_b64: Some(encode_output_b64(&[])),
            spv_sha256: None,
            compare: "full".into(),
            fc_specialization: None,
            fc_values: None,
            stage: None,
            entry: None,
            error: None,
        };

        let reason = incompatible_undefined_threadgroup_memory_golden(ll, &metal)
            .expect("simd lane read can observe unwritten threadgroup slots");
        assert!(reason.contains("thread_index_in_simdgroup"), "{reason}");
        assert!(reason.contains("unwritten threadgroup lanes"), "{reason}");
    }

    #[test]
    fn per_lane_threadgroup_store_and_load_stays_comparable() {
        let ll = r#"
@tg = internal addrspace(3) global [8 x i32] undef, align 4

define void @kernel(i16 %sg, i16 %lane, ptr addrspace(1) %out) {
entry:
  %li = zext i16 %lane to i64
  %p = getelementptr inbounds [8 x i32], ptr addrspace(3) @tg, i64 0, i64 %li
  store i32 7, ptr addrspace(3) %p, align 4
  call void @air.wg.barrier(i32 2, i32 1)
  %q = getelementptr inbounds [8 x i32], ptr addrspace(3) @tg, i64 0, i64 %li
  %v = load i32, ptr addrspace(3) %q, align 4
  store i32 %v, ptr addrspace(1) %out, align 4
  ret void
}

declare void @air.wg.barrier(i32, i32)

!air.kernel = !{!0}
!0 = !{ptr @kernel, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.simdgroup_index_in_threadgroup", !"air.arg_type_name", !"ushort", !"air.arg_name", !"sg"}
!4 = !{i32 1, !"air.thread_index_in_simdgroup", !"air.arg_type_name", !"ushort", !"air.arg_name", !"lane"}
!5 = !{i32 2, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
"#;
        let metal = MetalRow {
            air_sha256: "x".into(),
            shard: None,
            label: String::new(),
            status: "ok".into(),
            backend: "metal".into(),
            seed_profile: SEED_PROFILE.into(),
            plan_version: PLAN_VERSION,
            plan: infer_plan(ll),
            input_sha256: None,
            output_sha256: Some(sha256_hex(&[])),
            output_b64: Some(encode_output_b64(&[])),
            spv_sha256: None,
            compare: "full".into(),
            fc_specialization: None,
            fc_values: None,
            stage: None,
            entry: None,
            error: None,
        };

        assert!(
            incompatible_undefined_threadgroup_memory_golden(ll, &metal).is_none(),
            "per-lane initialized threadgroup slots should remain comparable"
        );
    }

    #[test]
    fn stale_scalar_bounded_control_layout_golden_is_missing() {
        let mut old_plan = infer_plan(SCALAR_ULONG_CONTROL_LL);
        let stride = old_plan
            .buffers
            .iter_mut()
            .find(|b| b.index == 7)
            .expect("stride buffer");
        stride.seed_layout.clear();
        let metal = MetalRow {
            air_sha256: "x".into(),
            shard: None,
            label: String::new(),
            status: "ok".into(),
            backend: "metal".into(),
            seed_profile: "deterministic_v2_bounded_control".into(),
            plan_version: PLAN_VERSION,
            plan: old_plan,
            input_sha256: None,
            output_sha256: Some(sha256_hex(&[0u8; 4])),
            output_b64: Some(encode_output_b64(&[0u8; 4])),
            spv_sha256: None,
            compare: "full".into(),
            fc_specialization: None,
            fc_values: None,
            stage: None,
            entry: None,
            error: None,
        };

        let reason = incompatible_bounded_control_seed_golden(SCALAR_ULONG_CONTROL_LL, &metal)
            .expect("stale scalar bounded-control layout");
        assert!(reason.contains("buffer 7"), "{reason}");
        assert!(reason.contains("typed AIR control metadata"), "{reason}");
        assert!(reason.contains("rebank Metal row"), "{reason}");
    }

    #[test]
    fn vector_control_struct_layout_rebanks_for_typed_float_seed() {
        let current_plan = infer_plan(VECTOR_CONTROL_STRUCT_LL);
        let uniforms = current_plan
            .buffers
            .iter()
            .find(|b| b.index == 1)
            .expect("uniform buffer");
        assert_eq!(
            uniforms.seed_layout,
            vec![ControlSeedField {
                offset: 96,
                size: 4,
                value: Some(0x3f80_0000)
            }]
        );

        let mut old_plan = current_plan.clone();
        old_plan
            .buffers
            .iter_mut()
            .find(|b| b.index == 1)
            .expect("uniform buffer")
            .seed_layout
            .clear();
        let metal = MetalRow {
            air_sha256: "x".into(),
            shard: None,
            label: String::new(),
            status: "ok".into(),
            backend: "metal".into(),
            seed_profile: "deterministic_v2_bounded_control".into(),
            plan_version: PLAN_VERSION,
            plan: old_plan,
            input_sha256: None,
            output_sha256: Some(sha256_hex(&[0u8; 4])),
            output_b64: Some(encode_output_b64(&[0u8; 4])),
            spv_sha256: None,
            compare: "full".into(),
            fc_specialization: None,
            fc_values: None,
            stage: None,
            entry: None,
            error: None,
        };

        let reason = incompatible_bounded_control_seed_golden(VECTOR_CONTROL_STRUCT_LL, &metal)
            .expect("stale vector control layout");
        assert!(reason.contains("buffer 1"), "{reason}");
        assert!(reason.contains("typed AIR control metadata"), "{reason}");
        assert!(reason.contains("rebank Metal row"), "{reason}");
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
    fn loop_guarded_bda_as_pointer_candidate_translates() {
        let guarded = match crate::loop_budget::classify_and_instrument(
            LOOP_GUARDED_BDA_AS_POINTER_LL,
            "bda_loop_guard",
        ) {
            crate::loop_budget::GuardPlan::Instrumented(text) => text,
            other => panic!("expected instrumented loop, got {other:?}"),
        };
        assert!(
            !guarded.contains("m2v.g.0:"),
            "metadata-bearing loop should be guarded in place:\n{guarded}"
        );
        assert!(
            guarded.contains("br i1 %m2v.0.leave, label %exit, label %loop, !llvm.loop !5"),
            "budgeted loop branch missing:\n{guarded}"
        );

        let tmp = crate::scratch_dir_for("loop_guarded_bda_as_pointer_candidate_translates");
        let spv = metal2vulkan::translate_sanitized_native(
            &guarded,
            metal2vulkan::passes::Stage::Kernel,
            &tmp,
        )
        .expect("loop-guarded BDA AS pointer candidate should translate");
        metal2vulkan::tools::spirv_val_bytes(&spv, &tmp)
            .expect("loop-guarded BDA AS pointer candidate should validate");
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn vulkan_candidate_translate_uses_plan_dispatch_shape() {
        let ll = r#"
target triple = "spirv-unknown-vulkan1.3"

define void @k(<2 x i32> %grid_size) {
entry:
  %sx = extractelement <2 x i32> %grid_size, i64 0
  %sy = extractelement <2 x i32> %grid_size, i64 1
  %sum = add i32 %sx, %sy
  %ok = icmp uge i32 %sum, 0
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.threads_per_grid", !"air.arg_type_name", !"uint2", !"air.arg_name", !"grid_size"}
"#;
        let plan = HarnessPlan {
            buffers: vec![],
            textures: vec![],
            output: PlanOutput {
                kind: "buffer".into(),
                index: 0,
                format: "RawBytes".into(),
                len: Some(4),
                w: None,
                h: None,
                d: None,
            },
            dispatch_grid: [64, 2, 1],
            dispatch_tg: [32, 2, 1],
        };
        let tmp = crate::scratch_dir_for("vulkan_candidate_translate_uses_plan_dispatch_shape");
        let spv = translate_candidate_spv_for_plan(ll, Stage::Kernel, &plan, &tmp)
            .expect("translate candidate");
        let _ = std::fs::remove_dir_all(&tmp);
        let asm = metal2vulkan::disassemble(&spv).expect("disassemble candidate");
        assert!(asm.contains("LocalSize 32 2 1"), "{asm}");
        assert!(!asm.contains("BuiltIn NumWorkgroups"), "{asm}");
        assert!(!asm.contains("OpIMul"), "{asm}");
        assert!(
            asm.lines()
                .any(|line| line.contains("OpConstant") && line.contains("64")),
            "{asm}"
        );
        assert!(
            asm.lines()
                .any(|line| line.contains("OpConstant") && line.contains("2")),
            "{asm}"
        );
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
        // One ULP above 1.0: well within the default ULP tolerance.
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
            fc_specialization: None,
            fc_values: None,
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
            None,
        );
        assert_eq!(status, "tolerance");
        assert!(observed.is_some());
        assert_eq!(
            tolerance.as_ref().map(|t| t.kind.as_str()),
            Some("AbsOrUlp")
        );
    }

    #[test]
    fn compare_fast_math_nonfinite_domain_result_requires_rebank() {
        let ll = r#"
define float @kernel(float %x) #0 {
entry:
  %r = tail call fast float @air.fast_rsqrt.f32(float %x)
  ret float %r
}
declare float @air.fast_rsqrt.f32(float)
attributes #0 = { "no-nans-fp-math"="true" "unsafe-fp-math"="true" }
"#;
        let golden = 1.0f32.to_le_bytes();
        let candidate = f32::NAN.to_bits().to_le_bytes();
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
            fc_specialization: None,
            fc_values: None,
            stage: None,
            entry: None,
            error: None,
        };
        let (status, observed, tolerance) = compare_candidate_to_metal(
            &candidate,
            &metal,
            &sha256_hex(&candidate),
            metal.output_sha256.as_deref().unwrap(),
            DataFormat::F32,
            Some(ll),
        );

        assert_eq!(status, "missing");
        assert_eq!(
            tolerance.as_ref().map(|t| t.kind.as_str()),
            Some("FastMathNonFiniteDomain")
        );
        assert_eq!(observed.and_then(|m| m.max_ulp), Some(u32::MAX));
        let error = candidate_compare_error(&status, &metal, tolerance.as_ref()).unwrap();
        assert!(error.contains("undefined fast-math domains"), "{error}");
    }

    #[test]
    fn compare_raw_bytes_from_pack_unorm_allows_one_lsb_channel_drift() {
        let ll = r#"
define void @kernel(ptr addrspace(1) %out) {
entry:
  %packed = tail call i32 @air.pack.unorm4x8.v4f16(<4 x half> <half 0xH3800, half 0xH3C00, half 0xH0000, half 0xH3400>)
  store i32 %packed, ptr addrspace(1) %out, align 4
  ret void
}

declare i32 @air.pack.unorm4x8.v4f16(<4 x half>)
"#;
        let golden = [10u8, 20, 30, 40, 50, 60, 70, 80];
        let candidate = [11u8, 19, 30, 40, 51, 60, 69, 80];
        let result = compare_to_golden(
            &candidate,
            &golden,
            DataFormat::RawBytes,
            tolerance_for_context(DataFormat::RawBytes, Some(ll)).as_ref(),
        );

        assert_eq!(result.status, "tolerance");
        assert_eq!(result.observed.as_ref().and_then(|m| m.max_abs), Some(1.0));
        assert_eq!(
            result.tolerance.as_ref().map(|t| t.kind.as_str()),
            Some("Abs")
        );
    }

    #[test]
    fn compare_raw_finite_struct_float_tolerates_only_float_fields() {
        let ll = r#"
define void @kernel(ptr addrspace(1) %out) {
entry:
  ret void
}
!air.compile_options = !{!0}
!0 = !{!"air.compile.fast_math_enable"}
"#;
        let mut plan = infer_plan("");
        plan.output.kind = "buffer".into();
        plan.output.index = 0;
        plan.output.format = "RawBytes".into();
        plan.output.len = Some(16);
        plan.buffers = vec![PlanBuffer {
            index: 0,
            len: 16,
            role: "InOut".into(),
            seed_tag: 1,
            seed_mode: SEED_MODE_FINITE_STRUCT_FLOAT.into(),
            seed_layout: vec![
                ControlSeedField {
                    offset: 0,
                    size: 4,
                    value: None,
                },
                ControlSeedField {
                    offset: 4,
                    size: 2,
                    value: None,
                },
            ],
            seed_stride: Some(8),
        }];
        let mut golden = vec![0u8; 16];
        golden[0..4].copy_from_slice(&1.0f32.to_le_bytes());
        golden[4..6].copy_from_slice(&0x3c00u16.to_le_bytes());
        let mut candidate = golden.clone();
        candidate[0..4].copy_from_slice(&f32::from_bits(1.0f32.to_bits() + 1).to_le_bytes());
        candidate[4..6].copy_from_slice(&0x3c01u16.to_le_bytes());
        let metal = MetalRow {
            air_sha256: "x".into(),
            shard: None,
            label: String::new(),
            status: "ok".into(),
            backend: "metal".into(),
            seed_profile: SEED_PROFILE.into(),
            plan_version: PLAN_VERSION,
            plan,
            input_sha256: None,
            output_sha256: Some(sha256_hex(&golden)),
            output_b64: Some(encode_output_b64(&golden)),
            spv_sha256: None,
            compare: "full".into(),
            fc_specialization: None,
            fc_values: None,
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
            Some(ll),
        );

        assert_eq!(status, "tolerance");
        assert_eq!(
            tolerance.as_ref().map(|t| t.kind.as_str()),
            Some("MaskedStructFloatAbsOrUlp")
        );
        assert!(observed
            .and_then(|m| m.max_ulp)
            .is_some_and(|ulp| ulp <= 32));
    }

    #[test]
    fn compare_raw_finite_struct_float_keeps_non_float_bytes_exact() {
        let ll = r#"
define void @kernel(ptr addrspace(1) %out) {
entry:
  ret void
}
!air.compile_options = !{!0}
!0 = !{!"air.compile.fast_math_enable"}
"#;
        let mut plan = infer_plan("");
        plan.output.kind = "buffer".into();
        plan.output.index = 0;
        plan.output.format = "RawBytes".into();
        plan.output.len = Some(16);
        plan.buffers = vec![PlanBuffer {
            index: 0,
            len: 16,
            role: "InOut".into(),
            seed_tag: 1,
            seed_mode: SEED_MODE_FINITE_STRUCT_FLOAT.into(),
            seed_layout: vec![ControlSeedField {
                offset: 0,
                size: 4,
                value: None,
            }],
            seed_stride: Some(8),
        }];
        let golden = vec![0u8; 16];
        let mut candidate = golden.clone();
        candidate[6] = 1;
        let metal = MetalRow {
            air_sha256: "x".into(),
            shard: None,
            label: String::new(),
            status: "ok".into(),
            backend: "metal".into(),
            seed_profile: SEED_PROFILE.into(),
            plan_version: PLAN_VERSION,
            plan,
            input_sha256: None,
            output_sha256: Some(sha256_hex(&golden)),
            output_b64: Some(encode_output_b64(&golden)),
            spv_sha256: None,
            compare: "full".into(),
            fc_specialization: None,
            fc_values: None,
            stage: None,
            entry: None,
            error: None,
        };
        let (status, _, tolerance) = compare_candidate_to_metal(
            &candidate,
            &metal,
            &sha256_hex(&candidate),
            metal.output_sha256.as_deref().unwrap(),
            DataFormat::RawBytes,
            Some(ll),
        );

        assert_eq!(status, "failure");
        assert_eq!(
            tolerance.as_ref().map(|t| t.kind.as_str()),
            Some("MaskedStructFloatAbsOrUlp")
        );
    }

    #[test]
    fn compare_plain_raw_bytes_remains_exact() {
        let result = compare_to_golden(
            &[2u8, 4, 6, 8],
            &[1u8, 4, 6, 8],
            DataFormat::RawBytes,
            tolerance_for_context(
                DataFormat::RawBytes,
                Some("define void @kernel() { ret void }"),
            )
            .as_ref(),
        );

        assert_eq!(result.status, "failure");
        assert!(result.tolerance.is_none());
    }

    #[test]
    fn compare_large_float_with_small_ulp_drift_uses_ulp_tolerance() {
        let golden = f32::from_bits(0x7600_0000);
        let candidate = f32::from_bits(golden.to_bits() + 27);
        let result = compare_to_golden(
            &candidate.to_le_bytes(),
            &golden.to_le_bytes(),
            DataFormat::F32,
            Some(&default_float_tolerance()),
        );

        assert_eq!(result.status, "tolerance");
        let observed = result.observed.expect("observed margins");
        assert!(
            observed.max_abs.unwrap() > default_float_tolerance().max_abs.unwrap(),
            "{observed:?}"
        );
        assert_eq!(observed.max_ulp, Some(27));
    }

    #[test]
    fn compare_float_nan_vs_finite_stays_failure() {
        let golden = f32::from_bits(0x7600_0000);
        let candidate = f32::from_bits(0x7fff_ffff);
        let result = compare_to_golden(
            &candidate.to_le_bytes(),
            &golden.to_le_bytes(),
            DataFormat::F32,
            Some(&default_float_tolerance()),
        );

        assert_eq!(result.status, "failure");
        assert!(
            result.observed.and_then(|m| m.max_ulp).unwrap()
                > default_float_tolerance().max_ulp.unwrap()
        );
    }

    #[test]
    fn compare_half_nan_payloads_classifies_tolerance() {
        let golden = 0x7e00u16.to_le_bytes();
        let candidate = 0x7fffu16.to_le_bytes();
        let result = compare_to_golden(
            &candidate,
            &golden,
            DataFormat::R16Float,
            Some(&default_float_tolerance()),
        );

        assert_eq!(result.status, "tolerance");
        assert_eq!(result.observed.and_then(|m| m.max_ulp), Some(0));
    }

    #[test]
    fn compare_half_signed_zero_to_subnormal_uses_ordered_ulp() {
        let golden = 0x8000u16.to_le_bytes();
        let candidate = 0x0003u16.to_le_bytes();
        let result = compare_to_golden(
            &candidate,
            &golden,
            DataFormat::R16Float,
            Some(&default_float_tolerance()),
        );

        assert_eq!(result.status, "tolerance");
        assert_eq!(result.observed.and_then(|m| m.max_ulp), Some(3));
    }

    #[test]
    fn compare_half_output_uses_half_absolute_tolerance() {
        let golden = 0x3800u16.to_le_bytes(); // 0.5
        let candidate = 0x3804u16.to_le_bytes(); // 0.501953125
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
            fc_specialization: None,
            fc_values: None,
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
            DataFormat::Rgba16Float,
            None,
        );

        assert_eq!(status, "tolerance");
        let observed = observed.expect("observed margins");
        assert_eq!(observed.max_abs, Some(0.001_953_125));
        assert_eq!(tolerance.and_then(|t| t.max_abs), Some(0.001_953_125));
    }

    #[test]
    fn compare_fast_math_half_output_uses_wider_absolute_tolerance() {
        let golden = 0xad3du16.to_le_bytes(); // approximately -0.08185
        let candidate = 0xad13u16.to_le_bytes(); // approximately -0.07928
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
            fc_specialization: None,
            fc_values: None,
            stage: None,
            entry: None,
            error: None,
        };
        let out_hash = sha256_hex(&candidate);
        let golden_hash = metal.output_sha256.clone().unwrap();
        let ordinary = compare_candidate_to_metal(
            &candidate,
            &metal,
            &out_hash,
            &golden_hash,
            DataFormat::Rgba16Float,
            None,
        );
        assert_eq!(ordinary.0, "failure");

        let fast_ll = r#"
define <4 x half> @frag(<4 x half> %x) #0 {
  %y = fmul fast <4 x half> %x, %x
  ret <4 x half> %y
}
attributes #0 = { "no-nans-fp-math"="true" }
"#;
        let (status, observed, tolerance) = compare_candidate_to_metal(
            &candidate,
            &metal,
            &out_hash,
            &golden_hash,
            DataFormat::Rgba16Float,
            Some(fast_ll),
        );
        assert_eq!(status, "tolerance");
        let observed = observed.expect("observed margins");
        assert!(observed.max_abs.unwrap() > 0.001_953_125, "{observed:?}");
        assert_eq!(tolerance.and_then(|t| t.max_abs), Some(0.003_906_25));
    }

    #[test]
    fn compare_sampled_fast_half_render_target_uses_texture_coordinate_tolerance() {
        let golden = 0x2c17u16.to_le_bytes(); // approximately 0.0639
        let candidate = 0x2badu16.to_le_bytes(); // approximately 0.0600
        let mut plan = infer_plan("");
        plan.output.kind = "render_target".into();
        plan.output.format = "Rgba16Float".into();
        let metal = metal_row_for_compare(&golden, plan, Some("Fragment"));
        let ll = r#"
define <4 x half> @frag(ptr addrspace(1) %tex) #0 {
  %r = tail call fast half @air.rsqrt.f16(half 0xH3C00)
  %uv = tail call fast <2 x float> @air.convert.f.v2f32.f.v2f16(<2 x half> zeroinitializer)
  %s = tail call { <4 x half>, i8 } @air.sample_texture_2d.v4f16(ptr addrspace(1) %tex, ptr addrspace(2) null, <2 x float> %uv, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %v = extractvalue { <4 x half>, i8 } %s, 0
  ret <4 x half> %v
}
declare half @air.rsqrt.f16(half)
declare <2 x float> @air.convert.f.v2f32.f.v2f16(<2 x half>)
declare { <4 x half>, i8 } @air.sample_texture_2d.v4f16(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32)
attributes #0 = { "no-nans-fp-math"="true" "unsafe-fp-math"="true" }
"#;
        let (status, observed, tolerance) = compare_candidate_to_metal(
            &candidate,
            &metal,
            &sha256_hex(&candidate),
            metal.output_sha256.as_deref().unwrap(),
            DataFormat::Rgba16Float,
            Some(ll),
        );

        assert_eq!(status, "tolerance");
        assert_eq!(observed.and_then(|m| m.max_abs), Some(0.003_936_767_6));
        assert_eq!(tolerance.and_then(|t| t.max_abs), Some(0.007_812_5));
    }

    #[test]
    fn compare_sampled_fast_half_render_target_counts_trig_coordinate_math() {
        let golden = 0x2c17u16.to_le_bytes(); // approximately 0.0639
        let candidate = 0x2b47u16.to_le_bytes(); // approximately 0.0569
        let mut plan = infer_plan("");
        plan.output.kind = "render_target".into();
        plan.output.format = "Rgba16Float".into();
        let metal = metal_row_for_compare(&golden, plan, Some("Fragment"));
        let ll = r#"
define <4 x half> @frag(ptr addrspace(1) %tex, ptr %cos_out) #0 {
  %sin = tail call fast half @air.sincos.f16(half 0xH3C00, ptr %cos_out)
  %uv = tail call fast <2 x float> @air.convert.f.v2f32.f.v2f16(<2 x half> zeroinitializer)
  %s = tail call { <4 x half>, i8 } @air.sample_texture_2d.v4f16(ptr addrspace(1) %tex, ptr addrspace(2) null, <2 x float> %uv, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %v = extractvalue { <4 x half>, i8 } %s, 0
  ret <4 x half> %v
}
declare half @air.sincos.f16(half, ptr)
declare <2 x float> @air.convert.f.v2f32.f.v2f16(<2 x half>)
declare { <4 x half>, i8 } @air.sample_texture_2d.v4f16(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32)
attributes #0 = { "no-nans-fp-math"="true" "unsafe-fp-math"="true" }
"#;
        let (status, observed, tolerance) = compare_candidate_to_metal(
            &candidate,
            &metal,
            &sha256_hex(&candidate),
            metal.output_sha256.as_deref().unwrap(),
            DataFormat::Rgba16Float,
            Some(ll),
        );

        assert_eq!(status, "tolerance");
        assert!(observed
            .and_then(|m| m.max_abs)
            .is_some_and(|abs| abs > 0.003_906_25));
        assert_eq!(tolerance.and_then(|t| t.max_abs), Some(0.007_812_5));
    }

    #[test]
    fn compare_sampled_fast_half_render_target_allows_small_sampling_drift() {
        let golden = 0x2c17u16.to_le_bytes(); // approximately 0.0639
        let candidate = 0x2b47u16.to_le_bytes(); // approximately 0.0569
        let mut plan = infer_plan("");
        plan.output.kind = "render_target".into();
        plan.output.format = "Rgba16Float".into();
        let metal = metal_row_for_compare(&golden, plan, Some("Fragment"));
        let ll = r#"
define <4 x half> @frag(ptr addrspace(1) %tex) #0 {
  %uv = tail call fast <2 x float> @air.convert.f.v2f32.f.v2f16(<2 x half> zeroinitializer)
  %s = tail call { <4 x half>, i8 } @air.sample_texture_2d.v4f16(ptr addrspace(1) %tex, ptr addrspace(2) null, <2 x float> %uv, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %v = extractvalue { <4 x half>, i8 } %s, 0
  ret <4 x half> %v
}
declare <2 x float> @air.convert.f.v2f32.f.v2f16(<2 x half>)
declare { <4 x half>, i8 } @air.sample_texture_2d.v4f16(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32)
attributes #0 = { "no-nans-fp-math"="true" "unsafe-fp-math"="true" }
"#;
        let (status, observed, tolerance) = compare_candidate_to_metal(
            &candidate,
            &metal,
            &sha256_hex(&candidate),
            metal.output_sha256.as_deref().unwrap(),
            DataFormat::Rgba16Float,
            Some(ll),
        );

        assert_eq!(status, "tolerance");
        assert!(observed
            .and_then(|m| m.max_abs)
            .is_some_and(|abs| abs > 0.003_906_25));
        assert_eq!(tolerance.and_then(|t| t.max_abs), Some(0.007_812_5));
    }

    #[test]
    fn compare_sampled_fast_half_buffer_output_keeps_existing_tolerance() {
        let golden = 0x2c17u16.to_le_bytes();
        let candidate = 0x2badu16.to_le_bytes();
        let mut plan = infer_plan("");
        plan.output.kind = "buffer".into();
        plan.output.format = "Rgba16Float".into();
        let metal = metal_row_for_compare(&golden, plan, Some("Fragment"));
        let ll = r#"
define <4 x half> @frag(ptr addrspace(1) %tex) #0 {
  %r = tail call fast half @air.rsqrt.f16(half 0xH3C00)
  %uv = tail call fast <2 x float> @air.convert.f.v2f32.f.v2f16(<2 x half> zeroinitializer)
  %s = tail call { <4 x half>, i8 } @air.sample_texture_2d.v4f16(ptr addrspace(1) %tex, ptr addrspace(2) null, <2 x float> %uv, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %v = extractvalue { <4 x half>, i8 } %s, 0
  ret <4 x half> %v
}
declare half @air.rsqrt.f16(half)
declare <2 x float> @air.convert.f.v2f32.f.v2f16(<2 x half>)
declare { <4 x half>, i8 } @air.sample_texture_2d.v4f16(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32)
attributes #0 = { "no-nans-fp-math"="true" "unsafe-fp-math"="true" }
"#;
        let (status, _, tolerance) = compare_candidate_to_metal(
            &candidate,
            &metal,
            &sha256_hex(&candidate),
            metal.output_sha256.as_deref().unwrap(),
            DataFormat::Rgba16Float,
            Some(ll),
        );

        assert_eq!(status, "failure");
        assert_eq!(tolerance.and_then(|t| t.max_abs), Some(0.003_906_25));
    }

    #[test]
    fn compare_half_abs_or_ulp_is_applied_per_lane() {
        let mut golden = Vec::new();
        let mut candidate = Vec::new();
        // Lane 0 exceeds the half absolute bound, but is only 6 ULP away.
        golden.extend_from_slice(&0x3adau16.to_le_bytes());
        candidate.extend_from_slice(&0x3ae0u16.to_le_bytes());
        // Lane 1 is 146 ULP away near zero, but is within the half absolute bound.
        golden.extend_from_slice(&0x221eu16.to_le_bytes());
        candidate.extend_from_slice(&0x22b0u16.to_le_bytes());

        let result = compare_to_golden(
            &candidate,
            &golden,
            DataFormat::Rgba16Float,
            Some(&float_tolerance_for_format(DataFormat::Rgba16Float)),
        );

        assert_eq!(result.status, "tolerance");
        let observed = result.observed.expect("observed margins");
        assert_eq!(observed.max_abs, Some(0.002_929_687_5));
        assert_eq!(observed.max_ulp, Some(146));
    }

    #[test]
    fn candidate_compare_format_prefers_current_buffer_air_type() {
        let ll = r#"
define void @kernel(ptr addrspace(1) %out) {
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @kernel, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 3, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_name", !"float"}
"#;
        let mut plan = infer_plan("");
        plan.output.kind = "buffer".into();
        plan.output.index = 3;
        plan.output.format = "RawBytes".into();
        let metal = MetalRow {
            air_sha256: "x".into(),
            shard: None,
            label: String::new(),
            status: "ok".into(),
            backend: "metal".into(),
            seed_profile: SEED_PROFILE.into(),
            plan_version: PLAN_VERSION,
            plan: plan.clone(),
            input_sha256: None,
            output_sha256: Some(sha256_hex(&0x7fc00000u32.to_le_bytes())),
            output_b64: Some(encode_output_b64(&0x7fc00000u32.to_le_bytes())),
            spv_sha256: None,
            compare: "none".into(),
            fc_specialization: None,
            fc_values: None,
            stage: None,
            entry: None,
            error: None,
        };

        let format = candidate_compare_format(ll, &plan, &metal);
        assert_eq!(format, DataFormat::F32);

        let candidate = 0x7fffffffu32.to_le_bytes();
        let golden = 0x7fc00000u32.to_le_bytes();
        let result = compare_to_golden(
            &candidate,
            &golden,
            format,
            Some(&default_float_tolerance()),
        );
        assert_eq!(result.status, "tolerance");
    }

    #[test]
    fn hash_only_float_golden_is_missing_not_failure() {
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
            output_sha256: Some(sha256_hex(&0x7fc00000u32.to_le_bytes())),
            output_b64: None,
            spv_sha256: None,
            compare: "full".into(),
            fc_specialization: None,
            fc_values: None,
            stage: None,
            entry: None,
            error: None,
        };

        let candidate = 0x7fffffffu32.to_le_bytes();
        let (status, observed, tolerance) = compare_candidate_to_metal(
            &candidate,
            &metal,
            &sha256_hex(&candidate),
            metal.output_sha256.as_deref().unwrap_or_default(),
            DataFormat::F32,
            None,
        );

        assert_eq!(status, "missing");
        assert_eq!(
            candidate_compare_error(&status, &metal, tolerance.as_ref()).as_deref(),
            Some("metal golden lacks output_b64 for float/tolerance comparison; rebank Metal row")
        );
        assert!(observed.is_none());
        assert!(tolerance.is_none());
    }

    #[test]
    fn compare_none_mismatch_is_smoke_not_failure() {
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
            output_sha256: Some(sha256_hex(&[1, 2, 3, 4])),
            output_b64: Some(encode_output_b64(&[1, 2, 3, 4])),
            spv_sha256: None,
            compare: "none".into(),
            fc_specialization: None,
            fc_values: None,
            stage: None,
            entry: None,
            error: None,
        };
        let candidate = [4, 3, 2, 1];
        let (status, observed, tolerance) = compare_candidate_to_metal(
            &candidate,
            &metal,
            &sha256_hex(&candidate),
            metal.output_sha256.as_deref().unwrap_or_default(),
            DataFormat::RawBytes,
            None,
        );

        assert_eq!(status, "smoke");
        assert!(observed.is_none());
        assert!(tolerance.is_none());
    }

    #[test]
    fn fast_float_integer_render_target_allows_byte_quantization_tolerance() {
        let ll = r#"
define i16 @frag(half %x) #0 {
  %r = tail call fast half @air.round.f16(half %x)
  %u = tail call i16 @air.convert.u.i16.f.f16(half %r)
  ret i16 %u
}

declare half @air.round.f16(half)
declare i16 @air.convert.u.i16.f.f16(half)

attributes #0 = { "no-nans-fp-math"="true" }

!air.fragment = !{!0}
!0 = !{ptr @frag, !1, !2}
!1 = !{!3}
!2 = !{}
!3 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"ushort"}
"#;
        let golden = [0xee, 0x87, 0xec, 0x95];
        let candidate = [0xec, 0x86, 0xec, 0x95];
        let metal = MetalRow {
            air_sha256: "x".into(),
            shard: None,
            label: String::new(),
            status: "ok".into(),
            backend: "metal".into(),
            seed_profile: SEED_PROFILE.into(),
            plan_version: PLAN_VERSION,
            plan: infer_plan(ll),
            input_sha256: None,
            output_sha256: Some(sha256_hex(&golden)),
            output_b64: Some(encode_output_b64(&golden)),
            spv_sha256: None,
            compare: "full".into(),
            fc_specialization: None,
            fc_values: None,
            stage: None,
            entry: None,
            error: None,
        };

        let (status, observed, tolerance) = compare_candidate_to_metal(
            &candidate,
            &metal,
            &sha256_hex(&candidate),
            metal.output_sha256.as_deref().unwrap_or_default(),
            DataFormat::R16Uint,
            Some(ll),
        );
        assert_eq!(status, "tolerance");
        assert_eq!(observed.and_then(|m| m.max_abs), Some(2.0));
        assert!(tolerance.is_some());

        let (status, _, tolerance) = compare_candidate_to_metal(
            &candidate,
            &metal,
            &sha256_hex(&candidate),
            metal.output_sha256.as_deref().unwrap_or_default(),
            DataFormat::R16Uint,
            Some(&ll.replace("\"no-nans-fp-math\"=\"true\"", "")),
        );
        assert_eq!(status, "failure");
        assert!(tolerance.is_none());
    }

    #[test]
    fn candidate_compare_format_uses_packed_float_buffer_lanes() {
        let ll = r#"
define void @kernel(ptr addrspace(1) %out) {
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @kernel, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 3, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 12, !"air.arg_type_name", !"packed_float3", !"air.arg_name", !"out"}
"#;
        let mut plan = infer_plan("");
        plan.output.kind = "buffer".into();
        plan.output.index = 3;
        plan.output.format = "RawBytes".into();
        let metal = MetalRow {
            air_sha256: "x".into(),
            shard: None,
            label: String::new(),
            status: "ok".into(),
            backend: "metal".into(),
            seed_profile: SEED_PROFILE.into(),
            plan_version: PLAN_VERSION,
            plan: plan.clone(),
            input_sha256: None,
            output_sha256: Some(sha256_hex(&0x7fc00000u32.to_le_bytes())),
            output_b64: Some(encode_output_b64(&0x7fc00000u32.to_le_bytes())),
            spv_sha256: None,
            compare: "full".into(),
            fc_specialization: None,
            fc_values: None,
            stage: None,
            entry: None,
            error: None,
        };

        let format = candidate_compare_format(ll, &plan, &metal);
        assert_eq!(format, DataFormat::Rgba32Float);
    }

    #[test]
    fn matrix_half_buffer_air_type_uses_float_output_lanes() {
        let ll = r#"
define void @kernel(ptr addrspace(1) %out) {
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @kernel, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_name", !"half2x4", !"air.arg_name", !"out"}
"#;
        let plan = infer_plan(ll);
        assert_eq!(plan.output.kind, "buffer");
        assert_eq!(plan.output.index, 1);
        assert_eq!(plan.output.format, "Rgba16Float");
        assert_eq!(
            plan.buffers
                .iter()
                .find(|buffer| buffer.index == 1)
                .map(|buffer| buffer.seed_mode.as_str()),
            Some(SEED_MODE_FINITE_FLOAT16)
        );

        let mut banked = plan.clone();
        banked.output.format = "RawBytes".into();
        let metal = MetalRow {
            air_sha256: "x".into(),
            shard: None,
            label: String::new(),
            status: "ok".into(),
            backend: "metal".into(),
            seed_profile: SEED_PROFILE.into(),
            plan_version: PLAN_VERSION,
            plan: banked,
            input_sha256: None,
            output_sha256: Some(sha256_hex(&0x7e00u16.to_le_bytes())),
            output_b64: Some(encode_output_b64(&0x7e00u16.to_le_bytes())),
            spv_sha256: None,
            compare: "full".into(),
            fc_specialization: None,
            fc_values: None,
            stage: None,
            entry: None,
            error: None,
        };

        assert!(incompatible_output_plan_golden(ll, &metal)
            .unwrap()
            .contains("Rgba16Float"));
        assert_eq!(
            candidate_compare_format(ll, &plan, &metal),
            DataFormat::Rgba16Float
        );
    }

    #[test]
    fn candidate_compare_format_uses_single_field_struct_buffer_lanes() {
        let ll = r#"
define void @kernel(ptr addrspace(1) %out) {
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @kernel, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.struct_type_info", !4, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"triangle_tessellation_factors_t", !"air.arg_name", !"factors"}
!4 = !{i32 0, i32 8, i32 0, !"half4", !"factors"}
"#;
        let plan = infer_plan(ll);
        assert_eq!(plan.output.format, "Rgba16Float");

        let metal = MetalRow {
            air_sha256: "x".into(),
            shard: None,
            label: String::new(),
            status: "ok".into(),
            backend: "metal".into(),
            seed_profile: SEED_PROFILE.into(),
            plan_version: PLAN_VERSION,
            plan: plan.clone(),
            input_sha256: None,
            output_sha256: Some(sha256_hex(&0x7e00u16.to_le_bytes())),
            output_b64: Some(encode_output_b64(&0x7e00u16.to_le_bytes())),
            spv_sha256: None,
            compare: "full".into(),
            fc_specialization: None,
            fc_values: None,
            stage: None,
            entry: None,
            error: None,
        };

        let format = candidate_compare_format(ll, &plan, &metal);
        assert_eq!(format, DataFormat::Rgba16Float);
        let candidate = 0x7fffu16.to_le_bytes();
        let (status, observed, _) = compare_candidate_to_metal(
            &candidate,
            &metal,
            &sha256_hex(&candidate),
            "",
            format,
            None,
        );
        assert_eq!(status, "tolerance");
        assert_eq!(observed.and_then(|m| m.max_ulp), Some(0));
    }

    #[test]
    fn candidate_compare_format_uses_multi_field_float_struct_lanes() {
        let ll = r#"
define void @kernel(ptr addrspace(1) %out) {
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @kernel, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.struct_type_info", !4, !"air.arg_type_size", i32 4240, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"ConvertedColorData", !"air.arg_name", !"out"}
!4 = !{!"air.struct_type_info", !5, i32 0, i32 144, i32 0, !"ConvertedColorMatrices", !"matrices", i32 144, i32 4, i32 1024, !"float", !"rLUT"}
!5 = !{i32 0, i32 48, i32 0, !"float3x3", !"awb_mtx", i32 48, i32 48, i32 0, !"float3x3", !"darkening_mtx", i32 96, i32 48, i32 0, !"float3x3", !"accessibility_mtx"}
"#;
        let plan = infer_plan(ll);
        assert_eq!(plan.output.format, "F32");

        let metal = MetalRow {
            air_sha256: "x".into(),
            shard: None,
            label: String::new(),
            status: "ok".into(),
            backend: "metal".into(),
            seed_profile: SEED_PROFILE.into(),
            plan_version: PLAN_VERSION,
            plan: plan.clone(),
            input_sha256: None,
            output_sha256: Some(sha256_hex(&0x3f800000u32.to_le_bytes())),
            output_b64: Some(encode_output_b64(&0x3f800000u32.to_le_bytes())),
            spv_sha256: None,
            compare: "full".into(),
            fc_specialization: None,
            fc_values: None,
            stage: None,
            entry: None,
            error: None,
        };

        let format = candidate_compare_format(ll, &plan, &metal);
        assert_eq!(format, DataFormat::F32);
        let candidate = 0x3f800001u32.to_le_bytes();
        let (status, observed, _) = compare_candidate_to_metal(
            &candidate,
            &metal,
            &sha256_hex(&candidate),
            "",
            format,
            None,
        );
        assert_eq!(status, "tolerance");
        assert_eq!(observed.and_then(|m| m.max_ulp), Some(1));
    }

    #[test]
    fn candidate_compare_format_uses_half_render_target_lanes() {
        let ll = r#"
define <4 x half> @frag() {
entry:
  ret <4 x half> zeroinitializer
}

!air.fragment = !{!0}
!0 = !{ptr @frag, !1, !2}
!1 = !{!3}
!2 = !{}
!3 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"half4"}
"#;
        let mut plan = infer_plan(ll);
        plan.output.format = "Rgba16Float".into();
        let golden = 0x7e00u16.to_le_bytes();
        let candidate = 0x7fffu16.to_le_bytes();
        let metal = MetalRow {
            air_sha256: "x".into(),
            shard: None,
            label: String::new(),
            status: "ok".into(),
            backend: "metal".into(),
            seed_profile: SEED_PROFILE.into(),
            plan_version: PLAN_VERSION,
            plan: plan.clone(),
            input_sha256: None,
            output_sha256: Some(sha256_hex(&golden)),
            output_b64: Some(encode_output_b64(&golden)),
            spv_sha256: None,
            compare: "full".into(),
            fc_specialization: None,
            fc_values: None,
            stage: None,
            entry: None,
            error: None,
        };

        let format = candidate_compare_format(ll, &plan, &metal);
        assert_eq!(format, DataFormat::Rgba16Float);
        let (status, observed, _) = compare_candidate_to_metal(
            &candidate,
            &metal,
            &sha256_hex(&candidate),
            "",
            format,
            None,
        );
        assert_eq!(status, "tolerance");
        assert_eq!(observed.and_then(|m| m.max_ulp), Some(0));
    }

    #[test]
    fn function_constant_golden_requires_explicit_mode() {
        let ll = r#"
@_Z1x.MTL_FC_INIT_1_b = internal addrspace(2) externally_initialized constant i8 undef, section "air.fc_initializer", align 1

define void @kernel() {
  ret void
}
"#;
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
            compare: "full".into(),
            fc_specialization: None,
            fc_values: None,
            stage: None,
            entry: None,
            error: None,
        };

        assert!(declares_air_function_constants(ll));
        assert!(incompatible_function_constant_golden(ll, &metal)
            .unwrap()
            .contains("lacks explicit function-constant specialization"));
        assert!(incompatible_function_constant_golden(
            "define void @kernel() { ret void }",
            &metal
        )
        .is_none());

        metal.fc_specialization = Some(FC_SPECIALIZATION_ZERO.into());
        assert!(incompatible_function_constant_golden(ll, &metal).is_none());

        metal.fc_specialization = Some(FC_SPECIALIZATION_VALUES.into());
        assert!(incompatible_function_constant_golden(ll, &metal)
            .unwrap()
            .contains("without fc_values"));
        metal.fc_values = Some(vec![FunctionConstantValueJson { index: 1, value: 1 }]);
        assert!(incompatible_function_constant_golden(ll, &metal).is_none());
    }

    #[test]
    fn zero_function_constant_divisor_golden_is_missing() {
        let ll = r#"
@_ZL2fc = internal unnamed_addr addrspace(2) global i16 zeroinitializer, align 2
@_Z2fc.MTL_FC_INIT_4_t = internal unnamed_addr addrspace(2) externally_initialized constant i16 undef, section "air.fc_initializer", align 2

define internal void @_GLOBAL__sub_I_k() section "air.static_init" {
  %1 = load i16, ptr addrspace(2) @_Z2fc.MTL_FC_INIT_4_t, align 2
  store i16 %1, ptr addrspace(2) @_ZL2fc, align 2
  ret void
}

define void @k(i32 %gid) {
  %2 = load i16, ptr addrspace(2) @_ZL2fc, align 2
  %3 = zext i16 %2 to i32
  %4 = sdiv i32 %gid, %3
  ret void
}
"#;
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
            compare: "full".into(),
            fc_specialization: Some(FC_SPECIALIZATION_ZERO.into()),
            fc_values: None,
            stage: None,
            entry: None,
            error: None,
        };

        let reason = incompatible_zero_function_constant_divisor_golden(ll, &metal)
            .expect("zero FC divisor");
        assert!(reason.contains("integer div/rem denominator"), "{reason}");

        metal.fc_specialization = Some(FC_SPECIALIZATION_VALUES.into());
        metal.fc_values = Some(vec![FunctionConstantValueJson { index: 4, value: 7 }]);
        assert!(incompatible_zero_function_constant_divisor_golden(ll, &metal).is_none());
    }

    #[test]
    fn specialized_function_constant_definedness_golden_is_missing() {
        let ll = r#"
@_Z2fc.MTL_FC_INIT_16_b = internal addrspace(2) externally_initialized constant i8 undef, section "air.fc_initializer", align 1

define void @kernel() {
  %defined = tail call i1 @air.is_function_constant_defined(ptr addrspace(2) @_Z2fc.MTL_FC_INIT_16_b)
  ret void
}

declare i1 @air.is_function_constant_defined(ptr addrspace(2))

!air.function_constants = !{!0}
!0 = !{ptr addrspace(2) @_Z2fc.MTL_FC_INIT_16_b, !"bool", !"fc", i32 16, i1 false}
"#;
        let mut metal = metal_row_for_compare(&[], infer_plan(ll), Some("Kernel"));
        metal.fc_specialization = Some(FC_SPECIALIZATION_ZERO.into());

        let reason = incompatible_function_constant_definedness_golden(ll, &metal)
            .expect("definedness specialization golden");
        assert!(
            reason.contains("air.is_function_constant_defined"),
            "{reason}"
        );

        metal.fc_specialization = None;
        assert!(incompatible_function_constant_definedness_golden(ll, &metal).is_none());
    }

    #[test]
    fn zero_function_constant_candidate_specializes_default_values() {
        fn inst(opcode: u16, operands: &[u32]) -> Vec<u32> {
            let mut out = vec![((operands.len() as u32 + 1) << 16) | opcode as u32];
            out.extend_from_slice(operands);
            out
        }

        fn string_words(value: &str) -> Vec<u32> {
            let mut bytes = value.as_bytes().to_vec();
            bytes.push(0);
            while !bytes.len().is_multiple_of(4) {
                bytes.push(0);
            }
            bytes
                .chunks_exact(4)
                .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect()
        }

        let mut words = vec![0x0723_0203, 0x0001_0300, 0, 8, 0];
        let mut name_ops = vec![4];
        name_ops.extend(string_words("_Z7enabled.MTL_FC_INIT_3_b"));
        words.extend(inst(5, &name_ops)); // OpName %4 "...MTL_FC_INIT_3..."
        words.extend(inst(21, &[1, 8, 0])); // %1 = OpTypeInt 8 0
        words.extend(inst(32, &[2, 6, 1])); // %2 = OpTypePointer Private %1
        words.extend(inst(42, &[1, 3])); // %3 = OpConstantNull %1
        words.extend(inst(59, &[2, 4, 6, 3])); // %4 = OpVariable %2 Private %3
        let spv = words
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect::<Vec<_>>();
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
            output_sha256: Some(sha256_hex(&[])),
            output_b64: Some(encode_output_b64(&[])),
            spv_sha256: None,
            compare: "full".into(),
            fc_specialization: Some(FC_SPECIALIZATION_ZERO.into()),
            fc_values: None,
            stage: None,
            entry: None,
            error: None,
        };

        let specialized =
            candidate_spv_for_metal_function_constants(spv, &metal).expect("zero specialize");
        let specialized_words = specialized
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect::<Vec<_>>();

        assert!(
            specialized_words.windows(4).any(|window| {
                window[0] == ((4u32 << 16) | 43) && window[1] == 1 && window[3] == 0
            }),
            "zero specialization should materialize OpConstant %int 0"
        );
    }

    #[test]
    fn stale_output_texture_format_golden_is_missing() {
        let ll = r#"
define void @kernel(ptr addrspace(1) %out, <2 x i16> %gid) {
  tail call void @air.write_texture_2d.i16.v4f16(ptr addrspace(1) %out, <2 x i16> %gid, <4 x half> zeroinitializer, i16 0, i32 2)
  ret void
}

declare void @air.write_texture_2d.i16.v4f16(ptr addrspace(1), <2 x i16>, <4 x half>, i16, i32)

!air.kernel = !{!0}
!0 = !{ptr @kernel, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 2, i32 1, !"air.write", !"air.arg_type_name", !"texture2d<half, write>", !"air.arg_name", !"out"}
!4 = !{i32 1, !"air.thread_position_in_grid", !"air.arg_type_name", !"ushort2", !"air.arg_name", !"gid"}
"#;
        let mut plan = infer_plan(ll);
        assert_eq!(plan.output.kind, "texture");
        assert_eq!(plan.output.format, "Rgba16Float");
        plan.output.format = "Rgba32Float".into();
        let metal = MetalRow {
            air_sha256: "x".into(),
            shard: None,
            label: String::new(),
            status: "ok".into(),
            backend: "metal".into(),
            seed_profile: SEED_PROFILE.into(),
            plan_version: PLAN_VERSION,
            plan,
            input_sha256: None,
            output_sha256: Some(sha256_hex(&[])),
            output_b64: Some(encode_output_b64(&[])),
            spv_sha256: None,
            compare: "full".into(),
            fc_specialization: None,
            fc_values: None,
            stage: None,
            entry: None,
            error: None,
        };

        let reason = incompatible_output_plan_golden(ll, &metal).expect("stale output plan");
        assert!(
            reason.contains("rebank Metal row") && reason.contains("Rgba32Float"),
            "{reason}"
        );
    }

    #[test]
    fn undef_lane_texture_write_golden_requires_rebank() {
        let ll = r#"
define void @kernel(ptr addrspace(1) %out, <2 x i32> %gid) {
  %texel = insertelement <4 x i16> undef, i16 7, i64 0
  tail call void @air.write_texture_2d.u.v4i16(ptr addrspace(1) %out, <2 x i32> %gid, <4 x i16> %texel, i32 0, i32 2)
  ret void
}

declare void @air.write_texture_2d.u.v4i16(ptr addrspace(1), <2 x i32>, <4 x i16>, i32, i32)

!air.kernel = !{!0}
!0 = !{ptr @kernel, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.write", !"air.arg_type_name", !"texture2d<ushort, write>", !"air.arg_name", !"out"}
!4 = !{i32 1, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint2", !"air.arg_name", !"gid"}
"#;
        let metal = MetalRow {
            air_sha256: "x".into(),
            shard: None,
            label: String::new(),
            status: "ok".into(),
            backend: "metal".into(),
            seed_profile: SEED_PROFILE.into(),
            plan_version: PLAN_VERSION,
            plan: infer_plan(ll),
            input_sha256: None,
            output_sha256: Some(sha256_hex(&[])),
            output_b64: Some(encode_output_b64(&[])),
            spv_sha256: None,
            compare: "full".into(),
            fc_specialization: None,
            fc_values: None,
            stage: None,
            entry: None,
            error: None,
        };

        let reason = incompatible_undefined_texture_write_lanes_golden(ll, &metal)
            .expect("undef texture lanes should require rebank");
        assert!(reason.contains("rebank Metal row"), "{reason}");
    }

    #[test]
    fn point_coord_golden_requires_topology_aware_rebank() {
        let ll = r#"
define <4 x float> @fragment(<2 x float> %coord) {
  %x = extractelement <2 x float> %coord, i64 0
  %v0 = insertelement <4 x float> poison, float %x, i64 0
  ret <4 x float> %v0
}

!air.fragment = !{!0}
!0 = !{ptr @fragment, !1, !2}
!1 = !{!3}
!2 = !{!4}
!3 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!4 = !{i32 0, !"air.point_coord", !"air.arg_type_name", !"float2", !"air.arg_name", !"coord"}
"#;
        let mut metal = MetalRow {
            air_sha256: "x".into(),
            shard: None,
            label: String::new(),
            status: "ok".into(),
            backend: "metal".into(),
            seed_profile: SEED_PROFILE.into(),
            plan_version: PLAN_VERSION,
            plan: infer_plan(ll),
            input_sha256: None,
            output_sha256: Some(sha256_hex(&[])),
            output_b64: Some(encode_output_b64(&[])),
            spv_sha256: None,
            compare: "full".into(),
            fc_specialization: None,
            fc_values: None,
            stage: None,
            entry: None,
            error: None,
        };

        assert!(incompatible_point_coord_golden(ll, &metal)
            .unwrap()
            .contains("topology-aware plan"));
        assert!(incompatible_point_coord_golden(
            &ll.replace("\"air.point_coord\"", "\"air.fragment_input\""),
            &metal,
        )
        .is_none());
        metal.plan_version = POINT_COORD_TOPOLOGY_PLAN_VERSION;
        assert!(incompatible_point_coord_golden(ll, &metal).is_none());
    }

    #[test]
    fn deterministic_float_seed_golden_with_nonfinite_input_is_missing() {
        let ll = r#"
define void @kernel(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
  %v = load float, ptr addrspace(1) %in, align 4
  %p = fmul fast float %v, 0.000000e+00
  store float %p, ptr addrspace(1) %out, align 4
  ret void
}

attributes #0 = { "no-nans-fp-math"="true" }

!air.kernel = !{!0}
!0 = !{ptr @kernel, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"out"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 3, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"in"}
"#;
        let mut old_plan = infer_plan(ll);
        let banked = old_plan.buffers.iter_mut().find(|b| b.index == 3).unwrap();
        banked.len = 256;
        banked.seed_mode = SEED_MODE_DETERMINISTIC.into();
        let raw = seeded_buffer_bytes(&BufferInput {
            index: banked.index,
            len: banked.len,
            role: BufferRole::Input,
            seed: Seed::Deterministic {
                tag: banked.seed_tag,
            },
        });
        assert!(contains_nonfinite_float_lane(&raw, 4));

        let metal = MetalRow {
            air_sha256: "x".into(),
            shard: None,
            label: String::new(),
            status: "ok".into(),
            backend: "metal".into(),
            seed_profile: "deterministic_v2_bounded_control".into(),
            plan_version: PLAN_VERSION,
            plan: old_plan,
            input_sha256: None,
            output_sha256: Some(sha256_hex(&0.0f32.to_le_bytes())),
            output_b64: Some(encode_output_b64(&0.0f32.to_le_bytes())),
            spv_sha256: None,
            compare: "full".into(),
            fc_specialization: None,
            fc_values: None,
            stage: None,
            entry: None,
            error: None,
        };

        assert!(incompatible_float_seed_golden(ll, &metal)
            .unwrap()
            .contains("deterministic f32 buffer 3"));
        assert!(incompatible_float_seed_golden(
            &ll.replace("fmul fast", "fmul")
                .replace("\"no-nans-fp-math\"=\"true\"", ""),
            &metal,
        )
        .is_none());
    }

    #[test]
    fn deterministic_bfloat_seed_golden_with_nonfinite_inout_is_missing() {
        let ll = r#"
define void @kernel(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
  %old = load bfloat, ptr addrspace(1) %out, align 2
  %v = load bfloat, ptr addrspace(1) %in, align 2
  %p = fmul fast bfloat %old, %v
  store bfloat %p, ptr addrspace(1) %out, align 2
  ret void
}

attributes #0 = { "no-nans-fp-math"="true" }

!air.kernel = !{!0}
!0 = !{ptr @kernel, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 3, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 2, !"air.arg_type_name", !"bfloat", !"air.arg_name", !"out"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 4, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 2, !"air.arg_type_name", !"bfloat", !"air.arg_name", !"in"}
"#;
        let mut old_plan = infer_plan(ll);
        let banked = old_plan.buffers.iter_mut().find(|b| b.index == 3).unwrap();
        banked.len = 256;
        banked.seed_mode = SEED_MODE_DETERMINISTIC.into();
        let raw = seeded_buffer_bytes(&BufferInput {
            index: banked.index,
            len: banked.len,
            role: BufferRole::InOut,
            seed: Seed::Deterministic {
                tag: banked.seed_tag,
            },
        });
        assert!(contains_nonfinite_bfloat_lane(&raw));

        let metal = MetalRow {
            air_sha256: "x".into(),
            shard: None,
            label: String::new(),
            status: "ok".into(),
            backend: "metal".into(),
            seed_profile: "deterministic_v2_bounded_control".into(),
            plan_version: PLAN_VERSION,
            plan: old_plan,
            input_sha256: None,
            output_sha256: Some(sha256_hex(&0u16.to_le_bytes())),
            output_b64: Some(encode_output_b64(&0u16.to_le_bytes())),
            spv_sha256: None,
            compare: "full".into(),
            fc_specialization: None,
            fc_values: None,
            stage: None,
            entry: None,
            error: None,
        };

        assert!(incompatible_float_seed_golden(ll, &metal)
            .unwrap()
            .contains("deterministic bf16 buffer 3"));
    }

    #[test]
    fn deterministic_float_texture_seed_golden_with_nonfinite_input_is_missing() {
        let ll = r#"
define <4 x half> @fragment(ptr addrspace(1) %tex, ptr addrspace(2) %sampler, <2 x float> %uv) #0 {
  %s = call { <4 x half>, i8 } @air.sample_texture_2d.v4f16(ptr addrspace(1) %tex, ptr addrspace(2) %sampler, <2 x float> %uv, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.0, float 0.0, i32 0)
  %c = extractvalue { <4 x half>, i8 } %s, 0
  %p = fmul fast <4 x half> %c, %c
  ret <4 x half> %p
}

declare { <4 x half>, i8 } @air.sample_texture_2d.v4f16(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32)

attributes #0 = { "no-nans-fp-math"="true" }

!air.fragment = !{!0}
!0 = !{ptr @fragment, !1, !2}
!1 = !{!3}
!2 = !{!4, !5, !6}
!3 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"half4"}
!4 = !{i32 0, !"air.texture", !"air.location_index", i32 2, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<half, sample>", !"air.arg_name", !"tex"}
!5 = !{i32 1, !"air.sampler", !"air.location_index", i32 2, i32 1, !"air.arg_type_name", !"sampler", !"air.arg_name", !"sampler"}
!6 = !{i32 2, !"air.fragment_input", !"air.arg_type_name", !"float2", !"air.arg_name", !"uv"}
"#;
        let mut old_plan = infer_plan(ll);
        let banked = old_plan.textures.iter_mut().find(|t| t.index == 2).unwrap();
        banked.seed_mode = SEED_MODE_DETERMINISTIC.into();
        let raw = seeded_texture_bytes(&TextureInput {
            index: banked.index,
            format: DataFormat::Rgba16Float,
            extent: Extent3d::new(banked.w, banked.h, banked.d),
            role: TextureRole::Sampled,
            seed: Seed::Deterministic {
                tag: banked.seed_tag,
            },
        });
        assert!(contains_nonfinite_float_lane(&raw, 2));

        let metal = MetalRow {
            air_sha256: "x".into(),
            shard: None,
            label: String::new(),
            status: "ok".into(),
            backend: "metal".into(),
            seed_profile: "deterministic_v3_finite_float_buffers".into(),
            plan_version: PLAN_VERSION,
            plan: old_plan,
            input_sha256: None,
            output_sha256: Some(sha256_hex(&0.0f32.to_le_bytes())),
            output_b64: Some(encode_output_b64(&0.0f32.to_le_bytes())),
            spv_sha256: None,
            compare: "full".into(),
            fc_specialization: None,
            fc_values: None,
            stage: None,
            entry: None,
            error: None,
        };

        assert!(incompatible_float_seed_golden(ll, &metal)
            .unwrap()
            .contains("deterministic f16 texture 2"));
    }

    #[test]
    fn deterministic_float_render_target_seed_golden_is_missing() {
        let ll = r#"
define <4 x half> @fragment(<4 x half> %color0) #0 {
  %bits = bitcast <4 x half> %color0 to <4 x i16>
  %half = bitcast <4 x i16> %bits to <4 x half>
  ret <4 x half> %half
}

attributes #0 = { "no-nans-fp-math"="true" }

!air.fragment = !{!0}
!air.compile_options = !{!5}
!0 = !{ptr @fragment, !1, !3}
!1 = !{!2}
!2 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"half4"}
!3 = !{!4}
!4 = !{i32 0, !"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"half4", !"air.arg_name", !"color0"}
!5 = !{!"air.compile.framebuffer_fetch_enable"}
"#;
        let metal = MetalRow {
            air_sha256: "x".into(),
            shard: None,
            label: String::new(),
            status: "ok".into(),
            backend: "metal".into(),
            seed_profile: "deterministic_v5_typed_bounded_control".into(),
            plan_version: PLAN_VERSION,
            plan: infer_plan(ll),
            input_sha256: None,
            output_sha256: Some(sha256_hex(&0u16.to_le_bytes())),
            output_b64: Some(encode_output_b64(&0u16.to_le_bytes())),
            spv_sha256: None,
            compare: "full".into(),
            fc_specialization: None,
            fc_values: None,
            stage: None,
            entry: None,
            error: None,
        };

        assert!(incompatible_float_seed_golden(ll, &metal)
            .unwrap()
            .contains("deterministic f16 render target seed"));
    }

    #[test]
    fn deterministic_float_render_target_seed_leak_in_golden_is_missing() {
        let ll = r#"
define <4 x half> @fragment() #0 {
  ret <4 x half> zeroinitializer
}

attributes #0 = { "no-nans-fp-math"="true" }

!air.fragment = !{!0}
!0 = !{ptr @fragment, !1, !3}
!1 = !{!2}
!2 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"half4"}
!3 = !{}
"#;
        let plan = infer_plan(ll);
        let raw = seeded_texture_bytes(&TextureInput {
            index: plan.output.index,
            format: DataFormat::Rgba16Float,
            extent: Extent3d::new(
                plan.output.w.unwrap(),
                plan.output.h.unwrap(),
                plan.output.d.unwrap(),
            ),
            role: TextureRole::ColorTarget,
            seed: Seed::Deterministic {
                tag: RENDER_TARGET_SEED_TAG,
            },
        });
        let metal = MetalRow {
            air_sha256: "x".into(),
            shard: None,
            label: String::new(),
            status: "ok".into(),
            backend: "metal".into(),
            seed_profile: "deterministic_v5_typed_bounded_control".into(),
            plan_version: PLAN_VERSION,
            plan,
            input_sha256: None,
            output_sha256: Some(sha256_hex(&raw)),
            output_b64: Some(encode_output_b64(&raw)),
            spv_sha256: None,
            compare: "full".into(),
            fc_specialization: None,
            fc_values: None,
            stage: None,
            entry: None,
            error: None,
        };

        assert!(incompatible_float_seed_golden(ll, &metal)
            .unwrap()
            .contains("render target seed bytes preserved in golden output"));
    }

    #[test]
    fn deterministic_float_texture_seed_golden_with_huge_finite_input_is_missing() {
        let ll = r#"
define <4 x half> @fragment(ptr addrspace(1) %tex, ptr addrspace(2) %sampler, <2 x float> %uv) #0 {
  %s = call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) %tex, ptr addrspace(2) %sampler, <2 x float> %uv, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.0, float 0.0, i32 0)
  %c = extractvalue { <4 x float>, i8 } %s, 0
  %p = tail call fast <4 x half> @air.convert.f.v4f16.f.v4f32(<4 x float> %c)
  ret <4 x half> %p
}

declare { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32)
declare <4 x half> @air.convert.f.v4f16.f.v4f32(<4 x float>)

attributes #0 = { "no-nans-fp-math"="true" }

!air.fragment = !{!0}
!air.compile_options = !{!7}
!0 = !{ptr @fragment, !1, !2}
!1 = !{!3}
!2 = !{!4, !5, !6}
!3 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"half4"}
!4 = !{i32 0, !"air.texture", !"air.location_index", i32 2, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"tex"}
!5 = !{i32 1, !"air.sampler", !"air.location_index", i32 2, i32 1, !"air.arg_type_name", !"sampler", !"air.arg_name", !"sampler"}
!6 = !{i32 2, !"air.fragment_input", !"air.arg_type_name", !"float2", !"air.arg_name", !"uv"}
!7 = !{!"air.compile.fast_math_enable"}
"#;
        let mut old_plan = infer_plan(ll);
        let banked = old_plan.textures.iter_mut().find(|t| t.index == 2).unwrap();
        banked.seed_mode = SEED_MODE_DETERMINISTIC.into();
        banked.w = 1;
        banked.h = 1;
        banked.d = 1;
        banked.seed_tag = (1..10_000)
            .find(|tag| {
                let input = TextureInput {
                    index: banked.index,
                    format: DataFormat::Rgba32Float,
                    extent: Extent3d::new(1, 1, 1),
                    role: TextureRole::Sampled,
                    seed: Seed::Deterministic { tag: *tag },
                };
                let raw = seeded_texture_bytes(&input);
                let finite = seeded_texture_bytes(&TextureInput {
                    seed: Seed::DeterministicFinite { tag: *tag },
                    ..input
                });
                raw != finite && !contains_nonfinite_float_lane(&raw, 4)
            })
            .expect("deterministic seed with huge finite f32 lane");

        let raw = seeded_texture_bytes(&TextureInput {
            index: banked.index,
            format: DataFormat::Rgba32Float,
            extent: Extent3d::new(1, 1, 1),
            role: TextureRole::Sampled,
            seed: Seed::Deterministic {
                tag: banked.seed_tag,
            },
        });
        assert!(!contains_nonfinite_float_lane(&raw, 4));

        let metal = MetalRow {
            air_sha256: "x".into(),
            shard: None,
            label: String::new(),
            status: "ok".into(),
            backend: "metal".into(),
            seed_profile: "deterministic_v3_finite_float_buffers".into(),
            plan_version: PLAN_VERSION,
            plan: old_plan,
            input_sha256: None,
            output_sha256: Some(sha256_hex(&0.0f32.to_le_bytes())),
            output_b64: Some(encode_output_b64(&0.0f32.to_le_bytes())),
            spv_sha256: None,
            compare: "full".into(),
            fc_specialization: None,
            fc_values: None,
            stage: None,
            entry: None,
            error: None,
        };

        assert!(incompatible_float_seed_golden(ll, &metal)
            .unwrap()
            .contains("deterministic f32 texture 2"));
    }

    #[test]
    fn sampled_fast_pow_f32_texture_golden_is_missing() {
        let ll = r#"
define <4 x half> @frag(ptr addrspace(1) %tex) #0 {
  %s = tail call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) %tex, ptr addrspace(2) null, <2 x float> zeroinitializer, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %v = extractvalue { <4 x float>, i8 } %s, 0
  %rgb = shufflevector <4 x float> %v, <4 x float> poison, <3 x i32> <i32 0, i32 1, i32 2>
  %pow = tail call fast <3 x float> @air.fast_pow.v3f32(<3 x float> %rgb, <3 x float> splat (float 0x40019999A0000000))
  %out3 = shufflevector <3 x float> %pow, <3 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 poison>
  %out4 = insertelement <4 x float> %out3, float 1.000000e+00, i64 3
  %out = tail call <4 x half> @air.convert.f.v4f16.f.v4f32(<4 x float> %out4)
  ret <4 x half> %out
}

declare { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32)
declare <3 x float> @air.fast_pow.v3f32(<3 x float>, <3 x float>)
declare <4 x half> @air.convert.f.v4f16.f.v4f32(<4 x float>)
attributes #0 = { "no-nans-fp-math"="true" "unsafe-fp-math"="true" }

!air.fragment = !{!0}
!0 = !{ptr @frag, !1, !2}
!1 = !{!3}
!2 = !{!4}
!3 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"half4"}
!4 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"tex"}
"#;

        let metal = metal_row_for_compare(&[], infer_plan(ll), Some("Fragment"));
        let reason = incompatible_sampled_fast_pow_texture_golden(ll, &metal)
            .expect("sampled fast pow texture golden");
        assert!(reason.contains("fast_pow"), "{reason}");
    }

    #[test]
    fn dependent_sampled_lookup_golden_is_missing() {
        let ll = r#"
define <4 x float> @frag(ptr addrspace(1) %src, ptr addrspace(1) %lut, <2 x float> %uv) #0 {
  %s = tail call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) %src, ptr addrspace(2) null, <2 x float> %uv, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %v = extractvalue { <4 x float>, i8 } %s, 0
  %z = extractelement <4 x float> %v, i64 2
  %scaled = fmul fast float %z, 6.300000e+01
  %ceil = tail call fast float @air.fast_ceil.f32(float %scaled)
  %row = fmul fast float %ceil, 1.250000e-01
  %floor = tail call fast float @air.fast_floor.f32(float %row)
  %x = insertelement <2 x float> poison, float %floor, i64 0
  %coord = insertelement <2 x float> %x, float %floor, i64 1
  %l = tail call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) %lut, ptr addrspace(2) null, <2 x float> %coord, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %out = extractvalue { <4 x float>, i8 } %l, 0
  ret <4 x float> %out
}

declare { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32)
declare float @air.fast_ceil.f32(float)
declare float @air.fast_floor.f32(float)
attributes #0 = { "no-nans-fp-math"="true" "unsafe-fp-math"="true" }

!air.fragment = !{!0}
!0 = !{ptr @frag, !1, !2}
!1 = !{!3}
!2 = !{!4, !5, !6}
!3 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!4 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"src"}
!5 = !{i32 1, !"air.texture", !"air.location_index", i32 1, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"lut"}
!6 = !{i32 2, !"air.fragment_input", !"air.arg_type_name", !"float2", !"air.arg_name", !"uv"}
"#;

        let metal = metal_row_for_compare(&[], infer_plan(ll), Some("Fragment"));
        let reason = incompatible_dependent_sampled_lookup_golden(ll, &metal)
            .expect("dependent sampled lookup golden");
        assert!(reason.contains("dependent texture lookup"), "{reason}");
    }

    #[test]
    fn dependent_sampled_half_lookup_golden_is_missing() {
        let ll = r#"
define <4 x half> @frag(ptr addrspace(1) %src, ptr addrspace(1) %lut, <2 x float> %uv) #0 {
  %s = tail call { <4 x half>, i8 } @air.sample_texture_2d.v4f16(ptr addrspace(1) %src, ptr addrspace(2) null, <2 x float> %uv, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %v = extractvalue { <4 x half>, i8 } %s, 0
  %x_h = extractelement <4 x half> %v, i64 0
  %x = fpext half %x_h to float
  %floor = tail call fast float @air.fast_floor.f32(float %x)
  %coord = insertelement <2 x float> %uv, float %floor, i64 0
  %l = tail call { <4 x half>, i8 } @air.sample_texture_2d.v4f16(ptr addrspace(1) %lut, ptr addrspace(2) null, <2 x float> %coord, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %out = extractvalue { <4 x half>, i8 } %l, 0
  ret <4 x half> %out
}

declare { <4 x half>, i8 } @air.sample_texture_2d.v4f16(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32)
declare float @air.fast_floor.f32(float)
attributes #0 = { "no-nans-fp-math"="true" "unsafe-fp-math"="true" }

!air.fragment = !{!0}
!0 = !{ptr @frag, !1, !2}
!1 = !{!3}
!2 = !{!4, !5, !6}
!3 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"half4"}
!4 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<half, sample>", !"air.arg_name", !"src"}
!5 = !{i32 1, !"air.texture", !"air.location_index", i32 1, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<half, sample>", !"air.arg_name", !"lut"}
!6 = !{i32 2, !"air.fragment_input", !"air.arg_type_name", !"float2", !"air.arg_name", !"uv"}
"#;

        let metal = metal_row_for_compare(&[], infer_plan(ll), Some("Fragment"));
        let reason = incompatible_dependent_sampled_half_lookup_golden(ll, &metal)
            .expect("dependent sampled half lookup golden");
        assert!(reason.contains("finite f16"), "{reason}");
    }

    #[test]
    fn sampled_half_fast_sqrt_render_target_golden_is_missing() {
        let ll = r#"
define <4 x half> @frag(ptr addrspace(1) %tex, <2 x float> %uv) #0 {
  %d = tail call fast float @air.dot.v2f32(<2 x float> %uv, <2 x float> %uv)
  %r = tail call fast float @air.fast_sqrt.f32(float %d)
  %coord = insertelement <2 x float> %uv, float %r, i64 0
  %s = tail call { <4 x half>, i8 } @air.sample_texture_2d.v4f16(ptr addrspace(1) %tex, ptr addrspace(2) null, <2 x float> %coord, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %v = extractvalue { <4 x half>, i8 } %s, 0
  ret <4 x half> %v
}

declare float @air.dot.v2f32(<2 x float>, <2 x float>)
declare float @air.fast_sqrt.f32(float)
declare { <4 x half>, i8 } @air.sample_texture_2d.v4f16(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32)
attributes #0 = { "no-nans-fp-math"="true" "unsafe-fp-math"="true" }

!air.fragment = !{!0}
!0 = !{ptr @frag, !1, !2}
!1 = !{!3}
!2 = !{!4, !5}
!3 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"half4"}
!4 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<half, sample>", !"air.arg_name", !"tex"}
!5 = !{i32 1, !"air.fragment_input", !"air.arg_type_name", !"float2", !"air.arg_name", !"uv"}
"#;

        let metal = metal_row_for_compare(&[], infer_plan(ll), Some("Fragment"));
        let reason = incompatible_sampled_half_fast_sqrt_render_target_golden(ll, &metal)
            .expect("sampled half fast sqrt render-target golden");
        assert!(reason.contains("fast_sqrt"), "{reason}");
    }

    #[test]
    fn sampled_half_exact_control_flow_golden_is_missing() {
        let ll = r#"
define <4 x half> @frag(ptr addrspace(1) %tex, <2 x float> %uv) #0 {
  %s = tail call { <4 x half>, i8 } @air.sample_texture_2d.v4f16(ptr addrspace(1) %tex, ptr addrspace(2) null, <2 x float> %uv, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %v = extractvalue { <4 x half>, i8 } %s, 0
  %x = extractelement <4 x half> %v, i64 0
  %is_zero = fcmp fast oeq half %x, 0xH0000
  br i1 %is_zero, label %zero, label %sampled

zero:
  ret <4 x half> zeroinitializer

sampled:
  ret <4 x half> %v
}

declare { <4 x half>, i8 } @air.sample_texture_2d.v4f16(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32)
attributes #0 = { "no-nans-fp-math"="true" "unsafe-fp-math"="true" }

!air.fragment = !{!0}
!0 = !{ptr @frag, !1, !2}
!1 = !{!3}
!2 = !{!4, !5}
!3 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"half4"}
!4 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<half, sample>", !"air.arg_name", !"tex"}
!5 = !{i32 1, !"air.fragment_input", !"air.arg_type_name", !"float2", !"air.arg_name", !"uv"}
"#;

        let metal = metal_row_for_compare(&[], infer_plan(ll), Some("Fragment"));
        let reason = incompatible_sampled_half_exact_control_flow_golden(ll, &metal)
            .expect("sampled half exact control flow golden");
        assert!(reason.contains("exact predicates"), "{reason}");
    }

    #[test]
    fn sampled_half_cube_fast_math_golden_is_missing() {
        let ll = r#"
define <4 x half> @frag(ptr addrspace(1) %cube, <3 x float> %normal) #0 {
  %d = tail call fast float @air.dot.v3f32(<3 x float> %normal, <3 x float> %normal)
  %r = tail call fast float @air.fast_rsqrt.f32(float %d)
  %splat = insertelement <3 x float> poison, float %r, i64 0
  %scale = shufflevector <3 x float> %splat, <3 x float> poison, <3 x i32> zeroinitializer
  %coord = fmul fast <3 x float> %scale, %normal
  %sample = tail call { <4 x half>, i8 } @air.sample_texture_cube.v4f16(ptr addrspace(1) %cube, ptr addrspace(2) null, <3 x float> %coord, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %out = extractvalue { <4 x half>, i8 } %sample, 0
  ret <4 x half> %out
}

declare float @air.dot.v3f32(<3 x float>, <3 x float>)
declare float @air.fast_rsqrt.f32(float)
declare { <4 x half>, i8 } @air.sample_texture_cube.v4f16(ptr addrspace(1), ptr addrspace(2), <3 x float>, i1, float, float, i32)
attributes #0 = { "no-nans-fp-math"="true" "unsafe-fp-math"="true" }

!air.fragment = !{!0}
!0 = !{ptr @frag, !1, !2}
!1 = !{!3}
!2 = !{!4, !5}
!3 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"half4"}
!4 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texturecube<half, sample>", !"air.arg_name", !"cube"}
!5 = !{i32 1, !"air.fragment_input", !"air.arg_type_name", !"float3", !"air.arg_name", !"normal"}
"#;

        let metal = metal_row_for_compare(&[], infer_plan(ll), Some("Fragment"));
        let reason = incompatible_sampled_half_cube_fast_math_golden(ll, &metal)
            .expect("sampled half cube fast math golden");
        assert!(reason.contains("f16 cube texture"), "{reason}");
    }

    #[test]
    fn sampled_half_buffer_fast_math_golden_is_missing() {
        let ll = r#"
define void @kernel(ptr addrspace(1) %uv_tex, ptr addrspace(1) %half_tex, ptr addrspace(1) %read_tex, ptr addrspace(1) %out) #0 {
  %uv_sample = tail call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) %uv_tex, ptr addrspace(2) null, <2 x float> zeroinitializer, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %uv_color = extractvalue { <4 x float>, i8 } %uv_sample, 0
  %uv = shufflevector <4 x float> %uv_color, <4 x float> poison, <2 x i32> <i32 0, i32 1>
  %sample = tail call { <4 x half>, i8 } @air.sample_texture_2d.v4f16(ptr addrspace(1) %half_tex, ptr addrspace(2) null, <2 x float> %uv, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %sample_color = extractvalue { <4 x half>, i8 } %sample, 0
  %read = tail call { <4 x half>, i8 } @air.read_texture_2d.i16.v4f16(ptr addrspace(1) %read_tex, ptr addrspace(2) null, <2 x i16> zeroinitializer, <2 x i16> zeroinitializer, i16 0, i32 1)
  %read_color = extractvalue { <4 x half>, i8 } %read, 0
  %sum = fadd fast <4 x half> %sample_color, %read_color
  %wide = tail call <4 x float> @air.convert.f.v4f32.f.v4f16(<4 x half> %sum)
  %xy = shufflevector <4 x float> %wide, <4 x float> poison, <2 x i32> <i32 0, i32 1>
  %pow = tail call fast <2 x float> @air.fast_pow.v2f32(<2 x float> %xy, <2 x float> splat (float 1.250000e+00))
  %rsqrt = tail call fast <2 x float> @air.fast_rsqrt.v2f32(<2 x float> %pow)
  %half = tail call <2 x half> @air.convert.f.v2f16.f.v2f32(<2 x float> %rsqrt)
  store <2 x half> %half, ptr addrspace(1) %out, align 4
  ret void
}

declare { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32)
declare { <4 x half>, i8 } @air.sample_texture_2d.v4f16(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32)
declare { <4 x half>, i8 } @air.read_texture_2d.i16.v4f16(ptr addrspace(1), ptr addrspace(2), <2 x i16>, <2 x i16>, i16, i32)
declare <4 x float> @air.convert.f.v4f32.f.v4f16(<4 x half>)
declare <2 x float> @air.fast_pow.v2f32(<2 x float>, <2 x float>)
declare <2 x float> @air.fast_rsqrt.v2f32(<2 x float>)
declare <2 x half> @air.convert.f.v2f16.f.v2f32(<2 x float>)
attributes #0 = { "unsafe-fp-math"="true" }

!air.kernel = !{!0}
!air.compile_options = !{!9}
!0 = !{ptr @kernel, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5, !6}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"uv_tex"}
!4 = !{i32 1, !"air.texture", !"air.location_index", i32 1, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<half, sample>", !"air.arg_name", !"half_tex"}
!5 = !{i32 2, !"air.texture", !"air.location_index", i32 2, i32 1, !"air.read", !"air.arg_type_name", !"texture2d<half, read>", !"air.arg_name", !"read_tex"}
!6 = !{i32 3, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"half2", !"air.arg_name", !"out"}
!9 = !{!"air.compile.fast_math_enable"}
"#;

        let metal = metal_row_for_compare(&[], infer_plan(ll), Some("Kernel"));
        let reason = incompatible_sampled_half_buffer_fast_math_golden(ll, &metal)
            .expect("sampled half buffer fast math golden");
        assert!(reason.contains("half buffer output"), "{reason}");
        assert!(reason.contains("fast_pow/fast_rsqrt"), "{reason}");
    }

    #[test]
    fn fast_no_nans_golden_with_nonfinite_float_output_is_missing() {
        let ll = r#"
define void @kernel(<3 x i32> %tid, ptr addrspace(1) %out) #0 {
  %xy = shufflevector <3 x i32> %tid, <3 x i32> poison, <2 x i32> <i32 0, i32 1>
  call void @air.write_texture_2d.v4f32(ptr addrspace(1) %out, <2 x i32> %xy, <4 x float> <float 0x7FF8000000000000, float 0.000000e+00, float 1.000000e+00, float 1.000000e+00>, i32 0, i32 2)
  ret void
}

declare void @air.write_texture_2d.v4f32(ptr addrspace(1), <2 x i32>, <4 x float>, i32, i32)

attributes #0 = { "no-nans-fp-math"="true" }

!air.kernel = !{!0}
!0 = !{ptr @kernel, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint3", !"air.arg_name", !"tid"}
!4 = !{i32 1, !"air.texture", !"air.location_index", i32 5, i32 1, !"air.write", !"air.arg_type_name", !"texture2d<float, write>", !"air.arg_name", !"out"}
"#;
        let plan = infer_plan(ll);
        assert_eq!(plan.output.kind, "texture");
        assert_eq!(plan.output.index, 5);
        assert_eq!(plan.output.format, "R32Float");
        let golden = [
            f32::NAN.to_le_bytes(),
            0.0f32.to_le_bytes(),
            1.0f32.to_le_bytes(),
            1.0f32.to_le_bytes(),
        ]
        .concat();
        let metal = MetalRow {
            air_sha256: "x".into(),
            shard: None,
            label: String::new(),
            status: "ok".into(),
            backend: "metal".into(),
            seed_profile: "deterministic_v3_finite_float_buffers".into(),
            plan_version: PLAN_VERSION,
            plan,
            input_sha256: None,
            output_sha256: Some(sha256_hex(&golden)),
            output_b64: Some(encode_output_b64(&golden)),
            spv_sha256: None,
            compare: "full".into(),
            fc_specialization: None,
            fc_values: None,
            stage: None,
            entry: None,
            error: None,
        };

        assert!(incompatible_float_output_golden(ll, &metal)
            .unwrap()
            .contains("output texture 5 contains f32 NaN/Inf"));
        assert!(incompatible_float_output_golden(
            &ll.replace("\"no-nans-fp-math\"=\"true\"", ""),
            &metal,
        )
        .is_none());
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
            fc_specialization: None,
            fc_values: None,
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
            None,
        );
        assert_eq!(status, "failure");
        assert!(candidate_compare_error(&status, &metal, tolerance.as_ref()).is_none());
        assert!(observed.is_none());
        assert!(tolerance.is_none());
    }

    #[test]
    fn execution_success_accepts_candidate_tolerance() {
        assert!(execution_status_is_success(RunBackend::Metal, "ok"));
        assert!(!execution_status_is_success(RunBackend::Metal, "tolerance"));
        assert!(!execution_status_is_success(RunBackend::Metal, "smoke"));
        assert!(execution_status_is_success(RunBackend::Vulkan, "ok"));
        assert!(execution_status_is_success(RunBackend::Vulkan, "tolerance"));
        assert!(execution_status_is_success(RunBackend::Vulkan, "smoke"));
        assert!(execution_status_is_success(
            RunBackend::MoltenVk,
            "tolerance"
        ));
        assert!(execution_status_is_success(RunBackend::MoltenVk, "smoke"));
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
    fn candidate_execution_driver_failures_are_quarantined() {
        for error in [
            "Vulkan validation runner skipped NVIDIA compute pipeline compiler crash",
            "Vulkan validation runner skipped NVIDIA graphics pipeline compiler crash",
            "graphics pipeline probe timed out after 15s",
            "wait for compute completion: the logical or physical device has been lost",
            "wait for vertex validation completion: the logical or physical device has been lost",
        ] {
            assert_eq!(candidate_execution_error_status(error), "quarantine");
        }
        assert_eq!(
            candidate_execution_error_status("create descriptor set: unsupported format"),
            "fallback"
        );
    }

    #[test]
    fn worker_exit_error_records_signal_but_not_regular_failure() {
        #[cfg(unix)]
        {
            use std::os::unix::process::ExitStatusExt;

            let signal = std::process::ExitStatus::from_raw(11);
            assert_eq!(
                worker_exit_error(signal).as_deref(),
                Some("worker terminated by signal 11")
            );

            let failure = std::process::ExitStatus::from_raw(1 << 8);
            assert_eq!(worker_exit_error(failure), None);

            let unusual = std::process::ExitStatus::from_raw(3 << 8);
            assert_eq!(
                worker_exit_error(unusual).as_deref(),
                Some("worker exited with status 3")
            );
        }
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
            fc_specialization: None,
            fc_values: None,
            stage: None,
            entry: None,
            error: None,
        };

        match candidate_ll_for_metal_compare(ll, "spin", &metal).unwrap() {
            Cow::Owned(text) => assert!(text.contains("m2v.g.0:"), "{text}"),
            Cow::Borrowed(_) => panic!("compare=none loop should be instrumented"),
        }
        assert!(incompatible_compare_none_loop_guard_golden(ll, "spin", &metal).is_none());

        metal.compare = "full".into();
        match candidate_ll_for_metal_compare(ll, "spin", &metal).unwrap() {
            Cow::Borrowed(text) => assert_eq!(text, ll),
            Cow::Owned(_) => panic!("compare=full should keep original LL"),
        }
    }

    #[test]
    fn compare_none_barrier_loop_golden_requires_rebank() {
        let ll = "\
define void @barrier_loop(ptr addrspace(1) %0) {
  br label %loop
loop:
  tail call void @air.wg.barrier(i32 2, i32 1)
  br label %loop
}

declare void @air.wg.barrier(i32, i32)
";
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
            output_sha256: Some(sha256_hex(&[])),
            output_b64: Some(encode_output_b64(&[])),
            spv_sha256: None,
            compare: "none".into(),
            fc_specialization: None,
            fc_values: None,
            stage: None,
            entry: None,
            error: None,
        };

        let reason = incompatible_compare_none_loop_guard_golden(ll, "barrier_loop", &metal)
            .expect("old compare=none barrier golden should require rebank");
        assert!(reason.contains("compare=none"), "{reason}");
        assert!(reason.contains("air.wg.barrier"), "{reason}");
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
    fn unsupported_fragment_color_arity_is_classified_without_panic_prefix() {
        let err = "newRenderPipelineStateWithDescriptor(inputStreamColorBlitFragment): \
fragment shader color output does not have enough components for the pixel format (MTLPixelFormatRGBA16Float)";

        assert_eq!(
            classify_metal_oracle_panic(err),
            format!("unsupported Metal fragment color output attachment arity: {err}")
        );
    }

    #[test]
    fn unsupported_rgb_fragment_color_output_is_preflighted() {
        let ll = r#"
define <3 x half> @inputStreamColorBlitFragment() {
  ret <3 x half> zeroinitializer
}

!air.fragment = !{!15}
!15 = !{ptr @inputStreamColorBlitFragment, !16, !18}
!16 = !{!17}
!17 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"half3"}
!18 = !{}
"#;

        let reason =
            unsupported_fragment_color_output_arity(ll).expect("half3 color should be unsupported");
        assert!(reason.contains("render target location 0"), "{reason}");
        assert!(reason.contains("\"half3\""), "{reason}");
        assert!(reason.contains("no renderable RGB"), "{reason}");
    }

    #[test]
    fn supported_rgba_fragment_color_output_is_not_preflighted() {
        let ll = r#"
define <4 x half> @frag() {
  ret <4 x half> zeroinitializer
}

!air.fragment = !{!15}
!15 = !{ptr @frag, !16, !18}
!16 = !{!17}
!17 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"half4"}
!18 = !{}
"#;

        assert!(unsupported_fragment_color_output_arity(ll).is_none());
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
    fn fragment_plan_uses_16bit_uint_render_target_for_ushort_output() {
        let ll = r#"
define <{ i16 }> @frag() {
  ret <{ i16 }> zeroinitializer
}

!air.fragment = !{!15}
!15 = !{ptr @frag, !16, !17}
!16 = !{!18}
!17 = !{}
!18 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"ushort", !"air.arg_name", !"color"}
"#;

        let plan = infer_plan(ll);
        assert_eq!(plan.output.kind, "render_target");
        assert_eq!(plan.output.format, "R16Uint");
        assert_eq!(
            parse_format(&plan.output.format).unwrap(),
            DataFormat::R16Uint
        );
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
    fn undef_fragment_color_output_golden_requires_rebank() {
        let ll = r#"
define <{ <4 x float>, float }> @frag(<4 x float> %position) {
  %depth = extractelement <4 x float> %position, i64 2
  %out = insertvalue <{ <4 x float>, float }> undef, float %depth, 1
  ret <{ <4 x float>, float }> %out
}

!air.fragment = !{!15}
!15 = !{ptr @frag, !16, !19}
!16 = !{!17, !18}
!17 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4", !"air.arg_name", !"color"}
!18 = !{!"air.depth", !"air.depth_qualifier", !"air.any", !"air.arg_type_name", !"float", !"air.arg_name", !"depth"}
!19 = !{!20}
!20 = !{i32 0, !"air.position", !"air.center", !"air.arg_type_name", !"float4", !"air.arg_name", !"position"}
"#;

        let plan = infer_plan(ll);
        assert_eq!(plan.output.format, "Rgba32Float");
        let tmp = crate::scratch_dir_for("undef_fragment_color_output_golden_requires_rebank");
        let spv = metal2vulkan::translate_sanitized_native(
            ll,
            metal2vulkan::passes::Stage::Fragment,
            &tmp,
        )
        .expect("translate");
        let _ = std::fs::remove_dir_all(&tmp);
        let reason = incompatible_undefined_fragment_color_output_golden(ll, &plan, &spv)
            .expect("undefined color output should invalidate golden");
        assert!(reason.contains("undefined"), "{reason}");
    }

    #[test]
    fn texture_metadata_becomes_plan_texture() {
        let ll = r#"
	define void @k(ptr addrspace(2) %0, <2 x i32> %gid) {
  ret void
}

!air.kernel = !{!15}
!15 = !{ptr @k, !16, !17}
!16 = !{}
!17 = !{!18, !19, !20}
!18 = !{i32 0, !"air.texture", !"air.location_index", i32 2, i32 1, !"air.read_write", !"air.arg_type_name", !"texture2d<half, read_write>", !"air.arg_name", !"tex"}
!19 = !{i32 1, !"air.sampler", !"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"sampler", !"air.arg_name", !"s"}
!20 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint2", !"air.arg_name", !"gid"}
"#;

        let plan = infer_plan(ll);
        assert_eq!(plan.textures.len(), 1);
        assert_eq!(plan.textures[0].index, 2);
        assert_eq!(plan.textures[0].role, "StorageReadWrite");
        assert_eq!(plan.textures[0].format, "Rgba16Float");
        assert_eq!(plan.output.kind, "texture");
        assert_eq!(plan.output.index, 2);
        assert_eq!(plan.dispatch_grid, [8, 8, 1]);
        assert_eq!(plan.dispatch_tg, [8, 8, 1]);

        let mut stale = plan.clone();
        stale.dispatch_grid = [64, 1, 1];
        stale.dispatch_tg = [64, 1, 1];
        let metal = MetalRow {
            air_sha256: "x".into(),
            shard: None,
            label: String::new(),
            status: "ok".into(),
            backend: "metal".into(),
            seed_profile: SEED_PROFILE.into(),
            plan_version: PLAN_VERSION,
            plan: stale,
            input_sha256: None,
            output_sha256: Some(sha256_hex(&[])),
            output_b64: Some(encode_output_b64(&[])),
            spv_sha256: None,
            compare: "full".into(),
            fc_specialization: None,
            fc_values: None,
            stage: Some("Kernel".into()),
            entry: Some("k".into()),
            error: None,
        };
        let reason = incompatible_output_plan_golden(ll, &metal).expect("stale dispatch plan");
        assert!(reason.contains("dispatch plan"), "{reason}");
    }

    #[test]
    fn texture_array_metadata_expands_plan_textures() {
        let ll = r#"
define <4 x float> @frag(ptr %textures, ptr addrspace(2) %view_id) {
  ret <4 x float> zeroinitializer
}

!air.fragment = !{!0}
!0 = !{ptr @frag, !1, !2}
!1 = !{!3}
!2 = !{!4, !5}
!3 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!4 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 2, !"air.sample", !"air.arg_type_name", !"array<texture2d<float, sample>, 2>", !"air.arg_name", !"material"}
!5 = !{i32 1, !"air.buffer", !"air.buffer_size", i32 1, !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_size", i32 1, !"air.arg_type_name", !"uchar", !"air.arg_name", !"view_id"}
"#;

        let plan = infer_plan(ll);
        assert_eq!(
            texture_plan_signature(&plan.textures),
            vec![
                "0:Rgba32Float:Sampled:8x8x1".to_string(),
                "1:Rgba32Float:Sampled:8x8x1".to_string(),
            ]
        );

        let mut stale = plan.clone();
        stale.textures.retain(|texture| texture.index == 0);
        let metal = metal_row_for_compare(&[], stale, Some("Fragment"));
        let reason =
            incompatible_texture_array_plan_golden(ll, &metal).expect("stale texture array plan");
        assert!(reason.contains("texture-array plan"), "{reason}");
    }

    #[test]
    fn texture2d_array_literal_layers_expand_plan_depth() {
        let ll = r#"
define <4 x float> @frag(ptr addrspace(1) %tex, <2 x float> %uv) {
  %s0 = tail call { <4 x half>, i8 } @air.sample_texture_2d_array.v4f16(ptr addrspace(1) %tex, ptr addrspace(2) null, <2 x float> %uv, i32 0, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %s2 = tail call { <4 x half>, i8 } @air.sample_texture_2d_array.v4f16(ptr addrspace(1) %tex, ptr addrspace(2) null, <2 x float> %uv, i32 2, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %v = extractvalue { <4 x half>, i8 } %s2, 0
  %out = tail call <4 x float> @air.convert.f.v4f32.f.v4f16(<4 x half> %v)
  ret <4 x float> %out
}

declare { <4 x half>, i8 } @air.sample_texture_2d_array.v4f16(ptr addrspace(1), ptr addrspace(2), <2 x float>, i32, i1, <2 x i32>, i1, float, float, i32)
declare <4 x float> @air.convert.f.v4f32.f.v4f16(<4 x half>)

!air.fragment = !{!0}
!0 = !{ptr @frag, !1, !2}
!1 = !{!3}
!2 = !{!4, !5}
!3 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!4 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d_array<half, sample>", !"air.arg_name", !"tex"}
!5 = !{i32 1, !"air.fragment_input", !"air.arg_type_name", !"float2", !"air.arg_name", !"uv"}
"#;

        let plan = infer_plan(ll);
        assert_eq!(plan.textures.len(), 1);
        assert_eq!(plan.textures[0].format, "Rgba16Float");
        assert_eq!(plan.textures[0].d, 3);

        let mut stale = plan.clone();
        stale.textures[0].d = 1;
        let metal = metal_row_for_compare(&[], stale, Some("Fragment"));
        let reason = incompatible_texture_array_plan_golden(ll, &metal)
            .expect("stale texture2d_array layer plan");
        assert!(reason.contains("texture-array plan"), "{reason}");
    }

    #[test]
    fn writable_float_texture_metadata_becomes_r32_output_format() {
        let ll = r#"
define void @k(ptr addrspace(1) %dst, <2 x i16> %gid) {
  tail call void @air.write_texture_2d.i16.v4f32(ptr addrspace(1) %dst, <2 x i16> %gid, <4 x float> zeroinitializer, i16 0, i32 2)
  ret void
}

declare void @air.write_texture_2d.i16.v4f32(ptr addrspace(1), <2 x i16>, <4 x float>, i16, i32)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.write", !"air.arg_type_name", !"texture2d<float, write>", !"air.arg_name", !"dst"}
!4 = !{i32 1, !"air.thread_position_in_grid", !"air.arg_type_name", !"ushort2", !"air.arg_name", !"gid"}
"#;

        let plan = infer_plan(ll);
        assert_eq!(plan.textures.len(), 1);
        assert_eq!(plan.textures[0].format, "R32Float");
        assert_eq!(plan.output.kind, "texture");
        assert_eq!(plan.output.format, "R32Float");
    }

    #[test]
    fn pointer_location_metadata_uses_static_default() {
        let ll = r#"
@tex_loc = internal addrspace(2) global i32 0, align 4
@buf_loc = internal addrspace(2) global i32 0, align 4

define internal void @_GLOBAL__sub_I_shader.metal() section "air.static_init" {
  store i32 4, ptr addrspace(2) @tex_loc, align 4
  store i32 7, ptr addrspace(2) @buf_loc, align 4
  ret void
}

define <4 x float> @frag(ptr addrspace(1) %tex, ptr addrspace(2) %buf) {
  ret <4 x float> zeroinitializer
}

!air.fragment = !{!0}
!0 = !{ptr @frag, !1, !2}
!1 = !{!3}
!2 = !{!4, !5}
!3 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!4 = !{i32 0, !"air.texture", !"air.location_index", ptr addrspace(2) @tex_loc, i32 1, !"air.read", !"air.arg_type_name", !"texture2d<half, read>", !"air.arg_name", !"tex"}
!5 = !{i32 1, !"air.buffer", !"air.buffer_size", i32 4, !"air.location_index", ptr addrspace(2) @buf_loc, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"buf"}
"#;

        let plan = infer_plan(ll);
        assert_eq!(plan.textures.len(), 1);
        assert_eq!(plan.textures[0].index, 4);
        assert_eq!(plan.textures[0].role, "StorageRead");
        assert!(plan.buffers.iter().any(|buffer| buffer.index == 7));

        let mut stale = plan.clone();
        stale.textures[0].index = 1;
        stale.buffers.iter_mut().for_each(|buffer| {
            if buffer.index == 7 {
                buffer.index = 1;
            }
        });
        let mut metal = metal_row_for_compare(&[], stale, Some("Fragment"));
        metal.fc_specialization = Some(FC_SPECIALIZATION_ZERO.into());
        let reason = incompatible_static_resource_plan_golden(ll, &metal)
            .expect("stale static resource plan");
        assert!(reason.contains("input resource plan"), "{reason}");
    }

    #[test]
    fn writable_half_buffer_metadata_becomes_output_format() {
        let ll = r#"
define void @k(ptr addrspace(1) %out) {
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 3, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 2, !"air.arg_type_align_size", i32 2, !"air.arg_type_name", !"half", !"air.arg_name", !"out"}
"#;

        let plan = infer_plan(ll);

        assert_eq!(plan.output.kind, "buffer");
        assert_eq!(plan.output.index, 3);
        assert_eq!(plan.output.format, "R16Float");
    }
}
