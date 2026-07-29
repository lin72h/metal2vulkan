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
pub const SEED_PROFILE: &str = "deterministic_v7_thread_indexed_inputs";
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
/// guard in `oracle_macos`, not from this timeout. Some large bounded kernels still spend minutes in
/// debug validation builds (for example ASTC decode compile+execute completed in 266.7s locally), so
/// keep the default high enough for those while retaining the env override for shorter local sweeps.
pub const DEFAULT_CASE_TIMEOUT_SECS: u64 = 300;
/// Log `# SLOW <air_sha256> …` when a case is still running (or finished) past this wall time.
pub const SLOW_CASE_SECS: u64 = 30;
const FC_SPECIALIZATION_ZERO: &str = "zero";
const FC_SPECIALIZATION_VALUES: &str = "values";
const INPUT_SPECIALIZATION_EXPLICIT: &str = "explicit";

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
    /// Validation-only marker that says `metal.plan` is a deliberately banked input plan and may
    /// be reused by forced Metal oracle reruns even when no function constants are involved.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_specialization: Option<String>,
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
        let max_abs = if sampled_half_render_target_palette_tolerance_applies(ll) {
            0.023_437_5
        } else {
            0.007_812_5
        };
        tolerance.max_abs = Some(tolerance.max_abs.unwrap_or(0.0).max(max_abs));
    }
    if sampled_f32_storage_texture_tolerance_applies(format, ll, metal) {
        let max_abs = if ll.is_some_and(|ll| {
            sampled_f32_storage_texture_uses_half_imageblock(ll)
                || sampled_f32_storage_texture_uses_half_coordinate_texture(ll)
        }) {
            0.003_906_25
        } else {
            0.001_953_125
        };
        tolerance.max_abs = Some(tolerance.max_abs.unwrap_or(0.0).max(max_abs));
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

fn sampled_half_render_target_palette_tolerance_applies(ll: Option<&str>) -> bool {
    let Some(ll) = ll else {
        return false;
    };
    (ll.contains("@air.dot.v2f16")
        || ll.contains("@air.dot.v3f16")
        || ll.contains("@air.dot.v4f16"))
        && (ll.contains("@air.mix.v2f16")
            || ll.contains("@air.mix.v3f16")
            || ll.contains("@air.mix.v4f16"))
}

fn sampled_f32_storage_texture_tolerance_applies(
    format: DataFormat,
    ll: Option<&str>,
    metal: &MetalRow,
) -> bool {
    if !matches!(
        format,
        DataFormat::R32Float | DataFormat::Rg32Float | DataFormat::Rgba32Float
    ) || metal.plan.output.kind != "texture"
    {
        return false;
    }
    let Some(ll) = ll else {
        return false;
    };
    if !ll.contains("@air.sample_texture") {
        return false;
    }
    infer_textures(ll).into_iter().any(|texture| {
        texture.role == "Sampled"
            && texture.format == "Rgba32Float"
            && texture.seed_mode == SEED_MODE_FINITE_FLOAT32
    })
}

fn sampled_f32_storage_texture_uses_half_imageblock(ll: &str) -> bool {
    ll.contains("@air.imageblock_data")
        && ll.contains("@air.write_imageblock_slice_to_texture_2d")
        && ll.contains(".v4f16")
}

fn sampled_f32_storage_texture_uses_half_coordinate_texture(ll: &str) -> bool {
    ll.contains("@air.read_texture_2d.v4f16")
        && ll.contains("@air.convert.f.v2f32.f.v2f16")
        && ll.contains("@air.sample_texture_2d.v4f32")
        && infer_textures(ll).into_iter().any(|texture| {
            texture.role == "StorageRead"
                && texture.format == "Rgba16Float"
                && texture.seed_mode == SEED_MODE_FINITE_FLOAT16
        })
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
    pub compile_missing: bool,
    pub only_air: Option<String>,
    pub only_air_list: Option<PathBuf>,
    pub jobs: usize,
    /// When true, process in-process (worker mode) and exit with outcome code.
    pub oneshot: bool,
    /// Per-case wall timeout for worker subprocesses (env override in CLI parsing).
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
            compile_missing: false,
            only_air: None,
            only_air_list: None,
            jobs: default_workers(),
            oneshot: false,
            timeout_secs: DEFAULT_CASE_TIMEOUT_SECS,
        }
    }

    fn reruns_existing_backend_rows(&self) -> bool {
        self.force
            || self.failed_only
            || self.only_status.is_some()
            || self.only_bucket.is_some()
            || self.contains.is_some()
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
            "--compile-missing" => cfg.compile_missing = true,
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
         --compile-missing translate+spirv-val Vulkan/MoltenVK rows that cannot compare to Metal;\n\
                         successful rows are recorded as smoke with the skip reason in error\n\
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
        RunBackend::Vulkan => status == "ok" || status == "tolerance" || status == "smoke",
        RunBackend::MoltenVk => status == "ok",
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

pub fn load_candidate_rows(path: &Path) -> HashMap<String, CandidateRow> {
    let mut map = HashMap::new();
    let Ok(file) = File::open(path) else {
        return map;
    };
    for line in BufReader::new(file).lines().map_while(Result::ok) {
        let t = line.trim();
        if t.is_empty() || t.starts_with('#') {
            continue;
        }
        if let Ok(row) = serde_json::from_str::<CandidateRow>(t) {
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

struct MetalStatusFields<'a> {
    status: &'a str,
    compare: &'a str,
    plan: HarnessPlan,
    stage: Option<Stage>,
    entry: Option<&'a str>,
    error: Option<String>,
}

fn metal_status_row(
    tr: &TranslateRow,
    src: &SourceFile,
    fields: MetalStatusFields<'_>,
) -> MetalRow {
    MetalRow {
        air_sha256: tr.air_sha256.clone(),
        shard: src.shard.clone(),
        label: src.label.clone(),
        status: fields.status.into(),
        backend: RunBackend::Metal.as_str().into(),
        seed_profile: SEED_PROFILE.into(),
        plan_version: PLAN_VERSION,
        plan: fields.plan,
        input_sha256: None,
        output_sha256: None,
        output_b64: None,
        spv_sha256: tr.spv_sha256.clone(),
        compare: fields.compare.into(),
        fc_specialization: None,
        fc_values: None,
        input_specialization: None,
        stage: fields.stage.map(|stage| format!("{stage:?}")),
        entry: fields.entry.map(str::to_string),
        error: fields.error,
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
            let row = metal_status_row(
                tr,
                src,
                MetalStatusFields {
                    status,
                    compare: metal_compare,
                    plan: infer_plan(""),
                    stage: None,
                    entry: None,
                    error: Some(error),
                },
            );
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
    let mut output = if stage == Stage::Fragment {
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
    resize_thread_indexed_input_buffers(ll_or_meta_text, dispatch_grid[0], &mut buffers);
    if stage == Stage::Kernel && output.kind == "buffer" {
        apply_output_stride_seed_values(ll_or_meta_text, output.index, &mut buffers);
    }
    if output.kind == "buffer" {
        if let Some(buffer) = buffers.iter().find(|b| b.index == output.index) {
            output.len = Some(buffer.len);
        }
    }

    HarnessPlan {
        buffers,
        textures,
        output,
        dispatch_grid,
        dispatch_tg,
    }
}

fn resize_thread_indexed_input_buffers(ll: &str, grid_x: u32, buffers: &mut [PlanBuffer]) {
    let grid_x = u64::from(grid_x.max(1));
    for buffer in buffers {
        if buffer.role != "Input" {
            continue;
        }
        let Some(type_name) = buffer_type_name_for_location(ll, buffer.index) else {
            continue;
        };
        let Some(required_len) = thread_indexed_input_required_len(ll, &type_name, grid_x) else {
            continue;
        };
        buffer.len = buffer.len.max(required_len);
    }
}

fn thread_indexed_input_required_len(ll: &str, type_name: &str, grid_x: u64) -> Option<usize> {
    let (llvm_ty, elem_bytes) = llvm_thread_indexed_input_type_and_size(type_name)?;
    if !module_has_dynamic_device_gep(ll, llvm_ty) {
        return None;
    }
    let required_elems =
        dynamic_device_gep_required_elements(ll, llvm_ty, grid_x).unwrap_or(grid_x);
    let required_bytes = required_elems.saturating_mul(elem_bytes as u64);
    usize::try_from(required_bytes).ok()
}

fn apply_output_stride_seed_values(ll: &str, output_index: u32, buffers: &mut [PlanBuffer]) {
    let mut required_by_buffer: HashMap<u32, u64> = HashMap::new();
    for req in output_stride_control_requirements(ll, output_index, buffers) {
        required_by_buffer
            .entry(req.buffer)
            .and_modify(|current| *current = (*current).max(req.min_row_bytes))
            .or_insert(req.min_row_bytes);
    }
    if required_by_buffer.is_empty() {
        return;
    }

    for buffer in buffers
        .iter_mut()
        .filter(|b| b.seed_mode == SEED_MODE_BOUNDED_CONTROL)
    {
        let Some(&required) = required_by_buffer.get(&buffer.index) else {
            continue;
        };
        for field in &mut buffer.seed_layout {
            if !bounded_control_seed_field_is_within_buffer(buffer.len, field) {
                continue;
            }
            let Some(max_value) = control_seed_field_max_value(field.size) else {
                continue;
            };
            if required > max_value {
                continue;
            }
            let current = field.value.unwrap_or(u64::from(BOUNDED_CONTROL_DIM));
            if current < required {
                field.value = Some(required);
            }
        }
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
    let local_array_index_control_bufs = buffers_with_loads_used_as_local_array_indices(ll);
    let bounded_control_module = module_uses_bounded_control_buffers(ll, &loop_bound_bufs);
    let readonly_buffers = readonly_entry_buffer_locations(ll);
    let writeonly_buffers = writeonly_entry_buffer_locations(ll);
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
        let control_param = is_control_param_buffer_meta(line, fixed_size);
        let size = if control_param {
            fixed_size
                .or_else(|| extract_i32_after(line, "air.arg_type_size"))
                .unwrap_or(DEFAULT_BUFFER_LEN as i32)
        } else {
            fixed_size.unwrap_or(DEFAULT_BUFFER_LEN as i32)
        };
        let loc = metadata_param_index(line)
            .and_then(|idx| locations.buffers.get(&idx).copied())
            .or_else(|| extract_i32_after(line, "air.location_index").map(|loc| loc as u32))
            .unwrap_or(0);
        let loc_u = loc;
        let role = if writeonly_buffers.contains(&loc_u) {
            "Output"
        } else if line.contains("air.read_write") {
            "InOut"
        } else if line.contains("!\"air.write\"") || line.contains("\"air.write\"") {
            "Output"
        } else {
            "Input"
        };
        let type_name = quoted_metadata_string_after(line, "air.arg_type_name");
        let atomic_counter_buffer = is_atomic_counter_air_type(type_name.as_deref());
        let seed_mode = if control_param
            || atomic_counter_buffer
            || loop_bound_bufs.contains(&loc_u)
            || stride_control_bufs.contains(&loc_u)
            || (role == "Input" && atomic_i32_load_bufs.contains(&loc_u))
        {
            SEED_MODE_BOUNDED_CONTROL
        } else if let Some(seed_mode) = finite_float_buffer_seed_mode(type_name.as_deref()) {
            seed_mode
        } else {
            SEED_MODE_DETERMINISTIC
        };
        let mut len = (size as usize).max(4);
        if fixed_size.is_none()
            && bounded_control_module
            && (role == "Input" || readonly_buffers.contains(&loc_u))
            && !writeonly_buffers.contains(&loc_u)
        {
            if let Some(payload_len) = bounded_control_float_payload_len(type_name.as_deref()) {
                len = len.max(payload_len);
            }
        }
        let finite_struct_seed =
            finite_struct_float_seed_layout(ll, line, len, bounded_control_module);
        let seed_mode = if seed_mode == SEED_MODE_DETERMINISTIC
            && role != "Output"
            && finite_struct_seed.is_some()
        {
            SEED_MODE_FINITE_STRUCT_FLOAT
        } else {
            seed_mode
        };
        let seed_layout = if seed_mode == SEED_MODE_BOUNDED_CONTROL {
            if atomic_counter_buffer {
                zero_u32_seed_layout(len)
            } else {
                bounded_control_seed_layout(
                    ll,
                    line,
                    len,
                    &stride_control_bufs,
                    &fdiv_denominator_control_bufs,
                    &local_array_index_control_bufs,
                )
            }
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
    entry_buffer_locations_with_arg_attr(ll, "writeonly")
}

fn readonly_entry_buffer_locations(ll: &str) -> HashSet<u32> {
    entry_buffer_locations_with_arg_attr(ll, "readonly")
}

fn entry_buffer_locations_with_arg_attr(ll: &str, attr: &str) -> HashSet<u32> {
    let Some(args) = primary_entry_function_args(ll) else {
        return HashSet::new();
    };
    let arg_to_buf = arg_index_to_buffer_location(ll);
    args.split(',')
        .enumerate()
        .filter(|(_, arg)| arg.split_whitespace().any(|token| token == attr))
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
    for arg in primary_entry_function_args(ll)
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
        Some(name) if name.starts_with("texturecube<") || name.starts_with("depthcube<") => 6,
        Some(name) if name.starts_with("texture2d_array<") => {
            max_literal_sample_texture_array_layer(ll)
                .and_then(|layer| layer.checked_add(1))
                .unwrap_or(DEFAULT_TEXTURE_EXTENT.depth)
                .max(DEFAULT_TEXTURE_EXTENT.depth)
        }
        Some(name)
            if name.starts_with("texturecube_array<") || name.starts_with("depthcube_array<") =>
        {
            1
        }
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

fn is_atomic_counter_air_type(type_name: Option<&str>) -> bool {
    type_name
        .map(|name| name.trim().trim_end_matches('*').trim() == "metal::_atomic")
        .unwrap_or(false)
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
    seed_control_int_fields: bool,
) -> Option<(usize, Vec<ControlSeedField>)> {
    let stride = extract_i32_after(buffer_meta_line, "air.arg_type_size")? as usize;
    if stride == 0 || stride > len.max(1) {
        return None;
    }
    let node = metadata_ref_after(buffer_meta_line, "air.struct_type_info")?;
    let mut fields = Vec::new();
    finite_struct_float_seed_fields(
        ll,
        node,
        0,
        stride,
        seed_control_int_fields,
        &mut fields,
        &mut Vec::new(),
    );
    (!fields.is_empty()).then_some((stride, fields))
}

fn finite_struct_float_seed_fields(
    ll: &str,
    node: u32,
    base_offset: usize,
    stride: usize,
    seed_control_int_fields: bool,
    fields: &mut Vec<ControlSeedField>,
    stack: &mut Vec<u32>,
) {
    if stack.contains(&node) {
        return;
    }
    let Some(line) = metadata_node_line(ll, node) else {
        return;
    };
    let Some(payload) = metadata_payload(line) else {
        return;
    };
    let tokens = metadata_tokens(payload);
    stack.push(node);
    let mut i = 0;
    let mut pending_nested_node = None;
    while i + 3 < tokens.len() {
        if metadata_quoted_token(tokens[i]) == Some("air.struct_type_info") {
            pending_nested_node = tokens.get(i + 1).and_then(|tok| metadata_ref_token(tok));
            i += 2;
            continue;
        }
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
        let offset = base_offset + offset as usize;
        let field_byte_size = byte_size as usize;
        let repeat_count = usize::try_from(repeat_count).unwrap_or(0).max(1);
        if let Some((elem_size, lanes)) = finite_float_field_shape(type_name, byte_size) {
            for element in 0..repeat_count {
                let element_offset = offset + element * field_byte_size;
                for lane in 0..lanes {
                    let lane_offset = element_offset + lane * elem_size;
                    if lane_offset.saturating_add(elem_size) <= stride {
                        fields.push(ControlSeedField {
                            offset: lane_offset,
                            size: elem_size,
                            value: None,
                        });
                    }
                }
            }
        } else if seed_control_int_fields {
            if let Some((size, value)) = finite_struct_control_int_field_seed(type_name, byte_size)
            {
                for element in 0..repeat_count {
                    let field_offset = offset + element * field_byte_size;
                    if field_offset.saturating_add(size) <= stride {
                        fields.push(ControlSeedField {
                            offset: field_offset,
                            size,
                            value: Some(value),
                        });
                    }
                }
            } else if let Some(nested_node) = pending_nested_node.take() {
                for element in 0..repeat_count {
                    finite_struct_float_seed_fields(
                        ll,
                        nested_node,
                        offset + element * field_byte_size,
                        stride,
                        seed_control_int_fields,
                        fields,
                        stack,
                    );
                }
            }
        } else if let Some(nested_node) = pending_nested_node.take() {
            for element in 0..repeat_count {
                finite_struct_float_seed_fields(
                    ll,
                    nested_node,
                    offset + element * field_byte_size,
                    stride,
                    seed_control_int_fields,
                    fields,
                    stack,
                );
            }
        }
        i += 5;
    }
    stack.pop();
}

fn finite_struct_control_int_field_seed(type_name: &str, byte_size: i32) -> Option<(usize, u64)> {
    let name = type_name.trim().trim_end_matches('*').trim();
    let (size, value) = match name {
        "bool" => (1, 0),
        "uchar" | "char" => (1, u64::from(BOUNDED_CONTROL_DIM)),
        "ushort" | "short" => (2, u64::from(BOUNDED_CONTROL_DIM)),
        "uint" | "int" => (4, u64::from(BOUNDED_CONTROL_DIM)),
        "ulong" | "long" => (8, u64::from(BOUNDED_CONTROL_DIM)),
        _ => return None,
    };
    (byte_size as usize == size).then_some((size, value))
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
    if !constant_as && !has_struct {
        return false;
    }
    let Some(sz) = fixed_size.or_else(|| {
        constant_as
            .then(|| extract_i32_after(line, "air.arg_type_size"))
            .flatten()
    }) else {
        return false;
    };
    if sz <= 0 || sz as usize > BOUNDED_CONTROL_MAX_BYTES {
        return false;
    }
    true
}

/// Lightweight IR scan: buffer locations whose scalar integer loads appear in an `icmp` that
/// feeds a `br` (trip-count / early-out class). Used to catch device-space counters that
/// are not tagged as constant-param structs.
///
/// Not a full relooper — false positives only force small integer seeds on that buffer,
/// which is safe for execution harnesses (goldens re-derived under the new seed profile).
fn buffers_with_loads_used_as_loop_bounds(ll: &str) -> HashSet<u32> {
    let Some(body) = primary_entry_function_body(ll) else {
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
            // Loop bounds are often loaded in a preheader and carried through loop phis
            // (or simple integer combinators) before the cyclic branch compares them.
            if rhs.starts_with("phi ")
                || rhs.starts_with("select ")
                || rhs.starts_with("add ")
                || rhs.starts_with("sub ")
            {
                if let Some(buf) = first_int_buf_operand(rhs, &int_from_buf) {
                    int_from_buf.insert(reg, buf);
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

fn first_int_buf_operand(rhs: &str, int_from_buf: &HashMap<&str, u32>) -> Option<u32> {
    for tok in rhs.split([',', ' ', '[', ']']) {
        let t = tok.trim();
        if let Some(name) = t.strip_prefix('%') {
            if let Some(&buf) = int_from_buf.get(name) {
                return Some(buf);
            }
        }
    }
    None
}

fn first_local_ptr_operand<'a>(rhs: &'a str, local_ptrs: &HashSet<&str>) -> Option<&'a str> {
    percent_operands(rhs)
        .into_iter()
        .find(|operand| local_ptrs.contains(operand))
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
    let (label, _) = line.split_once(':')?;
    (!label.is_empty() && !label.starts_with(';') && !label.contains(' ') && !label.contains('\t'))
        .then_some(label)
}

fn branch_labels(line: &str) -> Vec<&str> {
    if let Some(label) = line.trim().strip_prefix("br label %") {
        return vec![label
            .split(|c: char| c == ',' || c.is_whitespace())
            .next()
            .unwrap_or(label)];
    }
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
    let Some(body) = primary_entry_function_body(ll) else {
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
    let Some(body) = primary_entry_function_body(ll) else {
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

/// Buffer locations whose integer loads feed a dynamic inbounds GEP into a local alloca-backed
/// array. The default bounded-control integer value is 16; that is a useful small loop/count bound,
/// but it is out of range for local arrays with fewer elements.
fn buffers_with_loads_used_as_local_array_indices(ll: &str) -> HashSet<u32> {
    if !local_array_capacity_is_less_than_bounded_control_dim(ll) {
        return HashSet::new();
    }
    let Some(body) = primary_entry_function_body(ll) else {
        return HashSet::new();
    };
    let arg_to_buf = arg_index_to_buffer_location(ll);
    let arg_name_to_buf = arg_name_to_buffer_location(ll, &arg_to_buf);
    if arg_to_buf.is_empty() {
        return HashSet::new();
    }

    let mut ptr_buf: HashMap<&str, u32> = HashMap::new();
    let mut local_ptrs: HashSet<&str> = HashSet::new();
    let mut int_from_buf: HashMap<&str, u32> = HashMap::new();
    let mut out = HashSet::new();
    for line in body.lines() {
        let line = line.trim();
        let Some((reg, rhs)) = split_assign(line) else {
            continue;
        };
        if rhs.starts_with("alloca ") {
            local_ptrs.insert(reg);
            continue;
        }
        if rhs.starts_with("getelementptr") || rhs.starts_with("bitcast") {
            let local_ptr = first_local_ptr_operand(rhs, &local_ptrs).is_some();
            if rhs.starts_with("getelementptr") && rhs.contains("inbounds") && local_ptr {
                if let Some(buf) = first_int_buf_operand(rhs, &int_from_buf) {
                    out.insert(buf);
                }
            }
            if local_ptr {
                local_ptrs.insert(reg);
            }
            if let Some(buf) = first_buf_operand(rhs, &ptr_buf, &arg_to_buf, &arg_name_to_buf) {
                ptr_buf.insert(reg, buf);
            }
            continue;
        }
        if is_scalar_integer_load_rhs(rhs) {
            if let Some(buf) = first_buf_operand(rhs, &ptr_buf, &arg_to_buf, &arg_name_to_buf) {
                int_from_buf.insert(reg, buf);
            }
            continue;
        }
        if rhs.starts_with("zext ") || rhs.starts_with("sext ") || rhs.starts_with("trunc ") {
            if let Some(src) = first_percent_reg(rhs) {
                if let Some(&buf) = int_from_buf.get(src) {
                    int_from_buf.insert(reg, buf);
                }
            }
            continue;
        }
        if rhs.starts_with("phi ")
            || rhs.starts_with("select ")
            || rhs.starts_with("add ")
            || rhs.starts_with("sub ")
        {
            if let Some(buf) = first_int_buf_operand(rhs, &int_from_buf) {
                int_from_buf.insert(reg, buf);
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
    let reflected = reflected_entry_buffer_locations(ll);
    if !reflected.is_empty() {
        return reflected;
    }

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
        map.entry(ord as usize).or_insert(loc);
    }
    map
}

fn reflected_entry_buffer_locations(ll: &str) -> HashMap<usize, u32> {
    match stage_from_ll(ll) {
        Stage::Kernel => metal2vulkan::meta::parse_air_kernel_meta(ll)
            .map(|meta| {
                meta.roles
                    .into_iter()
                    .filter_map(|(arg, role)| match role {
                        KernRole::Buffer(loc) => Some((arg as usize, loc)),
                        _ => None,
                    })
                    .collect()
            })
            .unwrap_or_default(),
        Stage::Vertex => metal2vulkan::meta::parse_air_vertex_meta(ll)
            .map(|meta| {
                meta.roles
                    .into_iter()
                    .filter_map(|(arg, role)| match role {
                        VertRole::Buffer(loc) => Some((arg as usize, loc)),
                        _ => None,
                    })
                    .collect()
            })
            .unwrap_or_default(),
        Stage::Fragment => metal2vulkan::meta::parse_air_fragment_meta(ll)
            .map(|meta| {
                meta.roles
                    .into_iter()
                    .filter_map(|(arg, role)| match role {
                        FragRole::Buffer(loc) => Some((arg as usize, loc)),
                        _ => None,
                    })
                    .collect()
            })
            .unwrap_or_default(),
    }
}

fn arg_name_to_buffer_location(ll: &str, arg_to_buf: &HashMap<usize, u32>) -> HashMap<String, u32> {
    let Some(args) = primary_entry_function_args(ll) else {
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

fn zero_u32_seed_layout(len: usize) -> Vec<ControlSeedField> {
    (0..len)
        .step_by(4)
        .take_while(|offset| offset.saturating_add(4) <= len)
        .map(|offset| ControlSeedField {
            offset,
            size: 4,
            value: Some(0),
        })
        .collect()
}

fn bounded_control_seed_layout(
    ll: &str,
    buffer_meta_line: &str,
    len: usize,
    stride_control_bufs: &HashSet<u32>,
    fdiv_denominator_control_bufs: &HashSet<u32>,
    local_array_index_control_bufs: &HashSet<u32>,
) -> Vec<ControlSeedField> {
    let Some(node) = metadata_ref_after(buffer_meta_line, "air.struct_type_info") else {
        return scalar_bounded_control_seed_layout(
            buffer_meta_line,
            len,
            stride_control_bufs,
            fdiv_denominator_control_bufs,
            local_array_index_control_bufs,
        );
    };
    let loc = extract_i32_after(buffer_meta_line, "air.location_index").map(|v| v as u32);
    let seed_float_one = loc.is_some_and(|loc| fdiv_denominator_control_bufs.contains(&loc));
    let seed_int_zero = loc.is_some_and(|loc| local_array_index_control_bufs.contains(&loc));
    let mut fields = Vec::new();
    let ctx = BoundedControlSeedCtx {
        ll,
        len,
        seed_float_one,
        seed_int_zero,
    };
    bounded_control_seed_fields(&ctx, node, 0, None, &mut fields, &mut Vec::new());
    fields
}

struct BoundedControlSeedCtx<'a> {
    ll: &'a str,
    len: usize,
    seed_float_one: bool,
    seed_int_zero: bool,
}

fn bounded_control_seed_fields(
    ctx: &BoundedControlSeedCtx<'_>,
    node: u32,
    base_offset: usize,
    scalar_value_override: Option<u64>,
    fields: &mut Vec<ControlSeedField>,
    stack: &mut Vec<u32>,
) {
    if stack.contains(&node) {
        return;
    }
    let Some(line) = metadata_node_line(ctx.ll, node) else {
        return;
    };
    let Some(payload) = metadata_payload(line) else {
        return;
    };
    let tokens = metadata_tokens(payload);
    stack.push(node);
    let mut i = 0;
    let mut pending_nested_node = None;
    while i + 3 < tokens.len() {
        if metadata_quoted_token(tokens[i]) == Some("air.struct_type_info") {
            pending_nested_node = tokens.get(i + 1).and_then(|tok| metadata_ref_token(tok));
            i += 2;
            continue;
        }
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
        let field_name = tokens.get(i + 4).and_then(|tok| metadata_quoted_token(tok));
        let offset = base_offset.saturating_add(offset.max(0) as usize);
        let field_byte_size = byte_size.max(0) as usize;
        let repeat_count = usize::try_from(repeat_count).unwrap_or(0).max(1);
        if let Some(size) = bounded_control_field_seed_size(type_name, byte_size) {
            for element in 0..repeat_count {
                let field_offset = offset.saturating_add(element.saturating_mul(field_byte_size));
                if field_offset < ctx.len && field_offset.saturating_add(size) <= ctx.len {
                    fields.push(ControlSeedField {
                        offset: field_offset,
                        size,
                        value: scalar_value_override.or_else(|| {
                            bounded_control_field_seed_value(
                                type_name,
                                ctx.seed_float_one,
                                ctx.seed_int_zero,
                            )
                        }),
                    });
                }
            }
        } else if let Some(nested_node) = pending_nested_node.take() {
            let nested_scalar_value_override =
                result_dimension_struct_field(type_name, field_name).then_some(0);
            for element in 0..repeat_count {
                bounded_control_seed_fields(
                    ctx,
                    nested_node,
                    offset.saturating_add(element.saturating_mul(field_byte_size)),
                    nested_scalar_value_override,
                    fields,
                    stack,
                );
            }
        }
        i += 5;
    }
    stack.pop();
}

fn result_dimension_struct_field(type_name: &str, field_name: Option<&str>) -> bool {
    matches!(field_name, Some("result_dims" | "output_dims"))
        && type_name
            .trim()
            .trim_end_matches('*')
            .ends_with("TensorDimensions")
}

fn scalar_bounded_control_seed_layout(
    buffer_meta_line: &str,
    len: usize,
    stride_control_bufs: &HashSet<u32>,
    fdiv_denominator_control_bufs: &HashSet<u32>,
    local_array_index_control_bufs: &HashSet<u32>,
) -> Vec<ControlSeedField> {
    let Some(type_name) = quoted_metadata_string_after(buffer_meta_line, "air.arg_type_name")
    else {
        return Vec::new();
    };
    let Some(byte_size) = extract_i32_after(buffer_meta_line, "air.arg_type_size") else {
        return Vec::new();
    };
    let Some(size) = bounded_control_field_seed_size(&type_name, byte_size) else {
        return bounded_control_vector_seed_layout(&type_name, byte_size, len);
    };
    if size <= len {
        let loc = extract_i32_after(buffer_meta_line, "air.location_index").map(|v| v as u32);
        let value = loc
            .filter(|loc| stride_control_bufs.contains(loc))
            .map(|_| 1)
            .or_else(|| {
                let seed_float_one =
                    loc.is_some_and(|loc| fdiv_denominator_control_bufs.contains(&loc));
                let seed_int_zero =
                    loc.is_some_and(|loc| local_array_index_control_bufs.contains(&loc));
                bounded_control_field_seed_value(&type_name, seed_float_one, seed_int_zero)
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

fn bounded_control_vector_seed_layout(
    type_name: &str,
    byte_size: i32,
    len: usize,
) -> Vec<ControlSeedField> {
    let Some((lanes, scalar, elem_size)) = float_vector_shape(type_name) else {
        return Vec::new();
    };
    if byte_size as usize != lanes.saturating_mul(elem_size) || byte_size as usize > len {
        return Vec::new();
    }
    (0..lanes)
        .map(|lane| ControlSeedField {
            offset: lane.saturating_mul(elem_size),
            size: elem_size,
            value: bounded_control_float_lane_value(scalar, lane),
        })
        .collect()
}

fn float_vector_shape(type_name: &str) -> Option<(usize, &str, usize)> {
    let trimmed = type_name.trim().trim_end_matches('*').trim();
    let lanes = trimmed
        .chars()
        .last()
        .and_then(|ch| ch.to_digit(10))
        .and_then(|lanes| usize::try_from(lanes).ok())?;
    if !(2..=4).contains(&lanes) {
        return None;
    }
    let scalar = &trimmed[..trimmed.len() - 1];
    let elem_size = match scalar {
        "half" => 2,
        "float" => 4,
        "double" => 8,
        _ => return None,
    };
    Some((lanes, scalar, elem_size))
}

fn bounded_control_float_lane_value(scalar: &str, lane: usize) -> Option<u64> {
    match scalar {
        "half" => [0x3c00, 0x4000, 0x4200, 0x4400]
            .get(lane)
            .map(|value| *value as u64),
        "float" => Some(((lane + 1) as f32).to_bits() as u64),
        "double" => Some(((lane + 1) as f64).to_bits()),
        _ => None,
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

fn bounded_control_field_seed_value(
    type_name: &str,
    _seed_float_one: bool,
    seed_int_zero: bool,
) -> Option<u64> {
    let name = type_name.trim().trim_end_matches('*').trim();
    if seed_int_zero
        && matches!(
            name,
            "uchar" | "char" | "ushort" | "short" | "uint" | "int" | "ulong" | "long"
        )
    {
        return Some(0);
    }
    match name {
        "bool" => Some(0),
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
            if end > bytes.len() {
                continue;
            }
            if let Some(value) = field.value {
                let value_bytes = value.to_le_bytes();
                if matches!(field.size, 1 | 2 | 4 | 8) {
                    bytes[start..end].copy_from_slice(&value_bytes[..field.size]);
                }
            } else if matches!(field.size, 2 | 4) {
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

fn incompatible_multisample_texture_golden(ll: &str, _metal: &MetalRow) -> Option<String> {
    if !(ll.contains("texture2d_ms") || ll.contains("read_texture_2d_ms")) {
        return None;
    }
    Some(
        "metal golden uses AIR multisample texture reads, but the current validation plan records no \
         texture sample count and the macOS oracle binds seeded textures as single-sample resources; \
         rebank with a real multisample texture oracle or drop the row"
            .into(),
    )
}

fn incompatible_function_constant_texture_array_ref_golden(
    ll: &str,
    metal: &MetalRow,
) -> Option<String> {
    if !matches!(
        metal.fc_specialization.as_deref(),
        Some(FC_SPECIALIZATION_ZERO | FC_SPECIALIZATION_VALUES)
    ) {
        return None;
    }
    let has_fc_texture_metadata = ll.lines().any(|line| {
        line.contains(r#""air.function_constant""#)
            && line.contains(r#""air.texture""#)
            && line.contains("texture")
    });
    let mixes_plain_and_array = ll.contains("texture2d<") && ll.contains("texture2d_array<");
    if !has_fc_texture_metadata || !mixes_plain_and_array {
        return None;
    }
    Some(
        "metal golden uses function-constant-selected texture metadata mixing texture2d and \
         texture2d_array image-view families; the Vulkan validation runner cannot bind a single \
         compatible image view for that reflected descriptor set, so rebank with a concrete texture \
         family or drop the row"
            .into(),
    )
}

fn incompatible_function_constant_private_pointer_table_golden(
    ll: &str,
    metal: &MetalRow,
) -> Option<String> {
    if !matches!(
        metal.fc_specialization.as_deref(),
        Some(FC_SPECIALIZATION_ZERO | FC_SPECIALIZATION_VALUES)
    ) || !ll.contains("@air.is_function_constant_defined")
    {
        return None;
    }
    let has_private_constant_array = ll.lines().any(|line| {
        line.contains(" = internal ")
            && line.contains("addrspace(2) constant [")
            && line.contains(" x ")
    });
    let has_private_pointer_phi = ll.lines().any(|line| {
        line.contains(" = phi ptr addrspace(2) ")
            && line.contains("[ @")
            && line.contains(" ], [ @")
    });
    if !has_private_constant_array || !has_private_pointer_phi {
        return None;
    }
    Some(
        "metal golden specializes an AIR function constant whose static initializer selects among \
         private constant-table pointers; the current product path validates before the validation \
         FC-specialization helper can prune that unspecialized pointer phi, so this is not a \
         comparable Vulkan oracle yet"
            .into(),
    )
}

fn incompatible_function_constant_simdgroup_golden(ll: &str, metal: &MetalRow) -> Option<String> {
    if metal.compare == "none"
        || metal.stage.as_deref() != Some("Kernel")
        || !declares_air_function_constants(ll)
        || !matches!(
            metal.fc_specialization.as_deref(),
            Some(FC_SPECIALIZATION_VALUES | FC_SPECIALIZATION_ZERO)
        )
        || !(ll.contains("@air.simd_shuffle")
            || ll.contains("@air.simd_broadcast")
            || ll.contains("@air.simdgroup."))
        || !ll.contains("@air.wg.barrier")
        || !ll.contains("addrspace(3)")
    {
        return None;
    }
    Some(
        "metal golden specializes AIR function constants that enable simdgroup/threadgroup-memory \
         paths; subgroup topology, barriers, and threadgroup scratch scheduling are not a portable \
         byte oracle for this validation row, so rebank or drop the row"
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

fn function_constant_values_for_integer_divisors(ll: &str) -> Vec<(usize, u64)> {
    if !zero_function_constant_feeds_integer_divisor(ll)
        || ll.contains("@air.is_function_constant_defined")
    {
        return Vec::new();
    }
    declared_function_constant_indices(ll)
        .into_iter()
        .filter(|(ty, _)| is_integer_function_constant_type(ty))
        .map(|(_, index)| (index, 1))
        .collect()
}

fn function_constant_values_for_oracle_inputs(ll: &str) -> Vec<(usize, u64)> {
    let mut values = function_constant_values_for_barrier_loop_progress(ll);
    for value in function_constant_values_for_integer_divisors(ll) {
        if !values.iter().any(|(index, _)| *index == value.0) {
            values.push(value);
        }
    }
    values.sort_by_key(|(index, _)| *index);
    values
}

fn function_constant_values_for_barrier_loop_progress(ll: &str) -> Vec<(usize, u64)> {
    if !ll.contains("@air.wg.barrier(") || ll.contains("@air.is_function_constant_defined") {
        return Vec::new();
    }
    let colliding_conditional_buffers = colliding_conditional_buffer_function_constants(ll);
    let branch_bool_fcs = bool_function_constants_feeding_barrier_branches(ll);
    declared_function_constant_indices(ll)
        .into_iter()
        .filter_map(|(ty, index)| {
            if is_bool_function_constant_type(&ty)
                && branch_bool_fcs.contains(&index)
                && !colliding_conditional_buffers.contains(&index)
            {
                Some((index, 1))
            } else if is_integer_function_constant_type(&ty) {
                Some((index, 2))
            } else {
                None
            }
        })
        .collect()
}

#[derive(Debug)]
struct DeclaredFunctionConstant {
    ty: String,
    index: usize,
    global: String,
}

fn declared_function_constant_indices(ll: &str) -> Vec<(String, usize)> {
    declared_function_constants(ll)
        .into_iter()
        .map(|fc| (fc.ty, fc.index))
        .collect()
}

fn declared_function_constants(ll: &str) -> Vec<DeclaredFunctionConstant> {
    let node_ids = ll
        .lines()
        .find_map(|line| {
            let rest = line.trim().strip_prefix("!air.function_constants = !{")?;
            let rest = rest.strip_suffix('}')?;
            Some(
                rest.split(',')
                    .map(|s| s.trim().trim_start_matches('!'))
                    .filter(|s| !s.is_empty())
                    .collect::<Vec<_>>(),
            )
        })
        .unwrap_or_default();
    let mut out = Vec::new();
    for id in node_ids {
        let prefix = format!("!{id} = !{{");
        let Some(node) = ll
            .lines()
            .find(|line| line.trim_start().starts_with(&prefix))
        else {
            continue;
        };
        let mut quoted = node.split("!\"").skip(1).map(|s| s.split('"').next());
        let Some(Some(ty)) = quoted.next() else {
            continue;
        };
        let Some(index) = node.split("i32 ").nth(1).and_then(|s| {
            s.chars()
                .take_while(|c| c.is_ascii_digit())
                .collect::<String>()
                .parse::<usize>()
                .ok()
        }) else {
            continue;
        };
        let Some(global) = global_name_after(node, "ptr addrspace(") else {
            continue;
        };
        out.push(DeclaredFunctionConstant {
            ty: ty.to_string(),
            index,
            global,
        });
    }
    out
}

fn is_integer_function_constant_type(ty: &str) -> bool {
    matches!(
        ty,
        "char"
            | "uchar"
            | "short"
            | "ushort"
            | "int"
            | "uint"
            | "long"
            | "ulong"
            | "char2"
            | "uchar2"
            | "short2"
            | "ushort2"
            | "int2"
            | "uint2"
            | "long2"
            | "ulong2"
            | "char3"
            | "uchar3"
            | "short3"
            | "ushort3"
            | "int3"
            | "uint3"
            | "long3"
            | "ulong3"
            | "char4"
            | "uchar4"
            | "short4"
            | "ushort4"
            | "int4"
            | "uint4"
            | "long4"
            | "ulong4"
    )
}

fn is_bool_function_constant_type(ty: &str) -> bool {
    matches!(ty, "bool" | "bool2" | "bool3" | "bool4")
}

fn bool_function_constants_feeding_barrier_branches(ll: &str) -> HashSet<usize> {
    let global_sources = bool_function_constant_global_sources(ll);
    if global_sources.is_empty() {
        return HashSet::new();
    }
    let mut out = HashSet::new();
    let mut body = Vec::new();
    let mut in_func = false;
    for line in ll.lines() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("define ") {
            in_func = true;
            body.clear();
        }
        if in_func {
            body.push(line);
            if trimmed == "}" {
                collect_barrier_branch_bool_sources(&body, &global_sources, &mut out);
                in_func = false;
            }
        }
    }
    out
}

fn collect_barrier_branch_bool_sources(
    body: &[&str],
    global_sources: &HashMap<String, usize>,
    out: &mut HashSet<usize>,
) {
    if !body.iter().any(|line| line.contains("@air.wg.barrier(")) {
        return;
    }
    let mut reg_sources: HashMap<String, HashSet<usize>> = HashMap::new();
    for line in body {
        let Some((reg, rhs)) = split_assign(line) else {
            let trimmed = line.trim_start();
            if trimmed.starts_with("br i1") {
                if let Some(cond) = first_percent_reg(trimmed) {
                    if let Some(sources) = reg_sources.get(cond) {
                        out.extend(sources.iter().copied());
                    }
                }
            }
            continue;
        };
        let mut sources = HashSet::new();
        if rhs.starts_with("load ") {
            if let Some(global) = global_name_after(rhs, "ptr addrspace(") {
                if let Some(index) = global_sources.get(&global) {
                    sources.insert(*index);
                }
            }
        }
        for operand in percent_operands(rhs) {
            if operand == reg {
                continue;
            }
            if let Some(prev) = reg_sources.get(operand) {
                sources.extend(prev.iter().copied());
            }
        }
        if !sources.is_empty() {
            reg_sources.insert(reg.to_string(), sources);
        }
    }
}

fn bool_function_constant_global_sources(ll: &str) -> HashMap<String, usize> {
    let fc_global_to_index: HashMap<String, usize> = declared_function_constants(ll)
        .into_iter()
        .filter(|fc| is_bool_function_constant_type(&fc.ty))
        .map(|fc| (fc.global, fc.index))
        .collect();
    let mut value_sources: HashMap<String, usize> = HashMap::new();
    let mut global_sources = fc_global_to_index.clone();
    for line in ll.lines() {
        if let Some((result, global)) = load_result_and_global(line) {
            if let Some(index) = global_sources.get(&global) {
                value_sources.insert(ssa_key(&result).to_string(), *index);
            }
            continue;
        }
        if let Some((reg, rhs)) = split_assign(line) {
            let mut sources = HashSet::new();
            for operand in percent_operands(rhs) {
                if operand == reg {
                    continue;
                }
                if let Some(index) = value_sources.get(operand) {
                    sources.insert(*index);
                }
            }
            if sources.len() == 1 {
                value_sources.insert(reg.to_string(), *sources.iter().next().unwrap());
            }
            continue;
        }
        let Some((stored_value, dest_global)) = store_value_and_dest_global(line) else {
            continue;
        };
        if let Some(index) = value_sources.get(ssa_key(&stored_value)) {
            global_sources.insert(dest_global, *index);
        }
    }
    global_sources
}

fn colliding_conditional_buffer_function_constants(ll: &str) -> HashSet<usize> {
    let pred_to_fc = predicate_global_to_fc_initializer_global(ll);
    if pred_to_fc.is_empty() {
        return HashSet::new();
    }
    let fc_to_index: HashMap<String, usize> = declared_function_constants(ll)
        .into_iter()
        .map(|fc| (fc.global, fc.index))
        .collect();
    colliding_conditional_buffer_predicates(ll)
        .into_iter()
        .filter_map(|pred| {
            let fc_global = pred_to_fc.get(&pred)?;
            fc_to_index.get(fc_global).copied()
        })
        .collect()
}

fn colliding_conditional_buffer_predicates(ll: &str) -> HashSet<String> {
    let mut unconditional_locations = HashSet::new();
    let mut conditional = Vec::new();
    for line in ll.lines() {
        if !line.contains("air.buffer") || !line.contains("air.location_index") {
            continue;
        }
        if line.contains("air.texture") || line.contains("air.sampler") {
            continue;
        }
        if extract_i32_after(line, "air.address_space") == Some(3) {
            continue;
        }
        let Some(range) = metadata_location_range(line) else {
            continue;
        };
        if let Some(pred) = conditional_resource_predicate_global(ll, line) {
            conditional.push((pred, range));
        } else {
            unconditional_locations.extend(range);
        }
    }
    conditional
        .into_iter()
        .filter(|(_, range)| {
            range
                .clone()
                .any(|loc| unconditional_locations.contains(&loc))
        })
        .map(|(pred, _)| pred)
        .collect()
}

fn metadata_location_range(line: &str) -> Option<std::ops::Range<u32>> {
    let start =
        extract_i32_after(line, "air.location_index").and_then(|loc| u32::try_from(loc).ok())?;
    let count = literal_location_index_count(line).unwrap_or(1).max(1);
    Some(start..start.saturating_add(count))
}

fn conditional_resource_predicate_global(ll: &str, line: &str) -> Option<String> {
    let node = conditional_resource_predicate_node(line)?;
    let pred_line = metadata_node_line(ll, node)?;
    global_name_after(pred_line, "ptr addrspace(")
}

fn conditional_resource_predicate_node(line: &str) -> Option<u32> {
    let payload = metadata_payload(line)?;
    let tokens = metadata_tokens(payload);
    tokens.windows(2).find_map(|window| {
        (metadata_quoted_token(window[0]) == Some("air.function_constant"))
            .then(|| metadata_ref_token(window[1]))
            .flatten()
    })
}

fn predicate_global_to_fc_initializer_global(ll: &str) -> HashMap<String, String> {
    let mut value_sources: HashMap<String, String> = HashMap::new();
    let mut out = HashMap::new();
    for line in ll.lines() {
        if let Some((result, global)) = load_result_and_global(line) {
            if global.contains(".MTL_FC_INIT_") {
                value_sources.insert(ssa_key(&result).to_string(), global);
            }
            continue;
        }
        if line.contains("@air.normalize_function_constant_predicate") {
            if let Some(result) = instruction_result_name(line) {
                let result_key = ssa_key(&result);
                if let Some(source) = percent_operands(line)
                    .into_iter()
                    .find(|operand| *operand != result_key)
                    .and_then(|operand| value_sources.get(operand))
                    .cloned()
                {
                    value_sources.insert(result_key.to_string(), source);
                }
            }
            continue;
        }
        let Some((stored_value, dest_global)) = store_value_and_dest_global(line) else {
            continue;
        };
        if let Some(source) = value_sources.get(ssa_key(&stored_value)) {
            out.insert(dest_global, source.clone());
        }
    }
    out
}

fn ssa_key(value: &str) -> &str {
    value.trim().trim_start_matches('%')
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
    if current.kind == "buffer" {
        let current_buffer = current_plan
            .buffers
            .iter()
            .find(|buffer| buffer.index == current.index);
        let banked_buffer = metal
            .plan
            .buffers
            .iter()
            .find(|buffer| buffer.index == banked.index);
        if let (Some(current_buffer), Some(banked_buffer)) = (current_buffer, banked_buffer) {
            if output_buffer_plan_differs(current_buffer, banked_buffer) {
                return Some(format!(
                    "metal golden output buffer plan {} differs from current AIR output buffer \
                     plan {}; rebank Metal row",
                    plan_buffer_summary(banked_buffer),
                    plan_buffer_summary(current_buffer),
                ));
            }
        }
    }
    None
}

fn output_buffer_plan_differs(current: &PlanBuffer, banked: &PlanBuffer) -> bool {
    current.len != banked.len
        || current.role != banked.role
        || current.seed_mode != banked.seed_mode
        || current.seed_layout != banked.seed_layout
        || current.seed_stride != banked.seed_stride
}

fn plan_buffer_summary(buffer: &PlanBuffer) -> String {
    format!(
        "buffer {} role={} len={} seed={} stride={:?} layout_fields={}",
        buffer.index,
        buffer.role,
        buffer.len,
        buffer.seed_mode,
        buffer.seed_stride,
        buffer.seed_layout.len()
    )
}

fn incompatible_static_resource_plan_golden(ll: &str, metal: &MetalRow) -> Option<String> {
    if !matches!(
        metal.fc_specialization.as_deref(),
        Some(FC_SPECIALIZATION_ZERO | FC_SPECIALIZATION_VALUES)
    ) {
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

#[derive(Default)]
struct OwnedLoopInputFacts {
    fc_values: Vec<(usize, u64)>,
    arg_values: Vec<(String, i128)>,
    arg_float_values: Vec<(String, f64)>,
    arg_field_values: Vec<(String, Vec<i32>, i128)>,
    arg_upper_bounds: Vec<(String, i128)>,
    arg_vector_values: Vec<(String, usize, i128)>,
    arg_vector_upper_bounds: Vec<(String, usize, i128)>,
    texture_extents: Vec<(String, [i128; 3])>,
    imageblock_extent: Option<[i128; 2]>,
}

impl OwnedLoopInputFacts {
    fn as_loop_input_facts(&self) -> crate::loop_budget::LoopInputFacts<'_> {
        crate::loop_budget::LoopInputFacts {
            fc_values: &self.fc_values,
            arg_values: &self.arg_values,
            arg_float_values: &self.arg_float_values,
            arg_field_values: &self.arg_field_values,
            arg_upper_bounds: &self.arg_upper_bounds,
            arg_vector_values: &self.arg_vector_values,
            arg_vector_upper_bounds: &self.arg_vector_upper_bounds,
            texture_extents: &self.texture_extents,
            imageblock_extent: self.imageblock_extent,
        }
    }
}

fn loop_input_facts_for_metal_plan(ll: &str, entry: &str, metal: &MetalRow) -> OwnedLoopInputFacts {
    let arg_names = entry_arg_names(ll, entry);
    if arg_names.is_empty() {
        return OwnedLoopInputFacts::default();
    }
    let owned_inputs = plan_to_owned_inputs(&metal.plan).ok();
    let fc_values = metal
        .fc_values
        .as_deref()
        .filter(|values| !values.is_empty())
        .map(|values| {
            values
                .iter()
                .map(|value| (value.index as usize, value.value))
                .collect()
        })
        .unwrap_or_else(|| function_constant_values_for_oracle_inputs(ll));
    let mut facts = OwnedLoopInputFacts {
        fc_values,
        arg_values: exact_launch_scalar_arg_values(ll, &arg_names, &metal.plan),
        arg_float_values: Vec::new(),
        arg_field_values: Vec::new(),
        arg_upper_bounds: launch_scalar_arg_upper_bounds(ll, &arg_names, &metal.plan),
        arg_vector_values: exact_launch_vector_arg_values(ll, &arg_names, &metal.plan),
        arg_vector_upper_bounds: launch_vector_arg_upper_bounds(ll, &arg_names, &metal.plan),
        texture_extents: exact_texture_extent_values(ll, &arg_names, &metal.plan),
        imageblock_extent: exact_imageblock_extent_value(ll, &metal.plan),
    };
    if let Some(owned_inputs) = &owned_inputs {
        facts.arg_values.extend(exact_scalar_buffer_arg_values(
            ll,
            &arg_names,
            &owned_inputs.inputs,
        ));
        facts.arg_float_values.extend(exact_float_buffer_arg_values(
            ll,
            &arg_names,
            &owned_inputs.inputs,
        ));
        facts
            .arg_field_values
            .extend(exact_struct_buffer_arg_field_values(
                ll,
                &arg_names,
                &owned_inputs.inputs,
            ));
        facts
            .arg_vector_values
            .extend(exact_vector_buffer_arg_values(
                ll,
                &arg_names,
                &owned_inputs.inputs,
            ));
    }
    facts
}

fn exact_launch_scalar_arg_values(
    ll: &str,
    arg_names: &[String],
    plan: &HarnessPlan,
) -> Vec<(String, i128)> {
    let mut out = Vec::new();
    for line in ll.lines() {
        let Some(arg_ord) = metadata_param_index(line).and_then(|v| usize::try_from(v).ok()) else {
            continue;
        };
        let Some(arg_name) = arg_names.get(arg_ord) else {
            continue;
        };
        let Some(type_name) = quoted_metadata_string_after(line, "air.arg_type_name") else {
            continue;
        };
        if !is_scalar_integer_air_type(&type_name) {
            continue;
        }
        let degenerate_position = (line.contains(r#""air.thread_position_in_threadgroup""#)
            && plan.dispatch_tg[0] <= 1)
            || (line.contains(r#""air.thread_position_in_grid""#) && plan.dispatch_grid[0] <= 1)
            || (line.contains(r#""air.threadgroup_position_in_grid""#)
                && div_ceil_nonzero(plan.dispatch_grid[0], plan.dispatch_tg[0]) <= 1);
        let value = if line.contains(r#""air.threads_per_threadgroup""#) {
            Some(plan.dispatch_tg[0])
        } else if line.contains(r#""air.threads_per_grid""#) {
            Some(plan.dispatch_grid[0])
        } else if line.contains(r#""air.threadgroups_per_grid""#) {
            Some(div_ceil_nonzero(plan.dispatch_grid[0], plan.dispatch_tg[0]))
        } else if line.contains(r#""air.threads_per_simdgroup""#) {
            Some(32)
        } else if line.contains(r#""air.simdgroups_per_threadgroup""#) {
            Some(div_ceil_nonzero(plan.dispatch_tg[0], 32))
        } else if degenerate_position {
            Some(0)
        } else {
            None
        };
        if let Some(value) = value {
            out.push((arg_name.clone(), i128::from(value)));
        }
    }
    out
}

fn launch_scalar_arg_upper_bounds(
    ll: &str,
    arg_names: &[String],
    plan: &HarnessPlan,
) -> Vec<(String, i128)> {
    let mut out = Vec::new();
    for line in ll.lines() {
        let Some(arg_ord) = metadata_param_index(line).and_then(|v| usize::try_from(v).ok()) else {
            continue;
        };
        let Some(arg_name) = arg_names.get(arg_ord) else {
            continue;
        };
        let Some(type_name) = quoted_metadata_string_after(line, "air.arg_type_name") else {
            continue;
        };
        if !is_scalar_integer_air_type(&type_name) {
            continue;
        }
        let value = if line.contains(r#""air.thread_position_in_threadgroup""#) {
            Some(plan.dispatch_tg[0].saturating_sub(1))
        } else if line.contains(r#""air.thread_position_in_grid""#) {
            Some(plan.dispatch_grid[0].saturating_sub(1))
        } else if line.contains(r#""air.threadgroup_position_in_grid""#) {
            Some(div_ceil_nonzero(plan.dispatch_grid[0], plan.dispatch_tg[0]).saturating_sub(1))
        } else {
            None
        };
        if let Some(value) = value {
            out.push((arg_name.clone(), i128::from(value)));
        }
    }
    out
}

fn exact_launch_vector_arg_values(
    ll: &str,
    arg_names: &[String],
    plan: &HarnessPlan,
) -> Vec<(String, usize, i128)> {
    let mut out = Vec::new();
    for line in ll.lines() {
        let Some(arg_ord) = metadata_param_index(line).and_then(|v| usize::try_from(v).ok()) else {
            continue;
        };
        let Some(arg_name) = arg_names.get(arg_ord) else {
            continue;
        };
        let Some(type_name) = quoted_metadata_string_after(line, "air.arg_type_name") else {
            continue;
        };
        let Some(lanes) = integer_vector_lane_count(&type_name) else {
            continue;
        };
        for lane in 0..lanes {
            let value = if line.contains(r#""air.threads_per_threadgroup""#) {
                dispatch_lane(plan.dispatch_tg, lane)
            } else if line.contains(r#""air.threads_per_grid""#) {
                dispatch_lane(plan.dispatch_grid, lane)
            } else if line.contains(r#""air.threadgroups_per_grid""#) {
                Some(threadgroups_per_grid_lane(plan, lane))
            } else if line.contains(r#""air.thread_position_in_threadgroup""#) {
                dispatch_lane(plan.dispatch_tg, lane)
                    .filter(|value| *value <= 1)
                    .map(|_| 0)
            } else if line.contains(r#""air.thread_position_in_grid""#) {
                dispatch_lane(plan.dispatch_grid, lane)
                    .filter(|value| *value <= 1)
                    .map(|_| 0)
            } else if line.contains(r#""air.threadgroup_position_in_grid""#) {
                (threadgroups_per_grid_lane(plan, lane) <= 1).then_some(0)
            } else {
                None
            };
            if let Some(value) = value {
                out.push((arg_name.clone(), lane, i128::from(value)));
            }
        }
    }
    out
}

fn exact_vector_buffer_arg_values(
    ll: &str,
    arg_names: &[String],
    inputs: &Inputs,
) -> Vec<(String, usize, i128)> {
    let input_bytes = inputs
        .buffers
        .iter()
        .filter_map(|buffer| match buffer.seed {
            Seed::ExactBytes { bytes, .. } => Some((buffer.index, bytes)),
            _ => None,
        })
        .collect::<HashMap<_, _>>();
    if input_bytes.is_empty() {
        return Vec::new();
    }
    let locations = stage_resource_locations(ll);
    let mut out = Vec::new();
    for line in ll.lines() {
        let readable = line.contains(r#""air.read""#) || line.contains(r#""air.read_write""#);
        if !line.contains(r#""air.buffer""#)
            || !line.contains(r#""air.location_index""#)
            || !readable
        {
            continue;
        }
        let Some(arg_ord) = metadata_param_index(line).and_then(|v| usize::try_from(v).ok()) else {
            continue;
        };
        let Some(arg_name) = arg_names.get(arg_ord) else {
            continue;
        };
        let Some(raw_loc) = extract_i32_after(line, "air.location_index").map(|v| v as u32) else {
            continue;
        };
        let Some(type_name) = quoted_metadata_string_after(line, "air.arg_type_name") else {
            continue;
        };
        let Some((lanes, scalar_type, elem_size)) = integer_vector_shape(&type_name) else {
            continue;
        };
        let Some(bytes) = locations
            .buffers
            .get(&(arg_ord as u32))
            .and_then(|loc| input_bytes.get(loc))
            .or_else(|| input_bytes.get(&raw_loc))
        else {
            continue;
        };
        for lane in 0..lanes {
            let offset = lane.saturating_mul(elem_size);
            if let Some(value) =
                scalar_int_from_bytes(bytes.get(offset..).unwrap_or(&[]), scalar_type)
            {
                out.push((arg_name.clone(), lane, value));
            }
        }
    }
    out
}

fn launch_vector_arg_upper_bounds(
    ll: &str,
    arg_names: &[String],
    plan: &HarnessPlan,
) -> Vec<(String, usize, i128)> {
    let mut out = Vec::new();
    for line in ll.lines() {
        let Some(arg_ord) = metadata_param_index(line).and_then(|v| usize::try_from(v).ok()) else {
            continue;
        };
        let Some(arg_name) = arg_names.get(arg_ord) else {
            continue;
        };
        let Some(type_name) = quoted_metadata_string_after(line, "air.arg_type_name") else {
            continue;
        };
        let Some(lanes) = integer_vector_lane_count(&type_name) else {
            continue;
        };
        for lane in 0..lanes {
            let value = if line.contains(r#""air.thread_position_in_threadgroup""#) {
                dispatch_lane(plan.dispatch_tg, lane).map(|value| value.saturating_sub(1))
            } else if line.contains(r#""air.thread_position_in_grid""#) {
                dispatch_lane(plan.dispatch_grid, lane).map(|value| value.saturating_sub(1))
            } else if line.contains(r#""air.threadgroup_position_in_grid""#) {
                Some(threadgroups_per_grid_lane(plan, lane).saturating_sub(1))
            } else {
                None
            };
            if let Some(value) = value {
                out.push((arg_name.clone(), lane, i128::from(value)));
            }
        }
    }
    out
}

fn exact_scalar_buffer_arg_values(
    ll: &str,
    arg_names: &[String],
    inputs: &Inputs,
) -> Vec<(String, i128)> {
    let input_bytes = inputs
        .buffers
        .iter()
        .filter_map(|buffer| match buffer.seed {
            Seed::ExactBytes { bytes, .. } => Some((buffer.index, bytes)),
            _ => None,
        })
        .collect::<HashMap<_, _>>();
    if input_bytes.is_empty() {
        return Vec::new();
    }
    let locations = stage_resource_locations(ll);
    let mut out = Vec::new();
    for line in ll.lines() {
        let readable = line.contains(r#""air.read""#) || line.contains(r#""air.read_write""#);
        if !line.contains(r#""air.buffer""#)
            || !line.contains(r#""air.location_index""#)
            || !readable
        {
            continue;
        }
        let Some(arg_ord) = metadata_param_index(line).and_then(|v| usize::try_from(v).ok()) else {
            continue;
        };
        let Some(arg_name) = arg_names.get(arg_ord) else {
            continue;
        };
        let Some(raw_loc) = extract_i32_after(line, "air.location_index").map(|v| v as u32) else {
            continue;
        };
        let Some(type_name) = quoted_metadata_string_after(line, "air.arg_type_name") else {
            continue;
        };
        let Some(bytes) = locations
            .buffers
            .get(&(arg_ord as u32))
            .and_then(|loc| input_bytes.get(loc))
            .or_else(|| input_bytes.get(&raw_loc))
        else {
            continue;
        };
        let Some(value) = scalar_int_from_bytes(bytes, &type_name) else {
            continue;
        };
        out.push((arg_name.clone(), value));
    }
    out
}

fn exact_float_buffer_arg_values(
    ll: &str,
    arg_names: &[String],
    inputs: &Inputs,
) -> Vec<(String, f64)> {
    let input_bytes = inputs
        .buffers
        .iter()
        .filter_map(|buffer| match buffer.seed {
            Seed::ExactBytes { bytes, .. } => Some((buffer.index, bytes)),
            _ => None,
        })
        .collect::<HashMap<_, _>>();
    if input_bytes.is_empty() {
        return Vec::new();
    }
    let locations = stage_resource_locations(ll);
    let mut out = Vec::new();
    for line in ll.lines() {
        if !line.contains(r#""air.buffer""#)
            || !line.contains(r#""air.location_index""#)
            || !line.contains(r#""air.address_space", i32 2"#)
        {
            continue;
        }
        let Some(arg_ord) = metadata_param_index(line).and_then(|v| usize::try_from(v).ok()) else {
            continue;
        };
        let Some(arg_name) = arg_names.get(arg_ord) else {
            continue;
        };
        let Some(raw_loc) = extract_i32_after(line, "air.location_index").map(|v| v as u32) else {
            continue;
        };
        let Some(type_name) = quoted_metadata_string_after(line, "air.arg_type_name") else {
            continue;
        };
        let Some(bytes) = locations
            .buffers
            .get(&(arg_ord as u32))
            .and_then(|loc| input_bytes.get(loc))
            .or_else(|| input_bytes.get(&raw_loc))
        else {
            continue;
        };
        let Some(value) = scalar_float_from_bytes(bytes, &type_name) else {
            continue;
        };
        out.push((arg_name.clone(), value));
    }
    out
}

fn exact_struct_buffer_arg_field_values(
    ll: &str,
    arg_names: &[String],
    inputs: &Inputs,
) -> Vec<(String, Vec<i32>, i128)> {
    let input_bytes = inputs
        .buffers
        .iter()
        .filter_map(|buffer| match buffer.seed {
            Seed::ExactBytes { bytes, .. } => Some((buffer.index, bytes)),
            _ => None,
        })
        .collect::<HashMap<_, _>>();
    if input_bytes.is_empty() {
        return Vec::new();
    }
    let locations = stage_resource_locations(ll);
    let mut out = Vec::new();
    for line in ll.lines() {
        if !line.contains(r#""air.buffer""#)
            || !line.contains(r#""air.location_index""#)
            || !line.contains(r#""air.address_space", i32 2"#)
        {
            continue;
        }
        let Some(arg_ord) = metadata_param_index(line).and_then(|v| usize::try_from(v).ok()) else {
            continue;
        };
        let Some(arg_name) = arg_names.get(arg_ord) else {
            continue;
        };
        let Some(node) = metadata_ref_after(line, "air.struct_type_info") else {
            continue;
        };
        let Some(raw_loc) = extract_i32_after(line, "air.location_index").map(|v| v as u32) else {
            continue;
        };
        let Some(bytes) = locations
            .buffers
            .get(&(arg_ord as u32))
            .and_then(|loc| input_bytes.get(loc))
            .or_else(|| input_bytes.get(&raw_loc))
        else {
            continue;
        };
        let mut ctx = StructFieldCollectCtx {
            ll,
            bytes,
            arg_name,
            out: &mut out,
        };
        collect_struct_int_field_values(&mut ctx, node, 0, &[], &mut Vec::new());
    }
    out
}

struct StructFieldCollectCtx<'a, 'b> {
    ll: &'a str,
    bytes: &'a [u8],
    arg_name: &'a str,
    out: &'b mut Vec<(String, Vec<i32>, i128)>,
}

fn collect_struct_int_field_values(
    ctx: &mut StructFieldCollectCtx<'_, '_>,
    node: u32,
    base_offset: usize,
    base_path: &[i32],
    stack: &mut Vec<u32>,
) {
    if stack.contains(&node) {
        return;
    }
    let Some(line) = metadata_node_line(ctx.ll, node) else {
        return;
    };
    let Some(payload) = metadata_payload(line) else {
        return;
    };
    let tokens = metadata_tokens(payload);
    stack.push(node);
    let mut i = 0;
    let mut field_index = 0i32;
    let mut pending_nested_node = None;
    while i + 3 < tokens.len() {
        if metadata_quoted_token(tokens[i]) == Some("air.struct_type_info") {
            pending_nested_node = tokens.get(i + 1).and_then(|tok| metadata_ref_token(tok));
            i += 2;
            continue;
        }
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
        let offset = base_offset.saturating_add(offset.max(0) as usize);
        let field_byte_size = byte_size.max(0) as usize;
        let repeat_count = usize::try_from(repeat_count).unwrap_or(0).max(1);
        let mut field_path = base_path.to_vec();
        field_path.push(field_index);
        if is_scalar_integer_air_type(type_name) {
            for element in 0..repeat_count {
                let field_offset = offset.saturating_add(element.saturating_mul(field_byte_size));
                let end = field_offset.saturating_add(field_byte_size);
                if end > ctx.bytes.len() {
                    continue;
                }
                let mut path = field_path.clone();
                if repeat_count > 1 {
                    path.push(element as i32);
                }
                if let Some(value) = scalar_int_from_bytes(&ctx.bytes[field_offset..end], type_name)
                {
                    ctx.out.push((ctx.arg_name.to_string(), path, value));
                }
            }
        } else if let Some(nested_node) = pending_nested_node.take() {
            for element in 0..repeat_count {
                let nested_offset = offset.saturating_add(element.saturating_mul(field_byte_size));
                let mut nested_path = field_path.clone();
                if repeat_count > 1 {
                    nested_path.push(element as i32);
                }
                collect_struct_int_field_values(
                    ctx,
                    nested_node,
                    nested_offset,
                    &nested_path,
                    stack,
                );
            }
        }
        field_index += 1;
        i += 5;
    }
    stack.pop();
}

fn exact_texture_extent_values(
    ll: &str,
    arg_names: &[String],
    plan: &HarnessPlan,
) -> Vec<(String, [i128; 3])> {
    let input_extents = plan
        .textures
        .iter()
        .map(|texture| {
            (
                texture.index,
                [
                    i128::from(texture.w),
                    i128::from(texture.h),
                    i128::from(texture.d),
                ],
            )
        })
        .collect::<HashMap<_, _>>();
    if input_extents.is_empty() {
        return Vec::new();
    }
    let locations = stage_resource_locations(ll);
    let mut out = Vec::new();
    for line in ll.lines() {
        if !line.contains(r#""air.texture""#) || !line.contains(r#""air.location_index""#) {
            continue;
        }
        let Some(arg_ord) = metadata_param_index(line).and_then(|v| usize::try_from(v).ok()) else {
            continue;
        };
        let Some(arg_name) = arg_names.get(arg_ord) else {
            continue;
        };
        let Some(raw_loc) = extract_i32_after(line, "air.location_index").map(|v| v as u32) else {
            continue;
        };
        let Some(extent) = locations
            .textures
            .get(&(arg_ord as u32))
            .and_then(|loc| input_extents.get(loc))
            .or_else(|| input_extents.get(&raw_loc))
        else {
            continue;
        };
        out.push((arg_name.clone(), *extent));
    }
    out
}

fn exact_imageblock_extent_value(ll: &str, plan: &HarnessPlan) -> Option<[i128; 2]> {
    (ll.contains("@air.get_imageblock_width(") || ll.contains("@air.get_imageblock_height("))
        .then_some([
            i128::from(plan.dispatch_tg[0]),
            i128::from(plan.dispatch_tg[1]),
        ])
}

fn entry_arg_names(ll: &str, entry: &str) -> Vec<String> {
    let Some(line) = ll.lines().find(|line| {
        line.trim_start().starts_with("define ") && define_line_names_entry(line, entry)
    }) else {
        return Vec::new();
    };
    let Some(args) = define_args(line) else {
        return Vec::new();
    };
    args.split(',')
        .filter_map(|arg| {
            let (_, name) = arg.rsplit_once('%')?;
            let name = name.trim().trim_end_matches(|ch: char| {
                !(ch.is_ascii_alphanumeric() || ch == '_' || ch == '.')
            });
            (!name.is_empty()).then(|| name.to_string())
        })
        .collect()
}

fn define_line_names_entry(line: &str, entry: &str) -> bool {
    let Some(at) = line.find('@') else {
        return false;
    };
    let rest = &line[at + 1..];
    if let Some(quoted) = rest.strip_prefix('"') {
        return quoted
            .strip_prefix(entry)
            .is_some_and(|tail| tail.starts_with('"'));
    }
    rest.strip_prefix(entry).is_some_and(|tail| {
        tail.starts_with('(')
            || tail
                .chars()
                .next()
                .is_some_and(|ch| !(ch.is_ascii_alphanumeric() || ch == '_' || ch == '.'))
    })
}

fn define_args(line: &str) -> Option<&str> {
    let open = line.find('(')?;
    let close = line.rfind(") ")?;
    (open < close).then_some(&line[open + 1..close])
}

fn integer_vector_lane_count(type_name: &str) -> Option<usize> {
    integer_vector_shape(type_name).map(|(lanes, _, _)| lanes)
}

fn integer_vector_shape(type_name: &str) -> Option<(usize, &str, usize)> {
    let trimmed = type_name.trim().trim_end_matches('*').trim();
    let lanes = trimmed
        .chars()
        .last()
        .and_then(|ch| ch.to_digit(10))
        .and_then(|lanes| usize::try_from(lanes).ok())?;
    if !(2..=4).contains(&lanes) {
        return None;
    }
    let scalar = &trimmed[..trimmed.len() - 1];
    let elem_size = scalar_int_type_byte_size(scalar)?;
    Some((lanes, scalar, elem_size))
}

fn scalar_int_type_byte_size(type_name: &str) -> Option<usize> {
    match type_name.trim().trim_end_matches('*').trim() {
        "uchar" | "char" | "bool" => Some(1),
        "ushort" | "short" => Some(2),
        "uint" | "int" => Some(4),
        "ulong" | "long" => Some(8),
        _ => None,
    }
}

fn is_scalar_integer_air_type(type_name: &str) -> bool {
    matches!(
        type_name.trim().trim_end_matches('*').trim(),
        "uchar" | "char" | "bool" | "ushort" | "short" | "uint" | "int" | "ulong" | "long"
    )
}

fn dispatch_lane(values: [u32; 3], lane: usize) -> Option<u32> {
    values.get(lane).copied()
}

fn threadgroups_per_grid_lane(plan: &HarnessPlan, lane: usize) -> u32 {
    let grid = dispatch_lane(plan.dispatch_grid, lane).unwrap_or(1);
    let tg = dispatch_lane(plan.dispatch_tg, lane).unwrap_or(1);
    div_ceil_nonzero(grid, tg)
}

fn div_ceil_nonzero(n: u32, d: u32) -> u32 {
    let d = d.max(1);
    n.max(1).div_ceil(d)
}

fn scalar_int_from_bytes(bytes: &[u8], type_name: &str) -> Option<i128> {
    match type_name.trim().trim_end_matches('*').trim() {
        "bool" | "uchar" => bytes.first().map(|value| i128::from(*value)),
        "char" => bytes.first().map(|value| i128::from(*value as i8)),
        "ushort" => {
            let bytes: [u8; 2] = bytes.get(..2)?.try_into().ok()?;
            Some(i128::from(u16::from_le_bytes(bytes)))
        }
        "short" => {
            let bytes: [u8; 2] = bytes.get(..2)?.try_into().ok()?;
            Some(i128::from(i16::from_le_bytes(bytes)))
        }
        "uint" => {
            let bytes: [u8; 4] = bytes.get(..4)?.try_into().ok()?;
            Some(i128::from(u32::from_le_bytes(bytes)))
        }
        "int" => {
            let bytes: [u8; 4] = bytes.get(..4)?.try_into().ok()?;
            Some(i128::from(i32::from_le_bytes(bytes)))
        }
        "ulong" => {
            let bytes: [u8; 8] = bytes.get(..8)?.try_into().ok()?;
            Some(i128::from(u64::from_le_bytes(bytes)))
        }
        "long" => {
            let bytes: [u8; 8] = bytes.get(..8)?.try_into().ok()?;
            Some(i128::from(i64::from_le_bytes(bytes)))
        }
        _ => None,
    }
}

fn scalar_float_from_bytes(bytes: &[u8], type_name: &str) -> Option<f64> {
    match type_name.trim().trim_end_matches('*').trim() {
        "half" => {
            let bytes: [u8; 2] = bytes.get(..2)?.try_into().ok()?;
            Some(f16_bits_to_f64(u16::from_le_bytes(bytes)))
        }
        "float" => {
            let bytes: [u8; 4] = bytes.get(..4)?.try_into().ok()?;
            Some(f64::from(f32::from_le_bytes(bytes)))
        }
        "double" => {
            let bytes: [u8; 8] = bytes.get(..8)?.try_into().ok()?;
            Some(f64::from_le_bytes(bytes))
        }
        _ => None,
    }
}

fn f16_bits_to_f64(bits: u16) -> f64 {
    let sign = if bits & 0x8000 == 0 { 1.0 } else { -1.0 };
    let exp = (bits >> 10) & 0x1f;
    let frac = bits & 0x03ff;
    match (exp, frac) {
        (0, 0) => sign * 0.0,
        (0, _) => sign * f64::from(frac) * 2f64.powi(-24),
        (0x1f, 0) => sign * f64::INFINITY,
        (0x1f, _) => f64::NAN,
        _ => sign * (1.0 + f64::from(frac) / 1024.0) * 2f64.powi(i32::from(exp) - 15),
    }
}

fn incompatible_compare_none_loop_guard_golden(
    ll: &str,
    entry: &str,
    metal: &MetalRow,
) -> Option<String> {
    if metal.compare != "none" {
        return None;
    }
    let loop_facts = loop_input_facts_for_metal_plan(ll, entry, metal);
    match crate::loop_budget::classify_and_instrument_with_loop_input_facts(
        ll,
        entry,
        loop_facts.as_loop_input_facts(),
    ) {
        crate::loop_budget::GuardPlan::Quarantine(reason) => Some(format!(
            "metal golden compare=none cannot be reproduced by current loop guard: {reason}; \
             rebank Metal row"
        )),
        crate::loop_budget::GuardPlan::Instrumented(_)
        | crate::loop_budget::GuardPlan::LoopFree => None,
    }
}

fn incompatible_compare_none_simdgroup_matrix_smoke_golden(
    ll: &str,
    metal: &MetalRow,
) -> Option<String> {
    if metal.compare != "none" || metal.stage.as_deref() != Some("Kernel") {
        return None;
    }
    if !ll.contains("@air.simdgroup_matrix_8x8_multiply_accumulate.")
        || !ll.contains("@air.wg.barrier")
        || !ll.contains("addrspace(3)")
    {
        return None;
    }
    Some(
        "metal golden compare=none is a smoke-only simdgroup-matrix kernel with workgroup \
         barriers and threadgroup memory; the Vulkan smoke run is not a bounded semantic oracle on \
         this runner, so rebank or drop the row"
            .into(),
    )
}

fn incompatible_compare_none_raytracing_smoke_golden(ll: &str, metal: &MetalRow) -> Option<String> {
    if metal.compare != "none" || metal.stage.as_deref() != Some("Kernel") {
        return None;
    }
    if !(ll.contains(r#""air.instance_acceleration_structure""#)
        || ll.contains("acceleration_structure<"))
        || !ll.contains(r#""air.visible_function_table""#)
        || !ll.contains("raytracing")
    {
        return None;
    }
    Some(
        "metal golden compare=none is a smoke-only AIR raytracing kernel using acceleration \
         structure and visible-function-table ABI; this validation runner has no bounded Vulkan \
         semantic oracle for that row, so rebank or drop it"
            .into(),
    )
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

fn incompatible_subgroup_texture_write_race_golden(ll: &str, metal: &MetalRow) -> Option<String> {
    if metal.compare == "none"
        || metal.plan.output.kind != "texture"
        || metal.plan.dispatch_tg.iter().product::<u32>() <= 32
        || !ll.contains("@air.write_texture")
        || !(ll.contains("@air.simd_min.")
            || ll.contains("@air.simd_max.")
            || ll.contains("@air.simd_sum.")
            || ll.contains("@air.simd_prefix_"))
        || !ll.contains(r#""air.thread_index_in_simdgroup""#)
        || !ll.contains(r#""air.threadgroup_position_in_grid""#)
    {
        return None;
    }
    if !declares_lane_zero_subgroup_texture_write(ll) {
        return None;
    }
    Some(
        "metal golden compares a subgroup-reduction texture write gated only by \
         thread_index_in_simdgroup==0 while the workgroup contains multiple simdgroups; output \
         coordinates derive from threadgroup_position_in_grid, so multiple simdgroups race on the \
         same texels and the winner is backend-schedule-dependent"
            .into(),
    )
}

fn incompatible_many_to_one_texture_write_golden(ll: &str, metal: &MetalRow) -> Option<String> {
    if metal.compare == "none"
        || metal.stage.as_deref() != Some("Kernel")
        || metal.plan.output.kind != "texture"
        || !ll.contains(r#""air.thread_position_in_grid""#)
        || !ll.contains("@air.write_texture_2d")
    {
        return None;
    }
    let reduced_coords: HashSet<&str> = ll
        .lines()
        .filter_map(|line| {
            let (result, rhs) = line.split_once(" = ")?;
            let rhs = rhs.trim_start();
            let is_many_to_one = (rhs.starts_with("lshr <2 x i32> %")
                || rhs.starts_with("ashr <2 x i32> %")
                || rhs.starts_with("udiv <2 x i32> %"))
                && (rhs.contains("splat (i32 1)") || rhs.contains("<i32 1, i32 1>"));
            is_many_to_one.then_some(result.trim())
        })
        .collect();
    if reduced_coords.is_empty() {
        if !texture_array_write_omits_grid_lane(ll, &metal.plan) {
            return None;
        }
        return Some(
            "metal golden writes a texture2d_array output from multiple kernel lanes whose write \
             coordinates omit a varying grid dimension; multiple lanes race on the same texels and \
             the winner is backend-schedule-dependent, so rebank with a one-to-one launch/output \
             plan or drop the row"
                .into(),
        );
    }
    let uses_reduced_coord = ll.lines().any(|line| {
        line.contains("@air.write_texture_2d")
            && reduced_coords
                .iter()
                .any(|coord| line.contains(&format!("<2 x i32> {coord}")))
    });
    if !uses_reduced_coord {
        return None;
    }
    Some(
        "metal golden writes a texture from multiple kernel lanes to the same downscaled output \
         coordinate; the write race is order-dependent across Metal/Vulkan runners, so rebank with \
         a one-to-one launch/output plan or drop the row"
            .into(),
    )
}

fn texture_array_write_omits_grid_lane(ll: &str, plan: &HarnessPlan) -> bool {
    if !ll.contains("@air.write_texture_2d_array") {
        return false;
    }
    let mut coord_masks: HashMap<String, Vec<usize>> = HashMap::new();
    let mut lane_extracts: HashMap<String, usize> = HashMap::new();
    for line in ll.lines().map(str::trim) {
        let Some((result, rhs)) = split_assign(line) else {
            continue;
        };
        if rhs.starts_with("shufflevector <3 x i32> %") {
            if let Some(mask) = parse_i32_vector_literal(rhs).filter(|mask| mask.len() == 2) {
                coord_masks.insert(result.to_string(), mask);
            }
            continue;
        }
        if rhs.starts_with("extractelement <3 x i32> %") {
            if let Some(lane) = extractelement_i32_lane(rhs) {
                lane_extracts.insert(result.to_string(), lane);
            }
        }
    }
    if coord_masks.is_empty() {
        return false;
    }
    ll.lines()
        .filter(|line| line.contains("@air.write_texture_2d_array"))
        .any(|line| {
            coord_masks.iter().any(|(coord, mask)| {
                if !line.contains(&format!("<2 x i32> %{coord}")) {
                    return false;
                }
                (0..3).any(|lane| {
                    !mask.contains(&lane)
                        && plan.dispatch_grid.get(lane).copied().unwrap_or(1) > 1
                        && !lane_extracts.iter().any(|(reg, reg_lane)| {
                            *reg_lane == lane && line.contains(&format!("%{reg}"))
                        })
                })
            })
        })
}

fn parse_i32_vector_literal(rhs: &str) -> Option<Vec<usize>> {
    let start = rhs.rfind("<i32 ")?;
    let rest = &rhs[start + 1..];
    let (literal, _) = rest.split_once('>')?;
    literal
        .split(", ")
        .map(|part| {
            part.trim()
                .strip_prefix("i32 ")
                .and_then(|n| n.parse::<usize>().ok())
        })
        .collect()
}

fn extractelement_i32_lane(rhs: &str) -> Option<usize> {
    let lane = rhs
        .rsplit_once(", i64 ")
        .or_else(|| rhs.rsplit_once(", i32 "))?
        .1;
    lane.split(|ch: char| !ch.is_ascii_digit())
        .next()?
        .parse()
        .ok()
}

fn declares_lane_zero_subgroup_texture_write(ll: &str) -> bool {
    let Some(args) = primary_entry_function_args(ll) else {
        return false;
    };
    let lane_args = arg_names_for_metadata_key(args, ll, "air.thread_index_in_simdgroup")
        .into_iter()
        .map(|name| format!("%{name}"))
        .collect::<Vec<_>>();
    let tg_pos_args = arg_names_for_metadata_key(args, ll, "air.threadgroup_position_in_grid")
        .into_iter()
        .map(|name| format!("%{name}"))
        .collect::<Vec<_>>();
    if lane_args.is_empty() || tg_pos_args.is_empty() {
        return false;
    }
    let Some(body) = primary_entry_function_body(ll) else {
        return false;
    };

    let mut lane_zero_predicates = HashSet::new();
    let mut saw_lane_zero_branch = false;
    let mut tg_position_values = tg_pos_args.into_iter().collect::<HashSet<_>>();
    let mut texture_write_uses_tg_position = false;
    for line in body.lines().map(str::trim) {
        if line.starts_with('%')
            && line.contains(" = icmp eq ")
            && line.contains(", 0")
            && lane_args
                .iter()
                .any(|lane| llvm_line_uses_value(line, lane))
        {
            if let Some(value) = llvm_assignment_value(line) {
                lane_zero_predicates.insert(value.to_string());
            }
        }
        if line.starts_with("br i1 ")
            && lane_zero_predicates
                .iter()
                .any(|pred| llvm_line_uses_value(line, pred))
        {
            saw_lane_zero_branch = true;
        }
        if line.contains("@air.write_texture")
            && tg_position_values
                .iter()
                .any(|value| llvm_line_uses_value(line, value))
        {
            texture_write_uses_tg_position = true;
        }
        if line.starts_with('%')
            && tg_position_values
                .iter()
                .any(|value| llvm_line_uses_value(line, value))
        {
            if let Some(value) = llvm_assignment_value(line) {
                tg_position_values.insert(value.to_string());
            }
        }
    }
    saw_lane_zero_branch && texture_write_uses_tg_position
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
    let current = infer_plan(ll).buffers;
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

fn incompatible_overlapping_output_stride_golden(ll: &str, metal: &MetalRow) -> Option<String> {
    if metal.compare == "none"
        || metal.stage.as_deref() != Some("Kernel")
        || metal.plan.output.kind != "buffer"
    {
        return None;
    }
    let output_index = metal.plan.output.index;
    output_stride_control_requirements(ll, output_index, &metal.plan.buffers)
        .into_iter()
        .find(|req| {
            metal
                .plan
                .buffers
                .iter()
                .find(|buffer| buffer.index == req.buffer)
                .is_some_and(|buffer| bounded_control_output_stride_too_small(buffer, *req))
        })
        .map(|req| {
            format!(
                "metal golden seeds output byte-stride/control buffer too small while replay stores \
                 {} bytes per element to output buffer {output_index}; rebank Metal row \
                 with non-overlapping output stride",
                req.store_bytes
            )
        })
}

#[derive(Clone, Copy)]
struct OutputStrideRequirement {
    buffer: u32,
    store_bytes: usize,
    min_row_bytes: u64,
}

fn output_stride_control_requirements(
    ll: &str,
    output_index: u32,
    buffers: &[PlanBuffer],
) -> Vec<OutputStrideRequirement> {
    let stride_control_buffers: HashMap<u32, &PlanBuffer> = buffers
        .iter()
        .filter(|b| b.seed_mode == SEED_MODE_BOUNDED_CONTROL)
        .filter(|b| !b.seed_layout.is_empty())
        .map(|b| (b.index, b))
        .collect();
    if stride_control_buffers.is_empty() {
        return Vec::new();
    }

    let arg_to_buf = arg_index_to_buffer_location(ll);
    let arg_name_to_buf = arg_name_to_buffer_location(ll, &arg_to_buf);
    let mut ptr_buf: HashMap<&str, u32> = HashMap::new();
    let mut pointer_roots: HashMap<String, HashSet<u32>> = HashMap::new();
    let mut pointer_offsets: HashMap<String, HashSet<u32>> = HashMap::new();
    for arg in entry_function_args(ll)
        .into_iter()
        .flat_map(|args| args.split(',').enumerate())
    {
        let Some(&buf) = arg_to_buf.get(&arg.0) else {
            continue;
        };
        let Some(name) = arg.1.rsplit_once('%').map(|(_, name)| name.trim()) else {
            continue;
        };
        if name.is_empty() {
            continue;
        }
        ptr_buf.insert(name, buf);
        pointer_roots.insert(name.to_string(), HashSet::from([buf]));
        pointer_offsets.insert(name.to_string(), HashSet::new());
    }
    if let Some(entry) = entry_name_from_ll(ll) {
        for (ord, name) in entry_arg_names(ll, &entry).into_iter().enumerate() {
            let Some(&buf) = arg_to_buf.get(&ord) else {
                continue;
            };
            pointer_roots
                .entry(name.clone())
                .or_insert_with(|| HashSet::from([buf]));
            pointer_offsets.entry(name).or_default();
        }
    }

    let mut value_sources: HashMap<String, HashSet<u32>> = HashMap::new();
    let mut requirements = Vec::new();
    for line in ll.lines() {
        let trimmed = line.trim_start();
        if let Some((reg, rhs)) = split_assign(trimmed) {
            let operands = percent_operands(rhs);
            let mut sources = HashSet::new();
            if rhs.starts_with("load ") {
                if let Some(buf) = first_buf_operand(rhs, &ptr_buf, &arg_to_buf, &arg_name_to_buf) {
                    sources.insert(buf);
                }
                for operand in &operands {
                    if let Some(roots) = pointer_roots.get(*operand) {
                        sources.extend(roots.iter().copied());
                    }
                }
            }
            for operand in &operands {
                if let Some(prev) = value_sources.get(*operand) {
                    sources.extend(prev.iter().copied());
                }
            }
            if !sources.is_empty() {
                value_sources.insert(reg.to_string(), sources.clone());
            }

            let mut roots = HashSet::new();
            let mut offsets = HashSet::new();
            for operand in operands {
                if let Some(prev) = pointer_roots.get(operand) {
                    roots.extend(prev.iter().copied());
                }
                if let Some(prev) = pointer_offsets.get(operand) {
                    offsets.extend(prev.iter().copied());
                }
                if let Some(prev) = value_sources.get(operand) {
                    offsets.extend(prev.iter().copied());
                }
            }
            if !roots.is_empty()
                && (rhs.contains("getelementptr")
                    || rhs.starts_with("bitcast ")
                    || rhs.starts_with("select "))
            {
                pointer_roots.insert(reg.to_string(), roots);
                pointer_offsets.insert(reg.to_string(), offsets);
            }
            continue;
        }

        let Some(store_bytes) = llvm_store_value_byte_width(trimmed) else {
            continue;
        };
        if store_bytes <= 1 {
            continue;
        }
        let Some(ptr) = store_pointer_operand(trimmed) else {
            continue;
        };
        let roots_hit_output = pointer_roots
            .get(ptr)
            .is_some_and(|roots| roots.contains(&output_index));
        if !roots_hit_output {
            continue;
        }
        if let Some(offsets) = pointer_offsets.get(ptr) {
            for buf in offsets {
                if stride_control_buffers.contains_key(buf) {
                    requirements.push(OutputStrideRequirement {
                        buffer: *buf,
                        store_bytes,
                        min_row_bytes: output_stride_min_row_bytes(store_bytes),
                    });
                }
            }
        }
    }
    requirements
}

fn output_stride_min_row_bytes(store_bytes: usize) -> u64 {
    u64::from(BOUNDED_CONTROL_DIM).saturating_mul(store_bytes as u64)
}

fn bounded_control_output_stride_too_small(
    buffer: &PlanBuffer,
    req: OutputStrideRequirement,
) -> bool {
    buffer.seed_layout.iter().any(|field| {
        bounded_control_seed_field_is_within_buffer(buffer.len, field)
            && field.value.unwrap_or(u64::from(BOUNDED_CONTROL_DIM)) < req.min_row_bytes
    })
}

fn bounded_control_seed_field_is_within_buffer(len: usize, field: &ControlSeedField) -> bool {
    field.size <= 4
        && field.offset.saturating_add(field.size).min(len)
            == field.offset.saturating_add(field.size)
}

fn control_seed_field_max_value(size: usize) -> Option<u64> {
    match size {
        1 => Some(u64::from(u8::MAX)),
        2 => Some(u64::from(u16::MAX)),
        4 => Some(u64::from(u32::MAX)),
        _ => None,
    }
}

fn llvm_store_value_byte_width(line: &str) -> Option<usize> {
    let rest = line.strip_prefix("store ")?;
    let value_ty = llvm_store_value_type(rest)?;
    llvm_type_byte_width(value_ty)
}

fn llvm_store_value_type(rest: &str) -> Option<&str> {
    let rest = rest.trim_start();
    if rest.starts_with('<') {
        let end = rest.find('>')?;
        return Some(&rest[..=end]);
    }
    rest.split_whitespace().next()
}

fn llvm_type_byte_width(ty: &str) -> Option<usize> {
    let ty = ty.trim();
    if let Some(inner) = ty.strip_prefix('<').and_then(|s| s.strip_suffix('>')) {
        let (lanes, scalar) = inner.split_once(" x ")?;
        let lanes = lanes.trim().parse::<usize>().ok()?;
        return llvm_type_byte_width(scalar).map(|bytes| bytes.saturating_mul(lanes));
    }
    match ty {
        "half" => Some(2),
        "float" => Some(4),
        "double" => Some(8),
        "ptr" => Some(8),
        _ => ty
            .strip_prefix('i')
            .and_then(|bits| bits.parse::<usize>().ok())
            .map(|bits| bits.div_ceil(8)),
    }
}

fn store_pointer_operand(line: &str) -> Option<&str> {
    let (_, ptr_part) = line.split_once(',')?;
    first_percent_reg(ptr_part)
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

fn incompatible_bounded_control_strided_input_oob_golden(
    ll: &str,
    metal: &MetalRow,
) -> Option<String> {
    if metal.compare == "none"
        || metal.stage.as_deref() != Some("Kernel")
        || !ll.contains("load <2 x i32>, ptr addrspace(2)")
        || !ll.contains("@air.min.u.i32")
        || !module_has_dynamic_device_gep(ll, "i32")
    {
        return None;
    }
    let width = metal
        .plan
        .buffers
        .iter()
        .find(|buffer| {
            buffer.seed_mode == SEED_MODE_BOUNDED_CONTROL
                && buffer.len >= 8
                && buffer_type_name_for_location(ll, buffer.index).as_deref() == Some("uint2")
        })
        .and_then(|buffer| bounded_control_u32_at(buffer, 0))
        .unwrap_or(BOUNDED_CONTROL_DIM);
    let height = metal
        .plan
        .buffers
        .iter()
        .find(|buffer| {
            buffer.seed_mode == SEED_MODE_BOUNDED_CONTROL
                && buffer.len >= 8
                && buffer_type_name_for_location(ll, buffer.index).as_deref() == Some("uint2")
        })
        .and_then(|buffer| bounded_control_u32_at(buffer, 4))
        .unwrap_or(BOUNDED_CONTROL_DIM);
    let max_y = u64::from(
        metal.plan.dispatch_grid[1]
            .saturating_sub(1)
            .min(height.saturating_sub(1)),
    );
    let required_elems = max_y
        .saturating_mul(u64::from(width.max(1)))
        .saturating_add(u64::from(metal.plan.dispatch_grid[0].max(1)));
    for buffer in &metal.plan.buffers {
        if buffer.role == "Output" || buffer.seed_mode == SEED_MODE_BOUNDED_CONTROL {
            continue;
        }
        let Some(type_name) = buffer_type_name_for_location(ll, buffer.index) else {
            continue;
        };
        if !matches!(type_name.as_str(), "int" | "uint") {
            continue;
        }
        if required_elems.saturating_mul(4) <= buffer.len as u64 {
            continue;
        }
        return Some(format!(
            "metal golden seeds bounded_control width={} height={} for a strided device-buffer \
             lookup into buffer {} with {} uint elements; dispatch_grid={:?} can read past the \
             input buffer, so rebank with valid dimensions or a larger input buffer",
            width,
            height,
            buffer.index,
            buffer.len / 4,
            metal.plan.dispatch_grid
        ));
    }
    None
}

fn incompatible_finite_struct_control_index_golden(ll: &str, metal: &MetalRow) -> Option<String> {
    if metal.compare == "none" {
        return None;
    }
    let finite_struct_buffers: HashSet<u32> = metal
        .plan
        .buffers
        .iter()
        .filter(|buffer| buffer.seed_mode == SEED_MODE_FINITE_STRUCT_FLOAT)
        .map(|buffer| buffer.index)
        .collect();
    if finite_struct_buffers.is_empty() {
        return None;
    }
    let body = primary_entry_function_body(ll)?;
    let args = primary_entry_function_args(ll)?;
    let arg_to_buf = arg_index_to_buffer_location(ll);
    let arg_name_to_buf = arg_name_to_buffer_location_from_args(args, &arg_to_buf);
    if arg_to_buf.is_empty() {
        return None;
    }

    let mut ptr_buf: HashMap<&str, u32> = HashMap::new();
    let mut index_value_source: HashMap<&str, u32> = HashMap::new();
    for line in body.lines().map(str::trim) {
        let Some((reg, rhs)) = split_assign(line) else {
            continue;
        };
        if rhs.starts_with("getelementptr") || rhs.starts_with("bitcast") {
            let buf = first_buf_operand(rhs, &ptr_buf, &arg_to_buf, &arg_name_to_buf);
            if let Some(buf) = buf {
                ptr_buf.insert(reg, buf);
            }
            if rhs.starts_with("getelementptr")
                && buf.is_some_and(|buf| finite_struct_buffers.contains(&buf))
            {
                if let Some(source_buf) = trailing_dynamic_gep_index(rhs)
                    .and_then(normalize_percent_reg)
                    .and_then(|index| index_value_source.get(index))
                {
                    return Some(format!(
                        "metal golden uses finite_struct_float buffer {source_buf} with an integer \
                         struct field that feeds a dynamic struct-array index; rebank Metal row \
                         with bounded control/index fields"
                    ));
                }
            }
            continue;
        }
        if is_integer_load_rhs(rhs) {
            if let Some(buf) = first_buf_operand(rhs, &ptr_buf, &arg_to_buf, &arg_name_to_buf)
                .filter(|buf| finite_struct_buffers.contains(buf))
            {
                index_value_source.insert(reg, buf);
            }
            continue;
        }
        if let Some(src) = integer_cast_source(rhs).and_then(normalize_percent_reg) {
            if let Some(&buf) = index_value_source.get(src) {
                index_value_source.insert(reg, buf);
            }
        }
    }
    None
}

#[derive(Clone, Copy)]
struct ControlIndexSource {
    buffer: u32,
    value: u64,
}

fn incompatible_bounded_control_index_golden(ll: &str, metal: &MetalRow) -> Option<String> {
    if metal.compare == "none" {
        return None;
    }
    let bounded_buffers: HashMap<u32, &PlanBuffer> = metal
        .plan
        .buffers
        .iter()
        .filter(|buffer| buffer.seed_mode == SEED_MODE_BOUNDED_CONTROL)
        .map(|buffer| (buffer.index, buffer))
        .collect();
    if bounded_buffers.is_empty() {
        return None;
    }
    let arg_to_buf = arg_index_to_buffer_location(ll);
    if arg_to_buf.is_empty() {
        return None;
    }

    let mut arg_name_to_buf = HashMap::new();
    let mut ptr_buf: HashMap<&str, u32> = HashMap::new();
    let mut ptr_index_source: HashMap<&str, ControlIndexSource> = HashMap::new();
    let mut index_value_source: HashMap<&str, ControlIndexSource> = HashMap::new();
    for line in ll.lines().map(str::trim) {
        if let Some(args) = define_function_args(line) {
            arg_name_to_buf = arg_name_to_buffer_location_from_args(args, &arg_to_buf);
            ptr_buf.clear();
            ptr_index_source.clear();
            index_value_source.clear();
            continue;
        }

        let Some((reg, rhs)) = split_assign(line) else {
            continue;
        };

        if rhs.starts_with("getelementptr") {
            if let Some(bound) = fixed_array_gep_bound(rhs) {
                if let Some(source) = trailing_dynamic_gep_index(rhs)
                    .and_then(normalize_percent_reg)
                    .and_then(|index| index_value_source.get(index))
                    .copied()
                {
                    if source.value >= bound {
                        return Some(format!(
                            "metal golden seeds bounded_control buffer {} dynamic array index as \
                             {}, outside fixed {bound}-element array; rebank Metal row with valid \
                             control/index fields",
                            source.buffer, source.value
                        ));
                    }
                }
            }

            let buf = first_buf_operand(rhs, &ptr_buf, &arg_to_buf, &arg_name_to_buf);
            if let Some(buf) = buf {
                ptr_buf.insert(reg, buf);
            }
            if let Some(buffer) = buf.and_then(|buf| bounded_buffers.get(&buf).copied()) {
                if let Some(source) = bounded_control_gep_index_source(rhs, buffer) {
                    ptr_index_source.insert(reg, source);
                }
            }
            continue;
        }

        if rhs.starts_with("bitcast") {
            let buf = first_buf_operand(rhs, &ptr_buf, &arg_to_buf, &arg_name_to_buf);
            if let Some(buf) = buf {
                ptr_buf.insert(reg, buf);
            }
            if let Some(source) = percent_operands(rhs)
                .into_iter()
                .find_map(|operand| ptr_index_source.get(operand).copied())
            {
                ptr_index_source.insert(reg, source);
            }
            continue;
        }

        if is_integer_load_rhs(rhs) {
            if let Some(source) = load_pointer_operand(rhs)
                .and_then(|ptr| ptr_index_source.get(ptr))
                .copied()
                .or_else(|| {
                    first_buf_operand(rhs, &ptr_buf, &arg_to_buf, &arg_name_to_buf)
                        .and_then(|buf| bounded_buffers.get(&buf).copied())
                        .map(|buffer| ControlIndexSource {
                            buffer: buffer.index,
                            value: u64::from(BOUNDED_CONTROL_DIM),
                        })
                })
            {
                index_value_source.insert(reg, source);
            }
            continue;
        }

        if let Some(src) = integer_cast_source(rhs).and_then(normalize_percent_reg) {
            if let Some(&source) = index_value_source.get(src) {
                index_value_source.insert(reg, source);
            }
        }
    }
    None
}

fn incompatible_bounded_control_reflective_oob_golden(
    ll: &str,
    metal: &MetalRow,
) -> Option<String> {
    if metal.compare == "none" || metal.stage.as_deref() != Some("Kernel") {
        return None;
    }
    if metal.plan.output.kind != "buffer" || !ll.contains("@air.abs.s.i32") {
        return None;
    }
    let (finite_input, elem_size, elem_label, llvm_ty) =
        finite_float_input_for_reflective_oob(&metal.plan)?;
    if !module_has_dynamic_device_gep(ll, llvm_ty) {
        return None;
    }
    let has_reflective_index = ll.lines().any(|line| {
        let line = line.trim();
        line.contains(" sub ") && line.contains(" i32 %")
    }) && ll
        .lines()
        .any(|line| line.trim_start().starts_with("br i1 %"));
    if !has_reflective_index {
        return None;
    }

    let input_elems = finite_input.len / elem_size;
    if input_elems == 0 {
        return None;
    }

    let default_bounded_controls = metal
        .plan
        .buffers
        .iter()
        .filter(|buffer| buffer.seed_mode == SEED_MODE_BOUNDED_CONTROL)
        .filter(|buffer| {
            [0, 4, 12, 24]
                .into_iter()
                .all(|offset| bounded_control_u32_at(buffer, offset) == Some(BOUNDED_CONTROL_DIM))
        })
        .count();
    if default_bounded_controls < 2 {
        return None;
    }

    let dim = BOUNDED_CONTROL_DIM as usize;
    let min_reflected_linear_index = dim.saturating_mul(dim).saturating_add(1);
    if min_reflected_linear_index <= input_elems {
        return None;
    }
    Some(format!(
        "metal golden seeds bounded_control reflective-padding fields as {dim}, producing source \
         index {min_reflected_linear_index} past finite {elem_label} input buffer {} with {input_elems} \
         elements; rebank Metal row with valid padding/control fields",
        finite_input.index
    ))
}

fn incompatible_bounded_control_local_array_index_golden(
    ll: &str,
    metal: &MetalRow,
) -> Option<String> {
    if metal.compare == "none" {
        return None;
    }
    let index_control_bufs = buffers_with_loads_used_as_local_array_indices(ll);
    if index_control_bufs.is_empty() {
        return None;
    }
    let has_default_index_control = metal
        .plan
        .buffers
        .iter()
        .filter(|buffer| buffer.seed_mode == SEED_MODE_BOUNDED_CONTROL)
        .filter(|buffer| index_control_bufs.contains(&buffer.index))
        .any(|buffer| bounded_control_u32_at(buffer, 0) == Some(BOUNDED_CONTROL_DIM));
    if !has_default_index_control {
        return None;
    }
    Some(format!(
        "metal golden seeds bounded_control local-array index as {}, but the AIR indexes a smaller \
         stack array through a dynamic inbounds GEP; the result is undefined/robustness-dependent, \
         so rebank with an in-range index or drop the row",
        BOUNDED_CONTROL_DIM
    ))
}

fn local_array_capacity_is_less_than_bounded_control_dim(ll: &str) -> bool {
    ll.lines()
        .filter(|line| line.contains(" = type ") && line.contains("["))
        .any(|line| {
            let Some((_, after_bracket)) = line.split_once('[') else {
                return false;
            };
            let Some((count, _)) = after_bracket.split_once(" x ") else {
                return false;
            };
            count
                .trim()
                .parse::<u32>()
                .is_ok_and(|count| count < BOUNDED_CONTROL_DIM)
        })
}

fn incompatible_texture_indexed_float_buffer_golden(ll: &str, metal: &MetalRow) -> Option<String> {
    if metal.compare == "none"
        || metal.plan.output.kind != "texture"
        || !ll_has_fast_no_nans_float_semantics(ll)
        || !ll.contains("@air.read_texture")
        || !ll.contains(".u.v4i32")
        || !ll.contains("getelementptr inbounds float, ptr addrspace(1)")
    {
        return None;
    }
    let input = metal.plan.buffers.iter().find(|buffer| {
        buffer.role == "Input" && buffer.seed_mode == SEED_MODE_FINITE_FLOAT32 && buffer.len >= 4
    })?;
    let input_elems = input.len / 4;
    if input_elems >= 256 {
        return None;
    }
    let has_deterministic_uint_texture = metal.plan.textures.iter().any(|texture| {
        matches!(texture.role.as_str(), "StorageRead" | "StorageReadWrite")
            && texture.seed_mode == SEED_MODE_DETERMINISTIC
            && texture.format.ends_with("Uint")
    });
    if !has_deterministic_uint_texture || !texture_uint_read_feeds_float_buffer_gep(ll) {
        return None;
    }
    Some(format!(
        "metal golden uses deterministic unsigned texture data as an unchecked index into finite f32 \
         input buffer {} with {input_elems} elements under AIR fast/no-nans math; out-of-bounds buffer \
         reads are undefined/robustness-dependent, so rebank with bounded indices or drop Metal row",
        input.index
    ))
}

fn texture_uint_read_feeds_float_buffer_gep(ll: &str) -> bool {
    let mut read_results = HashSet::new();
    let mut vectors = HashSet::new();
    let mut scalars = HashSet::new();
    let mut indices = HashSet::new();
    for line in ll.lines().map(str::trim) {
        if line.contains("@air.read_texture") && line.contains(".u.v4i32") {
            if let Some(value) = llvm_assignment_value(line) {
                read_results.insert(value.to_string());
            }
            continue;
        }
        if line.contains("extractvalue")
            && read_results.iter().any(|v| llvm_line_uses_value(line, v))
        {
            if let Some(value) = llvm_assignment_value(line) {
                vectors.insert(value.to_string());
            }
            continue;
        }
        if line.contains("extractelement <4 x i32>")
            && vectors.iter().any(|v| llvm_line_uses_value(line, v))
        {
            if let Some(value) = llvm_assignment_value(line) {
                scalars.insert(value.to_string());
            }
            continue;
        }
        if line.contains(" = zext i32 ")
            && scalars.iter().any(|v| llvm_line_uses_value(line, v))
            && line.contains(" to i64")
        {
            if let Some(value) = llvm_assignment_value(line) {
                indices.insert(value.to_string());
            }
            continue;
        }
        if line.contains("getelementptr inbounds float, ptr addrspace(1)")
            && indices.iter().any(|v| llvm_line_uses_value(line, v))
        {
            return true;
        }
    }
    false
}

fn incompatible_fast_coordinate_buffer_lookup_golden(ll: &str, metal: &MetalRow) -> Option<String> {
    if metal.compare == "none"
        || !ll_has_fast_no_nans_float_semantics(ll)
        || !float_texture_or_render_target_output(ll, metal)
        || !ll.contains("@air.convert.u.")
        || !(ll.contains(" fdiv fast ") || ll.contains(" fdiv fast <"))
        || !ll.contains("getelementptr inbounds [")
        || !ll.contains(", ptr addrspace(1)")
    {
        return None;
    }
    let has_finite_float_input = metal.plan.buffers.iter().any(|buffer| {
        buffer.role == "Input"
            && matches!(
                buffer.seed_mode.as_str(),
                SEED_MODE_FINITE_FLOAT16 | SEED_MODE_FINITE_FLOAT32
            )
    });
    if !has_finite_float_input {
        return None;
    }
    let has_dynamic_device_array_gep = ll.lines().any(|line| {
        line.contains("getelementptr inbounds [")
            && line.contains(", ptr addrspace(1)")
            && (line.contains(", i64 %") || line.contains(", i32 %"))
    });
    if !has_dynamic_device_array_gep {
        return None;
    }
    Some(
        "metal golden computes a finite float-buffer lookup index through AIR fast floating-point \
         coordinate scaling before texture output; small Metal/Vulkan rounding differences can select \
         different buffer elements, so this seed is not a portable validation oracle; rebank or drop \
         Metal row"
            .into(),
    )
}

fn finite_float_input_for_reflective_oob(
    plan: &HarnessPlan,
) -> Option<(&PlanBuffer, usize, &'static str, &'static str)> {
    plan.buffers.iter().find_map(|buffer| {
        if buffer.role != "Input" {
            return None;
        }
        match buffer.seed_mode.as_str() {
            SEED_MODE_FINITE_FLOAT16 if buffer.len >= 2 => Some((buffer, 2, "f16", "half")),
            SEED_MODE_FINITE_FLOAT32 if buffer.len >= 4 => Some((buffer, 4, "f32", "float")),
            _ => None,
        }
    })
}

fn bounded_control_u32_at(buffer: &PlanBuffer, offset: usize) -> Option<u32> {
    if offset + 4 > buffer.len {
        return None;
    }
    match buffer
        .seed_layout
        .iter()
        .find(|field| field.offset == offset && field.size <= 4)
    {
        Some(field) => field
            .value
            .unwrap_or(u64::from(BOUNDED_CONTROL_DIM))
            .try_into()
            .ok(),
        None => Some(BOUNDED_CONTROL_DIM),
    }
}

fn define_function_args(line: &str) -> Option<&str> {
    let line = line.strip_prefix("define ")?;
    let header = line.split_once('{')?.0;
    let open = header.find('(')?;
    let close = header.rfind(')')?;
    (open < close).then_some(&header[open + 1..close])
}

fn fixed_array_gep_bound(rhs: &str) -> Option<u64> {
    let rhs = rhs.strip_prefix("getelementptr")?;
    let start = rhs.find('[')?;
    let rest = &rhs[start + 1..];
    let (bound, _) = rest.split_once(" x ")?;
    bound.trim().parse().ok()
}

fn bounded_control_gep_index_source(rhs: &str, buffer: &PlanBuffer) -> Option<ControlIndexSource> {
    let Some(field_index) = trailing_const_i32_gep_index(rhs) else {
        return Some(ControlIndexSource {
            buffer: buffer.index,
            value: u64::from(BOUNDED_CONTROL_DIM),
        });
    };
    let offset = usize::try_from(field_index).ok()?.checked_mul(4)?;
    let field = buffer
        .seed_layout
        .iter()
        .find(|field| field.offset == offset && field.size <= 8)?;
    Some(ControlIndexSource {
        buffer: buffer.index,
        value: field.value.unwrap_or(u64::from(BOUNDED_CONTROL_DIM)),
    })
}

fn trailing_const_i32_gep_index(rhs: &str) -> Option<u64> {
    let idx = rhs.rfind(", i32 ")?;
    let rest = &rhs[idx + ", i32 ".len()..];
    rest.split(|ch: char| !ch.is_ascii_digit())
        .next()?
        .parse()
        .ok()
}

fn load_pointer_operand(rhs: &str) -> Option<&str> {
    let (_, ptr_part) = rhs.split_once(',')?;
    first_percent_reg(ptr_part)
}

fn is_integer_load_rhs(rhs: &str) -> bool {
    let Some(rest) = rhs.strip_prefix("load ") else {
        return false;
    };
    let Some((ty, _)) = rest.split_once(',') else {
        return false;
    };
    is_scalar_integer_type(ty.trim())
}

fn integer_cast_source(rhs: &str) -> Option<&str> {
    let rest = rhs
        .strip_prefix("sext ")
        .or_else(|| rhs.strip_prefix("zext "))
        .or_else(|| rhs.strip_prefix("trunc "))?;
    let (from_ty, rest) = rest.split_once(' ')?;
    if !is_scalar_integer_type(from_ty.trim()) {
        return None;
    }
    let (src, to_ty) = rest.split_once(" to ")?;
    is_scalar_integer_type(to_ty.trim()).then_some(src.trim())
}

fn normalize_percent_reg(value: &str) -> Option<&str> {
    let value = value.trim();
    let name = value.strip_prefix('%')?;
    (!name.is_empty()).then_some(name)
}

fn is_scalar_integer_type(ty: &str) -> bool {
    ty.strip_prefix('i')
        .is_some_and(|bits| !bits.is_empty() && bits.chars().all(|c| c.is_ascii_digit()))
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
    let current_buffers = infer_plan(ll).buffers;
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
        if banked.seed_mode == current.seed_mode && banked.len != current.len {
            return Some(format!(
                "metal golden uses legacy {} buffer {} length {} now sized {}; rebank Metal row",
                finite_seed_label(&current.seed_mode),
                current.index,
                banked.len,
                current.len
            ));
        }
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

fn incompatible_finite_struct_half_fragment_golden(ll: &str, metal: &MetalRow) -> Option<String> {
    if metal.compare == "none"
        || metal.stage.as_deref() != Some("Fragment")
        || !ll_has_fast_no_nans_float_semantics(ll)
        || !float_texture_or_render_target_output(ll, metal)
        || !(ll.contains(" fdiv fast half ") || ll.contains(" fdiv fast <"))
    {
        return None;
    }
    let has_finite_struct_input = infer_buffers(ll)
        .into_iter()
        .any(|buffer| buffer.role != "Output" && buffer.seed_mode == SEED_MODE_FINITE_STRUCT_FLOAT);
    if !has_finite_struct_input {
        return None;
    }
    Some(
        "metal golden feeds finite_struct_float synthetic data through AIR fast half division before \
         float render-target output; Metal/Vulkan half division and rounding are not a portable \
         byte oracle for this seed, so rebank or drop the row"
            .into(),
    )
}

fn incompatible_barycentric_derivative_fragment_golden(
    ll: &str,
    metal: &MetalRow,
) -> Option<String> {
    if metal.compare == "none"
        || metal.stage.as_deref() != Some("Fragment")
        || metal.plan.output.kind != "render_target"
        || !ll.contains(r#""air.barycentric_coord""#)
        || !ll.contains("@air.fwidth.")
    {
        return None;
    }
    Some(
        "metal golden compares a fragment render target derived from AIR barycentric coordinates \
         and derivative/fwidth math; the current Vulkan harness does not reproduce Metal's exact \
         rasterization/interpolation derivative oracle for this synthetic draw, so rebank or drop \
         the row"
            .into(),
    )
}

fn incompatible_moltenvk_vertex_clip_distance_half_texture_golden(
    ll: &str,
    metal: &MetalRow,
) -> Option<String> {
    if metal.compare == "none"
        || metal.stage.as_deref() != Some("Vertex")
        || !ll.contains(r#""air.clip_distance""#)
        || !sampled_finite_half_texture(ll)
    {
        return None;
    }
    Some(
        "metal golden is a vertex shader that samples synthetic finite f16 texture data while \
         writing AIR clip_distance; MoltenVK rejects the validation runner's vertex-only \
         ClipDistance pipeline for this shape, so this row is not a portable MoltenVK byte oracle; \
         rebank or drop the Metal row"
            .into(),
    )
}

fn incompatible_moltenvk_sampled_half_render_target_exact_golden(
    ll: &str,
    metal: &MetalRow,
) -> Option<String> {
    if metal.compare == "none"
        || metal.stage.as_deref() != Some("Fragment")
        || metal.plan.output.kind != "render_target"
        || !ll_has_fast_no_nans_float_semantics(ll)
        || !(ll.contains("@air.sample_texture") || ll.contains("@air.gather_texture"))
        || !sampled_finite_half_texture(ll)
        || !float_texture_or_render_target_output(ll, metal)
    {
        return None;
    }
    Some(
        "metal golden samples synthetic finite f16 texture data before a float fragment \
         render-target output under AIR fast/no-nans math; MoltenVK exact byte comparison is not a \
         portable oracle for half texture filtering and render-target rounding in this seed; \
         rebank or drop the Metal row"
            .into(),
    )
}

fn incompatible_moltenvk_half_texture_output_exact_golden(
    ll: &str,
    metal: &MetalRow,
) -> Option<String> {
    if metal.compare == "none"
        || metal.stage.as_deref() != Some("Kernel")
        || metal.plan.output.kind != "texture"
        || !ll_has_fast_no_nans_float_semantics(ll)
        || !ll.contains("@air.write_texture")
        || !(ll.contains("@air.sample_texture")
            || ll.contains("@air.gather_texture")
            || ll.contains("@air.read_texture"))
        || !(sampled_finite_half_texture(ll) || storage_read_finite_half_texture(ll))
        || !float_texture_or_render_target_output(ll, metal)
    {
        return None;
    }
    Some(
        "metal golden samples or reads synthetic finite f16 texture data before a float texture \
         output under AIR fast/no-nans math; MoltenVK exact byte comparison is not a portable \
         oracle for half texture filtering, image reads, and texture write rounding in this seed; \
         rebank or drop the Metal row"
            .into(),
    )
}

fn incompatible_moltenvk_storage_f32_texture_output_exact_golden(
    ll: &str,
    metal: &MetalRow,
) -> Option<String> {
    if metal.compare == "none"
        || metal.stage.as_deref() != Some("Kernel")
        || metal.plan.output.kind != "texture"
        || !ll_has_fast_no_nans_float_semantics(ll)
        || !ll.contains("@air.read_texture")
        || !writes_f32_texture_output(ll)
        || !storage_read_finite_f32_texture(ll)
        || !float_texture_or_render_target_output(ll, metal)
    {
        return None;
    }
    Some(
        "metal golden reads synthetic finite f32 storage texture data through AIR fast/no-nans \
         math before float texture output; MoltenVK exact byte comparison is not a portable oracle \
         for texture read/write rounding, denorm flushing, and approximate math in this seed; \
         rebank or drop the Metal row"
            .into(),
    )
}

fn incompatible_moltenvk_sampled_f32_cube_buffer_exact_golden(
    ll: &str,
    metal: &MetalRow,
) -> Option<String> {
    if metal.compare == "none"
        || metal.stage.as_deref() != Some("Kernel")
        || metal.plan.output.kind != "buffer"
        || !ll_has_fast_no_nans_float_semantics(ll)
        || !ll.contains("@air.sample_texture_cube.v4f32")
        || !(ll.contains("@air.fast_sqrt.") || ll.contains("@air.fast_rsqrt."))
        || !sampled_finite_f32_texture(ll)
    {
        return None;
    }
    let format = current_output_format_for_plan(ll, &metal.plan.output)
        .and_then(|format| parse_format(format).ok())
        .or_else(|| parse_format(&metal.plan.output.format).ok())?;
    if !format.is_float_like() {
        return None;
    }
    Some(
        "metal golden samples synthetic finite f32 cube texture data through AIR fast sqrt/rsqrt \
         before float buffer output; Metal/Vulkan cube filtering and approximate math are not a \
         portable byte oracle for this seed, so rebank or drop Metal row"
            .into(),
    )
}

fn incompatible_moltenvk_fast_f32_input_buffer_exact_golden(
    ll: &str,
    metal: &MetalRow,
) -> Option<String> {
    if metal.compare == "none"
        || metal.stage.as_deref() != Some("Kernel")
        || metal.plan.output.kind != "buffer"
        || !ll_has_fast_no_nans_float_semantics(ll)
        || !metal.plan.textures.is_empty()
        || !ll.lines().any(|line| line.contains("store float "))
        || !ll_uses_approximate_fast_f32_math(ll)
    {
        return None;
    }
    let format = current_output_format_for_plan(ll, &metal.plan.output)
        .and_then(|format| parse_format(format).ok())
        .or_else(|| parse_format(&metal.plan.output.format).ok())?;
    if !format.is_float_like() {
        return None;
    }
    let finite_f32_inputs = metal
        .plan
        .buffers
        .iter()
        .filter(|buffer| buffer.role == "Input" && buffer.seed_mode == SEED_MODE_FINITE_FLOAT32)
        .count();
    if finite_f32_inputs < 2 {
        return None;
    }
    Some(
        "metal golden feeds finite f32 buffer inputs through AIR fast/no-nans approximate math \
         before float buffer output; MoltenVK exact byte comparison is not a portable oracle for \
         denorm flushing, signed-zero handling, and reassociation in this seed; rebank or drop \
         Metal row"
            .into(),
    )
}

fn incompatible_moltenvk_fast_half_buffer_output_exact_golden(
    ll: &str,
    metal: &MetalRow,
) -> Option<String> {
    if metal.compare == "none"
        || metal.stage.as_deref() != Some("Kernel")
        || metal.plan.output.kind != "buffer"
        || !ll_has_fast_no_nans_float_semantics(ll)
        || !(ll.contains("store half ") || ll.contains("store <3 x half>"))
        || !(ll.contains("@air.fast_sqrt.") || ll.contains("@air.fast_rsqrt."))
        || !metal
            .plan
            .buffers
            .iter()
            .any(|buffer| buffer.role == "Input" && buffer.seed_mode == SEED_MODE_FINITE_FLOAT16)
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
    Some(
        "metal golden writes finite f16 buffer output through AIR fast/no-nans approximate math; \
         MoltenVK exact byte comparison is not a portable oracle for half conversion, denorm \
         flushing, and reassociation in this seed; rebank or drop Metal row"
            .into(),
    )
}

fn incompatible_moltenvk_sampled_f32_render_target_exact_golden(
    ll: &str,
    metal: &MetalRow,
) -> Option<String> {
    if metal.compare == "none"
        || metal.stage.as_deref() != Some("Fragment")
        || metal.plan.output.kind != "render_target"
        || !ll_has_fast_no_nans_float_semantics(ll)
        || !ll.contains("@air.sample_texture")
        || !sampled_finite_f32_texture(ll)
        || !float_texture_or_render_target_output(ll, metal)
    {
        return None;
    }
    Some(
        "metal golden samples synthetic finite f32 texture data before a float fragment \
         render-target output under AIR fast/no-nans math; MoltenVK exact byte comparison is not a \
         portable oracle for texture filtering, denorm flushing, and render-target rounding in \
         this seed; rebank or drop the Metal row"
            .into(),
    )
}

fn incompatible_moltenvk_fast_f32_buffer_output_exact_golden(
    ll: &str,
    metal: &MetalRow,
) -> Option<String> {
    if metal.compare == "none"
        || metal.stage.as_deref() != Some("Kernel")
        || metal.plan.output.kind != "buffer"
        || !ll_has_fast_no_nans_float_semantics(ll)
        || !metal.plan.textures.is_empty()
    {
        return None;
    }
    let format = current_output_format_for_plan(ll, &metal.plan.output)
        .and_then(|format| parse_format(format).ok())
        .or_else(|| parse_format(&metal.plan.output.format).ok())?;
    if !format.is_float_like() {
        return None;
    }
    let has_finite_f32_output = metal.plan.buffers.iter().any(|buffer| {
        matches!(buffer.role.as_str(), "InOut" | "Output")
            && buffer.seed_mode == SEED_MODE_FINITE_FLOAT32
    });
    let has_control_input = metal.plan.buffers.iter().any(|buffer| {
        buffer.role != "Output"
            && matches!(
                buffer.seed_mode.as_str(),
                SEED_MODE_BOUNDED_CONTROL | SEED_MODE_DETERMINISTIC
            )
    });
    if !(has_finite_f32_output && has_control_input) {
        return None;
    }
    Some(
        "metal golden writes finite f32 buffer output under AIR fast/no-nans math; MoltenVK exact \
         byte comparison is not a portable oracle for denorm flushing, signed-zero handling, and \
         reassociation in this seed; rebank or drop the Metal row"
            .into(),
    )
}

fn incompatible_moltenvk_fast_raw_float_buffer_output_exact_golden(
    ll: &str,
    metal: &MetalRow,
) -> Option<String> {
    if metal.compare == "none"
        || metal.stage.as_deref() != Some("Kernel")
        || metal.plan.output.kind != "buffer"
        || metal.plan.output.format != "RawBytes"
        || !ll_has_fast_no_nans_float_semantics(ll)
        || !raw_buffer_writes_float_derived_bytes(ll)
    {
        return None;
    }
    Some(
        "metal golden compares raw buffer bytes written from AIR fast/no-nans float values; \
         MoltenVK exact byte comparison is not a portable oracle for float denorm flushing, \
         signed-zero handling, and raw padding bytes in this seed; rebank or drop the Metal row"
            .into(),
    )
}

fn incompatible_moltenvk_fast_half_render_target_exact_golden(
    ll: &str,
    metal: &MetalRow,
) -> Option<String> {
    if metal.compare == "none"
        || metal.stage.as_deref() != Some("Fragment")
        || metal.plan.output.kind != "render_target"
        || !ll_has_fast_no_nans_float_semantics(ll)
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
    Some(
        "metal golden writes a float fragment render target through AIR fast_sqrt under \
         fast/no-nans math; MoltenVK exact byte comparison is not a portable oracle for \
         approximate sqrt and render-target rounding in this seed; rebank or drop the Metal row"
            .into(),
    )
}

fn incompatible_moltenvk_integer_texture_fast_render_target_exact_golden(
    ll: &str,
    metal: &MetalRow,
) -> Option<String> {
    if metal.compare == "none"
        || metal.stage.as_deref() != Some("Fragment")
        || metal.plan.output.kind != "render_target"
        || !ll_has_fast_no_nans_float_semantics(ll)
        || !(ll.contains("@air.fast_pow.") || ll.contains("@air.fast_powr."))
        || !integer_input_texture(ll)
        || !float_texture_or_render_target_output(ll, metal)
    {
        return None;
    }
    Some(
        "metal golden reads deterministic integer texture data through AIR fast/no-nans pow before \
         float render-target output; MoltenVK exact byte comparison is not a portable oracle for \
         integer texture conversion, approximate pow, and half render-target rounding in this seed; \
         rebank or drop the Metal row"
            .into(),
    )
}

fn incompatible_moltenvk_scaled_integer_half_texture_output_exact_golden(
    ll: &str,
    metal: &MetalRow,
) -> Option<String> {
    if metal.compare == "none"
        || metal.stage.as_deref() != Some("Kernel")
        || metal.plan.output.kind != "texture"
        || !ll_has_fast_no_nans_float_semantics(ll)
        || !ll.contains("@air.write_texture_")
        || !ll.contains("v4f16")
        || !(ll.contains("@air.convert.f.v4f16.u.") || ll.contains("@air.convert.f.v4f16.s."))
        || !ll.contains("fmul fast <4 x half>")
        || !float_texture_or_render_target_output(ll, metal)
    {
        return None;
    }
    Some(
        "metal golden scales integer buffer data into f16 texture output under AIR fast/no-nans \
         math; Metal/MoltenVK half conversion and texture write rounding are not an exact byte \
         oracle for this seed, so rebank or drop the Metal row"
            .into(),
    )
}

fn sampled_finite_half_texture(ll: &str) -> bool {
    infer_textures(ll).into_iter().any(|texture| {
        texture.role == "Sampled"
            && texture.format == "Rgba16Float"
            && texture.seed_mode == SEED_MODE_FINITE_FLOAT16
    })
}

fn sampled_finite_f32_texture(ll: &str) -> bool {
    infer_textures(ll).into_iter().any(|texture| {
        texture.role == "Sampled"
            && texture.format == "Rgba32Float"
            && texture.seed_mode == SEED_MODE_FINITE_FLOAT32
    })
}

fn integer_input_texture(ll: &str) -> bool {
    infer_textures(ll).into_iter().any(|texture| {
        !matches!(texture.role.as_str(), "StorageWrite" | "ColorTarget")
            && parse_format(&texture.format)
                .ok()
                .is_some_and(data_format_is_integer_like)
    })
}

fn data_format_is_integer_like(format: DataFormat) -> bool {
    matches!(
        format,
        DataFormat::Rgba8Uint
            | DataFormat::Rgba8Sint
            | DataFormat::Rgba16Uint
            | DataFormat::Rgba16Sint
            | DataFormat::Rgba32Uint
            | DataFormat::Rgba32Sint
            | DataFormat::R32Uint
            | DataFormat::R32Sint
            | DataFormat::Rg32Uint
            | DataFormat::Rg32Sint
            | DataFormat::R16Uint
            | DataFormat::R16Sint
            | DataFormat::Rg16Uint
            | DataFormat::Rg16Sint
    )
}

fn storage_read_finite_half_texture(ll: &str) -> bool {
    infer_textures(ll).into_iter().any(|texture| {
        matches!(texture.role.as_str(), "StorageRead" | "StorageReadWrite")
            && texture.format == "Rgba16Float"
            && texture.seed_mode == SEED_MODE_FINITE_FLOAT16
    })
}

fn storage_read_finite_f32_texture(ll: &str) -> bool {
    infer_textures(ll).into_iter().any(|texture| {
        matches!(texture.role.as_str(), "StorageRead" | "StorageReadWrite")
            && matches!(
                texture.format.as_str(),
                "R32Float" | "Rg32Float" | "Rgba32Float"
            )
            && texture.seed_mode == SEED_MODE_FINITE_FLOAT32
    })
}

fn writes_f32_texture_output(ll: &str) -> bool {
    ll.contains("@air.write_texture_") && ll.contains("f32")
        || ll.contains("@air.write_imageblock_slice_to_texture_") && ll.contains("f32")
}

fn raw_buffer_writes_float_derived_bytes(ll: &str) -> bool {
    ll.contains("store float ")
        || ll.contains("store <")
        || ll.lines().any(|line| {
            line.contains("@air.pack.") && (line.contains("f16") || line.contains("f32"))
        })
}

fn ll_uses_approximate_fast_f32_math(ll: &str) -> bool {
    ll.contains("@air.fast_rsqrt.f32")
        || ll.contains("@air.fast_sqrt.f32")
        || ll.contains("@air.fast_sin.f32")
        || ll.contains("@air.fast_cos.f32")
        || ll.contains("@air.fast_pow.f32")
        || ll.contains("@air.fast_powr.f32")
        || ll.contains("@air.fast_fmod.f32")
}

fn float_texture_or_render_target_output(ll: &str, metal: &MetalRow) -> bool {
    if !matches!(metal.plan.output.kind.as_str(), "texture" | "render_target") {
        return false;
    }
    current_output_format_for_plan(ll, &metal.plan.output)
        .and_then(|format| parse_format(format).ok())
        .or_else(|| parse_format(&metal.plan.output.format).ok())
        .is_some_and(DataFormat::is_float_like)
}

fn static_normalized_linear_sampler(ll: &str) -> bool {
    ll.lines()
        .filter(|line| line.contains("@__air_sampler_state") && line.contains(" constant "))
        .filter_map(first_i64_literal)
        .any(air_sampler_word_is_normalized_linear)
}

fn air_sampler_word_is_normalized_linear(word: i64) -> bool {
    let word = word as u64;
    let min_linear = ((word >> 11) & 0x3) == 1;
    let mag_linear = ((word >> 9) & 0x3) == 1;
    let normalized_coordinates = word & 0x8000 == 0;
    min_linear && mag_linear && normalized_coordinates
}

fn first_i64_literal(line: &str) -> Option<i64> {
    let (_, after_i64) = line.split_once("i64 ")?;
    let token = after_i64
        .trim_start()
        .chars()
        .take_while(|c| c.is_ascii_digit() || *c == '-')
        .collect::<String>();
    (!token.is_empty()).then(|| token.parse().ok()).flatten()
}

fn incompatible_sampled_half_domain_sensitive_texture_golden(
    ll: &str,
    metal: &MetalRow,
) -> Option<String> {
    if metal.compare == "none"
        || !ll_has_fast_no_nans_float_semantics(ll)
        || !ll_has_domain_sensitive_float_math(ll)
        || !ll.contains("@air.sample_texture")
        || !sampled_finite_half_texture(ll)
        || !float_texture_or_render_target_output(ll, metal)
    {
        return None;
    }
    Some(
        "metal golden samples synthetic finite f16 texture data through AIR fast/no-nans \
         domain-sensitive math before float texture output; Metal/Vulkan texture sampling, \
         approximate math, and half-rounding are not a portable validation oracle for this seed; \
         rebank or drop Metal row"
            .into(),
    )
}

fn incompatible_fast_procedural_half_texture_golden(ll: &str, metal: &MetalRow) -> Option<String> {
    if metal.compare == "none"
        || !ll_has_fast_no_nans_float_semantics(ll)
        || !float_texture_or_render_target_output(ll, metal)
        || !ll.contains("@air.write_texture_2d.v4f16")
        || !(ll.contains("@air.fast_sin.") || ll.contains("@air.fast_cos."))
        || !ll.contains("@air.fast_fract.")
    {
        return None;
    }
    Some(
        "metal golden writes procedural f16 texture output from AIR fast trigonometric/fract math; \
         approximate transcendental results and half rounding are not a portable Metal/Vulkan \
         validation oracle for this synthetic seed; rebank or drop Metal row"
            .into(),
    )
}

fn incompatible_sampled_half_linear_filter_texture_golden(
    ll: &str,
    metal: &MetalRow,
) -> Option<String> {
    if metal.compare == "none"
        || !(ll.contains("@air.sample_texture_2d.v4f16")
            || ll.contains("@air.gather_texture_2d.v4f16"))
        || !sampled_finite_half_texture(ll)
        || !float_texture_or_render_target_output(ll, metal)
    {
        return None;
    }
    if ll.contains("@air.gather_texture_2d.v4f16")
        && ll.contains("@air.imageblock_data")
        && ll.contains("@air.write_imageblock_slice_to_texture_2d")
    {
        return Some(
            "metal golden gathers finite f16 texture data through an AIR imageblock before \
             texture output; Metal/Vulkan gather addressing and imageblock rounding are not a \
             portable validation oracle for this synthetic seed; rebank or drop Metal row"
                .into(),
        );
    }
    if ll.contains(" fmul fast <2 x float> ")
        && ll.contains("load float, ptr addrspace(1)")
        && ll.contains("@air.write_texture_2d.i16.v4f16")
    {
        return Some(
            "metal golden samples finite f16 texture data with buffer-scaled coordinates before \
             f16 texture output; Metal/Vulkan sampling coordinate handling and half rounding are \
             not a portable validation oracle for this seed; rebank or drop Metal row"
                .into(),
        );
    }
    if static_normalized_linear_sampler(ll)
        && ll.contains("@air.imageblock_data")
        && ll.contains("@air.write_imageblock_slice_to_texture_2d")
    {
        return Some(
            "metal golden stores normalized-linear sampled/gathered finite f16 texture data through an AIR \
             imageblock before texture output; Metal/Vulkan f16 interpolation and imageblock \
             rounding are not a portable validation oracle for this synthetic seed; rebank or drop \
             Metal row"
                .into(),
        );
    }
    if ll.contains(r#""air.sampler""#)
        && ll.contains("ptr addrspace(3)")
        && ll.contains("@air.wg.barrier")
    {
        return Some(
            "metal golden accumulates finite f16 texture samples selected by a runtime sampler \
             through threadgroup half storage before float texture output; the validation row does \
             not bank sampler semantics, and Metal/Vulkan f16 sampling/rounding is not a portable \
             oracle for this synthetic seed; rebank or drop Metal row"
                .into(),
        );
    }
    None
}

fn incompatible_storage_half_imageblock_texture_golden(
    ll: &str,
    metal: &MetalRow,
) -> Option<String> {
    if metal.compare == "none"
        || !ll.contains("@air.read_texture_2d.i16.v4f16")
        || !ll.contains("@air.imageblock_data")
        || !ll.contains("@air.write_imageblock_slice_to_texture_2d.i16.v4f16")
        || !ll.contains("@air.wg.barrier")
        || !storage_read_finite_half_texture(ll)
        || !float_texture_or_render_target_output(ll, metal)
    {
        return None;
    }
    Some(
        "metal golden copies synthetic finite f16 storage texture data through an AIR imageblock \
         before texture output; Metal imageblock slice layout/rounding is not a portable \
         Vulkan byte oracle for this seed; rebank or drop Metal row"
            .into(),
    )
}

fn incompatible_uninitialized_half_imageblock_texture_golden(
    ll: &str,
    metal: &MetalRow,
) -> Option<String> {
    if metal.compare == "none"
        || metal.stage.as_deref() != Some("Kernel")
        || !ll.contains("@air.imageblock_data")
        || !ll.contains("@air.write_imageblock_slice_to_texture_2d.i16.f16")
        || !float_texture_or_render_target_output(ll, metal)
        || ll
            .lines()
            .any(|line| line.contains("store ") && line.contains("ptr addrspace(4)"))
    {
        return None;
    }
    Some(
        "metal golden writes uninitialized scalar half AIR imageblock data into f16 texture output; \
         imageblock contents and scalar slice layout are not a portable Metal/Vulkan byte oracle \
         for this seed; rebank or drop Metal row"
            .into(),
    )
}

fn incompatible_sampled_f32_imageblock_texture_golden(
    ll: &str,
    metal: &MetalRow,
) -> Option<String> {
    if metal.compare == "none"
        || !ll.contains("@air.sample_texture_2d.v4f32")
        || !ll.contains("@air.imageblock_data")
        || !ll.contains("@air.write_imageblock_slice_to_texture_2d.v4f32")
        || !ll.contains("@air.wg.barrier")
        || !sampled_finite_f32_texture(ll)
        || !float_texture_or_render_target_output(ll, metal)
    {
        return None;
    }
    Some(
        "metal golden stores synthetic finite f32 sampled texture data through an AIR imageblock \
         before texture output; Metal imageblock slice layout and sampled-float rounding are not a \
         portable Vulkan byte oracle for this seed; rebank or drop Metal row"
            .into(),
    )
}

fn integer_texture_output(ll: &str, metal: &MetalRow) -> bool {
    if metal.plan.output.kind != "texture" {
        return false;
    }
    let Some(format) = current_output_format_for_plan(ll, &metal.plan.output)
        .and_then(|format| parse_format(format).ok())
        .or_else(|| parse_format(&metal.plan.output.format).ok())
    else {
        return false;
    };
    matches!(
        format,
        DataFormat::Rgba8Uint
            | DataFormat::Rgba8Sint
            | DataFormat::Rgba16Uint
            | DataFormat::Rgba16Sint
            | DataFormat::Rgba32Uint
            | DataFormat::Rgba32Sint
            | DataFormat::R32Uint
            | DataFormat::R32Sint
            | DataFormat::Rg32Uint
            | DataFormat::Rg32Sint
            | DataFormat::R16Uint
            | DataFormat::R16Sint
            | DataFormat::Rg16Uint
            | DataFormat::Rg16Sint
    )
}

fn incompatible_integer_gather_imageblock_texture_golden(
    ll: &str,
    metal: &MetalRow,
) -> Option<String> {
    if metal.compare == "none"
        || !ll.contains("@air.gather_texture_2d.s.v4i16")
        || !ll.contains("@air.imageblock_data")
        || !ll.contains("@air.write_imageblock_slice_to_texture_2d.i16.v4i16")
        || !ll.contains("@air.wg.barrier")
        || !ll.contains(r#""texture2d<short, sample>""#)
        || !ll.contains(r#""texture2d<short, write>""#)
        || !integer_texture_output(ll, metal)
    {
        return None;
    }
    Some(
        "metal golden gathers signed integer texture data through an AIR imageblock before \
         integer texture output; Metal imageblock slice layout and gather edge handling are not a \
         portable Vulkan byte oracle for this seed; rebank or drop Metal row"
            .into(),
    )
}

fn incompatible_sampled_half_cube_render_target_golden(
    ll: &str,
    metal: &MetalRow,
) -> Option<String> {
    if metal.compare == "none"
        || !ll_has_fast_no_nans_float_semantics(ll)
        || metal.plan.output.kind != "render_target"
        || !(ll.contains("@air.sample_texture_cube.v4f16")
            || ll.contains("@air.sample_texture_cube_array.v4f16"))
        || !ll.contains(r#""air.fragment_input""#)
        || !sampled_finite_half_texture(ll)
    {
        return None;
    }
    let format = current_output_format_for_plan(ll, &metal.plan.output)
        .and_then(|format| parse_format(format).ok())
        .or_else(|| parse_format(&metal.plan.output.format).ok())?;
    if !matches!(format, DataFormat::Rgba16Float | DataFormat::Rgba32Float) {
        return None;
    }
    Some(
        "metal golden samples synthetic finite f16 cube texture data from generated fragment \
         coordinates before float render-target output; Metal/Vulkan cube face selection, filtering, \
         and half rounding are not a portable validation oracle for this synthetic seed; rebank or \
         drop Metal row"
            .into(),
    )
}

fn incompatible_sampled_fast_pow_texture_golden(ll: &str, metal: &MetalRow) -> Option<String> {
    if !ll_has_fast_no_nans_float_semantics(ll)
        || !ll.contains("@air.sample_texture")
        || !(ll.contains("@air.fast_pow.") || ll.contains("@air.fast_powr."))
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
        "metal golden samples synthetic finite f32 texture data through AIR fast_pow/fast_powr \
         under fast/no-nans math; Metal/Vulkan linear sampling and approximate pow rounding are \
         not a portable validation oracle for this synthetic seed; rebank or drop Metal row"
            .into(),
    )
}

fn incompatible_sampled_fast_exp_texture_golden(ll: &str, metal: &MetalRow) -> Option<String> {
    if !ll_has_fast_no_nans_float_semantics(ll)
        || !ll.contains("@air.sample_texture")
        || !ll.contains("@air.fast_exp.")
        || !float_texture_or_render_target_output(ll, metal)
    {
        return None;
    }
    let has_sampled_f32_texture = infer_textures(ll).into_iter().any(|texture| {
        texture.role == "Sampled"
            && texture.format == "Rgba32Float"
            && texture.seed_mode == SEED_MODE_FINITE_FLOAT32
    });
    if !has_sampled_f32_texture {
        return None;
    }
    Some(
        "metal golden samples synthetic finite f32 texture data through AIR fast_exp under \
         fast/no-nans math before float texture output; Metal/Vulkan sampling and approximate exp \
         rounding are not a portable validation oracle for this seed, so rebank or drop the row"
            .into(),
    )
}

fn incompatible_sampled_f32_domain_math_texture_golden(
    ll: &str,
    metal: &MetalRow,
) -> Option<String> {
    if metal.compare == "none"
        || !ll_has_fast_no_nans_float_semantics(ll)
        || !ll.contains("@air.sample_texture")
        || !(ll_has_domain_sensitive_float_math(ll) || ll.contains("@air.log."))
        || ll.contains("@air.fast_pow.")
        || ll.contains("@air.fast_powr.")
        || ll.contains("@air.fast_exp.")
        || !sampled_finite_f32_texture(ll)
        || !float_texture_or_render_target_output(ll, metal)
    {
        return None;
    }
    Some(
        "metal golden samples synthetic finite f32 texture data through AIR fast/no-nans \
         domain-sensitive math before float output; Metal/Vulkan texture sampling and approximate \
         math are not a portable validation oracle for this seed; rebank or drop Metal row"
            .into(),
    )
}

fn incompatible_sampled_uint_float_render_target_golden(
    ll: &str,
    metal: &MetalRow,
) -> Option<String> {
    if metal.compare == "none"
        || metal.stage.as_deref() != Some("Fragment")
        || metal.plan.output.kind != "render_target"
        || !ll.contains("@air.sample_texture_2d.u.v4i32")
        || !ll.contains("@air.convert.f.f32.u.i32")
        || !float_texture_or_render_target_output(ll, metal)
    {
        return None;
    }
    Some(
        "metal golden samples synthetic uint texture data and converts it to float render-target \
         output; Metal/Vulkan integer sampling and float conversion are not an exact byte oracle \
         for this seed, so rebank or drop Metal row"
            .into(),
    )
}

fn incompatible_sampled_half_dot_render_target_golden(
    ll: &str,
    metal: &MetalRow,
) -> Option<String> {
    if metal.compare == "none"
        || metal.stage.as_deref() != Some("Fragment")
        || metal.plan.output.kind != "render_target"
        || !ll.contains("@air.sample_texture_2d.v4f16")
        || !ll.contains("@air.dot.v4f16")
        || !sampled_finite_half_texture(ll)
        || !float_texture_or_render_target_output(ll, metal)
    {
        return None;
    }
    Some(
        "metal golden samples synthetic finite f16 texture data through AIR half dot math before \
         float render-target output; Metal/Vulkan half sampling and dot rounding are not a \
         portable validation oracle for this seed; rebank or drop Metal row"
            .into(),
    )
}

fn incompatible_sampled_f32_dynamic_lod_render_target_golden(
    ll: &str,
    metal: &MetalRow,
) -> Option<String> {
    if metal.compare == "none"
        || !ll_has_fast_no_nans_float_semantics(ll)
        || metal.plan.output.kind != "render_target"
        || !ll.contains("@air.sample_texture_2d.v4f32")
        || !ll.contains(r#""air.fragment_input""#)
    {
        return None;
    }
    let format = current_output_format_for_plan(ll, &metal.plan.output)
        .and_then(|format| parse_format(format).ok())
        .or_else(|| parse_format(&metal.plan.output.format).ok())?;
    if !matches!(format, DataFormat::Rgba16Float | DataFormat::Rgba32Float) {
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
    if sampled_f32_count < 2 || !declares_dynamic_lod_sample(ll) {
        return None;
    }
    Some(
        "metal golden samples synthetic finite f32 texture data from generated fragment coordinates \
         with dynamic LOD before float render-target output; Metal/Vulkan sampling, LOD selection, \
         and fast-math rounding are not a portable validation oracle for this synthetic seed; \
         rebank or drop Metal row"
            .into(),
    )
}

fn incompatible_sampled_f32_texture_array_render_targets_golden(
    ll: &str,
    metal: &MetalRow,
) -> Option<String> {
    if metal.compare == "none"
        || metal.stage.as_deref() != Some("Fragment")
        || metal.plan.output.kind != "render_target"
        || !ll.contains("@air.sample_texture_2d.v4f32")
        || !ll.contains("array<texture2d<float, sample>")
        || !ll.contains("@__air_sampler_state")
        || !sampled_finite_f32_texture(ll)
        || !float_texture_or_render_target_output(ll, metal)
    {
        return None;
    }
    let render_targets = ll
        .lines()
        .filter(|line| line.contains(r#""air.render_target""#))
        .count();
    if render_targets < 2 {
        return None;
    }
    Some(
        "metal golden samples synthetic finite f32 texture-array inputs through a static sampler into \
         multiple float render targets; Metal/Vulkan interpolation and texture-array binding semantics \
         are not a portable validation oracle for this seed; rebank or drop Metal row"
            .into(),
    )
}

fn declares_dynamic_lod_sample(ll: &str) -> bool {
    ll.lines().any(|line| {
        line.contains("@air.sample_texture")
            && line.contains(", i1 true, float %")
            && line.contains(", float 0.000000e+00, i32 0)")
    })
}

fn incompatible_fragment_half_pow_render_target_golden(
    ll: &str,
    metal: &MetalRow,
) -> Option<String> {
    if metal.compare == "none"
        || !ll_has_fast_no_nans_float_semantics(ll)
        || metal.plan.output.kind != "render_target"
        || !(ll.contains("@air.pow.") || ll.contains("@air.fast_pow."))
        || !ll.contains(r#""air.fragment_input""#)
    {
        return None;
    }
    let format = current_output_format_for_plan(ll, &metal.plan.output)
        .and_then(|format| parse_format(format).ok())
        .or_else(|| parse_format(&metal.plan.output.format).ok())?;
    if !matches!(format, DataFormat::Rgba16Float | DataFormat::Rgba32Float) {
        return None;
    }
    if bounded_control_fragment_half_pow_inputs(ll, metal) {
        return Some(
            "metal golden feeds framebuffer half color through bounded-control gain/offset/gamma \
             AIR pow under fast/no-nans fragment render-target math; Metal/Vulkan pow and half \
             rounding are not a portable validation oracle for this synthetic seed; rebank or \
             drop Metal row"
                .into(),
        );
    }
    let signed_half_values: Vec<_> = ll
        .lines()
        .filter(|line| {
            line.contains("= fsub fast ") && (line.contains(" half ") || line.contains(" x half>"))
        })
        .filter_map(llvm_assignment_value)
        .collect();
    if signed_half_values.is_empty() {
        return None;
    }
    if ll.lines().any(|line| {
        (line.contains("@air.pow.") || line.contains("@air.fast_pow."))
            && line.contains("f16")
            && signed_half_values
                .iter()
                .any(|value| llvm_line_uses_value(line, value))
    }) {
        return Some(
            "metal golden feeds signed half values into AIR pow under fast/no-nans fragment \
             render-target math; negative-base pow behavior and half rounding are not a portable \
             Metal/Vulkan validation oracle for this synthetic seed; rebank or drop Metal row"
                .into(),
        );
    }
    None
}

fn incompatible_fragment_fast_pow_rsqrt_render_target_golden(
    ll: &str,
    metal: &MetalRow,
) -> Option<String> {
    if metal.compare == "none"
        || metal.stage.as_deref() != Some("Fragment")
        || metal.plan.output.kind != "render_target"
        || !ll_has_fast_no_nans_float_semantics(ll)
        || !ll.contains("@air.fast_pow.")
        || !ll.contains("@air.fast_rsqrt.")
        || !float_texture_or_render_target_output(ll, metal)
    {
        return None;
    }
    Some(
        "metal golden writes a float fragment render target through AIR fast_pow and fast_rsqrt \
         under fast/no-nans math; approximate reciprocal-square-root, pow, and reassociation \
         differences are not a portable Metal/Vulkan byte oracle for this seed; rebank or drop \
         the Metal row"
            .into(),
    )
}

fn bounded_control_fragment_half_pow_inputs(ll: &str, metal: &MetalRow) -> bool {
    if !ll.contains("@air.pow.v3f16")
        || !ll.contains("load float, ptr addrspace(2)")
        || !ll.contains("fptrunc float %")
        || !ll.contains("fmul fast <3 x half>")
        || !ll.contains("fadd fast <3 x half>")
        || !ll.contains("fdiv fast float 1.000000e+00")
    {
        return false;
    }
    metal
        .plan
        .buffers
        .iter()
        .any(|buffer| buffer.role != "Output" && buffer.seed_mode == SEED_MODE_BOUNDED_CONTROL)
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
        || !matches!(metal.plan.output.kind.as_str(), "render_target" | "texture")
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
    let has_sampled_half_dynamic_texture_lookup = metal.plan.output.kind == "texture"
        && float_texture_or_render_target_output(ll, metal)
        && sampled_half_count >= 2
        && ll.contains("@air.gather_texture_2d.v4f16")
        && ll.contains("@air.read_texture_2d")
        && ll.contains("@air.sample_texture_2d.v4f16")
        && ll.contains("@air.mix.v2f32")
        && ll.contains("fcmp fast")
        && has_dynamic_half_lane_extract(ll);
    if !(has_half_lookup_quantizer
        || has_sampled_half_to_3d_lookup
        || has_sampled_half_dynamic_texture_lookup)
    {
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

fn has_dynamic_half_lane_extract(ll: &str) -> bool {
    ll.lines().any(|line| {
        line.contains("extractelement <4 x half>")
            && (line.contains(", i16 %") || line.contains(", i32 %") || line.contains(", i64 %"))
    })
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
    if !sampled_finite_half_texture(ll) {
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
    if !sampled_finite_half_texture(ll) {
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

fn incompatible_fast_procedural_f32_texture_output_golden(
    ll: &str,
    metal: &MetalRow,
) -> Option<String> {
    if metal.compare == "none"
        || metal.stage.as_deref() != Some("Kernel")
        || metal.plan.output.kind != "texture"
        || !ll_has_fast_no_nans_float_semantics(ll)
        || !(ll.contains("@air.write_texture_") && ll.contains(".v4f32"))
        || !(ll.contains("@air.fast_fmod.")
            || ll.contains("@air.fast_fract.")
            || ll.contains("@air.fast_rsqrt."))
        || !float_texture_or_render_target_output(ll, metal)
    {
        return None;
    }
    Some(
        "metal golden writes procedural f32 texture output from AIR fast/no-nans approximate math; \
         Metal/Vulkan transcendental and normalization rounding are not a portable byte oracle for \
         this synthetic seed; rebank or drop Metal row"
            .into(),
    )
}

fn incompatible_fast_f32_buffer_output_golden(ll: &str, metal: &MetalRow) -> Option<String> {
    if metal.compare == "none"
        || metal.stage.as_deref() != Some("Kernel")
        || metal.plan.output.kind != "buffer"
        || !ll_has_fast_no_nans_float_semantics(ll)
        || !ll.contains("@air.fast_rsqrt.f32")
        || !ll.contains(" fdiv fast float ")
        || !ll.contains("@llvm.minnum.f32")
    {
        return None;
    }
    let format = current_output_format_for_plan(ll, &metal.plan.output)
        .and_then(|format| parse_format(format).ok())
        .or_else(|| parse_format(&metal.plan.output.format).ok())?;
    if !format.is_float_like() {
        return None;
    }
    let plan = infer_plan(ll);
    if !plan.textures.is_empty() {
        return None;
    }
    let finite_f32_inputs = plan
        .buffers
        .iter()
        .filter(|buffer| buffer.role == "Input" && buffer.seed_mode == SEED_MODE_FINITE_FLOAT32)
        .count();
    if finite_f32_inputs < 2 {
        return None;
    }
    Some(
        "metal golden feeds finite f32 buffer data through AIR fast/no-nans normalization and \
         min/max math before float buffer output; Metal/Vulkan approximate reciprocal-square-root, \
         division, and reassociation are not an exact byte oracle for this seed, so rebank or drop \
         Metal row"
            .into(),
    )
}

fn incompatible_sampled_f32_texture_output_golden(ll: &str, metal: &MetalRow) -> Option<String> {
    if metal.compare == "none"
        || metal.stage.as_deref() != Some("Kernel")
        || metal.plan.output.kind != "texture"
        || !ll.contains("@air.sample_texture_2d.v4f32")
        || !ll.contains("@air.write_texture_2d.v4f32")
        || !sampled_finite_f32_texture(ll)
        || !float_texture_or_render_target_output(ll, metal)
    {
        return None;
    }
    Some(
        "metal golden samples synthetic finite f32 texture data into f32 texture output; \
         Metal/Vulkan sampling and texture write rounding are not an exact byte oracle for this \
         seed, so rebank or drop Metal row"
            .into(),
    )
}

fn incompatible_storage_half_exp_texture_output_golden(
    ll: &str,
    metal: &MetalRow,
) -> Option<String> {
    if metal.compare == "none"
        || metal.stage.as_deref() != Some("Kernel")
        || metal.plan.output.kind != "texture"
        || !ll.contains("@air.read_texture_2d.i16.v4f16")
        || !ll.contains("@air.write_texture_2d.i16.v4f16")
        || !ll.contains("@air.exp.f16")
        || !float_texture_or_render_target_output(ll, metal)
    {
        return None;
    }
    Some(
        "metal golden reads synthetic finite f16 texture data through AIR exp before f16 texture \
         output; Metal/Vulkan half math and texture write rounding are not a portable byte oracle \
         for this seed; rebank or drop Metal row"
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
                if let Some(result) =
                    compare_finite_bfloat_raw_bytes(candidate, &golden, metal, format)
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

fn enforce_backend_compare_policy(
    backend: RunBackend,
    status: &mut String,
    error: &mut Option<String>,
) {
    if backend != RunBackend::MoltenVk || status == "ok" {
        return;
    }
    if status == "tolerance" {
        *status = "failure".into();
        if error.is_none() {
            *error = Some("MoltenVK output differs from Metal; exact byte match required".into());
        }
    } else if status == "smoke" {
        *status = "failure".into();
        if error.is_none() {
            *error = Some("MoltenVK compare=none smoke row is not an exact Metal match".into());
        }
    }
}

fn compile_missing_compare_smoke_error(
    status: &str,
    tolerance: Option<&ToleranceSpecJson>,
    error: Option<&str>,
) -> Option<String> {
    if status == "missing" && tolerance.is_some_and(|t| t.kind == "FastMathNonFiniteDomain") {
        Some(format!(
            "compile-only: {}",
            error.unwrap_or(
                "metal golden compares a non-finite result from AIR fast/no-nans domain-sensitive math"
            )
        ))
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

fn compare_finite_bfloat_raw_bytes(
    candidate: &[u8],
    golden: &[u8],
    metal: &MetalRow,
    format: DataFormat,
) -> Option<CompareResult> {
    if format != DataFormat::RawBytes {
        return None;
    }
    let output = metal.plan.buffers.iter().find(|buffer| {
        buffer.index == metal.plan.output.index
            && buffer.seed_mode == SEED_MODE_FINITE_BFLOAT16
            && matches!(
                buffer.role.as_str(),
                "InOut" | "Output" | "ReadWrite" | "Write"
            )
    })?;
    if output.len == 0 || !candidate.len().is_multiple_of(2) || !golden.len().is_multiple_of(2) {
        return None;
    }
    let policy = ToleranceSpecJson {
        kind: "BFloat16AbsOrUlp".into(),
        max_abs: Some(0.003_906_25),
        max_ulp: Some(32),
    };
    Some(compare_bfloat_raw_bytes(candidate, golden, &policy))
}

fn compare_bfloat_raw_bytes(
    candidate: &[u8],
    golden: &[u8],
    policy: &ToleranceSpecJson,
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
            tolerance: Some(policy.clone()),
        };
    }

    let mut max_abs = 0.0f32;
    let mut max_ulp = 0u32;
    let mut within = true;
    for (c, g) in candidate.chunks_exact(2).zip(golden.chunks_exact(2)) {
        let cu = u16::from_le_bytes([c[0], c[1]]);
        let gu = u16::from_le_bytes([g[0], g[1]]);
        let cf = bfloat_bits_to_f32(cu);
        let gf = bfloat_bits_to_f32(gu);
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
        let ulp = ordered_bfloat_ulp_key(cu).abs_diff(ordered_bfloat_ulp_key(gu));
        max_abs = max_abs.max(abs);
        max_ulp = max_ulp.max(ulp);
        within &= tolerance_policy_accepts(policy, abs, ulp);
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
        tolerance: Some(policy.clone()),
    }
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
        "AbsOrUlp" | "BFloat16AbsOrUlp" => {
            abs <= policy.max_abs.unwrap_or(0.0) || ulp <= policy.max_ulp.unwrap_or(0)
        }
        _ => false,
    }
}

fn ordered_bfloat_ulp_key(bits: u16) -> u32 {
    ordered_float_bits(bits as u32, 0x8000)
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

fn bfloat_bits_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
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
    if cfg.compile_missing {
        eprintln!("# compile-missing");
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
    let moltenvk_rows = if cfg.backend == RunBackend::Vulkan {
        load_candidate_rows(&cfg.corpus_dir.join(RunBackend::MoltenVk.ledger_file_name()))
    } else {
        HashMap::new()
    };

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
        let outcome = process_one(cfg, row, &metal_rows, &moltenvk_rows);
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

fn should_skip_existing_metal_compare_none(cfg: &RunConfig, row: &MetalRow) -> bool {
    !cfg.force && row.status == "ok" && row.compare == "none"
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
    if cfg.reruns_existing_backend_rows() {
        cmd.arg("--force");
    }
    if cfg.quiet {
        cmd.arg("--quiet");
    }
    if cfg.compile_missing {
        cmd.arg("--compile-missing");
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

#[derive(Debug, PartialEq, Eq)]
enum ProcessOutcome {
    Ok,
    Fail,
    Skip,
    Timeout,
}

fn candidate_metal_status_precheck(
    backend: RunBackend,
    metal: &MetalRow,
) -> Option<(&'static str, String, ProcessOutcome)> {
    if metal.status == "quarantine" {
        return Some((
            "quarantine",
            "metal status=quarantine".into(),
            ProcessOutcome::Skip,
        ));
    }
    if metal.status != "ok" || metal.output_sha256.is_none() {
        return Some((
            "missing",
            format!("metal status={}", metal.status),
            ProcessOutcome::Skip,
        ));
    }
    if backend == RunBackend::MoltenVk && metal.compare == "none" {
        return Some((
            "missing",
            "metal golden compare=none is not a full semantic golden; rebank Metal row".into(),
            ProcessOutcome::Skip,
        ));
    }
    None
}

fn compile_missing_candidate_smoke(
    cfg: &RunConfig,
    tr: &TranslateRow,
    src: &SourceFile,
    ll: &str,
    stage: Stage,
    golden_output_sha256: Option<String>,
    reason: String,
) -> ProcessOutcome {
    let tmp = crate::scratch_dir_for(&format!("corpus-smoke-{}", &tr.air_sha256[..12]));
    let plan = infer_plan(ll);
    let spv = match translate_candidate_spv_for_plan(ll, stage, &plan, &tmp) {
        Ok(spv) if !spv.is_empty() => spv,
        Ok(_) => {
            let _ = fs::remove_dir_all(&tmp);
            write_failure_row(
                cfg,
                tr,
                src,
                &format!("compile-missing smoke translate produced empty SPIR-V; skipped compare: {reason}"),
            );
            return ProcessOutcome::Fail;
        }
        Err(error) => {
            let _ = fs::remove_dir_all(&tmp);
            if compile_missing_unsupported_translate_error(&error) {
                let row = candidate_status_row(
                    cfg,
                    tr,
                    src,
                    "quarantine",
                    golden_output_sha256,
                    Some(format!(
                        "compile-missing smoke unsupported: {error}; skipped compare: {reason}"
                    )),
                );
                let _ = append_result_row(cfg, &row);
                return ProcessOutcome::Skip;
            }
            write_failure_row(
                cfg,
                tr,
                src,
                &format!("compile-missing smoke translate: {error}; skipped compare: {reason}"),
            );
            return ProcessOutcome::Fail;
        }
    };
    if let Err(error) = metal2vulkan::tools::spirv_val_bytes(&spv, &tmp) {
        let _ = fs::remove_dir_all(&tmp);
        if compile_missing_unsupported_validation_error(ll, &error) {
            let row = candidate_status_row(
                cfg,
                tr,
                src,
                "quarantine",
                golden_output_sha256,
                Some(format!(
                    "compile-missing smoke unsupported: {error}; skipped compare: {reason}"
                )),
            );
            let _ = append_result_row(cfg, &row);
            return ProcessOutcome::Skip;
        }
        if compile_missing_invalid_smoke_is_validation_quarantine(&reason) {
            let row = candidate_status_row(
                cfg,
                tr,
                src,
                "quarantine",
                golden_output_sha256,
                Some(format!(
                    "compile-missing smoke validation quarantine: {error}; skipped compare: {reason}"
                )),
            );
            let _ = append_result_row(cfg, &row);
            return ProcessOutcome::Skip;
        }
        write_failure_row(
            cfg,
            tr,
            src,
            &format!(
                "compile-missing smoke produced invalid SPIR-V: {error}; skipped compare: {reason}"
            ),
        );
        return ProcessOutcome::Fail;
    }
    let _ = fs::remove_dir_all(&tmp);
    let row = CandidateRow {
        air_sha256: tr.air_sha256.clone(),
        shard: src.shard.clone(),
        label: src.label.clone(),
        status: "smoke".into(),
        backend: cfg.backend.as_str().into(),
        output_sha256: None,
        output_b64: None,
        golden_output_sha256,
        spv_sha256: Some(sha256_hex(&spv)),
        tolerance: None,
        observed: None,
        error: Some(format!("compile-only: {reason}")),
    };
    if let Err(error) = append_result_row(cfg, &row) {
        eprintln!("    write ledger: {error}");
        return ProcessOutcome::Fail;
    }
    ProcessOutcome::Ok
}

fn compile_missing_unsupported_translate_error(error: &str) -> bool {
    error.contains("unsupported Metal visible function reference")
        || error.contains("unsupported Metal patch control point function")
}

fn compile_missing_unsupported_validation_error(ll: &str, error: &str) -> bool {
    let validation_class = metal2vulkan::native::classify_validation_error(error);
    ((error.contains("OpFunctionCall Argument") && error.contains("llvm_memcpy"))
        || validation_class == metal2vulkan::native::ValidationClass::CfgStructurization)
        && unsupported_byval_texture_array_memcpy(ll)
}

fn compile_missing_invalid_smoke_is_validation_quarantine(reason: &str) -> bool {
    reason.contains("rebank Metal row")
        || reason.contains("not a comparable Vulkan oracle yet")
        || reason.contains("compare=none")
}

fn unsupported_byval_texture_array_memcpy(ll: &str) -> bool {
    ll.contains("llvm.memcpy.")
        && ll.contains(" byval([")
        && ll.contains(" x ptr addrspace(1)])")
        && ll.contains(r#""air.texture""#)
        && ll.contains(r#""air.arg_type_name""#)
        && ll.contains("!\"array<texture")
}

fn moltenvk_ok_reference<'a>(
    cfg: &RunConfig,
    moltenvk_rows: &'a HashMap<String, CandidateRow>,
    air_sha256: &str,
    metal: &MetalRow,
) -> Option<&'a CandidateRow> {
    if cfg.backend != RunBackend::Vulkan {
        return None;
    }
    let metal_output = metal.output_sha256.as_deref()?;
    let row = moltenvk_rows.get(air_sha256)?;
    (row.status == "ok"
        && row.output_sha256.as_deref() == Some(metal_output)
        && row.golden_output_sha256.as_deref() == Some(metal_output)
        && row.output_b64.is_some())
    .then_some(row)
}

struct CandidateCompareReference<'a> {
    backend: &'static str,
    row: &'a CandidateRow,
    skipped_metal_reason: &'a str,
}

#[allow(clippy::too_many_arguments)]
fn try_run_vulkan_for_skipped_metal_compare(
    cfg: &RunConfig,
    tr: &TranslateRow,
    src: &SourceFile,
    ll: &str,
    stage: Stage,
    entry: &str,
    metal: &MetalRow,
    moltenvk_rows: &HashMap<String, CandidateRow>,
    skipped_metal_reason: &str,
) -> Option<ProcessOutcome> {
    if cfg.backend != RunBackend::Vulkan {
        return None;
    }
    let compare_reference =
        moltenvk_ok_reference(cfg, moltenvk_rows, &tr.air_sha256, metal).map(|reference| {
            CandidateCompareReference {
                backend: RunBackend::MoltenVk.as_str(),
                row: reference,
                skipped_metal_reason,
            }
        });
    Some(run_candidate(
        cfg,
        tr,
        src,
        ll,
        stage,
        entry,
        &metal.plan,
        metal,
        compare_reference,
        Some(skipped_metal_reason),
    ))
}

fn process_one(
    cfg: &RunConfig,
    tr: &TranslateRow,
    metal_rows: &HashMap<String, MetalRow>,
    moltenvk_rows: &HashMap<String, CandidateRow>,
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
            if should_skip_existing_metal_compare_none(cfg, m) {
                eprintln!("    skip: metal compare=none");
                return ProcessOutcome::Skip;
            }
        }
        let (plan, fc_values) = metal_oracle_inputs(&ll, metal_rows.get(&tr.air_sha256));
        return run_metal(cfg, tr, &src, &ll, stage, &entry, &plan, &fc_values);
    }

    // Vulkan / MoltenVK candidates: inputs must match the metal golden's banked plan. A stale
    // Metal golden is not a candidate regression; keep the diagnostic row but account it as a
    // skipped/not-comparable case in the parent runner.
    macro_rules! write_missing_candidate_and_skip {
        ($metal:expr, $golden:expr, $reason:expr) => {{
            if let Some(outcome) = try_run_vulkan_for_skipped_metal_compare(
                cfg,
                tr,
                &src,
                &ll,
                stage,
                &entry,
                $metal,
                moltenvk_rows,
                &$reason,
            ) {
                return outcome;
            }
            if cfg.compile_missing {
                return compile_missing_candidate_smoke(
                    cfg, tr, &src, &ll, stage, $golden, $reason,
                );
            }
            let row = candidate_status_row(cfg, tr, &src, "missing", $golden, Some($reason));
            let _ = append_result_row(cfg, &row);
            return ProcessOutcome::Skip;
        }};
    }
    let Some(metal) = metal_rows.get(&tr.air_sha256) else {
        let reason = "no metal golden row".to_string();
        if cfg.compile_missing {
            return compile_missing_candidate_smoke(cfg, tr, &src, &ll, stage, None, reason);
        }
        let row = candidate_status_row(cfg, tr, &src, "missing", None, Some(reason));
        let _ = append_result_row(cfg, &row);
        return ProcessOutcome::Skip;
    };
    if let Some((status, reason, outcome)) = candidate_metal_status_precheck(cfg.backend, metal) {
        if cfg.compile_missing && status == "missing" {
            return compile_missing_candidate_smoke(
                cfg,
                tr,
                &src,
                &ll,
                stage,
                metal.output_sha256.clone(),
                reason,
            );
        }
        let row = candidate_status_row(
            cfg,
            tr,
            &src,
            status,
            metal.output_sha256.clone(),
            Some(reason),
        );
        let _ = append_result_row(cfg, &row);
        return outcome;
    }
    if let Some(reason) = incompatible_function_constant_golden(&ll, metal) {
        write_missing_candidate_and_skip!(metal, metal.output_sha256.clone(), reason);
    }
    if let Some(reason) = incompatible_function_constant_texture_array_ref_golden(&ll, metal) {
        write_missing_candidate_and_skip!(metal, metal.output_sha256.clone(), reason);
    }
    if let Some(reason) = incompatible_function_constant_private_pointer_table_golden(&ll, metal) {
        write_missing_candidate_and_skip!(metal, metal.output_sha256.clone(), reason);
    }
    if let Some(reason) = incompatible_function_constant_simdgroup_golden(&ll, metal) {
        write_missing_candidate_and_skip!(metal, metal.output_sha256.clone(), reason);
    }
    if let Some(reason) = incompatible_zero_function_constant_divisor_golden(&ll, metal) {
        write_missing_candidate_and_skip!(metal, metal.output_sha256.clone(), reason);
    }
    if let Some(reason) = incompatible_output_plan_golden(&ll, metal) {
        write_missing_candidate_and_skip!(metal, metal.output_sha256.clone(), reason);
    }
    if let Some(reason) = incompatible_texture_array_plan_golden(&ll, metal) {
        write_missing_candidate_and_skip!(metal, metal.output_sha256.clone(), reason);
    }
    if let Some(reason) = incompatible_multisample_texture_golden(&ll, metal) {
        write_missing_candidate_and_skip!(metal, metal.output_sha256.clone(), reason);
    }
    if let Some(reason) = incompatible_static_resource_plan_golden(&ll, metal) {
        write_missing_candidate_and_skip!(metal, metal.output_sha256.clone(), reason);
    }
    if let Some(reason) = incompatible_point_coord_golden(&ll, metal) {
        write_missing_candidate_and_skip!(metal, metal.output_sha256.clone(), reason);
    }
    if let Some(reason) = incompatible_undefined_texture_write_lanes_golden(&ll, metal) {
        write_missing_candidate_and_skip!(metal, metal.output_sha256.clone(), reason);
    }
    if let Some(reason) = incompatible_bounded_control_seed_golden(&ll, metal) {
        write_missing_candidate_and_skip!(metal, metal.output_sha256.clone(), reason);
    }
    if let Some(reason) = incompatible_overlapping_output_stride_golden(&ll, metal) {
        write_missing_candidate_and_skip!(metal, metal.output_sha256.clone(), reason);
    }
    if let Some(reason) = incompatible_oob_vector_input_golden(&ll, metal) {
        write_missing_candidate_and_skip!(metal, metal.output_sha256.clone(), reason);
    }
    if let Some(reason) = incompatible_bounded_control_strided_input_oob_golden(&ll, metal) {
        write_missing_candidate_and_skip!(metal, metal.output_sha256.clone(), reason);
    }
    if let Some(reason) = incompatible_bounded_control_index_golden(&ll, metal) {
        write_missing_candidate_and_skip!(metal, metal.output_sha256.clone(), reason);
    }
    if let Some(reason) = incompatible_bounded_control_reflective_oob_golden(&ll, metal) {
        write_missing_candidate_and_skip!(metal, metal.output_sha256.clone(), reason);
    }
    if let Some(reason) = incompatible_bounded_control_local_array_index_golden(&ll, metal) {
        write_missing_candidate_and_skip!(metal, metal.output_sha256.clone(), reason);
    }
    if let Some(reason) = incompatible_texture_indexed_float_buffer_golden(&ll, metal) {
        write_missing_candidate_and_skip!(metal, metal.output_sha256.clone(), reason);
    }
    if let Some(reason) = incompatible_fast_coordinate_buffer_lookup_golden(&ll, metal) {
        write_missing_candidate_and_skip!(metal, metal.output_sha256.clone(), reason);
    }
    if let Some(reason) = incompatible_finite_struct_control_index_golden(&ll, metal) {
        write_missing_candidate_and_skip!(metal, metal.output_sha256.clone(), reason);
    }
    if let Some(reason) = incompatible_float_seed_golden(&ll, metal) {
        write_missing_candidate_and_skip!(metal, metal.output_sha256.clone(), reason);
    }
    if let Some(reason) = incompatible_float_output_golden(&ll, metal) {
        write_missing_candidate_and_skip!(metal, metal.output_sha256.clone(), reason);
    }
    if let Some(reason) = incompatible_finite_struct_half_fragment_golden(&ll, metal) {
        write_missing_candidate_and_skip!(metal, metal.output_sha256.clone(), reason);
    }
    if let Some(reason) = incompatible_barycentric_derivative_fragment_golden(&ll, metal) {
        write_missing_candidate_and_skip!(metal, metal.output_sha256.clone(), reason);
    }
    if cfg.backend == RunBackend::MoltenVk {
        if let Some(reason) =
            incompatible_moltenvk_vertex_clip_distance_half_texture_golden(&ll, metal)
        {
            write_missing_candidate_and_skip!(metal, metal.output_sha256.clone(), reason);
        }
        if let Some(reason) =
            incompatible_moltenvk_sampled_half_render_target_exact_golden(&ll, metal)
        {
            write_missing_candidate_and_skip!(metal, metal.output_sha256.clone(), reason);
        }
        if let Some(reason) = incompatible_moltenvk_half_texture_output_exact_golden(&ll, metal) {
            write_missing_candidate_and_skip!(metal, metal.output_sha256.clone(), reason);
        }
        if let Some(reason) =
            incompatible_moltenvk_storage_f32_texture_output_exact_golden(&ll, metal)
        {
            write_missing_candidate_and_skip!(metal, metal.output_sha256.clone(), reason);
        }
        if let Some(reason) =
            incompatible_moltenvk_sampled_f32_render_target_exact_golden(&ll, metal)
        {
            write_missing_candidate_and_skip!(metal, metal.output_sha256.clone(), reason);
        }
        if let Some(reason) = incompatible_moltenvk_fast_f32_buffer_output_exact_golden(&ll, metal)
        {
            write_missing_candidate_and_skip!(metal, metal.output_sha256.clone(), reason);
        }
        if let Some(reason) =
            incompatible_moltenvk_fast_raw_float_buffer_output_exact_golden(&ll, metal)
        {
            write_missing_candidate_and_skip!(metal, metal.output_sha256.clone(), reason);
        }
        if let Some(reason) = incompatible_moltenvk_sampled_f32_cube_buffer_exact_golden(&ll, metal)
        {
            write_missing_candidate_and_skip!(metal, metal.output_sha256.clone(), reason);
        }
        if let Some(reason) = incompatible_moltenvk_fast_f32_input_buffer_exact_golden(&ll, metal) {
            write_missing_candidate_and_skip!(metal, metal.output_sha256.clone(), reason);
        }
        if let Some(reason) = incompatible_moltenvk_fast_half_buffer_output_exact_golden(&ll, metal)
        {
            write_missing_candidate_and_skip!(metal, metal.output_sha256.clone(), reason);
        }
        if let Some(reason) = incompatible_moltenvk_fast_half_render_target_exact_golden(&ll, metal)
        {
            write_missing_candidate_and_skip!(metal, metal.output_sha256.clone(), reason);
        }
        if let Some(reason) =
            incompatible_moltenvk_integer_texture_fast_render_target_exact_golden(&ll, metal)
        {
            write_missing_candidate_and_skip!(metal, metal.output_sha256.clone(), reason);
        }
        if let Some(reason) =
            incompatible_moltenvk_scaled_integer_half_texture_output_exact_golden(&ll, metal)
        {
            write_missing_candidate_and_skip!(metal, metal.output_sha256.clone(), reason);
        }
    }
    if let Some(reason) = incompatible_sampled_half_linear_filter_texture_golden(&ll, metal) {
        write_missing_candidate_and_skip!(metal, metal.output_sha256.clone(), reason);
    }
    if let Some(reason) = incompatible_storage_half_imageblock_texture_golden(&ll, metal) {
        write_missing_candidate_and_skip!(metal, metal.output_sha256.clone(), reason);
    }
    if let Some(reason) = incompatible_uninitialized_half_imageblock_texture_golden(&ll, metal) {
        write_missing_candidate_and_skip!(metal, metal.output_sha256.clone(), reason);
    }
    if let Some(reason) = incompatible_sampled_f32_imageblock_texture_golden(&ll, metal) {
        write_missing_candidate_and_skip!(metal, metal.output_sha256.clone(), reason);
    }
    if let Some(reason) = incompatible_integer_gather_imageblock_texture_golden(&ll, metal) {
        write_missing_candidate_and_skip!(metal, metal.output_sha256.clone(), reason);
    }
    if let Some(reason) = incompatible_sampled_half_cube_render_target_golden(&ll, metal) {
        write_missing_candidate_and_skip!(metal, metal.output_sha256.clone(), reason);
    }
    if let Some(reason) = incompatible_sampled_half_domain_sensitive_texture_golden(&ll, metal) {
        write_missing_candidate_and_skip!(metal, metal.output_sha256.clone(), reason);
    }
    if let Some(reason) = incompatible_fast_procedural_half_texture_golden(&ll, metal) {
        write_missing_candidate_and_skip!(metal, metal.output_sha256.clone(), reason);
    }
    if let Some(reason) = incompatible_sampled_fast_pow_texture_golden(&ll, metal) {
        write_missing_candidate_and_skip!(metal, metal.output_sha256.clone(), reason);
    }
    if let Some(reason) = incompatible_sampled_fast_exp_texture_golden(&ll, metal) {
        write_missing_candidate_and_skip!(metal, metal.output_sha256.clone(), reason);
    }
    if let Some(reason) = incompatible_sampled_f32_domain_math_texture_golden(&ll, metal) {
        write_missing_candidate_and_skip!(metal, metal.output_sha256.clone(), reason);
    }
    if let Some(reason) = incompatible_sampled_uint_float_render_target_golden(&ll, metal) {
        write_missing_candidate_and_skip!(metal, metal.output_sha256.clone(), reason);
    }
    if let Some(reason) = incompatible_sampled_half_dot_render_target_golden(&ll, metal) {
        write_missing_candidate_and_skip!(metal, metal.output_sha256.clone(), reason);
    }
    if let Some(reason) = incompatible_sampled_f32_dynamic_lod_render_target_golden(&ll, metal) {
        write_missing_candidate_and_skip!(metal, metal.output_sha256.clone(), reason);
    }
    if let Some(reason) = incompatible_sampled_f32_texture_array_render_targets_golden(&ll, metal) {
        write_missing_candidate_and_skip!(metal, metal.output_sha256.clone(), reason);
    }
    if let Some(reason) = incompatible_fragment_half_pow_render_target_golden(&ll, metal) {
        write_missing_candidate_and_skip!(metal, metal.output_sha256.clone(), reason);
    }
    if let Some(reason) = incompatible_fragment_fast_pow_rsqrt_render_target_golden(&ll, metal) {
        write_missing_candidate_and_skip!(metal, metal.output_sha256.clone(), reason);
    }
    if let Some(reason) = incompatible_dependent_sampled_lookup_golden(&ll, metal) {
        write_missing_candidate_and_skip!(metal, metal.output_sha256.clone(), reason);
    }
    if let Some(reason) = incompatible_dependent_sampled_half_lookup_golden(&ll, metal) {
        write_missing_candidate_and_skip!(metal, metal.output_sha256.clone(), reason);
    }
    if let Some(reason) = incompatible_sampled_half_fast_sqrt_render_target_golden(&ll, metal) {
        write_missing_candidate_and_skip!(metal, metal.output_sha256.clone(), reason);
    }
    if let Some(reason) = incompatible_sampled_half_exact_control_flow_golden(&ll, metal) {
        write_missing_candidate_and_skip!(metal, metal.output_sha256.clone(), reason);
    }
    if let Some(reason) = incompatible_sampled_half_cube_fast_math_golden(&ll, metal) {
        write_missing_candidate_and_skip!(metal, metal.output_sha256.clone(), reason);
    }
    if let Some(reason) = incompatible_sampled_half_buffer_fast_math_golden(&ll, metal) {
        write_missing_candidate_and_skip!(metal, metal.output_sha256.clone(), reason);
    }
    if let Some(reason) = incompatible_fast_procedural_f32_texture_output_golden(&ll, metal) {
        write_missing_candidate_and_skip!(metal, metal.output_sha256.clone(), reason);
    }
    if let Some(reason) = incompatible_fast_f32_buffer_output_golden(&ll, metal) {
        write_missing_candidate_and_skip!(metal, metal.output_sha256.clone(), reason);
    }
    if let Some(reason) = incompatible_sampled_f32_texture_output_golden(&ll, metal) {
        write_missing_candidate_and_skip!(metal, metal.output_sha256.clone(), reason);
    }
    if let Some(reason) = incompatible_storage_half_exp_texture_output_golden(&ll, metal) {
        write_missing_candidate_and_skip!(metal, metal.output_sha256.clone(), reason);
    }
    if let Some(reason) = incompatible_compare_none_simdgroup_matrix_smoke_golden(&ll, metal) {
        write_missing_candidate_and_skip!(metal, metal.output_sha256.clone(), reason);
    }
    if let Some(reason) = incompatible_compare_none_raytracing_smoke_golden(&ll, metal) {
        write_missing_candidate_and_skip!(metal, metal.output_sha256.clone(), reason);
    }
    if let Some(reason) = incompatible_compare_none_loop_guard_golden(&ll, &entry, metal) {
        write_missing_candidate_and_skip!(metal, metal.output_sha256.clone(), reason);
    }
    if let Some(reason) = incompatible_nonportable_ptrtoint_golden(&ll, metal) {
        write_missing_candidate_and_skip!(metal, metal.output_sha256.clone(), reason);
    }
    if let Some(reason) = incompatible_parallel_dynamic_buffer_scatter_golden(&ll, metal) {
        write_missing_candidate_and_skip!(metal, metal.output_sha256.clone(), reason);
    }
    if let Some(reason) = incompatible_subgroup_texture_write_race_golden(&ll, metal) {
        write_missing_candidate_and_skip!(metal, metal.output_sha256.clone(), reason);
    }
    if let Some(reason) = incompatible_many_to_one_texture_write_golden(&ll, metal) {
        write_missing_candidate_and_skip!(metal, metal.output_sha256.clone(), reason);
    }
    if let Some(reason) = incompatible_undefined_threadgroup_memory_golden(&ll, metal) {
        write_missing_candidate_and_skip!(metal, metal.output_sha256.clone(), reason);
    }
    let plan = metal.plan.clone();
    run_candidate(cfg, tr, &src, &ll, stage, &entry, &plan, metal, None, None)
}

fn write_failure_row(cfg: &RunConfig, tr: &TranslateRow, src: &SourceFile, err: &str) {
    append_status_row(cfg, tr, src, "fallback", "full", None, err.into());
}

fn metal_oracle_inputs(ll: &str, metal: Option<&MetalRow>) -> (HarnessPlan, Vec<(usize, u64)>) {
    let current_plan = infer_plan(ll);
    let plan = metal
        .and_then(banked_explicit_input_plan)
        .unwrap_or(current_plan);
    let fc_values = metal
        .and_then(banked_function_constant_values)
        .unwrap_or_else(|| function_constant_values_for_oracle_inputs(ll));
    (plan, fc_values)
}

fn banked_function_constant_values(metal: &MetalRow) -> Option<Vec<(usize, u64)>> {
    if let Some(fc_values) = metal.fc_values.as_deref() {
        if !fc_values.is_empty() {
            return Some(
                fc_values
                    .iter()
                    .map(|value| (value.index as usize, value.value))
                    .collect(),
            );
        }
    }
    None
}

fn banked_explicit_input_plan(metal: &MetalRow) -> Option<HarnessPlan> {
    (metal.input_specialization.as_deref() == Some(INPUT_SPECIALIZATION_EXPLICIT))
        .then(|| metal.plan.clone())
}

/// Record a case the oracle refused to submit because it could not prove the GPU work bounded
/// (unbounded/uninstrumentable loop). `status=quarantine`, `compare=none`; counted as a failure
/// outcome with a quarantine status. A committed Metal kernel cannot be cancelled, so quarantining
/// is the only safe outcome — the case is recorded for visibility, not dispatched.
#[cfg(target_os = "macos")]
fn write_quarantine_row(
    cfg: &RunConfig,
    tr: &TranslateRow,
    src: &SourceFile,
    plan: &HarnessPlan,
    stage: Stage,
    entry: &str,
    reason: &str,
) {
    let row = metal_status_row(
        tr,
        src,
        MetalStatusFields {
            status: "quarantine",
            compare: "none",
            plan: plan.clone(),
            stage: Some(stage),
            entry: Some(entry),
            error: Some(format!("quarantined: {reason}")),
        },
    );
    let _ = append_result_row(cfg, &row);
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
    fc_values: &[(usize, u64)],
) -> ProcessOutcome {
    if stage == Stage::Fragment {
        if let Some(reason) = unsupported_fragment_color_output_arity(ll) {
            write_failure_row(cfg, tr, src, &reason);
            return ProcessOutcome::Fail;
        }
    }

    #[cfg(not(target_os = "macos"))]
    {
        let _ = (cfg, tr, src, ll, stage, entry, plan, fc_values);
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
        crate::oracle_macos::set_fc_values(fc_values.to_vec());
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
                    write_quarantine_row(cfg, tr, src, plan, stage, entry, reason.trim());
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
            input_specialization: None,
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

    let loop_facts = loop_input_facts_for_metal_plan(ll, entry, metal);
    match crate::loop_budget::classify_and_instrument_with_loop_input_facts(
        ll,
        entry,
        loop_facts.as_loop_input_facts(),
    ) {
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
        || error.contains("Vulkan device does not support SPIR-V")
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
) -> &'static str {
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
    status
}

fn speculative_quarantine_outcome(skipped_metal_reason: Option<&str>) -> ProcessOutcome {
    if skipped_metal_reason.is_some() {
        ProcessOutcome::Skip
    } else {
        ProcessOutcome::Fail
    }
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
    compare_reference: Option<CandidateCompareReference<'_>>,
    skipped_metal_reason: Option<&str>,
) -> ProcessOutcome {
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        let _ = (
            cfg,
            tr,
            src,
            ll,
            stage,
            entry,
            plan,
            metal,
            compare_reference,
            skipped_metal_reason,
        );
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
                return speculative_quarantine_outcome(skipped_metal_reason);
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
                if metal.compare == "none" {
                    let row = candidate_status_row(
                        cfg,
                        tr,
                        src,
                        "missing",
                        metal.output_sha256.clone(),
                        Some(format!(
                            "metal golden compare=none smoke candidate no longer translates: {e}; \
                             rebank or drop Metal row"
                        )),
                    );
                    let _ = append_result_row(cfg, &row);
                    return ProcessOutcome::Skip;
                }
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
            if cfg.compile_missing {
                let row = CandidateRow {
                    air_sha256: tr.air_sha256.clone(),
                    shard: src.shard.clone(),
                    label: src.label.clone(),
                    status: "smoke".into(),
                    backend: cfg.backend.as_str().into(),
                    output_sha256: None,
                    output_b64: None,
                    golden_output_sha256: metal.output_sha256.clone(),
                    spv_sha256: Some(sha256_hex(&spv)),
                    tolerance: None,
                    observed: None,
                    error: Some(format!("compile-only: {reason}")),
                };
                if let Err(e) = append_result_row(cfg, &row) {
                    eprintln!("    write ledger: {e}");
                    return ProcessOutcome::Fail;
                }
                return ProcessOutcome::Ok;
            }
            let row = candidate_status_row(
                cfg,
                tr,
                src,
                "missing",
                metal.output_sha256.clone(),
                Some(reason),
            );
            let _ = append_result_row(cfg, &row);
            return ProcessOutcome::Skip;
        }
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            crate::runner_linux::execute_result(stage, candidate_ll, &spv, &owned.inputs, &tmp)
        }));
        let _ = fs::remove_dir_all(&tmp);
        let candidate = match result {
            Ok(Ok(b)) => b,
            Ok(Err(error)) => {
                let status = write_candidate_execution_error_row(cfg, tr, src, metal, &error);
                return if status == "quarantine" {
                    speculative_quarantine_outcome(skipped_metal_reason)
                } else {
                    ProcessOutcome::Fail
                };
            }
            Err(payload) => {
                let detail = panic_payload_message(payload);
                write_failure_row(cfg, tr, src, &format!("vulkan execute panicked: {detail}"));
                return ProcessOutcome::Fail;
            }
        };
        let reference_metal;
        let compare_metal = if let Some(reference) = compare_reference.as_ref() {
            reference_metal = MetalRow {
                output_sha256: reference.row.output_sha256.clone(),
                output_b64: reference.row.output_b64.clone(),
                compare: "full".into(),
                ..metal.clone()
            };
            &reference_metal
        } else {
            metal
        };
        let golden_hash = compare_metal.output_sha256.clone().unwrap_or_default();
        let out_hash = sha256_hex(&candidate);
        let format = candidate_compare_format(candidate_ll, plan, compare_metal);
        let (mut status, observed, tolerance) = compare_candidate_to_metal(
            &candidate,
            compare_metal,
            &out_hash,
            &golden_hash,
            format,
            Some(candidate_ll),
        );
        let mut error = candidate_compare_error(&status, compare_metal, tolerance.as_ref());
        if let Some(reference) = compare_reference.as_ref() {
            if !matches!(status.as_str(), "ok" | "tolerance") && error.is_none() {
                error = Some(format!(
                    "Vulkan output differs from {} ok reference; skipped Metal compare reason: {}",
                    reference.backend, reference.skipped_metal_reason
                ));
            }
        }
        if let Some(reason) = skipped_metal_reason {
            if !matches!(status.as_str(), "ok" | "tolerance") {
                status = "missing".into();
                error = Some(reason.to_string());
            }
        }
        if cfg.compile_missing {
            if let Some(smoke_error) =
                compile_missing_compare_smoke_error(&status, tolerance.as_ref(), error.as_deref())
            {
                status = "smoke".into();
                error = Some(smoke_error);
            }
        }
        enforce_backend_compare_policy(cfg.backend, &mut status, &mut error);
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
        } else if row.status == "missing" {
            ProcessOutcome::Skip
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
            input_specialization: None,
            stage: stage.map(str::to_string),
            entry: None,
            error: None,
        }
    }

    fn candidate_row_for_compare(status: &str, output: Option<&[u8]>) -> CandidateRow {
        let output_sha256 = output.map(sha256_hex);
        CandidateRow {
            air_sha256: "x".into(),
            shard: None,
            label: String::new(),
            status: status.into(),
            backend: "moltenvk".into(),
            output_sha256: output_sha256.clone(),
            output_b64: output.map(encode_output_b64),
            golden_output_sha256: output_sha256,
            spv_sha256: None,
            tolerance: None,
            observed: None,
            error: None,
        }
    }

    #[test]
    fn vulkan_uses_only_complete_moltenvk_ok_reference_rows() {
        let mut cfg = RunConfig::from_manifest(RunBackend::Vulkan);
        let metal = metal_row_for_compare(&[1, 2, 3], infer_plan(""), Some("Kernel"));
        let mut rows = HashMap::new();

        rows.insert(
            "x".into(),
            candidate_row_for_compare("missing", Some(&[1, 2, 3])),
        );
        assert!(moltenvk_ok_reference(&cfg, &rows, "x", &metal).is_none());

        rows.insert("x".into(), candidate_row_for_compare("ok", None));
        assert!(moltenvk_ok_reference(&cfg, &rows, "x", &metal).is_none());

        rows.insert(
            "x".into(),
            candidate_row_for_compare("ok", Some(&[9, 9, 9])),
        );
        assert!(moltenvk_ok_reference(&cfg, &rows, "x", &metal).is_none());

        rows.insert(
            "x".into(),
            candidate_row_for_compare("ok", Some(&[1, 2, 3])),
        );
        assert!(moltenvk_ok_reference(&cfg, &rows, "x", &metal).is_some());

        cfg.backend = RunBackend::MoltenVk;
        assert!(moltenvk_ok_reference(&cfg, &rows, "x", &metal).is_none());
    }

    #[test]
    fn candidate_metal_quarantine_precheck_is_skip_not_failure() {
        let plan = infer_plan("");
        let mut metal = metal_row_for_compare(&[], plan, Some("Kernel"));
        assert!(candidate_metal_status_precheck(RunBackend::Vulkan, &metal).is_none());

        metal.status = "quarantine".into();
        let (status, reason, outcome) =
            candidate_metal_status_precheck(RunBackend::MoltenVk, &metal)
                .expect("quarantine precheck");
        assert_eq!(status, "quarantine");
        assert_eq!(reason, "metal status=quarantine");
        assert_eq!(outcome, ProcessOutcome::Skip);

        metal.status = "fallback".into();
        let (status, reason, outcome) =
            candidate_metal_status_precheck(RunBackend::MoltenVk, &metal)
                .expect("fallback precheck");
        assert_eq!(status, "missing");
        assert_eq!(reason, "metal status=fallback");
        assert_eq!(outcome, ProcessOutcome::Skip);

        metal.status = "ok".into();
        metal.output_sha256 = None;
        let (status, reason, outcome) =
            candidate_metal_status_precheck(RunBackend::MoltenVk, &metal)
                .expect("hashless ok precheck");
        assert_eq!(status, "missing");
        assert_eq!(reason, "metal status=ok");
        assert_eq!(outcome, ProcessOutcome::Skip);

        metal.output_sha256 = Some(sha256_hex(&[]));
        metal.compare = "none".into();
        assert!(candidate_metal_status_precheck(RunBackend::Vulkan, &metal).is_none());
        let (status, reason, outcome) =
            candidate_metal_status_precheck(RunBackend::MoltenVk, &metal)
                .expect("MoltenVK compare=none precheck");
        assert_eq!(status, "missing");
        assert_eq!(
            reason,
            "metal golden compare=none is not a full semantic golden; rebank Metal row"
        );
        assert_eq!(outcome, ProcessOutcome::Skip);
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

    const BOUNDED_CONTROL_FLOAT_PAYLOAD_LL: &str = r#"
%struct.Params = type { i32, [2 x i32], float, i32, i32 }

define void @k(ptr addrspace(1) readonly %input, ptr addrspace(1) writeonly %output, ptr addrspace(2) %params, i32 %group_id, i16 %lid, i16 %lsize) {
entry:
  %batch_size_ptr = getelementptr inbounds %struct.Params, ptr addrspace(2) %params, i64 0, i32 4
  %batch_size = load i32, ptr addrspace(2) %batch_size_ptr, align 4
  %stride0_ptr = getelementptr inbounds %struct.Params, ptr addrspace(2) %params, i64 0, i32 1, i64 0
  %stride0 = load i32, ptr addrspace(2) %stride0_ptr, align 4
  %stride1_ptr = getelementptr inbounds %struct.Params, ptr addrspace(2) %params, i64 0, i32 1, i64 1
  %stride1 = load i32, ptr addrspace(2) %stride1_ptr, align 4
  %channels_ptr = getelementptr inbounds %struct.Params, ptr addrspace(2) %params, i64 0, i32 3
  %channels = load i32, ptr addrspace(2) %channels_ptr, align 4
  %gid_base = mul nsw i32 %stride0, %group_id
  %lid32 = zext i16 %lid to i32
  %lsize32 = zext i16 %lsize to i32
  br label %batch_loop

batch_loop:
  %sum = phi float [ 0.000000e+00, %entry ], [ %sum_next, %batch_latch ]
  %batch = phi i32 [ 0, %entry ], [ %batch_next, %batch_latch ]
  %batch_base = mul nsw i32 %batch, %stride1
  %base = add nsw i32 %batch_base, %gid_base
  %active = icmp sgt i32 %channels, %lid32
  br i1 %active, label %lane_loop, label %batch_latch

lane_loop:
  %lane = phi i32 [ %lid32, %batch_loop ], [ %lane_next, %lane_loop ]
  %lane_sum = phi float [ %sum, %batch_loop ], [ %lane_sum_next, %lane_loop ]
  %idx = add nsw i32 %base, %lane
  %idx64 = sext i32 %idx to i64
  %ptr = getelementptr inbounds float, ptr addrspace(1) %input, i64 %idx64
  %value = load float, ptr addrspace(1) %ptr, align 4
  %lane_sum_next = fadd fast float %lane_sum, %value
  %lane_next = add nuw nsw i32 %lane, %lsize32
  %keep_lane = icmp slt i32 %lane_next, %channels
  br i1 %keep_lane, label %lane_loop, label %batch_latch

batch_latch:
  %sum_next = phi float [ %sum, %batch_loop ], [ %lane_sum_next, %lane_loop ]
  %batch_next = add nuw nsw i32 %batch, 1
  %keep_batch = icmp eq i32 %batch_next, %batch_size
  br i1 %keep_batch, label %done, label %batch_loop

done:
  %out_idx = sext i32 %group_id to i64
  %out = getelementptr inbounds float, ptr addrspace(1) %output, i64 %out_idx
  store float %sum_next, ptr addrspace(1) %out, align 4
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5, !7, !8, !9}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"input"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"output"}
!5 = !{i32 2, !"air.buffer", !"air.buffer_size", i32 24, !"air.location_index", i32 3, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !6, !"air.arg_type_size", i32 24, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"Params", !"air.arg_name", !"params"}
!6 = !{i32 0, i32 4, i32 0, !"int", !"flags", i32 4, i32 4, i32 2, !"int", !"strides", i32 12, i32 4, i32 0, !"float", !"scale", i32 16, i32 4, i32 0, !"int", !"channels", i32 20, i32 4, i32 0, !"int", !"batch_size"}
!7 = !{i32 3, !"air.threadgroup_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"group_id"}
!8 = !{i32 4, !"air.thread_position_in_threadgroup", !"air.arg_type_name", !"ushort", !"air.arg_name", !"lid"}
!9 = !{i32 5, !"air.threads_per_threadgroup", !"air.arg_type_name", !"ushort", !"air.arg_name", !"lsize"}
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

    const FLOAT2_DEPTH_DIVIDE_CONTROL_LL: &str = r#"
define float @depth_divide(ptr addrspace(2) %near_far, ptr addrspace(2) %bias, ptr addrspace(1) %tex, <2 x float> %uv) #0 {
  %sample = tail call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) %tex, ptr addrspace(2) null, <2 x float> %uv, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %color = extractvalue { <4 x float>, i8 } %sample, 0
  %disp = extractelement <4 x float> %color, i64 0
  %nf = load <2 x float>, ptr addrspace(2) %near_far, align 8
  %near = extractelement <2 x float> %nf, i64 0
  %far = extractelement <2 x float> %nf, i64 1
  %b = load float, ptr addrspace(2) %bias, align 4
  %den = fsub fast float %near, %far
  %num = fsub fast float %disp, %b
  %out = fdiv fast float %num, %den
  ret float %out
}

declare { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32)
attributes #0 = { "no-nans-fp-math"="true" "unsafe-fp-math"="true" }

!air.fragment = !{!0}
!0 = !{ptr @depth_divide, !1, !2}
!1 = !{!3}
!2 = !{!4, !5, !6, !7}
!3 = !{!"air.depth", !"air.depth_qualifier", !"air.any", !"air.arg_type_name", !"float", !"air.arg_name", !"z"}
!4 = !{i32 0, !"air.buffer", !"air.buffer_size", i32 8, !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"float2", !"air.arg_name", !"near_far"}
!5 = !{i32 1, !"air.buffer", !"air.buffer_size", i32 4, !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"bias"}
!6 = !{i32 2, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"tex"}
!7 = !{i32 3, !"air.fragment_input", !"air.arg_type_name", !"float2", !"air.arg_name", !"uv"}
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
  %done = icmp eq i32 %next, 257
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
    fn arg_type_sized_constant_struct_with_float_field_stays_bounded_control() {
        let plan = infer_plan(
            r#"
define void @steel_like(ptr addrspace(2) %params, ptr addrspace(1) %out) {
entry:
  %nq_ptr = getelementptr inbounds %Params, ptr addrspace(2) %params, i64 0, i32 7
  %nq = load i32, ptr addrspace(2) %nq_ptr, align 4
  %scale_ptr = getelementptr inbounds %Params, ptr addrspace(2) %params, i64 0, i32 6
  %scale = load float, ptr addrspace(2) %scale_ptr, align 4
  %keep = icmp sgt i32 %nq, 0
  br i1 %keep, label %write, label %exit
write:
  store float %scale, ptr addrspace(1) %out, align 4
  br label %exit
exit:
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @steel_like, !1, !2}
!1 = !{}
!2 = !{!3, !5}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 4, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !4, !"air.arg_type_size", i32 152, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"Params", !"air.arg_name", !"params"}
!4 = !{i32 0, i32 4, i32 0, !"int", !"B", i32 4, i32 4, i32 0, !"int", !"H", i32 8, i32 4, i32 0, !"int", !"D", i32 12, i32 4, i32 0, !"int", !"qL", i32 16, i32 4, i32 0, !"int", !"kL", i32 20, i32 4, i32 0, !"int", !"gqa_factor", i32 24, i32 4, i32 0, !"float", !"scale", i32 28, i32 4, i32 0, !"int", !"NQ", i32 32, i32 4, i32 0, !"int", !"NK", i32 36, i32 4, i32 0, !"int", !"NQ_aligned", i32 40, i32 4, i32 0, !"int", !"NK_aligned", i32 44, i32 4, i32 0, !"int", !"qL_rem", i32 48, i32 4, i32 0, !"int", !"kL_rem", i32 52, i32 4, i32 0, !"int", !"qL_off", i32 56, i32 8, i32 3, !"long", !"Q_strides", i32 80, i32 8, i32 3, !"long", !"K_strides", i32 104, i32 8, i32 3, !"long", !"V_strides", i32 128, i32 8, i32 3, !"long", !"O_strides"}
!5 = !{i32 1, !"air.buffer", !"air.location_index", i32 3, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"out"}
"#,
        );
        let params = plan.buffers.iter().find(|b| b.index == 4).unwrap();
        assert_eq!(params.len, 152);
        assert_eq!(params.seed_mode, SEED_MODE_BOUNDED_CONTROL);
        for (offset, size, value) in [
            (0usize, 4usize, None),
            (24, 4, Some(0x3f80_0000)),
            (28, 4, None),
            (56, 8, None),
            (128, 8, None),
        ] {
            assert!(
                params.seed_layout.iter().any(|field| {
                    field.offset == offset && field.size == size && field.value == value
                }),
                "missing bounded field ({offset}, {size}, {value:?}) in {:?}",
                params.seed_layout
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
    fn bounded_control_module_seeds_mixed_struct_integer_fields() {
        let mut plan = infer_plan(
            r#"
%struct.Payload = type { float, i32, i8, <2 x float> }

define void @kernel(ptr addrspace(2) %params, ptr addrspace(1) %queue) {
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @kernel, !1, !2}
!1 = !{}
!2 = !{!3, !5}
!3 = !{i32 0, !"air.buffer", !"air.buffer_size", i32 4, !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"params"}
!5 = !{i32 1, !"air.buffer", !"air.location_index", i32 7, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.struct_type_info", !6, !"air.arg_type_size", i32 20, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"Payload", !"air.arg_name", !"queue"}
!6 = !{i32 0, i32 4, i32 0, !"float", !"cost", i32 4, i32 4, i32 0, !"uint", !"count", i32 8, i32 1, i32 0, !"bool", !"enabled", i32 12, i32 8, i32 0, !"float2", !"bounds"}
"#,
        );
        let queue = plan.buffers.iter_mut().find(|b| b.index == 7).unwrap();
        queue.len = 64;
        assert_eq!(queue.seed_mode, SEED_MODE_FINITE_STRUCT_FLOAT);
        assert_eq!(queue.seed_stride, Some(20));
        assert!(
            queue.seed_layout.iter().any(|field| {
                field.offset == 4
                    && field.size == 4
                    && field.value == Some(u64::from(BOUNDED_CONTROL_DIM))
            }),
            "missing bounded integer field: {:?}",
            queue.seed_layout
        );
        assert!(
            queue
                .seed_layout
                .iter()
                .any(|field| { field.offset == 8 && field.size == 1 && field.value == Some(0) }),
            "missing bounded bool field: {:?}",
            queue.seed_layout
        );

        let owned = plan_to_owned_inputs(&plan).unwrap();
        let bytes =
            seeded_buffer_bytes(owned.inputs.buffers.iter().find(|b| b.index == 7).unwrap());
        for base in (0..60).step_by(20) {
            assert_eq!(
                u32::from_le_bytes(bytes[base + 4..base + 8].try_into().unwrap()),
                BOUNDED_CONTROL_DIM
            );
            assert_eq!(bytes[base + 8], 0);
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
            input_specialization: None,
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
    fn finite_struct_integer_field_dynamic_index_requires_rebank() {
        let ll = r#"
%struct.Payload = type { [4 x i16], [4 x float] }

define void @kernel(ptr addrspace(1) %src, ptr addrspace(1) %dst, i32 %tid) {
  %tid64 = zext i32 %tid to i64
  %idx_ptr = getelementptr inbounds %struct.Payload, ptr addrspace(1) %src, i64 0, i32 0, i64 %tid64
  %raw = load i16, ptr addrspace(1) %idx_ptr, align 2
  %idx = sext i16 %raw to i64
  %out = getelementptr inbounds %struct.Payload, ptr addrspace(1) %dst, i64 0, i32 1, i64 %idx
  store float 1.000000e+00, ptr addrspace(1) %out, align 4
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @kernel, !1, !2}
!1 = !{}
!2 = !{!3, !5, !7}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.struct_type_info", !4, !"air.arg_type_size", i32 24, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"Payload", !"air.arg_name", !"src"}
!4 = !{i32 0, i32 8, i32 4, !"short", !"idx", i32 8, i32 16, i32 4, !"float", !"values"}
!5 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.struct_type_info", !6, !"air.arg_type_size", i32 24, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"Payload", !"air.arg_name", !"dst"}
!6 = !{i32 0, i32 8, i32 4, !"short", !"idx", i32 8, i32 16, i32 4, !"float", !"values"}
!7 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid"}
"#;
        let plan = infer_plan(ll);
        assert_eq!(
            plan.buffers
                .iter()
                .find(|buffer| buffer.index == 0)
                .unwrap()
                .seed_mode,
            SEED_MODE_FINITE_STRUCT_FLOAT
        );
        assert_eq!(
            plan.buffers
                .iter()
                .find(|buffer| buffer.index == 1)
                .unwrap()
                .seed_mode,
            SEED_MODE_FINITE_STRUCT_FLOAT
        );
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
            input_specialization: None,
            stage: Some("Kernel".into()),
            entry: Some("kernel".into()),
            error: None,
        };

        let reason = incompatible_finite_struct_control_index_golden(ll, &metal).unwrap();
        assert!(
            reason.contains("finite_struct_float buffer 0")
                && reason.contains("dynamic struct-array index"),
            "{reason}"
        );
    }

    #[test]
    fn bounded_control_dynamic_fixed_array_index_requires_rebank() {
        let ll = r#"
%struct.Params = type { i32, i32 }

define void @kernel(ptr addrspace(2) %params, ptr addrspace(1) %out) {
  tail call fastcc void @impl(ptr addrspace(2) %params, ptr addrspace(1) %out)
  ret void
}

define internal fastcc void @impl(ptr addrspace(2) %params, ptr addrspace(1) %out) {
  %tmp = alloca [5 x i32], align 4
  %slot0 = getelementptr inbounds [5 x i32], ptr %tmp, i64 0, i64 0
  store i32 7, ptr %slot0, align 4
  %axis_ptr = getelementptr inbounds %struct.Params, ptr addrspace(2) %params, i64 0, i32 0
  %axis = load i32, ptr addrspace(2) %axis_ptr, align 4
  %axis64 = sext i32 %axis to i64
  %slot = getelementptr inbounds [5 x i32], ptr %tmp, i64 0, i64 %axis64
  %value = load i32, ptr %slot, align 4
  store i32 %value, ptr addrspace(1) %out, align 4
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @kernel, !1, !2}
!1 = !{}
!2 = !{!3, !5}
!3 = !{i32 0, !"air.buffer", !"air.buffer_size", i32 8, !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !4, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"Params", !"air.arg_name", !"params"}
!4 = !{i32 0, i32 4, i32 0, !"int", !"axis", i32 4, i32 4, i32 0, !"int", !"limit"}
!5 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"int", !"air.arg_name", !"out"}
"#;
        let plan = infer_plan(ll);
        let params = plan
            .buffers
            .iter()
            .find(|buffer| buffer.index == 0)
            .unwrap();
        assert_eq!(params.seed_mode, SEED_MODE_BOUNDED_CONTROL);
        assert!(
            params
                .seed_layout
                .iter()
                .any(|field| field.offset == 0 && field.size == 4 && field.value.is_none()),
            "{:?}",
            params.seed_layout
        );
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
            input_specialization: None,
            stage: Some("Kernel".into()),
            entry: Some("kernel".into()),
            error: None,
        };

        let reason = incompatible_bounded_control_index_golden(ll, &metal).unwrap();
        assert!(
            reason.contains("bounded_control buffer 0")
                && reason.contains("outside fixed 5-element array"),
            "{reason}"
        );
    }

    #[test]
    fn bounded_control_reflective_padding_oob_requires_rebank() {
        let ll = r#"
define void @kernel(ptr addrspace(1) %src, ptr addrspace(1) %dst, ptr addrspace(2) %src_ctl, ptr addrspace(2) %dst_ctl, <3 x i32> %gid) {
  %x = extractelement <3 x i32> %gid, i64 0
  %padded = sub i32 %x, 16
  %neg = icmp slt i32 %padded, 0
  br i1 %neg, label %reflect, label %direct

reflect:
  %abs = tail call i32 @air.abs.s.i32(i32 %padded)
  br label %join

direct:
  br label %join

join:
  %idx = phi i32 [ %abs, %reflect ], [ 257, %direct ]
  %idx64 = sext i32 %idx to i64
  %src_ptr = getelementptr inbounds half, ptr addrspace(1) %src, i64 %idx64
  %value = load half, ptr addrspace(1) %src_ptr, align 2
  store half %value, ptr addrspace(1) %dst, align 2
  ret void
}

declare i32 @air.abs.s.i32(i32)
"#;
        let control_layout = (0..14)
            .map(|field| ControlSeedField {
                offset: field * 4,
                size: 4,
                value: None,
            })
            .collect::<Vec<_>>();
        let plan = HarnessPlan {
            buffers: vec![
                PlanBuffer {
                    index: 0,
                    len: 512,
                    role: "Input".into(),
                    seed_tag: 1,
                    seed_mode: SEED_MODE_FINITE_FLOAT16.into(),
                    seed_layout: Vec::new(),
                    seed_stride: None,
                },
                PlanBuffer {
                    index: 1,
                    len: 256,
                    role: "InOut".into(),
                    seed_tag: 2,
                    seed_mode: SEED_MODE_FINITE_FLOAT16.into(),
                    seed_layout: Vec::new(),
                    seed_stride: None,
                },
                PlanBuffer {
                    index: 2,
                    len: 56,
                    role: "Input".into(),
                    seed_tag: 3,
                    seed_mode: SEED_MODE_BOUNDED_CONTROL.into(),
                    seed_layout: control_layout.clone(),
                    seed_stride: None,
                },
                PlanBuffer {
                    index: 3,
                    len: 56,
                    role: "Input".into(),
                    seed_tag: 4,
                    seed_mode: SEED_MODE_BOUNDED_CONTROL.into(),
                    seed_layout: control_layout,
                    seed_stride: None,
                },
            ],
            textures: Vec::new(),
            output: PlanOutput {
                kind: "buffer".into(),
                index: 1,
                format: "R16Float".into(),
                len: Some(256),
                w: None,
                h: None,
                d: None,
            },
            dispatch_grid: [64, 1, 1],
            dispatch_tg: [64, 1, 1],
        };
        let metal = metal_row_for_compare(&[], plan, Some("Kernel"));

        let reason = incompatible_bounded_control_reflective_oob_golden(ll, &metal)
            .expect("reflective padding OOB");
        assert!(reason.contains("reflective-padding"), "{reason}");
        assert!(
            reason.contains("past finite f16 input buffer 0"),
            "{reason}"
        );
    }

    #[test]
    fn bounded_control_reflective_padding_oob_covers_f32_inputs() {
        let ll = r#"
define void @kernel(ptr addrspace(1) %src, ptr addrspace(1) %dst, ptr addrspace(2) %src_ctl, ptr addrspace(2) %dst_ctl, <3 x i32> %gid) {
  %x = extractelement <3 x i32> %gid, i64 0
  %padded = sub i32 %x, 16
  %neg = icmp slt i32 %padded, 0
  br i1 %neg, label %reflect, label %direct

reflect:
  %abs = tail call i32 @air.abs.s.i32(i32 %padded)
  br label %join

direct:
  br label %join

join:
  %idx = phi i32 [ %abs, %reflect ], [ 257, %direct ]
  %idx64 = sext i32 %idx to i64
  %src_ptr = getelementptr inbounds float, ptr addrspace(1) %src, i64 %idx64
  %value = load float, ptr addrspace(1) %src_ptr, align 4
  store float %value, ptr addrspace(1) %dst, align 4
  ret void
}

declare i32 @air.abs.s.i32(i32)
"#;
        let control_layout = (0..14)
            .map(|field| ControlSeedField {
                offset: field * 4,
                size: 4,
                value: None,
            })
            .collect::<Vec<_>>();
        let plan = HarnessPlan {
            buffers: vec![
                PlanBuffer {
                    index: 0,
                    len: 1024,
                    role: "Input".into(),
                    seed_tag: 1,
                    seed_mode: SEED_MODE_FINITE_FLOAT32.into(),
                    seed_layout: Vec::new(),
                    seed_stride: None,
                },
                PlanBuffer {
                    index: 1,
                    len: 256,
                    role: "InOut".into(),
                    seed_tag: 2,
                    seed_mode: SEED_MODE_FINITE_FLOAT32.into(),
                    seed_layout: Vec::new(),
                    seed_stride: None,
                },
                PlanBuffer {
                    index: 2,
                    len: 56,
                    role: "Input".into(),
                    seed_tag: 3,
                    seed_mode: SEED_MODE_BOUNDED_CONTROL.into(),
                    seed_layout: control_layout.clone(),
                    seed_stride: None,
                },
                PlanBuffer {
                    index: 3,
                    len: 56,
                    role: "Input".into(),
                    seed_tag: 4,
                    seed_mode: SEED_MODE_BOUNDED_CONTROL.into(),
                    seed_layout: control_layout,
                    seed_stride: None,
                },
            ],
            textures: Vec::new(),
            output: PlanOutput {
                kind: "buffer".into(),
                index: 1,
                format: "F32".into(),
                len: Some(256),
                w: None,
                h: None,
                d: None,
            },
            dispatch_grid: [64, 1, 1],
            dispatch_tg: [64, 1, 1],
        };
        let metal = metal_row_for_compare(&[], plan, Some("Kernel"));

        let reason = incompatible_bounded_control_reflective_oob_golden(ll, &metal)
            .expect("reflective padding OOB");
        assert!(
            reason.contains("past finite f32 input buffer 0"),
            "{reason}"
        );
    }

    #[test]
    fn texture_indexed_float_buffer_oob_golden_is_missing() {
        let ll = r#"
define void @kernel(ptr addrspace(1) %hash, ptr addrspace(1) %input, ptr addrspace(1) %out, <2 x i16> %gid) #0 {
  %sam = tail call ptr addrspace(2) @air.get_read_sampler()
  %read = tail call { <4 x i32>, i8 } @air.read_texture_2d.i16.u.v4i32(ptr addrspace(1) %hash, ptr addrspace(2) %sam, <2 x i16> %gid, <2 x i16> zeroinitializer, i16 0, i32 1)
  %vec = extractvalue { <4 x i32>, i8 } %read, 0
  %lane = extractelement <4 x i32> %vec, i64 0
  %idx = zext i32 %lane to i64
  %ptr = getelementptr inbounds float, ptr addrspace(1) %input, i64 %idx
  %v = load float, ptr addrspace(1) %ptr, align 4
  %splat0 = insertelement <4 x float> poison, float %v, i64 0
  %splat = shufflevector <4 x float> %splat0, <4 x float> poison, <4 x i32> zeroinitializer
  tail call void @air.write_texture_2d.i16.v4f32(ptr addrspace(1) %out, <2 x i16> %gid, <4 x float> %splat, i16 0, i32 2)
  ret void
}

declare ptr addrspace(2) @air.get_read_sampler()
declare { <4 x i32>, i8 } @air.read_texture_2d.i16.u.v4i32(ptr addrspace(1), ptr addrspace(2), <2 x i16>, <2 x i16>, i16, i32)
declare void @air.write_texture_2d.i16.v4f32(ptr addrspace(1), <2 x i16>, <4 x float>, i16, i32)
attributes #0 = { "no-nans-fp-math"="true" "unsafe-fp-math"="true" }

!air.kernel = !{!0}
!air.compile_options = !{!8}
!0 = !{ptr @kernel, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5, !6}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.read", !"air.arg_type_name", !"texture2d<uint, read>", !"air.arg_name", !"hash"}
!4 = !{i32 1, !"air.buffer", !"air.buffer_size", i32 256, !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"input"}
!5 = !{i32 2, !"air.texture", !"air.location_index", i32 1, i32 1, !"air.write", !"air.arg_type_name", !"texture2d<float, write>", !"air.arg_name", !"out"}
!6 = !{i32 3, !"air.thread_position_in_grid", !"air.arg_type_name", !"ushort2", !"air.arg_name", !"gid"}
!8 = !{!"air.compile.fast_math_enable"}
"#;

        let metal = metal_row_for_compare(&[], infer_plan(ll), Some("Kernel"));
        let reason = incompatible_texture_indexed_float_buffer_golden(ll, &metal)
            .expect("texture-indexed finite f32 buffer OOB golden");
        assert!(reason.contains("unchecked index"), "{reason}");
        assert!(reason.contains("finite f32 input buffer 0"), "{reason}");
    }

    #[test]
    fn fast_coordinate_buffer_lookup_golden_is_missing() {
        let ll = r#"
define void @kernel(ptr addrspace(1) %input, ptr addrspace(1) %out, ptr addrspace(2) %size, <2 x i32> %gid) #0 {
  %dims = load <2 x i32>, ptr addrspace(2) %size, align 8
  %coord = tail call fast <2 x float> @air.convert.f.v2f32.u.v2i32(<2 x i32> %gid)
  %dims_f = tail call fast <2 x float> @air.convert.f.v2f32.u.v2i32(<2 x i32> %dims)
  %scaled = fdiv fast <2 x float> %coord, %dims_f
  %idx2 = tail call <2 x i32> @air.convert.u.v2i32.f.v2f32(<2 x float> %scaled)
  %y = extractelement <2 x i32> %idx2, i64 1
  %w = extractelement <2 x i32> %dims, i64 0
  %row = mul i32 %y, %w
  %x = extractelement <2 x i32> %idx2, i64 0
  %idx = add i32 %row, %x
  %idx64 = zext i32 %idx to i64
  %p0 = getelementptr inbounds [3 x half], ptr addrspace(1) %input, i64 %idx64, i64 0
  %h = load half, ptr addrspace(1) %p0, align 2
  %f = tail call fast float @air.convert.f.f32.f.f16(half %h)
  %v0 = insertelement <4 x float> poison, float %f, i64 0
  %v = shufflevector <4 x float> %v0, <4 x float> poison, <4 x i32> zeroinitializer
  tail call void @air.write_texture_2d.v4f32(ptr addrspace(1) %out, <2 x i32> %gid, <4 x float> %v, i32 0, i32 2)
  ret void
}

declare <2 x float> @air.convert.f.v2f32.u.v2i32(<2 x i32>)
declare <2 x i32> @air.convert.u.v2i32.f.v2f32(<2 x float>)
declare float @air.convert.f.f32.f.f16(half)
declare void @air.write_texture_2d.v4f32(ptr addrspace(1), <2 x i32>, <4 x float>, i32, i32)
attributes #0 = { "no-nans-fp-math"="true" "unsafe-fp-math"="true" }

!air.kernel = !{!0}
!air.compile_options = !{!7}
!0 = !{ptr @kernel, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5, !6}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 6, !"air.arg_type_align_size", i32 2, !"air.arg_type_name", !"packed_half3", !"air.arg_name", !"input"}
!4 = !{i32 1, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.write", !"air.arg_type_name", !"texture2d<float, write>", !"air.arg_name", !"out"}
!5 = !{i32 2, !"air.buffer", !"air.buffer_size", i32 8, !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_size", i32 8, !"air.arg_type_name", !"uint2", !"air.arg_name", !"size"}
!6 = !{i32 3, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint2", !"air.arg_name", !"gid"}
!7 = !{!"air.compile.fast_math_enable"}
"#;

        let metal = metal_row_for_compare(&[], infer_plan(ll), Some("Kernel"));
        let reason = incompatible_fast_coordinate_buffer_lookup_golden(ll, &metal)
            .expect("fast coordinate buffer lookup golden");
        assert!(reason.contains("finite float-buffer lookup"), "{reason}");
        assert!(reason.contains("rounding differences"), "{reason}");
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
    fn finite_struct_float_buffer_seed_expands_vector_array_fields() {
        let plan = infer_plan(
            r#"
define void @kernel(ptr addrspace(1) %in) {
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @kernel, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 9, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.struct_type_info", !4, !"air.arg_type_size", i32 24, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"Payload", !"air.arg_name", !"in"}
!4 = !{i32 0, i32 8, i32 3, !"float2", !"coords"}
"#,
        );
        let buf = plan.buffers.iter().find(|b| b.index == 9).unwrap();
        assert_eq!(buf.seed_mode, SEED_MODE_FINITE_STRUCT_FLOAT);
        assert_eq!(buf.seed_stride, Some(24));
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
                ControlSeedField {
                    offset: 16,
                    size: 4,
                    value: None,
                },
                ControlSeedField {
                    offset: 20,
                    size: 4,
                    value: None,
                },
            ]
        );
    }

    #[test]
    fn finite_struct_float_buffer_seed_expands_repeated_nested_struct_fields() {
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
!4 = !{!"air.struct_type_info", !5, i32 0, i32 8, i32 2, !"Pair", !"pairs"}
!5 = !{i32 0, i32 4, i32 0, !"float", !"x", i32 4, i32 4, i32 0, !"float", !"y"}
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
    fn bounded_control_float_vector_divisor_fields_seed_to_distinct_values() {
        let plan = infer_plan(FLOAT2_DEPTH_DIVIDE_CONTROL_LL);
        let near_far = plan.buffers.iter().find(|b| b.index == 0).unwrap();
        assert_eq!(near_far.seed_mode, SEED_MODE_BOUNDED_CONTROL);
        assert_eq!(
            near_far.seed_layout,
            vec![
                ControlSeedField {
                    offset: 0,
                    size: 4,
                    value: Some(0x3f80_0000),
                },
                ControlSeedField {
                    offset: 4,
                    size: 4,
                    value: Some(0x4000_0000),
                },
            ]
        );

        let owned = plan_to_owned_inputs(&plan).unwrap();
        let bytes =
            seeded_buffer_bytes(owned.inputs.buffers.iter().find(|b| b.index == 0).unwrap());
        assert_eq!(f32::from_le_bytes(bytes[0..4].try_into().unwrap()), 1.0);
        assert_eq!(f32::from_le_bytes(bytes[4..8].try_into().unwrap()), 2.0);
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
    fn bounded_control_seed_recurses_into_nested_struct_fields() {
        let plan = infer_plan(
            r#"
define void @mps_like(ptr addrspace(1) %params) {
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @mps_like, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.buffer", !"air.buffer_size", i32 64, !"air.location_index", i32 24, i32 1, !"air.read", !"air.address_space", i32 1, !"air.struct_type_info", !4, !"air.arg_type_size", i32 64, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"MPSNDArrayQuantizedGatherMultiplyParams", !"air.arg_name", !"params"}
!4 = !{!"air.struct_type_info", !5, i32 0, i32 48, i32 0, !"MPSNDArrayQuantizedMultiplyGenericParams", !"quantParams", i32 48, i32 4, i32 0, !"uint", !"vecPlaceholderIdx", i32 52, i32 4, i32 0, !"uint", !"w_experts_stride", i32 56, i32 4, i32 0, !"uint", !"scales_experts_stride", i32 60, i32 4, i32 0, !"uint", !"biases_experts_stride"}
!5 = !{i32 0, i32 4, i32 0, !"uint", !"x_ld", i32 4, i32 4, i32 0, !"uint", !"w_ld", i32 8, i32 4, i32 0, !"uint", !"scales_ld", i32 12, i32 4, i32 0, !"uint", !"biases_ld", i32 16, i32 4, i32 0, !"uint", !"x_batch_stride", i32 20, i32 4, i32 0, !"uint", !"w_batch_stride", i32 24, i32 4, i32 0, !"uint", !"scales_batch_stride", i32 28, i32 4, i32 0, !"uint", !"biases_batch_stride", i32 32, i32 4, i32 0, !"uint", !"B", i32 36, i32 4, i32 0, !"uint", !"M", i32 40, i32 4, i32 0, !"uint", !"N", i32 44, i32 4, i32 0, !"uint", !"K"}
"#,
        );
        let params = plan.buffers.iter().find(|b| b.index == 24).unwrap();
        assert_eq!(params.seed_mode, SEED_MODE_BOUNDED_CONTROL);
        for offset in [0usize, 44, 48, 60] {
            assert!(
                params.seed_layout.iter().any(|field| field.offset == offset
                    && field.size == 4
                    && field.value.is_none()),
                "missing bounded-control field at offset {offset}: {:?}",
                params.seed_layout
            );
        }

        let owned = plan_to_owned_inputs(&plan).unwrap();
        let bytes =
            seeded_buffer_bytes(owned.inputs.buffers.iter().find(|b| b.index == 24).unwrap());
        for offset in [0usize, 44, 48, 60] {
            let value = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
            assert_eq!(value, BOUNDED_CONTROL_DIM, "offset {offset}");
        }
    }

    #[test]
    fn bounded_control_result_dimensions_seed_to_zero() {
        let plan = infer_plan(
            r#"
define void @pool_like(ptr addrspace(2) %params) {
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @pool_like, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.buffer", !"air.buffer_size", i32 112, !"air.location_index", i32 4, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !4, !"air.arg_type_size", i32 112, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"PoolingParams", !"air.arg_name", !"params"}
!4 = !{i32 0, i32 4, i32 0, !"int", !"kernel_width", !"air.struct_type_info", !5, i32 80, i32 16, i32 0, !"TensorDimensions", !"result_dims", !"air.struct_type_info", !5, i32 96, i32 16, i32 0, !"TensorDimensions", !"result_strides"}
!5 = !{i32 0, i32 4, i32 0, !"int", !"n", i32 4, i32 4, i32 0, !"int", !"c", i32 8, i32 4, i32 0, !"int", !"h", i32 12, i32 4, i32 0, !"int", !"w"}
"#,
        );
        let params = plan.buffers.iter().find(|b| b.index == 4).unwrap();
        for offset in [80usize, 84, 88, 92] {
            assert!(
                params.seed_layout.iter().any(|field| field.offset == offset
                    && field.size == 4
                    && field.value == Some(0)),
                "missing zero result_dims field at offset {offset}: {:?}",
                params.seed_layout
            );
        }
        for offset in [96usize, 100, 104, 108] {
            assert!(
                params.seed_layout.iter().any(|field| field.offset == offset
                    && field.size == 4
                    && field.value.is_none()),
                "result_strides should keep bounded-control dim at offset {offset}: {:?}",
                params.seed_layout
            );
        }
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
    fn depth_cube_texture_plan_has_six_layers() {
        let plan = infer_plan(
            r#"
define <4 x half> @fragment(ptr addrspace(1) %depth, <3 x float> %coord) {
  %s = call { float, i8 } @air.sample_depth_cube.f32(ptr addrspace(1) %depth, ptr addrspace(2) null, i32 1, <3 x float> %coord, i1 true, float 0.0, float 0.0, i32 0)
  %d = extractvalue { float, i8 } %s, 0
  %c = insertelement <4 x half> zeroinitializer, half 0xH3C00, i64 0
  ret <4 x half> %c
}

declare { float, i8 } @air.sample_depth_cube.f32(ptr addrspace(1), ptr addrspace(2), i32, <3 x float>, i1, float, float, i32)

!air.fragment = !{!0}
!0 = !{ptr @fragment, !1, !2}
!1 = !{!3}
!2 = !{!4, !5}
!3 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"half4"}
!4 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"depthcube<float, sample>", !"air.arg_name", !"depth"}
!5 = !{i32 1, !"air.fragment_input", !"air.arg_type_name", !"float3", !"air.arg_name", !"coord"}
"#,
        );
        let texture = plan.textures.iter().find(|t| t.index == 0).unwrap();
        assert_eq!(texture.format, "R32Float");
        assert_eq!(texture.d, 6);
        assert_eq!(texture.seed_mode, SEED_MODE_FINITE_FLOAT32);
    }

    #[test]
    fn bounded_control_float_payload_sizes_readwrite_input() {
        let plan = infer_plan(BOUNDED_CONTROL_FLOAT_PAYLOAD_LL);
        let input = plan.buffers.iter().find(|b| b.index == 0).unwrap();
        let output = plan.buffers.iter().find(|b| b.index == 1).unwrap();
        let params = plan.buffers.iter().find(|b| b.index == 3).unwrap();

        assert_eq!(input.role, "InOut");
        assert_eq!(input.seed_mode, SEED_MODE_FINITE_FLOAT32);
        assert_eq!(
            input.len,
            BOUNDED_CONTROL_DIM as usize * BOUNDED_CONTROL_DIM as usize * 4
        );
        assert_eq!(output.len, DEFAULT_BUFFER_LEN);
        assert_eq!(plan.output.index, 1);
        assert_eq!(plan.output.len, Some(DEFAULT_BUFFER_LEN));
        assert_eq!(params.seed_mode, SEED_MODE_BOUNDED_CONTROL);
    }

    #[test]
    fn stale_bounded_control_float_payload_len_golden_is_missing() {
        let mut old_plan = infer_plan(BOUNDED_CONTROL_FLOAT_PAYLOAD_LL);
        old_plan
            .buffers
            .iter_mut()
            .find(|b| b.index == 0)
            .unwrap()
            .len = DEFAULT_BUFFER_LEN;
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
            output_sha256: Some(sha256_hex(&0.0f32.to_le_bytes())),
            output_b64: Some(encode_output_b64(&0.0f32.to_le_bytes())),
            spv_sha256: None,
            compare: "full".into(),
            fc_specialization: None,
            fc_values: None,
            input_specialization: None,
            stage: Some("Kernel".into()),
            entry: Some("k".into()),
            error: None,
        };

        let reason = incompatible_float_seed_golden(BOUNDED_CONTROL_FLOAT_PAYLOAD_LL, &metal)
            .expect("stale finite-float payload length");
        assert!(
            reason.contains("buffer 0 length 256 now sized 1024"),
            "{reason}"
        );
        assert!(reason.contains("rebank Metal row"), "{reason}");
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
    fn loop_bound_scan_sees_gep_load_from_readwrite_counter() {
        let ll = r#"
define void @bvh_like(i32 %tid, i32 %lid, ptr addrspace(1) %counters) {
entry:
  %count_ptr = getelementptr inbounds %struct.Counter, ptr addrspace(1) %counters, i64 0, i32 0
  %count = load i32, ptr addrspace(1) %count_ptr, align 4
  br label %loop
loop:
  %i = phi i32 [ 1, %entry ], [ %next, %loop ]
  %next = shl i32 %i, 1
  %keep = icmp ult i32 %next, %count
  br i1 %keep, label %loop, label %done
done:
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @bvh_like, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid"}
!4 = !{i32 1, !"air.thread_position_in_threadgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"lid"}
!5 = !{i32 2, !"air.buffer", !"air.location_index", i32 6, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.struct_type_info", !6, !"air.arg_type_size", i32 256, !"air.arg_type_name", !"Counter", !"air.arg_name", !"counters"}
!6 = !{i32 0, i32 4, i32 0, !"uint", !"count"}
"#;
        let hit = buffers_with_loads_used_as_loop_bounds(ll);
        assert!(
            hit.contains(&6),
            "expected buffer 6 (counter load -> icmp -> loop br), got {hit:?}"
        );
        let plan = infer_plan(ll);
        let counters = plan.buffers.iter().find(|b| b.index == 6).unwrap();
        assert_eq!(counters.seed_mode, SEED_MODE_BOUNDED_CONTROL);
    }

    #[test]
    fn loop_bound_scan_uses_air_entry_after_static_initializer() {
        let ll = r#"
@fc = internal unnamed_addr addrspace(2) externally_initialized constant i32 undef, section "air.fc_initializer", align 4
@g = internal unnamed_addr addrspace(2) global i32 undef, align 4

define internal void @_GLOBAL__sub_I_test() section "air.static_init" {
  %v = load i32, ptr addrspace(2) @fc, align 4
  store i32 %v, ptr addrspace(2) @g, align 4
  ret void
}

define void @k(i32 %tgid, ptr addrspace(1) %queue) {
entry:
  %idx = zext i32 %tgid to i64
  %count_ptr = getelementptr inbounds %struct.Queue, ptr addrspace(1) %queue, i64 %idx, i32 1
  %count = load i32, ptr addrspace(1) %count_ptr, align 4
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %next, %loop ]
  %step = load i32, ptr addrspace(2) @g, align 4
  %next = add i32 %step, %i
  %keep = icmp ult i32 %next, %count
  br i1 %keep, label %loop, label %done
done:
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.threadgroup_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"tgid"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 14, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.struct_type_info", !5, !"air.arg_type_size", i32 8, !"air.arg_type_name", !"Queue", !"air.arg_name", !"queue"}
!5 = !{i32 0, i32 4, i32 0, !"uint", !"base", i32 4, i32 4, i32 0, !"uint", !"count"}
"#;
        let hit = buffers_with_loads_used_as_loop_bounds(ll);
        assert!(
            hit.contains(&14),
            "expected buffer 14 from the AIR entry body, got {hit:?}"
        );
        let plan = infer_plan(ll);
        let queue = plan.buffers.iter().find(|b| b.index == 14).unwrap();
        assert_eq!(queue.seed_mode, SEED_MODE_BOUNDED_CONTROL);
    }

    #[test]
    fn loop_bound_scan_tracks_preheader_load_through_loop_phi() {
        let ll = r#"
define void @k(i32 %tid, ptr addrspace(1) %queue) {
entry:
  %idx = zext i32 %tid to i64
  %limit_ptr = getelementptr inbounds %struct.Queue, ptr addrspace(1) %queue, i64 %idx, i32 3
  %limit0 = load i32, ptr addrspace(1) %limit_ptr, align 4
  br label %loop
loop:
  %limit = phi i32 [ %limit0, %entry ], [ %limit1, %latch ]
  %i = phi i32 [ 0, %entry ], [ %next, %latch ]
  %next = add i32 %i, 16
  %keep = icmp ult i32 %next, %limit
  br i1 %keep, label %latch, label %done
latch:
  %limit1 = load i32, ptr addrspace(1) %limit_ptr, align 4
  br label %loop
done:
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 16, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.struct_type_info", !5, !"air.arg_type_size", i32 16, !"air.arg_type_name", !"Queue", !"air.arg_name", !"queue"}
!5 = !{i32 0, i32 4, i32 0, !"uint", !"a", i32 4, i32 4, i32 0, !"uint", !"b", i32 8, i32 4, i32 0, !"uint", !"c", i32 12, i32 4, i32 0, !"uint", !"limit"}
!6 = !{i32 1, !"air.indirect_constant", !"air.location_index", i32 99, i32 1, !"air.arg_type_name", !"uint", !"air.arg_name", !"nested"}
"#;
        let hit = buffers_with_loads_used_as_loop_bounds(ll);
        assert!(
            hit.contains(&16),
            "expected buffer 16 from preheader load carried through a loop phi, got {hit:?}"
        );
        assert!(
            !hit.contains(&99),
            "nested metadata must not override the entry buffer binding, got {hit:?}"
        );
        let plan = infer_plan(ll);
        let queue = plan.buffers.iter().find(|b| b.index == 16).unwrap();
        assert_eq!(queue.seed_mode, SEED_MODE_BOUNDED_CONTROL);
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
            input_specialization: None,
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
    fn atomic_counter_buffer_seeds_every_counter_lane_zero() {
        let ll = r#"
define void @scan(ptr addrspace(1) %counter, ptr addrspace(1) %scratch) {
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @scan, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_name", !"metal::_atomic", !"air.arg_name", !"counter"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 3, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_name", !"int", !"air.arg_name", !"scratch"}
"#;
        let plan = infer_plan(ll);
        let counter = plan
            .buffers
            .iter()
            .find(|b| b.index == 2)
            .expect("atomic counter buffer");
        let scratch = plan
            .buffers
            .iter()
            .find(|b| b.index == 3)
            .expect("plain int scratch buffer");

        assert_eq!(counter.seed_mode, SEED_MODE_BOUNDED_CONTROL);
        assert_eq!(counter.seed_layout.len(), counter.len / 4);
        assert!(counter
            .seed_layout
            .iter()
            .all(|field| field.size == 4 && field.value == Some(0)));
        assert!(
            bounded_control_buffer_bytes_with_layout(counter.len, &counter.seed_layout)
                .chunks_exact(4)
                .all(|lane| lane == [0, 0, 0, 0])
        );
        assert_eq!(scratch.seed_mode, SEED_MODE_DETERMINISTIC);
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
            input_specialization: None,
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
    fn bounded_control_bool_fields_seed_to_valid_range() {
        let ll = r#"
%struct.Config = type { i8, i8, i32 }

define void @flags(ptr addrspace(1) %dst, ptr addrspace(2) %config, i32 %tid) {
entry:
  %flagp = getelementptr inbounds %struct.Config, ptr addrspace(2) %config, i64 0, i32 0
  %flag = load i8, ptr addrspace(2) %flagp, align 4, !range !7
  %enabled = icmp ne i8 %flag, 0
  %value = select i1 %enabled, i8 1, i8 0
  %idx = zext i32 %tid to i64
  %out = getelementptr inbounds i8, ptr addrspace(1) %dst, i64 %idx
  store i8 %value, ptr addrspace(1) %out, align 1
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @flags, !1, !2}
!1 = !{}
!2 = !{!3, !4, !6}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 1, !"air.arg_type_name", !"uchar", !"air.arg_name", !"dst"}
!4 = !{i32 1, !"air.buffer", !"air.buffer_size", i32 8, !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !5, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"Config", !"air.arg_name", !"config"}
!5 = !{i32 0, i32 1, i32 0, !"bool", !"enabled", i32 1, i32 1, i32 0, !"bool", !"flip", i32 4, i32 4, i32 0, !"uint", !"count"}
!6 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid"}
!7 = !{i8 0, i8 2}
"#;
        let current_plan = infer_plan(ll);
        let config = current_plan
            .buffers
            .iter()
            .find(|b| b.index == 1)
            .expect("config buffer");
        assert_eq!(
            config.seed_layout,
            vec![
                ControlSeedField {
                    offset: 0,
                    size: 1,
                    value: Some(0)
                },
                ControlSeedField {
                    offset: 1,
                    size: 1,
                    value: Some(0)
                },
                ControlSeedField {
                    offset: 4,
                    size: 4,
                    value: None
                },
            ]
        );

        let owned = plan_to_owned_inputs(&current_plan).unwrap();
        let config_input = owned
            .inputs
            .buffers
            .iter()
            .find(|b| b.index == 1)
            .expect("config buffer");
        let bytes = seeded_buffer_bytes(config_input);
        assert_eq!(bytes[0], 0);
        assert_eq!(bytes[1], 0);
        assert_eq!(
            u32::from_le_bytes(bytes[4..8].try_into().unwrap()),
            BOUNDED_CONTROL_DIM
        );

        let mut old_plan = current_plan;
        for field in &mut old_plan
            .buffers
            .iter_mut()
            .find(|b| b.index == 1)
            .expect("config buffer")
            .seed_layout
        {
            if field.size == 1 {
                field.value = None;
            }
        }
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
            input_specialization: None,
            stage: Some("Kernel".into()),
            entry: Some("flags".into()),
            error: None,
        };

        let reason = incompatible_bounded_control_seed_golden(ll, &metal)
            .expect("stale bool bounded-control seed");
        assert!(reason.contains("buffer 1"), "{reason}");
        assert!(reason.contains("typed AIR control metadata"), "{reason}");
        assert!(reason.contains("rebank Metal row"), "{reason}");
    }

    #[test]
    fn loop_input_facts_include_bounded_control_struct_fields() {
        let ll = r#"
define void @kernel(ptr addrspace(2) %params) {
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @kernel, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 7, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !4, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"Params", !"air.arg_name", !"params"}
!4 = !{i32 0, i32 4, i32 0, !"uint", !"width", i32 4, i32 2, i32 0, !"ushort", !"height"}
"#;
        let arg_names = entry_arg_names(ll, "kernel");
        let plan = HarnessPlan {
            buffers: vec![PlanBuffer {
                index: 7,
                len: 8,
                role: "Input".into(),
                seed_tag: 1,
                seed_mode: SEED_MODE_BOUNDED_CONTROL.into(),
                seed_layout: vec![
                    ControlSeedField {
                        offset: 0,
                        size: 4,
                        value: Some(3),
                    },
                    ControlSeedField {
                        offset: 4,
                        size: 2,
                        value: Some(5),
                    },
                ],
                seed_stride: None,
            }],
            textures: Vec::new(),
            output: PlanOutput {
                kind: "buffer".into(),
                index: 7,
                format: "RawBytes".into(),
                len: Some(8),
                w: None,
                h: None,
                d: None,
            },
            dispatch_grid: [1, 1, 1],
            dispatch_tg: [1, 1, 1],
        };
        let inputs = plan_to_owned_inputs(&plan).unwrap();
        let mut fields = exact_struct_buffer_arg_field_values(ll, &arg_names, &inputs.inputs);
        fields.sort();
        assert_eq!(
            fields,
            vec![
                ("params".to_string(), vec![0], 3),
                ("params".to_string(), vec![1], 5),
            ]
        );
    }

    #[test]
    fn loop_input_facts_include_bounded_control_device_scalar() {
        let ll = r#"
define void @kernel(ptr addrspace(1) %counter) {
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @kernel, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 7, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"counter"}
"#;
        let arg_names = entry_arg_names(ll, "kernel");
        let plan = HarnessPlan {
            buffers: vec![PlanBuffer {
                index: 7,
                len: 4,
                role: "InOut".into(),
                seed_tag: 1,
                seed_mode: SEED_MODE_BOUNDED_CONTROL.into(),
                seed_layout: vec![ControlSeedField {
                    offset: 0,
                    size: 4,
                    value: Some(0),
                }],
                seed_stride: None,
            }],
            textures: Vec::new(),
            output: PlanOutput {
                kind: "buffer".into(),
                index: 1,
                format: "RawBytes".into(),
                len: Some(4),
                w: None,
                h: None,
                d: None,
            },
            dispatch_grid: [1, 1, 1],
            dispatch_tg: [1, 1, 1],
        };
        let inputs = plan_to_owned_inputs(&plan).unwrap();
        assert_eq!(
            exact_scalar_buffer_arg_values(ll, &arg_names, &inputs.inputs),
            vec![("counter".to_string(), 0)]
        );
    }

    #[test]
    fn loop_input_facts_include_bounded_control_vector_buffer_lanes() {
        let ll = r#"
define void @kernel(ptr addrspace(2) %tile_locations) {
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @kernel, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 7, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_name", !"ushort2", !"air.arg_name", !"tile_locations"}
"#;
        let arg_names = entry_arg_names(ll, "kernel");
        let plan = HarnessPlan {
            buffers: vec![PlanBuffer {
                index: 7,
                len: 4,
                role: "Input".into(),
                seed_tag: 1,
                seed_mode: SEED_MODE_BOUNDED_CONTROL.into(),
                seed_layout: vec![
                    ControlSeedField {
                        offset: 0,
                        size: 2,
                        value: Some(16),
                    },
                    ControlSeedField {
                        offset: 2,
                        size: 2,
                        value: Some(2),
                    },
                ],
                seed_stride: None,
            }],
            textures: Vec::new(),
            output: PlanOutput {
                kind: "buffer".into(),
                index: 7,
                format: "RawBytes".into(),
                len: Some(4),
                w: None,
                h: None,
                d: None,
            },
            dispatch_grid: [1, 1, 1],
            dispatch_tg: [1, 1, 1],
        };
        let inputs = plan_to_owned_inputs(&plan).unwrap();
        let mut lanes = exact_vector_buffer_arg_values(ll, &arg_names, &inputs.inputs);
        lanes.sort();
        assert_eq!(
            lanes,
            vec![
                ("tile_locations".to_string(), 0, 16),
                ("tile_locations".to_string(), 1, 2),
            ]
        );
    }

    #[test]
    fn loop_input_facts_include_bounded_control_float_buffer_value() {
        let ll = r#"
define void @kernel(ptr addrspace(2) %rank_mode) {
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @kernel, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"rank_mode"}
"#;
        let arg_names = entry_arg_names(ll, "kernel");
        let plan = HarnessPlan {
            buffers: vec![PlanBuffer {
                index: 0,
                len: 4,
                role: "Input".into(),
                seed_tag: 1,
                seed_mode: SEED_MODE_BOUNDED_CONTROL.into(),
                seed_layout: vec![ControlSeedField {
                    offset: 0,
                    size: 4,
                    value: Some(0),
                }],
                seed_stride: None,
            }],
            textures: Vec::new(),
            output: PlanOutput {
                kind: "buffer".into(),
                index: 0,
                format: "RawBytes".into(),
                len: Some(4),
                w: None,
                h: None,
                d: None,
            },
            dispatch_grid: [1, 1, 1],
            dispatch_tg: [1, 1, 1],
        };
        let inputs = plan_to_owned_inputs(&plan).unwrap();
        assert_eq!(
            exact_float_buffer_arg_values(ll, &arg_names, &inputs.inputs),
            vec![("rank_mode".to_string(), 0.0)]
        );
    }

    #[test]
    fn stale_byte_gep_stride_control_seed_golden_is_missing() {
        let ll = r#"
define void @byte_stride(ptr addrspace(1) %src, ptr addrspace(1) %dst, ptr addrspace(1) %stride, i32 %tid) {
entry:
  %s = load i32, ptr addrspace(1) %stride, align 4
  %base = mul i32 %tid, %s
  %idx = zext i32 %base to i64
  %p = getelementptr inbounds i8, ptr addrspace(1) %src, i64 %idx
  %q = bitcast ptr addrspace(1) %p to ptr addrspace(1)
  %v = load <4 x i16>, ptr addrspace(1) %q, align 8
  %out = tail call <4 x i8> @air.convert.u.v4i8.u.v4i16(<4 x i16> %v)
  %dstp = getelementptr inbounds i8, ptr addrspace(1) %dst, i64 %idx
  store <4 x i8> %out, ptr addrspace(1) %dstp, align 4
  ret void
}

declare <4 x i8> @air.convert.u.v4i8.u.v4i16(<4 x i16>)

!air.kernel = !{!0}
!0 = !{ptr @byte_stride, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5, !6}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 1, !"air.arg_type_name", !"uchar", !"air.arg_name", !"src"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 1, !"air.arg_type_name", !"uchar", !"air.arg_name", !"dst"}
!5 = !{i32 2, !"air.buffer", !"air.buffer_size", i32 4, !"air.location_index", i32 2, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"stride"}
!6 = !{i32 3, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid"}
"#;
        let current_plan = infer_plan(ll);
        let stride = current_plan
            .buffers
            .iter()
            .find(|b| b.index == 2)
            .expect("stride control");
        assert_eq!(stride.seed_mode, SEED_MODE_BOUNDED_CONTROL);

        let mut old_plan = current_plan;
        old_plan
            .buffers
            .iter_mut()
            .find(|b| b.index == 2)
            .expect("stride control")
            .seed_mode = SEED_MODE_DETERMINISTIC.into();
        let metal = MetalRow {
            air_sha256: "x".into(),
            shard: None,
            label: String::new(),
            status: "ok".into(),
            backend: "metal".into(),
            seed_profile: "deterministic_v5_typed_bounded_control".into(),
            plan_version: PLAN_VERSION,
            plan: old_plan,
            input_sha256: None,
            output_sha256: Some(sha256_hex(&[])),
            output_b64: Some(encode_output_b64(&[])),
            spv_sha256: None,
            compare: "full".into(),
            fc_specialization: None,
            fc_values: None,
            input_specialization: None,
            stage: Some("Kernel".into()),
            entry: Some("byte_stride".into()),
            error: None,
        };

        let reason = incompatible_bounded_control_seed_golden(ll, &metal)
            .expect("stale byte-GEP stride control seed");
        assert!(reason.contains("buffer 2"), "{reason}");
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
            input_specialization: None,
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
    fn thread_indexed_scalar_input_is_sized_to_dispatch_grid() {
        let ll = r#"
define void @copy(ptr addrspace(1) %src, ptr addrspace(1) %dst, i32 %tid) {
entry:
  %idx = zext i32 %tid to i64
  %p = getelementptr inbounds i32, ptr addrspace(1) %src, i64 %idx
  %v = load i32, ptr addrspace(1) %p, align 4
  %q = getelementptr inbounds i32, ptr addrspace(1) %dst, i64 %idx
  store i32 %v, ptr addrspace(1) %q, align 4
  ret void
}
!air.kernel = !{!0}
!0 = !{ptr @copy, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"src"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"dst"}
!5 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid"}
"#;
        let plan = infer_plan(ll);
        let src = plan
            .buffers
            .iter()
            .find(|buffer| buffer.index == 0)
            .expect("source buffer");

        assert_eq!(plan.dispatch_grid, [DEFAULT_DISPATCH_GRID_X as u32, 1, 1]);
        assert_eq!(src.len, DEFAULT_DISPATCH_GRID_X * 4);
        assert!(incompatible_oob_vector_input_golden(
            ll,
            &metal_row_for_compare(&[], plan, Some("Kernel"))
        )
        .is_none());
    }

    #[test]
    fn banked_function_constants_do_not_freeze_stale_metal_plan() {
        let ll = r#"
define void @copy(ptr addrspace(1) %src, ptr addrspace(1) %dst, i32 %tid) {
entry:
  %idx = zext i32 %tid to i64
  %p = getelementptr inbounds i32, ptr addrspace(1) %src, i64 %idx
  %v = load i32, ptr addrspace(1) %p, align 4
  %q = getelementptr inbounds i32, ptr addrspace(1) %dst, i64 %idx
  store i32 %v, ptr addrspace(1) %q, align 4
  ret void
}
!air.kernel = !{!0}
!0 = !{ptr @copy, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"src"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"dst"}
!5 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid"}
"#;
        let mut banked = metal_row_for_compare(&[], infer_plan(ll), Some("Kernel"));
        banked.plan.buffers.iter_mut().for_each(|buffer| {
            if buffer.index == 0 {
                buffer.len = 4;
            }
        });
        banked.fc_values = Some(vec![FunctionConstantValueJson {
            index: 11,
            value: 2,
        }]);

        let (plan, fc_values) = metal_oracle_inputs(ll, Some(&banked));
        let src = plan
            .buffers
            .iter()
            .find(|buffer| buffer.index == 0)
            .expect("source buffer");

        assert_eq!(src.len, DEFAULT_DISPATCH_GRID_X * 4);
        assert_eq!(fc_values, vec![(11, 2)]);
    }

    #[test]
    fn output_stride_seed_one_vector_store_is_sized_to_store_width() {
        let ll = r#"
define void @byte_output_stride(ptr addrspace(1) %src, ptr addrspace(1) %dst, ptr addrspace(1) %stride, i32 %tid) local_unnamed_addr #0 {
entry:
  %s = load i32, ptr addrspace(1) %stride, align 4
  %base = mul i32 %tid, %s
  %idx = zext i32 %base to i64
  %p = getelementptr inbounds i8, ptr addrspace(1) %src, i64 %idx
  %q = bitcast ptr addrspace(1) %p to ptr addrspace(1)
  %v = load <4 x i16>, ptr addrspace(1) %q, align 8
  %dstp = getelementptr inbounds i8, ptr addrspace(1) %dst, i64 %idx
  store <4 x i16> %v, ptr addrspace(1) %dstp, align 8
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @byte_output_stride, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5, !6}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 1, !"air.arg_type_name", !"uchar", !"air.arg_name", !"src"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 1, !"air.arg_type_name", !"uchar", !"air.arg_name", !"dst"}
!5 = !{i32 2, !"air.buffer", !"air.buffer_size", i32 4, !"air.location_index", i32 2, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"stride"}
!6 = !{i32 3, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid"}

attributes #0 = { nounwind }
"#;
        let plan = infer_plan(ll);
        let stride = plan
            .buffers
            .iter()
            .find(|b| b.index == 2)
            .expect("stride control");
        assert_eq!(stride.seed_mode, SEED_MODE_BOUNDED_CONTROL);
        assert_eq!(
            stride.seed_layout,
            vec![ControlSeedField {
                offset: 0,
                size: 4,
                value: Some(128)
            }]
        );
        let metal = metal_row_for_compare(&[], plan, Some("Kernel"));
        assert!(incompatible_overlapping_output_stride_golden(ll, &metal).is_none());

        let mut old_plan = metal.plan.clone();
        old_plan
            .buffers
            .iter_mut()
            .find(|b| b.index == 2)
            .expect("old stride control")
            .seed_layout[0]
            .value = Some(1);
        let metal = metal_row_for_compare(&[], old_plan, Some("Kernel"));
        let reason = incompatible_overlapping_output_stride_golden(ll, &metal)
            .expect("overlapping output stride should need rebank");
        assert!(
            reason.contains("output byte-stride/control buffer too small"),
            "{reason}"
        );
        assert!(reason.contains("stores 8 bytes"), "{reason}");
        assert!(reason.contains("rebank Metal row"), "{reason}");
    }

    #[test]
    fn output_stride_struct_field_is_sized_to_store_width() {
        let ll = r#"
%struct.Params = type { i32, i32, i32 }

define void @byte_output_stride_struct(ptr addrspace(1) %dst, ptr addrspace(2) %params, i32 %tid) local_unnamed_addr #0 {
entry:
  %stridep = getelementptr inbounds %struct.Params, ptr addrspace(2) %params, i64 0, i32 2
  %stride = load i32, ptr addrspace(2) %stridep, align 4
  %row = mul i32 %stride, %tid
  %row64 = zext i32 %row to i64
  %base = getelementptr inbounds i8, ptr addrspace(1) %dst, i64 %row64
  store float 1.000000e+00, ptr addrspace(1) %base, align 4
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @byte_output_stride_struct, !1, !2}
!1 = !{}
!2 = !{!3, !4, !6}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"dst"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !5, !"air.arg_type_size", i32 12, !"air.arg_type_name", !"Params", !"air.arg_name", !"params"}
!5 = !{!"air.struct_type_info", i32 0, i32 4, i32 0, !"uint", !"m", i32 4, i32 4, i32 0, !"uint", !"n", i32 8, i32 4, i32 0, !"uint", !"rowBytes"}
!6 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid"}

attributes #0 = { nounwind }
"#;
        let plan = infer_plan(ll);
        let params = plan
            .buffers
            .iter()
            .find(|b| b.index == 1)
            .expect("params control");
        assert_eq!(params.seed_mode, SEED_MODE_BOUNDED_CONTROL);
        assert!(params
            .seed_layout
            .iter()
            .any(|field| field.offset == 8 && field.value == Some(64)));
        let metal = metal_row_for_compare(&[], plan, Some("Kernel"));
        assert!(incompatible_overlapping_output_stride_golden(ll, &metal).is_none());

        let mut old_plan = metal.plan.clone();
        for field in &mut old_plan
            .buffers
            .iter_mut()
            .find(|b| b.index == 1)
            .expect("old params control")
            .seed_layout
        {
            field.value = None;
        }
        let metal = metal_row_for_compare(&[], old_plan, Some("Kernel"));
        let reason = incompatible_overlapping_output_stride_golden(ll, &metal)
            .expect("default bounded-control stride should need rebank");
        assert!(
            reason.contains("output byte-stride/control buffer too small"),
            "{reason}"
        );
        assert!(reason.contains("stores 4 bytes"), "{reason}");
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
    fn bounded_control_strided_input_oob_golden_is_missing() {
        let ll = r#"
define void @mask(ptr addrspace(1) readonly %input, ptr addrspace(1) %out, ptr addrspace(2) readonly %size, <2 x i16> %gid) #0 {
  %y16 = extractelement <2 x i16> %gid, i64 1
  %y = zext i16 %y16 to i32
  %dims = load <2 x i32>, ptr addrspace(2) %size, align 8
  %h = extractelement <2 x i32> %dims, i64 1
  %h_last = add i32 %h, -1
  %row = tail call i32 @air.min.u.i32(i32 %y, i32 %h_last)
  %x16 = extractelement <2 x i16> %gid, i64 0
  %x = zext i16 %x16 to i32
  %w = extractelement <2 x i32> %dims, i64 0
  %base = mul i32 %row, %w
  %idx = add i32 %base, %x
  %idx64 = zext i32 %idx to i64
  %p = getelementptr inbounds i32, ptr addrspace(1) %input, i64 %idx64
  %v = load i32, ptr addrspace(1) %p, align 4
  %zero = icmp eq i32 %v, 0
  br i1 %zero, label %one, label %none

one:
  tail call void @air.write_texture_2d.i16.v4f16(ptr addrspace(1) %out, <2 x i16> %gid, <4 x half> splat (half 0xH3C00), i16 0, i32 2)
  br label %done

none:
  tail call void @air.write_texture_2d.i16.v4f16(ptr addrspace(1) %out, <2 x i16> %gid, <4 x half> zeroinitializer, i16 0, i32 2)
  br label %done

done:
  ret void
}

declare i32 @air.min.u.i32(i32, i32)
declare void @air.write_texture_2d.i16.v4f16(ptr addrspace(1), <2 x i16>, <4 x half>, i16, i32)
attributes #0 = { "no-nans-fp-math"="true" "unsafe-fp-math"="true" }

!air.kernel = !{!0}
!0 = !{ptr @mask, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5, !6}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"input"}
!4 = !{i32 1, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.write", !"air.arg_type_name", !"texture2d<half, write>", !"air.arg_name", !"out"}
!5 = !{i32 2, !"air.buffer", !"air.buffer_size", i32 8, !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"uint2", !"air.arg_name", !"size"}
!6 = !{i32 3, !"air.thread_position_in_grid", !"air.arg_type_name", !"ushort2", !"air.arg_name", !"gid"}
"#;
        let metal = metal_row_for_compare(&[], infer_plan(ll), Some("Kernel"));
        let reason = incompatible_bounded_control_strided_input_oob_golden(ll, &metal)
            .expect("bounded-control strided input OOB");
        assert!(reason.contains("width=16 height=16"), "{reason}");
        assert!(reason.contains("read past the input buffer"), "{reason}");
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
            input_specialization: None,
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
            input_specialization: None,
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
            input_specialization: None,
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
            input_specialization: None,
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
            input_specialization: None,
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
        assert_eq!(
            plan.buffers
                .iter()
                .find(|buffer| buffer.index == 0)
                .map(|buffer| buffer.role.as_str()),
            Some("InOut")
        );
        assert_eq!(
            plan.buffers
                .iter()
                .find(|buffer| buffer.index == 1)
                .map(|buffer| buffer.role.as_str()),
            Some("Output")
        );
    }

    #[test]
    fn output_writeonly_buffer_plan_role_drift_requires_rebank() {
        let ll = r#"
define void @copy(ptr addrspace(1) noundef writeonly "air-buffer-no-alias" %out, i32 %tid) {
entry:
  %idx = zext i32 %tid to i64
  %dst = getelementptr i32, ptr addrspace(1) %out, i64 %idx
  store i32 %tid, ptr addrspace(1) %dst, align 4
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @copy, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.buffer", !"air.buffer_size", i32 256, !"air.location_index", i32 4, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!4 = !{i32 1, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid"}
"#;
        let current = infer_plan(ll);
        assert_eq!(
            current
                .buffers
                .iter()
                .find(|buffer| buffer.index == 4)
                .map(|buffer| buffer.role.as_str()),
            Some("Output")
        );

        let mut banked = current.clone();
        banked
            .buffers
            .iter_mut()
            .find(|buffer| buffer.index == 4)
            .expect("output buffer")
            .role = "InOut".into();
        let metal = metal_row_for_compare(&[], banked, Some("Kernel"));
        let reason = incompatible_output_plan_golden(ll, &metal)
            .expect("stale output writeonly buffer plan");
        assert!(reason.contains("output buffer plan"), "{reason}");
        assert!(reason.contains("role=InOut"), "{reason}");
        assert!(reason.contains("role=Output"), "{reason}");
    }

    #[test]
    fn subgroup_lane_zero_texture_write_race_requires_rebank() {
        let ll = r#"
@__air_sampler_state = internal addrspace(2) constant [2 x i64] zeroinitializer, align 8

define void @reduce(ptr addrspace(1) %src, ptr addrspace(1) %out, <2 x i16> %tgid, i16 %lane) #0 {
entry:
  %coord_f = tail call fast <2 x float> @air.convert.f.v2f32.u.v2i16(<2 x i16> %tgid)
  %sample = tail call { <4 x float>, i8 } @air.gather_texture_2d.v4f32(ptr addrspace(1) %src, ptr addrspace(2) @__air_sampler_state, <2 x float> %coord_f, i1 true, <2 x i32> zeroinitializer, i32 0, i32 0)
  %vec = extractvalue { <4 x float>, i8 } %sample, 0
  %x = extractelement <4 x float> %vec, i64 0
  %min = tail call fast float @air.simd_min.f32(float %x)
  %is_lane0 = icmp eq i16 %lane, 0
  br i1 %is_lane0, label %write, label %done

write:
  %coord = shl <2 x i16> %tgid, splat (i16 1)
  %value = insertelement <4 x float> zeroinitializer, float %min, i64 0
  tail call void @air.write_texture_2d.i16.v4f32(ptr addrspace(1) %out, <2 x i16> %coord, <4 x float> %value, i16 0, i32 2)
  br label %done

done:
  ret void
}

declare <2 x float> @air.convert.f.v2f32.u.v2i16(<2 x i16>)
declare { <4 x float>, i8 } @air.gather_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i32, i32)
declare float @air.simd_min.f32(float)
declare void @air.write_texture_2d.i16.v4f32(ptr addrspace(1), <2 x i16>, <4 x float>, i16, i32)
attributes #0 = { "no-nans-fp-math"="true" "unsafe-fp-math"="true" }

!air.kernel = !{!0}
!0 = !{ptr @reduce, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5, !6}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"src"}
!4 = !{i32 1, !"air.texture", !"air.location_index", i32 1, i32 1, !"air.write", !"air.arg_type_name", !"texture2d<float, write>", !"air.arg_name", !"out"}
!5 = !{i32 2, !"air.threadgroup_position_in_grid", !"air.arg_type_name", !"ushort2", !"air.arg_name", !"tgid"}
!6 = !{i32 3, !"air.thread_index_in_simdgroup", !"air.arg_type_name", !"ushort", !"air.arg_name", !"lane"}
"#;
        let mut plan = infer_plan(ll);
        plan.output.kind = "texture".into();
        plan.output.index = 1;
        plan.output.format = "R32Float".into();
        plan.dispatch_tg = [8, 8, 1];
        let metal = metal_row_for_compare(&[], plan, Some("Kernel"));

        let reason = incompatible_subgroup_texture_write_race_golden(ll, &metal)
            .expect("lane-zero subgroup texture write race");
        assert!(reason.contains("thread_index_in_simdgroup==0"), "{reason}");
        assert!(reason.contains("backend-schedule-dependent"), "{reason}");
    }

    #[test]
    fn downscaled_texture_write_race_requires_rebank() {
        let ll = r#"
define void @downscale(ptr addrspace(1) %src, ptr addrspace(1) %out, <2 x i32> %gid) #0 {
  %s = tail call { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1) %src, ptr addrspace(2) null, <2 x i32> %gid, <2 x i32> zeroinitializer, i32 0, i32 1)
  %v = extractvalue { <4 x float>, i8 } %s, 0
  %sat = tail call fast <4 x float> @air.fast_saturate.v4f32(<4 x float> %v)
  %dst = lshr <2 x i32> %gid, splat (i32 1)
  tail call void @air.write_texture_2d.v4f32(ptr addrspace(1) %out, <2 x i32> %dst, <4 x float> %sat, i32 0, i32 2)
  ret void
}

declare { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x i32>, <2 x i32>, i32, i32)
declare <4 x float> @air.fast_saturate.v4f32(<4 x float>)
declare void @air.write_texture_2d.v4f32(ptr addrspace(1), <2 x i32>, <4 x float>, i32, i32)
attributes #0 = { "no-nans-fp-math"="true" "unsafe-fp-math"="true" }

!air.kernel = !{!0}
!0 = !{ptr @downscale, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.read", !"air.arg_type_name", !"texture2d<float, read>", !"air.arg_name", !"src"}
!4 = !{i32 1, !"air.texture", !"air.location_index", i32 1, i32 1, !"air.write", !"air.arg_type_name", !"texture2d<float, write>", !"air.arg_name", !"out"}
!5 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint2", !"air.arg_name", !"gid"}
"#;
        let metal = metal_row_for_compare(&[], infer_plan(ll), Some("Kernel"));
        let reason = incompatible_many_to_one_texture_write_golden(ll, &metal)
            .expect("downscaled texture write race");
        assert!(reason.contains("downscaled output coordinate"), "{reason}");
        assert!(reason.contains("order-dependent"), "{reason}");
    }

    #[test]
    fn texture_array_write_omitting_grid_lane_requires_rebank() {
        let ll = r#"
define void @array_write(ptr addrspace(1) %out, ptr addrspace(1) readonly %src, <3 x i32> %gid) #0 {
  %z = extractelement <3 x i32> %gid, i64 2
  %x = extractelement <3 x i32> %gid, i64 0
  %idx = zext i32 %x to i64
  %p = getelementptr inbounds float, ptr addrspace(1) %src, i64 %idx
  %v0 = load float, ptr addrspace(1) %p, align 4
  %v = insertelement <4 x float> zeroinitializer, float %v0, i64 0
  %coord = shufflevector <3 x i32> %gid, <3 x i32> poison, <2 x i32> <i32 0, i32 2>
  tail call void @air.write_texture_2d_array.v4f32(ptr addrspace(1) %out, <2 x i32> %coord, i32 %z, <4 x float> %v, i32 0, i32 2)
  ret void
}

declare void @air.write_texture_2d_array.v4f32(ptr addrspace(1), <2 x i32>, i32, <4 x float>, i32, i32)
attributes #0 = { "no-nans-fp-math"="true" "unsafe-fp-math"="true" }

!air.kernel = !{!0}
!0 = !{ptr @array_write, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.write", !"air.arg_type_name", !"texture2d_array<float, write>", !"air.arg_name", !"out"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"src"}
!5 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint3", !"air.arg_name", !"gid"}
"#;
        let metal = metal_row_for_compare(&[], infer_plan(ll), Some("Kernel"));
        let reason = incompatible_many_to_one_texture_write_golden(ll, &metal)
            .expect("texture2d_array omitted grid lane race");
        assert!(reason.contains("texture2d_array"), "{reason}");
        assert!(reason.contains("omit a varying grid dimension"), "{reason}");
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
            input_specialization: None,
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
            input_specialization: None,
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
            input_specialization: None,
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
    fn stale_vector_float_bounded_control_layout_golden_is_missing() {
        let mut old_plan = infer_plan(FLOAT2_DEPTH_DIVIDE_CONTROL_LL);
        let near_far = old_plan
            .buffers
            .iter_mut()
            .find(|b| b.index == 0)
            .expect("near/far buffer");
        near_far.seed_layout.clear();
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
            input_specialization: None,
            stage: None,
            entry: None,
            error: None,
        };

        let reason =
            incompatible_bounded_control_seed_golden(FLOAT2_DEPTH_DIVIDE_CONTROL_LL, &metal)
                .expect("stale vector float bounded-control layout");
        assert!(reason.contains("buffer 0"), "{reason}");
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
            input_specialization: None,
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
            input_specialization: None,
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
    fn compare_bfloat_raw_output_uses_bfloat_tolerance() {
        let ll = r#"
define void @kernel(ptr addrspace(1) %out) {
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @kernel, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 2, !"air.arg_type_align_size", i32 2, !"air.arg_type_name", !"bfloat", !"air.arg_name", !"out"}
"#;
        let plan = infer_plan(ll);
        assert_eq!(plan.output.kind, "buffer");
        assert_eq!(plan.output.format, "RawBytes");
        assert_eq!(
            plan.buffers
                .iter()
                .find(|buffer| buffer.index == 1)
                .map(|buffer| buffer.seed_mode.as_str()),
            Some(SEED_MODE_FINITE_BFLOAT16)
        );

        let golden = 0xb81fu16.to_le_bytes();
        let candidate = 0xb81eu16.to_le_bytes();
        let metal = metal_row_for_compare(&golden, plan, Some("Kernel"));
        let out_hash = sha256_hex(&candidate);
        let golden_hash = metal.output_sha256.clone().unwrap();
        let (status, observed, tolerance) = compare_candidate_to_metal(
            &candidate,
            &metal,
            &out_hash,
            &golden_hash,
            DataFormat::RawBytes,
            Some(ll),
        );

        assert_eq!(status, "tolerance");
        assert_eq!(observed.and_then(|m| m.max_ulp), Some(1));
        assert_eq!(
            tolerance.as_ref().map(|t| t.kind.as_str()),
            Some("BFloat16AbsOrUlp")
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
            input_specialization: None,
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
    fn compile_missing_fast_math_nonfinite_domain_is_smoke() {
        let tolerance = ToleranceSpecJson {
            kind: "FastMathNonFiniteDomain".into(),
            max_abs: None,
            max_ulp: None,
        };
        let error = compile_missing_compare_smoke_error(
            "missing",
            Some(&tolerance),
            Some("metal golden compares a non-finite result from AIR fast/no-nans domain-sensitive math; rebank validation inputs away from undefined fast-math domains"),
        )
        .expect("smoke error");

        assert!(error.starts_with("compile-only: "), "{error}");
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
            input_specialization: None,
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
            input_specialization: None,
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
            input_specialization: None,
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
            input_specialization: None,
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
    fn compare_sampled_f32_storage_texture_uses_texture_tolerance() {
        let golden = 0.0f32.to_le_bytes();
        let candidate = 0.0015f32.to_le_bytes();
        let ll = r#"
define void @kernel(ptr addrspace(1) %src, ptr addrspace(1) %out) {
  %s = tail call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) %src, ptr addrspace(2) null, <2 x float> zeroinitializer, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  ret void
}

declare { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32)

!air.kernel = !{!0}
!0 = !{ptr @kernel, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"src"}
!4 = !{i32 1, !"air.texture", !"air.location_index", i32 1, i32 1, !"air.write", !"air.arg_type_name", !"texture2d<float, write>", !"air.arg_name", !"out"}
"#;
        let plan = infer_plan(ll);
        assert_eq!(plan.output.kind, "texture");
        assert_eq!(plan.output.format, "R32Float");
        let metal = metal_row_for_compare(&golden, plan, Some("Kernel"));

        let ordinary = compare_candidate_to_metal(
            &candidate,
            &metal,
            &sha256_hex(&candidate),
            metal.output_sha256.as_deref().unwrap(),
            DataFormat::R32Float,
            None,
        );
        assert_eq!(ordinary.0, "failure");

        let (status, observed, tolerance) = compare_candidate_to_metal(
            &candidate,
            &metal,
            &sha256_hex(&candidate),
            metal.output_sha256.as_deref().unwrap(),
            DataFormat::R32Float,
            Some(ll),
        );
        assert_eq!(status, "tolerance");
        assert_eq!(observed.and_then(|m| m.max_abs), Some(0.0015));
        assert_eq!(tolerance.and_then(|t| t.max_abs), Some(0.001_953_125));
    }

    #[test]
    fn compare_sampled_f32_half_imageblock_storage_texture_uses_half_step_tolerance() {
        let golden = 0.0f32.to_le_bytes();
        let candidate = 0.0029f32.to_le_bytes();
        let ll = r#"
define void @kernel(ptr addrspace(1) %src, ptr addrspace(1) %out, <2 x i16> %tid) {
  %s = tail call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) %src, ptr addrspace(2) null, <2 x float> zeroinitializer, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %v = extractvalue { <4 x float>, i8 } %s, 0
  %h = tail call fast <4 x half> @air.convert.f.v4f16.f.v4f32(<4 x float> %v)
  %slot = tail call ptr addrspace(4) @air.imageblock_data(<2 x i16> %tid, i32 0, i16 0)
  store <4 x half> %h, ptr addrspace(4) %slot
  tail call void @air.write_imageblock_slice_to_texture_2d.i16.v4f16(ptr addrspace(1) %out, ptr addrspace(4) %slot, i1 false, <2 x i16> zeroinitializer, <2 x i16> undef, <2 x i16> %tid, i16 0, i1 false, i32 2)
  ret void
}

declare { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32)
declare <4 x half> @air.convert.f.v4f16.f.v4f32(<4 x float>)
declare ptr addrspace(4) @air.imageblock_data(<2 x i16>, i32, i16)
declare void @air.write_imageblock_slice_to_texture_2d.i16.v4f16(ptr addrspace(1), ptr addrspace(4), i1, <2 x i16>, <2 x i16>, <2 x i16>, i16, i1, i32)

!air.kernel = !{!0}
!0 = !{ptr @kernel, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"src"}
!4 = !{i32 1, !"air.texture", !"air.location_index", i32 1, i32 1, !"air.write", !"air.arg_type_name", !"texture2d<float, write>", !"air.arg_name", !"out"}
!5 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"ushort2", !"air.arg_name", !"tid"}
"#;
        let metal = metal_row_for_compare(&golden, infer_plan(ll), Some("Kernel"));

        let (status, observed, tolerance) = compare_candidate_to_metal(
            &candidate,
            &metal,
            &sha256_hex(&candidate),
            metal.output_sha256.as_deref().unwrap(),
            DataFormat::R32Float,
            Some(ll),
        );

        assert_eq!(status, "tolerance");
        assert_eq!(observed.and_then(|m| m.max_abs), Some(0.0029));
        assert_eq!(tolerance.and_then(|t| t.max_abs), Some(0.003_906_25));
    }

    #[test]
    fn compare_sampled_f32_half_coordinate_texture_uses_half_step_tolerance() {
        let golden = 0.0f32.to_le_bytes();
        let candidate = 0.0038f32.to_le_bytes();
        let ll = r#"
define void @kernel(ptr addrspace(1) %warp, ptr addrspace(1) %src, ptr addrspace(1) %out, <2 x i32> %gid) {
  %r = tail call { <4 x half>, i8 } @air.read_texture_2d.v4f16(ptr addrspace(1) %warp, ptr addrspace(2) null, <2 x i32> %gid, <2 x i32> zeroinitializer, i32 0, i32 1)
  %rv = extractvalue { <4 x half>, i8 } %r, 0
  %xy_h = shufflevector <4 x half> %rv, <4 x half> poison, <2 x i32> <i32 0, i32 1>
  %xy = tail call fast <2 x float> @air.convert.f.v2f32.f.v2f16(<2 x half> %xy_h)
  %s = tail call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) %src, ptr addrspace(2) null, <2 x float> %xy, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %v = extractvalue { <4 x float>, i8 } %s, 0
  tail call void @air.write_texture_2d.v4f32(ptr addrspace(1) %out, <2 x i32> %gid, <4 x float> %v, i32 0, i32 2)
  ret void
}

declare { <4 x half>, i8 } @air.read_texture_2d.v4f16(ptr addrspace(1), ptr addrspace(2), <2 x i32>, <2 x i32>, i32, i32)
declare <2 x float> @air.convert.f.v2f32.f.v2f16(<2 x half>)
declare { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32)
declare void @air.write_texture_2d.v4f32(ptr addrspace(1), <2 x i32>, <4 x float>, i32, i32)

!air.kernel = !{!0}
!0 = !{ptr @kernel, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5, !6}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.read", !"air.arg_type_name", !"texture2d<half, read>", !"air.arg_name", !"warp"}
!4 = !{i32 1, !"air.texture", !"air.location_index", i32 3, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"src"}
!5 = !{i32 2, !"air.texture", !"air.location_index", i32 4, i32 1, !"air.write", !"air.arg_type_name", !"texture2d<float, write>", !"air.arg_name", !"out"}
!6 = !{i32 3, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint2", !"air.arg_name", !"gid"}
"#;
        let metal = metal_row_for_compare(&golden, infer_plan(ll), Some("Kernel"));

        let (status, observed, tolerance) = compare_candidate_to_metal(
            &candidate,
            &metal,
            &sha256_hex(&candidate),
            metal.output_sha256.as_deref().unwrap(),
            DataFormat::R32Float,
            Some(ll),
        );

        assert_eq!(status, "tolerance");
        assert_eq!(observed.and_then(|m| m.max_abs), Some(0.0038));
        assert_eq!(tolerance.and_then(|t| t.max_abs), Some(0.003_906_25));
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
    fn compare_sampled_fast_half_palette_allows_dot_mix_amplified_drift() {
        let golden = 0x3bb0u16.to_le_bytes(); // 0.9609375
        let candidate = 0x3bdeu16.to_le_bytes(); // 0.9833984375
        let mut plan = infer_plan("");
        plan.output.kind = "render_target".into();
        plan.output.format = "Rgba16Float".into();
        let metal = metal_row_for_compare(&golden, plan, Some("Fragment"));
        let ll = r#"
define <4 x half> @frag(ptr addrspace(1) %tex) #0 {
  %s = tail call { <4 x half>, i8 } @air.sample_texture_2d.v4f16(ptr addrspace(1) %tex, ptr addrspace(2) null, <2 x float> zeroinitializer, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %v = extractvalue { <4 x half>, i8 } %s, 0
  %rgb = shufflevector <4 x half> %v, <4 x half> poison, <3 x i32> <i32 0, i32 1, i32 2>
  %l = tail call fast half @air.dot.v3f16(<3 x half> %rgb, <3 x half> splat (half 0xH3C00))
  %t = insertelement <3 x half> poison, half %l, i64 0
  %tt = shufflevector <3 x half> %t, <3 x half> poison, <3 x i32> zeroinitializer
  %m = tail call fast <3 x half> @air.mix.v3f16(<3 x half> zeroinitializer, <3 x half> splat (half 0xH3C00), <3 x half> %tt)
  %out = shufflevector <3 x half> %m, <3 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 poison>
  ret <4 x half> %out
}
declare { <4 x half>, i8 } @air.sample_texture_2d.v4f16(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32)
declare half @air.dot.v3f16(<3 x half>, <3 x half>)
declare <3 x half> @air.mix.v3f16(<3 x half>, <3 x half>, <3 x half>)
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
        assert_eq!(observed.and_then(|m| m.max_abs), Some(0.022_460_938));
        assert_eq!(tolerance.and_then(|t| t.max_abs), Some(0.023_437_5));
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
            input_specialization: None,
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
            input_specialization: None,
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
            input_specialization: None,
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
            input_specialization: None,
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
            input_specialization: None,
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
            input_specialization: None,
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
            input_specialization: None,
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
            input_specialization: None,
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
            input_specialization: None,
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
            input_specialization: None,
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
    fn function_constant_texture_array_ref_view_family_requires_rebank() {
        let ll = r#"
@_ZL33_tex_loc = internal addrspace(2) global i32 1, align 4
@_ZL33_arr_loc = internal addrspace(2) global i32 33, align 4
@fc0 = internal addrspace(2) global i8 1, align 1
@fc1 = internal addrspace(2) global i8 1, align 1

define void @kernel(ptr addrspace(1) %a, ptr addrspace(1) %b) {
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @kernel, !1, !2}
!1 = !{}
!2 = !{!3, !5}
!3 = !{i32 0, !"air.function_constant", !4, !"air.texture", !"air.location_index", ptr addrspace(2) @_ZL33_tex_loc, i32 1, !"air.sample", !"air.arg_type_name", !"array_ref<texture2d<float, sample>>", !"air.arg_name", !"plain"}
!4 = !{ptr addrspace(2) @fc0, !"bool", !"usePlain", i32 1, i1 true}
!5 = !{i32 1, !"air.function_constant", !6, !"air.texture", !"air.location_index", ptr addrspace(2) @_ZL33_arr_loc, i32 1, !"air.sample", !"air.arg_type_name", !"array_ref<texture2d_array<float, sample>>", !"air.arg_name", !"arrayed"}
!6 = !{ptr addrspace(2) @fc1, !"bool", !"useArray", i32 2, i1 true}
"#;
        let mut metal = metal_row_for_compare(&[], infer_plan(ll), Some("Kernel"));
        metal.fc_specialization = Some(FC_SPECIALIZATION_VALUES.into());
        metal.fc_values = Some(vec![
            FunctionConstantValueJson { index: 1, value: 1 },
            FunctionConstantValueJson { index: 2, value: 1 },
        ]);

        let reason = incompatible_function_constant_texture_array_ref_golden(ll, &metal)
            .expect("mixed FC texture array_ref view family");
        assert!(reason.contains("texture metadata"), "{reason}");
        assert!(reason.contains("image-view"), "{reason}");

        metal.compare = "none".into();
        let reason = incompatible_function_constant_texture_array_ref_golden(ll, &metal)
            .expect("mixed FC texture array_ref smoke view family");
        assert!(reason.contains("image-view"), "{reason}");

        let read_ll = r#"
@fc0 = internal addrspace(2) global i8 1, align 1
@fc1 = internal addrspace(2) global i8 1, align 1

define void @kernel(ptr addrspace(1) %arrayed, ptr addrspace(1) %plain, ptr addrspace(1) %out) {
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @kernel, !1, !2}
!1 = !{}
!2 = !{!3, !5, !7}
!3 = !{i32 0, !"air.function_constant", !4, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.read", !"air.arg_type_name", !"texture2d_array<half, read>", !"air.arg_name", !"arrayed"}
!4 = !{ptr addrspace(2) @fc0, !"bool", !"useArrayed", i32 1, i1 true}
!5 = !{i32 1, !"air.function_constant", !6, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.read", !"air.arg_type_name", !"texture2d<half, read>", !"air.arg_name", !"plain"}
!6 = !{ptr addrspace(2) @fc1, !"bool", !"usePlain", i32 2, i1 true}
!7 = !{i32 2, !"air.texture", !"air.location_index", i32 1, i32 1, !"air.write", !"air.arg_type_name", !"texture2d<half, write>", !"air.arg_name", !"out"}
"#;
        let reason = incompatible_function_constant_texture_array_ref_golden(read_ll, &metal)
            .expect("mixed FC read texture view family");
        assert!(reason.contains("texture2d_array"), "{reason}");
    }

    #[test]
    fn function_constant_private_pointer_table_golden_is_missing() {
        let ll = r#"
@_Z7quality.MTL_FC_INIT_0_i = internal addrspace(2) externally_initialized constant i32 undef, section "air.fc_initializer", align 4
@small = internal unnamed_addr addrspace(2) constant [4 x <3 x half>] zeroinitializer, align 8
@large = internal unnamed_addr addrspace(2) constant [8 x <3 x half>] zeroinitializer, align 8

define internal void @init() section "air.static_init" {
  %q = load i32, ptr addrspace(2) @_Z7quality.MTL_FC_INIT_0_i, align 4
  %defined = tail call i1 @air.is_function_constant_defined(ptr addrspace(2) @_Z7quality.MTL_FC_INIT_0_i)
  %v = select i1 %defined, i32 %q, i32 0
  ret void
}

define void @frag(i32 %quality) {
  switch i32 %quality, label %small [
    i32 1, label %large
  ]
small:
  br label %merge
large:
  br label %merge
merge:
  %table = phi ptr addrspace(2) [ @small, %small ], [ @large, %large ]
  ret void
}

declare i1 @air.is_function_constant_defined(ptr addrspace(2))
!air.function_constants = !{!0}
!0 = !{ptr addrspace(2) @_Z7quality.MTL_FC_INIT_0_i, !"int", !"quality", i32 0, i1 false}
"#;
        let mut metal = metal_row_for_compare(&[], infer_plan(ll), Some("Fragment"));
        metal.fc_specialization = Some(FC_SPECIALIZATION_ZERO.into());

        let reason = incompatible_function_constant_private_pointer_table_golden(ll, &metal)
            .expect("FC private pointer table golden");
        assert!(
            reason.contains("private constant-table pointers"),
            "{reason}"
        );

        metal.fc_specialization = None;
        assert!(incompatible_function_constant_private_pointer_table_golden(ll, &metal).is_none());
    }

    #[test]
    fn multisample_texture_golden_requires_rebank() {
        let ll = r#"
define <{ <4 x i32> }> @frag(ptr addrspace(1) %tex, i32 %sampleId) {
entry:
  %coord = insertelement <2 x i32> undef, i32 0, i64 0
  %read = call { <4 x i32>, i8 } @air.read_texture_2d_ms.s.v4i32(ptr addrspace(1) %tex, i32 0, <2 x i32> %coord, i32 %sampleId, i32 1)
  %v = extractvalue { <4 x i32>, i8 } %read, 0
  %out = insertvalue <{ <4 x i32> }> undef, <4 x i32> %v, 0
  ret <{ <4 x i32> }> %out
}

declare { <4 x i32>, i8 } @air.read_texture_2d_ms.s.v4i32(ptr addrspace(1), i32, <2 x i32>, i32, i32)

!air.fragment = !{!0}
!0 = !{ptr @frag, !1, !3}
!1 = !{!2}
!2 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"int4"}
!3 = !{!4, !5}
!4 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.read", !"air.arg_type_name", !"texture2d_ms<int, read>", !"air.arg_name", !"tex"}
!5 = !{i32 1, !"air.sample_id", !"air.arg_type_name", !"uint", !"air.arg_name", !"sampleId"}
"#;
        let metal = metal_row_for_compare(&[], infer_plan(ll), Some("Fragment"));
        let reason = incompatible_multisample_texture_golden(ll, &metal)
            .expect("multisample texture golden should require rebank");
        assert!(reason.contains("single-sample"), "{reason}");
        assert!(reason.contains("rebank"), "{reason}");
        assert!(
            incompatible_multisample_texture_golden("define void @f() { ret void }", &metal)
                .is_none()
        );
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
            input_specialization: None,
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
    fn integer_divisor_function_constants_request_nonzero_values() {
        let ll = r#"
@_ZL2fc = internal unnamed_addr addrspace(2) global i16 zeroinitializer, align 2
@_Z2fc.MTL_FC_INIT_4_t = internal unnamed_addr addrspace(2) externally_initialized constant i16 undef, section "air.fc_initializer", align 2
@_Z2flt.MTL_FC_INIT_5_f = internal unnamed_addr addrspace(2) externally_initialized constant float undef, section "air.fc_initializer", align 4

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

!air.function_constants = !{!0, !1}
!0 = !{ptr addrspace(2) @_Z2fc.MTL_FC_INIT_4_t, !"ushort", !"fc", i32 4, i1 true}
!1 = !{ptr addrspace(2) @_Z2flt.MTL_FC_INIT_5_f, !"float", !"flt", i32 5, i1 true}
"#;
        assert_eq!(
            function_constant_values_for_integer_divisors(ll),
            vec![(4, 1)]
        );
    }

    #[test]
    fn barrier_loop_function_constants_request_nonzero_values() {
        let ll = r#"
define void @k() {
  br label %loop
loop:
  tail call void @air.wg.barrier(i32 2, i32 1)
  br label %loop
}

declare void @air.wg.barrier(i32, i32)
!air.function_constants = !{!0, !1, !2, !3}
!0 = !{ptr addrspace(2) @_Z2fc.MTL_FC_INIT_4_t, !"ushort", !"fc", i32 4, i1 true}
!1 = !{ptr addrspace(2) @_Z2flt.MTL_FC_INIT_5_f, !"float", !"flt", i32 5, i1 true}
!2 = !{ptr addrspace(2) @_Z2wide.MTL_FC_INIT_6_j, !"uint", !"wide", i32 6, i1 true}
!3 = !{ptr addrspace(2) @_Z2flag.MTL_FC_INIT_7_b, !"bool", !"flag", i32 7, i1 true}
"#;
        assert_eq!(
            function_constant_values_for_barrier_loop_progress(ll),
            vec![(4, 2), (6, 2)]
        );
    }

    #[test]
    fn barrier_loop_function_constants_include_branch_control_bools() {
        let ll = r#"
@_Z4flag.MTL_FC_INIT_7_b = internal unnamed_addr addrspace(2) externally_initialized constant i8 undef, section "air.fc_initializer", align 1
@_ZL4flag = internal unnamed_addr addrspace(2) global i8 undef, align 1

define internal void @_GLOBAL__sub_I_k() section "air.static_init" {
  %flag = load i8, ptr addrspace(2) @_Z4flag.MTL_FC_INIT_7_b, align 1
  store i8 %flag, ptr addrspace(2) @_ZL4flag, align 1
  ret void
}

define void @k() {
  %flag = load i8, ptr addrspace(2) @_ZL4flag, align 1
  %enabled = icmp ne i8 %flag, 0
  br i1 %enabled, label %loop, label %exit
loop:
  tail call void @air.wg.barrier(i32 2, i32 1)
  br label %exit
exit:
  ret void
}

declare void @air.wg.barrier(i32, i32)
!air.function_constants = !{!0}
!0 = !{ptr addrspace(2) @_Z4flag.MTL_FC_INIT_7_b, !"bool", !"flag", i32 7, i1 true}
"#;
        assert_eq!(
            function_constant_values_for_barrier_loop_progress(ll),
            vec![(7, 1)]
        );
    }

    #[test]
    fn barrier_loop_function_constants_do_not_activate_colliding_conditional_buffers() {
        let ll = r#"
@_Z4flag.MTL_FC_INIT_7_b = internal unnamed_addr addrspace(2) externally_initialized constant i8 undef, section "air.fc_initializer", align 1
@_Z4wide.MTL_FC_INIT_8_j = internal unnamed_addr addrspace(2) externally_initialized constant i32 undef, section "air.fc_initializer", align 4
@pred = internal addrspace(2) global i8 0, align 1

define internal void @_GLOBAL__sub_I_k() section "air.static_init" {
  %flag = load i8, ptr addrspace(2) @_Z4flag.MTL_FC_INIT_7_b, align 1
  %pred = tail call i8 @air.normalize_function_constant_predicate.i8(i8 %flag)
  store i8 %pred, ptr addrspace(2) @pred, align 1
  ret void
}

define void @k() {
  tail call void @air.wg.barrier(i32 2, i32 1)
  ret void
}

declare i8 @air.normalize_function_constant_predicate.i8(i8)
declare void @air.wg.barrier(i32, i32)
!air.kernel = !{!0}
!air.function_constants = !{!5, !6}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_name", !"uint", !"air.arg_name", !"base"}
!4 = !{i32 1, !"air.function_constant", !7, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_name", !"uint", !"air.arg_name", !"conditional"}
!5 = !{ptr addrspace(2) @_Z4flag.MTL_FC_INIT_7_b, !"bool", !"flag", i32 7, i1 true}
!6 = !{ptr addrspace(2) @_Z4wide.MTL_FC_INIT_8_j, !"uint", !"wide", i32 8, i1 true}
!7 = !{ptr addrspace(2) @pred, !"bool", !"flag"}
"#;
        assert_eq!(
            function_constant_values_for_barrier_loop_progress(ll),
            vec![(8, 2)]
        );
    }

    #[test]
    fn metal_oracle_inputs_only_bank_explicit_input_plans() {
        let mut plan = infer_plan("");
        plan.dispatch_grid = [1, 1, 1];
        plan.dispatch_tg = [1, 1, 1];
        let mut metal = metal_row_for_compare(&[], plan.clone(), Some("Kernel"));
        assert!(banked_explicit_input_plan(&metal).is_none());
        assert!(banked_function_constant_values(&metal).is_none());

        metal.input_specialization = Some(INPUT_SPECIALIZATION_EXPLICIT.into());
        let banked_plan =
            banked_explicit_input_plan(&metal).expect("explicit input row should bank inputs");
        assert_eq!(banked_plan.dispatch_grid, [1, 1, 1]);
        assert_eq!(banked_plan.dispatch_tg, [1, 1, 1]);

        metal.input_specialization = None;
        metal.fc_values = Some(vec![FunctionConstantValueJson {
            index: 18,
            value: 1,
        }]);
        assert!(banked_explicit_input_plan(&metal).is_none());
        let fc_values =
            banked_function_constant_values(&metal).expect("FC row should bank FC values");
        assert_eq!(fc_values, vec![(18, 1)]);

        let (current_plan, fc_values) = metal_oracle_inputs("", Some(&metal));
        assert_eq!(
            current_plan.dispatch_grid,
            [DEFAULT_DISPATCH_GRID_X as u32, 1, 1]
        );
        assert_eq!(fc_values, vec![(18, 1)]);
    }

    #[test]
    fn barrier_loop_function_constants_override_divisor_defaults() {
        let ll = r#"
@_ZL2fc = internal unnamed_addr addrspace(2) global i32 zeroinitializer, align 4
@_Z2fc.MTL_FC_INIT_4_j = internal unnamed_addr addrspace(2) externally_initialized constant i32 undef, section "air.fc_initializer", align 4

define internal void @_GLOBAL__sub_I_k() section "air.static_init" {
  %1 = load i32, ptr addrspace(2) @_Z2fc.MTL_FC_INIT_4_j, align 4
  store i32 %1, ptr addrspace(2) @_ZL2fc, align 4
  ret void
}

define void @k(i32 %gid) {
  %step = load i32, ptr addrspace(2) @_ZL2fc, align 4
  %q = sdiv i32 %gid, %step
  br label %loop
loop:
  %i = phi i32 [ 0, %0 ], [ %next, %loop ]
  tail call void @air.wg.barrier(i32 2, i32 1)
  %next = add i32 %i, %step
  %more = icmp ult i32 %next, 8
  br i1 %more, label %loop, label %exit
exit:
  ret void
}

declare void @air.wg.barrier(i32, i32)
!air.function_constants = !{!0}
!0 = !{ptr addrspace(2) @_Z2fc.MTL_FC_INIT_4_j, !"uint", !"fc", i32 4, i1 true}
"#;
        assert_eq!(function_constant_values_for_oracle_inputs(ll), vec![(4, 2)]);
    }

    #[test]
    fn definedness_function_constants_do_not_request_nonzero_values() {
        let ll = r#"
@_ZL2fc = internal unnamed_addr addrspace(2) global i16 zeroinitializer, align 2
@_Z2fc.MTL_FC_INIT_4_t = internal unnamed_addr addrspace(2) externally_initialized constant i16 undef, section "air.fc_initializer", align 2

define internal void @_GLOBAL__sub_I_k() section "air.static_init" {
  %1 = load i16, ptr addrspace(2) @_Z2fc.MTL_FC_INIT_4_t, align 2
  store i16 %1, ptr addrspace(2) @_ZL2fc, align 2
  ret void
}

define void @k(i32 %gid) {
  %defined = tail call i1 @air.is_function_constant_defined(ptr addrspace(2) @_Z2fc.MTL_FC_INIT_4_t)
  %2 = load i16, ptr addrspace(2) @_ZL2fc, align 2
  %3 = zext i16 %2 to i32
  %4 = sdiv i32 %gid, %3
  ret void
}

declare i1 @air.is_function_constant_defined(ptr addrspace(2))
!air.function_constants = !{!0}
!0 = !{ptr addrspace(2) @_Z2fc.MTL_FC_INIT_4_t, !"ushort", !"fc", i32 4, i1 true}
"#;
        assert!(function_constant_values_for_integer_divisors(ll).is_empty());
        assert!(function_constant_values_for_barrier_loop_progress(ll).is_empty());
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
            input_specialization: None,
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
            input_specialization: None,
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
            input_specialization: None,
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
            input_specialization: None,
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
            input_specialization: None,
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
    fn finite_float_seed_checker_uses_full_inferred_plan() {
        let ll = r#"
define void @kernel(ptr addrspace(1) %src, ptr addrspace(1) %dst, i32 %tid) #0 {
  %idx = zext i32 %tid to i64
  %p = getelementptr inbounds <3 x float>, ptr addrspace(1) %src, i64 %idx
  %v = load <3 x float>, ptr addrspace(1) %p, align 16
  %x = extractelement <3 x float> %v, i64 0
  %scaled = fmul fast float %x, 2.000000e+00
  %q = getelementptr inbounds float, ptr addrspace(1) %dst, i64 %idx
  store float %scaled, ptr addrspace(1) %q, align 4
  ret void
}

attributes #0 = { "no-nans-fp-math"="true" }

!air.kernel = !{!0}
!0 = !{ptr @kernel, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_name", !"float3", !"air.arg_name", !"src"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"dst"}
!5 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid"}
"#;
        let current_plan = infer_plan(ll);
        let src = current_plan
            .buffers
            .iter()
            .find(|b| b.index == 0)
            .expect("source buffer");
        assert_eq!(src.seed_mode, SEED_MODE_FINITE_FLOAT32);
        assert_eq!(src.len, 12 * DEFAULT_DISPATCH_GRID_X);

        let metal = MetalRow {
            air_sha256: "x".into(),
            shard: None,
            label: String::new(),
            status: "ok".into(),
            backend: "metal".into(),
            seed_profile: SEED_PROFILE.into(),
            plan_version: PLAN_VERSION,
            plan: current_plan,
            input_sha256: None,
            output_sha256: Some(sha256_hex(&0.0f32.to_le_bytes())),
            output_b64: Some(encode_output_b64(&0.0f32.to_le_bytes())),
            spv_sha256: None,
            compare: "full".into(),
            fc_specialization: None,
            fc_values: None,
            input_specialization: None,
            stage: None,
            entry: None,
            error: None,
        };

        assert!(incompatible_float_seed_golden(ll, &metal).is_none());
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
            input_specialization: None,
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
            input_specialization: None,
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
            input_specialization: None,
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
            input_specialization: None,
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
            input_specialization: None,
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
    fn sampled_f32_domain_math_texture_golden_is_missing() {
        let ll = r#"
define float @frag(ptr addrspace(1) %tex, <2 x float> %uv) #0 {
  %s = tail call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) %tex, ptr addrspace(2) null, <2 x float> %uv, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %v = extractvalue { <4 x float>, i8 } %s, 0
  %x = extractelement <4 x float> %v, i64 0
  %biased = fadd fast float %x, 1.000000e+00
  %out = tail call fast float @air.log.f32(float %biased)
  ret float %out
}

declare { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32)
declare float @air.log.f32(float)
attributes #0 = { "no-nans-fp-math"="true" "unsafe-fp-math"="true" }

!air.fragment = !{!0}
!air.compile_options = !{!5}
!0 = !{ptr @frag, !1, !2}
!1 = !{!3}
!2 = !{!4}
!3 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float"}
!4 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"tex"}
!5 = !{!"air.compile.fast_math_enable"}
"#;

        let metal = metal_row_for_compare(&[], infer_plan(ll), Some("Fragment"));
        let reason = incompatible_sampled_f32_domain_math_texture_golden(ll, &metal)
            .expect("sampled f32 domain math texture golden");
        assert!(reason.contains("finite f32 texture"), "{reason}");
    }

    #[test]
    fn sampled_uint_float_render_target_golden_is_missing() {
        let ll = r#"
define <4 x float> @frag(ptr addrspace(1) %tex, <2 x float> %uv) {
  %s = tail call { <4 x i32>, i8 } @air.sample_texture_2d.u.v4i32(ptr addrspace(1) %tex, ptr addrspace(2) null, <2 x float> %uv, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %v = extractvalue { <4 x i32>, i8 } %s, 0
  %x = extractelement <4 x i32> %v, i64 0
  %f = tail call float @air.convert.f.f32.u.i32(i32 %x)
  %scaled = fdiv float %f, 2.550000e+02
  %rgba = insertelement <4 x float> zeroinitializer, float %scaled, i64 0
  ret <4 x float> %rgba
}

declare { <4 x i32>, i8 } @air.sample_texture_2d.u.v4i32(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32)
declare float @air.convert.f.f32.u.i32(i32)

!air.fragment = !{!0}
!0 = !{ptr @frag, !1, !2}
!1 = !{!3}
!2 = !{!4}
!3 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!4 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<uint, sample>", !"air.arg_name", !"tex"}
"#;

        let metal = metal_row_for_compare(&[], infer_plan(ll), Some("Fragment"));
        let reason = incompatible_sampled_uint_float_render_target_golden(ll, &metal)
            .expect("sampled uint float render-target golden");
        assert!(reason.contains("synthetic uint texture"), "{reason}");
    }

    #[test]
    fn sampled_half_dot_render_target_golden_is_missing() {
        let ll = r#"
define <4 x float> @frag(ptr addrspace(1) %tex, <2 x float> %uv) {
  %s = tail call { <4 x half>, i8 } @air.sample_texture_2d.v4f16(ptr addrspace(1) %tex, ptr addrspace(2) null, <2 x float> %uv, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %v = extractvalue { <4 x half>, i8 } %s, 0
  %d = tail call half @air.dot.v4f16(<4 x half> %v, <4 x half> splat (half 0xH3C00))
  %f = tail call float @air.convert.f.f32.f.f16(half %d)
  %rgba = insertelement <4 x float> zeroinitializer, float %f, i64 0
  ret <4 x float> %rgba
}

declare { <4 x half>, i8 } @air.sample_texture_2d.v4f16(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32)
declare half @air.dot.v4f16(<4 x half>, <4 x half>)
declare float @air.convert.f.f32.f.f16(half)

!air.fragment = !{!0}
!0 = !{ptr @frag, !1, !2}
!1 = !{!3}
!2 = !{!4}
!3 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!4 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<half, sample>", !"air.arg_name", !"tex"}
"#;

        let metal = metal_row_for_compare(&[], infer_plan(ll), Some("Fragment"));
        let reason = incompatible_sampled_half_dot_render_target_golden(ll, &metal)
            .expect("sampled half dot render-target golden");
        assert!(reason.contains("half dot"), "{reason}");
    }

    #[test]
    fn fast_procedural_f32_texture_output_golden_is_missing() {
        let ll = r#"
define void @k(<2 x i16> %gid, ptr addrspace(1) %tex) #0 {
  %coord = tail call fast <2 x float> @air.convert.f.v2f32.u.v2i16(<2 x i16> %gid)
  %x = extractelement <2 x float> %coord, i64 0
  %m = tail call fast float @air.fast_fmod.f32(float %x, float 1.000000e+01)
  %r = tail call fast float @air.fast_rsqrt.f32(float %m)
  %v = insertelement <4 x float> zeroinitializer, float %r, i64 0
  tail call void @air.write_texture_2d.i16.v4f32(ptr addrspace(1) %tex, <2 x i16> %gid, <4 x float> %v, i16 0, i32 2)
  ret void
}

declare <2 x float> @air.convert.f.v2f32.u.v2i16(<2 x i16>)
declare float @air.fast_fmod.f32(float, float)
declare float @air.fast_rsqrt.f32(float)
declare void @air.write_texture_2d.i16.v4f32(ptr addrspace(1), <2 x i16>, <4 x float>, i16, i32)
attributes #0 = { "no-nans-fp-math"="true" "unsafe-fp-math"="true" }

!air.kernel = !{!0}
!air.compile_options = !{!4}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.write", !"air.arg_type_name", !"texture2d<float, write>", !"air.arg_name", !"tex"}
!4 = !{!"air.compile.fast_math_enable"}
"#;

        let metal = metal_row_for_compare(&[], infer_plan(ll), Some("Kernel"));
        let reason = incompatible_fast_procedural_f32_texture_output_golden(ll, &metal)
            .expect("fast procedural f32 texture output golden");
        assert!(reason.contains("procedural f32 texture output"), "{reason}");
    }

    #[test]
    fn fast_procedural_f32_texture3d_output_golden_is_missing() {
        let ll = r#"
define void @k(<3 x i16> %gid, ptr addrspace(1) %tex) #0 {
  %coord = tail call fast <3 x float> @air.convert.f.v3f32.u.v3i16(<3 x i16> %gid)
  %x = extractelement <3 x float> %coord, i64 0
  %m = tail call fast float @air.fast_fmod.f32(float %x, float 1.000000e+01)
  %r = tail call fast float @air.fast_rsqrt.f32(float %m)
  %v = insertelement <4 x float> zeroinitializer, float %r, i64 0
  tail call void @air.write_texture_3d.i16.v4f32(ptr addrspace(1) %tex, <3 x i16> %gid, <4 x float> %v, i16 0, i32 2)
  ret void
}

declare <3 x float> @air.convert.f.v3f32.u.v3i16(<3 x i16>)
declare float @air.fast_fmod.f32(float, float)
declare float @air.fast_rsqrt.f32(float)
declare void @air.write_texture_3d.i16.v4f32(ptr addrspace(1), <3 x i16>, <4 x float>, i16, i32)
attributes #0 = { "no-nans-fp-math"="true" "unsafe-fp-math"="true" }

!air.kernel = !{!0}
!air.compile_options = !{!4}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.write", !"air.arg_type_name", !"texture3d<float, write>", !"air.arg_name", !"tex"}
!4 = !{!"air.compile.fast_math_enable"}
"#;

        let metal = metal_row_for_compare(&[], infer_plan(ll), Some("Kernel"));
        let reason = incompatible_fast_procedural_f32_texture_output_golden(ll, &metal)
            .expect("fast procedural f32 texture3d output golden");
        assert!(reason.contains("procedural f32 texture output"), "{reason}");
    }

    #[test]
    fn fast_f32_buffer_output_golden_is_missing() {
        let ll = r#"
define void @k(ptr addrspace(1) %a, ptr addrspace(1) %b, ptr addrspace(1) %out) #0 {
  %x = load float, ptr addrspace(1) %a, align 4
  %y = load float, ptr addrspace(1) %b, align 4
  %sum = fadd fast float %x, %y
  %den = tail call fast float @air.fast_rsqrt.f32(float %sum)
  %q = fdiv fast float %x, %den
  %m = tail call fast float @llvm.minnum.f32(float %q, float %y)
  store float %m, ptr addrspace(1) %out, align 4
  ret void
}

declare float @air.fast_rsqrt.f32(float)
declare float @llvm.minnum.f32(float, float)
attributes #0 = { "no-nans-fp-math"="true" "unsafe-fp-math"="true" }

!air.kernel = !{!0}
!air.compile_options = !{!6}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"a"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"b"}
!5 = !{i32 2, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"out"}
!6 = !{!"air.compile.fast_math_enable"}
"#;

        let metal = metal_row_for_compare(&[], infer_plan(ll), Some("Kernel"));
        let reason = incompatible_fast_f32_buffer_output_golden(ll, &metal)
            .expect("fast f32 buffer output golden");
        assert!(reason.contains("finite f32 buffer data"), "{reason}");
    }

    #[test]
    fn sampled_f32_texture_output_golden_is_missing() {
        let ll = r#"
define void @k(<2 x i16> %gid, ptr addrspace(1) %src, ptr addrspace(1) %dst, <2 x float> %uv) {
  %s = tail call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) %src, ptr addrspace(2) null, <2 x float> %uv, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %v = extractvalue { <4 x float>, i8 } %s, 0
  tail call void @air.write_texture_2d.v4f32(ptr addrspace(1) %dst, <2 x i32> zeroinitializer, <4 x float> %v, i32 2)
  ret void
}

declare { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32)
declare void @air.write_texture_2d.v4f32(ptr addrspace(1), <2 x i32>, <4 x float>, i32)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"src"}
!4 = !{i32 1, !"air.texture", !"air.location_index", i32 1, i32 1, !"air.write", !"air.arg_type_name", !"texture2d<float, write>", !"air.arg_name", !"dst"}
"#;

        let metal = metal_row_for_compare(&[], infer_plan(ll), Some("Kernel"));
        let reason = incompatible_sampled_f32_texture_output_golden(ll, &metal)
            .expect("sampled f32 texture output golden");
        assert!(reason.contains("f32 texture output"), "{reason}");
    }

    #[test]
    fn storage_half_exp_texture_output_golden_is_missing() {
        let ll = r#"
define void @k(<2 x i16> %gid, ptr addrspace(1) %src, ptr addrspace(1) %dst) {
  %v = tail call <4 x half> @air.read_texture_2d.i16.v4f16(ptr addrspace(1) %src, <2 x i16> %gid, i32 0)
  %e = tail call half @air.exp.f16(half 0xH3C00)
  %out = insertelement <4 x half> %v, half %e, i64 0
  tail call void @air.write_texture_2d.i16.v4f16(ptr addrspace(1) %dst, <2 x i16> %gid, <4 x half> %out, i16 0, i32 2)
  ret void
}

declare <4 x half> @air.read_texture_2d.i16.v4f16(ptr addrspace(1), <2 x i16>, i32)
declare half @air.exp.f16(half)
declare void @air.write_texture_2d.i16.v4f16(ptr addrspace(1), <2 x i16>, <4 x half>, i16, i32)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.read", !"air.arg_type_name", !"texture2d<half, read>", !"air.arg_name", !"src"}
!4 = !{i32 1, !"air.texture", !"air.location_index", i32 1, i32 1, !"air.write", !"air.arg_type_name", !"texture2d<half, write>", !"air.arg_name", !"dst"}
"#;

        let metal = metal_row_for_compare(&[], infer_plan(ll), Some("Kernel"));
        let reason = incompatible_storage_half_exp_texture_output_golden(ll, &metal)
            .expect("storage half exp texture output golden");
        assert!(reason.contains("AIR exp"), "{reason}");
    }

    #[test]
    fn function_constant_simdgroup_threadgroup_golden_is_missing() {
        let ll = r#"
@_Z7enable0.MTL_FC_INIT_0_b = internal unnamed_addr addrspace(2) externally_initialized constant i8 undef, section "air.fc_initializer", align 1

define void @k(ptr addrspace(3) %scratch, i16 %lane) {
  %flag = load i8, ptr addrspace(2) @_Z7enable0.MTL_FC_INIT_0_b, align 1
  %enabled = icmp ne i8 %flag, 0
  br i1 %enabled, label %wide, label %done
wide:
  %v = tail call i16 @air.simd_shuffle_up.s.i16(i16 %lane, i16 1)
  %slot = getelementptr inbounds i32, ptr addrspace(3) %scratch, i64 0
  store i32 1, ptr addrspace(3) %slot, align 4
  tail call void @air.wg.barrier(i32 2, i32 1)
  br label %done
done:
  ret void
}

declare i16 @air.simd_shuffle_up.s.i16(i16, i16)
declare void @air.wg.barrier(i32, i32)

!air.kernel = !{!0}
!air.function_constants = !{!5}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 3, !"air.arg_type_name", !"int", !"air.arg_name", !"scratch"}
!4 = !{i32 1, !"air.thread_index_in_simdgroup", !"air.arg_type_name", !"ushort", !"air.arg_name", !"lane"}
!5 = !{ptr addrspace(2) @_Z7enable0.MTL_FC_INIT_0_b, !"bool", !"enable0", i32 0, i1 true}
"#;

        let mut metal = metal_row_for_compare(&[], infer_plan(ll), Some("Kernel"));
        metal.fc_specialization = Some(FC_SPECIALIZATION_VALUES.into());
        metal.fc_values = Some(vec![FunctionConstantValueJson { index: 0, value: 1 }]);
        let reason = incompatible_function_constant_simdgroup_golden(ll, &metal)
            .expect("function-constant simdgroup golden");
        assert!(reason.contains("simdgroup/threadgroup-memory"), "{reason}");
    }

    #[test]
    fn bounded_control_local_array_index_seeds_in_range() {
        let ll = r#"
%"struct.metal::array.4" = type { [4 x <3 x half>] }

define <4 x float> @frag(ptr addrspace(2) %index) {
  %arr = alloca %"struct.metal::array.4", align 8
  %idx = load i32, ptr addrspace(2) %index, align 4
  %wide = sext i32 %idx to i64
  %slot = getelementptr inbounds %"struct.metal::array.4", ptr %arr, i64 0, i32 0, i64 %wide
  %v = load <3 x half>, ptr %slot, align 8
  %out = tail call <3 x float> @air.convert.f.v3f32.f.v3f16(<3 x half> %v)
  %out4 = shufflevector <3 x float> %out, <3 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 poison>
  %alpha = insertelement <4 x float> %out4, float 1.000000e+00, i64 3
  ret <4 x float> %alpha
}

declare <3 x float> @air.convert.f.v3f32.f.v3f16(<3 x half>)

!air.fragment = !{!0}
!0 = !{ptr @frag, !1, !2}
!1 = !{!3}
!2 = !{!4}
!3 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!4 = !{i32 0, !"air.buffer", !"air.buffer_size", i32 4, !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"int", !"air.arg_name", !"index"}
"#;

        let plan = infer_plan(ll);
        let index = plan
            .buffers
            .iter()
            .find(|buffer| buffer.index == 0)
            .expect("index buffer");
        assert_eq!(index.seed_mode, SEED_MODE_BOUNDED_CONTROL);
        assert_eq!(bounded_control_u32_at(index, 0), Some(0));

        let metal = metal_row_for_compare(&[], plan, Some("Fragment"));
        assert!(incompatible_bounded_control_local_array_index_golden(ll, &metal).is_none());
    }

    #[test]
    fn bounded_control_local_array_loop_index_golden_is_comparable() {
        let ll = r#"
define void @k(ptr addrspace(2) %dims, ptr addrspace(1) %out) {
entry:
  %arr = alloca [2 x float], align 4
  %dim = load i32, ptr addrspace(2) %dims, align 4
  store float 1.000000e+00, ptr %arr, align 4
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %next, %loop ]
  %wide = zext i32 %i to i64
  %slot = getelementptr inbounds [2 x float], ptr %arr, i64 0, i64 %wide
  %v = load float, ptr %slot, align 4
  %dst = getelementptr inbounds float, ptr addrspace(1) %out, i64 %wide
  store float %v, ptr addrspace(1) %dst, align 4
  %next = add nuw nsw i32 %i, 1
  %done = icmp eq i32 %next, 2
  br i1 %done, label %exit, label %loop

exit:
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.buffer", !"air.buffer_size", i32 4, !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"int", !"air.arg_name", !"dims"}
!4 = !{i32 1, !"air.buffer", !"air.buffer_size", i32 8, !"air.location_index", i32 1, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float*", !"air.arg_name", !"out"}
"#;

        let metal = metal_row_for_compare(&[], infer_plan(ll), Some("Kernel"));
        assert!(incompatible_bounded_control_local_array_index_golden(ll, &metal).is_none());
    }

    #[test]
    fn finite_struct_half_div_fragment_golden_is_missing() {
        let ll = r#"
define <4 x half> @frag(ptr addrspace(1) %coeff, <4 x half> %color) #0 {
  %p = getelementptr inbounds half, ptr addrspace(1) %coeff, i64 0
  %den = load half, ptr addrspace(1) %p, align 2
  %splat0 = insertelement <3 x half> poison, half %den, i64 0
  %splat = shufflevector <3 x half> %splat0, <3 x half> poison, <3 x i32> zeroinitializer
  %rgb = shufflevector <4 x half> %color, <4 x half> poison, <3 x i32> <i32 0, i32 1, i32 2>
  %div = fdiv fast <3 x half> %rgb, %splat
  %out3 = shufflevector <3 x half> %div, <3 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 poison>
  %out = insertelement <4 x half> %out3, half 0xH3C00, i64 3
  ret <4 x half> %out
}

attributes #0 = { "no-nans-fp-math"="true" "unsafe-fp-math"="true" }

!air.fragment = !{!0}
!air.compile_options = !{!7}
!0 = !{ptr @frag, !1, !2}
!1 = !{!3}
!2 = !{!4, !5}
!3 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"half4"}
!4 = !{i32 0, !"air.buffer", !"air.buffer_size", i32 131072, !"air.location_index", i32 7, i32 1, !"air.read", !"air.address_space", i32 1, !"air.struct_type_info", !6, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 2, !"air.arg_type_name", !"Coeff", !"air.arg_name", !"coeff"}
!5 = !{i32 1, !"air.fragment_input", !"air.arg_type_name", !"half4", !"air.arg_name", !"color"}
!6 = !{i32 0, i32 2, i32 1, !"half"}
!7 = !{!"air.compile.fast_math_enable"}
"#;

        let metal = metal_row_for_compare(&[], infer_plan(ll), Some("Fragment"));
        let reason = incompatible_finite_struct_half_fragment_golden(ll, &metal)
            .expect("finite-struct half fragment golden");
        assert!(reason.contains("fast half division"), "{reason}");
    }

    #[test]
    fn barycentric_derivative_fragment_golden_is_missing() {
        let ll = r#"
define <4 x half> @frag(<3 x float> %bary) {
  %w = tail call <3 x float> @air.fwidth.v3f32(<3 x float> %bary)
  %x = extractelement <3 x float> %w, i64 0
  %c = insertelement <4 x float> zeroinitializer, float %x, i64 0
  %out = tail call <4 x half> @air.convert.f.v4f16.f.v4f32(<4 x float> %c)
  ret <4 x half> %out
}

declare <3 x float> @air.fwidth.v3f32(<3 x float>)
declare <4 x half> @air.convert.f.v4f16.f.v4f32(<4 x float>)

!air.fragment = !{!0}
!0 = !{ptr @frag, !1, !2}
!1 = !{!3}
!2 = !{!4}
!3 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"half4"}
!4 = !{i32 0, !"air.barycentric_coord", !"air.center", !"air.perspective", !"air.arg_type_name", !"float3", !"air.arg_name", !"bary"}
"#;

        let metal = metal_row_for_compare(&[], infer_plan(ll), Some("Fragment"));
        let reason = incompatible_barycentric_derivative_fragment_golden(ll, &metal)
            .expect("barycentric derivative fragment golden");
        assert!(reason.contains("barycentric"), "{reason}");
        assert!(reason.contains("derivative"), "{reason}");
    }

    #[test]
    fn moltenvk_vertex_clip_distance_half_texture_golden_is_missing() {
        let ll = r#"
define void @vert(ptr addrspace(1) %tex) {
  ret void
}

!air.vertex = !{!0}
!0 = !{ptr @vert, !1, !2}
!1 = !{!3, !4}
!2 = !{!5}
!3 = !{!"air.position", !"air.arg_type_name", !"float4", !"air.arg_name", !"position"}
!4 = !{!"air.clip_distance", !"air.clip_distance_array_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"clipDistance"}
!5 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<half, sample>", !"air.arg_name", !"tex"}
"#;

        let metal = metal_row_for_compare(&[], infer_plan(ll), Some("Vertex"));
        let reason = incompatible_moltenvk_vertex_clip_distance_half_texture_golden(ll, &metal)
            .expect("MoltenVK ClipDistance half-texture vertex golden");
        assert!(reason.contains("clip_distance"), "{reason}");
        assert!(reason.contains("MoltenVK"), "{reason}");

        let fragment_metal = metal_row_for_compare(&[], infer_plan(ll), Some("Fragment"));
        assert!(
            incompatible_moltenvk_vertex_clip_distance_half_texture_golden(ll, &fragment_metal)
                .is_none()
        );
    }

    #[test]
    fn moltenvk_sampled_half_render_target_exact_golden_is_missing() {
        let ll = r#"
define <4 x half> @frag(ptr addrspace(1) %tex, <2 x float> %uv) #0 {
  %s = tail call { <4 x half>, i8 } @air.sample_texture_2d.v4f16(ptr addrspace(1) %tex, ptr addrspace(2) null, <2 x float> %uv, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %v = extractvalue { <4 x half>, i8 } %s, 0
  %d = tail call fast half @air.dot.v4f16(<4 x half> %v, <4 x half> %v)
  %out = insertelement <4 x half> %v, half %d, i64 3
  ret <4 x half> %out
}

declare { <4 x half>, i8 } @air.sample_texture_2d.v4f16(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32)
declare half @air.dot.v4f16(<4 x half>, <4 x half>)
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
        let reason = incompatible_moltenvk_sampled_half_render_target_exact_golden(ll, &metal)
            .expect("MoltenVK sampled half render-target exact golden");
        assert!(reason.contains("finite f16 texture"), "{reason}");
        assert!(
            reason.contains("MoltenVK exact byte comparison"),
            "{reason}"
        );

        let mut compare_none = metal;
        compare_none.compare = "none".into();
        assert!(
            incompatible_moltenvk_sampled_half_render_target_exact_golden(ll, &compare_none)
                .is_none()
        );
    }

    #[test]
    fn moltenvk_half_texture_output_exact_golden_is_missing() {
        let ll = r#"
define void @kernel(ptr addrspace(1) readonly %src, ptr addrspace(1) %out, <2 x i16> %gid) #0 {
  %sam = tail call ptr addrspace(2) @air.get_read_sampler()
  %read = tail call { <4 x half>, i8 } @air.read_texture_2d.i16.v4f16(ptr addrspace(1) readonly %src, ptr addrspace(2) %sam, <2 x i16> %gid, <2 x i16> zeroinitializer, i16 0, i32 1)
  %v = extractvalue { <4 x half>, i8 } %read, 0
  %sum = fadd fast <4 x half> %v, %v
  tail call void @air.write_texture_2d.i16.v4f16(ptr addrspace(1) %out, <2 x i16> %gid, <4 x half> %sum, i16 0, i32 2)
  ret void
}

declare ptr addrspace(2) @air.get_read_sampler()
declare { <4 x half>, i8 } @air.read_texture_2d.i16.v4f16(ptr addrspace(1) readonly, ptr addrspace(2), <2 x i16>, <2 x i16>, i16, i32)
declare void @air.write_texture_2d.i16.v4f16(ptr addrspace(1), <2 x i16>, <4 x half>, i16, i32)
attributes #0 = { "no-nans-fp-math"="true" "unsafe-fp-math"="true" }

!air.kernel = !{!0}
!0 = !{ptr @kernel, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.read", !"air.arg_type_name", !"texture2d<half, read>", !"air.arg_name", !"src"}
!4 = !{i32 1, !"air.texture", !"air.location_index", i32 1, i32 1, !"air.write", !"air.arg_type_name", !"texture2d<half, write>", !"air.arg_name", !"out"}
!5 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"ushort2", !"air.arg_name", !"gid"}
"#;

        let metal = metal_row_for_compare(&[], infer_plan(ll), Some("Kernel"));
        let reason = incompatible_moltenvk_half_texture_output_exact_golden(ll, &metal)
            .expect("MoltenVK half texture-output exact golden");
        assert!(reason.contains("finite f16 texture data"), "{reason}");
        assert!(reason.contains("texture write rounding"), "{reason}");

        let fragment_metal = metal_row_for_compare(&[], infer_plan(ll), Some("Fragment"));
        assert!(
            incompatible_moltenvk_half_texture_output_exact_golden(ll, &fragment_metal).is_none()
        );
    }

    #[test]
    fn moltenvk_storage_f32_texture_output_exact_golden_is_missing() {
        let ll = r#"
define void @kernel(ptr addrspace(1) readonly %src, ptr addrspace(1) %out, <2 x i32> %gid) #0 {
  %read = tail call { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1) readonly %src, ptr addrspace(2) null, <2 x i32> %gid, <2 x i32> zeroinitializer, i32 0, i32 1)
  %v = extractvalue { <4 x float>, i8 } %read, 0
  %x = extractelement <4 x float> %v, i64 0
  %p = tail call fast float @air.fast_powr.f32(float %x, float 2.000000e+00)
  %out4 = insertelement <4 x float> zeroinitializer, float %p, i64 0
  tail call void @air.write_texture_2d.v4f32(ptr addrspace(1) %out, <2 x i32> %gid, <4 x float> %out4, i32 0, i32 2)
  ret void
}

declare { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1) readonly, ptr addrspace(2), <2 x i32>, <2 x i32>, i32, i32)
declare float @air.fast_powr.f32(float, float)
declare void @air.write_texture_2d.v4f32(ptr addrspace(1), <2 x i32>, <4 x float>, i32, i32)
attributes #0 = { "no-nans-fp-math"="true" "unsafe-fp-math"="true" }

!air.kernel = !{!0}
!0 = !{ptr @kernel, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.read", !"air.arg_type_name", !"texture2d<float, read>", !"air.arg_name", !"src"}
!4 = !{i32 1, !"air.texture", !"air.location_index", i32 1, i32 1, !"air.write", !"air.arg_type_name", !"texture2d<float, write>", !"air.arg_name", !"out"}
!5 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint2", !"air.arg_name", !"gid"}
"#;

        let metal = metal_row_for_compare(&[], infer_plan(ll), Some("Kernel"));
        let reason = incompatible_moltenvk_storage_f32_texture_output_exact_golden(ll, &metal)
            .expect("MoltenVK storage f32 texture-output exact golden");
        assert!(reason.contains("finite f32 storage texture"), "{reason}");
        assert!(
            reason.contains("MoltenVK exact byte comparison"),
            "{reason}"
        );
    }

    #[test]
    fn moltenvk_storage_r32_imageblock_texture_output_exact_golden_is_missing() {
        let ll = r#"
%"struct.metal::_imageblock_base.92" = type { ptr addrspace(4) }

define void @kernel(%"struct.metal::_imageblock_base.92" %ib, ptr addrspace(1) %tex, <2 x i16> %gid, <2 x i16> %tid) #0 {
  %read = tail call { <4 x float>, i8 } @air.read_texture_2d.i16.v4f32(ptr addrspace(1) readonly %tex, ptr addrspace(2) null, <2 x i16> %gid, <2 x i16> zeroinitializer, i16 0, i32 1)
  %v = extractvalue { <4 x float>, i8 } %read, 0
  %x = extractelement <4 x float> %v, i64 0
  %r = tail call fast float @air.fast_sqrt.f32(float %x)
  %out = insertelement <4 x float> zeroinitializer, float %r, i64 0
  %ptr = tail call ptr addrspace(4) @air.imageblock_data(<2 x i16> %tid, i32 0, i16 0)
  store <4 x float> %out, ptr addrspace(4) %ptr, align 16
  tail call void @air.wg.barrier(i32 8, i32 1)
  tail call void @air.write_imageblock_slice_to_texture_2d.i16.v4f32(ptr addrspace(1) %tex, ptr addrspace(4) %ptr, i1 true, <2 x i16> zeroinitializer, <2 x i16> splat (i16 8), <2 x i16> %gid, i16 0, i1 false, i32 2)
  ret void
}

declare { <4 x float>, i8 } @air.read_texture_2d.i16.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x i16>, <2 x i16>, i16, i32)
declare float @air.fast_sqrt.f32(float)
declare ptr addrspace(4) @air.imageblock_data(<2 x i16>, i32, i16)
declare void @air.wg.barrier(i32, i32)
declare void @air.write_imageblock_slice_to_texture_2d.i16.v4f32(ptr addrspace(1), ptr addrspace(4), i1, <2 x i16>, <2 x i16>, <2 x i16>, i16, i1, i32)
attributes #0 = { "no-nans-fp-math"="true" "unsafe-fp-math"="true" }

!air.kernel = !{!0}
!0 = !{ptr @kernel, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5, !6}
!3 = !{i32 0, !"air.imageblock", !"explicit", !"air.imageblock_data_size", i32 16, !"air.arg_type_name", !"imageblock<ibRGBA, layout_explicit>", !"air.arg_name", !"ib"}
!4 = !{i32 1, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.arg_type_name", !"texture2d<float, read_write>", !"air.arg_name", !"tex"}
!5 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"ushort2", !"air.arg_name", !"gid"}
!6 = !{i32 3, !"air.thread_position_in_threadgroup", !"air.arg_type_name", !"ushort2", !"air.arg_name", !"tid"}
"#;

        let mut metal = metal_row_for_compare(&[], infer_plan(ll), Some("Kernel"));
        metal.plan.output.format = "R32Float".into();
        let reason = incompatible_moltenvk_storage_f32_texture_output_exact_golden(ll, &metal)
            .expect("MoltenVK storage R32 imageblock exact golden");
        assert!(reason.contains("finite f32 storage texture"), "{reason}");
        assert!(reason.contains("texture read/write rounding"), "{reason}");
    }

    #[test]
    fn moltenvk_sampled_f32_render_target_exact_golden_is_missing() {
        let ll = r#"
define <4 x float> @frag(ptr addrspace(1) %tex, <2 x float> %uv) #0 {
  %s = tail call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) %tex, ptr addrspace(2) null, <2 x float> %uv, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %v = extractvalue { <4 x float>, i8 } %s, 0
  ret <4 x float> %v
}

declare { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32)
attributes #0 = { "no-nans-fp-math"="true" "unsafe-fp-math"="true" }

!air.fragment = !{!0}
!0 = !{ptr @frag, !1, !2}
!1 = !{!3}
!2 = !{!4, !5}
!3 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!4 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"tex"}
!5 = !{i32 1, !"air.fragment_input", !"air.arg_type_name", !"float2", !"air.arg_name", !"uv"}
"#;

        let metal = metal_row_for_compare(&[], infer_plan(ll), Some("Fragment"));
        let reason = incompatible_moltenvk_sampled_f32_render_target_exact_golden(ll, &metal)
            .expect("MoltenVK sampled f32 render-target exact golden");
        assert!(reason.contains("finite f32 texture"), "{reason}");
        assert!(reason.contains("render-target rounding"), "{reason}");
    }

    #[test]
    fn moltenvk_sampled_f32_cube_buffer_exact_golden_is_missing() {
        let ll = r#"
define void @kernel(ptr addrspace(1) %cube, ptr addrspace(1) %out) #0 {
  %s = tail call { <4 x float>, i8 } @air.sample_texture_cube.v4f32(ptr addrspace(1) %cube, ptr addrspace(2) null, <3 x float> splat (float 1.000000e+00), i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %v = extractvalue { <4 x float>, i8 } %s, 0
  %x = extractelement <4 x float> %v, i64 0
  %r = tail call fast float @air.fast_rsqrt.f32(float %x)
  store float %r, ptr addrspace(1) %out, align 4
  ret void
}

declare { <4 x float>, i8 } @air.sample_texture_cube.v4f32(ptr addrspace(1), ptr addrspace(2), <3 x float>, i1, float, float, i32)
declare float @air.fast_rsqrt.f32(float)
attributes #0 = { "no-nans-fp-math"="true" "unsafe-fp-math"="true" }

!air.kernel = !{!0}
!0 = !{ptr @kernel, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texturecube<float, sample>", !"air.arg_name", !"cube"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_name", !"float", !"air.arg_name", !"out"}
"#;

        let mut metal = metal_row_for_compare(&[], infer_plan(ll), Some("Kernel"));
        metal.plan.output.format = "F32".into();
        let reason = incompatible_moltenvk_sampled_f32_cube_buffer_exact_golden(ll, &metal)
            .expect("MoltenVK sampled f32 cube buffer exact golden");
        assert!(reason.contains("f32 cube texture"), "{reason}");
        assert!(reason.contains("approximate math"), "{reason}");
    }

    #[test]
    fn moltenvk_fast_f32_buffer_output_exact_golden_is_missing() {
        let ll = r#"
define void @kernel(ptr addrspace(1) %out, ptr addrspace(2) %control, i32 %idx) #0 {
  %c = load i32, ptr addrspace(2) %control, align 4
  %ok = icmp eq i32 %c, 0
  %value = select i1 %ok, float 0.000000e+00, float -0.000000e+00
  %dst = getelementptr float, ptr addrspace(1) %out, i32 %idx
  store float %value, ptr addrspace(1) %dst, align 4
  ret void
}

attributes #0 = { "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "unsafe-fp-math"="true" }

!air.kernel = !{!0}
!0 = !{ptr @kernel, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_name", !"float*", !"air.arg_name", !"out"}
!4 = !{i32 1, !"air.buffer", !"air.buffer_size", i32 4, !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_name", !"uint", !"air.arg_name", !"control"}
!5 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"idx"}
"#;

        let mut metal = metal_row_for_compare(&[], infer_plan(ll), Some("Kernel"));
        metal.plan.output.format = "Rgba32Float".into();
        metal.plan.buffers[0].seed_mode = SEED_MODE_FINITE_FLOAT32.into();
        let reason = incompatible_moltenvk_fast_f32_buffer_output_exact_golden(ll, &metal)
            .expect("MoltenVK fast f32 buffer-output exact golden");
        assert!(reason.contains("finite f32 buffer output"), "{reason}");
        assert!(reason.contains("denorm flushing"), "{reason}");
    }

    #[test]
    fn moltenvk_fast_f32_input_buffer_exact_golden_is_missing() {
        let ll = r#"
define void @kernel(ptr addrspace(1) %a, ptr addrspace(1) %b, ptr addrspace(1) %out) #0 {
  %x = load float, ptr addrspace(1) %a, align 4
  %y = load float, ptr addrspace(1) %b, align 4
  %s = fadd fast float %x, %y
  %r = tail call fast float @air.fast_rsqrt.f32(float %s)
  store float %r, ptr addrspace(1) %out, align 4
  ret void
}

declare float @air.fast_rsqrt.f32(float)
attributes #0 = { "no-nans-fp-math"="true" "unsafe-fp-math"="true" }

!air.kernel = !{!0}
!0 = !{ptr @kernel, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_name", !"float", !"air.arg_name", !"a"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_name", !"float", !"air.arg_name", !"b"}
!5 = !{i32 2, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_name", !"float", !"air.arg_name", !"out"}
"#;

        let mut metal = metal_row_for_compare(&[], infer_plan(ll), Some("Kernel"));
        metal.plan.output.format = "F32".into();
        let reason = incompatible_moltenvk_fast_f32_input_buffer_exact_golden(ll, &metal)
            .expect("MoltenVK finite f32 input buffer exact golden");
        assert!(reason.contains("finite f32 buffer inputs"), "{reason}");
        assert!(
            reason.contains("MoltenVK exact byte comparison"),
            "{reason}"
        );
    }

    #[test]
    fn moltenvk_fast_half_buffer_output_exact_golden_is_missing() {
        let ll = r#"
define void @kernel(ptr addrspace(1) %src, ptr addrspace(1) %out) #0 {
  %h = load half, ptr addrspace(1) %src, align 2
  %f = tail call fast float @air.convert.f.f32.f.f16(half %h)
  %r = tail call fast float @air.fast_sqrt.f32(float %f)
  %o = tail call fast half @air.convert.f.f16.f.f32(float %r)
  store half %o, ptr addrspace(1) %out, align 2
  ret void
}

declare float @air.convert.f.f32.f.f16(half)
declare float @air.fast_sqrt.f32(float)
declare half @air.convert.f.f16.f.f32(float)
attributes #0 = { "no-nans-fp-math"="true" "unsafe-fp-math"="true" }

!air.kernel = !{!0}
!0 = !{ptr @kernel, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_name", !"half", !"air.arg_name", !"src"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_name", !"half", !"air.arg_name", !"out"}
"#;

        let mut metal = metal_row_for_compare(&[], infer_plan(ll), Some("Kernel"));
        metal.plan.output.format = "R16Float".into();
        let reason = incompatible_moltenvk_fast_half_buffer_output_exact_golden(ll, &metal)
            .expect("MoltenVK fast half buffer-output exact golden");
        assert!(reason.contains("finite f16 buffer output"), "{reason}");
        assert!(reason.contains("half conversion"), "{reason}");
    }

    #[test]
    fn moltenvk_fast_half_render_target_exact_golden_is_missing() {
        let ll = r#"
define <4 x half> @frag(half %x) #0 {
  %s = tail call fast half @air.fast_sqrt.f16(half %x)
  %out = insertelement <4 x half> zeroinitializer, half %s, i64 0
  ret <4 x half> %out
}

declare half @air.fast_sqrt.f16(half)
attributes #0 = { "no-nans-fp-math"="true" "unsafe-fp-math"="true" }

!air.fragment = !{!0}
!0 = !{ptr @frag, !1, !2}
!1 = !{!3}
!2 = !{!4}
!3 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"half4"}
!4 = !{i32 0, !"air.fragment_input", !"air.arg_type_name", !"half", !"air.arg_name", !"x"}
"#;

        let metal = metal_row_for_compare(&[], infer_plan(ll), Some("Fragment"));
        let reason = incompatible_moltenvk_fast_half_render_target_exact_golden(ll, &metal)
            .expect("MoltenVK fast half render-target exact golden");
        assert!(reason.contains("fast_sqrt"), "{reason}");
        assert!(reason.contains("render-target rounding"), "{reason}");
    }

    #[test]
    fn moltenvk_fast_raw_float_buffer_output_exact_golden_is_missing() {
        let ll = r#"
define void @kernel(ptr addrspace(1) %out, float %x, i32 %idx) #0 {
  %dst = getelementptr float, ptr addrspace(1) %out, i32 %idx
  store float %x, ptr addrspace(1) %dst, align 4
  ret void
}

attributes #0 = { "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "unsafe-fp-math"="true" }

!air.kernel = !{!0}
!0 = !{ptr @kernel, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_name", !"float*", !"air.arg_name", !"out"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_name", !"float", !"air.arg_name", !"x"}
!5 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"idx"}
"#;

        let mut metal = metal_row_for_compare(&[], infer_plan(ll), Some("Kernel"));
        metal.plan.output.format = "RawBytes".into();
        let reason = incompatible_moltenvk_fast_raw_float_buffer_output_exact_golden(ll, &metal)
            .expect("MoltenVK fast raw float buffer-output exact golden");
        assert!(reason.contains("raw buffer bytes"), "{reason}");
        assert!(reason.contains("signed-zero"), "{reason}");
    }

    #[test]
    fn moltenvk_packed_unorm_raw_float_buffer_output_exact_golden_is_missing() {
        let ll = r#"
define void @kernel(ptr addrspace(1) %src, ptr addrspace(1) %out, i32 %idx) #0 {
  %srcp = getelementptr i32, ptr addrspace(1) %src, i32 %idx
  %raw = load i32, ptr addrspace(1) %srcp, align 1
  %unpacked = tail call fast <4 x half> @air.unpack.unorm4x8.v4f16(i32 %raw)
  %scaled = fmul fast <4 x half> %unpacked, %unpacked
  %packed = tail call i32 @air.pack.unorm4x8.v4f16(<4 x half> %scaled)
  %dstp = getelementptr i32, ptr addrspace(1) %out, i32 %idx
  store i32 %packed, ptr addrspace(1) %dstp, align 1
  ret void
}

declare <4 x half> @air.unpack.unorm4x8.v4f16(i32)
declare i32 @air.pack.unorm4x8.v4f16(<4 x half>)
attributes #0 = { "no-nans-fp-math"="true" "unsafe-fp-math"="true" }

!air.kernel = !{!0}
!0 = !{ptr @kernel, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_name", !"uchar", !"air.arg_name", !"src"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_name", !"uchar", !"air.arg_name", !"out"}
!5 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"idx"}
"#;

        let mut metal = metal_row_for_compare(&[], infer_plan(ll), Some("Kernel"));
        metal.plan.output.format = "RawBytes".into();
        let reason = incompatible_moltenvk_fast_raw_float_buffer_output_exact_golden(ll, &metal)
            .expect("MoltenVK packed unorm raw float buffer-output exact golden");
        assert!(reason.contains("raw buffer bytes"), "{reason}");
        assert!(reason.contains("float denorm"), "{reason}");
    }

    #[test]
    fn moltenvk_integer_texture_fast_render_target_exact_golden_is_missing() {
        let ll = r#"
define <4 x half> @frag(ptr addrspace(1) readonly %tex, <2 x i16> %coord) #0 {
  %r = tail call { <4 x i16>, i8 } @air.read_texture_2d.u.v4i16(ptr addrspace(1) readonly %tex, ptr addrspace(2) null, <2 x i16> %coord, <2 x i16> zeroinitializer, i16 0, i32 1)
  %u = extractvalue { <4 x i16>, i8 } %r, 0
  %f = tail call fast <4 x half> @air.convert.f.v4f16.u.v4i16(<4 x i16> %u)
  %x = extractelement <4 x half> %f, i64 0
  %p = tail call fast half @air.fast_pow.f16(half %x, half 2.000000e+00)
  %out = insertelement <4 x half> zeroinitializer, half %p, i64 0
  ret <4 x half> %out
}

declare { <4 x i16>, i8 } @air.read_texture_2d.u.v4i16(ptr addrspace(1) readonly, ptr addrspace(2), <2 x i16>, <2 x i16>, i16, i32)
declare <4 x half> @air.convert.f.v4f16.u.v4i16(<4 x i16>)
declare half @air.fast_pow.f16(half, half)
attributes #0 = { "no-nans-fp-math"="true" "unsafe-fp-math"="true" }

!air.fragment = !{!0}
!0 = !{ptr @frag, !1, !2}
!1 = !{!3}
!2 = !{!4, !5}
!3 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"half4"}
!4 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<ushort, sample>", !"air.arg_name", !"tex"}
!5 = !{i32 1, !"air.fragment_input", !"air.arg_type_name", !"ushort2", !"air.arg_name", !"coord"}
"#;

        let metal = metal_row_for_compare(&[], infer_plan(ll), Some("Fragment"));
        let reason =
            incompatible_moltenvk_integer_texture_fast_render_target_exact_golden(ll, &metal)
                .expect("MoltenVK integer texture fast render-target exact golden");
        assert!(reason.contains("integer texture data"), "{reason}");
        assert!(reason.contains("approximate pow"), "{reason}");
    }

    #[test]
    fn moltenvk_scaled_integer_half_texture_output_exact_golden_is_missing() {
        let ll = r#"
define void @kernel(ptr addrspace(1) readonly %src, ptr addrspace(1) %out, <2 x i16> %gid) #0 {
  %raw = load <4 x i8>, ptr addrspace(1) %src, align 4
  %h = tail call fast <4 x half> @air.convert.f.v4f16.u.v4i8(<4 x i8> %raw)
  %scaled = fmul fast <4 x half> %h, splat (half 0xH1C04)
  tail call void @air.write_texture_2d.i16.v4f16(ptr addrspace(1) %out, <2 x i16> %gid, <4 x half> %scaled, i16 0, i32 2)
  ret void
}

declare <4 x half> @air.convert.f.v4f16.u.v4i8(<4 x i8>)
declare void @air.write_texture_2d.i16.v4f16(ptr addrspace(1), <2 x i16>, <4 x half>, i16, i32)
attributes #0 = { "no-nans-fp-math"="true" "unsafe-fp-math"="true" }

!air.kernel = !{!0}
!0 = !{ptr @kernel, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_name", !"uchar4", !"air.arg_name", !"src"}
!4 = !{i32 1, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.write", !"air.arg_type_name", !"texture2d<half, write>", !"air.arg_name", !"out"}
!5 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"ushort2", !"air.arg_name", !"gid"}
"#;

        let metal = metal_row_for_compare(&[], infer_plan(ll), Some("Kernel"));
        let reason =
            incompatible_moltenvk_scaled_integer_half_texture_output_exact_golden(ll, &metal)
                .expect("MoltenVK scaled integer half texture-output exact golden");
        assert!(reason.contains("integer buffer data"), "{reason}");
        assert!(reason.contains("half conversion"), "{reason}");
    }

    #[test]
    fn sampled_fast_exp_f32_texture_golden_is_missing() {
        let ll = r#"
define void @kernel(ptr addrspace(1) %table, ptr addrspace(1) %out, <2 x i32> %gid) #0 {
  %coord = tail call fast <2 x float> @air.convert.f.v2f32.u.v2i32(<2 x i32> %gid)
  %s = tail call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) %table, ptr addrspace(2) null, <2 x float> %coord, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %v = extractvalue { <4 x float>, i8 } %s, 0
  %x = extractelement <4 x float> %v, i64 0
  %e = tail call fast float @air.fast_exp.f32(float %x)
  %out4 = insertelement <4 x float> zeroinitializer, float %e, i64 0
  tail call void @air.write_texture_2d.v4f32(ptr addrspace(1) %out, <2 x i32> %gid, <4 x float> %out4, i32 0, i32 2)
  ret void
}

declare <2 x float> @air.convert.f.v2f32.u.v2i32(<2 x i32>)
declare { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32)
declare float @air.fast_exp.f32(float)
declare void @air.write_texture_2d.v4f32(ptr addrspace(1), <2 x i32>, <4 x float>, i32, i32)
attributes #0 = { "no-nans-fp-math"="true" "unsafe-fp-math"="true" }

!air.kernel = !{!0}
!0 = !{ptr @kernel, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 4, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"table"}
!4 = !{i32 1, !"air.texture", !"air.location_index", i32 2, i32 1, !"air.write", !"air.arg_type_name", !"texture2d<float, write>", !"air.arg_name", !"out"}
!5 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint2", !"air.arg_name", !"gid"}
"#;

        let metal = metal_row_for_compare(&[], infer_plan(ll), Some("Kernel"));
        let reason = incompatible_sampled_fast_exp_texture_golden(ll, &metal)
            .expect("sampled fast exp texture golden");
        assert!(reason.contains("fast_exp"), "{reason}");
    }

    #[test]
    fn sampled_fast_powr_integer_render_target_golden_is_missing() {
        let ll = r#"
define i32 @frag(<2 x float> %uv, ptr addrspace(1) %tex, ptr addrspace(2) %sampler) #0 {
  %s = tail call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) %tex, ptr addrspace(2) %sampler, <2 x float> %uv, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %v = extractvalue { <4 x float>, i8 } %s, 0
  %rgb = shufflevector <4 x float> %v, <4 x float> poison, <3 x i32> <i32 0, i32 1, i32 2>
  %mag = tail call fast <3 x float> @air.fast_fabs.v3f32(<3 x float> %rgb)
  %pow = tail call fast <3 x float> @air.fast_powr.v3f32(<3 x float> %mag, <3 x float> splat (float 0x3FC4640000000000))
  %lane = extractelement <3 x float> %pow, i64 0
  %u = tail call i16 @air.convert.u.i16.f.f32(float %lane)
  %packed = zext i16 %u to i32
  ret i32 %packed
}

declare { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32)
declare <3 x float> @air.fast_fabs.v3f32(<3 x float>)
declare <3 x float> @air.fast_powr.v3f32(<3 x float>, <3 x float>)
declare i16 @air.convert.u.i16.f.f32(float)
attributes #0 = { "no-nans-fp-math"="true" "unsafe-fp-math"="true" }

!air.fragment = !{!0}
!0 = !{ptr @frag, !1, !2}
!1 = !{!3}
!2 = !{!4, !5, !6}
!3 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"uint"}
!4 = !{i32 0, !"air.fragment_input", !"air.arg_type_name", !"float2", !"air.arg_name", !"uv"}
!5 = !{i32 1, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"tex"}
!6 = !{i32 2, !"air.sampler", !"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"sampler", !"air.arg_name", !"sampler"}
"#;

        let metal = metal_row_for_compare(&[], infer_plan(ll), Some("Fragment"));
        let reason = incompatible_sampled_fast_pow_texture_golden(ll, &metal)
            .expect("sampled fast powr texture golden");
        assert!(reason.contains("fast_pow/fast_powr"), "{reason}");
    }

    #[test]
    fn sampled_f32_dynamic_lod_render_target_golden_is_missing() {
        let ll = r#"
define <4 x float> @frag(<2 x float> %uv, ptr addrspace(1) %src, ptr addrspace(1) %mask) #0 {
  %mask_sample = tail call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) %mask, ptr addrspace(2) null, <2 x float> %uv, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %mask_v = extractvalue { <4 x float>, i8 } %mask_sample, 0
  %alpha = extractelement <4 x float> %mask_v, i64 3
  %scaled = fmul fast float %alpha, 4.000000e+00
  %lod = tail call fast float @air.fast_log2.f32(float %scaled)
  %delta0 = insertelement <2 x float> poison, float %lod, i64 0
  %delta = shufflevector <2 x float> %delta0, <2 x float> poison, <2 x i32> zeroinitializer
  %coord = fadd fast <2 x float> %uv, %delta
  %sample = tail call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) %src, ptr addrspace(2) null, <2 x float> %coord, i1 true, <2 x i32> zeroinitializer, i1 true, float %lod, float 0.000000e+00, i32 0)
  %out = extractvalue { <4 x float>, i8 } %sample, 0
  ret <4 x float> %out
}

declare { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32)
declare float @air.fast_log2.f32(float)
attributes #0 = { "unsafe-fp-math"="true" }

!air.fragment = !{!0}
!0 = !{ptr @frag, !1, !2}
!1 = !{!3}
!2 = !{!4, !5, !6}
!3 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!4 = !{i32 0, !"air.fragment_input", !"air.arg_type_name", !"float2", !"air.arg_name", !"uv"}
!5 = !{i32 1, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"src"}
!6 = !{i32 2, !"air.texture", !"air.location_index", i32 1, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"mask"}
"#;

        let metal = metal_row_for_compare(&[], infer_plan(ll), Some("Fragment"));
        let reason = incompatible_sampled_f32_dynamic_lod_render_target_golden(ll, &metal)
            .expect("sampled f32 dynamic LOD render-target golden");
        assert!(reason.contains("dynamic LOD"), "{reason}");
    }

    #[test]
    fn sampled_f32_texture_array_render_targets_golden_is_missing() {
        let ll = r#"
@__air_sampler_state = internal addrspace(2) constant [2 x i64] [i64 34901797601055744, i64 0], align 8

define <{ <4 x float>, <4 x float> }> @frag(<2 x float> %uv0, <2 x float> %uv1, ptr readonly byval([2 x ptr addrspace(1)]) %textures) #0 {
  %p0 = getelementptr inbounds [2 x ptr addrspace(1)], ptr %textures, i64 0, i64 0
  %t0 = load ptr addrspace(1), ptr %p0, align 8
  %s0 = tail call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) %t0, ptr addrspace(2) @__air_sampler_state, <2 x float> %uv0, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %v0 = extractvalue { <4 x float>, i8 } %s0, 0
  %p1 = getelementptr inbounds [2 x ptr addrspace(1)], ptr %textures, i64 0, i64 1
  %t1 = load ptr addrspace(1), ptr %p1, align 8
  %s1 = tail call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) %t1, ptr addrspace(2) @__air_sampler_state, <2 x float> %uv1, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %v1 = extractvalue { <4 x float>, i8 } %s1, 0
  %o0 = insertvalue <{ <4 x float>, <4 x float> }> undef, <4 x float> %v0, 0
  %o1 = insertvalue <{ <4 x float>, <4 x float> }> %o0, <4 x float> %v1, 1
  ret <{ <4 x float>, <4 x float> }> %o1
}

declare { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32)
attributes #0 = { "no-nans-fp-math"="true" "unsafe-fp-math"="true" }

!air.fragment = !{!0}
!air.sampler_states = !{!7}
!0 = !{ptr @frag, !1, !2}
!1 = !{!3, !4}
!2 = !{!5, !6}
!3 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4", !"air.arg_name", !"left"}
!4 = !{!"air.render_target", i32 1, i32 0, !"air.arg_type_name", !"float4", !"air.arg_name", !"right"}
!5 = !{i32 0, !"air.fragment_input", !"air.arg_type_name", !"float2", !"air.arg_name", !"uv0"}
!6 = !{i32 2, !"air.texture", !"air.location_index", i32 0, i32 2, !"air.sample", !"air.arg_type_name", !"array<texture2d<float, sample>, 2>", !"air.arg_name", !"textures"}
!7 = !{!"air.sampler_state", ptr addrspace(2) @__air_sampler_state}
"#;

        let metal = metal_row_for_compare(&[], infer_plan(ll), Some("Fragment"));
        let reason = incompatible_sampled_f32_texture_array_render_targets_golden(ll, &metal)
            .expect("sampled f32 texture-array render-target golden");
        assert!(reason.contains("multiple float render targets"), "{reason}");
        assert!(reason.contains("texture-array"), "{reason}");
    }

    #[test]
    fn fragment_half_pow_signed_base_render_target_golden_is_missing() {
        let ll = r#"
define <4 x half> @frag(<2 x float> %uv) #0 {
  %u = extractelement <2 x float> %uv, i64 0
  %h = fptrunc float %u to half
  %base0 = insertelement <3 x half> poison, half %h, i64 0
  %base = shufflevector <3 x half> %base0, <3 x half> poison, <3 x i32> zeroinitializer
  %signed = fsub fast <3 x half> splat (half 0xH3C9A), %base
  %pow = tail call fast <3 x half> @air.pow.v3f16(<3 x half> %signed, <3 x half> splat (half 0xH3C00))
  %out3 = shufflevector <3 x half> %pow, <3 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 poison>
  %out = insertelement <4 x half> %out3, half 0xH3C00, i64 3
  ret <4 x half> %out
}

declare <3 x half> @air.pow.v3f16(<3 x half>, <3 x half>)
attributes #0 = { "no-nans-fp-math"="true" "unsafe-fp-math"="true" }

!air.fragment = !{!0}
!0 = !{ptr @frag, !1, !2}
!1 = !{!3}
!2 = !{!4}
!3 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"half4"}
!4 = !{i32 0, !"air.fragment_input", !"air.arg_type_name", !"float2", !"air.arg_name", !"uv"}
"#;

        let metal = metal_row_for_compare(&[], infer_plan(ll), Some("Fragment"));
        let reason = incompatible_fragment_half_pow_render_target_golden(ll, &metal)
            .expect("fragment half pow render-target golden");
        assert!(reason.contains("negative-base pow"), "{reason}");
    }

    #[test]
    fn fragment_fast_pow_rsqrt_render_target_golden_is_missing() {
        let ll = r#"
define <4 x float> @frag(float %x) #0 {
  %r = tail call fast float @air.fast_rsqrt.f32(float %x)
  %p = tail call fast float @air.fast_pow.f32(float %r, float 2.000000e+00)
  %out0 = insertelement <4 x float> zeroinitializer, float %p, i64 0
  ret <4 x float> %out0
}

declare float @air.fast_rsqrt.f32(float)
declare float @air.fast_pow.f32(float, float)
attributes #0 = { "no-nans-fp-math"="true" "unsafe-fp-math"="true" }

!air.fragment = !{!0}
!0 = !{ptr @frag, !1, !2}
!1 = !{!3}
!2 = !{!4}
!3 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!4 = !{i32 0, !"air.fragment_input", !"air.arg_type_name", !"float", !"air.arg_name", !"x"}
"#;

        let metal = metal_row_for_compare(&[], infer_plan(ll), Some("Fragment"));
        let reason = incompatible_fragment_fast_pow_rsqrt_render_target_golden(ll, &metal)
            .expect("fragment fast pow/rsqrt render-target golden");
        assert!(reason.contains("fast_pow"), "{reason}");
        assert!(reason.contains("fast_rsqrt"), "{reason}");
    }

    #[test]
    fn fragment_half_pow_bounded_control_render_target_golden_is_missing() {
        let ll = r#"
%"struct.Params" = type { float, float, float }

define <4 x half> @frag(<2 x float> %uv, ptr addrspace(2) %params, <4 x half> %color) #0 {
  %rgb = shufflevector <4 x half> %color, <4 x half> poison, <3 x i32> <i32 0, i32 1, i32 2>
  %gain_p = getelementptr inbounds %"struct.Params", ptr addrspace(2) %params, i64 0, i32 0
  %gain_f = load float, ptr addrspace(2) %gain_p, align 4
  %gain_h = fptrunc float %gain_f to half
  %gain0 = insertelement <3 x half> poison, half %gain_h, i64 0
  %gain = shufflevector <3 x half> %gain0, <3 x half> poison, <3 x i32> zeroinitializer
  %offset_p = getelementptr inbounds %"struct.Params", ptr addrspace(2) %params, i64 0, i32 1
  %offset_f = load float, ptr addrspace(2) %offset_p, align 4
  %offset_h = fptrunc float %offset_f to half
  %offset0 = insertelement <3 x half> poison, half %offset_h, i64 0
  %offset = shufflevector <3 x half> %offset0, <3 x half> poison, <3 x i32> zeroinitializer
  %scaled = fmul fast <3 x half> %rgb, %gain
  %base = fadd fast <3 x half> %scaled, %offset
  %gamma_p = getelementptr inbounds %"struct.Params", ptr addrspace(2) %params, i64 0, i32 2
  %gamma_f = load float, ptr addrspace(2) %gamma_p, align 4
  %inv = fdiv fast float 1.000000e+00, %gamma_f
  %inv_h = fptrunc float %inv to half
  %inv0 = insertelement <3 x half> poison, half %inv_h, i64 0
  %exp = shufflevector <3 x half> %inv0, <3 x half> poison, <3 x i32> zeroinitializer
  %pow = tail call fast <3 x half> @air.pow.v3f16(<3 x half> %base, <3 x half> %exp)
  %out3 = shufflevector <3 x half> %pow, <3 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 poison>
  %alpha = extractelement <4 x half> %color, i64 3
  %out = insertelement <4 x half> %out3, half %alpha, i64 3
  ret <4 x half> %out
}

declare <3 x half> @air.pow.v3f16(<3 x half>, <3 x half>)
attributes #0 = { "no-nans-fp-math"="true" "unsafe-fp-math"="true" }

!air.fragment = !{!0}
!0 = !{ptr @frag, !1, !2}
!1 = !{!3}
!2 = !{!4, !5, !6}
!3 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"half4"}
!4 = !{i32 0, !"air.fragment_input", !"air.arg_type_name", !"float2", !"air.arg_name", !"uv"}
!5 = !{i32 1, !"air.buffer", !"air.buffer_size", i32 12, !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !7, !"air.arg_type_size", i32 12, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"Params", !"air.arg_name", !"params"}
!6 = !{i32 2, !"air.render_target", i32 0, i32 1, !"air.arg_type_name", !"half4", !"air.arg_name", !"color"}
!7 = !{i32 0, i32 4, i32 0, !"float", !"gain", i32 4, i32 4, i32 0, !"float", !"offset", i32 8, i32 4, i32 0, !"float", !"gamma"}
"#;

        let metal = metal_row_for_compare(&[], infer_plan(ll), Some("Fragment"));
        let reason = incompatible_fragment_half_pow_render_target_golden(ll, &metal)
            .expect("bounded-control fragment half pow render-target golden");
        assert!(reason.contains("bounded-control"), "{reason}");
        assert!(reason.contains("AIR pow"), "{reason}");
    }

    #[test]
    fn sampled_half_domain_sensitive_texture_golden_is_missing() {
        let ll = r#"
@__air_sampler_state = internal addrspace(2) constant [2 x i64] [i64 34901797601050697, i64 0], align 8

define void @kernel(ptr addrspace(1) %src, ptr addrspace(1) %out, <2 x i16> %gid) #0 {
  %coord = tail call fast <2 x float> @air.convert.f.v2f32.u.v2i16(<2 x i16> %gid)
  %s = tail call { <4 x half>, i8 } @air.sample_texture_2d.v4f16(ptr addrspace(1) %src, ptr addrspace(2) @__air_sampler_state, <2 x float> %coord, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %v = extractvalue { <4 x half>, i8 } %s, 0
  %x_h = extractelement <4 x half> %v, i64 0
  %x = fpext half %x_h to float
  %log = tail call fast float @air.fast_log.f32(float %x)
  %root = tail call fast float @air.fast_sqrt.f32(float %log)
  %wide0 = insertelement <4 x float> poison, float %root, i64 0
  %wide = shufflevector <4 x float> %wide0, <4 x float> poison, <4 x i32> zeroinitializer
  tail call void @air.write_texture_2d.i16.v4f32(ptr addrspace(1) %out, <2 x i16> %gid, <4 x float> %wide, i16 0, i32 2)
  ret void
}

declare <2 x float> @air.convert.f.v2f32.u.v2i16(<2 x i16>)
declare { <4 x half>, i8 } @air.sample_texture_2d.v4f16(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32)
declare float @air.fast_log.f32(float)
declare float @air.fast_sqrt.f32(float)
declare void @air.write_texture_2d.i16.v4f32(ptr addrspace(1), <2 x i16>, <4 x float>, i16, i32)
attributes #0 = { "no-nans-fp-math"="true" "unsafe-fp-math"="true" }

!air.kernel = !{!0}
!air.sampler_states = !{!6}
!0 = !{ptr @kernel, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<half, sample>", !"air.arg_name", !"src"}
!4 = !{i32 1, !"air.texture", !"air.location_index", i32 1, i32 1, !"air.write", !"air.arg_type_name", !"texture2d<float, write>", !"air.arg_name", !"out"}
!5 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"ushort2", !"air.arg_name", !"gid"}
!6 = !{!"air.sampler_state", ptr addrspace(2) @__air_sampler_state}
"#;

        let metal = metal_row_for_compare(&[], infer_plan(ll), Some("Kernel"));
        let reason = incompatible_sampled_half_domain_sensitive_texture_golden(ll, &metal)
            .expect("sampled half domain-sensitive texture golden");
        assert!(reason.contains("domain-sensitive math"), "{reason}");
    }

    #[test]
    fn fast_procedural_half_texture_golden_is_missing() {
        let ll = r#"
define void @kernel(ptr addrspace(1) %out, <2 x i32> %gid) #0 {
  %coord = tail call fast <2 x float> @air.convert.f.v2f32.u.v2i32(<2 x i32> %gid)
  %x = extractelement <2 x float> %coord, i64 0
  %s = tail call fast float @air.fast_sin.f32(float %x)
  %f = tail call fast float @air.fast_fract.f32(float %s)
  %h = fptrunc float %f to half
  %v0 = insertelement <4 x half> zeroinitializer, half %h, i64 0
  tail call void @air.write_texture_2d.v4f16(ptr addrspace(1) %out, <2 x i32> %gid, <4 x half> %v0, i32 0, i32 2)
  ret void
}

declare <2 x float> @air.convert.f.v2f32.u.v2i32(<2 x i32>)
declare float @air.fast_sin.f32(float)
declare float @air.fast_fract.f32(float)
declare void @air.write_texture_2d.v4f16(ptr addrspace(1), <2 x i32>, <4 x half>, i32, i32)
attributes #0 = { "no-nans-fp-math"="true" "unsafe-fp-math"="true" }

!air.kernel = !{!0}
!0 = !{ptr @kernel, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.write", !"air.arg_type_name", !"texture2d<half, write>", !"air.arg_name", !"out"}
!4 = !{i32 1, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint2", !"air.arg_name", !"gid"}
"#;

        let metal = metal_row_for_compare(&[], infer_plan(ll), Some("Kernel"));
        let reason = incompatible_fast_procedural_half_texture_golden(ll, &metal)
            .expect("fast procedural half texture golden");
        assert!(reason.contains("fast trigonometric/fract"), "{reason}");
    }

    #[test]
    fn sampled_half_linear_imageblock_texture_golden_is_missing() {
        let ll = r#"
@__air_sampler_state = internal addrspace(2) constant [2 x i64] [i64 34901797601020489, i64 0], align 8

define void @kernel(ptr addrspace(4) %imageblock, ptr addrspace(1) %src, ptr addrspace(1) %out, <2 x i16> %gid) #0 {
  %coord_i = tail call fast <2 x float> @air.convert.f.v2f32.u.v2i16(<2 x i16> %gid)
  %coord = fmul fast <2 x float> %coord_i, splat (float 1.250000e-01)
  %s = tail call { <4 x half>, i8 } @air.sample_texture_2d.v4f16(ptr addrspace(1) %src, ptr addrspace(2) @__air_sampler_state, <2 x float> %coord, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %v = extractvalue { <4 x half>, i8 } %s, 0
  %slot = tail call ptr addrspace(4) @air.imageblock_data(<2 x i16> %gid, i32 0, i16 0)
  store <4 x half> %v, ptr addrspace(4) %slot
  tail call void @air.write_imageblock_slice_to_texture_2d.i16.v4f16(ptr addrspace(1) %out, ptr addrspace(4) %slot, i1 false, <2 x i16> zeroinitializer, <2 x i16> undef, <2 x i16> %gid, i16 0, i1 false, i32 2)
  ret void
}

declare <2 x float> @air.convert.f.v2f32.u.v2i16(<2 x i16>)
declare { <4 x half>, i8 } @air.sample_texture_2d.v4f16(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32)
declare ptr addrspace(4) @air.imageblock_data(<2 x i16>, i32, i16)
declare void @air.write_imageblock_slice_to_texture_2d.i16.v4f16(ptr addrspace(1), ptr addrspace(4), i1, <2 x i16>, <2 x i16>, <2 x i16>, i16, i1, i32)
attributes #0 = { "no-nans-fp-math"="true" "unsafe-fp-math"="true" }

!air.kernel = !{!0}
!air.sampler_states = !{!7}
!0 = !{ptr @kernel, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5, !6}
!3 = !{i32 0, !"air.imageblock", !"explicit", !"air.arg_type_name", !"imageblock<half4>", !"air.arg_name", !"imageblock"}
!4 = !{i32 1, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<half, sample>", !"air.arg_name", !"src"}
!5 = !{i32 2, !"air.texture", !"air.location_index", i32 1, i32 1, !"air.write", !"air.arg_type_name", !"texture2d<half, write>", !"air.arg_name", !"out"}
!6 = !{i32 3, !"air.thread_position_in_grid", !"air.arg_type_name", !"ushort2", !"air.arg_name", !"gid"}
!7 = !{!"air.sampler_state", ptr addrspace(2) @__air_sampler_state}
"#;

        let metal = metal_row_for_compare(&[], infer_plan(ll), Some("Kernel"));
        let reason = incompatible_sampled_half_linear_filter_texture_golden(ll, &metal)
            .expect("sampled half linear imageblock texture golden");
        assert!(reason.contains("imageblock"), "{reason}");
    }

    #[test]
    fn sampled_half_linear_scaled_texture_golden_is_missing() {
        let ll = r#"
@__air_sampler_state = internal addrspace(2) constant [2 x i64] [i64 34901797601055817, i64 0], align 8

define void @upsample(ptr addrspace(1) %src, ptr addrspace(1) %out, ptr addrspace(1) readonly %scale, <2 x i16> %gid) #0 {
  %coord_i = tail call fast <2 x float> @air.convert.f.v2f32.u.v2i16(<2 x i16> %gid)
  %center = fadd fast <2 x float> %coord_i, splat (float 5.000000e-01)
  %s = load float, ptr addrspace(1) %scale, align 4
  %lane = insertelement <2 x float> poison, float %s, i64 0
  %scale2 = shufflevector <2 x float> %lane, <2 x float> poison, <2 x i32> zeroinitializer
  %uv = fmul fast <2 x float> %scale2, %center
  %sample = tail call { <4 x half>, i8 } @air.sample_texture_2d.v4f16(ptr addrspace(1) %src, ptr addrspace(2) @__air_sampler_state, <2 x float> %uv, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %rgba = extractvalue { <4 x half>, i8 } %sample, 0
  tail call void @air.write_texture_2d.i16.v4f16(ptr addrspace(1) %out, <2 x i16> %gid, <4 x half> %rgba, i16 0, i32 2)
  ret void
}

declare <2 x float> @air.convert.f.v2f32.u.v2i16(<2 x i16>)
declare { <4 x half>, i8 } @air.sample_texture_2d.v4f16(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32)
declare void @air.write_texture_2d.i16.v4f16(ptr addrspace(1), <2 x i16>, <4 x half>, i16, i32)
attributes #0 = { "no-nans-fp-math"="true" "unsafe-fp-math"="true" }

!air.kernel = !{!0}
!air.sampler_states = !{!6}
!0 = !{ptr @upsample, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5, !7}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<half, sample>", !"air.arg_name", !"src"}
!4 = !{i32 1, !"air.texture", !"air.location_index", i32 1, i32 1, !"air.write", !"air.arg_type_name", !"texture2d<half, write>", !"air.arg_name", !"out"}
!5 = !{i32 2, !"air.buffer", !"air.buffer_size", i32 4, !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"scale"}
!6 = !{!"air.sampler_state", ptr addrspace(2) @__air_sampler_state}
!7 = !{i32 3, !"air.thread_position_in_grid", !"air.arg_type_name", !"ushort2", !"air.arg_name", !"gid"}
"#;

        let metal = metal_row_for_compare(&[], infer_plan(ll), Some("Kernel"));
        let reason = incompatible_sampled_half_linear_filter_texture_golden(ll, &metal)
            .expect("sampled half scaled linear texture golden");
        assert!(reason.contains("buffer-scaled coordinates"), "{reason}");
        assert!(reason.contains("f16 texture output"), "{reason}");
    }

    #[test]
    fn sampled_half_gather_imageblock_texture_golden_is_missing() {
        let ll = r#"
@__air_sampler_state = internal addrspace(2) constant [2 x i64] [i64 34901797601050697, i64 0], align 8

define void @kernel(ptr addrspace(1) %src, ptr addrspace(1) %out, <2 x i16> %gid) #0 {
  %coord = tail call fast <2 x float> @air.convert.f.v2f32.u.v2i16(<2 x i16> %gid)
  %g = tail call { <4 x half>, i8 } @air.gather_texture_2d.v4f16(ptr addrspace(1) %src, ptr addrspace(2) @__air_sampler_state, <2 x float> %coord, i1 true, <2 x i32> zeroinitializer, i32 0, i32 0)
  %v = extractvalue { <4 x half>, i8 } %g, 0
  %slot = tail call ptr addrspace(4) @air.imageblock_data(<2 x i16> %gid, i32 0, i16 0)
  store <4 x half> %v, ptr addrspace(4) %slot
  tail call void @air.write_imageblock_slice_to_texture_2d.i16.v4f16(ptr addrspace(1) %out, ptr addrspace(4) %slot, i1 true, <2 x i16> zeroinitializer, <2 x i16> %gid, <2 x i16> %gid, i16 0, i1 false, i32 2)
  ret void
}

declare <2 x float> @air.convert.f.v2f32.u.v2i16(<2 x i16>)
declare { <4 x half>, i8 } @air.gather_texture_2d.v4f16(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i32, i32)
declare ptr addrspace(4) @air.imageblock_data(<2 x i16>, i32, i16)
declare void @air.write_imageblock_slice_to_texture_2d.i16.v4f16(ptr addrspace(1), ptr addrspace(4), i1, <2 x i16>, <2 x i16>, <2 x i16>, i16, i1, i32)
attributes #0 = { "no-nans-fp-math"="true" "unsafe-fp-math"="true" }

!air.kernel = !{!0}
!air.sampler_states = !{!6}
!0 = !{ptr @kernel, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<half, sample>", !"air.arg_name", !"src"}
!4 = !{i32 1, !"air.texture", !"air.location_index", i32 1, i32 1, !"air.write", !"air.arg_type_name", !"texture2d<half, write>", !"air.arg_name", !"out"}
!5 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"ushort2", !"air.arg_name", !"gid"}
!6 = !{!"air.sampler_state", ptr addrspace(2) @__air_sampler_state}
"#;

        let metal = metal_row_for_compare(&[], infer_plan(ll), Some("Kernel"));
        let reason = incompatible_sampled_half_linear_filter_texture_golden(ll, &metal)
            .expect("sampled half gather imageblock texture golden");
        assert!(reason.contains("gathers finite f16"), "{reason}");
    }

    #[test]
    fn storage_half_imageblock_texture_golden_is_missing() {
        let ll = r#"
define void @kernel(ptr addrspace(1) readonly %src, ptr addrspace(1) %out, <2 x i16> %ib, <2 x i16> %gid) {
  %sam = tail call ptr addrspace(2) @air.get_read_sampler()
  %read = tail call { <4 x half>, i8 } @air.read_texture_2d.i16.v4f16(ptr addrspace(1) readonly %src, ptr addrspace(2) %sam, <2 x i16> %gid, <2 x i16> zeroinitializer, i16 0, i32 1)
  %v = extractvalue { <4 x half>, i8 } %read, 0
  %slot = tail call ptr addrspace(4) @air.imageblock_data(<2 x i16> %ib, i32 0, i16 0)
  store <4 x half> %v, ptr addrspace(4) %slot
  tail call void @air.wg.barrier(i32 8, i32 1)
  tail call void @air.write_imageblock_slice_to_texture_2d.i16.v4f16(ptr addrspace(1) %out, ptr addrspace(4) %slot, i1 false, <2 x i16> zeroinitializer, <2 x i16> undef, <2 x i16> %gid, i16 0, i1 false, i32 2)
  ret void
}

declare ptr addrspace(2) @air.get_read_sampler()
declare { <4 x half>, i8 } @air.read_texture_2d.i16.v4f16(ptr addrspace(1) readonly, ptr addrspace(2), <2 x i16>, <2 x i16>, i16, i32)
declare ptr addrspace(4) @air.imageblock_data(<2 x i16>, i32, i16)
declare void @air.wg.barrier(i32, i32)
declare void @air.write_imageblock_slice_to_texture_2d.i16.v4f16(ptr addrspace(1), ptr addrspace(4), i1, <2 x i16>, <2 x i16>, <2 x i16>, i16, i1, i32)

!air.kernel = !{!0}
!0 = !{ptr @kernel, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5, !6}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.read", !"air.arg_type_name", !"texture2d<half, read>", !"air.arg_name", !"src"}
!4 = !{i32 1, !"air.texture", !"air.location_index", i32 1, i32 1, !"air.write", !"air.arg_type_name", !"texture2d<half, write>", !"air.arg_name", !"out"}
!5 = !{i32 2, !"air.thread_position_in_threadgroup", !"air.arg_type_name", !"ushort2", !"air.arg_name", !"ib"}
!6 = !{i32 3, !"air.thread_position_in_grid", !"air.arg_type_name", !"ushort2", !"air.arg_name", !"gid"}
"#;

        let metal = metal_row_for_compare(&[], infer_plan(ll), Some("Kernel"));
        let reason = incompatible_storage_half_imageblock_texture_golden(ll, &metal)
            .expect("storage half imageblock texture golden");
        assert!(reason.contains("storage texture"), "{reason}");
        assert!(reason.contains("imageblock"), "{reason}");
    }

    #[test]
    fn uninitialized_half_imageblock_texture_golden_is_missing() {
        let ll = r#"
%"struct.metal::_imageblock_base" = type { ptr addrspace(4) }

define void @kernel(ptr addrspace(1) %out, %"struct.metal::_imageblock_base" %imageblock, <2 x i16> %gid, <2 x i16> %pid) #0 {
  %slot = tail call ptr addrspace(4) @air.imageblock_data(<2 x i16> %pid, i32 0, i16 0)
  tail call void @air.write_imageblock_slice_to_texture_2d.i16.f16(ptr addrspace(1) %out, ptr addrspace(4) %slot, i1 false, <2 x i16> zeroinitializer, <2 x i16> undef, <2 x i16> %gid, i16 0, i1 false, i32 2)
  ret void
}

declare ptr addrspace(4) @air.imageblock_data(<2 x i16>, i32, i16)
declare void @air.write_imageblock_slice_to_texture_2d.i16.f16(ptr addrspace(1), ptr addrspace(4), i1, <2 x i16>, <2 x i16>, <2 x i16>, i16, i1, i32)
attributes #0 = { "no-nans-fp-math"="true" "unsafe-fp-math"="true" }

!air.kernel = !{!0}
!0 = !{ptr @kernel, !1, !2}
!1 = !{}
!2 = !{!3, !4, !6, !7}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 1, i32 1, !"air.write", !"air.arg_type_name", !"texture2d<half, write>", !"air.arg_name", !"out"}
!4 = !{i32 1, !"air.imageblock", !"explicit", !"air.imageblock_data_size", i32 2, !"air.struct_type_info", !5, !"air.arg_type_align_size", i32 2, !"air.arg_type_name", !"imageblock<ImageBlockData, layout_explicit>", !"air.arg_name", !"imageblock"}
!5 = !{i32 0, i32 2, i32 0, !"half", !"depth"}
!6 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"ushort2", !"air.arg_name", !"gid"}
!7 = !{i32 3, !"air.thread_position_in_threadgroup", !"air.arg_type_name", !"ushort2", !"air.arg_name", !"pid"}
"#;

        let metal = metal_row_for_compare(&[], infer_plan(ll), Some("Kernel"));
        let reason = incompatible_uninitialized_half_imageblock_texture_golden(ll, &metal)
            .expect("uninitialized half imageblock texture golden");
        assert!(reason.contains("uninitialized scalar half"), "{reason}");
        assert!(reason.contains("imageblock"), "{reason}");
    }

    #[test]
    fn sampled_f32_imageblock_texture_golden_is_missing() {
        let ll = r#"
@__air_sampler_state = internal addrspace(2) constant [2 x i64] [i64 34901797601053257, i64 0], align 8

define void @kernel(ptr addrspace(1) readonly %src, ptr addrspace(1) %out, <2 x i32> %gid, <2 x i16> %ib) {
  %coord = tail call fast <2 x float> @air.convert.f.v2f32.u.v2i32(<2 x i32> %gid)
  %s = tail call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) readonly %src, ptr addrspace(2) @__air_sampler_state, <2 x float> %coord, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %v = extractvalue { <4 x float>, i8 } %s, 0
  %slot = tail call ptr addrspace(4) @air.imageblock_data(<2 x i16> %ib, i32 0, i16 0)
  store <4 x float> %v, ptr addrspace(4) %slot
  tail call void @air.wg.barrier(i32 8, i32 1)
  tail call void @air.write_imageblock_slice_to_texture_2d.v4f32(ptr addrspace(1) %out, ptr addrspace(4) %slot, i1 false, <2 x i16> zeroinitializer, <2 x i16> undef, <2 x i32> %gid, i32 0, i1 false, i32 2)
  ret void
}

declare <2 x float> @air.convert.f.v2f32.u.v2i32(<2 x i32>)
declare { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) readonly, ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32)
declare ptr addrspace(4) @air.imageblock_data(<2 x i16>, i32, i16)
declare void @air.wg.barrier(i32, i32)
declare void @air.write_imageblock_slice_to_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(4), i1, <2 x i16>, <2 x i16>, <2 x i32>, i32, i1, i32)

!air.kernel = !{!0}
!air.sampler_states = !{!6}
!0 = !{ptr @kernel, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"src"}
!4 = !{i32 1, !"air.texture", !"air.location_index", i32 1, i32 1, !"air.write", !"air.arg_type_name", !"texture2d<float, write>", !"air.arg_name", !"out"}
!5 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint2", !"air.arg_name", !"gid"}
!6 = !{!"air.sampler_state", ptr addrspace(2) @__air_sampler_state}
"#;

        let metal = metal_row_for_compare(&[], infer_plan(ll), Some("Kernel"));
        let reason = incompatible_sampled_f32_imageblock_texture_golden(ll, &metal)
            .expect("sampled f32 imageblock texture golden");
        assert!(reason.contains("finite f32"), "{reason}");
        assert!(reason.contains("imageblock"), "{reason}");
    }

    #[test]
    fn integer_gather_imageblock_texture_golden_is_missing() {
        let ll = r#"
@__air_sampler_state = internal addrspace(2) constant [2 x i64] [i64 34901797601050697, i64 0], align 8

define void @kernel(ptr addrspace(1) readonly %src, ptr addrspace(1) %out, <2 x i16> %gid, <2 x i16> %tid) {
  %coord = tail call fast <2 x float> @air.convert.f.v2f32.u.v2i16(<2 x i16> %gid)
  %g = tail call { <4 x i16>, i8 } @air.gather_texture_2d.s.v4i16(ptr addrspace(1) readonly %src, ptr addrspace(2) @__air_sampler_state, <2 x float> %coord, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0, i32 0)
  %v = extractvalue { <4 x i16>, i8 } %g, 0
  %slot = tail call ptr addrspace(4) @air.imageblock_data(<2 x i16> %tid, i32 0, i16 0)
  store <4 x i16> %v, ptr addrspace(4) %slot
  tail call void @air.wg.barrier(i32 8, i32 1)
  tail call void @air.write_imageblock_slice_to_texture_2d.i16.v4i16(ptr addrspace(1) %out, ptr addrspace(4) %slot, i1 true, <2 x i16> zeroinitializer, <2 x i16> %gid, <2 x i16> %gid, i16 0, i1 false, i32 2)
  ret void
}

declare <2 x float> @air.convert.f.v2f32.u.v2i16(<2 x i16>)
declare { <4 x i16>, i8 } @air.gather_texture_2d.s.v4i16(ptr addrspace(1) readonly, ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32, i32)
declare ptr addrspace(4) @air.imageblock_data(<2 x i16>, i32, i16)
declare void @air.wg.barrier(i32, i32)
declare void @air.write_imageblock_slice_to_texture_2d.i16.v4i16(ptr addrspace(1), ptr addrspace(4), i1, <2 x i16>, <2 x i16>, <2 x i16>, i16, i1, i32)
attributes #0 = { "no-nans-fp-math"="true" "unsafe-fp-math"="true" }

!air.kernel = !{!0}
!air.sampler_states = !{!6}
!0 = !{ptr @kernel, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5, !7}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<short, sample>", !"air.arg_name", !"src"}
!4 = !{i32 1, !"air.texture", !"air.location_index", i32 1, i32 1, !"air.write", !"air.arg_type_name", !"texture2d<short, write>", !"air.arg_name", !"out"}
!5 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"ushort2", !"air.arg_name", !"gid"}
!6 = !{!"air.sampler_state", ptr addrspace(2) @__air_sampler_state}
!7 = !{i32 3, !"air.thread_position_in_threadgroup", !"air.arg_type_name", !"ushort2", !"air.arg_name", !"tid"}
"#;

        let metal = metal_row_for_compare(&[], infer_plan(ll), Some("Kernel"));
        let reason = incompatible_integer_gather_imageblock_texture_golden(ll, &metal)
            .expect("integer gather imageblock texture golden");
        assert!(reason.contains("signed integer"), "{reason}");
        assert!(reason.contains("imageblock"), "{reason}");
    }

    #[test]
    fn sampled_half_runtime_sampler_threadgroup_texture_golden_is_missing() {
        let ll = r#"
define void @kernel(ptr addrspace(1) %src, ptr addrspace(1) %out, ptr addrspace(3) %scratch, ptr addrspace(2) %sam, <2 x i16> %gid) #0 {
  %coord = tail call fast <2 x float> @air.convert.f.v2f32.u.v2i16(<2 x i16> %gid)
  %s = tail call { <4 x half>, i8 } @air.sample_texture_2d.v4f16(ptr addrspace(1) %src, ptr addrspace(2) %sam, <2 x float> %coord, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %v = extractvalue { <4 x half>, i8 } %s, 0
  store <4 x half> %v, ptr addrspace(3) %scratch
  tail call void @air.wg.barrier(i32 2, i32 1)
  %loaded = load <4 x half>, ptr addrspace(3) %scratch
  %wide = tail call <4 x float> @air.convert.f.v4f32.f.v4f16(<4 x half> %loaded)
  tail call void @air.write_texture_2d.i16.v4f32(ptr addrspace(1) %out, <2 x i16> %gid, <4 x float> %wide, i16 0, i32 2)
  ret void
}

declare <2 x float> @air.convert.f.v2f32.u.v2i16(<2 x i16>)
declare { <4 x half>, i8 } @air.sample_texture_2d.v4f16(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32)
declare <4 x float> @air.convert.f.v4f32.f.v4f16(<4 x half>)
declare void @air.wg.barrier(i32, i32)
declare void @air.write_texture_2d.i16.v4f32(ptr addrspace(1), <2 x i16>, <4 x float>, i16, i32)
attributes #0 = { "no-nans-fp-math"="true" "unsafe-fp-math"="true" }

!air.kernel = !{!0}
!0 = !{ptr @kernel, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5, !6, !7}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<half, sample>", !"air.arg_name", !"src"}
!4 = !{i32 1, !"air.texture", !"air.location_index", i32 1, i32 1, !"air.write", !"air.arg_type_name", !"texture2d<float, write>", !"air.arg_name", !"out"}
!5 = !{i32 2, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 3, !"air.arg_type_size", i32 8, !"air.arg_type_name", !"threadgroup half4*", !"air.arg_name", !"scratch"}
!6 = !{i32 3, !"air.sampler", !"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"sampler", !"air.arg_name", !"sam"}
!7 = !{i32 4, !"air.thread_position_in_grid", !"air.arg_type_name", !"ushort2", !"air.arg_name", !"gid"}
"#;

        let metal = metal_row_for_compare(&[], infer_plan(ll), Some("Kernel"));
        let reason = incompatible_sampled_half_linear_filter_texture_golden(ll, &metal)
            .expect("sampled half runtime sampler threadgroup texture golden");
        assert!(reason.contains("runtime sampler"), "{reason}");
    }

    #[test]
    fn sampled_half_cube_array_render_target_golden_is_missing() {
        let ll = r#"
@__air_sampler_state = internal addrspace(2) constant [2 x i64] [i64 34901797601036873, i64 0], align 8

define <4 x half> @frag(<3 x float> %normal, i32 %layer, ptr addrspace(1) %cube) #0 {
  %z = extractelement <3 x float> %normal, i64 2
  %nz = fneg fast float %z
  %coord = insertelement <3 x float> %normal, float %nz, i64 2
  %s = tail call { <4 x half>, i8 } @air.sample_texture_cube_array.v4f16(ptr addrspace(1) %cube, ptr addrspace(2) @__air_sampler_state, <3 x float> %coord, i32 %layer, i1 true, float 0.000000e+00, float 0.000000e+00, i32 0)
  %v = extractvalue { <4 x half>, i8 } %s, 0
  ret <4 x half> %v
}

declare { <4 x half>, i8 } @air.sample_texture_cube_array.v4f16(ptr addrspace(1), ptr addrspace(2), <3 x float>, i32, i1, float, float, i32)
attributes #0 = { "no-nans-fp-math"="true" "unsafe-fp-math"="true" }

!air.fragment = !{!0}
!air.sampler_states = !{!6}
!0 = !{ptr @frag, !1, !2}
!1 = !{!3}
!2 = !{!4, !5, !7}
!3 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"half4"}
!4 = !{i32 0, !"air.fragment_input", !"air.arg_type_name", !"float3", !"air.arg_name", !"normal"}
!5 = !{i32 1, !"air.fragment_input", !"air.flat", !"air.arg_type_name", !"uint", !"air.arg_name", !"layer"}
!6 = !{!"air.sampler_state", ptr addrspace(2) @__air_sampler_state}
!7 = !{i32 2, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texturecube_array<half, sample>", !"air.arg_name", !"cube"}
"#;

        let metal = metal_row_for_compare(&[], infer_plan(ll), Some("Fragment"));
        let reason = incompatible_sampled_half_cube_render_target_golden(ll, &metal)
            .expect("sampled half cube-array render-target golden");
        assert!(reason.contains("cube texture"), "{reason}");
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
    fn dependent_sampled_half_texture_lookup_golden_is_missing() {
        let ll = r#"
@__air_sampler_state = internal addrspace(2) constant [2 x i64] [i64 34901797601050624, i64 0], align 8

define void @kernel(ptr addrspace(1) %src, ptr addrspace(1) %depth, ptr addrspace(1) %out, <2 x i16> %gid) #0 {
  %coord = tail call fast <2 x float> @air.convert.f.v2f32.u.v2i16(<2 x i16> %gid)
  %g = tail call { <4 x half>, i8 } @air.gather_texture_2d.v4f16(ptr addrspace(1) %src, ptr addrspace(2) @__air_sampler_state, <2 x float> %coord, i1 true, <2 x i32> zeroinitializer, i32 1, i32 0)
  %gv = extractvalue { <4 x half>, i8 } %g, 0
  %r = tail call { <4 x half>, i8 } @air.read_texture_2d.i16.v4f16(ptr addrspace(1) %depth, ptr addrspace(2) null, <2 x i16> %gid, <2 x i16> zeroinitializer, i16 0, i32 0)
  %rv = extractvalue { <4 x half>, i8 } %r, 0
  %sum = fadd fast <4 x half> %gv, %rv
  %mins = tail call fast <4 x half> @air.fmin.v4f16(<4 x half> %sum, <4 x half> zeroinitializer)
  %cmp4 = fcmp fast olt <4 x half> %mins, %sum
  %idx4 = select <4 x i1> %cmp4, <4 x i16> <i16 2, i16 3, i16 poison, i16 poison>, <4 x i16> <i16 0, i16 1, i16 poison, i16 poison>
  %a = extractelement <4 x i16> %idx4, i64 0
  %b = extractelement <4 x i16> %idx4, i64 1
  %choose_b = fcmp fast ogt half 0xH3C00, 0xH3800
  %idx = select i1 %choose_b, i16 %a, i16 %b
  %lane = extractelement <4 x half> %sum, i16 %idx
  %gate = fcmp fast oge half %lane, 0xH3800
  %gate_h = tail call fast half @air.convert.f.f16.u.i1(i1 %gate)
  %gate_f = fpext half %gate_h to float
  %splat0 = insertelement <2 x float> poison, float %gate_f, i64 0
  %splat = shufflevector <2 x float> %splat0, <2 x float> poison, <2 x i32> zeroinitializer
  %offset = fadd fast <2 x float> %coord, splat (float 1.000000e+00)
  %lookup = tail call fast <2 x float> @air.mix.v2f32(<2 x float> %coord, <2 x float> %offset, <2 x float> %splat)
  %s = tail call { <4 x half>, i8 } @air.sample_texture_2d.v4f16(ptr addrspace(1) %src, ptr addrspace(2) @__air_sampler_state, <2 x float> %lookup, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %outv = extractvalue { <4 x half>, i8 } %s, 0
  tail call void @air.write_texture_2d.i16.v4f16(ptr addrspace(1) %out, <2 x i16> %gid, <4 x half> %outv, i16 0, i32 2)
  ret void
}

declare <2 x float> @air.convert.f.v2f32.u.v2i16(<2 x i16>)
declare { <4 x half>, i8 } @air.gather_texture_2d.v4f16(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i32, i32)
declare { <4 x half>, i8 } @air.read_texture_2d.i16.v4f16(ptr addrspace(1), ptr addrspace(2), <2 x i16>, <2 x i16>, i16, i32)
declare <4 x half> @air.fmin.v4f16(<4 x half>, <4 x half>)
declare half @air.convert.f.f16.u.i1(i1)
declare <2 x float> @air.mix.v2f32(<2 x float>, <2 x float>, <2 x float>)
declare { <4 x half>, i8 } @air.sample_texture_2d.v4f16(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32)
declare void @air.write_texture_2d.i16.v4f16(ptr addrspace(1), <2 x i16>, <4 x half>, i16, i32)
attributes #0 = { "no-nans-fp-math"="true" "unsafe-fp-math"="true" }

!air.kernel = !{!0}
!air.sampler_states = !{!6}
!0 = !{ptr @kernel, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5, !7}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<half, sample>", !"air.arg_name", !"src"}
!4 = !{i32 1, !"air.texture", !"air.location_index", i32 1, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<half, sample>", !"air.arg_name", !"depth"}
!5 = !{i32 2, !"air.texture", !"air.location_index", i32 2, i32 1, !"air.write", !"air.arg_type_name", !"texture2d<half, write>", !"air.arg_name", !"out"}
!6 = !{!"air.sampler_state", ptr addrspace(2) @__air_sampler_state}
!7 = !{i32 3, !"air.thread_position_in_grid", !"air.arg_type_name", !"ushort2", !"air.arg_name", !"gid"}
"#;

        let metal = metal_row_for_compare(&[], infer_plan(ll), Some("Kernel"));
        let reason = incompatible_dependent_sampled_half_lookup_golden(ll, &metal)
            .expect("dependent sampled half texture lookup golden");
        assert!(reason.contains("dependent texture lookup"), "{reason}");
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
            input_specialization: None,
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
            input_specialization: None,
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
        assert!(!execution_status_is_success(
            RunBackend::MoltenVk,
            "tolerance"
        ));
        assert!(!execution_status_is_success(RunBackend::MoltenVk, "smoke"));
        assert!(!execution_status_is_success(
            RunBackend::MoltenVk,
            "failure"
        ));
    }

    #[test]
    fn moltenvk_compare_policy_requires_exact_output() {
        let mut status = "tolerance".to_string();
        let mut error = None;
        enforce_backend_compare_policy(RunBackend::MoltenVk, &mut status, &mut error);
        assert_eq!(status, "failure");
        assert_eq!(
            error.as_deref(),
            Some("MoltenVK output differs from Metal; exact byte match required")
        );

        let mut status = "tolerance".to_string();
        let mut error = None;
        enforce_backend_compare_policy(RunBackend::Vulkan, &mut status, &mut error);
        assert_eq!(status, "tolerance");
        assert!(error.is_none());

        let mut status = "smoke".to_string();
        let mut error = Some("existing smoke reason".to_string());
        enforce_backend_compare_policy(RunBackend::MoltenVk, &mut status, &mut error);
        assert_eq!(status, "failure");
        assert_eq!(error.as_deref(), Some("existing smoke reason"));
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
            "Vulkan device does not support SPIR-V StencilExportEXT capability",
        ] {
            assert_eq!(candidate_execution_error_status(error), "quarantine");
        }
        assert_eq!(
            candidate_execution_error_status("create descriptor set: unsupported format"),
            "fallback"
        );
    }

    #[test]
    fn speculative_candidate_quarantines_are_skips() {
        assert_eq!(
            speculative_quarantine_outcome(Some("skipped golden")),
            ProcessOutcome::Skip
        );
        assert_eq!(speculative_quarantine_outcome(None), ProcessOutcome::Fail);
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
    fn forced_metal_rerun_does_not_skip_existing_compare_none() {
        let mut cfg = RunConfig::from_manifest(RunBackend::Metal);
        let mut metal = metal_row_for_compare(&[], infer_plan(""), Some("Kernel"));
        metal.compare = "none".into();

        assert!(should_skip_existing_metal_compare_none(&cfg, &metal));

        cfg.force = true;
        assert!(!should_skip_existing_metal_compare_none(&cfg, &metal));

        cfg.force = false;
        cfg.only_status = Some("ok".into());
        assert!(cfg.reruns_existing_backend_rows());
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
            input_specialization: None,
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
    fn compare_none_candidate_loop_guard_uses_recorded_launch_plan_facts() {
        let ll = r#"
define void @stats_like(i16 %simd_size, i16 %simdgroups, ptr addrspace(1) %out) {
entry:
  %simd_size_wide = zext i16 %simd_size to i32
  %simdgroups_wide = zext i16 %simdgroups to i32
  %simd_size_minus_one = add nsw i32 %simd_size_wide, -1
  %rounded = add nsw i32 %simd_size_minus_one, %simdgroups_wide
  %groups_wide = sdiv i32 %rounded, %simd_size_wide
  %groups = trunc i32 %groups_wide to i16
  %run = icmp ugt i16 %groups, 1
  br i1 %run, label %loop, label %exit
loop:
  %i = phi i16 [ %groups, %entry ], [ %next, %loop ]
  tail call void @air.wg.barrier(i32 2, i32 1)
  %i_wide = zext i16 %i to i32
  %rounded_next = add nsw i32 %simd_size_minus_one, %i_wide
  %next_wide = sdiv i32 %rounded_next, %simd_size_wide
  %next = trunc i32 %next_wide to i16
  %more = icmp ugt i16 %next, 1
  br i1 %more, label %loop, label %exit
exit:
  ret void
}

declare void @air.wg.barrier(i32, i32)

!air.kernel = !{!0}
!0 = !{ptr @stats_like, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.threads_per_simdgroup", !"air.arg_type_name", !"ushort", !"air.arg_name", !"simd_size"}
!4 = !{i32 1, !"air.simdgroups_per_threadgroup", !"air.arg_type_name", !"ushort", !"air.arg_name", !"simdgroups"}
!5 = !{i32 2, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
"#;
        match crate::loop_budget::classify_and_instrument(ll, "stats_like") {
            crate::loop_budget::GuardPlan::Quarantine(reason) => {
                assert!(reason.contains("air.wg.barrier"), "{reason}");
            }
            other => panic!("expected facts-free loop guard to quarantine, got {other:?}"),
        }

        let metal = MetalRow {
            air_sha256: "x".into(),
            shard: None,
            label: "x".into(),
            status: "ok".into(),
            backend: "metal".into(),
            seed_profile: SEED_PROFILE.into(),
            plan_version: PLAN_VERSION,
            plan: HarnessPlan {
                buffers: vec![PlanBuffer {
                    index: 0,
                    len: 4,
                    role: "Output".into(),
                    seed_tag: 1,
                    seed_mode: SEED_MODE_DETERMINISTIC.into(),
                    seed_layout: Vec::new(),
                    seed_stride: None,
                }],
                textures: Vec::new(),
                output: PlanOutput {
                    kind: "buffer".into(),
                    index: 0,
                    format: "RawBytes".into(),
                    len: Some(4),
                    w: None,
                    h: None,
                    d: None,
                },
                dispatch_grid: [64, 1, 1],
                dispatch_tg: [64, 1, 1],
            },
            input_sha256: None,
            output_sha256: Some("gold".into()),
            output_b64: None,
            spv_sha256: None,
            compare: "none".into(),
            fc_specialization: None,
            fc_values: None,
            input_specialization: None,
            stage: Some("Kernel".into()),
            entry: Some("stats_like".into()),
            error: None,
        };

        match candidate_ll_for_metal_compare(ll, "stats_like", &metal).expect("candidate ll") {
            Cow::Borrowed(_) => {}
            Cow::Owned(text) => {
                panic!(
                    "expected exact launch facts to prove loop-free, got instrumentation:\n{text}"
                )
            }
        }
        assert!(incompatible_compare_none_loop_guard_golden(ll, "stats_like", &metal).is_none());
    }

    #[test]
    fn compare_none_candidate_loop_guard_uses_generated_fc_values() {
        let ll = r#"
@_Z21threadsPerThreadgroup.MTL_FC_INIT_11_j = internal unnamed_addr addrspace(2) externally_initialized constant i32 undef, section "air.fc_initializer", align 4
@_ZL21threadsPerThreadgroup = internal unnamed_addr addrspace(2) global i32 undef, align 4

define internal void @_GLOBAL__sub_I_test() section "air.static_init" {
  %t = load i32, ptr addrspace(2) @_Z21threadsPerThreadgroup.MTL_FC_INIT_11_j, align 4
  store i32 %t, ptr addrspace(2) @_ZL21threadsPerThreadgroup, align 4
  ret void
}

define void @fc_stride_like(i32 %lid) {
entry:
  %active = icmp ult i32 %lid, 48
  br i1 %active, label %preheader, label %exit
preheader:
  %stride = load i32, ptr addrspace(2) @_ZL21threadsPerThreadgroup, align 4
  br label %loop
loop:
  %i = phi i32 [ %lid, %preheader ], [ %next, %loop ]
  tail call void @air.wg.barrier(i32 2, i32 1)
  %next = add i32 %i, %stride
  %more = icmp ult i32 %next, 48
  br i1 %more, label %loop, label %exit
exit:
  ret void
}

declare void @air.wg.barrier(i32, i32)

!air.kernel = !{!0}
!air.function_constants = !{!5}
!0 = !{ptr @fc_stride_like, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.thread_position_in_threadgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"lid"}
!5 = !{ptr addrspace(2) @_Z21threadsPerThreadgroup.MTL_FC_INIT_11_j, !"uint", !"threadsPerThreadgroup", i32 11, i1 true}
"#;
        match crate::loop_budget::classify_and_instrument(ll, "fc_stride_like") {
            crate::loop_budget::GuardPlan::Quarantine(reason) => {
                assert!(reason.contains("air.wg.barrier"), "{reason}");
            }
            other => panic!("expected facts-free loop guard to quarantine, got {other:?}"),
        }

        let metal = MetalRow {
            air_sha256: "x".into(),
            shard: None,
            label: "x".into(),
            status: "ok".into(),
            backend: "metal".into(),
            seed_profile: SEED_PROFILE.into(),
            plan_version: PLAN_VERSION,
            plan: HarnessPlan {
                buffers: Vec::new(),
                textures: Vec::new(),
                output: PlanOutput {
                    kind: "buffer".into(),
                    index: 0,
                    format: "RawBytes".into(),
                    len: Some(4),
                    w: None,
                    h: None,
                    d: None,
                },
                dispatch_grid: [64, 1, 1],
                dispatch_tg: [64, 1, 1],
            },
            input_sha256: None,
            output_sha256: Some("gold".into()),
            output_b64: None,
            spv_sha256: None,
            compare: "none".into(),
            fc_specialization: None,
            fc_values: None,
            input_specialization: None,
            stage: Some("Kernel".into()),
            entry: Some("fc_stride_like".into()),
            error: None,
        };

        match candidate_ll_for_metal_compare(ll, "fc_stride_like", &metal).expect("candidate ll") {
            Cow::Borrowed(_) => {}
            Cow::Owned(text) => {
                panic!(
                    "expected generated FC facts to prove loop-free, got instrumentation:\n{text}"
                )
            }
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
            input_specialization: None,
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
    fn compare_none_simdgroup_matrix_smoke_golden_requires_rebank() {
        let ll = r#"
@tile = addrspace(3) global [64 x bfloat] undef, align 2

define void @kernel(<64 x float> %a, <64 x float> %b, <64 x float> %c) {
entry:
  call void @air.wg.barrier(i32 2, i32 1)
  %m = call <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f32.v64f32(<64 x float> %a, <64 x float> %b, <64 x float> %c)
  %x = extractelement <64 x float> %m, i64 0
  %bf = fptrunc float %x to bfloat
  store bfloat %bf, ptr addrspace(3) @tile, align 2
  ret void
}

declare void @air.wg.barrier(i32, i32)
declare <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f32.v64f32(<64 x float>, <64 x float>, <64 x float>)
"#;
        let mut metal = metal_row_for_compare(&[], infer_plan(ll), Some("Kernel"));
        metal.compare = "none".into();

        let reason = incompatible_compare_none_simdgroup_matrix_smoke_golden(ll, &metal)
            .expect("compare=none bfloat matrix smoke row");
        assert!(reason.contains("compare=none"), "{reason}");
        assert!(reason.contains("simdgroup-matrix"), "{reason}");
    }

    #[test]
    fn compare_none_half_simdgroup_matrix_smoke_golden_requires_rebank() {
        let ll = r#"
@tile = addrspace(3) global [64 x half] undef, align 2

define void @kernel(<64 x float> %a, <64 x float> %b, <64 x float> %c) {
entry:
  call void @air.wg.barrier(i32 2, i32 1)
  %m = call <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f32.v64f32(<64 x float> %a, <64 x float> %b, <64 x float> %c)
  %x = extractelement <64 x float> %m, i64 0
  %h = fptrunc float %x to half
  store half %h, ptr addrspace(3) @tile, align 2
  ret void
}

declare void @air.wg.barrier(i32, i32)
declare <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f32.v64f32(<64 x float>, <64 x float>, <64 x float>)
"#;
        let mut metal = metal_row_for_compare(&[], infer_plan(ll), Some("Kernel"));
        metal.compare = "none".into();

        let reason = incompatible_compare_none_simdgroup_matrix_smoke_golden(ll, &metal)
            .expect("compare=none half matrix smoke row");
        assert!(reason.contains("compare=none"), "{reason}");
        assert!(reason.contains("simdgroup-matrix"), "{reason}");
    }

    #[test]
    fn compare_none_float_simdgroup_matrix_smoke_golden_requires_rebank() {
        let ll = r#"
@tile = addrspace(3) global [64 x float] undef, align 4

define void @kernel(<64 x float> %a, <64 x float> %b, <64 x float> %c) {
entry:
  call void @air.wg.barrier(i32 2, i32 1)
  %m = call <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f32.v64f32(<64 x float> %a, <64 x float> %b, <64 x float> %c)
  store <64 x float> %m, ptr addrspace(3) @tile, align 4
  ret void
}

declare void @air.wg.barrier(i32, i32)
declare <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f32.v64f32(<64 x float>, <64 x float>, <64 x float>)
"#;
        let mut metal = metal_row_for_compare(&[], infer_plan(ll), Some("Kernel"));
        metal.compare = "none".into();

        let reason = incompatible_compare_none_simdgroup_matrix_smoke_golden(ll, &metal)
            .expect("compare=none float matrix smoke row");
        assert!(reason.contains("compare=none"), "{reason}");
        assert!(reason.contains("simdgroup-matrix"), "{reason}");
    }

    #[test]
    fn compare_none_raytracing_smoke_golden_requires_rebank() {
        let ll = r#"
define void @kernel(i32 %tid, ptr addrspace(1) %accel, ptr addrspace(1) %table) {
entry:
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @kernel, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid"}
!4 = !{i32 1, !"air.instance_acceleration_structure", !"air.location_index", i32 2, i32 1, !"air.read", !"air.arg_type_name", !"acceleration_structure<instancing, primitive_motion>", !"air.arg_name", !"raytracing_accel"}
!5 = !{i32 2, !"air.visible_function_table", !"air.location_index", i32 3, i32 1, !"air.read", !"air.arg_type_name", !"visible_function_table", !"air.arg_name", !"raytracing_table"}
"#;
        let mut metal = metal_row_for_compare(&[], infer_plan(ll), Some("Kernel"));
        metal.compare = "none".into();

        let reason = incompatible_compare_none_raytracing_smoke_golden(ll, &metal)
            .expect("compare=none raytracing smoke row");
        assert!(reason.contains("compare=none"), "{reason}");
        assert!(reason.contains("AIR raytracing"), "{reason}");
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
    fn compile_missing_patch_control_point_translate_error_is_quarantine() {
        assert!(compile_missing_unsupported_translate_error(
            "native emitter: unsupported Metal patch control point function; tessellation patch inputs are not yet lowered"
        ));
    }

    #[test]
    fn compile_missing_memcpy_validation_error_is_quarantine() {
        let ll = r#"
define <4 x float> @fragment_main(
    ptr readonly byval([3 x ptr addrspace(1)]) captures(none) %textures) {
  %dst = alloca [3 x ptr addrspace(1)], align 8
  call void @llvm.memcpy.p0.p0.i64(ptr %dst, ptr %textures, i64 24, i1 false)
  ret <4 x float> zeroinitializer
}

declare void @llvm.memcpy.p0.p0.i64(ptr, ptr, i64, i1)

!air.fragment = !{!1}
!1 = !{ptr @fragment_main, !2, !3}
!2 = !{}
!3 = !{!4}
!4 = !{i32 0, !"air.texture", !"air.location_index", i32 5, i32 3, !"air.sample",
       !"air.arg_type_name", !"array<texture2d<half, sample>, 3>"}
"#;
        let memcpy_err = "spirv-val failed:\nerror: line 618: OpFunctionCall Argument <id> \
'517[%517]'s type does not match Function <id> '3[%_ptr_Private_uchar]'s parameter type.\n\
  %518 = OpFunctionCall %void %llvm_memcpy_p0_p0_i64 %517 %263 %ulong_24 %false";
        let cfg_err = "spirv-val failed: error: line 526: Block '1818[%1818]' is already a merge block for another header";

        assert!(compile_missing_unsupported_validation_error(ll, memcpy_err));
        assert!(compile_missing_unsupported_validation_error(ll, cfg_err));
        assert!(!compile_missing_unsupported_validation_error(
            "", memcpy_err
        ));
        assert!(!compile_missing_unsupported_validation_error(
            ll,
            "spirv-val failed: error: line 1: unsupported binary version"
        ));
    }

    #[test]
    fn compile_missing_invalid_smoke_validation_precheck_is_quarantine() {
        assert!(compile_missing_invalid_smoke_is_validation_quarantine(
            "metal golden uses deterministic buffer 4 for AIR control/atomic counter input now seeded bounded_control; rebank Metal row"
        ));
        assert!(compile_missing_invalid_smoke_is_validation_quarantine(
            "metal golden specializes an AIR function constant whose static initializer selects among private constant-table pointers; the current product path validates before the validation FC-specialization helper can prune that unspecialized pointer phi, so this is not a comparable Vulkan oracle yet"
        ));
        assert!(!compile_missing_invalid_smoke_is_validation_quarantine(
            "metal status=fallback"
        ));
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
            input_specialization: None,
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

        metal.fc_specialization = Some(FC_SPECIALIZATION_VALUES.into());
        metal.fc_values = Some(vec![FunctionConstantValueJson { index: 1, value: 1 }]);
        let reason = incompatible_static_resource_plan_golden(ll, &metal)
            .expect("stale static resource plan with FC values");
        assert!(reason.contains("static-location plan"), "{reason}");
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
