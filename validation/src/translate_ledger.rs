use crate::air::{stage_from_ll, stage_name};
use crate::corpus_shards::{SourceData, SourceRef};
use crate::hash::sha256_bytes;
use crate::jsonl::to_sorted_json_string;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::thread;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranslateLedgerRow {
    pub air_sha256: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shard: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub spv_sha256: Option<String>,
    pub status: String,
    #[serde(default)]
    pub stage: String,
    #[serde(default)]
    pub label: String,
    #[serde(default)]
    pub kind: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TranslateFailureKind {
    LoadSource,
    TempDir,
    StageDetect,
    Translate,
    EmptySpirv,
}

#[derive(Debug, Clone)]
pub struct TranslateFailure {
    pub kind: TranslateFailureKind,
    pub message: String,
}

#[derive(Debug, Clone)]
pub struct TranslateReport {
    pub status: String,
    pub stage_name: &'static str,
    pub spv_sha256: Option<String>,
    pub spv_len: Option<usize>,
    pub failure: Option<TranslateFailure>,
}

struct TempDir {
    path: PathBuf,
}

impl TempDir {
    fn create(prefix: &str, air_sha256: &str) -> Result<Self, String> {
        let path = std::env::temp_dir().join(format!(
            "{prefix}-{}-{}",
            std::process::id(),
            &air_sha256[..16.min(air_sha256.len())]
        ));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path)
            .map_err(|e| format!("create temp dir {}: {e}", path.display()))?;
        Ok(Self { path })
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

pub fn default_workers() -> usize {
    thread::available_parallelism()
        .map(|n| n.get().saturating_mul(2))
        .unwrap_or(8)
        .max(1)
}

pub fn unique_sources_by_hash(sources: Vec<SourceRef>) -> HashMap<String, SourceRef> {
    let mut by_hash = HashMap::new();
    for source in sources {
        match by_hash.get(&source.air_sha256) {
            None => {
                by_hash.insert(source.air_sha256.clone(), source);
            }
            Some(previous) if source.label < previous.label => {
                by_hash.insert(source.air_sha256.clone(), source);
            }
            _ => {}
        }
    }
    by_hash
}

pub fn translate_source_data(source: &SourceData, tmp_prefix: &str) -> TranslateReport {
    if source.air_ll.is_empty() {
        return fallback_report(
            TranslateFailureKind::StageDetect,
            "source has empty air_ll".to_string(),
        );
    }
    let stage = stage_from_ll(&source.air_ll);
    let stage_name = stage_name(stage);
    let tmp = match TempDir::create(tmp_prefix, &source.air_sha256) {
        Ok(tmp) => tmp,
        Err(message) => {
            return TranslateReport {
                stage_name,
                ..fallback_report(TranslateFailureKind::TempDir, message)
            };
        }
    };
    let stage: metal2vulkan::passes::Stage = stage.into();
    match metal2vulkan::translate_sanitized_native(&source.air_ll, stage, &tmp.path) {
        Ok(spv) if !spv.is_empty() => TranslateReport {
            status: "ok".to_string(),
            stage_name,
            spv_sha256: Some(sha256_bytes(&spv)),
            spv_len: Some(spv.len()),
            failure: None,
        },
        Ok(_) => TranslateReport {
            stage_name,
            ..fallback_report(
                TranslateFailureKind::EmptySpirv,
                "translate returned empty module bytes (no error string)".to_string(),
            )
        },
        Err(error) => TranslateReport {
            stage_name,
            ..fallback_report(TranslateFailureKind::Translate, error)
        },
    }
}

pub fn translate_source_ref(
    source: &SourceRef,
    corpus_root: &Path,
    tmp_prefix: &str,
) -> TranslateLedgerRow {
    let report = match source.load(corpus_root) {
        Ok(loaded) => translate_source_data(&loaded, tmp_prefix),
        Err(error) => fallback_report(TranslateFailureKind::LoadSource, error),
    };
    row_from_source_ref(source, &report)
}

pub fn translate_all(
    sources: &[SourceRef],
    corpus_root: &Path,
    workers: usize,
    quiet: bool,
    verb: &str,
    tmp_prefix: &str,
) -> Vec<TranslateLedgerRow> {
    let n = sources.len();
    if n == 0 {
        return Vec::new();
    }
    let jobs = workers.min(n).max(1);
    eprintln!("# translate workers={jobs} sources={n}");

    let done = AtomicUsize::new(0);
    let rows: Mutex<Vec<TranslateLedgerRow>> = Mutex::new(Vec::with_capacity(n));
    let chunk_size = n.div_ceil(jobs);
    thread::scope(|scope| {
        for chunk in sources.chunks(chunk_size) {
            let done = &done;
            let rows = &rows;
            scope.spawn(move || {
                for source in chunk {
                    let row = translate_source_ref(source, corpus_root, tmp_prefix);
                    let i = done.fetch_add(1, Ordering::Relaxed) + 1;
                    if quiet {
                        if i == 1 || i == n || i.is_multiple_of(25) {
                            eprintln!("  [{i}/{n}] ...");
                        }
                    } else {
                        eprintln!(
                            "  [{i}/{n}] {verb} {:<10} {}  {}",
                            row.status,
                            source.label,
                            &source.air_sha256[..12.min(source.air_sha256.len())]
                        );
                    }
                    rows.lock().unwrap().push(row);
                }
            });
        }
    });

    let mut out = rows.into_inner().unwrap();
    sort_rows(&mut out);
    out
}

pub fn load_ledger_keys(path: &Path) -> Result<HashSet<String>, String> {
    let mut keys = HashSet::new();
    let file = match File::open(path) {
        Ok(file) => file,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(keys),
        Err(e) => return Err(format!("open ledger {}: {e}", path.display())),
    };
    for line in BufReader::new(file).lines() {
        let line = line.map_err(|e| format!("read ledger: {e}"))?;
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let row: TranslateLedgerRow = serde_json::from_str(trimmed)
            .map_err(|e| format!("parse ledger line: {e}: {trimmed}"))?;
        keys.insert(row.air_sha256);
    }
    Ok(keys)
}

pub fn load_ledger(path: &Path) -> Result<(HashMap<String, TranslateLedgerRow>, usize), String> {
    let mut by_hash = HashMap::new();
    let mut duplicates = 0usize;
    let file = match File::open(path) {
        Ok(file) => file,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok((by_hash, 0)),
        Err(e) => return Err(format!("open ledger {}: {e}", path.display())),
    };
    for line in BufReader::new(file).lines() {
        let line = line.map_err(|e| format!("read ledger: {e}"))?;
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let row: TranslateLedgerRow = serde_json::from_str(trimmed)
            .map_err(|e| format!("parse ledger line: {e}: {trimmed}"))?;
        if by_hash.contains_key(&row.air_sha256) {
            duplicates += 1;
        }
        by_hash.insert(row.air_sha256.clone(), row);
    }
    Ok((by_hash, duplicates))
}

pub fn load_ledger_row(path: &Path, air_sha256: &str) -> Option<TranslateLedgerRow> {
    let file = File::open(path).ok()?;
    let mut found = None;
    for line in BufReader::new(file).lines().map_while(Result::ok) {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let Ok(row) = serde_json::from_str::<TranslateLedgerRow>(trimmed) else {
            continue;
        };
        if row.air_sha256.eq_ignore_ascii_case(air_sha256) {
            found = Some(row);
        }
    }
    found
}

pub fn append_ledger_rows(path: &Path, rows: &[TranslateLedgerRow]) -> Result<(), String> {
    if rows.is_empty() {
        return Ok(());
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("mkdir {}: {e}", parent.display()))?;
    }

    let need_header = !path.is_file() || path.metadata().map(|m| m.len() == 0).unwrap_or(true);
    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|e| format!("open {}: {e}", path.display()))?;

    if need_header {
        write_ledger_header(&mut file)?;
    }

    let mut unique = BTreeMap::new();
    for row in rows {
        unique.insert(row.air_sha256.clone(), row);
    }
    for row in unique.values() {
        let line = to_sorted_json_string(row).map_err(|e| format!("json: {e}"))?;
        writeln!(file, "{line}").map_err(|e| format!("write ledger: {e}"))?;
    }
    file.flush().map_err(|e| format!("flush ledger: {e}"))
}

pub fn write_ledger(
    path: &Path,
    by_hash: &HashMap<String, TranslateLedgerRow>,
) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("mkdir {}: {e}", parent.display()))?;
    }

    let mut rows: Vec<_> = by_hash.values().cloned().collect();
    sort_rows(&mut rows);

    let tmp = path.with_extension("jsonl.tmp");
    {
        let mut file = File::create(&tmp).map_err(|e| format!("create {}: {e}", tmp.display()))?;
        write_ledger_header(&mut file)?;
        for row in &rows {
            let line = to_sorted_json_string(row).map_err(|e| format!("json: {e}"))?;
            writeln!(file, "{line}").map_err(|e| format!("write: {e}"))?;
        }
        file.flush().map_err(|e| format!("flush: {e}"))?;
    }
    fs::rename(&tmp, path)
        .map_err(|e| format!("rename {} -> {}: {e}", tmp.display(), path.display()))
}

fn row_from_source_ref(source: &SourceRef, report: &TranslateReport) -> TranslateLedgerRow {
    TranslateLedgerRow {
        air_sha256: source.air_sha256.clone(),
        shard: source.shard.clone(),
        spv_sha256: report.spv_sha256.clone(),
        status: report.status.clone(),
        stage: "auto".to_string(),
        label: source.label.clone(),
        kind: source.kind.clone(),
    }
}

fn fallback_report(kind: TranslateFailureKind, message: String) -> TranslateReport {
    TranslateReport {
        status: "fallback".to_string(),
        stage_name: "auto",
        spv_sha256: None,
        spv_len: None,
        failure: Some(TranslateFailure { kind, message }),
    }
}

fn sort_rows(rows: &mut [TranslateLedgerRow]) {
    rows.sort_by(|a, b| {
        a.kind
            .cmp(&b.kind)
            .then_with(|| a.label.cmp(&b.label))
            .then_with(|| a.air_sha256.cmp(&b.air_sha256))
    });
}

fn write_ledger_header(file: &mut File) -> Result<(), String> {
    writeln!(
        file,
        "# metal2vulkan ledger - hashes only, no shader bodies"
    )
    .map_err(|e| format!("write ledger header: {e}"))?;
    writeln!(
        file,
        "# schema: air_sha256 (unique), shard?, spv_sha256?, status=ok|fallback|timeout, stage, label, kind"
    )
    .map_err(|e| format!("write ledger schema: {e}"))
}
