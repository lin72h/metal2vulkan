use crate::corpus_run::{execution_failure_signature, execution_status_is_success, RunBackend};
use serde::Deserialize;
use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum LedgerKind {
    Translate,
    Metal,
    Vulkan,
    MoltenVk,
}

impl LedgerKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            LedgerKind::Translate => "translate",
            LedgerKind::Metal => "metal",
            LedgerKind::Vulkan => "vulkan",
            LedgerKind::MoltenVk => "moltenvk",
        }
    }

    pub const fn ledger_file_name(self) -> &'static str {
        match self {
            LedgerKind::Translate => "metal2vulkan-ledger.jsonl",
            LedgerKind::Metal => "metal2vulkan-ledger-metal.jsonl",
            LedgerKind::Vulkan => "metal2vulkan-ledger-vulkan.jsonl",
            LedgerKind::MoltenVk => "metal2vulkan-ledger-moltenvk.jsonl",
        }
    }

    pub const fn runner_bin(self) -> Option<&'static str> {
        match self {
            LedgerKind::Translate => None,
            LedgerKind::Metal => Some("corpus-run-metal"),
            LedgerKind::Vulkan => Some("corpus-run-vulkan"),
            LedgerKind::MoltenVk => Some("corpus-run-moltenvk"),
        }
    }

    pub const fn all() -> [Self; 4] {
        [
            LedgerKind::Translate,
            LedgerKind::Metal,
            LedgerKind::Vulkan,
            LedgerKind::MoltenVk,
        ]
    }
}

pub fn parse_ledger_kind(s: &str) -> Option<LedgerKind> {
    match s {
        "translate" | "mint" => Some(LedgerKind::Translate),
        "metal" => Some(LedgerKind::Metal),
        "vulkan" => Some(LedgerKind::Vulkan),
        "moltenvk" => Some(LedgerKind::MoltenVk),
        _ => None,
    }
}

#[derive(Debug, Clone)]
pub struct TriageRow {
    pub kind: LedgerKind,
    pub air_sha256: String,
    pub status: String,
    pub label: String,
    pub shard: Option<String>,
    pub spv_sha256: Option<String>,
    pub output_sha256: Option<String>,
    pub golden_output_sha256: Option<String>,
    pub error: Option<String>,
    pub has_tolerance: bool,
    pub signature: String,
}

impl TriageRow {
    pub fn is_success(&self) -> bool {
        match self.kind {
            LedgerKind::Translate => self.status == "ok",
            LedgerKind::Metal => execution_status_is_success(RunBackend::Metal, &self.status),
            LedgerKind::Vulkan => execution_status_is_success(RunBackend::Vulkan, &self.status),
            LedgerKind::MoltenVk => execution_status_is_success(RunBackend::MoltenVk, &self.status),
        }
    }

    pub fn matches_text(&self, needle: &str) -> bool {
        let needle = needle.to_ascii_lowercase();
        self.air_sha256.contains(&needle)
            || self.status.to_ascii_lowercase().contains(&needle)
            || self.label.to_ascii_lowercase().contains(&needle)
            || self.signature.to_ascii_lowercase().contains(&needle)
            || self
                .error
                .as_deref()
                .unwrap_or("")
                .to_ascii_lowercase()
                .contains(&needle)
    }
}

#[derive(Debug, Deserialize)]
struct RawRow {
    #[serde(default)]
    air_sha256: String,
    #[serde(default)]
    status: String,
    #[serde(default)]
    label: String,
    #[serde(default)]
    shard: Option<String>,
    #[serde(default)]
    spv_sha256: Option<String>,
    #[serde(default)]
    output_sha256: Option<String>,
    #[serde(default)]
    golden_output_sha256: Option<String>,
    #[serde(default)]
    error: Option<String>,
    #[serde(default)]
    tolerance: Option<serde_json::Value>,
}

pub fn ledger_path(ledger_dir: &Path, kind: LedgerKind) -> PathBuf {
    ledger_dir.join(kind.ledger_file_name())
}

pub fn load_rows(ledger_dir: &Path, kind: LedgerKind) -> Result<Vec<TriageRow>, String> {
    let path = ledger_path(ledger_dir, kind);
    let file = match File::open(&path) {
        Ok(file) => file,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(format!("open {}: {e}", path.display())),
    };

    let mut by_hash = HashMap::new();
    for (i, line) in BufReader::new(file).lines().enumerate() {
        let line = line.map_err(|e| format!("read {}:{}: {e}", path.display(), i + 1))?;
        let t = line.trim();
        if t.is_empty() || t.starts_with('#') {
            continue;
        }
        let raw: RawRow = serde_json::from_str(t)
            .map_err(|e| format!("parse {}:{}: {e}", path.display(), i + 1))?;
        if raw.air_sha256.is_empty() {
            continue;
        }
        let air_sha256 = raw.air_sha256.to_ascii_lowercase();
        let signature =
            execution_failure_signature(&raw.status, raw.error.as_deref(), raw.tolerance.is_some());
        by_hash.insert(
            air_sha256.clone(),
            TriageRow {
                kind,
                air_sha256,
                status: raw.status,
                label: raw.label,
                shard: raw.shard,
                spv_sha256: raw.spv_sha256,
                output_sha256: raw.output_sha256,
                golden_output_sha256: raw.golden_output_sha256,
                error: raw.error,
                has_tolerance: raw.tolerance.is_some(),
                signature,
            },
        );
    }
    let mut rows: Vec<_> = by_hash.into_values().collect();
    rows.sort_by(|a, b| {
        a.kind
            .cmp(&b.kind)
            .then(a.status.cmp(&b.status))
            .then(a.signature.cmp(&b.signature))
            .then(a.label.cmp(&b.label))
            .then(a.air_sha256.cmp(&b.air_sha256))
    });
    Ok(rows)
}

pub fn status_counts(rows: &[TriageRow]) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();
    for row in rows {
        *counts.entry(row.status.clone()).or_insert(0) += 1;
    }
    counts
}

pub fn failure_buckets(rows: &[TriageRow]) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();
    for row in rows.iter().filter(|row| !row.is_success()) {
        *counts.entry(row.signature.clone()).or_insert(0) += 1;
    }
    counts
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tolerance_is_success_for_both_candidate_backends() {
        let row = TriageRow {
            kind: LedgerKind::Vulkan,
            air_sha256: "a".repeat(64),
            status: "tolerance".into(),
            label: String::new(),
            shard: None,
            spv_sha256: None,
            output_sha256: None,
            golden_output_sha256: None,
            error: None,
            has_tolerance: true,
            signature: "within candidate tolerance".into(),
        };
        assert!(row.is_success());

        let moltenvk = TriageRow {
            kind: LedgerKind::MoltenVk,
            ..row
        };
        assert!(moltenvk.is_success());
    }

    #[test]
    fn signature_groups_vulkan_validation_noise() {
        let sig = execution_failure_signature(
            "fallback",
            Some(
                "vulkan execute panicked: create vertex validation pipeline: a validation error occurred",
            ),
            false,
        );
        assert_eq!(sig, "create vertex validation pipeline");

        let sig = execution_failure_signature(
            "fallback",
            Some(
                "vulkan execute panicked: create compute pipeline: a non-validation error occurred",
            ),
            false,
        );
        assert_eq!(sig, "create compute pipeline");
    }

    #[test]
    fn load_rows_keeps_last_row_for_hash() {
        let dir = crate::scratch_dir_for("triage-dedupe");
        let path = ledger_path(&dir, LedgerKind::MoltenVk);
        let hash = "abcd".repeat(16);
        std::fs::write(
            &path,
            format!(
                "{{\"air_sha256\":\"{hash}\",\"status\":\"fallback\",\"label\":\"old\",\"error\":\"vulkan execute panicked: create compute pipeline: a non-validation error occurred\"}}\n\
                 {{\"air_sha256\":\"{hash}\",\"status\":\"ok\",\"label\":\"new\"}}\n"
            ),
        )
        .expect("write test ledger");

        let rows = load_rows(&dir, LedgerKind::MoltenVk).expect("load rows");
        let _ = std::fs::remove_dir_all(&dir);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].status, "ok");
        assert_eq!(rows[0].label, "new");
    }
}
