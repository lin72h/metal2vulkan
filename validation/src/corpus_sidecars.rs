//! Committed corpus sidecars keyed by **AIR content SHA-256** (not short monorepo hashes).
//!
//! | File | Role |
//! |---|---|
//! | [`tolerances.jsonl`](../../corpus/tolerances.jsonl) | Per-AIR numeric compare tolerances |
//! | [`broken.jsonl`](../../corpus/broken.jsonl) | Cases that must not run / are known non-applicable |
//!
//! Both are **git-tracked JSONL** (one object per line) and contain **no shader bodies** — only
//! fingerprints + metadata. Private AIR stays under gitignored `corpus/local/`.
//!
//! Blank lines and `#` comment lines are ignored.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

// --- tolerances.jsonl ----------------------------------------------------------------------------

/// One banked numeric tolerance for a single AIR fingerprint (one JSONL row).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToleranceEntry {
    /// Lowercase hex SHA-256 of the AIR source bytes (`.ll` text or `.air` blob).
    pub air_sha256: String,
    /// Optional non-proprietary label (fixture basename, short tag).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    pub tolerance: ToleranceSpec,
}

/// Wire form of a tolerance (mirrors monorepo kinds; owned strings instead of `'static`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind")]
pub enum ToleranceSpec {
    Exact,
    Abs {
        max_abs: f32,
        reason: String,
    },
    Ulp {
        max_ulp: u32,
        reason: String,
    },
    RawF16Ulp {
        max_ulp: u32,
        reason: String,
    },
    RawU8Ulp {
        max_ulp: u32,
        reason: String,
    },
    AbsAndUlp {
        max_abs: f32,
        max_ulp: u32,
        reason: String,
    },
}

impl ToleranceSpec {
    pub fn reason(&self) -> Option<&str> {
        match self {
            ToleranceSpec::Exact => None,
            ToleranceSpec::Abs { reason, .. }
            | ToleranceSpec::Ulp { reason, .. }
            | ToleranceSpec::RawF16Ulp { reason, .. }
            | ToleranceSpec::RawU8Ulp { reason, .. }
            | ToleranceSpec::AbsAndUlp { reason, .. } => Some(reason.as_str()),
        }
    }

    /// Map into the crate's `'static` [`crate::Tolerance`] used by the predicate/oracle path.
    /// Reasons are leaked once (same approach as the monorepo corpus loader) so callers can keep
    /// using `Tolerance` without lifetime plumbing.
    pub fn to_tolerance(&self) -> crate::Tolerance {
        match self {
            ToleranceSpec::Exact => crate::Tolerance::Exact,
            ToleranceSpec::Abs { max_abs, reason } => crate::Tolerance::Abs {
                max_abs: *max_abs,
                reason: leak_reason(reason),
            },
            ToleranceSpec::Ulp { max_ulp, reason } => crate::Tolerance::Ulp {
                max_ulp: *max_ulp,
                reason: leak_reason(reason),
            },
            ToleranceSpec::RawF16Ulp { max_ulp, reason } => crate::Tolerance::RawF16Ulp {
                max_ulp: *max_ulp,
                reason: leak_reason(reason),
            },
            ToleranceSpec::RawU8Ulp { max_ulp, reason } => crate::Tolerance::RawU8Ulp {
                max_ulp: *max_ulp,
                reason: leak_reason(reason),
            },
            ToleranceSpec::AbsAndUlp {
                max_abs,
                max_ulp,
                reason,
            } => crate::Tolerance::AbsAndUlp {
                max_abs: *max_abs,
                max_ulp: *max_ulp,
                reason: leak_reason(reason),
            },
        }
    }
}

/// In-memory index of `tolerances.jsonl` (keyed by lowercase `air_sha256`).
#[derive(Debug, Clone, Default)]
pub struct TolerancesIndex {
    pub entries: HashMap<String, ToleranceEntry>,
}

// --- broken.jsonl --------------------------------------------------------------------------------

/// Why a case is excluded from normal validation runs (parent's `not_applicable` + friends).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BrokenCategory {
    /// Fixture cannot be byte-gated honestly (no valid reference / not synthesizable).
    #[default]
    NotApplicable,
    /// Harness cannot bind resources / stages needed for this AIR.
    HarnessGap,
    /// Known translator FALLBACK / structural gap (honest skip until fixed).
    TranslatorGap,
    /// Apple oracle is unsafe or non-deterministic under the synthetic fixture.
    OracleHazard,
    /// Catch-all with a free-form reason.
    Other,
}

/// One broken / not-applicable row (one JSONL line).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BrokenEntry {
    /// Lowercase hex SHA-256 of the AIR source bytes.
    pub air_sha256: String,
    /// Required human-readable reason (what is broken and why skip is honest).
    pub reason: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    #[serde(default)]
    pub category: BrokenCategory,
}

/// In-memory index of `broken.jsonl` (keyed by lowercase `air_sha256`).
#[derive(Debug, Clone, Default)]
pub struct BrokenIndex {
    pub entries: HashMap<String, BrokenEntry>,
}

// --- paths + loaders -----------------------------------------------------------------------------

pub fn tolerances_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("corpus/tolerances.jsonl")
}

pub fn broken_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("corpus/broken.jsonl")
}

/// Load committed tolerances JSONL (empty index if the file is missing).
pub fn load_tolerances() -> TolerancesIndex {
    load_tolerances_from(&tolerances_path())
}

pub fn load_tolerances_from(path: &Path) -> TolerancesIndex {
    let text = match fs::read_to_string(path) {
        Ok(text) => text,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return TolerancesIndex::default(),
        Err(e) => panic!("read {}: {e}", path.display()),
    };
    let mut entries = HashMap::new();
    for (line_index, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut row: ToleranceEntry = serde_json::from_str(line).unwrap_or_else(|e| {
            panic!(
                "parse tolerance JSONL {}:{}: {e}",
                path.display(),
                line_index + 1
            )
        });
        row.air_sha256 = row.air_sha256.to_ascii_lowercase();
        let key = row.air_sha256.clone();
        if entries.insert(key.clone(), row).is_some() {
            panic!(
                "duplicate air_sha256 in {}:{}: {key}",
                path.display(),
                line_index + 1
            );
        }
    }
    TolerancesIndex { entries }
}

/// Load committed broken JSONL (empty index if missing).
pub fn load_broken() -> BrokenIndex {
    load_broken_from(&broken_path())
}

pub fn load_broken_from(path: &Path) -> BrokenIndex {
    let text = match fs::read_to_string(path) {
        Ok(text) => text,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return BrokenIndex::default(),
        Err(e) => panic!("read {}: {e}", path.display()),
    };
    let mut entries = HashMap::new();
    for (line_index, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut row: BrokenEntry = serde_json::from_str(line).unwrap_or_else(|e| {
            panic!(
                "parse broken JSONL {}:{}: {e}",
                path.display(),
                line_index + 1
            )
        });
        row.air_sha256 = row.air_sha256.to_ascii_lowercase();
        let key = row.air_sha256.clone();
        if entries.insert(key.clone(), row).is_some() {
            panic!(
                "duplicate air_sha256 in {}:{}: {key}",
                path.display(),
                line_index + 1
            );
        }
    }
    BrokenIndex { entries }
}

/// Process-wide cache of the committed tolerances JSONL.
pub fn tolerances_cached() -> &'static TolerancesIndex {
    static CACHE: OnceLock<TolerancesIndex> = OnceLock::new();
    CACHE.get_or_init(load_tolerances)
}

/// Process-wide cache of the committed broken JSONL.
pub fn broken_cached() -> &'static BrokenIndex {
    static CACHE: OnceLock<BrokenIndex> = OnceLock::new();
    CACHE.get_or_init(load_broken)
}

pub fn tolerance_for_air_sha256(air_sha256: &str) -> Option<&'static ToleranceEntry> {
    tolerances_cached()
        .entries
        .get(&air_sha256.to_ascii_lowercase())
}

pub fn broken_for_air_sha256(air_sha256: &str) -> Option<&'static BrokenEntry> {
    broken_cached()
        .entries
        .get(&air_sha256.to_ascii_lowercase())
}

/// SHA-256 (lowercase hex) of `bytes`. Uses the system `shasum` / `sha256sum` so validation
/// stays free of a crypto crate; fine for offline tooling and sparse lookups.
pub fn air_sha256_hex(bytes: &[u8]) -> String {
    use std::io::Write;
    use std::process::{Command, Stdio};
    let mut child = Command::new("shasum")
        .args(["-a", "256"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .or_else(|_| {
            Command::new("sha256sum")
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .stderr(Stdio::null())
                .spawn()
        })
        .expect("shasum or sha256sum required to fingerprint AIR");
    {
        let mut stdin = child.stdin.take().expect("stdin");
        stdin.write_all(bytes).expect("write hash input");
    }
    let out = child.wait_with_output().expect("hash process");
    assert!(out.status.success(), "hash process failed");
    let s = String::from_utf8_lossy(&out.stdout);
    s.split_whitespace()
        .next()
        .expect("hash output")
        .trim()
        .to_ascii_lowercase()
}

fn leak_reason(reason: &str) -> &'static str {
    Box::leak(reason.to_owned().into_boxed_str())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn committed_sidecars_parse() {
        let t = load_tolerances();
        let b = load_broken();
        // Empty committed files are fine; load must not panic.
        let _ = (t.entries.len(), b.entries.len());
    }

    #[test]
    fn tolerance_spec_roundtrip_json() {
        let spec = ToleranceSpec::AbsAndUlp {
            max_abs: 1.5,
            max_ulp: 2,
            reason: "example".into(),
        };
        let json = serde_json::to_string(&spec).unwrap();
        let back: ToleranceSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(spec, back);
        let tol = back.to_tolerance();
        assert!(!tol.is_exact());
        assert_eq!(tol.reason(), Some("example"));
    }

    #[test]
    fn broken_jsonl_lookup_is_case_insensitive() {
        let path = std::env::temp_dir().join(format!(
            "m2v-broken-test-{}-{}.jsonl",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let row = BrokenEntry {
            air_sha256: "ABCD".into(),
            reason: "demo".into(),
            label: None,
            category: BrokenCategory::TranslatorGap,
        };
        fs::write(&path, format!("{}\n", serde_json::to_string(&row).unwrap())).unwrap();
        let loaded = load_broken_from(&path);
        assert!(loaded.entries.contains_key("abcd"));
        assert_eq!(loaded.entries["abcd"].reason, "demo");
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn tolerances_jsonl_loads_row() {
        let path = std::env::temp_dir().join(format!(
            "m2v-tol-test-{}-{}.jsonl",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let body = r#"# comment
{"air_sha256":"deadbeef","label":"sample","tolerance":{"kind":"Ulp","max_ulp":1,"reason":"demo ulp"}}
"#;
        fs::write(&path, body).unwrap();
        let loaded = load_tolerances_from(&path);
        let e = loaded.entries.get("deadbeef").expect("row");
        assert_eq!(e.label.as_deref(), Some("sample"));
        assert_eq!(e.tolerance.reason(), Some("demo ulp"));
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn air_sha256_hex_is_stable() {
        let a = air_sha256_hex(b"hello");
        let b = air_sha256_hex(b"hello");
        assert_eq!(a, b);
        assert_eq!(a.len(), 64);
        assert_ne!(a, air_sha256_hex(b"world"));
    }
}
