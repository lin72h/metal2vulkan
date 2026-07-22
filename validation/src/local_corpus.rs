//! Optional private metallib corpus support.
//!
//! Generated (gitignored) `tests/corpus_NN.rs` stubs call [`run_corpus_case`], which seeks one
//! JSONL record under `corpus/local/shards/` (or `METAL2VULKAN_CORPUS_DIR`) and runs a **translate
//! smoke** — not the monorepo’s full Metal-oracle byte gate.
//!
//! When the shard file is missing the test fails with a clear “private corpus not present”
//! message so a clean clone without local data never silently passes a half-generated suite.

use serde::Deserialize;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::process::Command;

/// Default relative root under the validation package (`validation/corpus/local`).
pub const DEFAULT_LOCAL_CORPUS_REL: &str = "corpus/local";

#[derive(Debug, Deserialize)]
struct CorpusRecord {
    id: String,
    hash: String,
    #[serde(rename = "fn")]
    #[allow(dead_code)]
    fn_name: Option<String>,
    stage: Option<String>,
    #[serde(default)]
    synth: bool,
    ignore_reason: Option<String>,
    air_ll: String,
}

/// Directory that holds `shards/shard_NN.jsonl`. Override with `METAL2VULKAN_CORPUS_DIR`.
pub fn corpus_root() -> PathBuf {
    if let Ok(dir) = std::env::var("METAL2VULKAN_CORPUS_DIR") {
        return PathBuf::from(dir);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(DEFAULT_LOCAL_CORPUS_REL)
}

/// Path to the committed public drift ledger (hashes only).
pub fn drift_ledger_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("corpus/drift-ledger.jsonl")
}

/// Seek one JSONL record and translate-smoke it.
///
/// Compatible with monorepo-generated `corpus_NN.rs` stubs:
/// `run_corpus_case(shard, byte_offset, byte_len)`.
pub fn run_corpus_case(shard: &str, byte_offset: u64, byte_len: usize) {
    let record = read_record(shard, byte_offset, byte_len);
    assert!(
        record.synth,
        "[{}] corpus case is not runnable: {}",
        record.id,
        record
            .ignore_reason
            .as_deref()
            .unwrap_or("metallib: not synthesized")
    );

    let air_sha = crate::air_sha256_hex(record.air_ll.as_bytes());
    if let Some(broken) = crate::broken_for_air_sha256(&air_sha) {
        eprintln!(
            "[{}] BROKEN air_sha256={} category={:?} reason={}",
            record.id, air_sha, broken.category, broken.reason
        );
        return;
    }

    let tmp = crate::scratch_dir_for(&format!("local-corpus-{}", record.hash));
    let air_path = tmp.join("case.ll");
    std::fs::write(&air_path, record.air_ll.as_bytes())
        .unwrap_or_else(|e| panic!("[{}] write temp AIR {}: {e}", record.id, air_path.display()));

    let stage = record.stage.as_deref().unwrap_or("Kernel");
    let m2v_stage = parse_stage(stage);
    let air_src = air_path.to_str().expect("temp AIR path is UTF-8");
    let spv = match metal2vulkan::translate(air_src, m2v_stage, &tmp) {
        Ok(bytes) => bytes,
        Err(e) => {
            let _ = std::fs::remove_dir_all(&tmp);
            panic!("[{}] translate failed (stage={stage}): {e}", record.id);
        }
    };

    if std::env::var_os("METAL2VULKAN_CORPUS_SKIP_VAL").is_none()
        && Command::new("spirv-val").arg("--version").output().is_ok()
    {
        if let Err(e) = metal2vulkan::tools::spirv_val_bytes(&spv, &tmp) {
            let _ = std::fs::remove_dir_all(&tmp);
            panic!("[{}] spirv-val failed: {e}", record.id);
        }
    }

    if std::env::var_os("METAL2VULKAN_CORPUS_DRIFT").is_some() {
        check_drift_against_ledger(&record, &spv);
    }

    eprintln!(
        "[{}] LOCAL_CORPUS_OK hash={} spv_bytes={}",
        record.id,
        record.hash,
        spv.len()
    );
    let _ = std::fs::remove_dir_all(&tmp);
}

fn read_record(shard: &str, byte_offset: u64, byte_len: usize) -> CorpusRecord {
    let path = corpus_root()
        .join("shards")
        .join(format!("shard_{shard}.jsonl"));
    assert!(
        path.is_file(),
        "private corpus shard missing: {} \
         (populate validation/corpus/local/shards/ or set METAL2VULKAN_CORPUS_DIR; \
         see validation/corpus/README.md)",
        path.display()
    );
    let mut file = File::open(&path).unwrap_or_else(|e| panic!("open {}: {e}", path.display()));
    file.seek(SeekFrom::Start(byte_offset))
        .unwrap_or_else(|e| panic!("seek {} @ {byte_offset}: {e}", path.display()));
    let mut buf = vec![0u8; byte_len];
    file.read_exact(&mut buf)
        .unwrap_or_else(|e| panic!("read {} len={byte_len}: {e}", path.display()));
    // Trim trailing newline if the recorded length included it.
    while buf.last() == Some(&b'\n') || buf.last() == Some(&b'\r') {
        buf.pop();
    }
    serde_json::from_slice(&buf).unwrap_or_else(|e| {
        panic!("parse corpus record shard={shard} off={byte_offset} len={byte_len}: {e}")
    })
}

fn parse_stage(stage: &str) -> metal2vulkan::passes::Stage {
    match stage {
        "Kernel" | "kernel" | "compute" | "Compute" => metal2vulkan::passes::Stage::Kernel,
        "Vertex" | "vertex" => metal2vulkan::passes::Stage::Vertex,
        "Fragment" | "fragment" => metal2vulkan::passes::Stage::Fragment,
        other => panic!("unsupported corpus stage {other:?} (Kernel|Vertex|Fragment)"),
    }
}

fn check_drift_against_ledger(record: &CorpusRecord, spv: &[u8]) {
    let ledger = drift_ledger_path();
    if !ledger.is_file() {
        return;
    }
    let air_hash = sha256_hex(record.air_ll.as_bytes());
    let spv_hash = sha256_hex(spv);
    let text = std::fs::read_to_string(&ledger)
        .unwrap_or_else(|e| panic!("read drift ledger {}: {e}", ledger.display()));
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let row: DriftRow = match serde_json::from_str(line) {
            Ok(r) => r,
            Err(_) => continue,
        };
        if row.air_sha256 != air_hash {
            continue;
        }
        match row.status.as_str() {
            "ok" => {
                let expected = row.spv_sha256.as_deref().unwrap_or("");
                assert_eq!(
                    spv_hash, expected,
                    "[{}] SPIR-V drift for air_sha256={air_hash}: got {spv_hash}, ledger {expected}",
                    record.id
                );
            }
            "fallback" => panic!(
                "[{}] translated ok but ledger marks air_sha256={air_hash} as fallback",
                record.id
            ),
            _ => {}
        }
        return;
    }
    // No ledger row: drift check is advisory; presence of METAL2VULKAN_CORPUS_DRIFT does not
    // require every private case to be banked.
}

#[derive(Debug, Deserialize)]
struct DriftRow {
    air_sha256: String,
    #[serde(default)]
    spv_sha256: Option<String>,
    status: String,
}

fn sha256_hex(bytes: &[u8]) -> String {
    use std::io::Write;
    // Prefer system shasum so we do not pull a crypto crate into validation for one helper.
    // Fall back to a pure-Rust FNV-style fingerprint is NOT acceptable for a public ledger —
    // require sha256sum/shasum.
    let mut child = Command::new("shasum")
        .args(["-a", "256"])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .spawn()
        .or_else(|_| {
            Command::new("sha256sum")
                .stdin(std::process::Stdio::piped())
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::null())
                .spawn()
        })
        .expect("shasum or sha256sum required for drift checks");
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
        .to_string()
}

/// True when the default local corpus shards directory exists (for optional discovery tests).
pub fn local_corpus_present() -> bool {
    corpus_root().join("shards").is_dir()
}

/// Resolve a path relative to the validation package for docs/tests.
pub fn validation_pkg_root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}
