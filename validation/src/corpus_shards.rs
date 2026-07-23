//! Shared JSONL shard handling for the private validation corpus.
//!
//! Harvested corpus rows are the durable source: each row carries sanitized LLVM text plus the
//! optional AIR bitcode payload. The old `corpus/local/air` filesystem mirror is intentionally not
//! part of source resolution anymore.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

pub use crate::hash::{hex_encode, sha256_bytes, sha256_file};
pub use crate::jsonl::sort_json;

pub const SHARD_COUNT: usize = 16;
pub const DEFAULT_LOCAL_CORPUS_REL: &str = "corpus/local";
pub const CORPUS_DIR_ENV: &str = "METAL2VULKAN_CORPUS_DIR";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardRecord {
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub hash: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shard: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lib: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lib_sha256: Option<String>,
    #[serde(rename = "fn", default, skip_serializing_if = "Option::is_none")]
    pub fn_name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stage: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub blob_b64: Option<String>,
    #[serde(default)]
    pub air_ll: String,
    #[serde(default)]
    pub air_sha256: String,
}

impl ShardRecord {
    pub fn normalized_air_sha256(&self) -> String {
        if self.air_sha256.len() == 64 && self.air_sha256.chars().all(|c| c.is_ascii_hexdigit()) {
            self.air_sha256.to_ascii_lowercase()
        } else {
            sha256_bytes(self.air_ll.as_bytes())
        }
    }

    pub fn label_or_default(&self) -> String {
        self.label
            .clone()
            .unwrap_or_else(|| format!("local/{}.ll", self.normalized_air_sha256()))
    }
}

#[derive(Debug, Clone)]
pub enum SourceStorage {
    PublicPath(PathBuf),
    ShardRow {
        shard: String,
        byte_offset: u64,
        byte_len: usize,
    },
}

#[derive(Debug, Clone)]
pub struct SourceRef {
    pub air_sha256: String,
    pub label: String,
    pub kind: String,
    pub shard: Option<String>,
    pub storage: SourceStorage,
}

#[derive(Debug, Clone)]
pub struct SourceData {
    pub air_sha256: String,
    pub label: String,
    pub kind: String,
    pub shard: Option<String>,
    pub air_ll: String,
    pub blob_b64: Option<String>,
    pub lib: Option<String>,
    pub lib_sha256: Option<String>,
    pub public_path: Option<PathBuf>,
}

impl SourceRef {
    pub fn load(&self, corpus_root: &Path) -> Result<SourceData, String> {
        match &self.storage {
            SourceStorage::PublicPath(path) => load_public_source(path, &self.label, &self.kind),
            SourceStorage::ShardRow {
                shard,
                byte_offset,
                byte_len,
            } => {
                let record = read_shard_record_at(corpus_root, shard, *byte_offset, *byte_len)
                    .map_err(|e| format!("{e}"))?;
                let air_sha256 = record.normalized_air_sha256();
                if air_sha256 != self.air_sha256 {
                    return Err(format!(
                        "shard source hash mismatch: ref={} row={air_sha256}",
                        self.air_sha256
                    ));
                }
                Ok(SourceData {
                    air_sha256,
                    label: record.label_or_default(),
                    kind: "private".into(),
                    shard: Some(shard.clone()),
                    air_ll: record.air_ll,
                    blob_b64: record.blob_b64,
                    lib: record.lib,
                    lib_sha256: record.lib_sha256,
                    public_path: None,
                })
            }
        }
    }
}

pub fn corpus_root_from_env_or_manifest() -> PathBuf {
    if let Ok(dir) = std::env::var(CORPUS_DIR_ENV) {
        return PathBuf::from(dir);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(DEFAULT_LOCAL_CORPUS_REL)
}

pub fn shards_dir(corpus_root: &Path) -> PathBuf {
    corpus_root.join("shards")
}

pub fn shard_name_for_index(index: usize) -> String {
    format!("shard_{:02}.jsonl", index % SHARD_COUNT)
}

pub fn shard_name_for_hash(hash: &str) -> String {
    shard_name_for_index(shard_index_for_hash(hash))
}

pub fn shard_index_for_hash(hash: &str) -> usize {
    let prefix = hash.get(..16).unwrap_or(hash);
    u64::from_str_radix(prefix, 16).unwrap_or(0) as usize % SHARD_COUNT
}

pub fn normalize_shard_name(shard: &str) -> String {
    let shard = shard.trim();
    if shard.starts_with("shard_") && shard.ends_with(".jsonl") {
        shard.to_string()
    } else if shard.starts_with("shard_") {
        format!("{shard}.jsonl")
    } else {
        format!("shard_{shard}.jsonl")
    }
}

pub fn shard_path(corpus_root: &Path, shard: &str) -> PathBuf {
    shards_dir(corpus_root).join(normalize_shard_name(shard))
}

pub fn existing_shard_paths(corpus_root: &Path) -> Vec<PathBuf> {
    let mut paths = Vec::new();
    let Ok(entries) = fs::read_dir(shards_dir(corpus_root)) else {
        return paths;
    };
    for ent in entries.flatten() {
        let path = ent.path();
        let Some(name) = path.file_name().and_then(|s| s.to_str()) else {
            continue;
        };
        if name.starts_with("shard_") && name.ends_with(".jsonl") && path.is_file() {
            paths.push(path);
        }
    }
    paths.sort();
    paths
}

pub fn read_shard_record_at(
    corpus_root: &Path,
    shard: &str,
    byte_offset: u64,
    byte_len: usize,
) -> std::io::Result<ShardRecord> {
    let path = shard_path(corpus_root, shard);
    let mut file = File::open(&path)?;
    file.seek(SeekFrom::Start(byte_offset))?;
    let mut buf = vec![0u8; byte_len];
    file.read_exact(&mut buf)?;
    while matches!(buf.last(), Some(b'\n' | b'\r')) {
        buf.pop();
    }
    serde_json::from_slice(&buf).map_err(std::io::Error::other)
}

pub fn find_shard_source_ref(
    corpus_root: &Path,
    shard_hint: &str,
    want: &str,
) -> Result<Option<SourceRef>, String> {
    let path = shard_path(corpus_root, shard_hint);
    if !path.is_file() {
        return Ok(None);
    }
    find_source_ref_in_shard_path(&path, want)
}

pub fn resolve_source(
    air_sha256: &str,
    label_hint: &str,
    kind_hint: &str,
    shard_hint: Option<&str>,
    public_dir: &Path,
    corpus_root: &Path,
) -> Result<Option<SourceData>, String> {
    let want = air_sha256.to_ascii_lowercase();
    if let Some(src) = try_public_label(&want, label_hint, kind_hint, public_dir)? {
        return src.load(corpus_root).map(Some);
    }

    let mut tried = HashSet::<String>::new();
    if let Some(shard) = shard_hint.filter(|s| !s.is_empty()) {
        let shard = normalize_shard_name(shard);
        tried.insert(shard.clone());
        if let Some(src) = find_shard_source_ref(corpus_root, &shard, &want)? {
            return src.load(corpus_root).map(Some);
        }
    }

    let shard = shard_name_for_hash(&want);
    if tried.insert(shard.clone()) {
        if let Some(src) = find_shard_source_ref(corpus_root, &shard, &want)? {
            return src.load(corpus_root).map(Some);
        }
    }

    for path in existing_shard_paths(corpus_root) {
        let Some(name) = path.file_name().and_then(|s| s.to_str()) else {
            continue;
        };
        if !tried.insert(name.to_string()) {
            continue;
        }
        if let Some(src) = find_source_ref_in_shard_path(&path, &want)? {
            return src.load(corpus_root).map(Some);
        }
    }
    Ok(None)
}

pub fn gather_source_refs(public_dir: &Path, corpus_root: &Path) -> Vec<SourceRef> {
    let mut sources = Vec::new();
    sources.extend(gather_public_source_refs(
        public_dir,
        public_dir,
        "public",
        "synthetic",
    ));
    sources.extend(gather_shard_source_refs(corpus_root));
    sources
}

pub fn gather_shard_source_refs(corpus_root: &Path) -> Vec<SourceRef> {
    let mut out = Vec::new();
    for path in existing_shard_paths(corpus_root) {
        let shard = path
            .file_name()
            .and_then(|s| s.to_str())
            .map(str::to_string)
            .unwrap_or_else(|| "shard_unknown.jsonl".into());
        let Ok(file) = File::open(&path) else {
            continue;
        };
        let mut reader = BufReader::new(file);
        let mut offset = 0u64;
        let mut line = Vec::new();
        loop {
            line.clear();
            let Ok(read) = reader.read_until(b'\n', &mut line) else {
                break;
            };
            if read == 0 {
                break;
            }
            let len = trim_line_ending_len(&line);
            if len != 0 {
                if let Ok(record) = serde_json::from_slice::<ShardRecord>(&line[..len]) {
                    out.push(source_ref_from_record(record, &shard, offset, len));
                }
            }
            offset += read as u64;
        }
    }
    out
}

pub fn trim_line_ending_len(line: &[u8]) -> usize {
    let mut len = line.len();
    if len != 0 && line[len - 1] == b'\n' {
        len -= 1;
    }
    if len != 0 && line[len - 1] == b'\r' {
        len -= 1;
    }
    len
}

fn load_public_source(path: &Path, label: &str, kind: &str) -> Result<SourceData, String> {
    if path.extension().and_then(|e| e.to_str()) != Some("ll") {
        return Err(format!(
            "public source {} needs a sibling .ll for shard-backed runners",
            path.display()
        ));
    }
    let air_ll = fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let air_sha256 = sha256_file(path).map_err(|e| format!("hash {}: {e}", path.display()))?;
    Ok(SourceData {
        air_sha256,
        label: label.to_string(),
        kind: kind.to_string(),
        shard: None,
        air_ll,
        blob_b64: None,
        lib: None,
        lib_sha256: None,
        public_path: Some(path.to_path_buf()),
    })
}

fn try_public_label(
    want: &str,
    label: &str,
    kind_hint: &str,
    public_dir: &Path,
) -> Result<Option<SourceRef>, String> {
    let Some(rel) = label.strip_prefix("public/") else {
        return Ok(None);
    };
    let path = public_dir.join(rel);
    if !path.is_file() {
        return Ok(None);
    }
    let got = sha256_file(&path).map_err(|e| format!("hash {}: {e}", path.display()))?;
    if got != want {
        return Ok(None);
    }
    Ok(Some(SourceRef {
        air_sha256: want.to_string(),
        label: label.to_string(),
        kind: if kind_hint.is_empty() {
            "synthetic".into()
        } else {
            kind_hint.into()
        },
        shard: None,
        storage: SourceStorage::PublicPath(path),
    }))
}

fn gather_public_source_refs(
    root: &Path,
    label_root: &Path,
    label_prefix: &str,
    kind: &str,
) -> Vec<SourceRef> {
    if !root.is_dir() {
        return Vec::new();
    }
    let mut by_stem: HashMap<(PathBuf, String), PathBuf> = HashMap::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = fs::read_dir(&dir) else {
            continue;
        };
        for ent in entries.flatten() {
            let path = ent.path();
            let Ok(ft) = ent.file_type() else {
                continue;
            };
            if ft.is_dir() {
                stack.push(path);
                continue;
            }
            if !ft.is_file() {
                continue;
            }
            let ext = path
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("")
                .to_ascii_lowercase();
            if ext != "ll" && ext != "air" {
                continue;
            }
            let parent = path
                .parent()
                .map(Path::to_path_buf)
                .unwrap_or_else(|| PathBuf::from("."));
            let stem = path
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_default();
            let key = (parent, stem);
            match by_stem.get(&key) {
                None => {
                    by_stem.insert(key, path);
                }
                Some(prev) if ext == "ll" && prev.extension().is_some_and(|e| e == "air") => {
                    by_stem.insert(key, path);
                }
                _ => {}
            }
        }
    }

    let mut out = Vec::with_capacity(by_stem.len());
    for path in by_stem.into_values() {
        let Ok(air_sha256) = sha256_file(&path) else {
            continue;
        };
        let rel = path
            .strip_prefix(label_root)
            .map(|p| p.display().to_string())
            .unwrap_or_else(|_| {
                path.file_name()
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_else(|| "unknown".into())
            });
        let label = format!("{label_prefix}/{rel}");
        out.push(SourceRef {
            air_sha256,
            label,
            kind: kind.to_string(),
            shard: None,
            storage: SourceStorage::PublicPath(path),
        });
    }
    out
}

fn find_source_ref_in_shard_path(path: &Path, want: &str) -> Result<Option<SourceRef>, String> {
    let shard = path
        .file_name()
        .and_then(|s| s.to_str())
        .map(str::to_string)
        .unwrap_or_else(|| "shard_unknown.jsonl".into());
    let file = File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let mut reader = BufReader::new(file);
    let mut offset = 0u64;
    let mut line = Vec::new();
    loop {
        line.clear();
        let read = reader
            .read_until(b'\n', &mut line)
            .map_err(|e| format!("read {}: {e}", path.display()))?;
        if read == 0 {
            break;
        }
        let len = trim_line_ending_len(&line);
        if len != 0 {
            let haystack = String::from_utf8_lossy(&line[..len]);
            if haystack.contains(want) {
                let record: ShardRecord = serde_json::from_slice(&line[..len])
                    .map_err(|e| format!("parse {} @ {offset}: {e}", path.display()))?;
                if record.normalized_air_sha256() == want {
                    return Ok(Some(source_ref_from_record(record, &shard, offset, len)));
                }
            }
        }
        offset += read as u64;
    }
    Ok(None)
}

fn source_ref_from_record(
    record: ShardRecord,
    shard: &str,
    byte_offset: u64,
    byte_len: usize,
) -> SourceRef {
    let air_sha256 = record.normalized_air_sha256();
    SourceRef {
        air_sha256: air_sha256.clone(),
        label: record.label_or_default(),
        kind: "private".into(),
        shard: Some(shard.to_string()),
        storage: SourceStorage::ShardRow {
            shard: shard.to_string(),
            byte_offset,
            byte_len,
        },
    }
}
