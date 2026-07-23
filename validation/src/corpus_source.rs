use crate::corpus_shards::{self, SourceData};
use base64::Engine as _;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct SourceFile {
    pub label: String,
    pub kind: String,
    pub air_sha256: String,
    pub shard: Option<String>,
    pub air_ll: String,
    pub blob_b64: Option<String>,
    pub lib: Option<String>,
    pub lib_sha256: Option<String>,
    pub public_path: Option<PathBuf>,
}

pub fn resolve_source(
    air_sha256: &str,
    label_hint: &str,
    kind_hint: &str,
    shard_hint: Option<&str>,
    public_dir: &Path,
    local_corpus: &Path,
) -> Option<SourceFile> {
    match corpus_shards::resolve_source(
        air_sha256,
        label_hint,
        kind_hint,
        shard_hint,
        public_dir,
        local_corpus,
    ) {
        Ok(Some(source)) => Some(source_file_from_data(source)),
        Ok(None) => None,
        Err(error) => {
            eprintln!("    source resolve error for {air_sha256}: {error}");
            None
        }
    }
}

pub fn load_ll_text(source: &SourceFile) -> Result<String, String> {
    if source.air_ll.is_empty() {
        Err(format!("source {} has empty air_ll", source.label))
    } else {
        Ok(source.air_ll.clone())
    }
}

pub fn air_blob_for_oracle(source: &SourceFile) -> Result<Vec<u8>, String> {
    if let Some(blob_b64) = source.blob_b64.as_deref().filter(|s| !s.is_empty()) {
        return base64::engine::general_purpose::STANDARD
            .decode(blob_b64)
            .map_err(|e| format!("decode AIR blob_b64: {e}"));
    }
    #[cfg(target_os = "macos")]
    {
        metal_as_ll_to_air(source)
    }
    #[cfg(not(target_os = "macos"))]
    {
        Err(format!(
            "no AIR blob_b64 for {} (metal-as only available on macOS)",
            source.label
        ))
    }
}

#[cfg(target_os = "macos")]
pub fn source_metallib_for_air(source: &SourceFile) -> Option<PathBuf> {
    let source_path = source.lib.as_deref().map(PathBuf::from)?;
    if !source_path.is_file() {
        return None;
    }
    if let Some(want_sha) = source.lib_sha256.as_deref().filter(|s| !s.is_empty()) {
        if !crate::hash::sha256_file(&source_path).is_ok_and(|sha| sha == want_sha) {
            return None;
        }
    }
    Some(source_path)
}

fn source_file_from_data(source: SourceData) -> SourceFile {
    SourceFile {
        label: source.label,
        kind: source.kind,
        air_sha256: source.air_sha256,
        shard: source.shard,
        air_ll: source.air_ll,
        blob_b64: source.blob_b64,
        lib: source.lib,
        lib_sha256: source.lib_sha256,
        public_path: source.public_path,
    }
}

#[cfg(target_os = "macos")]
fn metal_as_ll_to_air(source: &SourceFile) -> Result<Vec<u8>, String> {
    let tmp = crate::scratch_dir_for("corpus-run-metal-as");
    let mut text = source.air_ll.clone();
    if !text.contains("!air.version") {
        text.push_str(
            "\n!air.version = !{!999001}\n!999001 = !{i32 1, i32 8, i32 0}\n!air.language_version = !{!999002}\n!999002 = !{!\"Metal\", i32 2, i32 0, i32 0}\n",
        );
    }
    let in_ll = tmp.join("case.ll");
    let out_air = tmp.join("case.air");
    std::fs::write(&in_ll, text).map_err(|e| format!("write temp ll: {e}"))?;
    let status = std::process::Command::new("xcrun")
        .args([
            "metal-as",
            in_ll.to_str().ok_or("ll path utf8")?,
            "-o",
            out_air.to_str().ok_or("air path utf8")?,
        ])
        .output()
        .map_err(|e| format!("spawn metal-as: {e}"))?;
    if !status.status.success() {
        let _ = std::fs::remove_dir_all(&tmp);
        return Err(format!(
            "metal-as failed: {}",
            String::from_utf8_lossy(&status.stderr)
        ));
    }
    let bytes = std::fs::read(&out_air).map_err(|e| format!("read assembled air: {e}"))?;
    let _ = std::fs::remove_dir_all(&tmp);
    Ok(bytes)
}
