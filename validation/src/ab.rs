//! GPU-free old/new translation comparison.

use crate::hash::{sha256_bytes, sha256_file};
use crate::jsonl::to_sorted_json_string;
use crate::source::{find_source, public_sources, read_source_shard, source_shard_path, SourceRow};
use crate::ScratchDir;
use base64::Engine as _;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::Duration;
use wait_timeout::ChildExt;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AbClassification {
    UnchangedSpirv,
    ChangedSpirv,
    FallbackToSuccess,
    SuccessToFallback,
    ValidToInvalid,
    TimeoutOrToolFailure,
}

impl AbClassification {
    pub fn label(self) -> &'static str {
        match self {
            Self::UnchangedSpirv => "unchanged SPIR-V",
            Self::ChangedSpirv => "changed SPIR-V",
            Self::FallbackToSuccess => "fallback -> success",
            Self::SuccessToFallback => "success -> fallback",
            Self::ValidToInvalid => "valid -> invalid",
            Self::TimeoutOrToolFailure => "timeout/tool failure",
        }
    }
}

#[derive(Clone, Debug)]
pub struct AbOptions {
    pub corpus_root: PathBuf,
    pub old_binary: PathBuf,
    pub new_binary: PathBuf,
    pub selection: AbSelection,
    pub translator_options: Vec<String>,
    pub cache_dir: PathBuf,
    pub timeout: Duration,
    pub expect_no_change: bool,
    pub fail_on_unlisted_change: bool,
    pub spv_allowlist: HashSet<String>,
    pub fallback_to_success_allowlist: HashSet<String>,
}

#[derive(Clone, Debug, Default)]
pub struct AbSelection {
    pub air_sha256: Vec<String>,
    pub shards: Vec<usize>,
    pub canary: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AbResult {
    pub air_sha256: String,
    pub label: String,
    pub classification: AbClassification,
    pub old_spv_sha256: Option<String>,
    pub new_spv_sha256: Option<String>,
    pub allowed: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum TranslationStatus {
    Success,
    Fallback,
    Invalid,
    ToolFailure,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct TranslationResult {
    status: TranslationStatus,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    spv_b64: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    spv_sha256: Option<String>,
    stderr: String,
}

#[derive(Serialize)]
struct CacheKey<'a> {
    air_sha256: &'a str,
    translator_sha256: &'a str,
    stage: &'a str,
    translator_options: &'a [String],
    external_tool_identity: &'a str,
    translator_environment: &'a [(String, String)],
}

pub fn run(options: &AbOptions) -> Result<Vec<AbResult>, String> {
    validate_binary(&options.old_binary)?;
    validate_binary(&options.new_binary)?;
    let sources = selected_sources(&options.corpus_root, &options.selection)?;
    if sources.is_empty() {
        return Err("selection contains no AIR sources".into());
    }
    fs::create_dir_all(&options.cache_dir)
        .map_err(|error| format!("create cache {}: {error}", options.cache_dir.display()))?;
    let old_binary_sha = sha256_file(&options.old_binary)
        .map_err(|error| format!("hash {}: {error}", options.old_binary.display()))?;
    let new_binary_sha = sha256_file(&options.new_binary)
        .map_err(|error| format!("hash {}: {error}", options.new_binary.display()))?;
    let external_tool_identity = spirv_val_identity()?;
    let translator_environment = translator_environment();
    let mut results = Vec::new();
    for source in sources {
        let old = translate_cached(
            &source,
            &options.old_binary,
            &old_binary_sha,
            &external_tool_identity,
            &translator_environment,
            options,
        )?;
        let new = translate_cached(
            &source,
            &options.new_binary,
            &new_binary_sha,
            &external_tool_identity,
            &translator_environment,
            options,
        )?;
        let classification = classify(&old, &new);
        let allowed = policy_allows(
            classification,
            &source.air_sha256,
            options.expect_no_change,
            options.fail_on_unlisted_change,
            &options.spv_allowlist,
            &options.fallback_to_success_allowlist,
        );
        results.push(AbResult {
            air_sha256: source.air_sha256,
            label: source.label,
            classification,
            old_spv_sha256: old.spv_sha256,
            new_spv_sha256: new.spv_sha256,
            allowed,
        });
    }
    Ok(results)
}

pub fn selected_sources(root: &Path, selection: &AbSelection) -> Result<Vec<SourceRow>, String> {
    let mut sources = HashMap::<String, SourceRow>::new();
    if selection.canary {
        for source in public_sources()? {
            sources.insert(source.air_sha256.clone(), source);
        }
    }
    for &shard in &selection.shards {
        let path = source_shard_path(root, shard);
        if !path.is_file() {
            return Err(format!("source shard {} does not exist", path.display()));
        }
        for source in read_source_shard(&path)? {
            sources.insert(source.air_sha256.clone(), source);
        }
    }
    for hash in &selection.air_sha256 {
        let source = find_source(root, hash)?.ok_or_else(|| {
            format!("AIR {hash} was not found in source shards or public fixtures")
        })?;
        sources.insert(source.air_sha256.clone(), source);
    }
    let mut sources = sources.into_values().collect::<Vec<_>>();
    sources.sort_by(|left, right| left.air_sha256.cmp(&right.air_sha256));
    Ok(sources)
}

pub fn read_hash_list(path: &Path) -> Result<HashSet<String>, String> {
    let text = fs::read_to_string(path)
        .map_err(|error| format!("read hash list {}: {error}", path.display()))?;
    let mut hashes = HashSet::new();
    for (index, line) in text.lines().enumerate() {
        let hash = line.split('#').next().unwrap_or_default().trim();
        if hash.is_empty() {
            continue;
        }
        if hash.len() != 64 || !hash.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return Err(format!(
                "{}:{}: expected one SHA-256 per line",
                path.display(),
                index + 1
            ));
        }
        hashes.insert(hash.to_ascii_lowercase());
    }
    Ok(hashes)
}

pub fn change_is_allowed(
    classification: AbClassification,
    air_sha256: &str,
    spv_allowlist: &HashSet<String>,
    fallback_to_success_allowlist: &HashSet<String>,
) -> bool {
    match classification {
        AbClassification::UnchangedSpirv => true,
        AbClassification::ChangedSpirv => spv_allowlist.contains(air_sha256),
        AbClassification::FallbackToSuccess => fallback_to_success_allowlist.contains(air_sha256),
        AbClassification::SuccessToFallback
        | AbClassification::ValidToInvalid
        | AbClassification::TimeoutOrToolFailure => false,
    }
}

fn policy_allows(
    classification: AbClassification,
    air_sha256: &str,
    expect_no_change: bool,
    fail_on_unlisted_change: bool,
    spv_allowlist: &HashSet<String>,
    fallback_to_success_allowlist: &HashSet<String>,
) -> bool {
    if expect_no_change {
        classification == AbClassification::UnchangedSpirv
    } else if fail_on_unlisted_change {
        change_is_allowed(
            classification,
            air_sha256,
            spv_allowlist,
            fallback_to_success_allowlist,
        )
    } else {
        classification != AbClassification::TimeoutOrToolFailure
    }
}

fn validate_binary(path: &Path) -> Result<(), String> {
    if !path.is_file() {
        return Err(format!(
            "translator binary {} is not a file",
            path.display()
        ));
    }
    Ok(())
}

fn translate_cached(
    source: &SourceRow,
    binary: &Path,
    binary_sha: &str,
    external_tool_identity: &str,
    translator_environment: &[(String, String)],
    options: &AbOptions,
) -> Result<TranslationResult, String> {
    let stage = source.stage.to_ascii_lowercase();
    let key = to_sorted_json_string(CacheKey {
        air_sha256: &source.air_sha256,
        translator_sha256: binary_sha,
        stage: &stage,
        translator_options: &options.translator_options,
        external_tool_identity,
        translator_environment,
    })
    .map_err(|error| format!("serialize cache key: {error}"))?;
    let key = sha256_bytes(key.as_bytes());
    let cache_path = options.cache_dir.join(format!("{key}.json"));
    if cache_path.is_file() {
        let result: TranslationResult = serde_json::from_slice(
            &fs::read(&cache_path)
                .map_err(|error| format!("read cache {}: {error}", cache_path.display()))?,
        )
        .map_err(|error| format!("parse cache {}: {error}", cache_path.display()))?;
        validate_translation_result(&result)
            .map_err(|error| format!("invalid cache {}: {error}", cache_path.display()))?;
        return Ok(result);
    }
    let result = translate(source, binary, &stage, options)?;
    if result.status != TranslationStatus::ToolFailure {
        let bytes = serde_json::to_vec(&result)
            .map_err(|error| format!("serialize cache result: {error}"))?;
        let temporary = options
            .cache_dir
            .join(format!(".{key}.{}.tmp", std::process::id()));
        let write_result = (|| {
            let mut file = fs::File::create(&temporary)
                .map_err(|error| format!("create cache {}: {error}", temporary.display()))?;
            file.write_all(&bytes)
                .and_then(|()| file.sync_all())
                .map_err(|error| format!("write cache {}: {error}", temporary.display()))
        })();
        if let Err(error) = write_result {
            let _ = fs::remove_file(&temporary);
            return Err(error);
        }
        fs::rename(&temporary, &cache_path).map_err(|error| {
            let _ = fs::remove_file(&temporary);
            format!(
                "commit cache {} -> {}: {error}",
                temporary.display(),
                cache_path.display()
            )
        })?;
    }
    Ok(result)
}

fn translate(
    source: &SourceRow,
    binary: &Path,
    stage: &str,
    options: &AbOptions,
) -> Result<TranslationResult, String> {
    let scratch = ScratchDir::new("corpus-ab")?;
    let input = scratch.path().join("input.ll");
    let output = scratch.path().join("output.spv");
    let stdout_path = scratch.path().join("stdout.txt");
    let stderr_path = scratch.path().join("stderr.txt");
    fs::write(&input, &source.air_ll)
        .map_err(|error| format!("write {}: {error}", input.display()))?;
    let mut command = Command::new(binary);
    command
        .arg(&input)
        .arg(&output)
        .arg("--stage")
        .arg(stage)
        .args(&options.translator_options)
        .env("METAL2VULKAN_REPRO_DIR", scratch.path().join("repros"))
        .stdout(Stdio::from(fs::File::create(&stdout_path).map_err(
            |error| format!("create {}: {error}", stdout_path.display()),
        )?))
        .stderr(Stdio::from(fs::File::create(&stderr_path).map_err(
            |error| format!("create {}: {error}", stderr_path.display()),
        )?));
    configure_process_group(&mut command);
    let mut child = command
        .spawn()
        .map_err(|error| format!("spawn {}: {error}", binary.display()))?;
    let completed = child
        .wait_timeout(options.timeout)
        .map_err(|error| format!("wait for {}: {error}", binary.display()))?;
    if completed.is_none() {
        terminate_child(&mut child);
        return Ok(TranslationResult {
            status: TranslationStatus::ToolFailure,
            spv_b64: None,
            spv_sha256: None,
            stderr: format!("timeout after {} seconds", options.timeout.as_secs()),
        });
    }
    let status = completed.expect("checked above");
    let stdout = fs::read(&stdout_path)
        .map_err(|error| format!("read {}: {error}", stdout_path.display()))?;
    let stderr = fs::read(&stderr_path)
        .map_err(|error| format!("read {}: {error}", stderr_path.display()))?;
    let stderr = String::from_utf8_lossy(&stderr).into_owned();
    let spv = fs::read(&output).ok().filter(|bytes| !bytes.is_empty());
    let translation_status = if status.success() && spv.is_some() {
        TranslationStatus::Success
    } else if spv.is_some() && stderr.contains("spirv-val failed") {
        TranslationStatus::Invalid
    } else if status.code() == Some(2) {
        TranslationStatus::Fallback
    } else {
        TranslationStatus::ToolFailure
    };
    let spv_sha256 = spv.as_deref().map(sha256_bytes);
    let spv_b64 = spv
        .as_deref()
        .map(|bytes| base64::engine::general_purpose::STANDARD.encode(bytes));
    let mut diagnostics = stderr;
    if !stdout.is_empty() {
        diagnostics.push_str(&String::from_utf8_lossy(&stdout));
    }
    Ok(TranslationResult {
        status: translation_status,
        spv_b64,
        spv_sha256,
        stderr: diagnostics,
    })
}

#[cfg(unix)]
fn configure_process_group(command: &mut Command) {
    use std::os::unix::process::CommandExt as _;
    command.process_group(0);
}

#[cfg(not(unix))]
fn configure_process_group(_command: &mut Command) {}

#[cfg(unix)]
fn terminate_child(child: &mut Child) {
    unsafe extern "C" {
        fn kill(process: i32, signal: i32) -> i32;
    }
    let group = -(child.id() as i32);
    unsafe {
        let _ = kill(group, 15);
    }
    if child
        .wait_timeout(Duration::from_millis(100))
        .ok()
        .flatten()
        .is_none()
    {
        unsafe {
            let _ = kill(group, 9);
        }
        let _ = child.wait();
    }
}

#[cfg(not(unix))]
fn terminate_child(child: &mut Child) {
    let _ = child.kill();
    let _ = child.wait();
}

fn classify(old: &TranslationResult, new: &TranslationResult) -> AbClassification {
    match (old.status, new.status) {
        (TranslationStatus::Success, TranslationStatus::Success) => {
            if old.spv_b64 == new.spv_b64 {
                AbClassification::UnchangedSpirv
            } else {
                AbClassification::ChangedSpirv
            }
        }
        (TranslationStatus::Fallback, TranslationStatus::Success) => {
            AbClassification::FallbackToSuccess
        }
        (TranslationStatus::Success, TranslationStatus::Fallback) => {
            AbClassification::SuccessToFallback
        }
        (TranslationStatus::Success, TranslationStatus::Invalid) => {
            AbClassification::ValidToInvalid
        }
        _ => AbClassification::TimeoutOrToolFailure,
    }
}

fn validate_translation_result(result: &TranslationResult) -> Result<(), String> {
    match result.status {
        TranslationStatus::Success | TranslationStatus::Invalid => {
            let encoded = result
                .spv_b64
                .as_deref()
                .ok_or_else(|| "success/invalid result has no SPIR-V payload".to_string())?;
            let bytes = base64::engine::general_purpose::STANDARD
                .decode(encoded)
                .map_err(|error| format!("invalid SPIR-V base64: {error}"))?;
            let expected = result
                .spv_sha256
                .as_deref()
                .ok_or_else(|| "success/invalid result has no SPIR-V digest".to_string())?;
            if sha256_bytes(&bytes) != expected {
                return Err("SPIR-V payload digest mismatch".into());
            }
        }
        TranslationStatus::Fallback | TranslationStatus::ToolFailure => {
            if result.spv_b64.is_some() || result.spv_sha256.is_some() {
                return Err("fallback/tool-failure result contains SPIR-V".into());
            }
        }
    }
    Ok(())
}

fn translator_environment() -> Vec<(String, String)> {
    translator_environment_from(std::env::vars())
}

fn translator_environment_from(
    environment: impl IntoIterator<Item = (String, String)>,
) -> Vec<(String, String)> {
    let product_names = metal2vulkan::env_vars::REGISTRY
        .iter()
        .map(|variable| variable.name)
        .filter(|name| !name.contains('<') && *name != "METAL2VULKAN_REPRO_DIR")
        .collect::<HashSet<_>>();
    let mut values = environment
        .into_iter()
        .filter(|(name, _)| product_names.contains(name.as_str()))
        .collect::<Vec<_>>();
    values.sort();
    values
}

fn spirv_val_identity() -> Result<String, String> {
    let path = if let Some(path) = std::env::var_os("METAL2VULKAN_SPIRV_VAL") {
        PathBuf::from(path)
    } else {
        [
            "/opt/homebrew/opt/llvm/bin/spirv-val",
            "/usr/local/opt/llvm/bin/spirv-val",
            "/opt/homebrew/bin/spirv-val",
            "/usr/local/bin/spirv-val",
        ]
        .into_iter()
        .map(PathBuf::from)
        .find(|candidate| candidate.is_file())
        .or_else(|| find_on_path("spirv-val"))
        .ok_or_else(|| "spirv-val not found in product tool search path".to_string())?
    };
    let hash = sha256_file(&path)
        .map_err(|error| format!("hash external tool {}: {error}", path.display()))?;
    Ok(format!("{}:{hash}", path.display()))
}

fn find_on_path(name: &str) -> Option<PathBuf> {
    std::env::var_os("PATH").and_then(|path| {
        std::env::split_paths(&path)
            .map(|directory| directory.join(name))
            .find(|candidate| candidate.is_file())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allowlists_apply_only_to_their_exact_transition() {
        let hash = "11".repeat(32);
        let spv = [hash.clone()].into_iter().collect();
        let fallback = [hash.clone()].into_iter().collect();
        assert!(change_is_allowed(
            AbClassification::ChangedSpirv,
            &hash,
            &spv,
            &HashSet::new()
        ));
        assert!(change_is_allowed(
            AbClassification::FallbackToSuccess,
            &hash,
            &HashSet::new(),
            &fallback
        ));
        assert!(!change_is_allowed(
            AbClassification::SuccessToFallback,
            &hash,
            &spv,
            &fallback
        ));
    }

    #[test]
    fn pair_classification_covers_required_transitions() {
        fn result(status: TranslationStatus, bytes: Option<&str>) -> TranslationResult {
            TranslationResult {
                status,
                spv_b64: bytes.map(str::to_string),
                spv_sha256: None,
                stderr: String::new(),
            }
        }
        assert_eq!(
            classify(
                &result(TranslationStatus::Success, Some("a")),
                &result(TranslationStatus::Success, Some("a"))
            ),
            AbClassification::UnchangedSpirv
        );
        assert_eq!(
            classify(
                &result(TranslationStatus::Success, Some("a")),
                &result(TranslationStatus::Success, Some("b"))
            ),
            AbClassification::ChangedSpirv
        );
        assert_eq!(
            classify(
                &result(TranslationStatus::Fallback, None),
                &result(TranslationStatus::Success, Some("b"))
            ),
            AbClassification::FallbackToSuccess
        );
        assert_eq!(
            classify(
                &result(TranslationStatus::Success, Some("a")),
                &result(TranslationStatus::Invalid, Some("b"))
            ),
            AbClassification::ValidToInvalid
        );
    }

    #[test]
    fn strict_policies_reject_unlisted_changes() {
        let hash = "22".repeat(32);
        let empty = HashSet::new();
        assert!(!policy_allows(
            AbClassification::ChangedSpirv,
            &hash,
            true,
            false,
            &empty,
            &empty
        ));
        assert!(!policy_allows(
            AbClassification::ChangedSpirv,
            &hash,
            false,
            true,
            &empty,
            &empty
        ));
        assert!(policy_allows(
            AbClassification::UnchangedSpirv,
            &hash,
            true,
            true,
            &empty,
            &empty
        ));
    }

    #[test]
    fn cached_payload_digest_is_checked() {
        let result = TranslationResult {
            status: TranslationStatus::Success,
            spv_b64: Some(base64::engine::general_purpose::STANDARD.encode(b"spirv")),
            spv_sha256: Some(sha256_bytes(b"different")),
            stderr: String::new(),
        };
        assert!(validate_translation_result(&result).is_err());
    }

    #[test]
    fn cache_environment_includes_product_inputs_not_validation_controls() {
        let values = translator_environment_from([
            ("METAL2VULKAN_RELOOPER_MAX_BLOCKS".into(), "12".into()),
            ("METAL2VULKAN_HARVEST_LIMIT".into(), "1".into()),
            ("METAL2VULKAN_CORPUS_DIR".into(), "/tmp/corpus".into()),
            ("METAL2VULKAN_REPRO_DIR".into(), "/tmp/repros".into()),
        ]);
        assert_eq!(
            values,
            vec![("METAL2VULKAN_RELOOPER_MAX_BLOCKS".into(), "12".into())]
        );
    }

    #[cfg(unix)]
    #[test]
    fn translation_scratch_is_removed_for_every_process_outcome() {
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
                                && name.ends_with("-corpus-ab")
                        })
                })
                .collect()
        }

        let outer = ScratchDir::new("ab-process-outcomes").unwrap();
        let binary = outer.path().join("translator.sh");
        let source = SourceRow {
            air_sha256: sha256_bytes(b"air"),
            stage: "Kernel".into(),
            entry: "main".into(),
            air_ll: "air".into(),
            blob_b64: None,
            lib_sha256: "22".repeat(32),
            label: "test".into(),
        };
        let mut options = AbOptions {
            corpus_root: outer.path().into(),
            old_binary: binary.clone(),
            new_binary: binary.clone(),
            selection: AbSelection::default(),
            translator_options: vec![],
            cache_dir: outer.path().join("cache"),
            timeout: Duration::from_secs(2),
            expect_no_change: false,
            fail_on_unlisted_change: false,
            spv_allowlist: HashSet::new(),
            fallback_to_success_allowlist: HashSet::new(),
        };
        let baseline = scratch_paths();
        for (script, timeout, expected) in [
            (
                "#!/bin/sh\nprintf 'spv' > \"$2\"\n",
                Duration::from_secs(2),
                TranslationStatus::Success,
            ),
            (
                "#!/bin/sh\nexit 2\n",
                Duration::from_secs(2),
                TranslationStatus::Fallback,
            ),
            (
                "#!/bin/sh\nsleep 2\n",
                Duration::from_millis(30),
                TranslationStatus::ToolFailure,
            ),
            (
                "#!/bin/sh\nkill -TERM $$\n",
                Duration::from_secs(2),
                TranslationStatus::ToolFailure,
            ),
        ] {
            fs::write(&binary, script).unwrap();
            let mut permissions = fs::metadata(&binary).unwrap().permissions();
            permissions.set_mode(0o700);
            fs::set_permissions(&binary, permissions).unwrap();
            options.timeout = timeout;
            assert_eq!(
                translate(&source, &binary, "kernel", &options)
                    .unwrap()
                    .status,
                expected
            );
            assert_eq!(scratch_paths(), baseline);
        }
    }
}
