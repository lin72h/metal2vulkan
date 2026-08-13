use crate::case::AuthoredCase;
use crate::jsonl::to_sorted_json_string;
use crate::observation::{Backend, CandidateObservation, MetalObservation};
use crate::review::ReviewNote;
use crate::source::{find_source, shard_index_for_hash, shard_index_from_path, shard_name};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::collections::{BTreeMap, HashSet};
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

pub struct CorpusStore {
    root: PathBuf,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PutCaseResult {
    Added,
    Replaced { old_case_id: String },
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct TransactionManifest {
    entries: Vec<TransactionEntry>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct TransactionEntry {
    staged: String,
    target: String,
}

impl CorpusStore {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn recover_transactions(&self) -> Result<(), String> {
        let transactions = self.root.join(".transactions");
        let Ok(entries) = fs::read_dir(&transactions) else {
            return Ok(());
        };
        let mut paths = entries
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .filter(|path| path.is_dir())
            .collect::<Vec<_>>();
        paths.sort();
        for path in paths {
            let manifest_path = path.join("manifest.json");
            if !manifest_path.is_file() {
                fs::remove_dir_all(&path)
                    .map_err(|error| format!("remove incomplete {}: {error}", path.display()))?;
                continue;
            }
            let manifest: TransactionManifest = serde_json::from_slice(
                &fs::read(&manifest_path)
                    .map_err(|error| format!("read {}: {error}", manifest_path.display()))?,
            )
            .map_err(|error| format!("parse {}: {error}", manifest_path.display()))?;
            apply_transaction(&path, &self.root, &manifest)?;
        }
        Ok(())
    }

    pub fn read_all_cases(&self) -> Result<Vec<AuthoredCase>, String> {
        let mut rows = Vec::new();
        for path in shard_paths(&self.root.join("cases"))? {
            let index = shard_index_from_path(&path)?;
            let shard_rows = read_jsonl_if_exists(&path)?;
            validate_cases(&shard_rows, Some(index))?;
            rows.extend(shard_rows);
        }
        validate_cases(&rows, None)?;
        Ok(rows)
    }

    pub fn read_case_shard(&self, index: usize) -> Result<Vec<AuthoredCase>, String> {
        let rows = read_jsonl_if_exists(&self.root.join("cases").join(shard_name(index)))?;
        validate_cases(&rows, Some(index))?;
        Ok(rows)
    }

    pub fn find_case(&self, case_id: &str) -> Result<Option<AuthoredCase>, String> {
        Ok(self
            .read_all_cases()?
            .into_iter()
            .find(|case| case.case_id == case_id))
    }

    pub fn put_case(&self, case: AuthoredCase) -> Result<PutCaseResult, String> {
        self.recover_transactions()?;
        let computed = case.computed_case_id()?;
        if case.case_id != computed {
            return Err(format!(
                "case {} case_id mismatch: manifest={} computed={computed}",
                case.name, case.case_id
            ));
        }
        case.validate_literal_resources()
            .map_err(|errors| errors.join("; "))?;
        let review_air = case.air_sha256.clone();
        let index = shard_index_for_hash(&case.air_sha256)?;
        let mut cases = self.read_case_shard(index)?;
        if cases.iter().any(|row| row.case_id == case.case_id) {
            return Err(format!("duplicate semantic case_id {}", case.case_id));
        }
        let old = cases
            .iter()
            .position(|row| row.air_sha256 == case.air_sha256 && row.name == case.name)
            .map(|position| cases.remove(position));
        let old_case_id = old.as_ref().map(|row| row.case_id.as_str());
        cases.push(case);
        cases.sort_by(|left, right| {
            left.air_sha256
                .cmp(&right.air_sha256)
                .then_with(|| left.name.cmp(&right.name))
        });
        self.rewrite_aligned(index, &cases, old_case_id, Some(&review_air))?;
        Ok(match old {
            Some(old) => PutCaseResult::Replaced {
                old_case_id: old.case_id,
            },
            None => PutCaseResult::Added,
        })
    }

    pub fn delete_named_case(&self, air_sha256: &str, name: &str) -> Result<String, String> {
        self.recover_transactions()?;
        let index = shard_index_for_hash(air_sha256)?;
        let mut cases = self.read_case_shard(index)?;
        let position = cases
            .iter()
            .position(|row| row.air_sha256 == air_sha256 && row.name == name)
            .ok_or_else(|| format!("no case named {name:?} for AIR {air_sha256}"))?;
        let old = cases.remove(position);
        self.rewrite_aligned(index, &cases, Some(&old.case_id), None)?;
        Ok(old.case_id)
    }

    fn rewrite_aligned(
        &self,
        index: usize,
        cases: &[AuthoredCase],
        cascaded_case_id: Option<&str>,
        clear_review_air: Option<&str>,
    ) -> Result<(), String> {
        let name = shard_name(index);
        let case_path = PathBuf::from("cases").join(&name);
        let metal_path = PathBuf::from("observations/metal").join(&name);
        let moltenvk_path = PathBuf::from("observations/moltenvk").join(&name);
        let vulkan_path = PathBuf::from("observations/vulkan").join(&name);
        let review_path = PathBuf::from("reviews").join(&name);
        let mut metal: Vec<MetalObservation> = read_jsonl_if_exists(&self.root.join(&metal_path))?;
        let mut moltenvk: Vec<CandidateObservation> =
            read_jsonl_if_exists(&self.root.join(&moltenvk_path))?;
        let mut vulkan: Vec<CandidateObservation> =
            read_jsonl_if_exists(&self.root.join(&vulkan_path))?;
        let mut reviews: Vec<ReviewNote> = read_jsonl_if_exists(&self.root.join(&review_path))?;
        validate_aligned_observation_rows(&metal, index)?;
        validate_aligned_observation_rows(&moltenvk, index)?;
        validate_aligned_observation_rows(&vulkan, index)?;
        if moltenvk.iter().any(|row| row.backend != Backend::Moltenvk)
            || vulkan.iter().any(|row| row.backend != Backend::Vulkan)
        {
            return Err("candidate observation is stored in the wrong backend directory".into());
        }
        reject_duplicate_metal_slots(&metal)?;
        reject_duplicate_candidate_slots(&moltenvk)?;
        reject_duplicate_candidate_slots(&vulkan)?;
        validate_review_rows(&reviews, Some(index))?;
        if let Some(case_id) = cascaded_case_id {
            metal.retain(|row| row.case_id != case_id);
            moltenvk.retain(|row| row.case_id != case_id);
            vulkan.retain(|row| row.case_id != case_id);
        }
        if let Some(air_sha256) = clear_review_air {
            reviews.retain(|row| row.air_sha256 != air_sha256);
        }
        self.commit_files([
            (case_path, canonical_jsonl(cases)?),
            (metal_path, canonical_jsonl(&metal)?),
            (moltenvk_path, canonical_jsonl(&moltenvk)?),
            (vulkan_path, canonical_jsonl(&vulkan)?),
            (review_path, canonical_jsonl(&reviews)?),
        ])
    }

    pub fn upsert_review(&self, note: ReviewNote) -> Result<(), String> {
        self.recover_transactions()?;
        note.validate()?;
        if find_source(&self.root, &note.air_sha256)?.is_none() {
            return Err(format!("review references unknown AIR {}", note.air_sha256));
        }
        if self
            .read_all_cases()?
            .iter()
            .any(|case| case.air_sha256 == note.air_sha256)
        {
            return Err(format!(
                "AIR {} already has an authored case; a review note cannot replace evidence",
                note.air_sha256
            ));
        }
        let index = shard_index_for_hash(&note.air_sha256)?;
        let relative = PathBuf::from("reviews").join(shard_name(index));
        let mut rows: Vec<ReviewNote> = read_jsonl_if_exists(&self.root.join(&relative))?;
        validate_review_rows(&rows, Some(index))?;
        rows.retain(|row| row.air_sha256 != note.air_sha256);
        rows.push(note);
        rows.sort_by(|left, right| left.air_sha256.cmp(&right.air_sha256));
        self.commit_files([(relative, canonical_jsonl(&rows)?)])
    }

    /// Upsert review notes shard-by-shard without rebuilding the disposable index between rows.
    ///
    /// This preserves the same validation and canonical ordering as [`Self::upsert_review`], while
    /// allowing deterministic corpus triage to commit a selected batch efficiently. Review notes
    /// remain queue annotations and never become semantic evidence.
    pub fn upsert_reviews(&self, notes: Vec<ReviewNote>) -> Result<(), String> {
        self.recover_transactions()?;
        if notes.is_empty() {
            return Ok(());
        }

        let case_airs = self
            .read_all_cases()?
            .into_iter()
            .map(|case| case.air_sha256)
            .collect::<HashSet<_>>();
        let mut by_shard = BTreeMap::<usize, Vec<ReviewNote>>::new();
        let mut selected = HashSet::new();
        for note in notes {
            note.validate()?;
            if !selected.insert(note.air_sha256.clone()) {
                return Err(format!(
                    "duplicate review note for AIR {} in batch",
                    note.air_sha256
                ));
            }
            if case_airs.contains(&note.air_sha256) {
                return Err(format!(
                    "AIR {} already has an authored case; a review note cannot replace evidence",
                    note.air_sha256
                ));
            }
            by_shard
                .entry(shard_index_for_hash(&note.air_sha256)?)
                .or_default()
                .push(note);
        }

        for (index, shard_notes) in by_shard {
            let source_hashes = crate::source::read_source_shard(
                &crate::source::source_shard_path(&self.root, index),
            )?
            .into_iter()
            .map(|source| source.air_sha256)
            .collect::<HashSet<_>>();
            for note in &shard_notes {
                if !source_hashes.contains(&note.air_sha256)
                    && crate::source::public_sources()?
                        .iter()
                        .all(|source| source.air_sha256 != note.air_sha256)
                {
                    return Err(format!("review references unknown AIR {}", note.air_sha256));
                }
            }

            let relative = PathBuf::from("reviews").join(shard_name(index));
            let mut rows: Vec<ReviewNote> = read_jsonl_if_exists(&self.root.join(&relative))?;
            validate_review_rows(&rows, Some(index))?;
            let replacement_hashes = shard_notes
                .iter()
                .map(|note| note.air_sha256.as_str())
                .collect::<HashSet<_>>();
            rows.retain(|row| !replacement_hashes.contains(row.air_sha256.as_str()));
            rows.extend(shard_notes);
            rows.sort_by(|left, right| left.air_sha256.cmp(&right.air_sha256));
            validate_review_rows(&rows, Some(index))?;
            self.commit_files([(relative, canonical_jsonl(&rows)?)])?;
        }
        Ok(())
    }

    pub fn read_reviews(&self) -> Result<Vec<ReviewNote>, String> {
        let rows = self.read_reviews_for_index()?;
        for row in &rows {
            if find_source(&self.root, &row.air_sha256)?.is_none() {
                return Err(format!("review references unknown AIR {}", row.air_sha256));
            }
        }
        Ok(rows)
    }

    /// Read and structurally validate review shards without resolving source bodies.
    ///
    /// The index loader uses its `reviews.air_sha256 -> sources.air_sha256` foreign key for the
    /// membership check, avoiding one large source-shard parse per review note.
    pub(crate) fn read_reviews_for_index(&self) -> Result<Vec<ReviewNote>, String> {
        let mut rows = Vec::new();
        for path in shard_paths(&self.root.join("reviews"))? {
            let index = shard_index_from_path(&path)?;
            let shard_rows = read_jsonl_if_exists(&path)?;
            validate_review_rows(&shard_rows, Some(index))?;
            rows.extend(shard_rows);
        }
        validate_review_rows(&rows, None)?;
        Ok(rows)
    }

    pub fn upsert_metal(&self, row: MetalObservation) -> Result<(), String> {
        self.recover_transactions()?;
        row.validate_content()?;
        let index = shard_index_for_hash(&row.air_sha256)?;
        self.validate_observation_case(index, &row.case_id, &row.air_sha256, &row.input_sha256)?;
        let relative = PathBuf::from("observations/metal").join(shard_name(index));
        let mut rows: Vec<MetalObservation> = read_jsonl_if_exists(&self.root.join(&relative))?;
        validate_aligned_observation_rows(&rows, index)?;
        reject_duplicate_metal_slots(&rows)?;
        rows.retain(|old| {
            !(old.case_id == row.case_id && old.environment_id == row.environment_id)
        });
        rows.push(row);
        rows.sort_by(|left, right| {
            left.case_id
                .cmp(&right.case_id)
                .then_with(|| left.environment_id.cmp(&right.environment_id))
        });
        self.commit_files([(relative, canonical_jsonl(&rows)?)])
    }

    pub fn upsert_candidate(&self, row: CandidateObservation) -> Result<(), String> {
        self.recover_transactions()?;
        row.validate_content()?;
        let index = shard_index_for_hash(&row.air_sha256)?;
        self.validate_observation_case(index, &row.case_id, &row.air_sha256, &row.input_sha256)?;
        let relative = PathBuf::from("observations")
            .join(row.backend.directory())
            .join(shard_name(index));
        let mut rows: Vec<CandidateObservation> = read_jsonl_if_exists(&self.root.join(&relative))?;
        validate_aligned_observation_rows(&rows, index)?;
        if rows.iter().any(|existing| existing.backend != row.backend) {
            return Err("candidate observation is stored in the wrong backend directory".into());
        }
        reject_duplicate_candidate_slots(&rows)?;
        rows.retain(|old| {
            !(old.case_id == row.case_id
                && old.backend == row.backend
                && old.environment_id == row.environment_id)
        });
        rows.push(row);
        rows.sort_by(|left, right| {
            left.case_id
                .cmp(&right.case_id)
                .then_with(|| left.environment_id.cmp(&right.environment_id))
        });
        self.commit_files([(relative, canonical_jsonl(&rows)?)])
    }

    fn validate_observation_case(
        &self,
        index: usize,
        case_id: &str,
        air_sha256: &str,
        input_sha256: &str,
    ) -> Result<(), String> {
        let case = self
            .read_case_shard(index)?
            .into_iter()
            .find(|case| case.case_id == case_id)
            .ok_or_else(|| format!("observation references unknown case_id {case_id}"))?;
        if case.air_sha256 != air_sha256 {
            return Err(format!(
                "observation AIR {air_sha256} does not match case AIR {}",
                case.air_sha256
            ));
        }
        let expected = case.computed_input_sha256()?;
        if expected != input_sha256 {
            return Err(format!(
                "observation input {input_sha256} does not match case input {expected}"
            ));
        }
        Ok(())
    }

    pub fn read_metal(&self) -> Result<Vec<MetalObservation>, String> {
        let rows: Vec<MetalObservation> =
            read_aligned_observations(&self.root.join("observations/metal"))?;
        self.validate_observation_cases(
            rows.iter()
                .map(|row| (&row.case_id, &row.air_sha256, &row.input_sha256)),
        )?;
        reject_duplicate_metal_slots(&rows)?;
        Ok(rows)
    }

    pub fn read_candidates(&self, backend: Backend) -> Result<Vec<CandidateObservation>, String> {
        let rows: Vec<CandidateObservation> =
            read_aligned_observations(&self.root.join("observations").join(backend.directory()))?;
        for row in &rows {
            row.validate_content()?;
            if row.backend != backend {
                return Err(format!(
                    "{} observation found in {} directory",
                    row.backend.directory(),
                    backend.directory()
                ));
            }
        }
        self.validate_observation_cases(
            rows.iter()
                .map(|row| (&row.case_id, &row.air_sha256, &row.input_sha256)),
        )?;
        reject_duplicate_candidate_slots(&rows)?;
        Ok(rows)
    }

    fn validate_observation_cases<'a>(
        &self,
        rows: impl IntoIterator<Item = (&'a String, &'a String, &'a String)>,
    ) -> Result<(), String> {
        let cases = self.read_all_cases()?;
        let cases = cases
            .iter()
            .map(|case| (case.case_id.as_str(), case))
            .collect::<std::collections::HashMap<_, _>>();
        for (case_id, air_sha256, input_sha256) in rows {
            let case = cases
                .get(case_id.as_str())
                .ok_or_else(|| format!("observation references unknown case_id {case_id}"))?;
            if case.air_sha256 != *air_sha256 {
                return Err(format!(
                    "observation AIR {air_sha256} does not match case AIR {}",
                    case.air_sha256
                ));
            }
            let expected = case.computed_input_sha256()?;
            if expected != *input_sha256 {
                return Err(format!(
                    "observation input {input_sha256} does not match case input {expected}"
                ));
            }
        }
        Ok(())
    }

    fn commit_files<const N: usize>(&self, files: [(PathBuf, Vec<u8>); N]) -> Result<(), String> {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|error| format!("system clock: {error}"))?
            .as_nanos();
        let transaction = self
            .root
            .join(".transactions")
            .join(format!("{}-{nonce}", std::process::id()));
        fs::create_dir_all(&transaction)
            .map_err(|error| format!("create {}: {error}", transaction.display()))?;
        let mut entries = Vec::new();
        for (index, (target, bytes)) in files.into_iter().enumerate() {
            let staged = format!("file-{index:03}");
            let path = transaction.join(&staged);
            let mut file = File::create(&path)
                .map_err(|error| format!("create {}: {error}", path.display()))?;
            file.write_all(&bytes)
                .map_err(|error| format!("write {}: {error}", path.display()))?;
            file.sync_all()
                .map_err(|error| format!("fsync {}: {error}", path.display()))?;
            entries.push(TransactionEntry {
                staged,
                target: target
                    .to_str()
                    .ok_or_else(|| format!("non-UTF-8 target {}", target.display()))?
                    .into(),
            });
        }
        let manifest = TransactionManifest { entries };
        let manifest_path = transaction.join("manifest.json");
        let manifest_bytes = serde_json::to_vec(&manifest)
            .map_err(|error| format!("serialize transaction: {error}"))?;
        let mut file = File::create(&manifest_path)
            .map_err(|error| format!("create {}: {error}", manifest_path.display()))?;
        file.write_all(&manifest_bytes)
            .map_err(|error| format!("write {}: {error}", manifest_path.display()))?;
        file.sync_all()
            .map_err(|error| format!("fsync {}: {error}", manifest_path.display()))?;
        sync_directory(&transaction)?;
        apply_transaction(&transaction, &self.root, &manifest)
    }
}

fn apply_transaction(
    transaction: &Path,
    root: &Path,
    manifest: &TransactionManifest,
) -> Result<(), String> {
    for entry in &manifest.entries {
        let staged = transaction.join(&entry.staged);
        let target = root.join(&entry.target);
        if !staged.exists() {
            if target.exists() {
                continue;
            }
            return Err(format!(
                "transaction {} lost both staged and target file {}",
                transaction.display(),
                target.display()
            ));
        }
        let parent = target
            .parent()
            .ok_or_else(|| format!("target {} has no parent", target.display()))?;
        fs::create_dir_all(parent)
            .map_err(|error| format!("create {}: {error}", parent.display()))?;
        fs::rename(&staged, &target).map_err(|error| {
            format!(
                "rename {} to {}: {error}",
                staged.display(),
                target.display()
            )
        })?;
        sync_directory(parent)?;
    }
    fs::remove_dir_all(transaction)
        .map_err(|error| format!("remove {}: {error}", transaction.display()))?;
    if let Some(parent) = transaction.parent() {
        sync_directory(parent)?;
    }
    Ok(())
}

fn sync_directory(path: &Path) -> Result<(), String> {
    let file = OpenOptions::new()
        .read(true)
        .open(path)
        .map_err(|error| format!("open directory {}: {error}", path.display()))?;
    file.sync_all()
        .map_err(|error| format!("fsync directory {}: {error}", path.display()))
}

fn canonical_jsonl<T: Serialize>(rows: &[T]) -> Result<Vec<u8>, String> {
    let mut bytes = Vec::new();
    for row in rows {
        let line = to_sorted_json_string(row).map_err(|error| format!("serialize row: {error}"))?;
        bytes.extend_from_slice(line.as_bytes());
        bytes.push(b'\n');
    }
    Ok(bytes)
}

pub fn read_jsonl_if_exists<T: DeserializeOwned>(path: &Path) -> Result<Vec<T>, String> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    let file = File::open(path).map_err(|error| format!("open {}: {error}", path.display()))?;
    let mut rows = Vec::new();
    for (index, line) in BufReader::new(file).lines().enumerate() {
        let line =
            line.map_err(|error| format!("read {}:{}: {error}", path.display(), index + 1))?;
        if line.trim().is_empty() {
            continue;
        }
        rows.push(
            serde_json::from_str(&line)
                .map_err(|error| format!("parse {}:{}: {error}", path.display(), index + 1))?,
        );
    }
    Ok(rows)
}

fn shard_paths(directory: &Path) -> Result<Vec<PathBuf>, String> {
    let Ok(entries) = fs::read_dir(directory) else {
        return Ok(Vec::new());
    };
    let mut paths = entries
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("shard_") && name.ends_with(".jsonl"))
        })
        .collect::<Vec<_>>();
    paths.sort();
    Ok(paths)
}

trait AlignedObservation {
    fn air_sha256(&self) -> &str;
    fn validate_content(&self) -> Result<(), String>;
}

impl AlignedObservation for MetalObservation {
    fn air_sha256(&self) -> &str {
        &self.air_sha256
    }

    fn validate_content(&self) -> Result<(), String> {
        MetalObservation::validate_content(self)
    }
}

impl AlignedObservation for CandidateObservation {
    fn air_sha256(&self) -> &str {
        &self.air_sha256
    }

    fn validate_content(&self) -> Result<(), String> {
        CandidateObservation::validate_content(self)
    }
}

fn read_aligned_observations<T: DeserializeOwned + AlignedObservation>(
    directory: &Path,
) -> Result<Vec<T>, String> {
    let mut rows = Vec::new();
    for path in shard_paths(directory)? {
        let expected = shard_index_from_path(&path)?;
        let shard_rows: Vec<T> = read_jsonl_if_exists(&path)?;
        validate_aligned_observation_rows(&shard_rows, expected)?;
        rows.extend(shard_rows);
    }
    Ok(rows)
}

fn validate_aligned_observation_rows<T: AlignedObservation>(
    rows: &[T],
    expected: usize,
) -> Result<(), String> {
    for row in rows {
        row.validate_content()?;
        let actual = shard_index_for_hash(row.air_sha256())?;
        if actual != expected {
            return Err(format!(
                "observation for AIR {} belongs in shard {}, not {}",
                row.air_sha256(),
                actual,
                expected
            ));
        }
    }
    Ok(())
}

fn reject_duplicate_metal_slots(rows: &[MetalObservation]) -> Result<(), String> {
    let mut slots = std::collections::HashSet::new();
    for row in rows {
        if !slots.insert((&row.case_id, &row.environment_id)) {
            return Err(format!(
                "duplicate Metal experiment slot ({}, {})",
                row.case_id, row.environment_id
            ));
        }
    }
    Ok(())
}

fn reject_duplicate_candidate_slots(rows: &[CandidateObservation]) -> Result<(), String> {
    let mut slots = std::collections::HashSet::new();
    for row in rows {
        if !slots.insert((&row.case_id, row.backend, &row.environment_id)) {
            return Err(format!(
                "duplicate candidate experiment slot ({}, {:?}, {})",
                row.case_id, row.backend, row.environment_id
            ));
        }
    }
    Ok(())
}

fn validate_cases(rows: &[AuthoredCase], expected_shard: Option<usize>) -> Result<(), String> {
    let mut identities = std::collections::HashSet::new();
    let mut names = std::collections::HashSet::new();
    for row in rows {
        row.validate_literal_resources()
            .map_err(|errors| format!("case {}: {}", row.case_id, errors.join("; ")))?;
        let computed = row.computed_case_id()?;
        if computed != row.case_id {
            return Err(format!(
                "case {} identity mismatch: computed {computed}",
                row.case_id
            ));
        }
        if !identities.insert(&row.case_id) {
            return Err(format!("duplicate semantic case_id {}", row.case_id));
        }
        if !names.insert((&row.air_sha256, &row.name)) {
            return Err(format!(
                "duplicate case name {:?} for AIR {}",
                row.name, row.air_sha256
            ));
        }
        if let Some(expected) = expected_shard {
            let actual = shard_index_for_hash(&row.air_sha256)?;
            if actual != expected {
                return Err(format!(
                    "case {} is in shard {}, expected {}",
                    row.case_id, expected, actual
                ));
            }
        }
    }
    Ok(())
}

fn validate_review_rows(rows: &[ReviewNote], expected_shard: Option<usize>) -> Result<(), String> {
    let mut identities = std::collections::HashSet::new();
    for row in rows {
        row.validate()?;
        if !identities.insert(&row.air_sha256) {
            return Err(format!("duplicate review note for AIR {}", row.air_sha256));
        }
        if let Some(expected) = expected_shard {
            let actual = shard_index_for_hash(&row.air_sha256)?;
            if actual != expected {
                return Err(format!(
                    "review for AIR {} is in shard {}, expected {}",
                    row.air_sha256, expected, actual
                ));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::case::{
        BufferResource, Comparison, Dispatch, ExecutionSafety, OutputSelection, ResourceRole, Stage,
    };
    use crate::observation::{CandidateStatus, ComparisonResult, MetalStatus};
    use crate::ScratchDir;

    fn make_case(name: &str, initial: &str) -> AuthoredCase {
        let mut case = AuthoredCase {
            air_sha256: "11".repeat(32),
            case_id: String::new(),
            name: name.into(),
            entry: "main".into(),
            stage: Stage::Kernel,
            buffers: vec![BufferResource {
                binding: 0,
                role: ResourceRole::Output,
                bytes_b64: None,
                initial_bytes_b64: Some(initial.into()),
            }],
            argument_buffer_buffers: vec![],
            threadgroup_memory: vec![],
            imageblock: None,
            fragment_imageblock: None,
            acceleration_structures: vec![],
            visible_function_references: vec![],
            visible_function_tables: vec![],
            intersection_function_tables: vec![],
            argument_buffer_intersection_function_tables: vec![],
            textures: vec![],
            texture_arrays: vec![],
            argument_buffer_textures: vec![],
            samplers: vec![],
            render_targets: vec![],
            depth_stencil: None,
            vertex_inputs: vec![],
            vertex_observation: None,
            kernel_stage_inputs: vec![],
            function_constants: vec![],
            dispatch: Some(Dispatch {
                grid: [1, 1, 1],
                threads_per_threadgroup: [1, 1, 1],
            }),
            draw: None,
            tessellation: None,
            output: OutputSelection::Buffer {
                binding: 0,
                offset: 0,
                length: 4,
            },
            compare: Comparison::Exact,
            execution_safety: ExecutionSafety::LoopFree,
            rationale: None,
            authored_by: None,
        };
        case.case_id = case.computed_case_id().unwrap();
        case
    }

    fn candidate(
        case: &AuthoredCase,
        backend: Backend,
        environment_id: &str,
    ) -> CandidateObservation {
        let output_sha256 = crate::hash::sha256_bytes(&[42, 0, 0, 0]);
        CandidateObservation {
            case_id: case.case_id.clone(),
            air_sha256: case.air_sha256.clone(),
            input_sha256: case.computed_input_sha256().unwrap(),
            golden_output_sha256: output_sha256.clone(),
            spv_sha256: "33".repeat(32),
            translator_fingerprint: "44".repeat(32),
            candidate_output_sha256: output_sha256,
            output_b64: "KgAAAA==".into(),
            backend,
            environment_id: environment_id.into(),
            environment: serde_json::json!({}),
            executor_abi: "v1".into(),
            comparison: ComparisonResult::Exact,
            status: CandidateStatus::Match,
        }
    }

    #[test]
    fn replacing_named_case_cascades_only_old_identity() {
        let scratch = ScratchDir::new("store-cascade").unwrap();
        let store = CorpusStore::new(scratch.path());
        let first = make_case("slot", "AAAAAA==");
        let sibling = make_case("sibling", "AQAAAA==");
        store.put_case(first.clone()).unwrap();
        store.put_case(sibling.clone()).unwrap();
        for case in [&first, &sibling] {
            store
                .upsert_metal(MetalObservation {
                    case_id: case.case_id.clone(),
                    air_sha256: case.air_sha256.clone(),
                    input_sha256: case.computed_input_sha256().unwrap(),
                    metal_output_sha256: crate::hash::sha256_bytes(&[42, 0, 0, 0]),
                    output_b64: "KgAAAA==".into(),
                    environment_id: "env".into(),
                    environment: serde_json::json!({}),
                    oracle_abi: "v1".into(),
                    status: MetalStatus::Qualified,
                })
                .unwrap();
            store
                .upsert_candidate(candidate(case, Backend::Moltenvk, "mvk-env"))
                .unwrap();
            store
                .upsert_candidate(candidate(case, Backend::Vulkan, "vk-env"))
                .unwrap();
        }

        let replacement = make_case("slot", "AgAAAA==");
        assert_eq!(
            store.put_case(replacement.clone()).unwrap(),
            PutCaseResult::Replaced {
                old_case_id: first.case_id.clone()
            }
        );
        let cases = store.read_all_cases().unwrap();
        assert!(cases.iter().any(|row| row.case_id == replacement.case_id));
        assert!(cases.iter().any(|row| row.case_id == sibling.case_id));
        assert!(!cases.iter().any(|row| row.case_id == first.case_id));
        let metal = store.read_metal().unwrap();
        assert_eq!(metal.len(), 1);
        assert_eq!(metal[0].case_id, sibling.case_id);
        for backend in [Backend::Moltenvk, Backend::Vulkan] {
            let rows = store.read_candidates(backend).unwrap();
            assert_eq!(rows.len(), 1);
            assert_eq!(rows[0].case_id, sibling.case_id);
        }
    }

    #[test]
    fn exact_slot_upsert_replaces_instead_of_accumulating() {
        let scratch = ScratchDir::new("slot-upsert").unwrap();
        let store = CorpusStore::new(scratch.path());
        let case = make_case("slot", "AAAAAA==");
        store.put_case(case.clone()).unwrap();
        let mut row = MetalObservation {
            case_id: case.case_id.clone(),
            air_sha256: case.air_sha256.clone(),
            input_sha256: case.computed_input_sha256().unwrap(),
            metal_output_sha256: crate::hash::sha256_bytes(&[42, 0, 0, 0]),
            output_b64: "KgAAAA==".into(),
            environment_id: "env".into(),
            environment: serde_json::json!({}),
            oracle_abi: "v1".into(),
            status: MetalStatus::Qualified,
        };
        store.upsert_metal(row.clone()).unwrap();
        row.metal_output_sha256 = crate::hash::sha256_bytes(&[43, 0, 0, 0]);
        row.output_b64 = "KwAAAA==".into();
        store.upsert_metal(row.clone()).unwrap();
        assert_eq!(store.read_metal().unwrap(), vec![row]);

        let mut candidate = candidate(&case, Backend::Vulkan, "env");
        store.upsert_candidate(candidate.clone()).unwrap();
        candidate.spv_sha256 = "44".repeat(32);
        store.upsert_candidate(candidate.clone()).unwrap();
        assert_eq!(
            store.read_candidates(Backend::Vulkan).unwrap(),
            vec![candidate]
        );
    }

    #[test]
    fn deleting_named_case_cascades_only_its_evidence() {
        let scratch = ScratchDir::new("store-delete").unwrap();
        let store = CorpusStore::new(scratch.path());
        let removed = make_case("removed", "AAAAAA==");
        let kept = make_case("kept", "AQAAAA==");
        store.put_case(removed.clone()).unwrap();
        store.put_case(kept.clone()).unwrap();
        store
            .upsert_candidate(candidate(&removed, Backend::Vulkan, "env"))
            .unwrap();
        store
            .upsert_candidate(candidate(&kept, Backend::Vulkan, "env"))
            .unwrap();

        assert_eq!(
            store
                .delete_named_case(&removed.air_sha256, &removed.name)
                .unwrap(),
            removed.case_id
        );
        assert_eq!(store.read_all_cases().unwrap(), vec![kept.clone()]);
        assert_eq!(
            store.read_candidates(Backend::Vulkan).unwrap(),
            vec![candidate(&kept, Backend::Vulkan, "env")]
        );
    }

    #[test]
    fn prepared_transaction_is_recovered() {
        let scratch = ScratchDir::new("store-recovery").unwrap();
        let store = CorpusStore::new(scratch.path());
        let transaction = scratch.path().join(".transactions/interrupted");
        fs::create_dir_all(&transaction).unwrap();
        fs::write(transaction.join("file-000"), b"recovered").unwrap();
        fs::write(
            transaction.join("manifest.json"),
            serde_json::to_vec(&TransactionManifest {
                entries: vec![TransactionEntry {
                    staged: "file-000".into(),
                    target: "cases/recovered.txt".into(),
                }],
            })
            .unwrap(),
        )
        .unwrap();

        store.recover_transactions().unwrap();
        assert_eq!(
            fs::read(scratch.path().join("cases/recovered.txt")).unwrap(),
            b"recovered"
        );
        assert!(!transaction.exists());
    }

    #[test]
    fn canonical_readers_reject_misaligned_case_and_observation_rows() {
        let case_scratch = ScratchDir::new("misaligned-case").unwrap();
        let case_store = CorpusStore::new(case_scratch.path());
        let case = make_case("slot", "AAAAAA==");
        case_store.put_case(case.clone()).unwrap();
        let actual = shard_index_for_hash(&case.air_sha256).unwrap();
        let wrong = (actual + 1) % crate::source::SHARD_COUNT;
        fs::rename(
            case_scratch.path().join("cases").join(shard_name(actual)),
            case_scratch.path().join("cases").join(shard_name(wrong)),
        )
        .unwrap();
        assert!(case_store
            .read_all_cases()
            .unwrap_err()
            .contains("expected"));

        let observation_scratch = ScratchDir::new("misaligned-observation").unwrap();
        let observation_store = CorpusStore::new(observation_scratch.path());
        observation_store.put_case(case.clone()).unwrap();
        observation_store
            .upsert_metal(MetalObservation {
                case_id: case.case_id.clone(),
                air_sha256: case.air_sha256.clone(),
                input_sha256: case.computed_input_sha256().unwrap(),
                metal_output_sha256: crate::hash::sha256_bytes(&[42, 0, 0, 0]),
                output_b64: "KgAAAA==".into(),
                environment_id: "env".into(),
                environment: serde_json::json!({}),
                oracle_abi: "v1".into(),
                status: MetalStatus::Qualified,
            })
            .unwrap();
        fs::rename(
            observation_scratch
                .path()
                .join("observations/metal")
                .join(shard_name(actual)),
            observation_scratch
                .path()
                .join("observations/metal")
                .join(shard_name(wrong)),
        )
        .unwrap();
        assert!(observation_store
            .read_metal()
            .unwrap_err()
            .contains("belongs in shard"));
    }

    #[test]
    fn review_note_is_exact_slot_annotation_and_clears_when_authored() {
        let scratch = ScratchDir::new("review-note").unwrap();
        let store = CorpusStore::new(scratch.path());
        let source = crate::source::public_sources().unwrap().remove(0);
        let mut note = ReviewNote {
            air_sha256: source.air_sha256.clone(),
            reason: "requires inspection".into(),
            reviewed_by: "agent".into(),
        };
        store.upsert_review(note.clone()).unwrap();
        note.reason = "unsupported semantic input is not yet known".into();
        store.upsert_review(note.clone()).unwrap();
        assert_eq!(store.read_reviews().unwrap(), vec![note.clone()]);

        let mut case = make_case("authored", "AAAAAA==");
        case.air_sha256 = source.air_sha256;
        case.entry = source.entry;
        case.case_id = case.computed_case_id().unwrap();
        store.put_case(case).unwrap();
        assert!(store.read_reviews().unwrap().is_empty());
        assert!(store.upsert_review(note).is_err());
    }
}
