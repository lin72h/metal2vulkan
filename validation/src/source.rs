use crate::hash::sha256_bytes;
use crate::jsonl::to_sorted_json_string;
use base64::Engine as _;
use rusqlite::OptionalExtension;
use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

pub const SHARD_COUNT: usize = 64;
pub const SHARD_BITS: u8 = 6;
pub const DEFAULT_CORPUS_REL: &str = "corpus";
pub const CORPUS_DIR_ENV: &str = "METAL2VULKAN_CORPUS_DIR";

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SourceMergeStats {
    pub affected_shards: usize,
    pub inserted: usize,
    pub replaced: usize,
    pub duplicates: usize,
}

/// Physical I/O performed while resolving an indexed, bounded source selection.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct IndexedSourceReadStats {
    pub rows: usize,
    pub source_shards_opened: usize,
    pub source_bytes_read: u64,
    /// Legacy indexes can lack byte offsets. Only selected hash-derived shards are repaired.
    pub repair_shards_scanned: usize,
    pub repair_bytes_scanned: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct SourceIndexLocation {
    pub air_sha256: String,
    pub stage: String,
    pub entry: String,
    pub label: String,
    pub offset: i64,
    pub length: i64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourceRow {
    pub air_sha256: String,
    pub stage: String,
    pub entry: String,
    pub air_ll: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub blob_b64: Option<String>,
    pub lib_sha256: String,
    pub label: String,
}

#[derive(Deserialize)]
struct AnalysisRow {
    air_sha256: String,
    stage: String,
    entry: String,
    air_ll: String,
    lib_sha256: String,
    label: String,
}

impl SourceRow {
    pub fn validate(&self) -> Result<(), String> {
        let computed = sha256_bytes(self.air_ll.as_bytes());
        if self.air_sha256 != computed {
            return Err(format!(
                "source {} hash mismatch: row={} computed={computed}",
                self.label, self.air_sha256
            ));
        }
        if !matches!(self.stage.as_str(), "Kernel" | "Vertex" | "Fragment") {
            return Err(format!(
                "source {} has invalid stage {:?}",
                self.label, self.stage
            ));
        }
        if self.entry.is_empty() {
            return Err(format!("source {} has empty entry", self.label));
        }
        let (metadata_stage, metadata_entry) = crate::air::stage_entry_from_ll(&self.air_ll)
            .ok_or_else(|| format!("source {} has no stable AIR entry metadata", self.label))?;
        if metadata_stage != self.stage {
            return Err(format!(
                "source {} stage mismatch: row={} metadata={metadata_stage}",
                self.label, self.stage
            ));
        }
        if metadata_entry != self.entry {
            return Err(format!(
                "source {} entry mismatch: row={} metadata={metadata_entry}",
                self.label, self.entry
            ));
        }
        if self.label.trim().is_empty() {
            return Err("source label must not be empty".into());
        }
        if self.lib_sha256 != "owned-synthetic"
            && (self.lib_sha256.len() != 64
                || !self.lib_sha256.bytes().all(|byte| byte.is_ascii_hexdigit())
                || self
                    .lib_sha256
                    .bytes()
                    .any(|byte| byte.is_ascii_uppercase()))
        {
            return Err(format!(
                "source {} lib_sha256 must be lowercase SHA-256",
                self.label
            ));
        }
        if let Some(blob) = &self.blob_b64 {
            let decoded = base64::engine::general_purpose::STANDARD
                .decode(blob)
                .map_err(|error| format!("source {} has invalid blob_b64: {error}", self.label))?;
            if decoded.is_empty() {
                return Err(format!("source {} has empty blob_b64", self.label));
            }
        }
        Ok(())
    }
}

pub fn corpus_root() -> PathBuf {
    std::env::var_os(CORPUS_DIR_ENV)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(DEFAULT_CORPUS_REL))
}

pub fn shard_index_for_hash(hash: &str) -> Result<usize, String> {
    if hash.len() != 64 || !hash.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(format!("invalid SHA-256 {hash:?}"));
    }
    let byte = u8::from_str_radix(&hash[..2], 16)
        .map_err(|error| format!("invalid SHA-256 {hash:?}: {error}"))?;
    Ok((byte >> (8 - SHARD_BITS)) as usize)
}

pub fn shard_name(index: usize) -> String {
    format!("shard_{index:03}.jsonl")
}

pub fn shard_name_for_hash(hash: &str) -> Result<String, String> {
    shard_index_for_hash(hash).map(shard_name)
}

pub fn shard_index_from_path(path: &Path) -> Result<usize, String> {
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| format!("invalid shard path {}", path.display()))?;
    let digits = name
        .strip_prefix("shard_")
        .and_then(|name| name.strip_suffix(".jsonl"))
        .ok_or_else(|| format!("invalid shard filename {name:?}"))?;
    if digits.len() != 3 || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(format!("invalid shard filename {name:?}"));
    }
    let index = digits
        .parse::<usize>()
        .map_err(|error| format!("invalid shard filename {name:?}: {error}"))?;
    if index >= SHARD_COUNT {
        return Err(format!(
            "shard index {index} exceeds permanent {}-bucket mapping",
            SHARD_COUNT
        ));
    }
    Ok(index)
}

pub fn source_shards_dir(root: &Path) -> PathBuf {
    root.join("local/sources")
}

pub fn source_shard_path(root: &Path, index: usize) -> PathBuf {
    source_shards_dir(root).join(shard_name(index))
}

/// Return only hashes absent from the disposable source index.
///
/// A reharvest often rediscovers thousands of already indexed entry modules while collecting newly
/// retained non-entry dependency modules. Consulting SQLite before the shard merger prevents those
/// duplicates from reopening multi-gigabyte source buckets. A missing index conservatively returns
/// every requested hash; the canonical shard merger remains the trust boundary in that case.
pub fn unindexed_source_hashes(
    root: &Path,
    hashes: impl IntoIterator<Item = String>,
) -> Result<std::collections::HashSet<String>, String> {
    let hashes = hashes.into_iter().collect::<std::collections::HashSet<_>>();
    let index = root.join(".index.sqlite");
    if !index.is_file() || hashes.is_empty() {
        return Ok(hashes);
    }
    let connection = rusqlite::Connection::open(&index)
        .map_err(|error| format!("open source index {}: {error}", index.display()))?;
    let mut statement = connection
        .prepare("SELECT 1 FROM sources WHERE air_sha256=?1")
        .map_err(|error| format!("prepare indexed-source membership query: {error}"))?;
    let mut missing = std::collections::HashSet::new();
    for hash in hashes {
        let indexed = statement
            .query_row([&hash], |_| Ok(()))
            .optional()
            .map_err(|error| format!("query indexed source {hash}: {error}"))?
            .is_some();
        if !indexed {
            missing.insert(hash);
        }
    }
    Ok(missing)
}

/// Load the compact set of indexed entry AIR identities without opening source storage.
pub fn indexed_source_hashes(root: &Path) -> Result<std::collections::HashSet<String>, String> {
    let index = root.join(".index.sqlite");
    if !index.is_file() {
        return Ok(std::collections::HashSet::new());
    }
    let connection = rusqlite::Connection::open(&index)
        .map_err(|error| format!("open source index {}: {error}", index.display()))?;
    let mut statement = connection
        .prepare("SELECT air_sha256 FROM sources")
        .map_err(|error| format!("prepare indexed-source identity query: {error}"))?;
    let hashes = statement
        .query_map([], |row| row.get(0))
        .map_err(|error| format!("query indexed source identities: {error}"))?
        .collect::<Result<_, _>>()
        .map_err(|error| format!("read indexed source identities: {error}"))?;
    Ok(hashes)
}

pub fn write_source_shards(
    root: &Path,
    rows: impl IntoIterator<Item = SourceRow>,
) -> Result<(), String> {
    let mut buckets = vec![Vec::new(); SHARD_COUNT];
    for row in rows {
        row.validate()?;
        buckets[shard_index_for_hash(&row.air_sha256)?].push(row);
    }
    fs::create_dir_all(source_shards_dir(root))
        .map_err(|error| format!("create source shards: {error}"))?;
    remove_stale_source_temporaries(root)?;
    for (index, bucket) in buckets.iter_mut().enumerate() {
        write_source_bucket(root, index, bucket)?;
    }
    Ok(())
}

/// Merge harvested AIR into only the hash-aligned shards that receive rows.
///
/// The corpus may be several gigabytes, while one harvest batch commonly adds only a handful of
/// hashes. Reading and rewriting all 64 shards would turn the persistent index into dead weight.
/// Hash alignment makes each affected shard an independent canonical merge unit.
pub fn merge_source_shards(
    root: &Path,
    rows: impl IntoIterator<Item = SourceRow>,
) -> Result<SourceMergeStats, String> {
    let mut additions = std::collections::BTreeMap::<usize, Vec<SourceRow>>::new();
    for row in rows {
        row.validate()?;
        additions
            .entry(shard_index_for_hash(&row.air_sha256)?)
            .or_default()
            .push(row);
    }
    if additions.is_empty() {
        return Ok(SourceMergeStats::default());
    }
    fs::create_dir_all(source_shards_dir(root))
        .map_err(|error| format!("create source shards: {error}"))?;
    remove_stale_source_temporaries(root)?;

    let mut stats = SourceMergeStats {
        affected_shards: additions.len(),
        ..SourceMergeStats::default()
    };
    for (index, new_rows) in additions {
        let path = source_shard_path(root, index);
        let existing = if path.is_file() {
            read_source_shard(&path)?
        } else {
            Vec::new()
        };
        let mut by_hash = existing
            .into_iter()
            .map(|row| (row.air_sha256.clone(), row))
            .collect::<std::collections::BTreeMap<_, _>>();
        let mut changed = false;
        for row in new_rows {
            match by_hash.get(&row.air_sha256) {
                None => {
                    stats.inserted += 1;
                    changed = true;
                    by_hash.insert(row.air_sha256.clone(), row);
                }
                Some(previous) if row.lib_sha256 < previous.lib_sha256 => {
                    stats.replaced += 1;
                    changed = true;
                    by_hash.insert(row.air_sha256.clone(), row);
                }
                Some(_) => stats.duplicates += 1,
            }
        }
        if !changed {
            continue;
        }
        let mut merged = by_hash.into_values().collect::<Vec<_>>();
        let locations = write_source_bucket(root, index, &mut merged)?;
        crate::index::record_source_shard_write(root, index, &locations)?;
    }
    Ok(stats)
}

fn write_source_bucket(
    root: &Path,
    index: usize,
    bucket: &mut [SourceRow],
) -> Result<Vec<SourceIndexLocation>, String> {
    bucket.sort_by(|left, right| left.air_sha256.cmp(&right.air_sha256));
    let path = source_shard_path(root, index);
    let temporary =
        source_shards_dir(root).join(format!(".{}.{}.tmp", shard_name(index), std::process::id()));
    let result = (|| {
        let mut file = File::create(&temporary)
            .map_err(|error| format!("create {}: {error}", temporary.display()))?;
        let mut locations = Vec::with_capacity(bucket.len());
        let mut offset = 0u64;
        for row in bucket {
            let line = to_sorted_json_string(&*row)
                .map_err(|error| format!("serialize {}: {error}", row.air_sha256))?;
            let length = line
                .len()
                .checked_add(1)
                .ok_or_else(|| format!("source row {} length overflow", row.air_sha256))?;
            writeln!(file, "{line}")
                .map_err(|error| format!("write {}: {error}", temporary.display()))?;
            locations.push(SourceIndexLocation {
                air_sha256: row.air_sha256.clone(),
                stage: row.stage.clone(),
                entry: row.entry.clone(),
                label: row.label.clone(),
                offset: offset
                    .try_into()
                    .map_err(|_| format!("source offset in {} is too large", path.display()))?,
                length: length
                    .try_into()
                    .map_err(|_| format!("source row {} is too large", row.air_sha256))?,
            });
            offset = offset
                .checked_add(length as u64)
                .ok_or_else(|| format!("source shard {} size overflow", path.display()))?;
        }
        file.sync_all()
            .map_err(|error| format!("fsync {}: {error}", temporary.display()))?;
        fs::rename(&temporary, &path).map_err(|error| {
            format!(
                "rename {} to {}: {error}",
                temporary.display(),
                path.display()
            )
        })?;
        sync_directory(&source_shards_dir(root))?;
        Ok(locations)
    })();
    match result {
        Ok(locations) => Ok(locations),
        Err(error) => {
            let _ = fs::remove_file(&temporary);
            Err(error)
        }
    }
}

fn remove_stale_source_temporaries(root: &Path) -> Result<(), String> {
    let directory = source_shards_dir(root);
    let entries = fs::read_dir(&directory)
        .map_err(|error| format!("read source shards {}: {error}", directory.display()))?;
    for path in entries.filter_map(Result::ok).map(|entry| entry.path()) {
        let stale = path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.starts_with(".shard_") && name.ends_with(".tmp"));
        if stale {
            fs::remove_file(&path)
                .map_err(|error| format!("remove stale {}: {error}", path.display()))?;
        }
    }
    Ok(())
}

fn sync_directory(path: &Path) -> Result<(), String> {
    File::open(path)
        .and_then(|directory| directory.sync_all())
        .map_err(|error| format!("fsync directory {}: {error}", path.display()))
}

pub fn read_source_shard(path: &Path) -> Result<Vec<SourceRow>, String> {
    let expected_shard = shard_index_from_path(path)?;
    let file = File::open(path).map_err(|error| format!("open {}: {error}", path.display()))?;
    let mut rows = Vec::new();
    for (index, line) in BufReader::new(file).lines().enumerate() {
        let line =
            line.map_err(|error| format!("read {}:{}: {error}", path.display(), index + 1))?;
        if line.trim().is_empty() {
            continue;
        }
        let row: SourceRow = serde_json::from_str(&line)
            .map_err(|error| format!("parse {}:{}: {error}", path.display(), index + 1))?;
        row.validate()
            .map_err(|error| format!("{}:{}: {error}", path.display(), index + 1))?;
        let actual_shard = shard_index_for_hash(&row.air_sha256)?;
        if actual_shard != expected_shard {
            return Err(format!(
                "{}:{}: source {} belongs in shard {}, not {}",
                path.display(),
                index + 1,
                row.air_sha256,
                actual_shard,
                expected_shard
            ));
        }
        rows.push(row);
    }
    Ok(rows)
}

/// Stream the authoring-relevant projection of a canonical harvested shard.
///
/// Harvest performs full hash, AIR metadata, and blob validation before writing canonical shards.
/// Corpus-wide analysis does not need to allocate or base64-decode each retained AIR blob again;
/// it reads the AIR text and indexed identity while Serde skips `blob_b64`. Callers process one row
/// before the reader advances, so memory is bounded by the largest source row rather than corpus
/// size. Use [`read_source_shard`] at ingestion or other trust boundaries.
pub fn for_each_source_shard_analysis(
    path: &Path,
    mut consume: impl FnMut(SourceRow) -> Result<(), String>,
) -> Result<(), String> {
    let expected_shard = shard_index_from_path(path)?;
    let file = File::open(path).map_err(|error| format!("open {}: {error}", path.display()))?;
    let mut reader = BufReader::new(file);
    let mut line = Vec::new();
    let mut line_number = 0usize;
    loop {
        line.clear();
        if reader
            .read_until(b'\n', &mut line)
            .map_err(|error| format!("read {}: {error}", path.display()))?
            == 0
        {
            break;
        }
        line_number += 1;
        if line.iter().all(u8::is_ascii_whitespace) {
            continue;
        }
        consume(
            analysis_source_from_json(&line, expected_shard)
                .map_err(|error| format!("parse {}:{line_number}: {error}", path.display()))?,
        )
        .map_err(|error| format!("{}:{line_number}: {error}", path.display()))?;
    }
    Ok(())
}

/// Read selected source projections through one SQLite connection and at most one open handle per
/// selected shard.
///
/// The source index supplies exact JSONL byte locations. Legacy or stale locations are repaired by
/// scanning only the hash-derived selected shards before retrying. Harvest already validated the
/// canonical rows, so analysis skips the retained binary blob and does not repeat ingestion work.
pub fn for_each_indexed_source_analysis(
    root: &Path,
    index_path: &Path,
    hashes: &[String],
    consume: impl FnMut(SourceRow) -> Result<(), String>,
) -> Result<(), String> {
    for_each_indexed_source_analysis_with_stats(root, index_path, hashes, consume).map(|_| ())
}

pub fn for_each_indexed_source_analysis_with_stats(
    root: &Path,
    index_path: &Path,
    hashes: &[String],
    mut consume: impl FnMut(SourceRow) -> Result<(), String>,
) -> Result<IndexedSourceReadStats, String> {
    enum Location {
        Public(SourceRow),
        Private {
            hash: String,
            shard: usize,
            offset: u64,
            length: usize,
        },
    }

    let public = public_sources()?
        .into_iter()
        .map(|source| (source.air_sha256.clone(), source))
        .collect::<std::collections::HashMap<_, _>>();
    let mut locations = Vec::new();
    let mut stats = IndexedSourceReadStats::default();
    for attempt in 0..2 {
        locations.clear();
        let connection = rusqlite::Connection::open(index_path)
            .map_err(|error| format!("open index {}: {error}", index_path.display()))?;
        let mut statement = connection
            .prepare(
                "SELECT shard, source_offset, source_length, size FROM sources \
                 JOIN indexed_source_shards USING (shard) WHERE air_sha256=?1",
            )
            .map_err(|error| format!("prepare indexed source analysis: {error}"))?;
        let mut shard_sizes = std::collections::HashMap::new();
        let mut refresh = std::collections::BTreeSet::new();
        for hash in hashes {
            if let Some(source) = public.get(hash) {
                locations.push(Location::Public(source.clone()));
                continue;
            }
            let expected_shard = shard_index_for_hash(hash)?;
            let location = statement
                .query_row([hash], |row| {
                    Ok((
                        row.get::<_, usize>(0)?,
                        row.get::<_, Option<i64>>(1)?,
                        row.get::<_, Option<i64>>(2)?,
                        row.get::<_, u64>(3)?,
                    ))
                })
                .optional()
                .map_err(|error| format!("read indexed source {hash}: {error}"))?;
            let Some((shard, Some(offset), Some(length), indexed_size)) = location else {
                refresh.insert(expected_shard);
                continue;
            };
            if shard != expected_shard || offset < 0 || length <= 0 {
                refresh.insert(expected_shard);
                continue;
            }
            let current_size = *shard_sizes.entry(shard).or_insert_with(|| {
                fs::metadata(source_shard_path(root, shard))
                    .map(|metadata| metadata.len())
                    .unwrap_or(u64::MAX)
            });
            if current_size != indexed_size {
                refresh.insert(expected_shard);
                continue;
            }
            locations.push(Location::Private {
                hash: hash.clone(),
                shard,
                offset: offset as u64,
                length: length as usize,
            });
        }
        drop(statement);
        drop(connection);
        if refresh.is_empty() {
            break;
        }
        if attempt != 0 {
            return Err(format!(
                "source locations remain unavailable after repairing shards {}",
                refresh
                    .iter()
                    .map(usize::to_string)
                    .collect::<Vec<_>>()
                    .join(",")
            ));
        }
        for shard in refresh {
            stats.repair_shards_scanned += 1;
            stats.repair_bytes_scanned = stats
                .repair_bytes_scanned
                .checked_add(crate::index::index_source_shard_locations(
                    root, index_path, shard,
                )?)
                .ok_or_else(|| "source repair byte count overflow".to_string())?;
        }
    }
    if locations.len() != hashes.len() {
        return Err(format!(
            "indexed analysis resolved {} of {} requested sources",
            locations.len(),
            hashes.len()
        ));
    }

    let mut files = std::collections::HashMap::<usize, File>::new();
    for location in locations {
        let source = match location {
            Location::Public(source) => source,
            Location::Private {
                hash,
                shard,
                offset,
                length,
            } => {
                stats.source_bytes_read = stats
                    .source_bytes_read
                    .checked_add(length as u64)
                    .ok_or_else(|| "indexed source byte count overflow".to_string())?;
                let path = source_shard_path(root, shard);
                let file = match files.entry(shard) {
                    std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
                    std::collections::hash_map::Entry::Vacant(entry) => entry.insert(
                        File::open(&path)
                            .map_err(|error| format!("open {}: {error}", path.display()))?,
                    ),
                };
                file.seek(SeekFrom::Start(offset)).map_err(|error| {
                    format!("seek {} to source {hash}: {error}", path.display())
                })?;
                let mut bytes = vec![0; length];
                file.read_exact(&mut bytes).map_err(|error| {
                    format!(
                        "read indexed source {hash} from {}: {error}",
                        path.display()
                    )
                })?;
                let source = analysis_source_from_json(&bytes, shard)
                    .map_err(|error| format!("parse indexed source {hash}: {error}"))?;
                if source.air_sha256 != hash {
                    return Err(format!(
                        "indexed source location for {hash} resolves to {}",
                        source.air_sha256
                    ));
                }
                source
            }
        };
        consume(source)?;
        stats.rows += 1;
    }
    stats.source_shards_opened = files.len();
    Ok(stats)
}

fn analysis_source_from_json(bytes: &[u8], expected_shard: usize) -> Result<SourceRow, String> {
    let row: AnalysisRow =
        serde_json::from_slice(bytes).map_err(|error| format!("invalid source JSON: {error}"))?;
    let actual_shard = shard_index_for_hash(&row.air_sha256)?;
    if actual_shard != expected_shard {
        return Err(format!(
            "source {} belongs in shard {}, not {}",
            row.air_sha256, actual_shard, expected_shard
        ));
    }
    Ok(SourceRow {
        air_sha256: row.air_sha256,
        stage: row.stage,
        entry: row.entry,
        air_ll: row.air_ll,
        blob_b64: None,
        lib_sha256: row.lib_sha256,
        label: row.label,
    })
}

pub fn read_all_private_sources(root: &Path) -> Result<Vec<SourceRow>, String> {
    let directory = source_shards_dir(root);
    let Ok(entries) = fs::read_dir(&directory) else {
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
    let mut rows = Vec::new();
    for path in paths {
        rows.extend(read_source_shard(&path)?);
    }
    let mut seen = std::collections::HashSet::new();
    for row in &rows {
        if !seen.insert(&row.air_sha256) {
            return Err(format!("duplicate private source {}", row.air_sha256));
        }
    }
    Ok(rows)
}

pub fn find_source(root: &Path, hash: &str) -> Result<Option<SourceRow>, String> {
    if let Some(source) = find_public_source(hash)? {
        return Ok(Some(source));
    }
    let index = shard_index_for_hash(hash)?;
    if let Some(source) = find_indexed_private_source(root, hash, index)? {
        return Ok(Some(source));
    }
    let path = source_shard_path(root, index);
    if path.is_file() {
        if let Some(row) = read_source_shard(&path)?
            .into_iter()
            .find(|row| row.air_sha256 == hash)
        {
            return Ok(Some(row));
        }
    }
    Ok(None)
}

fn find_indexed_private_source(
    root: &Path,
    hash: &str,
    expected_shard: usize,
) -> Result<Option<SourceRow>, String> {
    find_indexed_private_source_with_refresh(root, hash, expected_shard, true)
}

fn find_indexed_private_source_with_refresh(
    root: &Path,
    hash: &str,
    expected_shard: usize,
    refresh_missing_location: bool,
) -> Result<Option<SourceRow>, String> {
    let index_path = root.join(".index.sqlite");
    if !index_path.is_file() {
        return Ok(None);
    }
    let connection = match rusqlite::Connection::open(&index_path) {
        Ok(connection) => connection,
        Err(_) => return Ok(None),
    };
    let location = match connection
        .query_row(
            "SELECT s.shard, s.source_offset, s.source_length, t.size \
             FROM sources s JOIN indexed_source_shards t ON t.shard=s.shard \
             WHERE s.air_sha256=?1",
            [hash],
            |row| {
                Ok((
                    row.get::<_, usize>(0)?,
                    row.get::<_, Option<i64>>(1)?,
                    row.get::<_, Option<i64>>(2)?,
                    row.get::<_, u64>(3)?,
                ))
            },
        )
        .optional()
    {
        Ok(Some(location)) => location,
        Ok(None) => {
            drop(connection);
            if refresh_missing_location {
                crate::index::index_source_shard_locations(root, &index_path, expected_shard)?;
                return find_indexed_private_source_with_refresh(root, hash, expected_shard, false);
            }
            return Ok(None);
        }
        Err(_) => return Ok(None),
    };
    let (shard, offset, length, indexed_size) = location;
    let (Some(offset), Some(length)) = (offset, length) else {
        drop(connection);
        if refresh_missing_location {
            crate::index::index_source_shard_locations(root, &index_path, expected_shard)?;
            return find_indexed_private_source_with_refresh(root, hash, expected_shard, false);
        }
        return Ok(None);
    };
    if shard != expected_shard || offset < 0 || length <= 0 {
        return Ok(None);
    }
    let path = source_shard_path(root, shard);
    let current_size = match fs::metadata(&path) {
        Ok(metadata) => metadata.len(),
        Err(_) => return Ok(None),
    };
    if current_size != indexed_size {
        drop(connection);
        if refresh_missing_location {
            crate::index::index_source_shard_locations(root, &index_path, expected_shard)?;
            return find_indexed_private_source_with_refresh(root, hash, expected_shard, false);
        }
        return Ok(None);
    }
    let mut file =
        File::open(&path).map_err(|error| format!("open {}: {error}", path.display()))?;
    file.seek(SeekFrom::Start(offset as u64))
        .map_err(|error| format!("seek {} to source {hash}: {error}", path.display()))?;
    let mut bytes = vec![0u8; length as usize];
    file.read_exact(&mut bytes).map_err(|error| {
        format!(
            "read indexed source {hash} from {}: {error}",
            path.display()
        )
    })?;
    let row: SourceRow = serde_json::from_slice(&bytes).map_err(|error| {
        format!(
            "parse indexed source {hash} from {}: {error}",
            path.display()
        )
    })?;
    row.validate()
        .map_err(|error| format!("indexed source {hash} from {}: {error}", path.display()))?;
    if row.air_sha256 != hash {
        return Err(format!(
            "indexed source location for {hash} resolves to {}",
            row.air_sha256
        ));
    }
    Ok(Some(row))
}

pub fn public_sources() -> Result<Vec<SourceRow>, String> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/public");
    let mut paths = fs::read_dir(&root)
        .map_err(|error| format!("read {}: {error}", root.display()))?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|extension| extension == "ll"))
        .collect::<Vec<_>>();
    paths.sort();
    paths.into_iter().map(public_source_row).collect()
}

fn find_public_source(hash: &str) -> Result<Option<SourceRow>, String> {
    Ok(public_sources()?
        .into_iter()
        .find(|row| row.air_sha256 == hash))
}

fn public_source_row(path: PathBuf) -> Result<SourceRow, String> {
    let air_ll = fs::read_to_string(&path)
        .map_err(|error| format!("read public source {}: {error}", path.display()))?;
    let (stage, entry) = crate::air::stage_entry_from_ll(&air_ll)
        .ok_or_else(|| format!("{} has no stable AIR entry metadata", path.display()))?;
    Ok(SourceRow {
        air_sha256: sha256_bytes(air_ll.as_bytes()),
        stage: stage.into(),
        entry,
        air_ll,
        blob_b64: None,
        lib_sha256: "owned-synthetic".into(),
        label: format!(
            "public/{}",
            path.file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("fixture.ll")
        ),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ScratchDir;

    #[test]
    fn leading_six_bits_select_stable_aligned_shards() {
        assert_eq!(
            shard_index_for_hash(&format!("00{}", "0".repeat(62))).unwrap(),
            0
        );
        assert_eq!(
            shard_index_for_hash(&format!("03{}", "0".repeat(62))).unwrap(),
            0
        );
        assert_eq!(
            shard_index_for_hash(&format!("04{}", "0".repeat(62))).unwrap(),
            1
        );
        assert_eq!(
            shard_index_for_hash(&format!("ff{}", "0".repeat(62))).unwrap(),
            63
        );
        assert_eq!(shard_name(7), "shard_007.jsonl");
    }

    #[test]
    fn analysis_reader_skips_blob_decoding_and_streams_air_projection() {
        let scratch = ScratchDir::new("source-analysis-reader").unwrap();
        let directory = source_shards_dir(scratch.path());
        fs::create_dir_all(&directory).unwrap();
        let path = source_shard_path(scratch.path(), 0);
        let row = serde_json::json!({
            "air_ll": "define void @main() { ret void }",
            "air_sha256": "00".repeat(32),
            "blob_b64": "deliberately not base64",
            "entry": "main",
            "label": "analysis/test",
            "lib_sha256": "11".repeat(32),
            "stage": "Kernel"
        });
        fs::write(
            &path,
            format!("{}\n", crate::jsonl::to_sorted_json_string(&row).unwrap()),
        )
        .unwrap();
        let mut seen = Vec::new();
        for_each_source_shard_analysis(&path, |source| {
            seen.push(source);
            Ok(())
        })
        .unwrap();
        assert_eq!(seen.len(), 1);
        assert_eq!(seen[0].air_ll, "define void @main() { ret void }");
        assert_eq!(seen[0].blob_b64, None);
    }

    #[test]
    fn source_shard_output_is_byte_deterministic() {
        let scratch = ScratchDir::new("source-shards").unwrap();
        let ll = "define void @main() { ret void }\n!air.kernel = !{!0}\n!0 = !{ptr @main}";
        let row = SourceRow {
            air_sha256: sha256_bytes(ll.as_bytes()),
            stage: "Kernel".into(),
            entry: "main".into(),
            air_ll: ll.into(),
            blob_b64: Some("YmxvYg==".into()),
            lib_sha256: "22".repeat(32),
            label: "local/test.ll".into(),
        };
        write_source_shards(scratch.path(), [row.clone()]).unwrap();
        let first = fs::read_dir(source_shards_dir(scratch.path()))
            .unwrap()
            .filter_map(Result::ok)
            .map(|entry| (entry.file_name(), fs::read(entry.path()).unwrap()))
            .collect::<Vec<_>>();
        fs::write(
            source_shards_dir(scratch.path()).join(".shard_000.jsonl.interrupted.tmp"),
            b"partial",
        )
        .unwrap();
        write_source_shards(scratch.path(), [row]).unwrap();
        let second = fs::read_dir(source_shards_dir(scratch.path()))
            .unwrap()
            .filter_map(Result::ok)
            .map(|entry| (entry.file_name(), fs::read(entry.path()).unwrap()))
            .collect::<Vec<_>>();
        assert_eq!(first, second);
    }

    #[test]
    fn incremental_merge_never_opens_or_rewrites_an_unrelated_shard() {
        let scratch = ScratchDir::new("source-shard-merge").unwrap();
        let mut selected = public_sources().unwrap().remove(0);
        selected.air_ll.push_str("\n; selected private source\n");
        selected.air_sha256 = sha256_bytes(selected.air_ll.as_bytes());
        selected.lib_sha256 = "22".repeat(32);
        selected.label = "local/selected.ll".into();
        let selected_shard = shard_index_for_hash(&selected.air_sha256).unwrap();
        let mut unrelated = selected.clone();
        for nonce in 0..u32::MAX {
            unrelated.air_ll = format!("{}\n; unrelated {nonce}\n", selected.air_ll);
            unrelated.air_sha256 = sha256_bytes(unrelated.air_ll.as_bytes());
            if shard_index_for_hash(&unrelated.air_sha256).unwrap() != selected_shard {
                break;
            }
        }
        unrelated.label = "local/unrelated.ll".into();
        let unrelated_shard = shard_index_for_hash(&unrelated.air_sha256).unwrap();
        write_source_shards(scratch.path(), [selected.clone(), unrelated]).unwrap();

        let unrelated_path = source_shard_path(scratch.path(), unrelated_shard);
        let unrelated_bytes = fs::read(&unrelated_path).unwrap();
        let unreadable_canonical_data = vec![b'x'; unrelated_bytes.len()];
        fs::write(&unrelated_path, &unreadable_canonical_data).unwrap();

        selected.lib_sha256 = "11".repeat(32);
        let stats = merge_source_shards(scratch.path(), [selected.clone()]).unwrap();
        assert_eq!(
            stats,
            SourceMergeStats {
                affected_shards: 1,
                inserted: 0,
                replaced: 1,
                duplicates: 0,
            }
        );
        assert_eq!(
            fs::read(&unrelated_path).unwrap(),
            unreadable_canonical_data
        );
        assert_eq!(
            read_source_shard(&source_shard_path(scratch.path(), selected_shard))
                .unwrap()
                .remove(0)
                .lib_sha256,
            selected.lib_sha256
        );
    }

    #[test]
    fn indexed_membership_filters_reharvest_before_source_shards_are_opened() {
        let scratch = ScratchDir::new("source-index-membership").unwrap();
        let index = scratch.path().join(".index.sqlite");
        let connection = rusqlite::Connection::open(&index).unwrap();
        connection
            .execute_batch("CREATE TABLE sources (air_sha256 TEXT PRIMARY KEY);")
            .unwrap();
        let known = "11".repeat(32);
        let missing = "22".repeat(32);
        connection
            .execute("INSERT INTO sources VALUES (?1)", [&known])
            .unwrap();
        drop(connection);

        assert_eq!(
            unindexed_source_hashes(scratch.path(), [known, missing.clone()]).unwrap(),
            [missing].into_iter().collect()
        );
        assert_eq!(
            indexed_source_hashes(scratch.path()).unwrap(),
            ["11".repeat(32)].into_iter().collect()
        );
    }

    #[test]
    fn duplicate_merge_does_not_replace_an_unchanged_shard_file() {
        use std::os::unix::fs::MetadataExt as _;

        let scratch = ScratchDir::new("source-duplicate-no-rewrite").unwrap();
        let mut row = public_sources().unwrap().remove(0);
        row.air_ll.push_str("\n; duplicate no-rewrite fixture\n");
        row.air_sha256 = sha256_bytes(row.air_ll.as_bytes());
        row.lib_sha256 = "11".repeat(32);
        row.label = "local/duplicate.ll".into();
        write_source_shards(scratch.path(), [row.clone()]).unwrap();
        let path = source_shard_path(
            scratch.path(),
            shard_index_for_hash(&row.air_sha256).unwrap(),
        );
        let inode = fs::metadata(&path).unwrap().ino();

        let stats = merge_source_shards(scratch.path(), [row]).unwrap();
        assert_eq!(stats.inserted, 0);
        assert_eq!(stats.replaced, 0);
        assert_eq!(stats.duplicates, 1);
        assert_eq!(fs::metadata(path).unwrap().ino(), inode);
    }

    #[test]
    fn public_source_lookup_survives_an_unrelated_private_bucket_file() {
        let scratch = ScratchDir::new("public-source-fallback").unwrap();
        let public = public_sources().unwrap().remove(0);
        let path = source_shard_path(
            scratch.path(),
            shard_index_for_hash(&public.air_sha256).unwrap(),
        );
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, b"").unwrap();
        assert_eq!(
            find_source(scratch.path(), &public.air_sha256)
                .unwrap()
                .unwrap(),
            public
        );
    }

    #[test]
    fn source_reader_rejects_a_misaligned_row() {
        let scratch = ScratchDir::new("misaligned-source").unwrap();
        let ll = "define void @main() { ret void }\n!air.kernel = !{!0}\n!0 = !{ptr @main}";
        let row = SourceRow {
            air_sha256: sha256_bytes(ll.as_bytes()),
            stage: "Kernel".into(),
            entry: "main".into(),
            air_ll: ll.into(),
            blob_b64: None,
            lib_sha256: "22".repeat(32),
            label: "local/test.ll".into(),
        };
        let actual = shard_index_for_hash(&row.air_sha256).unwrap();
        let wrong = (actual + 1) % SHARD_COUNT;
        let path = scratch.path().join(shard_name(wrong));
        fs::write(&path, format!("{}\n", to_sorted_json_string(row).unwrap())).unwrap();
        assert!(read_source_shard(&path)
            .unwrap_err()
            .contains("belongs in shard"));
    }
}
