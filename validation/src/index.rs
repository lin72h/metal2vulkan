use crate::candidate::EXECUTOR_ABI;
use crate::metal::ORACLE_ABI;
use crate::observation::{Backend, CandidateDependencies, TRANSLATOR_FINGERPRINT};
use crate::source::{
    public_sources, shard_index_from_path, source_shards_dir, SourceIndexLocation,
};
use crate::store::CorpusStore;
use memchr::memmem;
use rusqlite::{params, Connection, OptionalExtension, Transaction};
use std::collections::HashSet;
use std::fs;
use std::fs::File;
use std::io::{BufRead, BufReader, Seek};
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QueueRow {
    pub air_sha256: String,
    pub stage: String,
    pub entry: String,
    pub label: String,
    pub state: QueueState,
    pub review_reason: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QueueState {
    Unplanned,
    Authored,
    MetalQualified,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EvidenceState {
    Current,
    Failed,
    Missing,
    Stale,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct IndexSyncStats {
    pub rebuilt: bool,
    pub source_shards_scanned: usize,
    pub source_bytes_scanned: u64,
}

impl EvidenceState {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Current => "current",
            Self::Failed => "failed",
            Self::Missing => "missing",
            Self::Stale => "stale",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EvidenceRow {
    pub case_id: String,
    pub name: String,
    pub metal: EvidenceState,
    pub moltenvk: EvidenceState,
    pub vulkan: EvidenceState,
    pub translation_error: Option<String>,
    pub slots: Vec<EvidenceSlot>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EvidenceSlot {
    pub backend: &'static str,
    pub environment_id: String,
    pub state: EvidenceState,
}

impl QueueState {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Unplanned => "unplanned",
            Self::Authored => "authored",
            Self::MetalQualified => "metal-qualified",
        }
    }
}

pub fn default_index_path(root: &Path) -> PathBuf {
    root.join(".index.sqlite")
}

pub fn rebuild_index(root: &Path, destination: &Path) -> Result<(), String> {
    let temporary = destination.with_extension(format!("sqlite.{}.tmp", std::process::id()));
    if temporary.exists() {
        fs::remove_file(&temporary)
            .map_err(|error| format!("remove {}: {error}", temporary.display()))?;
    }
    if let Some(parent) = destination.parent() {
        fs::create_dir_all(parent)
            .map_err(|error| format!("create {}: {error}", parent.display()))?;
    }
    let mut cleanup = RemoveOnDrop::new(temporary.clone());
    let mut connection = Connection::open(&temporary)
        .map_err(|error| format!("open {}: {error}", temporary.display()))?;
    connection
        .execute_batch(
            "PRAGMA foreign_keys=ON;
             PRAGMA journal_mode=DELETE;
             PRAGMA synchronous=FULL;
             CREATE TABLE sources (
               air_sha256 TEXT PRIMARY KEY,
               stage TEXT NOT NULL,
               entry TEXT NOT NULL,
               label TEXT NOT NULL,
               shard INTEGER,
               source_offset INTEGER,
               source_length INTEGER
             );
             CREATE TABLE cases (
               case_id TEXT PRIMARY KEY,
               air_sha256 TEXT NOT NULL REFERENCES sources(air_sha256),
               name TEXT NOT NULL,
               input_sha256 TEXT NOT NULL,
               UNIQUE(air_sha256, name)
             );
             CREATE TABLE reviews (
               air_sha256 TEXT PRIMARY KEY REFERENCES sources(air_sha256),
               reason TEXT NOT NULL,
               reviewed_by TEXT NOT NULL
             );
             CREATE TABLE metal_observations (
               case_id TEXT NOT NULL REFERENCES cases(case_id),
               environment_id TEXT NOT NULL,
               input_sha256 TEXT NOT NULL,
               output_sha256 TEXT NOT NULL,
               oracle_abi TEXT NOT NULL,
               PRIMARY KEY(case_id, environment_id)
             );
             CREATE TABLE candidate_observations (
               case_id TEXT NOT NULL REFERENCES cases(case_id),
               backend TEXT NOT NULL,
               environment_id TEXT NOT NULL,
               input_sha256 TEXT NOT NULL,
               golden_output_sha256 TEXT NOT NULL,
               spv_sha256 TEXT NOT NULL,
               translator_fingerprint TEXT NOT NULL,
               executor_abi TEXT NOT NULL,
               status TEXT NOT NULL,
               PRIMARY KEY(case_id, backend, environment_id)
             );
             CREATE TABLE indexed_source_shards (
               shard INTEGER PRIMARY KEY,
               size INTEGER NOT NULL,
               modified_ns INTEGER NOT NULL
             );
             CREATE TABLE index_metadata (
               key TEXT PRIMARY KEY,
               value TEXT NOT NULL
             );
             INSERT INTO index_metadata (key, value)
               VALUES ('source_shard_tracking', 'complete');
             CREATE TABLE triage_analysis (
               air_sha256 TEXT PRIMARY KEY,
               analyzer_abi TEXT NOT NULL,
               result_json TEXT NOT NULL
             );
             CREATE INDEX cases_air_sha256 ON cases(air_sha256);
             CREATE INDEX metal_case_dependencies ON metal_observations(case_id, input_sha256, oracle_abi);",
        )
        .map_err(|error| format!("create index schema: {error}"))?;
    let transaction = connection
        .transaction()
        .map_err(|error| format!("begin index transaction: {error}"))?;

    {
        let mut insert = transaction
            .prepare(
                "INSERT INTO sources (air_sha256, stage, entry, label, shard, source_offset, source_length) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            )
            .map_err(|error| format!("prepare source insert: {error}"))?;
        let mut public_hashes = HashSet::new();
        for source in public_sources()? {
            let air_sha256 = source.air_sha256;
            insert_source(
                &mut insert,
                &SourceIndexRow {
                    air_sha256: air_sha256.clone(),
                    stage: source.stage,
                    entry: source.entry,
                    label: source.label,
                    lib_sha256: source.lib_sha256,
                    source_offset: None,
                    source_length: None,
                },
                None,
            )?;
            public_hashes.insert(air_sha256);
        }

        let source_dir = source_shards_dir(root);
        if let Ok(entries) = fs::read_dir(&source_dir) {
            let mut paths = entries
                .filter_map(Result::ok)
                .map(|entry| entry.path())
                .filter(|path| {
                    path.extension()
                        .is_some_and(|extension| extension == "jsonl")
                })
                .collect::<Vec<_>>();
            paths.sort();
            let mut private_hashes = HashSet::new();
            for path in paths {
                let expected_shard = shard_index_from_path(&path)?;
                read_source_index_shard(&path, |source| {
                    let actual_shard = crate::source::shard_index_for_hash(&source.air_sha256)?;
                    if actual_shard != expected_shard {
                        return Err(format!(
                            "source {} belongs in shard {}, not {}",
                            source.air_sha256, actual_shard, expected_shard
                        ));
                    }
                    if !private_hashes.insert(source.air_sha256.clone()) {
                        return Err(format!("duplicate private source {}", source.air_sha256));
                    }
                    if !public_hashes.contains(&source.air_sha256) {
                        insert_source(&mut insert, &source, Some(actual_shard))?;
                    }
                    Ok(())
                })?;
                let stamp = source_shard_stamp(&path)?;
                transaction
                    .execute(
                        "INSERT INTO indexed_source_shards (shard, size, modified_ns) VALUES (?1, ?2, ?3)",
                        params![expected_shard, stamp.size, stamp.modified_ns],
                    )
                    .map_err(|error| format!("index source shard stamp: {error}"))?;
            }
        }
    }

    let store = CorpusStore::new(root);
    for note in store.read_reviews()? {
        transaction
            .execute(
                "INSERT INTO reviews (air_sha256, reason, reviewed_by) VALUES (?1, ?2, ?3)",
                params![note.air_sha256, note.reason, note.reviewed_by],
            )
            .map_err(|error| format!("index review note: {error}"))?;
    }
    for case in store.read_all_cases()? {
        transaction
            .execute(
                "INSERT INTO cases (case_id, air_sha256, name, input_sha256) VALUES (?1, ?2, ?3, ?4)",
                params![case.case_id, case.air_sha256, case.name, case.computed_input_sha256()?],
            )
            .map_err(|error| format!("index case: {error}"))?;
    }
    for observation in store.read_metal()? {
        transaction
            .execute(
                "INSERT INTO metal_observations (case_id, environment_id, input_sha256, output_sha256, oracle_abi) VALUES (?1, ?2, ?3, ?4, ?5)",
                params![observation.case_id, observation.environment_id, observation.input_sha256, observation.metal_output_sha256, observation.oracle_abi],
            )
            .map_err(|error| format!("index Metal observation: {error}"))?;
    }
    for backend in [Backend::Moltenvk, Backend::Vulkan] {
        for observation in store.read_candidates(backend)? {
            transaction
                .execute(
                    "INSERT INTO candidate_observations (case_id, backend, environment_id, input_sha256, golden_output_sha256, spv_sha256, translator_fingerprint, executor_abi, status) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
                    params![observation.case_id, backend.directory(), observation.environment_id, observation.input_sha256, observation.golden_output_sha256, observation.spv_sha256, observation.translator_fingerprint, observation.executor_abi, format!("{:?}", observation.status).to_ascii_lowercase()],
                )
                .map_err(|error| format!("index {} observation: {error}", backend.directory()))?;
        }
    }
    transaction
        .commit()
        .map_err(|error| format!("commit index: {error}"))?;
    connection
        .execute_batch("PRAGMA optimize;")
        .map_err(|error| format!("optimize index: {error}"))?;
    drop(connection);
    fs::rename(&temporary, destination).map_err(|error| {
        format!(
            "rename {} to {}: {error}",
            temporary.display(),
            destination.display()
        )
    })?;
    cleanup.disarm();
    Ok(())
}

/// Incrementally synchronize the disposable index.
///
/// Source bodies dominate corpus size, so unchanged source shards are identified by their file
/// stamp and never opened. Authored cases and evidence are compact and are refreshed atomically on
/// every sync.
pub fn sync_index(root: &Path, destination: &Path) -> Result<(), String> {
    sync_index_with_stats(root, destination).map(|_| ())
}

pub fn sync_index_with_stats(root: &Path, destination: &Path) -> Result<IndexSyncStats, String> {
    if !destination.is_file() {
        let paths = source_shard_paths(root)?;
        let source_bytes_scanned = paths.iter().try_fold(0u64, |total, path| {
            source_shard_stamp(path).and_then(|stamp| {
                total
                    .checked_add(stamp.size)
                    .ok_or_else(|| "source shard byte count overflow".to_string())
            })
        })?;
        rebuild_index(root, destination)?;
        return Ok(IndexSyncStats {
            rebuilt: true,
            source_shards_scanned: paths.len(),
            source_bytes_scanned,
        });
    }
    let mut stats = IndexSyncStats::default();
    let mut connection = Connection::open(destination)
        .map_err(|error| format!("open index {}: {error}", destination.display()))?;
    ensure_source_location_columns(&connection)?;
    ensure_candidate_fingerprint_column(&connection)?;
    connection
        .execute_batch(
            "PRAGMA foreign_keys=ON;
             CREATE TABLE IF NOT EXISTS indexed_source_shards (
               shard INTEGER PRIMARY KEY,
               size INTEGER NOT NULL,
               modified_ns INTEGER NOT NULL
             );
             CREATE TABLE IF NOT EXISTS index_metadata (
               key TEXT PRIMARY KEY,
               value TEXT NOT NULL
             );
             CREATE TABLE IF NOT EXISTS triage_analysis (
               air_sha256 TEXT PRIMARY KEY,
               analyzer_abi TEXT NOT NULL,
               result_json TEXT NOT NULL
             );
             CREATE INDEX IF NOT EXISTS cases_air_sha256 ON cases(air_sha256);
             CREATE INDEX IF NOT EXISTS metal_case_dependencies ON metal_observations(case_id, input_sha256, oracle_abi);",
        )
        .map_err(|error| format!("upgrade index schema: {error}"))?;
    // Do not infer migration completion from a nonempty stamp table. Harvest can publish one
    // changed shard before the first sync of a legacy index, leaving a deliberately partial table.
    // Only this marker, committed after all shard stamps, proves missing stamps mean new data.
    let tracking_complete = connection
        .query_row(
            "SELECT 1 FROM index_metadata WHERE key='source_shard_tracking' AND value='complete'",
            [],
            |_| Ok(()),
        )
        .optional()
        .map_err(|error| format!("inspect source-shard migration marker: {error}"))?
        .is_some();
    let transaction = connection
        .transaction()
        .map_err(|error| format!("begin index sync: {error}"))?;

    transaction
        .execute_batch(
            "DELETE FROM candidate_observations;
             DELETE FROM metal_observations;
             DELETE FROM reviews;
             DELETE FROM cases;",
        )
        .map_err(|error| format!("clear compact index rows: {error}"))?;

    let paths = source_shard_paths(root)?;
    let current_shards = paths
        .iter()
        .map(|path| shard_index_from_path(path))
        .collect::<Result<HashSet<_>, _>>()?;
    let tracked_shards = {
        let mut statement = transaction
            .prepare("SELECT shard FROM indexed_source_shards")
            .map_err(|error| format!("prepare tracked-shard query: {error}"))?;
        let shards = statement
            .query_map([], |row| row.get::<_, usize>(0))
            .map_err(|error| format!("query tracked shards: {error}"))?
            .collect::<Result<HashSet<_>, _>>()
            .map_err(|error| format!("read tracked shards: {error}"))?;
        shards
    };
    for removed in tracked_shards.difference(&current_shards) {
        transaction
            .execute("DELETE FROM sources WHERE shard=?1", [removed])
            .map_err(|error| format!("remove source shard {removed}: {error}"))?;
        transaction
            .execute(
                "DELETE FROM indexed_source_shards WHERE shard=?1",
                [removed],
            )
            .map_err(|error| format!("remove source shard stamp {removed}: {error}"))?;
    }

    transaction
        .execute(
            "DELETE FROM sources WHERE shard IS NULL OR label LIKE 'public/%'",
            [],
        )
        .map_err(|error| format!("refresh public sources: {error}"))?;
    let mut insert = transaction
        .prepare(
            "INSERT OR IGNORE INTO sources (air_sha256, stage, entry, label, shard, source_offset, source_length) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
        )
        .map_err(|error| format!("prepare source sync insert: {error}"))?;
    for source in public_sources()? {
        insert_source(
            &mut insert,
            &SourceIndexRow {
                air_sha256: source.air_sha256,
                stage: source.stage,
                entry: source.entry,
                label: source.label,
                lib_sha256: source.lib_sha256,
                source_offset: None,
                source_length: None,
            },
            None,
        )?;
    }
    for path in paths {
        let shard = shard_index_from_path(&path)?;
        let stamp = source_shard_stamp(&path)?;
        let tracked = transaction
            .query_row(
                "SELECT size, modified_ns FROM indexed_source_shards WHERE shard=?1",
                [shard],
                |row| {
                    Ok(SourceShardStamp {
                        size: row.get(0)?,
                        modified_ns: row.get(1)?,
                    })
                },
            )
            .optional()
            .map_err(|error| format!("read source shard {shard} stamp: {error}"))?;
        // A legacy index already contains the source projection. Treat it as the migration
        // baseline and install per-shard stamps from metadata alone. Reading every source body to
        // add bookkeeping would defeat the index; missing byte locations are reconciled lazily by
        // scanning only the hash-derived shard on first lookup.
        let bootstrap_unchanged = !tracking_complete;
        // Harvest rewrites only buckets that received a hash, so the complete file stamp is a
        // reliable shard-local invalidation key. This also notices same-size replacements.
        let tracked_content_unchanged = tracked == Some(stamp);
        if tracked_content_unchanged || bootstrap_unchanged {
            transaction
                .execute(
                    "INSERT OR REPLACE INTO indexed_source_shards (shard, size, modified_ns) VALUES (?1, ?2, ?3)",
                    params![shard, stamp.size, stamp.modified_ns],
                )
                .map_err(|error| format!("record source shard {shard} stamp: {error}"))?;
            continue;
        }
        transaction
            .execute("DELETE FROM sources WHERE shard=?1", [shard])
            .map_err(|error| format!("replace source shard {shard}: {error}"))?;
        stats.source_shards_scanned += 1;
        stats.source_bytes_scanned = stats
            .source_bytes_scanned
            .checked_add(stamp.size)
            .ok_or_else(|| "source shard byte count overflow".to_string())?;
        read_source_index_shard(&path, |source| {
            let actual = crate::source::shard_index_for_hash(&source.air_sha256)?;
            if actual != shard {
                return Err(format!(
                    "source {} belongs in shard {}, not {}",
                    source.air_sha256, actual, shard
                ));
            }
            insert_source(&mut insert, &source, Some(shard))
        })?;
        transaction
            .execute(
                "INSERT OR REPLACE INTO indexed_source_shards (shard, size, modified_ns) VALUES (?1, ?2, ?3)",
                params![shard, stamp.size, stamp.modified_ns],
            )
            .map_err(|error| format!("record source shard {shard} stamp: {error}"))?;
    }
    drop(insert);
    transaction
        .execute(
            "INSERT OR REPLACE INTO index_metadata (key, value) VALUES ('source_shard_tracking', 'complete')",
            [],
        )
        .map_err(|error| format!("record source-shard migration completion: {error}"))?;
    insert_compact_rows(&transaction, root)?;
    transaction
        .commit()
        .map_err(|error| format!("commit index sync: {error}"))?;
    Ok(stats)
}

/// Publish the compact projection of a source shard that was just atomically replaced.
///
/// The shard writer already serialized every row, so reopening a multi-gigabyte bucket merely to
/// rediscover metadata and byte offsets is wasted work. If this update fails or the process exits
/// between the shard rename and this transaction, the file stamp remains different and the normal
/// synchronizer repairs exactly that shard on its next run.
pub(crate) fn record_source_shard_write(
    root: &Path,
    shard: usize,
    locations: &[SourceIndexLocation],
) -> Result<(), String> {
    let destination = default_index_path(root);
    if !destination.is_file() {
        return Ok(());
    }
    let stamp = source_shard_stamp(&crate::source::source_shard_path(root, shard))?;
    let mut connection = Connection::open(&destination)
        .map_err(|error| format!("open index {}: {error}", destination.display()))?;
    ensure_source_location_columns(&connection)?;
    connection
        .execute_batch(
            "PRAGMA foreign_keys=ON;
             CREATE TABLE IF NOT EXISTS indexed_source_shards (
               shard INTEGER PRIMARY KEY,
               size INTEGER NOT NULL,
               modified_ns INTEGER NOT NULL
             );",
        )
        .map_err(|error| format!("upgrade source-shard tracking: {error}"))?;
    let transaction = connection
        .transaction()
        .map_err(|error| format!("begin source-shard index update: {error}"))?;
    {
        let mut upsert = transaction
            .prepare(
                "INSERT INTO sources
                   (air_sha256, stage, entry, label, shard, source_offset, source_length)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
                 ON CONFLICT(air_sha256) DO UPDATE SET
                   stage=excluded.stage,
                   entry=excluded.entry,
                   label=excluded.label,
                   shard=excluded.shard,
                   source_offset=excluded.source_offset,
                   source_length=excluded.source_length
                 WHERE sources.shard IS NOT NULL",
            )
            .map_err(|error| format!("prepare source-shard index update: {error}"))?;
        for location in locations {
            upsert
                .execute(params![
                    location.air_sha256,
                    location.stage,
                    location.entry,
                    location.label,
                    shard,
                    location.offset,
                    location.length,
                ])
                .map_err(|error| format!("index source {}: {error}", location.air_sha256))?;
        }
    }
    transaction
        .execute(
            "INSERT OR REPLACE INTO indexed_source_shards (shard, size, modified_ns)
             VALUES (?1, ?2, ?3)",
            params![shard, stamp.size, stamp.modified_ns],
        )
        .map_err(|error| format!("record source shard {shard} stamp: {error}"))?;
    transaction
        .commit()
        .map_err(|error| format!("commit source-shard index update: {error}"))
}

/// Return cases whose exact Metal and MoltenVK environment slots are not current.
///
/// This query uses only compact index rows. In particular, it never opens an AIR source shard.
pub fn pending_macos_refresh_case_ids(
    index: &Path,
    metal_environment_id: &str,
    environment_id: &str,
) -> Result<HashSet<String>, String> {
    let connection = Connection::open(index)
        .map_err(|error| format!("open index {}: {error}", index.display()))?;
    let mut statement = connection
        .prepare(
            "SELECT c.case_id
             FROM cases c
             WHERE NOT EXISTS (
               SELECT 1 FROM metal_observations m
               WHERE m.case_id=c.case_id
                 AND m.environment_id=?1
                 AND m.input_sha256=c.input_sha256
                 AND m.oracle_abi=?2
             ) OR NOT EXISTS (
               SELECT 1
               FROM metal_observations m
               JOIN candidate_observations o
                 ON o.case_id=c.case_id
                AND o.input_sha256=c.input_sha256
                AND o.golden_output_sha256=m.output_sha256
               WHERE m.case_id=c.case_id
                 AND m.environment_id=?1
                 AND m.input_sha256=c.input_sha256
                 AND m.oracle_abi=?2
                 AND o.backend='moltenvk'
                 AND o.environment_id=?3
                 AND o.translator_fingerprint=?4
                 AND o.executor_abi=?5
                 AND o.status='match'
             )",
        )
        .map_err(|error| format!("prepare refresh query: {error}"))?;
    let result = statement
        .query_map(
            params![
                metal_environment_id,
                ORACLE_ABI,
                environment_id,
                TRANSLATOR_FINGERPRINT,
                EXECUTOR_ABI
            ],
            |row| row.get(0),
        )
        .map_err(|error| format!("query refresh cases: {error}"))?
        .collect::<Result<HashSet<_>, _>>()
        .map_err(|error| format!("read refresh cases: {error}"));
    result
}

struct SourceIndexRow {
    air_sha256: String,
    stage: String,
    entry: String,
    lib_sha256: String,
    label: String,
    source_offset: Option<i64>,
    source_length: Option<i64>,
}

impl SourceIndexRow {
    fn validate_metadata(&self) -> Result<(), String> {
        crate::source::shard_index_for_hash(&self.air_sha256)?;
        if self
            .air_sha256
            .bytes()
            .any(|byte| byte.is_ascii_uppercase())
        {
            return Err(format!("source {} hash must be lowercase", self.air_sha256));
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
        Ok(())
    }
}

fn read_source_index_shard(
    path: &Path,
    mut consume: impl FnMut(SourceIndexRow) -> Result<(), String>,
) -> Result<(), String> {
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
        let source_offset = reader
            .stream_position()
            .map_err(|error| format!("locate {}:{line_number}: {error}", path.display()))?
            .checked_sub(line.len() as u64)
            .ok_or_else(|| format!("invalid source offset at {}:{line_number}", path.display()))?;
        let mut row = source_index_row_from_canonical_json(&line)
            .map_err(|error| format!("parse {}:{line_number}: {error}", path.display()))?;
        row.source_offset = Some(source_offset.try_into().map_err(|_| {
            format!(
                "source offset at {}:{line_number} is too large",
                path.display()
            )
        })?);
        row.source_length = Some(line.len().try_into().map_err(|_| {
            format!(
                "source row at {}:{line_number} is too large",
                path.display()
            )
        })?);
        row.validate_metadata()
            .map_err(|error| format!("{}:{line_number}: {error}", path.display()))?;
        consume(row).map_err(|error| format!("{}:{line_number}: {error}", path.display()))?;
    }
    Ok(())
}

/// Extract the small index projection without decoding the potentially very large `air_ll` and
/// `blob_b64` JSON strings. Source shards are written by `write_source_shards` with sorted keys;
/// ingestion performs the full source/hash/metadata validation before that canonical write.
fn source_index_row_from_canonical_json(line: &[u8]) -> Result<SourceIndexRow, String> {
    const METADATA_START: &[u8] = b"\",\"air_sha256\":";
    let start = memmem::find(line, METADATA_START)
        .ok_or_else(|| "canonical source row has no air_sha256 boundary".to_string())?
        + 2;
    let metadata = &line[start..];
    Ok(SourceIndexRow {
        air_sha256: extract_json_string(metadata, b"air_sha256")?,
        stage: extract_json_string(metadata, b"stage")?,
        entry: extract_json_string(metadata, b"entry")?,
        lib_sha256: extract_json_string(metadata, b"lib_sha256")?,
        label: extract_json_string(metadata, b"label")?,
        source_offset: None,
        source_length: None,
    })
}

fn extract_json_string(bytes: &[u8], key: &[u8]) -> Result<String, String> {
    let mut pattern = Vec::with_capacity(key.len() + 4);
    pattern.push(b'"');
    pattern.extend_from_slice(key);
    pattern.extend_from_slice(b"\":\"");
    let value_start = memmem::find(bytes, &pattern).ok_or_else(|| {
        format!(
            "canonical source row has no {}",
            String::from_utf8_lossy(key)
        )
    })? + pattern.len();
    let value = &bytes[value_start..];
    let mut escaped = false;
    let end = value
        .iter()
        .position(|byte| {
            if escaped {
                escaped = false;
                false
            } else if *byte == b'\\' {
                escaped = true;
                false
            } else {
                *byte == b'"'
            }
        })
        .ok_or_else(|| {
            format!(
                "canonical source row has unterminated {}",
                String::from_utf8_lossy(key)
            )
        })?;
    let mut quoted = Vec::with_capacity(end + 2);
    quoted.push(b'"');
    quoted.extend_from_slice(&value[..end]);
    quoted.push(b'"');
    serde_json::from_slice(&quoted).map_err(|error| {
        format!(
            "canonical source row has invalid {}: {error}",
            String::from_utf8_lossy(key)
        )
    })
}

fn insert_source(
    statement: &mut rusqlite::Statement<'_>,
    source: &SourceIndexRow,
    shard: Option<usize>,
) -> Result<(), String> {
    statement
        .execute(params![
            source.air_sha256,
            source.stage,
            source.entry,
            source.label,
            shard,
            source.source_offset,
            source.source_length,
        ])
        .map_err(|error| format!("index source: {error}"))?;
    Ok(())
}

fn ensure_source_location_columns(connection: &Connection) -> Result<(), String> {
    let mut statement = connection
        .prepare("PRAGMA table_info(sources)")
        .map_err(|error| format!("inspect source index columns: {error}"))?;
    let columns = statement
        .query_map([], |row| row.get::<_, String>(1))
        .map_err(|error| format!("query source index columns: {error}"))?
        .collect::<Result<HashSet<_>, _>>()
        .map_err(|error| format!("read source index columns: {error}"))?;
    drop(statement);
    if !columns.contains("source_offset") {
        connection
            .execute("ALTER TABLE sources ADD COLUMN source_offset INTEGER", [])
            .map_err(|error| format!("add source_offset index column: {error}"))?;
    }
    if !columns.contains("source_length") {
        connection
            .execute("ALTER TABLE sources ADD COLUMN source_length INTEGER", [])
            .map_err(|error| format!("add source_length index column: {error}"))?;
    }
    Ok(())
}

fn ensure_candidate_fingerprint_column(connection: &Connection) -> Result<(), String> {
    let mut statement = connection
        .prepare("PRAGMA table_info(candidate_observations)")
        .map_err(|error| format!("inspect candidate index columns: {error}"))?;
    let columns = statement
        .query_map([], |row| row.get::<_, String>(1))
        .map_err(|error| format!("query candidate index columns: {error}"))?
        .collect::<Result<HashSet<_>, _>>()
        .map_err(|error| format!("read candidate index columns: {error}"))?;
    drop(statement);
    if !columns.contains("translator_fingerprint") {
        connection
            .execute(
                "ALTER TABLE candidate_observations ADD COLUMN translator_fingerprint TEXT NOT NULL DEFAULT ''",
                [],
            )
            .map_err(|error| format!("add translator_fingerprint index column: {error}"))?;
    }
    Ok(())
}

/// Lazily backfill byte locations for one private source shard.
///
/// Older indexes already know each source's shard and never need a corpus-wide migration. The first
/// lookup into a shard scans that shard alone; subsequent lookups seek directly to the recorded row.
pub(crate) fn index_source_shard_locations(
    root: &Path,
    index_path: &Path,
    shard: usize,
) -> Result<u64, String> {
    if !index_path.is_file() {
        return Ok(0);
    }
    let path = crate::source::source_shard_path(root, shard);
    let stamp = source_shard_stamp(&path)?;
    let mut connection = Connection::open(index_path)
        .map_err(|error| format!("open index {}: {error}", index_path.display()))?;
    ensure_source_location_columns(&connection)?;
    let transaction = connection
        .transaction()
        .map_err(|error| format!("begin source-location update: {error}"))?;
    transaction
        .execute(
            "UPDATE sources SET source_offset=NULL, source_length=NULL WHERE shard=?1",
            [shard],
        )
        .map_err(|error| format!("clear source locations for shard {shard}: {error}"))?;
    {
        let mut update = transaction
            .prepare(
                "UPDATE sources SET stage=?1, entry=?2, label=?3, source_offset=?4, \
                 source_length=?5 WHERE air_sha256=?6 AND shard=?7",
            )
            .map_err(|error| format!("prepare source-location update: {error}"))?;
        let mut insert = transaction
            .prepare(
                "INSERT OR IGNORE INTO sources \
                 (air_sha256, stage, entry, label, shard, source_offset, source_length) \
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            )
            .map_err(|error| format!("prepare source-location insert: {error}"))?;
        read_source_index_shard(&path, |source| {
            update
                .execute(params![
                    source.stage,
                    source.entry,
                    source.label,
                    source.source_offset,
                    source.source_length,
                    source.air_sha256,
                    shard,
                ])
                .map_err(|error| format!("record source location: {error}"))?;
            insert_source(&mut insert, &source, Some(shard))?;
            Ok(())
        })?;
    }
    transaction
        .execute(
            "INSERT OR REPLACE INTO indexed_source_shards (shard, size, modified_ns) \
             VALUES (?1, ?2, ?3)",
            params![shard, stamp.size, stamp.modified_ns],
        )
        .map_err(|error| format!("record source shard {shard} stamp: {error}"))?;
    transaction
        .commit()
        .map_err(|error| format!("commit source-location update: {error}"))?;
    Ok(stamp.size)
}

fn insert_compact_rows(transaction: &Transaction<'_>, root: &Path) -> Result<(), String> {
    let store = CorpusStore::new(root);
    for note in store.read_reviews_for_index()? {
        transaction
            .execute(
                "INSERT INTO reviews (air_sha256, reason, reviewed_by) VALUES (?1, ?2, ?3)",
                params![note.air_sha256, note.reason, note.reviewed_by],
            )
            .map_err(|error| format!("index review note: {error}"))?;
    }
    for case in store.read_all_cases()? {
        transaction
            .execute(
                "INSERT INTO cases (case_id, air_sha256, name, input_sha256) VALUES (?1, ?2, ?3, ?4)",
                params![
                    case.case_id,
                    case.air_sha256,
                    case.name,
                    case.computed_input_sha256()?
                ],
            )
            .map_err(|error| format!("index case: {error}"))?;
    }
    for observation in store.read_metal()? {
        transaction
            .execute(
                "INSERT INTO metal_observations (case_id, environment_id, input_sha256, output_sha256, oracle_abi) VALUES (?1, ?2, ?3, ?4, ?5)",
                params![observation.case_id, observation.environment_id, observation.input_sha256, observation.metal_output_sha256, observation.oracle_abi],
            )
            .map_err(|error| format!("index Metal observation: {error}"))?;
    }
    for backend in [Backend::Moltenvk, Backend::Vulkan] {
        for observation in store.read_candidates(backend)? {
            transaction
                .execute(
                    "INSERT INTO candidate_observations (case_id, backend, environment_id, input_sha256, golden_output_sha256, spv_sha256, translator_fingerprint, executor_abi, status) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
                    params![observation.case_id, backend.directory(), observation.environment_id, observation.input_sha256, observation.golden_output_sha256, observation.spv_sha256, observation.translator_fingerprint, observation.executor_abi, format!("{:?}", observation.status).to_ascii_lowercase()],
                )
                .map_err(|error| format!("index {} observation: {error}", backend.directory()))?;
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SourceShardStamp {
    size: u64,
    modified_ns: i64,
}

fn source_shard_stamp(path: &Path) -> Result<SourceShardStamp, String> {
    let metadata = fs::metadata(path)
        .map_err(|error| format!("read metadata for {}: {error}", path.display()))?;
    Ok(SourceShardStamp {
        size: metadata.len(),
        modified_ns: metadata
            .modified()
            .map_err(|error| format!("read modification time for {}: {error}", path.display()))?
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|error| format!("invalid modification time for {}: {error}", path.display()))?
            .as_nanos()
            .try_into()
            .map_err(|_| format!("modification time for {} is out of range", path.display()))?,
    })
}

fn source_shard_paths(root: &Path) -> Result<Vec<PathBuf>, String> {
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
    Ok(paths)
}

struct RemoveOnDrop {
    path: PathBuf,
    armed: bool,
}

impl RemoveOnDrop {
    fn new(path: PathBuf) -> Self {
        Self { path, armed: true }
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for RemoveOnDrop {
    fn drop(&mut self) {
        if self.armed {
            let _ = fs::remove_file(&self.path);
        }
    }
}

pub fn check_index(root: &Path, index: &Path) -> Result<(), String> {
    if !index.is_file() {
        return Err(format!("index {} does not exist", index.display()));
    }
    let rebuilt = index.with_extension(format!("sqlite.{}.check", std::process::id()));
    let result = (|| {
        fs::copy(index, &rebuilt).map_err(|error| {
            format!(
                "copy index {} to {}: {error}",
                index.display(),
                rebuilt.display()
            )
        })?;
        sync_index(root, &rebuilt)?;
        let current = logical_snapshot(index)?;
        let expected = logical_snapshot(&rebuilt)?;
        if current != expected {
            return Err(format!(
                "index {} differs from canonical corpus shards; rebuild it",
                index.display()
            ));
        }
        Ok(())
    })();
    if rebuilt.exists() {
        fs::remove_file(&rebuilt)
            .map_err(|error| format!("remove {}: {error}", rebuilt.display()))?;
    }
    result
}

fn logical_snapshot(index: &Path) -> Result<Vec<String>, String> {
    let connection = Connection::open(index)
        .map_err(|error| format!("open index {}: {error}", index.display()))?;
    connection
        .query_row("PRAGMA integrity_check", [], |row| row.get::<_, String>(0))
        .map_err(|error| format!("check index integrity: {error}"))
        .and_then(|result| {
            if result == "ok" {
                Ok(())
            } else {
                Err(format!("index integrity check failed: {result}"))
            }
        })?;
    let queries = [
        "SELECT 'source|' || quote(air_sha256) || '|' || quote(stage) || '|' || quote(entry) || '|' || quote(label) || '|' || quote(shard) || '|' || quote(source_offset) || '|' || quote(source_length) FROM sources ORDER BY air_sha256",
        "SELECT 'review|' || quote(air_sha256) || '|' || quote(reason) || '|' || quote(reviewed_by) FROM reviews ORDER BY air_sha256",
        "SELECT 'case|' || quote(case_id) || '|' || quote(air_sha256) || '|' || quote(name) || '|' || quote(input_sha256) FROM cases ORDER BY case_id",
        "SELECT 'metal|' || quote(case_id) || '|' || quote(environment_id) || '|' || quote(input_sha256) || '|' || quote(output_sha256) || '|' || quote(oracle_abi) FROM metal_observations ORDER BY case_id, environment_id",
        "SELECT 'candidate|' || quote(case_id) || '|' || quote(backend) || '|' || quote(environment_id) || '|' || quote(input_sha256) || '|' || quote(golden_output_sha256) || '|' || quote(spv_sha256) || '|' || quote(translator_fingerprint) || '|' || quote(executor_abi) || '|' || quote(status) FROM candidate_observations ORDER BY case_id, backend, environment_id",
        "SELECT 'source-shard|' || quote(shard) || '|' || quote(size) || '|' || quote(modified_ns) FROM indexed_source_shards ORDER BY shard",
    ];
    let mut snapshot = Vec::new();
    for query in queries {
        let mut statement = connection
            .prepare(query)
            .map_err(|error| format!("prepare index check: {error}"))?;
        let rows = statement
            .query_map([], |row| row.get::<_, String>(0))
            .map_err(|error| format!("query index check: {error}"))?;
        snapshot.extend(
            rows.collect::<Result<Vec<_>, _>>()
                .map_err(|error| format!("read index check: {error}"))?,
        );
    }
    Ok(snapshot)
}

pub fn select_queue(
    index: &Path,
    state: QueueState,
    limit: usize,
) -> Result<Vec<QueueRow>, String> {
    let connection = Connection::open(index)
        .map_err(|error| format!("open index {}: {error}", index.display()))?;
    let matching_metal = format!(
        "m.case_id=c.case_id AND m.input_sha256=c.input_sha256 AND m.oracle_abi='{ORACLE_ABI}'"
    );
    let condition = match state {
        QueueState::Unplanned => {
            "NOT EXISTS (SELECT 1 FROM cases c WHERE c.air_sha256=s.air_sha256)".to_string()
        }
        QueueState::Authored => format!(
            "EXISTS (SELECT 1 FROM cases c WHERE c.air_sha256=s.air_sha256) AND EXISTS (SELECT 1 FROM cases c WHERE c.air_sha256=s.air_sha256 AND NOT EXISTS (SELECT 1 FROM metal_observations m WHERE {matching_metal}))"
        ),
        QueueState::MetalQualified => format!(
            "EXISTS (SELECT 1 FROM cases c WHERE c.air_sha256=s.air_sha256) AND NOT EXISTS (SELECT 1 FROM cases c WHERE c.air_sha256=s.air_sha256 AND NOT EXISTS (SELECT 1 FROM metal_observations m WHERE {matching_metal}))"
        ),
    };
    let sql = format!(
        "SELECT s.air_sha256, s.stage, s.entry, s.label, r.reason FROM sources s LEFT JOIN reviews r ON r.air_sha256=s.air_sha256 WHERE {condition} ORDER BY s.air_sha256 LIMIT ?1"
    );
    let mut statement = connection
        .prepare(&sql)
        .map_err(|error| format!("prepare queue query: {error}"))?;
    let rows = statement
        .query_map([limit as i64], |row| {
            Ok(QueueRow {
                air_sha256: row.get(0)?,
                stage: row.get(1)?,
                entry: row.get(2)?,
                label: row.get(3)?,
                state,
                review_reason: row.get(4)?,
            })
        })
        .map_err(|error| format!("query queue: {error}"))?;
    rows.collect::<Result<Vec<_>, _>>()
        .map_err(|error| format!("read queue: {error}"))
}

pub fn status_counts(index: &Path) -> Result<Vec<(String, u64)>, String> {
    let connection = Connection::open(index)
        .map_err(|error| format!("open index {}: {error}", index.display()))?;
    let sql = format!(
            "SELECT state, COUNT(*) FROM (
               SELECT CASE
                 WHEN NOT EXISTS (SELECT 1 FROM cases c WHERE c.air_sha256=s.air_sha256) THEN 'unplanned'
                 WHEN EXISTS (SELECT 1 FROM cases c WHERE c.air_sha256=s.air_sha256 AND NOT EXISTS (SELECT 1 FROM metal_observations m WHERE m.case_id=c.case_id AND m.input_sha256=c.input_sha256 AND m.oracle_abi='{ORACLE_ABI}')) THEN 'authored'
                 ELSE 'metal-qualified'
               END AS state
               FROM sources s
             ) GROUP BY state ORDER BY state"
        );
    let mut statement = connection
        .prepare(&sql)
        .map_err(|error| format!("prepare status query: {error}"))?;
    let result = statement
        .query_map([], |row| Ok((row.get(0)?, row.get::<_, i64>(1)? as u64)))
        .map_err(|error| format!("query status: {error}"))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| format!("read status: {error}"));
    result
}

pub fn evidence_status(root: &Path) -> Result<Vec<EvidenceRow>, String> {
    let store = CorpusStore::new(root);
    let mut cases = store.read_all_cases()?;
    cases.sort_by(|left, right| {
        left.air_sha256
            .cmp(&right.air_sha256)
            .then_with(|| left.name.cmp(&right.name))
    });
    let metals = store.read_metal()?;
    let moltenvk = store.read_candidates(Backend::Moltenvk)?;
    let vulkan = store.read_candidates(Backend::Vulkan)?;
    let mut rows = Vec::new();
    for case in cases {
        let case_metals = metals
            .iter()
            .filter(|observation| observation.case_id == case.case_id)
            .collect::<Vec<_>>();
        let current_metals = case_metals
            .iter()
            .copied()
            .filter(|observation| observation.dependency_matches(&case, ORACLE_ABI))
            .collect::<Vec<_>>();
        let metal = if case_metals.is_empty() {
            EvidenceState::Missing
        } else if current_metals.len() != case_metals.len() {
            EvidenceState::Stale
        } else {
            EvidenceState::Current
        };
        let moltenvk_state = candidate_state(&case, Backend::Moltenvk, &current_metals, &moltenvk);
        let vulkan_state = candidate_state(&case, Backend::Vulkan, &current_metals, &vulkan);
        let mut slots = case_metals
            .iter()
            .map(|observation| EvidenceSlot {
                backend: "metal",
                environment_id: observation.environment_id.clone(),
                state: if observation.dependency_matches(&case, ORACLE_ABI) {
                    EvidenceState::Current
                } else {
                    EvidenceState::Stale
                },
            })
            .collect::<Vec<_>>();
        for (backend, observations) in [
            (Backend::Moltenvk, moltenvk.as_slice()),
            (Backend::Vulkan, vulkan.as_slice()),
        ] {
            slots.extend(
                observations
                    .iter()
                    .filter(|observation| observation.case_id == case.case_id)
                    .map(|observation| EvidenceSlot {
                        backend: backend.directory(),
                        environment_id: observation.environment_id.clone(),
                        state: candidate_slot_state(&case, backend, &current_metals, observation),
                    }),
            );
        }
        slots.sort_by(|left, right| {
            left.backend
                .cmp(right.backend)
                .then_with(|| left.environment_id.cmp(&right.environment_id))
        });
        rows.push(EvidenceRow {
            case_id: case.case_id,
            name: case.name,
            metal,
            moltenvk: moltenvk_state,
            vulkan: vulkan_state,
            translation_error: None,
            slots,
        });
    }
    Ok(rows)
}

fn candidate_state(
    case: &crate::case::AuthoredCase,
    backend: Backend,
    metals: &[&crate::observation::MetalObservation],
    candidates: &[crate::observation::CandidateObservation],
) -> EvidenceState {
    let candidates = candidates
        .iter()
        .filter(|observation| observation.case_id == case.case_id)
        .collect::<Vec<_>>();
    if candidates.is_empty() {
        return EvidenceState::Missing;
    }
    let states = candidates
        .iter()
        .map(|candidate| candidate_slot_state(case, backend, metals, candidate))
        .collect::<Vec<_>>();
    summarize_existing_states(&states)
}

fn summarize_existing_states(states: &[EvidenceState]) -> EvidenceState {
    if states.contains(&EvidenceState::Failed) {
        EvidenceState::Failed
    } else if states.contains(&EvidenceState::Stale) {
        EvidenceState::Stale
    } else if states.is_empty() {
        EvidenceState::Missing
    } else {
        EvidenceState::Current
    }
}

fn candidate_slot_state(
    case: &crate::case::AuthoredCase,
    backend: Backend,
    metals: &[&crate::observation::MetalObservation],
    candidate: &crate::observation::CandidateObservation,
) -> EvidenceState {
    metals
        .iter()
        .find_map(|metal| {
            candidate
                .dependency_matches(&CandidateDependencies {
                    case,
                    metal,
                    spv_sha256: &candidate.spv_sha256,
                    translator_fingerprint: TRANSLATOR_FINGERPRINT,
                    backend,
                    environment_id: &candidate.environment_id,
                    executor_abi: EXECUTOR_ABI,
                })
                .then_some(match candidate.status {
                    crate::observation::CandidateStatus::Match => EvidenceState::Current,
                    crate::observation::CandidateStatus::Mismatch => EvidenceState::Failed,
                })
        })
        .unwrap_or(EvidenceState::Stale)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::case::{
        AuthoredCase, BufferResource, Comparison, Dispatch, ExecutionSafety, OutputSelection,
        ResourceRole, Stage,
    };
    use crate::observation::{
        CandidateObservation, CandidateStatus, ComparisonResult, MetalObservation, MetalStatus,
    };
    use crate::review::ReviewNote;
    use crate::ScratchDir;

    fn synthetic_public_case() -> AuthoredCase {
        let source = public_sources().unwrap().remove(0);
        let mut case = AuthoredCase {
            air_sha256: source.air_sha256,
            case_id: String::new(),
            name: "index-test".into(),
            entry: source.entry,
            stage: Stage::Kernel,
            buffers: vec![BufferResource {
                binding: 0,
                role: ResourceRole::Output,
                bytes_b64: None,
                initial_bytes_b64: Some("AAAAAA==".into()),
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

    fn synthetic_observations(case: &AuthoredCase) -> (MetalObservation, CandidateObservation) {
        let input_sha256 = case.computed_input_sha256().unwrap();
        let output_b64 = "KgAAAA==";
        let output_sha256 = crate::hash::sha256_bytes(&[42, 0, 0, 0]);
        let metal = MetalObservation {
            case_id: case.case_id.clone(),
            air_sha256: case.air_sha256.clone(),
            input_sha256: input_sha256.clone(),
            metal_output_sha256: output_sha256.clone(),
            output_b64: output_b64.into(),
            environment_id: "metal-env".into(),
            environment: serde_json::json!({}),
            oracle_abi: ORACLE_ABI.into(),
            status: MetalStatus::Qualified,
        };
        let candidate = CandidateObservation {
            case_id: case.case_id.clone(),
            air_sha256: case.air_sha256.clone(),
            input_sha256,
            golden_output_sha256: output_sha256.clone(),
            spv_sha256: "33".repeat(32),
            translator_fingerprint: TRANSLATOR_FINGERPRINT.into(),
            candidate_output_sha256: output_sha256,
            output_b64: output_b64.into(),
            backend: Backend::Moltenvk,
            environment_id: "moltenvk-env".into(),
            environment: serde_json::json!({}),
            executor_abi: EXECUTOR_ABI.into(),
            comparison: ComparisonResult::Exact,
            status: CandidateStatus::Match,
        };
        (metal, candidate)
    }

    #[test]
    fn rebuilding_index_twice_is_logically_equivalent() {
        let scratch = ScratchDir::new("index-rebuild").unwrap();
        let case = synthetic_public_case();
        let (metal, candidate) = synthetic_observations(&case);
        let store = CorpusStore::new(scratch.path());
        store.put_case(case.clone()).unwrap();
        store.upsert_metal(metal).unwrap();
        store.upsert_candidate(candidate).unwrap();

        let source = public_sources()
            .unwrap()
            .into_iter()
            .find(|source| source.air_sha256 != case.air_sha256)
            .unwrap();
        store
            .upsert_review(ReviewNote {
                air_sha256: source.air_sha256.clone(),
                reason: "needs an explicit semantic input".into(),
                reviewed_by: "test".into(),
            })
            .unwrap();
        let first = scratch.path().join("first.sqlite");
        let second = scratch.path().join("second.sqlite");
        rebuild_index(scratch.path(), &first).unwrap();
        rebuild_index(scratch.path(), &second).unwrap();
        assert_eq!(
            status_counts(&first).unwrap(),
            status_counts(&second).unwrap()
        );
        assert_eq!(
            select_queue(&first, QueueState::Unplanned, 100).unwrap(),
            select_queue(&second, QueueState::Unplanned, 100).unwrap()
        );
        assert_eq!(
            select_queue(&first, QueueState::Unplanned, 100)
                .unwrap()
                .into_iter()
                .find(|row| row.air_sha256 == source.air_sha256)
                .unwrap()
                .review_reason
                .as_deref(),
            Some("needs an explicit semantic input")
        );
        check_index(scratch.path(), &first).unwrap();
    }

    #[test]
    fn incremental_sync_reindexes_only_a_changed_source_shard() {
        let scratch = ScratchDir::new("index-incremental").unwrap();
        let mut source = public_sources().unwrap().into_iter().next().unwrap();
        source.air_ll.push_str("\n; private index fixture\n");
        source.air_sha256 = crate::hash::sha256_bytes(source.air_ll.as_bytes());
        source.lib_sha256 = "owned-synthetic".into();
        source.label = "private/before".into();
        crate::source::write_source_shards(scratch.path(), [source.clone()]).unwrap();
        let index = scratch.path().join("index.sqlite");
        rebuild_index(scratch.path(), &index).unwrap();

        source.label = "private/after-a-changed-shard".into();
        source.lib_sha256 = "00".repeat(32);
        crate::source::merge_source_shards(scratch.path(), [source.clone()]).unwrap();
        let stats = sync_index_with_stats(scratch.path(), &index).unwrap();
        assert_eq!(stats.source_shards_scanned, 1);
        assert_eq!(
            stats.source_bytes_scanned,
            fs::metadata(crate::source::source_shard_path(
                scratch.path(),
                crate::source::shard_index_for_hash(&source.air_sha256).unwrap()
            ))
            .unwrap()
            .len()
        );

        let row = select_queue(&index, QueueState::Unplanned, usize::MAX)
            .unwrap()
            .into_iter()
            .find(|row| row.air_sha256 == source.air_sha256)
            .unwrap();
        assert_eq!(row.label, source.label);
        check_index(scratch.path(), &index).unwrap();
    }

    #[test]
    fn source_merge_publishes_locations_without_a_followup_body_scan() {
        let scratch = ScratchDir::new("index-merge-handoff").unwrap();
        let mut source = public_sources().unwrap().into_iter().next().unwrap();
        source
            .air_ll
            .push_str("\n; indexed merge handoff fixture\n");
        source.air_sha256 = crate::hash::sha256_bytes(source.air_ll.as_bytes());
        source.lib_sha256 = "owned-synthetic".into();
        source.label = "private/before".into();
        crate::source::write_source_shards(scratch.path(), [source.clone()]).unwrap();
        let index = default_index_path(scratch.path());
        rebuild_index(scratch.path(), &index).unwrap();

        source.label = "private/after-direct-handoff".into();
        source.lib_sha256 = "00".repeat(32);
        crate::source::merge_source_shards(scratch.path(), [source.clone()]).unwrap();

        let stats = sync_index_with_stats(scratch.path(), &index).unwrap();
        assert_eq!(stats.source_shards_scanned, 0);
        assert_eq!(stats.source_bytes_scanned, 0);
        let connection = Connection::open(&index).unwrap();
        let indexed = connection
            .query_row(
                "SELECT label, source_offset, source_length FROM sources WHERE air_sha256=?1",
                [&source.air_sha256],
                |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, Option<i64>>(1)?,
                        row.get::<_, Option<i64>>(2)?,
                    ))
                },
            )
            .unwrap();
        assert_eq!(indexed.0, source.label);
        assert!(indexed.1.is_some());
        assert!(indexed.2.is_some());
        let read_stats = crate::source::for_each_indexed_source_analysis_with_stats(
            scratch.path(),
            &index,
            std::slice::from_ref(&source.air_sha256),
            |_| Ok(()),
        )
        .unwrap();
        assert_eq!(read_stats.rows, 1);
        assert_eq!(read_stats.source_shards_opened, 1);
        assert_eq!(read_stats.source_bytes_read, indexed.2.unwrap() as u64);
        assert_eq!(read_stats.repair_shards_scanned, 0);
        assert_eq!(read_stats.repair_bytes_scanned, 0);
        check_index(scratch.path(), &index).unwrap();
    }

    #[test]
    fn legacy_migration_and_lookup_never_open_an_unrelated_shard() {
        let scratch = ScratchDir::new("index-lazy-locations").unwrap();
        let mut source = public_sources().unwrap().into_iter().next().unwrap();
        source
            .air_ll
            .push_str("\n; private lazy-location fixture\n");
        source.air_sha256 = crate::hash::sha256_bytes(source.air_ll.as_bytes());
        source.lib_sha256 = "owned-synthetic".into();
        source.label = "private/lazy".into();
        let source_shard = crate::source::shard_index_for_hash(&source.air_sha256).unwrap();
        let mut other = source.clone();
        for nonce in 0..u32::MAX {
            other.air_ll = format!("{}\n; unrelated shard {nonce}\n", source.air_ll);
            other.air_sha256 = crate::hash::sha256_bytes(other.air_ll.as_bytes());
            if crate::source::shard_index_for_hash(&other.air_sha256).unwrap() != source_shard {
                break;
            }
        }
        other.label = "private/unrelated".into();
        crate::source::write_source_shards(scratch.path(), [source.clone(), other.clone()])
            .unwrap();
        let index = scratch.path().join(".index.sqlite");
        rebuild_index(scratch.path(), &index).unwrap();
        let connection = Connection::open(&index).unwrap();
        connection
            .execute(
                "UPDATE sources SET source_offset=NULL, source_length=NULL WHERE air_sha256 IN (?1, ?2)",
                [&source.air_sha256, &other.air_sha256],
            )
            .unwrap();
        connection
            .execute("DROP TABLE indexed_source_shards", [])
            .unwrap();
        connection.execute("DROP TABLE index_metadata", []).unwrap();
        connection
            .execute_batch(
                "CREATE TABLE indexed_source_shards (
                   shard INTEGER PRIMARY KEY,
                   size INTEGER NOT NULL,
                   modified_ns INTEGER NOT NULL
                 );
                 INSERT INTO indexed_source_shards (shard, size, modified_ns)
                   VALUES (63, 1, 1);",
            )
            .unwrap();
        drop(connection);

        // Preserve readable metadata but make any attempted body scan fail. Both migration and a
        // lookup in `source_shard` must leave this unrelated shard unopened.
        let unrelated_path = crate::source::source_shard_path(
            scratch.path(),
            crate::source::shard_index_for_hash(&other.air_sha256).unwrap(),
        );
        let unrelated_len = fs::metadata(&unrelated_path).unwrap().len() as usize;
        fs::write(&unrelated_path, vec![b'x'; unrelated_len]).unwrap();

        let stats = sync_index_with_stats(scratch.path(), &index).unwrap();
        assert_eq!(stats.source_shards_scanned, 0);
        assert_eq!(stats.source_bytes_scanned, 0);
        let connection = Connection::open(&index).unwrap();
        let before = connection
            .query_row(
                "SELECT source_offset FROM sources WHERE air_sha256=?1",
                [&source.air_sha256],
                |row| row.get::<_, Option<i64>>(0),
            )
            .unwrap();
        assert_eq!(before, None);
        drop(connection);

        assert_eq!(
            crate::source::find_source(scratch.path(), &source.air_sha256)
                .unwrap()
                .unwrap()
                .label,
            source.label
        );
        let connection = Connection::open(&index).unwrap();
        let after = connection
            .query_row(
                "SELECT source_offset FROM sources WHERE air_sha256=?1",
                [&source.air_sha256],
                |row| row.get::<_, Option<i64>>(0),
            )
            .unwrap();
        assert!(after.is_some());
        let unrelated = connection
            .query_row(
                "SELECT source_offset FROM sources WHERE air_sha256=?1",
                [&other.air_sha256],
                |row| row.get::<_, Option<i64>>(0),
            )
            .unwrap();
        assert_eq!(unrelated, None, "lookup must not scan an unrelated shard");
    }

    #[test]
    fn refresh_selection_uses_only_exact_compact_evidence_slots() {
        let scratch = ScratchDir::new("index-refresh-selection").unwrap();
        let case = synthetic_public_case();
        CorpusStore::new(scratch.path())
            .put_case(case.clone())
            .unwrap();
        let index = scratch.path().join("index.sqlite");
        rebuild_index(scratch.path(), &index).unwrap();
        let input_sha256 = case.computed_input_sha256().unwrap();
        let connection = Connection::open(&index).unwrap();
        connection
            .execute(
                "INSERT INTO metal_observations \
                 (case_id, environment_id, input_sha256, output_sha256, oracle_abi) \
                 VALUES (?1, 'metal-env', ?2, 'output', ?3)",
                params![case.case_id, input_sha256, ORACLE_ABI],
            )
            .unwrap();
        connection
            .execute(
                "INSERT INTO candidate_observations \
                 (case_id, backend, environment_id, input_sha256, golden_output_sha256, \
                  spv_sha256, translator_fingerprint, executor_abi, status) \
                 VALUES (?1, 'moltenvk', 'vulkan-env', ?2, 'output', 'spv', ?3, ?4, 'match')",
                params![
                    case.case_id,
                    input_sha256,
                    TRANSLATOR_FINGERPRINT,
                    EXECUTOR_ABI
                ],
            )
            .unwrap();
        drop(connection);

        assert!(
            pending_macos_refresh_case_ids(&index, "metal-env", "vulkan-env")
                .unwrap()
                .is_empty()
        );
        assert!(
            pending_macos_refresh_case_ids(&index, "other-metal-env", "vulkan-env")
                .unwrap()
                .contains(&case.case_id)
        );
        assert!(
            pending_macos_refresh_case_ids(&index, "metal-env", "other-vulkan-env")
                .unwrap()
                .contains(&case.case_id)
        );
    }

    #[test]
    fn status_summary_does_not_hide_failed_or_stale_slots() {
        assert_eq!(
            summarize_existing_states(&[EvidenceState::Current, EvidenceState::Stale]),
            EvidenceState::Stale
        );
        assert_eq!(
            summarize_existing_states(&[EvidenceState::Current, EvidenceState::Failed]),
            EvidenceState::Failed
        );
        assert_eq!(summarize_existing_states(&[]), EvidenceState::Missing);
    }

    #[test]
    fn failed_index_rebuild_removes_its_temporary_database() {
        let scratch = ScratchDir::new("index-failure-cleanup").unwrap();
        fs::create_dir_all(scratch.path().join("cases")).unwrap();
        fs::write(scratch.path().join("cases/shard_000.jsonl"), b"{}\n").unwrap();
        let destination = scratch.path().join("index.sqlite");
        assert!(rebuild_index(scratch.path(), &destination).is_err());
        assert!(!destination
            .with_extension(format!("sqlite.{}.tmp", std::process::id()))
            .exists());
    }
}
