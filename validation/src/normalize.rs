//! Explicit, recoverable migration from legacy sanitized AIR identities to the current contract.

use crate::case::{AuthoredCase, IntersectionFunctionTableEntry};
use crate::hash::sha256_bytes;
use crate::jsonl::to_sorted_json_string;
use crate::library_module::{self, LibraryModuleRow};
use crate::observation::Backend;
use crate::source::{self, SourceRow, SHARD_COUNT};
use crate::store::CorpusStore;
use rusqlite::{params, Connection, OptionalExtension, Transaction};
use std::collections::{BTreeMap, HashMap};
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NormalizationStats {
    pub source_rows_before: usize,
    pub source_rows_after: usize,
    pub module_rows_before: usize,
    pub module_rows_after: usize,
    pub cases_rewritten: usize,
    pub cases_deduplicated: usize,
    pub reviews_superseded: usize,
    pub observations_invalidated: usize,
    pub observations_deduplicated: usize,
}

struct PreparedDirectory(PathBuf);

impl Drop for PreparedDirectory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// Normalize every private AIR identity and all committed references in one recoverable publish.
pub fn normalize_air_identities(root: &Path) -> Result<NormalizationStats, String> {
    let store = CorpusStore::new(root);
    store.recover_transactions()?;
    let prepared_path = root.join(format!(".normalize-air.{}", std::process::id()));
    if prepared_path.exists() {
        fs::remove_dir_all(&prepared_path)
            .map_err(|error| format!("remove stale {}: {error}", prepared_path.display()))?;
    }
    fs::create_dir_all(&prepared_path)
        .map_err(|error| format!("create {}: {error}", prepared_path.display()))?;
    let prepared = PreparedDirectory(prepared_path);
    let database = prepared.0.join("identities.sqlite");
    let mut connection = Connection::open(&database)
        .map_err(|error| format!("open {}: {error}", database.display()))?;
    connection
        .execute_batch(
            "PRAGMA journal_mode=DELETE;
             PRAGMA synchronous=FULL;
             CREATE TABLE sources (
               hash TEXT PRIMARY KEY, stage TEXT NOT NULL, entry TEXT NOT NULL,
               air_ll TEXT NOT NULL, blob_b64 TEXT, label TEXT NOT NULL
             );
             CREATE TABLE source_memberships (
               hash TEXT NOT NULL, library TEXT NOT NULL, PRIMARY KEY(hash, library)
             );
             CREATE TABLE modules (
               hash TEXT PRIMARY KEY, air_ll TEXT NOT NULL, blob_b64 TEXT NOT NULL,
               label TEXT NOT NULL
             );
             CREATE TABLE module_memberships (
               hash TEXT NOT NULL, library TEXT NOT NULL, PRIMARY KEY(hash, library)
             );",
        )
        .map_err(|error| format!("create normalization database: {error}"))?;

    let mut stats = NormalizationStats::default();
    let mut source_map = HashMap::new();
    let mut module_map = HashMap::new();
    ingest_sources(root, &mut connection, &mut source_map, &mut stats)?;
    ingest_modules(root, &mut connection, &mut module_map, &mut stats)?;

    let old_cases = store.read_all_cases_for_identity_migration()?;
    let old_metal = store.read_metal_for_identity_migration()?;
    let old_moltenvk = store.read_candidates_for_identity_migration(Backend::Moltenvk)?;
    let old_vulkan = store.read_candidates_for_identity_migration(Backend::Vulkan)?;
    let old_reviews = store.read_reviews()?;
    let (cases, case_map) = remap_cases(old_cases, &source_map, &module_map, &mut stats)?;
    let case_airs = cases
        .iter()
        .map(|case| case.air_sha256.as_str())
        .collect::<std::collections::HashSet<_>>();
    let reviews = remap_reviews(old_reviews, &source_map, &case_airs, &mut stats)?;

    let mut files = write_normalized_air(&prepared.0, &connection, &mut stats)?;
    files.extend(write_aligned_rows(
        &prepared.0,
        "cases",
        cases,
        |row: &AuthoredCase| &row.air_sha256,
    )?);
    files.extend(write_aligned_rows(
        &prepared.0,
        "reviews",
        reviews,
        |row: &crate::review::ReviewNote| &row.air_sha256,
    )?);
    let observations_before = old_metal.len() + old_moltenvk.len() + old_vulkan.len();
    let metal = old_metal
        .into_iter()
        .filter(|row| {
            case_map
                .get(&row.case_id)
                .is_some_and(|new| new == &row.case_id)
        })
        .collect::<Vec<_>>();
    let moltenvk = old_moltenvk
        .into_iter()
        .filter(|row| {
            case_map
                .get(&row.case_id)
                .is_some_and(|new| new == &row.case_id)
        })
        .collect::<Vec<_>>();
    let vulkan = old_vulkan
        .into_iter()
        .filter(|row| {
            case_map
                .get(&row.case_id)
                .is_some_and(|new| new == &row.case_id)
        })
        .collect::<Vec<_>>();
    stats.observations_invalidated =
        observations_before - metal.len() - moltenvk.len() - vulkan.len();
    let (metal, metal_deduplicated) = deduplicate_exact_rows(
        metal,
        |row| (row.case_id.clone(), row.environment_id.clone()),
        "Metal observation",
    )?;
    let (moltenvk, moltenvk_deduplicated) = deduplicate_exact_rows(
        moltenvk,
        |row| (row.case_id.clone(), row.environment_id.clone()),
        "MoltenVK observation",
    )?;
    let (vulkan, vulkan_deduplicated) = deduplicate_exact_rows(
        vulkan,
        |row| (row.case_id.clone(), row.environment_id.clone()),
        "Vulkan observation",
    )?;
    stats.observations_deduplicated =
        metal_deduplicated + moltenvk_deduplicated + vulkan_deduplicated;
    files.extend(write_aligned_rows(
        &prepared.0,
        "observations/metal",
        metal,
        |row: &crate::observation::MetalObservation| &row.air_sha256,
    )?);
    files.extend(write_aligned_rows(
        &prepared.0,
        "observations/moltenvk",
        moltenvk,
        |row: &crate::observation::CandidateObservation| &row.air_sha256,
    )?);
    files.extend(write_aligned_rows(
        &prepared.0,
        "observations/vulkan",
        vulkan,
        |row: &crate::observation::CandidateObservation| &row.air_sha256,
    )?);

    drop(connection);
    let mut deletions = Vec::new();
    files.retain(|(target, prepared)| {
        let empty = fs::metadata(prepared).is_ok_and(|metadata| metadata.len() == 0);
        if empty {
            let _ = fs::remove_file(prepared);
            deletions.push(target.clone());
        }
        !empty
    });
    store.commit_prepared_files_and_deletions(files, deletions)?;
    crate::index::rebuild_index(root, &crate::index::default_index_path(root))?;
    Ok(stats)
}

fn ingest_sources(
    root: &Path,
    connection: &mut Connection,
    mapping: &mut HashMap<String, String>,
    stats: &mut NormalizationStats,
) -> Result<(), String> {
    for shard in 0..SHARD_COUNT {
        let path = source::source_shard_path(root, shard);
        if !path.is_file() {
            continue;
        }
        let transaction = connection
            .transaction()
            .map_err(|error| format!("begin source normalization: {error}"))?;
        source::for_each_source_shard(&path, |mut row| {
            stats.source_rows_before += 1;
            let old = row.air_sha256.clone();
            row.air_ll = metal2vulkan::tools::sanitize_ll_text_with_datalayout(&row.air_ll).0;
            row.air_sha256 = sha256_bytes(row.air_ll.as_bytes());
            mapping.insert(old, row.air_sha256.clone());
            insert_source(&transaction, &row)
        })?;
        transaction
            .commit()
            .map_err(|error| format!("commit source normalization: {error}"))?;
    }
    Ok(())
}

fn insert_source(transaction: &Transaction<'_>, row: &SourceRow) -> Result<(), String> {
    let existing = transaction
        .query_row(
            "SELECT stage, entry, air_ll, blob_b64, label FROM sources WHERE hash=?1",
            [&row.air_sha256],
            |record| {
                Ok((
                    record.get::<_, String>(0)?,
                    record.get::<_, String>(1)?,
                    record.get::<_, String>(2)?,
                    record.get::<_, Option<String>>(3)?,
                    record.get::<_, String>(4)?,
                ))
            },
        )
        .optional()
        .map_err(|error| format!("query normalized source {}: {error}", row.air_sha256))?;
    if let Some((stage, entry, air_ll, blob, label)) = existing {
        if (stage.as_str(), entry.as_str(), air_ll.as_str())
            != (row.stage.as_str(), row.entry.as_str(), row.air_ll.as_str())
        {
            return Err(format!("canonical source collision {}", row.air_sha256));
        }
        let preferred_blob =
            if source::source_blob_is_preferred(blob.as_deref(), row.blob_b64.as_deref()) {
                row.blob_b64.as_ref()
            } else {
                blob.as_ref()
            };
        transaction
            .execute(
                "UPDATE sources SET blob_b64=?2, label=?3 WHERE hash=?1",
                params![row.air_sha256, preferred_blob, label.min(row.label.clone())],
            )
            .map_err(|error| format!("merge normalized source {}: {error}", row.air_sha256))?;
    } else {
        transaction
            .execute(
                "INSERT INTO sources VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                params![
                    row.air_sha256,
                    row.stage,
                    row.entry,
                    row.air_ll,
                    row.blob_b64,
                    row.label
                ],
            )
            .map_err(|error| format!("insert normalized source {}: {error}", row.air_sha256))?;
    }
    for library in &row.lib_sha256s {
        transaction
            .execute(
                "INSERT OR IGNORE INTO source_memberships VALUES (?1, ?2)",
                params![row.air_sha256, library],
            )
            .map_err(|error| format!("merge source membership: {error}"))?;
    }
    Ok(())
}

fn ingest_modules(
    root: &Path,
    connection: &mut Connection,
    mapping: &mut HashMap<String, String>,
    stats: &mut NormalizationStats,
) -> Result<(), String> {
    for shard in 0..SHARD_COUNT {
        let path = library_module::library_module_shard_path(root, shard);
        if !path.is_file() {
            continue;
        }
        let transaction = connection
            .transaction()
            .map_err(|error| format!("begin module normalization: {error}"))?;
        library_module::for_each_library_module_shard(&path, |mut row| {
            stats.module_rows_before += 1;
            let old = row.module_sha256.clone();
            row.air_ll = metal2vulkan::tools::sanitize_ll_text_with_datalayout(&row.air_ll).0;
            row.module_sha256 = sha256_bytes(row.air_ll.as_bytes());
            mapping.insert(old, row.module_sha256.clone());
            insert_module(&transaction, &row)
        })?;
        transaction
            .commit()
            .map_err(|error| format!("commit module normalization: {error}"))?;
    }
    Ok(())
}

fn insert_module(transaction: &Transaction<'_>, row: &LibraryModuleRow) -> Result<(), String> {
    let existing = transaction
        .query_row(
            "SELECT air_ll, blob_b64, label FROM modules WHERE hash=?1",
            [&row.module_sha256],
            |record| {
                Ok((
                    record.get::<_, String>(0)?,
                    record.get::<_, String>(1)?,
                    record.get::<_, String>(2)?,
                ))
            },
        )
        .optional()
        .map_err(|error| format!("query normalized module {}: {error}", row.module_sha256))?;
    if let Some((air_ll, blob, label)) = existing {
        if air_ll != row.air_ll {
            return Err(format!("canonical module collision {}", row.module_sha256));
        }
        transaction
            .execute(
                "UPDATE modules SET blob_b64=?2, label=?3 WHERE hash=?1",
                params![
                    row.module_sha256,
                    blob.min(row.blob_b64.clone()),
                    label.min(row.label.clone())
                ],
            )
            .map_err(|error| format!("merge normalized module {}: {error}", row.module_sha256))?;
    } else {
        transaction
            .execute(
                "INSERT INTO modules VALUES (?1, ?2, ?3, ?4)",
                params![row.module_sha256, row.air_ll, row.blob_b64, row.label],
            )
            .map_err(|error| format!("insert normalized module {}: {error}", row.module_sha256))?;
    }
    for library in &row.lib_sha256s {
        transaction
            .execute(
                "INSERT OR IGNORE INTO module_memberships VALUES (?1, ?2)",
                params![row.module_sha256, library],
            )
            .map_err(|error| format!("merge module membership: {error}"))?;
    }
    Ok(())
}

fn remap_cases(
    cases: Vec<AuthoredCase>,
    source_map: &HashMap<String, String>,
    module_map: &HashMap<String, String>,
    stats: &mut NormalizationStats,
) -> Result<(Vec<AuthoredCase>, HashMap<String, String>), String> {
    let mut identities = BTreeMap::<String, AuthoredCase>::new();
    let mut case_map = HashMap::new();
    for mut case in cases {
        let old_case_id = case.case_id.clone();
        if let Some(hash) = source_map.get(&case.air_sha256) {
            case.air_sha256 = hash.clone();
        }
        for reference in &mut case.visible_function_references {
            remap_module_hash(&mut reference.module_sha256, module_map);
        }
        for table in &mut case.visible_function_tables {
            for entry in &mut table.entries {
                remap_module_hash(&mut entry.module_sha256, module_map);
            }
        }
        for table in case
            .intersection_function_tables
            .iter_mut()
            .map(|table| &mut table.entries)
            .chain(
                case.argument_buffer_intersection_function_tables
                    .iter_mut()
                    .map(|table| &mut table.entries),
            )
        {
            for entry in table {
                if let IntersectionFunctionTableEntry::Linked { module_sha256, .. } = entry {
                    remap_module_hash(module_sha256, module_map);
                }
            }
        }
        case.case_id = case.computed_case_id()?;
        case_map.insert(old_case_id.clone(), case.case_id.clone());
        if old_case_id != case.case_id {
            stats.cases_rewritten += 1;
        }
        match identities.get_mut(&case.case_id) {
            Some(existing) => {
                existing.name = existing.name.clone().min(case.name);
                existing.rationale = merge_annotation(
                    "rationale",
                    &existing.rationale,
                    &case.rationale,
                    &case.case_id,
                )?;
                existing.authored_by = merge_annotation(
                    "authored_by",
                    &existing.authored_by,
                    &case.authored_by,
                    &case.case_id,
                )?;
                stats.cases_deduplicated += 1;
            }
            None => {
                identities.insert(case.case_id.clone(), case);
            }
        }
    }
    let mut slots = BTreeMap::<(String, String), String>::new();
    let mut cases = identities.into_values().collect::<Vec<_>>();
    for case in &cases {
        let key = (case.air_sha256.clone(), case.name.clone());
        if let Some(existing) = slots.insert(key, case.case_id.clone()) {
            return Err(format!(
                "normalization gives case name {:?} conflicting semantics {} and {}",
                case.name, existing, case.case_id
            ));
        }
    }
    cases.sort_by(|left, right| {
        left.air_sha256
            .cmp(&right.air_sha256)
            .then_with(|| left.name.cmp(&right.name))
    });
    Ok((cases, case_map))
}

fn merge_annotation(
    field: &str,
    current: &Option<String>,
    candidate: &Option<String>,
    case_id: &str,
) -> Result<Option<String>, String> {
    match (current, candidate) {
        (Some(current), Some(candidate)) if current != candidate => Err(format!(
            "duplicate semantic case {case_id} has conflicting {field} annotations"
        )),
        (Some(current), _) => Ok(Some(current.clone())),
        (None, Some(candidate)) => Ok(Some(candidate.clone())),
        (None, None) => Ok(None),
    }
}

fn deduplicate_exact_rows<T, K>(
    rows: Vec<T>,
    key: impl Fn(&T) -> K,
    label: &str,
) -> Result<(Vec<T>, usize), String>
where
    T: PartialEq,
    K: Ord + std::fmt::Debug,
{
    let mut unique = BTreeMap::new();
    let mut duplicates = 0;
    for row in rows {
        let key = key(&row);
        match unique.get(&key) {
            Some(existing) if existing == &row => duplicates += 1,
            Some(_) => return Err(format!("conflicting {label} slot {key:?}")),
            None => {
                unique.insert(key, row);
            }
        }
    }
    Ok((unique.into_values().collect(), duplicates))
}

fn remap_module_hash(hash: &mut String, mapping: &HashMap<String, String>) {
    if let Some(canonical) = mapping.get(hash) {
        hash.clone_from(canonical);
    }
}

fn remap_reviews(
    reviews: Vec<crate::review::ReviewNote>,
    source_map: &HashMap<String, String>,
    case_airs: &std::collections::HashSet<&str>,
    stats: &mut NormalizationStats,
) -> Result<Vec<crate::review::ReviewNote>, String> {
    let mut rows = BTreeMap::new();
    for mut review in reviews {
        if let Some(hash) = source_map.get(&review.air_sha256) {
            review.air_sha256 = hash.clone();
        }
        if case_airs.contains(review.air_sha256.as_str()) {
            stats.reviews_superseded += 1;
            continue;
        }
        match rows.get(&review.air_sha256) {
            Some(existing) if existing == &review => {}
            Some(_) => {
                return Err(format!(
                    "normalization gives AIR {} conflicting review notes",
                    review.air_sha256
                ));
            }
            None => {
                rows.insert(review.air_sha256.clone(), review);
            }
        }
    }
    Ok(rows.into_values().collect())
}

fn write_normalized_air(
    prepared: &Path,
    connection: &Connection,
    stats: &mut NormalizationStats,
) -> Result<Vec<(PathBuf, PathBuf)>, String> {
    let mut files = Vec::new();
    let mut source_writers = prepared_shard_writers(prepared, "local/sources", &mut files)?;
    let mut source_memberships = connection
        .prepare("SELECT library FROM source_memberships WHERE hash=?1 ORDER BY library")
        .map_err(|error| format!("prepare normalized source memberships: {error}"))?;
    let mut sources = connection
        .prepare("SELECT hash, stage, entry, air_ll, blob_b64, label FROM sources ORDER BY hash")
        .map_err(|error| format!("prepare normalized sources: {error}"))?;
    let rows = sources
        .query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, Option<String>>(4)?,
                row.get::<_, String>(5)?,
            ))
        })
        .map_err(|error| format!("query normalized sources: {error}"))?;
    for row in rows {
        let (hash, stage, entry, air_ll, blob_b64, label) =
            row.map_err(|error| format!("read normalized source: {error}"))?;
        let lib_sha256s = source_memberships
            .query_map([&hash], |row| row.get(0))
            .map_err(|error| format!("query source memberships: {error}"))?
            .collect::<Result<Vec<String>, _>>()
            .map_err(|error| format!("read source memberships: {error}"))?;
        let row = SourceRow {
            air_sha256: hash.clone(),
            stage,
            entry,
            air_ll,
            blob_b64,
            lib_sha256s,
            label,
        };
        row.validate()?;
        write_json_line(
            &mut source_writers[source::shard_index_for_hash(&hash)?],
            &row,
        )?;
        stats.source_rows_after += 1;
    }
    finish_writers(&mut source_writers)?;

    let mut module_writers = prepared_shard_writers(prepared, "local/library-modules", &mut files)?;
    let mut module_memberships = connection
        .prepare("SELECT library FROM module_memberships WHERE hash=?1 ORDER BY library")
        .map_err(|error| format!("prepare normalized module memberships: {error}"))?;
    let mut modules = connection
        .prepare("SELECT hash, air_ll, blob_b64, label FROM modules ORDER BY hash")
        .map_err(|error| format!("prepare normalized modules: {error}"))?;
    let rows = modules
        .query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, String>(3)?,
            ))
        })
        .map_err(|error| format!("query normalized modules: {error}"))?;
    for row in rows {
        let (hash, air_ll, blob_b64, label) =
            row.map_err(|error| format!("read normalized module: {error}"))?;
        let lib_sha256s = module_memberships
            .query_map([&hash], |row| row.get(0))
            .map_err(|error| format!("query module memberships: {error}"))?
            .collect::<Result<Vec<String>, _>>()
            .map_err(|error| format!("read module memberships: {error}"))?;
        let row = LibraryModuleRow {
            module_sha256: hash.clone(),
            air_ll,
            blob_b64,
            lib_sha256s,
            label,
        };
        row.validate()?;
        write_json_line(
            &mut module_writers[source::shard_index_for_hash(&hash)?],
            &row,
        )?;
        stats.module_rows_after += 1;
    }
    finish_writers(&mut module_writers)?;
    Ok(files)
}

fn write_aligned_rows<T: serde::Serialize>(
    prepared: &Path,
    relative_dir: &str,
    rows: Vec<T>,
    hash: impl Fn(&T) -> &String,
) -> Result<Vec<(PathBuf, PathBuf)>, String> {
    let mut buckets = (0..SHARD_COUNT)
        .map(|_| Vec::new())
        .collect::<Vec<Vec<String>>>();
    for row in rows {
        let shard = source::shard_index_for_hash(hash(&row))?;
        buckets[shard].push(
            to_sorted_json_string(&row)
                .map_err(|error| format!("serialize aligned row: {error}"))?,
        );
    }
    let mut files = Vec::new();
    let mut writers = prepared_shard_writers(prepared, relative_dir, &mut files)?;
    for (writer, lines) in writers.iter_mut().zip(&mut buckets) {
        lines.sort();
        for line in lines {
            writeln!(writer, "{line}")
                .map_err(|error| format!("write prepared {relative_dir}: {error}"))?;
        }
    }
    finish_writers(&mut writers)?;
    Ok(files)
}

fn prepared_shard_writers(
    prepared: &Path,
    relative_dir: &str,
    files: &mut Vec<(PathBuf, PathBuf)>,
) -> Result<Vec<BufWriter<File>>, String> {
    let directory = prepared.join("files").join(relative_dir);
    fs::create_dir_all(&directory)
        .map_err(|error| format!("create {}: {error}", directory.display()))?;
    (0..SHARD_COUNT)
        .map(|shard| {
            let name = source::shard_name(shard);
            let path = directory.join(&name);
            let file = File::create(&path)
                .map_err(|error| format!("create {}: {error}", path.display()))?;
            files.push((PathBuf::from(relative_dir).join(name), path));
            Ok(BufWriter::new(file))
        })
        .collect()
}

fn write_json_line(
    writer: &mut BufWriter<File>,
    row: &impl serde::Serialize,
) -> Result<(), String> {
    let line = to_sorted_json_string(row).map_err(|error| format!("serialize row: {error}"))?;
    writeln!(writer, "{line}").map_err(|error| format!("write normalized row: {error}"))
}

fn finish_writers(writers: &mut [BufWriter<File>]) -> Result<(), String> {
    for writer in writers {
        writer
            .flush()
            .map_err(|error| format!("flush normalized shard: {error}"))?;
        writer
            .get_ref()
            .sync_all()
            .map_err(|error| format!("fsync normalized shard: {error}"))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::case::LinkedFunctionResource;
    use crate::observation::{MetalObservation, MetalStatus};
    use crate::ScratchDir;

    #[test]
    fn normalization_deduplicates_air_and_invalidates_changed_evidence_atomically() {
        let scratch = ScratchDir::new("normalize-air-identities").unwrap();
        let root = scratch.path();
        let canonical_entry = crate::case::ATTACHMENTLESS_FRAGMENT_AIR;
        let entry_a = format!("; ModuleID = '/scratch/a.air'\n{canonical_entry}");
        let entry_b = format!("; ModuleID = '/scratch/b.air'\n{canonical_entry}");
        let source_a = source_row(&entry_a, "11", "YQ==");
        let source_b = source_row(&entry_b, "22", "Yg==");
        source::write_source_shards(root, [source_a.clone(), source_b.clone()]).unwrap();

        let canonical_module = "define void @linked() { ret void }\n";
        let module_a = module_row(
            &format!("; ModuleID = '/scratch/a-linked.air'\n{canonical_module}"),
            "11",
            "YQ==",
        );
        let module_b = module_row(
            &format!("; ModuleID = '/scratch/b-linked.air'\n{canonical_module}"),
            "22",
            "Yg==",
        );
        library_module::merge_library_module_shards(root, [module_a.clone(), module_b.clone()])
            .unwrap();

        let store = CorpusStore::new(root);
        let mut case_a = crate::case::attachmentless_fragment_test_case(
            source_a.air_sha256.clone(),
            "fragment_no_writes".into(),
        );
        case_a.visible_function_references = vec![LinkedFunctionResource {
            module_sha256: module_a.module_sha256.clone(),
            function: "linked".into(),
        }];
        case_a.case_id = case_a.computed_case_id().unwrap();
        let old_case_id = case_a.case_id.clone();
        store.put_case(case_a.clone()).unwrap();
        let mut case_b = case_a.clone();
        case_b.air_sha256 = source_b.air_sha256.clone();
        case_b.visible_function_references[0].module_sha256 = module_b.module_sha256.clone();
        case_b.name = "alternate-evidence-name".into();
        case_b.case_id = case_b.computed_case_id().unwrap();
        store.put_case(case_b).unwrap();
        let empty_hash = sha256_bytes(&[]);
        store
            .upsert_metal(MetalObservation {
                case_id: old_case_id,
                air_sha256: source_a.air_sha256.clone(),
                input_sha256: case_a.computed_input_sha256().unwrap(),
                metal_output_sha256: empty_hash,
                output_b64: String::new(),
                environment_id: "test-metal".into(),
                environment: serde_json::json!({}),
                oracle_abi: "test-v1".into(),
                status: MetalStatus::Qualified,
            })
            .unwrap();
        crate::index::rebuild_index(root, &crate::index::default_index_path(root)).unwrap();

        let stats = normalize_air_identities(root).unwrap();
        assert_eq!(stats.source_rows_before, 2);
        assert_eq!(stats.source_rows_after, 1);
        assert_eq!(stats.module_rows_before, 2);
        assert_eq!(stats.module_rows_after, 1);
        assert_eq!(stats.cases_rewritten, 2);
        assert_eq!(stats.cases_deduplicated, 1);
        assert_eq!(stats.observations_invalidated, 1);

        let canonical_entry =
            metal2vulkan::tools::sanitize_ll_text_with_datalayout(canonical_entry).0;
        let canonical_entry_hash = sha256_bytes(canonical_entry.as_bytes());
        let sources = source::read_source_shard(&source::source_shard_path(
            root,
            source::shard_index_for_hash(&canonical_entry_hash).unwrap(),
        ))
        .unwrap();
        let source = sources
            .iter()
            .find(|row| row.air_sha256 == canonical_entry_hash)
            .unwrap();
        assert_eq!(source.lib_sha256s, vec!["11".repeat(32), "22".repeat(32)]);
        assert!(source.blob_b64.is_some());

        let canonical_module_hash = sha256_bytes(canonical_module.as_bytes());
        let module = library_module::find_library_module(root, &canonical_module_hash)
            .unwrap()
            .unwrap();
        assert_eq!(module.lib_sha256s, vec!["11".repeat(32), "22".repeat(32)]);
        let cases = store.read_all_cases().unwrap();
        assert_eq!(cases.len(), 1);
        assert_eq!(cases[0].name, "alternate-evidence-name");
        assert_eq!(cases[0].air_sha256, canonical_entry_hash);
        assert_eq!(
            cases[0].visible_function_references[0].module_sha256,
            canonical_module_hash
        );
        assert!(store.read_metal().unwrap().is_empty());
        crate::index::check_index(root, &crate::index::default_index_path(root)).unwrap();

        let second = normalize_air_identities(root).unwrap();
        assert_eq!(second.source_rows_before, 1);
        assert_eq!(second.source_rows_after, 1);
        assert_eq!(second.module_rows_before, 1);
        assert_eq!(second.module_rows_after, 1);
        assert_eq!(second.cases_rewritten, 0);
        assert_eq!(second.cases_deduplicated, 0);
        assert_eq!(second.observations_invalidated, 0);
    }

    fn source_row(air_ll: &str, library_byte: &str, blob: &str) -> SourceRow {
        SourceRow {
            air_sha256: sha256_bytes(air_ll.as_bytes()),
            stage: "Fragment".into(),
            entry: "fragment_no_writes".into(),
            air_ll: air_ll.into(),
            blob_b64: Some(blob.into()),
            lib_sha256s: vec![library_byte.repeat(32)],
            label: format!("local/{library_byte}.ll"),
        }
    }

    fn module_row(air_ll: &str, library_byte: &str, blob: &str) -> LibraryModuleRow {
        LibraryModuleRow {
            module_sha256: sha256_bytes(air_ll.as_bytes()),
            air_ll: air_ll.into(),
            blob_b64: blob.into(),
            lib_sha256s: vec![library_byte.repeat(32)],
            label: format!("local/library-module/{library_byte}.ll"),
        }
    }
}
