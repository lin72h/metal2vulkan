//! Private AIR modules that are not shader entry points.
//!
//! Visible/intersection function implementations are commonly packaged as separate AIR blobs with
//! no shader-stage entry metadata. They are dependencies of authored entry cases, not independently
//! executable queue rows, so harvest retains them separately instead of dropping them.

use crate::case::{
    AuthoredCase, FunctionTableEntry, FunctionTableResource, IntersectionFunctionSignature,
    IntersectionFunctionTableEntry, LinkedFunctionResource,
};
use crate::hash::sha256_bytes;
use crate::jsonl::to_sorted_json_string;
use crate::source::{shard_index_for_hash, shard_name};
use base64::Engine as _;
use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LibraryModuleRow {
    pub module_sha256: String,
    pub air_ll: String,
    pub blob_b64: String,
    pub lib_sha256s: Vec<String>,
    pub label: String,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LibraryModuleMergeStats {
    pub affected_shards: usize,
    pub inserted: usize,
    pub merged_memberships: usize,
    pub duplicates: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LibraryModuleIndexSyncStats {
    pub shards_scanned: usize,
    pub bytes_scanned: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct LibraryModuleShardStamp {
    size: u64,
    modified_ns: u64,
}

/// Fully checked function-table dependencies for one authored entry point.
///
/// Case checking resolves these once by content hash and verifies that every implementation
/// defines the exact authored symbol. Executors consume this object directly, so they cannot
/// accidentally use a different module lookup or silently omit an authored table entry.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ResolvedLinkedFunctions {
    pub references: Vec<ResolvedFunctionReference>,
    pub visible: Vec<ResolvedFunctionTable>,
    pub intersection: Vec<ResolvedIntersectionFunctionTable>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResolvedFunctionReference {
    pub function: String,
    pub module: LibraryModuleRow,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResolvedFunctionTable {
    pub binding: u32,
    pub size: u32,
    pub entries: Vec<ResolvedFunctionEntry>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResolvedFunctionEntry {
    pub index: u32,
    pub function: String,
    pub module: LibraryModuleRow,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResolvedIntersectionFunctionTable {
    pub location: ResolvedIntersectionFunctionTableLocation,
    pub size: u32,
    pub entries: Vec<ResolvedIntersectionFunctionEntry>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResolvedIntersectionFunctionTableLocation {
    Direct {
        binding: u32,
    },
    ArgumentBuffer {
        buffer_binding: u32,
        field_offset: u32,
    },
}

impl ResolvedIntersectionFunctionTableLocation {
    pub fn buffer_binding(self) -> u32 {
        match self {
            Self::Direct { binding } => binding,
            Self::ArgumentBuffer { buffer_binding, .. } => buffer_binding,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ResolvedIntersectionFunctionEntry {
    Linked(ResolvedFunctionEntry),
    OpaqueTriangle {
        index: u32,
        signature: Vec<IntersectionFunctionSignature>,
    },
}

impl ResolvedIntersectionFunctionEntry {
    pub fn index(&self) -> u32 {
        match self {
            Self::Linked(entry) => entry.index,
            Self::OpaqueTriangle { index, .. } => *index,
        }
    }
}

impl ResolvedLinkedFunctions {
    pub fn is_empty(&self) -> bool {
        self.references.is_empty() && self.visible.is_empty() && self.intersection.is_empty()
    }

    pub fn all_dependencies(&self) -> impl Iterator<Item = (&str, &LibraryModuleRow)> {
        self.references
            .iter()
            .map(|reference| (reference.function.as_str(), &reference.module))
            .chain(self.visible.iter().flat_map(|table| {
                table
                    .entries
                    .iter()
                    .map(|entry| (entry.function.as_str(), &entry.module))
            }))
            .chain(self.intersection.iter().flat_map(|table| {
                table.entries.iter().filter_map(|entry| match entry {
                    ResolvedIntersectionFunctionEntry::Linked(entry) => {
                        Some((entry.function.as_str(), &entry.module))
                    }
                    ResolvedIntersectionFunctionEntry::OpaqueTriangle { .. } => None,
                })
            }))
    }
}

/// Resolve every authored table entry through its one hash-derived module shard.
///
/// All errors are accumulated so an author can repair a manifest in one pass. Repeated references
/// to the same module are loaded once, while the returned tables retain the authored ordering that
/// literal validation already proved canonical.
pub fn resolve_linked_functions(
    root: &Path,
    case: &AuthoredCase,
) -> Result<ResolvedLinkedFunctions, Vec<String>> {
    let mut modules = HashMap::<String, Result<Option<LibraryModuleRow>, String>>::new();
    let mut defined_functions = HashMap::<String, Vec<String>>::new();
    let mut errors = Vec::new();
    let references = resolve_reference_kind(
        root,
        &case.visible_function_references,
        &mut modules,
        &mut defined_functions,
        &mut errors,
    );
    let visible = resolve_table_kind(
        root,
        "visible-function",
        &case.visible_function_tables,
        &mut modules,
        &mut defined_functions,
        &mut errors,
    );
    let authored_intersection = case
        .intersection_function_tables
        .iter()
        .map(|table| {
            (
                ResolvedIntersectionFunctionTableLocation::Direct {
                    binding: table.binding,
                },
                table.size,
                table.entries.as_slice(),
            )
        })
        .chain(
            case.argument_buffer_intersection_function_tables
                .iter()
                .map(|table| {
                    (
                        ResolvedIntersectionFunctionTableLocation::ArgumentBuffer {
                            buffer_binding: table.buffer_binding,
                            field_offset: table.field_offset,
                        },
                        table.size,
                        table.entries.as_slice(),
                    )
                }),
        )
        .collect::<Vec<_>>();
    let linked_intersection = authored_intersection
        .iter()
        .map(|(location, size, entries)| FunctionTableResource {
            binding: location.buffer_binding(),
            size: *size,
            entries: entries
                .iter()
                .filter_map(|entry| match entry {
                    IntersectionFunctionTableEntry::Linked {
                        index,
                        module_sha256,
                        function,
                    } => Some(FunctionTableEntry {
                        index: *index,
                        module_sha256: module_sha256.clone(),
                        function: function.clone(),
                    }),
                    IntersectionFunctionTableEntry::OpaqueTriangle { .. } => None,
                })
                .collect(),
        })
        .collect::<Vec<_>>();
    let resolved_linked = resolve_table_kind(
        root,
        "intersection-function",
        &linked_intersection,
        &mut modules,
        &mut defined_functions,
        &mut errors,
    );
    let intersection = authored_intersection
        .iter()
        .zip(resolved_linked)
        .map(|((location, size, entries), resolved)| {
            let mut linked = resolved
                .entries
                .into_iter()
                .map(|entry| (entry.index, entry))
                .collect::<HashMap<_, _>>();
            ResolvedIntersectionFunctionTable {
                location: *location,
                size: *size,
                entries: entries
                    .iter()
                    .filter_map(|entry| match entry {
                        IntersectionFunctionTableEntry::Linked { index, .. } => linked
                            .remove(index)
                            .map(ResolvedIntersectionFunctionEntry::Linked),
                        IntersectionFunctionTableEntry::OpaqueTriangle { index, signature } => {
                            Some(ResolvedIntersectionFunctionEntry::OpaqueTriangle {
                                index: *index,
                                signature: signature.clone(),
                            })
                        }
                    })
                    .collect(),
            }
        })
        .collect::<Vec<_>>();
    let mut linked_names = HashMap::<&str, &str>::new();
    for (function, module_sha256) in references
        .iter()
        .map(|reference| {
            (
                reference.function.as_str(),
                reference.module.module_sha256.as_str(),
            )
        })
        .chain(
            visible
                .iter()
                .flat_map(|table| &table.entries)
                .chain(intersection.iter().flat_map(|table| {
                    table.entries.iter().filter_map(|entry| match entry {
                        ResolvedIntersectionFunctionEntry::Linked(entry) => Some(entry),
                        ResolvedIntersectionFunctionEntry::OpaqueTriangle { .. } => None,
                    })
                }))
                .map(|entry| (entry.function.as_str(), entry.module.module_sha256.as_str())),
        )
    {
        if let Some(previous_module) = linked_names.insert(function, module_sha256) {
            if previous_module != module_sha256 {
                errors.push(format!(
                    "linked function {:?} is defined by both module {} and module {}; Metal requires linked function names to be unique",
                    function, previous_module, module_sha256
                ));
            }
        }
    }
    if errors.is_empty() {
        Ok(ResolvedLinkedFunctions {
            references,
            visible,
            intersection,
        })
    } else {
        Err(errors)
    }
}

fn resolve_reference_kind(
    root: &Path,
    references: &[LinkedFunctionResource],
    modules: &mut HashMap<String, Result<Option<LibraryModuleRow>, String>>,
    defined_functions: &mut HashMap<String, Vec<String>>,
    errors: &mut Vec<String>,
) -> Vec<ResolvedFunctionReference> {
    references
        .iter()
        .filter_map(|reference| {
            let module = modules
                .entry(reference.module_sha256.clone())
                .or_insert_with(|| find_library_module(root, &reference.module_sha256));
            let module = match module {
                Ok(Some(module)) => module,
                Ok(None) => {
                    errors.push(format!(
                        "visible function reference {:?} module {} is not harvested",
                        reference.function, reference.module_sha256
                    ));
                    return None;
                }
                Err(error) => {
                    errors.push(error.clone());
                    return None;
                }
            };
            let names = defined_functions
                .entry(module.module_sha256.clone())
                .or_insert_with(|| defined_function_names(&module.air_ll));
            if !names.contains(&reference.function) {
                errors.push(format!(
                    "visible function reference module {} does not define function {:?}",
                    reference.module_sha256, reference.function
                ));
                return None;
            }
            Some(ResolvedFunctionReference {
                function: reference.function.clone(),
                module: module.clone(),
            })
        })
        .collect()
}

fn resolve_table_kind(
    root: &Path,
    kind: &str,
    tables: &[FunctionTableResource],
    modules: &mut HashMap<String, Result<Option<LibraryModuleRow>, String>>,
    defined_functions: &mut HashMap<String, Vec<String>>,
    errors: &mut Vec<String>,
) -> Vec<ResolvedFunctionTable> {
    tables
        .iter()
        .map(|table| {
            let entries = table
                .entries
                .iter()
                .filter_map(|entry| {
                    let module = modules
                        .entry(entry.module_sha256.clone())
                        .or_insert_with(|| find_library_module(root, &entry.module_sha256));
                    let module = match module {
                        Ok(Some(module)) => module,
                        Ok(None) => {
                            errors.push(format!(
                                "{kind} table binding {} entry {} module {} is not harvested",
                                table.binding, entry.index, entry.module_sha256
                            ));
                            return None;
                        }
                        Err(error) => {
                            errors.push(error.clone());
                            return None;
                        }
                    };
                    let names = defined_functions
                        .entry(module.module_sha256.clone())
                        .or_insert_with(|| defined_function_names(&module.air_ll));
                    if !names.contains(&entry.function) {
                        errors.push(format!(
                            "{kind} table binding {} entry {} module {} does not define function {:?}",
                            table.binding, entry.index, entry.module_sha256, entry.function
                        ));
                        return None;
                    }
                    Some(ResolvedFunctionEntry {
                        index: entry.index,
                        function: entry.function.clone(),
                        module: module.clone(),
                    })
                })
                .collect();
            ResolvedFunctionTable {
                binding: table.binding,
                size: table.size,
                entries,
            }
        })
        .collect()
}

impl LibraryModuleRow {
    pub fn validate(&self) -> Result<(), String> {
        let computed = sha256_bytes(self.air_ll.as_bytes());
        if self.module_sha256 != computed {
            return Err(format!(
                "library module {} hash mismatch: row={} computed={computed}",
                self.label, self.module_sha256
            ));
        }
        if !self
            .air_ll
            .lines()
            .any(|line| line.trim_start().starts_with("define "))
        {
            return Err(format!(
                "library module {} contains no function definition",
                self.label
            ));
        }
        let blob = base64::engine::general_purpose::STANDARD
            .decode(&self.blob_b64)
            .map_err(|error| format!("library module {} has invalid blob: {error}", self.label))?;
        if blob.is_empty() {
            return Err(format!("library module {} has an empty blob", self.label));
        }
        if self.lib_sha256s.is_empty() {
            return Err(format!(
                "library module {} has no parent library",
                self.label
            ));
        }
        let mut previous = None;
        for hash in &self.lib_sha256s {
            validate_hash("parent library", hash)?;
            if previous.is_some_and(|value: &String| value >= hash) {
                return Err(format!(
                    "library module {} parent libraries must be sorted and unique",
                    self.label
                ));
            }
            previous = Some(hash);
        }
        if self.label.trim().is_empty() {
            return Err("library module label must not be empty".into());
        }
        Ok(())
    }
}

pub fn library_modules_dir(root: &Path) -> PathBuf {
    root.join("local/library-modules")
}

pub fn library_module_shard_path(root: &Path, shard: usize) -> PathBuf {
    library_modules_dir(root).join(shard_name(shard))
}

pub fn read_library_module_shard(path: &Path) -> Result<Vec<LibraryModuleRow>, String> {
    let mut rows = Vec::new();
    for_each_library_module_shard(path, |row| {
        rows.push(row);
        Ok(())
    })?;
    Ok(rows)
}

/// Stream fully validated dependency modules while retaining at most one decoded row at a time.
pub fn for_each_library_module_shard(
    path: &Path,
    mut consume: impl FnMut(LibraryModuleRow) -> Result<(), String>,
) -> Result<(), String> {
    let expected = crate::source::shard_index_from_path(path)?;
    let file = File::open(path).map_err(|error| format!("open {}: {error}", path.display()))?;
    for (line_index, line) in BufReader::new(file).lines().enumerate() {
        let line =
            line.map_err(|error| format!("read {}:{}: {error}", path.display(), line_index + 1))?;
        if line.trim().is_empty() {
            continue;
        }
        let row: LibraryModuleRow = serde_json::from_str(&line)
            .map_err(|error| format!("parse {}:{}: {error}", path.display(), line_index + 1))?;
        row.validate()
            .map_err(|error| format!("{}:{}: {error}", path.display(), line_index + 1))?;
        let actual = shard_index_for_hash(&row.module_sha256)?;
        if actual != expected {
            return Err(format!(
                "{}:{}: library module {} belongs in shard {}, not {}",
                path.display(),
                line_index + 1,
                row.module_sha256,
                actual,
                expected
            ));
        }
        consume(row)?;
    }
    Ok(())
}

/// Resolve one explicitly authored dependency by its content hash.
///
/// The hash determines the only shard that may contain the module; no corpus-wide search is ever
/// needed for case checking or execution.
pub fn find_library_module(
    root: &Path,
    module_sha256: &str,
) -> Result<Option<LibraryModuleRow>, String> {
    let shard = shard_index_for_hash(module_sha256)?;
    let path = library_module_shard_path(root, shard);
    if !path.is_file() {
        return Ok(None);
    }
    Ok(read_library_module_shard(&path)?
        .into_iter()
        .find(|row| row.module_sha256 == module_sha256))
}

/// Incrementally index retained dependency-module identities, library memberships, symbols, and
/// exact JSONL byte locations.
///
/// A missing index is an explicit one-time migration. After it is warm, unchanged module shards
/// are identified from their stamps and are not opened.
pub fn sync_library_module_index(
    root: &Path,
    index: &Path,
) -> Result<LibraryModuleIndexSyncStats, String> {
    let mut connection = Connection::open(index)
        .map_err(|error| format!("open index {}: {error}", index.display()))?;
    connection
        .execute_batch(
            "PRAGMA foreign_keys=ON;
             CREATE TABLE IF NOT EXISTS library_modules (
               module_sha256 TEXT PRIMARY KEY,
               shard INTEGER NOT NULL,
               source_offset INTEGER NOT NULL,
               source_length INTEGER NOT NULL
             );
             CREATE TABLE IF NOT EXISTS library_module_memberships (
               module_sha256 TEXT NOT NULL REFERENCES library_modules(module_sha256) ON DELETE CASCADE,
               lib_sha256 TEXT NOT NULL,
               PRIMARY KEY(module_sha256, lib_sha256)
             );
             CREATE TABLE IF NOT EXISTS library_module_symbols (
               module_sha256 TEXT NOT NULL REFERENCES library_modules(module_sha256) ON DELETE CASCADE,
               symbol TEXT NOT NULL,
               PRIMARY KEY(module_sha256, symbol)
             );
             CREATE TABLE IF NOT EXISTS indexed_library_module_shards (
               shard INTEGER PRIMARY KEY,
               size INTEGER NOT NULL,
               modified_ns INTEGER NOT NULL
             );
             CREATE INDEX IF NOT EXISTS library_memberships_by_library
               ON library_module_memberships(lib_sha256, module_sha256);
             CREATE INDEX IF NOT EXISTS library_symbols_by_symbol
               ON library_module_symbols(symbol, module_sha256);",
        )
        .map_err(|error| format!("upgrade library-module index schema: {error}"))?;

    let paths = library_module_shard_paths(root)?;
    let current_shards = paths
        .iter()
        .map(|path| crate::source::shard_index_from_path(path))
        .collect::<Result<std::collections::HashSet<_>, _>>()?;
    let transaction = connection
        .transaction()
        .map_err(|error| format!("begin library-module index sync: {error}"))?;
    let tracked_shards = {
        let mut statement = transaction
            .prepare("SELECT shard FROM indexed_library_module_shards")
            .map_err(|error| format!("prepare tracked library-module shards: {error}"))?;
        let shards = statement
            .query_map([], |row| row.get::<_, usize>(0))
            .map_err(|error| format!("query tracked library-module shards: {error}"))?
            .collect::<Result<std::collections::HashSet<_>, _>>()
            .map_err(|error| format!("read tracked library-module shards: {error}"))?;
        shards
    };
    for shard in tracked_shards.difference(&current_shards) {
        transaction
            .execute("DELETE FROM library_modules WHERE shard=?1", [shard])
            .map_err(|error| format!("remove library-module shard {shard}: {error}"))?;
        transaction
            .execute(
                "DELETE FROM indexed_library_module_shards WHERE shard=?1",
                [shard],
            )
            .map_err(|error| format!("remove library-module shard stamp {shard}: {error}"))?;
    }

    let mut stats = LibraryModuleIndexSyncStats::default();
    for path in paths {
        let shard = crate::source::shard_index_from_path(&path)?;
        let stamp = library_module_shard_stamp(&path)?;
        let tracked = transaction
            .query_row(
                "SELECT size, modified_ns FROM indexed_library_module_shards WHERE shard=?1",
                [shard],
                |row| {
                    Ok(LibraryModuleShardStamp {
                        size: row.get(0)?,
                        modified_ns: row.get(1)?,
                    })
                },
            )
            .optional()
            .map_err(|error| format!("read library-module shard {shard} stamp: {error}"))?;
        if tracked == Some(stamp) {
            continue;
        }
        transaction
            .execute("DELETE FROM library_modules WHERE shard=?1", [shard])
            .map_err(|error| format!("replace library-module shard {shard}: {error}"))?;
        index_library_module_shard(&transaction, &path, shard)?;
        transaction
            .execute(
                "INSERT OR REPLACE INTO indexed_library_module_shards (shard, size, modified_ns)
                 VALUES (?1, ?2, ?3)",
                params![shard, stamp.size, stamp.modified_ns],
            )
            .map_err(|error| format!("record library-module shard {shard}: {error}"))?;
        stats.shards_scanned += 1;
        stats.bytes_scanned = stats
            .bytes_scanned
            .checked_add(stamp.size)
            .ok_or_else(|| "library-module shard byte count overflow".to_string())?;
    }
    transaction
        .commit()
        .map_err(|error| format!("commit library-module index sync: {error}"))?;
    Ok(stats)
}

/// Resolve direct AIR visible-function references only when the retained module is unique across
/// every parent library of the entry AIR, including transitive dependency references.
pub fn resolve_indexed_visible_references(
    root: &Path,
    index: &Path,
    entry_ll: &str,
    parent_library_sha256s: &[String],
) -> Result<Option<Vec<ResolvedFunctionReference>>, String> {
    let mut pending = metal2vulkan::linked_functions::visible_function_reference_symbols(entry_ll)?;
    if pending.is_empty() {
        return Ok(Some(Vec::new()));
    }
    let connection = Connection::open_with_flags(
        index,
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )
    .map_err(|error| format!("open index {}: {error}", index.display()))?;
    let mut resolved = BTreeMap::<String, ResolvedFunctionReference>::new();
    while let Some(symbol) = pending.pop() {
        if resolved.contains_key(&symbol) {
            continue;
        }
        let hashes = indexed_modules_for_symbol(&connection, parent_library_sha256s, &symbol)?;
        if hashes.len() != 1 {
            return Ok(None);
        }
        let module = find_indexed_library_module(root, &connection, &hashes[0])?
            .ok_or_else(|| format!("indexed library module {} is unavailable", hashes[0]))?;
        pending.extend(
            metal2vulkan::linked_functions::visible_function_reference_symbols(&module.air_ll)?,
        );
        resolved.insert(
            symbol.clone(),
            ResolvedFunctionReference {
                function: symbol,
                module,
            },
        );
    }
    Ok(Some(resolved.into_values().collect()))
}

fn indexed_modules_for_symbol(
    connection: &Connection,
    libraries: &[String],
    symbol: &str,
) -> Result<Vec<String>, String> {
    let mut statement = connection
        .prepare(
            "SELECT m.module_sha256
             FROM library_module_memberships m
             JOIN library_module_symbols s ON s.module_sha256=m.module_sha256
             WHERE m.lib_sha256=?1 AND s.symbol=?2
             ORDER BY m.module_sha256",
        )
        .map_err(|error| format!("prepare linked-symbol lookup: {error}"))?;
    let mut hashes = BTreeSet::new();
    for library in libraries {
        let rows = statement
            .query_map(params![library, symbol], |row| row.get(0))
            .map_err(|error| format!("query linked symbol {symbol:?}: {error}"))?;
        for hash in rows {
            hashes.insert(hash.map_err(|error| format!("read linked symbol {symbol:?}: {error}"))?);
        }
    }
    Ok(hashes.into_iter().collect())
}

fn find_indexed_library_module(
    root: &Path,
    connection: &Connection,
    hash: &str,
) -> Result<Option<LibraryModuleRow>, String> {
    let location = connection
        .query_row(
            "SELECT shard, source_offset, source_length FROM library_modules WHERE module_sha256=?1",
            [hash],
            |row| Ok((row.get::<_, usize>(0)?, row.get::<_, u64>(1)?, row.get::<_, usize>(2)?)),
        )
        .optional()
        .map_err(|error| format!("query library module {hash}: {error}"))?;
    let Some((shard, offset, length)) = location else {
        return Ok(None);
    };
    let path = library_module_shard_path(root, shard);
    let mut file =
        File::open(&path).map_err(|error| format!("open {}: {error}", path.display()))?;
    file.seek(SeekFrom::Start(offset))
        .map_err(|error| format!("seek {}: {error}", path.display()))?;
    let mut bytes = vec![0; length];
    file.read_exact(&mut bytes)
        .map_err(|error| format!("read {}: {error}", path.display()))?;
    let row: LibraryModuleRow = serde_json::from_slice(&bytes)
        .map_err(|error| format!("parse indexed library module {hash}: {error}"))?;
    if row.module_sha256 != hash {
        return Err(format!(
            "indexed library-module location for {hash} contains {}",
            row.module_sha256
        ));
    }
    Ok(Some(row))
}

fn library_module_shard_paths(root: &Path) -> Result<Vec<PathBuf>, String> {
    let directory = library_modules_dir(root);
    if !directory.is_dir() {
        return Ok(Vec::new());
    }
    let mut paths = fs::read_dir(&directory)
        .map_err(|error| format!("read {}: {error}", directory.display()))?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.extension()
                .is_some_and(|extension| extension == "jsonl")
        })
        .collect::<Vec<_>>();
    paths.sort();
    Ok(paths)
}

fn library_module_shard_stamp(path: &Path) -> Result<LibraryModuleShardStamp, String> {
    let metadata =
        fs::metadata(path).map_err(|error| format!("read metadata {}: {error}", path.display()))?;
    let modified_ns = metadata
        .modified()
        .map_err(|error| format!("read modification time {}: {error}", path.display()))?
        .duration_since(UNIX_EPOCH)
        .map_err(|error| format!("invalid modification time {}: {error}", path.display()))?
        .as_nanos()
        .try_into()
        .map_err(|_| format!("modification time overflow for {}", path.display()))?;
    Ok(LibraryModuleShardStamp {
        size: metadata.len(),
        modified_ns,
    })
}

fn index_library_module_shard(
    transaction: &rusqlite::Transaction<'_>,
    path: &Path,
    shard: usize,
) -> Result<(), String> {
    let mut reader = BufReader::new(
        File::open(path).map_err(|error| format!("open {}: {error}", path.display()))?,
    );
    let mut offset = 0u64;
    let mut line = Vec::new();
    loop {
        line.clear();
        let length = reader
            .read_until(b'\n', &mut line)
            .map_err(|error| format!("read {}: {error}", path.display()))?;
        if length == 0 {
            break;
        }
        let row_bytes = line.strip_suffix(b"\n").unwrap_or(&line);
        if row_bytes.is_empty() {
            offset += length as u64;
            continue;
        }
        let row: LibraryModuleRow = serde_json::from_slice(row_bytes)
            .map_err(|error| format!("parse {} at byte {offset}: {error}", path.display()))?;
        row.validate()
            .map_err(|error| format!("{} at byte {offset}: {error}", path.display()))?;
        let actual = shard_index_for_hash(&row.module_sha256)?;
        if actual != shard {
            return Err(format!(
                "library module {} belongs in shard {actual}, not {shard}",
                row.module_sha256
            ));
        }
        transaction
            .execute(
                "INSERT INTO library_modules (module_sha256, shard, source_offset, source_length)
                 VALUES (?1, ?2, ?3, ?4)",
                params![row.module_sha256, shard, offset, row_bytes.len()],
            )
            .map_err(|error| format!("index library module {}: {error}", row.module_sha256))?;
        for library in &row.lib_sha256s {
            transaction
                .execute(
                    "INSERT INTO library_module_memberships (module_sha256, lib_sha256)
                     VALUES (?1, ?2)",
                    params![row.module_sha256, library],
                )
                .map_err(|error| format!("index library membership: {error}"))?;
        }
        for symbol in defined_function_names(&row.air_ll) {
            transaction
                .execute(
                    "INSERT INTO library_module_symbols (module_sha256, symbol) VALUES (?1, ?2)",
                    params![row.module_sha256, symbol],
                )
                .map_err(|error| format!("index library symbol: {error}"))?;
        }
        offset += length as u64;
    }
    Ok(())
}

pub fn defined_function_names(ll: &str) -> Vec<String> {
    ll.lines()
        .filter_map(|line| {
            let line = line.trim_start();
            if !line.starts_with("define ") {
                return None;
            }
            let at = line.find('@')?;
            llvm_symbol(&line[at + 1..])
        })
        .collect()
}

fn llvm_symbol(text: &str) -> Option<String> {
    if let Some(quoted) = text.strip_prefix('"') {
        let mut escaped = false;
        let end = quoted.char_indices().find_map(|(index, ch)| {
            if escaped {
                escaped = false;
                None
            } else if ch == '\\' {
                escaped = true;
                None
            } else if ch == '"' {
                Some(index)
            } else {
                None
            }
        })?;
        Some(quoted[..end].to_string())
    } else {
        let end = text.find('(')?;
        let name = text[..end].trim();
        (!name.is_empty()).then(|| name.to_string())
    }
}

pub fn merge_library_module_shards(
    root: &Path,
    rows: impl IntoIterator<Item = LibraryModuleRow>,
) -> Result<LibraryModuleMergeStats, String> {
    let mut additions = BTreeMap::<usize, Vec<LibraryModuleRow>>::new();
    for row in rows {
        row.validate()?;
        additions
            .entry(shard_index_for_hash(&row.module_sha256)?)
            .or_default()
            .push(row);
    }
    if additions.is_empty() {
        return Ok(LibraryModuleMergeStats::default());
    }
    fs::create_dir_all(library_modules_dir(root))
        .map_err(|error| format!("create library-module shards: {error}"))?;
    remove_stale_temporaries(root)?;
    let mut stats = LibraryModuleMergeStats {
        affected_shards: additions.len(),
        ..LibraryModuleMergeStats::default()
    };
    for (shard, additions) in additions {
        let path = library_module_shard_path(root, shard);
        let existing = if path.is_file() {
            read_library_module_shard(&path)?
        } else {
            Vec::new()
        };
        let mut merged = existing
            .into_iter()
            .map(|row| (row.module_sha256.clone(), row))
            .collect::<BTreeMap<_, _>>();
        for addition in additions {
            match merged.get_mut(&addition.module_sha256) {
                None => {
                    stats.inserted += 1;
                    merged.insert(addition.module_sha256.clone(), addition);
                }
                Some(existing) => {
                    let before = existing.lib_sha256s.len();
                    existing.lib_sha256s.extend(addition.lib_sha256s);
                    existing.lib_sha256s.sort();
                    existing.lib_sha256s.dedup();
                    stats.merged_memberships += existing.lib_sha256s.len() - before;
                    if addition.blob_b64 < existing.blob_b64 {
                        existing.blob_b64 = addition.blob_b64;
                    }
                    if existing.lib_sha256s.len() == before {
                        stats.duplicates += 1;
                    }
                }
            }
        }
        write_library_module_bucket(root, shard, merged.into_values().collect())?;
    }
    Ok(stats)
}

fn write_library_module_bucket(
    root: &Path,
    shard: usize,
    mut rows: Vec<LibraryModuleRow>,
) -> Result<(), String> {
    rows.sort_by(|left, right| left.module_sha256.cmp(&right.module_sha256));
    let path = library_module_shard_path(root, shard);
    let temporary = library_modules_dir(root).join(format!(
        ".{}.{}.tmp",
        shard_name(shard),
        std::process::id()
    ));
    let result = (|| {
        let mut file = File::create(&temporary)
            .map_err(|error| format!("create {}: {error}", temporary.display()))?;
        for row in rows {
            let line = to_sorted_json_string(&row)
                .map_err(|error| format!("serialize {}: {error}", row.module_sha256))?;
            writeln!(file, "{line}")
                .map_err(|error| format!("write {}: {error}", temporary.display()))?;
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
        File::open(library_modules_dir(root))
            .and_then(|directory| directory.sync_all())
            .map_err(|error| {
                format!(
                    "fsync directory {}: {error}",
                    library_modules_dir(root).display()
                )
            })
    })();
    if let Err(error) = result {
        let _ = fs::remove_file(&temporary);
        return Err(error);
    }
    Ok(())
}

fn remove_stale_temporaries(root: &Path) -> Result<(), String> {
    let directory = library_modules_dir(root);
    for path in fs::read_dir(&directory)
        .map_err(|error| format!("read {}: {error}", directory.display()))?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
    {
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

fn validate_hash(label: &str, hash: &str) -> Result<(), String> {
    if hash.len() != 64
        || !hash.bytes().all(|byte| byte.is_ascii_hexdigit())
        || hash.bytes().any(|byte| byte.is_ascii_uppercase())
    {
        Err(format!("{label} must be a lowercase SHA-256"))
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ScratchDir;

    fn row(ll: &str, library: &str) -> LibraryModuleRow {
        LibraryModuleRow {
            module_sha256: sha256_bytes(ll.as_bytes()),
            air_ll: ll.into(),
            blob_b64: base64::engine::general_purpose::STANDARD.encode(b"owned bitcode"),
            lib_sha256s: vec![library.into()],
            label: "local/library-module.ll".into(),
        }
    }

    #[test]
    fn merge_is_shard_local_and_preserves_all_library_memberships() {
        let scratch = ScratchDir::new("library-module-merge").unwrap();
        let first = row("define void @visible() { ret void }", &"11".repeat(32));
        let first_shard = shard_index_for_hash(&first.module_sha256).unwrap();
        let mut unrelated = first.clone();
        for nonce in 0..u32::MAX {
            unrelated = row(
                &format!("define void @other() {{ ret void }}\n; {nonce}"),
                &"22".repeat(32),
            );
            if shard_index_for_hash(&unrelated.module_sha256).unwrap() != first_shard {
                break;
            }
        }
        merge_library_module_shards(scratch.path(), [first.clone(), unrelated.clone()]).unwrap();
        let unrelated_path = library_module_shard_path(
            scratch.path(),
            shard_index_for_hash(&unrelated.module_sha256).unwrap(),
        );
        let unrelated_bytes = fs::read(&unrelated_path).unwrap();
        fs::write(&unrelated_path, vec![b'x'; unrelated_bytes.len()]).unwrap();

        let same_module_other_library = row(&first.air_ll, &"33".repeat(32));
        let stats =
            merge_library_module_shards(scratch.path(), [same_module_other_library]).unwrap();
        assert_eq!(stats.affected_shards, 1);
        assert_eq!(stats.merged_memberships, 1);
        assert_eq!(
            fs::read(&unrelated_path).unwrap(),
            vec![b'x'; unrelated_bytes.len()]
        );
        assert_eq!(
            read_library_module_shard(&library_module_shard_path(scratch.path(), first_shard))
                .unwrap()[0]
                .lib_sha256s,
            vec!["11".repeat(32), "33".repeat(32)]
        );
    }

    #[test]
    fn indexed_visible_references_resolve_across_all_entry_library_memberships() {
        let scratch = ScratchDir::new("indexed-visible-reference").unwrap();
        let entry_library = "11".repeat(32);
        let dependency_library = "22".repeat(32);
        let dependency = row("define void @linked() { ret void }\n", &dependency_library);
        merge_library_module_shards(scratch.path(), [dependency.clone()]).unwrap();
        let index = scratch.path().join("index.sqlite");
        Connection::open(&index).unwrap();

        let cold = sync_library_module_index(scratch.path(), &index).unwrap();
        assert_eq!(cold.shards_scanned, 1);
        let warm = sync_library_module_index(scratch.path(), &index).unwrap();
        assert_eq!(warm, LibraryModuleIndexSyncStats::default());

        let entry = "define void @main() { ret void }\n!air.visible_function_references = !{!0}\n!0 = !{!\"air.visible_function_reference\", ptr @linked.MTL_VISIBLE_FN_REF, !\"linked\"}\n";
        let resolved = resolve_indexed_visible_references(
            scratch.path(),
            &index,
            entry,
            &[entry_library, dependency_library],
        )
        .unwrap()
        .unwrap();
        assert_eq!(resolved.len(), 1);
        assert_eq!(resolved[0].function, "linked");
        assert_eq!(resolved[0].module.module_sha256, dependency.module_sha256);
    }

    #[test]
    fn indexed_visible_references_leave_ambiguous_definitions_for_authoring() {
        let scratch = ScratchDir::new("ambiguous-visible-reference").unwrap();
        let library = "11".repeat(32);
        let first = row("define void @linked() { ret void }\n; first\n", &library);
        let second = row("define void @linked() { ret void }\n; second\n", &library);
        merge_library_module_shards(scratch.path(), [first, second]).unwrap();
        let index = scratch.path().join("index.sqlite");
        Connection::open(&index).unwrap();
        sync_library_module_index(scratch.path(), &index).unwrap();

        let entry = "define void @main() { ret void }\n!air.visible_function_references = !{!0}\n!0 = !{!\"air.visible_function_reference\", ptr @linked.MTL_VISIBLE_FN_REF, !\"linked\"}\n";
        assert_eq!(
            resolve_indexed_visible_references(
                scratch.path(),
                &index,
                entry,
                std::slice::from_ref(&library),
            )
            .unwrap(),
            None
        );
    }

    #[test]
    fn module_requires_real_function_body_and_exact_identity() {
        let mut module = row("declare void @visible()", &"11".repeat(32));
        assert!(module
            .validate()
            .unwrap_err()
            .contains("no function definition"));
        module.air_ll = "define void @visible() { ret void }".into();
        assert!(module.validate().unwrap_err().contains("hash mismatch"));
    }

    #[test]
    fn function_names_cover_plain_and_quoted_llvm_symbols() {
        assert_eq!(
            defined_function_names(
                "define void @plain() { ret void }\ndefine float @\"quoted name\"(float %x) { ret float %x }\ndeclare void @ignored()"
            ),
            ["plain", "quoted name"]
        );
    }

    #[test]
    fn direct_reference_resolves_an_explicit_cross_library_module() {
        let scratch = ScratchDir::new("direct-linked-function").unwrap();
        let module = row(
            "define i32 @linked(i32 %value) { ret i32 %value }",
            &"33".repeat(32),
        );
        merge_library_module_shards(scratch.path(), [module.clone()]).unwrap();
        let mut modules = HashMap::new();
        let mut names = HashMap::new();
        let mut errors = Vec::new();
        let resolved = resolve_reference_kind(
            scratch.path(),
            &[LinkedFunctionResource {
                module_sha256: module.module_sha256.clone(),
                function: "linked".into(),
            }],
            &mut modules,
            &mut names,
            &mut errors,
        );
        assert!(errors.is_empty(), "{errors:?}");
        assert_eq!(resolved.len(), 1);
        assert_eq!(resolved[0].module, module);
    }

    #[test]
    fn shard_count_matches_source_store_contract() {
        assert_eq!(crate::source::SHARD_COUNT, 64);
    }
}
