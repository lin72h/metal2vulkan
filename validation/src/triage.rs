use crate::check::has_cfg_cycle;
use crate::hash::sha256_bytes;
use crate::requirement::ToolingRequirement;
use crate::source::SourceRow;
use rusqlite::{params, Connection, OptionalExtension, Transaction};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::Path;

pub const ANALYZER_ABI: &str = "structural-triage-v38";

const RESOURCE_KINDS: [&str; 13] = [
    "air.buffer",
    "air.texture",
    "air.sampler",
    "air.imageblock",
    "air.visible_function_table",
    "air.intersection_function_table",
    "air.acceleration_structure",
    "air.indirect_command_buffer",
    "air.threadgroup_position_in_grid",
    "air.thread_position_in_grid",
    "air.thread_position_in_threadgroup",
    "air.vertex_input",
    "air.render_target",
];

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuralTriage {
    pub signature: String,
    pub has_cfg_cycle: bool,
    pub air_calls: BTreeMap<String, usize>,
    pub unrecognized_air_intrinsics: BTreeMap<String, usize>,
    pub resource_markers: BTreeMap<String, usize>,
    pub uses_device_addresses: bool,
    pub tooling_requirements: Vec<ToolingRequirement>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AuditTarget {
    VisibleFunctionTables,
    RayIntersections,
    DeviceAddressHierarchy,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct AuthoringCapabilitySummary {
    pub total_sources: usize,
    pub classified_sources: usize,
    pub unresolved_sources: usize,
    pub requirements: BTreeMap<ToolingRequirement, usize>,
    pub unrecognized_air_intrinsics: BTreeMap<String, usize>,
}

impl AuthoringCapabilitySummary {
    pub fn remaining_sources(&self) -> usize {
        self.total_sources.saturating_sub(self.classified_sources)
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct VisibleFunctionTableAudit {
    pub lookup_uses: BTreeMap<String, usize>,
    pub table_operands: BTreeMap<String, usize>,
    pub queries: BTreeMap<String, usize>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RayIntersectionAudit {
    pub calls: BTreeMap<String, usize>,
    pub signatures: BTreeMap<String, usize>,
    pub extracted_fields: BTreeMap<String, usize>,
    pub table_operands: BTreeMap<String, usize>,
    pub contract_errors: BTreeMap<String, usize>,
    pub malformed_calls: usize,
    pub product_supported: bool,
}

impl RayIntersectionAudit {
    pub fn shape(&self) -> String {
        let mut parts = Vec::new();
        for (callee, count) in &self.calls {
            parts.push(format!("call:{callee}={count}"));
        }
        for (field, count) in &self.extracted_fields {
            parts.push(format!("field:{field}={count}"));
        }
        for (kind, count) in &self.table_operands {
            parts.push(format!("table:{kind}={count}"));
        }
        for (error, count) in &self.contract_errors {
            parts.push(format!("contract:{error}={count}"));
        }
        if self.malformed_calls != 0 {
            parts.push(format!("malformed={}", self.malformed_calls));
        }
        parts.push(format!("product_supported={}", self.product_supported));
        parts.join(",")
    }
}

impl VisibleFunctionTableAudit {
    pub fn shape(&self) -> String {
        let mut parts = Vec::new();
        for (kind, count) in &self.lookup_uses {
            parts.push(format!("use:{kind}={count}"));
        }
        for (kind, count) in &self.table_operands {
            parts.push(format!("table:{kind}={count}"));
        }
        for (kind, count) in &self.queries {
            parts.push(format!("query:{kind}={count}"));
        }
        if parts.is_empty() {
            "marker_only".into()
        } else {
            parts.join(",")
        }
    }

    pub fn has_unsupported_use(&self) -> bool {
        self.lookup_uses
            .keys()
            .any(|kind| kind.contains(".unsupported_"))
            || self.table_operands.contains_key("derived")
    }

    /// Whether exact authored table contents are a semantic input to this source.
    ///
    /// A dead lookup has no observable table dependency. Queries, calls, nullness/sentinel probes,
    /// and helper-threaded callbacks do; they can be specialized exactly only after the caller
    /// supplies the table size and populated linked modules.
    pub fn requires_authored_linkage(&self) -> bool {
        !self.has_unsupported_use()
            && (!self.queries.is_empty()
                || self
                    .lookup_uses
                    .keys()
                    .any(|kind| !kind.ends_with(".unused")))
    }
}

/// Select sources from the versioned triage cache by an exact tooling requirement.
///
/// This query touches only SQLite. The caller can then use indexed byte locations to read the
/// bounded set of source rows it intends to audit.
pub fn select_cached_requirement(
    index: &Path,
    requirement: ToolingRequirement,
    limit: usize,
) -> Result<Vec<String>, String> {
    select_cached_requirement_after(index, requirement, None, limit)
}

pub fn select_cached_requirement_after(
    index: &Path,
    requirement: ToolingRequirement,
    after: Option<&str>,
    limit: usize,
) -> Result<Vec<String>, String> {
    let connection = Connection::open(index)
        .map_err(|error| format!("open index {}: {error}", index.display()))?;
    ensure_cache_table(&connection)?;
    let mut statement = connection
        .prepare(
            "SELECT s.air_sha256 FROM sources s JOIN triage_analysis t USING (air_sha256) \
             WHERE t.analyzer_abi=?1 AND (?2 IS NULL OR s.air_sha256>?2) AND EXISTS (\
               SELECT 1 FROM json_each(t.result_json, '$.tooling_requirements') r \
               WHERE r.value=?3\
             ) ORDER BY s.air_sha256 LIMIT ?4",
        )
        .map_err(|error| format!("prepare cached requirement query: {error}"))?;
    let hashes = statement
        .query_map(
            params![ANALYZER_ABI, after, requirement.as_str(), limit as i64],
            |row| row.get::<_, String>(0),
        )
        .map_err(|error| format!("query cached requirement {requirement}: {error}"))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| format!("read cached requirement {requirement}: {error}"))?;
    Ok(hashes)
}

/// Select a supported structural family for focused regression auditing.
///
/// Audit targets are deliberately independent of [`ToolingRequirement`]: gaining support must not
/// make the corresponding regression audit select zero rows.
pub fn select_cached_audit_target_after(
    index: &Path,
    target: AuditTarget,
    after: Option<&str>,
    limit: usize,
) -> Result<Vec<String>, String> {
    let connection = Connection::open(index)
        .map_err(|error| format!("open index {}: {error}", index.display()))?;
    ensure_cache_table(&connection)?;
    let predicate = match target {
        AuditTarget::VisibleFunctionTables => {
            "coalesce(json_extract(t.result_json, '$.resource_markers.\"air.visible_function_table\"'), 0) > 0"
        }
        AuditTarget::RayIntersections => {
            "EXISTS (SELECT 1 FROM json_each(t.result_json, '$.air_calls') call WHERE call.key LIKE 'air.intersect.%')"
        }
        AuditTarget::DeviceAddressHierarchy => {
            "json_extract(t.result_json, '$.uses_device_addresses') = 1"
        }
    };
    let sql = format!(
        "SELECT s.air_sha256 FROM sources s JOIN triage_analysis t USING (air_sha256) \
         WHERE t.analyzer_abi=?1 AND (?2 IS NULL OR s.air_sha256>?2) AND {predicate} \
         ORDER BY s.air_sha256 LIMIT ?3"
    );
    let mut statement = connection
        .prepare(&sql)
        .map_err(|error| format!("prepare cached {target:?} audit query: {error}"))?;
    let hashes = statement
        .query_map(params![ANALYZER_ABI, after, limit as i64], |row| {
            row.get::<_, String>(0)
        })
        .map_err(|error| format!("query cached {target:?} audit: {error}"))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| format!("read cached {target:?} audit: {error}"))?;
    Ok(hashes)
}

/// Select source identities that do not have facts under the current analyzer contract.
///
/// This is intentionally independent of case/review state: authored rows are part of the same
/// capability census as unplanned rows. The query touches only SQLite; callers read the returned
/// rows through their indexed byte locations.
pub fn select_uncached(index: &Path, limit: usize) -> Result<Vec<String>, String> {
    let connection = Connection::open(index)
        .map_err(|error| format!("open index {}: {error}", index.display()))?;
    ensure_cache_table(&connection)?;
    let mut statement = connection
        .prepare(
            "SELECT s.air_sha256 FROM sources s LEFT JOIN triage_analysis t USING (air_sha256) \
             WHERE t.air_sha256 IS NULL OR t.analyzer_abi<>?1 \
             ORDER BY s.air_sha256 LIMIT ?2",
        )
        .map_err(|error| format!("prepare uncached triage query: {error}"))?;
    let hashes = statement
        .query_map(params![ANALYZER_ABI, limit as i64], |row| {
            row.get::<_, String>(0)
        })
        .map_err(|error| format!("query uncached triage rows: {error}"))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| format!("read uncached triage rows: {error}"))?;
    Ok(hashes)
}

/// Select every indexed source identity, irrespective of its triage-cache state.
///
/// This supports deliberate full regression sweeps without deleting the disposable index or
/// weakening the normal incremental path.
pub fn select_all(index: &Path, limit: usize) -> Result<Vec<String>, String> {
    let connection = Connection::open(index)
        .map_err(|error| format!("open index {}: {error}", index.display()))?;
    let mut statement = connection
        .prepare("SELECT air_sha256 FROM sources ORDER BY air_sha256 LIMIT ?1")
        .map_err(|error| format!("prepare full triage query: {error}"))?;
    let hashes = statement
        .query_map([limit as i64], |row| row.get::<_, String>(0))
        .map_err(|error| format!("query full triage rows: {error}"))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| format!("read full triage rows: {error}"))?;
    Ok(hashes)
}

/// Summarize the exact current authoring-capability contract using only the disposable index.
pub fn authoring_capability_summary(index: &Path) -> Result<AuthoringCapabilitySummary, String> {
    let connection = Connection::open(index)
        .map_err(|error| format!("open index {}: {error}", index.display()))?;
    ensure_cache_table(&connection)?;
    let total_sources = connection
        .query_row("SELECT count(*) FROM sources", [], |row| row.get(0))
        .map_err(|error| format!("count authoring-capability sources: {error}"))?;
    let classified_sources = connection
        .query_row(
            "SELECT count(*) FROM sources s JOIN triage_analysis t USING (air_sha256) \
             WHERE t.analyzer_abi=?1",
            [ANALYZER_ABI],
            |row| row.get(0),
        )
        .map_err(|error| format!("count classified authoring-capability sources: {error}"))?;
    let unresolved_sources = connection
        .query_row(
            "SELECT count(*) FROM sources s JOIN triage_analysis t USING (air_sha256) \
             WHERE t.analyzer_abi=?1 \
               AND (json_array_length(t.result_json, '$.tooling_requirements')>0 \
                    OR EXISTS (SELECT 1 FROM json_each(\
                        t.result_json, '$.unrecognized_air_intrinsics'))) ",
            [ANALYZER_ABI],
            |row| row.get(0),
        )
        .map_err(|error| format!("count unresolved authoring-capability sources: {error}"))?;
    let mut statement = connection
        .prepare(
            "SELECT requirement.value, count(*) \
             FROM sources s JOIN triage_analysis t USING (air_sha256), \
                  json_each(t.result_json, '$.tooling_requirements') requirement \
             WHERE t.analyzer_abi=?1 \
             GROUP BY requirement.value ORDER BY requirement.value",
        )
        .map_err(|error| format!("prepare authoring-capability requirements: {error}"))?;
    let requirements = statement
        .query_map([ANALYZER_ABI], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, usize>(1)?))
        })
        .map_err(|error| format!("query authoring-capability requirements: {error}"))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| format!("read authoring-capability requirements: {error}"))?
        .into_iter()
        .map(|(requirement, count)| Ok((requirement.parse()?, count)))
        .collect::<Result<BTreeMap<_, _>, String>>()?;
    let mut statement = connection
        .prepare(
            "SELECT intrinsic.key, sum(CAST(intrinsic.value AS INTEGER)) \
             FROM sources s JOIN triage_analysis t USING (air_sha256), \
                  json_each(t.result_json, '$.unrecognized_air_intrinsics') intrinsic \
             WHERE t.analyzer_abi=?1 \
             GROUP BY intrinsic.key ORDER BY intrinsic.key",
        )
        .map_err(|error| format!("prepare unrecognized AIR intrinsic summary: {error}"))?;
    let unrecognized_air_intrinsics = statement
        .query_map([ANALYZER_ABI], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, usize>(1)?))
        })
        .map_err(|error| format!("query unrecognized AIR intrinsic summary: {error}"))?
        .collect::<Result<BTreeMap<_, _>, _>>()
        .map_err(|error| format!("read unrecognized AIR intrinsic summary: {error}"))?;
    Ok(AuthoringCapabilitySummary {
        total_sources,
        classified_sources,
        unresolved_sources,
        requirements,
        unrecognized_air_intrinsics,
    })
}

pub fn read_cached(
    index: &Path,
    hashes: &[String],
) -> Result<BTreeMap<String, StructuralTriage>, String> {
    let connection = Connection::open(index)
        .map_err(|error| format!("open index {}: {error}", index.display()))?;
    ensure_cache_table(&connection)?;
    let mut cached = BTreeMap::new();
    {
        let mut statement = connection
            .prepare(
                "SELECT air_sha256, analyzer_abi, result_json FROM triage_analysis \
                 WHERE air_sha256=?1 AND analyzer_abi=?2",
            )
            .map_err(|error| format!("prepare triage cache query: {error}"))?;
        for requested_hash in hashes {
            let row = statement
                .query_row(params![requested_hash, ANALYZER_ABI], |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, String>(2)?,
                    ))
                })
                .optional()
                .map_err(|error| format!("query triage cache for {requested_hash}: {error}"))?;
            let Some((hash, _abi, json)) = row else {
                continue;
            };
            let mut result: StructuralTriage = serde_json::from_str(&json)
                .map_err(|error| format!("parse cached triage result for {hash}: {error}"))?;
            finalize_requirements(&mut result);
            cached.insert(hash, result);
        }
    }
    Ok(cached)
}

pub fn write_cached<'a>(
    index: &Path,
    results: impl IntoIterator<Item = (&'a str, &'a StructuralTriage)>,
) -> Result<(), String> {
    let mut connection = Connection::open(index)
        .map_err(|error| format!("open index {}: {error}", index.display()))?;
    ensure_cache_table(&connection)?;
    let transaction = connection
        .transaction()
        .map_err(|error| format!("begin triage cache update: {error}"))?;
    write_cached_transaction(&transaction, results)?;
    transaction
        .commit()
        .map_err(|error| format!("commit triage cache update: {error}"))
}

/// Recompute product-owned AIR disposition from cached exact call inventories.
///
/// This is the fast path for a lowering/recognition contract change: the structural analyzer has
/// already paid to parse every source and retained `air_calls` by exact ABI symbol. Reopening AIR
/// shards would make the index pointless. Keyset batches keep memory bounded independently of
/// corpus size and avoid holding a SQLite read cursor while committing each update batch.
pub fn reclassify_cached_air_intrinsics(index: &Path) -> Result<usize, String> {
    const BATCH: usize = 1024;
    let mut after = String::new();
    let mut updated = 0usize;
    loop {
        let rows = {
            let connection = Connection::open(index)
                .map_err(|error| format!("open index {}: {error}", index.display()))?;
            ensure_cache_table(&connection)?;
            let mut statement = connection
                .prepare(
                    "SELECT air_sha256, result_json FROM triage_analysis \
                     WHERE analyzer_abi=?1 AND air_sha256>?2 \
                     ORDER BY air_sha256 LIMIT ?3",
                )
                .map_err(|error| format!("prepare cached AIR reclassification: {error}"))?;
            let batch = statement
                .query_map(params![ANALYZER_ABI, after, BATCH as i64], |row| {
                    Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
                })
                .map_err(|error| format!("query cached AIR reclassification: {error}"))?
                .collect::<Result<Vec<_>, _>>()
                .map_err(|error| format!("read cached AIR reclassification: {error}"))?;
            batch
        };
        if rows.is_empty() {
            break;
        }
        after = rows.last().expect("nonempty batch").0.clone();
        let mut classified = Vec::with_capacity(rows.len());
        for (hash, json) in rows {
            let mut result: StructuralTriage = serde_json::from_str(&json)
                .map_err(|error| format!("decode cached triage row {hash}: {error}"))?;
            result.unrecognized_air_intrinsics =
                metal2vulkan::air_intrinsics::unrecognized_air_intrinsics_from_counts(
                    &result.air_calls,
                );
            classified.push((hash, result));
        }
        write_cached(
            index,
            classified
                .iter()
                .map(|(hash, result)| (hash.as_str(), result)),
        )?;
        updated += classified.len();
    }
    Ok(updated)
}

fn ensure_cache_table(connection: &Connection) -> Result<(), String> {
    connection
        .execute_batch(
            "CREATE TABLE IF NOT EXISTS triage_analysis (
               air_sha256 TEXT PRIMARY KEY,
               analyzer_abi TEXT NOT NULL,
               result_json TEXT NOT NULL
             );",
        )
        .map_err(|error| format!("create triage cache: {error}"))
}

fn write_cached_transaction<'a>(
    transaction: &Transaction<'_>,
    results: impl IntoIterator<Item = (&'a str, &'a StructuralTriage)>,
) -> Result<(), String> {
    let mut insert = transaction
        .prepare(
            "INSERT INTO triage_analysis (air_sha256, analyzer_abi, result_json) \
             VALUES (?1, ?2, ?3) ON CONFLICT(air_sha256) DO UPDATE SET \
             analyzer_abi=excluded.analyzer_abi, result_json=excluded.result_json",
        )
        .map_err(|error| format!("prepare triage cache update: {error}"))?;
    for (hash, result) in results {
        let json = serde_json::to_string(result)
            .map_err(|error| format!("serialize triage result for {hash}: {error}"))?;
        insert
            .execute(params![hash, ANALYZER_ABI, json])
            .map_err(|error| format!("cache triage result for {hash}: {error}"))?;
    }
    Ok(())
}

pub fn classify(source: &SourceRow) -> StructuralTriage {
    classify_inner(source, true)
}

/// Classify corpus-wide support requirements without deriving the expensive per-call structural
/// signature. The blocker and tooling results are identical to [`classify`].
pub fn classify_summary(source: &SourceRow) -> StructuralTriage {
    classify_inner(source, false)
}

fn classify_inner(source: &SourceRow, include_signature: bool) -> StructuralTriage {
    let cycle = has_cfg_cycle(&source.air_ll);
    let calls = crate::executor_contract::air_call_counts(&source.air_ll);
    let resource_markers = resource_marker_counts(&source.air_ll);
    let signature = if include_signature {
        structural_signature(source, cycle, &calls, &resource_markers)
    } else {
        String::new()
    };
    let unrecognized_air_intrinsics =
        metal2vulkan::air_intrinsics::unrecognized_air_intrinsics_from_counts(&calls);
    let tooling_requirements = tooling_requirements(source);
    let mut result = StructuralTriage {
        signature,
        has_cfg_cycle: cycle,
        air_calls: calls,
        unrecognized_air_intrinsics,
        resource_markers,
        uses_device_addresses: source.air_ll.contains("inttoptr"),
        tooling_requirements,
    };
    finalize_requirements(&mut result);
    result
}

fn finalize_requirements(result: &mut StructuralTriage) {
    result.tooling_requirements.sort();
    result.tooling_requirements.dedup();
}

fn tooling_requirements(source: &SourceRow) -> Vec<ToolingRequirement> {
    let mut requirements = crate::executor_contract::unsupported_air_requirements(&source.air_ll)
        .into_iter()
        .collect::<Vec<_>>();
    if source.air_ll.contains("@air.intersect.")
        && !metal2vulkan::meta::air_intersection_calls_are_supported(&source.air_ll)
        && !opaque_intersection_is_authorable(source)
    {
        requirements.push(ToolingRequirement::RayIntersectionLowering);
    }
    if source.air_ll.contains("!\"air.visible_function_table\"")
        && audit_visible_function_tables(source).has_unsupported_use()
    {
        requirements.push(ToolingRequirement::VisibleFunctionTable);
    }
    let (stage, authored_stage) = match source.stage.as_str() {
        "Kernel" => (
            metal2vulkan::passes::Stage::Kernel,
            crate::case::Stage::Kernel,
        ),
        "Vertex" => (
            metal2vulkan::passes::Stage::Vertex,
            crate::case::Stage::Vertex,
        ),
        "Fragment" => (
            metal2vulkan::passes::Stage::Fragment,
            crate::case::Stage::Fragment,
        ),
        _ => return requirements,
    };
    if let Ok(reflection) = metal2vulkan::reflect_sanitized(
        &source.air_ll,
        stage,
        metal2vulkan::passes::TransformOptions::default(),
    ) {
        requirements.extend(crate::executor_contract::unsupported_source_requirements(
            &source.air_ll,
            &reflection,
        ));
        requirements.extend(
            crate::executor_contract::unsupported_observation_requirements(
                authored_stage,
                &reflection,
            ),
        );
    }
    requirements
}

/// Build the smallest authored opaque-triangle table that removes every callback-bearing ray call
/// in one captured entry module.
///
/// This is shared by the requirement classifier and the executable translation audit so the
/// census cannot claim authorability using a looser model than the product path actually receives.
pub fn authored_opaque_intersection_linkage(
    source: &SourceRow,
) -> Option<metal2vulkan::linked_functions::LinkedFunctionLinkage> {
    use metal2vulkan::linked_functions::{
        IntersectionFunctionEntry, IntersectionFunctionTable, IntersectionFunctionTableSource,
        LinkedFunctionLinkage,
    };

    let families = source
        .air_ll
        .lines()
        .filter_map(|line| {
            let (_, instruction) = line.split_once(" = ")?;
            let (callee, _, _) = ray_call_shape(instruction)?;
            metal2vulkan::meta::AirIntersectionFamily::parse(&callee)
                .ok()
                .flatten()
                .filter(|family| family.intersection_function_buffer)
        })
        .collect::<Vec<_>>();
    if families.len() != 1 {
        return None;
    }
    let family = &families[0];
    let signature = metal2vulkan::linked_functions::opaque_triangle_signature(family)?;
    let stage = match source.stage.as_str() {
        "Kernel" => metal2vulkan::passes::Stage::Kernel,
        "Vertex" => metal2vulkan::passes::Stage::Vertex,
        "Fragment" => metal2vulkan::passes::Stage::Fragment,
        _ => return None,
    };
    let Ok(reflection) = metal2vulkan::reflect_sanitized(
        &source.air_ll,
        stage,
        metal2vulkan::passes::TransformOptions::default(),
    ) else {
        return None;
    };
    let direct = reflection.bindings.iter().filter_map(|binding| {
        (binding.kind == metal2vulkan::reflect::ResourceKind::IntersectionFunctionTable)
            .then_some(binding.param_index)
            .flatten()
            .map(|parameter_index| IntersectionFunctionTableSource::Parameter { parameter_index })
    });
    let embedded = reflection.argument_buffer_fields.iter().map(|field| {
        IntersectionFunctionTableSource::ArgumentBuffer {
            buffer_parameter_index: field.buffer_param_index,
            field_ordinal: field.field_ordinal,
            field_offset: field.field_offset,
        }
    });
    direct.chain(embedded).find_map(|table_source| {
        let linkage = LinkedFunctionLinkage {
            visible_references: vec![],
            visible_tables: vec![],
            intersection_tables: vec![IntersectionFunctionTable {
                source: table_source,
                size: 1,
                entries: vec![IntersectionFunctionEntry::OpaqueTriangle {
                    index: 0,
                    signature: signature.clone(),
                }],
            }],
        };
        let fully_specialized =
            metal2vulkan::linked_functions::specialize_opaque_triangle_intersection_tables(
                &source.air_ll,
                &source.entry,
                &linkage,
            )
            .is_ok_and(|specialized| {
                !specialized.lines().any(|line| {
                    line.contains("call ")
                        && line.contains("@air.intersect.intersection_function_buffer")
                })
            });
        fully_specialized.then_some(linkage)
    })
}

/// Build the authored intersection-table linkage needed by translation-only census cases.
///
/// Callback-bearing ray queries use the opaque-triangle contract above. An AIR setter may also be
/// specialized when its destination is a top-level table whose ABI has no callback types: the
/// authored table is then necessarily all-null, and Logical SPIR-V presents that authored table
/// directly instead of attempting to mutate a nonexistent runtime function-table object.
pub fn authored_intersection_linkage(
    source: &SourceRow,
) -> Option<metal2vulkan::linked_functions::LinkedFunctionLinkage> {
    authored_opaque_intersection_linkage(source)
        .or_else(|| authored_empty_intersection_setter_linkage(source))
}

fn authored_empty_intersection_setter_linkage(
    source: &SourceRow,
) -> Option<metal2vulkan::linked_functions::LinkedFunctionLinkage> {
    use metal2vulkan::linked_functions::{
        IntersectionFunctionTable, IntersectionFunctionTableSource, LinkedFunctionLinkage,
    };

    if !source
        .air_ll
        .contains("call void @air.set_buffer_intersection_function_table.")
        || !source.air_ll.contains("!\"intersection_function_table<>\"")
    {
        return None;
    }
    let stage = match source.stage.as_str() {
        "Kernel" => metal2vulkan::passes::Stage::Kernel,
        "Vertex" => metal2vulkan::passes::Stage::Vertex,
        "Fragment" => metal2vulkan::passes::Stage::Fragment,
        _ => return None,
    };
    let reflection = metal2vulkan::reflect_sanitized(
        &source.air_ll,
        stage,
        metal2vulkan::passes::TransformOptions::default(),
    )
    .ok()?;
    let intersection_tables = reflection
        .bindings
        .iter()
        .filter_map(|binding| {
            (binding.kind == metal2vulkan::reflect::ResourceKind::IntersectionFunctionTable)
                .then_some(binding.param_index?)
                .map(|parameter_index| IntersectionFunctionTable {
                    source: IntersectionFunctionTableSource::Parameter { parameter_index },
                    size: 1,
                    entries: vec![],
                })
        })
        .collect::<Vec<_>>();
    if intersection_tables.is_empty() {
        return None;
    }
    let linkage = LinkedFunctionLinkage {
        visible_references: vec![],
        visible_tables: vec![],
        intersection_tables,
    };
    metal2vulkan::linked_functions::specialize_opaque_triangle_intersection_tables(
        &source.air_ll,
        &source.entry,
        &linkage,
    )
    .ok()
    .filter(|specialized| {
        !specialized
            .lines()
            .any(|line| line.contains("call void @air.set_buffer_intersection_function_table."))
    })
    .map(|_| linkage)
}

fn opaque_intersection_is_authorable(source: &SourceRow) -> bool {
    authored_opaque_intersection_linkage(source).is_some()
}

fn structural_signature(
    source: &SourceRow,
    cycle: bool,
    calls: &BTreeMap<String, usize>,
    resource_markers: &BTreeMap<String, usize>,
) -> String {
    let accesses = ["air.read", "air.write", "air.read_write", "air.sample"];
    let mut features = vec![
        format!("stage={}", source.stage),
        format!("cycle={cycle}"),
        format!(
            "defines={}",
            source
                .air_ll
                .lines()
                .filter(|line| line.trim_start().starts_with("define "))
                .count()
        ),
    ];
    for (kind, count) in resource_markers {
        features.push(format!("{kind}={count}"));
    }
    for access in accesses {
        features.push(format!(
            "{access}={}",
            source.air_ll.matches(access).count()
        ));
    }
    for (call, count) in calls {
        features.push(format!("call:{call}={count}"));
    }
    sha256_bytes(features.join("\n").as_bytes())
}

fn resource_marker_counts(ll: &str) -> BTreeMap<String, usize> {
    RESOURCE_KINDS
        .into_iter()
        .map(|kind| (kind.to_string(), ll.matches(kind).count()))
        .collect()
}

/// Describe how visible-function-table values flow through one sanitized AIR entry module.
///
/// Categories intentionally mirror the product specialization contract. A direct call or a chain
/// of `bitcast`/`addrspacecast` operations is supported; every other escape is named explicitly so
/// a corpus audit cannot mistake an unhandled pointer path for coverage.
#[derive(Clone)]
struct VisiblePointerTrace {
    lookup: usize,
    cast_depth: usize,
}

#[derive(Clone, Copy)]
enum VisibleSlotKind {
    Constant,
    Dynamic,
    UnsupportedType,
}

struct VisibleLookup {
    slot: VisibleSlotKind,
    observed_use: bool,
}

/// Catalog exact AIR ray-query ABI shapes and the result fields the module actually observes.
///
/// This is deliberately descriptive rather than a support heuristic. Product lowering can use the
/// catalog as a completeness target without treating an unrecognized family as equivalent to a
/// known one.
pub fn audit_ray_intersections(source: &SourceRow) -> RayIntersectionAudit {
    let mut audit = RayIntersectionAudit {
        product_supported: metal2vulkan::meta::air_intersection_calls_are_supported(&source.air_ll),
        ..RayIntersectionAudit::default()
    };
    let entry_parameters = entry_parameter_values(&source.air_ll, &source.entry);
    let mut results = HashMap::<String, String>::new();
    let mut null_tables = HashSet::<String>::new();
    for line in source.air_ll.lines() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("define ") || trimmed == "}" {
            results.clear();
            null_tables.clear();
            continue;
        }
        let Some((defined, instruction)) = trimmed.split_once(" = ") else {
            continue;
        };
        let defined = defined.trim();
        if instruction.contains("@air.get_null_intersection_function_table(") {
            null_tables.insert(defined.to_string());
            continue;
        }
        if instruction.contains("@air.intersect.") {
            let Some((callee, return_type, argument_count)) = ray_call_shape(instruction) else {
                audit.malformed_calls += 1;
                continue;
            };
            *audit.calls.entry(callee.clone()).or_default() += 1;
            *audit
                .signatures
                .entry(format!(
                    "{callee}|return={return_type}|arguments={argument_count}"
                ))
                .or_default() += 1;
            match metal2vulkan::meta::AirIntersectionFamily::parse(&callee) {
                Ok(Some(family)) => {
                    let expected_result = family.llvm_result_type();
                    if return_type != expected_result {
                        *audit
                            .contract_errors
                            .entry(format!(
                                "{callee}:result:{return_type}:expected:{expected_result}"
                            ))
                            .or_default() += 1;
                    }
                    let expected_arguments = family.argument_count();
                    if argument_count != expected_arguments {
                        *audit
                            .contract_errors
                            .entry(format!(
                                "{callee}:arguments:{argument_count}:expected:{expected_arguments}"
                            ))
                            .or_default() += 1;
                    }
                }
                Ok(None) => unreachable!("ray audit selected an air.intersect symbol"),
                Err(error) => {
                    *audit.contract_errors.entry(error).or_default() += 1;
                }
            }
            if let Ok(Some(family)) = metal2vulkan::meta::AirIntersectionFamily::parse(&callee) {
                let table = ray_call_arguments(instruction)
                    .and_then(|arguments| {
                        arguments
                            .get(family.intersection_table_argument_index())
                            .copied()
                    })
                    .map(audit_value_operand);
                let kind = match table {
                    Some(value) if null_tables.contains(value) => "null",
                    Some(value) if entry_parameters.contains(value) => "entry_parameter",
                    Some(_) => "derived_or_helper_parameter",
                    None => "missing",
                };
                *audit.table_operands.entry(kind.into()).or_default() += 1;
            }
            results.insert(defined.to_string(), callee);
            continue;
        }
        if !instruction.starts_with("extractvalue ") {
            continue;
        }
        for (value, callee) in &results {
            if !contains_llvm_value(instruction, value) {
                continue;
            }
            let Some((_, indices)) = instruction.split_once(&format!("{value},")) else {
                continue;
            };
            let path = indices
                .split(',')
                .map(str::trim)
                .take_while(|index| index.parse::<u32>().is_ok())
                .collect::<Vec<_>>()
                .join(".");
            if !path.is_empty() {
                *audit
                    .extracted_fields
                    .entry(format!("{callee}[{path}]"))
                    .or_default() += 1;
            }
            break;
        }
    }
    audit
}

fn ray_call_shape(instruction: &str) -> Option<(String, String, usize)> {
    let at = instruction.find("@air.intersect.")?;
    let open = instruction[at..].find('(')? + at;
    let close = audit_matching_paren(instruction, open)?;
    let callee = instruction[at + 1..open].to_string();
    let call = instruction[..at].rfind("call ")? + 5;
    let mut return_type = instruction[call..at].trim();
    while let Some((first, rest)) = return_type.split_once(char::is_whitespace) {
        if matches!(
            first,
            "fast"
                | "nnan"
                | "ninf"
                | "nsz"
                | "arcp"
                | "contract"
                | "afn"
                | "reassoc"
                | "fastcc"
                | "coldcc"
                | "ccc"
        ) {
            return_type = rest.trim_start();
        } else {
            break;
        }
    }
    if return_type.is_empty() {
        return None;
    }
    let arguments = instruction[open + 1..close].trim();
    let argument_count = if arguments.is_empty() {
        0
    } else {
        audit_split_top_level(arguments, ',').len()
    };
    Some((
        callee,
        return_type.split_whitespace().collect::<Vec<_>>().join(" "),
        argument_count,
    ))
}

fn ray_call_arguments(instruction: &str) -> Option<Vec<&str>> {
    let at = instruction.find("@air.intersect.")?;
    let open = instruction[at..].find('(')? + at;
    let close = audit_matching_paren(instruction, open)?;
    Some(audit_split_top_level(&instruction[open + 1..close], ','))
}

pub fn audit_visible_function_tables(source: &SourceRow) -> VisibleFunctionTableAudit {
    let entry_parameters = entry_parameter_values(&source.air_ll, &source.entry);
    let table_flow = visible_table_parameter_flow(source);
    let defined_functions = defined_function_globals(&source.air_ll);
    let mut result = VisibleFunctionTableAudit::default();
    let mut pointers = HashMap::<String, VisiblePointerTrace>::new();
    let mut sentinel_integers = HashMap::<String, VisiblePointerTrace>::new();
    let mut lookups = Vec::<VisibleLookup>::new();
    let mut current_function = None::<String>;
    for line in source.air_ll.lines() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("define ") {
            current_function = definition_global(trimmed);
            pointers.clear();
            sentinel_integers.clear();
            continue;
        }
        if trimmed == "}" {
            current_function = None;
            pointers.clear();
            sentinel_integers.clear();
            continue;
        }
        let rooted_tables = current_function
            .as_ref()
            .and_then(|function| table_flow.get(function));
        let in_entry = current_function
            .as_deref()
            .is_some_and(|function| function_matches_entry(function, &source.entry));
        if trimmed.contains("call ") && trimmed.contains("@air.get_size_visible_function_table(") {
            record_table_query(
                &mut result,
                "size",
                trimmed,
                &entry_parameters,
                rooted_tables,
                in_entry,
            );
        }
        if trimmed.contains("call ") && trimmed.contains("@air.is_null_visible_function_table(") {
            record_table_query(
                &mut result,
                "is_null",
                trimmed,
                &entry_parameters,
                rooted_tables,
                in_entry,
            );
        }

        let Some((defined, instruction)) = trimmed.split_once(" = ") else {
            record_pointer_escapes(
                trimmed,
                None,
                &pointers,
                &mut lookups,
                &defined_functions,
                &mut result,
            );
            continue;
        };
        let defined = defined.trim();
        if instruction.contains("@air.get_function_pointer_visible_function_table(") {
            let Some(arguments) = audit_call_arguments(instruction) else {
                *result
                    .lookup_uses
                    .entry("malformed_lookup".into())
                    .or_default() += 1;
                continue;
            };
            if arguments.len() < 2 {
                *result
                    .lookup_uses
                    .entry("malformed_lookup".into())
                    .or_default() += 1;
                continue;
            }
            let table = audit_value_operand(arguments[0]);
            let table_kind = if in_entry && entry_parameters.contains(table) {
                "direct_entry_parameter"
            } else if rooted_tables.is_some_and(|tables| tables.contains(table)) {
                "threaded_entry_parameter"
            } else {
                "derived"
            };
            *result.table_operands.entry(table_kind.into()).or_default() += 1;
            let slot = if !arguments[1].trim_start().starts_with("i32 ") {
                VisibleSlotKind::UnsupportedType
            } else if audit_value_operand(arguments[1]).parse::<u32>().is_ok() {
                VisibleSlotKind::Constant
            } else {
                VisibleSlotKind::Dynamic
            };
            let lookup = lookups.len();
            lookups.push(VisibleLookup {
                slot,
                observed_use: false,
            });
            pointers.insert(
                defined.to_string(),
                VisiblePointerTrace {
                    lookup,
                    cast_depth: 0,
                },
            );
            continue;
        }

        if instruction.starts_with("bitcast ") || instruction.starts_with("addrspacecast ") {
            if let Some(source_value) = audit_cast_source_value(instruction) {
                if let Some(mut trace) = pointers.get(source_value).cloned() {
                    trace.cast_depth += 1;
                    pointers.insert(defined.to_string(), trace);
                    continue;
                }
            }
        }

        if let Some(pointer) = audit_null_pointer_comparison(instruction) {
            if let Some(trace) = pointers.get(pointer) {
                let lookup = &mut lookups[trace.lookup];
                lookup.observed_use = true;
                let slot = visible_slot_kind(lookup.slot);
                *result
                    .lookup_uses
                    .entry(format!("{slot}.null_compare"))
                    .or_default() += 1;
                continue;
            }
        }

        if instruction.starts_with("ptrtoint ") {
            if let Some(source_value) = audit_cast_source_value(instruction) {
                if let Some(trace) = pointers.get(source_value).cloned() {
                    sentinel_integers.insert(defined.to_string(), trace);
                    continue;
                }
            }
        }

        if instruction.starts_with("trunc ") {
            if let Some(source_value) = audit_cast_source_value(instruction) {
                if let Some(trace) = sentinel_integers.get(source_value).cloned() {
                    sentinel_integers.insert(defined.to_string(), trace);
                    continue;
                }
            }
        }

        if let Some(integer) = audit_opaque_sentinel_comparison(instruction) {
            if let Some(trace) = sentinel_integers.get(integer) {
                let lookup = &mut lookups[trace.lookup];
                lookup.observed_use = true;
                let slot = visible_slot_kind(lookup.slot);
                *result
                    .lookup_uses
                    .entry(format!("{slot}.sentinel_compare"))
                    .or_default() += 1;
                continue;
            }
        }

        if let Some(callee) = audit_indirect_call_callee(trimmed) {
            if let Some(trace) = pointers.get(callee) {
                let lookup = &mut lookups[trace.lookup];
                lookup.observed_use = true;
                let slot = visible_slot_kind(lookup.slot);
                let path = if trace.cast_depth == 0 {
                    "direct_call"
                } else {
                    "cast_call"
                };
                *result
                    .lookup_uses
                    .entry(format!("{slot}.{path}"))
                    .or_default() += 1;
                record_pointer_escapes(
                    trimmed,
                    Some(callee),
                    &pointers,
                    &mut lookups,
                    &defined_functions,
                    &mut result,
                );
                record_integer_escapes(trimmed, &sentinel_integers, &mut lookups, &mut result);
                continue;
            }
        }
        record_integer_escapes(trimmed, &sentinel_integers, &mut lookups, &mut result);
        record_pointer_escapes(
            trimmed,
            None,
            &pointers,
            &mut lookups,
            &defined_functions,
            &mut result,
        );
    }
    for lookup in lookups {
        if !lookup.observed_use {
            let slot = visible_slot_kind(lookup.slot);
            *result
                .lookup_uses
                .entry(format!("{slot}.unused"))
                .or_default() += 1;
        }
    }
    result
}

fn record_integer_escapes(
    line: &str,
    integers: &HashMap<String, VisiblePointerTrace>,
    lookups: &mut [VisibleLookup],
    result: &mut VisibleFunctionTableAudit,
) {
    for (integer, trace) in integers {
        if !contains_llvm_value(line, integer) {
            continue;
        }
        let lookup = &mut lookups[trace.lookup];
        lookup.observed_use = true;
        let slot = visible_slot_kind(lookup.slot);
        *result
            .lookup_uses
            .entry(format!("{slot}.unsupported_ptrtoint_escape"))
            .or_default() += 1;
    }
}

fn visible_slot_kind(slot: VisibleSlotKind) -> &'static str {
    match slot {
        VisibleSlotKind::Constant => "constant",
        VisibleSlotKind::Dynamic => "dynamic",
        VisibleSlotKind::UnsupportedType => "unsupported_slot_type",
    }
}

fn record_table_query(
    result: &mut VisibleFunctionTableAudit,
    query: &str,
    instruction: &str,
    entry_parameters: &HashSet<String>,
    rooted_tables: Option<&HashSet<String>>,
    in_entry: bool,
) {
    let table_kind = audit_call_arguments(instruction)
        .and_then(|arguments| arguments.first().copied())
        .map(audit_value_operand)
        .map_or("malformed", |table| {
            if in_entry && entry_parameters.contains(table) {
                "direct_entry_parameter"
            } else if rooted_tables.is_some_and(|tables| tables.contains(table)) {
                "threaded_entry_parameter"
            } else {
                "derived"
            }
        });
    *result
        .queries
        .entry(format!("{query}.{table_kind}"))
        .or_default() += 1;
}

fn record_pointer_escapes(
    line: &str,
    handled_callee: Option<&str>,
    pointers: &HashMap<String, VisiblePointerTrace>,
    lookups: &mut [VisibleLookup],
    defined_functions: &HashSet<String>,
    result: &mut VisibleFunctionTableAudit,
) {
    for (pointer, trace) in pointers {
        if handled_callee == Some(pointer.as_str()) || !contains_llvm_value(line, pointer) {
            continue;
        }
        let lookup = &mut lookups[trace.lookup];
        lookup.observed_use = true;
        let slot = visible_slot_kind(lookup.slot);
        if named_defined_call(line, defined_functions) {
            *result
                .lookup_uses
                .entry(format!("{slot}.helper_parameter"))
                .or_default() += 1;
            continue;
        }
        let kind = unsupported_pointer_use(line);
        *result
            .lookup_uses
            .entry(format!("{slot}.unsupported_{kind}"))
            .or_default() += 1;
    }
}

fn defined_function_globals(ll: &str) -> HashSet<String> {
    ll.lines()
        .filter_map(|line| {
            let line = line.trim_start();
            if !line.starts_with("define ") {
                return None;
            }
            let at = line.find('@')?;
            let open = line[at..].find('(')? + at;
            Some(line[at..open].trim_end().to_string())
        })
        .collect()
}

fn named_defined_call(line: &str, defined_functions: &HashSet<String>) -> bool {
    let Some(call) = line.find("call ") else {
        return false;
    };
    let Some(relative_at) = line[call + 5..].find('@') else {
        return false;
    };
    let at = call + 5 + relative_at;
    let Some(relative_open) = line[at..].find('(') else {
        return false;
    };
    defined_functions.contains(line[at..at + relative_open].trim_end())
}

fn unsupported_pointer_use(line: &str) -> &'static str {
    let padded = format!(" {line} ");
    for (needle, kind) in [
        (" phi ", "phi"),
        (" select ", "select"),
        (" store ", "store"),
        (" load ", "load"),
        (" ptrtoint ", "ptrtoint"),
        (" icmp ", "compare"),
        (" call ", "call_argument"),
        (" ret ", "return"),
    ] {
        if padded.contains(needle) {
            return kind;
        }
    }
    "other"
}

fn audit_null_pointer_comparison(instruction: &str) -> Option<&str> {
    let operands = instruction
        .strip_prefix("icmp eq ")
        .or_else(|| instruction.strip_prefix("icmp ne "))?;
    let operands = audit_split_top_level(operands, ',');
    if operands.len() != 2 {
        return None;
    }
    let left = audit_value_operand(operands[0]);
    let right = audit_value_operand(operands[1]);
    match (left, right) {
        (pointer, "null") if pointer.starts_with('%') => Some(pointer),
        ("null", pointer) if pointer.starts_with('%') => Some(pointer),
        _ => None,
    }
}

fn audit_opaque_sentinel_comparison(instruction: &str) -> Option<&str> {
    let operands = instruction
        .strip_prefix("icmp eq ")
        .or_else(|| instruction.strip_prefix("icmp ne "))?;
    let operands = audit_split_top_level(operands, ',');
    if operands.len() != 2 {
        return None;
    }
    let left = audit_value_operand(operands[0]);
    let right = audit_value_operand(operands[1]);
    match (left, right) {
        (integer, "1") if integer.starts_with('%') => Some(integer),
        ("1", integer) if integer.starts_with('%') => Some(integer),
        _ => None,
    }
}

fn contains_llvm_value(line: &str, value: &str) -> bool {
    line.match_indices(value).any(|(start, _)| {
        let end = start + value.len();
        let boundary = |byte: Option<&u8>| {
            byte.is_none_or(|byte| {
                !byte.is_ascii_alphanumeric() && !matches!(byte, b'_' | b'.' | b'$' | b'-')
            })
        };
        boundary(
            start
                .checked_sub(1)
                .and_then(|index| line.as_bytes().get(index)),
        ) && boundary(line.as_bytes().get(end))
    })
}

fn entry_parameter_values(ll: &str, entry: &str) -> HashSet<String> {
    let globals = [
        format!("@{entry}("),
        format!("@\"{}\"(", entry.replace('"', "\\22")),
    ];
    let mut signature = String::new();
    let mut collecting = false;
    for line in ll.lines() {
        let trimmed = line.trim_start();
        if !collecting
            && trimmed.starts_with("define ")
            && globals.iter().any(|global| trimmed.contains(global))
        {
            collecting = true;
        }
        if collecting {
            signature.push_str(trimmed);
            signature.push(' ');
            if trimmed.contains('{') {
                break;
            }
        }
    }
    let Some(open) = signature.find('(') else {
        return HashSet::new();
    };
    let Some(close) = audit_matching_paren(&signature, open) else {
        return HashSet::new();
    };
    audit_split_top_level(&signature[open + 1..close], ',')
        .into_iter()
        .filter_map(|parameter| {
            parameter
                .split_whitespace()
                .last()
                .filter(|value| value.starts_with('%'))
                .map(str::to_string)
        })
        .collect()
}

fn visible_table_parameter_flow(source: &SourceRow) -> HashMap<String, HashSet<String>> {
    let roots = match source.stage.as_str() {
        "Kernel" => metal2vulkan::meta::parse_air_kernel_meta(&source.air_ll)
            .into_iter()
            .flat_map(|meta| meta.roles)
            .filter_map(|(index, role)| {
                matches!(role, metal2vulkan::meta::KernRole::VisibleFunctionTable(_))
                    .then_some(index)
            })
            .collect::<Vec<_>>(),
        "Vertex" => metal2vulkan::meta::parse_air_vertex_meta(&source.air_ll)
            .into_iter()
            .flat_map(|meta| meta.roles)
            .filter_map(|(index, role)| {
                matches!(role, metal2vulkan::meta::VertRole::VisibleFunctionTable(_))
                    .then_some(index)
            })
            .collect::<Vec<_>>(),
        "Fragment" => metal2vulkan::meta::parse_air_fragment_meta(&source.air_ll)
            .into_iter()
            .flat_map(|meta| meta.roles)
            .filter_map(|(index, role)| {
                matches!(role, metal2vulkan::meta::FragRole::VisibleFunctionTable(_))
                    .then_some(index)
            })
            .collect::<Vec<_>>(),
        _ => Vec::new(),
    };
    metal2vulkan::linked_functions::trace_visible_function_table_parameters(
        &source.air_ll,
        &source.entry,
        &roots,
    )
    .unwrap_or_default()
}

fn definition_global(line: &str) -> Option<String> {
    let at = line.find('@')?;
    let open = line[at..].find('(')? + at;
    Some(line[at..open].trim_end().to_string())
}

fn function_matches_entry(global: &str, entry: &str) -> bool {
    global == format!("@{entry}") || global == format!("@\"{}\"", entry.replace('"', "\\22"))
}

fn audit_call_arguments(instruction: &str) -> Option<Vec<&str>> {
    let open = instruction.find('(')?;
    let close = audit_matching_paren(instruction, open)?;
    Some(audit_split_top_level(&instruction[open + 1..close], ','))
}

fn audit_value_operand(argument: &str) -> &str {
    argument.split_whitespace().last().unwrap_or_default()
}

fn audit_cast_source_value(instruction: &str) -> Option<&str> {
    let (_, source_and_destination) = instruction.split_once(' ')?;
    let (source, _) = source_and_destination.rsplit_once(" to ")?;
    source
        .split_whitespace()
        .last()
        .filter(|value| value.starts_with('%'))
}

fn audit_indirect_call_callee(line: &str) -> Option<&str> {
    let call = line.find("call ")?;
    let after_call = &line[call + 5..];
    let open = after_call.find('(')? + call + 5;
    let head = &line[..open];
    let start = head.rfind(char::is_whitespace).map_or(0, |index| index + 1);
    let callee = &line[start..open];
    callee.starts_with('%').then_some(callee)
}

fn audit_matching_paren(text: &str, open: usize) -> Option<usize> {
    let mut depth = 0usize;
    for (relative, byte) in text.as_bytes()[open..].iter().enumerate() {
        match byte {
            b'(' => depth += 1,
            b')' => {
                depth = depth.checked_sub(1)?;
                if depth == 0 {
                    return Some(open + relative);
                }
            }
            _ => {}
        }
    }
    None
}

fn audit_split_top_level(text: &str, delimiter: char) -> Vec<&str> {
    let mut depth = 0usize;
    let mut start = 0usize;
    let mut parts = Vec::new();
    for (index, ch) in text.char_indices() {
        match ch {
            '(' | '[' | '{' | '<' => depth += 1,
            ')' | ']' | '}' | '>' => depth = depth.saturating_sub(1),
            _ if ch == delimiter && depth == 0 => {
                parts.push(text[start..index].trim());
                start = index + ch.len_utf8();
            }
            _ => {}
        }
    }
    parts.push(text[start..].trim());
    parts
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ScratchDir;

    fn source(ll: &str) -> SourceRow {
        SourceRow {
            air_sha256: "11".repeat(32),
            stage: "Kernel".into(),
            entry: "main".into(),
            air_ll: ll.into(),
            blob_b64: None,
            lib_sha256s: vec!["22".repeat(32)],
            label: "local/test.ll".into(),
        }
    }

    #[test]
    fn cyclic_work_stays_actionable_without_entry_name_signatures() {
        let first =
            source("define void @first() {\nentry:\n br label %loop\nloop:\n br label %loop\n}");
        let mut renamed = first.clone();
        renamed.entry = "renamed".into();
        renamed.air_ll = renamed.air_ll.replace("@first", "@renamed");
        let first = classify(&first);
        let renamed = classify(&renamed);
        assert_eq!(first.signature, renamed.signature);
        assert!(first.tooling_requirements.is_empty());
    }

    #[test]
    fn unrecognized_air_families_are_first_class_product_gaps_with_exact_counts() {
        let row = source(
            "define void @main() {\nentry:\n %a = call i32 @air.future_tensor_op.i32(i32 1)\n %b = call i32 @air.future_tensor_op.i32(i32 2)\n ret void\n}\ndeclare i32 @air.future_tensor_op.i32(i32)",
        );
        let result = classify(&row);
        assert_eq!(
            result.unrecognized_air_intrinsics,
            BTreeMap::from([("air.future_tensor_op.i32".into(), 2)])
        );
        assert!(result.tooling_requirements.is_empty());
    }

    #[test]
    fn recognized_and_linkage_intrinsics_do_not_create_unknown_family_gaps() {
        let row = source(
            "define void @main(ptr addrspace(1) %table) {\nentry:\n %a = call i32 @air.abs.s.i32(i32 -1)\n %b = call ptr @air.get_function_pointer_visible_function_table(ptr addrspace(1) %table, i32 0)\n ret void\n}\ndeclare i32 @air.abs.s.i32(i32)\ndeclare ptr @air.get_function_pointer_visible_function_table(ptr addrspace(1), i32)\n!0 = !{i32 0, !\"air.visible_function_table\"}",
        );
        let result = classify(&row);
        assert!(result.unrecognized_air_intrinsics.is_empty());
        assert!(result.tooling_requirements.is_empty());
    }

    #[test]
    fn unused_visible_table_marker_is_already_authorable() {
        let row = source(
            "define void @main(ptr addrspace(1) %table) {\nentry:\n %f = call ptr @air.get_function_pointer_visible_function_table(ptr addrspace(1) %table, i32 7)\n ret void\n}\n!0 = !{i32 0, !\"air.visible_function_table\"}",
        );
        let result = classify(&row);
        assert!(result.tooling_requirements.is_empty());
        assert!(!audit_visible_function_tables(&row).requires_authored_linkage());
    }

    #[test]
    fn visible_table_audit_matches_supported_constant_cast_call_shape() {
        let row = source(
            "define void @main(ptr addrspace(1) %table, ptr %out) {\nentry:\n %f = call ptr @air.get_function_pointer_visible_function_table(ptr addrspace(1) %table, i32 7)\n %null = icmp eq ptr %f, null\n %typed = bitcast ptr %f to ptr\n %value = call i32 %typed(ptr %out)\n ret void\n}\n!0 = !{i32 0, !\"air.visible_function_table\"}",
        );
        let audit = audit_visible_function_tables(&row);
        assert_eq!(audit.lookup_uses["constant.cast_call"], 1);
        assert_eq!(audit.lookup_uses["constant.null_compare"], 1);
        assert_eq!(audit.table_operands["direct_entry_parameter"], 1);
        assert!(!audit.has_unsupported_use());
        assert!(audit.requires_authored_linkage());
    }

    #[test]
    fn visible_table_audit_exposes_dynamic_calls_and_pointer_escapes() {
        let row = source(
            "define void @main(ptr addrspace(1) %table, i32 %slot, i1 %choose) {\nentry:\n %f = call ptr @air.get_function_pointer_visible_function_table(ptr addrspace(1) %table, i32 %slot)\n %selected = select i1 %choose, ptr %f, ptr null\n %value = call i32 %f()\n ret void\n}\n!0 = !{i32 0, !\"air.visible_function_table\"}",
        );
        let audit = audit_visible_function_tables(&row);
        assert_eq!(audit.lookup_uses["dynamic.direct_call"], 1);
        assert_eq!(audit.lookup_uses["dynamic.unsupported_select"], 1);
        assert!(audit.has_unsupported_use());
        assert!(!audit.requires_authored_linkage());
        assert_eq!(
            classify(&row).tooling_requirements,
            [ToolingRequirement::VisibleFunctionTable]
        );
    }

    #[test]
    fn visible_table_audit_accepts_only_the_reserved_opaque_sentinel_probe() {
        let supported = source(
            "define void @main(ptr addrspace(1) %table, i32 %slot) {\nentry:\n %f = call ptr @air.get_function_pointer_visible_function_table(ptr addrspace(1) %table, i32 %slot)\n %wide = ptrtoint ptr %f to i64\n %low = trunc i64 %wide to i32\n %opaque = icmp eq i32 %low, 1\n %value = call i32 %f()\n ret void\n}\n!0 = !{i32 0, !\"air.visible_function_table\"}",
        );
        let audit = audit_visible_function_tables(&supported);
        assert_eq!(audit.lookup_uses["dynamic.sentinel_compare"], 1);
        assert_eq!(audit.lookup_uses["dynamic.direct_call"], 1);
        assert!(!audit.has_unsupported_use());
        assert!(classify(&supported).tooling_requirements.is_empty());

        let unsupported = source(
            "define i64 @main(ptr addrspace(1) %table, i32 %slot) {\nentry:\n %f = call ptr @air.get_function_pointer_visible_function_table(ptr addrspace(1) %table, i32 %slot)\n %wide = ptrtoint ptr %f to i64\n %changed = add i64 %wide, 2\n ret i64 %changed\n}\n!0 = !{i32 0, !\"air.visible_function_table\"}",
        );
        let audit = audit_visible_function_tables(&unsupported);
        assert_eq!(audit.lookup_uses["dynamic.unsupported_ptrtoint_escape"], 1);
        assert!(audit.has_unsupported_use());
    }

    #[test]
    fn callback_free_intersection_table_is_not_a_separate_tooling_gap() {
        let row = source(
            "define void @main(ptr addrspace(1) %table) {\nentry:\n %hit = call { i32, float, i32, i32, ptr addrspace(1), i32, i32 } @air.intersect.instancing(<3 x float> zeroinitializer, <3 x float> zeroinitializer, float 0.0, float 1.0, ptr addrspace(1) null, i32 255, ptr addrspace(1) %table, ptr null, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 1, i32 -1, i32 -1, i32 0, i1 false, i1 false)\n ret void\n}\n!0 = !{i32 0, !\"air.intersection_function_table\"}",
        );
        assert!(classify(&row).tooling_requirements.is_empty());
    }

    #[test]
    fn callback_family_with_authored_opaque_table_is_not_a_tooling_gap() {
        let row = source(
            r#"define void @main(ptr addrspace(1) %table, ptr addrspace(1) %as) {
entry:
 %hit = call { i32, float, i32, i32, ptr addrspace(1), <2 x float>, i1 } @air.intersect.intersection_function_buffer.triangle_data(<3 x float> zeroinitializer, <3 x float> zeroinitializer, float 0.0, float 1.0, ptr addrspace(1) %as, ptr addrspace(1) %table, i64 1, i64 8, ptr null, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 -1, i32 -1, i32 0, i1 false, i1 false)
 ret void
}
!air.kernel = !{!0}
!0 = !{ptr @main, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.intersection_function_table", !"air.location_index", i32 6, i32 1, !"air.read"}
!4 = !{i32 1, !"air.primitive_acceleration_structure", !"air.location_index", i32 5, i32 1, !"air.read"}
"#,
        );
        assert!(opaque_intersection_is_authorable(&row));
        assert!(classify(&row).tooling_requirements.is_empty());
    }

    #[test]
    fn empty_intersection_table_setter_has_exact_authored_linkage() {
        let row = source(
            r#"define void @main(ptr addrspace(1) %destination, ptr addrspace(1) %source, i32 %index) {
entry:
 call void @air.set_buffer_intersection_function_table.p1i8(ptr addrspace(1) %destination, ptr addrspace(1) %source, i32 %index)
 ret void
}
declare void @air.set_buffer_intersection_function_table.p1i8(ptr addrspace(1), ptr addrspace(1), i32)
!air.kernel = !{!0}
!0 = !{ptr @main, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.intersection_function_table", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.arg_type_name", !"intersection_function_table<>"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 1}
!5 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint"}
"#,
        );
        let linkage = authored_intersection_linkage(&row).expect("empty table is authorable");
        let specialized =
            metal2vulkan::linked_functions::specialize_opaque_triangle_intersection_tables(
                &row.air_ll,
                &row.entry,
                &linkage,
            )
            .unwrap();
        assert!(!specialized.lines().any(|line| {
            line.contains("call void @air.set_buffer_intersection_function_table.")
        }));
    }

    #[test]
    fn visible_table_audit_treats_internal_helper_threading_as_supported() {
        let row = source(
            "define void @main(ptr addrspace(1) %table, i32 %slot) {\nentry:\n %f = call ptr @air.get_function_pointer_visible_function_table(ptr addrspace(1) %table, i32 %slot)\n %typed = bitcast ptr %f to ptr\n %value = call i32 @invoke(ptr %typed)\n ret void\n}\ndefine internal i32 @invoke(ptr %callback) {\nentry:\n %value = call i32 %callback()\n ret i32 %value\n}\n!0 = !{i32 0, !\"air.visible_function_table\"}",
        );
        let audit = audit_visible_function_tables(&row);
        assert_eq!(audit.lookup_uses["dynamic.helper_parameter"], 1);
        assert!(!audit.has_unsupported_use());
    }

    #[test]
    fn visible_table_audit_traces_the_table_through_an_internal_helper() {
        let row = source(
            "define void @main(ptr addrspace(1) %table, i32 %slot) {\nentry:\n %value = call i32 @invoke(ptr addrspace(1) %table, i32 %slot)\n ret void\n}\ndefine internal i32 @invoke(ptr addrspace(1) %functions, i32 %slot) {\nentry:\n %f = call ptr @air.get_function_pointer_visible_function_table(ptr addrspace(1) %functions, i32 %slot)\n %value = call i32 %f()\n ret i32 %value\n}\ndeclare ptr @air.get_function_pointer_visible_function_table(ptr addrspace(1), i32)\n!air.kernel = !{!0}\n!0 = !{ptr @main, !1, !2}\n!1 = !{}\n!2 = !{!3}\n!3 = !{i32 0, !\"air.visible_function_table\", !\"air.location_index\", i32 1, i32 1, !\"air.read\"}",
        );
        let audit = audit_visible_function_tables(&row);
        assert_eq!(audit.table_operands["threaded_entry_parameter"], 1);
        assert_eq!(audit.lookup_uses["dynamic.direct_call"], 1);
        assert!(!audit.has_unsupported_use());
        assert!(audit.requires_authored_linkage());
        assert!(classify(&row).tooling_requirements.is_empty());
    }

    #[test]
    fn cached_requirement_selection_never_reads_source_bodies() {
        let scratch = ScratchDir::new("triage-requirement-selection").unwrap();
        let index = scratch.path().join("index.sqlite");
        let connection = Connection::open(&index).unwrap();
        connection
            .execute_batch(
                "CREATE TABLE sources (air_sha256 TEXT PRIMARY KEY, stage TEXT NOT NULL);\n\
                 CREATE TABLE triage_analysis (\n\
                   air_sha256 TEXT PRIMARY KEY,\n\
                   analyzer_abi TEXT NOT NULL,\n\
                   result_json TEXT NOT NULL\n\
                 );\n\
                 INSERT INTO sources VALUES ('a', 'Kernel'), ('b', 'Kernel');",
            )
            .unwrap();
        let mut matching = classify(&source(
            "define void @main() { ret void }\n!0 = !{i32 0, !\"air.visible_function_table\"}",
        ));
        matching.tooling_requirements = vec![ToolingRequirement::VisibleFunctionTable];
        let unrelated = classify(&source("define void @main() { ret void }"));
        drop(connection);
        write_cached(&index, [("a", &matching), ("b", &unrelated)]).unwrap();
        assert_eq!(
            select_cached_requirement(&index, ToolingRequirement::VisibleFunctionTable, 200)
                .unwrap(),
            ["a"]
        );
        assert!(select_cached_requirement_after(
            &index,
            ToolingRequirement::VisibleFunctionTable,
            Some("a"),
            200,
        )
        .unwrap()
        .is_empty());
    }

    #[test]
    fn focused_audits_select_supported_structure_not_only_tooling_gaps() {
        let scratch = ScratchDir::new("triage-audit-targets").unwrap();
        let index = scratch.path().join("index.sqlite");
        let connection = Connection::open(&index).unwrap();
        connection
            .execute_batch(
                "CREATE TABLE sources (air_sha256 TEXT PRIMARY KEY, stage TEXT NOT NULL);\n\
                 INSERT INTO sources VALUES ('device', 'Kernel'), ('ray', 'Kernel'), ('visible', 'Kernel');",
            )
            .unwrap();
        drop(connection);

        let visible = classify(&source(
            "define void @main() { ret void }\n!0 = !{i32 0, !\"air.visible_function_table\"}",
        ));
        let ray = classify(&source(
            "define void @main() { call void @air.intersect.synthetic() ret void }",
        ));
        let device = classify(&source(
            "define ptr @main(i64 %address) { %pointer = inttoptr i64 %address to ptr ret ptr %pointer }",
        ));
        assert!(visible.tooling_requirements.is_empty());
        write_cached(
            &index,
            [("visible", &visible), ("ray", &ray), ("device", &device)],
        )
        .unwrap();

        assert_eq!(
            select_cached_audit_target_after(&index, AuditTarget::VisibleFunctionTables, None, 10,)
                .unwrap(),
            ["visible"]
        );
        assert_eq!(
            select_cached_audit_target_after(&index, AuditTarget::RayIntersections, None, 10)
                .unwrap(),
            ["ray"]
        );
        assert_eq!(
            select_cached_audit_target_after(
                &index,
                AuditTarget::DeviceAddressHierarchy,
                None,
                10,
            )
            .unwrap(),
            ["device"]
        );
    }

    #[test]
    fn authoring_capability_census_includes_every_source_state() {
        let scratch = ScratchDir::new("authoring-capability-census").unwrap();
        let index = scratch.path().join("index.sqlite");
        let connection = Connection::open(&index).unwrap();
        connection
            .execute_batch(
                "CREATE TABLE sources (air_sha256 TEXT PRIMARY KEY, stage TEXT NOT NULL);\n\
                 INSERT INTO sources VALUES \
                    ('authored', 'Kernel'), ('reviewed', 'Kernel'), \
                    ('unknown-air', 'Kernel'), ('unplanned', 'Kernel');",
            )
            .unwrap();
        drop(connection);

        let supported = classify(&source("define void @main() { ret void }"));
        let mut unsupported = supported.clone();
        unsupported.tooling_requirements = vec![ToolingRequirement::IndirectCommandBuffer];
        let mut unknown_air = supported.clone();
        unknown_air.unrecognized_air_intrinsics =
            BTreeMap::from([("air.future_tensor.i32".into(), 3)]);
        write_cached(
            &index,
            [
                ("authored", &supported),
                ("reviewed", &unsupported),
                ("unknown-air", &unknown_air),
            ],
        )
        .unwrap();

        assert_eq!(select_uncached(&index, 100).unwrap(), ["unplanned"]);
        assert_eq!(select_uncached(&index, 0).unwrap(), Vec::<String>::new());
        assert_eq!(
            select_all(&index, 100).unwrap(),
            ["authored", "reviewed", "unknown-air", "unplanned"]
        );
        assert_eq!(select_all(&index, 2).unwrap(), ["authored", "reviewed"]);
        let summary = authoring_capability_summary(&index).unwrap();
        assert_eq!(summary.total_sources, 4);
        assert_eq!(summary.classified_sources, 3);
        assert_eq!(summary.remaining_sources(), 1);
        assert_eq!(summary.unresolved_sources, 2);
        assert_eq!(
            summary.requirements[&ToolingRequirement::IndirectCommandBuffer],
            1
        );
        assert_eq!(
            summary.unrecognized_air_intrinsics["air.future_tensor.i32"],
            3
        );
    }

    #[test]
    fn acceleration_structure_contract_is_actionable_not_a_review_blocker() {
        let row = source(
            "define void @main(ptr addrspace(1) %as) { ret void }\n!0 = !{i32 0, !\"air.instance_acceleration_structure\"}",
        );
        let result = classify(&row);
        assert!(result.tooling_requirements.is_empty());
    }

    #[test]
    fn callback_free_primitive_intersection_is_supported_end_to_end() {
        let resource_only = source(
            "define void @main(ptr addrspace(1) %as) { ret void }\n!0 = !{i32 0, !\"air.primitive_acceleration_structure\"}",
        );
        assert!(classify(&resource_only).tooling_requirements.is_empty());

        let intersect = source(
            "define void @main(ptr addrspace(1) %as) { %hit = call i1 @air.intersect.triangle_data(ptr addrspace(1) %as) ret void }\n!0 = !{i32 0, !\"air.primitive_acceleration_structure\"}",
        );
        assert!(classify(&intersect).tooling_requirements.is_empty());
    }

    #[test]
    fn multi_level_device_address_walk_is_supported_end_to_end() {
        let row = source(
            "define void @main(ptr addrspace(1) %as, ptr addrspace(1) %table, i64 %address, ptr %ids, ptr %user_ids) {\nentry:\n %hit = call { i32, float, i32, i32, ptr addrspace(1), i8 } @air.intersect.multi_level_instancing(<3 x float> zeroinitializer, <3 x float> zeroinitializer, float 0.0, float 1.0, ptr addrspace(1) %as, i32 255, ptr addrspace(1) %table, ptr null, i64 0, i8 2, ptr %ids, ptr %user_ids, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 -1, i32 -1, i32 0, i1 false, i1 false)\n %child = inttoptr i64 %address to ptr addrspace(1)\n ret void\n}",
        );
        assert!(classify(&row).tooling_requirements.is_empty());
    }

    #[test]
    fn ray_audit_preserves_family_return_layout_and_observed_fields() {
        let row = source(
            "define void @main() {\nentry:\n %hit = call { i32, float, ptr addrspace(1) } @air.intersect.instancing.world_space_data(i32 0, ptr null)\n %kind = extractvalue { i32, float, ptr addrspace(1) } %hit, 0\n %opaque = extractvalue { i32, float, ptr addrspace(1) } %hit, 2\n ret void\n}",
        );
        let audit = audit_ray_intersections(&row);
        assert_eq!(audit.calls["air.intersect.instancing.world_space_data"], 1);
        assert_eq!(
            audit.signatures["air.intersect.instancing.world_space_data|return={ i32, float, ptr addrspace(1) }|arguments=2"],
            1
        );
        assert_eq!(
            audit.extracted_fields["air.intersect.instancing.world_space_data[0]"],
            1
        );
        assert_eq!(
            audit.extracted_fields["air.intersect.instancing.world_space_data[2]"],
            1
        );
        assert_eq!(audit.malformed_calls, 0);
    }

    #[test]
    fn ray_audit_uses_the_shared_table_operand_position() {
        let row = source(
            "define void @main(ptr addrspace(1) %table) {\nentry:\n %null = call ptr addrspace(1) @air.get_null_intersection_function_table()\n %a = call { i32, float, i32, i32, ptr addrspace(1), <2 x float>, i1 } @air.intersect.triangle_data(i32 0, i32 0, i32 0, i32 0, i32 0, ptr addrspace(1) %table, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)\n %b = call { i32, float, i32, i32, ptr addrspace(1), <2 x float>, i1 } @air.intersect.triangle_data(i32 0, i32 0, i32 0, i32 0, i32 0, ptr addrspace(1) %null, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)\n ret void\n}",
        );
        let audit = audit_ray_intersections(&row);
        assert_eq!(audit.table_operands["entry_parameter"], 1);
        assert_eq!(audit.table_operands["null"], 1);
        assert!(audit.contract_errors.is_empty());
    }

    #[test]
    fn tooling_requirements_are_independent_not_precedence_ordered() {
        let mut row = source(
            "define void @main() { ret void }\n!0 = !{i32 0, !\"air.imageblock\"}\n!1 = !{i32 1, !\"air.command_buffer\", !\"air.location_index\", i32 0}\n!2 = !{i32 2, !\"air.texture\", !\"air.arg_type_name\", !\"array_ref<texture2d<float, sample>>\"}",
        );
        row.stage = "Fragment".into();
        let result = classify(&row);
        assert_eq!(
            result.tooling_requirements,
            [ToolingRequirement::IndirectCommandBuffer]
        );
    }

    #[test]
    fn unknown_implicit_imageblock_suffix_is_not_a_false_clean_census_row() {
        let supported = source(
            "define void @main() { %v = call <2 x half> @air.load.implicit_imageblock.v2f16(i32 0, <2 x i16> zeroinitializer, i32 0, i16 0) ret void }\ndeclare <2 x half> @air.load.implicit_imageblock.v2f16(i32, <2 x i16>, i32, i16)",
        );
        assert!(classify(&supported).tooling_requirements.is_empty());

        let unknown = source(
            "define void @main() { %v = call <3 x half> @air.load.implicit_imageblock.v3f16(i32 0, <2 x i16> zeroinitializer, i32 0, i16 0) ret void }\ndeclare <3 x half> @air.load.implicit_imageblock.v3f16(i32, <2 x i16>, i32, i16)",
        );
        assert_eq!(
            classify(&unknown).tooling_requirements,
            [ToolingRequirement::ImplicitImageblockLiteral]
        );
    }

    #[test]
    fn cache_reuses_only_the_current_analyzer_contract() {
        let scratch = ScratchDir::new("triage-cache").unwrap();
        let index = scratch.path().join("index.sqlite");
        let connection = Connection::open(&index).unwrap();
        connection
            .execute_batch(
                "CREATE TABLE sources (air_sha256 TEXT PRIMARY KEY, stage TEXT NOT NULL);\n\
                 INSERT INTO sources VALUES ('air', 'Kernel');",
            )
            .unwrap();
        drop(connection);
        let result = classify(&source("define void @main() { ret void }"));
        write_cached(&index, [("air", &result)]).unwrap();
        let hashes = vec!["air".to_string()];
        assert_eq!(
            read_cached(&index, &hashes).unwrap().get("air"),
            Some(&result)
        );

        let connection = Connection::open(&index).unwrap();
        connection
            .execute(
                "UPDATE triage_analysis SET analyzer_abi='obsolete' WHERE air_sha256='air'",
                [],
            )
            .unwrap();
        assert!(read_cached(&index, &hashes).unwrap().is_empty());
    }

    #[test]
    fn cached_air_inventory_reclassifies_without_source_rows() {
        let scratch = ScratchDir::new("triage-cache-air-reclassify").unwrap();
        let index = scratch.path().join("index.sqlite");
        let connection = Connection::open(&index).unwrap();
        connection
            .execute_batch(
                "CREATE TABLE sources (air_sha256 TEXT PRIMARY KEY, stage TEXT NOT NULL);\n\
                 INSERT INTO sources VALUES ('air', 'Kernel');",
            )
            .unwrap();
        drop(connection);
        let symbol =
            "air.simdgroup_matrix_16x16x16_multiply_accumulate.f.f.v8f32.v8f16.v8f16.v8f32";
        let mut stale = classify(&source("define void @main() { ret void }"));
        stale.air_calls.insert(symbol.into(), 3);
        stale.unrecognized_air_intrinsics.insert(symbol.into(), 3);
        write_cached(&index, [("air", &stale)]).unwrap();

        assert_eq!(reclassify_cached_air_intrinsics(&index).unwrap(), 1);
        let current = read_cached(&index, &["air".into()]).unwrap();
        assert!(current["air"].unrecognized_air_intrinsics.is_empty());
        assert_eq!(current["air"].air_calls[symbol], 3);
    }

    #[test]
    fn analyzer_upgrade_reopens_every_row_when_the_contract_affects_every_stage() {
        let scratch = ScratchDir::new("triage-cache-upgrade-scope").unwrap();
        let index = scratch.path().join("index.sqlite");
        let connection = Connection::open(&index).unwrap();
        connection
            .execute_batch(
                "CREATE TABLE sources (air_sha256 TEXT PRIMARY KEY, stage TEXT NOT NULL);\n\
                 INSERT INTO sources VALUES ('affected', 'Kernel'), ('supported', 'Vertex'), ('unrelated', 'Kernel'), ('fragment', 'Fragment');",
            )
            .unwrap();
        drop(connection);
        let mut affected = classify(&source("define void @main() { ret void }"));
        affected.air_calls.insert(
            "air.intersect.intersection_function_buffer.triangle_data".into(),
            1,
        );
        let supported_marker = classify(&source(
            "define void @main() { ret void }\n!0 = !{i32 0, !\"air.visible_function_table\"}",
        ));
        let unrelated = classify(&source("define void @main() { ret void }"));
        let fragment = unrelated.clone();
        write_cached(
            &index,
            [
                ("affected", &affected),
                ("supported", &supported_marker),
                ("unrelated", &unrelated),
                ("fragment", &fragment),
            ],
        )
        .unwrap();
        let connection = Connection::open(&index).unwrap();
        connection
            .execute(
                "UPDATE triage_analysis SET analyzer_abi=?1",
                ["structural-triage-v13"],
            )
            .unwrap();
        drop(connection);

        let hashes = ["affected", "supported", "unrelated", "fragment"]
            .map(str::to_string)
            .to_vec();
        assert!(read_cached(&index, &hashes).unwrap().is_empty());
        assert_eq!(
            select_uncached(&index, 10).unwrap(),
            ["affected", "fragment", "supported", "unrelated"]
        );

        let connection = Connection::open(&index).unwrap();
        let versions = ["affected", "supported", "unrelated", "fragment"]
            .into_iter()
            .map(|hash| {
                connection
                    .query_row(
                        "SELECT analyzer_abi FROM triage_analysis WHERE air_sha256=?1",
                        [hash],
                        |row| row.get::<_, String>(0),
                    )
                    .unwrap()
            })
            .collect::<Vec<_>>();
        assert_eq!(versions, ["structural-triage-v13"; 4]);
    }

    #[test]
    fn framebuffer_fetch_uses_the_shared_supported_reflection_contract() {
        let mut row = source(
            r#"define <4 x float> @frag(<4 x float> %color) { ret <4 x float> %color }
!air.fragment = !{!0}
!0 = !{ptr @frag, !1, !3}
!1 = !{!2}
!2 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!3 = !{!4}
!4 = !{i32 0, !"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
"#,
        );
        row.stage = "Fragment".into();
        assert!(classify(&row).tooling_requirements.is_empty());
    }

    #[test]
    fn observation_requirements_use_the_executor_type_and_linkage_contract() {
        let mut supported = source(
            r#"!air.fragment = !{!0}
!0 = !{ptr @frag, !1, !3}
!1 = !{!2}
!2 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!3 = !{!4}
!4 = !{i32 0, !"air.fragment_input", !"user(payload)", !"air.arg_type_name", !"bool4", !"air.arg_name", !"not.valid"}
"#,
        );
        supported.stage = "Fragment".into();
        assert!(classify(&supported).tooling_requirements.is_empty());

        let unsupported = SourceRow {
            air_ll: supported.air_ll.replace("!\"bool4\"", "!\"double2\""),
            ..supported
        };
        assert_eq!(
            classify(&unsupported).tooling_requirements,
            [ToolingRequirement::FragmentVaryingObservationType]
        );
    }

    #[test]
    fn vertex_without_position_is_an_explicit_observation_gap() {
        let mut row = source(
            r#"!air.vertex = !{!0}
!0 = !{ptr @vertex, !1, !2}
!1 = !{!3}
!2 = !{}
!3 = !{!"air.vertex_output", !"user(payload)", !"air.arg_type_name", !"float4", !"air.arg_name", !"payload"}
"#,
        );
        row.stage = "Vertex".into();
        assert_eq!(
            classify(&row).tooling_requirements,
            [ToolingRequirement::VertexSideEffectObservation]
        );
    }
}
