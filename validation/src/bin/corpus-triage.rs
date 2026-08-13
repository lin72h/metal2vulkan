use metal2vulkan_validation::index::{default_index_path, select_queue, QueueRow, QueueState};
use metal2vulkan_validation::requirement::ToolingRequirement;
use metal2vulkan_validation::source::{
    corpus_root, for_each_indexed_source_analysis, for_each_indexed_source_analysis_with_stats,
};
use metal2vulkan_validation::translation_audit::{
    select_translation_audit_batch, translation_audit_summary, write_translation_audit_results,
    SelectionMode, TranslationAuditResult, TranslationAuditStatus,
};
use metal2vulkan_validation::triage::{
    authored_intersection_linkage, authoring_capability_summary, classify, classify_summary,
    read_cached, reclassify_cached_air_intrinsics, select_cached_audit_target_after,
    select_uncached, write_cached, AuditTarget, StructuralTriage,
};
use metal2vulkan_validation::ScratchDir;
use std::alloc::{GlobalAlloc, Layout, System};
use std::borrow::Cow;
use std::collections::BTreeMap;
use std::fs;
use std::io::Write as _;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::time::{Duration, Instant};
use wait_timeout::ChildExt as _;

const TRANSLATION_TIMEOUT: Duration = Duration::from_secs(30);
const TRANSLATION_MEMORY_LIMIT_BYTES: u64 = 512 * 1024 * 1024;
const LARGE_TRANSLATION_SOURCE_BYTES: usize = 1024 * 1024;
const MAX_LARGE_TRANSLATION_JOBS: usize = 4;
// Leave headroom inside the resident-memory contract for allocator metadata, fragmentation, stacks,
// and mapped runtime pages that the live-allocation counter cannot see. The parent independently
// enforces the full 512 MiB RSS ceiling, so this is an early fail-safe rather than a second contract.
const TRANSLATION_ALLOCATION_LIMIT_BYTES: usize = 416 * 1024 * 1024;
static LIVE_ALLOCATED_BYTES: AtomicUsize = AtomicUsize::new(0);
static ALLOCATION_LIMIT_BYTES: AtomicUsize = AtomicUsize::new(usize::MAX);

struct TranslationBudgetAllocator;

#[global_allocator]
static GLOBAL_ALLOCATOR: TranslationBudgetAllocator = TranslationBudgetAllocator;

unsafe impl GlobalAlloc for TranslationBudgetAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if !reserve_allocation(layout.size()) {
            return std::ptr::null_mut();
        }
        let allocation = unsafe { System.alloc(layout) };
        if allocation.is_null() {
            LIVE_ALLOCATED_BYTES.fetch_sub(layout.size(), Ordering::Relaxed);
        }
        allocation
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) };
        LIVE_ALLOCATED_BYTES.fetch_sub(layout.size(), Ordering::Relaxed);
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if new_size > layout.size() && !reserve_allocation(new_size - layout.size()) {
            return std::ptr::null_mut();
        }
        let allocation = unsafe { System.realloc(ptr, layout, new_size) };
        if allocation.is_null() {
            if new_size > layout.size() {
                LIVE_ALLOCATED_BYTES.fetch_sub(new_size - layout.size(), Ordering::Relaxed);
            }
        } else if new_size < layout.size() {
            LIVE_ALLOCATED_BYTES.fetch_sub(layout.size() - new_size, Ordering::Relaxed);
        }
        allocation
    }
}

fn reserve_allocation(bytes: usize) -> bool {
    LIVE_ALLOCATED_BYTES
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |live| {
            live.checked_add(bytes)
                .filter(|next| *next <= ALLOCATION_LIMIT_BYTES.load(Ordering::Relaxed))
        })
        .is_ok()
}

fn enable_translation_memory_budget() {
    let baseline = LIVE_ALLOCATED_BYTES.load(Ordering::Relaxed);
    ALLOCATION_LIMIT_BYTES.store(
        baseline.saturating_add(TRANSLATION_ALLOCATION_LIMIT_BYTES),
        Ordering::Relaxed,
    );
}

fn main() {
    let worker = std::env::args().nth(1).as_deref() == Some("--translation-worker");
    let result = if worker {
        run_translation_worker()
    } else {
        run()
    };
    if let Err(error) = result {
        eprintln!("corpus-triage: {error}");
        std::process::exit(1);
    }
}

#[derive(Clone, Copy)]
enum AuditKind {
    AuthoringCapabilities,
    VisibleFunctionTables,
    RayIntersections,
    DeviceAddressHierarchy,
    Translation,
}

enum TranslationSelection<'a> {
    Indexed {
        after: Option<&'a str>,
        limit: usize,
        mode: SelectionMode,
    },
    HashFile(&'a std::path::Path),
}

impl AuditKind {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "authoring-capabilities" => Ok(Self::AuthoringCapabilities),
            "visible-function-tables" => Ok(Self::VisibleFunctionTables),
            "ray-intersections" => Ok(Self::RayIntersections),
            "device-address-hierarchy" => Ok(Self::DeviceAddressHierarchy),
            "translation" => Ok(Self::Translation),
            _ => Err(format!(
                "unknown audit {value:?}; expected authoring-capabilities, visible-function-tables, ray-intersections, device-address-hierarchy, or translation"
            )),
        }
    }
}

fn run() -> Result<(), String> {
    let mut root = corpus_root();
    let mut index = None;
    let mut limit = 200usize;
    let mut summary_only = false;
    let mut audit = None;
    let mut after = None;
    let mut current_fingerprint = false;
    let mut retry_failures = false;
    let mut reclassify_all = false;
    let mut hash_file = None;
    let mut jobs = std::thread::available_parallelism()
        .map(|parallelism| parallelism.get())
        .unwrap_or(1);
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--corpus" => root = PathBuf::from(required(&mut args, "--corpus")?),
            "--index" => index = Some(PathBuf::from(required(&mut args, "--index")?)),
            "--limit" => {
                limit = required(&mut args, "--limit")?
                    .parse()
                    .map_err(|error| format!("invalid --limit: {error}"))?
            }
            "--jobs" => {
                jobs = required(&mut args, "--jobs")?
                    .parse()
                    .map_err(|error| format!("invalid --jobs: {error}"))?
            }
            "--summary-only" => summary_only = true,
            "--audit" => audit = Some(AuditKind::parse(&required(&mut args, "--audit")?)?),
            "--after" => after = Some(required(&mut args, "--after")?),
            "--current-fingerprint" => current_fingerprint = true,
            "--retry-failures" => retry_failures = true,
            "--reclassify-all" => reclassify_all = true,
            "--hash-file" => hash_file = Some(PathBuf::from(required(&mut args, "--hash-file")?)),
            "-h" | "--help" => {
                println!(
                    "usage: corpus-triage [--corpus DIR] [--index PATH] [--limit N] [--jobs N] [--summary-only] [--audit authoring-capabilities [--reclassify-all] | --audit visible-function-tables|ray-intersections|device-address-hierarchy|translation [--after SHA256] [--current-fingerprint | --retry-failures | --hash-file PATH]]\n\n--jobs defaults to all available logical CPU cores; lower it to constrain aggregate worker memory. --hash-file reads a whitespace-delimited SHA-256 from the first field of each non-empty line and audits exactly those indexed sources."
                );
                return Ok(());
            }
            _ => return Err(format!("unknown argument {arg:?}")),
        }
    }
    if limit == 0 {
        return Err("--limit must be greater than zero".into());
    }
    if jobs == 0 {
        return Err("--jobs must be greater than zero".into());
    }
    if after.is_some() && audit.is_none() {
        return Err("--after requires --audit".into());
    }
    if current_fingerprint && !matches!(audit, Some(AuditKind::Translation)) {
        return Err("--current-fingerprint requires --audit translation".into());
    }
    if retry_failures && !matches!(audit, Some(AuditKind::Translation)) {
        return Err("--retry-failures requires --audit translation".into());
    }
    if reclassify_all && !matches!(audit, Some(AuditKind::AuthoringCapabilities)) {
        return Err("--reclassify-all requires --audit authoring-capabilities".into());
    }
    if hash_file.is_some() && !matches!(audit, Some(AuditKind::Translation)) {
        return Err("--hash-file requires --audit translation".into());
    }
    if hash_file.is_some() && (after.is_some() || current_fingerprint || retry_failures) {
        return Err(
            "--hash-file is mutually exclusive with --after, --current-fingerprint, and --retry-failures"
                .into(),
        );
    }
    if current_fingerprint && retry_failures {
        return Err("--current-fingerprint and --retry-failures are mutually exclusive".into());
    }
    if after.as_deref().is_some_and(|hash| {
        hash.len() != 64
            || !hash
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    }) {
        return Err("--after must be a lowercase SHA-256".into());
    }

    let index = index.unwrap_or_else(|| default_index_path(&root));
    if let Some(audit) = audit {
        return match audit {
            AuditKind::AuthoringCapabilities => {
                if after.is_some() {
                    return Err(
                        "--after is not accepted by the authoring-capabilities audit".into(),
                    );
                }
                audit_authoring_capabilities(&root, &index, limit, jobs, reclassify_all)
            }
            AuditKind::VisibleFunctionTables => {
                audit_visible_function_tables(&root, &index, after.as_deref(), limit, summary_only)
            }
            AuditKind::RayIntersections => {
                audit_ray_intersections(&root, &index, after.as_deref(), limit, summary_only)
            }
            AuditKind::DeviceAddressHierarchy => audit_device_address_hierarchy(
                &root,
                &index,
                after.as_deref(),
                limit,
                jobs,
                summary_only,
            ),
            AuditKind::Translation => audit_translation(
                &root,
                &index,
                jobs,
                summary_only,
                if let Some(path) = hash_file.as_deref() {
                    TranslationSelection::HashFile(path)
                } else {
                    TranslationSelection::Indexed {
                        after: after.as_deref(),
                        limit,
                        mode: if retry_failures {
                            SelectionMode::RetryCurrentFailures
                        } else if current_fingerprint {
                            SelectionMode::CurrentFingerprint
                        } else {
                            SelectionMode::Discovery
                        },
                    }
                },
            ),
        };
    }
    let queue = select_queue(&index, QueueState::Unplanned, limit)?;
    let selected_hashes = queue
        .iter()
        .map(|row| row.air_sha256.clone())
        .collect::<Vec<_>>();
    let mut results = read_cached(&index, &selected_hashes)?;
    let missing = queue
        .iter()
        .filter(|row| !results.contains_key(&row.air_sha256))
        .map(|row| row.air_sha256.clone())
        .collect::<Vec<_>>();
    let mut fresh = BTreeMap::new();
    for_each_indexed_source_analysis(&root, &index, &missing, |source| {
        fresh.insert(source.air_sha256.clone(), classify(&source));
        Ok(())
    })?;
    write_cached(
        &index,
        fresh.iter().map(|(hash, result)| (hash.as_str(), result)),
    )?;
    results.append(&mut fresh);
    let mut summary = TriageSummary::default();
    for row in queue {
        let result = results
            .get(&row.air_sha256)
            .ok_or_else(|| format!("selected AIR {} has no triage result", row.air_sha256))?;
        summary.process(row, result, summary_only);
    }
    println!(
        "summary\tselected={}\tactionable={}\tsignatures={}",
        summary.actionable,
        summary.actionable,
        if summary_only {
            "skipped".to_string()
        } else {
            summary.signatures.len().to_string()
        }
    );
    for (stage, count) in summary.stages {
        println!("stage\t{stage}\t{count}");
    }
    for (requirement, count) in summary.requirements {
        println!("requirement\t{requirement}\t{count}");
    }
    if !summary_only {
        for (signature, count) in summary.signatures {
            println!("signature\t{signature}\t{count}");
        }
    }
    Ok(())
}

fn audit_authoring_capabilities(
    root: &std::path::Path,
    index: &std::path::Path,
    limit: usize,
    jobs: usize,
    reclassify_all: bool,
) -> Result<(), String> {
    let started = Instant::now();
    let hashes = (!reclassify_all)
        .then(|| select_uncached(index, limit))
        .transpose()?
        .unwrap_or_default();
    let selection_elapsed = started.elapsed();
    let source_started = Instant::now();
    let (selected, read_stats) = if reclassify_all {
        (
            reclassify_cached_air_intrinsics(index)?,
            metal2vulkan_validation::source::IndexedSourceReadStats::default(),
        )
    } else {
        (
            hashes.len(),
            classify_and_cache_parallel(root, index, &hashes, jobs)?,
        )
    };
    let source_elapsed = source_started.elapsed();
    let summary = authoring_capability_summary(index)?;
    println!(
        "authoring-capability-summary\tselected={}\ttotal={}\tclassified={}\tremaining={}\tunresolved={}\tindex_select_ms={}\tclassify_cache_ms={}\tsource_shards_opened={}\tsource_bytes_read={}\trepair_shards_scanned={}\trepair_bytes_scanned={}",
        selected,
        summary.total_sources,
        summary.classified_sources,
        summary.remaining_sources(),
        summary.unresolved_sources,
        selection_elapsed.as_millis(),
        source_elapsed.as_millis(),
        read_stats.source_shards_opened,
        read_stats.source_bytes_read,
        read_stats.repair_shards_scanned,
        read_stats.repair_bytes_scanned,
    );
    for (requirement, count) in &summary.requirements {
        println!("authoring-requirement\t{requirement}\t{count}");
    }
    for (intrinsic, count) in &summary.unrecognized_air_intrinsics {
        println!("unrecognized-air-intrinsic\t{intrinsic}\t{count}");
    }
    if summary.remaining_sources() == 0 && summary.unresolved_sources != 0 {
        return Err(format!(
            "{} source(s) still have validation support gaps",
            summary.unresolved_sources
        ));
    }
    Ok(())
}

fn classify_and_cache_parallel(
    root: &std::path::Path,
    index: &std::path::Path,
    hashes: &[String],
    jobs: usize,
) -> Result<metal2vulkan_validation::source::IndexedSourceReadStats, String> {
    const CACHE_BATCH: usize = 1024;
    let worker_count = jobs.min(hashes.len().max(1));
    std::thread::scope(|scope| {
        let (result_sender, result_receiver) =
            mpsc::sync_channel::<(String, StructuralTriage)>(worker_count.saturating_mul(2));
        let writer = scope.spawn(move || -> Result<(), String> {
            let mut batch = Vec::with_capacity(CACHE_BATCH);
            for result in result_receiver {
                batch.push(result);
                if batch.len() == CACHE_BATCH {
                    write_triage_batch(index, &batch)?;
                    batch.clear();
                }
            }
            write_triage_batch(index, &batch)
        });
        // A source can be tens of MiB. Keep only one parsed row waiting outside the workers; a
        // worker-count-sized queue would turn the channel itself into an unbounded-in-practice
        // corpus cache and can exceed the process memory budget on adjacent large rows.
        let (source_sender, source_receiver) =
            mpsc::sync_channel::<metal2vulkan_validation::source::SourceRow>(1);
        let source_receiver = Arc::new(Mutex::new(source_receiver));
        let mut workers = Vec::with_capacity(worker_count);
        for _ in 0..worker_count {
            let source_receiver = Arc::clone(&source_receiver);
            let result_sender = result_sender.clone();
            workers.push(scope.spawn(move || loop {
                let source = match source_receiver.lock() {
                    Ok(receiver) => receiver.recv(),
                    Err(_) => break,
                };
                let Ok(source) = source else {
                    break;
                };
                let hash = source.air_sha256.clone();
                let result = classify_summary(&source);
                if result_sender.send((hash, result)).is_err() {
                    break;
                }
            }));
        }
        drop(result_sender);
        let read = for_each_indexed_source_analysis_with_stats(root, index, hashes, |source| {
            source_sender
                .send(source)
                .map_err(|_| "authoring-capability classifier stopped unexpectedly".to_string())
        });
        drop(source_sender);
        for worker in workers {
            worker
                .join()
                .map_err(|_| "authoring-capability classifier panicked".to_string())?;
        }
        writer
            .join()
            .map_err(|_| "authoring-capability cache writer panicked".to_string())??;
        read
    })
}

fn write_triage_batch(
    index: &std::path::Path,
    batch: &[(String, StructuralTriage)],
) -> Result<(), String> {
    if batch.is_empty() {
        return Ok(());
    }
    write_cached(
        index,
        batch.iter().map(|(hash, result)| (hash.as_str(), result)),
    )
}

fn audit_translation(
    root: &std::path::Path,
    index: &std::path::Path,
    jobs: usize,
    summary_only: bool,
    selection: TranslationSelection<'_>,
) -> Result<(), String> {
    let started = Instant::now();
    let hashes = match selection {
        TranslationSelection::Indexed { after, limit, mode } => {
            select_translation_audit_batch(index, mode, after, limit)?
        }
        TranslationSelection::HashFile(path) => read_hash_file(path)?,
    };
    let selection_elapsed = started.elapsed();
    let source_started = Instant::now();
    let mut sources = Vec::with_capacity(hashes.len());
    let read_stats = for_each_indexed_source_analysis_with_stats(root, index, &hashes, |source| {
        sources.push(source);
        Ok(())
    })?;
    let source_elapsed = source_started.elapsed();

    let sources = Arc::new(sources);
    let (large_work, small_work) = translation_work_lanes(&sources);
    let large_work = Arc::new(large_work);
    let small_work = Arc::new(small_work);
    let next_large = AtomicUsize::new(0);
    let next_small = AtomicUsize::new(0);
    let (sender, receiver) = mpsc::channel();
    let mut results = std::iter::repeat_with(|| None)
        .take(sources.len())
        .collect::<Vec<Option<TranslationAuditResult>>>();
    let mut elapsed_ms = vec![0u128; sources.len()];
    let translation_started = Instant::now();
    std::thread::scope(|scope| {
        let worker_count = jobs.min(sources.len().max(1));
        let large_worker_count = MAX_LARGE_TRANSLATION_JOBS
            .min(large_work.len())
            .min(worker_count);
        for worker_index in 0..worker_count {
            let sources = Arc::clone(&sources);
            let large_work = Arc::clone(&large_work);
            let small_work = Arc::clone(&small_work);
            let sender = sender.clone();
            let next_large = &next_large;
            let next_small = &next_small;
            let handles_large = worker_index < large_worker_count;
            scope.spawn(move || loop {
                let source_index = handles_large
                    .then(|| {
                        large_work
                            .get(next_large.fetch_add(1, Ordering::Relaxed))
                            .copied()
                    })
                    .flatten()
                    .or_else(|| {
                        small_work
                            .get(next_small.fetch_add(1, Ordering::Relaxed))
                            .copied()
                    });
                let Some(source_index) = source_index else {
                    break;
                };
                let source = &sources[source_index];
                let started = Instant::now();
                let result = audit_translation_source(source);
                if sender
                    .send((source_index, result, started.elapsed().as_millis()))
                    .is_err()
                {
                    break;
                }
            });
        }
        drop(sender);
        let mut checkpoint = Vec::with_capacity(10);
        for completed in 1..=sources.len() {
            let (result_index, result, result_elapsed_ms) = receiver
                .recv()
                .map_err(|error| format!("translation audit worker stopped: {error}"))?;
            checkpoint.push(result.clone());
            results[result_index] = Some(result);
            elapsed_ms[result_index] = result_elapsed_ms;
            if completed == sources.len() || completed.is_multiple_of(10) {
                // Persist before reporting progress. A killed or interrupted long-running census
                // therefore loses at most the current short checkpoint, and its next resumable
                // selection skips every completed row instead of repeating the whole batch.
                write_translation_audit_results(index, &checkpoint)?;
                checkpoint.clear();
                eprintln!(
                    "# translation-audit progress={completed}/{} jobs={jobs}",
                    sources.len()
                );
            }
        }
        Ok::<(), String>(())
    })?;
    let translation_elapsed = translation_started.elapsed();
    let results = results
        .into_iter()
        .map(|result| result.expect("every translation audit worker returned a result"))
        .collect::<Vec<_>>();

    let mut translated = 0usize;
    let mut authored_linkage = 0usize;
    let mut failures = BTreeMap::<String, usize>::new();
    for result in &results {
        match result.status {
            TranslationAuditStatus::Translated => {
                translated += 1;
                if !summary_only {
                    println!("translation-audit\t{}\ttranslated", result.air_sha256);
                }
            }
            TranslationAuditStatus::AuthoredLinkageRequired => {
                authored_linkage += 1;
                if !summary_only {
                    println!(
                        "translation-audit\t{}\tauthored-linkage-required",
                        result.air_sha256
                    );
                }
            }
            TranslationAuditStatus::Failed => {
                let shape = result.failure_shape.as_deref().unwrap_or("unknown");
                *failures.entry(shape.to_string()).or_default() += 1;
                if !summary_only {
                    println!("translation-audit\t{}\tfailed\t{shape}", result.air_sha256);
                }
            }
        }
    }
    let census = translation_audit_summary(index)?;
    let slowest = elapsed_ms
        .iter()
        .enumerate()
        .max_by_key(|(_, elapsed)| *elapsed)
        .map(|(index, elapsed)| (sources[index].air_sha256.as_str(), *elapsed))
        .unwrap_or(("none", 0));
    println!(
        "translation-audit-summary\tselected={}\ttranslated={}\tauthored_linkage_required={}\tfailed={}\tfailure_shapes={}\tindex_select_ms={}\tsource_read_ms={}\ttranslate_validate_ms={}\tslowest_air_sha256={}\tslowest_ms={}\tindexed_rows={}\tsource_shards_opened={}\tsource_bytes_read={}\trepair_shards_scanned={}\trepair_bytes_scanned={}\tdiscovery_covered={}\tdiscovery_remaining={}\tcurrent_attempted={}\tcurrent_translated={}\tcurrent_authored_linkage_required={}\tcurrent_failed={}\tcurrent_remaining={}",
        results.len(),
        translated,
        authored_linkage,
        results.len() - translated - authored_linkage,
        failures.len(),
        selection_elapsed.as_millis(),
        source_elapsed.as_millis(),
        translation_elapsed.as_millis(),
        slowest.0,
        slowest.1,
        read_stats.rows,
        read_stats.source_shards_opened,
        read_stats.source_bytes_read,
        read_stats.repair_shards_scanned,
        read_stats.repair_bytes_scanned,
        census.discovery_covered,
        census.discovery_remaining,
        census.current_attempted,
        census.current_translated,
        census.current_authored_linkage_required,
        census.current_failed,
        census.current_remaining,
    );
    for (shape, count) in failures {
        println!("translation-audit-failure\t{count}\t{shape}");
    }
    Ok(())
}

fn read_hash_file(path: &std::path::Path) -> Result<Vec<String>, String> {
    let contents = fs::read_to_string(path)
        .map_err(|error| format!("read hash file {}: {error}", path.display()))?;
    let mut hashes = std::collections::BTreeSet::new();
    for (line_index, line) in contents.lines().enumerate() {
        let Some(hash) = line.split_whitespace().next() else {
            continue;
        };
        if hash.len() != 64
            || !hash
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(format!(
                "{}:{} first field must be a lowercase SHA-256",
                path.display(),
                line_index + 1
            ));
        }
        hashes.insert(hash.to_string());
    }
    if hashes.is_empty() {
        return Err(format!("hash file {} is empty", path.display()));
    }
    Ok(hashes.into_iter().collect())
}

fn translation_work_lanes(
    sources: &[metal2vulkan_validation::source::SourceRow],
) -> (Vec<usize>, Vec<usize>) {
    let mut large = Vec::new();
    let mut small = Vec::new();
    for (index, source) in sources.iter().enumerate() {
        if source.air_ll.len() >= LARGE_TRANSLATION_SOURCE_BYTES {
            large.push(index);
        } else {
            small.push(index);
        }
    }
    let largest_first = |left: &usize, right: &usize| {
        sources[*right]
            .air_ll
            .len()
            .cmp(&sources[*left].air_ll.len())
            .then_with(|| left.cmp(right))
    };
    large.sort_by(largest_first);
    small.sort_by(largest_first);
    (large, small)
}

fn audit_translation_source(
    source: &metal2vulkan_validation::source::SourceRow,
) -> TranslationAuditResult {
    match audit_translation_source_in_worker(source) {
        Ok(result) => result,
        Err(error) => TranslationAuditResult {
            air_sha256: source.air_sha256.clone(),
            status: TranslationAuditStatus::Failed,
            failure_shape: Some(normalize_failure_shape(&error)),
            detail: Some(bounded_detail(&error)),
        },
    }
}

fn audit_translation_source_in_worker(
    source: &metal2vulkan_validation::source::SourceRow,
) -> Result<TranslationAuditResult, String> {
    let scratch = ScratchDir::new("translation-audit-worker")?;
    let translation_tmp = scratch.path().join("translation");
    fs::create_dir(&translation_tmp)
        .map_err(|error| format!("create {}: {error}", translation_tmp.display()))?;
    let stdout_path = scratch.path().join("stdout.json");
    let stderr_path = scratch.path().join("stderr.txt");
    let executable = std::env::current_exe()
        .map_err(|error| format!("resolve corpus-triage executable: {error}"))?;
    let mut command = Command::new(executable);
    command
        .arg("--translation-worker")
        .arg(&translation_tmp)
        .stdin(Stdio::piped())
        .stdout(Stdio::from(fs::File::create(&stdout_path).map_err(
            |error| format!("create {}: {error}", stdout_path.display()),
        )?))
        .stderr(Stdio::from(fs::File::create(&stderr_path).map_err(
            |error| format!("create {}: {error}", stderr_path.display()),
        )?));
    configure_process_group(&mut command);
    let input = serde_json::to_vec(source)
        .map_err(|error| format!("encode translation worker input: {error}"))?;
    let started = Instant::now();
    let mut child = command
        .spawn()
        .map_err(|error| format!("spawn translation worker: {error}"))?;
    let mut child_stdin = child
        .stdin
        .take()
        .ok_or("translation worker stdin unavailable")?;
    // Stream the row concurrently so decoding is covered by the same wall/RSS watchdog as
    // translation. A synchronous write can fill the pipe while the child is parsing a large JSON
    // string, postponing both limits until most of the work and allocation have already happened.
    let input_writer = std::thread::spawn(move || child_stdin.write_all(&input));
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break Ok(status),
            Ok(None) => {}
            Err(error) => {
                terminate_child(&mut child);
                break Err(format!("poll translation worker: {error}"));
            }
        }
        if started.elapsed() >= TRANSLATION_TIMEOUT {
            terminate_child(&mut child);
            let stderr = fs::read_to_string(&stderr_path).unwrap_or_default();
            break Err(format!(
                "translation timeout after {} seconds: {}",
                TRANSLATION_TIMEOUT.as_secs(),
                bounded_detail(&stderr)
            ));
        }
        match worker_resident_bytes(child.id()) {
            Ok(Some(resident)) if resident > TRANSLATION_MEMORY_LIMIT_BYTES => {
                terminate_child(&mut child);
                let stderr = fs::read_to_string(&stderr_path).unwrap_or_default();
                break Err(format!(
                    "translation exceeded {} MiB resident-memory budget (measured {} MiB): {}",
                    TRANSLATION_MEMORY_LIMIT_BYTES / (1024 * 1024),
                    resident / (1024 * 1024),
                    single_line(&stderr)
                ));
            }
            Ok(_) => {}
            Err(error) => {
                terminate_child(&mut child);
                break Err(error);
            }
        }
        std::thread::sleep(Duration::from_millis(10));
    };
    let input_result = input_writer
        .join()
        .map_err(|_| "translation worker input writer panicked".to_string())?;
    let status = status?;
    let stderr = fs::read_to_string(&stderr_path).unwrap_or_default();
    if let Err(error) = input_result {
        return Err(format!(
            "write translation worker input: {error}: {}",
            single_line(&stderr)
        ));
    }
    if !status.success() {
        return Err(format!(
            "translation worker failed under {} MiB memory limit: {}",
            TRANSLATION_MEMORY_LIMIT_BYTES / (1024 * 1024),
            single_line(&stderr)
        ));
    }
    let output = fs::read(&stdout_path)
        .map_err(|error| format!("read {}: {error}", stdout_path.display()))?;
    serde_json::from_slice(&output)
        .map_err(|error| format!("decode translation worker result: {error}"))
}

fn run_translation_worker() -> Result<(), String> {
    enable_translation_memory_budget();
    let translation_tmp = std::env::args_os()
        .nth(2)
        .map(PathBuf::from)
        .ok_or("translation worker scratch path is unavailable")?;
    let mut source: metal2vulkan_validation::source::SourceRow =
        serde_json::from_reader(std::io::stdin().lock())
            .map_err(|error| format!("decode translation worker input: {error}"))?;
    // Translation consumes sanitized AIR and authored linkage metadata, never the original encoded
    // bitcode. Release that potentially multi-megabyte transport field before parsing the AIR graph.
    source.blob_b64 = None;
    let result = audit_translation_source_inline(&source, &translation_tmp);
    serde_json::to_writer(std::io::stdout().lock(), &result)
        .map_err(|error| format!("encode translation worker result: {error}"))?;
    Ok(())
}

fn audit_translation_source_inline(
    source: &metal2vulkan_validation::source::SourceRow,
    translation_tmp: &std::path::Path,
) -> TranslationAuditResult {
    let outcome = translate_and_validate_source(source, translation_tmp);
    match outcome {
        Ok(()) => TranslationAuditResult {
            air_sha256: source.air_sha256.clone(),
            status: TranslationAuditStatus::Translated,
            failure_shape: None,
            detail: None,
        },
        Err(error) if visible_linkage_is_the_only_missing_input(source, &error) => {
            TranslationAuditResult {
                air_sha256: source.air_sha256.clone(),
                status: TranslationAuditStatus::AuthoredLinkageRequired,
                failure_shape: None,
                detail: Some(bounded_detail(&error)),
            }
        }
        Err(error) => TranslationAuditResult {
            air_sha256: source.air_sha256.clone(),
            status: TranslationAuditStatus::Failed,
            failure_shape: Some(normalize_failure_shape(&error)),
            detail: Some(bounded_detail(&error)),
        },
    }
}

fn translate_and_validate_source(
    source: &metal2vulkan_validation::source::SourceRow,
    translation_tmp: &std::path::Path,
) -> Result<(), String> {
    let stage = match source.stage.as_str() {
        "Kernel" => metal2vulkan::passes::Stage::Kernel,
        "Vertex" => metal2vulkan::passes::Stage::Vertex,
        "Fragment" => metal2vulkan::passes::Stage::Fragment,
        other => return Err(format!("unknown stage {other:?}")),
    };
    let linkage = authored_intersection_linkage(source).unwrap_or_default();
    let prepared = if linkage.is_empty() {
        Cow::Borrowed(source.air_ll.as_str())
    } else {
        Cow::Owned(
            metal2vulkan::linked_functions::specialize_opaque_triangle_intersection_tables(
                &source.air_ll,
                &source.entry,
                &linkage,
            )?,
        )
    };

    // `METAL2VULKAN_RETRY_DUMP` promises the first invalid candidate, not whichever later tier
    // happens to fail last. The primary-only gate owns that diagnostic contract. Keep its extra
    // emit+validation strictly default-off, then run the ordinary complete cascade unchanged.
    if metal2vulkan::env_vars::retry_dump().is_some() {
        let _ = metal2vulkan::translate_native_primary_validated(&prepared, stage, translation_tmp);
    }

    let options = metal2vulkan::passes::TransformOptions {
        raster_sample_count: (stage == metal2vulkan::passes::Stage::Fragment).then_some(1),
        ..metal2vulkan::passes::TransformOptions::default()
    };
    let spv = metal2vulkan::translate_sanitized_native_with_options(
        &prepared,
        stage,
        translation_tmp,
        options,
    )?;
    metal2vulkan::tools::spirv_val_bytes(&spv, translation_tmp)
}

#[cfg(target_os = "macos")]
fn worker_resident_bytes(pid: u32) -> Result<Option<u64>, String> {
    let mut usage = std::mem::MaybeUninit::<libc::rusage_info_v2>::zeroed();
    let result = unsafe {
        libc::proc_pid_rusage(
            pid as libc::c_int,
            libc::RUSAGE_INFO_V2,
            usage.as_mut_ptr().cast::<libc::rusage_info_t>(),
        )
    };
    if result == 0 {
        Ok(Some(unsafe { usage.assume_init() }.ri_resident_size))
    } else if std::io::Error::last_os_error().raw_os_error() == Some(libc::ESRCH) {
        Ok(None)
    } else {
        Err(format!(
            "read translation worker resident memory: {}",
            std::io::Error::last_os_error()
        ))
    }
}

#[cfg(target_os = "linux")]
fn worker_resident_bytes(pid: u32) -> Result<Option<u64>, String> {
    let path = format!("/proc/{pid}/status");
    let status = match fs::read_to_string(&path) {
        Ok(status) => status,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(format!("read {path}: {error}")),
    };
    let resident_kib = status
        .lines()
        .find_map(|line| line.strip_prefix("VmRSS:"))
        .and_then(|value| value.split_whitespace().next())
        .and_then(|value| value.parse::<u64>().ok());
    Ok(resident_kib.map(|kib| kib * 1024))
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn worker_resident_bytes(_pid: u32) -> Result<Option<u64>, String> {
    Ok(None)
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
    let group = -(child.id() as i32);
    unsafe {
        let _ = libc::kill(group, libc::SIGTERM);
    }
    if child
        .wait_timeout(Duration::from_millis(100))
        .ok()
        .flatten()
        .is_none()
    {
        unsafe {
            let _ = libc::kill(group, libc::SIGKILL);
        }
        let _ = child.wait();
    }
}

#[cfg(not(unix))]
fn terminate_child(child: &mut Child) {
    let _ = child.kill();
    let _ = child.wait();
}

fn visible_linkage_is_the_only_missing_input(
    source: &metal2vulkan_validation::source::SourceRow,
    error: &str,
) -> bool {
    if error.contains("unsupported Metal visible function reference") {
        return source.air_ll.contains("!air.visible_function_references");
    }
    error.contains("unsupported indirect call through function pointer")
        && source.air_ll.contains("air.visible_function_table")
        && !metal2vulkan_validation::triage::audit_visible_function_tables(source)
            .has_unsupported_use()
}

fn bounded_detail(error: &str) -> String {
    let detail = single_line(error);
    detail.chars().take(4096).collect()
}

fn normalize_failure_shape(error: &str) -> String {
    let error = single_line(error);
    if error.starts_with("translation timeout after ") {
        return "translation timeout after # seconds".into();
    }
    let mut normalized = String::with_capacity(error.len().min(1024));
    let mut digits = false;
    for ch in error.chars().take(4096) {
        if ch.is_ascii_digit() {
            if !digits {
                normalized.push('#');
                digits = true;
            }
        } else {
            digits = false;
            normalized.push(ch);
        }
        if normalized.len() >= 1024 {
            break;
        }
    }
    normalized
}

fn audit_visible_function_tables(
    root: &std::path::Path,
    index: &std::path::Path,
    after: Option<&str>,
    limit: usize,
    summary_only: bool,
) -> Result<(), String> {
    let hashes =
        select_cached_audit_target_after(index, AuditTarget::VisibleFunctionTables, after, limit)?;
    let mut shapes = BTreeMap::<String, usize>::new();
    let mut uses = BTreeMap::<String, usize>::new();
    let mut table_operands = BTreeMap::<String, usize>::new();
    let mut queries = BTreeMap::<String, usize>::new();
    let mut unsupported_sources = 0usize;
    for_each_indexed_source_analysis(root, index, &hashes, |source| {
        let audit = metal2vulkan_validation::triage::audit_visible_function_tables(&source);
        let shape = audit.shape();
        *shapes.entry(shape.clone()).or_default() += 1;
        if audit.has_unsupported_use() {
            unsupported_sources += 1;
        }
        merge_counts(&mut uses, &audit.lookup_uses);
        merge_counts(&mut table_operands, &audit.table_operands);
        merge_counts(&mut queries, &audit.queries);
        if !summary_only {
            println!("visible-function-table\t{}\t{shape}", source.air_sha256);
        }
        Ok(())
    })?;
    println!(
        "visible-function-table-summary\tselected={}\tfirst={}\tlast={}\tunsupported_sources={}\tshapes={}",
        hashes.len(),
        hashes.first().map_or("none", String::as_str),
        hashes.last().map_or("none", String::as_str),
        unsupported_sources,
        shapes.len()
    );
    for (kind, count) in uses {
        println!("visible-function-table-use\t{kind}\t{count}");
    }
    for (kind, count) in table_operands {
        println!("visible-function-table-operand\t{kind}\t{count}");
    }
    for (kind, count) in queries {
        println!("visible-function-table-query\t{kind}\t{count}");
    }
    for (shape, count) in shapes {
        println!("visible-function-table-shape\t{count}\t{shape}");
    }
    Ok(())
}

fn audit_ray_intersections(
    root: &std::path::Path,
    index: &std::path::Path,
    after: Option<&str>,
    limit: usize,
    summary_only: bool,
) -> Result<(), String> {
    let hashes =
        select_cached_audit_target_after(index, AuditTarget::RayIntersections, after, limit)?;
    let mut calls = BTreeMap::<String, usize>::new();
    let mut signatures = BTreeMap::<String, usize>::new();
    let mut fields = BTreeMap::<String, usize>::new();
    let mut table_operands = BTreeMap::<String, usize>::new();
    let mut contract_errors = BTreeMap::<String, usize>::new();
    let mut shapes = BTreeMap::<String, usize>::new();
    let mut malformed_sources = 0usize;
    let mut product_supported_sources = 0usize;
    let mut opaque_authorable_sources = 0usize;
    let mut refreshed = BTreeMap::new();
    for_each_indexed_source_analysis(root, index, &hashes, |source| {
        let classified = classify(&source);
        opaque_authorable_sources += usize::from(
            !classified
                .tooling_requirements
                .contains(&ToolingRequirement::RayIntersectionLowering),
        );
        refreshed.insert(source.air_sha256.clone(), classified);
        let audit = metal2vulkan_validation::triage::audit_ray_intersections(&source);
        let shape = audit.shape();
        *shapes.entry(shape.clone()).or_default() += 1;
        malformed_sources += usize::from(audit.malformed_calls != 0);
        product_supported_sources += usize::from(audit.product_supported);
        merge_counts(&mut calls, &audit.calls);
        merge_counts(&mut signatures, &audit.signatures);
        merge_counts(&mut fields, &audit.extracted_fields);
        merge_counts(&mut table_operands, &audit.table_operands);
        merge_counts(&mut contract_errors, &audit.contract_errors);
        if !summary_only {
            println!("ray-intersection\t{}\t{shape}", source.air_sha256);
        }
        Ok(())
    })?;
    write_cached(
        index,
        refreshed
            .iter()
            .map(|(hash, result)| (hash.as_str(), result)),
    )?;
    println!(
        "ray-intersection-summary\tselected={}\tfirst={}\tlast={}\tmalformed_sources={}\tproduct_supported_sources={}\topaque_authorable_sources={}\tshapes={}",
        hashes.len(),
        hashes.first().map_or("none", String::as_str),
        hashes.last().map_or("none", String::as_str),
        malformed_sources,
        product_supported_sources,
        opaque_authorable_sources,
        shapes.len()
    );
    for (callee, count) in calls {
        println!("ray-intersection-call\t{callee}\t{count}");
    }
    for (signature, count) in signatures {
        println!("ray-intersection-signature\t{count}\t{signature}");
    }
    for (field, count) in fields {
        println!("ray-intersection-field\t{field}\t{count}");
    }
    for (kind, count) in table_operands {
        println!("ray-intersection-table\t{kind}\t{count}");
    }
    for (error, count) in contract_errors {
        println!("ray-intersection-contract-error\t{count}\t{error}");
    }
    for (shape, count) in shapes {
        println!("ray-intersection-shape\t{count}\t{shape}");
    }
    Ok(())
}

fn audit_device_address_hierarchy(
    root: &std::path::Path,
    index: &std::path::Path,
    after: Option<&str>,
    limit: usize,
    jobs: usize,
    summary_only: bool,
) -> Result<(), String> {
    let started = Instant::now();
    let hashes =
        select_cached_audit_target_after(index, AuditTarget::DeviceAddressHierarchy, after, limit)?;
    let selection_elapsed = started.elapsed();
    let mut sources = Vec::with_capacity(hashes.len());
    let source_started = Instant::now();
    let read_stats = for_each_indexed_source_analysis_with_stats(root, index, &hashes, |source| {
        sources.push(source);
        Ok(())
    })?;
    let source_elapsed = source_started.elapsed();
    let translation_started = Instant::now();
    let sources = Arc::new(sources);
    let next = AtomicUsize::new(0);
    let (sender, receiver) = mpsc::channel();
    let mut results = std::iter::repeat_with(|| None)
        .take(sources.len())
        .collect::<Vec<Option<Result<TranslationAuditStatus, String>>>>();
    std::thread::scope(|scope| {
        for _ in 0..jobs.min(sources.len().max(1)) {
            let sources = Arc::clone(&sources);
            let sender = sender.clone();
            let next = &next;
            scope.spawn(move || loop {
                let index = next.fetch_add(1, Ordering::Relaxed);
                let Some(source) = sources.get(index) else {
                    break;
                };
                if sender
                    .send((index, audit_device_address_source(source)))
                    .is_err()
                {
                    break;
                }
            });
        }
        drop(sender);
        for completed in 1..=sources.len() {
            let (index, result) = receiver
                .recv()
                .map_err(|error| format!("device-address audit worker stopped: {error}"))?;
            results[index] = Some(result);
            if completed == sources.len() || completed.is_multiple_of(10) {
                eprintln!(
                    "# device-address-hierarchy progress={completed}/{} jobs={jobs}",
                    sources.len()
                );
            }
        }
        Ok::<(), String>(())
    })?;
    let translation_elapsed = translation_started.elapsed();

    let mut translated = 0usize;
    let mut authored_linkage = 0usize;
    let mut failures = BTreeMap::<String, usize>::new();
    for (source, result) in sources.iter().zip(results) {
        match result.expect("every device-address audit worker returned a result") {
            Ok(TranslationAuditStatus::Translated) => {
                translated += 1;
                if !summary_only {
                    println!(
                        "device-address-hierarchy\t{}\ttranslated",
                        source.air_sha256
                    );
                }
            }
            Ok(TranslationAuditStatus::AuthoredLinkageRequired) => {
                authored_linkage += 1;
                if !summary_only {
                    println!(
                        "device-address-hierarchy\t{}\tauthored-linkage-required",
                        source.air_sha256
                    );
                }
            }
            Ok(TranslationAuditStatus::Failed) => {
                unreachable!("failed device-address results are returned as errors")
            }
            Err(error) => {
                let error = single_line(&error);
                *failures.entry(error.clone()).or_default() += 1;
                if !summary_only {
                    println!(
                        "device-address-hierarchy\t{}\tfailed\t{error}",
                        source.air_sha256
                    );
                }
            }
        }
    }
    println!(
        "device-address-hierarchy-summary\tselected={}\tfirst={}\tlast={}\ttranslated={}\tauthored_linkage_required={}\tfailed={}\tfailure_shapes={}\tindex_select_ms={}\tsource_read_ms={}\ttranslate_validate_ms={}\tindexed_rows={}\tsource_shards_opened={}\tsource_bytes_read={}\trepair_shards_scanned={}\trepair_bytes_scanned={}",
        hashes.len(),
        hashes.first().map_or("none", String::as_str),
        hashes.last().map_or("none", String::as_str),
        translated,
        authored_linkage,
        hashes.len() - translated - authored_linkage,
        failures.len(),
        selection_elapsed.as_millis(),
        source_elapsed.as_millis(),
        translation_elapsed.as_millis(),
        read_stats.rows,
        read_stats.source_shards_opened,
        read_stats.source_bytes_read,
        read_stats.repair_shards_scanned,
        read_stats.repair_bytes_scanned
    );
    for (error, count) in failures {
        println!("device-address-hierarchy-failure\t{count}\t{error}");
    }
    Ok(())
}

fn audit_device_address_source(
    source: &metal2vulkan_validation::source::SourceRow,
) -> Result<TranslationAuditStatus, String> {
    let result = audit_translation_source(source);
    match result.status {
        TranslationAuditStatus::Translated | TranslationAuditStatus::AuthoredLinkageRequired => {
            Ok(result.status)
        }
        TranslationAuditStatus::Failed => Err(result
            .detail
            .or(result.failure_shape)
            .unwrap_or_else(|| "bounded translation audit failed without detail".into())),
    }
}

fn merge_counts(target: &mut BTreeMap<String, usize>, source: &BTreeMap<String, usize>) {
    for (kind, count) in source {
        *target.entry(kind.clone()).or_default() += count;
    }
}

#[derive(Default)]
struct TriageSummary {
    signatures: BTreeMap<String, usize>,
    actionable: usize,
    requirements: BTreeMap<ToolingRequirement, usize>,
    stages: BTreeMap<String, usize>,
}

impl TriageSummary {
    fn process(&mut self, source: QueueRow, result: &StructuralTriage, summary_only: bool) {
        *self.stages.entry(source.stage.clone()).or_default() += 1;
        for requirement in &result.tooling_requirements {
            *self.requirements.entry(*requirement).or_default() += 1;
        }
        if !summary_only {
            *self.signatures.entry(result.signature.clone()).or_default() += 1;
        }
        self.actionable += 1;
        if !summary_only {
            let requirements = result
                .tooling_requirements
                .iter()
                .map(|requirement| requirement.as_str())
                .collect::<Vec<_>>()
                .join(",");
            println!(
                "actionable\t{}\t{}\t{}\t{}\t{}",
                source.air_sha256,
                source.stage,
                result.signature,
                source.entry,
                if requirements.is_empty() {
                    "authoring_review"
                } else {
                    requirements.as_str()
                }
            );
        }
    }
}

fn single_line(value: &str) -> String {
    value.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn required(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, String> {
    args.next()
        .ok_or_else(|| format!("{flag} requires a value"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn direct_visible_reference_failure_is_an_authored_linkage_input() {
        let source = metal2vulkan_validation::source::SourceRow {
            air_sha256: "11".repeat(32),
            stage: "Kernel".into(),
            entry: "main".into(),
            air_ll: "define void @main() { ret void }\n!air.visible_function_references = !{!0}\n!0 = !{!\"air.visible_function_reference\", ptr @linked.MTL_VISIBLE_FN_REF, !\"linked\"}\n".into(),
            blob_b64: None,
            lib_sha256: "22".repeat(32),
            label: "local/test.ll".into(),
        };
        assert!(visible_linkage_is_the_only_missing_input(
            &source,
            "native emitter: unsupported Metal visible function reference; dynamic linked functions are not expressible in Logical SPIR-V"
        ));
    }

    #[test]
    fn translation_work_separates_large_sources_and_orders_each_lane_largest_first() {
        let source = |hash: &str, bytes: usize| metal2vulkan_validation::source::SourceRow {
            air_sha256: hash.repeat(64),
            stage: "Kernel".into(),
            entry: "main".into(),
            air_ll: "x".repeat(bytes),
            blob_b64: None,
            lib_sha256: "22".repeat(32),
            label: format!("local/{hash}.ll"),
        };
        let sources = vec![
            source("a", 10),
            source("b", LARGE_TRANSLATION_SOURCE_BYTES + 1),
            source("c", LARGE_TRANSLATION_SOURCE_BYTES),
            source("d", 100),
        ];
        assert_eq!(translation_work_lanes(&sources), (vec![1, 2], vec![3, 0]));
    }

    #[test]
    fn hash_file_accepts_audit_manifests_and_deduplicates_sources() {
        let scratch = ScratchDir::new("corpus-triage-hash-file").unwrap();
        let path = scratch.path().join("sources.txt");
        fs::write(
            &path,
            format!(
                "{} Fragment first\n\n{} Kernel second\n{} duplicate\n",
                "11".repeat(32),
                "22".repeat(32),
                "11".repeat(32)
            ),
        )
        .unwrap();
        assert_eq!(
            read_hash_file(&path).unwrap(),
            vec!["11".repeat(32), "22".repeat(32)]
        );
    }

    #[test]
    fn hash_file_rejects_noncanonical_hashes() {
        let scratch = ScratchDir::new("corpus-triage-bad-hash-file").unwrap();
        let path = scratch.path().join("sources.txt");
        fs::write(&path, format!("{}\n", "AA".repeat(32))).unwrap();
        let error = read_hash_file(&path).unwrap_err();
        assert!(error.contains("first field must be a lowercase SHA-256"));
    }

    #[test]
    fn timeout_shape_excludes_bounded_phase_diagnostics() {
        assert_eq!(
            normalize_failure_shape(
                "translation timeout after 30 seconds: [retry-debug] passes: cfg repair start"
            ),
            "translation timeout after # seconds"
        );
    }
}
