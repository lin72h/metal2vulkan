use metal2vulkan_validation::index::{
    default_index_path, pending_macos_refresh_case_ids, sync_index_with_stats,
};
use metal2vulkan_validation::observation::Backend;
use metal2vulkan_validation::source::corpus_root;
use metal2vulkan_validation::store::CorpusStore;
use std::collections::HashSet;
use std::path::PathBuf;
use std::time::Instant;

fn main() {
    if let Err(error) = run() {
        eprintln!("corpus-refresh: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    if !cfg!(target_os = "macos") {
        return Err("batched Metal qualification and MoltenVK comparison requires macOS".into());
    }
    let mut root = corpus_root();
    let mut case_ids = Vec::new();
    let mut metal_environment_id = None;
    let mut environment_id = None;
    let mut force_all = false;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--corpus" => root = PathBuf::from(required(&mut args, "--corpus")?),
            "--case-id" => case_ids.push(required(&mut args, "--case-id")?),
            "--metal-environment-id" => {
                metal_environment_id = Some(required(&mut args, "--metal-environment-id")?)
            }
            "--environment-id" => environment_id = Some(required(&mut args, "--environment-id")?),
            "--all" => force_all = true,
            "-h" | "--help" => {
                println!(
                    "usage: corpus-refresh [--corpus DIR] [--case-id HASH ... | --all] \\\n\
                     --metal-environment-id ID --environment-id ID\n\
                     With neither selector, refreshes only stale or missing exact environment slots."
                );
                return Ok(());
            }
            _ => return Err(format!("unknown argument {arg:?}")),
        }
    }
    let metal_environment_id =
        metal_environment_id.ok_or_else(|| "--metal-environment-id is required".to_string())?;
    let environment_id =
        environment_id.ok_or_else(|| "--environment-id is required".to_string())?;
    let requested = case_ids.iter().cloned().collect::<HashSet<_>>();
    if requested.len() != case_ids.len() {
        return Err("duplicate --case-id".into());
    }
    if force_all && !requested.is_empty() {
        return Err("--all and --case-id are mutually exclusive".into());
    }
    let index = default_index_path(&root);
    let index_started = Instant::now();
    let initial_sync = sync_index_with_stats(&root, &index)?;
    eprintln!(
        "# index sync: {:.3}s, source_shards_scanned={}, source_bytes_scanned={}, rebuilt={}",
        index_started.elapsed().as_secs_f64(),
        initial_sync.source_shards_scanned,
        initial_sync.source_bytes_scanned,
        initial_sync.rebuilt
    );
    let pending = if requested.is_empty() && !force_all {
        Some(pending_macos_refresh_case_ids(
            &index,
            &metal_environment_id,
            &environment_id,
        )?)
    } else {
        None
    };
    let all_cases = CorpusStore::new(&root).read_all_cases()?;
    let cases = if let Some(pending) = &pending {
        all_cases
            .into_iter()
            .filter(|case| pending.contains(&case.case_id))
            .collect::<Vec<_>>()
    } else if requested.is_empty() {
        all_cases
    } else {
        let selected = all_cases
            .into_iter()
            .filter(|case| requested.contains(&case.case_id))
            .collect::<Vec<_>>();
        let found = selected
            .iter()
            .map(|case| case.case_id.as_str())
            .collect::<HashSet<_>>();
        let mut missing = requested
            .iter()
            .filter(|case_id| !found.contains(case_id.as_str()))
            .cloned()
            .collect::<Vec<_>>();
        missing.sort();
        if !missing.is_empty() {
            return Err(format!("unknown case IDs: {}", missing.join(", ")));
        }
        selected
    };
    eprintln!("# cases selected: {}", cases.len());
    let mut failures = Vec::new();
    for case in cases {
        let case_id = case.case_id.clone();
        let metal_started = Instant::now();
        let metal = match metal2vulkan_validation::metal::qualify_case(
            &root,
            case.clone(),
            &metal_environment_id,
        ) {
            Ok(metal) => metal,
            Err(error) => {
                eprintln!("case {case_id}: metal failed: {error}");
                failures.push(format!("case {case_id}: Metal: {error}"));
                continue;
            }
        };
        let metal_elapsed = metal_started.elapsed();
        let candidate_started = Instant::now();
        let candidate = match metal2vulkan_validation::candidate::execute_case(
            &root,
            case,
            Backend::Moltenvk,
            &metal_environment_id,
            &environment_id,
        ) {
            Ok(candidate) => candidate,
            Err(error) => {
                eprintln!("case {case_id}: MoltenVK failed: {error}");
                failures.push(format!("case {case_id}: MoltenVK: {error}"));
                continue;
            }
        };
        let candidate_elapsed = candidate_started.elapsed();
        println!(
            "{}\tmetal={}\tmoltenvk={:?}\tmetal_seconds={:.3}\tmoltenvk_seconds={:.3}",
            case_id,
            metal.metal_output_sha256,
            candidate.status,
            metal_elapsed.as_secs_f64(),
            candidate_elapsed.as_secs_f64()
        );
    }
    let final_sync_started = Instant::now();
    let final_sync = sync_index_with_stats(&root, &index)?;
    eprintln!(
        "# final index sync: {:.3}s, source_shards_scanned={}, source_bytes_scanned={}",
        final_sync_started.elapsed().as_secs_f64(),
        final_sync.source_shards_scanned,
        final_sync.source_bytes_scanned
    );
    if !failures.is_empty() {
        return Err(format!(
            "{} case execution(s) failed; first: {}",
            failures.len(),
            failures[0]
        ));
    }
    Ok(())
}

fn required(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, String> {
    args.next()
        .ok_or_else(|| format!("{flag} requires a value"))
}
