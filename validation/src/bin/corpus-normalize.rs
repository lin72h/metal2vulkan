use metal2vulkan_validation::normalize::normalize_air_identities;
use metal2vulkan_validation::source::corpus_root;
use std::path::PathBuf;

fn main() {
    if let Err(error) = run() {
        eprintln!("corpus-normalize: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let mut root = corpus_root();
    let mut apply = false;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--corpus" => {
                root = PathBuf::from(
                    args.next()
                        .ok_or_else(|| "--corpus requires a directory".to_string())?,
                );
            }
            "--apply" => apply = true,
            "-h" | "--help" => {
                println!("usage: corpus-normalize [--corpus DIR] --apply");
                return Ok(());
            }
            _ => return Err(format!("unknown argument {arg:?}")),
        }
    }
    if !apply {
        return Err("refusing to rewrite the corpus without explicit --apply".into());
    }
    let stats = normalize_air_identities(&root)?;
    println!(
        "normalized sources={}->{} modules={}->{} cases_rewritten={} cases_deduplicated={} reviews_superseded={} observations_invalidated={} observations_deduplicated={}",
        stats.source_rows_before,
        stats.source_rows_after,
        stats.module_rows_before,
        stats.module_rows_after,
        stats.cases_rewritten,
        stats.cases_deduplicated,
        stats.reviews_superseded,
        stats.observations_invalidated,
        stats.observations_deduplicated,
    );
    Ok(())
}
