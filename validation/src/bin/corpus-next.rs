use metal2vulkan_validation::index::{default_index_path, select_queue, sync_index, QueueState};
use metal2vulkan_validation::review::ReviewNote;
use metal2vulkan_validation::source::corpus_root;
use metal2vulkan_validation::store::CorpusStore;
use std::path::PathBuf;

fn main() {
    if let Err(error) = run() {
        eprintln!("corpus-next: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let mut root = corpus_root();
    let mut index = None;
    let mut limit = 1usize;
    let mut review_air = None;
    let mut reason = None;
    let mut reviewed_by = None;
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
            "--review-air" => review_air = Some(required(&mut args, "--review-air")?),
            "--reason" => reason = Some(required(&mut args, "--reason")?),
            "--reviewed-by" => reviewed_by = Some(required(&mut args, "--reviewed-by")?),
            "-h" | "--help" => {
                println!(
                    "usage: corpus-next [--corpus DIR] [--index PATH] [--limit N]\n       corpus-next [--corpus DIR] [--index PATH] --review-air HASH --reason TEXT --reviewed-by ID"
                );
                return Ok(());
            }
            _ => return Err(format!("unknown argument {arg:?}")),
        }
    }
    let index = index.unwrap_or_else(|| default_index_path(&root));
    match (review_air, reason, reviewed_by) {
        (Some(air_sha256), Some(reason), Some(reviewed_by)) => {
            CorpusStore::new(&root).upsert_review(ReviewNote {
                air_sha256: air_sha256.clone(),
                reason,
                reviewed_by,
            })?;
            sync_index(&root, &index)?;
            println!("reviewed\t{air_sha256}");
            return Ok(());
        }
        (None, None, None) => {}
        _ => {
            return Err(
                "--review-air, --reason, and --reviewed-by must be supplied together".into(),
            )
        }
    }
    if limit == 0 {
        return Err("--limit must be greater than zero".into());
    }
    for row in select_queue(&index, QueueState::Unplanned, limit)? {
        println!(
            "{}\t{}\t{}\t{}\t{}\t{}",
            row.air_sha256,
            row.stage,
            row.entry,
            row.label,
            row.state.as_str(),
            row.review_reason
                .as_deref()
                .map(single_line)
                .unwrap_or_default()
        );
    }
    Ok(())
}

fn single_line(value: &str) -> String {
    value.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn required(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, String> {
    args.next()
        .ok_or_else(|| format!("{flag} requires a value"))
}
