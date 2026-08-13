use metal2vulkan_validation::index::{
    default_index_path, evidence_status, select_queue, status_counts, EvidenceState, QueueState,
};
use metal2vulkan_validation::source::corpus_root;
use std::path::PathBuf;

fn main() {
    if let Err(error) = run() {
        eprintln!("corpus-status: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let mut root = corpus_root();
    let mut index = None;
    let mut limit = 20usize;
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
            "-h" | "--help" => {
                println!("usage: corpus-status [--corpus DIR] [--index PATH] [--limit N]");
                return Ok(());
            }
            _ => return Err(format!("unknown argument {arg:?}")),
        }
    }
    let index = index.unwrap_or_else(|| default_index_path(&root));
    for (state, count) in status_counts(&index)? {
        println!("{state}\t{count}");
    }
    for row in select_queue(&index, QueueState::Authored, limit)? {
        println!(
            "action\t{}\t{}\t{}\t{}\t{}",
            row.state.as_str(),
            row.air_sha256,
            row.entry,
            row.label,
            row.review_reason
                .as_deref()
                .map(single_line)
                .unwrap_or_default()
        );
    }
    for row in evidence_status(&root)?.into_iter().take(limit) {
        if row.metal != EvidenceState::Current
            || row.moltenvk != EvidenceState::Current
            || row.vulkan != EvidenceState::Current
        {
            println!(
                "evidence\t{}\tmetal={}\tmoltenvk={}\tvulkan={}\t{}",
                row.case_id,
                row.metal.as_str(),
                row.moltenvk.as_str(),
                row.vulkan.as_str(),
                row.translation_error.as_deref().unwrap_or(&row.name)
            );
            for slot in row.slots {
                println!(
                    "slot\t{}\t{}\t{}\t{}",
                    row.case_id,
                    slot.backend,
                    slot.environment_id,
                    slot.state.as_str()
                );
            }
        }
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
