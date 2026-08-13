use metal2vulkan_validation::index::{
    check_index, default_index_path, rebuild_index, status_counts, sync_index_with_stats,
};
use metal2vulkan_validation::source::corpus_root;
use std::path::PathBuf;

fn main() {
    if let Err(error) = run() {
        eprintln!("corpus-index: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let mut root = corpus_root();
    let mut destination = None;
    let mut check = false;
    let mut rebuild = false;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--corpus" => root = PathBuf::from(required(&mut args, "--corpus")?),
            "--index" => destination = Some(PathBuf::from(required(&mut args, "--index")?)),
            "--check" => check = true,
            "--rebuild" => rebuild = true,
            "-h" | "--help" => {
                println!("usage: corpus-index [--corpus DIR] [--index PATH] [--check | --rebuild]");
                return Ok(());
            }
            _ => return Err(format!("unknown argument {arg:?}")),
        }
    }
    let destination = destination.unwrap_or_else(|| default_index_path(&root));
    if check && rebuild {
        return Err("--check and --rebuild are mutually exclusive".into());
    }
    if check {
        check_index(&root, &destination)?;
        println!("checked {}", destination.display());
    } else if rebuild {
        rebuild_index(&root, &destination)?;
        println!("rebuilt {}", destination.display());
    } else {
        let stats = sync_index_with_stats(&root, &destination)?;
        println!(
            "synchronized {} (source_shards_scanned={} source_bytes_scanned={} rebuilt={})",
            destination.display(),
            stats.source_shards_scanned,
            stats.source_bytes_scanned,
            stats.rebuilt
        );
    }
    for (state, count) in status_counts(&destination)? {
        println!("{state}\t{count}");
    }
    Ok(())
}

fn required(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, String> {
    args.next()
        .ok_or_else(|| format!("{flag} requires a value"))
}
