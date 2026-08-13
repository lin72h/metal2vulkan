use metal2vulkan_validation::ab::{read_hash_list, run, AbOptions, AbSelection};
use metal2vulkan_validation::source::corpus_root;
use std::collections::HashSet;
use std::path::PathBuf;
use std::time::Duration;

fn main() {
    match run_command() {
        Ok(true) => {}
        Ok(false) => std::process::exit(1),
        Err(error) => {
            eprintln!("corpus-ab: {error}");
            std::process::exit(2);
        }
    }
}

fn run_command() -> Result<bool, String> {
    let mut root = corpus_root();
    let mut old = None;
    let mut new = None;
    let mut selection = AbSelection::default();
    let mut air_lists = Vec::new();
    let mut translator_options = Vec::new();
    let mut cache_dir = PathBuf::from(".cache/validation-corpus-ab");
    let mut timeout = 60u64;
    let mut expect_no_change = false;
    let mut fail_on_unlisted_change = false;
    let mut spv_allowlist = HashSet::new();
    let mut fallback_allowlist = HashSet::new();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--corpus" => root = PathBuf::from(required(&mut args, "--corpus")?),
            "--old" => old = Some(PathBuf::from(required(&mut args, "--old")?)),
            "--new" => new = Some(PathBuf::from(required(&mut args, "--new")?)),
            "--air-sha256" => selection
                .air_sha256
                .push(required(&mut args, "--air-sha256")?),
            "--air-list" => air_lists.push(PathBuf::from(required(&mut args, "--air-list")?)),
            "--shard" => {
                let shard: usize = required(&mut args, "--shard")?
                    .parse()
                    .map_err(|error| format!("invalid --shard: {error}"))?;
                if shard >= 64 {
                    return Err("--shard must be in 0..64".into());
                }
                selection.shards.push(shard);
            }
            "--canary" => selection.canary = true,
            "--expect-no-change" => expect_no_change = true,
            "--fail-on-unlisted-change" => fail_on_unlisted_change = true,
            "--allow-spv-change" => {
                spv_allowlist.extend(read_hash_list(&PathBuf::from(required(
                    &mut args,
                    "--allow-spv-change",
                )?))?);
            }
            "--allow-fallback-to-success" => {
                fallback_allowlist.extend(read_hash_list(&PathBuf::from(required(
                    &mut args,
                    "--allow-fallback-to-success",
                )?))?);
            }
            "--translator-option" => {
                translator_options.push(required(&mut args, "--translator-option")?)
            }
            "--cache" => cache_dir = PathBuf::from(required(&mut args, "--cache")?),
            "--timeout-secs" => {
                timeout = required(&mut args, "--timeout-secs")?
                    .parse()
                    .map_err(|error| format!("invalid --timeout-secs: {error}"))?;
            }
            "-h" | "--help" => {
                print_help();
                return Ok(true);
            }
            _ => return Err(format!("unknown argument {arg:?}")),
        }
    }
    for path in air_lists {
        selection.air_sha256.extend(read_hash_list(&path)?);
    }
    let options = AbOptions {
        corpus_root: root,
        old_binary: old.ok_or("--old is required")?,
        new_binary: new.ok_or("--new is required")?,
        selection,
        translator_options,
        cache_dir,
        timeout: Duration::from_secs(timeout),
        expect_no_change,
        fail_on_unlisted_change,
        spv_allowlist,
        fallback_to_success_allowlist: fallback_allowlist,
    };
    let results = run(&options)?;
    let mut clean = true;
    for result in &results {
        println!(
            "{}\t{}\t{}\t{}",
            result.classification.label(),
            if result.allowed { "allowed" } else { "FAIL" },
            result.air_sha256,
            result.label
        );
        clean &= result.allowed;
    }
    println!(
        "summary\ttotal={}\tfailed={}",
        results.len(),
        results.iter().filter(|result| !result.allowed).count()
    );
    Ok(clean)
}

fn required(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, String> {
    args.next()
        .ok_or_else(|| format!("{flag} requires a value"))
}

fn print_help() {
    println!(
        "usage: corpus-ab --old PATH --new PATH SELECTION [POLICY]\n\
         selection: --air-sha256 HASH | --air-list PATH | --shard ID | --canary\n\
         policy: --expect-no-change --allow-spv-change PATH\n\
                 --allow-fallback-to-success PATH --fail-on-unlisted-change\n\
         execution: --translator-option ARG --cache DIR --timeout-secs N"
    );
}
