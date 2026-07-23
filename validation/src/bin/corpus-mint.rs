//! Additive AIR→SPIR-V hash mint for the committed metal2vulkan ledger.
//!
//! Always scans **both**:
//! - `validation/fixtures/public/**/*.{ll,air}`
//! - `validation/corpus/local/shards/shard_NN.jsonl`
//!
//! Builds a unique set of source content hashes (`air_sha256`). For each hash that
//! is **not** already in `validation/corpus/metal2vulkan-ledger.jsonl`, runs the translator
//! (stage auto) and **adds** a ledger row. Existing rows are never re-translated or
//! overwritten.
//!
//! Usage (from repo root):
//! ```text
//! cargo run -p metal2vulkan-validation --release --bin corpus-mint
//! cargo run -p metal2vulkan-validation --release --bin corpus-mint -- --jobs 8
//! ```
//!
//! Optional: `--ledger PATH`, `--jobs N` (default: CPU cores × 2), `--quiet`,
//! `--dry-run` (scan/hash/filter only; do not translate or write).

use metal2vulkan_validation::corpus_shards::{self, SourceRef};
use metal2vulkan_validation::translate_ledger;
use std::path::PathBuf;
use std::time::Instant;

fn main() {
    let mut ledger_path: Option<PathBuf> = None;
    let mut jobs: Option<usize> = None;
    let mut quiet = false;
    let mut dry_run = false;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-h" | "--help" => {
                print_usage();
                return;
            }
            "--quiet" => quiet = true,
            "--dry-run" => dry_run = true,
            "--ledger" => {
                let p = args
                    .next()
                    .unwrap_or_else(|| fatal("--ledger requires a path"));
                ledger_path = Some(PathBuf::from(p));
            }
            "--jobs" => {
                let n = args
                    .next()
                    .unwrap_or_else(|| fatal("--jobs requires a number"));
                jobs = Some(
                    n.parse::<usize>()
                        .unwrap_or_else(|_| fatal(&format!("bad --jobs {n}"))),
                );
            }
            other if other.starts_with("--ledger=") => {
                ledger_path = Some(PathBuf::from(other.trim_start_matches("--ledger=")));
            }
            other if other.starts_with("--jobs=") => {
                let n = other.trim_start_matches("--jobs=");
                jobs = Some(
                    n.parse::<usize>()
                        .unwrap_or_else(|_| fatal(&format!("bad --jobs {n}"))),
                );
            }
            other => fatal(&format!("unknown arg: {other}")),
        }
    }

    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    // validation/ is CARGO_MANIFEST_DIR; public fixtures live under validation/fixtures/public
    // in this tree layout (see AGENTS.md / corpus README).
    let public_dir = manifest_dir.join("fixtures/public");
    let local_corpus = corpus_shards::corpus_root_from_env_or_manifest();
    let ledger =
        ledger_path.unwrap_or_else(|| manifest_dir.join("corpus/metal2vulkan-ledger.jsonl"));

    let workers = jobs
        .unwrap_or_else(translate_ledger::default_workers)
        .max(1);

    eprintln!("# corpus-mint");
    eprintln!("# ledger  {}", ledger.display());
    eprintln!("# public  {}", public_dir.display());
    eprintln!("# local   {}", local_corpus.display());
    eprintln!("# jobs    {workers}");
    if dry_run {
        eprintln!("# dry-run (no translate, no ledger write)");
    }

    let t0 = Instant::now();
    let existing = translate_ledger::load_ledger_keys(&ledger).unwrap_or_else(|e| fatal(&e));
    eprintln!("# ledger rows (unique air_sha256): {}", existing.len());

    let sources = corpus_shards::gather_source_refs(&public_dir, &local_corpus);
    eprintln!(
        "# source refs (public fixtures + shard rows): {}",
        sources.len()
    );

    // Unique by content hash; prefer stable label (lexicographically smaller).
    let by_hash = translate_ledger::unique_sources_by_hash(sources);
    let unique = by_hash.len();
    eprintln!("# unique air_sha256 among sources: {unique}");

    let mut todo: Vec<SourceRef> = by_hash
        .into_values()
        .filter(|s| !existing.contains(&s.air_sha256))
        .collect();
    todo.sort_by(|a, b| {
        a.label
            .cmp(&b.label)
            .then_with(|| a.air_sha256.cmp(&b.air_sha256))
    });

    let already = unique - todo.len();
    eprintln!("# already in ledger: {already}  to mint: {}", todo.len());

    if todo.is_empty() {
        eprintln!("# nothing new to mint ({:.1}s)", t0.elapsed().as_secs_f64());
        return;
    }

    if dry_run {
        eprintln!(
            "# RESULT: dry-run would mint {} new hash(es) ({:.1}s)",
            todo.len(),
            t0.elapsed().as_secs_f64()
        );
        return;
    }

    let new_rows =
        translate_ledger::translate_all(&todo, &local_corpus, workers, quiet, "mint", "m2v-mint");
    let n_ok = new_rows.iter().filter(|r| r.status == "ok").count();
    let n_fb = new_rows.len() - n_ok;
    translate_ledger::append_ledger_rows(&ledger, &new_rows).unwrap_or_else(|e| fatal(&e));
    eprintln!(
        "# RESULT: added {} row(s) (ok={n_ok} fallback={n_fb}) → {}  ({:.1}s)",
        new_rows.len(),
        ledger.display(),
        t0.elapsed().as_secs_f64()
    );
}

fn print_usage() {
    eprintln!(
        "usage: corpus-mint [--ledger PATH] [--jobs N] [--quiet] [--dry-run]\n\
         \n\
         Scans public fixtures + local JSONL shards, mints metal2vulkan-ledger rows only for\n\
         air_sha256 values not already present (never rewrites existing rows).\n\
         --dry-run  hash + filter only; do not translate or write the ledger."
    );
}

fn fatal(msg: &str) -> ! {
    eprintln!("corpus-mint: {msg}");
    std::process::exit(64);
}
