//! Re-mint metal2vulkan-ledger rows (full ledger by default).
//!
//! Reads `validation/corpus/metal2vulkan-ledger.jsonl`, re-translates matching sources
//! under public fixtures + local shards (stage auto), and **updates** those rows in
//! place (full ledger rewrite; rows without a public fixture or shard source are left alone).
//!
//! ```text
//! # remint every banked hash that is still on disk
//! cargo run -p metal2vulkan-validation --release --bin corpus-remint
//!
//! # only status != ok
//! cargo run -p metal2vulkan-validation --release --bin corpus-remint -- --failed-only
//! ```
//!
//! Optional: `--ledger PATH`, `--jobs N`, `--quiet`, `--dry-run`.

use metal2vulkan_validation::corpus_shards::{self, SourceRef};
use metal2vulkan_validation::translate_ledger::{self, TranslateLedgerRow};
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

const PROGRAM: &str = "corpus-remint";

struct RemintOptions {
    ledger: PathBuf,
    public_dir: PathBuf,
    local_corpus: PathBuf,
    jobs: usize,
    quiet: bool,
    dry_run: bool,
    /// When true, only rows with `status != "ok"`.
    failed_only: bool,
}

fn main() {
    let Some(opts) = parse_args() else {
        return;
    };
    let code = run(opts);
    std::process::exit(code);
}

fn parse_args() -> Option<RemintOptions> {
    let mut ledger_path: Option<PathBuf> = None;
    let mut jobs: Option<usize> = None;
    let mut quiet = false;
    let mut dry_run = false;
    let mut failed_only = false;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-h" | "--help" => {
                print_usage();
                return None;
            }
            "--quiet" => quiet = true,
            "--dry-run" => dry_run = true,
            "--failed-only" => failed_only = true,
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
    Some(RemintOptions {
        ledger: ledger_path
            .unwrap_or_else(|| manifest_dir.join("corpus/metal2vulkan-ledger.jsonl")),
        public_dir: manifest_dir.join("fixtures/public"),
        local_corpus: corpus_shards::corpus_root_from_env_or_manifest(),
        jobs: jobs
            .unwrap_or_else(translate_ledger::default_workers)
            .max(1),
        quiet,
        dry_run,
        failed_only,
    })
}

fn print_usage() {
    eprintln!(
        "usage: {PROGRAM} [--ledger PATH] [--jobs N] [--quiet] [--dry-run] [--failed-only]\n\
         \n\
         Re-translates metal2vulkan-ledger rows whose AIR is present under public fixtures\n\
         or local JSONL shards, and rewrites those rows in the ledger.\n\
         Default: remint every ledger row with a public fixture or shard source.\n\
         --failed-only  only remint rows with status != ok\n\
         --dry-run      list remintable rows only; do not translate or write."
    );
}

fn run(opts: RemintOptions) -> i32 {
    eprintln!("# {PROGRAM}");
    eprintln!("# ledger  {}", opts.ledger.display());
    eprintln!("# public  {}", opts.public_dir.display());
    eprintln!("# local   {}", opts.local_corpus.display());
    eprintln!("# jobs    {}", opts.jobs);
    eprintln!(
        "# mode    {}",
        if opts.failed_only {
            "failed-only (status != ok)"
        } else {
            "all ledger rows with source in public fixtures or shards"
        }
    );
    if opts.dry_run {
        eprintln!("# dry-run (no translate, no ledger write)");
    }

    let t0 = Instant::now();
    let (mut by_hash, n_dup) =
        translate_ledger::load_ledger(&opts.ledger).unwrap_or_else(|e| fatal(&e));
    if by_hash.is_empty() {
        fatal(&format!("no ledger rows at {}", opts.ledger.display()));
    }
    if n_dup > 0 {
        eprintln!("# load: collapsed {n_dup} duplicate air_sha256 row(s)");
    }
    eprintln!("# ledger unique rows: {}", by_hash.len());

    let mut candidates: Vec<TranslateLedgerRow> = by_hash
        .values()
        .filter(|r| !opts.failed_only || r.status != "ok")
        .cloned()
        .collect();
    candidates.sort_by(|a, b| {
        a.label
            .cmp(&b.label)
            .then_with(|| a.air_sha256.cmp(&b.air_sha256))
    });

    if opts.failed_only {
        let mut status_hist: HashMap<String, usize> = HashMap::new();
        for r in &candidates {
            *status_hist.entry(r.status.clone()).or_default() += 1;
        }
        let hist = status_hist
            .iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect::<Vec<_>>()
            .join(" ");
        eprintln!(
            "# non-ok rows: {}{}",
            candidates.len(),
            if hist.is_empty() {
                String::new()
            } else {
                format!(" ({hist})")
            }
        );
    } else {
        eprintln!("# candidate rows: {}", candidates.len());
    }

    if candidates.is_empty() {
        eprintln!("# nothing to remint ({:.1}s)", t0.elapsed().as_secs_f64());
        return 0;
    }

    eprintln!("# indexing sources…");
    let sources = corpus_shards::gather_source_refs(&opts.public_dir, &opts.local_corpus);
    let by_content = translate_ledger::unique_sources_by_hash(sources);
    eprintln!(
        "# unique source hashes in fixtures/shards: {}",
        by_content.len()
    );

    let mut todo: Vec<SourceRef> = Vec::new();
    let mut missing = 0usize;
    for row in &candidates {
        match by_content.get(&row.air_sha256) {
            Some(src) => {
                todo.push(SourceRef {
                    air_sha256: row.air_sha256.clone(),
                    shard: row.shard.clone().or_else(|| src.shard.clone()),
                    label: if row.label.is_empty() {
                        src.label.clone()
                    } else {
                        row.label.clone()
                    },
                    kind: if row.kind.is_empty() {
                        src.kind.clone()
                    } else {
                        row.kind.clone()
                    },
                    storage: src.storage.clone(),
                });
            }
            None => {
                missing += 1;
                if !opts.quiet {
                    eprintln!(
                        "  missing   {}  air={} (source not on disk; row kept)",
                        row.label,
                        &row.air_sha256[..12.min(row.air_sha256.len())]
                    );
                }
            }
        }
    }
    todo.sort_by(|a, b| {
        a.label
            .cmp(&b.label)
            .then_with(|| a.air_sha256.cmp(&b.air_sha256))
    });
    eprintln!("# remintable: {}  missing_source: {missing}", todo.len());

    if todo.is_empty() {
        eprintln!(
            "# no selected rows have sources on disk ({:.1}s)",
            t0.elapsed().as_secs_f64()
        );
        return 0;
    }

    if opts.dry_run {
        eprintln!(
            "# RESULT: dry-run would remint {} row(s) ({:.1}s)",
            todo.len(),
            t0.elapsed().as_secs_f64()
        );
        return 0;
    }

    let updated = translate_ledger::translate_all(
        &todo,
        &opts.local_corpus,
        opts.jobs,
        opts.quiet,
        "remint",
        "m2v-remint",
    );
    let n_ok = updated.iter().filter(|r| r.status == "ok").count();
    let n_still = updated.len() - n_ok;
    for row in &updated {
        by_hash.insert(row.air_sha256.clone(), row.clone());
    }
    translate_ledger::write_ledger(&opts.ledger, &by_hash).unwrap_or_else(|e| fatal(&e));
    eprintln!(
        "# RESULT: reminted {} (now_ok={n_ok} still_failed={n_still} missing_source={missing}) → {}  ({:.1}s)",
        updated.len(),
        opts.ledger.display(),
        t0.elapsed().as_secs_f64()
    );
    0
}

fn fatal(msg: &str) -> ! {
    eprintln!("{PROGRAM}: {msg}");
    std::process::exit(64);
}
