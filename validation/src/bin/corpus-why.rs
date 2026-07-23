//! Diagnose a single corpus / ledger `air_sha256`: locate the source, try translate, print why.
//!
//! ```text
//! cargo run -p metal2vulkan-validation --release --bin corpus-why -- <air_sha256>
//! cargo run -p metal2vulkan-validation --release --bin corpus-why -- \
//!   e5d28c29d84f7bfed83ddeca7c34e8dbe226beccd9666246caef04ad88c88487
//! ```
//!
//! Looks up the hash in `corpus/metal2vulkan-ledger.jsonl` (if present), finds the matching
//! `.ll` under public fixtures or a private JSONL shard, runs stage auto + native translate, and
//! prints the success summary or the **actual** translator error (mint/remint only bank
//! `status=fallback` without the message).

use metal2vulkan_validation::corpus_shards::{self, SourceData};
use metal2vulkan_validation::translate_ledger::{self, TranslateFailureKind, TranslateLedgerRow};
use std::path::{Path, PathBuf};
use std::time::Instant;

const PROGRAM: &str = "corpus-why";

fn main() {
    let mut want: Option<String> = None;
    let mut ledger_path: Option<PathBuf> = None;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-h" | "--help" => {
                print_usage();
                return;
            }
            "--ledger" => {
                let p = args
                    .next()
                    .unwrap_or_else(|| fatal("--ledger requires a path"));
                ledger_path = Some(PathBuf::from(p));
            }
            other if other.starts_with("--ledger=") => {
                ledger_path = Some(PathBuf::from(other.trim_start_matches("--ledger=")));
            }
            other if other.starts_with('-') => fatal(&format!("unknown flag: {other}")),
            other => {
                if want.is_some() {
                    fatal("only one air_sha256 argument is accepted");
                }
                want = Some(normalize_hash(other));
            }
        }
    }

    let want = want.unwrap_or_else(|| {
        print_usage();
        fatal("missing air_sha256");
    });
    if want.len() != 64 || !want.chars().all(|c| c.is_ascii_hexdigit()) {
        fatal(&format!(
            "air_sha256 must be 64 lowercase hex chars, got len={} {want:?}",
            want.len()
        ));
    }

    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let public_dir = manifest_dir.join("fixtures/public");
    let local_corpus = corpus_shards::corpus_root_from_env_or_manifest();
    let ledger =
        ledger_path.unwrap_or_else(|| manifest_dir.join("corpus/metal2vulkan-ledger.jsonl"));

    eprintln!("# {PROGRAM}");
    eprintln!("# air_sha256  {want}");
    eprintln!("# ledger      {}", ledger.display());
    eprintln!("# public      {}", public_dir.display());
    eprintln!("# local       {}", local_corpus.display());

    let row = translate_ledger::load_ledger_row(&ledger, &want);
    if let Some(ref r) = row {
        eprintln!(
            "# ledger row  status={} kind={} stage={}",
            r.status, r.kind, r.stage
        );
        if !r.label.is_empty() {
            eprintln!("#             label={}", r.label);
        }
        if let Some(ref spv) = r.spv_sha256 {
            eprintln!("#             spv_sha256={spv}");
        }
    } else {
        eprintln!("# ledger row  (not present)");
    }

    let t0 = Instant::now();
    let src = match resolve_source(&want, row.as_ref(), &public_dir, &local_corpus) {
        Ok(Some(s)) => s,
        Ok(None) => {
            eprintln!(
                "# RESULT: cannot locate source ({:.2}s)",
                t0.elapsed().as_secs_f64()
            );
            eprintln!("why: no public fixture or shard row hashes to {want}");
            std::process::exit(2);
        }
        Err(why) => {
            eprintln!(
                "# RESULT: cannot locate source ({:.2}s)",
                t0.elapsed().as_secs_f64()
            );
            eprintln!("why: {why}");
            std::process::exit(2);
        }
    };
    if let Some(path) = src.public_path.as_ref() {
        eprintln!("# source      {}", path.display());
    } else if let Some(shard) = src.shard.as_deref() {
        eprintln!("# source      {shard}");
    } else {
        eprintln!("# source      {}", src.label);
    }
    eprintln!("# label       {}", src.label);
    eprintln!("# kind        {}", src.kind);
    if let Some(shard) = src.shard.as_deref() {
        eprintln!("# shard       {shard}");
    }

    // Verify content hash matches the requested air_sha256.
    let got = src.air_sha256.clone();
    if got != want {
        eprintln!(
            "# RESULT: source hash mismatch ({:.2}s)",
            t0.elapsed().as_secs_f64()
        );
        eprintln!("why: source {} hashes to {got}, expected {want}", src.label);
        std::process::exit(2);
    }

    eprintln!("# translate   stage=auto (detect + native emit)...");
    let report = translate_ledger::translate_source_data(&src, "m2v-why");
    if report.stage_name != "auto" {
        eprintln!("# stage       {} (from !air.* metadata)", report.stage_name);
    }
    if report.status == "ok" {
        let spv_hash = report.spv_sha256.as_deref().unwrap_or("");
        eprintln!(
            "# RESULT: ok  spv_bytes={} spv_sha256={spv_hash} ({:.2}s)",
            report.spv_len.unwrap_or(0),
            t0.elapsed().as_secs_f64()
        );
        if let Some(ref r) = row {
            if r.status != "ok" {
                eprintln!(
                    "note: ledger had status={} - translator now succeeds; remint would update",
                    r.status
                );
            } else if let Some(ref old) = r.spv_sha256 {
                if old != spv_hash {
                    eprintln!(
                        "note: ledger spv_sha256 differs (would be DRIFT on a check):\n\
                         \tledger {old}\n\
                         \tgot    {spv_hash}"
                    );
                } else {
                    eprintln!("note: matches banked ledger spv_sha256");
                }
            }
        }
        std::process::exit(0);
    }

    let failure = report
        .failure
        .as_ref()
        .expect("fallback report should carry a failure");
    match failure.kind {
        TranslateFailureKind::TempDir => {
            eprintln!("# RESULT: tmp failed ({:.2}s)", t0.elapsed().as_secs_f64());
            eprintln!("why: {}", failure.message);
            std::process::exit(2);
        }
        TranslateFailureKind::StageDetect => {
            eprintln!(
                "# RESULT: FALLBACK at stage detect ({:.2}s)",
                t0.elapsed().as_secs_f64()
            );
            eprintln!("why: {}", failure.message);
            std::process::exit(1);
        }
        TranslateFailureKind::EmptySpirv => {
            eprintln!(
                "# RESULT: FALLBACK empty SPIR-V ({:.2}s)",
                t0.elapsed().as_secs_f64()
            );
            eprintln!("why: {}", failure.message);
            std::process::exit(1);
        }
        TranslateFailureKind::Translate | TranslateFailureKind::LoadSource => {
            eprintln!(
                "# RESULT: FALLBACK at translate ({:.2}s)",
                t0.elapsed().as_secs_f64()
            );
            eprintln!("why: {}", failure.message);
            std::process::exit(1);
        }
    }
}

fn print_usage() {
    eprintln!(
        "usage: {PROGRAM} <air_sha256> [--ledger PATH]\n\
         \n\
         Locate the AIR/LL for the given content hash under public fixtures or\n\
         local JSONL shards, run metal2vulkan (stage auto), and print the failure reason\n\
         (or ok + spv_sha256). Does not modify the ledger."
    );
}

fn fatal(msg: &str) -> ! {
    eprintln!("{PROGRAM}: {msg}");
    std::process::exit(64);
}

fn normalize_hash(s: &str) -> String {
    s.trim().trim_start_matches("0x").to_ascii_lowercase()
}

/// Resolve source for `want` hash: try the ledger's public label or shard hint first.
fn resolve_source(
    want: &str,
    row: Option<&TranslateLedgerRow>,
    public_dir: &Path,
    local_corpus: &Path,
) -> Result<Option<SourceData>, String> {
    corpus_shards::resolve_source(
        want,
        row.map(|r| r.label.as_str()).unwrap_or(""),
        row.map(|r| r.kind.as_str()).unwrap_or(""),
        row.and_then(|r| r.shard.as_deref()),
        public_dir,
        local_corpus,
    )
}
