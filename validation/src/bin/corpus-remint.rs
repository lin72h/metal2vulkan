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
//! Optional: `--ledger PATH`, `--jobs N`, `--quiet`, `--dry-run`, `--status S`,
//! `--contains TEXT`, `--skip N`, `--limit N`, `--case-timeout-secs N`.

use metal2vulkan_validation::corpus_shards::SourceStorage;
use metal2vulkan_validation::corpus_shards::{self, SourceRef};
use metal2vulkan_validation::translate_ledger::{self, TranslateLedgerRow};
use std::collections::HashMap;
use std::io::Read;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};
use wait_timeout::ChildExt;

const PROGRAM: &str = "corpus-remint";
const DEFAULT_CASE_TIMEOUT_SECS: u64 = 60;

struct RemintOptions {
    ledger: PathBuf,
    public_dir: PathBuf,
    local_corpus: PathBuf,
    jobs: usize,
    quiet: bool,
    dry_run: bool,
    /// When true, only rows with `status != "ok"`.
    failed_only: bool,
    status: Option<String>,
    contains: Option<String>,
    skip: usize,
    limit: Option<usize>,
    case_timeout_secs: u64,
}

struct TranslateOneOptions {
    local_corpus: PathBuf,
    source: SourceRef,
}

fn main() {
    let Some(mode) = parse_args() else {
        return;
    };
    let code = match mode {
        Mode::Remint(opts) => run(opts),
        Mode::TranslateOne(opts) => run_translate_one(opts),
    };
    std::process::exit(code);
}

enum Mode {
    Remint(RemintOptions),
    TranslateOne(TranslateOneOptions),
}

fn parse_args() -> Option<Mode> {
    let mut ledger_path: Option<PathBuf> = None;
    let mut jobs: Option<usize> = None;
    let mut quiet = false;
    let mut dry_run = false;
    let mut failed_only = false;
    let mut status: Option<String> = None;
    let mut contains: Option<String> = None;
    let mut skip = 0usize;
    let mut limit: Option<usize> = None;
    let mut case_timeout_secs = DEFAULT_CASE_TIMEOUT_SECS;
    let mut translate_one = false;
    let mut local_corpus: Option<PathBuf> = None;
    let mut one_air_sha256: Option<String> = None;
    let mut one_label: Option<String> = None;
    let mut one_kind: Option<String> = None;
    let mut one_shard: Option<String> = None;
    let mut one_public_path: Option<PathBuf> = None;
    let mut one_byte_offset: Option<u64> = None;
    let mut one_byte_len: Option<usize> = None;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-h" | "--help" => {
                print_usage();
                return None;
            }
            "--translate-one" => translate_one = true,
            "--quiet" => quiet = true,
            "--dry-run" => dry_run = true,
            "--failed-only" => failed_only = true,
            "--local-corpus" => {
                let p = args
                    .next()
                    .unwrap_or_else(|| fatal("--local-corpus requires a path"));
                local_corpus = Some(PathBuf::from(p));
            }
            "--air-sha256" => {
                one_air_sha256 = Some(
                    args.next()
                        .unwrap_or_else(|| fatal("--air-sha256 requires text")),
                );
            }
            "--label" => {
                one_label = Some(
                    args.next()
                        .unwrap_or_else(|| fatal("--label requires text")),
                );
            }
            "--kind" => {
                one_kind = Some(args.next().unwrap_or_else(|| fatal("--kind requires text")));
            }
            "--shard" => {
                one_shard = Some(
                    args.next()
                        .unwrap_or_else(|| fatal("--shard requires text")),
                );
            }
            "--public-path" => {
                let p = args
                    .next()
                    .unwrap_or_else(|| fatal("--public-path requires a path"));
                one_public_path = Some(PathBuf::from(p));
            }
            "--byte-offset" => {
                let n = args
                    .next()
                    .unwrap_or_else(|| fatal("--byte-offset requires a number"));
                one_byte_offset = Some(
                    n.parse::<u64>()
                        .unwrap_or_else(|_| fatal(&format!("bad --byte-offset {n}"))),
                );
            }
            "--byte-len" => {
                let n = args
                    .next()
                    .unwrap_or_else(|| fatal("--byte-len requires a number"));
                one_byte_len = Some(
                    n.parse::<usize>()
                        .unwrap_or_else(|_| fatal(&format!("bad --byte-len {n}"))),
                );
            }
            "--status" => {
                status = Some(
                    args.next()
                        .unwrap_or_else(|| fatal("--status requires text")),
                );
            }
            "--contains" => {
                contains = Some(
                    args.next()
                        .unwrap_or_else(|| fatal("--contains requires text")),
                );
            }
            "--limit" => {
                let n = args
                    .next()
                    .unwrap_or_else(|| fatal("--limit requires a number"));
                limit = Some(
                    n.parse::<usize>()
                        .unwrap_or_else(|_| fatal(&format!("bad --limit {n}"))),
                );
            }
            "--skip" => {
                let n = args
                    .next()
                    .unwrap_or_else(|| fatal("--skip requires a number"));
                skip = n
                    .parse::<usize>()
                    .unwrap_or_else(|_| fatal(&format!("bad --skip {n}")));
            }
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
            "--case-timeout-secs" => {
                let n = args
                    .next()
                    .unwrap_or_else(|| fatal("--case-timeout-secs requires a number"));
                case_timeout_secs = n
                    .parse::<u64>()
                    .unwrap_or_else(|_| fatal(&format!("bad --case-timeout-secs {n}")));
            }
            other if other.starts_with("--ledger=") => {
                ledger_path = Some(PathBuf::from(other.trim_start_matches("--ledger=")));
            }
            other if other.starts_with("--local-corpus=") => {
                local_corpus = Some(PathBuf::from(other.trim_start_matches("--local-corpus=")));
            }
            other if other.starts_with("--jobs=") => {
                let n = other.trim_start_matches("--jobs=");
                jobs = Some(
                    n.parse::<usize>()
                        .unwrap_or_else(|_| fatal(&format!("bad --jobs {n}"))),
                );
            }
            other if other.starts_with("--status=") => {
                status = Some(other.trim_start_matches("--status=").to_string());
            }
            other if other.starts_with("--contains=") => {
                contains = Some(other.trim_start_matches("--contains=").to_string());
            }
            other if other.starts_with("--limit=") => {
                let n = other.trim_start_matches("--limit=");
                limit = Some(
                    n.parse::<usize>()
                        .unwrap_or_else(|_| fatal(&format!("bad --limit {n}"))),
                );
            }
            other if other.starts_with("--skip=") => {
                let n = other.trim_start_matches("--skip=");
                skip = n
                    .parse::<usize>()
                    .unwrap_or_else(|_| fatal(&format!("bad --skip {n}")));
            }
            other if other.starts_with("--case-timeout-secs=") => {
                let n = other.trim_start_matches("--case-timeout-secs=");
                case_timeout_secs = n
                    .parse::<u64>()
                    .unwrap_or_else(|_| fatal(&format!("bad --case-timeout-secs {n}")));
            }
            other => fatal(&format!("unknown arg: {other}")),
        }
    }

    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let local_corpus = local_corpus.unwrap_or_else(corpus_shards::corpus_root_from_env_or_manifest);
    if translate_one {
        return Some(Mode::TranslateOne(TranslateOneOptions {
            local_corpus,
            source: build_translate_one_source(
                one_air_sha256,
                one_label,
                one_kind,
                one_shard,
                one_public_path,
                one_byte_offset,
                one_byte_len,
            ),
        }));
    }

    Some(Mode::Remint(RemintOptions {
        ledger: ledger_path
            .unwrap_or_else(|| manifest_dir.join("corpus/metal2vulkan-ledger.jsonl")),
        public_dir: manifest_dir.join("fixtures/public"),
        local_corpus,
        jobs: jobs
            .unwrap_or_else(translate_ledger::default_workers)
            .max(1),
        quiet,
        dry_run,
        failed_only,
        status,
        contains,
        skip,
        limit,
        case_timeout_secs,
    }))
}

fn print_usage() {
    eprintln!(
        "usage: {PROGRAM} [--ledger PATH] [--jobs N] [--quiet] [--dry-run] [--failed-only]\n\
         \t\t[--status STATUS] [--contains TEXT] [--skip N] [--limit N]\n\
         \t\t[--case-timeout-secs N]\n\
         \n\
         Re-translates metal2vulkan-ledger rows whose AIR is present under public fixtures\n\
         or local JSONL shards, and rewrites those rows in the ledger.\n\
         Default: remint every ledger row with a public fixture or shard source.\n\
         --failed-only  only remint rows with status != ok\n\
         --status S     only remint rows with status S\n\
         --contains T   only remint rows whose label/status/stage/kind/hash contains T\n\
         --skip N       skip N selected rows after stable sorting, before --limit\n\
         --limit N      remint at most N selected rows after stable sorting\n\
         --case-timeout-secs N\n\
                       per-row subprocess wall timeout; 0 keeps legacy in-process remint\n\
         --dry-run      list remintable rows only; do not translate or write."
    );
}

fn run(opts: RemintOptions) -> i32 {
    eprintln!("# {PROGRAM}");
    eprintln!("# ledger  {}", opts.ledger.display());
    eprintln!("# public  {}", opts.public_dir.display());
    eprintln!("# local   {}", opts.local_corpus.display());
    eprintln!("# jobs    {}", opts.jobs);
    if opts.case_timeout_secs == 0 {
        eprintln!("# timeout disabled (legacy in-process translate)");
    } else {
        eprintln!("# timeout {}s per row", opts.case_timeout_secs);
    }
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
    if let Some(status) = opts.status.as_deref() {
        eprintln!("# filter-status {status}");
    }
    if let Some(contains) = opts.contains.as_deref() {
        eprintln!("# filter-contains {contains}");
    }
    if opts.skip > 0 {
        eprintln!("# skip    {}", opts.skip);
    }
    if let Some(limit) = opts.limit {
        eprintln!("# limit   {limit}");
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
        .filter(|r| opts.status.as_deref().is_none_or(|s| r.status == s))
        .filter(|r| {
            opts.contains
                .as_deref()
                .is_none_or(|needle| row_contains(r, needle))
        })
        .cloned()
        .collect();
    candidates.sort_by(|a, b| {
        a.label
            .cmp(&b.label)
            .then_with(|| a.air_sha256.cmp(&b.air_sha256))
    });
    let unbounded_candidates = candidates.len();
    if opts.skip > 0 {
        if opts.skip >= candidates.len() {
            candidates.clear();
        } else {
            candidates.drain(..opts.skip);
        }
    }
    if let Some(limit) = opts.limit {
        candidates.truncate(limit);
    }

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
    if opts.limit.is_some() || opts.skip > 0 {
        eprintln!(
            "# selected rows after skip/limit: {}/{}",
            candidates.len(),
            unbounded_candidates
        );
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

    let updated = if opts.case_timeout_secs == 0 {
        translate_ledger::translate_all(
            &todo,
            &opts.local_corpus,
            opts.jobs,
            opts.quiet,
            "remint",
            "m2v-remint",
        )
    } else {
        translate_all_subprocess(
            &todo,
            &opts.local_corpus,
            opts.jobs,
            opts.quiet,
            opts.case_timeout_secs,
        )
    };
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

fn build_translate_one_source(
    air_sha256: Option<String>,
    label: Option<String>,
    kind: Option<String>,
    shard: Option<String>,
    public_path: Option<PathBuf>,
    byte_offset: Option<u64>,
    byte_len: Option<usize>,
) -> SourceRef {
    let air_sha256 = air_sha256.unwrap_or_else(|| fatal("--translate-one requires --air-sha256"));
    let label = label.unwrap_or_else(|| fatal("--translate-one requires --label"));
    let kind = kind.unwrap_or_else(|| fatal("--translate-one requires --kind"));
    let storage = if let Some(path) = public_path {
        SourceStorage::PublicPath(path)
    } else {
        SourceStorage::ShardRow {
            shard: shard
                .clone()
                .unwrap_or_else(|| fatal("--translate-one shard rows require --shard")),
            byte_offset: byte_offset
                .unwrap_or_else(|| fatal("--translate-one shard rows require --byte-offset")),
            byte_len: byte_len
                .unwrap_or_else(|| fatal("--translate-one shard rows require --byte-len")),
        }
    };
    SourceRef {
        air_sha256,
        label,
        kind,
        shard,
        storage,
    }
}

fn run_translate_one(opts: TranslateOneOptions) -> i32 {
    let row =
        translate_ledger::translate_source_ref(&opts.source, &opts.local_corpus, "m2v-remint");
    match serde_json::to_string(&row) {
        Ok(line) => {
            println!("{line}");
            0
        }
        Err(e) => {
            eprintln!("{PROGRAM}: serialize worker row: {e}");
            1
        }
    }
}

fn translate_all_subprocess(
    sources: &[SourceRef],
    local_corpus: &std::path::Path,
    workers: usize,
    quiet: bool,
    timeout_secs: u64,
) -> Vec<TranslateLedgerRow> {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;
    use std::thread;

    let n = sources.len();
    if n == 0 {
        return Vec::new();
    }
    let jobs = workers.min(n).max(1);
    eprintln!("# translate workers={jobs} sources={n} timeout={timeout_secs}s");

    let done = AtomicUsize::new(0);
    let rows: Mutex<Vec<TranslateLedgerRow>> = Mutex::new(Vec::with_capacity(n));
    let chunk_size = n.div_ceil(jobs);
    thread::scope(|scope| {
        for chunk in sources.chunks(chunk_size) {
            let done = &done;
            let rows = &rows;
            scope.spawn(move || {
                for source in chunk {
                    let row = translate_one_subprocess(source, local_corpus, timeout_secs);
                    let i = done.fetch_add(1, Ordering::Relaxed) + 1;
                    if quiet {
                        if i == 1 || i == n || i.is_multiple_of(25) {
                            eprintln!("  [{i}/{n}] ...");
                        }
                    } else {
                        eprintln!(
                            "  [{i}/{n}] remint {:<10} {}  {}",
                            row.status,
                            source.label,
                            &source.air_sha256[..12.min(source.air_sha256.len())]
                        );
                    }
                    rows.lock().unwrap().push(row);
                }
            });
        }
    });

    let mut out = rows.into_inner().unwrap();
    out.sort_by(|a, b| {
        a.kind
            .cmp(&b.kind)
            .then_with(|| a.label.cmp(&b.label))
            .then_with(|| a.air_sha256.cmp(&b.air_sha256))
    });
    out
}

fn translate_one_subprocess(
    source: &SourceRef,
    local_corpus: &std::path::Path,
    timeout_secs: u64,
) -> TranslateLedgerRow {
    let exe = match std::env::current_exe() {
        Ok(exe) => exe,
        Err(e) => return error_row(source, "fallback", &format!("current_exe: {e}")),
    };
    let mut cmd = Command::new(exe);
    cmd.arg("--translate-one")
        .arg("--local-corpus")
        .arg(local_corpus)
        .arg("--air-sha256")
        .arg(&source.air_sha256)
        .arg("--label")
        .arg(&source.label)
        .arg("--kind")
        .arg(&source.kind);
    if let Some(shard) = source.shard.as_deref() {
        cmd.arg("--shard").arg(shard);
    }
    match &source.storage {
        SourceStorage::PublicPath(path) => {
            cmd.arg("--public-path").arg(path);
        }
        SourceStorage::ShardRow {
            shard,
            byte_offset,
            byte_len,
        } => {
            if source.shard.is_none() {
                cmd.arg("--shard").arg(shard);
            }
            cmd.arg("--byte-offset")
                .arg(byte_offset.to_string())
                .arg("--byte-len")
                .arg(byte_len.to_string());
        }
    }
    let backtrace = std::env::var("RUST_BACKTRACE").unwrap_or_else(|_| "1".into());
    cmd.env("RUST_BACKTRACE", backtrace);
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        cmd.process_group(0);
    }
    cmd.stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let mut child = match cmd.spawn() {
        Ok(child) => child,
        Err(e) => return error_row(source, "fallback", &format!("spawn worker: {e}")),
    };

    match child
        .wait_timeout(Duration::from_secs(timeout_secs))
        .unwrap_or_else(|e| fatal(&format!("wait worker: {e}")))
    {
        Some(status) => {
            let mut stdout = String::new();
            if let Some(mut pipe) = child.stdout.take() {
                let _ = pipe.read_to_string(&mut stdout);
            }
            let mut stderr = String::new();
            if let Some(mut pipe) = child.stderr.take() {
                let _ = pipe.read_to_string(&mut stderr);
            }
            if !status.success() {
                let message = if stderr.trim().is_empty() {
                    format!("worker exited with {status}")
                } else {
                    stderr.trim().to_string()
                };
                return error_row(source, "fallback", &message);
            }
            match serde_json::from_str::<TranslateLedgerRow>(stdout.trim()) {
                Ok(row) => row,
                Err(e) => error_row(source, "fallback", &format!("parse worker row: {e}")),
            }
        }
        None => {
            kill_child(&mut child);
            let _ = child.wait();
            error_row(source, "timeout", &format!("timeout after {timeout_secs}s"))
        }
    }
}

fn kill_child(child: &mut std::process::Child) {
    #[cfg(unix)]
    unsafe {
        let pgid = child.id() as i32;
        let _ = libc::kill(-pgid, libc::SIGKILL);
    }
    #[cfg(not(unix))]
    {
        let _ = child.kill();
    }
}

fn error_row(source: &SourceRef, status: &str, _error: &str) -> TranslateLedgerRow {
    TranslateLedgerRow {
        air_sha256: source.air_sha256.clone(),
        shard: source.shard.clone(),
        spv_sha256: None,
        status: status.to_string(),
        stage: "auto".to_string(),
        label: source.label.clone(),
        kind: source.kind.clone(),
    }
}

fn row_contains(row: &TranslateLedgerRow, needle: &str) -> bool {
    [
        row.air_sha256.as_str(),
        row.status.as_str(),
        row.stage.as_str(),
        row.label.as_str(),
        row.kind.as_str(),
        row.shard.as_deref().unwrap_or(""),
        row.spv_sha256.as_deref().unwrap_or(""),
    ]
    .iter()
    .any(|value| value.contains(needle))
}

fn fatal(msg: &str) -> ! {
    eprintln!("{PROGRAM}: {msg}");
    std::process::exit(64);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timeout_row_preserves_source_identity() {
        let source = SourceRef {
            air_sha256: "a".repeat(64),
            label: "local/a.ll".into(),
            kind: "private".into(),
            shard: Some("shard_00.jsonl".into()),
            storage: SourceStorage::ShardRow {
                shard: "shard_00.jsonl".into(),
                byte_offset: 7,
                byte_len: 11,
            },
        };

        let row = error_row(&source, "timeout", "timeout after 1s");

        assert_eq!(row.air_sha256, source.air_sha256);
        assert_eq!(row.shard, source.shard);
        assert_eq!(row.label, source.label);
        assert_eq!(row.kind, source.kind);
        assert_eq!(row.status, "timeout");
        assert!(row.spv_sha256.is_none());
    }
}
