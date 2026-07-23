//! Summarize validation ledgers and print failure-focused rerun commands.

use metal2vulkan_validation::corpus_triage::{
    failure_buckets, ledger_path, load_rows, parse_ledger_kind, status_counts, LedgerKind,
    TriageRow,
};
use std::path::PathBuf;

const PROGRAM: &str = "corpus-triage";
const DEFAULT_LIMIT: usize = 20;

#[derive(Debug)]
struct Options {
    ledger_dir: PathBuf,
    kinds: Vec<LedgerKind>,
    status: Option<String>,
    contains: Option<String>,
    include_success: bool,
    limit: usize,
    commands: bool,
}

fn main() {
    let Some(opts) = parse_args() else {
        return;
    };
    std::process::exit(run(opts));
}

fn parse_args() -> Option<Options> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut opts = Options {
        ledger_dir: manifest_dir.join("corpus"),
        kinds: LedgerKind::all().into(),
        status: None,
        contains: None,
        include_success: false,
        limit: DEFAULT_LIMIT,
        commands: false,
    };

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-h" | "--help" => {
                print_usage();
                return None;
            }
            "--commands" => opts.commands = true,
            "--include-success" => opts.include_success = true,
            "--ledger-dir" => {
                let p = args
                    .next()
                    .unwrap_or_else(|| fatal("--ledger-dir requires a path"));
                opts.ledger_dir = PathBuf::from(p);
            }
            "--backend" => {
                let backend = args
                    .next()
                    .unwrap_or_else(|| fatal("--backend requires a value"));
                opts.kinds = parse_backends(&backend);
            }
            "--status" => {
                opts.status = Some(
                    args.next()
                        .unwrap_or_else(|| fatal("--status requires a value")),
                );
            }
            "--contains" => {
                opts.contains = Some(
                    args.next()
                        .unwrap_or_else(|| fatal("--contains requires text")),
                );
            }
            "--limit" => {
                let n = args
                    .next()
                    .unwrap_or_else(|| fatal("--limit requires a number"));
                opts.limit = n
                    .parse::<usize>()
                    .unwrap_or_else(|_| fatal(&format!("bad --limit {n}")));
            }
            other if other.starts_with("--ledger-dir=") => {
                opts.ledger_dir = PathBuf::from(other.trim_start_matches("--ledger-dir="));
            }
            other if other.starts_with("--backend=") => {
                opts.kinds = parse_backends(other.trim_start_matches("--backend="));
            }
            other if other.starts_with("--status=") => {
                opts.status = Some(other.trim_start_matches("--status=").to_string());
            }
            other if other.starts_with("--contains=") => {
                opts.contains = Some(other.trim_start_matches("--contains=").to_string());
            }
            other if other.starts_with("--limit=") => {
                let n = other.trim_start_matches("--limit=");
                opts.limit = n
                    .parse::<usize>()
                    .unwrap_or_else(|_| fatal(&format!("bad --limit {n}")));
            }
            other => fatal(&format!("unknown arg: {other}")),
        }
    }

    Some(opts)
}

fn parse_backends(s: &str) -> Vec<LedgerKind> {
    if s == "all" {
        return LedgerKind::all().into();
    }
    let mut out = Vec::new();
    for part in s.split(',') {
        let part = part.trim();
        let Some(kind) = parse_ledger_kind(part) else {
            fatal(&format!("unknown backend {part:?}"));
        };
        out.push(kind);
    }
    out.sort();
    out.dedup();
    out
}

fn run(opts: Options) -> i32 {
    eprintln!("# {PROGRAM}");
    eprintln!("# ledger-dir {}", opts.ledger_dir.display());
    eprintln!(
        "# selected rows: {}success status={:?} contains={:?} limit={}",
        if opts.include_success {
            "including "
        } else {
            "non-"
        },
        opts.status,
        opts.contains,
        opts.limit
    );

    let mut selected = Vec::new();
    let mut had_error = false;
    for kind in &opts.kinds {
        let path = ledger_path(&opts.ledger_dir, *kind);
        let rows = match load_rows(&opts.ledger_dir, *kind) {
            Ok(rows) => rows,
            Err(e) => {
                eprintln!("# {} ERROR: {e}", kind.as_str());
                had_error = true;
                continue;
            }
        };
        eprintln!("# {} {} rows={}", kind.as_str(), path.display(), rows.len());
        print_counts("status", &status_counts(&rows));
        print_counts("bucket", &failure_buckets(&rows));

        selected.extend(rows.into_iter().filter(|row| selected_by(&opts, row)));
    }

    selected.sort_by(|a, b| {
        a.kind
            .cmp(&b.kind)
            .then(a.status.cmp(&b.status))
            .then(a.signature.cmp(&b.signature))
            .then(a.label.cmp(&b.label))
            .then(a.air_sha256.cmp(&b.air_sha256))
    });

    let shown = if opts.limit == 0 {
        0
    } else {
        selected.len().min(opts.limit)
    };
    eprintln!("# rows matched={} shown={shown}", selected.len());
    for row in selected.iter().take(shown) {
        print_row(row, opts.commands);
    }

    if had_error {
        1
    } else {
        0
    }
}

fn selected_by(opts: &Options, row: &TriageRow) -> bool {
    if let Some(status) = opts.status.as_deref() {
        if row.status != status {
            return false;
        }
    } else if !opts.include_success && row.is_success() {
        return false;
    }
    if let Some(needle) = opts.contains.as_deref() {
        if !row.matches_text(needle) {
            return false;
        }
    }
    true
}

fn print_counts(label: &str, counts: &std::collections::BTreeMap<String, usize>) {
    if counts.is_empty() {
        eprintln!("  {label}: (none)");
        return;
    }
    let mut items: Vec<_> = counts.iter().collect();
    items.sort_by(|(a_key, a_count), (b_key, b_count)| {
        b_count.cmp(a_count).then_with(|| a_key.cmp(b_key))
    });
    for (key, count) in items {
        eprintln!("  {label}: {count:>6} {key}");
    }
}

fn print_row(row: &TriageRow, commands: bool) {
    eprintln!(
        "{} {} {} {}",
        row.kind.as_str(),
        row.status,
        row.air_sha256,
        row.label
    );
    if let Some(shard) = row.shard.as_deref() {
        eprintln!("    shard: {shard}");
    }
    eprintln!("    bucket: {}", row.signature);
    if row.has_tolerance {
        eprintln!("    tolerance: present");
    }
    if let Some(spv) = row.spv_sha256.as_deref() {
        eprintln!("    spv: {spv}");
    }
    if let Some(output) = row.output_sha256.as_deref() {
        eprintln!("    output: {output}");
    }
    if let Some(golden) = row.golden_output_sha256.as_deref() {
        eprintln!("    golden: {golden}");
    }
    if let Some(error) = row.error.as_deref() {
        let first = error.lines().next().unwrap_or("").trim();
        if !first.is_empty() {
            eprintln!("    error: {first}");
        }
    }
    if commands {
        print_commands(row);
    }
}

fn print_commands(row: &TriageRow) {
    eprintln!(
        "    translate: cargo run -p metal2vulkan-validation --release --bin corpus-why -- {}",
        row.air_sha256
    );
    if row.kind != LedgerKind::Translate {
        if row.kind != LedgerKind::Metal {
            let label = if row.status == "missing" {
                "metal"
            } else {
                "metal-if-needed"
            };
            eprintln!(
                "    {label}: cargo run -p metal2vulkan-validation --release --bin corpus-run-metal -- --air-sha256 {} --force --jobs 1",
                row.air_sha256
            );
        }
        if let Some(bin) = row.kind.runner_bin() {
            eprintln!(
                "    rerun: cargo run -p metal2vulkan-validation --release --bin {bin} -- --air-sha256 {} --force --jobs 1",
                row.air_sha256
            );
        }
    } else {
        eprintln!(
            "    remint-failures: cargo run -p metal2vulkan-validation --release --bin corpus-remint -- --failed-only --jobs 1"
        );
    }
}

fn print_usage() {
    eprintln!(
        "usage: {PROGRAM} [--ledger-dir DIR] [--backend all|translate|metal|vulkan|moltenvk]\n\
                \t\t[--status STATUS] [--contains TEXT] [--include-success]\n\
                \t\t[--limit N] [--commands]\n\
         \n\
         Summarizes translate/execution JSONL ledgers, groups non-success rows by a stable\n\
         failure bucket, and lists failure rows for single-case iteration.\n\
         Defaults: all ledgers, non-success rows only, --limit {DEFAULT_LIMIT}."
    );
}

fn fatal(msg: &str) -> ! {
    eprintln!("{PROGRAM}: {msg}");
    std::process::exit(64);
}
