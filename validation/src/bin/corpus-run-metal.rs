//! Bank Metal oracle goldens into `metal2vulkan-ledger-metal.jsonl` (see `plan.md`).
//!
//! ```text
//! cargo run -p metal2vulkan-validation --release --bin corpus-run-metal -- --dry-run
//! cargo run -p metal2vulkan-validation --release --bin corpus-run-metal
//! ```

fn main() {
    let Some(cfg) = metal2vulkan_validation::corpus_run::parse_run_args(
        metal2vulkan_validation::corpus_run::RunBackend::Metal,
    ) else {
        return;
    };
    std::process::exit(metal2vulkan_validation::corpus_run::run_driver(&cfg));
}
