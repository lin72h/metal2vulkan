//! Compare Linux Vulkan candidates to Metal goldens (`metal2vulkan-ledger-vulkan.jsonl`).
//!
//! ```text
//! cargo run -p metal2vulkan-validation --release --bin corpus-run-vulkan -- --dry-run
//! cargo run -p metal2vulkan-validation --release --bin corpus-run-vulkan
//! ```

fn main() {
    let Some(cfg) = metal2vulkan_validation::corpus_run::parse_run_args(
        metal2vulkan_validation::corpus_run::RunBackend::Vulkan,
    ) else {
        return;
    };
    std::process::exit(metal2vulkan_validation::corpus_run::run_driver(&cfg));
}
