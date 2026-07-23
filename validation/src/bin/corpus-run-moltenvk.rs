//! Compare MoltenVK candidates to Metal goldens (`metal2vulkan-ledger-moltenvk.jsonl`).
//!
//! ```text
//! cargo run -p metal2vulkan-validation --release --bin corpus-run-moltenvk -- --dry-run
//! cargo run -p metal2vulkan-validation --release --bin corpus-run-moltenvk
//! ```

fn main() {
    let Some(cfg) = metal2vulkan_validation::corpus_run::parse_run_args(
        metal2vulkan_validation::corpus_run::RunBackend::MoltenVk,
    ) else {
        return;
    };
    std::process::exit(metal2vulkan_validation::corpus_run::run_driver(&cfg));
}
