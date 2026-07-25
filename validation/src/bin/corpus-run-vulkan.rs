//! Compare Linux Vulkan candidates to Metal goldens (`metal2vulkan-ledger-vulkan.jsonl`).
//!
//! ```text
//! cargo run -p metal2vulkan-validation --release --bin corpus-run-vulkan -- --dry-run
//! cargo run -p metal2vulkan-validation --release --bin corpus-run-vulkan
//! ```

fn main() {
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    {
        let mut args = std::env::args_os();
        let _program = args.next();
        match args
            .next()
            .and_then(|arg| arg.into_string().ok())
            .as_deref()
        {
            Some("--graphics-pipeline-probe") => {
                std::process::exit(
                    metal2vulkan_validation::runner_linux::run_graphics_pipeline_probe_args(args),
                );
            }
            Some("--vertex-pipeline-probe") => {
                std::process::exit(
                    metal2vulkan_validation::runner_linux::run_vertex_pipeline_probe_args(args),
                );
            }
            Some("--compute-pipeline-probe") => {
                std::process::exit(
                    metal2vulkan_validation::runner_linux::run_compute_pipeline_probe_args(args),
                );
            }
            _ => {}
        }
    }

    let Some(cfg) = metal2vulkan_validation::corpus_run::parse_run_args(
        metal2vulkan_validation::corpus_run::RunBackend::Vulkan,
    ) else {
        return;
    };
    std::process::exit(metal2vulkan_validation::corpus_run::run_driver(&cfg));
}
