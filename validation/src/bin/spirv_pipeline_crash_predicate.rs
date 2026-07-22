//! Interestingness predicate for `spirv-reduce` over NVIDIA compute-pipeline crashes.
//!
//! Usage: `spirv_pipeline_crash_predicate <candidate.spv>`.
//!
//! The candidate is interesting iff it validates for Vulkan 1.3 and the sibling
//! `spirv_pipeline_probe --pipeline` process terminates with SIGSEGV / exit 139.

use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

#[cfg(unix)]
use std::os::unix::process::ExitStatusExt;

fn main() {
    let args = std::env::args_os().skip(1).collect::<Vec<_>>();
    if args.len() == 1 && matches!(args[0].to_str(), Some("-h" | "--help")) {
        println!("usage: spirv_pipeline_crash_predicate <candidate.spv>");
        return;
    }
    if args.len() != 1 {
        eprintln!("usage: spirv_pipeline_crash_predicate <candidate.spv>");
        std::process::exit(2);
    }
    let spv = PathBuf::from(&args[0]);
    if !valid_spirv(&spv) {
        std::process::exit(1);
    }
    match run_probe(&spv) {
        ProbeResult::Segv => std::process::exit(0),
        ProbeResult::Clean | ProbeResult::OtherFailure | ProbeResult::TimedOut => {
            std::process::exit(1)
        }
        ProbeResult::InfrastructureError => std::process::exit(2),
    }
}

fn valid_spirv(spv: &Path) -> bool {
    let spirv_val = std::env::var_os("SPIRV_VAL").unwrap_or_else(|| OsString::from("spirv-val"));
    matches!(
        Command::new(spirv_val)
            .args(["--target-env", "vulkan1.3"])
            .arg(spv)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status(),
        Ok(status) if status.success()
    )
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ProbeResult {
    Segv,
    Clean,
    OtherFailure,
    TimedOut,
    InfrastructureError,
}

fn run_probe(spv: &Path) -> ProbeResult {
    let probe = match probe_path() {
        Some(probe) => probe,
        None => return ProbeResult::InfrastructureError,
    };
    let mut child = match Command::new(probe)
        .arg("--pipeline")
        .arg(spv)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
    {
        Ok(child) => child,
        Err(_) => return ProbeResult::InfrastructureError,
    };
    let deadline = Instant::now() + timeout();
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                if status.code() == Some(139) {
                    return ProbeResult::Segv;
                }
                #[cfg(unix)]
                if status.signal() == Some(11) {
                    return ProbeResult::Segv;
                }
                return if status.success() {
                    ProbeResult::Clean
                } else {
                    ProbeResult::OtherFailure
                };
            }
            Ok(None) if Instant::now() < deadline => {
                thread::sleep(Duration::from_millis(50));
            }
            Ok(None) => {
                let _ = child.kill();
                let _ = child.wait();
                return ProbeResult::TimedOut;
            }
            Err(_) => return ProbeResult::InfrastructureError,
        }
    }
}

fn probe_path() -> Option<PathBuf> {
    if let Some(path) = std::env::var_os("SPIRV_PIPELINE_PROBE") {
        return Some(PathBuf::from(path));
    }
    let mut path = std::env::current_exe().ok()?;
    path.set_file_name("spirv_pipeline_probe");
    Some(path)
}

fn timeout() -> Duration {
    let millis = std::env::var("SPIRV_PIPELINE_TIMEOUT_MS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(15_000);
    Duration::from_millis(millis)
}
