use metal2vulkan_validation::spirv_delta::{classify_spirv_delta, SpirvDelta};
use std::fs;
use std::path::{Path, PathBuf};

fn main() {
    let args = std::env::args_os().skip(1).collect::<Vec<_>>();
    if args.first().is_some_and(|arg| arg == "list") {
        assert_eq!(
            args.len(),
            4,
            "usage: spirv_delta list <keys> <before-directory> <after-directory>"
        );
        classify_list(
            Path::new(&args[1]),
            Path::new(&args[2]),
            Path::new(&args[3]),
        );
        return;
    }
    assert_eq!(args.len(), 2, "usage: spirv_delta <before.spv> <after.spv>");
    let before = PathBuf::from(&args[0]);
    let after = PathBuf::from(&args[1]);

    let before_bytes =
        fs::read(&before).unwrap_or_else(|error| panic!("read {}: {error}", before.display()));
    let after_bytes =
        fs::read(&after).unwrap_or_else(|error| panic!("read {}: {error}", after.display()));
    let verdict = classify_spirv_delta(&before_bytes, &after_bytes)
        .unwrap_or_else(|error| panic!("classify SPIR-V delta: {error}"));
    println!("{}", verdict_text(verdict));
}

fn classify_list(keys_path: &Path, before_dir: &Path, after_dir: &Path) {
    let keys = fs::read_to_string(keys_path)
        .unwrap_or_else(|error| panic!("read {}: {error}", keys_path.display()));
    let mut saw_other = false;
    for key in keys.lines().map(str::trim).filter(|line| !line.is_empty()) {
        let stem = key.replace('/', "-");
        let before = before_dir.join(format!("{stem}.spv"));
        let after = after_dir.join(format!("{stem}.spv"));
        let before_bytes =
            fs::read(&before).unwrap_or_else(|error| panic!("read {}: {error}", before.display()));
        let after_bytes =
            fs::read(&after).unwrap_or_else(|error| panic!("read {}: {error}", after.display()));
        let verdict = classify_spirv_delta(&before_bytes, &after_bytes)
            .unwrap_or_else(|error| panic!("classify {key}: {error}"));
        saw_other |= matches!(verdict, SpirvDelta::Other { .. });
        println!("{key} {}", verdict_text(verdict));
    }
    if saw_other {
        std::process::exit(1);
    }
}

fn verdict_text(verdict: SpirvDelta) -> String {
    match verdict {
        SpirvDelta::Dc0 => "DC0".into(),
        SpirvDelta::Dc1 => "DC1".into(),
        SpirvDelta::Dc2 => "DC2".into(),
        SpirvDelta::Other {
            first_offending_line,
        } => format!("OTHER {first_offending_line}"),
    }
}
