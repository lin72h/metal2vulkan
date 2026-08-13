use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};

fn main() {
    let product = PathBuf::from("..");
    let mut files = vec![product.join("Cargo.toml")];
    collect_files(&product.join("src"), &mut files);
    files.sort();

    let mut hash = Sha256::new();
    for path in files {
        println!("cargo:rerun-if-changed={}", path.display());
        let relative = path.strip_prefix(&product).expect("product file");
        let bytes = fs::read(&path).unwrap_or_else(|error| {
            panic!(
                "read translator fingerprint input {}: {error}",
                path.display()
            )
        });
        hash.update(relative.to_string_lossy().as_bytes());
        hash.update([0]);
        hash.update((bytes.len() as u64).to_le_bytes());
        hash.update(bytes);
    }
    println!(
        "cargo:rustc-env=METAL2VULKAN_PRODUCT_FINGERPRINT={:x}",
        hash.finalize()
    );
}

fn collect_files(directory: &Path, output: &mut Vec<PathBuf>) {
    let mut entries = fs::read_dir(directory)
        .unwrap_or_else(|error| panic!("read {}: {error}", directory.display()))
        .map(|entry| entry.expect("directory entry").path())
        .collect::<Vec<_>>();
    entries.sort();
    for path in entries {
        if path.is_dir() {
            collect_files(&path, output);
        } else if path.is_file() {
            output.push(path);
        }
    }
}
