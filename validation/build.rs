use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};

fn main() {
    let product = PathBuf::from("..");
    let mut product_files = vec![product.join("Cargo.toml")];
    collect_files(&product.join("src"), &mut product_files);
    product_files.sort();

    let mut product_hash = Sha256::new();
    hash_files(&mut product_hash, &product, &product_files);
    println!(
        "cargo:rustc-env=METAL2VULKAN_PRODUCT_FINGERPRINT={:x}",
        product_hash.clone().finalize()
    );

    // Translation-audit facts also depend on the worker boundary, validation, scheduling, and
    // classification code in this crate. Keep that cache key separate from candidate observations,
    // whose dependency contract is the product translator itself.
    let validation = PathBuf::from(".");
    let mut audit_files = vec![validation.join("Cargo.toml")];
    collect_files(&validation.join("src"), &mut audit_files);
    audit_files.sort();
    let mut audit_hash = product_hash;
    audit_hash.update(b"metal2vulkan-translation-audit\0");
    hash_files(&mut audit_hash, &validation, &audit_files);
    println!(
        "cargo:rustc-env=METAL2VULKAN_TRANSLATION_AUDIT_FINGERPRINT={:x}",
        audit_hash.finalize()
    );
}

fn hash_files(hash: &mut Sha256, root: &Path, files: &[PathBuf]) {
    for path in files {
        println!("cargo:rerun-if-changed={}", path.display());
        let relative = path.strip_prefix(root).expect("fingerprint input");
        let bytes = fs::read(path)
            .unwrap_or_else(|error| panic!("read fingerprint input {}: {error}", path.display()));
        hash.update(relative.to_string_lossy().as_bytes());
        hash.update([0]);
        hash.update((bytes.len() as u64).to_le_bytes());
        hash.update(bytes);
    }
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
