use metal2vulkan_validation::case::AuthoredCase;
use metal2vulkan_validation::check::check_case;
use metal2vulkan_validation::source::corpus_root;
use metal2vulkan_validation::store::CorpusStore;
use std::fs;
use std::path::PathBuf;

fn main() {
    if let Err(error) = run() {
        eprintln!("corpus-case-check: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let mut root = corpus_root();
    let mut manifest = None;
    let mut case_id = None;
    let mut install = false;
    let mut delete_air = None;
    let mut delete_name = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--corpus" => root = PathBuf::from(required(&mut args, "--corpus")?),
            "--manifest" => manifest = Some(PathBuf::from(required(&mut args, "--manifest")?)),
            "--case-id" => case_id = Some(required(&mut args, "--case-id")?),
            "--install" => install = true,
            "--delete-air" => delete_air = Some(required(&mut args, "--delete-air")?),
            "--delete-name" => delete_name = Some(required(&mut args, "--delete-name")?),
            "-h" | "--help" => {
                println!(
                    "usage: corpus-case-check [--corpus DIR] (--manifest PATH [--install] | --case-id HASH | --delete-air HASH --delete-name NAME)"
                );
                return Ok(());
            }
            _ => return Err(format!("unknown argument {arg:?}")),
        }
    }
    let store = CorpusStore::new(&root);
    match (delete_air, delete_name) {
        (Some(air), Some(name)) if manifest.is_none() && case_id.is_none() && !install => {
            let removed = store.delete_named_case(&air, &name)?;
            println!("deleted {removed}");
            return Ok(());
        }
        (None, None) => {}
        _ => {
            return Err(
                "--delete-air and --delete-name must be used together and without case checking"
                    .into(),
            )
        }
    }
    let mut case = match (manifest, case_id) {
        (Some(path), None) => {
            let bytes = fs::read(&path)
                .map_err(|error| format!("read manifest {}: {error}", path.display()))?;
            serde_json::from_slice::<AuthoredCase>(&bytes)
                .map_err(|error| format!("parse manifest {}: {error}", path.display()))?
        }
        (None, Some(case_id)) => store
            .find_case(&case_id)?
            .ok_or_else(|| format!("case_id {case_id} not found"))?,
        _ => return Err("select exactly one of --manifest or --case-id".into()),
    };
    if case.case_id.is_empty() {
        case.case_id = case.computed_case_id()?;
    }
    let checked = check_case(&root, case).map_err(|errors| errors.join("\n  - "))?;
    println!(
        "checked {} input_sha256={}",
        checked.case.case_id, checked.input_sha256
    );
    if install {
        println!("installed {:?}", store.put_case(checked.case)?);
    }
    Ok(())
}

fn required(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, String> {
    args.next()
        .ok_or_else(|| format!("{flag} requires a value"))
}
