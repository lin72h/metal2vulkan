use metal2vulkan_validation::metal::qualify_case;
use metal2vulkan_validation::source::corpus_root;
use metal2vulkan_validation::store::CorpusStore;
use std::path::PathBuf;

fn main() {
    if let Err(error) = run() {
        eprintln!("corpus-metal: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let mut root = corpus_root();
    let mut case_id = None;
    let mut environment_id = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--corpus" => root = PathBuf::from(required(&mut args, "--corpus")?),
            "--case-id" => case_id = Some(required(&mut args, "--case-id")?),
            "--environment-id" => environment_id = Some(required(&mut args, "--environment-id")?),
            "-h" | "--help" => {
                println!("usage: corpus-metal [--corpus DIR] --case-id HASH --environment-id ID");
                return Ok(());
            }
            _ => return Err(format!("unknown argument {arg:?}")),
        }
    }
    let case_id = case_id.ok_or("--case-id is required")?;
    let environment_id = environment_id.ok_or("--environment-id is required")?;
    let case = CorpusStore::new(&root)
        .find_case(&case_id)?
        .ok_or_else(|| format!("case_id {case_id} not found"))?;
    let observation = qualify_case(&root, case, &environment_id)?;
    println!(
        "qualified {} output_sha256={}",
        observation.case_id, observation.metal_output_sha256
    );
    Ok(())
}

fn required(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, String> {
    args.next()
        .ok_or_else(|| format!("{flag} requires a value"))
}
