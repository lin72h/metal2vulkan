use metal2vulkan_validation::candidate::execute_case;
use metal2vulkan_validation::observation::Backend;
use metal2vulkan_validation::source::corpus_root;
use metal2vulkan_validation::store::CorpusStore;
use std::path::PathBuf;

pub fn main(backend: Backend) {
    if let Err(error) = run(backend) {
        eprintln!("corpus-{}: {error}", backend.directory());
        std::process::exit(1);
    }
}

fn run(backend: Backend) -> Result<(), String> {
    let mut root = corpus_root();
    let mut case_id = None;
    let mut metal_environment_id = None;
    let mut environment_id = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--corpus" => root = PathBuf::from(required(&mut args, "--corpus")?),
            "--case-id" => case_id = Some(required(&mut args, "--case-id")?),
            "--metal-environment-id" => {
                metal_environment_id = Some(required(&mut args, "--metal-environment-id")?)
            }
            "--environment-id" => environment_id = Some(required(&mut args, "--environment-id")?),
            "-h" | "--help" => {
                println!(
                    "usage: corpus-{} [--corpus DIR] --case-id HASH --metal-environment-id ID --environment-id ID",
                    backend.directory()
                );
                return Ok(());
            }
            _ => return Err(format!("unknown argument {arg:?}")),
        }
    }
    let case_id = case_id.ok_or("--case-id is required")?;
    let metal_environment_id = metal_environment_id.ok_or("--metal-environment-id is required")?;
    let environment_id = environment_id.ok_or("--environment-id is required")?;
    let case = CorpusStore::new(&root)
        .find_case(&case_id)?
        .ok_or_else(|| format!("case_id {case_id} not found"))?;
    let observation = execute_case(&root, case, backend, &metal_environment_id, &environment_id)?;
    println!(
        "{:?} {} candidate_output_sha256={}",
        observation.status, observation.case_id, observation.candidate_output_sha256
    );
    if observation.status == metal2vulkan_validation::observation::CandidateStatus::Mismatch {
        return Err("candidate output does not match the exact Metal observation".into());
    }
    Ok(())
}

fn required(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, String> {
    args.next()
        .ok_or_else(|| format!("{flag} requires a value"))
}
