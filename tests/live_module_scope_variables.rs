//! A translated module declares no global variable it never uses.
//!
//! A module-scope `OpVariable` is not inert bookkeeping. A `StorageBuffer` or `UniformConstant` one
//! carries `DescriptorSet` and `Binding` decorations and sits in the entry point's interface, which
//! is where a consumer that builds a `VkDescriptorSetLayout` from the module reads its
//! requirements. A variable that outlived its last use therefore becomes a descriptor the shader
//! never touches but the application still has to bind.
//!
//! Nothing catches that on its own. `finalize` filters the interface to the variables the bodies
//! reference — but the instruction-deleting work that runs after it (pointer and access-chain
//! closures, dead-value elimination, constant-CFG pruning, CFG construction) can delete the last
//! use of a variable that was live when the interface was written, and from that point the
//! arrangement is self-sustaining: the global collector roots liveness AT the entry-point
//! interface, so the interface entry keeps the variable alive and the variable justifies the
//! interface entry. `passes::drop_unreferenced_global_variables` breaks that cycle at each
//! boundary; this file is the end-to-end check that no boundary is missing one.
//!
//! `prune.rs` already states this standard for blocks — "Dead code is a defect regardless of the
//! consumer" — and prunes unreachable ones on every translate path. This is the same standard for
//! module-scope variables.

use metal2vulkan::passes::Stage;
use metal2vulkan::{disassemble, translate_sanitized_native};
use std::path::{Path, PathBuf};

/// Every global `OpVariable` the module declares that no instruction inside a function names.
///
/// Reads the crate's own disassembly rather than the SPIR-V words: it prints raw ids and one
/// instruction per line, and the module-scope section is exactly what precedes the first
/// `OpFunction`. Function-scope (`Function` storage class) variables are declared after that line
/// and are deliberately out of scope here — they are local slots, not interface or descriptor
/// declarations.
fn global_variables_no_instruction_references(spirv: &[u8]) -> Vec<String> {
    let text = disassemble(spirv).expect("disassemble the translated module");
    let mut declared: Vec<(String, String)> = Vec::new();
    let mut referenced: Vec<String> = Vec::new();
    let mut inside_a_function = false;
    for line in text.lines() {
        let tokens = line.split_whitespace().collect::<Vec<_>>();
        if tokens.contains(&"OpFunction") {
            inside_a_function = true;
        }
        if inside_a_function {
            referenced.extend(
                tokens
                    .iter()
                    .filter(|token| is_id(token))
                    .map(|token| (*token).to_string()),
            );
            continue;
        }
        // `%<id> = OpVariable %<pointer type> <storage class>`
        if let ([id, "=", "OpVariable", ..], Some(_)) = (tokens.as_slice(), tokens.get(4)) {
            if is_id(id) {
                declared.push(((*id).to_string(), line.trim().to_string()));
            }
        }
    }
    declared
        .into_iter()
        .filter(|(id, _)| !referenced.contains(id))
        .map(|(_, line)| line)
        .collect()
}

/// A raw SPIR-V id as the crate's disassembler prints it: `%` and decimal digits, no friendly name.
fn is_id(token: &str) -> bool {
    token.strip_prefix('%').is_some_and(|digits| {
        !digits.is_empty() && digits.bytes().all(|byte| byte.is_ascii_digit())
    })
}

/// Scratch for one subject. `spirv-val` writes a fixed file name inside it and these tests run
/// concurrently, so each subject gets its own directory.
fn scratch(label: &str) -> PathBuf {
    let directory = std::env::temp_dir().join(format!(
        "m2v_live_globals_{}_{}",
        std::process::id(),
        label.replace(['/', '.'], "_")
    ));
    std::fs::create_dir_all(&directory).expect("scratch directory");
    directory
}

/// How many global variables the sweep saw. A guard that inspected modules with no global variables
/// at all would pass on anything, so the callers below require this to be substantial.
fn assert_no_unused_globals(label: &str, spirv: &[u8]) -> usize {
    let unused = global_variables_no_instruction_references(spirv);
    assert!(
        unused.is_empty(),
        "{label} declares {} module-scope variable(s) no instruction references:\n  {}",
        unused.len(),
        unused.join("\n  ")
    );
    disassemble(spirv)
        .expect("disassemble the translated module")
        .lines()
        .take_while(|line| !line.split_whitespace().any(|token| token == "OpFunction"))
        .filter(|line| line.split_whitespace().any(|token| token == "OpVariable"))
        .count()
}

#[test]
fn no_public_fixture_translates_to_a_module_with_an_unused_global_variable() {
    let mut variables = 0;
    let mut checked = 0;
    for path in public_fixtures() {
        let source = std::fs::read_to_string(&path).expect("read fixture");
        let label = path
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_default();
        let Some(stage) = stage_of(&source) else {
            continue;
        };
        let Ok(spirv) = translate_sanitized_native(&source, stage, &scratch(&label)) else {
            continue;
        };
        variables += assert_no_unused_globals(&label, &spirv);
        checked += 1;
    }
    assert!(
        checked >= 20 && variables >= 40,
        "only {checked} fixtures with {variables} global variables were inspected, so this swept \
         almost nothing"
    );
}

/// The same contract for every stage each fixture translates under.
///
/// The stage is an input like any other, and translating a fixture under a stage its AIR does not
/// declare reaches interface paths the declared stage never does — a buffer parameter the declared
/// stage binds to a descriptor becomes an unbound Private placeholder under another. Those
/// placeholder rewrites are exactly the kind that replace a variable's last use.
#[test]
fn no_fixture_leaves_an_unused_global_variable_under_any_stage_it_translates_under() {
    let mut variables = 0;
    let mut checked = 0;
    for path in public_fixtures() {
        let source = std::fs::read_to_string(&path).expect("read fixture");
        let name = path
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_default();
        for stage in [Stage::Vertex, Stage::Fragment, Stage::Kernel] {
            let label = format!("{name}-{stage:?}");
            let Ok(spirv) = translate_sanitized_native(&source, stage, &scratch(&label)) else {
                continue;
            };
            variables += assert_no_unused_globals(&label, &spirv);
            checked += 1;
        }
    }
    assert!(
        checked >= 30 && variables >= 60,
        "only {checked} fixture/stage pairs with {variables} global variables were inspected, so \
         this swept almost nothing"
    );
}

/// The stage the AIR declares. The library's `detect_stage` sanitizes from a file path; these
/// fixtures are already sanitized text, and they name their stage the same way.
fn stage_of(source: &str) -> Option<Stage> {
    if source.contains("!air.vertex =") {
        Some(Stage::Vertex)
    } else if source.contains("!air.fragment =") {
        Some(Stage::Fragment)
    } else if source.contains("!air.kernel =") {
        Some(Stage::Kernel)
    } else {
        None
    }
}

fn public_fixtures() -> Vec<PathBuf> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("validation/fixtures/public");
    let mut paths = std::fs::read_dir(&root)
        .unwrap_or_else(|error| panic!("read {}: {error}", root.display()))
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|extension| extension == "ll"))
        .collect::<Vec<_>>();
    paths.sort();
    paths
}
