//! A reflected `access` covers what the module does through that descriptor.
//!
//! `ResourceAccess` is what a consumer reads to decide whether a buffer's contents have to be
//! staged before the dispatch, whether the results have to be read back, and which barriers the
//! resource needs. `ReadOnly` on a buffer the shader writes is not a lost optimization: it is a
//! missing barrier, and, for a consumer that places read-only resources in read-only memory, a
//! write to memory that does not accept one.
//!
//! The classification comes from AIR metadata, tightened by the specialized entry's parameter
//! attributes. Neither of those is the module. Measured over 2880 corpus sources before this was
//! reconciled: 20 buffers reflected `ReadOnly` were stored through, 9 reflected `WriteOnly` were
//! loaded from, and 3 reflected `Unused` were both.
//!
//! Only one direction is a defect. Reflection may report `ReadWrite` for a buffer the module only
//! reads: the walk that resolves an access to its descriptor follows the pointer graph it can, and
//! an access through a device address it cannot attribute would be missed, so a narrower answer
//! would be a guess. This file checks the direction that is not.

use metal2vulkan::passes::{Stage, TransformOptions};
use metal2vulkan::reflect::{ResourceAccess, ResourceKind, ShaderReflection};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// A kernel that contradicts both of its declared buffer accesses.
///
/// `declared_read` carries AIR's `air.read` and is stored through; `declared_write` carries
/// `air.write` and is loaded from. Both are ordinary Metal — AIR's declared access is not a
/// guarantee about the body — and both used to be reported as the declaration rather than as what
/// the shader does.
const CONTRADICTED_ACCESS: &str = r#"target triple = "spirv-unknown-vulkan1.2"
define void @k(ptr addrspace(1) %declared_read, ptr addrspace(1) %declared_write) {
entry:
  store i32 7, ptr addrspace(1) %declared_read, align 4
  %v = load i32, ptr addrspace(1) %declared_write, align 4
  %g = getelementptr inbounds i32, ptr addrspace(1) %declared_write, i64 1
  store i32 %v, ptr addrspace(1) %g, align 4
  ret void
}
!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.buffer", !"air.buffer_size", i32 4, !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"declared_read"}
!4 = !{i32 1, !"air.buffer", !"air.buffer_size", i32 8, !"air.location_index", i32 1, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"declared_write"}
"#;

#[test]
fn a_declared_access_the_body_contradicts_is_widened_to_what_the_body_does() {
    let (spirv, reflection) = translate(CONTRADICTED_ACCESS, Stage::Kernel, "contradicted_access");
    assert_access_covers_the_module("the contradicted-access kernel", &spirv, &reflection);

    for metal_index in [0, 1] {
        let resource = reflection
            .binding_at(ResourceKind::Buffer, metal_index)
            .unwrap_or_else(|| panic!("buffer {metal_index} is reflected"));
        assert_eq!(
            resource.access,
            Some(ResourceAccess::ReadWrite),
            "buffer {metal_index} is both read and written by the module"
        );
    }
}

/// The same contract over every committed fixture, at the stage its AIR declares.
#[test]
fn every_public_fixture_reflects_an_access_that_covers_its_module() {
    let mut checked = 0;
    let mut accesses = 0;
    for path in public_fixtures() {
        let source = std::fs::read_to_string(&path).expect("read fixture");
        let label = path
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_default();
        let Some(stage) = stage_of(&source) else {
            continue;
        };
        let Ok((spirv, reflection)) = metal2vulkan::translate_sanitized_native_reflected(
            &source,
            stage,
            &scratch(&label),
            TransformOptions::default(),
        ) else {
            continue;
        };
        accesses += assert_access_covers_the_module(&label, &spirv, &reflection);
        checked += 1;
    }
    assert!(
        checked >= 20 && accesses >= 10,
        "only {checked} fixtures carrying {accesses} buffer accesses were inspected, so this swept \
         almost nothing"
    );
}

/// Require each reflected buffer access to permit what the module does through that binding, and
/// return how many were checked.
fn assert_access_covers_the_module(
    label: &str,
    spirv: &[u8],
    reflection: &ShaderReflection,
) -> usize {
    let observed = accesses_the_module_performs(spirv);
    let mut checked = 0;
    for resource in &reflection.bindings {
        if !matches!(
            resource.kind,
            ResourceKind::Buffer | ResourceKind::KernelStageInput
        ) {
            continue;
        }
        let (Some(access), Some(location)) = (resource.access, resource.descriptor) else {
            continue;
        };
        let Some((reads, writes)) = observed.get(&location.binding).copied() else {
            continue;
        };
        let permitted = match access {
            ResourceAccess::ReadOnly => (true, false),
            ResourceAccess::WriteOnly => (false, true),
            ResourceAccess::ReadWrite => (true, true),
            ResourceAccess::Unused => (false, false),
            // Texture classifications; a buffer binding never carries one.
            ResourceAccess::Sampled | ResourceAccess::Storage => continue,
        };
        assert!(
            (!reads || permitted.0) && (!writes || permitted.1),
            "{label} reflects {access:?} for {:?}({}) at binding {}, but the module reads={reads} \
             writes={writes} through it, so a consumer would stage or barrier it wrongly",
            resource.kind,
            resource.metal_index,
            location.binding
        );
        checked += 1;
    }
    checked
}

/// Whether the module reads and/or writes through each descriptor binding.
///
/// Follows the pointer graph from each decorated variable through the chain and copy instructions
/// that keep a pointer's root, then classifies the loads and stores that land on it. This
/// deliberately does NOT follow a pointer through memory, so it under-reports rather than
/// over-reports, which is the safe direction for an assertion that a reflected access is wide
/// enough.
fn accesses_the_module_performs(spirv: &[u8]) -> HashMap<u32, (bool, bool)> {
    let text = metal2vulkan::disassemble(spirv).expect("disassemble the translated module");
    let mut binding_of: HashMap<String, u32> = HashMap::new();
    for line in text.lines() {
        if let ["OpDecorate", target, "Binding", value] =
            line.split_whitespace().collect::<Vec<_>>().as_slice()
        {
            if let Ok(value) = value.parse::<u32>() {
                binding_of.insert((*target).to_string(), value);
            }
        }
    }
    // Root every derived pointer at the variable it came from. One pass per fixed point: a chain
    // can be defined after the chain it extends only inside a function body, and instructions are
    // in definition order there, but a loop header's phi is not, so iterate until nothing changes.
    let mut root: HashMap<String, String> = binding_of
        .keys()
        .map(|id| (id.clone(), id.clone()))
        .collect();
    loop {
        let mut changed = false;
        for line in text.lines() {
            let tokens = line.split_whitespace().collect::<Vec<_>>();
            let (result, opcode, rest) = match tokens.as_slice() {
                [result, "=", opcode, rest @ ..] => (*result, *opcode, rest),
                _ => continue,
            };
            if !matches!(
                opcode,
                "OpAccessChain"
                    | "OpInBoundsAccessChain"
                    | "OpPtrAccessChain"
                    | "OpInBoundsPtrAccessChain"
                    | "OpCopyObject"
            ) {
                continue;
            }
            // `%r = Op... %type %base ...` — the base pointer is the operand after the result type.
            let Some(base) = rest.get(1) else { continue };
            if root.contains_key(result) {
                continue;
            }
            if let Some(base_root) = root.get(*base).cloned() {
                root.insert(result.to_string(), base_root);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    let mut observed: HashMap<u32, (bool, bool)> = HashMap::new();
    for line in text.lines() {
        let tokens = line.split_whitespace().collect::<Vec<_>>();
        let (pointer, reads, writes) = match tokens.as_slice() {
            [_, "=", "OpLoad", _, pointer, ..] => (*pointer, true, false),
            ["OpStore", pointer, ..] => (*pointer, false, true),
            // Every SPIR-V atomic reads its pointee; all but `OpAtomicLoad` write it back.
            [_, "=", opcode, _, pointer, ..] if opcode.starts_with("OpAtomic") => {
                (*pointer, true, *opcode != "OpAtomicLoad")
            }
            _ => continue,
        };
        let Some(root) = root.get(pointer) else {
            continue;
        };
        let Some(binding) = binding_of.get(root) else {
            continue;
        };
        let slot = observed.entry(*binding).or_default();
        slot.0 |= reads;
        slot.1 |= writes;
    }
    observed
}

fn translate(source: &str, stage: Stage, label: &str) -> (Vec<u8>, ShaderReflection) {
    metal2vulkan::translate_sanitized_native_reflected(
        source,
        stage,
        &scratch(label),
        TransformOptions::default(),
    )
    .unwrap_or_else(|error| panic!("{label} translates: {error}"))
}

/// Scratch for one subject. `spirv-val` writes into it and these tests run concurrently, so each
/// subject gets its own directory.
fn scratch(label: &str) -> PathBuf {
    let directory = std::env::temp_dir().join(format!(
        "m2v_reflected_access_{}_{}",
        std::process::id(),
        label.replace(['/', '.'], "_")
    ));
    std::fs::create_dir_all(&directory).expect("scratch directory");
    directory
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
