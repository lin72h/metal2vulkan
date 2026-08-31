//! Reflection and the module agree on which descriptors exist.
//!
//! `ShaderReflection` is what a consumer builds its `VkDescriptorSetLayout` from. If the translated
//! module decorates a variable with a `DescriptorSet`/`Binding` that reflection does not report,
//! the layout the consumer builds does not cover the shader, and the mismatch surfaces at pipeline
//! creation or as a read of an unbound descriptor — never as a translation error. The two
//! descriptions have to close over each other.
//!
//! Reflection may legitimately report MORE than the module declares: Metal can declare a resource
//! the shader never touches, and dropping the binding would silently renumber a consumer's
//! expectations. Only the other direction is a defect, and that is what this file checks.
//!
//! The first case below is where the direction went wrong. `air.get_read_sampler()` exists because
//! AIR threads a sampler pointer into `texture.read(coord)`, which in Metal is sampler-less; the
//! interface pass materialized a real `OpTypeSampler` descriptor so the value would be typed, and
//! `lower_read` then discarded the value. No Metal argument corresponds to that sampler, so
//! reflection had nothing to report — while the module demanded the binding anyway.

use metal2vulkan::passes::{Stage, TransformOptions};
use metal2vulkan::reflect::{ShaderReflection, SAMPLER_BINDING_RANGE};
use metal2vulkan::translate_sanitized_native_reflected;
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

/// Every `(set, binding)` a module-scope variable in `spirv` is decorated with.
///
/// Reads the crate's own disassembly, which prints raw ids and one instruction per line. A
/// descriptor decoration always targets a variable, so pairing the two decorations by target id is
/// enough; a variable carrying only one of them would not be a usable descriptor and is left out
/// rather than guessed at.
fn bindings_the_module_declares(spirv: &[u8]) -> BTreeSet<(u32, u32)> {
    let text = metal2vulkan::disassemble(spirv).expect("disassemble the translated module");
    let mut sets: Vec<(String, u32)> = Vec::new();
    let mut bindings: Vec<(String, u32)> = Vec::new();
    for line in text.lines() {
        let tokens = line.split_whitespace().collect::<Vec<_>>();
        let ["OpDecorate", target, decoration, value] = tokens.as_slice() else {
            continue;
        };
        let Ok(value) = value.parse::<u32>() else {
            continue;
        };
        match *decoration {
            "DescriptorSet" => sets.push(((*target).to_string(), value)),
            "Binding" => bindings.push(((*target).to_string(), value)),
            _ => {}
        }
    }
    bindings
        .into_iter()
        .filter_map(|(target, binding)| {
            sets.iter()
                .find(|(other, _)| *other == target)
                .map(|(_, set)| (*set, binding))
        })
        .collect()
}

/// Every `(set, binding)` reflection reports, from all three places it carries one: ordinary
/// resources, implicit imageblock attachment planes, and custom fragment imageblock members.
fn bindings_reflection_reports(reflection: &ShaderReflection) -> BTreeSet<(u32, u32)> {
    let set = reflection.descriptor_layout.set;
    let mut reported = BTreeSet::new();
    for resource in &reflection.bindings {
        if let Some(location) = &resource.descriptor {
            for index in 0..location.count {
                reported.insert((location.set, location.binding + index));
            }
        }
    }
    for attachment in &reflection.implicit_imageblock_attachments {
        reported.insert((set, attachment.binding));
    }
    for member in reflection
        .fragment_imageblock
        .iter()
        .flat_map(|imageblock| &imageblock.members)
    {
        if let Some(binding) = member.binding {
            reported.insert((set, binding));
        }
    }
    reported
}

/// Scratch for one subject. `spirv-val` writes a fixed file name inside it and these tests run
/// concurrently, so each subject gets its own directory.
fn scratch(label: &str) -> PathBuf {
    let directory = std::env::temp_dir().join(format!(
        "m2v_reflected_bindings_{}_{}",
        std::process::id(),
        label.replace(['/', '.'], "_")
    ));
    std::fs::create_dir_all(&directory).expect("scratch directory");
    directory
}

/// The bindings the module declared, after requiring reflection to cover every one of them.
fn assert_reflection_covers_declarations(
    label: &str,
    spirv: &[u8],
    reflection: &ShaderReflection,
) -> BTreeSet<(u32, u32)> {
    let declared = bindings_the_module_declares(spirv);
    let reported = bindings_reflection_reports(reflection);
    let unreported = declared.difference(&reported).collect::<Vec<_>>();
    assert!(
        unreported.is_empty(),
        "{label} decorates {unreported:?} on a variable but reflection does not report those \
         bindings, so a descriptor-set layout built from the reflection would not cover the module"
    );
    declared
}

/// A kernel that copies one texel with `texture.read(coord)`.
///
/// Metal's `read` takes no sampler. AIR still passes one, from `air.get_read_sampler()`, and the
/// read lowering discards it — so the sampler this module ends up needing, if any, is a sampler
/// nothing ever samples with.
const SAMPLER_LESS_TEXTURE_READ: &str = r#"target triple = "spirv-unknown-vulkan1.2"

define void @copy_texel(ptr addrspace(1) %src, ptr addrspace(1) %dst, <2 x i32> %gid) {
entry:
  %sampler = tail call ptr addrspace(2) @air.get_read_sampler()
  %read = tail call { <4 x half>, i8 } @air.read_texture_2d_array.v4f16(ptr addrspace(1) %src, ptr addrspace(2) %sampler, <2 x i32> %gid, i32 0, <2 x i32> zeroinitializer, i32 0, i32 0)
  %texel = extractvalue { <4 x half>, i8 } %read, 0
  tail call void @air.write_texture_2d_array.v4f16(ptr addrspace(1) %dst, <2 x i32> %gid, i32 0, <4 x half> %texel, i32 0, i32 2)
  ret void
}

declare ptr addrspace(2) @air.get_read_sampler()
declare { <4 x half>, i8 } @air.read_texture_2d_array.v4f16(ptr addrspace(1), ptr addrspace(2), <2 x i32>, i32, <2 x i32>, i32, i32)
declare void @air.write_texture_2d_array.v4f16(ptr addrspace(1), <2 x i32>, i32, <4 x half>, i32, i32)

!air.kernel = !{!0}
!0 = !{ptr @copy_texel, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 1, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d_array<half, sample>", !"air.arg_name", !"src"}
!4 = !{i32 1, !"air.texture", !"air.location_index", i32 2, i32 1, !"air.write", !"air.arg_type_name", !"texture2d_array<half, write>", !"air.arg_name", !"dst"}
!5 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint2", !"air.arg_name", !"gid"}
"#;

#[test]
fn a_sampler_less_texture_read_needs_no_sampler_descriptor() {
    let (spirv, reflection) = translate_sanitized_native_reflected(
        SAMPLER_LESS_TEXTURE_READ,
        Stage::Kernel,
        &scratch("sampler_less_texture_read"),
        TransformOptions::default(),
    )
    .expect("the read kernel translates");

    let declared = assert_reflection_covers_declarations("the read kernel", &spirv, &reflection);
    // The two textures the AIR declares, and nothing else.
    let samplers = declared
        .iter()
        .filter(|(_, binding)| SAMPLER_BINDING_RANGE.contains(*binding))
        .collect::<Vec<_>>();
    assert!(
        samplers.is_empty(),
        "the kernel reads a texture and never samples one, so it must not demand a sampler \
         descriptor; it declares {samplers:?}"
    );
    let text = metal2vulkan::disassemble(&spirv).expect("disassemble");
    assert!(
        !text.contains("OpTypeSampler"),
        "no sampler is bound, so no sampler type should survive:\n{text}"
    );
    assert_eq!(declared.len(), 2, "one sampled and one write texture");
}

/// The same contract over every committed fixture, at the stage its AIR declares.
///
/// The stage is deliberately not varied here, unlike the sweeps in `deterministic_output.rs`.
/// Reflection's stage-specific sections describe the stage the AIR declares — a kernel's tile
/// imageblock planes are reported by `from_kernel` and by nothing else — so asking what a compute
/// module reflects when it is translated as a vertex shader asks about a pairing Metal cannot
/// produce. (That pairing does expose a real reporting gap; it is not this file's subject.)
#[test]
fn every_public_fixture_reflects_every_descriptor_it_declares() {
    let mut declared = 0;
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
        let Ok((spirv, reflection)) = translate_sanitized_native_reflected(
            &source,
            stage,
            &scratch(&label),
            TransformOptions::default(),
        ) else {
            continue;
        };
        declared += assert_reflection_covers_declarations(&label, &spirv, &reflection).len();
        checked += 1;
    }
    assert!(
        checked >= 20 && declared >= 20,
        "only {checked} fixtures declaring {declared} bindings were inspected, so this swept \
         almost nothing"
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
