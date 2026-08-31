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
//! The same asymmetry applies to how MANY descriptors a binding holds. `docs/REFLECTION.md` tells a
//! consumer to group entries by `(set, binding, descriptor type)` and take the largest reported
//! `count`; that number has to reach the largest array the module declares there, or the layout is
//! smaller than the array the shader indexes into. The last test checks that direction.
//!
//! Two of the cases below are where the direction went wrong, and they are the same defect twice.
//! `air.get_read_sampler()` exists because AIR threads a sampler pointer into `texture.read(coord)`,
//! which in Metal is sampler-less; `air.get_null_texture_*()` exists because a function-constant
//! gated optional attachment still has to yield a texture handle. The interface pass materialized a
//! real `OpTypeSampler` / `OpTypeImage` descriptor so each value would be typed, and the consumer of
//! that value then turned out to be nothing at all. No Metal argument corresponds to either, so
//! reflection had nothing to report — while the module demanded the binding anyway.

use metal2vulkan::meta::TextureDimension;
use metal2vulkan::passes::{Stage, TransformOptions};
use metal2vulkan::reflect::{
    ResourceKind, ShaderReflection, SAMPLER_BINDING_RANGE, TEXTURE_BINDING_RANGE,
};
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

/// A kernel that asks whether a placeholder texture is bound, and does nothing else with it.
///
/// `air.get_null_texture_2d()` is what a function-constant-gated optional attachment resolves to
/// once the constants are folded off. `air.is_null_texture` on that handle folds to a constant, so
/// the image the interface pass materialized to type the handle ends up with no consumer -- and a
/// descriptor with no consumer is a demand on the pipeline layout for a texture the shader never
/// reads. `native_get_null_texture_models_unmodeled_placeholder` is the other half: a placeholder
/// something DOES read keeps its descriptor.
const UNCONSUMED_NULL_TEXTURE: &str = r#"target triple = "spirv-unknown-vulkan1.2"

define void @probe_optional_attachment(ptr addrspace(1) %out) {
entry:
  %tex = call ptr addrspace(1) @air.get_null_texture_2d()
  %isnull = call i1 @air.is_null_texture_2d(ptr addrspace(1) %tex)
  %flag = zext i1 %isnull to i32
  store i32 %flag, ptr addrspace(1) %out, align 4
  ret void
}

declare ptr addrspace(1) @air.get_null_texture_2d()
declare i1 @air.is_null_texture_2d(ptr addrspace(1))

!air.kernel = !{!0}
!0 = !{ptr @probe_optional_attachment, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.buffer", !"air.buffer_size", i32 4, !"air.location_index", i32 0, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
"#;

#[test]
fn a_placeholder_texture_nothing_reads_needs_no_texture_descriptor() {
    let (spirv, reflection) = translate_sanitized_native_reflected(
        UNCONSUMED_NULL_TEXTURE,
        Stage::Kernel,
        &scratch("unconsumed_null_texture"),
        TransformOptions::default(),
    )
    .expect("the optional-attachment probe translates");

    let declared =
        assert_reflection_covers_declarations("the optional-attachment probe", &spirv, &reflection);
    let textures = declared
        .iter()
        .filter(|(_, binding)| TEXTURE_BINDING_RANGE.contains(*binding))
        .collect::<Vec<_>>();
    assert!(
        textures.is_empty(),
        "the kernel only asks whether the placeholder is bound and never reads it, so it must not \
         demand a texture descriptor; it declares {textures:?}"
    );
    let text = metal2vulkan::disassemble(&spirv).expect("disassemble");
    assert!(
        !text.contains("OpTypeImage"),
        "no image is bound, so no image type should survive:\n{text}"
    );
    assert_eq!(declared.len(), 1, "the one output buffer");
}

/// A kernel that queries the size of the placeholder texture, so something does read through it.
///
/// This is the other half of `UNCONSUMED_NULL_TEXTURE`: the value has a consumer, so the descriptor
/// is not retracted and the consumer of the shader has to bind an image at it. No Metal argument
/// corresponds to that binding, so nothing in the AIR metadata could describe it -- reflection
/// reports it from the finished module instead.
const CONSUMED_NULL_TEXTURE: &str = r#"target triple = "spirv-unknown-vulkan1.2"

define void @measure_optional_attachment(ptr addrspace(1) %out) {
entry:
  %tex = call ptr addrspace(1) @air.get_null_texture_2d()
  %width = call i32 @air.get_width_texture_2d(ptr addrspace(1) %tex, i32 0)
  store i32 %width, ptr addrspace(1) %out, align 4
  ret void
}

declare ptr addrspace(1) @air.get_null_texture_2d()
declare i32 @air.get_width_texture_2d(ptr addrspace(1), i32)

!air.kernel = !{!0}
!0 = !{ptr @measure_optional_attachment, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.buffer", !"air.buffer_size", i32 4, !"air.location_index", i32 0, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
"#;

#[test]
fn a_placeholder_texture_the_shader_reads_is_reported_as_a_descriptor() {
    let (spirv, reflection) = translate_sanitized_native_reflected(
        CONSUMED_NULL_TEXTURE,
        Stage::Kernel,
        &scratch("consumed_null_texture"),
        TransformOptions::default(),
    )
    .expect("the optional-attachment measurement translates");

    let declared = assert_reflection_covers_declarations(
        "the optional-attachment measurement",
        &spirv,
        &reflection,
    );
    let placeholder = reflection
        .bindings
        .iter()
        .find(|resource| resource.kind == ResourceKind::SynthesizedNullTexture)
        .expect("the placeholder the module reads through is reported");
    let location = placeholder.descriptor.expect("it consumes a descriptor");
    assert!(
        TEXTURE_BINDING_RANGE.contains(location.binding),
        "a placeholder image belongs in the sampled-texture band, not at {}",
        location.binding
    );
    assert!(
        declared.contains(&(location.set, location.binding)),
        "the reported binding is one the module decorates; module declares {declared:?}"
    );
    let shape = placeholder
        .texture_shape
        .expect("a consumer needs the shape of the image it has to bind");
    assert_eq!(shape.dimension, TextureDimension::D2);
    assert!(!shape.arrayed && !shape.writable);
}

/// A fragment shader that reads its render target back through the implicit imageblock.
///
/// `air.load.implicit_imageblock.*` is detected from the module body, not from the stage, and the
/// interface pass materializes a descriptor-backed plane wherever it lowers one. Reflection has to
/// describe that plane whichever stage declared the entry.
const FRAGMENT_IMPLICIT_IMAGEBLOCK: &str = r#"define <2 x half> @read_back_render_target(<4 x float> %position) {
entry:
  %value = call <2 x half> @air.load.implicit_imageblock.v2f16(i32 0, <2 x i16> zeroinitializer, i32 0, i16 0)
  ret <2 x half> %value
}

declare <2 x half> @air.load.implicit_imageblock.v2f16(i32, <2 x i16>, i32, i16)

!air.fragment = !{!0}
!0 = !{ptr @read_back_render_target, !1, !2}
!1 = !{!3}
!2 = !{!4}
!3 = !{i32 0, !"air.render_target", !"air.location_index", i32 0, i32 0, !"air.arg_type_name", !"half2"}
!4 = !{i32 0, !"air.position", !"air.center", !"air.no_perspective", !"air.arg_type_name", !"float4", !"air.arg_name", !"position"}
"#;

#[test]
fn a_fragment_shader_reflects_the_imageblock_plane_it_declares() {
    let (spirv, reflection) = translate_sanitized_native_reflected(
        FRAGMENT_IMPLICIT_IMAGEBLOCK,
        Stage::Fragment,
        &scratch("fragment_implicit_imageblock"),
        TransformOptions::default(),
    )
    .expect("the render-target read-back translates");

    let declared = assert_reflection_covers_declarations(
        "the render-target read-back fragment",
        &spirv,
        &reflection,
    );
    assert_eq!(
        reflection.implicit_imageblock_attachments.len(),
        1,
        "the shader loads one implicit imageblock plane"
    );
    assert!(
        declared.contains(&(
            reflection.descriptor_layout.set,
            reflection.implicit_imageblock_attachments[0].binding
        )),
        "the reported plane is the binding the module decorates; module declares {declared:?}"
    );
}

/// A kernel that indexes a runtime-sized texture argument, which binds as a descriptor ARRAY.
///
/// `array_ref<texture2d<...>>` has no fixed length in AIR, so the module declares the descriptor-ABI
/// capacity and reflection reports the same number as its `count`. A consumer that sized the layout
/// from a smaller count would build one the shader indexes past.
const RUNTIME_TEXTURE_ARRAY: &str = r#"target triple = "spirv-unknown-vulkan1.2"
%"struct.metal::texture2d" = type { ptr addrspace(1) }

define void @k(ptr readonly captures(none) %imgs, ptr addrspace(1) %out) {
entry:
  %tex = load ptr addrspace(1), ptr %imgs, align 8
  %w = tail call i32 @air.get_width_texture_2d(ptr addrspace(1) %tex, i32 0)
  store i32 %w, ptr addrspace(1) %out, align 4
  ret void
}
declare i32 @air.get_width_texture_2d(ptr addrspace(1), i32)
!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"array_ref<texture2d<float, sample>>", !"air.arg_name", !"imgs"}
!4 = !{i32 1, !"air.buffer", !"air.buffer_size", i32 4, !"air.location_index", i32 0, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
"#;

#[test]
fn a_descriptor_array_reports_at_least_as_many_descriptors_as_the_module_declares() {
    let (spirv, reflection) = translate_sanitized_native_reflected(
        RUNTIME_TEXTURE_ARRAY,
        Stage::Kernel,
        &scratch("runtime_texture_array"),
        TransformOptions::default(),
    )
    .expect("the descriptor-array kernel translates");

    assert_reflection_covers_declarations("the descriptor-array kernel", &spirv, &reflection);
    let declared = array_lengths_the_module_declares(&spirv);
    assert!(
        declared.values().any(|length| *length > 1),
        "this fixture is only meaningful if the module declares a descriptor array; got {declared:?}"
    );
    assert_counts_cover_declared_arrays("the descriptor-array kernel", &spirv, &reflection);
}

/// Require the largest reported `count` at each binding to reach the largest array the module
/// declares there, and return how many bindings carried an array.
fn assert_counts_cover_declared_arrays(
    label: &str,
    spirv: &[u8],
    reflection: &ShaderReflection,
) -> usize {
    let mut reported = std::collections::BTreeMap::<u32, u32>::new();
    for location in reflection
        .bindings
        .iter()
        .filter_map(|resource| resource.descriptor)
    {
        let slot = reported.entry(location.binding).or_default();
        *slot = (*slot).max(location.count);
    }
    let declared = array_lengths_the_module_declares(spirv);
    let mut checked = 0;
    for (binding, length) in &declared {
        // A binding the module declares but reflection does not report is the other direction, and
        // `assert_reflection_covers_declarations` is what says so.
        let Some(count) = reported.get(binding) else {
            continue;
        };
        assert!(
            count >= length,
            "{label} reports at most {count} descriptor(s) at binding {binding}, but the module \
             declares an array of {length} there, so a layout built from the reflection is smaller \
             than the array the shader indexes"
        );
        checked += 1;
    }
    checked
}

/// The largest descriptor-array length the module declares at each binding; `1` for a scalar.
///
/// Reads the crate's own disassembly, which prints raw ids and one instruction per line.
fn array_lengths_the_module_declares(spirv: &[u8]) -> std::collections::BTreeMap<u32, u32> {
    let text = metal2vulkan::disassemble(spirv).expect("disassemble the translated module");
    let mut constants: Vec<(String, u32)> = Vec::new();
    let mut arrays: Vec<(String, String)> = Vec::new();
    let mut pointees: Vec<(String, String)> = Vec::new();
    let mut variables: Vec<(String, String)> = Vec::new();
    let mut bindings: Vec<(String, u32)> = Vec::new();
    for line in text.lines() {
        let tokens = line.split_whitespace().collect::<Vec<_>>();
        match tokens.as_slice() {
            ["OpDecorate", target, "Binding", value] => {
                if let Ok(value) = value.parse::<u32>() {
                    bindings.push(((*target).to_string(), value));
                }
            }
            [result, "=", "OpConstant", _, value] => {
                if let Ok(value) = value.parse::<u32>() {
                    constants.push(((*result).to_string(), value));
                }
            }
            [result, "=", "OpTypeArray", _, length] => {
                arrays.push(((*result).to_string(), (*length).to_string()));
            }
            [result, "=", "OpTypePointer", _, pointee] => {
                pointees.push(((*result).to_string(), (*pointee).to_string()));
            }
            [result, "=", "OpVariable", pointer, _] => {
                variables.push(((*result).to_string(), (*pointer).to_string()));
            }
            _ => {}
        }
    }
    let find = |pairs: &Vec<(String, String)>, key: &str| {
        pairs
            .iter()
            .find(|(id, _)| id == key)
            .map(|(_, value)| value.clone())
    };
    let mut declared = std::collections::BTreeMap::<u32, u32>::new();
    for (variable, binding) in bindings {
        let Some(pointer) = find(&variables, &variable) else {
            continue;
        };
        let Some(pointee) = find(&pointees, &pointer) else {
            continue;
        };
        let length = match find(&arrays, &pointee) {
            // A runtime array has no length to compare against; it is bounded by the layout, which
            // is what the count reports.
            Some(length) => match constants.iter().find(|(id, _)| *id == length) {
                Some((_, value)) => *value,
                None => continue,
            },
            None => 1,
        };
        let slot = declared.entry(binding).or_insert(1);
        *slot = (*slot).max(length);
    }
    declared
}

/// The same contract over every committed fixture, at the stage its AIR declares.
///
/// The stage is not varied here, unlike the sweeps in `deterministic_output.rs`. Reflection is
/// parsed from the stage metadata the AIR carries, so asking what a module reflects at a stage it
/// does not declare asks about a description that has no source — the parse returns nothing and the
/// reflection is a stub, whatever the module happens to contain.
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
        assert_counts_cover_declared_arrays(&label, &spirv, &reflection);
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
