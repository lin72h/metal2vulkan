//! The same AIR must translate to the same bytes.
//!
//! This is not a nicety. The validation ladder compares a candidate's output by SHA-256, consumers
//! cache translated SPIR-V by content hash, and a bug report is only reproducible if the module
//! that produced it can be produced again. A translator whose output depends on hash-map iteration
//! order breaks all three, and it does so silently: every run is individually correct, valid, and
//! passes `spirv-val`.
//!
//! Rust seeds each `HashMap`'s hasher differently, including two maps built in the same process, so
//! translating one input twice in one test is enough to expose an emission decision that follows
//! iteration order. Both tests below do exactly that.

use metal2vulkan::passes::{Stage, TransformOptions};
use metal2vulkan::reflect::ResourceKind;
use metal2vulkan::{translate_sanitized_native, translate_sanitized_native_reflected};
use std::path::{Path, PathBuf};

/// Scratch for one subject. Each gets its own directory: `spirv-val` writes a fixed file name
/// inside it, and the tests in this file run concurrently.
fn scratch(label: &str) -> PathBuf {
    let directory = std::env::temp_dir().join(format!(
        "m2v_determinism_{}_{}",
        std::process::id(),
        label.replace(['/', '.'], "_")
    ));
    std::fs::create_dir_all(&directory).expect("scratch directory");
    directory
}

/// How many times each input is translated. Iteration order is a permutation, so a two-element
/// collection agrees half the time by chance; repeating shrinks that to nothing.
const RUNS: usize = 6;

fn assert_deterministic(label: &str, source: &str, stage: Stage) {
    let tmp = scratch(label);
    let first = translate_sanitized_native(source, stage, &tmp)
        .unwrap_or_else(|error| panic!("{label} must translate to be checked: {error}"));
    for run in 1..RUNS {
        let again = translate_sanitized_native(source, stage, &tmp)
            .unwrap_or_else(|error| panic!("{label} run {run}: {error}"));
        assert_eq!(
            again.len(),
            first.len(),
            "{label} translated to {} bytes and then to {} bytes",
            first.len(),
            again.len()
        );
        assert!(
            again == first,
            "{label} translated to different bytes on run {run}; some emission decision is \
             following hash-map iteration order rather than a stable order of the input"
        );
    }
    let _ = std::fs::remove_dir_all(&tmp);
}

/// A kernel whose buffers are addressed by device address.
///
/// The entry prologue materializes one address per direct buffer parameter, and the position each
/// one lands at fixes which address-table slot the module reads there. That list used to be built
/// by iterating the emitter's raw-offset map, so the prologue came out in a different order on
/// every run — a semantically equivalent module with different bytes. Four buffers make an
/// accidental agreement between two runs vanishingly unlikely.
///
/// The AIR is written here rather than added to `validation/fixtures/public/`, because a fixture
/// there is expected to have a Metal source to qualify against and this one is authored directly.
const DEVICE_ADDRESS_KERNEL: &str = r#"
target triple = "air64_v28-apple-macosx26.5.0"

%Words = type { [4 x i32] }
%Handles = type { ptr addrspace(1), ptr addrspace(1) }

define void @k(ptr addrspace(1) %a, ptr addrspace(1) %b, ptr addrspace(1) %c, ptr addrspace(2) %handles) {
entry:
  %hp0 = getelementptr inbounds %Handles, ptr addrspace(2) %handles, i64 0, i32 0
  %h0 = load ptr addrspace(1), ptr addrspace(2) %hp0
  %hv0 = load i32, ptr addrspace(1) %h0
  %ap = getelementptr inbounds %Words, ptr addrspace(1) %a, i64 0, i32 0, i64 0
  %av = load i32, ptr addrspace(1) %ap
  %bp = getelementptr inbounds %Words, ptr addrspace(1) %b, i64 0, i32 0, i64 0
  %bv = load i32, ptr addrspace(1) %bp
  %s = add i32 %av, %bv
  %t = add i32 %s, %hv0
  %cp = getelementptr inbounds %Words, ptr addrspace(1) %c, i64 0, i32 0, i64 0
  store i32 %t, ptr addrspace(1) %cp
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4, !6, !7}
!3 = !{i32 0, !"air.buffer", !"air.buffer_size", i32 16, !"air.struct_type_info", !5, !"air.location_index", i32 0, i32 1, !"air.read", !"air.arg_type_name", !"Words", !"air.arg_name", !"a"}
!4 = !{i32 1, !"air.buffer", !"air.buffer_size", i32 16, !"air.struct_type_info", !5, !"air.location_index", i32 1, i32 1, !"air.read", !"air.arg_type_name", !"Words", !"air.arg_name", !"b"}
!6 = !{i32 2, !"air.buffer", !"air.buffer_size", i32 16, !"air.struct_type_info", !5, !"air.location_index", i32 2, i32 1, !"air.read_write", !"air.arg_type_name", !"Words", !"air.arg_name", !"c"}
!7 = !{i32 3, !"air.buffer", !"air.buffer_size", i32 16, !"air.struct_type_info", !5, !"air.location_index", i32 3, i32 1, !"air.read", !"air.arg_type_name", !"Handles", !"air.arg_name", !"handles"}
!5 = !{i32 0, i32 16, i32 0, !"uint", !"v0", i32 4, i32 4, i32 0, !"uint", !"v1", i32 8, i32 4, i32 0, !"uint", !"v2", i32 12, i32 4, i32 0, !"uint", !"v3"}
"#;

/// The device-address prologue is the path this test exists for; assert it is actually taken, so a
/// change that stops using an address table cannot leave the test below passing over nothing.
#[test]
fn the_device_address_kernel_really_reads_an_address_table() {
    let (_, reflection) = translate_sanitized_native_reflected(
        DEVICE_ADDRESS_KERNEL,
        Stage::Kernel,
        &scratch("address-table-probe"),
        TransformOptions::default(),
    )
    .expect("the device-address kernel translates");
    assert!(
        reflection
            .bindings
            .iter()
            .any(|binding| binding.kind == ResourceKind::BufferAddressTable),
        "the kernel no longer lowers to a buffer-address table, so the determinism test below is \
         no longer covering the prologue it was written for"
    );
}

#[test]
fn a_device_address_kernel_translates_to_the_same_bytes_every_time() {
    assert_deterministic(
        "the device-address kernel",
        DEVICE_ADDRESS_KERNEL,
        Stage::Kernel,
    );
}

/// A kernel whose only resource parameter is never bound.
///
/// Without `!air.kernel` metadata naming it, `%u` has no descriptor to bind to, so the pipeline
/// re-classes it into a null-initialized Private placeholder root and rewrites every load from it
/// into a copy of that type's zero. One `OpConstantNull` is minted per distinct load result type,
/// and the order they are minted in is the order they are declared in the module.
///
/// That order used to come from a `HashSet` of the result types, so this kernel's three nulls came
/// out in a different permutation on every run. Three distinct load types make an accidental
/// agreement between two runs a one-in-six event, which `RUNS` repetitions drive to nothing.
const UNBOUND_BUFFER_KERNEL: &str = r#"
target triple = "air64_v28-apple-macosx26.5.0"

%Uniforms = type { <2 x float>, <3 x float>, <4 x float> }

define void @k(ptr addrspace(1) %out, ptr addrspace(2) %u) {
entry:
  %p2 = getelementptr inbounds %Uniforms, ptr addrspace(2) %u, i64 0, i32 0
  %v2 = load <2 x float>, ptr addrspace(2) %p2
  %p3 = getelementptr inbounds %Uniforms, ptr addrspace(2) %u, i64 0, i32 1
  %v3 = load <3 x float>, ptr addrspace(2) %p3
  %p4 = getelementptr inbounds %Uniforms, ptr addrspace(2) %u, i64 0, i32 2
  %v4 = load <4 x float>, ptr addrspace(2) %p4
  %a = fadd <2 x float> %v2, %v2
  %b = fadd <3 x float> %v3, %v3
  %c = fadd <4 x float> %v4, %v4
  %s0 = extractelement <2 x float> %a, i64 0
  %s1 = extractelement <3 x float> %b, i64 0
  %s2 = extractelement <4 x float> %c, i64 0
  %t0 = fadd float %s0, %s1
  %t1 = fadd float %t0, %s2
  store float %t1, ptr addrspace(1) %out
  ret void
}
"#;

/// The zero-root rewrite is the path this test exists for; assert the module really takes it, so a
/// change that starts binding `%u` cannot leave the test below passing over nothing.
#[test]
fn the_unbound_buffer_kernel_really_zeroes_its_absent_resource() {
    let bytes = translate_sanitized_native(
        UNBOUND_BUFFER_KERNEL,
        Stage::Kernel,
        &scratch("unbound-buffer-probe"),
    )
    .expect("the unbound-buffer kernel translates");
    let text = metal2vulkan::disassemble(&bytes).expect("disassemble");
    // Each rewritten load is an `OpCopyObject` of an `OpConstantNull` — three loads, three distinct
    // types, so three nulls minted in one pass. (The module declares a fourth null elsewhere, as the
    // initializer of the Private placeholder variable itself, which this rewrite does not mint.)
    let nulls = text
        .lines()
        .filter_map(|line| line.split_once(" = OpConstantNull"))
        .map(|(id, _)| id.trim().to_string())
        .collect::<Vec<_>>();
    let copied_nulls = text
        .lines()
        .filter(|line| line.contains("OpCopyObject"))
        .filter(|line| {
            nulls
                .iter()
                .any(|null| line.split_whitespace().any(|word| word == null))
        })
        .count();
    assert_eq!(
        copied_nulls, 3,
        "expected the absent resource's three distinct load types to each become a copy of that \
         type's null, got {copied_nulls}; the determinism test below no longer covers the mint \
         order it was written for\n{text}"
    );
}

#[test]
fn an_unbound_buffer_kernel_translates_to_the_same_bytes_every_time() {
    assert_deterministic(
        "the unbound-buffer kernel",
        UNBOUND_BUFFER_KERNEL,
        Stage::Kernel,
    );
}

/// The same contract across every committed fixture, so an emission decision that starts following
/// iteration order anywhere in the pipeline fails here rather than in a consumer's cache.
#[test]
fn every_public_fixture_translates_to_the_same_bytes_every_time() {
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
        // A fixture that does not translate has nothing to compare; the suites that own those
        // inputs report on them.
        if translate_sanitized_native(&source, stage, &scratch(&label)).is_err() {
            continue;
        }
        assert_deterministic(&label, &source, stage);
        checked += 1;
    }
    assert!(
        checked >= 20,
        "only {checked} public fixtures translated, so this swept almost nothing"
    );
}

/// Determinism does not depend on the stage being the one the AIR declares.
///
/// `translate_sanitized_native` takes the stage as a parameter, so the stage is an input like any
/// other and the output has to be a function of it. Translating each fixture at every stage reaches
/// pipeline paths the declared stage never does — a buffer parameter that the declared stage binds
/// to a descriptor becomes an unbound Private placeholder under another, which is exactly the
/// rewrite the kernel above was reduced from. Anything that does not translate under a given stage
/// is skipped: this test is about byte stability, not about which stages accept which module.
#[test]
fn every_public_fixture_is_stable_under_every_stage_it_translates_under() {
    let mut checked = 0;
    for path in public_fixtures() {
        let source = std::fs::read_to_string(&path).expect("read fixture");
        let name = path
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_default();
        for stage in [Stage::Vertex, Stage::Fragment, Stage::Kernel] {
            let label = format!("{name}-{stage:?}");
            if translate_sanitized_native(&source, stage, &scratch(&label)).is_err() {
                continue;
            }
            assert_deterministic(&label, &source, stage);
            checked += 1;
        }
    }
    assert!(
        checked >= 30,
        "only {checked} fixture/stage pairs translated, so this swept almost nothing"
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
