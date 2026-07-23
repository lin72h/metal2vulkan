#[cfg(test)]
use crate::texture::texture_kind_from_type_name;
use crate::texture::{
    fragment_writes_depth, texture_kind, texture_layer_count, texture_output_extent,
    texture_seed_bytes, texture_seed_extent, TextureKind,
};
use crate::{
    scratch_dir_for, seeded_buffer_bytes, seeded_render_target_bytes, BlendMode, DataFormat,
    Extent3d, Inputs, Output, Stage, TextureRole,
};
use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine as _;
use core::ffi::c_void;
use core::ptr::NonNull;
use objc2::rc::{autoreleasepool, Retained};
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSArray, NSString, NSURL};
use objc2_metal::{
    MTLAttributeFormat, MTLBlendFactor, MTLBlendOperation, MTLBuffer, MTLCommandBuffer,
    MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue, MTLCompareFunction,
    MTLComputeCommandEncoder, MTLComputePipelineDescriptor, MTLComputePipelineState,
    MTLCreateSystemDefaultDevice, MTLDepthStencilDescriptor, MTLDepthStencilState, MTLDevice,
    MTLFunction, MTLFunctionConstantValues, MTLFunctionDescriptor, MTLLibrary, MTLLinkedFunctions,
    MTLLoadAction, MTLOrigin, MTLPipelineOption, MTLPixelFormat, MTLPrimitiveType, MTLRegion,
    MTLRenderCommandEncoder, MTLRenderPassDescriptor, MTLRenderPipelineDescriptor,
    MTLResourceOptions, MTLResourceUsage, MTLSamplerDescriptor, MTLSamplerState, MTLSize,
    MTLStageInputOutputDescriptor, MTLStepFunction, MTLStorageMode, MTLStoreAction, MTLTexture,
    MTLTextureDescriptor, MTLTextureType, MTLTextureUsage, MTLVertexDescriptor, MTLVertexFormat,
    MTLVertexStepFunction,
};
use std::fs;
use std::path::Path;
use std::process::Command;

#[link(name = "CoreGraphics", kind = "framework")]
extern "C" {}

type MetalBuffer = Retained<ProtocolObject<dyn MTLBuffer>>;
type MetalComputePipeline = Retained<ProtocolObject<dyn MTLComputePipelineState>>;
type MetalDepthStencilState = Retained<ProtocolObject<dyn MTLDepthStencilState>>;
type MetalSampler = Retained<ProtocolObject<dyn MTLSamplerState>>;
type MetalTexture = Retained<ProtocolObject<dyn MTLTexture>>;

const WORKGROUP_MEMORY_ELEMENTS: usize = 512;

const VALIDATION_VERTEX_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct Metal2VulkanValidationVertexOut {
    float4 position [[position]];
    float2 coord [[user(coord)]];
};

vertex Metal2VulkanValidationVertexOut metal2vulkan_validation_fullscreen_vertex(uint vid [[vertex_id]]) {
    float2 positions[3] = {
        float2(-1.0, -1.0),
        float2( 3.0, -1.0),
        float2(-1.0,  3.0),
    };
    float2 coords[3] = {
        float2(0.0, 0.0),
        float2(2.0, 0.0),
        float2(0.0, 2.0),
    };
    Metal2VulkanValidationVertexOut out;
    out.position = float4(positions[vid], 0.0, 1.0);
    out.coord = coords[vid];
    return out;
}
"#;

const VALIDATION_FRAGMENT_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

fragment half4 metal2vulkan_validation_empty_fragment() {
    return half4(0.0);
}
"#;

pub fn execute_metallib_blob(
    blob_b64: &str,
    function_name: &str,
    stage: Stage,
    inputs: &crate::Inputs,
    sanitized_ll: &str,
    source_metallib: Option<&Path>,
) -> Vec<u8> {
    autoreleasepool(|_| {
        let tmp = scratch_dir_for("oracle-metallib");
        let air_path = tmp.join("case.air");
        let metallib_path = tmp.join("case.metallib");
        let blob = BASE64
            .decode(blob_b64.as_bytes())
            .unwrap_or_else(|e| panic!("decode AIR blob for {function_name}: {e}"));
        fs::write(&air_path, blob).unwrap_or_else(|e| panic!("write {}: {e}", air_path.display()));
        let mut effective_name = function_name.to_string();
        // One case per `--oneshot` worker; reset the per-thread compare mode before we classify.
        set_oracle_compare(OracleCompare::Full);

        // Structural infinite-loop guard. A committed Metal command buffer CANNOT be cancelled —
        // an unbounded compute loop pins the GPU until the machine is rebooted (killing the CPU
        // worker does not stop it), and no choice of input data can prevent that (halting problem).
        // So we bound GPU work BEFORE `commit()`: disassemble to Apple-dialect IR, compose the
        // existing threadgroup zero-init, then classify. Loop-free kernels take the byte-identical
        // fast path; loopy kernels are instrumented with a per-thread back-edge budget so no
        // dispatch can run unbounded; anything we cannot instrument-and-verify is quarantined.
        //
        // Deterministic threadgroup memory (module addrspace(3) globals / addrspace(3) kernel args)
        // still gets a zero-fill prologue injected through the same textual round-trip, so reads of
        // never-written threadgroup slots see ZERO instead of leftover GPU memory — the same defined
        // refinement metal2vulkan's Workgroup zero-init pass applies on the candidate side.
        let module_text = disassembled_module_text(&air_path)
            .unwrap_or_else(|e| panic!("m2v-quarantine: metal-objdump failed: {e}"));
        let zero_init = if sanitized_ll.contains("addrspace(3)") {
            insert_threadgroup_zero_init(&module_text, function_name)
        } else {
            None
        };
        let analysis_text = zero_init.as_deref().unwrap_or(&module_text);

        match crate::loop_budget::classify_and_instrument(analysis_text, function_name) {
            crate::loop_budget::GuardPlan::Quarantine(reason) => {
                // Never submit work we cannot prove bounded. Surfaced to the driver via the
                // `m2v-quarantine:` panic sentinel (caught by `catch_oracle_unwind`).
                panic!("m2v-quarantine: {reason}");
            }
            crate::loop_budget::GuardPlan::Instrumented(text) => {
                // Candidate runners use compare=none to apply the same loop-budget transform before
                // translating their SPIR-V, so the golden and candidate both cover bounded work.
                set_oracle_compare(OracleCompare::MetalOnly);
                match link_module_text_with_rename_retry(
                    &text,
                    &air_path,
                    &metallib_path,
                    function_name,
                    "case_guarded",
                ) {
                    Ok(entry) => effective_name = entry,
                    // A rejection means our instrumentation produced IR metal-as won't accept —
                    // quarantine rather than submit (the round-trip verifies the transform).
                    Err(e) => panic!("m2v-quarantine: instrumented metallib rejected: {e}"),
                }
            }
            crate::loop_budget::GuardPlan::LoopFree => {
                if let Some(text) = zero_init {
                    match link_module_text_with_rename_retry(
                        &text,
                        &air_path,
                        &metallib_path,
                        function_name,
                        "case_zeroinit",
                    ) {
                        Ok(entry) => effective_name = entry,
                        Err(e) => panic!("threadgroup zero-init relink failed: {e}"),
                    }
                } else if let Err(e) = command_stdout(
                    "xcrun",
                    &[
                        "metallib",
                        air_path.to_str().expect("AIR scratch path is not UTF-8"),
                        "-o",
                        metallib_path
                            .to_str()
                            .expect("metallib scratch path is not UTF-8"),
                    ],
                ) {
                    // `xcrun metallib` injects stdlib libcall symbols while linking; a kernel whose
                    // entry symbol collides with one of them (e.g. a kernel literally named `memcpy`)
                    // dies with "LLVM ERROR: multiple symbols ('<sym>')!". Retry by renaming the
                    // colliding symbol (taken from the error message, never from a name table)
                    // in Apple's own textual IR dialect and reassembling.
                    let Some(sym) = multiple_symbols_name(&e) else {
                        panic!("xcrun failed: {e}");
                    };
                    effective_name = rebuild_metallib_with_renamed_symbol(
                        &air_path,
                        &metallib_path,
                        &sym,
                        function_name,
                    )
                    .unwrap_or_else(|retry_err| {
                        panic!("xcrun failed: {e}; rename retry for {sym:?} failed: {retry_err}")
                    });
                    eprintln!(
                        "[oracle] metallib symbol collision on {sym:?}; relinked with renamed entry {effective_name:?}"
                    );
                }
            }
        }

        let device =
            MTLCreateSystemDefaultDevice().expect("MTLCreateSystemDefaultDevice returned nil");
        let library_path = if effective_name == function_name
            && sanitized_ll.contains("air.visible_function_references")
        {
            source_metallib.unwrap_or(&metallib_path)
        } else {
            &metallib_path
        };
        let library = load_library_from_url(&device, library_path);
        match stage {
            Stage::Kernel => execute_compute_library(
                &device,
                &library,
                &effective_name,
                inputs,
                Some(sanitized_ll),
            ),
            Stage::Fragment => execute_render_fragment_library(
                &device,
                &library,
                &effective_name,
                inputs,
                Some(sanitized_ll),
            ),
            Stage::Vertex => execute_vertex_library(
                &device,
                &library,
                &effective_name,
                inputs,
                Some(sanitized_ll),
            ),
        }
    })
}

/// Parse the colliding symbol out of a Metal linker "LLVM ERROR: multiple symbols ('<sym>')!"
/// failure. Returns None for any other failure shape.
fn multiple_symbols_name(err: &str) -> Option<String> {
    let start = err.find("multiple symbols ('")? + "multiple symbols ('".len();
    let rest = &err[start..];
    let end = rest.find("')")?;
    let sym = &rest[..end];
    (!sym.is_empty()).then(|| sym.to_string())
}

/// Retry path for a metallib symbol collision: disassemble the AIR object with Apple's own
/// `metal-objdump` (its textual dialect round-trips through `metal-as`, unlike Homebrew
/// `llvm-dis` output), rename every reference to the colliding symbol, reassemble, and relink.
/// Returns the renamed entry-point name to execute.
fn rebuild_metallib_with_renamed_symbol(
    air_path: &Path,
    metallib_path: &Path,
    sym: &str,
    function_name: &str,
) -> Result<String, String> {
    let text = disassembled_module_text(air_path)?;
    let (renamed, entry) = rename_symbol_in_module_text(&text, sym, function_name);
    assemble_and_link(&renamed, air_path, metallib_path, "case_renamed")?;
    Ok(entry)
}

/// Assemble+link Apple-dialect module `text` into `metallib_path`, retrying the stdlib libcall
/// symbol-collision rename once (the same collision `xcrun metallib` hits). `Ok(entry)` returns the
/// entry name to execute (renamed when the entry itself collided). A non-collision rejection is
/// returned as `Err` so the caller can decide: the zero-init path treats it as a hard error, the
/// loop-budget path treats it as a quarantine (the round-trip IS the verification of the transform).
fn link_module_text_with_rename_retry(
    text: &str,
    air_path: &Path,
    metallib_path: &Path,
    function_name: &str,
    stem: &str,
) -> Result<String, String> {
    match assemble_and_link(text, air_path, metallib_path, stem) {
        Ok(()) => Ok(function_name.to_string()),
        Err(e) => {
            let Some(sym) = multiple_symbols_name(&e) else {
                return Err(e);
            };
            let (renamed, entry) = rename_symbol_in_module_text(text, &sym, function_name);
            assemble_and_link(
                &renamed,
                air_path,
                metallib_path,
                &format!("{stem}_renamed"),
            )
            .map_err(|retry_err| format!("{e}; rename retry for {sym:?} failed: {retry_err}"))?;
            eprintln!(
                "[oracle] metallib symbol collision on {sym:?}; relinked with renamed entry {entry:?}"
            );
            Ok(entry)
        }
    }
}

/// How a candidate backend (`corpus-run-vulkan` / `-moltenvk`) should prepare this oracle golden.
/// Set per-case by [`execute_metallib_blob`]: a kernel that needed loop-budget instrumentation is
/// [`OracleCompare::MetalOnly`] because candidate runners must translate a guarded LL copy before
/// dispatching on Vulkan / MoltenVK.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OracleCompare {
    Full,
    MetalOnly,
}

thread_local! {
    static LAST_ORACLE_COMPARE: std::cell::Cell<OracleCompare> =
        const { std::cell::Cell::new(OracleCompare::Full) };
}

/// The compare mode selected by the most recent [`execute_metallib_blob`] on this thread. Corpus
/// runs one case per `--oneshot` worker, so there is no cross-case bleed.
pub fn last_oracle_compare_mode() -> OracleCompare {
    LAST_ORACLE_COMPARE.with(|c| c.get())
}

fn set_oracle_compare(mode: OracleCompare) {
    LAST_ORACLE_COMPARE.with(|c| c.set(mode));
}

/// Apple's textual module dump for an AIR object: `metal-objdump --disassemble-all` with the
/// banner lines dropped (the module text starts at `; ModuleID`). This dialect round-trips
/// through `metal-as`, unlike Homebrew `llvm-dis` output.
fn disassembled_module_text(air_path: &Path) -> Result<String, String> {
    let air = air_path.to_str().ok_or("AIR scratch path is not UTF-8")?;
    let dis = command_stdout("xcrun", &["metal-objdump", "--disassemble-all", air])?;
    let module_start = dis
        .find("; ModuleID")
        .ok_or("metal-objdump output has no module text")?;
    Ok(dis[module_start..].to_string())
}

/// Reassemble edited Apple-dialect module text (`metal-as`) and relink it (`metallib`) into
/// `metallib_path`. `stem` names the scratch `.ll`/`.air` intermediates next to `air_path`.
fn assemble_and_link(
    module_text: &str,
    air_path: &Path,
    metallib_path: &Path,
    stem: &str,
) -> Result<(), String> {
    let scratch = air_path.parent().ok_or("AIR scratch path has no parent")?;
    let ll_path = scratch.join(format!("{stem}.ll"));
    let out_air = scratch.join(format!("{stem}.air"));
    fs::write(&ll_path, module_text).map_err(|e| format!("write {}: {e}", ll_path.display()))?;
    command_stdout(
        "xcrun",
        &[
            "metal-as",
            ll_path.to_str().ok_or("scratch ll path is not UTF-8")?,
            "-o",
            out_air.to_str().ok_or("scratch air path is not UTF-8")?,
        ],
    )?;
    command_stdout(
        "xcrun",
        &[
            "metallib",
            out_air.to_str().ok_or("scratch air path is not UTF-8")?,
            "-o",
            metallib_path.to_str().ok_or("metallib path is not UTF-8")?,
        ],
    )?;
    Ok(())
}

/// Rename every word-boundary reference to `@<sym>` (a symbol that collides with a stdlib libcall
/// the metallib linker injects). Returns the edited text plus the entry name to execute (renamed
/// when the entry itself was the colliding symbol).
fn rename_symbol_in_module_text(
    module_text: &str,
    sym: &str,
    function_name: &str,
) -> (String, String) {
    let renamed_sym = format!("{sym}__m2v_oracle");
    let mut text = String::with_capacity(module_text.len());
    let mut rest = module_text;
    // Word-boundary rename of `@<sym>` so a symbol that prefixes another (e.g. `memcpy` vs
    // `memcpy2`) is not clobbered.
    let needle = format!("@{sym}");
    while let Some(pos) = rest.find(&needle) {
        let after = pos + needle.len();
        let boundary = rest[after..]
            .chars()
            .next()
            .map(|ch| !(ch.is_ascii_alphanumeric() || ch == '_' || ch == '.' || ch == '$'))
            .unwrap_or(true);
        text.push_str(&rest[..pos]);
        text.push('@');
        text.push_str(if boundary { &renamed_sym } else { sym });
        rest = &rest[after..];
    }
    text.push_str(rest);
    // Alias-scope metadata embeds the bare function name.
    let text = text.replace(
        &format!("\"air-alias-scopes({sym})\""),
        &format!("\"air-alias-scopes({renamed_sym})\""),
    );
    let entry = if function_name == sym {
        renamed_sym
    } else {
        function_name.to_string()
    };
    (text, entry)
}

/// Inject a deterministic zero-fill prologue for every threadgroup region of the entry kernel into
/// Apple-dialect module text: module-level `addrspace(3)` globals get whole-aggregate
/// `store <type> zeroinitializer` lines, and each `addrspace(3)*` kernel argument gets a store of
/// `[WORKGROUP_MEMORY_ELEMENTS x <pointee>] zeroinitializer` (exactly the region the harness binds
/// via `setThreadgroupMemoryLength`), followed by one `air.wg.barrier` ordering the fill before the
/// kernel body. Every invocation writes the same zero bytes, so the racy fill is
/// value-deterministic. Returns `None` when the entry has no threadgroup regions (nothing to do).
/// Purely structural: it keys on address space 3 in the IR, never on any name from a specific shader.
fn insert_threadgroup_zero_init(module_text: &str, entry: &str) -> Option<String> {
    // Module-level threadgroup globals: `@g = internal ... addrspace(3) global <type> undef, align N`.
    let mut prologue: Vec<String> = Vec::new();
    for line in module_text.lines() {
        let Some((name, rest)) = line.split_once(" = ") else {
            continue;
        };
        if !name.starts_with('@') {
            continue;
        }
        let marker = " addrspace(3) global ";
        let Some(pos) = rest.find(marker) else {
            continue;
        };
        let tail = &rest[pos + marker.len()..];
        let (body, align) = match tail.rsplit_once(", align ") {
            Some((body, align)) => (body, Some(align.trim())),
            None => (tail, None),
        };
        // Threadgroup globals carry a placeholder initializer (their contents are undefined at
        // dispatch); anything else is not a plain threadgroup scratch declaration — skip it.
        let Some(ty) = body
            .strip_suffix(" undef")
            .or_else(|| body.strip_suffix(" zeroinitializer"))
        else {
            continue;
        };
        let align_suffix = align.map(|a| format!(", align {a}")).unwrap_or_default();
        prologue.push(format!(
            "  store {ty} zeroinitializer, {ty} addrspace(3)* {name}{align_suffix}\n"
        ));
    }

    // The entry kernel's define line + its threadgroup pointer arguments.
    let define_prefix = format!("define void @{entry}(");
    let mut define_line_idx = None;
    let lines: Vec<&str> = module_text.lines().collect();
    for (idx, line) in lines.iter().enumerate() {
        if line.starts_with(&define_prefix) {
            define_line_idx = Some(idx);
            break;
        }
    }
    let define_line_idx = define_line_idx?;
    let define_line = lines[define_line_idx];
    let open = define_line.find('(')?;
    // Match the define's outer parameter list (types nest parens/angles via structs and vectors).
    let mut depth = 0usize;
    let mut close = None;
    for (offset, ch) in define_line[open..].char_indices() {
        match ch {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    close = Some(open + offset);
                    break;
                }
            }
            _ => {}
        }
    }
    let params_text = &define_line[open + 1..close?];
    let mut depth = 0i32;
    let mut param = String::new();
    let mut params: Vec<String> = Vec::new();
    for ch in params_text.chars() {
        match ch {
            '(' | '<' | '[' | '{' => depth += 1,
            ')' | '>' | ']' | '}' => depth -= 1,
            ',' if depth == 0 => {
                params.push(std::mem::take(&mut param));
                continue;
            }
            _ => {}
        }
        param.push(ch);
    }
    if !param.trim().is_empty() {
        params.push(param);
    }
    for (index, param) in params.iter().enumerate() {
        let param = param.trim();
        let Some(split) = param.rfind(" addrspace(3)*") else {
            continue;
        };
        let pointee = &param[..split];
        let Some(name) = param
            .split_whitespace()
            .next_back()
            .filter(|token| token.starts_with('%'))
        else {
            continue;
        };
        let array_ty = format!("[{WORKGROUP_MEMORY_ELEMENTS} x {pointee}]");
        let cast = format!("%\"m2v.zi.arg{index}\"");
        prologue.push(format!(
            "  {cast} = bitcast {pointee} addrspace(3)* {name} to {array_ty} addrspace(3)*\n"
        ));
        prologue.push(format!(
            "  store {array_ty} zeroinitializer, {array_ty} addrspace(3)* {cast}, align 4\n"
        ));
    }

    if prologue.is_empty() {
        return None;
    }
    prologue.push("  tail call void @air.wg.barrier(i32 2, i32 1)\n".to_string());

    // Insert right inside the entry block: after the define line, skipping an explicit block label
    // if the printer emitted one (labels sit at column 0 and end with ':').
    let mut insert_after = define_line_idx;
    if let Some(next) = lines.get(define_line_idx + 1) {
        let is_label = !next.starts_with([' ', '\t', ';'])
            && next.trim_end().ends_with(':')
            && !next.trim_start().is_empty();
        if is_label {
            insert_after += 1;
        }
    }
    let mut out =
        String::with_capacity(module_text.len() + prologue.iter().map(String::len).sum::<usize>());
    for (idx, line) in lines.iter().enumerate() {
        out.push_str(line);
        out.push('\n');
        if idx == insert_after {
            for inserted in &prologue {
                out.push_str(inserted);
            }
        }
    }
    // The prologue's barrier needs the AIR declaration when the kernel itself never barriers
    // (probe the ORIGINAL text: the prologue always references the symbol).
    if !module_text.contains("@air.wg.barrier(") {
        out.push_str("\ndeclare void @air.wg.barrier(i32, i32)\n");
    }
    Some(out)
}

fn command_stdout(cmd: &str, args: &[&str]) -> Result<String, String> {
    let output = Command::new(cmd)
        .args(args)
        .output()
        .map_err(|e| format!("spawn {cmd}: {e}"))?;
    if !output.status.success() {
        return Err(format!(
            "status={} stderr={}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

// Create a Metal function with an **empty** `MTLFunctionConstantValues`. This matches how `metal2vulkan`
// handles `[[function_constant]]`s: every function constant is treated as undefined and folded to
// its disabled default (booleans → false, scalar loads → 0). Supplying no constant values makes
// `is_function_constant_defined` return false and lets Metal use each constant's declared default,
// so the oracle pipeline and the translated SPIR-V select the same code path. Functions with no
// function constants are unaffected. A pipeline that genuinely requires an unset constant fails
// here (the collector gates those out via `kernel_harness_gap_reason`).
thread_local! {
    /// Per-case request (set by `the optional oracle path` from the override's `zero_fc` flag) to specialize
    /// the oracle with explicit-zero function constants. A thread-local (not a parameter threaded
    /// through the whole execute path) keeps the signature churn contained; oracle tests run one
    /// case per thread so there is no cross-case bleed.
    static ZERO_FC_CASE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// Per-case explicit function-constant values (set from the override's `fc_values`): FC index ->
    /// value. When non-empty the oracle is specialized with these values (any declared FC not listed
    /// gets 0), matching the same values baked into the translated SPIR-V via
    /// `metal2vulkan::specialize_function_constants`. This is how FC kernels that hang / divide-by-zero
    /// under the disabled-default (all-zero) specialization get a derivable, byte-comparable golden.
    static FC_VALUES_CASE: std::cell::RefCell<Vec<(usize, u64)>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// Set the per-case zero-FC specialization request for the current thread. `the optional oracle path` calls
/// this before invoking the oracle so the specialization matches the case's override.
pub fn set_zero_fc(on: bool) {
    ZERO_FC_CASE.with(|c| c.set(on));
}

/// Set the per-case explicit function-constant values for the current thread. `the optional oracle path` calls
/// this before invoking the oracle so the oracle and the translated SPIR-V specialize identically.
pub fn set_fc_values(values: Vec<(usize, u64)>) {
    FC_VALUES_CASE.with(|c| *c.borrow_mut() = values);
}

fn fc_values_requested() -> Vec<(usize, u64)> {
    FC_VALUES_CASE.with(|c| c.borrow().clone())
}

fn zero_fc_requested() -> bool {
    ZERO_FC_CASE.with(|c| c.get()) || std::env::var("METAL2VULKAN_ZERO_FC").is_ok_and(|v| v != "0")
}

fn new_specialized_function(
    library: &ProtocolObject<dyn MTLLibrary>,
    entry: &str,
    sanitized_ll: Option<&str>,
) -> Retained<ProtocolObject<dyn MTLFunction>> {
    let function_name = NSString::from_str(entry);
    // A module whose `air.fc_initializer`s are `undef` (no declared default) reads GARBAGE for
    // every function constant under an EMPTY constant set — Metal specializes the pipeline with
    // uninitialized FC values. For a kernel that uses an FC directly (e.g. a loop-bound radius),
    // that garbage can make the dispatch loop an unbounded number of times (oracle hang) AND
    // diverge from metal2vulkan, which folds every FC to its disabled ZERO default. Specializing
    // the oracle with an EXPLICIT ZERO for every declared FC makes the reference take the exact
    // same disabled-default code path as the translator — the honest way to give these
    // otherwise-unrunnable FC kernels a byte-comparable golden. Opt-in per run so existing goldens
    // (captured under empty specialization) are untouched until deliberately reseeded.
    let explicit_fc = fc_values_requested();
    if !explicit_fc.is_empty() {
        // Explicit values requested: specialize with them (declared FCs not listed default to 0),
        // identical to the values baked into the translated SPIR-V.
        if let Some(function) =
            new_fc_specialized_function(library, entry, sanitized_ll, &explicit_fc)
        {
            return function;
        }
    }
    if zero_fc_requested() {
        if let Some(function) = new_fc_specialized_function(library, entry, sanitized_ll, &[]) {
            return function;
        }
    }
    let constants = MTLFunctionConstantValues::new();
    match library.newFunctionWithName_constantValues_error(&function_name, &constants) {
        Ok(function) => function,
        Err(empty_err) => {
            if let Some(values) = dynamic_resource_location_fc_values(sanitized_ll) {
                if let Some(function) =
                    new_fc_specialized_function(library, entry, sanitized_ll, &values)
                {
                    return function;
                }
            }
            // Apple's MTLCompilerService can CRASH (XPC connection interrupted) specializing a
            // function whose `air.fc_initializer` is undef under an EMPTY constant set (the
            // interpolateRGB* Portrait family). Supplying an EXPLICIT ZERO for every declared
            // function constant is semantically identical to metal2vulkan's folding (every FC folds
            // to its disabled zero default) and avoids the front-end crash, so retry with zeros
            // before giving up. Only reached when the empty-set specialization already failed, so
            // no currently-passing case changes behavior.
            new_fc_specialized_function(library, entry, sanitized_ll, &[]).unwrap_or_else(|| {
                panic!("Metal function {entry:?} could not be specialized: {empty_err}")
            })
        }
    }
}

/// Build the function specializing every declared function constant to an EXPLICIT value: the value
/// in `values` for that FC index, or 0 if unlisted. All-zero (`values` empty) is semantically
/// identical to metal2vulkan's folding (every FC → its disabled zero default); non-zero values match
/// the constants baked into the translated SPIR-V by `metal2vulkan::specialize_function_constants`,
/// giving otherwise-unrunnable FC kernels (all-zero → udiv/0 / unbounded loop) a byte-comparable
/// golden. Returns `None` when the module declares no function constants (nothing to specialize).
fn new_fc_specialized_function(
    library: &ProtocolObject<dyn MTLLibrary>,
    entry: &str,
    sanitized_ll: Option<&str>,
    values: &[(usize, u64)],
) -> Option<Retained<ProtocolObject<dyn MTLFunction>>> {
    let declared = sanitized_ll
        .map(declared_function_constants)
        .unwrap_or_default();
    if declared.is_empty() {
        return None;
    }
    let value_for: std::collections::HashMap<usize, u64> = values.iter().copied().collect();
    let function_name = NSString::from_str(entry);
    let specialized = MTLFunctionConstantValues::new();
    for (ty, index) in &declared {
        let Some(data_type) = mtl_data_type_for_fc(ty) else {
            panic!("Metal function {entry:?}: FC path unsupported for constant type {ty:?}");
        };
        // Little-endian value bytes; `setConstantValue_type_atIndex` reads `data_type`-many bytes
        // from the front, so the low N bytes are correct for any scalar-int width. Unlisted FCs → 0.
        let bytes = value_for.get(index).copied().unwrap_or(0).to_le_bytes();
        unsafe {
            specialized.setConstantValue_type_atIndex(
                std::ptr::NonNull::new(bytes.as_ptr() as *mut _).unwrap(),
                data_type,
                *index,
            );
        }
    }
    Some(
        library
            .newFunctionWithName_constantValues_error(&function_name, &specialized)
            .unwrap_or_else(|err| {
                panic!("Metal function {entry:?} could not be FC-specialized: {err}")
            }),
    )
}

/// Parse the module's `!air.function_constants` list into `(type-name, constant-index)` pairs.
/// Each referenced node has the stable AIR shape
/// `!{ptr @..., !"<type>", !"<name>", i32 <index>, i1 <...>}`.
fn declared_function_constants(sanitized_ll: &str) -> Vec<(String, usize)> {
    let node_ids: Vec<&str> = sanitized_ll
        .lines()
        .find_map(|line| {
            let rest = line.trim().strip_prefix("!air.function_constants = !{")?;
            let rest = rest.strip_suffix('}')?;
            Some(
                rest.split(',')
                    .map(|s| s.trim().trim_start_matches('!'))
                    .filter(|s| !s.is_empty())
                    .collect(),
            )
        })
        .unwrap_or_default();
    let mut out = vec![];
    for id in node_ids {
        let prefix = format!("!{id} = !{{");
        let Some(node) = sanitized_ll
            .lines()
            .find(|line| line.trim_start().starts_with(&prefix))
        else {
            continue;
        };
        // Fields: `ptr @init, !"<type>", !"<name>", i32 <index>, i1 <...>` — take the first quoted
        // string as the type and the first `i32 N` as the index.
        let mut quoted = node.split("!\"").skip(1).map(|s| s.split('"').next());
        let Some(Some(ty)) = quoted.next() else {
            continue;
        };
        let Some(index) = node.split("i32 ").nth(1).and_then(|s| {
            s.chars()
                .take_while(|c| c.is_ascii_digit())
                .collect::<String>()
                .parse::<usize>()
                .ok()
        }) else {
            continue;
        };
        out.push((ty.to_string(), index));
    }
    out
}

fn dynamic_resource_location_fc_values(sanitized_ll: Option<&str>) -> Option<Vec<(usize, u64)>> {
    let ll = sanitized_ll?;
    let has_dynamic_resource_location = ll.lines().any(|line| {
        (line.contains(r#""air.texture""#) || line.contains(r#""air.buffer""#))
            && line.contains(r#""air.function_constant""#)
            && line.contains(r#""air.location_index", ptr addrspace(2)"#)
    });
    if !has_dynamic_resource_location {
        return None;
    }

    let values: Vec<_> = declared_function_constants(ll)
        .into_iter()
        .filter(|(ty, _)| is_integer_fc_type(ty))
        .map(|(_, index)| (index, 1))
        .collect();
    (!values.is_empty()).then_some(values)
}

fn is_integer_fc_type(ty: &str) -> bool {
    matches!(
        ty,
        "char"
            | "char2"
            | "char3"
            | "char4"
            | "uchar"
            | "uchar2"
            | "uchar3"
            | "uchar4"
            | "short"
            | "short2"
            | "short3"
            | "short4"
            | "ushort"
            | "ushort2"
            | "ushort3"
            | "ushort4"
            | "int"
            | "int2"
            | "int3"
            | "int4"
            | "uint"
            | "uint2"
            | "uint3"
            | "uint4"
            | "long"
            | "long2"
            | "long3"
            | "long4"
            | "ulong"
            | "ulong2"
            | "ulong3"
            | "ulong4"
    )
}

/// Map an AIR function-constant type name to the `MTLDataType` used to set its (zero) value.
fn mtl_data_type_for_fc(ty: &str) -> Option<objc2_metal::MTLDataType> {
    use objc2_metal::MTLDataType as T;
    Some(match ty {
        "bool" => T::Bool,
        "bool2" => T::Bool2,
        "bool3" => T::Bool3,
        "bool4" => T::Bool4,
        "int" => T::Int,
        "int2" => T::Int2,
        "int3" => T::Int3,
        "int4" => T::Int4,
        "uint" => T::UInt,
        "uint2" => T::UInt2,
        "uint3" => T::UInt3,
        "uint4" => T::UInt4,
        "short" => T::Short,
        "short2" => T::Short2,
        "short3" => T::Short3,
        "short4" => T::Short4,
        "ushort" => T::UShort,
        "ushort2" => T::UShort2,
        "ushort3" => T::UShort3,
        "ushort4" => T::UShort4,
        "char" => T::Char,
        "char2" => T::Char2,
        "char3" => T::Char3,
        "char4" => T::Char4,
        "uchar" => T::UChar,
        "uchar2" => T::UChar2,
        "uchar3" => T::UChar3,
        "uchar4" => T::UChar4,
        "long" => T::Long,
        "long2" => T::Long2,
        "long3" => T::Long3,
        "long4" => T::Long4,
        "ulong" => T::ULong,
        "ulong2" => T::ULong2,
        "ulong3" => T::ULong3,
        "ulong4" => T::ULong4,
        "float" => T::Float,
        "float2" => T::Float2,
        "float3" => T::Float3,
        "float4" => T::Float4,
        "half" => T::Half,
        "half2" => T::Half2,
        "half3" => T::Half3,
        "half4" => T::Half4,
        _ => return None,
    })
}

fn execute_compute_library(
    device: &ProtocolObject<dyn MTLDevice>,
    library: &ProtocolObject<dyn MTLLibrary>,
    entry: &str,
    inputs: &Inputs,
    sanitized_ll: Option<&str>,
) -> Vec<u8> {
    let function = new_specialized_function(library, entry, sanitized_ll);
    let stage_inputs = sanitized_ll.map(compute_stage_inputs).unwrap_or_default();
    let stage_input_buffer_index = free_attribute_buffer_index(inputs);
    let pipeline = new_compute_pipeline_state(
        device,
        library,
        entry,
        sanitized_ll,
        &function,
        &stage_inputs,
        stage_input_buffer_index,
    )
    .unwrap_or_else(|pso_err| {
        // PSO creation can fail even when empty-set specialization succeeded: an argument
        // whose presence is predicated on a function constant with an undef
        // `air.fc_initializer` (e.g. an `[[function_constant]]`-guarded imageblock arg)
        // defers the failure to the backend compile ("Encountered unlowered function call
        // to air.imageblock_data"). Zero-specializing every declared FC resolves the
        // predicate exactly the way metal2vulkan folds FCs, so retry PSO creation once
        // with explicit zeros before giving up.
        let retry_values =
            dynamic_resource_location_fc_values(sanitized_ll).unwrap_or_else(fc_values_requested);
        let zero_fn = new_fc_specialized_function(library, entry, sanitized_ll, &retry_values)
            .unwrap_or_else(|| panic!("newComputePipelineStateWithFunction({entry}): {pso_err}"));
        new_compute_pipeline_state(
            device,
            library,
            entry,
            sanitized_ll,
            &zero_fn,
            &stage_inputs,
            stage_input_buffer_index,
        )
        .unwrap_or_else(|zero_err| {
            panic!(
                "newComputePipelineStateWithFunction({entry}): {pso_err} \
                         (zero-FC retry: {zero_err})"
            )
        })
    });
    let queue = device
        .newCommandQueue()
        .expect("MTLDevice::newCommandQueue returned nil");
    let command_buffer = queue
        .commandBuffer()
        .expect("MTLCommandQueue::commandBuffer returned nil");
    let encoder = command_buffer
        .computeCommandEncoder()
        .expect("MTLCommandBuffer::computeCommandEncoder returned nil");

    encoder.setComputePipelineState(&pipeline);
    let buffers = make_buffers(device, inputs);
    let textures = make_textures(device, inputs, sanitized_ll);
    let default_sampler = make_default_sampler(device);
    let stage_input_buffer = make_compute_stage_input_buffer(device, &stage_inputs, inputs);
    for (index, buffer) in &buffers {
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&**buffer), 0, *index as usize);
        }
    }
    if let Some(buffer) = &stage_input_buffer {
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&**buffer), 0, stage_input_buffer_index as usize);
        }
        encoder.setStageInRegion(MTLRegion {
            origin: MTLOrigin { x: 0, y: 0, z: 0 },
            size: mtl_size(inputs.dispatch.threads_per_grid),
        });
    }
    for (index, texture) in &textures {
        unsafe {
            encoder.setTexture_atIndex(Some(&**texture), *index as usize);
        }
    }
    // Argument-buffer-embedded textures: write the seeded texture's GPU resource ID into the arg
    // buffer's handle slot (element 0) so Metal's `read_texture` on the embedded handle resolves to
    // our deterministic texture, and declare the texture used-through-argument-buffer so the GPU may
    // access it. This is what replaces the previous garbage golden (an unseeded handle read arbitrary
    // memory). The Vulkan runner reads the SAME seeded texture through a standalone descriptor.
    for binding in inputs.embedded_textures {
        let Some((_, texture)) = textures
            .iter()
            .find(|(idx, _)| *idx == binding.texture_index)
        else {
            continue;
        };
        let resource_id = texture.gpuResourceID().to_raw();
        if let Some((_, buffer)) = buffers.iter().find(|(idx, _)| *idx == binding.buffer_index) {
            unsafe {
                let dst = buffer
                    .contents()
                    .as_ptr()
                    .cast::<u8>()
                    .add(binding.field_offset as usize);
                std::ptr::copy_nonoverlapping(resource_id.to_le_bytes().as_ptr(), dst, 8);
            }
        }
        encoder.useResource_usage(ProtocolObject::from_ref(&**texture), MTLResourceUsage::Read);
    }
    unsafe {
        encoder.setSamplerState_atIndex(Some(&*default_sampler), 0);
    }
    bind_threadgroup_memory(&encoder, sanitized_ll);
    encoder.dispatchThreads_threadsPerThreadgroup(
        mtl_size(inputs.dispatch.threads_per_grid),
        mtl_size(inputs.dispatch.threads_per_threadgroup),
    );
    encoder.endEncoding();
    command_buffer.commit();
    command_buffer.waitUntilCompleted();
    if command_buffer.status() != MTLCommandBufferStatus::Completed {
        if let Some(error) = command_buffer.error() {
            panic!("Metal command buffer failed: {error}");
        }
        panic!(
            "Metal command buffer ended with status {:?}",
            command_buffer.status()
        );
    }

    match inputs.output {
        Output::Buffer { index, len, .. } => {
            let buffer = buffers
                .iter()
                .find_map(|(buffer_index, buffer)| (*buffer_index == index).then_some(buffer))
                .unwrap_or_else(|| panic!("output buffer index {index} was not bound"));
            assert!(
                buffer.length() >= len,
                "output buffer index {index} has length {}, expected at least {len}",
                buffer.length()
            );
            unsafe {
                let ptr = buffer.contents().as_ptr().cast::<u8>();
                std::slice::from_raw_parts(ptr, len).to_vec()
            }
        }
        Output::Texture {
            index,
            format,
            extent,
        } => {
            let texture = textures
                .iter()
                .find_map(|(texture_index, texture)| (*texture_index == index).then_some(texture))
                .unwrap_or_else(|| panic!("output texture index {index} was not bound"));
            let kind = texture_kind(sanitized_ll, index);
            read_texture(
                texture,
                format,
                texture_output_extent(extent, kind),
                kind == TextureKind::Dim2dArray,
            )
        }
        Output::RenderTarget { .. } => {
            panic!("objc2-metal oracle currently supports compute buffer/texture outputs only")
        }
    }
}

fn new_compute_pipeline_state(
    device: &ProtocolObject<dyn MTLDevice>,
    library: &ProtocolObject<dyn MTLLibrary>,
    entry: &str,
    sanitized_ll: Option<&str>,
    function: &ProtocolObject<dyn MTLFunction>,
    stage_inputs: &[VertexInput],
    stage_input_buffer_index: u32,
) -> Result<MetalComputePipeline, Retained<objc2_foundation::NSError>> {
    let linked_names = sanitized_ll
        .map(visible_function_reference_names)
        .unwrap_or_default();
    if linked_names.is_empty() && stage_inputs.is_empty() {
        return device.newComputePipelineStateWithFunction_error(function);
    }
    if !linked_names.is_empty() && stage_inputs.is_empty() {
        if let Ok(pipeline) = device.newComputePipelineStateWithFunction_error(function) {
            return Ok(pipeline);
        }
    }

    let descriptor = MTLComputePipelineDescriptor::new();
    descriptor.setComputeFunction(Some(function));
    if !stage_inputs.is_empty() {
        let stage_descriptor =
            make_compute_stage_input_descriptor(stage_inputs, stage_input_buffer_index);
        descriptor.setStageInputDescriptor(Some(&stage_descriptor));
    }

    if !linked_names.is_empty() {
        descriptor.setMaxCallStackDepth((linked_names.len() + 1).max(2));

        let linked_functions = MTLLinkedFunctions::new();
        let functions: Vec<_> = linked_names
            .iter()
            .filter(|name| name.as_str() != entry)
            .filter_map(|name| new_linked_function(library, name))
            .collect();
        let functions = NSArray::from_retained_slice(&functions);
        linked_functions.setFunctions(Some(&functions));
        descriptor.setLinkedFunctions(Some(&linked_functions));
    }

    device.newComputePipelineStateWithDescriptor_options_reflection_error(
        &descriptor,
        MTLPipelineOption::None,
        None,
    )
}

fn new_linked_function(
    library: &ProtocolObject<dyn MTLLibrary>,
    entry: &str,
) -> Option<Retained<ProtocolObject<dyn MTLFunction>>> {
    let function_name = NSString::from_str(entry);
    library
        .newFunctionWithName(&function_name)
        .or_else(|| {
            let constants = MTLFunctionConstantValues::new();
            library
                .newFunctionWithName_constantValues_error(&function_name, &constants)
                .ok()
        })
        .or_else(|| {
            let descriptor = MTLFunctionDescriptor::functionDescriptor();
            descriptor.setName(Some(&function_name));
            library.newFunctionWithDescriptor_error(&descriptor).ok()
        })
}

/// Parse `!air.visible_function_references` into the stable AIR-visible function names that
/// must be linked into a compute pipeline. Each node shape is
/// `!{!"air.visible_function_reference", ptr @..., !"<visible-name>"}`.
fn visible_function_reference_names(sanitized_ll: &str) -> Vec<String> {
    let Some(node_ids) = metadata_list_node_ids(sanitized_ll, "!air.visible_function_references")
    else {
        return vec![];
    };

    let mut out = Vec::new();
    for id in node_ids {
        let prefix = format!("!{id} = !{{");
        let Some(node) = sanitized_ll
            .lines()
            .find(|line| line.trim_start().starts_with(&prefix))
        else {
            continue;
        };
        let strings = metadata_quoted_strings(node);
        if strings
            .first()
            .is_none_or(|tag| *tag != "air.visible_function_reference")
        {
            continue;
        }
        if let Some(name) = strings.last() {
            if !name.is_empty() && !out.iter().any(|seen| seen == name) {
                out.push((*name).to_string());
            }
        }
    }
    out
}

fn metadata_list_node_ids<'a>(sanitized_ll: &'a str, name: &str) -> Option<Vec<&'a str>> {
    sanitized_ll.lines().find_map(|line| {
        let rest = line.trim().strip_prefix(name)?;
        let rest = rest.trim_start().strip_prefix("= !{")?;
        let rest = rest.strip_suffix('}')?;
        Some(
            rest.split(',')
                .map(|s| s.trim().trim_start_matches('!'))
                .filter(|s| !s.is_empty())
                .collect(),
        )
    })
}

fn metadata_quoted_strings(line: &str) -> Vec<&str> {
    line.split("!\"")
        .skip(1)
        .filter_map(|s| s.split('"').next())
        .collect()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ThreadgroupMemoryBinding {
    index: usize,
    byte_len: usize,
}

fn bind_threadgroup_memory(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    sanitized_ll: Option<&str>,
) {
    let Some(ll) = sanitized_ll else { return };
    for binding in threadgroup_memory_bindings(ll) {
        unsafe {
            encoder.setThreadgroupMemoryLength_atIndex(binding.byte_len, binding.index);
        }
    }
}

fn threadgroup_memory_bindings(ll: &str) -> Vec<ThreadgroupMemoryBinding> {
    ll.lines()
        .filter(|line| {
            line.contains(r#""air.buffer""#)
                && metadata_i32_after(line, "air.address_space") == Some(3)
        })
        .filter_map(|line| {
            let index = metadata_i32_after(line, "air.location_index")? as usize;
            let elem_size = metadata_i32_after(line, "air.arg_type_size").unwrap_or(4) as usize;
            Some(ThreadgroupMemoryBinding {
                index,
                byte_len: elem_size.max(1) * WORKGROUP_MEMORY_ELEMENTS,
            })
        })
        .collect()
}

fn metadata_i32_after(body: &str, marker: &str) -> Option<u32> {
    let marker = format!("!\"{marker}\"");
    let tail = body.get(body.find(&marker)? + marker.len()..)?;
    let mut tokens = tail.split_whitespace().peekable();
    while let Some(token) = tokens.next() {
        if token == "i32" {
            let value = tokens.peek()?.trim_end_matches(',');
            if let Ok(parsed) = value.parse() {
                return Some(parsed);
            }
        }
    }
    None
}

fn execute_render_fragment_library(
    device: &ProtocolObject<dyn MTLDevice>,
    library: &ProtocolObject<dyn MTLLibrary>,
    fragment_entry: &str,
    inputs: &Inputs,
    sanitized_ll: Option<&str>,
) -> Vec<u8> {
    let (format, extent) = match inputs.output {
        Output::RenderTarget { format, extent } => (format, extent),
        other => panic!("fragment render cases must use RenderTarget output, got {other:?}"),
    };
    assert_eq!(
        extent.depth, 1,
        "objc2-metal oracle currently supports 2D render targets only"
    );
    assert_eq!(
        inputs.render.target, extent,
        "render target extent must match render pass target extent"
    );
    let depth_output = is_depth_format(format);
    let writes_depth = depth_output || sanitized_ll.is_some_and(fragment_writes_depth);

    let validation_vertex_src = validation_vertex_src_for_fragment(sanitized_ll);
    let validation_vertex_library = compile_library(device, &validation_vertex_src);
    let vertex_name = NSString::from_str("metal2vulkan_validation_fullscreen_vertex");
    let vertex_function = validation_vertex_library
        .newFunctionWithName(&vertex_name)
        .expect("validation fullscreen vertex function not found");
    let fragment_function = new_specialized_function(library, fragment_entry, sanitized_ll);

    let pipeline_descriptor = MTLRenderPipelineDescriptor::new();
    pipeline_descriptor.setVertexFunction(Some(&vertex_function));
    pipeline_descriptor.setFragmentFunction(Some(&fragment_function));
    if depth_output {
        pipeline_descriptor.setDepthAttachmentPixelFormat(metal_pixel_format(format));
    } else {
        let color_attachments = pipeline_descriptor.colorAttachments();
        let color_attachment = unsafe { color_attachments.objectAtIndexedSubscript(0) };
        color_attachment.setPixelFormat(metal_pixel_format(format));
        match inputs.render.blend {
            BlendMode::Replace => {
                color_attachment.setBlendingEnabled(false);
            }
            BlendMode::SourceOver => {
                color_attachment.setBlendingEnabled(true);
                color_attachment.setSourceRGBBlendFactor(MTLBlendFactor::SourceAlpha);
                color_attachment.setDestinationRGBBlendFactor(MTLBlendFactor::OneMinusSourceAlpha);
                color_attachment.setRgbBlendOperation(MTLBlendOperation::Add);
                color_attachment.setSourceAlphaBlendFactor(MTLBlendFactor::One);
                color_attachment
                    .setDestinationAlphaBlendFactor(MTLBlendFactor::OneMinusSourceAlpha);
                color_attachment.setAlphaBlendOperation(MTLBlendOperation::Add);
            }
        }
    }
    if writes_depth && !depth_output {
        pipeline_descriptor.setDepthAttachmentPixelFormat(MTLPixelFormat::Depth32Float);
    }
    let pipeline = device
        .newRenderPipelineStateWithDescriptor_error(&pipeline_descriptor)
        .unwrap_or_else(|e| panic!("newRenderPipelineStateWithDescriptor({fragment_entry}): {e}"));
    let depth_stencil_state = writes_depth.then(|| make_depth_stencil_state(device));

    let target = make_render_target(device, format, extent);
    let mut target_bytes = seeded_render_target_bytes(format, extent);
    write_texture_bytes(
        &target,
        format,
        extent,
        TextureKind::Plain,
        &mut target_bytes,
    );
    let depth_target = if writes_depth && !depth_output {
        let depth_target = make_render_target(device, DataFormat::Depth32Float, extent);
        let mut depth_bytes = seeded_render_target_bytes(DataFormat::Depth32Float, extent);
        write_texture_bytes(
            &depth_target,
            DataFormat::Depth32Float,
            extent,
            TextureKind::Plain,
            &mut depth_bytes,
        );
        Some(depth_target)
    } else {
        None
    };
    let pass_descriptor = MTLRenderPassDescriptor::renderPassDescriptor();
    if depth_output {
        let pass_depth_attachment = pass_descriptor.depthAttachment();
        pass_depth_attachment.setTexture(Some(&target));
        pass_depth_attachment.setLoadAction(MTLLoadAction::Load);
        pass_depth_attachment.setStoreAction(MTLStoreAction::Store);
    } else {
        let pass_color_attachments = pass_descriptor.colorAttachments();
        let pass_color_attachment = unsafe { pass_color_attachments.objectAtIndexedSubscript(0) };
        pass_color_attachment.setTexture(Some(&target));
        pass_color_attachment.setLoadAction(MTLLoadAction::Load);
        pass_color_attachment.setStoreAction(MTLStoreAction::Store);
        if let Some(depth_target) = &depth_target {
            let pass_depth_attachment = pass_descriptor.depthAttachment();
            pass_depth_attachment.setTexture(Some(&**depth_target));
            pass_depth_attachment.setLoadAction(MTLLoadAction::Load);
            pass_depth_attachment.setStoreAction(MTLStoreAction::Store);
        }
    }

    let queue = device
        .newCommandQueue()
        .expect("MTLDevice::newCommandQueue returned nil");
    let command_buffer = queue
        .commandBuffer()
        .expect("MTLCommandQueue::commandBuffer returned nil");
    let encoder = command_buffer
        .renderCommandEncoderWithDescriptor(&pass_descriptor)
        .expect("MTLCommandBuffer::renderCommandEncoderWithDescriptor returned nil");
    encoder.setRenderPipelineState(&pipeline);
    if let Some(depth_stencil_state) = &depth_stencil_state {
        encoder.setDepthStencilState(Some(&**depth_stencil_state));
    }
    let buffers = make_buffers(device, inputs);
    let textures = make_textures(device, inputs, sanitized_ll);
    for (index, buffer) in &buffers {
        unsafe {
            encoder.setFragmentBuffer_offset_atIndex(Some(&**buffer), 0, *index as usize);
        }
    }
    for (index, texture) in &textures {
        unsafe {
            encoder.setFragmentTexture_atIndex(Some(&**texture), *index as usize);
        }
    }
    let default_sampler = make_default_sampler(device);
    unsafe {
        encoder.setFragmentSamplerState_atIndex(Some(&*default_sampler), 0);
    }
    unsafe {
        encoder.drawPrimitives_vertexStart_vertexCount(
            MTLPrimitiveType::Triangle,
            0,
            inputs.render.vertex_count as usize,
        );
    }
    encoder.endEncoding();
    command_buffer.commit();
    command_buffer.waitUntilCompleted();
    if command_buffer.status() != MTLCommandBufferStatus::Completed {
        if let Some(error) = command_buffer.error() {
            panic!("Metal command buffer failed: {error}");
        }
        panic!(
            "Metal command buffer ended with status {:?}",
            command_buffer.status()
        );
    }

    read_texture(&target, format, extent, false)
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct FragmentInput {
    user_name: String,
    user_attribute: Option<String>,
    type_name: String,
}

fn validation_vertex_src_for_fragment(sanitized_ll: Option<&str>) -> String {
    let inputs = sanitized_ll
        .map(fragment_inputs)
        .unwrap_or_default()
        .into_iter()
        .filter(|input| validation_vertex_value(&input.type_name).is_some())
        .collect::<Vec<_>>();
    if inputs.is_empty() {
        return VALIDATION_VERTEX_SRC.to_string();
    }
    let field_names = inputs
        .iter()
        .enumerate()
        .map(|(index, input)| validation_vertex_field_name(input, index))
        .collect::<Vec<_>>();
    let position_field = validation_vertex_position_field_name(&field_names);

    let mut src = String::from(
        r#"
#include <metal_stdlib>
using namespace metal;

struct Metal2VulkanValidationVertexOut {
"#,
    );
    src.push_str(&format!("    float4 {position_field} [[position]];\n"));
    for (index, input) in inputs.iter().enumerate() {
        src.push_str(&format!(
            "    {} {}{};\n",
            input.type_name,
            field_names[index],
            validation_vertex_user_attribute(input)
        ));
    }
    src.push_str(
        r#"};

vertex Metal2VulkanValidationVertexOut metal2vulkan_validation_fullscreen_vertex(uint vid [[vertex_id]]) {
    float2 positions[3] = {
        float2(-1.0, -1.0),
        float2( 3.0, -1.0),
        float2(-1.0,  3.0),
    };
    float2 coords[3] = {
        float2(0.0, 0.0),
        float2(2.0, 0.0),
        float2(0.0, 2.0),
    };
    float2 coord = coords[vid];
    Metal2VulkanValidationVertexOut out;
"#,
    );
    src.push_str(&format!(
        "    out.{position_field} = float4(positions[vid], 0.0, 1.0);\n"
    ));
    for (index, input) in inputs.iter().enumerate() {
        let value = validation_vertex_value(&input.type_name).expect("input type was filtered");
        src.push_str(&format!("    out.{} = {value};\n", field_names[index]));
    }
    src.push_str(
        r#"    return out;
}
"#,
    );
    src
}

fn validation_vertex_field_name(input: &FragmentInput, fallback_index: usize) -> String {
    if is_msl_identifier(&input.user_name) {
        input.user_name.clone()
    } else {
        format!("user{fallback_index}")
    }
}

fn validation_vertex_user_attribute(input: &FragmentInput) -> String {
    input
        .user_attribute
        .as_deref()
        .filter(|name| is_msl_identifier(name))
        .map(|name| format!(" [[user({name})]]"))
        .unwrap_or_default()
}

fn validation_vertex_position_field_name(field_names: &[String]) -> String {
    let base = "metal2vulkan_validation_position";
    if !field_names.iter().any(|name| name == base) {
        return base.to_string();
    }
    for index in 0.. {
        let candidate = format!("{base}_{index}");
        if !field_names.iter().any(|name| name == &candidate) {
            return candidate;
        }
    }
    unreachable!("unbounded suffix search must find a free field name")
}

fn fragment_inputs(sanitized_ll: &str) -> Vec<FragmentInput> {
    sanitized_ll
        .lines()
        .filter(|line| line.contains(r#""air.fragment_input""#))
        .filter(|line| !line.contains(r#""air.arg_unused""#))
        .filter_map(|line| {
            let user_attribute =
                metadata_string_after(line, "air.fragment_input").and_then(user_attribute_name);
            let user_name = metadata_string_after(line, "air.arg_name")?;
            let type_name = metadata_string_after(line, "air.arg_type_name")?;
            Some(FragmentInput {
                user_name,
                user_attribute,
                type_name,
            })
        })
        .collect()
}

fn user_attribute_name(interpolation: String) -> Option<String> {
    interpolation
        .strip_prefix("user(")?
        .strip_suffix(')')
        .map(str::to_string)
}

fn validation_vertex_value(type_name: &str) -> Option<&'static str> {
    Some(match type_name {
        "float" => "0.25f",
        "float2" => "coord",
        "float3" => "float3(coord, 0.5f)",
        "float4" => "float4(coord, 0.5f, 1.0f)",
        "half" => "half(0.25)",
        "half2" => "half2(coord)",
        "half3" => "half3(half2(coord), half(0.5))",
        "half4" => "half4(half2(coord), half(0.5), half(1.0))",
        "int" => "int(1)",
        "int2" => "int2(1, 2)",
        "int3" => "int3(1, 2, 3)",
        "int4" => "int4(1, 2, 3, 4)",
        "uint" => "uint(1)",
        "uint2" => "uint2(1, 2)",
        "uint3" => "uint3(1, 2, 3)",
        "uint4" => "uint4(1, 2, 3, 4)",
        _ => return None,
    })
}

fn metadata_string_after(line: &str, marker: &str) -> Option<String> {
    let marker = format!("!\"{marker}\", !\"");
    let tail = line.get(line.find(&marker)? + marker.len()..)?;
    let end = tail.find('"')?;
    Some(tail[..end].to_string())
}

fn is_msl_identifier(name: &str) -> bool {
    let mut chars = name.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    (first == '_' || first.is_ascii_alphabetic())
        && chars.all(|ch| ch == '_' || ch.is_ascii_alphanumeric())
}

fn execute_vertex_library(
    device: &ProtocolObject<dyn MTLDevice>,
    library: &ProtocolObject<dyn MTLLibrary>,
    entry: &str,
    inputs: &Inputs,
    sanitized_ll: Option<&str>,
) -> Vec<u8> {
    let (index, len) = match inputs.output {
        Output::Buffer { index, len, .. } => (index, len),
        other => panic!("standalone vertex cases must use Buffer output, got {other:?}"),
    };
    assert_eq!(
        inputs.render.target.depth, 1,
        "objc2-metal oracle currently supports 2D vertex validation targets only"
    );

    let function = new_specialized_function(library, entry, sanitized_ll);
    let validation_fragment_library = compile_library(device, VALIDATION_FRAGMENT_SRC);
    let fragment_name = NSString::from_str("metal2vulkan_validation_empty_fragment");
    let fragment_function = validation_fragment_library
        .newFunctionWithName(&fragment_name)
        .expect("validation empty fragment function not found");

    let vertex_inputs = sanitized_ll.map(vertex_inputs).unwrap_or_default();
    let vertex_input_buffer_index = free_attribute_buffer_index(inputs);
    let pipeline_descriptor = MTLRenderPipelineDescriptor::new();
    pipeline_descriptor.setVertexFunction(Some(&function));
    pipeline_descriptor.setFragmentFunction(Some(&fragment_function));
    let vertex_descriptor = if vertex_inputs.is_empty() {
        None
    } else {
        let descriptor = make_vertex_descriptor(&vertex_inputs, vertex_input_buffer_index);
        pipeline_descriptor.setVertexDescriptor(Some(&descriptor));
        Some(descriptor)
    };
    let color_attachments = pipeline_descriptor.colorAttachments();
    let color_attachment = unsafe { color_attachments.objectAtIndexedSubscript(0) };
    color_attachment.setPixelFormat(MTLPixelFormat::RGBA8Unorm);
    let pipeline = device
        .newRenderPipelineStateWithDescriptor_error(&pipeline_descriptor)
        .unwrap_or_else(|e| panic!("newRenderPipelineStateWithDescriptor({entry}): {e}"));
    let _vertex_descriptor = vertex_descriptor;

    let target = make_render_target(device, DataFormat::Rgba8Unorm, inputs.render.target);
    let mut target_bytes = seeded_render_target_bytes(DataFormat::Rgba8Unorm, inputs.render.target);
    write_texture_bytes(
        &target,
        DataFormat::Rgba8Unorm,
        inputs.render.target,
        TextureKind::Plain,
        &mut target_bytes,
    );
    let pass_descriptor = MTLRenderPassDescriptor::renderPassDescriptor();
    let pass_color_attachments = pass_descriptor.colorAttachments();
    let pass_color_attachment = unsafe { pass_color_attachments.objectAtIndexedSubscript(0) };
    pass_color_attachment.setTexture(Some(&target));
    pass_color_attachment.setLoadAction(MTLLoadAction::Load);
    pass_color_attachment.setStoreAction(MTLStoreAction::Store);

    let queue = device
        .newCommandQueue()
        .expect("MTLDevice::newCommandQueue returned nil");
    let command_buffer = queue
        .commandBuffer()
        .expect("MTLCommandQueue::commandBuffer returned nil");
    let encoder = command_buffer
        .renderCommandEncoderWithDescriptor(&pass_descriptor)
        .expect("MTLCommandBuffer::renderCommandEncoderWithDescriptor returned nil");
    encoder.setRenderPipelineState(&pipeline);
    let buffers = make_buffers(device, inputs);
    let vertex_input_buffer = make_vertex_input_buffer(device, &vertex_inputs);
    let textures = make_textures(device, inputs, sanitized_ll);
    for (buffer_index, buffer) in &buffers {
        unsafe {
            encoder.setVertexBuffer_offset_atIndex(Some(&**buffer), 0, *buffer_index as usize);
        }
    }
    if let Some(buffer) = &vertex_input_buffer {
        unsafe {
            encoder.setVertexBuffer_offset_atIndex(
                Some(&**buffer),
                0,
                vertex_input_buffer_index as usize,
            );
        }
    }
    for (texture_index, texture) in &textures {
        unsafe {
            encoder.setVertexTexture_atIndex(Some(&**texture), *texture_index as usize);
        }
    }
    let default_sampler = make_default_sampler(device);
    unsafe {
        encoder.setVertexSamplerState_atIndex(Some(&*default_sampler), 0);
        encoder.drawPrimitives_vertexStart_vertexCount(
            MTLPrimitiveType::Triangle,
            0,
            inputs.render.vertex_count as usize,
        );
    }
    encoder.endEncoding();
    command_buffer.commit();
    command_buffer.waitUntilCompleted();
    if command_buffer.status() != MTLCommandBufferStatus::Completed {
        if let Some(error) = command_buffer.error() {
            panic!("Metal command buffer failed: {error}");
        }
        panic!(
            "Metal command buffer ended with status {:?}",
            command_buffer.status()
        );
    }

    let buffer = buffers
        .iter()
        .find_map(|(buffer_index, buffer)| (*buffer_index == index).then_some(buffer))
        .unwrap_or_else(|| panic!("output buffer index {index} was not bound"));
    assert!(
        buffer.length() >= len,
        "output buffer index {index} has length {}, expected at least {len}",
        buffer.length()
    );
    unsafe {
        let ptr = buffer.contents().as_ptr().cast::<u8>();
        std::slice::from_raw_parts(ptr, len).to_vec()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct VertexInput {
    location: u32,
    type_name: String,
}

fn vertex_inputs(sanitized_ll: &str) -> Vec<VertexInput> {
    sanitized_ll
        .lines()
        .filter(|line| line.contains(r#""air.vertex_input""#))
        .filter_map(|line| {
            let location = metadata_i32_after(line, "air.location_index")?;
            let type_name = metadata_string_after(line, "air.arg_type_name")?;
            Some(VertexInput {
                location,
                type_name,
            })
        })
        .collect()
}

fn compute_stage_inputs(sanitized_ll: &str) -> Vec<VertexInput> {
    sanitized_ll
        .lines()
        .filter(|line| line.contains(r#""air.stage_in""#))
        .filter_map(|line| {
            let location = metadata_i32_after(line, "air.location_index")?;
            let type_name = metadata_string_after(line, "air.arg_type_name")?;
            Some(VertexInput {
                location,
                type_name,
            })
        })
        .collect()
}

fn free_attribute_buffer_index(inputs: &Inputs) -> u32 {
    (0..31)
        .find(|index| !inputs.buffers.iter().any(|buffer| buffer.index == *index))
        .unwrap_or(30)
}

fn make_compute_stage_input_descriptor(
    inputs: &[VertexInput],
    buffer_index: u32,
) -> Retained<MTLStageInputOutputDescriptor> {
    let descriptor = MTLStageInputOutputDescriptor::stageInputOutputDescriptor();
    let attributes = descriptor.attributes();
    let mut offset = 0usize;
    for input in inputs {
        assert!(
            input.location < 31,
            "stage input attribute location {} exceeds Metal validation descriptor limit",
            input.location
        );
        let (format, size) =
            stage_attribute_format_and_size(&input.type_name).unwrap_or_else(|| {
                panic!(
                    "objc2-metal oracle does not support stage input type {:?}",
                    input.type_name
                )
            });
        let attribute = unsafe { attributes.objectAtIndexedSubscript(input.location as usize) };
        attribute.setFormat(format);
        attribute.setOffset(offset);
        unsafe {
            attribute.setBufferIndex(buffer_index as usize);
        }
        offset += size;
    }

    let layouts = descriptor.layouts();
    let layout = unsafe { layouts.objectAtIndexedSubscript(buffer_index as usize) };
    layout.setStride(offset.max(1));
    layout.setStepFunction(MTLStepFunction::ThreadPositionInGridX);
    layout.setStepRate(1);
    descriptor
}

fn make_compute_stage_input_buffer(
    device: &ProtocolObject<dyn MTLDevice>,
    inputs: &[VertexInput],
    oracle_inputs: &Inputs,
) -> Option<MetalBuffer> {
    if inputs.is_empty() {
        return None;
    }

    let stride = attribute_stride(inputs, stage_attribute_format_and_size, "stage input");
    let elements = oracle_inputs.dispatch.threads_per_grid[0].max(1) as usize;
    let mut bytes = Vec::with_capacity(stride * elements);
    for element in 0..elements {
        for input in inputs {
            append_vertex_attribute_value(&mut bytes, &input.type_name, element);
        }
    }
    let ptr = NonNull::new(bytes.as_mut_ptr().cast::<c_void>()).expect("Vec pointer is null");
    Some(
        unsafe {
            device.newBufferWithBytes_length_options(
                ptr,
                bytes.len(),
                MTLResourceOptions::StorageModeShared,
            )
        }
        .unwrap_or_else(|| {
            panic!(
                "new compute stage input buffer(length={}) returned nil",
                bytes.len()
            )
        }),
    )
}

fn make_vertex_descriptor(
    inputs: &[VertexInput],
    buffer_index: u32,
) -> Retained<MTLVertexDescriptor> {
    let descriptor = MTLVertexDescriptor::vertexDescriptor();
    let attributes = descriptor.attributes();
    let mut offset = 0usize;
    for input in inputs {
        assert!(
            input.location < 31,
            "vertex attribute location {} exceeds Metal validation descriptor limit",
            input.location
        );
        let (format, size) = vertex_format_and_size(&input.type_name).unwrap_or_else(|| {
            panic!(
                "objc2-metal oracle does not support vertex input type {:?}",
                input.type_name
            )
        });
        let attribute = unsafe { attributes.objectAtIndexedSubscript(input.location as usize) };
        attribute.setFormat(format);
        unsafe {
            attribute.setOffset(offset);
            attribute.setBufferIndex(buffer_index as usize);
        }
        offset += size;
    }

    let layouts = descriptor.layouts();
    let layout = unsafe { layouts.objectAtIndexedSubscript(buffer_index as usize) };
    unsafe {
        layout.setStride(offset.max(1));
        layout.setStepRate(1);
    }
    layout.setStepFunction(MTLVertexStepFunction::PerVertex);
    descriptor
}

fn make_vertex_input_buffer(
    device: &ProtocolObject<dyn MTLDevice>,
    inputs: &[VertexInput],
) -> Option<MetalBuffer> {
    if inputs.is_empty() {
        return None;
    }

    let stride = attribute_stride(inputs, vertex_format_and_size, "vertex input");
    let mut bytes = Vec::with_capacity(stride * 3);
    for vertex in 0..3 {
        for input in inputs {
            append_vertex_attribute_value(&mut bytes, &input.type_name, vertex);
        }
    }
    let ptr = NonNull::new(bytes.as_mut_ptr().cast::<c_void>()).expect("Vec pointer is null");
    Some(
        unsafe {
            device.newBufferWithBytes_length_options(
                ptr,
                bytes.len(),
                MTLResourceOptions::StorageModeShared,
            )
        }
        .unwrap_or_else(|| {
            panic!(
                "new vertex input buffer(length={}) returned nil",
                bytes.len()
            )
        }),
    )
}

fn attribute_stride<F, T>(inputs: &[VertexInput], format_and_size: F, label: &str) -> usize
where
    F: Fn(&str) -> Option<(T, usize)>,
{
    inputs
        .iter()
        .map(|input| {
            format_and_size(&input.type_name)
                .unwrap_or_else(|| panic!("unsupported {label} type {:?}", input.type_name))
                .1
        })
        .sum()
}

fn stage_attribute_format_and_size(type_name: &str) -> Option<(MTLAttributeFormat, usize)> {
    Some(match type_name {
        "float" => (MTLAttributeFormat::Float, 4),
        "float2" => (MTLAttributeFormat::Float2, 8),
        "float3" => (MTLAttributeFormat::Float3, 12),
        "float4" => (MTLAttributeFormat::Float4, 16),
        "half" => (MTLAttributeFormat::Half, 2),
        "half2" => (MTLAttributeFormat::Half2, 4),
        "half3" => (MTLAttributeFormat::Half3, 6),
        "half4" => (MTLAttributeFormat::Half4, 8),
        "int" => (MTLAttributeFormat::Int, 4),
        "int2" => (MTLAttributeFormat::Int2, 8),
        "int3" => (MTLAttributeFormat::Int3, 12),
        "int4" => (MTLAttributeFormat::Int4, 16),
        "uint" => (MTLAttributeFormat::UInt, 4),
        "uint2" => (MTLAttributeFormat::UInt2, 8),
        "uint3" => (MTLAttributeFormat::UInt3, 12),
        "uint4" => (MTLAttributeFormat::UInt4, 16),
        "short" => (MTLAttributeFormat::Short, 2),
        "short2" => (MTLAttributeFormat::Short2, 4),
        "short3" => (MTLAttributeFormat::Short3, 6),
        "short4" => (MTLAttributeFormat::Short4, 8),
        "ushort" => (MTLAttributeFormat::UShort, 2),
        "ushort2" => (MTLAttributeFormat::UShort2, 4),
        "ushort3" => (MTLAttributeFormat::UShort3, 6),
        "ushort4" => (MTLAttributeFormat::UShort4, 8),
        "char" => (MTLAttributeFormat::Char, 1),
        "char2" => (MTLAttributeFormat::Char2, 2),
        "char3" => (MTLAttributeFormat::Char3, 3),
        "char4" => (MTLAttributeFormat::Char4, 4),
        "uchar" => (MTLAttributeFormat::UChar, 1),
        "uchar2" => (MTLAttributeFormat::UChar2, 2),
        "uchar3" => (MTLAttributeFormat::UChar3, 3),
        "uchar4" => (MTLAttributeFormat::UChar4, 4),
        _ => return None,
    })
}

fn vertex_format_and_size(type_name: &str) -> Option<(MTLVertexFormat, usize)> {
    Some(match type_name {
        "float" => (MTLVertexFormat::Float, 4),
        "float2" => (MTLVertexFormat::Float2, 8),
        "float3" => (MTLVertexFormat::Float3, 12),
        "float4" => (MTLVertexFormat::Float4, 16),
        "half" => (MTLVertexFormat::Half, 2),
        "half2" => (MTLVertexFormat::Half2, 4),
        "half3" => (MTLVertexFormat::Half3, 6),
        "half4" => (MTLVertexFormat::Half4, 8),
        "int" => (MTLVertexFormat::Int, 4),
        "int2" => (MTLVertexFormat::Int2, 8),
        "int3" => (MTLVertexFormat::Int3, 12),
        "int4" => (MTLVertexFormat::Int4, 16),
        "uint" => (MTLVertexFormat::UInt, 4),
        "uint2" => (MTLVertexFormat::UInt2, 8),
        "uint3" => (MTLVertexFormat::UInt3, 12),
        "uint4" => (MTLVertexFormat::UInt4, 16),
        "short" => (MTLVertexFormat::Short, 2),
        "short2" => (MTLVertexFormat::Short2, 4),
        "short3" => (MTLVertexFormat::Short3, 6),
        "short4" => (MTLVertexFormat::Short4, 8),
        "ushort" => (MTLVertexFormat::UShort, 2),
        "ushort2" => (MTLVertexFormat::UShort2, 4),
        "ushort3" => (MTLVertexFormat::UShort3, 6),
        "ushort4" => (MTLVertexFormat::UShort4, 8),
        "char" => (MTLVertexFormat::Char, 1),
        "char2" => (MTLVertexFormat::Char2, 2),
        "char3" => (MTLVertexFormat::Char3, 3),
        "char4" => (MTLVertexFormat::Char4, 4),
        "uchar" => (MTLVertexFormat::UChar, 1),
        "uchar2" => (MTLVertexFormat::UChar2, 2),
        "uchar3" => (MTLVertexFormat::UChar3, 3),
        "uchar4" => (MTLVertexFormat::UChar4, 4),
        _ => return None,
    })
}

fn append_vertex_attribute_value(out: &mut Vec<u8>, type_name: &str, vertex: usize) {
    let floats = vertex_float_values(vertex);
    match type_name {
        "float" => push_f32s(out, &floats[0..1]),
        "float2" => push_f32s(out, &floats[0..2]),
        "float3" => push_f32s(out, &floats[0..3]),
        "float4" => push_f32s(out, &floats),
        "half" => push_half_zeros(out, 1),
        "half2" => push_half_zeros(out, 2),
        "half3" => push_half_zeros(out, 3),
        "half4" => push_half_zeros(out, 4),
        "int" => push_i32s(out, &[1 + vertex as i32]),
        "int2" => push_i32s(out, &[1 + vertex as i32, 2]),
        "int3" => push_i32s(out, &[1 + vertex as i32, 2, 3]),
        "int4" => push_i32s(out, &[1 + vertex as i32, 2, 3, 4]),
        "uint" => push_u32s(out, &[1 + vertex as u32]),
        "uint2" => push_u32s(out, &[1 + vertex as u32, 2]),
        "uint3" => push_u32s(out, &[1 + vertex as u32, 2, 3]),
        "uint4" => push_u32s(out, &[1 + vertex as u32, 2, 3, 4]),
        "short" => push_i16s(out, &[1 + vertex as i16]),
        "short2" => push_i16s(out, &[1 + vertex as i16, 2]),
        "short3" => push_i16s(out, &[1 + vertex as i16, 2, 3]),
        "short4" => push_i16s(out, &[1 + vertex as i16, 2, 3, 4]),
        "ushort" => push_u16s(out, &[1 + vertex as u16]),
        "ushort2" => push_u16s(out, &[1 + vertex as u16, 2]),
        "ushort3" => push_u16s(out, &[1 + vertex as u16, 2, 3]),
        "ushort4" => push_u16s(out, &[1 + vertex as u16, 2, 3, 4]),
        "char" => out.push(1 + vertex as u8),
        "char2" => out.extend_from_slice(&[1 + vertex as u8, 2]),
        "char3" => out.extend_from_slice(&[1 + vertex as u8, 2, 3]),
        "char4" => out.extend_from_slice(&[1 + vertex as u8, 2, 3, 4]),
        "uchar" => out.push(1 + vertex as u8),
        "uchar2" => out.extend_from_slice(&[1 + vertex as u8, 2]),
        "uchar3" => out.extend_from_slice(&[1 + vertex as u8, 2, 3]),
        "uchar4" => out.extend_from_slice(&[1 + vertex as u8, 2, 3, 4]),
        _ => panic!("unsupported vertex input type {type_name:?}"),
    }
}

fn vertex_float_values(vertex: usize) -> [f32; 4] {
    match vertex {
        0 => [-1.0, -1.0, 0.0, 1.0],
        1 => [3.0, -1.0, 0.0, 1.0],
        _ => [-1.0, 3.0, 0.0, 1.0],
    }
}

fn push_f32s(out: &mut Vec<u8>, values: &[f32]) {
    for value in values {
        out.extend_from_slice(&value.to_le_bytes());
    }
}

fn push_i32s(out: &mut Vec<u8>, values: &[i32]) {
    for value in values {
        out.extend_from_slice(&value.to_le_bytes());
    }
}

fn push_u32s(out: &mut Vec<u8>, values: &[u32]) {
    for value in values {
        out.extend_from_slice(&value.to_le_bytes());
    }
}

fn push_i16s(out: &mut Vec<u8>, values: &[i16]) {
    for value in values {
        out.extend_from_slice(&value.to_le_bytes());
    }
}

fn push_u16s(out: &mut Vec<u8>, values: &[u16]) {
    for value in values {
        out.extend_from_slice(&value.to_le_bytes());
    }
}

fn push_half_zeros(out: &mut Vec<u8>, components: usize) {
    out.extend(std::iter::repeat_n(0, components * 2));
}

fn compile_library(
    device: &ProtocolObject<dyn MTLDevice>,
    metal_src: &str,
) -> Retained<ProtocolObject<dyn MTLLibrary>> {
    let source = NSString::from_str(metal_src);
    device
        .newLibraryWithSource_options_error(&source, None)
        .unwrap_or_else(|e| panic!("newLibraryWithSource failed: {e}"))
}

fn load_library_from_url(
    device: &ProtocolObject<dyn MTLDevice>,
    path: &Path,
) -> Retained<ProtocolObject<dyn MTLLibrary>> {
    let path = NSString::from_str(
        path.to_str()
            .unwrap_or_else(|| panic!("metallib path is not UTF-8: {}", path.display())),
    );
    let url = NSURL::fileURLWithPath(&path);
    device
        .newLibraryWithURL_error(&url)
        .unwrap_or_else(|e| panic!("newLibraryWithURL failed: {e}"))
}

fn make_default_sampler(device: &ProtocolObject<dyn MTLDevice>) -> MetalSampler {
    let descriptor = MTLSamplerDescriptor::new();
    device
        .newSamplerStateWithDescriptor(&descriptor)
        .expect("newSamplerStateWithDescriptor returned nil")
}

fn make_depth_stencil_state(device: &ProtocolObject<dyn MTLDevice>) -> MetalDepthStencilState {
    let descriptor = MTLDepthStencilDescriptor::new();
    descriptor.setDepthCompareFunction(MTLCompareFunction::Always);
    descriptor.setDepthWriteEnabled(true);
    device
        .newDepthStencilStateWithDescriptor(&descriptor)
        .expect("newDepthStencilStateWithDescriptor returned nil")
}

fn is_depth_format(format: DataFormat) -> bool {
    matches!(
        format,
        DataFormat::Depth32Float | DataFormat::Depth24Stencil8
    )
}

fn make_buffers(
    device: &ProtocolObject<dyn MTLDevice>,
    inputs: &Inputs,
) -> Vec<(u32, MetalBuffer)> {
    inputs
        .buffers
        .iter()
        .map(|input| {
            let mut bytes = seeded_buffer_bytes(input);
            let buffer = if bytes.is_empty() {
                device
                    .newBufferWithLength_options(0, MTLResourceOptions::StorageModeShared)
                    .unwrap_or_else(|| panic!("newBufferWithLength({}) returned nil", input.len))
            } else {
                let ptr =
                    NonNull::new(bytes.as_mut_ptr().cast::<c_void>()).expect("Vec pointer is null");
                unsafe {
                    device.newBufferWithBytes_length_options(
                        ptr,
                        bytes.len(),
                        MTLResourceOptions::StorageModeShared,
                    )
                }
                .unwrap_or_else(|| {
                    panic!("newBufferWithBytes(length={}) returned nil", bytes.len())
                })
            };
            (input.index, buffer)
        })
        .collect()
}

fn make_textures(
    device: &ProtocolObject<dyn MTLDevice>,
    inputs: &Inputs,
    sanitized_ll: Option<&str>,
) -> Vec<(u32, MetalTexture)> {
    inputs
        .textures
        .iter()
        .map(|input| {
            let kind = texture_kind(sanitized_ll, input.index);
            let shape_extent = texture_seed_extent(input.extent, kind);
            let mut bytes = texture_seed_bytes(input, kind, shape_extent);
            let descriptor = texture_descriptor(input.format, shape_extent, kind);
            descriptor.setUsage(metal_texture_usage(input.role));
            descriptor.setStorageMode(MTLStorageMode::Shared);
            let texture = device
                .newTextureWithDescriptor(&descriptor)
                .unwrap_or_else(|| panic!("newTextureWithDescriptor({:?}) returned nil", input));
            if !bytes.is_empty() {
                write_texture_bytes(&texture, input.format, shape_extent, kind, &mut bytes);
            }
            (input.index, texture)
        })
        .collect()
}

fn write_texture_bytes(
    texture: &MetalTexture,
    format: DataFormat,
    extent: Extent3d,
    kind: TextureKind,
    bytes: &mut [u8],
) {
    let stride = format
        .bytes_per_pixel()
        .unwrap_or_else(|| panic!("texture format {format:?} has no pixel stride"));
    let width = extent.width as usize;
    let height = extent.height as usize;
    let depth = extent.depth as usize;
    if matches!(kind, TextureKind::Dim2dArray | TextureKind::Cube) {
        let layer_stride = width * height * stride;
        for layer in 0..texture_layer_count(extent, kind) {
            let ptr = NonNull::new(unsafe { bytes.as_mut_ptr().add(layer * layer_stride) }.cast())
                .expect("slice pointer is null");
            let region = MTLRegion {
                origin: MTLOrigin { x: 0, y: 0, z: 0 },
                size: MTLSize {
                    width,
                    height,
                    depth: 1,
                },
            };
            unsafe {
                texture.replaceRegion_mipmapLevel_slice_withBytes_bytesPerRow_bytesPerImage(
                    region,
                    0,
                    layer,
                    ptr,
                    width * stride,
                    layer_stride,
                );
            }
        }
    } else if depth > 1 {
        let ptr = NonNull::new(bytes.as_mut_ptr().cast::<c_void>()).expect("slice pointer is null");
        let region = MTLRegion {
            origin: MTLOrigin { x: 0, y: 0, z: 0 },
            size: MTLSize {
                width,
                height,
                depth,
            },
        };
        unsafe {
            texture.replaceRegion_mipmapLevel_slice_withBytes_bytesPerRow_bytesPerImage(
                region,
                0,
                0,
                ptr,
                width * stride,
                width * height * stride,
            );
        }
    } else {
        let ptr = NonNull::new(bytes.as_mut_ptr().cast::<c_void>()).expect("slice pointer is null");
        let region = MTLRegion {
            origin: MTLOrigin { x: 0, y: 0, z: 0 },
            size: MTLSize {
                width,
                height,
                depth,
            },
        };
        unsafe {
            texture.replaceRegion_mipmapLevel_withBytes_bytesPerRow(region, 0, ptr, width * stride);
        }
    }
}

fn make_render_target(
    device: &ProtocolObject<dyn MTLDevice>,
    format: DataFormat,
    extent: Extent3d,
) -> MetalTexture {
    let descriptor = unsafe {
        MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
            metal_pixel_format(format),
            extent.width as usize,
            extent.height as usize,
            false,
        )
    };
    descriptor.setUsage(MTLTextureUsage::RenderTarget);
    descriptor.setStorageMode(MTLStorageMode::Shared);
    device
        .newTextureWithDescriptor(&descriptor)
        .unwrap_or_else(|| {
            panic!(
                "new render target texture {:?} {:?} returned nil",
                format, extent
            )
        })
}

fn read_texture(
    texture: &MetalTexture,
    format: DataFormat,
    extent: Extent3d,
    is_2d_array: bool,
) -> Vec<u8> {
    let stride = format
        .bytes_per_pixel()
        .unwrap_or_else(|| panic!("texture output {format:?} has no pixel stride"));
    let width = extent.width as usize;
    let height = extent.height as usize;
    let depth = extent.depth as usize;
    let mut bytes = vec![0u8; width * height * depth * stride];
    if !bytes.is_empty() {
        let ptr = NonNull::new(bytes.as_mut_ptr().cast::<c_void>()).expect("Vec pointer is null");
        let region = MTLRegion {
            origin: MTLOrigin { x: 0, y: 0, z: 0 },
            size: MTLSize {
                width,
                height,
                depth,
            },
        };
        if is_2d_array {
            let layer_stride = width * height * stride;
            for layer in 0..depth.max(1) {
                let ptr = NonNull::new(
                    unsafe { bytes.as_mut_ptr().add(layer * layer_stride) }.cast::<c_void>(),
                )
                .expect("Vec pointer is null");
                let region = MTLRegion {
                    origin: MTLOrigin { x: 0, y: 0, z: 0 },
                    size: MTLSize {
                        width,
                        height,
                        depth: 1,
                    },
                };
                unsafe {
                    texture.getBytes_bytesPerRow_bytesPerImage_fromRegion_mipmapLevel_slice(
                        ptr,
                        width * stride,
                        layer_stride,
                        region,
                        0,
                        layer,
                    );
                }
            }
        } else if depth > 1 {
            unsafe {
                texture.getBytes_bytesPerRow_bytesPerImage_fromRegion_mipmapLevel_slice(
                    ptr,
                    width * stride,
                    width * height * stride,
                    region,
                    0,
                    0,
                );
            }
        } else {
            unsafe {
                texture.getBytes_bytesPerRow_fromRegion_mipmapLevel(ptr, width * stride, region, 0);
            }
        }
    }
    bytes
}

fn texture_descriptor(
    format: DataFormat,
    extent: Extent3d,
    kind: TextureKind,
) -> Retained<MTLTextureDescriptor> {
    let descriptor = MTLTextureDescriptor::new();
    descriptor.setTextureType(metal_texture_type(extent, kind));
    descriptor.setPixelFormat(metal_pixel_format(format));
    unsafe {
        descriptor.setWidth(extent.width as usize);
        descriptor.setHeight(extent.height as usize);
        if kind == TextureKind::Dim2dArray {
            descriptor.setDepth(1);
            descriptor.setArrayLength(extent.depth.max(1) as usize);
        } else if kind == TextureKind::Cube {
            descriptor.setDepth(1);
            descriptor.setArrayLength(1);
        } else {
            descriptor.setDepth(extent.depth as usize);
        }
        descriptor.setMipmapLevelCount(1);
        descriptor.setSampleCount(1);
    }
    descriptor
}

fn metal_texture_type(extent: Extent3d, kind: TextureKind) -> MTLTextureType {
    match kind {
        TextureKind::Dim1d => MTLTextureType::Type1D,
        TextureKind::Dim3d => MTLTextureType::Type3D,
        TextureKind::Dim2dArray => MTLTextureType::Type2DArray,
        TextureKind::Cube => MTLTextureType::TypeCube,
        TextureKind::Plain if extent.depth > 1 => MTLTextureType::Type3D,
        TextureKind::Plain => MTLTextureType::Type2D,
    }
}

fn metal_texture_usage(role: TextureRole) -> MTLTextureUsage {
    match role {
        TextureRole::Sampled | TextureRole::StorageRead | TextureRole::InputAttachment => {
            MTLTextureUsage::ShaderRead
        }
        TextureRole::StorageWrite => MTLTextureUsage::ShaderWrite,
        TextureRole::StorageReadWrite => MTLTextureUsage::ShaderRead | MTLTextureUsage::ShaderWrite,
        TextureRole::ColorTarget => MTLTextureUsage::RenderTarget,
    }
}

fn metal_pixel_format(format: DataFormat) -> MTLPixelFormat {
    match format {
        DataFormat::Rgba8Unorm => MTLPixelFormat::RGBA8Unorm,
        DataFormat::Rgba8Uint => MTLPixelFormat::RGBA8Uint,
        DataFormat::Rgba8Sint => MTLPixelFormat::RGBA8Sint,
        DataFormat::Rgba16Uint => MTLPixelFormat::RGBA16Uint,
        DataFormat::R32Uint => MTLPixelFormat::R32Uint,
        DataFormat::Rg32Uint => MTLPixelFormat::RG32Uint,
        DataFormat::Rgba32Uint => MTLPixelFormat::RGBA32Uint,
        DataFormat::R32Sint => MTLPixelFormat::R32Sint,
        DataFormat::Rg32Sint => MTLPixelFormat::RG32Sint,
        DataFormat::Rgba32Sint => MTLPixelFormat::RGBA32Sint,
        DataFormat::R16Float => MTLPixelFormat::R16Float,
        DataFormat::Rg16Float => MTLPixelFormat::RG16Float,
        DataFormat::Rgba16Float => MTLPixelFormat::RGBA16Float,
        DataFormat::Rg32Float => MTLPixelFormat::RG32Float,
        DataFormat::Rgba32Float => MTLPixelFormat::RGBA32Float,
        DataFormat::R32Float => MTLPixelFormat::R32Float,
        DataFormat::Depth32Float => MTLPixelFormat::Depth32Float,
        DataFormat::Depth24Stencil8 => MTLPixelFormat::Depth24Unorm_Stencil8,
        _ => panic!("unsupported Metal texture format {format:?}"),
    }
}

fn mtl_size(size: [u32; 3]) -> MTLSize {
    MTLSize {
        width: size[0] as usize,
        height: size[1] as usize,
        depth: size[2] as usize,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn texture3d_type_name_uses_3d_texture_type_even_for_depth_one() {
        let kind = texture_kind_from_type_name(Some("texture3d<float, write>"));

        assert_eq!(kind, TextureKind::Dim3d);
        assert_eq!(
            metal_texture_type(Extent3d::new(8, 8, 1), kind),
            MTLTextureType::Type3D
        );
        assert_eq!(
            texture_seed_extent(Extent3d::new(8, 8, 1), kind),
            Extent3d::new(8, 8, 1)
        );
        assert_eq!(texture_layer_count(Extent3d::new(8, 8, 1), kind), 1);
    }

    #[test]
    fn threadgroup_memory_bindings_use_location_and_arg_size() {
        let ll = r#"
!22 = !{i32 3, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 3, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"tg_mem"}
"#;

        assert_eq!(
            threadgroup_memory_bindings(ll),
            vec![ThreadgroupMemoryBinding {
                index: 0,
                byte_len: 2048,
            }]
        );
    }

    #[test]
    fn threadgroup_memory_bindings_ignore_device_buffers() {
        let ll = r#"
!22 = !{i32 3, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"device_mem"}
"#;

        assert!(threadgroup_memory_bindings(ll).is_empty());
    }

    #[test]
    fn dynamic_resource_location_fcs_use_minimal_positive_values() {
        let ll = r#"
!air.function_constants = !{!47, !48}
!18 = !{i32 0, !"air.function_constant", !19, !"air.texture", !"air.location_index", ptr addrspace(2) @loc, i32 1, !"air.write", !"air.arg_type_name", !"texture2d<float, write>", !"air.arg_name", !"dest"}
!47 = !{ptr addrspace(2) @fc_u, !"uint", !"Count", i32 125, i1 true}
!48 = !{ptr addrspace(2) @fc_b, !"bool", !"Enabled", i32 126, i1 true}
"#;

        assert_eq!(
            dynamic_resource_location_fc_values(Some(ll)),
            Some(vec![(125, 1)])
        );
    }

    #[test]
    fn validation_vertex_matches_fragment_user_inputs() {
        let ll = r#"
!20 = !{i32 1, !"air.fragment_input", !"generated(9texcoord0Dv2_f)", !"air.center", !"air.perspective", !"air.arg_type_name", !"float2", !"air.arg_name", !"texcoord0"}
!21 = !{i32 2, !"air.fragment_input", !"generated(5colorDv4_f)", !"air.center", !"air.perspective", !"air.arg_type_name", !"float4", !"air.arg_name", !"color"}
!22 = !{i32 3, !"air.fragment_input", !"generated(12outlinecolorDv4_Dh)", !"air.center", !"air.perspective", !"air.arg_type_name", !"half4", !"air.arg_name", !"outlinecolor"}
"#;

        assert_eq!(
            fragment_inputs(ll),
            vec![
                FragmentInput {
                    user_name: "texcoord0".to_string(),
                    user_attribute: None,
                    type_name: "float2".to_string(),
                },
                FragmentInput {
                    user_name: "color".to_string(),
                    user_attribute: None,
                    type_name: "float4".to_string(),
                },
                FragmentInput {
                    user_name: "outlinecolor".to_string(),
                    user_attribute: None,
                    type_name: "half4".to_string(),
                },
            ]
        );
        let src = validation_vertex_src_for_fragment(Some(ll));
        assert!(src.contains("float2 texcoord0;"));
        assert!(src.contains("float4 color;"));
        assert!(src.contains("half4 outlinecolor;"));
        assert!(src.contains("out.texcoord0 = coord;"));
        assert!(src.contains("out.color = float4(coord, 0.5f, 1.0f);"));
        assert!(src.contains("out.outlinecolor = half4(half2(coord), half(0.5), half(1.0));"));
    }

    #[test]
    fn validation_vertex_emits_explicit_user_attribute_inputs() {
        let ll = r#"
!20 = !{i32 1, !"air.fragment_input", !"user(texturecoord)", !"air.center", !"air.perspective", !"air.arg_type_name", !"float2", !"air.arg_name", !"texCoord"}
"#;

        assert_eq!(
            fragment_inputs(ll),
            vec![FragmentInput {
                user_name: "texCoord".to_string(),
                user_attribute: Some("texturecoord".to_string()),
                type_name: "float2".to_string(),
            }]
        );
        let src = validation_vertex_src_for_fragment(Some(ll));
        assert!(src.contains("float2 texCoord [[user(texturecoord)]];"));
        assert!(src.contains("out.texCoord = coord;"));
    }

    #[test]
    fn validation_vertex_keeps_position_user_input_distinct_from_builtin() {
        let ll = r#"
!20 = !{i32 1, !"air.fragment_input", !"generated(8positionDv3_f)", !"air.center", !"air.perspective", !"air.arg_type_name", !"float3", !"air.arg_name", !"position"}
"#;

        let src = validation_vertex_src_for_fragment(Some(ll));
        assert!(src.contains("float4 metal2vulkan_validation_position [[position]];"));
        assert!(src.contains("float3 position;"));
        assert!(src
            .contains("out.metal2vulkan_validation_position = float4(positions[vid], 0.0, 1.0);"));
        assert!(src.contains("out.position = float3(coord, 0.5f);"));
    }

    #[test]
    fn vertex_inputs_parse_attribute_locations_and_payload_size() {
        let ll = r#"
!20 = !{i32 0, !"air.vertex_input", !"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"float3", !"air.arg_name", !"position"}
!21 = !{i32 1, !"air.vertex_input", !"air.location_index", i32 6, i32 1, !"air.arg_type_name", !"float2", !"air.arg_name", !"texcoord0"}
"#;
        let inputs = vertex_inputs(ll);
        assert_eq!(
            inputs,
            vec![
                VertexInput {
                    location: 0,
                    type_name: "float3".to_string(),
                },
                VertexInput {
                    location: 6,
                    type_name: "float2".to_string(),
                },
            ]
        );

        let mut bytes = Vec::new();
        for input in &inputs {
            append_vertex_attribute_value(&mut bytes, &input.type_name, 0);
        }
        assert_eq!(bytes.len(), 20);
    }

    #[test]
    fn compute_stage_inputs_parse_function_constant_decorated_attributes() {
        let ll = r#"
!19 = !{i32 1, !"air.stage_in", !"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"float3", !"air.arg_name", !"position"}
!20 = !{i32 2, !"air.function_constant", !21, !"air.stage_in", !"air.location_index", i32 1, i32 1, !"air.arg_type_name", !"float3", !"air.arg_name", !"normal"}
"#;
        let inputs = compute_stage_inputs(ll);
        assert_eq!(
            inputs,
            vec![
                VertexInput {
                    location: 0,
                    type_name: "float3".to_string(),
                },
                VertexInput {
                    location: 1,
                    type_name: "float3".to_string(),
                },
            ]
        );
        assert_eq!(
            attribute_stride(&inputs, stage_attribute_format_and_size, "stage input"),
            24
        );
    }

    #[test]
    fn visible_function_reference_names_follow_air_metadata() {
        let ll = r#"
!air.visible_function_references = !{!31, !32, !33, !32}
!31 = !{!"air.visible_function_reference", ptr @first.MTL_VISIBLE_FN_REF, !"first"}
!32 = !{!"air.visible_function_reference", ptr @second.MTL_VISIBLE_FN_REF, !"second"}
!33 = !{!"not_a_visible_function_reference", !"ignored"}
"#;

        assert_eq!(
            visible_function_reference_names(ll),
            vec!["first".to_string(), "second".to_string()]
        );
    }

    /// End-to-end on the real Apple toolchain, but WITHOUT dispatching to the GPU: compile a kernel
    /// with a data-dependent (effectively unbounded) loop to AIR, disassemble it, run the loop-budget
    /// instrumentation, and prove `metal-as` + `metallib` accept the instrumented module. This is the
    /// dialect-validity check for the transform (opaque `ptr`, alloca budget, phi-predecessor rename)
    /// — the riskiest assumption — with none of the wedge risk of actually running the loop.
    /// Skips cleanly when the Metal toolchain is unavailable.
    #[test]
    fn instrumented_infinite_loop_kernel_reassembles_and_links() {
        if command_stdout("xcrun", &["--find", "metal"]).is_err() {
            eprintln!("skip: no Metal toolchain");
            return;
        }
        let src = r#"
#include <metal_stdlib>
using namespace metal;
kernel void m2v_spin(device uint *buf [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
    // Data-dependent trip count `n` (from the buffer) with a side-effecting accumulator: the
    // optimizer cannot bound or fold this, so it survives as a real loop — the exact wedge shape.
    uint n = buf[tid];
    uint acc = tid;
    for (uint i = 0; i < n; i++) { acc = acc * 1664525u + 1013904223u; }
    buf[tid] = acc;
}
"#;
        let tmp = scratch_dir_for("loopbudget-roundtrip");
        let metal = tmp.join("case.metal");
        let air = tmp.join("case.air");
        let metallib = tmp.join("case.metallib");
        fs::write(&metal, src).expect("write metal");
        if command_stdout(
            "xcrun",
            &[
                "-sdk",
                "macosx",
                "metal",
                "-c",
                "-o",
                air.to_str().unwrap(),
                metal.to_str().unwrap(),
            ],
        )
        .is_err()
        {
            eprintln!("skip: metal -c failed");
            return;
        }
        let text = disassembled_module_text(&air).expect("metal-objdump");
        match crate::loop_budget::classify_and_instrument(&text, "m2v_spin") {
            crate::loop_budget::GuardPlan::Instrumented(instrumented) => {
                assert!(instrumented.contains("m2v.exit:"), "no exit block");
                assert!(
                    instrumented.contains("%m2v.bd = alloca i32"),
                    "no budget alloca"
                );
                // The crux: the instrumented IR is valid Apple IR that re-links to a metallib.
                assemble_and_link(&instrumented, &air, &metallib, "roundtrip_test")
                    .expect("instrumented IR must re-assemble + link via metal-as/metallib");
                assert!(metallib.is_file(), "no metallib produced");
            }
            other => panic!("expected Instrumented for a while-loop kernel, got {other:?}"),
        }
    }
}
