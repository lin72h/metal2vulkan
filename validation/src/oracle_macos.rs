#[cfg(test)]
use crate::texture::texture_kind_from_type_name;
use crate::texture::{texture_kind, texture_seed_bytes, TextureKind};
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
use objc2_foundation::{NSString, NSURL};
use objc2_metal::{
    MTLBlendFactor, MTLBlendOperation, MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus,
    MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder, MTLCreateSystemDefaultDevice,
    MTLDevice, MTLFunction, MTLFunctionConstantValues, MTLLibrary, MTLLoadAction, MTLOrigin,
    MTLPixelFormat, MTLPrimitiveType, MTLRegion, MTLRenderCommandEncoder, MTLRenderPassDescriptor,
    MTLRenderPipelineDescriptor, MTLResourceOptions, MTLResourceUsage, MTLSamplerDescriptor,
    MTLSamplerState, MTLSize, MTLStorageMode, MTLStoreAction, MTLTexture, MTLTextureDescriptor,
    MTLTextureType, MTLTextureUsage,
};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

#[link(name = "CoreGraphics", kind = "framework")]
extern "C" {}

type MetalBuffer = Retained<ProtocolObject<dyn MTLBuffer>>;
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

pub fn assert_toolchain_pinned() {
    let lock_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("TOOLCHAIN.lock");
    let current =
        current_toolchain_lock().unwrap_or_else(|e| panic!("toolchain probe failed: {e}"));
    match fs::read_to_string(&lock_path) {
        Ok(pinned) if normalize_lock(&pinned) == normalize_lock(&current) => {}
        Ok(pinned) => {
            panic!(
                "Apple toolchain drifted from {}\n--- pinned ---\n{}\n--- current ---\n{}",
                lock_path.display(),
                pinned,
                current
            );
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            fs::write(&lock_path, &current)
                .unwrap_or_else(|err| panic!("write {}: {err}", lock_path.display()));
            eprintln!("created {}", lock_path.display());
        }
        Err(e) => panic!("read {}: {e}", lock_path.display()),
    }
}

pub fn compile_to_sanitized_ll(metal_src: &str, stage: Stage) -> String {
    let tmp = scratch_dir_for(match stage {
        Stage::Kernel => "oracle-kernel",
        Stage::Vertex => "oracle-vertex",
        Stage::Fragment => "oracle-fragment",
    });
    let src = tmp.join("case.metal");
    let air = tmp.join("case.air");
    let ll = tmp.join("case.ll");
    fs::write(&src, metal_src).unwrap_or_else(|e| panic!("write {}: {e}", src.display()));

    run(
        "xcrun",
        &[
            "-sdk",
            "macosx",
            "metal",
            "-c",
            "-o",
            air.to_str().unwrap(),
            src.to_str().unwrap(),
        ],
    );
    let llvm_dis = llvm_dis_program();
    run(
        &llvm_dis,
        &[air.to_str().unwrap(), "-o", ll.to_str().unwrap()],
    );
    let text = fs::read_to_string(&ll).unwrap_or_else(|e| panic!("read {}: {e}", ll.display()));
    sanitize_llvm_ir(&text)
}

pub fn execute(metal_src: &str, stage: Stage, inputs: &crate::Inputs) -> Vec<u8> {
    autoreleasepool(|_| match stage {
        Stage::Kernel => execute_compute(metal_src, inputs),
        Stage::Fragment => execute_render_fragment(metal_src, inputs),
        Stage::Vertex => execute_vertex(metal_src, inputs),
    })
}

pub fn execute_metallib_blob(
    blob_b64: &str,
    function_name: &str,
    stage: Stage,
    inputs: &crate::Inputs,
    sanitized_ll: &str,
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
        // Deterministic threadgroup memory: a kernel with threadgroup regions (module addrspace(3)
        // globals or addrspace(3) kernel arguments) gets a zero-fill prologue injected through the
        // Apple textual round-trip, so reads of never-written threadgroup slots see ZERO instead of
        // leftover GPU memory — the same defined refinement metal2vulkan's Workgroup zero-init pass
        // applies on the candidate side. The addrspace(3) probe on the sanitized IR is purely
        // structural and skips the round-trip for the (vast) majority of kernels without it.
        let zero_init_text = if sanitized_ll.contains("addrspace(3)") {
            let text = disassembled_module_text(&air_path)
                .unwrap_or_else(|e| panic!("metal-objdump for threadgroup zero-init: {e}"));
            insert_threadgroup_zero_init(&text, function_name)
        } else {
            None
        };
        if let Some(text) = zero_init_text {
            if let Err(e) = assemble_and_link(&text, &air_path, &metallib_path, "case_zeroinit") {
                // Same stdlib-libcall symbol collision the plain path retries on (below): apply the
                // rename to the SAME instrumented text and relink.
                let Some(sym) = multiple_symbols_name(&e) else {
                    panic!("threadgroup zero-init relink failed: {e}");
                };
                let (renamed, entry) = rename_symbol_in_module_text(&text, &sym, function_name);
                assemble_and_link(&renamed, &air_path, &metallib_path, "case_zeroinit_renamed")
                    .unwrap_or_else(|retry_err| {
                        panic!(
                            "threadgroup zero-init relink failed: {e}; rename retry for {sym:?} \
                             failed: {retry_err}"
                        )
                    });
                effective_name = entry;
                eprintln!(
                    "[oracle] metallib symbol collision on {sym:?}; relinked with renamed entry {effective_name:?}"
                );
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

        let device =
            MTLCreateSystemDefaultDevice().expect("MTLCreateSystemDefaultDevice returned nil");
        let library = load_library_from_url(&device, &metallib_path);
        match stage {
            Stage::Kernel => execute_compute_library(
                &device,
                &library,
                &effective_name,
                inputs,
                Some(sanitized_ll),
            ),
            Stage::Vertex | Stage::Fragment => {
                panic!("metallib oracle currently supports kernel stages only")
            }
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

fn current_toolchain_lock() -> Result<String, String> {
    let metal = command_stdout("xcrun", &["metal", "--version"])?;
    let swift = command_stdout("swift", &["--version"])?;
    Ok(format!(
        "xcrun metal --version\n{}\n\nswift --version\n{}\n",
        metal.trim_end(),
        swift.trim_end()
    ))
}

fn normalize_lock(s: &str) -> String {
    // The lock pins the toolchain *version* (`metal --version` / `swift --version`), which is what
    // determines golden output. The `InstalledDir:` line is an ephemeral filesystem mount path — the
    // Metal toolchain re-mounts under a fresh cryptexd / DVTDownloads directory on reinstall while the
    // version string is byte-identical — so it carries no version semantics and must not trip the drift
    // guard. Drop it from the comparison; real version drift (the version / target lines) is still caught.
    s.lines()
        .map(str::trim_end)
        .filter(|line| !line.trim_start().starts_with("InstalledDir:"))
        .collect::<Vec<_>>()
        .join("\n")
}

fn run(cmd: &str, args: &[&str]) {
    command_stdout(cmd, args).unwrap_or_else(|e| panic!("{cmd} failed: {e}"));
}

fn llvm_dis_program() -> String {
    if let Some(path) = env::var_os("LLVM_DIS") {
        return path.to_string_lossy().into_owned();
    }
    if let Some(path) = find_on_path("llvm-dis") {
        return path.display().to_string();
    }
    for path in [
        "/opt/homebrew/opt/llvm/bin/llvm-dis",
        "/usr/local/opt/llvm/bin/llvm-dis",
    ] {
        let path = Path::new(path);
        if path.is_file() {
            return path.display().to_string();
        }
    }
    "llvm-dis".to_string()
}

fn find_on_path(program: &str) -> Option<PathBuf> {
    env::var_os("PATH").and_then(|path| {
        env::split_paths(&path)
            .map(|dir| dir.join(program))
            .find(|path| path.is_file())
    })
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

fn sanitize_llvm_ir(text: &str) -> String {
    text.lines()
        .filter_map(|line| {
            let trimmed = line.trim_start();
            if trimmed.starts_with("target triple")
                || trimmed.starts_with("target datalayout")
                || trimmed.starts_with("@llvm.global_ctors")
                || trimmed.starts_with("@llvm.global_dtors")
                || trimmed.starts_with(';')
            {
                None
            } else if trimmed.starts_with("source_filename = ") {
                Some("source_filename = \"case.metal\"".to_string())
            } else if let Some(line) = normalize_source_file_metadata(line) {
                Some(line)
            } else {
                Some(line.to_string())
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn normalize_source_file_metadata(line: &str) -> Option<String> {
    let (id, value) = line.split_once(" = !{!\"")?;
    if !value.trim_end().ends_with("case.metal\"}") {
        return None;
    }
    Some(format!("{id} = !{{!\"case.metal\"}}"))
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

fn execute_compute(metal_src: &str, inputs: &Inputs) -> Vec<u8> {
    let device = MTLCreateSystemDefaultDevice().expect("MTLCreateSystemDefaultDevice returned nil");
    let library = compile_library(&device, metal_src);
    let entry = kernel_function_name(metal_src);
    execute_compute_library(&device, &library, &entry, inputs, None)
}

fn execute_compute_library(
    device: &ProtocolObject<dyn MTLDevice>,
    library: &ProtocolObject<dyn MTLLibrary>,
    entry: &str,
    inputs: &Inputs,
    sanitized_ll: Option<&str>,
) -> Vec<u8> {
    let function = new_specialized_function(library, entry, sanitized_ll);
    let pipeline = device
        .newComputePipelineStateWithFunction_error(&function)
        .unwrap_or_else(|pso_err| {
            // PSO creation can fail even when empty-set specialization succeeded: an argument
            // whose presence is predicated on a function constant with an undef
            // `air.fc_initializer` (e.g. an `[[function_constant]]`-guarded imageblock arg)
            // defers the failure to the backend compile ("Encountered unlowered function call
            // to air.imageblock_data"). Zero-specializing every declared FC resolves the
            // predicate exactly the way metal2vulkan folds FCs, so retry PSO creation once
            // with explicit zeros before giving up.
            let zero_fn =
                new_fc_specialized_function(library, entry, sanitized_ll, &fc_values_requested())
                    .unwrap_or_else(|| {
                        panic!("newComputePipelineStateWithFunction({entry}): {pso_err}")
                    });
            device
                .newComputePipelineStateWithFunction_error(&zero_fn)
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
    for (index, buffer) in &buffers {
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&**buffer), 0, *index as usize);
        }
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

fn execute_render_fragment(metal_src: &str, inputs: &Inputs) -> Vec<u8> {
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

    let device = MTLCreateSystemDefaultDevice().expect("MTLCreateSystemDefaultDevice returned nil");
    let render_src = format!("{VALIDATION_VERTEX_SRC}\n{metal_src}");
    let library = compile_library(&device, &render_src);
    let vertex_name = NSString::from_str("metal2vulkan_validation_fullscreen_vertex");
    let vertex_function = library
        .newFunctionWithName(&vertex_name)
        .expect("validation fullscreen vertex function not found");
    let fragment_entry = fragment_function_name(metal_src);
    let fragment_name = NSString::from_str(&fragment_entry);
    let fragment_function = library
        .newFunctionWithName(&fragment_name)
        .unwrap_or_else(|| panic!("Metal fragment function {fragment_entry:?} not found"));

    let pipeline_descriptor = MTLRenderPipelineDescriptor::new();
    pipeline_descriptor.setVertexFunction(Some(&vertex_function));
    pipeline_descriptor.setFragmentFunction(Some(&fragment_function));
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
            color_attachment.setDestinationAlphaBlendFactor(MTLBlendFactor::OneMinusSourceAlpha);
            color_attachment.setAlphaBlendOperation(MTLBlendOperation::Add);
        }
    }
    let pipeline = device
        .newRenderPipelineStateWithDescriptor_error(&pipeline_descriptor)
        .unwrap_or_else(|e| panic!("newRenderPipelineStateWithDescriptor({fragment_entry}): {e}"));

    let target = make_render_target(&device, format, extent);
    let mut target_bytes = seeded_render_target_bytes(format, extent);
    write_texture_bytes(
        &target,
        format,
        extent,
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
    let buffers = make_buffers(&device, inputs);
    let textures = make_textures(&device, inputs, None);
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
    let default_sampler = make_default_sampler(&device);
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

fn execute_vertex(metal_src: &str, inputs: &Inputs) -> Vec<u8> {
    let (index, len) = match inputs.output {
        Output::Buffer { index, len, .. } => (index, len),
        other => panic!("standalone vertex cases must use Buffer output, got {other:?}"),
    };
    assert_eq!(
        inputs.render.target.depth, 1,
        "objc2-metal oracle currently supports 2D vertex validation targets only"
    );

    let device = MTLCreateSystemDefaultDevice().expect("MTLCreateSystemDefaultDevice returned nil");
    let library = compile_library(&device, metal_src);
    let entry = vertex_function_name(metal_src);
    let function_name = NSString::from_str(&entry);
    let function = library
        .newFunctionWithName(&function_name)
        .unwrap_or_else(|| panic!("Metal vertex function {entry:?} not found"));

    let pipeline_descriptor = MTLRenderPipelineDescriptor::new();
    pipeline_descriptor.setVertexFunction(Some(&function));
    pipeline_descriptor.setRasterizationEnabled(false);
    let color_attachments = pipeline_descriptor.colorAttachments();
    let color_attachment = unsafe { color_attachments.objectAtIndexedSubscript(0) };
    color_attachment.setPixelFormat(MTLPixelFormat::RGBA8Unorm);
    let pipeline = device
        .newRenderPipelineStateWithDescriptor_error(&pipeline_descriptor)
        .unwrap_or_else(|e| panic!("newRenderPipelineStateWithDescriptor({entry}): {e}"));

    let target = make_render_target(&device, DataFormat::Rgba8Unorm, inputs.render.target);
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
    let buffers = make_buffers(&device, inputs);
    let textures = make_textures(&device, inputs, None);
    for (buffer_index, buffer) in &buffers {
        unsafe {
            encoder.setVertexBuffer_offset_atIndex(Some(&**buffer), 0, *buffer_index as usize);
        }
    }
    for (texture_index, texture) in &textures {
        unsafe {
            encoder.setVertexTexture_atIndex(Some(&**texture), *texture_index as usize);
        }
    }
    let default_sampler = make_default_sampler(&device);
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

fn texture_seed_extent(extent: Extent3d, kind: TextureKind) -> Extent3d {
    match kind {
        TextureKind::Dim1d => Extent3d::new(extent.width, 1, 1),
        TextureKind::Cube => Extent3d::new(extent.width, extent.height, 6),
        TextureKind::Plain | TextureKind::Dim2dArray | TextureKind::Dim3d => extent,
    }
}

/// The extent an output texture's readback actually covers, derived from the texture's declared
/// kind rather than the caller's (2D-shaped) contract extent. A 1D texture holds one row of
/// `width` texels regardless of the contract's `h`; a cube readback covers the single face the
/// harness compares. Reading a larger region than the texture stores would silently return zeros
/// (Metal) or zero padding (Vulkan) — never real texel data — so the contract length must follow
/// the texture's real shape.
fn texture_output_extent(extent: Extent3d, kind: TextureKind) -> Extent3d {
    match kind {
        TextureKind::Dim1d => Extent3d::new(extent.width, 1, 1),
        TextureKind::Cube => Extent3d::new(extent.width, extent.height, 1),
        TextureKind::Plain | TextureKind::Dim2dArray | TextureKind::Dim3d => extent,
    }
}

fn texture_layer_count(extent: Extent3d, kind: TextureKind) -> usize {
    match kind {
        TextureKind::Dim2dArray => extent.depth.max(1) as usize,
        TextureKind::Cube => 6,
        TextureKind::Dim1d | TextureKind::Dim3d | TextureKind::Plain => 1,
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
        DataFormat::Rgba16Float => MTLPixelFormat::RGBA16Float,
        DataFormat::Rgba32Float => MTLPixelFormat::RGBA32Float,
        DataFormat::R32Float => MTLPixelFormat::R32Float,
        _ => panic!("unsupported Metal texture format {format:?}"),
    }
}

fn kernel_function_name(metal_src: &str) -> String {
    stage_function_name(metal_src, "kernel")
}

fn fragment_function_name(metal_src: &str) -> String {
    stage_function_name(metal_src, "fragment")
}

fn vertex_function_name(metal_src: &str) -> String {
    stage_function_name(metal_src, "vertex")
}

fn stage_function_name(metal_src: &str, stage_keyword: &str) -> String {
    for (pos, _) in metal_src.match_indices(stage_keyword) {
        if !is_ident_boundary(metal_src[..pos].chars().next_back())
            || !is_ident_boundary(metal_src[pos + stage_keyword.len()..].chars().next())
        {
            continue;
        }
        let after_keyword = &metal_src[pos + stage_keyword.len()..];
        let paren = after_keyword
            .find('(')
            .unwrap_or_else(|| panic!("{stage_keyword} declaration has no parameter list"));
        let head = &after_keyword[..paren];
        if let Some(name) = head
            .rsplit(|ch: char| !(ch.is_ascii_alphanumeric() || ch == '_'))
            .find(|token| !token.is_empty())
        {
            return name.to_string();
        }
    }
    panic!("no {stage_keyword} function declaration found in Metal source");
}

fn is_ident_boundary(ch: Option<char>) -> bool {
    ch.map(|ch| !(ch.is_ascii_alphanumeric() || ch == '_'))
        .unwrap_or(true)
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
    fn sanitizer_strips_target_and_global_constructor_lines() {
        let ll = r#"; ModuleID = 'x'
target triple = "air64-apple-macosx"
target datalayout = "e"
@llvm.global_ctors = appending global [0 x { i32, ptr, ptr }] []
define void @main() { ret void }
"#;
        let san = sanitize_llvm_ir(ll);
        assert!(!san.contains("target triple"));
        assert!(!san.contains("target datalayout"));
        assert!(!san.contains("@llvm.global_ctors"));
        assert!(san.contains("define void @main()"));
    }

    #[test]
    fn sanitizer_normalizes_source_paths() {
        let ll = r#"; ModuleID = '/tmp/random/case.air'
source_filename = "/tmp/random/case.metal"
!air.source_file_name = !{!0}
!0 = !{!"/tmp/random/case.metal"}
"#;
        let san = sanitize_llvm_ir(ll);
        assert!(san.contains("source_filename = \"case.metal\""));
        assert!(san.contains("!0 = !{!\"case.metal\"}"));
        assert!(!san.contains("/tmp/random"));
    }

    #[test]
    fn finds_kernel_entry_point() {
        let src = r#"
#include <metal_stdlib>
using namespace metal;
kernel void add_one(device uint *out [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
    out[tid] = 1;
}
"#;
        assert_eq!(kernel_function_name(src), "add_one");
    }

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
}
