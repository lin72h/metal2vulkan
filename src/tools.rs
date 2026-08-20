//! External-tool runners for metal2vulkan. Production SPIR-V emission is native Rust; the remaining
//! spawns (`llvm-dis` and `spirv-val`) are bounded by a timeout so a stuck tool can't hang the
//! translator.

use crate::native;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Condvar, Mutex, OnceLock};
use std::time::Duration;

/// Hard cap on any single external-tool invocation.
pub const TOOL_TIMEOUT_SECS: u64 = 60;

/// Marker substring every timeout error carries, so callers can tell a TIMEOUT (the tool was
/// starved/killed before producing a verdict) apart from a real non-zero-exit failure.
pub const TIMEOUT_MARKER: &str = "timed out after";

/// Marker substring an error carries when the tool did NOT render a verdict — it was killed by a
/// signal (e.g. the OS OOM-killer under concurrent memory pressure) or could not be spawned at all
/// (a transient `fork` EAGAIN under heavy load). Like a timeout, this is NOT a clean non-zero-exit
/// failure: the tool never decided. Callers that treat the tool's exit as a verdict (spirv-val
/// validity) MUST retry a no-verdict rather than reading it as a real failure, or the result becomes
/// nondeterministic (a valid module dropped as a false failure whenever its validator was killed).
pub const NO_VERDICT_MARKER: &str = "no verdict";

/// Global bound on the number of spirv-val subprocesses running at once, INDEPENDENT of the emit
/// parallelism (`multi-threaded validation runs`). Validation is memory-heavy — the huge Apple
/// BVH-builder modules make spirv-val allocate hundreds of MB — so spawning one validator per gate
/// worker exhausts RAM on a constrained host and the OS OOM-killer terminates spirv-val mid-run. A
/// signal-kill renders NO verdict; while [`spirv_val`]'s escalating-cap loop retries a NO_VERDICT,
/// sustained memory pressure keeps EVERY retry killed, so a genuinely-valid module is ultimately
/// dropped as a false failure and the adopt-if-validates retries fall back to the base emit error —
/// inflating the measured frontier (observed: `--threads 8` reported 60 where `--threads 1` reports
/// the true ~15, the extra 45 being false OOM-kills of valid retry modules). Bounding concurrent
/// validators keeps peak memory under the ceiling so each one actually renders its verdict, making the
/// floor a pure function of the bytes regardless of `--threads`. The cap is small by default (3) and
/// overridable with `METAL2VULKAN_VAL_PAR`; emit stays fully parallel, only the validator spawns serialize.
fn val_gate() -> &'static (Mutex<usize>, Condvar) {
    static GATE: OnceLock<(Mutex<usize>, Condvar)> = OnceLock::new();
    GATE.get_or_init(|| (Mutex::new(0usize), Condvar::new()))
}

fn val_par_limit() -> usize {
    crate::env_vars::val_par()
}

/// RAII permit for a concurrent spirv-val slot — blocks until one of [`val_par_limit`] slots is free,
/// releasing it (and waking a waiter) on drop.
struct ValPermit;
impl ValPermit {
    fn acquire() -> ValPermit {
        let limit = val_par_limit();
        let (lock, cv) = val_gate();
        let mut n = lock.lock().unwrap();
        while *n >= limit {
            n = cv.wait(n).unwrap();
        }
        *n += 1;
        ValPermit
    }
}
impl Drop for ValPermit {
    fn drop(&mut self) {
        let (lock, cv) = val_gate();
        let mut n = lock.lock().unwrap();
        *n = n.saturating_sub(1);
        cv.notify_one();
    }
}

/// Run a tool with the default [`TOOL_TIMEOUT_SECS`] cap. See [`run_with_timeout`].
pub fn run(cmd: &str, args: &[&str]) -> Result<(Vec<u8>, Vec<u8>), String> {
    run_with_timeout(cmd, args, TOOL_TIMEOUT_SECS)
}

/// Run a tool with an explicit timeout (seconds); return `(stdout, stderr)` on success, or
/// `Err(message)` on failure/timeout. Drains stdout/stderr on threads so large output can't
/// deadlock the pipe. A timeout error contains [`TIMEOUT_MARKER`].
pub fn run_with_timeout(
    cmd: &str,
    args: &[&str],
    timeout_secs: u64,
) -> Result<(Vec<u8>, Vec<u8>), String> {
    use std::io::Read;
    use std::process::Stdio;
    use wait_timeout::ChildExt;

    let bin = tool_bin(cmd);
    let mut child = Command::new(&bin)
        .args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        // A failed spawn under heavy load is usually transient (`fork` EAGAIN when the process/
        // thread table is saturated), NOT a missing tool — tag it NO_VERDICT so a validity caller
        // retries rather than reading it as "invalid".
        .map_err(|e| {
            format!(
                "cannot run {cmd} ({}) [{NO_VERDICT_MARKER}]: {e}",
                bin.display()
            )
        })?;
    let mut so = child.stdout.take().unwrap();
    let mut se = child.stderr.take().unwrap();
    let to = std::thread::spawn(move || {
        let mut v = Vec::new();
        let _ = so.read_to_end(&mut v);
        v
    });
    let te = std::thread::spawn(move || {
        let mut v = Vec::new();
        let _ = se.read_to_end(&mut v);
        v
    });
    match child.wait_timeout(Duration::from_secs(timeout_secs)) {
        Ok(Some(status)) => {
            let out = to.join().unwrap_or_default();
            let err = te.join().unwrap_or_default();
            if status.success() {
                Ok((out, err))
            } else if status.code().is_none() {
                // No exit code => the tool was terminated by a signal (on Unix), e.g. the OS
                // OOM-killer under concurrent memory pressure. It never rendered a verdict, so tag
                // NO_VERDICT: a validity caller must retry, not treat this as a real failure. A
                // genuinely-invalid input, by contrast, makes the tool EXIT (a code) in ms.
                Err(format!(
                    "{cmd} killed by signal [{NO_VERDICT_MARKER}]:\n{}",
                    String::from_utf8_lossy(&err)
                ))
            } else {
                Err(format!("{cmd} failed:\n{}", String::from_utf8_lossy(&err)))
            }
        }
        Ok(None) => {
            let _ = child.kill();
            let _ = to.join();
            let _ = te.join();
            Err(format!("{cmd} {TIMEOUT_MARKER} {timeout_secs}s -- killed"))
        }
        Err(e) => {
            let _ = child.kill();
            Err(format!("waiting on {cmd} [{NO_VERDICT_MARKER}]: {e}"))
        }
    }
}

fn tool_bin(cmd: &str) -> PathBuf {
    if let Some(path) = crate::env_vars::tool_path_override(cmd) {
        return PathBuf::from(path);
    }
    for dir in [
        "/opt/homebrew/opt/llvm/bin",
        "/usr/local/opt/llvm/bin",
        "/opt/homebrew/bin",
        "/usr/local/bin",
    ] {
        let candidate = Path::new(dir).join(cmd);
        if candidate.is_file() {
            return candidate;
        }
    }
    PathBuf::from(cmd)
}

/// Vulkan SPIR-V triple the sanitizer rewrites AIR modules to.
pub const VULKAN_TRIPLE: &str = "spirv-unknown-vulkan1.3";

/// `src` (.air bitcode or .ll) -> sanitized `.ll` text ready for the native emitter.
/// Rewrites the target triple to the Vulkan one, preserves the AIR datalayout for executable layout
/// lowering, and drops LLVM metadata/runtime root globals.
pub fn air_to_sanitized_ll(src: &str, tmp: &Path) -> Result<String, String> {
    Ok(air_to_sanitized_ll_with_datalayout(src, tmp)?.0)
}

/// Like [`air_to_sanitized_ll`] but also returns the AIR `target datalayout` string it preserves. The
/// sanitizer preserves that line as the executable source-layout contract and also returns its value
/// for reflection, avoiding a second source read. The sanitized text is byte-identical to
/// [`air_to_sanitized_ll`].
pub fn air_to_sanitized_ll_with_datalayout(
    src: &str,
    tmp: &Path,
) -> Result<(String, Option<String>), String> {
    let ll_text = if src.ends_with(".ll") {
        std::fs::read_to_string(src).map_err(|e| format!("read {src}: {e}"))?
    } else {
        let ll = tmp.join("k.ll");
        let text = (|| {
            run("llvm-dis", &[src, "-o", ll.to_str().unwrap()])?;
            std::fs::read_to_string(&ll).map_err(|e| format!("read {}: {e}", ll.display()))
        })();
        // Intermediate only needed for llvm-dis → read; drop even on tool/read failure.
        let _ = std::fs::remove_file(&ll);
        text?
    };

    Ok(sanitize_ll_text_with_datalayout(&ll_text))
}

/// Sanitize LLVM IR text that is already in `.ll` form.
///
/// This is the text-only core used by [`air_to_sanitized_ll_with_datalayout`]: rewrite the target
/// triple to Vulkan, preserve the AIR datalayout, drop llvm-dis's input-path `ModuleID` comment, and
/// remove LLVM metadata/runtime root globals that are not part of the shader entry body.
pub fn sanitize_ll_text_with_datalayout(ll_text: &str) -> (String, Option<String>) {
    let mut out = String::with_capacity(ll_text.len());
    let mut datalayout = None;
    for line in ll_text.lines() {
        let t = line.trim_start();
        if t.starts_with("target triple") {
            out.push_str(&format!("target triple = \"{VULKAN_TRIPLE}\"\n"));
            continue;
        }
        if t.starts_with("target datalayout") {
            datalayout = datalayout_value(t);
            out.push_str(line);
            out.push('\n');
            continue;
        }
        // llvm-dis derives this comment from its input filename. Validation disassembles through
        // a uniquely named scratch directory, so retaining it makes identical bitcode acquire a
        // different corpus identity on every harvest.
        if t.starts_with("; ModuleID =") {
            continue;
        }
        if t.starts_with("@llvm.global_ctors")
            || t.starts_with("@llvm.global_dtors")
            || t.starts_with("@llvm.used")
            || t.starts_with("@llvm.compiler.used")
        {
            continue;
        }
        out.push_str(line);
        out.push('\n');
    }
    (out, datalayout)
}

/// The quoted value of a `target datalayout = "..."` line.
fn datalayout_value(line: &str) -> Option<String> {
    let start = line.find('"')?;
    let rest = &line[start + 1..];
    let end = rest.find('"')?;
    Some(rest[..end].to_string())
}

/// Emit Vulkan SPIR-V from sanitized `.ll`, returning the raw SPIR-V binary words.
///
/// This is an in-process native Rust emission boundary. Unsupported IR shapes return explicit
/// errors.
pub fn emit_vulkan_spirv(san_ll: &str, _tmp: &Path) -> Result<Vec<u8>, String> {
    native::emit_vulkan_spirv(san_ll)
}

pub(crate) fn emit_vulkan_spirv_with_sidecar(
    san_ll: &str,
    _tmp: &Path,
    kern: Option<&crate::meta::KernMeta>,
    entry_name: Option<&str>,
    buffer_layouts: Option<&HashMap<u32, crate::meta::AirType>>,
) -> Result<crate::emit_sidecar::EmittedSpirv, String> {
    native::emit_vulkan_spirv_with_sidecar(san_ll, kern, entry_name, buffer_layouts)
}

/// Like [`emit_vulkan_spirv`], but enables the reject-only construct-tree own-arm candidate. Used only
/// by an adopt-if-validates retry tier after primary CFG validation failure.
pub fn emit_vulkan_spirv_construct_tree(san_ll: &str, _tmp: &Path) -> Result<Vec<u8>, String> {
    native::emit_vulkan_spirv_construct_tree(san_ll)
}

pub(crate) fn emit_vulkan_spirv_construct_tree_with_sidecar(
    san_ll: &str,
    _tmp: &Path,
    kern: Option<&crate::meta::KernMeta>,
    entry_name: Option<&str>,
    buffer_layouts: Option<&HashMap<u32, crate::meta::AirType>>,
) -> Result<crate::emit_sidecar::EmittedSpirv, String> {
    native::emit_vulkan_spirv_construct_tree_with_sidecar(san_ll, kern, entry_name, buffer_layouts)
}

/// Like [`emit_vulkan_spirv`], but models every device/constant buffer param raw. Used by the R4
/// ground-truth retry when the default typed emission produces a mistyped buffer access.
pub fn emit_vulkan_spirv_all_buffers_raw(san_ll: &str, _tmp: &Path) -> Result<Vec<u8>, String> {
    native::emit_vulkan_spirv_all_buffers_raw(san_ll)
}

pub(crate) fn emit_vulkan_spirv_all_buffers_raw_with_sidecar(
    san_ll: &str,
    _tmp: &Path,
    kern: Option<&crate::meta::KernMeta>,
    entry_name: Option<&str>,
    buffer_layouts: Option<&HashMap<u32, crate::meta::AirType>>,
) -> Result<crate::emit_sidecar::EmittedSpirv, String> {
    native::emit_vulkan_spirv_all_buffers_raw_with_sidecar(san_ll, kern, entry_name, buffer_layouts)
}

/// Like [`emit_vulkan_spirv_all_buffers_raw`], but also models threadgroup (`addrspace(3)`) buffer
/// params raw. The second tier of the R4 ground-truth retry, used only after the device/constant-only
/// raw retry itself fails spirv-val.
pub fn emit_vulkan_spirv_all_buffers_raw_with_workgroup(
    san_ll: &str,
    _tmp: &Path,
) -> Result<Vec<u8>, String> {
    native::emit_vulkan_spirv_all_buffers_raw_with_workgroup(san_ll)
}

pub(crate) fn emit_vulkan_spirv_all_buffers_raw_with_workgroup_sidecar(
    san_ll: &str,
    _tmp: &Path,
    kern: Option<&crate::meta::KernMeta>,
    entry_name: Option<&str>,
    buffer_layouts: Option<&HashMap<u32, crate::meta::AirType>>,
) -> Result<crate::emit_sidecar::EmittedSpirv, String> {
    native::emit_vulkan_spirv_all_buffers_raw_with_workgroup_sidecar(
        san_ll,
        kern,
        entry_name,
        buffer_layouts,
    )
}

/// Emit the all-device/constant-buffer raw view with repair skipped for the W2 relooper's exclusive
/// consumption. The caller must rebuild and validate its CFG before adopting any bytes.
pub fn emit_vulkan_spirv_all_buffers_raw_relooper_feed(
    san_ll: &str,
    _tmp: &Path,
) -> Result<Vec<u8>, String> {
    native::emit_vulkan_spirv_all_buffers_raw_relooper_feed(san_ll)
}

pub(crate) fn emit_vulkan_spirv_all_buffers_raw_relooper_feed_with_sidecar(
    san_ll: &str,
    _tmp: &Path,
    kern: Option<&crate::meta::KernMeta>,
    entry_name: Option<&str>,
    buffer_layouts: Option<&HashMap<u32, crate::meta::AirType>>,
) -> Result<crate::emit_sidecar::EmittedSpirv, String> {
    native::emit_vulkan_spirv_all_buffers_raw_relooper_feed_with_sidecar(
        san_ll,
        kern,
        entry_name,
        buffer_layouts,
    )
}

/// Like [`emit_vulkan_spirv_all_buffers_raw`], but also enables BDA device-pointer modeling (a device
/// pointer loaded from a buffer becomes its real 64-bit PhysicalStorageBuffer64 address). The honest
/// retry tier for the `raw store for Ptr(1)` BDA frontier class — see
/// [`native::emit_vulkan_spirv_all_buffers_raw_bda`].
pub fn emit_vulkan_spirv_all_buffers_raw_bda(san_ll: &str, _tmp: &Path) -> Result<Vec<u8>, String> {
    native::emit_vulkan_spirv_all_buffers_raw_bda(san_ll)
}

pub(crate) fn emit_vulkan_spirv_all_buffers_raw_bda_with_sidecar(
    san_ll: &str,
    _tmp: &Path,
    kern: Option<&crate::meta::KernMeta>,
    entry_name: Option<&str>,
    buffer_layouts: Option<&HashMap<u32, crate::meta::AirType>>,
) -> Result<crate::emit_sidecar::EmittedSpirv, String> {
    native::emit_vulkan_spirv_all_buffers_raw_bda_with_sidecar(
        san_ll,
        kern,
        entry_name,
        buffer_layouts,
    )
}

pub(crate) fn emit_vulkan_spirv_all_buffers_raw_bda_relooper_feed_with_sidecar(
    san_ll: &str,
    _tmp: &Path,
    kern: Option<&crate::meta::KernMeta>,
    entry_name: Option<&str>,
    buffer_layouts: Option<&HashMap<u32, crate::meta::AirType>>,
) -> Result<crate::emit_sidecar::EmittedSpirv, String> {
    native::emit_vulkan_spirv_all_buffers_raw_bda_relooper_feed_with_sidecar(
        san_ll,
        kern,
        entry_name,
        buffer_layouts,
    )
}

/// Like [`emit_vulkan_spirv`], but inlines non-recursive internal helper calls and promotes the
/// resulting entry-stored non-escaping Function allocas (scalar-replacement + aggregate fold) before
/// emission. The honest retry tier for the MPS NDArray multi-destination (TopK) `missing pointer
/// storage` frontier class — a device-buffer-pointer array staged through a Function struct forwarded
/// by value into a helper. See [`native::emit_vulkan_spirv_inline_sroa`].
pub fn emit_vulkan_spirv_inline_sroa(san_ll: &str, _tmp: &Path) -> Result<Vec<u8>, String> {
    native::emit_vulkan_spirv_inline_sroa(san_ll)
}

pub(crate) fn emit_vulkan_spirv_inline_sroa_with_sidecar(
    san_ll: &str,
    _tmp: &Path,
    kern: Option<&crate::meta::KernMeta>,
    entry_name: Option<&str>,
    buffer_layouts: Option<&HashMap<u32, crate::meta::AirType>>,
) -> Result<crate::emit_sidecar::EmittedSpirv, String> {
    native::emit_vulkan_spirv_inline_sroa_with_sidecar(san_ll, kern, entry_name, buffer_layouts)
}

pub(crate) fn emit_vulkan_spirv_pointer_select_consumer_inline_with_sidecar(
    san_ll: &str,
    selected_pointer: &str,
    _tmp: &Path,
    kern: Option<&crate::meta::KernMeta>,
    entry_name: Option<&str>,
    buffer_layouts: Option<&HashMap<u32, crate::meta::AirType>>,
) -> Result<crate::emit_sidecar::EmittedSpirv, String> {
    native::emit_vulkan_spirv_pointer_select_consumer_inline_with_sidecar(
        san_ll,
        selected_pointer,
        kern,
        entry_name,
        buffer_layouts,
    )
}

/// Like [`emit_vulkan_spirv_inline_sroa`], but ALSO models device/constant buffers raw so the byte-
/// addressed float/uint traffic surviving the inline+SROA collapse becomes Logical-legal word-offset
/// access. The escalation tier for the TopK multi-destination kernels whose lowered `%10` device-
/// pointer array feeds typed loads/stores at byte offsets. See
/// [`native::emit_vulkan_spirv_inline_sroa_raw`].
pub fn emit_vulkan_spirv_inline_sroa_raw(san_ll: &str, _tmp: &Path) -> Result<Vec<u8>, String> {
    native::emit_vulkan_spirv_inline_sroa_raw(san_ll)
}

pub(crate) fn emit_vulkan_spirv_inline_sroa_raw_with_sidecar(
    san_ll: &str,
    _tmp: &Path,
    kern: Option<&crate::meta::KernMeta>,
    entry_name: Option<&str>,
    buffer_layouts: Option<&HashMap<u32, crate::meta::AirType>>,
) -> Result<crate::emit_sidecar::EmittedSpirv, String> {
    native::emit_vulkan_spirv_inline_sroa_raw_with_sidecar(san_ll, kern, entry_name, buffer_layouts)
}

/// Like [`emit_vulkan_spirv_inline_sroa_raw`], but ALSO enables the R2 cross-arm restructure — the
/// missing composition for the straddle-loop-merge + cross-binding-phi cluster (05), where the raw
/// model needs a restructured CFG before the following PSB rewrite can dissolve the cross-binding phi.
/// See [`native::emit_vulkan_spirv_inline_sroa_raw_cfg_restructure`]. Adopt-if-validates at the caller.
pub fn emit_vulkan_spirv_inline_sroa_raw_cfg_restructure(
    san_ll: &str,
    _tmp: &Path,
) -> Result<Vec<u8>, String> {
    native::emit_vulkan_spirv_inline_sroa_raw_cfg_restructure(san_ll)
}

pub(crate) fn emit_vulkan_spirv_inline_sroa_raw_cfg_restructure_with_sidecar(
    san_ll: &str,
    _tmp: &Path,
    kern: Option<&crate::meta::KernMeta>,
    entry_name: Option<&str>,
    buffer_layouts: Option<&HashMap<u32, crate::meta::AirType>>,
) -> Result<crate::emit_sidecar::EmittedSpirv, String> {
    native::emit_vulkan_spirv_inline_sroa_raw_cfg_restructure_with_sidecar(
        san_ll,
        kern,
        entry_name,
        buffer_layouts,
    )
}

/// Validate SPIR-V bytes against `vulkan1.3`. Returns `Ok(())` if valid.
///
/// A spirv-val TIMEOUT is NOT a validation verdict — it means the validator was killed (starved by
/// CPU saturation) before it could decide. Under a parallel gate run, a starved spirv-val on a large
/// module exceeds the wall-clock cap, and treating that as "invalid" would make the adopt-if-validates
/// retries DROP a module that is in fact valid (inflating the measured frontier with false failures).
/// So a timeout is retried with an escalating cap (which also lets contending workers drain); only a
/// real spirv-val FAILURE (a clean non-zero exit carrying validation errors) is returned. Determinism:
/// the verdict is a pure function of the bytes, so the retry never changes a real result — it only
/// converts a starvation timeout into the true verdict.
pub fn spirv_val(spv_path: &str) -> Result<(), String> {
    // Bound concurrent validators so memory-heavy spirv-val runs are not OOM-killed under a parallel
    // gate (see [`val_gate`]). Held across the whole escalating-cap loop so the in-flight validator
    // count never exceeds the limit. Byte-neutral / verdict-neutral: serializing spawns cannot change
    // what spirv-val decides, only whether it lives long enough to decide.
    let _permit = ValPermit::acquire();
    let args = ["--target-env", "vulkan1.3", spv_path];
    // Escalating wall-clock caps. A spirv-val TIMEOUT is not a verdict (the validator was
    // CPU-starved before deciding); only a clean non-zero exit is a real FAILURE. Critically,
    // an INVALID module fails FAST — spirv-val reports the first error and exits in milliseconds,
    // never consuming the cap — so a large *final* cap can only ever be spent confirming a module
    // that is in fact VALID. That makes the floor DETERMINISTIC: a translator-valid module is
    // never dropped as a starvation false-failure regardless of host contention / --threads. The
    // earlier short caps let contending workers drain first; the 600s tail is the safety net for
    // the heaviest valid modules under extreme oversubscription. (Measured: --threads 3 on this
    // 8-core box flaked ~4 cfg/PSB retry cases between identical runs at the old 240s tail.)
    const CAPS_SECS: [u64; 4] = [60, 120, 240, 600];
    for (i, &cap) in CAPS_SECS.iter().enumerate() {
        let last = i + 1 == CAPS_SECS.len();
        match run_with_timeout("spirv-val", &args, cap) {
            Ok(_) => return Ok(()),
            // A TIMEOUT or a NO_VERDICT (signal-kill / spawn-failure) means spirv-val never decided
            // — the validator was starved or OOM-killed under concurrency, not the module rejected.
            // Retry with a larger cap (which also lets contending workers drain). Only a clean
            // non-zero EXIT is a real INVALID verdict, and an invalid module produces that in ms, so
            // the escalating retries never loop on a genuinely-invalid input. This makes the verdict
            // a pure function of the bytes: a valid module is never dropped as a false failure just
            // because its validator was killed (which otherwise flakes the measured floor).
            Err(e) if (e.contains(TIMEOUT_MARKER) || e.contains(NO_VERDICT_MARKER)) && !last => {
                continue
            }
            Err(e) => return Err(e),
        }
    }
    unreachable!("CAPS_SECS is non-empty")
}

/// Validate in-memory SPIR-V bytes by writing them to a temp file and invoking `spirv-val`.
pub fn spirv_val_bytes(spv: &[u8], tmp: &Path) -> Result<(), String> {
    let path = tmp.join("a2v_val.spv");
    std::fs::write(&path, spv).map_err(|e| format!("spirv_val_bytes write: {e}"))?;
    let result = spirv_val(path.to_str().ok_or("spirv_val_bytes: bad tmp path")?);
    // spirv-val only needs the path for the subprocess lifetime.
    let _ = std::fs::remove_file(&path);
    result
}

/// Best-effort filename stem (drop the final extension).
pub fn strip_ext(p: &str) -> String {
    match p.rfind('.') {
        Some(i) => p[..i].to_string(),
        None => p.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitizer_captures_and_preserves_target_datalayout() {
        // The same datalayout survives both as a captured reflection string and in the sanitized
        // text handed to executable layout lowering.
        let tmp = std::env::temp_dir();
        let src = tmp.join("m2v_r4_datalayout_test.ll");
        std::fs::write(
            &src,
            "target datalayout = \"e-m:o-i64:64-i128:128-n32:64-S128\"\n\
             target triple = \"air64-apple-macosx\"\n\
             define void @k() {\n  ret void\n}\n",
        )
        .unwrap();
        let (san, datalayout) =
            air_to_sanitized_ll_with_datalayout(src.to_str().unwrap(), &tmp).unwrap();
        assert_eq!(
            datalayout.as_deref(),
            Some("e-m:o-i64:64-i128:128-n32:64-S128")
        );
        assert!(san.contains("target datalayout = \"e-m:o-i64:64-i128:128-n32:64-S128\""));
        assert!(san.contains(&format!("target triple = \"{VULKAN_TRIPLE}\"")));
        // The plain wrapper returns byte-identical sanitized text.
        assert_eq!(
            air_to_sanitized_ll(src.to_str().unwrap(), &tmp).unwrap(),
            san
        );
        std::fs::remove_file(&src).ok();
    }

    #[test]
    fn text_sanitizer_matches_file_sanitizer_rules() {
        let (san, datalayout) = sanitize_ll_text_with_datalayout(
            "; ModuleID = '/tmp/random/case.air'\n\
             target datalayout = \"e-p:64:64\"\n\
             target triple = \"air64-apple-ios\"\n\
             @llvm.global_ctors = appending global [0 x { i32, ptr, ptr }] []\n\
             @llvm.compiler.used = appending global [0 x ptr] [], section \"llvm.metadata\"\n\
             define void @k() {\n  ret void\n}\n",
        );

        assert_eq!(datalayout.as_deref(), Some("e-p:64:64"));
        assert!(san.contains(&format!("target triple = \"{VULKAN_TRIPLE}\"")));
        assert!(san.contains("target datalayout = \"e-p:64:64\""));
        assert!(!san.contains("ModuleID"));
        assert!(!san.contains("@llvm.global_ctors"));
        assert!(!san.contains("@llvm.compiler.used"));
        assert!(san.contains("define void @k()"));
    }

    #[test]
    fn sanitizer_identity_ignores_llvm_dis_scratch_path() {
        let first = sanitize_ll_text_with_datalayout(
            "; ModuleID = '/tmp/first/case.air'\nsource_filename = \"stable\"\ndefine void @k() { ret void }\n",
        )
        .0;
        let second = sanitize_ll_text_with_datalayout(
            "; ModuleID = '/tmp/second/case.air'\nsource_filename = \"stable\"\ndefine void @k() { ret void }\n",
        )
        .0;
        assert_eq!(first, second);
    }

    #[test]
    fn datalayout_value_extracts_quoted_string() {
        assert_eq!(
            datalayout_value("target datalayout = \"e-p:32:32\""),
            Some("e-p:32:32".to_string())
        );
        assert_eq!(datalayout_value("target datalayout = malformed"), None);
    }
}
