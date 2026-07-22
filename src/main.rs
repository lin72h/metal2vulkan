//! metal2vulkan CLI — host translator used by fixtures, coverage tools, and replay/oracle experiments.
//! Usage:  metal2vulkan <in.air|.ll> <out.spv> --stage vertex|fragment|passthrough|kernel
//! Prints a line containing `PASS` on success (spirv-val vulkan1.3 clean), nonzero exit + `FALLBACK`
//! on any failure.

use metal2vulkan::passes::{Stage, TransformOptions};
use metal2vulkan::reflect::ShaderReflection;
use metal2vulkan::{
    detect_stage, tools, translate_passthrough, translate_reflected_with_options,
    translate_with_options,
};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};
use std::{env, fs, process};

struct ReproContext {
    original_args: Vec<String>,
    src: String,
    out: String,
    requested_stage: String,
    effective_stage: Option<String>,
}

impl ReproContext {
    fn stage_for_repro(&self) -> &str {
        self.effective_stage
            .as_deref()
            .unwrap_or(&self.requested_stage)
    }
}

fn main() {
    let original_args: Vec<String> = env::args().collect();
    let args: Vec<String> = original_args.iter().skip(1).cloned().collect();
    // Default to auto-detecting the stage from the AIR's !air.* metadata (the stage is intrinsic to
    // the module). An explicit --stage overrides. This stops the whole class of "ran a fragment as
    // --stage vertex" mis-mappings when translating captured guest AIR.
    let mut stage = "auto".to_string();
    let mut emit_meta: Option<String> = None;
    let mut simd_cluster32 = false;
    let mut pos: Vec<String> = vec![];
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--help" | "-h" => {
                println!(
                    "usage: metal2vulkan <in.air|.ll> <out.spv> [--stage auto|vertex|fragment|passthrough|kernel] [--emit-meta out.json] [--simd-cluster32]  (default stage: auto)\n"
                );
                print!("{}", metal2vulkan::env_vars::help_text());
                process::exit(0);
            }
            "--stage" => {
                if let Some(v) = args.get(i + 1) {
                    stage = v.clone();
                }
                i += 2;
            }
            "--emit-meta" => {
                emit_meta = args.get(i + 1).cloned();
                i += 2;
            }
            // M-D2: cluster simd reductions to Metal's 32-lane simdgroup (see TransformOptions).
            "--simd-cluster32" => {
                simd_cluster32 = true;
                i += 1;
            }
            // accept (and ignore) the legacy flags so the harness/contract is a superset.
            "--entry" | "--local" => {
                i += 2;
            }
            _ => {
                pos.push(args[i].clone());
                i += 1;
            }
        }
    }
    if pos.is_empty() {
        eprintln!(
            "usage: metal2vulkan <in.air|.ll> <out.spv> [--stage auto|vertex|fragment|passthrough|kernel]  (default: auto)"
        );
        fail("no input");
    }
    let src = pos[0].clone();
    let out = pos
        .get(1)
        .cloned()
        .unwrap_or_else(|| format!("{}.vk.spv", tools::strip_ext(&src)));
    let mut repro = ReproContext {
        original_args,
        src: src.clone(),
        out: out.clone(),
        requested_stage: stage.clone(),
        effective_stage: None,
    };

    let tmp = env::temp_dir().join(format!("metal2vulkan_{}", process::id()));
    let _ = fs::create_dir_all(&tmp);

    let options = TransformOptions {
        simd_cluster32,
        ..TransformOptions::default()
    };
    let want_meta = emit_meta.is_some();
    let translated = match stage.as_str() {
        "passthrough" => translate_passthrough(&src, &tmp).map(|spv| (spv, None)),
        "auto" => match detect_stage(&src, &tmp) {
            Ok(st) => {
                let name = match st {
                    Stage::Vertex => "vertex",
                    Stage::Fragment => "fragment",
                    Stage::Kernel => "kernel",
                };
                repro.effective_stage = Some(name.to_string());
                eprintln!("metal2vulkan: auto-detected stage {name}");
                translate_maybe_reflect(&src, st, &tmp, want_meta, options)
            }
            Err(e) => {
                eprintln!("metal2vulkan: {e}");
                fail_with_repro("stage auto-detect failed", &repro, Some(&e), &tmp);
            }
        },
        "vertex" => {
            repro.effective_stage = Some("vertex".to_string());
            translate_maybe_reflect(&src, Stage::Vertex, &tmp, want_meta, options)
        }
        "fragment" => {
            repro.effective_stage = Some("fragment".to_string());
            translate_maybe_reflect(&src, Stage::Fragment, &tmp, want_meta, options)
        }
        "kernel" | "compute" => {
            repro.effective_stage = Some("kernel".to_string());
            translate_maybe_reflect(&src, Stage::Kernel, &tmp, want_meta, options)
        }
        other => {
            eprintln!(
                "metal2vulkan: unsupported --stage {other} (only vertex|fragment|passthrough|kernel|compute)"
            );
            fail_with_repro("bad stage", &repro, None, &tmp);
        }
    };
    let (spv, reflection) = match translated {
        Ok(v) => v,
        Err(e) => {
            eprintln!("metal2vulkan: {e}");
            fail_with_repro("translate failed", &repro, Some(&e), &tmp);
        }
    };
    if let Err(e) = fs::write(&out, &spv) {
        eprintln!("metal2vulkan: cannot write {out}: {e}");
        fail_with_repro("write failed", &repro, Some(&e.to_string()), &tmp);
    }

    match tools::spirv_val(&out) {
        Ok(()) => {
            println!("metal2vulkan: wrote {out}; spirv-val vulkan1.3: PASS");
        }
        Err(e) => {
            eprintln!("{e}");
            fail_with_repro("spirv-val failed", &repro, Some(&e), &tmp);
        }
    }

    if let Some(meta_path) = &emit_meta {
        match reflection.as_ref() {
            Some(r) => match write_meta(meta_path, r) {
                Ok(()) => println!("metal2vulkan: wrote reflection {meta_path}"),
                Err(e) => {
                    eprintln!("metal2vulkan: {e}");
                    fail_with_repro("emit-meta failed", &repro, Some(&e), &tmp);
                }
            },
            None => {
                eprintln!("metal2vulkan: --emit-meta is not supported for --stage passthrough");
                fail_with_repro("emit-meta unsupported for stage", &repro, None, &tmp);
            }
        }
    }

    // process::exit skips Drop; clean the work dir on the success path too.
    let _ = fs::remove_dir_all(&tmp);
}

/// Translate, also returning the reflection when `--emit-meta` was requested (else `None`, to skip the
/// small extra metadata build on the common path).
fn translate_maybe_reflect(
    src: &str,
    stage: Stage,
    tmp: &Path,
    want_meta: bool,
    options: TransformOptions,
) -> Result<(Vec<u8>, Option<ShaderReflection>), String> {
    if want_meta {
        let (spv, reflection) = translate_reflected_with_options(src, stage, tmp, options)?;
        Ok((spv, Some(reflection)))
    } else {
        Ok((translate_with_options(src, stage, tmp, options)?, None))
    }
}

/// Serialize the reflection to pretty JSON at `path`. Requires the `serde` feature (which pulls
/// serde_json); a build without it fails `--emit-meta` visibly rather than silently dropping it.
#[cfg(feature = "serde")]
fn write_meta(path: &str, reflection: &ShaderReflection) -> Result<(), String> {
    let json = serde_json::to_string_pretty(reflection)
        .map_err(|e| format!("serialize reflection: {e}"))?;
    fs::write(path, json).map_err(|e| format!("cannot write {path}: {e}"))
}

#[cfg(not(feature = "serde"))]
fn write_meta(_path: &str, _reflection: &ShaderReflection) -> Result<(), String> {
    Err("--emit-meta requires building metal2vulkan with --features serde".to_string())
}

fn fail(why: &str) -> ! {
    eprintln!("metal2vulkan: FALLBACK ({why})");
    process::exit(2);
}

fn fail_with_repro(why: &str, ctx: &ReproContext, detail: Option<&str>, tmp: &Path) -> ! {
    match write_repro(ctx, why, detail) {
        Ok(dir) => eprintln!("metal2vulkan: wrote repro {}", dir.display()),
        Err(e) => eprintln!("metal2vulkan: repro write failed: {e}"),
    }
    // process::exit does not run Drop — remove the CLI work dir explicitly.
    let _ = fs::remove_dir_all(tmp);
    fail(why)
}

fn write_repro(ctx: &ReproContext, why: &str, detail: Option<&str>) -> Result<PathBuf, String> {
    let base = metal2vulkan::env_vars::repro_dir()
        .map(PathBuf::from)
        .unwrap_or_else(|| env::temp_dir().join("metal2vulkan-repros"));
    fs::create_dir_all(&base).map_err(|e| format!("create {}: {e}", base.display()))?;

    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|e| format!("time before UNIX_EPOCH: {e}"))?
        .as_nanos();
    let dir = base.join(format!("metal2vulkan-repro-{}-{stamp}", process::id()));
    fs::create_dir(&dir).map_err(|e| format!("create {}: {e}", dir.display()))?;

    let input_name = repro_input_name(&ctx.src);
    let copied_input = dir.join(&input_name);
    fs::copy(&ctx.src, &copied_input)
        .map_err(|e| format!("copy {} -> {}: {e}", ctx.src, copied_input.display()))?;

    let repro_out = dir.join("out.spv");
    let stage = ctx.stage_for_repro();
    let repro_command = format!(
        "metal2vulkan {} {} --stage {}\n",
        shell_quote(&input_name),
        shell_quote(repro_out.file_name().unwrap().to_string_lossy().as_ref()),
        shell_quote(stage)
    );
    fs::write(dir.join("repro-command.txt"), &repro_command)
        .map_err(|e| format!("write repro-command.txt: {e}"))?;

    let repro_script = format!(
        "#!/usr/bin/env sh\nset -eu\nDIR=$(CDPATH= cd -- \"$(dirname -- \"$0\")\" && pwd)\nmetal2vulkan \"$DIR/{input_name}\" \"$DIR/out.spv\" --stage {}\n",
        shell_quote(stage)
    );
    let script_path = dir.join("repro.sh");
    fs::write(&script_path, repro_script).map_err(|e| format!("write repro.sh: {e}"))?;
    make_executable(&script_path)?;

    let mut failure = String::new();
    failure.push_str(&format!("why: {why}\n"));
    failure.push_str(&format!("source: {}\n", ctx.src));
    failure.push_str(&format!("output: {}\n", ctx.out));
    failure.push_str(&format!("requested_stage: {}\n", ctx.requested_stage));
    failure.push_str(&format!("effective_stage: {}\n", ctx.stage_for_repro()));
    failure.push_str(&format!(
        "original_command: {}\n",
        shell_join(&ctx.original_args)
    ));
    if let Some(detail) = detail {
        failure.push_str("\ndetail:\n");
        failure.push_str(detail);
        if !detail.ends_with('\n') {
            failure.push('\n');
        }
    }
    fs::write(dir.join("failure.txt"), failure).map_err(|e| format!("write failure.txt: {e}"))?;

    Ok(dir)
}

fn repro_input_name(src: &str) -> String {
    let ext = Path::new(src)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("input");
    match ext {
        "air" | "ll" => format!("input.{ext}"),
        _ => "input.bin".to_string(),
    }
}

fn shell_join(args: &[String]) -> String {
    args.iter()
        .map(|arg| shell_quote(arg))
        .collect::<Vec<_>>()
        .join(" ")
}

fn shell_quote(s: &str) -> String {
    if s.is_empty() {
        return "''".to_string();
    }
    if s.bytes()
        .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-' | b'.' | b'/' | b':' | b'='))
    {
        return s.to_string();
    }
    format!("'{}'", s.replace('\'', "'\\''"))
}

#[cfg(unix)]
fn make_executable(path: &Path) -> Result<(), String> {
    use std::os::unix::fs::PermissionsExt;
    let mut perms = fs::metadata(path)
        .map_err(|e| format!("stat {}: {e}", path.display()))?
        .permissions();
    perms.set_mode(0o755);
    fs::set_permissions(path, perms).map_err(|e| format!("chmod {}: {e}", path.display()))
}

#[cfg(not(unix))]
fn make_executable(_path: &Path) -> Result<(), String> {
    Ok(())
}
