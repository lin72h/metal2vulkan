use crate::{meta, tools};
use std::collections::BTreeSet;
use std::path::Path;

#[derive(Clone, Debug)]
struct PassTy {
    spirv: String,
    base: &'static str,
    lanes: u32,
}

fn passthrough_type(air: &str) -> Result<PassTy, String> {
    let air = air.trim();
    let (base, rest) = if let Some(rest) = air.strip_prefix("float") {
        ("float", rest)
    } else if let Some(rest) = air.strip_prefix("half") {
        ("half", rest)
    } else {
        return Err(format!("passthrough: unsupported varying type {air}"));
    };
    let lanes = if rest.is_empty() {
        1
    } else {
        rest.parse::<u32>()
            .map_err(|_| format!("passthrough: unsupported varying type {air}"))?
    };
    if !(1..=4).contains(&lanes) {
        return Err(format!("passthrough: unsupported varying type {air}"));
    }
    let spirv = if lanes == 1 {
        base.to_string()
    } else {
        format!("v{lanes}{base}")
    };
    Ok(PassTy { spirv, base, lanes })
}

fn emit_float_value(
    p: &mut Vec<String>,
    prefix: &str,
    idx: usize,
    lanes: u32,
) -> Result<String, String> {
    match lanes {
        1 => Ok("%ux".to_string()),
        2 => {
            let id = format!("%{prefix}{idx}");
            p.push(format!("{id} = OpCompositeConstruct %v2float %ux %uy"));
            Ok(id)
        }
        3 => {
            let id = format!("%{prefix}{idx}");
            p.push(format!(
                "{id} = OpCompositeConstruct %v3float %ux %uy %float_0"
            ));
            Ok(id)
        }
        4 => {
            let id = format!("%{prefix}{idx}");
            p.push(format!(
                "{id} = OpCompositeConstruct %v4float %ux %uy %float_0 %float_1"
            ));
            Ok(id)
        }
        _ => Err(format!("passthrough: unsupported lane count {lanes}")),
    }
}

fn passthrough_vertex_spvasm(meta: &meta::FragMeta) -> Result<String, String> {
    let mut locs: Vec<u32> = meta
        .roles
        .iter()
        .filter_map(|(_, role)| match role {
            meta::FragRole::Varying(loc) => Some(*loc),
            _ => None,
        })
        .collect();
    locs.sort_unstable();
    locs.dedup();

    let mut varyings = Vec::with_capacity(locs.len());
    for loc in locs {
        let ty_name = meta
            .varying_type(loc)
            .ok_or_else(|| format!("passthrough: missing type for varying location {loc}"))?;
        varyings.push((loc, passthrough_type(ty_name)?));
    }

    let mut out_types: BTreeSet<String> = BTreeSet::new();
    out_types.insert("v4float".to_string());
    for (_, ty) in &varyings {
        out_types.insert(ty.spirv.clone());
    }
    let has_half = varyings.iter().any(|(_, ty)| ty.base == "half");

    let mut p: Vec<String> = vec![];
    if has_half {
        p.push("OpCapability Float16".into());
    }
    p.push("OpCapability Shader".into());
    p.push("OpMemoryModel Logical GLSL450".into());

    let mut iface = vec!["%glpos".to_string(), "%vidx".to_string()];
    for i in 0..varyings.len() {
        iface.push(format!("%vout{i}"));
    }
    p.push(format!(
        "OpEntryPoint Vertex %main \"main\" {}",
        iface.join(" ")
    ));
    p.push("OpDecorate %glpos BuiltIn Position".into());
    p.push("OpDecorate %vidx BuiltIn VertexIndex".into());
    for (i, (loc, _)) in varyings.iter().enumerate() {
        p.push(format!("OpDecorate %vout{i} Location {loc}"));
    }

    p.push("%void = OpTypeVoid".into());
    p.push("%fnty = OpTypeFunction %void".into());
    p.push("%float = OpTypeFloat 32".into());
    if has_half {
        p.push("%half = OpTypeFloat 16".into());
    }
    p.push("%int = OpTypeInt 32 1".into());
    p.push("%v2float = OpTypeVector %float 2".into());
    p.push("%v3float = OpTypeVector %float 3".into());
    p.push("%v4float = OpTypeVector %float 4".into());
    if has_half {
        p.push("%v2half = OpTypeVector %half 2".into());
        p.push("%v3half = OpTypeVector %half 3".into());
        p.push("%v4half = OpTypeVector %half 4".into());
    }
    p.push("%_ptr_Input_int = OpTypePointer Input %int".into());
    for ty in &out_types {
        p.push(format!("%_ptr_Output_{ty} = OpTypePointer Output %{ty}"));
    }
    p.push("%float_0 = OpConstant %float 0".into());
    p.push("%float_1 = OpConstant %float 1".into());
    p.push("%float_2 = OpConstant %float 2".into());
    p.push("%float_4 = OpConstant %float 4".into());
    p.push("%int_1 = OpConstant %int 1".into());
    p.push("%glpos = OpVariable %_ptr_Output_v4float Output".into());
    p.push("%vidx = OpVariable %_ptr_Input_int Input".into());
    for (i, (_, ty)) in varyings.iter().enumerate() {
        p.push(format!(
            "%vout{i} = OpVariable %_ptr_Output_{} Output",
            ty.spirv
        ));
    }

    p.push("%main = OpFunction %void None %fnty".into());
    p.push("%entry = OpLabel".into());
    p.push("%idx = OpLoad %int %vidx".into());
    p.push("%ax_i = OpBitwiseAnd %int %idx %int_1".into());
    p.push("%ay_i = OpShiftRightArithmetic %int %idx %int_1".into());
    p.push("%axf = OpConvertSToF %float %ax_i".into());
    p.push("%ayf = OpConvertSToF %float %ay_i".into());
    p.push("%px0 = OpFMul %float %axf %float_4".into());
    p.push("%px = OpFSub %float %px0 %float_1".into());
    p.push("%py0 = OpFMul %float %ayf %float_4".into());
    // Metal's viewport maps top-row pixels to positive clip-space Y, while this Vulkan harness
    // uses the default positive-height viewport. Flip only the generated passthrough geometry so
    // screen-space interpolation of synthesized varyings matches the Metal oracle.
    p.push("%py = OpFSub %float %float_1 %py0".into());
    p.push("%pos = OpCompositeConstruct %v4float %px %py %float_0 %float_1".into());
    p.push("OpStore %glpos %pos".into());
    p.push("%ux = OpFMul %float %axf %float_2".into());
    p.push("%uy = OpFMul %float %ayf %float_2".into());

    for (i, (_, ty)) in varyings.iter().enumerate() {
        let val = if ty.base == "float" {
            emit_float_value(&mut p, "uvf", i, ty.lanes)?
        } else {
            let fval = emit_float_value(&mut p, "uvf", i, ty.lanes)?;
            let hval = format!("%uvh{i}");
            p.push(format!("{hval} = OpFConvert %{} {fval}", ty.spirv));
            hval
        };
        p.push(format!("OpStore %vout{i} {val}"));
    }
    p.push("OpReturn".into());
    p.push("OpFunctionEnd".into());
    Ok(p.join("\n") + "\n")
}

fn assemble_spvasm(asm: &str, tmp: &Path, stem: &str) -> Result<Vec<u8>, String> {
    let asmf = tmp.join(format!("{stem}.spvasm"));
    let spvf = tmp.join(format!("{stem}.spv"));
    std::fs::write(&asmf, asm).map_err(|e| format!("write {}: {e}", asmf.display()))?;
    // Unified subprocess handling (S3): `tools::run` bounds the tool with a timeout and resolves it
    // via the shared tool-bin path, and its failure string is `"spirv-as failed:\n<stderr>"` —
    // byte-identical to the raw runner this replaced.
    let asmf_s = asmf.to_str().ok_or("passthrough: bad asm path")?;
    let spvf_s = spvf.to_str().ok_or("passthrough: bad spv path")?;
    let assembled = tools::run(
        "spirv-as",
        &["--target-env", "vulkan1.3", asmf_s, "-o", spvf_s],
    );
    let bytes = match assembled {
        Ok(_) => std::fs::read(&spvf).map_err(|e| format!("read {}: {e}", spvf.display())),
        Err(e) => Err(e),
    };
    // Intermediates only needed for spirv-as I/O.
    let _ = std::fs::remove_file(&asmf);
    let _ = std::fs::remove_file(&spvf);
    bytes
}

fn passthrough_sanitized_ll(src: &str, tmp: &Path) -> Result<String, String> {
    let ll_text = if src.ends_with(".ll") {
        std::fs::read_to_string(src).map_err(|e| format!("read {src}: {e}"))?
    } else {
        let ll = tmp.join("passthrough.ll");
        // Unified subprocess handling (S3) — see the `spirv-as` call above; failure string
        // `"llvm-dis failed:\n<stderr>"` is byte-identical to the raw runner replaced here.
        let ll_s = ll.to_str().ok_or("passthrough: bad ll path")?;
        let text = (|| {
            tools::run("llvm-dis", &[src, "-o", ll_s])?;
            std::fs::read_to_string(&ll).map_err(|e| format!("read {}: {e}", ll.display()))
        })();
        let _ = std::fs::remove_file(&ll);
        text?
    };

    let mut out = String::with_capacity(ll_text.len());
    for line in ll_text.lines() {
        let t = line.trim_start();
        if t.starts_with("target triple") {
            out.push_str(&format!("target triple = \"{}\"\n", tools::VULKAN_TRIPLE));
            continue;
        }
        if t.starts_with("target datalayout")
            || t.starts_with("@llvm.global_ctors")
            || t.starts_with("@llvm.global_dtors")
            || t.starts_with("@llvm.used")
            || t.starts_with("@llvm.compiler.used")
        {
            continue;
        }
        out.push_str(line);
        out.push('\n');
    }
    Ok(out)
}

/// Generate a fullscreen-triangle vertex shader whose output interface matches a fragment shader's
/// `[[stage_in]]` inputs. Used when APV pipelines bind a built-in vertex slot rather than AIR vertex code.
pub fn translate_passthrough(src: &str, tmp: &Path) -> Result<Vec<u8>, String> {
    let san_ll = passthrough_sanitized_ll(src, tmp)?;
    let frag = meta::parse_air_fragment_meta(&san_ll)
        .ok_or_else(|| "passthrough: source has no !air.fragment metadata".to_string())?;
    let asm = passthrough_vertex_spvasm(&frag)?;
    assemble_spvasm(&asm, tmp, "passthrough")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn passthrough_vertex_flips_clip_y_for_vulkan_viewport() {
        let mut frag = meta::FragMeta::default();
        frag.roles.push((1, meta::FragRole::Varying(0)));
        frag.varying_types.insert(0, "float2".to_string());

        let asm = passthrough_vertex_spvasm(&frag).unwrap();
        assert!(asm.contains("%px = OpFSub %float %px0 %float_1"));
        assert!(asm.contains("%py = OpFSub %float %float_1 %py0"));
    }
}
