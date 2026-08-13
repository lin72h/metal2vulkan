/// A Metal `[[function_constant(N)]]` discovered from its AIR `air.fc_initializer` initializer
/// global. The default value is NOT recoverable from the IR — function constants are externally
/// specialized (`MTLFunctionConstantValues` / `specialize_function_constants`) and the initializer
/// global is `externally_initialized ... undef` — so only the index, symbol name, and LLVM type are
/// carried.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FunctionConstant {
    /// The `[[function_constant(N)]]` index / SPIR-V spec-id.
    pub index: u32,
    /// The demangled-free base symbol (the mangled name before the `.MTL_FC_INIT_` marker).
    pub name: String,
    /// The LLVM-IR type of the constant (`i32`, `i1`, `<4 x i32>`, `float`, …).
    pub type_name: String,
    /// Itanium ABI type encoding carried after `MTL_FC_INIT_<index>_` (`j`, `Dv4_j`, `Dh`, …).
    /// Unlike LLVM's signless integer type, this preserves the Metal scalar signedness and lanes
    /// needed to bind an exact `MTLFunctionConstantValues` value.
    pub abi_type_encoding: String,
}

/// Scan LLVM-IR for `[[function_constant]]` initializer globals — module-scope declarations named
/// `@<base>.MTL_FC_INIT_<N>_<suffix>` (Apple's stable FC ABI marker, `section "air.fc_initializer"`).
/// Keys ONLY on that documented marker, never on a shader-specific name. Returns one entry per
/// distinct index, sorted, so a consumer can discover the module's spec-ids without scanning SPIR-V.
pub fn parse_function_constants(ll: &str) -> Vec<FunctionConstant> {
    let mut out: Vec<FunctionConstant> = Vec::new();
    for line in ll.lines() {
        let t = line.trim_start();
        if !t.starts_with('@') || !t.contains(".MTL_FC_INIT_") {
            continue;
        }
        let Some(eq) = t.find(" = ") else {
            continue;
        };
        let Some((base, marker)) = t[1..eq].trim().split_once(".MTL_FC_INIT_") else {
            continue;
        };
        let digits: String = marker.chars().take_while(|c| c.is_ascii_digit()).collect();
        let Ok(index) = digits.parse::<u32>() else {
            continue;
        };
        let abi_type_encoding = marker
            .strip_prefix(&digits)
            .and_then(|suffix| suffix.strip_prefix('_'))
            .unwrap_or_default()
            .to_string();
        if out.iter().any(|f| f.index == index) {
            continue;
        }
        out.push(FunctionConstant {
            index,
            name: base.to_string(),
            type_name: fc_global_decl_type(&t[eq + 3..]).unwrap_or_default(),
            abi_type_encoding,
        });
    }
    out.sort_by_key(|f| f.index);
    out
}

/// The declared LLVM type of a `constant`/`global` definition body (the token after the
/// `constant`/`global` keyword): a balanced `<...>` vector/array or the first scalar token.
fn fc_global_decl_type(decl: &str) -> Option<String> {
    let after = decl
        .split(" constant ")
        .nth(1)
        .or_else(|| decl.split(" global ").nth(1))?;
    let s = after.trim_start();
    if let Some(rest) = s.strip_prefix('<') {
        let end = rest.find('>')?;
        Some(format!("<{}>", &rest[..end]))
    } else {
        s.split_whitespace().next().map(str::to_string)
    }
}
