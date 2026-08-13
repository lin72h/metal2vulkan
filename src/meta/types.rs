use std::collections::HashMap;

/// A reconstructed AIR aggregate type built from a buffer's `air.struct_type_info` metadata. It
/// restores the source layout when an emitted buffer parameter is represented as a bare element
/// pointer (homogeneous/nested aggregates → `T*` with physical/multi-level access chains). The
/// access-chain indices are valid struct-navigation indices for this reconstructed type.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum AirType {
    /// 32-bit scalar leaf.
    Scalar(AirScalar),
    /// `floatN` / `uintN` / `intN` vector.
    Vec { scalar: AirScalar, lanes: u32 },
    /// `packed_floatN` / `packed_uintN` / `packed_intN`, laid out as scalar array stride 4.
    PackedVec { scalar: AirScalar, lanes: u32 },
    /// Fixed-size metadata array, e.g. `uint[4]` represented as `i32 array_len = 4`.
    Array { elem: Box<AirType>, len: u32 },
    /// `floatCxR`/`halfCxR` matrix → `{ [cols x vec(rows)] }` (matches Metal's
    /// `metal::matrix` wrapper struct).
    Matrix {
        scalar: AirScalar,
        cols: u32,
        rows: u32,
    },
    /// Nested struct, members in declaration order with explicit AIR byte offsets.
    Struct(Vec<AirMember>),
}

#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct AirMember {
    pub offset: u32,
    pub ty: AirType,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum AirScalar {
    Float,
    Half,
    UInt,
    SInt,
    ULong,
    SLong,
    UShort,
    SShort,
    UChar,
    Bool,
}

/// Map a primitive AIR type-name (`float`, `float2`, `packed_float3`, `float3x4`, `half4`, ...)
/// to an `AirType`. Unknown user struct/class names return `None`.
pub fn primitive_air_type_from_name(name: &str) -> Option<AirType> {
    let raw = name.trim();
    let (n, packed) = match raw.strip_prefix("packed_") {
        Some(n) => (n, true),
        None => (raw, false),
    };
    for (base, scalar) in [
        ("ulong", AirScalar::ULong),
        ("long", AirScalar::SLong),
        ("ushort", AirScalar::UShort),
        ("short", AirScalar::SShort),
        ("uchar", AirScalar::UChar),
        // `char` (signed 8-bit) shares i8 storage with `uchar`; SPIR-V integer types are
        // signedness-agnostic, so the storage type is identical. Without this entry the
        // struct-member fallback in `air_type_from_name` silently mistypes `char` as `float`,
        // widening an i8 member to 32 bits (seen in MTLGPUBVHBuilderQueueEntry).
        ("char", AirScalar::UChar),
        ("float", AirScalar::Float),
        ("half", AirScalar::Half),
        ("uint", AirScalar::UInt),
        ("int", AirScalar::SInt),
        ("bool", AirScalar::Bool),
    ] {
        if let Some(rest) = n.strip_prefix(base) {
            return parse_dims(rest, scalar, packed);
        }
    }
    None
}

/// Map a primitive AIR type-name inside `air.struct_type_info`; historical callers default unknown
/// names to float leaves because metadata occasionally names opaque helper structs without nested
/// layout. New external users should prefer `primitive_air_type_from_name`.
fn air_type_from_name(name: &str) -> AirType {
    primitive_air_type_from_name(name).unwrap_or(AirType::Scalar(AirScalar::Float))
}

fn parse_dims(rest: &str, scalar: AirScalar, packed: bool) -> Option<AirType> {
    if rest.is_empty() {
        return Some(AirType::Scalar(scalar));
    }
    if let Some((c, r)) = rest.split_once('x') {
        let cols = c.trim().parse().ok()?;
        let rows = r.trim().parse().ok()?;
        return Some(AirType::Matrix { scalar, cols, rows });
    }
    match rest.trim().parse::<u32>() {
        Ok(n) if n >= 1 && packed => Some(AirType::PackedVec { scalar, lanes: n }),
        Ok(n) if n >= 2 => Some(AirType::Vec { scalar, lanes: n }),
        _ => None,
    }
}

/// One token of a metadata node body, for parsing `air.struct_type_info`.
pub(super) enum Tok {
    Int(u32),
    Str(String),
    Ref(u32),
}

/// Tokenize a metadata node body into Int(`i32 N`) / Str(`!"..."`) / Ref(`!N`) tokens, dropping
/// anything else (`ptr @f`, addrspace globals, …).
pub(super) fn tokenize(body: &str) -> Vec<Tok> {
    let mut out = vec![];
    for field in body.split(',') {
        let f = field.trim();
        if let Some(rest) = f.strip_prefix("i32 ") {
            if let Ok(v) = rest.trim().parse::<u32>() {
                out.push(Tok::Int(v));
            }
        } else if let Some(rest) = f.strip_prefix("!\"") {
            out.push(Tok::Str(rest.strip_suffix('"').unwrap_or(rest).to_string()));
        } else if let Some(rest) = f.strip_prefix('!') {
            if let Ok(v) = rest.trim().parse::<u32>() {
                out.push(Tok::Ref(v));
            }
        }
    }
    out
}

/// Parse an `air.struct_type_info` node into an `AirType::Struct`. Each member is a 5-tuple
/// `i32 offset, i32 size, i32 array_len, !"type", !"name"`, optionally PREFIXED by
/// `!"air.struct_type_info", !N` when the member is itself a nested struct (then recurse into `!N`).
pub(super) fn parse_struct_info(
    nodes: &HashMap<u32, String>,
    id: u32,
    depth: u32,
) -> Option<AirType> {
    if depth > 16 {
        return None;
    }
    let body = nodes.get(&id)?;
    let toks = tokenize(body);
    let mut members = vec![];
    let mut i = 0;
    while i < toks.len() {
        // Optional nested-struct prefix.
        let mut nested = None;
        if let (Some(Tok::Str(s)), Some(Tok::Ref(x))) = (toks.get(i), toks.get(i + 1)) {
            if s == "air.struct_type_info" {
                nested = Some(*x);
                i += 2;
            }
        }
        // Member tuple: offset, size, array_len, type-name (name string follows, ignored).
        let (offset, size, array_len, tyname) = match (
            toks.get(i),
            toks.get(i + 1),
            toks.get(i + 2),
            toks.get(i + 3),
        ) {
            (
                Some(Tok::Int(offset)),
                Some(Tok::Int(size)),
                Some(Tok::Int(array_len)),
                Some(Tok::Str(t)),
            ) => (*offset, *size, *array_len, t.clone()),
            _ => break,
        };
        i += 5; // 3 ints + type + name
        let mut mt = match nested {
            Some(x) => {
                let nested_ty = parse_struct_info(nodes, x, depth + 1)?;
                if nested_offsets_are_strict(&nested_ty) {
                    nested_ty
                } else {
                    storage_air_type_for_size(size)
                }
            }
            None => air_type_from_name(&tyname),
        };
        if array_len > 0 {
            mt = AirType::Array {
                elem: Box::new(mt),
                len: array_len,
            };
        }
        members.push(AirMember { offset, ty: mt });
        while i < toks.len() && !struct_member_starts_at(&toks, i) {
            i += 1;
        }
    }
    if members.is_empty() {
        None
    } else {
        Some(AirType::Struct(members))
    }
}

pub(super) fn struct_member_starts_at(toks: &[Tok], mut i: usize) -> bool {
    if let (Some(Tok::Str(s)), Some(Tok::Ref(_))) = (toks.get(i), toks.get(i + 1)) {
        if s == "air.struct_type_info" {
            i += 2;
        }
    }
    matches!(
        (
            toks.get(i),
            toks.get(i + 1),
            toks.get(i + 2),
            toks.get(i + 3),
            toks.get(i + 4),
        ),
        (
            Some(Tok::Int(_)),
            Some(Tok::Int(_)),
            Some(Tok::Int(_)),
            Some(Tok::Str(_)),
            Some(Tok::Str(_)),
        )
    )
}

fn nested_offsets_are_strict(ty: &AirType) -> bool {
    match ty {
        AirType::Struct(members) => members
            .windows(2)
            .all(|pair| pair[1].offset > pair[0].offset),
        _ => true,
    }
}

fn storage_air_type_for_size(size: u32) -> AirType {
    match size {
        0 | 4 => AirType::Scalar(AirScalar::UInt),
        1 => AirType::Scalar(AirScalar::UChar),
        2 => AirType::Scalar(AirScalar::UShort),
        8 => AirType::Scalar(AirScalar::ULong),
        n if n % 4 == 0 => AirType::Array {
            elem: Box::new(AirType::Scalar(AirScalar::UInt)),
            len: n / 4,
        },
        n => AirType::Array {
            elem: Box::new(AirType::Scalar(AirScalar::UChar)),
            len: n,
        },
    }
}

/// The `air.struct_type_info` ref (`!N`) named directly in a buffer-arg node body, if any.
pub(super) fn struct_info_ref(body: &str) -> Option<u32> {
    let p = body.find("air.struct_type_info")?;
    let after = &body[p..];
    let bang = after.find(", !")? + 3;
    let digits: String = after[bang..]
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect();
    digits.parse().ok()
}
