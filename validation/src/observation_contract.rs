use crate::case::TextureFormat;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ObservationScalar {
    Float,
    Half,
    Int,
    Uint,
    Short,
    Ushort,
    Bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ObservationType {
    pub scalar: ObservationScalar,
    pub lanes: usize,
}

impl ObservationType {
    pub fn parse(type_name: &str) -> Option<Self> {
        let split = type_name
            .bytes()
            .position(|byte| byte.is_ascii_digit())
            .unwrap_or(type_name.len());
        let (base, lanes) = type_name.split_at(split);
        let lanes = if lanes.is_empty() {
            1
        } else {
            lanes.parse().ok()?
        };
        if !(1..=4).contains(&lanes) {
            return None;
        }
        let scalar = match base {
            "float" => ObservationScalar::Float,
            "half" => ObservationScalar::Half,
            "int" => ObservationScalar::Int,
            "uint" => ObservationScalar::Uint,
            "short" => ObservationScalar::Short,
            "ushort" => ObservationScalar::Ushort,
            "bool" => ObservationScalar::Bool,
            _ => return None,
        };
        Some(Self { scalar, lanes })
    }

    pub fn attachment_format(self) -> TextureFormat {
        match self.scalar {
            ObservationScalar::Float | ObservationScalar::Half => TextureFormat::Rgba32Float,
            ObservationScalar::Int | ObservationScalar::Short => TextureFormat::Rgba32Sint,
            ObservationScalar::Uint | ObservationScalar::Ushort | ObservationScalar::Bool => {
                TextureFormat::Rgba32Uint
            }
        }
    }

    pub fn metal_output_base(self) -> &'static str {
        match self.scalar {
            ObservationScalar::Float | ObservationScalar::Half => "float",
            ObservationScalar::Int | ObservationScalar::Short => "int",
            ObservationScalar::Uint | ObservationScalar::Ushort | ObservationScalar::Bool => "uint",
        }
    }

    pub fn requires_flat_interpolation(self) -> bool {
        !matches!(
            self.scalar,
            ObservationScalar::Float | ObservationScalar::Half
        )
    }
}

pub fn metal_user_attribute(semantic: Option<&str>) -> Option<&str> {
    semantic.filter(|semantic| semantic.starts_with("user(") && semantic.ends_with(')'))
}

pub fn metal_field_name(
    location: u32,
    name: Option<&str>,
    semantic: Option<&str>,
) -> Result<String, String> {
    if let Some(name) = name.filter(|name| is_msl_identifier(name)) {
        return Ok(name.to_string());
    }
    if metal_user_attribute(semantic).is_some() {
        return Ok(format!("metal2vulkan_varying_{location}"));
    }
    match name {
        Some(name) => Err(format!(
            "varying {location} has invalid Metal identifier {name:?} and no explicit user semantic"
        )),
        None => Err(format!(
            "varying {location} has neither a Metal-linkable field name nor an explicit user semantic"
        )),
    }
}

fn is_msl_identifier(name: &str) -> bool {
    let mut bytes = name.bytes();
    bytes
        .next()
        .is_some_and(|byte| byte == b'_' || byte.is_ascii_alphabetic())
        && bytes.all(|byte| byte == b'_' || byte.is_ascii_alphanumeric())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn observation_types_have_one_shared_attachment_contract() {
        assert_eq!(
            ObservationType::parse("half3").unwrap().attachment_format(),
            TextureFormat::Rgba32Float
        );
        assert_eq!(
            ObservationType::parse("bool4").unwrap().attachment_format(),
            TextureFormat::Rgba32Uint
        );
        assert!(ObservationType::parse("double2").is_none());
        assert!(ObservationType::parse("float8").is_none());
    }

    #[test]
    fn explicit_semantic_removes_the_source_identifier_dependency() {
        assert_eq!(
            metal_field_name(7, Some("not.valid"), Some("user(payload)")),
            Ok("metal2vulkan_varying_7".into())
        );
        assert!(metal_field_name(7, None, Some("generated(payload)")).is_err());
    }
}
