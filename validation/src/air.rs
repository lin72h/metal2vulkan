use crate::case::Stage;

pub fn stage_label_from_ll(ll: &str) -> Option<&'static str> {
    if ll.contains("!air.vertex =") {
        Some("Vertex")
    } else if ll.contains("!air.fragment =") {
        Some("Fragment")
    } else if ll.contains("!air.kernel =") {
        Some("Kernel")
    } else {
        None
    }
}

pub fn stage_from_ll(ll: &str) -> Stage {
    match stage_label_from_ll(ll) {
        Some("Vertex") => Stage::Vertex,
        Some("Fragment") => Stage::Fragment,
        _ => Stage::Kernel,
    }
}

pub fn stage_name(stage: Stage) -> &'static str {
    match stage {
        Stage::Vertex => "vertex",
        Stage::Fragment => "fragment",
        Stage::Kernel => "kernel",
    }
}

pub fn entry_name_from_ll(ll: &str) -> Option<String> {
    stage_entry_from_ll(ll)
        .map(|(_, entry)| entry)
        .or_else(|| first_define_name(ll))
}

pub fn stage_entry_from_ll(ll: &str) -> Option<(&'static str, String)> {
    for key in ["kernel", "vertex", "fragment"] {
        let needle = format!("!air.{key} = !{{!");
        if let Some(pos) = ll.find(&needle) {
            let rest = &ll[pos + needle.len()..];
            let id: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
            if id.is_empty() {
                continue;
            }
            let node = format!("!{id} = !{{");
            if let Some(npos) = ll.find(&node) {
                let body = &ll[npos + node.len()..];
                if let Some(name) = symbol_after_ptr_at(body) {
                    let stage = match key {
                        "kernel" => "Kernel",
                        "vertex" => "Vertex",
                        "fragment" => "Fragment",
                        _ => unreachable!(),
                    };
                    return Some((stage, name));
                }
            }
        }
    }
    None
}

fn first_define_name(ll: &str) -> Option<String> {
    for line in ll.lines() {
        let line = line.trim();
        let Some(rest) = line.strip_prefix("define ") else {
            continue;
        };
        if let Some(at) = rest.find('@') {
            if let Some(name) = symbol_after_at(&rest[at..]) {
                return Some(name);
            }
        }
    }
    None
}

fn symbol_after_ptr_at(s: &str) -> Option<String> {
    let p = s.find("ptr @")?;
    symbol_after_at(&s[p + 4..])
}

fn symbol_after_at(s: &str) -> Option<String> {
    let rest = s.strip_prefix('@')?;
    if let Some(rest) = rest.strip_prefix('"') {
        let mut out = String::new();
        let mut escaped = false;
        for ch in rest.chars() {
            if escaped {
                out.push(ch);
                escaped = false;
                continue;
            }
            match ch {
                '\\' => escaped = true,
                '"' => break,
                _ => out.push(ch),
            }
        }
        return (!out.is_empty()).then_some(out);
    }
    let name: String = rest
        .chars()
        .take_while(|c| c.is_ascii_alphanumeric() || *c == '_' || *c == '.' || *c == '$')
        .collect();
    (!name.is_empty()).then_some(name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stage_detection_uses_air_metadata() {
        assert_eq!(stage_from_ll("!air.kernel = !{!0}"), Stage::Kernel);
        assert_eq!(stage_from_ll("!air.vertex = !{!0}"), Stage::Vertex);
        assert_eq!(stage_from_ll("!air.fragment = !{!0}"), Stage::Fragment);
        assert_eq!(stage_from_ll(""), Stage::Kernel);
    }

    #[test]
    fn entry_name_prefers_air_stage_metadata() {
        let ll = r#"
define void @helper() {
  ret void
}

define void @"persona::ksDepthDilate"(ptr addrspace(2) %0) {
  ret void
}

!air.kernel = !{!15}
!15 = !{ptr @"persona::ksDepthDilate", !16, !17}
!16 = !{}
!17 = !{!18}
!18 = !{i32 0, !"air.buffer", !"air.location_index", i32 0}
"#;

        assert_eq!(
            entry_name_from_ll(ll),
            Some("persona::ksDepthDilate".into())
        );
        assert_eq!(
            stage_entry_from_ll(ll),
            Some(("Kernel", "persona::ksDepthDilate".into()))
        );
    }

    #[test]
    fn stage_entry_requires_a_well_formed_stage_node() {
        let ll = "define void @k() { ret void }\n!air.kernel = !{!0}\n!0 = !{i32 1}";
        assert_eq!(stage_entry_from_ll(ll), None);
        assert_eq!(entry_name_from_ll(ll), Some("k".into()));
    }
}
