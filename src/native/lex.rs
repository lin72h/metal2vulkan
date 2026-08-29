//! Leaf lexer and literal helpers for native LLVM IR parsing.

pub(super) fn split_top_level(s: &str, sep: char) -> Vec<&str> {
    let mut depth = 0i32;
    let mut start = 0usize;
    let mut out = Vec::new();
    let mut in_string = false;
    let mut escaped = false;
    for (i, c) in s.char_indices() {
        if in_string {
            if escaped {
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else if c == '"' {
                in_string = false;
            }
            continue;
        }
        if c == '"' {
            in_string = true;
            continue;
        }
        match c {
            '<' | '[' | '{' | '(' => depth += 1,
            '>' | ']' | '}' | ')' => depth -= 1,
            _ if c == sep && depth == 0 => {
                out.push(s[start..i].trim());
                start = i + c.len_utf8();
            }
            _ => {}
        }
    }
    out.push(s[start..].trim());
    out
}

pub(super) fn split_top_level_whitespace(s: &str) -> Vec<&str> {
    let mut depth = 0i32;
    let mut start = None;
    let mut out = Vec::new();
    let mut in_string = false;
    let mut escaped = false;
    for (i, c) in s.char_indices() {
        if in_string {
            if escaped {
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else if c == '"' {
                in_string = false;
            }
            continue;
        }
        if c == '"' {
            if start.is_none() {
                start = Some(i);
            }
            in_string = true;
            continue;
        }
        match c {
            '<' | '[' | '{' | '(' => {
                if start.is_none() {
                    start = Some(i);
                }
                depth += 1;
            }
            '>' | ']' | '}' | ')' => depth -= 1,
            c if c.is_whitespace() && depth == 0 => {
                if let Some(st) = start.take() {
                    out.push(&s[st..i]);
                }
            }
            _ => {
                if start.is_none() {
                    start = Some(i);
                }
            }
        }
    }
    if let Some(st) = start {
        out.push(&s[st..]);
    }
    out
}

pub(super) fn strip_comment(s: &str) -> &str {
    let mut in_string = false;
    let mut escaped = false;
    for (i, c) in s.char_indices() {
        if in_string {
            if escaped {
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else if c == '"' {
                in_string = false;
            }
            continue;
        }
        if c == '"' {
            in_string = true;
            continue;
        }
        if c == ';' {
            return &s[..i];
        }
    }
    s
}

pub(super) fn matching_paren(s: &str, open: usize) -> Option<usize> {
    matching_delim(s, open, '(', ')')
}

pub(super) fn matching_angle(s: &str, open: usize) -> Option<usize> {
    matching_delim(s, open, '<', '>')
}

pub(super) fn matching_delim(s: &str, open: usize, left: char, right: char) -> Option<usize> {
    if s[open..].chars().next()? != left {
        return None;
    }
    let mut depth = 0i32;
    for (i, c) in s.char_indices().skip_while(|(i, _)| *i < open) {
        if c == left {
            depth += 1;
        } else if c == right {
            depth -= 1;
            if depth == 0 {
                return Some(i);
            }
        }
    }
    None
}

pub(super) fn parse_u32(s: &str) -> Result<u32, String> {
    s.parse()
        .map_err(|e| format!("native emitter: expected u32 integer `{s}`: {e}"))
}

pub(super) fn parse_float_literal(s: &str) -> Result<f64, String> {
    let looks_float = s.contains('.') || s.contains('e') || s.contains('E');
    if !looks_float {
        return Err(format!("native emitter: not a decimal float literal `{s}`"));
    }
    s.parse()
        .map_err(|e| format!("native emitter: expected float `{s}`: {e}"))
}

pub(super) fn parse_float32_hex_literal(s: &str) -> Option<u32> {
    let hex = s.strip_prefix("f0x")?;
    if hex.len() != 8 {
        return None;
    }
    u32::from_str_radix(hex, 16).ok()
}

pub(super) fn parse_half_literal(s: &str) -> Option<u16> {
    let hex = s.strip_prefix("0xH")?;
    u16::from_str_radix(hex, 16).ok()
}

pub(super) fn parse_bfloat_literal(s: &str) -> Option<u16> {
    let hex = s.strip_prefix("0xR")?;
    u16::from_str_radix(hex, 16).ok()
}

pub(super) fn parse_hex_literal(s: &str) -> Option<u64> {
    let hex = s.strip_prefix("0x")?;
    u64::from_str_radix(hex, 16).ok()
}

pub(super) fn parse_i64_literal(s: &str) -> Result<i64, String> {
    if !s.starts_with('-') {
        return Err(format!("native emitter: not a signed integer `{s}`"));
    }
    s.parse()
        .map_err(|e| format!("native emitter: expected signed integer `{s}`: {e}"))
}

pub(super) fn parse_u64_literal(s: &str) -> Result<u64, String> {
    if let Some(hex) = s.strip_prefix("0x") {
        u64::from_str_radix(hex, 16)
            .map_err(|e| format!("native emitter: expected hex integer `{s}`: {e}"))
    } else {
        s.parse()
            .map_err(|e| format!("native emitter: expected integer `{s}`: {e}"))
    }
}

#[cfg(test)]
mod tests {
    //! Leaf-lexer edge cases (refactor T8). This file had zero coverage; these tests pin the
    //! string/comment/nesting behaviour every higher-level parser re-lexes on top of, so the S7/S8
    //! refactors can't silently regress it. No external tools — part of the pure-static gate (T2).
    use super::*;

    #[test]
    fn float32_bit_literal_requires_exact_width() {
        assert_eq!(parse_float32_hex_literal("f0x358637BD"), Some(0x3586_37bd));
        assert_eq!(parse_float32_hex_literal("f0x00000000"), Some(0));
        assert_eq!(parse_float32_hex_literal("f0x1234567"), None);
        assert_eq!(parse_float32_hex_literal("f0x123456789"), None);
        assert_eq!(parse_float32_hex_literal("f0x1234567g"), None);
    }

    #[test]
    fn split_top_level_respects_nesting_and_strings() {
        assert_eq!(split_top_level("a, b, c", ','), vec!["a", "b", "c"]);
        // separators inside brackets/angles/braces/parens are not top-level
        assert_eq!(
            split_top_level("<2 x i32>, i8, [4 x i8]", ','),
            vec!["<2 x i32>", "i8", "[4 x i8]"]
        );
        assert_eq!(split_top_level("f(a, b), c", ','), vec!["f(a, b)", "c"]);
        // a separator inside a quoted string is literal
        assert_eq!(
            split_top_level(r#"%"a,b", i8"#, ','),
            vec![r#"%"a,b""#, "i8"]
        );
        // an escaped quote does not close the string
        assert_eq!(split_top_level(r#""a\",b""#, ','), vec![r#""a\",b""#]);
        // empty input yields one empty field, trimmed
        assert_eq!(split_top_level("", ','), vec![""]);
        assert_eq!(split_top_level("  x  ", ','), vec!["x"]);
    }

    #[test]
    fn split_top_level_whitespace_groups_bracketed_and_quoted() {
        assert_eq!(split_top_level_whitespace("i32 %0"), vec!["i32", "%0"]);
        // a bracketed group with inner whitespace stays one token
        assert_eq!(
            split_top_level_whitespace("<2 x i32> %v"),
            vec!["<2 x i32>", "%v"]
        );
        // a quoted token with inner whitespace stays one token
        assert_eq!(
            split_top_level_whitespace(r#"%"a b" %c"#),
            vec![r#"%"a b""#, "%c"]
        );
        // leading/trailing/duplicate whitespace collapses away
        assert_eq!(split_top_level_whitespace("   a   b   "), vec!["a", "b"]);
        assert!(split_top_level_whitespace("").is_empty());
    }

    #[test]
    fn strip_comment_ignores_semicolons_in_strings() {
        assert_eq!(strip_comment("add i32 %a ; a comment").trim(), "add i32 %a");
        // a semicolon inside a quoted string is not a comment start
        assert_eq!(strip_comment(r#"call @"a;b"()"#), r#"call @"a;b"()"#);
        // escaped quote keeps the string open across the semicolon
        assert_eq!(strip_comment(r#""x\";y""#), r#""x\";y""#);
        // no comment => unchanged
        assert_eq!(strip_comment("plain text"), "plain text");
    }

    #[test]
    fn matching_delim_finds_balanced_close_or_none() {
        assert_eq!(matching_paren("(a (b) c)", 0), Some(8));
        assert_eq!(matching_angle("<2 x <4 x i8>>", 0), Some(13));
        // unbalanced returns None rather than panicking
        assert_eq!(matching_paren("(a (b)", 0), None);
        // wrong opening char at `open` returns None
        assert_eq!(matching_paren("[a]", 0), None);
        // open past a leading prefix
        assert_eq!(matching_paren("ptr (x)", 4), Some(6));
    }

    #[test]
    fn literal_parsers_cover_prefixes_and_errors() {
        assert_eq!(parse_half_literal("0xH3C00"), Some(0x3C00));
        assert_eq!(parse_half_literal("3.0"), None); // no 0xH prefix
        assert_eq!(parse_bfloat_literal("0xR3F80"), Some(0x3F80));
        assert_eq!(parse_bfloat_literal("0xH3C00"), None); // wrong prefix
        assert_eq!(parse_hex_literal("0xFF"), Some(255));
        assert_eq!(parse_hex_literal("255"), None); // needs 0x
        assert_eq!(parse_u32("42").unwrap(), 42);
        assert!(parse_u32("-1").is_err());
        assert!(parse_u32("4.0").is_err());
        // i64 helper only accepts a leading '-'
        assert_eq!(parse_i64_literal("-7").unwrap(), -7);
        assert!(parse_i64_literal("7").is_err());
        // u64 helper takes hex or decimal
        assert_eq!(parse_u64_literal("0x10").unwrap(), 16);
        assert_eq!(parse_u64_literal("16").unwrap(), 16);
        assert!(parse_u64_literal("nope").is_err());
        // float helper requires a float-looking token
        assert_eq!(parse_float_literal("1.5").unwrap(), 1.5);
        assert_eq!(parse_float_literal("1e3").unwrap(), 1000.0);
        assert!(parse_float_literal("10").is_err()); // no '.'/'e' => not a float literal
    }
}

/// Parse LLVM IR special float literals: `+qnan`, `-qnan`, `qnan`, `snan`, `-snan`,
/// `+inf`, `-inf`, `inf`, `nan`. Returns the IEEE 754 f32 bit pattern.
pub(super) fn parse_float_special_literal(s: &str) -> Option<u32> {
    let s = s.trim();
    // Strip leading + if present
    let s = s.strip_prefix('+').unwrap_or(s);
    match s {
        "qnan" => Some(0x7FC0_0000),       // quiet NaN (positive)
        "-qnan" => Some(0xFFC0_0000),       // quiet NaN (negative)
        "snan" => Some(0x7F80_0001),       // signaling NaN
        "-snan" => Some(0xFF80_0001),      // signaling NaN (negative)
        "inf" => Some(0x7F80_0000),         // positive infinity
        "-inf" => Some(0xFF80_0000),        // negative infinity
        "nan" => Some(0x7FC0_0000),         // generic NaN
        "-nan" => Some(0xFFC0_0000),       // generic NaN (negative)
        _ => None,
    }
}
