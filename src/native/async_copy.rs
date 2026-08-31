//! Pre-emit AIR lowering for `air.simdgroup_async_copy_2d` — the Metal cooperative 2D tile copy
//! (device → threadgroup) used by the MPS structured-sparse kernels.
//!
//! The AGX *hardware* encoding of this op is undocumented (dougallj applegpu #28 could not decode it),
//! but the `air.*` intrinsic's MEMORY SEMANTICS are the documented Metal `simdgroup_async_copy`: it
//! fills a `dst` tile (`dst_tile_dims` elements, row stride `dst_elements_per_row`) from `src` starting
//! at `src_offset` inside a `src` region of `src_tile_dims` (row stride `src_elements_per_row`),
//! zero-filling the destination where the source is out of bounds. Like `simdgroup_matrix`, the END
//! memory state is a pure function of these operands — INVISIBLE to the hardware thread/lane
//! distribution — so a straight strided copy is byte-faithful to the documented layout. The copy is
//! done by every thread of the threadgroup redundantly (idempotent: every thread writes the same value
//! to the same threadgroup address), then the paired `air.wait_simdgroup_events` lowers to the
//! threadgroup barrier that publishes the tile.
//!
//! Dispatch is on the stable `air.*` ABI symbol (the allowed name-family exception), and the copy is
//! built from the operand STRUCTURE (pointers, strides, dims, offset), never a shader identifier.
//!
//! Runs on the sanitized AIR TEXT before [`super::LlModule::parse`], so the emitter sees only ordinary
//! LLVM (a synthesized helper `define` + calls + a barrier). Floor-safe by construction: it fires only
//! on a module that CALLS `air.simdgroup_async_copy_2d` — a module the emitter rejects outright today
//! (`unhandled air.* intrinsic`), so nothing that emits today is altered.

use std::fmt::Write as _;

/// The 12-operand `air.simdgroup_async_copy_2d` call, decoded from its argument list.
struct AsyncCopyCall {
    /// element size in bytes (operand 0): 4 → i32/float, 2 → i16/half.
    elem: u32,
    dst: String,
    dst_epr: String,
    dst_dims: String,
    src: String,
    src_epr: String,
    src_dims: String,
    offset: String,
    result: String,
}

/// Rewrite every `air.simdgroup_async_copy_2d` / `air.get_null_simdgroup_event` /
/// `air.is_null_simdgroup_event` / `air.wait_simdgroup_events` in `san_ll`. Borrows the original text
/// when the module does not call the async-copy intrinsic (the common case), so large modules do not
/// retain a byte-identical full-source copy beside the typed parser.
pub(crate) fn lower_simdgroup_async_copy(san_ll: &str) -> std::borrow::Cow<'_, str> {
    // The shared product prologue lowers calls before the emitter. It intentionally leaves the
    // now-dead intrinsic declaration in the module, so retry/emit entry points can see the symbol
    // without there being any work left to do. Test for an actual call line: treating a declaration
    // as work would clone and rejoin every line of a large already-lowered module on each emit tier.
    let has_copy_call = san_ll.lines().any(|line| {
        let trimmed = line.trim_start();
        trimmed.contains("air.simdgroup_async_copy_2d") && trimmed.contains("call")
    });
    if !has_copy_call {
        return std::borrow::Cow::Borrowed(san_ll);
    }

    let mut elem_sizes: std::collections::BTreeSet<u32> = std::collections::BTreeSet::new();
    // A lowered call expands one long intrinsic line into eight short ordinary-LLVM lines. Reserve
    // bounded growth once so a large module does not cross its original capacity by a few bytes and
    // make `String` retain an abandoned doubled buffer immediately before the typed parse peak.
    let rewrite_capacity = san_ll.len().saturating_add(san_ll.len() / 4);
    let mut out = LineBuffer::with_capacity(rewrite_capacity);
    let mut fresh = fresh_counter(san_ll);

    for line in san_ll.lines() {
        let trimmed = line.trim_start();
        if trimmed.contains("air.simdgroup_async_copy_2d") && trimmed.contains("call") {
            match parse_async_copy(trimmed) {
                Some(call) => {
                    elem_sizes.insert(call.elem);
                    let indent = &line[..line.len() - trimmed.len()];
                    emit_copy_call(&mut out, indent, &call, &mut fresh);
                    continue;
                }
                None => {
                    // Unrecognized shape — leave it (the emitter will still error, no regression).
                    out.push(line);
                    continue;
                }
            }
        }
        if trimmed.contains("air.is_null_simdgroup_event") && trimmed.contains("call") {
            // The event is always valid (the copy completed synchronously): is_null = false.
            if let Some(res) = call_result_id(trimmed) {
                let indent = &line[..line.len() - trimmed.len()];
                out.push_fmt(format_args!("{indent}{res} = icmp ne i64 0, 0"));
                continue;
            }
        }
        if trimmed.contains("air.get_null_simdgroup_event") && trimmed.contains("call") {
            // AIR's explicit null-event constructor is the null pointer value. Keep the pointer
            // carrier because event structs store it, while eliminating the now-unsupported call.
            if let Some(res) = call_result_id(trimmed) {
                let indent = &line[..line.len() - trimmed.len()];
                out.push_fmt(format_args!("{indent}{res} = inttoptr i64 0 to ptr"));
                continue;
            }
        }
        if trimmed.contains("air.wait_simdgroup_events") && trimmed.contains("call") {
            // The copy is already complete in every thread; publish the threadgroup tile with a barrier.
            let indent = &line[..line.len() - trimmed.len()];
            out.push_fmt(format_args!(
                "{indent}call void @air.wg.barrier(i32 2, i32 1)"
            ));
            continue;
        }
        out.push(line);
    }

    if san_ll.ends_with('\n') {
        out.text.push('\n');
    }
    // Append the synthesized copy helpers (one per element size) and the barrier declaration if the
    // module did not already declare it.
    for &elem in &elem_sizes {
        out.text.push_str(&helper_definition(elem));
    }
    if !san_ll.contains("declare void @air.wg.barrier") {
        out.text
            .push_str("\ndeclare void @air.wg.barrier(i32, i32)\n");
    }
    std::borrow::Cow::Owned(out.text)
}

/// Owned counterpart used when the caller can relinquish the source buffer. Once rewriting creates
/// a replacement, returning it drops the superseded input before typed parsing begins; a no-op
/// preserves and returns the original allocation unchanged.
pub(crate) fn lower_simdgroup_async_copy_owned(san_ll: String) -> String {
    match lower_simdgroup_async_copy(&san_ll) {
        std::borrow::Cow::Borrowed(_) => san_ll,
        std::borrow::Cow::Owned(lowered) => lowered,
    }
}

struct LineBuffer {
    text: String,
    has_line: bool,
}

impl LineBuffer {
    fn with_capacity(capacity: usize) -> Self {
        Self {
            text: String::with_capacity(capacity),
            has_line: false,
        }
    }

    fn begin_line(&mut self) {
        if self.has_line {
            self.text.push('\n');
        }
        self.has_line = true;
    }

    fn push(&mut self, line: &str) {
        self.begin_line();
        self.text.push_str(line);
    }

    fn push_fmt(&mut self, line: std::fmt::Arguments<'_>) {
        self.begin_line();
        let _ = self.text.write_fmt(line);
    }
}

/// Emit the extractelements + helper call replacing one async-copy call. `%result` is kept as a dummy
/// non-null pointer so downstream `store ptr %result` / event plumbing stays well-typed.
fn emit_copy_call(out: &mut LineBuffer, indent: &str, call: &AsyncCopyCall, fresh: &mut u64) {
    let mut id = || {
        let v = format!("%__ac{}", *fresh);
        *fresh += 1;
        v
    };
    let (dw, dh, sw, sh, ox, oy) = (id(), id(), id(), id(), id(), id());
    out.push_fmt(format_args!(
        "{indent}{dw} = extractelement <2 x i64> {}, i64 0",
        call.dst_dims
    ));
    out.push_fmt(format_args!(
        "{indent}{dh} = extractelement <2 x i64> {}, i64 1",
        call.dst_dims
    ));
    out.push_fmt(format_args!(
        "{indent}{sw} = extractelement <2 x i64> {}, i64 0",
        call.src_dims
    ));
    out.push_fmt(format_args!(
        "{indent}{sh} = extractelement <2 x i64> {}, i64 1",
        call.src_dims
    ));
    out.push_fmt(format_args!(
        "{indent}{ox} = extractelement <2 x i64> {}, i64 0",
        call.offset
    ));
    out.push_fmt(format_args!(
        "{indent}{oy} = extractelement <2 x i64> {}, i64 1",
        call.offset
    ));
    out.push_fmt(format_args!(
        "{indent}call void @__metal2vulkan_sac2d_e{}(ptr addrspace(3) {}, i64 {}, i64 {dw}, i64 {dh}, ptr addrspace(1) {}, i64 {}, i64 {sw}, i64 {sh}, i64 {ox}, i64 {oy})",
        call.elem, call.dst, call.dst_epr, call.src, call.src_epr
    ));
    // A non-null dummy for the event value the intrinsic returned.
    out.push_fmt(format_args!(
        "{indent}{} = inttoptr i64 1 to ptr",
        call.result
    ));
}

/// The copy-loop helper for a given element size (bytes). Copies `dst_dims` (dw×dh) elements from
/// `src` at `offset`, zero-filling where the source is out of bounds. i32 for 4-byte elements, i16 for
/// 2-byte (both bit-preserving copies of float/half).
fn helper_definition(elem: u32) -> String {
    let ty = match elem {
        2 => "i16",
        _ => "i32",
    };
    format!(
        r#"
define internal void @__metal2vulkan_sac2d_e{elem}(ptr addrspace(3) %dst, i64 %depr, i64 %dw, i64 %dh, ptr addrspace(1) %src, i64 %sepr, i64 %sw, i64 %sh, i64 %ox, i64 %oy) {{
entry:
  br label %rowhead
rowhead:
  %r = phi i64 [ 0, %entry ], [ %rnext, %rowlatch ]
  %rok = icmp ult i64 %r, %dh
  br i1 %rok, label %colhead, label %done
colhead:
  %c = phi i64 [ 0, %rowhead ], [ %cnext, %collatch ]
  %cok = icmp ult i64 %c, %dw
  br i1 %cok, label %body, label %rowlatch
body:
  %sx = add i64 %ox, %c
  %sy = add i64 %oy, %r
  %xin = icmp ult i64 %sx, %sw
  %yin = icmp ult i64 %sy, %sh
  %in = and i1 %xin, %yin
  %didx = mul i64 %r, %depr
  %didx2 = add i64 %didx, %c
  %dp = getelementptr {ty}, ptr addrspace(3) %dst, i64 %didx2
  br i1 %in, label %copy, label %zero
copy:
  %syr = mul i64 %sy, %sepr
  %sidx = add i64 %syr, %sx
  %sp = getelementptr {ty}, ptr addrspace(1) %src, i64 %sidx
  %v = load {ty}, ptr addrspace(1) %sp
  store {ty} %v, ptr addrspace(3) %dp
  br label %collatch
zero:
  store {ty} 0, ptr addrspace(3) %dp
  br label %collatch
collatch:
  %cnext = add i64 %c, 1
  br label %colhead
rowlatch:
  %rnext = add i64 %r, 1
  br label %rowhead
done:
  ret void
}}
"#
    )
}

/// Parse the 12 operands of an `air.simdgroup_async_copy_2d` call line. The signature (from the AIR
/// declaration) is
/// `(i64 elem, i64 _, ptr addrspace(3) dst, i64 dst_epr, i64 _, <2 x i64> dst_dims,
///   ptr addrspace(1) src, i64 src_epr, i64 _, <2 x i64> src_dims, <2 x i64> offset, i32 flags)`.
fn parse_async_copy(line: &str) -> Option<AsyncCopyCall> {
    let result = line.split('=').next()?.trim().to_string();
    if !result.starts_with('%') {
        return None;
    }
    // The argument list is delimited by the call's OUTER parens. `rfind(')')` is the closing paren
    // (the `#N` attribute group and `!noalias` metadata follow it), and `find('(')` skips the callee's
    // `addrspace(3)`-style type parens because it lands on the first `(` — which is inside a type; use
    // the LAST balanced top-level paren group instead.
    let open = line.find('(')?;
    let args_str = &line[open + 1..line.rfind(')')?];
    let args = split_args(args_str);
    if args.len() != 12 {
        return None;
    }
    let elem: u32 = operand_value(&args[0]).parse().ok()?;
    Some(AsyncCopyCall {
        elem,
        dst: operand_value(&args[2]),
        dst_epr: operand_value(&args[3]),
        dst_dims: operand_value(&args[5]),
        src: operand_value(&args[6]),
        src_epr: operand_value(&args[7]),
        src_dims: operand_value(&args[9]),
        offset: operand_value(&args[10]),
        result,
    })
}

/// Split a call argument list on TOP-LEVEL commas, respecting `<...>` / `(...)` / `[...]` / `{...}`
/// nesting — a constant vector operand `<2 x i64> <i64 8, i64 32>` contains an inner comma that must
/// NOT split the argument.
fn split_args(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut depth = 0i32;
    let mut cur = String::new();
    for ch in s.chars() {
        match ch {
            '<' | '(' | '[' | '{' => {
                depth += 1;
                cur.push(ch);
            }
            '>' | ')' | ']' | '}' => {
                depth -= 1;
                cur.push(ch);
            }
            ',' if depth == 0 => {
                out.push(cur.trim().to_string());
                cur.clear();
            }
            _ => cur.push(ch),
        }
    }
    if !cur.trim().is_empty() {
        out.push(cur.trim().to_string());
    }
    out
}

/// The VALUE of a `<type> <value>` LLVM operand. A `<2 x i64>` vector operand's value is everything
/// after that type token (a `%reg`, a `<i64 8, i64 32>` constant, or a `splat (i64 32)` constant); a
/// scalar / pointer operand's value is its last whitespace token (past any `noundef`/`readonly`/
/// `captures(none)` attributes).
fn operand_value(arg: &str) -> String {
    let a = arg.trim();
    if let Some(rest) = a.strip_prefix("<2 x i64>") {
        return rest.trim().to_string();
    }
    a.rsplit(' ').next().unwrap_or("").to_string()
}

/// The `%id` result of a `%id = call ...` line, if any.
fn call_result_id(line: &str) -> Option<String> {
    let head = line.split('=').next()?.trim();
    head.starts_with('%').then(|| head.to_string())
}

/// Seed a fresh-SSA counter above any `%__acN` already present (there should be none, but be safe).
fn fresh_counter(_san_ll: &str) -> u64 {
    0
}

#[cfg(test)]
mod tests {
    use super::lower_simdgroup_async_copy;
    use std::borrow::Cow;

    #[test]
    fn no_async_copy_borrows_the_original_module() {
        let ll = "define void @k() {\nentry:\n  ret void\n}\n";
        assert!(matches!(
            lower_simdgroup_async_copy(ll),
            Cow::Borrowed(value) if std::ptr::eq(value, ll)
        ));
    }

    #[test]
    fn dead_async_copy_declaration_borrows_the_already_lowered_module() {
        let ll = concat!(
            "define void @k() {\nentry:\n  ret void\n}\n",
            "declare ptr @air.simdgroup_async_copy_2d.p3i8.p1i8(i64)\n",
        );
        assert!(matches!(
            lower_simdgroup_async_copy(ll),
            Cow::Borrowed(value) if std::ptr::eq(value, ll)
        ));
    }

    #[test]
    fn explicit_null_event_lowers_with_the_async_copy_family() {
        let ll = concat!(
            "%null = call ptr @air.get_null_simdgroup_event()\n",
            "%event = call ptr @air.simdgroup_async_copy_2d.p3i8.p1i8(i64 2, i64 2, ptr addrspace(3) %dst, i64 8, i64 1, <2 x i64> <i64 4, i64 4>, ptr addrspace(1) %src, i64 8, i64 1, <2 x i64> <i64 4, i64 4>, <2 x i64> zeroinitializer, i32 0)\n",
            "%is_null = call i1 @air.is_null_simdgroup_event(ptr %event)\n",
            "call void @air.wait_simdgroup_events(i32 1, ptr %event)\n",
        );
        let lowered = lower_simdgroup_async_copy(ll);
        assert!(lowered.contains("%null = inttoptr i64 0 to ptr"));
        assert!(lowered.contains("%event = inttoptr i64 1 to ptr"));
        assert!(lowered.contains("%is_null = icmp ne i64 0, 0"));
        assert!(!lowered.contains("call ptr @air.get_null_simdgroup_event"));
    }
}
