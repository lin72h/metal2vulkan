use crate::{DataFormat, Expected, Tolerance};

pub fn assert_conforms(id: &str, candidate: &[u8], expected: &Expected) {
    assert!(
        !expected.is_empty(),
        "[{id}] expected golden is empty; run on macOS first"
    );
    if candidate.len() != expected.bytes.len() {
        dump_mismatch(id, candidate, expected.bytes, None);
    }
    assert_eq!(
        candidate.len(),
        expected.bytes.len(),
        "[{id}] LENGTH mismatch: candidate={} expected={}",
        candidate.len(),
        expected.bytes.len()
    );

    if candidate == expected.bytes {
        eprintln!("[{id}] MATCH byte-exact");
        return;
    }

    if expected.tolerance.is_exact() {
        let diff = first_diff(candidate, expected.bytes);
        dump_mismatch(id, candidate, expected.bytes, Some(diff));
        panic!("[{id}] MISMATCH first_diff={}", diff);
    }

    assert!(
        tolerance_components_supported(expected.format, expected.bytes.len(), expected.tolerance),
        "[{id}] MISMATCH: tolerance declared for unsupported format {:?}",
        expected.format
    );

    let report = compare_with_tolerance(candidate, expected);
    if report.ok {
        eprintln!(
            "[{id}] TOLERANCE max_abs={} max_ulp={} reason={}",
            report.max_abs_seen,
            report.max_ulp_seen,
            expected.tolerance.reason().unwrap_or("unspecified")
        );
    } else {
        dump_mismatch(id, candidate, expected.bytes, Some(report.component));
        panic!(
            "[{id}] MISMATCH tolerance exceeded at component {}: abs={} ulp={} reason={}",
            report.component,
            report.max_abs_seen,
            report.max_ulp_seen,
            expected.tolerance.reason().unwrap_or("unspecified")
        );
    }
}

fn dump_mismatch(id: &str, candidate: &[u8], expected: &[u8], first_diff: Option<usize>) {
    let Some(root) = std::env::var_os("METAL2VULKAN_DUMP_MISMATCH_DIR") else {
        return;
    };
    let dir = std::path::PathBuf::from(root).join(sanitize_id(id));
    if let Err(e) = std::fs::create_dir_all(&dir) {
        eprintln!(
            "[{id}] failed to create mismatch dump {}: {e}",
            dir.display()
        );
        return;
    }
    if let Err(e) = std::fs::write(dir.join("candidate.bin"), candidate) {
        eprintln!("[{id}] failed to write mismatch candidate: {e}");
        return;
    }
    if let Err(e) = std::fs::write(dir.join("expected.bin"), expected) {
        eprintln!("[{id}] failed to write mismatch expected: {e}");
        return;
    }
    let summary = format!(
        "id={id}\ncandidate_len={}\nexpected_len={}\nfirst_diff={}\n",
        candidate.len(),
        expected.len(),
        first_diff
            .map(|value| value.to_string())
            .unwrap_or_else(|| "n/a".to_string())
    );
    if let Err(e) = std::fs::write(dir.join("summary.txt"), summary) {
        eprintln!("[{id}] failed to write mismatch summary: {e}");
    }
}

fn sanitize_id(id: &str) -> String {
    id.chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-') {
                ch
            } else {
                '_'
            }
        })
        .collect()
}

#[derive(Clone, Copy, Debug)]
struct ToleranceReport {
    ok: bool,
    component: usize,
    max_abs_seen: f32,
    max_ulp_seen: u32,
}

fn first_diff(candidate: &[u8], expected: &[u8]) -> usize {
    candidate
        .iter()
        .zip(expected.iter())
        .position(|(a, b)| a != b)
        .unwrap_or(candidate.len().min(expected.len()))
}

fn compare_with_tolerance(candidate: &[u8], expected: &Expected) -> ToleranceReport {
    let mut report = ToleranceReport {
        ok: true,
        component: 0,
        max_abs_seen: 0.0,
        max_ulp_seen: 0,
    };

    for (component, (c, e)) in components(
        candidate,
        expected.bytes,
        expected.format,
        expected.tolerance,
    )
    .into_iter()
    .enumerate()
    {
        if c.value.is_nan() && e.value.is_nan() {
            continue;
        }
        let abs = (c.value - e.value).abs();
        let ulp = c.ulp_distance(e);
        report.max_abs_seen = report.max_abs_seen.max(abs);
        report.max_ulp_seen = report.max_ulp_seen.max(ulp);
        if !within(expected.tolerance, abs, ulp) {
            report.ok = false;
            report.component = component;
            return report;
        }
    }
    report
}

fn within(tolerance: Tolerance, abs: f32, ulp: u32) -> bool {
    match tolerance {
        Tolerance::Exact => abs == 0.0 && ulp == 0,
        Tolerance::Abs { max_abs, .. } => abs <= max_abs,
        Tolerance::Ulp { max_ulp, .. } => ulp <= max_ulp,
        Tolerance::RawF16Ulp { max_ulp, .. } => ulp <= max_ulp,
        Tolerance::RawU8Ulp { max_ulp, .. } => ulp <= max_ulp,
        Tolerance::AbsAndUlp {
            max_abs, max_ulp, ..
        } => abs <= max_abs || ulp <= max_ulp,
    }
}

fn tolerance_components_supported(format: DataFormat, len: usize, tolerance: Tolerance) -> bool {
    match tolerance {
        Tolerance::RawF16Ulp { .. } => format == DataFormat::RawBytes && len.is_multiple_of(2),
        Tolerance::RawU8Ulp { .. } => format == DataFormat::RawBytes,
        _ => format.is_float_like() || (format == DataFormat::RawBytes && len.is_multiple_of(4)),
    }
}

#[derive(Clone, Copy, Debug)]
struct Component {
    value: f32,
    bits: ComponentBits,
}

impl Component {
    fn ulp_distance(self, other: Component) -> u32 {
        match (self.bits, other.bits) {
            (ComponentBits::F32(a), ComponentBits::F32(b)) => {
                ordered_f32(a).abs_diff(ordered_f32(b))
            }
            (ComponentBits::F16(a), ComponentBits::F16(b)) => {
                ordered_f16(a).abs_diff(ordered_f16(b))
            }
            (ComponentBits::U8(a), ComponentBits::U8(b)) => a.abs_diff(b) as u32,
            _ => u32::MAX,
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum ComponentBits {
    F32(u32),
    F16(u16),
    U8(u8),
}

fn components(
    candidate: &[u8],
    expected: &[u8],
    format: DataFormat,
    tolerance: Tolerance,
) -> Vec<(Component, Component)> {
    if matches!(tolerance, Tolerance::RawU8Ulp { .. }) {
        return candidate
            .iter()
            .zip(expected.iter())
            .map(|(c, e)| {
                (
                    Component {
                        value: *c as f32,
                        bits: ComponentBits::U8(*c),
                    },
                    Component {
                        value: *e as f32,
                        bits: ComponentBits::U8(*e),
                    },
                )
            })
            .collect();
    }
    if matches!(tolerance, Tolerance::RawF16Ulp { .. }) {
        return candidate
            .chunks_exact(2)
            .zip(expected.chunks_exact(2))
            .map(|(c, e)| {
                let c_bits = u16::from_le_bytes(c.try_into().unwrap());
                let e_bits = u16::from_le_bytes(e.try_into().unwrap());
                (
                    Component {
                        value: f16_to_f32(c_bits),
                        bits: ComponentBits::F16(c_bits),
                    },
                    Component {
                        value: f16_to_f32(e_bits),
                        bits: ComponentBits::F16(e_bits),
                    },
                )
            })
            .collect();
    }
    match format {
        DataFormat::RawBytes
        | DataFormat::F32
        | DataFormat::Rgba32Float
        | DataFormat::R32Float
        | DataFormat::Depth32Float => candidate
            .chunks_exact(4)
            .zip(expected.chunks_exact(4))
            .map(|(c, e)| {
                let c_bits = u32::from_le_bytes(c.try_into().unwrap());
                let e_bits = u32::from_le_bytes(e.try_into().unwrap());
                (
                    Component {
                        value: f32::from_bits(c_bits),
                        bits: ComponentBits::F32(c_bits),
                    },
                    Component {
                        value: f32::from_bits(e_bits),
                        bits: ComponentBits::F32(e_bits),
                    },
                )
            })
            .collect(),
        DataFormat::Rgba16Float => candidate
            .chunks_exact(2)
            .zip(expected.chunks_exact(2))
            .map(|(c, e)| {
                let c_bits = u16::from_le_bytes(c.try_into().unwrap());
                let e_bits = u16::from_le_bytes(e.try_into().unwrap());
                (
                    Component {
                        value: f16_to_f32(c_bits),
                        bits: ComponentBits::F16(c_bits),
                    },
                    Component {
                        value: f16_to_f32(e_bits),
                        bits: ComponentBits::F16(e_bits),
                    },
                )
            })
            .collect(),
        DataFormat::Rgba8Unorm => candidate
            .iter()
            .zip(expected.iter())
            .map(|(c, e)| {
                (
                    Component {
                        value: *c as f32 / 255.0,
                        bits: ComponentBits::U8(*c),
                    },
                    Component {
                        value: *e as f32 / 255.0,
                        bits: ComponentBits::U8(*e),
                    },
                )
            })
            .collect(),
        DataFormat::U32
        | DataFormat::I32
        | DataFormat::Rgba8Uint
        | DataFormat::Rgba8Sint
        | DataFormat::Rgba16Uint
        | DataFormat::Depth24Stencil8 => Vec::new(),
    }
}

fn ordered_f32(bits: u32) -> i32 {
    let signed = bits as i32;
    if signed < 0 {
        i32::MIN - signed
    } else {
        signed
    }
}

fn ordered_f16(bits: u16) -> i32 {
    let signed = bits as i16 as i32;
    if signed < 0 {
        i16::MIN as i32 - signed
    } else {
        signed
    }
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exp = ((bits >> 10) & 0x1f) as i32;
    let frac = (bits & 0x03ff) as u32;
    let f32_bits = if exp == 0 {
        if frac == 0 {
            sign
        } else {
            let mut frac_norm = frac;
            let mut shift = 0;
            while (frac_norm & 0x0400) == 0 {
                frac_norm <<= 1;
                shift += 1;
            }
            frac_norm &= 0x03ff;
            let exp32 = (127 - 15 - shift) as u32;
            sign | (exp32 << 23) | (frac_norm << 13)
        }
    } else if exp == 0x1f {
        sign | 0x7f80_0000 | (frac << 13)
    } else {
        let exp32 = (exp - 15 + 127) as u32;
        sign | (exp32 << 23) | (frac << 13)
    };
    f32::from_bits(f32_bits)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_exact_formats_require_exact_bytes() {
        let expected = Expected::exact(DataFormat::U32, &[1, 0, 0, 0]);
        assert_conforms("exact", &[1, 0, 0, 0], &expected);
        let failed =
            std::panic::catch_unwind(|| assert_conforms("exact", &[2, 0, 0, 0], &expected));
        assert!(failed.is_err());
    }

    #[test]
    fn rgba8unorm_abs_tolerance_compares_normalized_components() {
        let expected = Expected::with_tolerance(
            DataFormat::Rgba8Unorm,
            Tolerance::Abs {
                max_abs: 1.0 / 255.0,
                reason: "one channel step",
            },
            &[10, 20, 30, 40],
        );
        assert_conforms("rgba8unorm", &[11, 20, 30, 40], &expected);
    }

    #[test]
    fn f32_ulp_tolerance_uses_float_ordering() {
        static ONE: [u8; 4] = [0x00, 0x00, 0x80, 0x3f];
        static NEXT: [u8; 4] = [0x01, 0x00, 0x80, 0x3f];
        let expected = Expected::with_tolerance(
            DataFormat::F32,
            Tolerance::Ulp {
                max_ulp: 1,
                reason: "one f32 ulp",
            },
            &ONE,
        );
        assert_conforms("f32", &NEXT, &expected);
    }

    #[test]
    fn tolerance_treats_nan_payloads_as_matching() {
        static APPLE_NAN: [u8; 8] = [0x00, 0x7e, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
        static VULKAN_NAN: [u8; 8] = [0xff, 0x7f, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
        let expected = Expected::with_tolerance(
            DataFormat::Rgba16Float,
            Tolerance::Ulp {
                max_ulp: 0,
                reason: "nan payloads are not portable",
            },
            &APPLE_NAN,
        );
        assert_conforms("f16-nan-payload", &VULKAN_NAN, &expected);
    }

    #[test]
    fn rawbytes_tolerance_treats_f32_nan_payloads_as_matching() {
        static APPLE_NAN: [u8; 12] = [
            0x00, 0x00, 0xc0, 0x7f, 0x00, 0x00, 0xc0, 0x7f, 0x00, 0x00, 0xc0, 0x7f,
        ];
        static VULKAN_NAN: [u8; 12] = [
            0xff, 0xff, 0xff, 0x7f, 0xff, 0xff, 0xff, 0x7f, 0xff, 0xff, 0xff, 0x7f,
        ];
        let expected = Expected::with_tolerance(
            DataFormat::RawBytes,
            Tolerance::Ulp {
                max_ulp: 0,
                reason: "raw f32 nan payloads are not portable",
            },
            &APPLE_NAN,
        );

        assert_conforms("rawbytes-f32-nan-payload", &VULKAN_NAN, &expected);
    }

    #[test]
    fn rawbytes_f16_ulp_tolerance_uses_half_words() {
        static APPLE: [u8; 4] = [0xa0, 0x34, 0x8d, 0x1c];
        static VULKAN: [u8; 4] = [0xa0, 0x34, 0x8e, 0x1c];
        let expected = Expected::with_tolerance(
            DataFormat::RawBytes,
            Tolerance::RawF16Ulp {
                max_ulp: 1,
                reason: "raw half words",
            },
            &APPLE,
        );

        assert_conforms("rawbytes-f16-high-halfword", &VULKAN, &expected);
    }

    #[test]
    fn rawbytes_u8_ulp_tolerance_uses_bytes() {
        static APPLE: [u8; 4] = [219, 251, 255, 255];
        static VULKAN: [u8; 4] = [218, 251, 255, 255];
        let expected = Expected::with_tolerance(
            DataFormat::RawBytes,
            Tolerance::RawU8Ulp {
                max_ulp: 1,
                reason: "raw packed bytes",
            },
            &APPLE,
        );

        assert_conforms("rawbytes-u8-pack-rounding", &VULKAN, &expected);
    }
}
