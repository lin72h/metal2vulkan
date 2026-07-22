//! Fast iteration harness: translate a sanitized AIR .ll through the native emitter and spirv-val.
//! Usage: cargo run -q --example translate_native -- <in.ll> [kernel|fragment|vertex] [out.spv]
use metal2vulkan::passes::Stage;

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .expect("usage: translate_native <in.ll> [stage] [out.spv]");
    let stage = match args.next().as_deref() {
        Some("fragment") => Stage::Fragment,
        Some("vertex") => Stage::Vertex,
        _ => Stage::Kernel,
    };
    let out = args.next();
    let san_ll = std::fs::read_to_string(&path).expect("read input");
    let tmp = std::env::temp_dir().join(format!("translate_native_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    match metal2vulkan::translate_sanitized_native(&san_ll, stage, &tmp) {
        Err(e) => {
            eprintln!("FALLBACK: {e}");
            std::process::exit(2);
        }
        Ok(spv) => {
            if let Some(out) = &out {
                std::fs::write(out, &spv).unwrap();
            }
            match metal2vulkan::tools::spirv_val_bytes(&spv, &tmp) {
                Ok(()) => println!("PASS spirv-val ({} bytes)", spv.len()),
                Err(e) => {
                    eprintln!("INVALID-SPIRV: {e}");
                    std::process::exit(3);
                }
            }
        }
    }
    let _ = std::fs::remove_dir_all(&tmp);
}
