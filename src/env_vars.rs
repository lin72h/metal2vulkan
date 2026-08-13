//! One registry for every `METAL2VULKAN_*` environment variable the crate honors (refactor S8).
//!
//! Environment reads previously lived at scattered `std::env::var` sites with subtly
//! different idioms (`var_os().is_some()` vs `var().is_ok()`, `map_or(true, |v| v != "0")` vs
//! `is_ok_and(|v| v == "0")`, ad-hoc integer parses) and none were documented in `--help`. Every var
//! now has exactly one typed accessor here plus a `REGISTRY` entry (name/default/effect) that
//! `metal2vulkan --help` prints.
//!
//! Accessor kinds:
//!   * **presence flag** — set iff the var is present (any value, incl. empty). Backed by
//!     `var_os().is_some()`, which also matches non-UTF-8 values; the two former `var().is_ok()`
//!     sites (`REPAIR_STATS`, `DBG_RAWBYTE`) are unaffected for every UTF-8 value.
//!   * **default-on flag** — on unless the value is exactly `"0"` (`var().map_or(true, |v| v != "0")`).
//!   * **integer** — parsed, with a default (and, for `VAL_PAR`, a `>= 1` floor).
//!   * **path** — the raw `OsString` value if present.
//!     All accessors re-read the environment live (the process env does not change mid-run, so this
//!     matches the former mix of cached and live reads).

use std::ffi::OsString;

/// One row in the documented registry. `default`/`effect` are for `--help` only.
pub struct EnvVar {
    pub name: &'static str,
    pub default: &'static str,
    pub effect: &'static str,
}

/// Every `METAL2VULKAN_*` var, grouped by kind, for `--help`. Order is stable (grouped, then
/// alphabetical) so the help text is deterministic.
pub const REGISTRY: &[EnvVar] = &[
    // integers
    EnvVar {
        name: "METAL2VULKAN_VAL_PAR",
        default: "3",
        effect: "max concurrent spirv-val processes (>=1)",
    },
    EnvVar {
        name: "METAL2VULKAN_RELOOPER_MAX_BLOCKS",
        default: "1024",
        effect: "requested relooper block-count cap (hard maximum: 1024)",
    },
    // paths
    EnvVar {
        name: "METAL2VULKAN_REPRO_DIR",
        default: "$TMPDIR/metal2vulkan-repros",
        effect: "base directory for FALLBACK repro bundles",
    },
    EnvVar {
        name: "METAL2VULKAN_RETRY_DUMP",
        default: "unset",
        effect: "path to dump a failing retry or corpus-audit SPIR-V module (debug)",
    },
    EnvVar {
        name: "METAL2VULKAN_<TOOL>",
        default: "PATH search",
        effect: "absolute per-tool override (for example METAL2VULKAN_LLVM_DIS or \
                 METAL2VULKAN_SPIRV_VAL)",
    },
    // presence-flag debug/trace toggles (set to any value to enable)
    EnvVar {
        name: "METAL2VULKAN_DBG_RAWBYTE",
        default: "off",
        effect: "trace raw-byte GEP indexing",
    },
    EnvVar {
        name: "METAL2VULKAN_RETRY_DEBUG",
        default: "off",
        effect: "trace each retry tier's emit/validate to stderr",
    },
    EnvVar {
        name: "METAL2VULKAN_TIER_CENSUS",
        default: "off",
        effect: "print the adopted retry tier for each translation",
    },
    EnvVar {
        name: "METAL2VULKAN_PARAM_POINTEE_DBG",
        default: "off",
        effect: "trace parameter-pointee sidecar mismatches",
    },
    EnvVar {
        name: "METAL2VULKAN_POINTEE_DBG",
        default: "off",
        effect: "trace typed-IR pointee-carrier mismatches",
    },
    EnvVar {
        name: "METAL2VULKAN_WHOLE_PART",
        default: "off",
        effect:
            "UNSAFE diagnostic: probe whole-composite local-pointer carriers; may emit invalid \
                 SPIR-V and must not be used as a product feature",
    },
    EnvVar {
        name: "METAL2VULKAN_REINTERP_REAL",
        default: "off",
        effect: "UNSAFE diagnostic: probe same-width float/integer pointer retyping; known \
                 nonconformant and never a product feature",
    },
    EnvVar {
        name: "METAL2VULKAN_STRADDLE_ADMIT",
        default: "off",
        effect:
            "UNSAFE diagnostic: bypass one structured-CFG straddle check; may emit invalid SPIR-V",
    },
    EnvVar {
        name: "METAL2VULKAN_PTR_NETWORK_WHY",
        default: "off",
        effect: "trace pointer phi/select networks with non-uniform pointee types (byte-neutral)",
    },
    EnvVar {
        name: "METAL2VULKAN_STORAGE_DBG",
        default: "off",
        effect: "trace storage-class inference",
    },
    EnvVar {
        name: "METAL2VULKAN_TEX_DBG",
        default: "off",
        effect: "trace texture lowering",
    },
    EnvVar {
        name: "METAL2VULKAN_TIR_DBG",
        default: "off",
        effect: "trace typed-IR construction",
    },
    EnvVar {
        name: "METAL2VULKAN_TIR_ONLY",
        default: "off",
        effect: "panic if an op falls back to the string parser (migration gate)",
    },
    EnvVar {
        name: "METAL2VULKAN_WHY",
        default: "off",
        effect: "print the structured-CFG admit/reject reason",
    },
    EnvVar {
        name: "METAL2VULKAN_RELOOP_WHY",
        default: "off",
        effect: "trace relooper entry, block counts, and exact bailout reasons (byte-neutral)",
    },
    EnvVar {
        name: "METAL2VULKAN_SPI_WHY",
        default: "off",
        effect: "trace exact structured-plan rejection points and plan flags (byte-neutral)",
    },
    // default-off measurement / gate substrates
    EnvVar {
        name: "METAL2VULKAN_CONVERGE_INLOOP",
        default: "off",
        effect:
            "measurement override: force in-loop merge convergence on every structured-plan attempt",
    },
    // default-off diagnostics (read-only eprintln, never change bytes)
    EnvVar {
        name: "METAL2VULKAN_UNMODELED_WHY",
        default: "off",
        effect: "trace synthesized unmodeled-pointer placeholders and their provenance",
    },
    EnvVar {
        name: "METAL2VULKAN_FLM_WHY",
        default: "off",
        effect: "trace loop-merge plans, selection collisions, and merge phis",
    },
    EnvVar {
        name: "METAL2VULKAN_EXIT_WHY",
        default: "off",
        effect: "trace illegal structured-exit edges and their innermost constructs",
    },
    EnvVar {
        name: "METAL2VULKAN_SWITCH_TAIL_WHY",
        default: "off",
        effect: "trace switch-case shared-continuation cloning candidates",
    },
];

/// Multi-line `--help` block listing every registry var.
pub fn help_text() -> String {
    let mut out = String::from(
        "environment variables (debug toggles are enabled by presence, regardless of value):\n",
    );
    for v in REGISTRY {
        out.push_str(&format!(
            "  {:<32} {} [default: {}]\n",
            v.name, v.effect, v.default
        ));
    }
    out
}

fn present(name: &str) -> bool {
    std::env::var_os(name).is_some()
}

// --- integers ---------------------------------------------------------------

/// Max concurrent spirv-val slots. Default 3; a parsed value `< 1` falls back to the default.
pub fn val_par() -> usize {
    std::env::var("METAL2VULKAN_VAL_PAR")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&n| n >= 1)
        .unwrap_or(3)
}

/// Requested relooper block-count cap. The native relooper applies its hard product safety maximum
/// after reading this value, so the override can lower the cap but cannot raise it above 1024.
pub fn relooper_max_blocks(fallback: usize) -> usize {
    std::env::var("METAL2VULKAN_RELOOPER_MAX_BLOCKS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(fallback)
}

// --- paths ------------------------------------------------------------------

pub fn repro_dir() -> Option<OsString> {
    std::env::var_os("METAL2VULKAN_REPRO_DIR")
}

pub fn retry_dump() -> Option<OsString> {
    std::env::var_os("METAL2VULKAN_RETRY_DUMP")
}

/// Optional path override for an external tool `cmd` (`llvm-dis`, `spirv-val`, …), read from
/// `METAL2VULKAN_<CMD>` (uppercased, `-`→`_`). A dynamic family — one var per tool the pipeline
/// shells out to — documented in `REGISTRY` as a single `METAL2VULKAN_<TOOL>` family row rather than
/// one entry per tool. Absent → the caller searches the known tool dirs then `PATH`.
pub fn tool_path_override(cmd: &str) -> Option<OsString> {
    std::env::var_os(format!(
        "METAL2VULKAN_{}",
        cmd.replace('-', "_").to_ascii_uppercase()
    ))
}

// --- presence-flag debug/trace toggles --------------------------------------

pub fn dbg_rawbyte() -> bool {
    present("METAL2VULKAN_DBG_RAWBYTE")
}
pub fn retry_debug() -> bool {
    present("METAL2VULKAN_RETRY_DEBUG")
}
pub fn tier_census() -> bool {
    present("METAL2VULKAN_TIER_CENSUS")
}
pub fn param_pointee_dbg() -> bool {
    present("METAL2VULKAN_PARAM_POINTEE_DBG")
}
pub fn pointee_dbg() -> bool {
    present("METAL2VULKAN_POINTEE_DBG")
}
pub fn whole_part() -> bool {
    present("METAL2VULKAN_WHOLE_PART")
}
pub fn reinterp_real() -> bool {
    present("METAL2VULKAN_REINTERP_REAL")
}
pub fn straddle_admit() -> bool {
    present("METAL2VULKAN_STRADDLE_ADMIT")
}
pub fn ptr_network_why() -> bool {
    present("METAL2VULKAN_PTR_NETWORK_WHY")
}
/// Analysis-only diagnostic: log every unmodeled-pointer PLACEHOLDER the native emitter synthesizes
/// (`emit_private_zero_pointer_value` — a null Private stand-in for a pointer computation the emitter
/// could not model). Surfaces the SPIR-V result id, pointee, and the emitter callsite ("site") that
/// gave up, so the provenance shape behind an M-B2 blocker can be identified and modeled. Default-off,
/// read-only (eprintln) → BC/G4/G5-neutral.
pub fn unmodeled_why() -> bool {
    present("METAL2VULKAN_UNMODELED_WHY")
}
/// Analysis-only diagnostic: log each illegal structured-exit edge `exit_check` finds
/// (`EXIT-ILLEGAL innermost-header/merge/block->succ`), surfacing why a block escapes its innermost
/// construct. Default-off, read-only (eprintln) → BC/G4/G5-neutral.
pub fn exit_why() -> bool {
    present("METAL2VULKAN_EXIT_WHY")
}
/// Analysis-only diagnostic: log the switch-case shared-continuation candidates
/// `privatize_dominated_region`'s deep-shared clone loop considers each round. Default-off, read-only
/// (eprintln) → BC/G4/G5-neutral.
pub fn switch_tail_why() -> bool {
    present("METAL2VULKAN_SWITCH_TAIL_WHY")
}
/// Analysis-only diagnostic: log every loop plan `forest_loop_merges` processes with its restructure
/// kind, natural merge block, whether that merge collides with an outer selection
/// (`merge_collides_with_outer_selection`), and whether the merge carries a phi. Surfaces the exact
/// loop-merge ⇄ selection-merge collision sites behind the dominant `cond-phi-shared/loop-role/
/// merge-inloop` reject class (the largest frontier structured-plan reject bucket), so the collision
/// topology of an M-B1 blocker can be characterized. Default-off, read-only (eprintln) → BC/G4/G5-neutral.
pub fn flm_why() -> bool {
    present("METAL2VULKAN_FLM_WHY")
}
/// Analysis-only diagnostic: log each relooper invocation's function count and, per function, its
/// block count and the exact bail reason (`too-few-blocks`/`too-many-blocks`/`empty-block`/
/// `unhandled-terminator`/`non-spillable-demote`). Surfaces WHY the relooper — repair's designed
/// replacement — cannot rescue an M-B2 NO_REPAIR blocker (05 bails on `non-spillable-demote`: its
/// loop-carried buffer pointers cannot be register-demoted under logical addressing). Default-off,
/// read-only (eprintln) → BC/G4/G5-neutral.
pub fn reloop_why() -> bool {
    present("METAL2VULKAN_RELOOP_WHY")
}
/// Measurement override (default-off): force the in-loop merge-collision convergence on ALL
/// `structured_plan` attempts, not just the reject-triggered 4th one that is now the DEFAULT. The
/// default 4th attempt fires only on base-REJECTING functions (byte-identical for admitting ones);
/// setting this forces the broadening on the base attempts too, to measure the large-scale byte churn
/// the unconditional broad form would cause (≈2170 rows vs the bounded reject-triggered set).
pub fn converge_inloop() -> bool {
    present("METAL2VULKAN_CONVERGE_INLOOP")
}
/// TEMP diagnostic (Keystone 2): log where `structured_plan_inner` returns None, tagged with the
/// converge_inloop flag + block count, so the ACTUAL residual reject of a merge-inloop function under
/// the 4th (converge=true) attempt can be observed (the `--structured-why` LABEL is computed with
/// converge=false and does not name why the converge attempt fails). Read-only eprintln → byte-neutral.
pub fn spi_why() -> bool {
    present("METAL2VULKAN_SPI_WHY")
}
pub fn storage_dbg() -> bool {
    present("METAL2VULKAN_STORAGE_DBG")
}
pub fn tex_dbg() -> bool {
    present("METAL2VULKAN_TEX_DBG")
}
pub fn tir_dbg() -> bool {
    present("METAL2VULKAN_TIR_DBG")
}
pub fn tir_only() -> bool {
    present("METAL2VULKAN_TIR_ONLY")
}
pub fn why() -> bool {
    present("METAL2VULKAN_WHY")
}
