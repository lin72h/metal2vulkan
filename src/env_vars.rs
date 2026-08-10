//! One registry for every `METAL2VULKAN_*` environment variable the crate honors (refactor S8).
//!
//! Before this module the ~19 vars were read at ~33 scattered `std::env::var` sites with subtly
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
        effect: "path to dump a failing inline+SROA module (debug)",
    },
    EnvVar {
        name: "METAL2VULKAN_<TOOL>",
        default: "PATH search",
        effect: "dynamic family: per-external-tool path override — METAL2VULKAN_<CMD> (uppercased, \
                 -> _) gives an absolute path to that tool binary (e.g. METAL2VULKAN_LLVM_DIS, \
                 METAL2VULKAN_SPIRV_VAL). Absent → search the known tool dirs then PATH",
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
        effect: "print `[tier-census] <tier>` per translate: which retry tier (if any) was \
                 adopted (M-C1 cascade-redundancy telemetry)",
    },
    EnvVar {
        name: "METAL2VULKAN_PARAM_POINTEE_DBG",
        default: "off",
        effect: "enumerate S18 param-pointee sidecar divergences (--param-pointee-check)",
    },
    EnvVar {
        name: "METAL2VULKAN_POINTEE_DBG",
        default: "off",
        effect: "enumerate M2 pointee-carrier divergences (--tir-pointee-check)",
    },
    EnvVar {
        name: "METAL2VULKAN_WHOLE_PART",
        default: "off",
        effect: "M-A2(b) whole-vs-part: upgrade a SCALAR local-pointer pointee to the use-implied \
                 WHOLE composite carrier (Vector(S,N)/[N x S] of the same scalar S) when the pointer \
                 is dereferenced as the whole. Excludes raw_offsets/unmodeled/byte-view and any \
                 pointer participating in a phi/select pointer merge. Byte-changing; default OFF — \
                 the 112 changed banked cases MoltenVK-conform (109 byte-exact + 3 flag-independent \
                 FP-drift) ONLY because the retry cascade rescues its INVALID primary emits: the \
                 upgrade widens a load's result to the whole vector WITHOUT retyping the feeding \
                 access-chain pointer, so a bitcast-aliased whole-vector load through a scalar-element \
                 access chain emits `OpLoad <4 x half> from half*` (spirv-val invalid — see the \
                 native_pointer_bitcast_vector_load_store_uses_scalar_lanes test). The flip is blocked \
                 by that partial-retyping soundness gap, NOT by G8 hardware; it needs consistent \
                 def-site pointer-network retyping (M-A2(c)/M-B1 keystone) first. Frozen by dead-end \
                 A9 (read-side prefer-carrier retyping is proven unsound); kept as measurement \
                 substrate only",
    },
    EnvVar {
        name: "METAL2VULKAN_REINTERP_REAL",
        default: "off",
        effect: "M-A2(a) Float<->Int reinterpret (DIAGNOSTIC/UNSOUND — do NOT flip): upgrade a SCALAR \
                 local-pointer pointee to a use-implied carrier scalar of the SAME bit width but \
                 different kind (Float(32)<->Int(32), Half(16)<->Int(16), …). Enumerable via \
                 --reinterp-real-check. Proven NON-conformant: a G7 MoltenVK sample miscompiles the \
                 topk_common_matrix_float family (naive prefer-carrier picks one arm of a genuine \
                 reinterpret). Frozen by dead-end A9 (read-side prefer-carrier retyping is proven \
                 unsound). Kept default-off as the measurement substrate for the eventual sound \
                 def-site-unambiguous version; never flip in this naive form",
    },
    EnvVar {
        name: "METAL2VULKAN_STRADDLE_ADMIT",
        default: "off",
        effect: "M-B1 straddle DIAGNOSTIC (default-off, NOT a fix): bypass the \
                 `selection:straddle-loop-merge` self-check so the structured plan is admitted anyway. \
                 For the enclosing-guard early-return shape (05/b00a8a8d — top-level `if(!c) return` \
                 guards whose false arm is the OpReturn block that doubles as the loop merge) this lets \
                 the downstream synth run and exposes the NEXT blocker (a byte-view pointer-phi) the \
                 CFG reject otherwise masks. Flag-on emits invalid SPIR-V for genuine straddles, so it \
                 is a single-case probe knob for phi/pointee work, never enabled at large scale",
    },
    EnvVar {
        name: "METAL2VULKAN_PTR_NETWORK_WHY",
        default: "off",
        effect: "M-A2/M-B1 DIAGNOSTIC (default-off, read-only): per function, print each pointer \
                 network (connected component over phi result↔incoming + select result↔arm edges) \
                 whose recorded pointees are non-uniform, tagged whole-vs-part / reinterpret-mix / \
                 unclassified. Builds the grouping M-A2's def-site finest-granularity recording needs \
                 and quantifies its scope. Changes no bytes",
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
        effect: "M-B DIAGNOSTIC (default-off, read-only): per relooper invocation, print the function \
                 count handed in (RELOOP-ENTER) and, per function, its block count and the reason \
                 rewrite_to_relooper bails (RELOOP-FN / RELOOP-BAIL: too-few/too-many-blocks, \
                 empty-block, unhandled-terminator, non-spillable-demote). Surfaces WHY the relooper \
                 (repair's designed replacement) cannot rescue an M-B2 NO_REPAIR blocker — e.g. 05 \
                 bails on non-spillable pointer demotion (loop-carried buffer pointers). Changes no bytes",
    },
    EnvVar {
        name: "METAL2VULKAN_SPI_WHY",
        default: "off",
        effect: "Keystone-2 DIAGNOSTIC (default-off, read-only eprintln): log the exact point + the \
                 (converge_inloop, break_aware) flags at which structured_plan_inner returns \
                 None for a function, so the ACTUAL residual reject of a merge-inloop case can be seen \
                 (the --structured-why LABEL is computed with converge=false and does not name why the \
                 converge/protect attempts fail). On a branch-no-merge reject it also dumps the sblocks \
                 skeleton. Changes no bytes",
    },
    // default-off measurement / gate substrates
    EnvVar {
        name: "METAL2VULKAN_CONVERGE_INLOOP",
        default: "off",
        effect: "Measurement override: force the in-loop merge-collision convergence on ALL \
                 structured_plan attempts, not just the reject-triggered 4th (the default). Measures the \
                 large-scale byte churn (~2170 rows) the unconditional broad form would cause vs the \
                 bounded reject-triggered set",
    },
    // default-off diagnostics (read-only eprintln, never change bytes)
    EnvVar {
        name: "METAL2VULKAN_UNMODELED_WHY",
        default: "off",
        effect: "log every unmodeled-pointer PLACEHOLDER the emitter synthesizes \
                 (emit_private_zero_pointer_value) with result id, pointee, and callsite, so the \
                 provenance behind a blocker can be identified",
    },
    EnvVar {
        name: "METAL2VULKAN_FLM_WHY",
        default: "off",
        effect: "log every loop plan forest_loop_merges processes with its restructure kind, natural \
                 merge block, whether that merge collides with an outer selection, and whether it \
                 carries a phi — the loop-merge/selection-merge collision topology behind the \
                 cond-phi-shared/loop-role/merge-inloop reject class",
    },
    EnvVar {
        name: "METAL2VULKAN_EXIT_WHY",
        default: "off",
        effect: "log each illegal structured-exit edge exit_check finds (EXIT-ILLEGAL \
                 innermost-header/merge/block->succ), surfacing why a block escapes its innermost \
                 construct",
    },
    EnvVar {
        name: "METAL2VULKAN_SWITCH_TAIL_WHY",
        default: "off",
        effect: "log the switch-case shared-continuation candidates privatize_dominated_region's \
                 deep-shared clone loop considers each round",
    },
];

/// Multi-line `--help` block listing every registry var.
pub fn help_text() -> String {
    let mut out = String::from("environment variables:\n");
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
