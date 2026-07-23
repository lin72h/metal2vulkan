# metal2vulkan architecture

A map of the AIR→SPIR-V translator: the primary pipeline, the failure-triggered retry cascade, and
how the pieces fit together. This document describes **structure**. Regression coverage for this
public tree is the unit/integration suite plus optional local A/B and oracle tooling.

## The pipeline

Unlike the legacy `metal2vulkanspirv` path (OpenCL/Kernel SPIR-V backend plus heavy post-rewrite),
this crate emits `OpCapability Shader` / `Logical GLSL450` SPIR-V **directly** from sanitized AIR
LLVM IR, structured by construction where the CFG admits it.

```
.air|.ll
  └─ tools::air_to_sanitized_ll          llvm-dis (if .air) + sanitize → textual LLVM IR
  └─ lower_async_copy_if_enabled         shared pre-emit AIR rewrites
  └─ meta::parse_air_{frag,vert,kern}    stage interface metadata (bindings, attributes)
  └─ native emit (tools::llc_vulkan_spirv → native::emit_vulkan_spirv)
        parse (native/parse.rs, lex.rs) → LlModule
        typed IR                        (native/tir/, native/ir/)
        structured-by-construction emit (native/emitter/**, native/cfg/**)
          └─ structured_plan admits → Op*Merge; else unrepaired emit + relooper on retry
  └─ retained SpirvModule + EmitSidecar
  └─ stage/resource/lowering passes     (passes/**)
  └─ finish_module                      id canonicalization, portability normalization
  └─ assemble → SPIR-V words
  └─ spirv-val (vulkan1.3)              [validation gate; not part of the bytes]
```

`translate_native_no_retry` is this path up to (but not including) spirv-val and the retry cascade.
It is the pure primary emit used when you want “what the emitter produced first,” without rescue
tiers.

### Text vs typed IR

The production body path is **parse once → typed IR → emit**. There is no mid-pipeline
`Vec<String>` function body:

- LLVM-IR text is read in the parser (`parse.rs` / `lex.rs`) and in `meta/` for AIR metadata.
- Instruction and terminator **semantics** are carried on typed `TirInst` / `TirBlock` fields
  (`operands`, `call`, `phi_incoming`, etc.). The emitter walks those carriers
  (`emit_body_inst`); unmigrated opcodes fail visibly rather than re-lexing text.
- The CFG structurizer (`cfg/**`) mutates typed block carriers (`BodyBlock.typed`), not string
  lines. Roles such as loop-merge / terminal-exit are typed tags on `BodyBlock`, not name-decoded
  decisions in production control flow.
- Four pre-emit rewrites still run as **text→text** on the sanitized AIR string before parse:
  `async_copy`, `vec_scalar_merge`, `sroa`, and `inline` (including the `inline_sroa*` retry
  variants). They are intentional pre-parse passes, not mid-pipeline body text.

### Reflection

Each translation decodes AIR metadata once into a `StageMeta` shared by emission, passes,
reflection, and retry re-emissions. Public `translate_*_reflected` entry points return
`(Vec<u8>, reflect::ShaderReflection)` with **bytes identical** to the non-reflected call.

Descriptor ABI bases (buffer = index, texture = 32+n, sampler = 64+n, color = 96+n, set 0) live in
`reflect` and are consumed by the resource pass so decorated bindings and reported reflection share
one contract. Consumer-oriented guide: [`REFLECTION.md`](REFLECTION.md).

### Retained module seam

The native emitter and the typed passes share one crate-owned `SpirvModule`. Emission returns an
internal `EmittedSpirv` (module + `EmitSidecar`); `finish_module` hands both into the pass
context. The validating-primary path assembles once after passes complete.

`SpirvModule` owns section layout, load/assemble, result-id allocation, and instruction/block
nodes. Retries may re-enter from bytes, but they parse back into the same module type. There is no production dependency on an external SPIR-V builder; optional validation
helpers may use `rspirv` as an offline oracle only.

Related ownership:

| Area | Location |
|---|---|
| Layout rules (AIR / SPIR-V / tight / padded) | crate `layout` module |
| Value/type queries on retained SPIR-V | `passes/value_queries.rs` |
| Access-chain / subword / Workgroup / Private | `passes/access/` |
| Structured-CFG repairs after AIR-call phase | `passes/cfg_repair/` |
| AIR/LLVM intrinsic lowering | `passes/air_calls/` (+ `images/`) |
| Workgroup interface + zero-init | `passes/workgroup/` |
| Resource discovery, bindings, rewrites | `passes/resources/` |
| Stage inputs / outputs | `passes/stage_input/`, `passes/stage_output/` |
| Finalize, GC, caps, type singletons | `passes/finalize.rs`, `module_cleanup.rs`, `type_singletons.rs` |

The emit sidecar carries Word-id-keyed facts (buffer-address words, local-pointer field stores,
static/dynamic field loads, AIR struct offsets). Producers append; inlining and resource rewrites
remap them. There is no cross-seam `OpName` marker protocol for facts.

### Inlining

Helpers are inlined on the producer side where structural rules allow (one-block and multi-block
leaf helpers, constructor CFGs). Typed SSA, pointee/raw-buffer facts, and sidecar IDs move with
the splice. Residual indirect/bodiless/conflicting cases stay as calls. Dead-function pruning and
chained-access composition remain in their post-inline cleanup phase; `passes/emitted_inline/` is
the producer-side inliner.

### Structurizer

Emission is structured-by-construction when possible:

1. `native::cfg` builds a loop/selection forest from the AIR CFG.
2. `structured_plan` decides whether that forest is expressible as Vulkan structured control flow.
3. **Admit** → emitter walks the plan (`OpLoopMerge` / `OpSelectionMerge`), then
   `repair_pre_phi_incoming_materializations` relocates illegal in-phi-block access chains.
4. **Reject** → function emits with inferred merges unrepaired; the retry cascade’s **relooper**
   (`native/relooper.rs`) strips merges and rebuilds structured CFG. The relooper is the sole
   structuring fallback after the old post-hoc repair roster was deleted.

Between reject and relooper, a few specialized constructions may still admit a shape
(`structured_plan_divergent_exit`, construct-tree own-arm retry for a narrow class). Adoption of
retry-produced modules is always **spirv-val gated**.

Diagnose admit/reject reasons with `METAL2VULKAN_WHY=1` on a local translate of the module under
study.

## Retry cascade

When the primary emit fails validation (or the emitter returns an error), translation walks a fixed
ladder of retry **tiers**. Mechanisms live in `retry.rs`; routing lives in
`translate_sanitized_with_meta` as a match on `native::classify_{validation,emit}_error`.

Two invariants:

1. **Adopt only if the candidate validates.** A module that already passed spirv-val never enters
   the cascade; a non-validating retry result is discarded.
2. **Tier order is load-bearing.** Example: value-select before PhysicalStorageBuffer (device-address)
   lowering, because some drivers cannot pipeline BDA modules as compute; inline+SROA before relooper
   where that ordering has been proven safer. Do not reorder tiers casually.

Classification of validator/emitter text into `ValidationClass` / `EmitErrorClass` is confined to
`native` (`classify_validation_error` / `classify_emit_error`). The cascade then routes by class:

| Class | Typical tier order (census labels under `METAL2VULKAN_TIER_CENSUS`) |
|---|---|
| PointerTyping | `fc_promote_logical` → raw → prune → subword pack → raw+relooper → `fc_promote_psb` |
| CfgStructurization | relooper → relooper (huge) → raw+relooper |
| LogicalPointerPhi | phi-index legalization → prune |
| CrossBindingPointerMerge | value-select → PSB → raw+PSB → prune |
| Other (validation) | prune → raw+relooper → value-select |
| Emit PointerTyping / Other | raw / BDA / PSB / inline-SROA combinations |

CFG shapes that the structurizer rejects now fail **validation** (unrepaired merges), not a separate
CFG emit-error arm. Emit-error routing is essentially `{PointerTyping, Other}`.

### Primary-path rewrites

Before the cascade, several validation-gated rewrites run on the primary path itself
(`apply_primary_emit_rewrites` and related helpers in `lib.rs`). Each is a no-op on modules that do
not carry its shape. Examples: cross-binding value-domain lowering, logical-pointer-phi
legalization, multi-entry loop split, and reject-triggered structured-plan admission handlers.
These run unconditionally on the product path; `src/env_vars.rs` is for diagnostics, tool overrides,
and default-off measurement substrates only—not product feature gates.

## Verification

Match the check to the change. The full developer playbook (harvest → bank hashes → A/B while
refactoring, what to commit, recipes by change type) lives in **[`VALIDATION.md`](VALIDATION.md)**.

| Check | Command / tool | Catches |
|---|---|---|
| Lint | `cargo clippy` | style / obvious bugs |
| Unit + integration | `cargo test -- --test-threads=1` | translator regressions |
| SPIR-V delta class | validation `classify_spirv_delta` | id/order vs semantic drift |
| Byte A/B | `scripts/metal2vulkan-ab/` | drift on a local sample |
| Translate ledger | `corpus-mint` / `corpus-remint` / `corpus-why` | AIR/SPIR-V sha256 pins |
| Execution ledgers | `corpus-run-metal` / `-vulkan` / `-moltenvk` | plan + `output_b64` + digests (JSONL) |
| Private corpus (optional) | `validation/corpus/local/` | local shard source for ledgers / A/B |
| Metal oracle (optional) | `validation` on macOS | host Metal execution |
| Vulkan executor (optional) | `validation` runner | candidate vs oracle bytes |

Always rebuild the translator binary before measuring. Default CI is synthetic-only; private
metallib harvest is gitignored under `validation/corpus/local/` (see that README).

## Hard-won constraints

- **spirv-val validity ≠ end-to-end correctness.** Retyping a load/store carrier only to silence the
  validator is unsound if it changes meaning.
- **Post-hoc repair was deleted for a reason.** Remaining “repair” is narrow and structural (e.g.
  phi-incoming materialization relocation), not a large merge-rewrite roster. Broad repair deletion
  is byte-changing and needs execution evidence, not only a reject census of zero.
- **No name-keyed translation.** Decide from IR structure, types, storage classes, and the AIR
  metadata ABI—not from shader/function names observed in particular workloads. Stable `air.*` /
  `llvm.*` ABI symbols are the allowed exception.
- **Admission is reject-triggered and spirv-val gated.** Do not invent structured exits from
  dominance alone as a primary admission path.
