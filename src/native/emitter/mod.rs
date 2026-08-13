use super::cfg::{
    block_index_by_label, funnel_shared_branch_dispatches, id_ref_operand,
    infer_bounded_branch_merges_by_header, infer_branch_merges, infer_direct_branch_merges,
    infer_direct_switch_merges, infer_loop_merges, infer_switch_merges,
    lower_unstructured_switches, refunnel_one_deep_shared_arm, LoopMergeInfo,
};
use super::ir::{
    LlDeclaration, LlFunction, LlGep, LlGlobal, LlModule, LlType, LlTypeCapability, LlValue,
    TypedValue,
};
use super::parse::{
    fcmp_predicate, float_compare_result_type, icmp_predicate, int_compare_result_type,
    is_ignored_call_line, matching_paren, parse_type, parse_typed_value, parse_value,
    split_top_level, split_top_level_whitespace, strip_call_prefix, switch_literal_operand, LlCall,
    LlLoad,
};
use crate::spirv_module::Operand;
use crate::spirv_module::{Block, Function, Instruction, Module, ModuleHeader};
use crate::types::TypeInterner;
use spirv::{
    AddressingModel, Capability, FunctionControl, GlslStd450Op, LoopControl, MemoryModel,
    MemorySemantics, Op, Scope, SelectionControl, SourceLanguage, StorageClass, Word,
};
use std::collections::{HashMap, HashSet};

mod air_struct_offsets;
mod body;
mod control;
mod functions;
mod helpers;
mod memory;
mod ops;
mod pointer_network;
mod pointers;
mod types;

#[cfg(test)]
mod layout_tests;

use helpers::*;

pub(super) struct Emitter {
    module: Module,
    ir: LlModule,
    emit_sidecar: crate::emit_sidecar::EmitSidecar,
    /// SPIR-V type/constant dedup caches (see `crate::types`). The builder methods that emit the
    /// instructions live on `Emitter`; the module owns result-id allocation.
    interner: TypeInterner,
    glsl_ext: Option<Word>,
    values: HashMap<String, (Word, LlType)>,
    global_values: HashMap<String, (Word, LlType)>,
    /// Module globals (kebab here: `@name`) that are the pointer operand of an integer atomic
    /// (`air.atomic.*.i32`) but are declared with a 32-bit-bitcastable *non*-integer scalar pointee
    /// (e.g. a `float` threadgroup scratch slot used for the atomic-min/max bit-pattern idiom).
    /// Under Logical addressing an integer atomic needs an `i32`-typed pointer to that exact memory,
    /// which — for a global — only exists if the variable itself is declared `i32`. Reinterpreting
    /// the float pointer would require an illegal logical-pointer `OpBitcast`, so `emit_global`
    /// declares these as `i32` instead, and the existing scalar value-reinterpret load/store paths
    /// (32-bit `OpBitcast` on the *value*) carry the float accesses. Computed structurally from the
    /// `air.atomic.*.i32` ABI symbol family before globals emit; excludes any global also accessed by
    /// a float atomic (`air.atomic.*.f32`), which would then need the opposite reinterpret.
    int_atomic_reinterpret_globals: HashSet<String>,
    /// Constant globals accessed through a GEP source type that differs from the declared type
    /// (a byte-table reinterpret view); when all-i8-leaf, declared as a flat `[N x i8]` so every
    /// view lowers through the byte-array raw paths. Populated before globals emit.
    byte_view_reinterpret_globals: HashSet<String>,
    /// Constant globals accessed through a GEP source type that differs from the declared type
    /// whose declared layout is a padding-free scalar image. These are declared as `[N x scalar]`
    /// and AIR aggregate GEP views linearize to scalar indices from that flat root.
    flat_scalar_reinterpret_globals: HashMap<String, LlType>,
    gep_provenance: HashMap<String, GepProvenance>,
    selected_pointers: HashMap<String, SelectedPointer>,
    selected_load_pointers: HashMap<String, SelectedLoadPointer>,
    vector_word_roots: HashMap<Word, VectorWordRoot>,
    vector_word_pointers: HashMap<String, VectorWordPointer>,
    local_pointer_fields: HashMap<LocalPointerField, TypedValue>,
    raw_memcpy_shadows: HashMap<String, Vec<(u64, Word)>>,
    imageblock_data_scratch: Option<(Word, LlType)>,
    dynamic_pointer_tables: HashMap<String, DynamicPointerTable>,
    forward_geps: HashMap<String, LlGep>,
    pointer_storage: HashMap<String, StorageClass>,
    pointer_pointees: HashMap<String, LlType>,
    local_alloca_pointees: HashMap<String, LlType>,
    pointer_nullness: HashMap<String, Word>,
    /// BDA-mode integer sources for `inttoptr` results in the current function. Collected from the
    /// typed graph before block emission so a loop-header address phi can reserve a backedge value
    /// whose `inttoptr` definition appears in the latch.
    bda_inttoptr_sources: HashMap<String, TypedValue>,
    /// Runtime 64-bit addresses for descriptor-rooted direct buffer parameters. In BDA mode these
    /// are materialized once in the entry block from the reflected buffer-address sidecar, then used
    /// as ordinary dominating integer values by address-domain pointer merges.
    bda_direct_addresses: HashMap<String, Word>,
    /// The two little-endian 32-bit words of a serialized 64-bit pointer loaded through a raw
    /// buffer view. Logical SPIR-V pointers cannot represent or compare this wire payload, so raw
    /// pointer equality and stores operate on these integer words instead.
    pointer_payload_words: HashMap<String, (Word, Word)>,
    /// Pointer SSA values used by a non-null equality comparison in the current function. Built from
    /// the typed def/use graph before emission so a raw load can retain its serialized payload even
    /// when dynamic offset alignment is known only from the load's explicit `align` contract.
    pointer_payload_values: HashSet<String>,
    pointer_phi_values: HashSet<String>,
    pointer_phi_incoming_values: HashSet<String>,
    tir_phi_incomings: HashMap<String, Vec<(LlValue, String)>>,
    function_param_pointees: HashMap<(String, usize), LlType>,
    function_param_nonnull: HashSet<(String, usize)>,
    /// Residual helper pointer parameters whose authored LLVM nullness is observed. Logical SPIR-V
    /// pointers cannot carry a null value portably, so calls append one Boolean shadow for exactly
    /// these parameters and the callee binds that shadow to its ordinary pointer SSA name.
    function_param_nullness: HashSet<(String, usize)>,
    direct_param_values: HashSet<String>,
    direct_param_indices: HashMap<String, u32>,
    param_values: HashSet<String>,
    inline_parameter_substitutions: Vec<(Word, Word)>,
    raw_buffer_params: HashSet<String>,
    /// Constant descriptor-relative byte cursors passed to raw helper parameters, keyed by
    /// `(callee, parameter)`. The emitted-helper inliner can then keep a non-zero caller GEP rooted
    /// in the real StorageBuffer instead of the ordinary Private placeholder value.
    raw_call_param_offsets: HashMap<(String, String), RawBufferOffset>,
    /// Current function's entry params declared `air.buffer` in kernel metadata (data pointers,
    /// never textures/samplers). Rebuilt per emit_function from `ir.metadata_data_buffer_params`.
    data_buffer_params: HashSet<String>,
    raw_offsets: HashMap<String, RawBufferOffset>,
    int_alignments: HashMap<String, u64>,
    unmodeled_pointers: HashSet<String>,
    /// AIR occasionally embeds a numeric Workgroup address directly in an atomic operand. Logical
    /// SPIR-V has no integer-to-Workgroup-pointer conversion, so preserve equality of identical
    /// numeric addresses with one module-scope atomic slot per address.
    workgroup_i32_addresses: HashMap<u64, Word>,
    /// SSA value names bound by `air.get_null_texture_*`. A strict subset of `unmodeled_pointers`
    /// (many other placeholders land there too), so `air.is_null_texture` keys on THIS set — a
    /// value synthesized as a null texture answers TRUE, everything else stays on the default path.
    null_texture_values: HashSet<String>,
    function_ids: HashMap<String, Word>,
    block_labels: HashMap<String, Word>,
    branch_merges: HashMap<(String, String), String>,
    branch_merges_by_header: HashMap<String, String>,
    /// The current function was planned by the construct tree, whose merge ownership is keyed by
    /// header rather than by a potentially shared target pair. This is per-function provenance:
    /// enabling the construct-tree retry for the module does not change how normally planned helper
    /// functions look up their merges.
    branch_merges_header_only: bool,
    loop_merges: HashMap<String, LoopMergeInfo>,
    switch_merges: HashMap<String, String>,
    current_block: Option<String>,
    /// R3 graph-driven emission: the current function's instruction operands resolved by the typed SSA
    /// IR (`tir`), keyed by result `%name`. Emission for the opcode shapes tir lowers (binary ops) reads
    /// its operands from here instead of re-lexing the instruction text; absent/`Unresolved` entries
    /// fall back to the string path. Rebuilt per function in `emit_function` (empty if
    /// `build_from_blocks` fails, so emission degrades to the string path — floor-safe).
    tir_operands: HashMap<String, Vec<crate::native::tir::TirOperand>>,
    /// R3 graph-driven emission: the typed SSA IR's resolved RESULT TYPE for each `%name`, keyed by
    /// result name (the dual of `tir_operands`, which carries operand types). Emitters that today
    /// re-lex a result/destination type from the instruction text (e.g. the `<conv> .. to <dstty>`
    /// destination type) read it from here instead, retiring that text parse. Byte-neutral: tir
    /// computes the same `parse_type(dst)` the text path does, so `resolve_type` of either is identical.
    /// Absent entries fall back to the string parse. Rebuilt per function in `emit_function`.
    tir_result_types: HashMap<String, crate::native::ir::LlType>,
    /// R3 graph-driven emission: the typed SSA IR's comparison PREDICATE token for each `icmp`/`fcmp`
    /// result (`eq`/`slt`/`oeq`/...), keyed by result name. The compare emitters read the predicate
    /// from here instead of re-lexing it from the instruction text, retiring that structural-literal
    /// parse. Byte-neutral: `icmp_predicate`/`fcmp_predicate` over this stored token yields the same
    /// `Op` the text path derives. Absent entries fall back to the string parse. Rebuilt per function in
    /// `emit_function`.
    tir_predicates: HashMap<String, String>,
    /// R3 graph-driven emission: the typed SSA IR's explicit memory ALIGNMENT (`align N`) for each
    /// `load` result, keyed by result name. The `load` emitter reads the alignment from here instead of
    /// re-lexing the trailing `, align N` field, retiring that structural-literal parse. Byte-neutral:
    /// tir computes it via the same `parse_memory_alignment` the text path uses. Absent entries fall
    /// back to the parsed `LlLoad.align`. In the graph walk `store` (result-LESS) sources its alignment
    /// straight from `inst.mem_align` at the dispatch site instead.
    tir_aligns: HashMap<String, Option<u64>>,
    /// R3 graph-driven emission: the typed SSA IR's `getelementptr` SOURCE element type for each gep
    /// result, keyed by result name. The `getelementptr` emitter builds its `LlGep` from this (plus the
    /// base/index operands in `tir_operands`) instead of re-parsing the line with `parse_gep`, retiring
    /// that emit-time re-lex on the R4-critical pointer path. Byte-identical: tir computes the source_ty
    /// via the same `parse_gep`. Absent entries fall back to the string parse. Rebuilt per function in
    /// `emit_function`.
    tir_gep_source_types: HashMap<String, crate::native::ir::LlType>,
    /// M1 (pointer-typing rewrite): the USE-based pointee carried on every pointer SSA value, keyed by
    /// result `%name`. Sourced once per function from the structurized typed-IR graph
    /// (`tir::build_from_blocks(..).use_pointees`, the SAME graph `tir_operands` is built from), it gives
    /// emission the pointee a pointer is actually *dereferenced* as — a `load`/`store` type, a GEP source
    /// element type, an atomic element type — propagated across `select`/`phi`/`freeze` pointer merges to
    /// a fixpoint (see [`crate::native::tir::TirFunction::use_pointees`]). This is the data carrier the
    /// whole-module pointer-typing rewrite needs to retire the ~42 name-keyed pointee/storage side-tables
    /// and the illegal pointer-`OpBitcast` reinterpret fallback: storage is already a pure function of
    /// address space (`pointer_storage_for`/`llvm_pointer_storage`), so this pointee half is what was
    /// missing at every pointer def. M2 (S20) first consumer: `pointer_pointee_for_value`
    /// (`pointers.rs`) reads it as a FALLBACK for a local pointer the def-time side-tables left without
    /// a pointee — the emitter-recorded pointee stays authoritative (checked first), so the carrier only
    /// ADDS answers where the emitter had `None`, never overrides. Floor-safe (G4 3 / G5 62). Later M2
    /// slices widen consumption to the divergence set behind the BC byte-drift gate / the frontier
    /// battery. Rebuilt per function in `emit_function`.
    tir_use_pointees: HashMap<String, crate::native::ir::LlType>,
    /// The pointers `tir_use_pointees` resolved that ALSO carry a byte (`i8`) view — a byte-cursor
    /// `getelementptr i8` / `i8` load-store / byte atomic (see
    /// [`crate::native::tir::TirFunction::byte_view_pointers`]). The M2 byte→real pointee upgrade in
    /// `pointer_pointee_for_value` must skip these: their carrier is wider than `i8`, but the emitter
    /// still emits their byte cursor as a `uchar`-result `OpPtrAccessChain`, so upgrading the pointee
    /// strands it (globally-invalid SPIR-V). Only the pure-widening subset (absent from this set) is
    /// safe to flip. Rebuilt per function in `emit_function`.
    tir_byte_view_pointers: std::collections::HashSet<String>,
    /// M-A2 def-site network-pointee recording: the uniform pointee to record at every def
    /// site of a pointer network (connected component over phi/select edges) whose tir-carrier
    /// granularity is consistent across the component. Consulted by `pointer_meta_for_value` for a
    /// network member so `pointer_merge_meta` reconciles on the carrier type instead of the byte-view
    /// `Int(8)` the raw recording flattens the byte-addressed arm to. Rebuilt per function in
    /// `emit_function`; empty (no effect) unless the flag is set.
    network_pointees: HashMap<String, crate::native::ir::LlType>,
    /// When true, `emit_function` attempts the R2 cross-arm restructure (full-closure tail duplication
    /// + return unification) for a function `structured_plan` rejects, taking the transformed blocks if
    ///   they then admit a structured plan. This produces a CANDIDATE that the caller (the
    ///   `inline_sroa_raw_cfg_restructure` retry tier) keeps ONLY if the whole module then passes
    ///   spirv-val — the adopt-if-VALIDATES discipline. The DEFAULT emit path leaves this false so a
    ///   reject emits its inferred merges unrepaired and the unsafe in-emit adopt-if-ADMITS can never
    ///   hijack an admitting case. Set via [`with_cfg_restructure`].
    cfg_restructure: bool,
    /// When true, `emit_function` attempts the reject-only construct-tree own-arm candidate for a
    /// function `structured_plan` rejects. The default emitter leaves this false; the retry cascade sets
    /// it only after the primary module fails CFG validation and adopts only a validating module.
    construct_tree: bool,
    /// When true, `emit_function` SKIPS the structured-plan attempt entirely — even for a function
    /// `structured_plan` WOULD admit — emitting the raw blocks without branch/loop inferred merge
    /// hints or structuring. Switches still need a merge target to encode `OpSwitch`; the W2 relooper
    /// strips that stale hint and rebuilds a structured CFG from scratch. (The DEFAULT path already
    /// emits a *reject* unstructured since the W4 repair-roster deletion; this flag forces the same for
    /// an admitting function, so a guaranteed-unstructured complete module always exists to reloop.)
    /// Set via [`with_relooper_feed`]; never on the production path (the caller adopts only the
    /// relooper's validating output).
    relooper_feed: bool,
    /// M1 storage-carrier measurement: when set, `emit_function` snapshots its final per-value
    /// `pointer_storage` map (the emitter's stateful storage derivation) into `storage_snapshots`,
    /// keyed by function name, so the validation harness can compare it against the from-tir
    /// derivation (`tir::derive_pointer_storage`). Off in production emission; set via
    /// [`emit_collecting_storage`].
    capture_storage: bool,
    storage_snapshots: Vec<(String, HashMap<String, StorageClass>)>,
    /// M2 pointee-carrier measurement (the pointer-typing rewrite, the pointee half). When set,
    /// `emit_function` snapshots its final per-value `pointer_pointees` map (the emitter's stateful
    /// ground-truth pointee derivation, populated across GEP/load/phi/select/bitcast/buffer sites)
    /// into `pointee_snapshots`, keyed by function name, so the validation harness can compare it
    /// against the from-tir `use_pointees` carrier (`tir::TirFunction::use_pointees`) — the reconciliation
    /// set an M2 consumer must settle before flipping a pointer def from the side-tables to the carrier.
    /// Off in production emission; set via [`emit_collecting_pointees`].
    capture_pointees: bool,
    pointee_snapshots: Vec<(String, HashMap<String, LlType>)>,
    /// When true, a device pointer (`addrspace(1)`) LOADED from a buffer word is modeled as its real
    /// 64-bit address (an `OpConvertUToPtr` PhysicalStorageBuffer64 pointer) instead of being dropped to
    /// a Private null placeholder — so the kernel can STORE it (a verbatim 8-byte copy) and DEREFERENCE
    /// it (address + struct/array offset). This is the honest lowering of the "BDA" frontier class
    /// (Apple BVH builders that load/store/deref a device pointer): byte-correct by construction (the
    /// stored bytes are the exact loaded address; the deref is `address + offset` with no tag-bit
    /// manipulation), valid SPIR-V under `buffer_device_address`. Off by default (the default Logical
    /// emit is byte-identical); set via [`with_bda_device_pointers`] for the adopt-if-validates BDA
    /// retry tier. See [`RawBufferOffset::device_addr_base`].
    bda_device_pointers: bool,
    /// Set true the first time a device-address (`device_addr_base`) leaf load/store is emitted, so
    /// [`emit`] flips the module to the `PhysicalStorageBuffer64` addressing model (+ `Int64` /
    /// `PhysicalStorageBufferAddresses` caps + the `SPV_KHR_physical_storage_buffer` extension). Only
    /// ever set in BDA mode.
    used_device_address: bool,
}

#[derive(Clone, Debug)]
struct GepProvenance {
    root: Word,
    addrspace: u32,
    source_ty: LlType,
    indices: Vec<TypedValue>,
    root_indices: Option<Vec<TypedValue>>,
    root_is_indexed_container: bool,
}

#[derive(Clone, Debug)]
struct SelectedPointer {
    cond: Word,
    true_value: LlValue,
    false_value: LlValue,
    ty: LlType,
}

#[derive(Clone, Debug)]
struct SelectedLoadPointer {
    cond: Word,
    true_ptr: Option<Word>,
    false_ptr: Option<Word>,
    true_storage: StorageClass,
    false_storage: StorageClass,
    pointee: LlType,
    true_raw: Option<RawBufferOffset>,
    false_raw: Option<RawBufferOffset>,
    /// The arms are already-typed pointers whose modeled pointee is unknown (e.g. a select between
    /// pointers into distinct buffers, which cannot be a legal Logical-SPIR-V `OpSelect`). The load
    /// drives the value type: load each arm with the load's result type and select the values.
    load_typed: bool,
}

#[derive(Clone, Debug)]
struct DynamicPointerTable {
    selector: Word,
    selector_bits: u32,
    entries: Vec<(u32, TypedValue)>,
}

#[derive(Clone, Debug)]
struct VectorWordRoot {
    storage: StorageClass,
    vector_ty: LlType,
    lanes: u32,
    lanes_per_word: u32,
    words_per_vector: u32,
    base_is_vector_pointer: bool,
}

#[derive(Clone, Debug)]
struct VectorWordPointer {
    base: Word,
    storage: StorageClass,
    vector_ty: LlType,
    lanes: u32,
    lanes_per_word: u32,
    words_per_vector: u32,
    base_is_vector_pointer: bool,
    word_index: Word,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct LocalPointerField {
    root: Word,
    indices: Vec<u32>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct PointerMeta {
    storage: StorageClass,
    pointee: Option<LlType>,
}

#[derive(Clone, Debug)]
struct RawBufferOffset {
    const_off: i64,
    dyn_terms: Vec<(TypedValue, i64)>,
    root: String,
    addrspace: u32,
    unmodelable: bool,
    /// When `Some(addr)`, this offset is NOT rooted at a descriptor-bound buffer but at a runtime
    /// 64-bit device ADDRESS value `addr` (a pointer the kernel loaded from device memory). The
    /// offset is then `addr + const_off + Σ(dyn_terms)` in bytes, and a leaf load/store through it is
    /// lowered with `OpConvertUToPtr` to a `PhysicalStorageBuffer` pointer (BDA mode only — see
    /// [`Emitter::bda_device_pointers`]). GEP folds offsets into `const_off`/`dyn_terms` exactly as for
    /// a descriptor-rooted offset, so the existing [`Emitter::apply_raw_gep`] machinery is reused. `None`
    /// for every descriptor-bound buffer offset (the default).
    device_addr_base: Option<Word>,
}

impl RawBufferOffset {
    fn root(root: String, addrspace: u32) -> Self {
        Self {
            const_off: 0,
            dyn_terms: vec![],
            root,
            addrspace,
            unmodelable: false,
            device_addr_base: None,
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum RawSubwordLane {
    Static(u32),
    Dynamic(Word),
}

impl Emitter {
    pub(super) fn require_capability(&mut self, capability: Capability) {
        if self.module.capabilities.iter().any(|inst| {
            matches!(
                inst.operands.as_slice(),
                [Operand::Capability(existing)] if *existing == capability
            )
        }) {
            return;
        }
        self.module.capabilities.push(Instruction::new(
            Op::Capability,
            None,
            None,
            vec![Operand::Capability(capability)],
        ));
    }

    pub(super) fn require_extension(&mut self, extension: &str) {
        if self.module.extensions.iter().any(|inst| {
            matches!(
                inst.operands.as_slice(),
                [Operand::LiteralString(existing)] if existing == extension
            )
        }) {
            return;
        }
        self.module.extensions.push(Instruction::new(
            Op::Extension,
            None,
            None,
            vec![Operand::LiteralString(extension.to_string())],
        ));
    }

    pub(super) fn new(mut ir: LlModule) -> Self {
        ir.inline_simple_static_initializers();
        ir.fold_static_initializer_constants();
        ir.inline_ordinary_leaf_helpers();
        let mut module = Module::new();
        module.capabilities.push(Instruction::new(
            Op::Capability,
            None,
            None,
            vec![Operand::Capability(Capability::Shader)],
        ));
        module.memory_model = Some(Instruction::new(
            Op::MemoryModel,
            None,
            None,
            vec![
                Operand::AddressingModel(AddressingModel::Logical),
                Operand::MemoryModel(MemoryModel::GLSL450),
            ],
        ));
        module.debug_string_source.push(Instruction::new(
            Op::Source,
            None,
            None,
            vec![
                Operand::SourceLanguage(SourceLanguage::Unknown),
                Operand::LiteralBit32(0),
            ],
        ));
        Self {
            module,
            ir,
            emit_sidecar: crate::emit_sidecar::EmitSidecar::default(),
            interner: TypeInterner::new(),
            glsl_ext: None,
            values: HashMap::new(),
            global_values: HashMap::new(),
            int_atomic_reinterpret_globals: HashSet::new(),
            byte_view_reinterpret_globals: HashSet::new(),
            flat_scalar_reinterpret_globals: HashMap::new(),
            gep_provenance: HashMap::new(),
            selected_pointers: HashMap::new(),
            selected_load_pointers: HashMap::new(),
            vector_word_roots: HashMap::new(),
            vector_word_pointers: HashMap::new(),
            local_pointer_fields: HashMap::new(),
            raw_memcpy_shadows: HashMap::new(),
            imageblock_data_scratch: None,
            dynamic_pointer_tables: HashMap::new(),
            forward_geps: HashMap::new(),
            pointer_storage: HashMap::new(),
            pointer_pointees: HashMap::new(),
            local_alloca_pointees: HashMap::new(),
            pointer_nullness: HashMap::new(),
            bda_inttoptr_sources: HashMap::new(),
            bda_direct_addresses: HashMap::new(),
            pointer_payload_words: HashMap::new(),
            pointer_payload_values: HashSet::new(),
            pointer_phi_values: HashSet::new(),
            pointer_phi_incoming_values: HashSet::new(),
            tir_phi_incomings: HashMap::new(),
            function_param_pointees: HashMap::new(),
            function_param_nonnull: HashSet::new(),
            function_param_nullness: HashSet::new(),
            direct_param_values: HashSet::new(),
            direct_param_indices: HashMap::new(),
            param_values: HashSet::new(),
            inline_parameter_substitutions: Vec::new(),
            raw_buffer_params: HashSet::new(),
            raw_call_param_offsets: HashMap::new(),
            data_buffer_params: HashSet::new(),
            raw_offsets: HashMap::new(),
            int_alignments: HashMap::new(),
            unmodeled_pointers: HashSet::new(),
            workgroup_i32_addresses: HashMap::new(),
            null_texture_values: HashSet::new(),
            function_ids: HashMap::new(),
            block_labels: HashMap::new(),
            branch_merges: HashMap::new(),
            branch_merges_by_header: HashMap::new(),
            branch_merges_header_only: false,
            loop_merges: HashMap::new(),
            switch_merges: HashMap::new(),
            current_block: None,
            tir_operands: HashMap::new(),
            tir_result_types: HashMap::new(),
            tir_predicates: HashMap::new(),
            tir_aligns: HashMap::new(),
            tir_gep_source_types: HashMap::new(),
            tir_use_pointees: HashMap::new(),
            network_pointees: HashMap::new(),
            tir_byte_view_pointers: std::collections::HashSet::new(),
            cfg_restructure: false,
            construct_tree: false,
            relooper_feed: false,
            capture_storage: false,
            storage_snapshots: Vec::new(),
            capture_pointees: false,
            pointee_snapshots: Vec::new(),
            bda_device_pointers: false,
            used_device_address: false,
        }
    }

    /// Enable BDA device-pointer modeling (see [`Self::bda_device_pointers`]). The caller marks device
    /// buffers raw and adopts the result only if it independently passes spirv-val, so the default
    /// Logical emission is never altered.
    pub(super) fn with_bda_device_pointers(mut self) -> Self {
        self.bda_device_pointers = true;
        self
    }

    /// Enable the R2 cross-arm restructure retry for this emitter (see [`Self::cfg_restructure`]). Used
    /// by the `inline_sroa_raw_cfg_restructure` emit variant the translate pipeline drives as an
    /// adopt-if-validates fallback; the default emitter leaves it off.
    pub(super) fn with_cfg_restructure(mut self) -> Self {
        self.cfg_restructure = true;
        self
    }

    /// Enable the construct-tree own-arm retry for this emitter. Used only by an adopt-if-validates
    /// retry tier; the default emitter leaves it off.
    pub(super) fn with_construct_tree(mut self) -> Self {
        self.construct_tree = true;
        self
    }

    /// Enable the "emit unstructured for the relooper" path (see [`Self::relooper_feed`]). The caller
    /// (`raw_reemit_relooper_feed`, the last raw fallback of `raw_then_relooper`) feeds the result
    /// straight to the W2 relooper and adopts only its validating output, so the unstructured
    /// intermediate is never shipped.
    pub(super) fn with_relooper_feed(mut self) -> Self {
        self.relooper_feed = true;
        self
    }

    pub(super) fn glsl_ext_inst_import(&mut self) -> Word {
        if let Some(id) = self.glsl_ext {
            return id;
        }
        for inst in &self.module.ext_inst_imports {
            if let Some(Operand::LiteralString(s)) = inst.operands.first() {
                if s == "GLSL.std.450" {
                    let id = inst.result_id.expect("GLSL import result id");
                    self.glsl_ext = Some(id);
                    return id;
                }
            }
        }
        let id = self.fresh();
        self.module.ext_inst_imports.push(Instruction::new(
            Op::ExtInstImport,
            None,
            Some(id),
            vec![Operand::LiteralString("GLSL.std.450".into())],
        ));
        self.glsl_ext = Some(id);
        id
    }

    fn infer_function_param_pointees(&mut self, functions: &[LlFunction]) -> Result<(), String> {
        let functions_by_name: HashMap<String, &LlFunction> = functions
            .iter()
            .map(|function| (function.name.clone(), function))
            .collect();
        let mut inferred = HashMap::new();
        let mut ambiguous = HashSet::new();
        for function in functions {
            let local_pointees = self.local_call_arg_pointees(function)?;
            for inst in function.carrier_insts() {
                let Some(call_result) = &inst.emit_scan_call else {
                    continue;
                };
                let call = call_result.as_ref().as_ref().map_err(|e| e.clone())?;
                let Some(callee) = functions_by_name.get(&call.callee) else {
                    continue;
                };
                for (index, arg) in call.args.iter().enumerate() {
                    if index >= callee.params.len() {
                        break;
                    }
                    if !matches!(self.resolve_type(&callee.params[index].1)?, LlType::Ptr(_)) {
                        continue;
                    }
                    let Some(pointee) = self.call_arg_pointee(&arg.value, &local_pointees) else {
                        continue;
                    };
                    let callee_param = &callee.params[index].0;
                    let callee_param_ty = &callee.params[index].1;
                    if !self.callee_param_accepts_call_pointee(
                        &callee.name,
                        callee_param,
                        callee_param_ty,
                        &pointee,
                    ) {
                        continue;
                    }
                    let key = (callee.name.clone(), index);
                    if ambiguous.contains(&key) {
                        continue;
                    }
                    if let Some(existing) = inferred.get(&key) {
                        if !types_compatible(existing, &pointee) {
                            inferred.remove(&key);
                            ambiguous.insert(key);
                        }
                    } else {
                        inferred.insert(key, pointee);
                    }
                }
            }
        }
        self.function_param_pointees = inferred;
        Ok(())
    }

    pub(super) fn infer_function_param_nonnull(
        &self,
        functions: &[LlFunction],
    ) -> Result<HashSet<(String, usize)>, String> {
        let functions_by_name: HashMap<String, &LlFunction> = functions
            .iter()
            .map(|function| (function.name.clone(), function))
            .collect();
        let mut all_nonnull = HashSet::new();
        let mut seen = HashSet::new();
        let mut rejected = HashSet::new();
        for function in functions {
            let local_nonnull = self.local_call_arg_nonnull_values(function)?;
            for inst in function.carrier_insts() {
                let Some(call_result) = &inst.emit_scan_call else {
                    continue;
                };
                let call = call_result.as_ref().as_ref().map_err(|e| e.clone())?;
                let Some(callee) = functions_by_name.get(&call.callee) else {
                    continue;
                };
                for (index, arg) in call.args.iter().enumerate() {
                    if index >= callee.params.len() {
                        break;
                    }
                    if !matches!(self.resolve_type(&callee.params[index].1)?, LlType::Ptr(_)) {
                        continue;
                    }
                    let key = (callee.name.clone(), index);
                    if rejected.contains(&key) {
                        continue;
                    }
                    seen.insert(key.clone());
                    if self.call_arg_known_nonnull(&arg.value, &local_nonnull) {
                        all_nonnull.insert(key);
                    } else {
                        all_nonnull.remove(&key);
                        rejected.insert(key);
                    }
                }
            }
        }
        all_nonnull.retain(|key| seen.contains(key) && !rejected.contains(key));
        Ok(all_nonnull)
    }

    pub(super) fn infer_function_param_nullness(
        &self,
        functions: &[LlFunction],
    ) -> Result<HashSet<(String, usize)>, String> {
        let function_param_nonnull = self.infer_function_param_nonnull(functions)?;
        let mut required = HashSet::new();
        for function in functions {
            for inst in function.carrier_insts() {
                if !matches!(inst.cmp_predicate.as_deref(), Some("eq" | "ne")) {
                    continue;
                }
                let operands = inst
                    .operands
                    .iter()
                    .map(crate::native::tir::TirOperand::as_typed_value)
                    .collect::<Option<Vec<_>>>();
                let Some(operands) = operands else {
                    continue;
                };
                let [lhs, rhs] = operands.as_slice() else {
                    continue;
                };
                if !matches!(self.resolve_type(&lhs.ty)?, LlType::Ptr(_)) {
                    continue;
                }
                let observed = match (&lhs.value, &rhs.value) {
                    (LlValue::Zero, LlValue::Local(name))
                    | (LlValue::Local(name), LlValue::Zero) => name,
                    _ => continue,
                };
                if let Some(index) = function
                    .params
                    .iter()
                    .position(|(name, _)| name == observed)
                {
                    required.insert((function.name.clone(), index));
                }
            }
        }

        // A helper may forward one of its parameters directly to another helper that observes
        // nullness. Propagate that ABI requirement back through direct parameter-to-parameter calls;
        // ordinary SSA aliases carry their already-recorded Boolean at emission time.
        let mut changed = true;
        while changed {
            changed = false;
            for function in functions {
                for inst in function.carrier_insts() {
                    let Some(call) = inst.call.as_ref() else {
                        continue;
                    };
                    for (callee_index, argument) in call.args.iter().enumerate() {
                        if !required.contains(&(call.callee.clone(), callee_index)) {
                            continue;
                        }
                        let LlValue::Local(argument_name) = &argument.value else {
                            continue;
                        };
                        let Some(caller_index) = function
                            .params
                            .iter()
                            .position(|(name, _)| name == argument_name)
                        else {
                            continue;
                        };
                        changed |= required.insert((function.name.clone(), caller_index));
                    }
                }
            }
        }
        required.retain(|key| {
            !self.ir.entry_functions.contains(&key.0) && !function_param_nonnull.contains(key)
        });
        Ok(required)
    }

    fn local_call_arg_nonnull_values(
        &self,
        function: &LlFunction,
    ) -> Result<HashSet<String>, String> {
        let mut nonnull = HashSet::new();
        if self.ir.entry_functions.contains(&function.name) {
            for (param, ty) in &function.params {
                if matches!(self.resolve_type(ty)?, LlType::Ptr(_)) {
                    nonnull.insert(param.clone());
                }
            }
        }
        for inst in function.carrier_insts() {
            if inst.opcode == "alloca" {
                if let Some(name) = &inst.result {
                    nonnull.insert(name.clone());
                }
            }
        }

        let mut changed = true;
        while changed {
            changed = false;
            for inst in function.carrier_insts() {
                let Some(name) = &inst.result else {
                    continue;
                };
                if nonnull.contains(name) {
                    continue;
                }
                // `bitcast` carries the parsed src + dst TEXT (`convert_dst_type` stays emit-time); re-parse
                // the dst text to keep the reader's `parse_type(dst_text)? -> resolve_type?` propagation.
                if let Some((src, dst_text)) = inst.bitcast.as_deref() {
                    if matches!(self.resolve_type(&parse_type(dst_text)?)?, LlType::Ptr(_))
                        && self.call_arg_known_nonnull(&src.value, &nonnull)
                    {
                        nonnull.insert(name.clone());
                        changed = true;
                    }
                    continue;
                }
                if let Some(gep) = &inst.gep {
                    if self.call_arg_known_nonnull(&gep.base.value, &nonnull)
                        || self.inbounds_gep_has_nonzero_constant_offset(gep)?
                    {
                        nonnull.insert(name.clone());
                        changed = true;
                    }
                }
            }
        }
        Ok(nonnull)
    }

    fn inbounds_gep_has_nonzero_constant_offset(&self, gep: &LlGep) -> Result<bool, String> {
        if !gep.inbounds {
            return Ok(false);
        }
        let Some(first) = const_index(gep.indices.first()) else {
            return Ok(false);
        };
        let source_ty = self.resolve_type(&gep.source_ty)?;
        if first != 0 && self.raw_type_size_align(&source_ty)?.0 != 0 {
            return Ok(true);
        }
        Ok(self
            .constant_aggregate_gep_offset(&source_ty, &gep.indices[1..])?
            .is_some_and(|(offset, _)| offset != 0))
    }

    fn call_arg_known_nonnull(&self, value: &LlValue, local_nonnull: &HashSet<String>) -> bool {
        match value {
            LlValue::Local(name) => local_nonnull.contains(name),
            LlValue::Global(_) => true,
            LlValue::Gep(gep) => self.call_arg_known_nonnull(&gep.base.value, local_nonnull),
            _ => false,
        }
    }

    fn local_call_arg_pointees(
        &self,
        function: &LlFunction,
    ) -> Result<HashMap<String, LlType>, String> {
        let mut pointees = HashMap::new();
        for (param, ty) in &function.params {
            if !matches!(ty, LlType::Ptr(_)) {
                continue;
            }
            if let Some(pointee) = self
                .ir
                .ptr_pointees
                .get(&(function.name.clone(), param.clone()))
                .cloned()
            {
                pointees.insert(param.clone(), pointee);
            }
        }

        let mut changed = true;
        while changed {
            changed = false;
            for inst in function.carrier_insts() {
                let Some(name) = &inst.result else {
                    continue;
                };
                if pointees.contains_key(name) {
                    continue;
                }
                if let Some((src, dst_text)) = inst.bitcast.as_deref() {
                    let dst_ty = self.resolve_type(&parse_type(dst_text)?)?;
                    if matches!(dst_ty, LlType::Ptr(_)) {
                        if let LlValue::Local(src_name) = &src.value {
                            if let Some(pointee) = pointees.get(src_name).cloned() {
                                pointees.insert(name.clone(), pointee);
                                changed = true;
                            }
                        }
                    }
                    continue;
                }
                if let Some(gep) = &inst.gep {
                    let LlValue::Local(base_name) = &gep.base.value else {
                        continue;
                    };
                    if pointees.contains_key(base_name) {
                        let pointee =
                            gep_pointee(&self.resolve_type(&gep.source_ty)?, &gep.indices)?;
                        pointees.insert(name.clone(), pointee);
                        changed = true;
                    }
                }
            }
        }
        Ok(pointees)
    }

    fn call_arg_pointee(
        &self,
        value: &LlValue,
        local_pointees: &HashMap<String, LlType>,
    ) -> Option<LlType> {
        match value {
            LlValue::Local(name) => local_pointees.get(name).cloned(),
            LlValue::Global(name) => self.pointer_pointees.get(name).cloned(),
            _ => None,
        }
    }

    fn callee_param_accepts_call_pointee(
        &self,
        callee_name: &str,
        callee_param: &str,
        callee_param_ty: &LlType,
        call_pointee: &LlType,
    ) -> bool {
        if matches!(callee_param_ty, LlType::Ptr(3)) {
            return true;
        }
        self.ir
            .ptr_pointees
            .get(&(callee_name.to_string(), callee_param.to_string()))
            .is_none_or(|local_pointee| types_compatible(local_pointee, call_pointee))
    }

    /// Drop body-less `OpFunction`s (emitted from `declare`s) that no `OpFunctionCall` references.
    /// An inlined intrinsic (e.g. `air.rhadd.u.i16`) leaves its declaration behind as an empty
    /// function shell, which is invalid SPIR-V; removing the *unreferenced* ones is strictly a
    /// correctness improvement (genuinely-called declarations are kept untouched).
    fn remove_dead_empty_functions(&mut self) {
        let mut called: HashSet<Word> = HashSet::new();
        for f in &self.module.functions {
            for b in &f.blocks {
                for inst in &b.instructions {
                    if inst.class.opcode == Op::FunctionCall {
                        if let Some(Operand::IdRef(callee)) = inst.operands.first() {
                            called.insert(*callee);
                        }
                    }
                }
            }
        }
        let mut removed: HashSet<Word> = HashSet::new();
        self.module.functions.retain(|f| {
            if !f.blocks.is_empty() {
                return true;
            }
            match f.def.as_ref().and_then(|d| d.result_id) {
                Some(id) if !called.contains(&id) => {
                    removed.insert(id);
                    false
                }
                _ => true,
            }
        });
        if !removed.is_empty() {
            self.module.debug_names.retain(|inst| {
                !(inst.class.opcode == Op::Name
                    && matches!(inst.operands.first(), Some(Operand::IdRef(id)) if removed.contains(id)))
            });
        }
    }

    pub(super) fn emit_with_sidecar(
        mut self,
        buffer_layouts: Option<&HashMap<u32, crate::meta::AirType>>,
    ) -> Result<(Module, crate::emit_sidecar::EmitSidecar), String> {
        self.emit_inner()?;
        self.record_air_struct_offsets(buffer_layouts);
        if self.used_device_address {
            self.lower_bda_null_aggregate_pointers()?;
            self.switch_to_physical_storage_buffer64();
        }
        Ok((self.module, self.emit_sidecar))
    }

    /// Flip the module to the `PhysicalStorageBuffer64` addressing model and declare the capabilities +
    /// extension a BDA (`OpConvertUToPtr` → `PhysicalStorageBuffer` pointer) module requires. Called from
    /// [`emit_with_sidecar`](Self::emit_with_sidecar) only when at least one device-address leaf was
    /// emitted (BDA mode). Mirrors the equivalent block in the `native::psb` byte-rewrite pass.
    fn switch_to_physical_storage_buffer64(&mut self) {
        if let Some(mm) = self.module.memory_model.as_mut() {
            if let Some(op) = mm.operands.get_mut(0) {
                *op = Operand::AddressingModel(AddressingModel::PhysicalStorageBuffer64);
            }
        }
        self.require_capability(Capability::PhysicalStorageBufferAddresses);
        self.require_capability(Capability::Int64);
        self.require_extension("SPV_KHR_physical_storage_buffer");
    }

    /// PhysicalStorageBuffer64 does not permit logical pointers as aggregate values. Callback-free
    /// AIR intersection lowering can leave one such value behind as an opaque result field that is
    /// structurally always null. Re-type only fields reached exclusively by a null-pointer
    /// `OpCompositeInsert` to the integer address representation used by the BDA path. Any other use
    /// leaves the pointer untouched, so an unsupported live logical-pointer aggregate still fails
    /// validation rather than acquiring invented semantics.
    fn lower_bda_null_aggregate_pointers(&mut self) -> Result<(), String> {
        let null_pointers =
            self.module
                .types_global_values
                .iter()
                .filter_map(|inst| {
                    (inst.class.opcode == Op::ConstantNull)
                        .then_some((inst.result_id?, inst.result_type?))
                })
                .filter(|(_, ty)| {
                    self.module.types_global_values.iter().any(|inst| {
                        inst.class.opcode == Op::TypePointer && inst.result_id == Some(*ty)
                    })
                })
                .collect::<HashMap<_, _>>();
        if null_pointers.is_empty() {
            return Ok(());
        }

        let type_defs = self
            .module
            .types_global_values
            .iter()
            .filter_map(|inst| inst.result_id.map(|id| (id, inst.clone())))
            .collect::<HashMap<_, _>>();
        let mut fields_by_null: HashMap<Word, Vec<(Word, usize)>> = HashMap::new();
        let mut unsupported_use = HashSet::new();
        for function in &self.module.functions {
            for inst in function
                .blocks
                .iter()
                .flat_map(|block| block.instructions.iter())
            {
                for (operand_index, operand) in inst.operands.iter().enumerate() {
                    let Operand::IdRef(id) = operand else {
                        continue;
                    };
                    let Some(pointer_ty) = null_pointers.get(id).copied() else {
                        continue;
                    };
                    if inst.class.opcode != Op::CompositeInsert || operand_index != 0 {
                        unsupported_use.insert(*id);
                        continue;
                    }
                    let Some(mut aggregate_ty) = inst.result_type else {
                        unsupported_use.insert(*id);
                        continue;
                    };
                    let mut target = None;
                    for index in inst.operands.iter().skip(2) {
                        let Operand::LiteralBit32(member) = index else {
                            target = None;
                            break;
                        };
                        let Some(def) = type_defs.get(&aggregate_ty) else {
                            target = None;
                            break;
                        };
                        let Some(Operand::IdRef(member_ty)) = def.operands.get(*member as usize)
                        else {
                            target = None;
                            break;
                        };
                        target = Some((aggregate_ty, *member as usize, *member_ty));
                        aggregate_ty = *member_ty;
                    }
                    match target {
                        Some((owner, member, leaf_ty)) if leaf_ty == pointer_ty => {
                            fields_by_null.entry(*id).or_default().push((owner, member));
                        }
                        _ => {
                            unsupported_use.insert(*id);
                        }
                    }
                }
            }
        }
        fields_by_null.retain(|id, fields| !fields.is_empty() && !unsupported_use.contains(id));
        if fields_by_null.is_empty() {
            return Ok(());
        }

        let address_ty = self.type_id(&LlType::Int(64))?;
        let zero = self.const_signed_int(64, 0)?;
        let lowered_nulls = fields_by_null.keys().copied().collect::<HashSet<_>>();
        let mut address_values = lowered_nulls.clone();
        let mut lowered_fields = fields_by_null
            .into_values()
            .flatten()
            .collect::<HashSet<_>>();
        let result_types = self
            .module
            .types_global_values
            .iter()
            .chain(self.module.functions.iter().flat_map(|function| {
                function.parameters.iter().chain(
                    function
                        .blocks
                        .iter()
                        .flat_map(|block| block.instructions.iter()),
                )
            }))
            .filter_map(|inst| Some((inst.result_id?, inst.result_type?)))
            .collect::<HashMap<_, _>>();
        loop {
            let mut changed = false;
            for function in &self.module.functions {
                for inst in function
                    .blocks
                    .iter()
                    .flat_map(|block| block.instructions.iter())
                {
                    match inst.class.opcode {
                        Op::CompositeExtract => {
                            let Some(Operand::IdRef(composite)) = inst.operands.first() else {
                                continue;
                            };
                            let Some(mut aggregate_ty) = result_types.get(composite).copied() else {
                                continue;
                            };
                            let mut target = None;
                            for index in inst.operands.iter().skip(1) {
                                let Operand::LiteralBit32(member) = index else {
                                    target = None;
                                    break;
                                };
                                let Some(def) = type_defs.get(&aggregate_ty) else {
                                    target = None;
                                    break;
                                };
                                let Some(Operand::IdRef(member_ty)) =
                                    def.operands.get(*member as usize)
                                else {
                                    target = None;
                                    break;
                                };
                                target = Some((aggregate_ty, *member as usize));
                                aggregate_ty = *member_ty;
                            }
                            if target.is_some_and(|field| lowered_fields.contains(&field)) {
                                if let Some(result) = inst.result_id {
                                    changed |= address_values.insert(result);
                                }
                            }
                        }
                        Op::CompositeInsert => {
                            let Some(Operand::IdRef(object)) = inst.operands.first() else {
                                continue;
                            };
                            if !address_values.contains(object) {
                                continue;
                            }
                            let Some(mut aggregate_ty) = inst.result_type else {
                                continue;
                            };
                            let mut target = None;
                            for index in inst.operands.iter().skip(2) {
                                let Operand::LiteralBit32(member) = index else {
                                    target = None;
                                    break;
                                };
                                let Some(def) = type_defs.get(&aggregate_ty) else {
                                    target = None;
                                    break;
                                };
                                let Some(Operand::IdRef(member_ty)) =
                                    def.operands.get(*member as usize)
                                else {
                                    target = None;
                                    break;
                                };
                                target = Some((aggregate_ty, *member as usize));
                                aggregate_ty = *member_ty;
                            }
                            if let Some(field) = target {
                                changed |= lowered_fields.insert(field);
                            }
                        }
                        Op::CopyObject
                            if inst.operands.iter().any(|operand| {
                                matches!(operand, Operand::IdRef(id) if address_values.contains(id))
                            }) =>
                        {
                            if let Some(result) = inst.result_id {
                                changed |= address_values.insert(result);
                            }
                        }
                        _ => {}
                    }
                }
            }
            if !changed {
                break;
            }
        }
        for inst in &mut self.module.types_global_values {
            let Some(owner) = inst.result_id else {
                continue;
            };
            for (member, operand) in inst.operands.iter_mut().enumerate() {
                if lowered_fields.contains(&(owner, member)) {
                    *operand = Operand::IdRef(address_ty);
                }
            }
        }
        for function in &mut self.module.functions {
            for inst in function
                .blocks
                .iter_mut()
                .flat_map(|block| block.instructions.iter_mut())
            {
                if inst
                    .result_id
                    .is_some_and(|id| address_values.contains(&id))
                {
                    inst.result_type = Some(address_ty);
                }
                for operand in &mut inst.operands {
                    if matches!(operand, Operand::IdRef(id) if lowered_nulls.contains(id)) {
                        *operand = Operand::IdRef(zero);
                    }
                }
            }
        }
        Ok(())
    }

    /// Run the full emission for its side effect of populating `storage_snapshots` (one per function's
    /// final `pointer_storage`), and return them — the M1 storage-carrier measurement entry. Discards
    /// the module. Mirrors [`emit`] exactly except for the capture flag, so the snapshots reflect the
    /// production storage derivation.
    pub(super) fn emit_collecting_storage(
        mut self,
    ) -> Result<Vec<(String, HashMap<String, StorageClass>)>, String> {
        self.capture_storage = true;
        self.emit_inner()?;
        Ok(self.storage_snapshots)
    }

    /// Run the full emission for its side effect of populating `pointee_snapshots` (one per function's
    /// final `pointer_pointees`) and return them — the M2 pointee-carrier measurement entry. Discards
    /// the module. Mirrors [`emit`] exactly except for the capture flag, so the snapshots reflect the
    /// production pointee derivation.
    pub(super) fn emit_collecting_pointees(
        mut self,
    ) -> Result<Vec<(String, HashMap<String, LlType>)>, String> {
        self.capture_pointees = true;
        self.emit_inner()?;
        Ok(self.pointee_snapshots)
    }

    /// Run the full emission for its side effect of populating `function_param_pointees` (the
    /// call-site-inferred `(function, param-index) -> pointee` map — the S18 advisory sidecar's
    /// pointer-pointee facts) and return it. Discards the module. Mirrors [`emit`] exactly except
    /// that it hands back the sidecar instead of the bytes, so it reflects the production inference.
    pub(super) fn emit_collecting_param_pointees(
        mut self,
    ) -> Result<HashMap<(String, usize), LlType>, String> {
        self.emit_inner()?;
        Ok(self.function_param_pointees)
    }

    fn emit_inner(&mut self) -> Result<(), String> {
        let globals = self.ir.globals.clone();
        let functions = self.ir.functions.clone();
        let declarations = self.ir.declarations.clone();
        self.int_atomic_reinterpret_globals =
            Self::scan_int_atomic_reinterpret_globals(&globals, &functions);
        self.byte_view_reinterpret_globals = self.scan_byte_view_reinterpret_globals()?;
        self.flat_scalar_reinterpret_globals = self.scan_flat_scalar_reinterpret_globals()?;
        for global in &globals {
            self.emit_global(global)?;
        }
        for f in &functions {
            let id = self.fresh();
            self.function_ids.insert(f.name.clone(), id);
        }
        for decl in &declarations {
            let id = self.fresh();
            self.function_ids.insert(decl.name.clone(), id);
        }
        for decl in &declarations {
            self.emit_declaration(decl)?;
        }
        self.infer_function_param_pointees(&functions)?;
        self.function_param_nonnull = self.infer_function_param_nonnull(&functions)?;
        self.function_param_nullness = self.infer_function_param_nullness(&functions)?;
        for f in &functions {
            self.emit_function_with_raw_retry(f)?;
        }
        // The old residual inliner removed migrated helper bodies and dead types after emission,
        // but left capabilities requested while materializing those types. Replay only declarations
        // still missing after surviving functions emit: `require_capability` preserves the exact
        // order of every capability the live module already requested.
        let mut retained_type_capabilities = self
            .ir
            .preinlined_helper_type_capabilities
            .iter()
            .copied()
            .collect::<Vec<_>>();
        retained_type_capabilities.sort_unstable();
        for capability in retained_type_capabilities {
            self.require_capability(match capability {
                LlTypeCapability::Float16 => Capability::Float16,
                LlTypeCapability::Int8 => Capability::Int8,
                LlTypeCapability::Int16 => Capability::Int16,
                LlTypeCapability::Int64 => Capability::Int64,
            });
        }
        self.rewrite_scalar_pointer_arithmetic_access_chains();
        self.remove_dead_empty_functions();
        self.inject_static_initializer_calls(&functions)?;
        let mut header = ModuleHeader::new(self.module.id_bound());
        // Match the SPIR-V version LLVM's Vulkan backend emitted for this pipeline.
        header.set_version(1, 4);
        self.module.header = Some(header);
        let module = std::mem::take(&mut self.module);
        let emit_sidecar = std::mem::take(&mut self.emit_sidecar);
        (self.module, self.emit_sidecar) = crate::passes::inline_all_emitted_helpers(
            module,
            emit_sidecar,
            self.ir.entry_name.as_deref(),
        )?;
        self.rewrite_private_scalar_offset_access_chains();
        self.repair_truncated_access_chain_indices();
        Ok(())
    }

    /// Emit a function, and if it fails specifically because the logical pointer model cannot
    /// represent a buffer access, mark the function's buffer params raw and retry. Two trigger classes,
    /// both raised ONLY when a buffer access genuinely cannot be lowered logically:
    /// - **pointer-merge** (`pointer merge pointee mismatch`): a `select`/`phi` merges pointers with
    ///   incompatible pointees (the aggregate-vs-element reinterpret) — the dominant pointer-merge
    ///   frontier class.
    /// - **integer atomic on a float-typed buffer** (`atomic i32 pointer targets ...`): the
    ///   atomic-float-min/max idiom (`air.atomic.global.{min,max}.s.i32` on a `<3 x float>*` bounding
    ///   box, reinterpreting the float bits as a signed int) — `atomic_i32_pointer_id` cannot form an
    ///   `i32*` from a float/vector pointee under Logical addressing, but the raw path (`raw_offsets` →
    ///   `emit_raw_word_pointer_for_access`) lowers it as a uint-word atomic on the `RuntimeArray<uint>`
    ///   backing.
    ///
    /// This is a *precise* raw trigger: it fires ONLY on a genuine lowering failure, so a function that
    /// already translates (and therefore can be banked) never reaches it and is never marked raw — the
    /// over-marking that regressed earlier whole-buffer attempts is impossible by construction. A failed
    /// first attempt pushes only well-formed, unused type/constant globals (the function's blocks are
    /// not appended to the module until it fully succeeds), so the retry is safe without snapshotting
    /// module state; `emit_function` rebuilds `raw_buffer_params` from `self.ir.raw_buffer_params` on
    /// each call, so the retry observes the new marking.
    fn emit_function_with_raw_retry(&mut self, f: &LlFunction) -> Result<(), String> {
        match self.emit_function(f) {
            Err(e)
                if e.contains("pointer merge pointee mismatch")
                    || e.contains("atomic i32 pointer targets") =>
            {
                if self.mark_function_buffers_raw(f) {
                    self.emit_function(f)
                } else {
                    Err(e)
                }
            }
            other => other,
        }
    }

    /// Mark every device/constant buffer pointer param of `f` raw in `self.ir.raw_buffer_params`.
    /// Returns whether any param was newly marked (so a retry that would change nothing is skipped).
    fn mark_function_buffers_raw(&mut self, f: &LlFunction) -> bool {
        let mut newly_marked = false;
        for (name, ty) in &f.params {
            if matches!(ty, LlType::Ptr(1 | 2)) {
                let key = (f.name.clone(), name.clone());
                if self.ir.raw_buffer_params.insert(key) {
                    newly_marked = true;
                }
            }
        }
        newly_marked
    }

    fn rewrite_scalar_pointer_arithmetic_access_chains(&mut self) {
        let mut pointer_storage = HashMap::new();
        let aggregate_types = self
            .module
            .types_global_values
            .iter()
            .filter(|inst| {
                matches!(
                    inst.class.opcode,
                    Op::TypeStruct | Op::TypeArray | Op::TypeRuntimeArray | Op::TypeMatrix
                )
            })
            .filter_map(|inst| inst.result_id)
            .collect::<HashSet<_>>();
        let mut pointer_pointees = HashMap::new();
        for inst in &self.module.types_global_values {
            if inst.class.opcode == Op::TypePointer {
                if let (
                    Some(result),
                    Some(Operand::StorageClass(storage)),
                    Some(Operand::IdRef(pointee)),
                ) = (inst.result_id, inst.operands.first(), inst.operands.get(1))
                {
                    pointer_storage.insert(result, *storage);
                    pointer_pointees.insert(result, *pointee);
                }
            }
        }

        let mut id_types = HashMap::new();
        for inst in &self.module.types_global_values {
            if let (Some(result), Some(result_type)) = (inst.result_id, inst.result_type) {
                id_types.insert(result, result_type);
            }
        }
        for function in &self.module.functions {
            for inst in &function.parameters {
                if let (Some(result), Some(result_type)) = (inst.result_id, inst.result_type) {
                    id_types.insert(result, result_type);
                }
            }
            for block in &function.blocks {
                if let Some(label) = &block.label {
                    if let (Some(result), Some(result_type)) = (label.result_id, label.result_type)
                    {
                        id_types.insert(result, result_type);
                    }
                }
                for inst in &block.instructions {
                    if let (Some(result), Some(result_type)) = (inst.result_id, inst.result_type) {
                        id_types.insert(result, result_type);
                    }
                }
            }
        }

        for function in &mut self.module.functions {
            for block in &mut function.blocks {
                for inst in &mut block.instructions {
                    if inst.class.opcode != Op::InBoundsAccessChain {
                        continue;
                    }
                    let Some(result_type) = inst.result_type else {
                        continue;
                    };
                    if !pointer_storage
                        .get(&result_type)
                        .is_some_and(|storage| ptr_access_chain_allowed_storage(*storage))
                    {
                        continue;
                    }
                    if pointer_pointees
                        .get(&result_type)
                        .is_some_and(|pointee| aggregate_types.contains(pointee))
                    {
                        continue;
                    }
                    let Some(Operand::IdRef(base)) = inst.operands.first() else {
                        continue;
                    };
                    if id_types.get(base) != Some(&result_type) {
                        continue;
                    }
                    *inst = Self::inst(
                        Op::PtrAccessChain,
                        inst.result_type,
                        inst.result_id,
                        inst.operands.clone(),
                    );
                }
            }
        }
    }

    fn rewrite_private_scalar_offset_access_chains(&mut self) {
        let mut defs = HashMap::new();
        for inst in &self.module.types_global_values {
            if let Some(id) = inst.result_id {
                defs.insert(id, inst.clone());
            }
        }
        for function in &self.module.functions {
            for param in &function.parameters {
                if let Some(id) = param.result_id {
                    defs.insert(id, param.clone());
                }
            }
            for block in &function.blocks {
                if let Some(label) = &block.label {
                    if let Some(id) = label.result_id {
                        defs.insert(id, label.clone());
                    }
                }
                for inst in &block.instructions {
                    if let Some(id) = inst.result_id {
                        defs.insert(id, inst.clone());
                    }
                }
            }
        }

        let mut pointer_info = HashMap::new();
        for inst in &self.module.types_global_values {
            if inst.class.opcode == Op::TypePointer {
                let (
                    Some(ptr),
                    Some(Operand::StorageClass(storage)),
                    Some(Operand::IdRef(pointee)),
                ) = (inst.result_id, inst.operands.first(), inst.operands.get(1))
                else {
                    continue;
                };
                pointer_info.insert(ptr, (*storage, *pointee));
            }
        }

        let mut insertions = Vec::new();
        let mut next_defs = defs.clone();
        for fi in 0..self.module.functions.len() {
            for bi in 0..self.module.functions[fi].blocks.len() {
                let mut ii = 0usize;
                while ii < self.module.functions[fi].blocks[bi].instructions.len() {
                    let inst = self.module.functions[fi].blocks[bi].instructions[ii].clone();
                    if inst.class.opcode != Op::InBoundsAccessChain || inst.operands.len() != 2 {
                        ii += 1;
                        continue;
                    }
                    let Some(result_type) = inst.result_type else {
                        ii += 1;
                        continue;
                    };
                    let Some((StorageClass::Private, pointee_ty)) =
                        pointer_info.get(&result_type).copied()
                    else {
                        ii += 1;
                        continue;
                    };
                    if !spirv_type_is_scalar(&defs, pointee_ty) {
                        ii += 1;
                        continue;
                    }
                    let Some(Operand::IdRef(base)) = inst.operands.first() else {
                        ii += 1;
                        continue;
                    };
                    let Some(base_def) = next_defs.get(base).cloned() else {
                        ii += 1;
                        continue;
                    };
                    if !matches!(
                        base_def.class.opcode,
                        Op::AccessChain | Op::InBoundsAccessChain
                    ) || base_def.result_type != Some(result_type)
                        || base_def.operands.len() < 2
                    {
                        ii += 1;
                        continue;
                    }
                    let Some(Operand::IdRef(root)) = base_def.operands.first() else {
                        ii += 1;
                        continue;
                    };
                    let Some(root_ptr_ty) = result_type_of_id(&defs, *root) else {
                        ii += 1;
                        continue;
                    };
                    let Some((StorageClass::Private, mut cur_ty)) =
                        pointer_info.get(&root_ptr_ty).copied()
                    else {
                        ii += 1;
                        continue;
                    };
                    for index in &base_def.operands[1..base_def.operands.len() - 1] {
                        let Some(next) = access_chain_step_type(&defs, cur_ty, index) else {
                            cur_ty = 0;
                            break;
                        };
                        cur_ty = next;
                    }
                    if cur_ty == 0
                        || access_chain_step_type(&defs, cur_ty, base_def.operands.last().unwrap())
                            != Some(pointee_ty)
                    {
                        ii += 1;
                        continue;
                    }
                    let Some(Operand::IdRef(offset)) = inst.operands.get(1) else {
                        ii += 1;
                        continue;
                    };
                    let Some(Operand::IdRef(base_last)) = base_def.operands.last() else {
                        ii += 1;
                        continue;
                    };
                    let Some(merged) =
                        self.merge_access_indices(*base_last, *offset, &defs, &mut insertions)
                    else {
                        ii += 1;
                        continue;
                    };
                    let mut operands = base_def.operands.clone();
                    *operands
                        .last_mut()
                        .expect("base access-chain operand length checked") =
                        Operand::IdRef(merged);
                    self.module.functions[fi].blocks[bi].instructions[ii].operands = operands;
                    if let Some(id) = inst.result_id {
                        let mut updated = inst;
                        updated.operands = self.module.functions[fi].blocks[bi].instructions[ii]
                            .operands
                            .clone();
                        next_defs.insert(id, updated);
                    }
                    if !insertions.is_empty() {
                        let pending = std::mem::take(&mut insertions);
                        self.module.functions[fi].blocks[bi]
                            .instructions
                            .splice(ii..ii, pending);
                        ii += 1;
                    }
                    ii += 1;
                }
            }
        }
    }

    fn repair_truncated_access_chain_indices(&mut self) {
        let mut defs = HashMap::new();
        for inst in &self.module.types_global_values {
            if let Some(id) = inst.result_id {
                defs.insert(id, inst.clone());
            }
        }
        for function in &self.module.functions {
            for param in &function.parameters {
                if let Some(id) = param.result_id {
                    defs.insert(id, param.clone());
                }
            }
            for block in &function.blocks {
                if let Some(label) = &block.label {
                    if let Some(id) = label.result_id {
                        defs.insert(id, label.clone());
                    }
                }
                for inst in &block.instructions {
                    if let Some(id) = inst.result_id {
                        defs.insert(id, inst.clone());
                    }
                }
            }
        }

        let mut pointer_info = HashMap::new();
        for inst in &self.module.types_global_values {
            if inst.class.opcode == Op::TypePointer {
                let (
                    Some(ptr),
                    Some(Operand::StorageClass(storage)),
                    Some(Operand::IdRef(pointee)),
                ) = (inst.result_id, inst.operands.first(), inst.operands.get(1))
                else {
                    continue;
                };
                pointer_info.insert(ptr, (*storage, *pointee));
            }
        }

        for fi in 0..self.module.functions.len() {
            for bi in 0..self.module.functions[fi].blocks.len() {
                for ii in 0..self.module.functions[fi].blocks[bi].instructions.len() {
                    let inst = self.module.functions[fi].blocks[bi].instructions[ii].clone();
                    if !matches!(inst.class.opcode, Op::AccessChain | Op::InBoundsAccessChain)
                        || inst.operands.len() < 2
                    {
                        continue;
                    }
                    let Some(result_type) = inst.result_type else {
                        continue;
                    };
                    let Some((_, pointee_ty)) = pointer_info.get(&result_type).copied() else {
                        continue;
                    };
                    let Some(Operand::IdRef(base)) = inst.operands.first() else {
                        continue;
                    };
                    let Some(base_ty) = result_type_of_id(&defs, *base) else {
                        continue;
                    };
                    let Some((_, mut cur_ty)) = pointer_info.get(&base_ty).copied() else {
                        continue;
                    };
                    let mut valid = true;
                    for index in &inst.operands[1..] {
                        let Some(next) = access_chain_step_type(&defs, cur_ty, index) else {
                            valid = false;
                            break;
                        };
                        cur_ty = next;
                    }
                    if !valid || cur_ty == pointee_ty {
                        continue;
                    }
                    let Some(extra_count) = zero_tail_indices_to_pointee(&defs, cur_ty, pointee_ty)
                    else {
                        continue;
                    };
                    if extra_count == 0 {
                        continue;
                    }
                    let Some(index_ty) = inst.operands[1..].iter().rev().find_map(|operand| {
                        let Operand::IdRef(id) = operand else {
                            return None;
                        };
                        result_type_of_id(&defs, *id)
                    }) else {
                        continue;
                    };
                    let zero = self.get_or_create_index_const(index_ty, 0);
                    let mut operands = inst.operands;
                    operands.extend(std::iter::repeat_n(Operand::IdRef(zero), extra_count));
                    self.module.functions[fi].blocks[bi].instructions[ii].operands = operands;
                }
            }
        }
    }

    fn merge_access_indices(
        &mut self,
        lhs: Word,
        rhs: Word,
        defs: &HashMap<Word, Instruction>,
        insertions: &mut Vec<Instruction>,
    ) -> Option<Word> {
        match (const_int_value(defs, lhs), const_int_value(defs, rhs)) {
            (Some(a), Some(b)) => {
                let ty = result_type_of_id(defs, lhs).or_else(|| result_type_of_id(defs, rhs))?;
                return Some(self.get_or_create_index_const(ty, a.checked_add(b)?));
            }
            (Some(0), _) => return Some(rhs),
            (_, Some(0)) => return Some(lhs),
            _ => {}
        }
        let lhs_ty = result_type_of_id(defs, lhs);
        let rhs_ty = result_type_of_id(defs, rhs);
        let (ty, lhs_id, rhs_id) = match (
            lhs_ty,
            rhs_ty,
            const_int_value(defs, lhs),
            const_int_value(defs, rhs),
        ) {
            (Some(a), Some(b), _, _) if a == b => (a, lhs, rhs),
            (_, Some(ty), Some(value), _) => (ty, self.get_or_create_index_const(ty, value), rhs),
            (Some(ty), _, _, Some(value)) => (ty, lhs, self.get_or_create_index_const(ty, value)),
            _ => return None,
        };
        let id = self.fresh();
        insertions.push(Self::inst(
            Op::IAdd,
            Some(ty),
            Some(id),
            vec![Operand::IdRef(lhs_id), Operand::IdRef(rhs_id)],
        ));
        Some(id)
    }

    fn get_or_create_index_const(&mut self, ty: Word, value: u64) -> Word {
        let operands = index_const_operands(&self.module.types_global_values, ty, value)
            .unwrap_or_else(|| vec![Operand::LiteralBit32(value as u32)]);
        for inst in &self.module.types_global_values {
            if inst.class.opcode == Op::Constant
                && inst.result_type == Some(ty)
                && inst.operands == operands
            {
                return inst.result_id.expect("constant has result id");
            }
        }
        let id = self.fresh();
        self.module.types_global_values.push(Self::inst(
            Op::Constant,
            Some(ty),
            Some(id),
            operands,
        ));
        id
    }

    pub(super) fn fresh(&mut self) -> Word {
        self.module.fresh_id()
    }

    pub(super) fn inst(
        op: Op,
        result_type: Option<Word>,
        result_id: Option<Word>,
        operands: Vec<Operand>,
    ) -> Instruction {
        Instruction::new(op, result_type, result_id, operands)
    }
}

fn ptr_access_chain_allowed_storage(storage: StorageClass) -> bool {
    matches!(
        storage,
        StorageClass::Workgroup | StorageClass::StorageBuffer | StorageClass::PhysicalStorageBuffer
    )
}

fn result_type_of_id(defs: &HashMap<Word, Instruction>, id: Word) -> Option<Word> {
    defs.get(&id).and_then(|inst| inst.result_type)
}

fn const_int_value(defs: &HashMap<Word, Instruction>, id: Word) -> Option<u64> {
    let inst = defs.get(&id)?;
    if inst.class.opcode != Op::Constant {
        return None;
    }
    match inst.operands.as_slice() {
        [Operand::LiteralBit32(value)] => Some(u64::from(*value)),
        [Operand::LiteralBit32(lo), Operand::LiteralBit32(hi)] => {
            Some(u64::from(*lo) | (u64::from(*hi) << 32))
        }
        _ => None,
    }
}

fn index_const_operands(types: &[Instruction], ty: Word, value: u64) -> Option<Vec<Operand>> {
    let ty = types
        .iter()
        .find(|inst| inst.result_id == Some(ty) && inst.class.opcode == Op::TypeInt)?;
    let bits = match ty.operands.first()? {
        Operand::LiteralBit32(bits) => *bits,
        _ => return None,
    };
    match bits {
        64 => Some(vec![
            Operand::LiteralBit32(value as u32),
            Operand::LiteralBit32((value >> 32) as u32),
        ]),
        _ => Some(vec![Operand::LiteralBit32(value as u32)]),
    }
}

fn spirv_type_is_scalar(defs: &HashMap<Word, Instruction>, ty: Word) -> bool {
    defs.get(&ty).is_some_and(|inst| {
        matches!(
            inst.class.opcode,
            Op::TypeBool | Op::TypeFloat | Op::TypeInt
        )
    })
}

fn zero_tail_indices_to_pointee(
    defs: &HashMap<Word, Instruction>,
    mut cur_ty: Word,
    pointee_ty: Word,
) -> Option<usize> {
    let mut count = 0usize;
    while cur_ty != pointee_ty {
        let def = defs.get(&cur_ty)?;
        cur_ty = match def.class.opcode {
            Op::TypeArray | Op::TypeRuntimeArray | Op::TypeVector | Op::TypeMatrix => {
                let Operand::IdRef(elem) = def.operands.first()? else {
                    return None;
                };
                *elem
            }
            _ => return None,
        };
        count += 1;
    }
    Some(count)
}

fn access_chain_step_type(
    defs: &HashMap<Word, Instruction>,
    composite_ty: Word,
    index: &Operand,
) -> Option<Word> {
    let def = defs.get(&composite_ty)?;
    match def.class.opcode {
        Op::TypeStruct => {
            let Operand::IdRef(index) = index else {
                return None;
            };
            let member = const_int_value(defs, *index)? as usize;
            match def.operands.get(member)? {
                Operand::IdRef(member_ty) => Some(*member_ty),
                _ => None,
            }
        }
        Op::TypeArray | Op::TypeRuntimeArray | Op::TypeVector | Op::TypeMatrix => {
            match def.operands.first()? {
                Operand::IdRef(elem) => Some(*elem),
                _ => None,
            }
        }
        _ => None,
    }
}
