//! Pointer-network connected-component analysis (M-A2 / M-B1 structural prerequisite).
//!
//! Today each pointer phi/select is reconciled INDEPENDENTLY in `pointer_merge_meta` — the
//! merge-participant sets (`pointer_phi_values`, `pointer_phi_incoming_values`, `selected_pointers`)
//! are flat membership sets, NOT a grouping. That is the wall behind the three M-B1 blockers and the
//! unsound M-A2(a)/(b) read-side flags: a pointee typed differently at two def sites of ONE
//! phi/select network (for example, a loop-carried device pointer whose incomings dereference as
//! `<4 x float>` on one arm and scalar `float` on the other) errors with a pointer-merge pointee
//! mismatch, and no read-side override can fix it because SPIR-V logical addressing forbids
//! `OpBitcast` between pointer types.
//!
//! The sound fix (plan §M-A2, ~line 325) must first BUILD the network — the transitive closure over
//! phi result ↔ incoming and select result ↔ arm edges — then census each component's deref
//! granularities so a later pass can record ONE finest granularity uniformly at every def site. This
//! module is that first step: a pure union-find + granularity census over a function's body, with the
//! per-component classification that decides whether uniform recording is even sound:
//!
//! - [`NetworkClass::Uniform`] — 0/1 distinct recorded pointee: nothing to reconcile.
//! - [`NetworkClass::WholeVsPart`] — all pointees peel to ONE scalar element (`Float` and
//!   `Vector(Float,4)`): the sound whole-vs-part case, finest = the shared scalar.
//! - [`NetworkClass::ReinterpretMix`] — pointees peel to ≥2 DISTINCT scalars (`Float` and `Int(32)`):
//!   a genuine reinterpret; network widening is UNSOUND (dead-end #14, needs per-use retyping).
//! - [`NetworkClass::Unclassified`] — mixed with a non-scalar-family relationship (structs, nested
//!   pointers): not a widening candidate.
//!
//! The analysis is read-only. The emitter consumes its structural classifications when choosing the
//! representation of pointer networks.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use crate::native::cfg::BodyBlock;
use crate::native::ir::{LlType, LlValue};
use crate::native::tir::TirOperand;

/// How a pointer network's per-def-site deref granularities relate — decides whether recording one
/// finest pointee uniformly across the component is sound.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(in crate::native) enum NetworkClass {
    /// 0 or 1 distinct recorded pointee across the whole component — already consistent.
    Uniform,
    /// Mixed pointees that all peel to ONE scalar element. Carries that shared scalar (the finest,
    /// sound uniform-recording target — e.g. `float*` for `{Float, Vector(Float,4)}`).
    WholeVsPart(LlType),
    /// Mixed pointees peeling to ≥2 DISTINCT scalar elements (a genuine reinterpret). Network
    /// widening cannot pick one arm soundly (dead-end #14) — needs per-use retyping.
    ReinterpretMix,
    /// Mixed pointees with no scalar-family relationship the census models (structs, nested
    /// pointers). Left for a human; not a widening candidate.
    Unclassified,
}

/// One connected component of pointer SSA values joined by phi-incoming and select-arm edges.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(in crate::native) struct PointerNetwork {
    /// SSA names (`%`-prefixed, matching `pointer_pointees` keys) in this component, sorted.
    pub(in crate::native) members: Vec<String>,
    /// Distinct recorded pointees among members, deduped and sorted for determinism.
    pub(in crate::native) pointees: Vec<LlType>,
    pub(in crate::native) class: NetworkClass,
}

/// Union-find over `%`-prefixed SSA names. A node not yet seen is its own root.
#[derive(Default)]
struct UnionFind {
    parent: HashMap<String, String>,
}

impl UnionFind {
    fn find(&mut self, x: &str) -> String {
        let mut root = x.to_string();
        while let Some(p) = self.parent.get(&root) {
            if p == &root {
                break;
            }
            root = p.clone();
        }
        // Path compression: point every node on the walk straight at the root.
        let mut cur = x.to_string();
        while cur != root {
            let next = self
                .parent
                .get(&cur)
                .cloned()
                .unwrap_or_else(|| cur.clone());
            self.parent.insert(cur.clone(), root.clone());
            if next == cur {
                break;
            }
            cur = next;
        }
        root
    }

    fn union(&mut self, a: &str, b: &str) {
        self.parent
            .entry(a.to_string())
            .or_insert_with(|| a.to_string());
        self.parent
            .entry(b.to_string())
            .or_insert_with(|| b.to_string());
        let ra = self.find(a);
        let rb = self.find(b);
        if ra != rb {
            self.parent.insert(ra, rb);
        }
    }
}

/// Peel `Vector`/`Array` element wrappers to the innermost scalar/aggregate. `Vector(Float,4)` and
/// `Array(Float,128)` both peel to `Float`; a `Struct`/`Named`/scalar peels to itself.
fn peel_scalar(ty: &LlType) -> &LlType {
    match ty {
        LlType::Vector(e, _) | LlType::Array(e, _) => peel_scalar(e),
        other => other,
    }
}

/// A pointee whose finest element is a primitive numeric scalar — the only family whole-vs-part
/// widening understands (a coarser access is re-expressible via access-chain indices).
fn is_primitive_scalar(ty: &LlType) -> bool {
    matches!(
        ty,
        LlType::Float | LlType::Half | LlType::BFloat | LlType::Int(_)
    )
}

/// Classify a component from its distinct recorded pointee set (already deduped).
fn classify(pointees: &[LlType]) -> NetworkClass {
    if pointees.len() <= 1 {
        return NetworkClass::Uniform;
    }
    let mut scalars: Vec<&LlType> = Vec::new();
    for p in pointees {
        let s = peel_scalar(p);
        if !is_primitive_scalar(s) {
            return NetworkClass::Unclassified;
        }
        if !scalars.contains(&s) {
            scalars.push(s);
        }
    }
    match scalars.as_slice() {
        [only] => NetworkClass::WholeVsPart((*only).clone()),
        _ => NetworkClass::ReinterpretMix,
    }
}

/// Extract pointer-network edges from a function body: for each pointer `phi`, an edge from the
/// result to every `%local` incoming; for each pointer `select`, an edge from the result to every
/// `%local` arm. Globals/constants are not def sites we retype, so only `%local` endpoints join a
/// network.
fn pointer_edges(blocks: &[BodyBlock]) -> Vec<(String, String)> {
    let mut edges = Vec::new();
    for block in blocks {
        if let Some(carrier) = &block.typed {
            pointer_edges_from_carrier(carrier, &mut edges);
        }
    }
    edges
}

/// Walk a block's typed instructions for the same phi-ptr / select-ptr / bitcast-ptr edges the line
/// scan below extracts. Byte-identical to the line scan by construction: a `phi ptr` lowers to an
/// inst with `opcode == "phi"` and a `Ptr` phi type; a pointer `select` to `opcode == "select"` with
/// both value arms `Ptr`-typed; a `bitcast ptr … to ptr` to the `bitcast` carrier (`Ptr` source + a
/// `ptr` destination-type text) — the same `parse_phi`/`parse_typed_value`/`resolve_bitcast` the line
/// scan re-lexed.
fn pointer_edges_from_carrier(
    carrier: &crate::native::tir::TirBlock,
    edges: &mut Vec<(String, String)>,
) {
    for inst in &carrier.insts {
        let Some(name) = &inst.result else {
            continue;
        };
        if inst.opcode == "phi" {
            if let Some((ty, incomings)) = &inst.phi_incoming() {
                if matches!(ty, LlType::Ptr(_)) {
                    for (value, _pred) in incomings {
                        if let LlValue::Local(inc) = value {
                            edges.push((name.clone(), inc.clone()));
                        }
                    }
                }
            }
        } else if inst.opcode == "select" && inst.operands.len() == 3 {
            // `select <cty> <c>, <ty> <a>, <ty> <b>`: both value arms must be `ptr`-typed (the line
            // scan returns None otherwise); then join the `%local` arms.
            let arms = [&inst.operands[1], &inst.operands[2]];
            if arms.iter().all(|arm| is_ptr_operand(arm)) {
                for arm in arms {
                    if let Some(tv) = arm.as_typed_value() {
                        if let LlValue::Local(local) = tv.value {
                            edges.push((name.clone(), local));
                        }
                    }
                }
            }
        } else if inst.opcode == "bitcast" {
            // A `bitcast ptr %src to ptr` is a pointer-identity alias (a no-op in opaque-pointer LLVM):
            // result and source are the same pointer, so they belong to one network. Load-bearing for
            // the access census: a scalar `float` arm can reach its phi through a pointer bitcast;
            // without this edge the scalar access is invisible and a whole-vs-part network can
            // masquerade as Uniform.
            if let Some((src, dst)) = inst.bitcast() {
                if matches!(src.ty, LlType::Ptr(_)) && dst.trim_start().starts_with("ptr") {
                    if let LlValue::Local(local) = &src.value {
                        edges.push((name.clone(), local.clone()));
                    }
                }
            }
        } else if inst.opcode == "freeze" && matches!(inst.result_ty, Some(LlType::Ptr(_))) {
            if let Some(TirOperand::Value {
                name: local,
                ty: LlType::Ptr(_),
            }) = inst.operands.first()
            {
                edges.push((name.clone(), local.clone()));
            }
        }
    }
}

/// Pointer components whose only roots are `null`/`undef` and whose remaining producers are
/// transparent pointer aliases or GEPs rooted back in the same component. These arise after literal
/// specialization removes the concrete arm of a pointer induction. Retaining its separately-carried
/// nullness is sufficient while the pointer payload may be represented by a correctly typed `OpUndef`.
/// Source blocks may still contain consumers on paths discarded by the selected CFG and cleanup
/// construction; the final product module retains only the separately represented nullness.
pub(in crate::native) fn null_rooted_pointer_networks(blocks: &[BodyBlock]) -> Vec<Vec<String>> {
    let definitions = blocks
        .iter()
        .filter_map(|block| block.typed.as_ref())
        .flat_map(|block| &block.insts)
        .filter_map(|inst| inst.result.as_ref().map(|result| (result.as_str(), inst)))
        .collect::<HashMap<_, _>>();
    let mut result = Vec::new();
    for members in build_null_rooted_components(blocks) {
        let member_set = members.iter().map(String::as_str).collect::<HashSet<_>>();
        let mut saw_nullish = false;
        let valid = members.iter().all(|name| {
            let Some(inst) = definitions.get(name.as_str()) else {
                return false;
            };
            let pointer_value = |value: &LlValue, saw_nullish: &mut bool| match value {
                LlValue::Local(local) => member_set.contains(local.as_str()),
                LlValue::Zero | LlValue::Undef => {
                    *saw_nullish = true;
                    true
                }
                _ => false,
            };
            match inst.opcode.as_str() {
                "phi" => inst.phi_values().is_some_and(|values| {
                    values
                        .into_iter()
                        .all(|value| pointer_value(value, &mut saw_nullish))
                }),
                "select" => inst.select_arms().as_ref().is_some_and(|arms| {
                    pointer_value(&arms.0.value, &mut saw_nullish)
                        && pointer_value(&arms.1.value, &mut saw_nullish)
                }),
                "bitcast" | "freeze" => inst
                    .operands
                    .first()
                    .and_then(TirOperand::as_typed_value)
                    .is_some_and(|value| pointer_value(&value.value, &mut saw_nullish)),
                "getelementptr" => inst
                    .gep()
                    .as_deref()
                    .is_some_and(|gep| pointer_value(&gep.base.value, &mut saw_nullish)),
                _ => false,
            }
        });
        if valid && saw_nullish {
            result.push(members);
        }
    }
    result
}

/// Components for null-root closure include GEP result↔base edges in addition to the ordinary
/// phi/select/transparent-alias graph. A GEP is itself an allowed producer in this classification;
/// omitting its edge can split one recurrence into two apparent components and hide the concrete root
/// (or make an otherwise closed component appear to reference an external producer).
fn build_null_rooted_components(blocks: &[BodyBlock]) -> Vec<Vec<String>> {
    let mut edges = pointer_edges(blocks);
    for block in blocks {
        let Some(carrier) = &block.typed else {
            continue;
        };
        for inst in &carrier.insts {
            let (Some(result), Some(gep)) = (&inst.result, inst.gep().as_deref()) else {
                continue;
            };
            if let LlValue::Local(base) = &gep.base.value {
                edges.push((result.clone(), base.clone()));
            }
        }
    }
    build_components_from_edges(edges)
}

#[cfg(test)]
pub(in crate::native) fn null_rooted_pointer_network_members(
    blocks: &[BodyBlock],
) -> HashSet<String> {
    null_rooted_pointer_networks(blocks)
        .into_iter()
        .flatten()
        .collect()
}

/// Whether a typed operand is `ptr`-typed (the dual of the line scan's arm `starts_with("ptr")`).
fn is_ptr_operand(op: &crate::native::tir::TirOperand) -> bool {
    op.as_typed_value()
        .is_some_and(|tv| matches!(tv.ty, LlType::Ptr(_)))
}

/// Group the pointer SSA names of `blocks` into connected components over the phi/select edges.
/// Shared by every census; each returned member list is sorted and deduped.
fn build_components(blocks: &[BodyBlock]) -> Vec<Vec<String>> {
    build_components_from_edges(pointer_edges(blocks))
}

fn build_components_from_edges(edges: Vec<(String, String)>) -> Vec<Vec<String>> {
    let mut uf = UnionFind::default();
    let mut nodes: BTreeSet<String> = BTreeSet::new();
    for (a, b) in &edges {
        uf.union(a, b);
        nodes.insert(a.clone());
        nodes.insert(b.clone());
    }
    let mut groups: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for n in &nodes {
        let root = uf.find(n);
        groups.entry(root).or_default().push(n.clone());
    }
    groups
        .into_values()
        .map(|mut members| {
            members.sort();
            members.dedup();
            members
        })
        .collect()
}

/// Census a component from a per-member type lookup (each member may map to one or several types).
fn census<'a, F, I>(members: Vec<String>, types_of: F) -> PointerNetwork
where
    F: Fn(&str) -> I,
    I: IntoIterator<Item = &'a LlType>,
{
    let mut distinct: Vec<LlType> = Vec::new();
    for m in &members {
        for p in types_of(m) {
            if !distinct.contains(p) {
                distinct.push(p.clone());
            }
        }
    }
    distinct.sort_by_key(|t| format!("{t:?}"));
    let class = classify(&distinct);
    PointerNetwork {
        members,
        pointees: distinct,
        class,
    }
}

/// Build every pointer network in `blocks` and census each against the recorded `pointees`.
pub(in crate::native) fn analyze_pointer_networks(
    blocks: &[BodyBlock],
    pointees: &HashMap<String, LlType>,
) -> Vec<PointerNetwork> {
    build_components(blocks)
        .into_iter()
        .map(|members| census(members, |m| pointees.get(m)))
        .collect()
}

/// Census every pointer network against the TRUE IR ACCESS WIDTH — the element types the members are
/// actually dereferenced/stepped at (`load`/`store`/`getelementptr`), NOT the element-scalar carrier.
/// This is the census the M-A2 def-site fix must consult: the `use_pointees` carrier reports the
/// innermost scalar, so a member accessed whole-vector (`load <4 x float>`) is reported `Float` and a
/// real whole-vs-part component is mislabelled `Uniform` (measured: seeding that flattened scalar
/// regresses 11 frontier cases). Here a member loaded/stepped at `<4 x float>` while another is stepped
/// at scalar `float` makes the component census `{Vector(Float,4), Float}` → `WholeVsPart(Float)`, the
/// finest expressible granularity — exactly what a sound recording+scalarization pass needs. Read-only.
pub(in crate::native) fn analyze_networks_by_access(blocks: &[BodyBlock]) -> Vec<PointerNetwork> {
    let access = access_pointees(blocks);
    build_components(blocks)
        .into_iter()
        .map(|members| census(members, |m| access.get(m).into_iter().flatten()))
        .collect()
}

/// Buffer parameters whose pointer-merge component genuinely reinterprets storage between distinct
/// scalar families. Logical SPIR-V cannot assign one pointee to such a component. Trace every member
/// back through typed pointer identities, GEPs, selects, and phis. This primary-construction path is
/// needed when the merged carrier crosses a phi and either still contains the function-constant
/// select or has one surviving root after function-constant pruning. A select-only component can
/// keep using its existing value-domain lowering. The concrete roots must be function-constant
/// buffer alternatives at one Metal location, and a pruned single root must have another metadata
/// alternative at that location. This gives the roots one descriptor identity despite their
/// different source pointees. When every concrete leaf satisfies that contract, return all exact
/// roots for byte-address modeling. An unknown producer rejects the complete component, preventing
/// partial or mismatched pointer representations.
pub(in crate::native) fn reinterpret_mix_buffer_params(
    blocks: &[BodyBlock],
    buffer_params: &BTreeMap<String, u32>,
) -> BTreeSet<String> {
    let mut sources = HashMap::<String, Vec<LlValue>>::new();
    let mut phi_results = HashSet::new();
    let mut select_results = HashSet::new();
    for block in blocks {
        let Some(carrier) = &block.typed else {
            continue;
        };
        for inst in &carrier.insts {
            let (Some(result), Some(LlType::Ptr(_))) = (&inst.result, &inst.result_ty) else {
                continue;
            };
            let values = if let Some(gep) = inst.gep().as_deref() {
                vec![gep.base.value.clone()]
            } else if let Some((source, _)) = inst.bitcast() {
                vec![source.value]
            } else if let Some((_, incoming)) = inst.phi_incoming().as_ref() {
                phi_results.insert(result.clone());
                incoming.iter().map(|(value, _)| value.clone()).collect()
            } else if let Some((true_value, false_value)) = inst.select_arms().as_deref() {
                select_results.insert(result.clone());
                vec![true_value.value.clone(), false_value.value.clone()]
            } else if matches!(inst.opcode.as_str(), "addrspacecast" | "freeze") {
                inst.operands
                    .first()
                    .and_then(|operand| operand.as_typed_value())
                    .map(|value| vec![value.value])
                    .unwrap_or_default()
            } else {
                Vec::new()
            };
            if !values.is_empty() {
                sources.insert(result.clone(), values);
            }
        }
    }

    fn roots_for(
        name: &str,
        buffer_params: &BTreeMap<String, u32>,
        sources: &HashMap<String, Vec<LlValue>>,
        select_results: &HashSet<String>,
        visiting: &mut HashSet<String>,
        members: &mut BTreeSet<String>,
        saw_select: &mut bool,
    ) -> Option<BTreeSet<String>> {
        members.insert(name.to_string());
        *saw_select |= select_results.contains(name);
        if buffer_params.contains_key(name) {
            return Some(BTreeSet::from([name.to_string()]));
        }
        if !visiting.insert(name.to_string()) {
            return Some(BTreeSet::new());
        }
        let result = sources.get(name).and_then(|values| {
            let mut roots = BTreeSet::new();
            for value in values {
                match value {
                    LlValue::Local(source) => {
                        roots.extend(roots_for(
                            source,
                            buffer_params,
                            sources,
                            select_results,
                            visiting,
                            members,
                            saw_select,
                        )?);
                    }
                    LlValue::Zero | LlValue::Undef => {}
                    _ => return None,
                }
            }
            Some(roots)
        });
        visiting.remove(name);
        result
    }

    let mut selected = BTreeSet::new();
    let access = access_pointees(blocks);
    for phi in &phi_results {
        let mut members = BTreeSet::new();
        let mut saw_select = false;
        let Some(roots) = roots_for(
            phi,
            buffer_params,
            &sources,
            &select_results,
            &mut HashSet::new(),
            &mut members,
            &mut saw_select,
        ) else {
            continue;
        };
        let mut pointees = Vec::new();
        for pointee in members
            .iter()
            .filter_map(|member| access.get(member))
            .flatten()
        {
            if !pointees.contains(pointee) {
                pointees.push(pointee.clone());
            }
        }
        let root_locations = roots
            .iter()
            .filter_map(|root| buffer_params.get(root))
            .copied()
            .collect::<BTreeSet<_>>();
        let one_shared_location = root_locations.len() == 1;
        let metadata_alternatives = root_locations.first().map_or(0, |location| {
            buffer_params
                .values()
                .filter(|candidate| *candidate == location)
                .count()
        });
        let recurrent_phi_carrier = members
            .iter()
            .filter(|member| phi_results.contains(*member))
            .take(2)
            .count()
            >= 2;
        let function_constant_shape = (saw_select && roots.len() >= 2)
            || (roots.len() == 1 && metadata_alternatives >= 2 && recurrent_phi_carrier);
        if function_constant_shape
            && one_shared_location
            && matches!(classify(&pointees), NetworkClass::ReinterpretMix)
        {
            selected.extend(roots);
        }
    }
    selected
}

/// Collect, per pointer SSA name, the distinct element types it is dereferenced or stepped at across the
/// function body: a `load T`/`store T` derefs the pointer at `T`; a `getelementptr T, ptr %base, …`
/// steps `%base` at `T` (element stride) and defines a result whose pointee is `gep_pointee(T, idx)`.
/// A pointer never directly accessed (e.g. a bare phi arm) contributes nothing — the other members of
/// its network carry the width. Non-parseable lines are skipped (floor-safe: an unseen access can only
/// widen the census toward "mixed", never falsely narrow it to uniform).
fn access_pointees(blocks: &[BodyBlock]) -> HashMap<String, Vec<LlType>> {
    use crate::native::emitter::helpers::gep_pointee;

    let mut map: HashMap<String, Vec<LlType>> = HashMap::new();
    let mut record = |name: &str, ty: LlType| {
        let entry = map.entry(name.to_string()).or_default();
        if !entry.contains(&ty) {
            entry.push(ty);
        }
    };
    for block in blocks {
        // Typed walk (the sole substrate): `load`/`store` carry the deref type on `result_ty` / the
        // value operand and the pointer on `operands`, and `getelementptr` carries the full `parse_gep`
        // result on `inst.gep()`.
        if let Some(carrier) = &block.typed {
            {
                for inst in &carrier.insts {
                    if inst.opcode == "load" {
                        if let (Some(ty), Some(ptr)) = (
                            &inst.result_ty,
                            inst.operands.first().and_then(local_operand),
                        ) {
                            record(&ptr, ty.clone());
                        }
                    } else if inst.opcode == "store" {
                        if let (Some(val), Some(ptr)) = (
                            inst.operands.first().and_then(|o| o.as_typed_value()),
                            inst.operands.get(1).and_then(local_operand),
                        ) {
                            record(&ptr, val.ty);
                        }
                    } else if inst.opcode == "getelementptr" {
                        if let Some(gep) = &inst.gep() {
                            if let LlValue::Local(base) = &gep.base.value {
                                record(base, gep.source_ty.clone());
                            }
                            if let (Some(result), Ok(pointee)) =
                                (&inst.result, gep_pointee(&gep.source_ty, &gep.indices))
                            {
                                record(result, pointee);
                            }
                        }
                    }
                }
            }
        }
    }
    map
}

/// The `%local` name of a typed operand, or `None` (constant/unresolved). Shared by the access census.
fn local_operand(op: &crate::native::tir::TirOperand) -> Option<String> {
    match op.as_typed_value()?.value {
        LlValue::Local(name) => Some(name),
        _ => None,
    }
}

/// Buffer roots whose typed load spans beyond a struct member. Logical SPIR-V access chains preserve
/// member boundaries, so a wider load cannot be constructed by incrementing that member index as if
/// it were an array stride. Select byte-addressed storage before emission for exactly those roots.
pub(in crate::native) fn cross_member_widening_load_roots(
    blocks: &[BodyBlock],
    buffer_params: &HashSet<String>,
    named_types: &HashMap<String, LlType>,
) -> BTreeSet<String> {
    use crate::native::emitter::helpers::{bitcast_width, gep_parent_before_last, gep_pointee};

    let mut sources = HashMap::<String, Vec<String>>::new();
    let mut cross_member_gep_pointee_bits = HashMap::<String, u32>::new();
    let mut loads = Vec::<(String, LlType)>::new();
    fn resolve_type(
        ty: &LlType,
        named_types: &HashMap<String, LlType>,
        visiting: &mut HashSet<String>,
    ) -> Option<LlType> {
        match ty {
            LlType::Named(name) => {
                if !visiting.insert(name.clone()) {
                    return None;
                }
                let resolved = resolve_type(named_types.get(name)?, named_types, visiting);
                visiting.remove(name);
                resolved
            }
            LlType::Vector(elem, 1) => resolve_type(elem, named_types, visiting),
            LlType::Vector(elem, lanes) => Some(LlType::Vector(
                Box::new(resolve_type(elem, named_types, visiting)?),
                *lanes,
            )),
            LlType::Array(elem, len) => Some(LlType::Array(
                Box::new(resolve_type(elem, named_types, visiting)?),
                *len,
            )),
            LlType::Struct(fields) => fields
                .iter()
                .map(|field| resolve_type(field, named_types, visiting))
                .collect::<Option<Vec<_>>>()
                .map(LlType::Struct),
            LlType::Int(1) => Some(LlType::Bool),
            other => Some(other.clone()),
        }
    }
    for block in blocks {
        let Some(carrier) = &block.typed else {
            continue;
        };
        for inst in &carrier.insts {
            if inst.opcode == "load" {
                if let (Some(result_ty), Some(pointer)) = (
                    inst.result_ty.clone(),
                    inst.operands.first().and_then(local_operand),
                ) {
                    loads.push((pointer, result_ty));
                }
            }
            let Some(result) = &inst.result else {
                continue;
            };
            let operands = if let Some(gep) = inst.gep().as_deref() {
                let source_ty = resolve_type(&gep.source_ty, named_types, &mut HashSet::new());
                if matches!(
                    source_ty
                        .as_ref()
                        .and_then(|source_ty| { gep_parent_before_last(source_ty, &gep.indices) }),
                    Some(LlType::Struct(_))
                ) {
                    if let Some(bits) = source_ty
                        .as_ref()
                        .and_then(|source_ty| gep_pointee(source_ty, &gep.indices).ok())
                        .and_then(|ty| bitcast_width(&ty))
                    {
                        cross_member_gep_pointee_bits.insert(result.clone(), bits);
                    }
                }
                match &gep.base.value {
                    LlValue::Local(base) => vec![base.clone()],
                    _ => Vec::new(),
                }
            } else if let Some((source, _)) = inst.bitcast() {
                match source.value {
                    LlValue::Local(source) => vec![source],
                    _ => Vec::new(),
                }
            } else if let Some((_, incoming)) = inst.phi_incoming().as_ref() {
                incoming
                    .iter()
                    .filter_map(|(value, _)| match value {
                        LlValue::Local(source) => Some(source.clone()),
                        _ => None,
                    })
                    .collect()
            } else if let Some((true_value, false_value)) = inst.select_arms().as_deref() {
                [true_value, false_value]
                    .into_iter()
                    .filter_map(|value| match &value.value {
                        LlValue::Local(source) => Some(source.clone()),
                        _ => None,
                    })
                    .collect()
            } else if matches!(inst.opcode.as_str(), "addrspacecast" | "freeze") {
                inst.operands
                    .first()
                    .and_then(local_operand)
                    .into_iter()
                    .collect()
            } else {
                Vec::new()
            };
            if !operands.is_empty() {
                sources.insert(result.clone(), operands);
            }
        }
    }

    fn collect_roots(
        name: &str,
        sources: &HashMap<String, Vec<String>>,
        buffer_params: &HashSet<String>,
        visiting: &mut HashSet<String>,
        roots: &mut BTreeSet<String>,
    ) {
        if buffer_params.contains(name) {
            roots.insert(name.to_string());
            return;
        }
        if !visiting.insert(name.to_string()) {
            return;
        }
        if let Some(parents) = sources.get(name) {
            for parent in parents {
                collect_roots(parent, sources, buffer_params, visiting, roots);
            }
        }
        visiting.remove(name);
    }

    let mut roots = BTreeSet::new();
    for (pointer, load_ty) in loads {
        let Some(load_bits) = bitcast_width(&load_ty) else {
            continue;
        };
        let mut stack = vec![pointer.clone()];
        let mut seen = HashSet::new();
        let mut crosses = false;
        while let Some(name) = stack.pop() {
            if !seen.insert(name.clone()) {
                continue;
            }
            if cross_member_gep_pointee_bits
                .get(&name)
                .is_some_and(|pointee_bits| load_bits > *pointee_bits)
            {
                crosses = true;
                break;
            }
            if let Some(parents) = sources.get(&name) {
                stack.extend(parents.iter().cloned());
            }
        }
        if crosses {
            collect_roots(
                &pointer,
                &sources,
                buffer_params,
                &mut HashSet::new(),
                &mut roots,
            );
        }
    }
    roots
}

/// Pointer names that are stepped as an ARRAY of a bare primitive scalar — a
/// `getelementptr <scalar>, ptr %base, <index>` whose element type is a primitive scalar (not an
/// aggregate) and whose single index is non-zero or dynamic. Such a `%base` implicitly addresses an
/// array of that scalar (the byte/word scratch model), so recording its network's pointee as the bare
/// scalar mis-DECLARES the object: the emitter then declares `%base` as a scalar `OpVariable` /
/// scalar-pointee pointer and every non-identity step becomes an `OpInBoundsAccessChain` /
/// `OpPtrAccessChain` into a non-composite scalar (spirv-val "reached non-composite type" /
/// "not a logical pointer"). Seeding these networks is the inconsistent partial retyping the seed's
/// scalar recording cannot make sound WITHOUT re-declaring the object as an array + re-striding every
/// index (the M-A2(c) #2/#3 keystone work); until that exists, they are excluded from the seed so the
/// recording stays a strict subset of the sound cases. A leading constant `0` (identity / true
/// aggregate descent) does NOT qualify — only a genuine scalar-stride step.
pub(in crate::native) fn array_indexed_scalar_bases(blocks: &[BodyBlock]) -> BTreeSet<String> {
    use crate::native::emitter::helpers::const_index;
    use crate::native::ir::LlGep;

    let mut set = BTreeSet::new();
    // Flag `%base` of a bare-scalar-element GEP stepped by exactly one non-zero/dynamic index (it
    // implicitly addresses an array of that scalar). An aggregate `source_ty` (its leading `0` descends
    // a real struct/array) is declared correctly regardless of the seed.
    let mut consider = |gep: &LlGep| {
        let LlValue::Local(base) = &gep.base.value else {
            return;
        };
        if !is_primitive_scalar(&gep.source_ty) || gep.indices.len() != 1 {
            return;
        }
        if const_index(gep.indices.first()) != Some(0) {
            set.insert(base.clone());
        }
    };
    for block in blocks {
        // Typed walk (the sole substrate): the carrier holds each GEP's full `parse_gep` result on
        // `inst.gep()`.
        if let Some(carrier) = &block.typed {
            for inst in &carrier.insts {
                if let Some(gep) = &inst.gep() {
                    consider(gep);
                }
            }
        }
    }
    set
}

#[cfg(test)]
mod tests {
    use super::*;

    fn block(lines: &[&str]) -> BodyBlock {
        let mut lines: Vec<String> = lines.iter().map(|l| l.to_string()).collect();
        // Give the fixture a terminator so it lowers to a typed carrier — the production state after
        // `populate_typed_carriers`, so the scanners exercise their carrier branch. `ret void` matches
        // none of the phi/select/bitcast/load/store/gep patterns, so the census is unchanged.
        lines.push("ret void".to_string());
        let typed = crate::native::tir::lower_block_carrier("entry", &lines, &HashMap::new());
        assert!(typed.is_some(), "fixture must lower: {lines:?}");
        BodyBlock {
            name: "entry".to_string(),
            role: crate::native::cfg::BlockRole::Normal,
            typed: typed.map(Into::into),
        }
    }

    fn net_containing<'a>(nets: &'a [PointerNetwork], member: &str) -> &'a PointerNetwork {
        nets.iter()
            .find(|n| n.members.iter().any(|m| m == member))
            .expect("network with member")
    }

    #[test]
    fn peels_vector_and_array_to_scalar() {
        assert_eq!(
            peel_scalar(&LlType::Vector(Box::new(LlType::Float), 4)),
            &LlType::Float
        );
        assert_eq!(
            peel_scalar(&LlType::Array(Box::new(LlType::Half), 8)),
            &LlType::Half
        );
        assert_eq!(peel_scalar(&LlType::Float), &LlType::Float);
    }

    #[test]
    fn classifies_whole_vs_part_as_finest_scalar() {
        let c = classify(&[LlType::Float, LlType::Vector(Box::new(LlType::Float), 4)]);
        assert_eq!(c, NetworkClass::WholeVsPart(LlType::Float));
    }

    #[test]
    fn classifies_same_scalar_across_two_vector_widths() {
        let c = classify(&[
            LlType::Vector(Box::new(LlType::Float), 2),
            LlType::Vector(Box::new(LlType::Float), 4),
        ]);
        assert_eq!(c, NetworkClass::WholeVsPart(LlType::Float));
    }

    #[test]
    fn classifies_reinterpret_mix_for_different_scalars() {
        assert_eq!(
            classify(&[LlType::Float, LlType::Int(32)]),
            NetworkClass::ReinterpretMix
        );
    }

    #[test]
    fn classifies_struct_mix_as_unclassified() {
        let c = classify(&[
            LlType::Float,
            LlType::Struct(vec![LlType::Half, LlType::Half]),
        ]);
        assert_eq!(c, NetworkClass::Unclassified);
    }

    #[test]
    fn recognizes_closed_null_rooted_pointer_recurrence() {
        let blocks = [block(&[
            "%cursor = phi ptr addrspace(2) [ null, %entry ], [ %next, %loop ]",
            "%next = getelementptr float, ptr addrspace(2) %cursor, i64 %stride",
        ])];

        assert_eq!(
            null_rooted_pointer_network_members(&blocks),
            HashSet::from(["%cursor".to_string(), "%next".to_string()])
        );
    }

    #[test]
    fn gep_edges_close_half_whole_part_null_recurrence() {
        let blocks = [block(&[
            "%cursor = phi ptr addrspace(2) [ null, %entry ], [ %next, %loop ]",
            "%wide_next = getelementptr <4 x half>, ptr addrspace(2) %cursor, i64 %index",
            "%next = bitcast ptr addrspace(2) %wide_next to ptr addrspace(2)",
            "%wide = load <4 x half>, ptr addrspace(2) %next",
            "%narrow = load half, ptr addrspace(2) %cursor",
        ])];
        let null_members = null_rooted_pointer_network_members(&blocks);
        let network = analyze_networks_by_access(&blocks)
            .into_iter()
            .find(|network| network.members.iter().any(|member| member == "%cursor"))
            .expect("pointer network");

        assert_eq!(network.class, NetworkClass::WholeVsPart(LlType::Half));
        assert!(
            network
                .members
                .iter()
                .all(|member| null_members.contains(member)),
            "the access network must be one closed null-rooted component"
        );
    }

    #[test]
    fn concrete_pointer_root_disqualifies_null_rooted_recurrence() {
        let blocks = [block(&[
            "%cursor = phi ptr addrspace(2) [ %base, %entry ], [ %next, %loop ]",
            "%next = getelementptr float, ptr addrspace(2) %cursor, i64 %stride",
        ])];

        assert!(null_rooted_pointer_network_members(&blocks).is_empty());
    }

    #[test]
    fn reinterpret_mix_select_marks_only_its_traced_buffer_roots() {
        let blocks = [block(&[
            "%fp = getelementptr float, ptr addrspace(2) %floats, i64 0",
            "%hp = getelementptr half, ptr addrspace(2) %halves, i64 0",
            "%fv = bitcast ptr addrspace(2) %fp to ptr addrspace(2)",
            "%hv = bitcast ptr addrspace(2) %hp to ptr addrspace(2)",
            "%picked = select i1 %condition, ptr addrspace(2) %hv, ptr addrspace(2) %fv",
            "%vector = getelementptr <4 x float>, ptr addrspace(2) %picked, i64 1",
            "%merged = phi ptr addrspace(2) [ %vector, %left ], [ %picked, %right ]",
            "%value = load <4 x float>, ptr addrspace(2) %merged",
        ])];
        let eligible = BTreeMap::from([
            ("%floats".to_string(), 29),
            ("%halves".to_string(), 29),
            ("%unrelated".to_string(), 30),
        ]);

        assert_eq!(
            reinterpret_mix_buffer_params(&blocks, &eligible),
            BTreeSet::from(["%floats".to_string(), "%halves".to_string()])
        );
    }

    #[test]
    fn reinterpret_mix_select_without_pointer_phi_keeps_value_domain_lowering() {
        let blocks = [block(&[
            "%fp = getelementptr float, ptr addrspace(2) %storage, i64 0",
            "%hp = getelementptr half, ptr addrspace(2) %storage, i64 0",
            "%picked = select i1 %condition, ptr addrspace(2) %hp, ptr addrspace(2) %fp",
            "%value = load <4 x float>, ptr addrspace(2) %picked",
        ])];
        let eligible = BTreeMap::from([("%storage".to_string(), 1)]);

        assert!(reinterpret_mix_buffer_params(&blocks, &eligible).is_empty());
    }

    #[test]
    fn reinterpret_mix_after_function_constant_pruning_marks_the_surviving_root() {
        let blocks = [block(&[
            "%hp = getelementptr half, ptr addrspace(2) %halves, i64 0",
            "%view = bitcast ptr addrspace(2) %hp to ptr addrspace(2)",
            "%vector = getelementptr <4 x float>, ptr addrspace(2) %view, i64 1",
            "%merged = phi ptr addrspace(2) [ %vector, %left ], [ %view, %right ]",
            "%next = getelementptr <4 x float>, ptr addrspace(2) %merged, i64 1",
            "%recurred = phi ptr addrspace(2) [ %next, %body ], [ %merged, %entry ]",
            "%value = load <4 x float>, ptr addrspace(2) %recurred",
        ])];
        let eligible = BTreeMap::from([("%floats".to_string(), 29), ("%halves".to_string(), 29)]);

        assert_eq!(
            reinterpret_mix_buffer_params(&blocks, &eligible),
            BTreeSet::from(["%halves".to_string()])
        );
    }

    #[test]
    fn single_phi_after_function_constant_pruning_is_not_raw_modeled() {
        let blocks = [block(&[
            "%hp = getelementptr half, ptr addrspace(2) %halves, i64 0",
            "%view = bitcast ptr addrspace(2) %hp to ptr addrspace(2)",
            "%vector = getelementptr <4 x float>, ptr addrspace(2) %view, i64 1",
            "%merged = phi ptr addrspace(2) [ %vector, %left ], [ %view, %right ]",
            "%value = load <4 x float>, ptr addrspace(2) %merged",
        ])];
        let eligible = BTreeMap::from([("%floats".to_string(), 29), ("%halves".to_string(), 29)]);

        assert!(reinterpret_mix_buffer_params(&blocks, &eligible).is_empty());
    }

    #[test]
    fn reinterpret_mix_with_different_buffer_locations_is_not_raw_modeled() {
        let blocks = [block(&[
            "%fp = getelementptr float, ptr addrspace(2) %floats, i64 0",
            "%hp = getelementptr half, ptr addrspace(2) %halves, i64 0",
            "%picked = select i1 %condition, ptr addrspace(2) %hp, ptr addrspace(2) %fp",
            "%merged = phi ptr addrspace(2) [ %picked, %left ], [ %fp, %right ]",
            "%value = load <4 x float>, ptr addrspace(2) %merged",
        ])];
        let eligible = BTreeMap::from([("%floats".to_string(), 28), ("%halves".to_string(), 29)]);

        assert!(reinterpret_mix_buffer_params(&blocks, &eligible).is_empty());
    }

    #[test]
    fn reinterpret_mix_with_unknown_pointer_producer_marks_no_partial_roots() {
        let blocks = [block(&[
            "%fp = getelementptr float, ptr addrspace(2) %floats, i64 0",
            "%opaque = call ptr addrspace(2) @pointer_source()",
            "%picked = select i1 %condition, ptr addrspace(2) %opaque, ptr addrspace(2) %fp",
            "%merged = phi ptr addrspace(2) [ %picked, %left ], [ %fp, %right ]",
            "%value = load <4 x i32>, ptr addrspace(2) %merged",
        ])];
        let eligible = BTreeMap::from([("%floats".to_string(), 1)]);

        assert!(reinterpret_mix_buffer_params(&blocks, &eligible).is_empty());
    }

    #[test]
    fn single_pointee_is_uniform() {
        assert_eq!(classify(&[LlType::Float]), NetworkClass::Uniform);
        assert_eq!(classify(&[]), NetworkClass::Uniform);
    }

    /// Reproduces the `05/b00a8a8d` shape: a single loop-carried phi `%144` whose incomings deref at
    /// mixed granularity (`%148` a `<4 x float>`-stride GEP, `%142` a scalar `float` GEP/bitcast) —
    /// the pointee-network wall behind M-B1 case 05. The union-find must fuse `%144`/`%148`/`%142`
    /// into ONE component, and the census must call it whole-vs-part with finest = `float`.
    #[test]
    fn groups_mixed_granularity_loop_phi_as_whole_vs_part() {
        let blocks = [block(&[
            "%144 = phi ptr [ %148, %loop ], [ %142, %entry ]",
            "%148 = getelementptr <4 x float>, ptr %144, i64 %125",
            "%142 = getelementptr float, ptr %129, i64 %140",
            "%147 = load <4 x float>, ptr %144",
        ])];
        let mut pointees = HashMap::new();
        pointees.insert(
            "%144".to_string(),
            LlType::Vector(Box::new(LlType::Float), 4),
        );
        pointees.insert(
            "%148".to_string(),
            LlType::Vector(Box::new(LlType::Float), 4),
        );
        pointees.insert("%142".to_string(), LlType::Float);

        let nets = analyze_pointer_networks(&blocks, &pointees);
        let net = net_containing(&nets, "%144");
        assert!(net.members.contains(&"%148".to_string()));
        assert!(net.members.contains(&"%142".to_string()));
        assert_eq!(net.class, NetworkClass::WholeVsPart(LlType::Float));
    }

    /// The access-width census (`analyze_networks_by_access`) must recover 05's whole-vs-part shape
    /// DIRECTLY from the IR loads/geps — no recorded `pointees` map — where the carrier census flattens
    /// it to `Uniform [Float]`. `%144` is loaded/stepped `<4 x float>`, `%142` stepped scalar `float`;
    /// the component censuses `{Vector(Float,4), Float}` → `WholeVsPart(Float)`.
    #[test]
    fn access_census_classifies_05_phi_as_whole_vs_part() {
        let blocks = [block(&[
            "%144 = phi ptr [ %148, %loop ], [ %142, %entry ]",
            "%148 = getelementptr <4 x float>, ptr %144, i64 %125",
            "%142 = getelementptr float, ptr %129, i64 %140",
            "%147 = load <4 x float>, ptr %144",
        ])];
        let nets = analyze_networks_by_access(&blocks);
        let net = net_containing(&nets, "%144");
        assert!(net.members.contains(&"%148".to_string()));
        assert!(net.members.contains(&"%142".to_string()));
        assert_eq!(net.class, NetworkClass::WholeVsPart(LlType::Float));
    }

    /// 05's real IR reaches the scalar arm through a `bitcast` of a `gep float`; the bitcast alias edge
    /// must fuse `%141`/`%142` into the network so the access census sees the `float` deref and reports
    /// whole-vs-part rather than a false Uniform.
    #[test]
    fn access_census_follows_bitcast_alias_to_scalar_arm() {
        let blocks = [block(&[
            "%144 = phi ptr addrspace(1) [ %148, %loop ], [ %142, %entry ]",
            "%148 = getelementptr <4 x float>, ptr addrspace(1) %144, i64 %125",
            "%141 = getelementptr float, ptr addrspace(1) %129, i64 %140",
            "%142 = bitcast ptr addrspace(1) %141 to ptr addrspace(1)",
            "%147 = load <4 x float>, ptr addrspace(1) %144",
        ])];
        let nets = analyze_networks_by_access(&blocks);
        let net = net_containing(&nets, "%144");
        assert!(net.members.contains(&"%141".to_string()));
        assert!(net.members.contains(&"%142".to_string()));
        assert_eq!(net.class, NetworkClass::WholeVsPart(LlType::Float));
    }

    /// A network whose members are all accessed at ONE width censuses `Uniform` under the access census
    /// — a store counts as a deref too, so a scalar-only loop pointer stays uniform.
    #[test]
    fn access_census_uniform_when_all_scalar() {
        let blocks = [block(&[
            "%p = phi ptr [ %q, %loop ], [ %r, %entry ]",
            "%q = getelementptr float, ptr %p, i64 1",
            "store float %v, ptr %r",
        ])];
        let nets = analyze_networks_by_access(&blocks);
        let net = net_containing(&nets, "%p");
        assert_eq!(net.class, NetworkClass::Uniform);
        assert_eq!(net.pointees, vec![LlType::Float]);
    }

    #[test]
    fn cross_member_widening_load_selects_byte_addressed_root() {
        let blocks = [block(&[
            "%field = getelementptr %Payload, ptr addrspace(2) %buffer, i64 0, i32 1",
            "%alias = bitcast ptr addrspace(2) %field to ptr addrspace(2)",
            "%value = load <3 x float>, ptr addrspace(2) %alias",
        ])];
        let params = HashSet::from(["%buffer".to_string()]);
        let named_types = HashMap::from([(
            "%Payload".to_string(),
            LlType::Struct(vec![LlType::Int(32), LlType::Float, LlType::Int(32)]),
        )]);
        assert_eq!(
            cross_member_widening_load_roots(&blocks, &params, &named_types),
            BTreeSet::from(["%buffer".to_string()])
        );
    }

    #[test]
    fn array_element_widening_load_keeps_structured_root() {
        let blocks = [block(&[
            "%element = getelementptr [4 x float], ptr addrspace(2) %buffer, i64 0, i64 1",
            "%value = load <3 x float>, ptr addrspace(2) %element",
        ])];
        let params = HashSet::from(["%buffer".to_string()]);
        assert!(cross_member_widening_load_roots(&blocks, &params, &HashMap::new()).is_empty());
    }

    #[test]
    fn groups_select_arms_into_one_network() {
        let blocks = [block(&[
            "%m = select i1 %c, ptr addrspace(1) %pa, ptr addrspace(1) %pb",
        ])];
        let mut pointees = HashMap::new();
        pointees.insert("%pa".to_string(), LlType::Float);
        pointees.insert("%pb".to_string(), LlType::Float);
        let nets = analyze_pointer_networks(&blocks, &pointees);
        let net = net_containing(&nets, "%m");
        assert!(net.members.contains(&"%pa".to_string()));
        assert!(net.members.contains(&"%pb".to_string()));
        assert_eq!(net.class, NetworkClass::Uniform);
    }

    #[test]
    fn transitively_fuses_chained_phis() {
        let blocks = [block(&[
            "%a = phi ptr [ %b, %x ], [ %c, %y ]",
            "%d = phi ptr [ %a, %z ], [ %e, %w ]",
        ])];
        let pointees = HashMap::new();
        let nets = analyze_pointer_networks(&blocks, &pointees);
        // All five names collapse into a single component through the shared %a edge.
        let net = net_containing(&nets, "%e");
        for name in ["%a", "%b", "%c", "%d", "%e"] {
            assert!(net.members.contains(&name.to_string()), "missing {name}");
        }
        assert_eq!(
            nets.iter()
                .filter(|n| n.members.contains(&"%a".to_string()))
                .count(),
            1
        );
    }

    #[test]
    fn disjoint_networks_stay_separate() {
        let blocks = [block(&[
            "%a = phi ptr [ %b, %x ], [ %c, %y ]",
            "%p = phi ptr [ %q, %x ], [ %r, %y ]",
        ])];
        let pointees = HashMap::new();
        let nets = analyze_pointer_networks(&blocks, &pointees);
        let na = net_containing(&nets, "%a");
        let np = net_containing(&nets, "%p");
        assert!(!na.members.contains(&"%q".to_string()));
        assert!(!np.members.contains(&"%b".to_string()));
    }

    /// A bare-scalar element GEP stepped by a NON-ZERO or DYNAMIC index flags its base as
    /// array-indexed: `%p` is stepped `getelementptr float, ptr %p, i64 1` (const non-zero) and `%q`
    /// `getelementptr float, ptr %q, i64 %i` (dynamic) — both address an array of `float`, so seeding
    /// their network's pointee as the scalar `float` would mis-declare the object.
    #[test]
    fn array_indexed_scalar_bases_flags_nonzero_and_dynamic_scalar_steps() {
        let blocks = [block(&[
            "%r1 = getelementptr float, ptr %p, i64 1",
            "%r2 = getelementptr float, ptr %q, i64 %i",
        ])];
        let flagged = array_indexed_scalar_bases(&blocks);
        assert!(flagged.contains("%p"));
        assert!(flagged.contains("%q"));
    }

    /// A leading constant-`0` step (identity / true aggregate descent) does NOT flag the base: a
    /// `getelementptr float, ptr %p, i64 0` is an identity pointer, and an aggregate `source_ty`
    /// (`%struct`/`[N x float]`) is declared correctly regardless of the seed, so neither base is a
    /// scalar-array object the seed would mis-declare.
    #[test]
    fn array_indexed_scalar_bases_ignores_identity_and_aggregate_steps() {
        let blocks = [block(&[
            "%r1 = getelementptr float, ptr %p, i64 0",
            "%r2 = getelementptr [8 x float], ptr %q, i64 0, i64 %i",
            "%r3 = getelementptr %struct.S, ptr %s, i64 0, i32 2",
        ])];
        let flagged = array_indexed_scalar_bases(&blocks);
        assert!(!flagged.contains("%p"));
        assert!(!flagged.contains("%q"));
        assert!(!flagged.contains("%s"));
    }
}
