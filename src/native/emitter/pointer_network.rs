//! Pointer-network connected-component analysis (M-A2 / M-B1 structural prerequisite).
//!
//! Today each pointer phi/select is reconciled INDEPENDENTLY in `pointer_merge_meta` — the
//! merge-participant sets (`pointer_phi_values`, `pointer_phi_incoming_values`, `selected_pointers`)
//! are flat membership sets, NOT a grouping. That is the wall behind the three M-B1 blockers and the
//! unsound M-A2(a)/(b) read-side flags: a pointee typed differently at two def sites of ONE
//! phi/select network (e.g. `05/b00a8a8d`'s `%144`, a loop-carried device pointer whose incomings
//! deref as `<4 x float>` on one arm and scalar `float` on the other) errors `pointer merge pointee
//! mismatch Float vs Vector(Float,4)`, and no read-side override can fix it because SPIR-V logical
//! addressing forbids `OpBitcast` between pointer types.
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
//! Analysis-only: this module reads the IR and the emitter's recorded pointees; it changes no bytes.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use crate::native::cfg::BodyBlock;
use crate::native::ir::{LlType, LlValue};

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
            if let Some((ty, incomings)) = &inst.phi_incoming {
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
            // the access census — 05's scalar `float` arm reaches its phi through
            // `%142 = bitcast (%141 = gep float …)`; without this edge `%141`'s Float access is invisible
            // and the whole-vs-part network masquerades as Uniform.
            if let Some((src, dst)) = &inst.bitcast {
                if matches!(src.ty, LlType::Ptr(_)) && dst.trim_start().starts_with("ptr") {
                    if let LlValue::Local(local) = &src.value {
                        edges.push((name.clone(), local.clone()));
                    }
                }
            }
        }
    }
}

/// Whether a typed operand is `ptr`-typed (the dual of the line scan's arm `starts_with("ptr")`).
fn is_ptr_operand(op: &crate::native::tir::TirOperand) -> bool {
    op.as_typed_value()
        .is_some_and(|tv| matches!(tv.ty, LlType::Ptr(_)))
}

/// Group the pointer SSA names of `blocks` into connected components over the phi/select edges.
/// Shared by every census; each returned member list is sorted and deduped.
fn build_components(blocks: &[BodyBlock]) -> Vec<Vec<String>> {
    let edges = pointer_edges(blocks);
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
        // result on `inst.gep`.
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
                        if let Some(gep) = &inst.gep {
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
        // `inst.gep`.
        if let Some(carrier) = &block.typed {
            for inst in &carrier.insts {
                if let Some(gep) = &inst.gep {
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
            typed,
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
