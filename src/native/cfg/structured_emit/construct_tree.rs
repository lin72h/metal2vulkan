//! Bounded ownership and edge-route analysis for the reject-only construct-tree re-nester.
//!
//! This module does not participate in production admission yet. It converts an explicit parent-first
//! construct claim graph into one owner per original block and one finite ancestor route per original
//! edge; its `renest` child materializes the bounded typed regional construct used by the R1 fixture
//! proof. R2+ derives these claims from one live reject class at a time.

use super::{synthetic_block, BlockRole, BodyBlock};
use std::collections::HashSet;

pub(in crate::native) mod renest;

const ROUTE_PREFIX: &str = "%metal2vulkan.ctroute.";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::native) enum ConstructKind {
    Root,
    Loop,
    Selection,
    Switch,
    Arm,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(in crate::native) struct ConstructNode {
    pub(in crate::native) name: String,
    pub(in crate::native) parent: Option<usize>,
    pub(in crate::native) kind: ConstructKind,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(in crate::native) struct ClaimedBlock {
    pub(in crate::native) name: String,
    pub(in crate::native) claims: Vec<usize>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::native) struct OriginalEdge {
    pub(in crate::native) from: usize,
    pub(in crate::native) to: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(in crate::native) struct ConstructRoute {
    pub(in crate::native) edge: OriginalEdge,
    /// Constructs exited from the source owner toward the least common ancestor.
    pub(in crate::native) exits: Vec<usize>,
    /// Constructs entered from the least common ancestor toward the target owner.
    pub(in crate::native) enters: Vec<usize>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(in crate::native) struct ConstructTreePlan {
    /// Canonical least-common-ancestor owner for each original block.
    pub(in crate::native) owners: Vec<usize>,
    /// Exactly one route for each original input edge, in input order.
    pub(in crate::native) routes: Vec<ConstructRoute>,
    pub(in crate::native) max_depth: usize,
    pub(in crate::native) ownership_lifts: usize,
    /// Conservative materialized block ceiling: `V + 3C + E(2D + 1)`.
    pub(in crate::native) block_bound: usize,
}

fn ancestry(nodes: &[ConstructNode], mut node: usize) -> Vec<usize> {
    let mut out = vec![node];
    while let Some(parent) = nodes[node].parent {
        out.push(parent);
        node = parent;
    }
    out
}

fn least_common_ancestor(nodes: &[ConstructNode], claims: &[usize]) -> usize {
    let first = ancestry(nodes, claims[0]);
    first
        .into_iter()
        .find(|candidate| {
            claims
                .iter()
                .all(|claim| ancestry(nodes, *claim).contains(candidate))
        })
        .expect("validated tree always has root as a common ancestor")
}

/// Plan canonical ownership and finite boundary routes from an explicit construct claim graph.
///
/// The graph is intentionally index-based: later claim derivation can construct it without assigning
/// protocol-like numeric meanings. Indices are local identities only. Analysis reads the supplied
/// nodes/blocks/edges once and never analyzes generated output.
pub(in crate::native) fn plan_construct_tree(
    nodes: &[ConstructNode],
    blocks: &[ClaimedBlock],
    edges: &[OriginalEdge],
) -> Result<ConstructTreePlan, String> {
    let Some(root) = nodes.first() else {
        return Err("construct-tree:no-root".to_string());
    };
    if root.parent.is_some() || root.kind != ConstructKind::Root {
        return Err("construct-tree:invalid-root".to_string());
    }

    let mut depths = vec![0usize; nodes.len()];
    for (index, node) in nodes.iter().enumerate().skip(1) {
        let Some(parent) = node.parent else {
            return Err(format!(
                "construct-tree:detached-construct name={}",
                node.name
            ));
        };
        if parent >= index {
            return Err(format!(
                "construct-tree:not-parent-first name={} parent={parent} index={index}",
                node.name
            ));
        }
        depths[index] = depths[parent]
            .checked_add(1)
            .ok_or_else(|| "construct-tree:depth-overflow".to_string())?;
    }
    let max_depth = depths.iter().copied().max().unwrap_or(0);

    let mut owners = Vec::with_capacity(blocks.len());
    let mut ownership_lifts = 0usize;
    for block in blocks {
        if block.claims.is_empty() {
            return Err(format!("construct-tree:no-claim block={}", block.name));
        }
        if let Some(claim) = block
            .claims
            .iter()
            .copied()
            .find(|claim| *claim >= nodes.len())
        {
            return Err(format!(
                "construct-tree:claim-out-of-range block={} claim={claim} constructs={}",
                block.name,
                nodes.len()
            ));
        }
        let owner = least_common_ancestor(nodes, &block.claims);
        let deepest_claim = block
            .claims
            .iter()
            .map(|claim| depths[*claim])
            .max()
            .unwrap_or(depths[owner]);
        ownership_lifts = ownership_lifts
            .checked_add(deepest_claim.saturating_sub(depths[owner]))
            .ok_or_else(|| "construct-tree:lift-overflow".to_string())?;
        owners.push(owner);
    }

    let mut routes = Vec::with_capacity(edges.len());
    for &edge in edges {
        if edge.from >= blocks.len() || edge.to >= blocks.len() {
            return Err(format!(
                "construct-tree:edge-out-of-range from={} to={} blocks={}",
                edge.from,
                edge.to,
                blocks.len()
            ));
        }
        let from_path = ancestry(nodes, owners[edge.from]);
        let to_path = ancestry(nodes, owners[edge.to]);
        let lca = *from_path
            .iter()
            .find(|node| to_path.contains(node))
            .expect("validated tree always shares root");
        let exits = from_path
            .into_iter()
            .take_while(|node| *node != lca)
            .collect();
        let mut enters: Vec<usize> = to_path
            .into_iter()
            .take_while(|node| *node != lca)
            .collect();
        enters.reverse();
        routes.push(ConstructRoute {
            edge,
            exits,
            enters,
        });
    }

    let edge_factor = max_depth
        .checked_mul(2)
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| "construct-tree:bound-overflow".to_string())?;
    let block_bound = blocks
        .len()
        .checked_add(
            nodes
                .len()
                .checked_mul(3)
                .ok_or_else(|| "construct-tree:bound-overflow".to_string())?,
        )
        .and_then(|value| {
            edges
                .len()
                .checked_mul(edge_factor)
                .and_then(|edge_blocks| value.checked_add(edge_blocks))
        })
        .ok_or_else(|| "construct-tree:bound-overflow".to_string())?;

    debug_assert!(ownership_lifts <= blocks.len().saturating_mul(max_depth));
    debug_assert_eq!(routes.len(), edges.len());

    Ok(ConstructTreePlan {
        owners,
        routes,
        max_depth,
        ownership_lifts,
        block_bound,
    })
}

/// Materialize every non-empty planned boundary route as a typed chain of pass-through gateways.
///
/// Each original block is cloned into the transaction exactly once and never duplicated. For an
/// original edge `A -> B`, the source terminator targets the first gateway, each gateway targets the
/// next, and the last targets `B`; `B`'s phi predecessor `A` is rewritten to the last gateway. This
/// preserves both lazy execution and the original incoming value without a global spill. The function
/// is transactional: every plan/source invariant is checked before the cloned CFG is mutated.
pub(in crate::native) fn materialize_construct_routes(
    blocks: &[BodyBlock],
    plan: &ConstructTreePlan,
) -> Result<Vec<BodyBlock>, String> {
    if blocks.len() != plan.owners.len() {
        return Err(format!(
            "construct-tree:block-plan-mismatch blocks={} owners={}",
            blocks.len(),
            plan.owners.len()
        ));
    }
    let mut names = HashSet::with_capacity(blocks.len());
    for block in blocks {
        if !names.insert(block.name.as_str()) {
            return Err(format!(
                "construct-tree:duplicate-block-name block={}",
                block.name
            ));
        }
        if block.typed.is_none() {
            return Err(format!(
                "construct-tree:missing-carrier block={}",
                block.name
            ));
        }
    }

    let mut seen_edges = HashSet::with_capacity(plan.routes.len());
    for route in &plan.routes {
        let edge = route.edge;
        if edge.from >= blocks.len() || edge.to >= blocks.len() {
            return Err(format!(
                "construct-tree:planned-edge-out-of-range from={} to={} blocks={}",
                edge.from,
                edge.to,
                blocks.len()
            ));
        }
        if !seen_edges.insert((edge.from, edge.to)) {
            return Err(format!(
                "construct-tree:duplicate-edge from={} to={}",
                blocks[edge.from].name, blocks[edge.to].name
            ));
        }
        let source = blocks[edge.from]
            .typed
            .as_ref()
            .expect("carrier checked above");
        if !source
            .terminator
            .successors()
            .iter()
            .any(|successor| *successor == blocks[edge.to].name)
        {
            return Err(format!(
                "construct-tree:not-a-source-edge from={} to={}",
                blocks[edge.from].name, blocks[edge.to].name
            ));
        }
    }

    let mut out = blocks.to_vec();
    let mut gateways = Vec::new();
    for (route_index, route) in plan.routes.iter().enumerate() {
        let step_count = route.exits.len() + route.enters.len();
        if step_count == 0 {
            continue;
        }
        let source_name = blocks[route.edge.from].name.clone();
        let target_name = blocks[route.edge.to].name.clone();
        let route_names: Vec<String> = (0..step_count)
            .map(|step| format!("{ROUTE_PREFIX}{route_index}.{step}"))
            .collect();
        if let Some(source) = out[route.edge.from].typed.as_mut() {
            source.redirect_successor(&target_name, &route_names[0]);
        }
        if let Some(target) = out[route.edge.to].typed.as_mut() {
            target.rewrite_phi_predecessor(
                &source_name,
                route_names.last().expect("non-empty route"),
            );
        }
        for (step, name) in route_names.iter().enumerate() {
            let target = route_names
                .get(step + 1)
                .cloned()
                .unwrap_or_else(|| target_name.clone());
            gateways.push(synthetic_block(
                name.clone(),
                vec![format!("br label {target}")],
                BlockRole::ConstructTreeRoute,
            ));
        }
    }
    out.extend(gateways);
    if out.len() > plan.block_bound {
        return Err(format!(
            "construct-tree:materialization-bound blocks={} bound={}",
            out.len(),
            plan.block_bound
        ));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::native::tir;
    use std::collections::HashMap;

    fn bb(name: &str, lines: &[&str]) -> BodyBlock {
        let lines: Vec<String> = lines.iter().map(|line| line.to_string()).collect();
        BodyBlock {
            name: name.to_string(),
            role: BlockRole::Normal,
            typed: tir::lower_block_carrier(name, &lines, &HashMap::new()),
        }
    }

    fn deep_single_owner_nodes() -> Vec<ConstructNode> {
        vec![
            ConstructNode {
                name: "root".to_string(),
                parent: None,
                kind: ConstructKind::Root,
            },
            ConstructNode {
                name: "selection".to_string(),
                parent: Some(0),
                kind: ConstructKind::Selection,
            },
            ConstructNode {
                name: "arm".to_string(),
                parent: Some(1),
                kind: ConstructKind::Arm,
            },
            ConstructNode {
                name: "inner".to_string(),
                parent: Some(2),
                kind: ConstructKind::Selection,
            },
        ]
    }

    fn linear_claims() -> Vec<ClaimedBlock> {
        vec![
            ClaimedBlock {
                name: "%entry".to_string(),
                claims: vec![3],
            },
            ClaimedBlock {
                name: "%def".to_string(),
                claims: vec![3],
            },
            ClaimedBlock {
                name: "%use".to_string(),
                claims: vec![3],
            },
        ]
    }

    #[test]
    fn r1_core_rejects_non_parent_first_constructs() {
        let nodes = vec![
            ConstructNode {
                name: "root".to_string(),
                parent: None,
                kind: ConstructKind::Root,
            },
            ConstructNode {
                name: "bad".to_string(),
                parent: Some(1),
                kind: ConstructKind::Selection,
            },
        ];
        assert_eq!(
            plan_construct_tree(&nodes, &[], &[]),
            Err("construct-tree:not-parent-first name=bad parent=1 index=1".to_string())
        );
    }

    #[test]
    fn r1_core_lifts_sibling_claim_to_parent_and_routes_once() {
        let nodes = vec![
            ConstructNode {
                name: "root".to_string(),
                parent: None,
                kind: ConstructKind::Root,
            },
            ConstructNode {
                name: "selection".to_string(),
                parent: Some(0),
                kind: ConstructKind::Selection,
            },
            ConstructNode {
                name: "left".to_string(),
                parent: Some(1),
                kind: ConstructKind::Arm,
            },
            ConstructNode {
                name: "right".to_string(),
                parent: Some(1),
                kind: ConstructKind::Arm,
            },
        ];
        let blocks = vec![
            ClaimedBlock {
                name: "left-block".to_string(),
                claims: vec![2],
            },
            ClaimedBlock {
                name: "shared".to_string(),
                claims: vec![2, 3],
            },
        ];
        let edges = vec![OriginalEdge { from: 0, to: 1 }];
        let plan = plan_construct_tree(&nodes, &blocks, &edges).unwrap();
        assert_eq!(plan.owners, vec![2, 1]);
        assert_eq!(plan.routes.len(), 1);
        assert_eq!(plan.routes[0].exits, vec![2]);
        assert!(plan.routes[0].enters.is_empty());
        assert_eq!(plan.ownership_lifts, 1);
    }

    #[test]
    fn r1_materializer_splits_boundary_edges_and_rewrites_phi_predecessors() {
        let nodes = vec![
            ConstructNode {
                name: "root".to_string(),
                parent: None,
                kind: ConstructKind::Root,
            },
            ConstructNode {
                name: "selection".to_string(),
                parent: Some(0),
                kind: ConstructKind::Selection,
            },
            ConstructNode {
                name: "left".to_string(),
                parent: Some(1),
                kind: ConstructKind::Arm,
            },
            ConstructNode {
                name: "right".to_string(),
                parent: Some(1),
                kind: ConstructKind::Arm,
            },
        ];
        let blocks = vec![
            bb("%entry", &["br i1 %c, label %left, label %right"]),
            bb("%left", &["br label %merge"]),
            bb("%right", &["br label %merge"]),
            bb(
                "%merge",
                &["%value = phi i32 [ 7, %left ], [ 9, %right ]", "ret void"],
            ),
        ];
        let claims = vec![
            ClaimedBlock {
                name: "%entry".to_string(),
                claims: vec![1],
            },
            ClaimedBlock {
                name: "%left".to_string(),
                claims: vec![2],
            },
            ClaimedBlock {
                name: "%right".to_string(),
                claims: vec![3],
            },
            ClaimedBlock {
                name: "%merge".to_string(),
                claims: vec![2, 3],
            },
        ];
        let edges = vec![
            OriginalEdge { from: 0, to: 1 },
            OriginalEdge { from: 0, to: 2 },
            OriginalEdge { from: 1, to: 3 },
            OriginalEdge { from: 2, to: 3 },
        ];
        let plan = plan_construct_tree(&nodes, &claims, &edges).unwrap();
        let out = materialize_construct_routes(&blocks, &plan).unwrap();

        assert_eq!(
            out.iter()
                .filter(|block| block.name.starts_with(ROUTE_PREFIX))
                .count(),
            4,
            "one typed gateway for each single-level boundary edge"
        );
        assert_eq!(
            out.iter()
                .filter(|block| !block.name.starts_with(ROUTE_PREFIX))
                .map(|block| block.name.as_str())
                .collect::<Vec<_>>(),
            ["%entry", "%left", "%right", "%merge"],
            "every original body is retained exactly once"
        );
        assert!(out.iter().all(|block| block.typed.is_some()));

        let merge = out.iter().find(|block| block.name == "%merge").unwrap();
        let phi = merge
            .typed
            .as_ref()
            .unwrap()
            .insts
            .iter()
            .find(|inst| inst.result.as_deref() == Some("%value"))
            .and_then(|inst| inst.phi_incoming.as_ref())
            .map(|(_, incoming)| incoming)
            .unwrap();
        assert_eq!(phi[0].1, format!("{ROUTE_PREFIX}2.0"));
        assert_eq!(phi[1].1, format!("{ROUTE_PREFIX}3.0"));
        assert!(
            super::super::structured_plan(&out).is_some(),
            "typed gateway insertion must preserve an already-structured diamond"
        );
    }

    #[test]
    fn r1_materializer_rejects_an_edge_absent_from_the_typed_cfg() {
        let nodes = vec![ConstructNode {
            name: "root".to_string(),
            parent: None,
            kind: ConstructKind::Root,
        }];
        let blocks = vec![bb("%a", &["ret void"]), bb("%b", &["ret void"])];
        let claims = vec![
            ClaimedBlock {
                name: "%a".to_string(),
                claims: vec![0],
            },
            ClaimedBlock {
                name: "%b".to_string(),
                claims: vec![0],
            },
        ];
        let plan =
            plan_construct_tree(&nodes, &claims, &[OriginalEdge { from: 0, to: 1 }]).unwrap();
        assert_eq!(
            materialize_construct_routes(&blocks, &plan).unwrap_err(),
            "construct-tree:not-a-source-edge from=%a to=%b"
        );
    }

    #[test]
    fn r1_renester_carries_cross_case_scalar_values_through_typed_state_slots() {
        let nodes = deep_single_owner_nodes();
        let blocks = vec![
            bb("%entry", &["br label %def"]),
            bb("%def", &["%x = add i32 1, 2", "br label %use"]),
            bb("%use", &["%y = add i32 %x, 3", "ret void"]),
        ];
        let edges = vec![
            OriginalEdge { from: 0, to: 1 },
            OriginalEdge { from: 1, to: 2 },
        ];
        let plan = plan_construct_tree(&nodes, &linear_claims(), &edges).unwrap();
        let out = renest::materialize_construct_tree_roles(&blocks, &plan).unwrap();

        let header = out
            .iter()
            .find(|block| block.name == "%metal2vulkan.ct.header")
            .unwrap()
            .typed
            .as_ref()
            .unwrap();
        assert!(
            header
                .insts
                .iter()
                .any(|inst| inst.result.as_deref() == Some("%metal2vulkan.ct.vslot.0")),
            "the cross-case scalar def must have a current state phi"
        );

        let use_block = out
            .iter()
            .find(|block| block.name == "%use")
            .unwrap()
            .typed
            .as_ref()
            .unwrap();
        let y = use_block
            .insts
            .iter()
            .find(|inst| inst.result.as_deref() == Some("%y"))
            .unwrap();
        assert_eq!(
            y.uses,
            vec!["%metal2vulkan.ct.vslot.0".to_string()],
            "later cases read the current state slot, not the out-of-scope source SSA name"
        );

        let def_block = out
            .iter()
            .find(|block| block.name == "%def")
            .unwrap()
            .typed
            .as_ref()
            .unwrap();
        assert!(
            def_block
                .insts
                .iter()
                .any(|inst| inst.result.as_deref() == Some("%x")),
            "the defining case must keep the original SSA result instead of renaming the def"
        );

        let merge = out
            .iter()
            .find(|block| block.name == "%metal2vulkan.ct.merge")
            .unwrap()
            .typed
            .as_ref()
            .unwrap();
        let next_slot = merge
            .insts
            .iter()
            .find(|inst| inst.result.as_deref() == Some("%metal2vulkan.ct.nextvslot.0"))
            .and_then(|inst| inst.phi_incoming.as_ref())
            .map(|(_, incoming)| incoming)
            .unwrap();
        assert!(
            next_slot
                .iter()
                .any(|(value, pred)| matches!(value, crate::native::ir::LlValue::Local(name) if name == "%x")
                    && pred == "%metal2vulkan.ct.casejoin.1"),
            "the defining case must publish its local result into the next state slot"
        );
    }

    #[test]
    fn r1_renester_keeps_cross_case_pointer_values_rejected() {
        let nodes = deep_single_owner_nodes();
        let blocks = vec![
            bb("%entry", &["br label %def"]),
            bb("%def", &["%p = alloca i32, align 4", "br label %use"]),
            bb("%use", &["%v = load i32, ptr %p, align 4", "ret void"]),
        ];
        let edges = vec![
            OriginalEdge { from: 0, to: 1 },
            OriginalEdge { from: 1, to: 2 },
        ];
        let plan = plan_construct_tree(&nodes, &linear_claims(), &edges).unwrap();
        let error = renest::materialize_construct_tree_roles(&blocks, &plan).unwrap_err();
        assert_eq!(
            error,
            "construct-tree:cross-case-pointer-value owner=%def value=%p"
        );
    }
}
