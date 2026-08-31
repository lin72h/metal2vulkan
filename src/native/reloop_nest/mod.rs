//! A nesting structurizer for reducible control flow.
//!
//! The state-machine constructor in [`super::relooper`] can express any CFG, and that generality is
//! why it makes every non-entry block a sibling case of one `OpSwitch` inside one loop. Sibling
//! cases do not dominate each other, so every value crossing a block boundary has to be
//! register-demoted to a function-scope variable. On a large function the result is one loop
//! containing the whole program plus thousands of variables whose live ranges all span it. That
//! module is valid SPIR-V, and it is also a shape a driver's shader compiler can take unbounded
//! time on: promoting those variables back to SSA inside a single enormous loop makes the dominance
//! frontier work superlinear in (variables x blocks).
//!
//! Reducible control flow does not need that. This module derives the relooper shape tree (see
//! [`shape`]) and emits it as genuinely nested SPIR-V loop and selection constructs. Nesting
//! keeps the original CFG's paths, so ordinary SSA values are emitted untouched; only `OpPhi` —
//! whose meaning is its predecessor list, which is exactly what nesting rewrites — is demoted.
//!
//! Keeping paths is not the same as keeping dominance. An edge that has to leave more than one
//! construct is staged through a merge dispatch, and its destination is then reached from that
//! merge rather than from the block that left, so a definition inside the construct can stop
//! dominating a use beyond it. The state machine repaired that incidentally, by demoting every
//! crossing value. This pass does not, so [`structure_selected_functions`] checks the function it
//! emitted against the module's value-flow contract and discards it when the check fails.
//!
//! Dominance is not the whole contract either. An edge that leaves more than one construct stages
//! its destination in a function-scope flow variable that the dispatches it passes through read
//! back. That lives in memory, where dominance says nothing, so [`verify`] checks separately that
//! no path reaches a dispatch without having written it.
//!
//! It is a strict addition, not a replacement. Irreducible graphs, any shape the emitter cannot
//! express, and any nesting that does not survive those checks stay on the state-machine
//! constructor.

mod emit;
mod shape;
mod verify;

use super::relooper::{block_label, decode_term, Term, TypeCtx};
use crate::spirv_module::{Function, Module};
use shape::Graph;
use spirv::Word;
use std::collections::{BTreeMap, HashMap, HashSet};

/// Restructure every selected function whose control flow this module can nest. Returns the result
/// ids it rewrote, so the caller can leave the rest to the state-machine constructor.
pub(crate) fn structure_selected_functions(
    module: &mut Module,
    selected: &HashSet<Word>,
) -> HashSet<Word> {
    let mut next_id = module
        .header
        .as_ref()
        .map(|header| header.bound)
        .unwrap_or(1);
    // Verifying the emitted nesting needs the result type of every value it can name. Snapshot the
    // module's before its functions are borrowed mutably; result ids are unique module-wide, so the
    // ids each emitted function adds can simply be folded into the same map.
    let mut value_types = module
        .all_inst_iter()
        .filter_map(|instruction| Some((instruction.result_id?, instruction.result_type?)))
        .collect::<HashMap<_, _>>();
    let mut tc = TypeCtx::new(module, &mut next_id);
    let mut structured = HashSet::new();
    for function in &mut module.functions {
        let Some(id) = function.def.as_ref().and_then(|def| def.result_id) else {
            continue;
        };
        if !selected.contains(&id) || function.blocks.len() < 2 {
            continue;
        }
        let before = function.blocks.len();
        match structure_function(function, &mut tc) {
            Ok(emit::Structured {
                blocks,
                flow_variable,
            }) => {
                let candidate = std::mem::replace(&mut function.blocks, blocks);
                value_types.extend(function.blocks.iter().flat_map(|block| {
                    block.instructions.iter().filter_map(|instruction| {
                        Some((instruction.result_id?, instruction.result_type?))
                    })
                }));
                // A nesting is adopted only if it really is a well-formed construction. That is the
                // same question the caller asked to select construction in the first place, so it
                // is asked the same way, then extended to everything else a control-flow rewrite
                // owns: construct nesting, structured exits, and value flow.
                //
                // Value flow has to be checked because nesting keeps ordinary values in registers.
                // That is only sound where it preserved the paths that made their definitions
                // dominate their uses, and it does not always: an edge staged through a construct's
                // merge dispatch reaches its destination from that merge rather than from the block
                // it left. The state machine this pass replaces repaired such a function
                // incidentally, by demoting every crossing value; nesting keeps them, so it has to
                // prove the definitions still reach.
                let broken =
                    super::rewrites::blocks_have_unowned_selection_header(&function.blocks)
                        .then(|| "nesting left an unowned construct".to_string())
                        .or_else(|| {
                            super::rewrites::function_has_unowned_backedge(function)
                                .then(|| "nesting left an unowned back edge".to_string())
                        })
                        .or_else(|| {
                            super::owned_cfg::owned_function_construction_error(
                                function,
                                &value_types,
                            )
                        })
                        .or_else(|| {
                            verify::reads_the_flow_variable_unwritten(
                                &function.blocks,
                                flow_variable,
                            )
                        });
                if let Some(reason) = broken {
                    function.blocks = candidate;
                    if crate::env_vars::reloop_why() {
                        eprintln!("NEST-DECLINE blocks={before} {reason}");
                    }
                } else {
                    if crate::env_vars::reloop_why() {
                        eprintln!("NEST-ADOPT blocks={before} after={}", function.blocks.len());
                    }
                    structured.insert(id);
                }
            }
            Err(reason) => {
                if crate::env_vars::reloop_why() {
                    eprintln!("NEST-DECLINE blocks={before} {reason}");
                }
            }
        }
    }
    tc.flush(module);
    if let Some(header) = module.header.as_mut() {
        header.bound = next_id;
    }
    structured
}

fn structure_function(
    function: &Function,
    tc: &mut TypeCtx<'_>,
) -> Result<emit::Structured, String> {
    let graph = build_graph(function)?;
    if !graph.is_reducible() {
        return Err("irreducible control flow".to_string());
    }
    let tree = shape::calculate(&graph).ok_or_else(|| "empty shape tree".to_string())?;
    let structured = emit::structure_function(function, &graph, &tree, tc)?;
    Ok(emit::Structured {
        blocks: reverse_postorder(structured.blocks),
        flow_variable: structured.flow_variable,
    })
}

fn build_graph(function: &Function) -> Result<Graph, String> {
    let entry = function
        .blocks
        .first()
        .and_then(block_label)
        .ok_or_else(|| "function without an entry block".to_string())?;
    let mut successors = BTreeMap::new();
    for block in &function.blocks {
        let label = block_label(block).ok_or_else(|| "block without a label".to_string())?;
        let term = block
            .instructions
            .last()
            .and_then(decode_term)
            .ok_or_else(|| "unhandled terminator".to_string())?;
        let mut targets: Vec<Word> = Vec::new();
        let push = |target: Word, targets: &mut Vec<Word>| {
            if !targets.contains(&target) {
                targets.push(target);
            }
        };
        match term {
            Term::Branch(target) => push(target, &mut targets),
            Term::BranchCond(_, on_true, on_false) => {
                push(on_true, &mut targets);
                push(on_false, &mut targets);
            }
            Term::Switch(_, default, cases) => {
                push(default, &mut targets);
                for (_, case) in cases {
                    push(case, &mut targets);
                }
            }
            Term::Return | Term::ReturnValue(_) | Term::Unreachable | Term::Kill(_) => {}
        }
        successors.insert(label, targets);
    }
    for targets in successors.values() {
        for target in targets {
            if !successors.contains_key(target) {
                return Err("branch to a block outside the function".to_string());
            }
        }
    }
    Ok(Graph::new(entry, successors))
}

/// SPIR-V requires a block to appear before every block it dominates. A reverse postorder of the
/// emitted graph satisfies that; blocks no path reaches (declared merges for terminators whose arms
/// all leave the construct) keep their relative order at the end.
fn reverse_postorder(blocks: Vec<crate::spirv_module::Block>) -> Vec<crate::spirv_module::Block> {
    let mut by_label = BTreeMap::new();
    let mut order = Vec::new();
    for block in blocks {
        let Some(label) = block_label(&block) else {
            order.push((None, block));
            continue;
        };
        order.push((Some(label), block));
    }
    for (label, block) in &order {
        if let Some(label) = label {
            by_label.insert(*label, successors_of(block));
        }
    }
    let Some(entry) = order.first().and_then(|(label, _)| *label) else {
        return order.into_iter().map(|(_, block)| block).collect();
    };
    let mut seen = HashSet::new();
    let mut postorder = Vec::new();
    let mut stack = vec![(entry, 0usize)];
    seen.insert(entry);
    while let Some((label, index)) = stack.pop() {
        let targets = by_label.get(&label).cloned().unwrap_or_default();
        if index < targets.len() {
            stack.push((label, index + 1));
            let target = targets[index];
            if by_label.contains_key(&target) && seen.insert(target) {
                stack.push((target, 0));
            }
        } else {
            postorder.push(label);
        }
    }
    postorder.reverse();
    let mut position = postorder
        .iter()
        .enumerate()
        .map(|(index, label)| (*label, index))
        .collect::<BTreeMap<_, _>>();
    let unreachable_base = postorder.len();
    let mut trailing = unreachable_base;
    for (label, _) in &order {
        if let Some(label) = label {
            if !position.contains_key(label) {
                position.insert(*label, trailing);
                trailing += 1;
            }
        }
    }
    let mut sorted = order;
    sorted.sort_by_key(|(label, _)| {
        label
            .and_then(|label| position.get(&label).copied())
            .unwrap_or(usize::MAX)
    });
    sorted.into_iter().map(|(_, block)| block).collect()
}

fn successors_of(block: &crate::spirv_module::Block) -> Vec<Word> {
    let Some(term) = block.instructions.last().and_then(decode_term) else {
        return Vec::new();
    };
    let mut targets = Vec::new();
    match term {
        Term::Branch(target) => targets.push(target),
        Term::BranchCond(_, on_true, on_false) => {
            targets.push(on_true);
            targets.push(on_false);
        }
        Term::Switch(_, default, cases) => {
            targets.push(default);
            targets.extend(cases.into_iter().map(|(_, label)| label));
        }
        Term::Return | Term::ReturnValue(_) | Term::Unreachable | Term::Kill(_) => {}
    }
    targets
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    fn scratch() -> std::path::PathBuf {
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "metal2vulkan_reloop_nest_{}_{}",
            std::process::id(),
            n
        ));
        let _ = std::fs::create_dir_all(&dir);
        dir
    }

    /// Assemble spvasm through the local `spirv-as`; `None` when the toolchain is absent so the
    /// test no-ops rather than failing for an unrelated reason.
    fn assemble(spvasm: &str) -> Option<Vec<u8>> {
        if std::process::Command::new("spirv-as")
            .arg("--version")
            .output()
            .is_err()
        {
            return None;
        }
        let dir = scratch();
        let source = dir.join("in.spvasm");
        let out = dir.join("in.spv");
        std::fs::write(&source, spvasm).unwrap();
        let status = std::process::Command::new("spirv-as")
            .args(["--target-env", crate::tools::VULKAN_TARGET_ENV])
            .arg(&source)
            .arg("-o")
            .arg(&out)
            .output()
            .unwrap();
        assert!(
            status.status.success(),
            "spirv-as: {}",
            String::from_utf8_lossy(&status.stderr)
        );
        Some(std::fs::read(&out).unwrap())
    }

    fn validates(spv: &[u8]) -> bool {
        let dir = scratch();
        let path = dir.join("m.spv");
        std::fs::write(&path, spv).unwrap();
        let status = std::process::Command::new("spirv-val")
            .args(["--target-env", crate::tools::VULKAN_TARGET_ENV])
            .arg(&path)
            .output()
            .unwrap();
        if !status.status.success() {
            eprintln!("spirv-val: {}", String::from_utf8_lossy(&status.stderr));
        }
        status.status.success()
    }

    /// Nest every function of `spv` and return the reassembled module, or `None` when the
    /// structurizer declined.
    fn nested_bytes(spv: &[u8]) -> Option<Vec<u8>> {
        let mut module = crate::spirv_module::load_bytes(spv).expect("load");
        let selected = module
            .functions
            .iter()
            .filter_map(|function| function.def.as_ref().and_then(|def| def.result_id))
            .collect::<HashSet<_>>();
        let structured = structure_selected_functions(&mut module, &selected);
        if structured.is_empty() {
            return None;
        }
        crate::native::add_native_module_capabilities(&mut module);
        Some(
            module
                .assemble()
                .iter()
                .flat_map(|word| word.to_le_bytes())
                .collect(),
        )
    }

    fn disassemble(spv: &[u8]) -> String {
        let dir = scratch();
        let path = dir.join("m.spv");
        std::fs::write(&path, spv).unwrap();
        let output = std::process::Command::new("spirv-dis")
            .arg(&path)
            .output()
            .unwrap();
        String::from_utf8_lossy(&output.stdout).into_owned()
    }

    const LOOP_WITH_PHI: &str = r#"
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %main "main"
               OpExecutionMode %main LocalSize 1 1 1
       %void = OpTypeVoid
       %uint = OpTypeInt 32 0
       %bool = OpTypeBool
     %uint_0 = OpConstant %uint 0
     %uint_1 = OpConstant %uint 1
     %uint_8 = OpConstant %uint 8
      %fnvoid = OpTypeFunction %void
       %main = OpFunction %void None %fnvoid
      %entry = OpLabel
               OpBranch %header
     %header = OpLabel
          %i = OpPhi %uint %uint_0 %entry %next %latch
       %cmp = OpULessThan %bool %i %uint_8
               OpBranchConditional %cmp %body %exit
       %body = OpLabel
       %even = OpIEqual %bool %i %uint_1
               OpBranchConditional %even %odd %latch
        %odd = OpLabel
               OpBranch %latch
      %latch = OpLabel
       %next = OpIAdd %uint %i %uint_1
               OpBranch %header
       %exit = OpLabel
               OpReturn
               OpFunctionEnd
"#;

    #[test]
    fn nested_loop_with_phi_is_structured_and_validates() {
        let Some(spv) = assemble(LOOP_WITH_PHI) else {
            return;
        };
        let nested = nested_bytes(&spv).expect("reducible control flow is nested");
        assert!(validates(&nested));
        let asm = disassemble(&nested);
        assert!(asm.contains("OpLoopMerge"), "{asm}");
        // The whole function must not have become one dispatch: no state-machine switch.
        assert!(!asm.contains("OpSwitch"), "{asm}");
    }

    #[test]
    fn nesting_keeps_ordinary_values_in_registers() {
        let Some(spv) = assemble(LOOP_WITH_PHI) else {
            return;
        };
        let nested = nested_bytes(&spv).expect("nested");
        assert!(validates(&nested));
        let asm = disassemble(&nested);
        // No variable at all. The comparison, the add and the condition never left SSA, and the
        // induction phi went back to one on the synthesized loop header, where its two edges meet.
        let variables = asm.matches("OpVariable").count();
        assert_eq!(variables, 0, "{asm}");
        assert!(asm.contains("OpPhi"), "{asm}");
        assert!(asm.contains("OpIAdd"), "{asm}");
    }

    /// A value defined in one arm of a selection and used after the merge. The state-machine
    /// constructor repairs this by demoting every crossing value; the nesting keeps values in
    /// registers, so it cannot, and must decline rather than ship a use its definition never
    /// reaches. Earlier stages hand construction such functions, so this is a real input, not a
    /// hypothetical malformed one.
    const USE_WITHOUT_DOMINATING_DEFINITION: &str = r#"
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %main "main"
               OpExecutionMode %main LocalSize 1 1 1
       %void = OpTypeVoid
       %uint = OpTypeInt 32 0
       %bool = OpTypeBool
     %uint_0 = OpConstant %uint 0
     %uint_1 = OpConstant %uint 1
    %fnvoid = OpTypeFunction %void
       %main = OpFunction %void None %fnvoid
      %entry = OpLabel
       %cond = OpIEqual %bool %uint_0 %uint_1
               OpBranchConditional %cond %arm %other
        %arm = OpLabel
      %value = OpIAdd %uint %uint_0 %uint_1
               OpBranch %join
      %other = OpLabel
               OpBranch %join
       %join = OpLabel
       %used = OpIAdd %uint %value %uint_1
               OpReturn
               OpFunctionEnd
"#;

    #[test]
    fn nesting_declines_a_use_its_definition_does_not_dominate() {
        let Some(spv) = assemble(USE_WITHOUT_DOMINATING_DEFINITION) else {
            return;
        };
        assert!(
            nested_bytes(&spv).is_none(),
            "a function whose value flow the nesting cannot honour must stay on the state machine"
        );
    }

    /// `while (c) { if (d) { p(); if (e) break; } q(); }` — the break leaves a selection AND the
    /// loop on one edge, which SPIR-V cannot express as a single branch.
    const NESTED_BREAK: &str = r#"
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %main "main"
               OpExecutionMode %main LocalSize 1 1 1
       %void = OpTypeVoid
       %uint = OpTypeInt 32 0
       %bool = OpTypeBool
     %uint_0 = OpConstant %uint 0
     %uint_1 = OpConstant %uint 1
     %uint_3 = OpConstant %uint 3
     %uint_8 = OpConstant %uint 8
     %fnvoid = OpTypeFunction %void
       %main = OpFunction %void None %fnvoid
      %entry = OpLabel
               OpBranch %header
     %header = OpLabel
          %i = OpPhi %uint %uint_0 %entry %next %latch
          %c = OpULessThan %bool %i %uint_8
               OpBranchConditional %c %body %exit
       %body = OpLabel
          %d = OpIEqual %bool %i %uint_1
               OpBranchConditional %d %then %q
       %then = OpLabel
          %p = OpIMul %uint %i %uint_3
          %e = OpIEqual %bool %p %uint_3
               OpBranchConditional %e %exit %q
          %q = OpLabel
               OpBranch %latch
      %latch = OpLabel
       %next = OpIAdd %uint %i %uint_1
               OpBranch %header
       %exit = OpLabel
               OpReturn
               OpFunctionEnd
"#;

    #[test]
    fn a_break_leaving_two_constructs_is_staged_and_validates() {
        let Some(spv) = assemble(NESTED_BREAK) else {
            return;
        };
        let nested = nested_bytes(&spv).expect("reducible control flow is nested");
        assert!(validates(&nested));
        let asm = disassemble(&nested);
        assert!(asm.contains("OpLoopMerge"), "{asm}");
        // The loop is a real loop, not a case of a whole-function dispatch.
        assert_eq!(asm.matches("OpLoopMerge").count(), 1, "{asm}");
    }

    #[test]
    fn irreducible_control_flow_is_declined() {
        let irreducible = r#"
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %main "main"
               OpExecutionMode %main LocalSize 1 1 1
       %void = OpTypeVoid
       %bool = OpTypeBool
       %true = OpConstantTrue %bool
      %fnvoid = OpTypeFunction %void
       %main = OpFunction %void None %fnvoid
      %entry = OpLabel
               OpSelectionMerge %b None
               OpBranchConditional %true %a %b
          %a = OpLabel
               OpBranch %b
          %b = OpLabel
               OpBranchConditional %true %a %exit
       %exit = OpLabel
               OpReturn
               OpFunctionEnd
"#;
        // spirv-as accepts this only because the module is not validated at assembly time; the
        // point is that the structurizer must recognize the graph and decline rather than emit a
        // nesting that does not preserve its entries.
        let Some(spv) = assemble(irreducible) else {
            return;
        };
        let mut module = crate::spirv_module::load_bytes(&spv).expect("load");
        let function = module.functions.first().expect("one function");
        let graph = build_graph(function).expect("graph");
        assert!(!graph.is_reducible());
        let selected = module
            .functions
            .iter()
            .filter_map(|function| function.def.as_ref().and_then(|def| def.result_id))
            .collect::<HashSet<_>>();
        assert!(structure_selected_functions(&mut module, &selected).is_empty());
    }
}
