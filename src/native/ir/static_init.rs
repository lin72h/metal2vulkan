use super::*;
use crate::native::cfg::BodyBlock;
use crate::native::tir::{RetEmit, TirTerminator};

struct StaticInitializer {
    ordinal: usize,
    name: String,
    blocks: Vec<BodyBlock>,
}

fn terminator_targets(term: &TirTerminator, target: &str) -> bool {
    match term {
        TirTerminator::Br(label) => label == target,
        TirTerminator::BrCond { t, f, .. } => t == target || f == target,
        TirTerminator::Switch { default, cases, .. } => {
            default == target || cases.iter().any(|(_, label)| label == target)
        }
        TirTerminator::Ret(_) | TirTerminator::Unreachable => false,
    }
}

fn eligible_static_initializer(function: &LlFunction) -> Option<Vec<BodyBlock>> {
    if !function.params.is_empty() {
        return None;
    }
    let carriers = function
        .blocks
        .iter()
        .map(|block| block.typed.as_ref())
        .collect::<Option<Vec<_>>>()?;
    if carriers.iter().any(|block| {
        block.insts.iter().any(|instruction| {
            instruction.opcode == "alloca"
                || (matches!(
                    instruction.opcode.as_str(),
                    "call" | "tail" | "musttail" | "notail"
                ) && instruction.call.is_none())
        })
    }) {
        return None;
    }
    let returns = carriers
        .iter()
        .filter(|block| {
            matches!(block.terminator, TirTerminator::Ret(None))
                && matches!(block.ret, RetEmit::Void)
        })
        .count();
    if returns != 1
        || carriers.iter().any(|block| {
            matches!(
                block.terminator,
                TirTerminator::Ret(Some(_)) | TirTerminator::Unreachable
            )
        })
    {
        return None;
    }
    if carriers.len() > 1 {
        let entry = &carriers[0].label;
        if carriers
            .iter()
            .any(|block| terminator_targets(&block.terminator, entry))
        {
            return None;
        }
    }
    Some(function.blocks.clone())
}

impl LlModule {
    /// Inline the one-block suffix of AIR static initializers into the typed entry.
    ///
    /// Multi-block bodies retain their independently emitted CFG and join the producer's complete
    /// typed SPIR-V call-graph closure immediately before serialization. Keeping this early
    /// transform straight-line-only avoids combining caller/callee inference domains before
    /// emission.
    pub(in crate::native) fn inline_simple_static_initializers(&mut self) {
        let Some(entry_name) = self
            .entry_name
            .clone()
            .or_else(|| self.functions.first().map(|function| function.name.clone()))
        else {
            return;
        };
        let Some(entry_index) = self
            .functions
            .iter()
            .position(|function| function.name == entry_name)
        else {
            return;
        };
        if self.functions[entry_index]
            .blocks
            .first()
            .and_then(|block| block.typed.as_ref())
            .is_none()
        {
            return;
        }

        let initializers = self
            .functions
            .iter()
            .enumerate()
            .filter(|(_, function)| {
                function.name != entry_name && function.name.starts_with("_GLOBAL__sub_I")
            })
            .map(|(ordinal, function)| {
                (
                    ordinal,
                    function.name.clone(),
                    (function.blocks.len() == 1)
                        .then(|| eligible_static_initializer(function))
                        .flatten(),
                )
            })
            .collect::<Vec<_>>();
        // Residual calls are injected at entry start. Therefore only an eligible SUFFIX can move
        // ahead of the seam without reordering around an earlier/later callful constructor.
        let suffix_start = initializers
            .iter()
            .rposition(|(_, _, blocks)| blocks.is_none())
            .map_or(0, |index| index + 1);
        let initializers = initializers
            .into_iter()
            .skip(suffix_start)
            .filter_map(|(ordinal, name, blocks)| {
                blocks.map(|blocks| StaticInitializer {
                    ordinal,
                    name,
                    blocks,
                })
            })
            .collect::<Vec<_>>();
        if initializers.is_empty() {
            return;
        }

        let source_pointees = self.ptr_pointees.clone();
        let mut pointees = Vec::new();
        let mut preinlined = Vec::new();
        let entry = &mut self.functions[entry_index];
        let mut cursor = 0;
        for StaticInitializer {
            ordinal,
            name,
            mut blocks,
        } in initializers
        {
            let mut rename = blocks
                .iter()
                .filter_map(|block| block.typed.as_ref())
                .flat_map(|block| block.insts.iter())
                .filter_map(|instruction| instruction.result.as_ref())
                .map(|result| {
                    (
                        result.clone(),
                        format!(
                            "%metal2vulkan.static_init.{ordinal}.{}",
                            result.trim_start_matches('%')
                        ),
                    )
                })
                .collect::<HashMap<_, _>>();
            for block in &blocks {
                rename.insert(
                    block.name.clone(),
                    format!(
                        "%metal2vulkan.static_init.{ordinal}.block.{}",
                        block.name.trim_start_matches('%')
                    ),
                );
            }
            for block in &mut blocks {
                let typed = block
                    .typed
                    .as_mut()
                    .expect("eligible static initializer has typed blocks");
                typed.rename(&rename);
                block.name = typed.label.clone();
            }
            for ((function, local), pointee) in &source_pointees {
                if function == &name {
                    if let Some(renamed) = rename.get(local) {
                        pointees.push(((entry_name.clone(), renamed.clone()), pointee.clone()));
                    }
                }
            }
            preinlined.push(name);

            let block = blocks[0]
                .typed
                .as_mut()
                .expect("eligible static initializer has a typed block");
            let inserted = block.insts.len();
            entry.blocks[0]
                .typed
                .as_mut()
                .expect("entry carrier checked before static-initializer mutation")
                .insts
                .splice(cursor..cursor, block.insts.drain(..));
            cursor += inserted;
        }

        self.ptr_pointees.extend(pointees);
        self.preinlined_static_initializers.extend(preinlined);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn static_initializer_suffix_carries_direct_calls_in_source_order() {
        let ll = r#"
@a = internal addrspace(2) global i8 0

define internal void @_GLOBAL__sub_I_leaf() {
  store i8 1, ptr addrspace(2) @a
  ret void
}

define internal void @_GLOBAL__sub_I_callful() {
  call void @air.test()
  ret void
}

define void @main() {
  store i8 2, ptr addrspace(2) @a
  ret void
}

declare void @air.test()
"#;
        let mut module =
            LlModule::parse_with_stage_meta(ll, None, Some("main")).expect("typed module");
        module.inline_simple_static_initializers();
        assert_eq!(module.preinlined_static_initializers.len(), 2);
        let entry = module
            .functions
            .iter()
            .find(|function| function.name == "main")
            .expect("entry");
        assert_eq!(
            entry
                .carrier_insts()
                .map(|instruction| instruction.opcode.as_str())
                .collect::<Vec<_>>(),
            ["store", "call", "store"],
            "direct-call initializer must retain source order in the typed suffix"
        );
    }

    #[test]
    fn static_initializer_three_block_cfg_remains_independent_until_emitted_closure() {
        let ll = r#"
@a = internal addrspace(2) global i8 0

define internal void @_GLOBAL__sub_I_before() {
  store i8 1, ptr addrspace(2) @a
  ret void
}

define internal void @_GLOBAL__sub_I_cfg() {
entry:
  %c = icmp eq i8 0, 0
  br i1 %c, label %then, label %merge
then:
  store i8 2, ptr addrspace(2) @a
  br label %merge
merge:
  %v = phi i8 [ 3, %entry ], [ 4, %then ]
  store i8 %v, ptr addrspace(2) @a
  ret void
}

define internal void @_GLOBAL__sub_I_after() {
  store i8 5, ptr addrspace(2) @a
  ret void
}

define void @main() {
  store i8 9, ptr addrspace(2) @a
  ret void
}
"#;
        let mut module =
            LlModule::parse_with_stage_meta(ll, None, Some("main")).expect("typed module");
        module.inline_simple_static_initializers();
        assert_eq!(
            module.preinlined_static_initializers,
            HashSet::from(["_GLOBAL__sub_I_after".to_string()]),
            "only the straight-line suffix crosses the pre-emission inference boundary"
        );
        let entry = module
            .functions
            .iter()
            .find(|function| function.name == "main")
            .expect("entry");
        assert_eq!(
            entry
                .carrier_insts()
                .map(|instruction| instruction.opcode.as_str())
                .collect::<Vec<_>>(),
            ["store", "store"],
            "the post-CFG straight-line constructor stays before the entry body"
        );
    }

    #[test]
    fn static_initializer_two_block_cfg_remains_independent_until_emitted_closure() {
        let ll = r#"
@a = internal addrspace(2) global i8 0

define internal void @_GLOBAL__sub_I_cfg() {
entry:
  br label %exit
exit:
  store i8 1, ptr addrspace(2) @a
  ret void
}

define void @main() {
  store i8 9, ptr addrspace(2) @a
  ret void
}
"#;
        let mut module =
            LlModule::parse_with_stage_meta(ll, None, Some("main")).expect("typed module");
        module.inline_simple_static_initializers();
        assert!(module.preinlined_static_initializers.is_empty());
        let entry = module
            .functions
            .iter()
            .find(|function| function.name == "main")
            .expect("entry");
        assert_eq!(entry.blocks.len(), 1);
    }

    #[test]
    fn static_initializer_chained_cfgs_remain_independent_until_emitted_closure() {
        let ll = r#"
define internal void @_GLOBAL__sub_I_first() {
entry:
  %c = icmp eq i8 0, 0
  br i1 %c, label %then, label %merge
then:
  br label %merge
merge:
  ret void
}

define internal void @_GLOBAL__sub_I_second() {
entry:
  %c = icmp eq i8 0, 0
  br i1 %c, label %then, label %merge
then:
  br label %merge
merge:
  ret void
}

define void @main() {
  ret void
}
"#;
        let mut module =
            LlModule::parse_with_stage_meta(ll, None, Some("main")).expect("typed module");
        module.inline_simple_static_initializers();
        assert!(
            module.preinlined_static_initializers.is_empty(),
            "constructor CFGs retain their independent pre-emission domains"
        );
        let entry = module
            .functions
            .iter()
            .find(|function| function.name == "main")
            .expect("entry");
        assert_eq!(entry.blocks.len(), 1);
    }

    #[test]
    fn static_initializer_suffix_leaves_indirect_calls_residual() {
        let ll = r#"
@a = internal addrspace(2) global i8 0

define internal void @_GLOBAL__sub_I_indirect() {
  %fn = inttoptr i64 0 to ptr
  call void %fn()
  ret void
}

define void @main() {
  store i8 2, ptr addrspace(2) @a
  ret void
}
"#;
        let mut module =
            LlModule::parse_with_stage_meta(ll, None, Some("main")).expect("typed module");
        module.inline_simple_static_initializers();
        assert!(
            module.preinlined_static_initializers.is_empty(),
            "indirect-call constructor must remain for the emitted-graph closure"
        );
        let entry = module
            .functions
            .iter()
            .find(|function| function.name == "main")
            .expect("entry");
        assert_eq!(entry.carrier_insts().count(), 1);
    }
}
