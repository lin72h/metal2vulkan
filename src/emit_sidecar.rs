//! Typed facts carried beside native-emitted SPIR-V across the retained emit → passes seam.
//!
//! The sidecar is the sole carrier for these facts. It exists only through `passes::transform`;
//! every consumer runs before final id canonicalization.

use spirv::Word;
use std::collections::HashMap;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct EmitSidecar {
    /// Emitted `OpTypeStruct` id -> exact AIR member offsets, including backend padding members.
    pub(crate) air_struct_offsets: HashMap<Word, Vec<u32>>,
    pub(crate) buffer_address_words: Vec<BufferAddressWord>,
    pub(crate) local_pointer_field_stores: Vec<LocalPointerFieldStore>,
    pub(crate) local_pointer_field_loads: Vec<LocalPointerFieldLoad>,
    pub(crate) local_pointer_dynamic_field_loads: Vec<LocalPointerDynamicFieldLoad>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct BufferAddressWord {
    pub(crate) id: Word,
    pub(crate) param_index: u32,
    pub(crate) component: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct LocalPointerFieldStore {
    pub(crate) id: Word,
    pub(crate) source: Word,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct LocalPointerFieldLoad {
    pub(crate) id: Word,
    pub(crate) root: Word,
    pub(crate) indices: Vec<u32>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct LocalPointerDynamicFieldLoad {
    pub(crate) id: Word,
    pub(crate) root: Word,
    pub(crate) prefix: Vec<u32>,
    pub(crate) suffix: Vec<u32>,
}

impl EmitSidecar {
    /// Apply a whole-module value-id substitution to every typed fact.
    ///
    /// Typed helper inlining emits cloned bodies against opaque parameter ids, then replaces those
    /// ids with caller arguments. The sidecar crosses the same seam as the module and must observe
    /// the identical substitution or a field-load root can point at an id that no longer exists.
    pub(crate) fn remap_ids(&mut self, remap: &HashMap<Word, Word>) {
        let replace = |id: &mut Word| {
            if let Some(replacement) = remap.get(id) {
                *id = *replacement;
            }
        };
        for fact in &mut self.buffer_address_words {
            replace(&mut fact.id);
        }
        for fact in &mut self.local_pointer_field_stores {
            replace(&mut fact.id);
            replace(&mut fact.source);
        }
        for fact in &mut self.local_pointer_field_loads {
            replace(&mut fact.id);
            replace(&mut fact.root);
        }
        for fact in &mut self.local_pointer_dynamic_field_loads {
            replace(&mut fact.id);
            replace(&mut fact.root);
        }
        self.air_struct_offsets = std::mem::take(&mut self.air_struct_offsets)
            .into_iter()
            .map(|(id, offsets)| (remap.get(&id).copied().unwrap_or(id), offsets))
            .collect();
    }

    pub(crate) fn clone_inlined_local_pointer_field_loads(&mut self, remap: &HashMap<Word, Word>) {
        let clones = self
            .local_pointer_field_loads
            .iter()
            .filter_map(|fact| {
                let id = remap.get(&fact.id).copied()?;
                Some(LocalPointerFieldLoad {
                    id,
                    root: remap.get(&fact.root).copied().unwrap_or(fact.root),
                    indices: fact.indices.clone(),
                })
            })
            .collect::<Vec<_>>();
        self.local_pointer_field_loads.extend(clones);
    }

    pub(crate) fn remap_local_pointer_field_store_sources(&mut self, remap: &HashMap<Word, Word>) {
        for fact in &mut self.local_pointer_field_stores {
            if let Some(source) = remap.get(&fact.source) {
                fact.source = *source;
            }
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct EmittedSpirv {
    pub(crate) module: crate::spirv_module::Module,
    pub(crate) sidecar: EmitSidecar,
}

impl EmittedSpirv {
    pub(crate) fn into_bytes(self) -> Vec<u8> {
        self.module
            .assemble()
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inlining_clones_load_facts_and_remaps_store_sources() {
        let mut sidecar = EmitSidecar {
            local_pointer_field_stores: vec![LocalPointerFieldStore { id: 10, source: 20 }],
            local_pointer_field_loads: vec![LocalPointerFieldLoad {
                id: 30,
                root: 40,
                indices: vec![1, 2],
            }],
            local_pointer_dynamic_field_loads: vec![LocalPointerDynamicFieldLoad {
                id: 50,
                root: 60,
                prefix: vec![3],
                suffix: vec![4],
            }],
            ..EmitSidecar::default()
        };
        let remap = HashMap::from([(20, 120), (30, 130), (40, 140)]);

        sidecar.clone_inlined_local_pointer_field_loads(&remap);
        sidecar.remap_local_pointer_field_store_sources(&remap);

        assert_eq!(
            sidecar.local_pointer_field_stores,
            vec![LocalPointerFieldStore {
                id: 10,
                source: 120
            }]
        );
        assert_eq!(
            sidecar.local_pointer_field_loads,
            vec![
                LocalPointerFieldLoad {
                    id: 30,
                    root: 40,
                    indices: vec![1, 2],
                },
                LocalPointerFieldLoad {
                    id: 130,
                    root: 140,
                    indices: vec![1, 2],
                },
            ]
        );
        assert_eq!(
            sidecar.local_pointer_dynamic_field_loads,
            vec![LocalPointerDynamicFieldLoad {
                id: 50,
                root: 60,
                prefix: vec![3],
                suffix: vec![4],
            }]
        );
    }

    #[test]
    fn whole_module_substitution_remaps_every_sidecar_id_field() {
        let mut sidecar = EmitSidecar {
            air_struct_offsets: HashMap::from([(5, vec![0, 16])]),
            buffer_address_words: vec![BufferAddressWord {
                id: 10,
                param_index: 2,
                component: 1,
            }],
            local_pointer_field_stores: vec![LocalPointerFieldStore { id: 20, source: 21 }],
            local_pointer_field_loads: vec![LocalPointerFieldLoad {
                id: 30,
                root: 31,
                indices: vec![4],
            }],
            local_pointer_dynamic_field_loads: vec![LocalPointerDynamicFieldLoad {
                id: 40,
                root: 41,
                prefix: vec![5],
                suffix: vec![6],
            }],
        };
        let remap = HashMap::from([
            (10, 110),
            (20, 120),
            (21, 121),
            (30, 130),
            (31, 131),
            (40, 140),
            (41, 141),
            (5, 105),
        ]);

        sidecar.remap_ids(&remap);

        assert_eq!(sidecar.air_struct_offsets.get(&105), Some(&vec![0, 16]));
        assert!(!sidecar.air_struct_offsets.contains_key(&5));
        assert_eq!(sidecar.buffer_address_words[0].id, 110);
        assert_eq!(sidecar.local_pointer_field_stores[0].id, 120);
        assert_eq!(sidecar.local_pointer_field_stores[0].source, 121);
        assert_eq!(sidecar.local_pointer_field_loads[0].id, 130);
        assert_eq!(sidecar.local_pointer_field_loads[0].root, 131);
        assert_eq!(sidecar.local_pointer_dynamic_field_loads[0].id, 140);
        assert_eq!(sidecar.local_pointer_dynamic_field_loads[0].root, 141);
    }
}
