//! Typed facts carried beside native-emitted SPIR-V across the retained emit → passes seam.
//!
//! The sidecar is the sole carrier for these facts. It exists only through `passes::transform`;
//! every consumer runs before final id canonicalization.

use spirv::Word;
use std::collections::{HashMap, HashSet};

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct EmitSidecar {
    /// The emitter intentionally modeled every device/constant buffer through the raw byte/word
    /// representation. Interface construction uses this typed fact to preserve that representation's
    /// established closure ordering rather than applying the default typed-buffer fallback early.
    pub(crate) all_device_buffers_raw: bool,
    /// Final interface construction must represent any remaining cross-binding StorageBuffer
    /// closure in the address domain. Both all-raw constructors set this before passes begin so the
    /// resource phase owns the complete descriptor-pointer graph.
    pub(crate) construct_cross_binding_addresses: bool,
    /// Functions whose exact source CFG was rejected by the ordinary ownership planner. Same-CFG
    /// representation retries reuse this immutable result before trying construct-tree ownership.
    pub(crate) ordinary_plan_rejected_functions: HashSet<String>,
    /// Functions whose exact source CFG was rejected by the complete source-ownership ladder.
    /// Same-CFG representation retries can proceed directly to the bounded fallback constructor.
    pub(crate) ownership_plan_rejected_functions: HashSet<String>,
    /// Surviving functions whose final owned instruction graph contains a conditional or switch
    /// header without an adjacent merge declaration after source ownership planning and lowering.
    pub(crate) post_lowering_cfg_construction_functions: HashSet<String>,
    /// Parsed source vector ABI rules, threaded through every retry candidate into final layout.
    pub(crate) air_data_layout: Option<crate::layout::AirDataLayout>,
    /// Emitted `OpTypeStruct` id -> exact AIR member offsets, including backend padding members.
    pub(crate) air_struct_offsets: HashMap<Word, Vec<u32>>,
    /// One outcome for every entry buffer carrying `air.struct_type_info`. This keeps an exact AIR
    /// layout that could not be associated with the emitted type graph from disappearing silently.
    pub(crate) air_struct_layout_mappings: Vec<AirStructLayoutMapping>,
    /// Entry buffer ordinals whose AIR aggregate layout replaced a non-aggregate source pointee.
    /// Their native raw-word access paths are flat element indices and must be routed through member
    /// 0 of the existing `{ RuntimeArray<uint> }` interface block.
    pub(crate) flat_raw_buffer_params: HashSet<u32>,
    pub(crate) buffer_address_words: Vec<BufferAddressWord>,
    /// Exact constant byte address of a GEP result relative to one buffer root. This preserves the
    /// source aggregate layout across the native-emitter → remodeled-interface seam, where an
    /// overlapping struct/union may intentionally become a raw word block.
    pub(crate) buffer_access_offsets: Vec<BufferAccessOffset>,
    /// Affine byte address of a dynamic GEP relative to one buffer root. Each term is an emitted
    /// integer index id and its exact source-layout byte stride.
    pub(crate) buffer_access_affine_offsets: Vec<BufferAccessAffineOffset>,
    /// Final descriptor variable -> original emitted parameter pointee type. Raw interface blocks
    /// deliberately discard that aggregate shape; constant late access paths still need it as the
    /// exact AIR layout oracle.
    pub(crate) buffer_root_source_types: HashMap<Word, Word>,
    /// Pointer-handle value loaded from a fixed byte offset of a buffer parameter root. The root id
    /// is remapped with helper parameters during inlining, so the fact reaches the entry parameter
    /// without retaining a callee-local ordinal.
    pub(crate) buffer_pointer_field_loads: Vec<BufferPointerFieldLoad>,
    /// Pointer handle loaded from element `index` of a buffer parameter whose AIR element is one
    /// serialized 64-bit opaque handle (for example `array_ref<texture2d<...>>`).
    pub(crate) buffer_pointer_dynamic_field_loads: Vec<BufferPointerDynamicFieldLoad>,
    pub(crate) local_pointer_field_stores: Vec<LocalPointerFieldStore>,
    pub(crate) local_pointer_field_loads: Vec<LocalPointerFieldLoad>,
    pub(crate) local_pointer_dynamic_field_loads: Vec<LocalPointerDynamicFieldLoad>,
    /// Exact logical pointer carried by an integer payload slot in a by-value AIR aggregate. The
    /// aggregate and path survive emitted helper inlining; interface binding remaps `source` to the
    /// concrete descriptor value before resource-wrapper collapse resolves matching extracts.
    pub(crate) aggregate_pointer_values: Vec<AggregatePointerValue>,
    /// Result ids emitted as typed sentinels for the stable `llvm.agx2.cluster.num` ABI intrinsic.
    /// The final interface pass replaces each sentinel with the AGX2 physical-cluster number derived
    /// from Vulkan `LocalInvocationId` and the caller-supplied kernel local size.
    pub(crate) agx2_cluster_numbers: Vec<Word>,
}

#[derive(Debug)]
pub(crate) struct EmissionFailure {
    pub(crate) error: String,
    pub(crate) ordinary_plan_rejected_functions: HashSet<String>,
    pub(crate) ownership_plan_rejected_functions: HashSet<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct AirStructLayoutMapping {
    pub(crate) param_index: u32,
    pub(crate) struct_ty: Option<Word>,
    pub(crate) status: AirStructLayoutMappingStatus,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AirStructLayoutMappingStatus {
    MappedNatural,
    MappedExplicit,
    ParameterMissing,
    ParameterIsNotPointer,
    MetadataIsNotStruct,
    EmittedShapeMismatch,
    NonIncreasingOffsets,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct BufferAddressWord {
    pub(crate) id: Word,
    pub(crate) param_index: u32,
    pub(crate) component: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct BufferAccessOffset {
    pub(crate) id: Word,
    pub(crate) root: Word,
    pub(crate) byte_offset: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct BufferAccessAffineOffset {
    pub(crate) id: Word,
    pub(crate) root: Word,
    pub(crate) constant: u64,
    pub(crate) terms: Vec<(Word, u64)>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct BufferPointerFieldLoad {
    pub(crate) id: Word,
    pub(crate) root: Word,
    pub(crate) byte_offset: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct BufferPointerDynamicFieldLoad {
    pub(crate) id: Word,
    pub(crate) root: Word,
    pub(crate) byte_offset: u64,
    pub(crate) index: Word,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct LocalPointerFieldStore {
    pub(crate) id: Word,
    pub(crate) source: Word,
    pub(crate) root: Word,
    pub(crate) indices: Vec<u32>,
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
    pub(crate) index: Word,
    pub(crate) suffix: Vec<u32>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct AggregatePointerValue {
    pub(crate) aggregate: Word,
    pub(crate) source: Word,
    pub(crate) indices: Vec<u32>,
}

impl EmitSidecar {
    /// Every SPIR-V id referenced by a typed fact crossing the emitter-to-passes seam.
    pub(crate) fn referenced_ids(&self) -> HashSet<Word> {
        let mut ids = self
            .buffer_address_words
            .iter()
            .map(|fact| fact.id)
            .collect::<HashSet<_>>();
        for fact in &self.buffer_access_offsets {
            ids.extend([fact.id, fact.root]);
        }
        for fact in &self.buffer_access_affine_offsets {
            ids.extend([fact.id, fact.root]);
            ids.extend(fact.terms.iter().map(|(index, _)| *index));
        }
        for fact in &self.buffer_pointer_field_loads {
            ids.extend([fact.id, fact.root]);
        }
        for fact in &self.buffer_pointer_dynamic_field_loads {
            ids.extend([fact.id, fact.root, fact.index]);
        }
        for fact in &self.local_pointer_field_stores {
            ids.extend([fact.id, fact.source, fact.root]);
        }
        for fact in &self.local_pointer_field_loads {
            ids.extend([fact.id, fact.root]);
        }
        for fact in &self.local_pointer_dynamic_field_loads {
            ids.extend([fact.id, fact.root, fact.index]);
        }
        for fact in &self.aggregate_pointer_values {
            ids.extend([fact.aggregate, fact.source]);
        }
        ids.extend(self.agx2_cluster_numbers.iter().copied());
        ids.extend(self.air_struct_offsets.keys().copied());
        ids.extend(
            self.air_struct_layout_mappings
                .iter()
                .filter_map(|mapping| mapping.struct_ty),
        );
        for (&root, &source_ty) in &self.buffer_root_source_types {
            ids.extend([root, source_ty]);
        }
        ids
    }

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
        for fact in &mut self.buffer_access_offsets {
            replace(&mut fact.id);
            replace(&mut fact.root);
        }
        for fact in &mut self.buffer_access_affine_offsets {
            replace(&mut fact.id);
            replace(&mut fact.root);
            for (index, _) in &mut fact.terms {
                replace(index);
            }
        }
        for fact in &mut self.buffer_pointer_field_loads {
            replace(&mut fact.id);
            replace(&mut fact.root);
        }
        for fact in &mut self.buffer_pointer_dynamic_field_loads {
            replace(&mut fact.id);
            replace(&mut fact.root);
            replace(&mut fact.index);
        }
        for fact in &mut self.local_pointer_field_stores {
            replace(&mut fact.id);
            replace(&mut fact.source);
            replace(&mut fact.root);
        }
        for fact in &mut self.local_pointer_field_loads {
            replace(&mut fact.id);
            replace(&mut fact.root);
        }
        for fact in &mut self.local_pointer_dynamic_field_loads {
            replace(&mut fact.id);
            replace(&mut fact.root);
            replace(&mut fact.index);
        }
        for fact in &mut self.aggregate_pointer_values {
            replace(&mut fact.aggregate);
            replace(&mut fact.source);
        }
        for id in &mut self.agx2_cluster_numbers {
            replace(id);
        }
        for mapping in &mut self.air_struct_layout_mappings {
            if let Some(struct_ty) = &mut mapping.struct_ty {
                replace(struct_ty);
            }
        }
        self.air_struct_offsets = self
            .air_struct_offsets
            .iter()
            .map(|(&id, offsets)| (remap.get(&id).copied().unwrap_or(id), offsets.clone()))
            .collect();
        self.buffer_root_source_types = self
            .buffer_root_source_types
            .iter()
            .map(|(&root, &source_ty)| {
                (
                    remap.get(&root).copied().unwrap_or(root),
                    remap.get(&source_ty).copied().unwrap_or(source_ty),
                )
            })
            .collect();
    }

    pub(crate) fn clone_inlined_facts(&mut self, remap: &HashMap<Word, Word>) {
        let clones = self
            .buffer_access_offsets
            .iter()
            .filter_map(|fact| {
                Some(BufferAccessOffset {
                    id: remap.get(&fact.id).copied()?,
                    root: remap.get(&fact.root).copied().unwrap_or(fact.root),
                    byte_offset: fact.byte_offset,
                })
            })
            .collect::<Vec<_>>();
        self.buffer_access_offsets.extend(clones);
        let clones = self
            .buffer_access_affine_offsets
            .iter()
            .filter_map(|fact| {
                Some(BufferAccessAffineOffset {
                    id: remap.get(&fact.id).copied()?,
                    root: remap.get(&fact.root).copied().unwrap_or(fact.root),
                    constant: fact.constant,
                    terms: fact
                        .terms
                        .iter()
                        .map(|(index, stride)| {
                            (remap.get(index).copied().unwrap_or(*index), *stride)
                        })
                        .collect(),
                })
            })
            .collect::<Vec<_>>();
        self.buffer_access_affine_offsets.extend(clones);
        let clones = self
            .local_pointer_field_stores
            .iter()
            .filter_map(|fact| {
                Some(LocalPointerFieldStore {
                    id: remap.get(&fact.id).copied().unwrap_or(fact.id),
                    source: remap.get(&fact.source).copied().unwrap_or(fact.source),
                    root: remap.get(&fact.root).copied()?,
                    indices: fact.indices.clone(),
                })
            })
            .collect::<Vec<_>>();
        self.local_pointer_field_stores.extend(clones);
        let clones = self
            .buffer_pointer_field_loads
            .iter()
            .filter_map(|fact| {
                Some(BufferPointerFieldLoad {
                    id: remap.get(&fact.id).copied()?,
                    root: remap.get(&fact.root).copied().unwrap_or(fact.root),
                    byte_offset: fact.byte_offset,
                })
            })
            .collect::<Vec<_>>();
        self.buffer_pointer_field_loads.extend(clones);
        let clones = self
            .buffer_pointer_dynamic_field_loads
            .iter()
            .filter_map(|fact| {
                Some(BufferPointerDynamicFieldLoad {
                    id: remap.get(&fact.id).copied()?,
                    root: remap.get(&fact.root).copied().unwrap_or(fact.root),
                    byte_offset: fact.byte_offset,
                    index: remap.get(&fact.index).copied().unwrap_or(fact.index),
                })
            })
            .collect::<Vec<_>>();
        self.buffer_pointer_dynamic_field_loads.extend(clones);
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
        let clones = self
            .local_pointer_dynamic_field_loads
            .iter()
            .filter_map(|fact| {
                let id = remap.get(&fact.id).copied()?;
                Some(LocalPointerDynamicFieldLoad {
                    id,
                    root: remap.get(&fact.root).copied().unwrap_or(fact.root),
                    prefix: fact.prefix.clone(),
                    index: remap.get(&fact.index).copied().unwrap_or(fact.index),
                    suffix: fact.suffix.clone(),
                })
            })
            .collect::<Vec<_>>();
        self.local_pointer_dynamic_field_loads.extend(clones);
        let clones = self
            .aggregate_pointer_values
            .iter()
            .filter_map(|fact| {
                Some(AggregatePointerValue {
                    aggregate: remap.get(&fact.aggregate).copied()?,
                    source: remap.get(&fact.source).copied().unwrap_or(fact.source),
                    indices: fact.indices.clone(),
                })
            })
            .collect::<Vec<_>>();
        self.aggregate_pointer_values.extend(clones);
        let clones = self
            .agx2_cluster_numbers
            .iter()
            .filter_map(|id| remap.get(id).copied())
            .collect::<Vec<_>>();
        self.agx2_cluster_numbers.extend(clones);
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
            ordinary_plan_rejected_functions: HashSet::new(),
            ownership_plan_rejected_functions: HashSet::new(),
            post_lowering_cfg_construction_functions: HashSet::new(),
            buffer_access_offsets: vec![BufferAccessOffset {
                id: 82,
                root: 92,
                byte_offset: 28,
            }],
            buffer_pointer_field_loads: vec![BufferPointerFieldLoad {
                id: 80,
                root: 90,
                byte_offset: 16,
            }],
            buffer_pointer_dynamic_field_loads: vec![BufferPointerDynamicFieldLoad {
                id: 81,
                root: 91,
                byte_offset: 8,
                index: 71,
            }],
            local_pointer_field_stores: vec![LocalPointerFieldStore {
                id: 10,
                source: 20,
                root: 40,
                indices: vec![1, 2],
            }],
            local_pointer_field_loads: vec![LocalPointerFieldLoad {
                id: 30,
                root: 40,
                indices: vec![1, 2],
            }],
            local_pointer_dynamic_field_loads: vec![LocalPointerDynamicFieldLoad {
                id: 50,
                root: 60,
                prefix: vec![3],
                index: 70,
                suffix: vec![4],
            }],
            ..EmitSidecar::default()
        };
        let remap = HashMap::from([
            (20, 120),
            (30, 130),
            (40, 140),
            (50, 150),
            (60, 160),
            (80, 180),
            (90, 190),
            (81, 181),
            (91, 191),
            (82, 182),
            (92, 192),
        ]);

        sidecar.clone_inlined_facts(&remap);
        sidecar.remap_local_pointer_field_store_sources(&remap);

        assert_eq!(
            sidecar.buffer_access_offsets,
            vec![
                BufferAccessOffset {
                    id: 82,
                    root: 92,
                    byte_offset: 28,
                },
                BufferAccessOffset {
                    id: 182,
                    root: 192,
                    byte_offset: 28,
                },
            ]
        );

        assert_eq!(
            sidecar.buffer_pointer_field_loads,
            vec![
                BufferPointerFieldLoad {
                    id: 80,
                    root: 90,
                    byte_offset: 16,
                },
                BufferPointerFieldLoad {
                    id: 180,
                    root: 190,
                    byte_offset: 16,
                },
            ]
        );
        assert_eq!(
            sidecar.buffer_pointer_dynamic_field_loads,
            vec![
                BufferPointerDynamicFieldLoad {
                    id: 81,
                    root: 91,
                    byte_offset: 8,
                    index: 71,
                },
                BufferPointerDynamicFieldLoad {
                    id: 181,
                    root: 191,
                    byte_offset: 8,
                    index: 71,
                },
            ]
        );

        assert_eq!(
            sidecar.local_pointer_field_stores,
            vec![
                LocalPointerFieldStore {
                    id: 10,
                    source: 120,
                    root: 40,
                    indices: vec![1, 2],
                },
                LocalPointerFieldStore {
                    id: 10,
                    source: 120,
                    root: 140,
                    indices: vec![1, 2],
                },
            ]
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
            vec![
                LocalPointerDynamicFieldLoad {
                    id: 50,
                    root: 60,
                    prefix: vec![3],
                    index: 70,
                    suffix: vec![4],
                },
                LocalPointerDynamicFieldLoad {
                    id: 150,
                    root: 160,
                    prefix: vec![3],
                    index: 70,
                    suffix: vec![4],
                },
            ]
        );
    }

    #[test]
    fn whole_module_substitution_remaps_every_sidecar_id_field() {
        let mut sidecar = EmitSidecar {
            all_device_buffers_raw: false,
            construct_cross_binding_addresses: false,
            ordinary_plan_rejected_functions: HashSet::new(),
            ownership_plan_rejected_functions: HashSet::new(),
            post_lowering_cfg_construction_functions: HashSet::new(),
            air_data_layout: None,
            air_struct_offsets: HashMap::from([(5, vec![0, 16])]),
            air_struct_layout_mappings: vec![AirStructLayoutMapping {
                param_index: 2,
                struct_ty: Some(5),
                status: AirStructLayoutMappingStatus::MappedNatural,
            }],
            flat_raw_buffer_params: HashSet::from([2]),
            buffer_address_words: vec![BufferAddressWord {
                id: 10,
                param_index: 2,
                component: 1,
            }],
            buffer_access_offsets: vec![BufferAccessOffset {
                id: 16,
                root: 17,
                byte_offset: 28,
            }],
            buffer_access_affine_offsets: vec![BufferAccessAffineOffset {
                id: 23,
                root: 24,
                constant: 32,
                terms: vec![(25, 48)],
            }],
            buffer_root_source_types: HashMap::from([(18, 19)]),
            buffer_pointer_field_loads: vec![BufferPointerFieldLoad {
                id: 11,
                root: 12,
                byte_offset: 24,
            }],
            buffer_pointer_dynamic_field_loads: vec![BufferPointerDynamicFieldLoad {
                id: 13,
                root: 14,
                byte_offset: 16,
                index: 15,
            }],
            local_pointer_field_stores: vec![LocalPointerFieldStore {
                id: 20,
                source: 21,
                root: 22,
                indices: vec![3],
            }],
            local_pointer_field_loads: vec![LocalPointerFieldLoad {
                id: 30,
                root: 31,
                indices: vec![4],
            }],
            local_pointer_dynamic_field_loads: vec![LocalPointerDynamicFieldLoad {
                id: 40,
                root: 41,
                prefix: vec![5],
                index: 42,
                suffix: vec![6],
            }],
            aggregate_pointer_values: vec![AggregatePointerValue {
                aggregate: 43,
                source: 44,
                indices: vec![1, 0],
            }],
            agx2_cluster_numbers: vec![50],
        };
        let remap = HashMap::from([
            (10, 110),
            (11, 111),
            (12, 112),
            (13, 113),
            (14, 114),
            (15, 115),
            (16, 116),
            (17, 117),
            (18, 118),
            (19, 119),
            (20, 120),
            (21, 121),
            (22, 122),
            (23, 123),
            (24, 124),
            (25, 125),
            (30, 130),
            (31, 131),
            (40, 140),
            (41, 141),
            (42, 142),
            (43, 143),
            (44, 144),
            (5, 105),
            (50, 150),
        ]);

        sidecar.remap_ids(&remap);
        assert_eq!(
            sidecar.aggregate_pointer_values,
            vec![AggregatePointerValue {
                aggregate: 143,
                source: 144,
                indices: vec![1, 0],
            }]
        );
        assert_eq!(
            sidecar.buffer_root_source_types,
            HashMap::from([(118, 119)])
        );

        assert_eq!(sidecar.air_struct_offsets.get(&105), Some(&vec![0, 16]));
        assert!(!sidecar.air_struct_offsets.contains_key(&5));
        assert_eq!(sidecar.air_struct_layout_mappings[0].struct_ty, Some(105));
        assert_eq!(sidecar.buffer_address_words[0].id, 110);
        assert_eq!(sidecar.buffer_access_offsets[0].id, 116);
        assert_eq!(sidecar.buffer_access_offsets[0].root, 117);
        assert_eq!(sidecar.buffer_access_affine_offsets[0].id, 123);
        assert_eq!(sidecar.buffer_access_affine_offsets[0].root, 124);
        assert_eq!(
            sidecar.buffer_access_affine_offsets[0].terms,
            vec![(125, 48)]
        );
        assert_eq!(sidecar.buffer_pointer_field_loads[0].id, 111);
        assert_eq!(sidecar.buffer_pointer_field_loads[0].root, 112);
        assert_eq!(sidecar.buffer_pointer_dynamic_field_loads[0].id, 113);
        assert_eq!(sidecar.buffer_pointer_dynamic_field_loads[0].root, 114);
        assert_eq!(sidecar.buffer_pointer_dynamic_field_loads[0].index, 115);
        assert_eq!(sidecar.local_pointer_field_stores[0].id, 120);
        assert_eq!(sidecar.local_pointer_field_stores[0].source, 121);
        assert_eq!(sidecar.local_pointer_field_stores[0].root, 122);
        assert_eq!(sidecar.local_pointer_field_loads[0].id, 130);
        assert_eq!(sidecar.local_pointer_field_loads[0].root, 131);
        assert_eq!(sidecar.local_pointer_dynamic_field_loads[0].id, 140);
        assert_eq!(sidecar.local_pointer_dynamic_field_loads[0].root, 141);
        assert_eq!(sidecar.local_pointer_dynamic_field_loads[0].index, 142);
        assert_eq!(sidecar.agx2_cluster_numbers, vec![150]);
    }
}
