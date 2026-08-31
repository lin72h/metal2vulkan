use crate::case::{
    AccelerationStructureKind, AuthoredCase, ExecutionSafety, OutputSelection, ResourceRole, Stage,
    TextureFormat, TextureType, VertexObservation,
};
use crate::library_module::ResolvedLinkedFunctions;
use crate::source::{find_source, SourceRow};
use base64::Engine as _;
use metal2vulkan::reflect::{ResourceKind, ShaderReflection};
use std::collections::{HashMap, HashSet};
use std::path::Path;

pub struct CheckedCase {
    pub case: AuthoredCase,
    pub source: SourceRow,
    pub reflection: ShaderReflection,
    pub linked_functions: ResolvedLinkedFunctions,
    pub input_sha256: String,
}

pub struct CheckedCaseContract {
    pub case: AuthoredCase,
    pub reflection: ShaderReflection,
    pub linked_functions: ResolvedLinkedFunctions,
    pub input_sha256: String,
}

pub fn check_case(root: &Path, case: AuthoredCase) -> Result<CheckedCase, Vec<String>> {
    let (mut errors, input_sha256) = check_case_identity(&case);
    let source = match find_source(root, &case.air_sha256) {
        Ok(Some(source)) => source,
        Ok(None) => {
            errors.push(format!(
                "AIR {} does not exist in aligned source shards or public fixtures",
                case.air_sha256
            ));
            return Err(errors);
        }
        Err(error) => {
            errors.push(error);
            return Err(errors);
        }
    };
    let checked =
        check_case_against_source_with_identity(root, case, &source, errors, input_sha256)?;
    Ok(CheckedCase {
        case: checked.case,
        source,
        reflection: checked.reflection,
        linked_functions: checked.linked_functions,
        input_sha256: checked.input_sha256,
    })
}

/// Check an authored case against an already selected exact AIR row.
///
/// Translation workers use this boundary so case validation does not reopen or rescan a source
/// shard after the indexed row has already been handed to the isolated worker.
pub fn check_case_against_source(
    root: &Path,
    case: AuthoredCase,
    source: &SourceRow,
) -> Result<CheckedCaseContract, Vec<String>> {
    let (errors, input_sha256) = check_case_identity(&case);
    check_case_against_source_with_identity(root, case, source, errors, input_sha256)
}

fn check_case_identity(case: &AuthoredCase) -> (Vec<String>, String) {
    let mut errors = case.validate_literal_resources().err().unwrap_or_default();
    match case.computed_case_id() {
        Ok(computed) if computed != case.case_id => errors.push(format!(
            "case_id mismatch: manifest={} computed={computed}",
            case.case_id
        )),
        Err(error) => errors.push(error),
        _ => {}
    }
    let input_sha256 = match case.computed_input_sha256() {
        Ok(digest) => digest,
        Err(error) => {
            errors.push(error);
            String::new()
        }
    };
    (errors, input_sha256)
}

fn check_case_against_source_with_identity(
    root: &Path,
    case: AuthoredCase,
    source: &SourceRow,
    mut errors: Vec<String>,
    input_sha256: String,
) -> Result<CheckedCaseContract, Vec<String>> {
    if source.air_sha256 != case.air_sha256 {
        errors.push(format!(
            "AIR hash mismatch: case={} source={}",
            case.air_sha256, source.air_sha256
        ));
    }
    if source.stage != case.stage.metadata_label() {
        errors.push(format!(
            "stage mismatch: manifest={:?} AIR metadata={}",
            case.stage, source.stage
        ));
    }
    if source.entry != case.entry {
        errors.push(format!(
            "entry mismatch: manifest={:?} AIR metadata={:?}",
            case.entry, source.entry
        ));
    }
    validate_execution_safety(case.execution_safety, &source.air_ll, &mut errors);
    for resource in &case.acceleration_structures {
        let marker = match resource.kind {
            AccelerationStructureKind::Instance => "!\"air.instance_acceleration_structure\"",
            AccelerationStructureKind::Primitive => "!\"air.primitive_acceleration_structure\"",
        };
        if !source.air_ll.contains(marker) {
            errors.push(format!(
                "authored {:?} acceleration-structure binding {} has no matching AIR argument",
                resource.kind, resource.binding
            ));
        }
    }
    let linked_functions = match crate::library_module::resolve_linked_functions(root, &case) {
        Ok(tables) => tables,
        Err(table_errors) => {
            errors.extend(table_errors);
            ResolvedLinkedFunctions::default()
        }
    };
    validate_visible_reference_closure(&source.air_ll, &linked_functions, &mut errors);

    let reflection = match reflect(&source.air_ll, &case, &linked_functions) {
        Ok(reflection) => reflection,
        Err(error) => {
            errors.push(format!("product reflection failed: {error}"));
            return Err(errors);
        }
    };
    if let Err(error) = crate::executor_contract::require_reflection(
        &case,
        &source.air_ll,
        &reflection,
        "shared executor",
    ) {
        errors.push(error);
    }
    validate_reflection(&case, &reflection, &mut errors);
    if errors.is_empty() {
        Ok(CheckedCaseContract {
            case,
            reflection,
            linked_functions,
            input_sha256,
        })
    } else {
        Err(errors)
    }
}

fn validate_visible_reference_closure(
    entry_ll: &str,
    linked: &ResolvedLinkedFunctions,
    errors: &mut Vec<String>,
) {
    let linkage = metal2vulkan::linked_functions::LinkedFunctionLinkage {
        visible_references: linked
            .references
            .iter()
            .map(
                |reference| metal2vulkan::linked_functions::LinkedFunctionReference {
                    symbol: reference.function.clone(),
                    module_ll: reference.module.air_ll.clone(),
                },
            )
            .collect(),
        visible_tables: vec![],
        intersection_tables: vec![],
    };
    let modules = std::iter::once(entry_ll).chain(linked.visible.iter().flat_map(|table| {
        table
            .entries
            .iter()
            .map(|entry| entry.module.air_ll.as_str())
    }));
    let modules = modules.chain(linked.intersection.iter().flat_map(|table| {
        table.entries.iter().filter_map(|entry| match entry {
            crate::library_module::ResolvedIntersectionFunctionEntry::Linked(entry) => {
                Some(entry.module.air_ll.as_str())
            }
            crate::library_module::ResolvedIntersectionFunctionEntry::OpaqueTriangle { .. } => None,
        })
    }));
    let mut seen = HashSet::new();
    for module in modules {
        if seen.insert(module) {
            if let Err(error) =
                metal2vulkan::linked_functions::specialize_visible_function_references(
                    module, &linkage,
                )
            {
                errors.push(error);
            }
        }
    }
}

/// Build the exact product linkage described by checked authored resources and reflection.
///
/// Both candidate execution and translation audits use this mapping so table bindings, embedded
/// table fields, linked callbacks, and opaque intersection entries cannot drift between paths.
pub fn product_linkage(
    reflection: &ShaderReflection,
    linked: &ResolvedLinkedFunctions,
) -> Result<metal2vulkan::linked_functions::LinkedFunctionLinkage, String> {
    fn linked_tables(
        reflection: &ShaderReflection,
        tables: &[crate::library_module::ResolvedFunctionTable],
        kind: ResourceKind,
        label: &str,
    ) -> Result<Vec<metal2vulkan::linked_functions::LinkedFunctionTable>, String> {
        tables
            .iter()
            .map(|table| {
                let parameter_index = reflection
                    .bindings
                    .iter()
                    .find(|binding| binding.kind == kind && binding.metal_index == table.binding)
                    .and_then(|binding| binding.param_index)
                    .ok_or_else(|| {
                        format!(
                            "{label} function table binding {} has no reflected entry parameter",
                            table.binding,
                        )
                    })?;
                Ok(metal2vulkan::linked_functions::LinkedFunctionTable {
                    parameter_index,
                    size: table.size,
                    entries: table
                        .entries
                        .iter()
                        .map(|entry| metal2vulkan::linked_functions::LinkedFunction {
                            index: entry.index,
                            symbol: entry.function.clone(),
                            module_ll: entry.module.air_ll.clone(),
                        })
                        .collect(),
                })
            })
            .collect()
    }
    let visible_tables = linked_tables(
        reflection,
        &linked.visible,
        ResourceKind::VisibleFunctionTable,
        "visible",
    )?;
    let intersection_tables = linked
        .intersection
        .iter()
        .map(|table| {
            let source = match table.location {
                crate::library_module::ResolvedIntersectionFunctionTableLocation::Direct {
                    binding: table_binding,
                } => {
                    let parameter_index = reflection
                        .bindings
                        .iter()
                        .find(|binding| {
                            binding.kind == ResourceKind::IntersectionFunctionTable
                                && binding.metal_index == table_binding
                        })
                        .and_then(|binding| binding.param_index)
                        .ok_or_else(|| {
                            format!(
                                "intersection function table binding {table_binding} has no reflected entry parameter"
                            )
                        })?;
                    metal2vulkan::linked_functions::IntersectionFunctionTableSource::Parameter {
                        parameter_index,
                    }
                }
                crate::library_module::ResolvedIntersectionFunctionTableLocation::ArgumentBuffer {
                    buffer_binding,
                    field_offset,
                } => {
                    let field = reflection
                        .argument_buffer_fields
                        .iter()
                        .find(|field| {
                            field.buffer_index == buffer_binding
                                && field.field_offset == field_offset
                        })
                        .ok_or_else(|| {
                            format!(
                                "argument-buffer intersection function table at buffer {buffer_binding} offset {field_offset} has no reflected field"
                            )
                        })?;
                    metal2vulkan::linked_functions::IntersectionFunctionTableSource::ArgumentBuffer {
                        buffer_parameter_index: field.buffer_param_index,
                        field_ordinal: field.field_ordinal,
                        field_offset,
                    }
                }
            };
            let entries = table
                .entries
                .iter()
                .map(|entry| match entry {
                    crate::library_module::ResolvedIntersectionFunctionEntry::Linked(entry) => {
                        metal2vulkan::linked_functions::IntersectionFunctionEntry::Linked(
                            metal2vulkan::linked_functions::LinkedFunction {
                                index: entry.index,
                                symbol: entry.function.clone(),
                                module_ll: entry.module.air_ll.clone(),
                            },
                        )
                    }
                    crate::library_module::ResolvedIntersectionFunctionEntry::OpaqueTriangle {
                        index,
                        signature,
                    } => {
                        use crate::case::IntersectionFunctionSignature as Source;
                        use metal2vulkan::linked_functions::IntersectionFunctionSignature as Target;
                        metal2vulkan::linked_functions::IntersectionFunctionEntry::OpaqueTriangle {
                            index: *index,
                            signature: signature
                                .iter()
                                .map(|flag| match flag {
                                    Source::Instancing => Target::Instancing,
                                    Source::TriangleData => Target::TriangleData,
                                    Source::WorldSpaceData => Target::WorldSpaceData,
                                    Source::InstanceMotion => Target::InstanceMotion,
                                    Source::PrimitiveMotion => Target::PrimitiveMotion,
                                    Source::ExtendedLimits => Target::ExtendedLimits,
                                    Source::MaxLevels => Target::MaxLevels,
                                    Source::IntersectionFunctionBuffer => {
                                        Target::IntersectionFunctionBuffer
                                    }
                                    Source::UserData => Target::UserData,
                                })
                                .collect(),
                        }
                    }
                })
                .collect();
            Ok(metal2vulkan::linked_functions::IntersectionFunctionTable {
                source,
                size: table.size,
                entries,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    let visible_references = linked
        .references
        .iter()
        .map(
            |reference| metal2vulkan::linked_functions::LinkedFunctionReference {
                symbol: reference.function.clone(),
                module_ll: reference.module.air_ll.clone(),
            },
        )
        .collect();
    Ok(metal2vulkan::linked_functions::LinkedFunctionLinkage {
        visible_references,
        visible_tables,
        intersection_tables,
    })
}

fn validate_execution_safety(safety: ExecutionSafety, ll: &str, errors: &mut Vec<String>) {
    if has_cfg_cycle(ll) && safety == ExecutionSafety::LoopFree {
        errors.push(
            "AIR contains a CFG cycle; use execution_safety=authored_bounded and identify the finite input/function-constant bound in rationale"
                .into(),
        );
    }
}

fn reflect(
    air_ll: &str,
    case: &AuthoredCase,
    linked: &ResolvedLinkedFunctions,
) -> Result<ShaderReflection, String> {
    let function_constants = crate::literal::function_constants(case)?
        .into_iter()
        .map(|constant| (constant.index, constant.bytes))
        .collect::<Vec<_>>();
    let initial = metal2vulkan::reflect_sanitized_specialized(
        air_ll,
        case.stage.product(),
        crate::case::product_transform_options(case)?,
        &function_constants,
    )?;
    let linkage = product_linkage(&initial, linked)?;
    let linked_air = if linkage.is_empty() {
        std::borrow::Cow::Borrowed(air_ll)
    } else {
        std::borrow::Cow::Owned(metal2vulkan::specialize_linked_module(
            air_ll,
            case.stage.product(),
            &linkage,
        )?)
    };
    metal2vulkan::reflect_sanitized_specialized(
        linked_air.as_ref(),
        case.stage.product(),
        crate::case::product_transform_options_with_reflection(case, &initial)?,
        &function_constants,
    )
}

fn validate_reflection(
    case: &AuthoredCase,
    reflection: &ShaderReflection,
    errors: &mut Vec<String>,
) {
    validate_tessellation(case, reflection, errors);
    let reflected_buffers = reflection
        .bindings
        .iter()
        .filter(|binding| binding.kind == ResourceKind::Buffer)
        .map(|binding| binding.metal_index)
        .collect::<HashSet<_>>();
    let manifest_buffers = case
        .buffers
        .iter()
        .map(|buffer| buffer.binding)
        .chain(case.device_buffer_arrays.iter().map(|array| array.binding))
        .collect::<HashSet<_>>();
    report_set_difference("buffer", &reflected_buffers, &manifest_buffers, errors);
    for binding in reflection
        .bindings
        .iter()
        .filter(|binding| binding.kind == ResourceKind::Buffer)
    {
        let is_array = binding
            .type_name
            .as_deref()
            .is_some_and(metal2vulkan::meta::is_device_buffer_array_type_name);
        let authored_as_array = case
            .device_buffer_arrays
            .iter()
            .any(|array| array.binding == binding.metal_index);
        if is_array != authored_as_array {
            errors.push(format!(
                "buffer {} {} a device-buffer array in AIR metadata, but the manifest {}",
                binding.metal_index,
                if is_array { "is" } else { "is not" },
                if authored_as_array {
                    "authors it as one"
                } else {
                    "does not author it as one"
                }
            ));
        }
    }

    let reflected_argument_buffer_buffers = reflection
        .bindings
        .iter()
        .filter_map(|binding| {
            (binding.kind == ResourceKind::EmbeddedArgBufferBuffer)
                .then_some(binding.embedded_source)
                .flatten()
                .map(|source| (source.buffer_index, source.field_offset))
        })
        .collect::<HashSet<_>>();
    let manifest_argument_buffer_buffers = case
        .argument_buffer_buffers
        .iter()
        .map(|buffer| (buffer.buffer_binding, buffer.field_offset))
        .collect::<HashSet<_>>();
    for (owner, offset) in
        reflected_argument_buffer_buffers.difference(&manifest_argument_buffer_buffers)
    {
        errors.push(format!(
            "reflection requires argument-buffer buffer {owner}+{offset}, but the manifest does not declare it"
        ));
    }
    for (owner, offset) in
        manifest_argument_buffer_buffers.difference(&reflected_argument_buffer_buffers)
    {
        errors.push(format!(
            "manifest declares argument-buffer buffer {owner}+{offset}, but product reflection does not"
        ));
    }
    for resource in &case.argument_buffer_buffers {
        let Some(binding) = reflection.bindings.iter().find(|binding| {
            binding.kind == ResourceKind::EmbeddedArgBufferBuffer
                && binding.embedded_source.is_some_and(|source| {
                    source.buffer_index == resource.buffer_binding
                        && source.field_offset == resource.field_offset
                })
        }) else {
            continue;
        };
        if resource.role != crate::case::ResourceRole::Input
            && binding.access == Some(metal2vulkan::reflect::ResourceAccess::ReadOnly)
        {
            errors.push(format!(
                "argument-buffer buffer {}+{} is authored as {:?}, but reflection is read-only",
                resource.buffer_binding, resource.field_offset, resource.role
            ));
        }
    }

    let reflected_threadgroup_memory = reflection
        .bindings
        .iter()
        .filter(|binding| binding.kind == ResourceKind::ThreadgroupBuffer)
        .map(|binding| binding.metal_index)
        .collect::<HashSet<_>>();
    let manifest_threadgroup_memory = case
        .threadgroup_memory
        .iter()
        .map(|resource| resource.binding)
        .collect::<HashSet<_>>();
    report_set_difference(
        "threadgroup memory",
        &reflected_threadgroup_memory,
        &manifest_threadgroup_memory,
        errors,
    );

    let reflected_stage_inputs = reflection
        .bindings
        .iter()
        .filter(|binding| binding.kind == ResourceKind::KernelStageInput)
        .filter_map(|binding| binding.stage_input_location)
        .collect::<HashSet<_>>();
    let manifest_stage_inputs = case
        .kernel_stage_inputs
        .iter()
        .map(|resource| resource.location)
        .collect::<HashSet<_>>();
    report_set_difference(
        "kernel stage input",
        &reflected_stage_inputs,
        &manifest_stage_inputs,
        errors,
    );
    for resource in &case.kernel_stage_inputs {
        let Some(binding) = reflection.bindings.iter().find(|binding| {
            binding.kind == ResourceKind::KernelStageInput
                && binding.stage_input_location == Some(resource.location)
        }) else {
            continue;
        };
        if binding.type_name.as_deref() != Some(resource.format.air_type_name()) {
            errors.push(format!(
                "kernel stage input {} format {:?} does not match reflected AIR type {:?}",
                resource.location, resource.format, binding.type_name
            ));
        }
        if binding.descriptor.is_none() {
            errors.push(format!(
                "kernel stage input {} has no reflected descriptor",
                resource.location
            ));
        }
    }
    for resource in &case.threadgroup_memory {
        let Some(binding) = reflection.bindings.iter().find(|binding| {
            binding.kind == ResourceKind::ThreadgroupBuffer
                && binding.metal_index == resource.binding
        }) else {
            continue;
        };
        if binding
            .declared_size
            .is_some_and(|minimum| resource.length < minimum)
        {
            errors.push(format!(
                "threadgroup-memory binding {} length {} is smaller than reflected element size {}",
                resource.binding,
                resource.length,
                binding.declared_size.unwrap_or_default()
            ));
        }
    }

    match (
        reflection.imageblock_layouts.is_empty(),
        case.imageblock.as_ref(),
    ) {
        (true, Some(_)) => errors.push(
            "manifest declares an imageblock, but product reflection has no imageblock layout"
                .into(),
        ),
        (false, None) => errors.push(format!(
            "reflection requires {} imageblock layout(s), but the manifest declares no imageblock dimensions",
            reflection.imageblock_layouts.len()
        )),
        _ => {}
    }
    validate_implicit_imageblock_attachments(case, reflection, errors);
    validate_fragment_imageblock(case, reflection, errors);
    if !selected_output_is_reflected_writable(case, reflection) {
        errors.push("selected output is not reflected as shader-writable".into());
    }

    let reflected_acceleration_structures = reflection
        .bindings
        .iter()
        .filter(|binding| {
            matches!(
                binding.kind,
                ResourceKind::AccelerationStructureShadow
                    | ResourceKind::PrimitiveAccelerationStructure
            )
        })
        .map(|binding| binding.metal_index)
        .collect::<HashSet<_>>();
    let manifest_acceleration_structures = case
        .acceleration_structures
        .iter()
        .map(|resource| resource.binding)
        .collect::<HashSet<_>>();
    report_set_difference(
        "acceleration structure",
        &reflected_acceleration_structures,
        &manifest_acceleration_structures,
        errors,
    );
    for resource in &case.acceleration_structures {
        let expected = match resource.kind {
            AccelerationStructureKind::Instance => ResourceKind::AccelerationStructureShadow,
            AccelerationStructureKind::Primitive => ResourceKind::PrimitiveAccelerationStructure,
        };
        if reflection.bindings.iter().any(|binding| {
            binding.metal_index == resource.binding
                && matches!(
                    binding.kind,
                    ResourceKind::AccelerationStructureShadow
                        | ResourceKind::PrimitiveAccelerationStructure
                )
                && binding.kind != expected
        }) {
            errors.push(format!(
                "acceleration-structure binding {} kind {:?} does not match product reflection {:?}",
                resource.binding, resource.kind, expected
            ));
        }
    }

    for (label, kind, authored) in [
        (
            "visible-function table",
            ResourceKind::VisibleFunctionTable,
            case.visible_function_tables
                .iter()
                .map(|table| table.binding)
                .collect::<HashSet<_>>(),
        ),
        (
            "intersection-function table",
            ResourceKind::IntersectionFunctionTable,
            case.intersection_function_tables
                .iter()
                .map(|table| table.binding)
                .collect::<HashSet<_>>(),
        ),
    ] {
        let reflected = reflection
            .bindings
            .iter()
            .filter(|binding| binding.kind == kind)
            .map(|binding| binding.metal_index)
            .collect::<HashSet<_>>();
        report_set_difference(label, &reflected, &authored, errors);
    }
    let reflected_embedded_fields = reflection
        .argument_buffer_fields
        .iter()
        .map(|field| (field.buffer_index, field.field_offset))
        .collect::<HashSet<_>>();
    let authored_embedded_tables = case
        .argument_buffer_intersection_function_tables
        .iter()
        .map(|table| (table.buffer_binding, table.field_offset))
        .collect::<HashSet<_>>();
    for (buffer, offset) in authored_embedded_tables.difference(&reflected_embedded_fields) {
        errors.push(format!(
            "manifest declares argument-buffer intersection-function table at buffer {buffer} offset {offset}, but AIR reflection has no such indirect field"
        ));
    }

    let reflected_textures = reflection
        .bindings
        .iter()
        .filter(|binding| {
            matches!(
                binding.kind,
                ResourceKind::Texture | ResourceKind::StorageImage
            )
        })
        .map(|binding| binding.metal_index)
        .collect::<HashSet<_>>();
    let manifest_textures = case
        .textures
        .iter()
        .filter(|texture| {
            !case.texture_arrays.iter().any(|array| {
                array.overrides_texture_at_base
                    && array.binding == texture.binding
                    && reflection.bindings.iter().any(|binding| {
                        binding.kind == ResourceKind::TextureArray
                            && binding.metal_index == array.binding
                    })
            })
        })
        .map(|texture| texture.binding)
        .collect::<HashSet<_>>();
    report_set_difference("texture", &reflected_textures, &manifest_textures, errors);
    for resource in &case.textures {
        let alternatives = reflection
            .bindings
            .iter()
            .filter(|binding| {
                matches!(
                    binding.kind,
                    ResourceKind::Texture | ResourceKind::StorageImage
                ) && binding.metal_index == resource.binding
            })
            .map(|binding| binding.texture_shape)
            .collect::<Vec<_>>();
        if alternatives.is_empty() {
            continue;
        }
        validate_texture_alternatives(
            &format!("texture binding {}", resource.binding),
            resource.texture_type,
            resource.format,
            &alternatives,
            errors,
        );
    }

    let reflected_texture_arrays = reflection
        .bindings
        .iter()
        .filter(|binding| binding.kind == ResourceKind::TextureArray)
        .map(|binding| binding.metal_index)
        .collect::<HashSet<_>>();
    let manifest_texture_arrays = case
        .texture_arrays
        .iter()
        .filter(|array| {
            !(array.overrides_texture_at_base
                && reflection.bindings.iter().any(|binding| {
                    matches!(
                        binding.kind,
                        ResourceKind::Texture | ResourceKind::StorageImage
                    ) && binding.metal_index == array.binding
                }))
        })
        .map(|texture| texture.binding)
        .collect::<HashSet<_>>();
    report_set_difference(
        "texture array",
        &reflected_texture_arrays,
        &manifest_texture_arrays,
        errors,
    );
    for resource in &case.texture_arrays {
        let alternatives = reflection
            .bindings
            .iter()
            .filter(|binding| {
                binding.kind == ResourceKind::TextureArray
                    && binding.metal_index == resource.binding
            })
            .collect::<Vec<_>>();
        if alternatives.is_empty() {
            continue;
        }
        let fixed_lengths = alternatives
            .iter()
            .filter_map(|binding| binding.texture_shape?.array_length)
            .collect::<Vec<_>>();
        if !fixed_lengths.is_empty()
            && !fixed_lengths
                .iter()
                .any(|length| *length as usize == resource.elements.len())
        {
            errors.push(format!(
                "texture-array binding {} has {} authored elements, but AIR alternatives declare {:?}",
                resource.binding, resource.elements.len(), fixed_lengths
            ));
        }
        if alternatives.iter().any(|binding| {
            binding.descriptor.is_none_or(|descriptor| {
                descriptor.count != metal2vulkan::meta::TEXTURE_HANDLE_ARRAY_DESCRIPTOR_COUNT
            })
        }) {
            errors.push(format!(
                "texture-array binding {} does not expose the {}-descriptor product contract for every alternative",
                resource.binding,
                metal2vulkan::meta::TEXTURE_HANDLE_ARRAY_DESCRIPTOR_COUNT
            ));
        }
        if resource.role != crate::case::ResourceRole::Input
            && !alternatives.iter().any(|binding| {
                binding.access == Some(metal2vulkan::reflect::ResourceAccess::Storage)
            })
        {
            errors.push(format!(
                "texture-array binding {} is authored as {:?}, but reflection is read-only",
                resource.binding, resource.role
            ));
        }
        validate_texture_alternatives(
            &format!("texture-array binding {}", resource.binding),
            resource.texture_type,
            resource.format,
            &alternatives
                .iter()
                .map(|binding| binding.texture_shape)
                .collect::<Vec<_>>(),
            errors,
        );
    }

    let reflected_argument_buffer_textures = reflection
        .bindings
        .iter()
        .filter(|binding| binding.kind == ResourceKind::EmbeddedArgBufferTexture)
        .flat_map(|binding| {
            let source = binding.embedded_source;
            let count = binding
                .descriptor
                .map(|descriptor| descriptor.count)
                .unwrap_or(1);
            (0..count).filter_map(move |element| {
                let source = source?;
                let field_offset = source.field_offset.checked_add(element.checked_mul(8)?)?;
                Some((source.buffer_index, field_offset))
            })
        })
        .collect::<HashSet<_>>();
    let manifest_argument_buffer_textures = case
        .argument_buffer_textures
        .iter()
        .map(|texture| (texture.buffer_binding, texture.field_offset))
        .collect::<HashSet<_>>();
    let mut missing = reflected_argument_buffer_textures
        .difference(&manifest_argument_buffer_textures)
        .copied()
        .collect::<Vec<_>>();
    let mut extra = manifest_argument_buffer_textures
        .difference(&reflected_argument_buffer_textures)
        .copied()
        .collect::<Vec<_>>();
    missing.sort_unstable();
    extra.sort_unstable();
    for (buffer_binding, field_offset) in missing {
        errors.push(format!(
            "reflection requires argument-buffer texture {buffer_binding}+{field_offset}, but the manifest does not declare it"
        ));
    }
    for (buffer_binding, field_offset) in extra {
        errors.push(format!(
            "manifest declares argument-buffer texture {buffer_binding}+{field_offset}, but product reflection does not"
        ));
    }
    for resource in &case.argument_buffer_textures {
        let alternatives = reflection
            .bindings
            .iter()
            .filter(|binding| {
                if binding.kind != ResourceKind::EmbeddedArgBufferTexture {
                    return false;
                }
                let Some(source) = binding.embedded_source else {
                    return false;
                };
                let count = binding
                    .descriptor
                    .map(|descriptor| descriptor.count)
                    .unwrap_or(1);
                source.buffer_index == resource.buffer_binding
                    && resource.field_offset >= source.field_offset
                    && (resource.field_offset - source.field_offset) % 8 == 0
                    && (resource.field_offset - source.field_offset) / 8 < count
            })
            .collect::<Vec<_>>();
        if alternatives.is_empty() {
            continue;
        }
        if resource.role != crate::case::ResourceRole::Input
            && !alternatives.iter().any(|binding| {
                binding
                    .texture_shape
                    .as_ref()
                    .is_some_and(|shape| shape.writable)
            })
        {
            errors.push(format!(
                "argument-buffer texture {}+{} is authored as {:?}, but reflection is read-only",
                resource.buffer_binding, resource.field_offset, resource.role
            ));
        }
        validate_texture_alternatives(
            &format!(
                "argument-buffer texture {}+{}",
                resource.buffer_binding, resource.field_offset
            ),
            resource.texture_type,
            resource.format,
            &alternatives
                .iter()
                .map(|binding| binding.texture_shape)
                .collect::<Vec<_>>(),
            errors,
        );
    }

    let reflected_samplers = reflection
        .bindings
        .iter()
        .filter(|binding| binding.kind == ResourceKind::Sampler)
        .map(|binding| binding.metal_index)
        .collect::<HashSet<_>>();
    let manifest_samplers = case
        .samplers
        .iter()
        .map(|sampler| sampler.binding)
        .collect::<HashSet<_>>();
    report_set_difference("sampler", &reflected_samplers, &manifest_samplers, errors);

    let reflected_constants = reflection
        .function_constants
        .iter()
        .map(|constant| constant.index)
        .collect::<HashSet<_>>();
    let manifest_constants = case
        .function_constants
        .iter()
        .map(|constant| constant.index)
        .collect::<HashSet<_>>();
    report_set_difference(
        "function constant",
        &reflected_constants,
        &manifest_constants,
        errors,
    );
    for reflected in &reflection.function_constants {
        let Some(authored) = case
            .function_constants
            .iter()
            .find(|constant| constant.index == reflected.index)
        else {
            continue;
        };
        let Some((scalar_type, lanes)) =
            crate::case::ScalarType::from_metal_abi_type_encoding(&reflected.abi_type_encoding)
        else {
            errors.push(format!(
                "function constant {} has unsupported Metal ABI type {:?}",
                reflected.index, reflected.abi_type_encoding
            ));
            continue;
        };
        if authored.scalar_type != scalar_type || authored.lanes != lanes {
            errors.push(format!(
                "function constant {} requires {scalar_type:?}x{lanes}, got {:?}x{}",
                reflected.index, authored.scalar_type, authored.lanes
            ));
        }
    }

    let reflected_vertex_inputs = reflection
        .vertex_attributes
        .iter()
        .map(|attribute| attribute.location)
        .collect::<HashSet<_>>();
    let manifest_vertex_inputs = case
        .vertex_inputs
        .iter()
        .map(|input| input.location)
        .collect::<HashSet<_>>();
    report_set_difference(
        "vertex input",
        &reflected_vertex_inputs,
        &manifest_vertex_inputs,
        errors,
    );

    if case.stage == Stage::Fragment {
        let reflected_targets = reflection
            .render_targets
            .iter()
            .map(|target| target.location)
            .chain(
                reflection
                    .bindings
                    .iter()
                    .filter(|binding| binding.kind == ResourceKind::ColorInput)
                    .map(|binding| binding.metal_index),
            )
            .collect::<HashSet<_>>();
        let manifest_targets = case
            .render_targets
            .iter()
            .map(|target| target.index)
            .collect::<HashSet<_>>();
        report_set_difference(
            "render target",
            &reflected_targets,
            &manifest_targets,
            errors,
        );
        for target in &case.render_targets {
            let type_names = reflection
                .render_targets
                .iter()
                .filter(|reflected| reflected.location == target.index)
                .filter_map(|reflected| reflected.type_name.as_deref())
                .chain(
                    reflection
                        .bindings
                        .iter()
                        .filter(|binding| {
                            binding.kind == ResourceKind::ColorInput
                                && binding.metal_index == target.index
                        })
                        .filter_map(|binding| binding.type_name.as_deref()),
                );
            for type_name in type_names {
                if !render_target_format_matches_type(target.format, type_name) {
                    errors.push(format!(
                        "render target {} format {:?} is incompatible with AIR type {type_name}",
                        target.index, target.format
                    ));
                }
            }
        }
        let has_depth = case
            .depth_stencil
            .as_ref()
            .is_some_and(|attachment| attachment.initial_depth_b64.is_some());
        let has_stencil = case
            .depth_stencil
            .as_ref()
            .is_some_and(|attachment| attachment.initial_stencil_b64.is_some());
        if reflection.depth_members.is_empty() && has_depth {
            errors.push("manifest has a depth aspect absent from fragment reflection".into());
        }
        if !reflection.depth_members.is_empty() && !has_depth {
            errors.push("fragment reflection requires an authored depth aspect".into());
        }
        if reflection.stencil_members.is_empty() && has_stencil {
            errors.push("manifest has a stencil aspect absent from fragment reflection".into());
        }
        if !reflection.stencil_members.is_empty() && !has_stencil {
            errors.push("fragment reflection requires an authored stencil aspect".into());
        }
    }
    validate_vertex_observation(case, reflection, errors);
}

fn selected_output_is_reflected_writable(
    case: &AuthoredCase,
    reflection: &ShaderReflection,
) -> bool {
    use metal2vulkan::reflect::ResourceAccess;

    let writable = |access: Option<ResourceAccess>| {
        !matches!(
            access,
            Some(ResourceAccess::Unused | ResourceAccess::ReadOnly | ResourceAccess::Sampled)
        )
    };
    match &case.output {
        OutputSelection::None => {
            !reflection.bindings.iter().any(|binding| {
                matches!(
                    binding.kind,
                    ResourceKind::Buffer
                        | ResourceKind::StorageImage
                        | ResourceKind::TextureArray
                        | ResourceKind::EmbeddedArgBufferBuffer
                        | ResourceKind::EmbeddedArgBufferTexture
                ) && writable(binding.access)
            }) && reflection.render_targets.is_empty()
                && reflection.depth_members.is_empty()
                && reflection.stencil_members.is_empty()
                && !reflection
                    .implicit_imageblock_attachments
                    .iter()
                    .any(|attachment| writable(Some(attachment.access)))
                && !reflection
                    .fragment_imageblock
                    .as_ref()
                    .is_some_and(|imageblock| {
                        imageblock
                            .members
                            .iter()
                            .any(|member| member.binding.is_some() && writable(Some(member.access)))
                    })
                && !(case.stage == Stage::Vertex
                    && (reflection
                        .vertex_builtins
                        .is_some_and(|builtins| builtins.writes_position)
                        || !reflection.varyings.is_empty()))
        }
        OutputSelection::Buffer { binding, .. } => reflection.bindings.iter().any(|reflected| {
            reflected.kind == ResourceKind::Buffer
                && reflected.metal_index == *binding
                && writable(reflected.access)
        }),
        OutputSelection::ArgumentBufferBuffer {
            buffer_binding,
            field_offset,
            ..
        } => reflection.bindings.iter().any(|reflected| {
            reflected.kind == ResourceKind::EmbeddedArgBufferBuffer
                && reflected.embedded_source.is_some_and(|source| {
                    source.buffer_index == *buffer_binding && source.field_offset == *field_offset
                })
                && writable(reflected.access)
        }),
        OutputSelection::DeviceBufferArrayElement { binding, .. } => {
            reflection.bindings.iter().any(|reflected| {
                reflected.kind == ResourceKind::Buffer
                    && reflected.metal_index == *binding
                    && writable(reflected.access)
            })
        }
        OutputSelection::Texture { binding, .. }
        | OutputSelection::TextureArrayElement { binding, .. } => {
            reflection.bindings.iter().any(|reflected| {
                reflected.metal_index == *binding
                    && matches!(
                        reflected.kind,
                        ResourceKind::StorageImage | ResourceKind::TextureArray
                    )
                    && writable(reflected.access)
            })
        }
        OutputSelection::ArgumentBufferTexture {
            buffer_binding,
            field_offset,
            ..
        } => reflection.bindings.iter().any(|reflected| {
            reflected.kind == ResourceKind::EmbeddedArgBufferTexture
                && reflected.embedded_source.is_some_and(|source| {
                    source.buffer_index == *buffer_binding && source.field_offset == *field_offset
                })
                && writable(reflected.access)
        }),
        OutputSelection::RenderTarget { index, .. } => {
            reflection
                .render_targets
                .iter()
                .any(|target| target.location == *index)
                || reflection
                    .implicit_imageblock_attachments
                    .iter()
                    .any(|attachment| {
                        attachment.attachment == *index && writable(Some(attachment.access))
                    })
                || (case.stage == Stage::Vertex && *index == 0 && case.vertex_observation.is_some())
        }
        OutputSelection::Depth { .. } => !reflection.depth_members.is_empty(),
        OutputSelection::Stencil { .. } => !reflection.stencil_members.is_empty(),
        OutputSelection::FragmentImageblock { semantic, .. } => reflection
            .fragment_imageblock
            .as_ref()
            .is_some_and(|imageblock| {
                imageblock.members.iter().any(|member| {
                    member.semantic == *semantic
                        && member.binding.is_some()
                        && writable(Some(member.access))
                })
            }),
    }
}

fn validate_tessellation(
    case: &AuthoredCase,
    reflection: &ShaderReflection,
    errors: &mut Vec<String>,
) {
    let (Some(authored), Some(interface)) = (&case.tessellation, &reflection.tessellation) else {
        if case.tessellation.is_some() != reflection.tessellation.is_some() {
            errors.push("manifest and AIR tessellation execution do not match".into());
        }
        return;
    };
    let (edge_count, inside_count) = match interface.domain {
        metal2vulkan::meta::PatchDomain::Triangle => (3, 1),
        metal2vulkan::meta::PatchDomain::Quad => (4, 2),
        metal2vulkan::meta::PatchDomain::Isoline => (2, 0),
    };
    for (patch, factors) in authored.factors.iter().enumerate() {
        if factors.edge_f16.len() != edge_count || factors.inside_f16.len() != inside_count {
            errors.push(format!(
                "tessellation patch {patch} requires {edge_count} edge and {inside_count} inside factors for {:?}",
                interface.domain
            ));
        }
    }
    validate_tessellation_attributes(
        "control-point",
        &authored.control_points,
        &interface.control_point_attributes,
        authored.factors.len(),
        interface.control_point_count as usize,
        errors,
    );
    validate_tessellation_attributes(
        "patch",
        &authored.patch_inputs,
        &interface.patch_attributes,
        authored.factors.len(),
        1,
        errors,
    );
}

fn validate_tessellation_attributes(
    label: &str,
    authored: &[crate::case::AttributeInput],
    reflected: &[metal2vulkan::reflect::TessellationAttribute],
    patch_count: usize,
    records_per_patch: usize,
    errors: &mut Vec<String>,
) {
    let authored_locations = authored
        .iter()
        .map(|input| input.location)
        .collect::<HashSet<_>>();
    let reflected_locations = reflected
        .iter()
        .map(|input| input.location)
        .collect::<HashSet<_>>();
    report_set_difference(
        &format!("tessellation {label} input"),
        &reflected_locations,
        &authored_locations,
        errors,
    );
    for input in authored {
        if let Some(expected) = reflected
            .iter()
            .find(|reflected| reflected.location == input.location)
            .and_then(|reflected| reflected.type_name.as_deref())
        {
            if input.format.air_type_name() != expected {
                errors.push(format!(
                    "tessellation {label} input {} format {:?} is incompatible with AIR type {expected}",
                    input.location, input.format
                ));
            }
        }
        let required = patch_count
            .checked_mul(records_per_patch)
            .and_then(|records| records.checked_mul(input.stride as usize));
        if let (Some(required), Ok(bytes)) = (
            required,
            base64::engine::general_purpose::STANDARD.decode(&input.bytes_b64),
        ) {
            if bytes.len() < required {
                errors.push(format!(
                    "tessellation {label} input {} has {} bytes, execution requires at least {required}",
                    input.location,
                    bytes.len()
                ));
            }
        }
    }
}

fn validate_texture_shape(
    label: &str,
    authored_type: TextureType,
    authored_format: TextureFormat,
    reflected: Option<metal2vulkan::meta::TextureShape>,
    errors: &mut Vec<String>,
) {
    use metal2vulkan::meta::{
        TextureComponent, TextureDimension, TextureFormat as ReflectedFormat,
    };

    let Some(shape) = reflected else {
        errors.push(format!("{label} has no reflected texture shape"));
        return;
    };
    let expected_type = match (shape.dimension, shape.arrayed, shape.multisampled) {
        (TextureDimension::Buffer, false, false) => Some(TextureType::Buffer),
        (TextureDimension::D1, false, false) => Some(TextureType::D1),
        (TextureDimension::D1, true, false) => Some(TextureType::D1Array),
        (TextureDimension::D2, false, false) => Some(TextureType::D2),
        (TextureDimension::D2, true, false) => Some(TextureType::D2Array),
        (TextureDimension::D2, false, true) => Some(TextureType::D2Multisample),
        (TextureDimension::D2, true, true) => Some(TextureType::D2MultisampleArray),
        (TextureDimension::D3, false, false) => Some(TextureType::D3),
        (TextureDimension::Cube, false, false) => Some(TextureType::Cube),
        (TextureDimension::Cube, true, false) => Some(TextureType::CubeArray),
        _ => None,
    };
    match expected_type {
        Some(expected) if expected != authored_type => errors.push(format!(
            "{label} requires texture type {expected:?}, got {authored_type:?}"
        )),
        None => errors.push(format!(
            "{label} has no authored texture type for reflected shape {shape:?}"
        )),
        _ => {}
    }

    let exact_storage_format = shape.storage_format.map(|format| match format {
        ReflectedFormat::R8 => TextureFormat::R8Unorm,
        ReflectedFormat::Rgba8 => TextureFormat::Rgba8Unorm,
        ReflectedFormat::R16f => TextureFormat::R16Float,
        ReflectedFormat::R16ui => TextureFormat::R16Uint,
        ReflectedFormat::Rg16f => TextureFormat::Rg16Float,
        ReflectedFormat::Rg32f => TextureFormat::Rg32Float,
        ReflectedFormat::R32f => TextureFormat::R32Float,
        ReflectedFormat::R32i => TextureFormat::R32Sint,
        ReflectedFormat::R32ui => TextureFormat::R32Uint,
        ReflectedFormat::Rgba32i => TextureFormat::Rgba32Sint,
        ReflectedFormat::Rgba32ui => TextureFormat::Rgba32Uint,
        ReflectedFormat::Rgba32f => TextureFormat::Rgba32Float,
        ReflectedFormat::Rgba16f => TextureFormat::Rgba16Float,
        ReflectedFormat::Rgba8ui => TextureFormat::Rgba8Uint,
        ReflectedFormat::Rgba16ui => TextureFormat::Rgba16Uint,
        ReflectedFormat::Rgba8i => TextureFormat::Rgba8Sint,
    });
    let format_matches = exact_storage_format.map_or_else(
        || match shape.component {
            TextureComponent::Float => matches!(
                authored_format,
                TextureFormat::R8Unorm
                    | TextureFormat::Rgba8Unorm
                    | TextureFormat::Rg32Float
                    | TextureFormat::Rgba16Float
                    | TextureFormat::R32Float
                    | TextureFormat::Rgba32Float
                    | TextureFormat::Depth32Float
            ),
            TextureComponent::Uint => matches!(
                authored_format,
                TextureFormat::Rgba8Uint
                    | TextureFormat::R16Uint
                    | TextureFormat::Rgba16Uint
                    | TextureFormat::R32Uint
                    | TextureFormat::Rgba32Uint
            ),
            TextureComponent::Sint => matches!(
                authored_format,
                TextureFormat::Rgba8Sint | TextureFormat::R32Sint | TextureFormat::Rgba32Sint
            ),
        },
        |expected| expected == authored_format,
    );
    if !format_matches {
        if let Some(expected) = exact_storage_format {
            errors.push(format!(
                "{label} requires storage format {expected:?}, got {authored_format:?}"
            ));
        } else {
            errors.push(format!(
                "{label} format {authored_format:?} does not match reflected {:?} component class",
                shape.component
            ));
        }
    }
}

fn validate_texture_alternatives(
    label: &str,
    authored_type: TextureType,
    authored_format: TextureFormat,
    reflected: &[Option<metal2vulkan::meta::TextureShape>],
    errors: &mut Vec<String>,
) {
    let attempts = reflected
        .iter()
        .map(|shape| {
            let mut attempt = Vec::new();
            validate_texture_shape(label, authored_type, authored_format, *shape, &mut attempt);
            attempt
        })
        .collect::<Vec<_>>();
    if attempts.iter().any(Vec::is_empty) {
        return;
    }
    if let Some(first) = attempts.into_iter().next() {
        errors.extend(first);
    }
}

fn render_target_format_matches_type(format: TextureFormat, type_name: &str) -> bool {
    let scalar = type_name
        .trim()
        .trim_end_matches(|ch: char| ch.is_ascii_digit());
    match scalar {
        "float" | "half" => matches!(
            format,
            TextureFormat::R8Unorm
                | TextureFormat::Rgba8Unorm
                | TextureFormat::R16Float
                | TextureFormat::Rg16Float
                | TextureFormat::Rg32Float
                | TextureFormat::Rgba16Float
                | TextureFormat::R32Float
                | TextureFormat::Rgba32Float
        ),
        "uint" | "ushort" | "uchar" | "bool" => {
            matches!(
                format,
                TextureFormat::Rgba8Uint
                    | TextureFormat::R16Uint
                    | TextureFormat::Rgba16Uint
                    | TextureFormat::R32Uint
                    | TextureFormat::Rgba32Uint
            )
        }
        "int" | "short" | "char" => {
            matches!(
                format,
                TextureFormat::Rgba8Sint | TextureFormat::R32Sint | TextureFormat::Rgba32Sint
            )
        }
        _ => false,
    }
}

fn validate_implicit_imageblock_attachments(
    case: &AuthoredCase,
    reflection: &ShaderReflection,
    errors: &mut Vec<String>,
) {
    let has_implicit_attachments = !reflection.implicit_imageblock_attachments.is_empty();
    match (
        has_implicit_attachments,
        case.imageblock
            .as_ref()
            .and_then(|imageblock| imageblock.implicit_coverage),
    ) {
        (true, None) => errors.push(
            "implicit imageblock attachments require an authored imageblock.implicit_coverage"
                .into(),
        ),
        (false, Some(_)) => errors.push(
            "imageblock.implicit_coverage is valid only for an implicit imageblock layout".into(),
        ),
        _ => {}
    }
    let reflected = reflection
        .implicit_imageblock_attachments
        .iter()
        .map(|attachment| attachment.attachment)
        .collect::<HashSet<_>>();
    let manifest = case
        .render_targets
        .iter()
        .map(|target| target.index)
        .collect::<HashSet<_>>();
    if case.stage == Stage::Kernel {
        report_set_difference(
            "implicit imageblock attachment",
            &reflected,
            &manifest,
            errors,
        );
    }
    let Some(imageblock) = case.imageblock.as_ref() else {
        return;
    };
    for attachment in &reflection.implicit_imageblock_attachments {
        let Some(target) = case
            .render_targets
            .iter()
            .find(|target| target.index == attachment.attachment)
        else {
            continue;
        };
        if target.dimensions != imageblock.dimensions {
            errors.push(format!(
                "implicit imageblock attachment {} dimensions {:?} must equal imageblock dimensions {:?}",
                attachment.attachment, target.dimensions, imageblock.dimensions
            ));
        }
        let expected = match attachment.format {
            metal2vulkan::meta::TextureFormat::R16f => TextureFormat::R16Float,
            metal2vulkan::meta::TextureFormat::Rg16f => TextureFormat::Rg16Float,
            metal2vulkan::meta::TextureFormat::Rg32f => TextureFormat::Rg32Float,
            metal2vulkan::meta::TextureFormat::R32f => TextureFormat::R32Float,
            metal2vulkan::meta::TextureFormat::R32ui => TextureFormat::R32Uint,
            metal2vulkan::meta::TextureFormat::Rgba16f => TextureFormat::Rgba16Float,
            metal2vulkan::meta::TextureFormat::Rgba32f => TextureFormat::Rgba32Float,
            other => {
                errors.push(format!(
                    "implicit imageblock attachment {} has unsupported reflected format {other:?}",
                    attachment.attachment
                ));
                continue;
            }
        };
        if target.format != expected {
            errors.push(format!(
                "implicit imageblock attachment {} requires format {expected:?}, got {:?}",
                attachment.attachment, target.format
            ));
        }
    }
}

fn validate_fragment_imageblock(
    case: &AuthoredCase,
    reflection: &ShaderReflection,
    errors: &mut Vec<String>,
) {
    let authored = case.fragment_imageblock.as_ref();
    let reflected = reflection.fragment_imageblock.as_ref();
    match (authored, reflected) {
        (None, None) => return,
        (Some(_), None) => {
            errors.push(
                "manifest declares fragment_imageblock, but product reflection has no custom fragment imageblock"
                    .into(),
            );
            return;
        }
        (None, Some(_)) => {
            errors.push(
                "reflection requires a custom fragment imageblock, but the manifest declares none"
                    .into(),
            );
            return;
        }
        (Some(_), Some(_)) => {}
    }
    let authored = authored.expect("matched above");
    let reflected = reflected.expect("matched above");
    let attachment_dimensions = case
        .render_targets
        .first()
        .map(|target| target.dimensions)
        .or_else(|| case.depth_stencil.as_ref().map(|target| target.dimensions));
    if attachment_dimensions.is_some_and(|dimensions| dimensions != authored.dimensions) {
        errors.push(format!(
            "fragment imageblock dimensions {:?} must equal graphics attachment dimensions {:?}",
            authored.dimensions,
            attachment_dimensions.expect("checked above")
        ));
    }
    for member in reflected
        .members
        .iter()
        .filter(|member| member.binding.is_some())
    {
        let Some(resource) = authored
            .members
            .iter()
            .find(|resource| resource.semantic == member.semantic)
        else {
            errors.push(format!(
                "reflected fragment imageblock member {} has no authored plane",
                member.semantic
            ));
            continue;
        };
        let expected_role = match member.access {
            metal2vulkan::reflect::ResourceAccess::ReadOnly => ResourceRole::Input,
            metal2vulkan::reflect::ResourceAccess::WriteOnly => ResourceRole::Output,
            metal2vulkan::reflect::ResourceAccess::ReadWrite => ResourceRole::InOut,
            metal2vulkan::reflect::ResourceAccess::Unused => continue,
            other => {
                errors.push(format!(
                    "fragment imageblock member {} has invalid reflected access {other:?}",
                    member.semantic
                ));
                continue;
            }
        };
        if resource.role != expected_role {
            errors.push(format!(
                "fragment imageblock member {} role {:?} must match reflected access {:?}",
                member.semantic, resource.role, member.access
            ));
        }
        let reflected_format =
            crate::case::FragmentImageblockFormat::from_air_type(&member.type_name, member.size);
        if reflected_format.is_none() {
            errors.push(format!(
                "fragment imageblock member {} has unsupported reflected type {} size {}",
                member.semantic, member.type_name, member.size
            ));
        } else if reflected_format != Some(resource.format) {
            errors.push(format!(
                "fragment imageblock member {} format {:?} must match reflected type {} size {}",
                member.semantic, resource.format, member.type_name, member.size
            ));
        }
    }
    for resource in &authored.members {
        if !reflected
            .members
            .iter()
            .any(|member| member.binding.is_some() && member.semantic == resource.semantic)
        {
            errors.push(format!(
                "authored fragment imageblock member {} is not an accessed reflected member",
                resource.semantic
            ));
        }
    }
}

fn validate_vertex_observation(
    case: &AuthoredCase,
    reflection: &ShaderReflection,
    errors: &mut Vec<String>,
) {
    let Some(observation) = case.vertex_observation else {
        return;
    };
    let Some(target) = case.render_targets.first() else {
        return;
    };
    let expected_format = match observation {
        VertexObservation::Position => {
            if !reflection
                .vertex_builtins
                .is_some_and(|builtins| builtins.writes_position)
            {
                errors
                    .push("vertex observation selects position, but AIR does not write it".into());
            }
            TextureFormat::Rgba32Float
        }
        VertexObservation::Varying { location } => {
            let Some(varying) = reflection
                .varyings
                .iter()
                .find(|varying| varying.location == location)
            else {
                errors.push(format!(
                    "vertex observation selects undeclared varying location {location}"
                ));
                return;
            };
            match varying
                .type_name
                .as_deref()
                .and_then(crate::observation_contract::ObservationType::parse)
            {
                Some(observation_type) => observation_type.attachment_format(),
                None => {
                    errors.push(format!(
                        "vertex varying {location} has unsupported or missing observer type {:?}",
                        varying.type_name
                    ));
                    return;
                }
            }
        }
    };
    if target.format != expected_format {
        errors.push(format!(
            "vertex observation requires render target format {expected_format:?}, got {:?}",
            target.format
        ));
    }
}

fn report_set_difference(
    kind: &str,
    reflected: &HashSet<u32>,
    manifest: &HashSet<u32>,
    errors: &mut Vec<String>,
) {
    let mut missing = reflected.difference(manifest).copied().collect::<Vec<_>>();
    let mut extra = manifest.difference(reflected).copied().collect::<Vec<_>>();
    missing.sort_unstable();
    extra.sort_unstable();
    for binding in missing {
        errors.push(format!(
            "reflection requires {kind} {binding}, but the manifest does not declare it"
        ));
    }
    for binding in extra {
        errors.push(format!(
            "manifest declares {kind} {binding}, but product reflection does not"
        ));
    }
}

#[cfg(test)]
mod resource_contract_tests {
    use super::*;
    use crate::case::{
        BufferResource, Comparison, Dispatch, ExecutionSafety, FunctionTableEntry,
        FunctionTableResource, OutputSelection, ResourceRole, ThreadgroupMemoryResource,
    };

    #[test]
    fn selected_identity_output_requires_a_reflected_write() {
        let ll = include_str!("../fixtures/public/kernel_implicit_imageblock_half2.ll");
        let case = crate::case::narrow_implicit_imageblock_test_case(
            crate::hash::sha256_bytes(ll.as_bytes()),
            "kernel_implicit_imageblock_half2".into(),
        );
        let writable = reflect(ll, &case, &ResolvedLinkedFunctions::default()).unwrap();
        assert!(selected_output_is_reflected_writable(&case, &writable));

        let read_only_ll = ll.replace(
            "  call void @air.store.implicit_imageblock.v2f16(<2 x half> %value, i32 0, <2 x i16> %position, i32 0, i16 0)\n",
            "",
        );
        let read_only = reflect(&read_only_ll, &case, &ResolvedLinkedFunctions::default()).unwrap();
        assert!(!selected_output_is_reflected_writable(&case, &read_only));
    }

    #[test]
    fn exact_empty_output_requires_no_reflected_write() {
        let ll = r#"
define void @no_output() {
entry:
  ret void
}
!air.kernel = !{!0}
!0 = !{ptr @no_output, !1, !1}
!1 = !{}
"#;
        let mut case: AuthoredCase = serde_json::from_value(serde_json::json!({
            "air_sha256": crate::hash::sha256_bytes(ll.as_bytes()),
            "case_id": "test-no-output",
            "name": "no-output",
            "entry": "no_output",
            "stage": "kernel",
            "dispatch": {"grid": [1, 1, 1], "threads_per_threadgroup": [1, 1, 1]},
            "output": {"kind": "none"},
            "compare": {"kind": "exact"},
            "execution_safety": "loop_free"
        }))
        .unwrap();
        case.case_id = case.computed_case_id().unwrap();
        case.validate_literal_resources().unwrap();
        let reflection = reflect(ll, &case, &ResolvedLinkedFunctions::default()).unwrap();
        assert!(selected_output_is_reflected_writable(&case, &reflection));

        let mut vertex_case = case.clone();
        vertex_case.stage = Stage::Vertex;
        let mut vertex_output = reflection.clone();
        vertex_output.vertex_builtins = Some(metal2vulkan::reflect::VertexBuiltins {
            writes_position: true,
            ..Default::default()
        });
        assert!(!selected_output_is_reflected_writable(
            &vertex_case,
            &vertex_output
        ));
        vertex_output.vertex_builtins = None;
        vertex_output.varyings.push(metal2vulkan::reflect::Varying {
            location: 0,
            type_name: Some("float4".into()),
            name: None,
            user_semantic: None,
        });
        assert!(!selected_output_is_reflected_writable(
            &vertex_case,
            &vertex_output
        ));

        let writable_ll = include_str!("../fixtures/public/kernel_vector_function_constant.ll");
        let mut writable_case = crate::case::vector_function_constant_test_case(
            crate::hash::sha256_bytes(writable_ll.as_bytes()),
            "kernel_vector_function_constant".into(),
        );
        let writable_reflection = reflect(
            writable_ll,
            &writable_case,
            &ResolvedLinkedFunctions::default(),
        )
        .unwrap();
        writable_case.output = OutputSelection::None;
        assert!(!selected_output_is_reflected_writable(
            &writable_case,
            &writable_reflection
        ));
    }

    #[test]
    fn threadgroup_memory_is_authored_and_reflected_without_a_descriptor() {
        let source = crate::source::public_sources()
            .unwrap()
            .into_iter()
            .find(|source| source.entry == "threadgroup_word")
            .unwrap();
        let mut case = AuthoredCase {
            air_sha256: source.air_sha256,
            case_id: String::new(),
            name: "threadgroup-word".into(),
            entry: source.entry,
            stage: Stage::Kernel,
            buffers: vec![BufferResource {
                binding: 1,
                role: ResourceRole::Output,
                bytes_b64: None,
                initial_bytes_b64: Some("q6urqw==".into()),
            }],
            argument_buffer_buffers: vec![],
            device_buffer_arrays: vec![],
            threadgroup_memory: vec![ThreadgroupMemoryResource {
                binding: 0,
                length: 4,
            }],
            imageblock: None,
            fragment_imageblock: None,
            acceleration_structures: vec![],
            visible_function_references: vec![],
            visible_function_tables: vec![],
            intersection_function_tables: vec![],
            argument_buffer_intersection_function_tables: vec![],
            textures: vec![],
            texture_arrays: vec![],
            argument_buffer_textures: vec![],
            samplers: vec![],
            render_targets: vec![],
            depth_stencil: None,
            vertex_inputs: vec![],
            vertex_observation: None,
            kernel_stage_inputs: vec![],
            function_constants: vec![],
            dispatch: Some(Dispatch {
                grid: [1, 1, 1],
                threads_per_threadgroup: [1, 1, 1],
            }),
            draw: None,
            tessellation: None,
            output: OutputSelection::Buffer {
                binding: 1,
                offset: 0,
                length: 4,
            },
            compare: Comparison::Exact,
            execution_safety: ExecutionSafety::LoopFree,
            rationale: None,
            authored_by: Some("test".into()),
        };
        case.case_id = case.computed_case_id().unwrap();
        let checked = check_case(&crate::source::corpus_root(), case).unwrap();
        let reflected = checked
            .reflection
            .bindings
            .iter()
            .find(|binding| binding.kind == ResourceKind::ThreadgroupBuffer)
            .unwrap();
        assert_eq!(reflected.metal_index, 0);
        assert_eq!(reflected.declared_size, Some(4));
        assert_eq!(reflected.descriptor, None);
    }

    #[test]
    fn function_table_entry_resolves_exact_cross_library_module_and_symbol() {
        let scratch = crate::ScratchDir::new("function-table-check").unwrap();
        let air_ll = r#"
define void @main(ptr addrspace(1) %output, ptr addrspace(1) %table) {
entry:
  store i32 42, ptr addrspace(1) %output, align 4
  ret void
}
!air.kernel = !{!0}
!0 = !{ptr @main, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.write", !"air.arg_type_name", !"uint"}
!4 = !{i32 1, !"air.visible_function_table", !"air.location_index", i32 3, i32 1, !"air.read", !"air.arg_type_name", !"visible_function_table"}
"#;
        let entry_library = "11".repeat(32);
        let callback_library = "22".repeat(32);
        let source = SourceRow {
            air_sha256: crate::hash::sha256_bytes(air_ll.as_bytes()),
            stage: "Kernel".into(),
            entry: "main".into(),
            air_ll: air_ll.into(),
            blob_b64: None,
            lib_sha256s: vec![entry_library],
            label: "local/function-table-entry.ll".into(),
        };
        crate::source::write_source_shards(scratch.path(), [source.clone()]).unwrap();
        let module_ll =
            "define <4 x float> @shade(ptr %payload) { ret <4 x float> zeroinitializer }";
        let module_sha256 = crate::hash::sha256_bytes(module_ll.as_bytes());
        crate::library_module::merge_library_module_shards(
            scratch.path(),
            [crate::library_module::LibraryModuleRow {
                module_sha256: module_sha256.clone(),
                air_ll: module_ll.into(),
                blob_b64: base64::engine::general_purpose::STANDARD.encode(b"owned bitcode"),
                lib_sha256s: vec![callback_library.clone()],
                label: "local/library-module/shade.ll".into(),
            }],
        )
        .unwrap();
        let mut case = AuthoredCase {
            air_sha256: source.air_sha256,
            case_id: String::new(),
            name: "function-table".into(),
            entry: source.entry,
            stage: Stage::Kernel,
            buffers: vec![BufferResource {
                binding: 0,
                role: ResourceRole::Output,
                bytes_b64: None,
                initial_bytes_b64: Some("q6urqw==".into()),
            }],
            argument_buffer_buffers: vec![],
            device_buffer_arrays: vec![],
            threadgroup_memory: vec![],
            imageblock: None,
            fragment_imageblock: None,
            acceleration_structures: vec![],
            visible_function_references: vec![],
            visible_function_tables: vec![FunctionTableResource {
                binding: 3,
                size: 1,
                entries: vec![FunctionTableEntry {
                    index: 0,
                    module_sha256,
                    function: "shade".into(),
                }],
            }],
            intersection_function_tables: vec![],
            argument_buffer_intersection_function_tables: vec![],
            textures: vec![],
            texture_arrays: vec![],
            argument_buffer_textures: vec![],
            samplers: vec![],
            render_targets: vec![],
            depth_stencil: None,
            vertex_inputs: vec![],
            vertex_observation: None,
            kernel_stage_inputs: vec![],
            function_constants: vec![],
            dispatch: Some(Dispatch {
                grid: [1, 1, 1],
                threads_per_threadgroup: [1, 1, 1],
            }),
            draw: None,
            tessellation: None,
            output: OutputSelection::Buffer {
                binding: 0,
                offset: 0,
                length: 4,
            },
            compare: Comparison::Exact,
            execution_safety: ExecutionSafety::LoopFree,
            rationale: None,
            authored_by: Some("test".into()),
        };
        case.case_id = case.computed_case_id().unwrap();
        let checked = check_case(scratch.path(), case).unwrap();
        assert_eq!(checked.linked_functions.visible.len(), 1);
        assert_eq!(checked.linked_functions.visible[0].binding, 3);
        assert_eq!(
            checked.linked_functions.visible[0].entries[0].function,
            "shade"
        );
        assert_eq!(
            checked.linked_functions.visible[0].entries[0]
                .module
                .lib_sha256s,
            vec![callback_library]
        );
        assert!(checked.reflection.bindings.iter().any(|binding| {
            binding.kind == ResourceKind::VisibleFunctionTable
                && binding.metal_index == 3
                && binding.descriptor.is_none()
        }));
    }
}

pub fn has_cfg_cycle(ll: &str) -> bool {
    let mut graph = HashMap::<String, Vec<String>>::new();
    let mut current = None::<String>;
    let mut in_function = false;
    for line in ll.lines() {
        let line = line.trim();
        if line.starts_with("define ") {
            in_function = true;
            current = Some("<entry>".into());
            graph.entry("<entry>".into()).or_default();
            continue;
        }
        if !in_function {
            continue;
        }
        if line == "}" {
            if graph_has_cycle(&graph) {
                return true;
            }
            graph.clear();
            current = None;
            in_function = false;
            continue;
        }
        if let Some(label) = parse_label(line) {
            current = Some(label.clone());
            graph.entry(label).or_default();
            continue;
        }
        if let Some(block) = current.as_ref() {
            let destinations = branch_destinations(line);
            graph.entry(block.clone()).or_default().extend(destinations);
        }
    }
    graph_has_cycle(&graph)
}

fn parse_label(line: &str) -> Option<String> {
    let label = line.split(';').next()?.trim().strip_suffix(':')?;
    if label.is_empty() || label.contains(char::is_whitespace) {
        return None;
    }
    Some(label.trim_matches('"').into())
}

fn branch_destinations(line: &str) -> Vec<String> {
    let instruction = line
        .split_once(" = ")
        .map_or(line, |(_, instruction)| instruction.trim_start());
    let opcode = instruction.split_whitespace().next().unwrap_or_default();
    if !matches!(
        opcode,
        "br" | "switch"
            | "indirectbr"
            | "invoke"
            | "callbr"
            | "catchswitch"
            | "catchret"
            | "cleanupret"
    ) {
        return Vec::new();
    }
    let bytes = instruction.as_bytes();
    let mut destinations = Vec::new();
    let mut offset = 0;
    while let Some(relative) = instruction[offset..].find("label %") {
        let start = offset + relative + "label %".len();
        let tail = &instruction[start..];
        let (name, consumed) = if let Some(quoted) = tail.strip_prefix('"') {
            match quoted.find('"') {
                Some(end) => (&quoted[..end], end + 2),
                None => break,
            }
        } else {
            let end = tail
                .find(|ch: char| {
                    !(ch.is_ascii_alphanumeric() || matches!(ch, '_' | '.' | '$' | '-'))
                })
                .unwrap_or(tail.len());
            (&tail[..end], end)
        };
        if !name.is_empty() {
            destinations.push(name.into());
        }
        offset = start + consumed;
        if offset >= bytes.len() {
            break;
        }
    }
    destinations
}

fn graph_has_cycle(graph: &HashMap<String, Vec<String>>) -> bool {
    fn visit(
        node: &str,
        graph: &HashMap<String, Vec<String>>,
        visiting: &mut HashSet<String>,
        visited: &mut HashSet<String>,
    ) -> bool {
        if visiting.contains(node) {
            return true;
        }
        if visited.contains(node) {
            return false;
        }
        visiting.insert(node.into());
        if graph.get(node).is_some_and(|edges| {
            edges
                .iter()
                .any(|next| visit(next, graph, visiting, visited))
        }) {
            return true;
        }
        visiting.remove(node);
        visited.insert(node.into());
        false
    }
    let mut visiting = HashSet::new();
    let mut visited = HashSet::new();
    graph
        .keys()
        .any(|node| visit(node, graph, &mut visiting, &mut visited))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn safety_check_distinguishes_cycle_from_acyclic_cfg() {
        let acyclic = "define void @k() {\nentry:\n br label %done\ndone:\n ret void\n}";
        let cyclic = "define void @k() {\nentry:\n br label %loop\nloop:\n br i1 true, label %loop, label %done\ndone:\n ret void\n}";
        assert!(!has_cfg_cycle(acyclic));
        assert!(has_cfg_cycle(cyclic));

        let commented = "define void @k() {\nentry:\n br label %loop\nloop: ; preds = %entry, %loop\n br i1 true, label %loop, label %done\ndone: ; preds = %loop\n ret void\n}";
        assert!(has_cfg_cycle(commented));

        let exceptional = "define void @k() personality ptr null {\nentry:\n %x = invoke i32 @f() to label %done unwind label %entry\ndone:\n ret void\n}";
        assert!(has_cfg_cycle(exceptional));

        let mut errors = Vec::new();
        validate_execution_safety(ExecutionSafety::AuthoredBounded, cyclic, &mut errors);
        assert!(errors.is_empty());
        validate_execution_safety(ExecutionSafety::LoopFree, cyclic, &mut errors);
        assert_eq!(errors.len(), 1);
    }

    #[test]
    fn render_target_formats_match_air_component_classes() {
        assert!(render_target_format_matches_type(
            TextureFormat::Rgba16Float,
            "half4"
        ));
        assert!(render_target_format_matches_type(
            TextureFormat::R32Uint,
            "uint"
        ));
        assert!(!render_target_format_matches_type(
            TextureFormat::Rgba32Float,
            "int4"
        ));
        assert!(!render_target_format_matches_type(
            TextureFormat::Rgba32Sint,
            "mystery4"
        ));
    }

    #[test]
    fn texture_shapes_require_exact_storage_formats() {
        use metal2vulkan::meta::texture_shape_from_name;

        let mut errors = Vec::new();
        validate_texture_shape(
            "uint storage texture",
            TextureType::D2,
            TextureFormat::Rgba8Uint,
            Some(texture_shape_from_name("texture2d<uint, write>")),
            &mut errors,
        );
        assert!(errors.is_empty());

        validate_texture_shape(
            "uint storage texture",
            TextureType::D2,
            TextureFormat::Rgba32Uint,
            Some(texture_shape_from_name("texture2d<uint, write>")),
            &mut errors,
        );
        assert_eq!(errors.len(), 1);
        assert!(errors[0].contains("requires storage format Rgba8Uint"));
    }

    #[test]
    fn texture_shape_may_match_a_function_constant_alternative() {
        use metal2vulkan::meta::texture_shape_from_name;

        let mut errors = Vec::new();
        validate_texture_alternatives(
            "conditional texture",
            TextureType::D2,
            TextureFormat::Rgba16Float,
            &[
                Some(texture_shape_from_name("texture2d_array<half, read>")),
                Some(texture_shape_from_name("texture2d<half, read>")),
            ],
            &mut errors,
        );
        assert!(errors.is_empty());
    }
}
