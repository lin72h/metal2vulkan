use crate::case::{AuthoredCase, Stage, VertexObservation};
use crate::check::check_case;
use crate::hash::sha256_bytes;
use crate::literal::LiteralResources;
use crate::metal::ORACLE_ABI;
use crate::observation::{
    Backend, CandidateDependencies, CandidateObservation, CandidateStatus, ComparisonResult,
    TRANSLATOR_FINGERPRINT,
};
use crate::store::CorpusStore;
use crate::ScratchDir;
use base64::Engine as _;
use metal2vulkan::reflect::ShaderReflection;
use std::path::Path;

pub const EXECUTOR_ABI: &str = "vulkan-literal-resources-v27";

pub fn execute_case(
    root: &Path,
    case: AuthoredCase,
    backend: Backend,
    metal_environment_id: &str,
    environment_id: &str,
) -> Result<CandidateObservation, String> {
    if metal_environment_id.trim().is_empty() || environment_id.trim().is_empty() {
        return Err("Metal and candidate environment IDs must not be empty".into());
    }
    validate_backend_host(backend)?;
    let checked = check_case(root, case).map_err(|errors| errors.join("; "))?;
    crate::executor_contract::require_case(&checked.case, "candidate executor")?;
    let resources = LiteralResources::prepare(&checked.case)?;
    let store = CorpusStore::new(root);
    let metal = store
        .read_metal()?
        .into_iter()
        .find(|row| {
            row.case_id == checked.case.case_id && row.environment_id == metal_environment_id
        })
        .ok_or_else(|| {
            format!(
                "no Metal observation for case {} environment {}",
                checked.case.case_id, metal_environment_id
            )
        })?;
    if !metal.dependency_matches(&checked.case, ORACLE_ABI) {
        return Err(
            "Metal observation dependencies do not match the checked case and oracle ABI".into(),
        );
    }
    let scratch = ScratchDir::new("candidate-translate")?;
    let options = crate::case::product_transform_options(&checked.case)?;
    let linked_functions = candidate_linkage(&checked)?;
    let mut spv = if linked_functions.is_empty() {
        metal2vulkan::translate_sanitized_native_with_options(
            &checked.source.air_ll,
            checked.case.stage.product(),
            scratch.path(),
            options,
        )?
    } else {
        metal2vulkan::translate_sanitized_native_linked_with_options(
            &checked.source.air_ll,
            checked.case.stage.product(),
            scratch.path(),
            options,
            &linked_functions,
        )?
    };
    if !resources.function_constants.is_empty() {
        spv = metal2vulkan::specialize_function_constant_bytes(
            &spv,
            &resources.function_constant_values(),
        )?;
    }
    let companion_spv = match checked.case.stage {
        Stage::Kernel => None,
        Stage::Vertex if checked.case.is_rasterization_disabled_vertex() => None,
        Stage::Fragment | Stage::Vertex => {
            let companion_ll = scratch.path().join("graphics-companion.ll");
            std::fs::write(&companion_ll, &checked.source.air_ll)
                .map_err(|error| format!("write {}: {error}", companion_ll.display()))?;
            let source_path = companion_ll
                .to_str()
                .ok_or_else(|| "graphics companion path is not UTF-8".to_string())?;
            Some(match checked.case.stage {
                Stage::Fragment => {
                    metal2vulkan::translate_passthrough(source_path, scratch.path())?
                }
                Stage::Vertex => metal2vulkan::translate_vertex_observer(
                    source_path,
                    match checked.case.vertex_observation {
                        Some(VertexObservation::Position) => None,
                        Some(VertexObservation::Varying { location }) => Some(location),
                        None => unreachable!("rasterization-disabled vertex handled above"),
                    },
                    scratch.path(),
                )?,
                Stage::Kernel => unreachable!(),
            })
        }
    };
    let tessellation_spv = tessellation_companion_spvasm(&resources, &checked.reflection)?
        .map(|assembly| assemble_spvasm(&assembly, "tessellation-companion"))
        .transpose()?;
    let mut pipeline_modules = vec![spv.as_slice()];
    pipeline_modules.extend(tessellation_spv.as_deref());
    pipeline_modules.extend(companion_spv.as_deref());
    let spv_sha256 = pipeline_spv_sha256(&pipeline_modules);
    let (output, environment) = platform::execute(
        &checked.case,
        &resources,
        &checked.reflection,
        &spv,
        companion_spv.as_deref(),
        tessellation_spv.as_deref(),
        backend,
    )?;
    let candidate_output_sha256 = sha256_bytes(&output);
    let status = if candidate_output_sha256 == metal.metal_output_sha256
        && base64::engine::general_purpose::STANDARD
            .decode(&metal.output_b64)
            .is_ok_and(|golden| golden == output)
    {
        CandidateStatus::Match
    } else {
        CandidateStatus::Mismatch
    };
    let observation = CandidateObservation {
        case_id: checked.case.case_id.clone(),
        air_sha256: checked.case.air_sha256.clone(),
        input_sha256: checked.input_sha256,
        golden_output_sha256: metal.metal_output_sha256.clone(),
        spv_sha256,
        translator_fingerprint: TRANSLATOR_FINGERPRINT.into(),
        candidate_output_sha256,
        output_b64: base64::engine::general_purpose::STANDARD.encode(output),
        backend,
        environment_id: environment_id.into(),
        environment,
        executor_abi: EXECUTOR_ABI.into(),
        comparison: ComparisonResult::Exact,
        status,
    };
    let dependencies = CandidateDependencies {
        case: &checked.case,
        metal: &metal,
        spv_sha256: &observation.spv_sha256,
        translator_fingerprint: TRANSLATOR_FINGERPRINT,
        backend,
        environment_id,
        executor_abi: EXECUTOR_ABI,
    };
    if !observation.dependency_matches(&dependencies) {
        return Err("constructed candidate observation failed its own dependency check".into());
    }
    store.upsert_candidate(observation.clone())?;
    Ok(observation)
}

fn candidate_linkage(
    checked: &crate::check::CheckedCase,
) -> Result<metal2vulkan::linked_functions::LinkedFunctionLinkage, String> {
    fn linked_tables(
        checked: &crate::check::CheckedCase,
        tables: &[crate::library_module::ResolvedFunctionTable],
        kind: metal2vulkan::reflect::ResourceKind,
        label: &str,
    ) -> Result<Vec<metal2vulkan::linked_functions::LinkedFunctionTable>, String> {
        tables
            .iter()
            .map(|table| {
                let parameter_index = checked
                    .reflection
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
        checked,
        &checked.linked_functions.visible,
        metal2vulkan::reflect::ResourceKind::VisibleFunctionTable,
        "visible",
    )?;
    let intersection_tables = checked
        .linked_functions
        .intersection
        .iter()
        .map(|table| {
            let source = match table.location {
                crate::library_module::ResolvedIntersectionFunctionTableLocation::Direct {
                    binding: table_binding,
                } => {
                    let parameter_index = checked
                        .reflection
                        .bindings
                        .iter()
                        .find(|binding| {
                            binding.kind
                                == metal2vulkan::reflect::ResourceKind::IntersectionFunctionTable
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
                    let field = checked
                        .reflection
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
    let visible_references = checked
        .linked_functions
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

fn pipeline_spv_sha256(modules: &[&[u8]]) -> String {
    if modules.len() == 1 {
        return sha256_bytes(modules[0]);
    }
    let mut pipeline = Vec::new();
    for module in modules {
        pipeline.extend_from_slice(&(module.len() as u64).to_le_bytes());
        pipeline.extend_from_slice(module);
    }
    sha256_bytes(&pipeline)
}

fn assemble_spvasm(assembly: &str, label: &str) -> Result<Vec<u8>, String> {
    let scratch = ScratchDir::new(label)?;
    let asm = scratch.path().join("module.spvasm");
    let spv = scratch.path().join("module.spv");
    std::fs::write(&asm, assembly).map_err(|error| format!("write {}: {error}", asm.display()))?;
    let asm_path = asm
        .to_str()
        .ok_or_else(|| format!("{label} assembly path is not UTF-8"))?;
    let spv_path = spv
        .to_str()
        .ok_or_else(|| format!("{label} output path is not UTF-8"))?;
    metal2vulkan::tools::run(
        "spirv-as",
        &["--target-env", "vulkan1.3", asm_path, "-o", spv_path],
    )?;
    let bytes = std::fs::read(&spv).map_err(|error| format!("read {}: {error}", spv.display()))?;
    metal2vulkan::tools::run("spirv-val", &["--target-env", "vulkan1.3", spv_path])?;
    Ok(bytes)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum TessScalar {
    Half,
    Float,
    Short,
    Ushort,
    Int,
    Uint,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct TessType {
    scalar: TessScalar,
    lanes: u32,
}

impl TessType {
    fn from_format(format: crate::case::AttributeFormat) -> Self {
        use crate::case::AttributeFormat as F;
        let (scalar, lanes) = match format {
            F::Char
            | F::Char2
            | F::Char3
            | F::Char4
            | F::Uchar
            | F::Uchar2
            | F::Uchar3
            | F::Uchar4 => unreachable!("8-bit tessellation interfaces are rejected by contract"),
            F::Short => (TessScalar::Short, 1),
            F::Short2 => (TessScalar::Short, 2),
            F::Short3 => (TessScalar::Short, 3),
            F::Short4 => (TessScalar::Short, 4),
            F::Ushort => (TessScalar::Ushort, 1),
            F::Ushort2 => (TessScalar::Ushort, 2),
            F::Ushort3 => (TessScalar::Ushort, 3),
            F::Ushort4 => (TessScalar::Ushort, 4),
            F::Half => (TessScalar::Half, 1),
            F::Half2 => (TessScalar::Half, 2),
            F::Half3 => (TessScalar::Half, 3),
            F::Half4 => (TessScalar::Half, 4),
            F::Float => (TessScalar::Float, 1),
            F::Float2 => (TessScalar::Float, 2),
            F::Float3 => (TessScalar::Float, 3),
            F::Float4 => (TessScalar::Float, 4),
            F::Int => (TessScalar::Int, 1),
            F::Int2 => (TessScalar::Int, 2),
            F::Int3 => (TessScalar::Int, 3),
            F::Int4 => (TessScalar::Int, 4),
            F::Uint => (TessScalar::Uint, 1),
            F::Uint2 => (TessScalar::Uint, 2),
            F::Uint3 => (TessScalar::Uint, 3),
            F::Uint4 => (TessScalar::Uint, 4),
        };
        Self { scalar, lanes }
    }

    fn from_air(name: &str) -> Option<Self> {
        let (scalar, suffix) = [
            (TessScalar::Ushort, "ushort"),
            (TessScalar::Short, "short"),
            (TessScalar::Float, "float"),
            (TessScalar::Half, "half"),
            (TessScalar::Uint, "uint"),
            (TessScalar::Int, "int"),
        ]
        .into_iter()
        .find_map(|(scalar, prefix)| name.strip_prefix(prefix).map(|suffix| (scalar, suffix)))?;
        let lanes = if suffix.is_empty() {
            1
        } else {
            suffix.parse().ok()?
        };
        (1..=4).contains(&lanes).then_some(Self { scalar, lanes })
    }

    fn id(self) -> String {
        let scalar = match self.scalar {
            TessScalar::Half => "half",
            TessScalar::Float => "float",
            TessScalar::Short => "short",
            TessScalar::Ushort => "ushort",
            TessScalar::Int => "int",
            TessScalar::Uint => "uint",
        };
        if self.lanes == 1 {
            scalar.into()
        } else {
            format!("v{}{scalar}", self.lanes)
        }
    }

    fn byte_size(self) -> usize {
        let scalar = match self.scalar {
            TessScalar::Half | TessScalar::Short | TessScalar::Ushort => 2,
            TessScalar::Float | TessScalar::Int | TessScalar::Uint => 4,
        };
        scalar * self.lanes as usize
    }

    fn is_16_bit(self) -> bool {
        matches!(
            self.scalar,
            TessScalar::Half | TessScalar::Short | TessScalar::Ushort
        )
    }
}

fn tessellation_companion_spvasm(
    resources: &LiteralResources,
    reflection: &ShaderReflection,
) -> Result<Option<String>, String> {
    let (Some(authored), Some(interface)) = (&resources.tessellation, &reflection.tessellation)
    else {
        return Ok(None);
    };
    let control_point_count = interface.control_point_count;
    let mut types = std::collections::BTreeSet::new();
    types.insert(TessType {
        scalar: TessScalar::Float,
        lanes: 1,
    });
    types.insert(TessType {
        scalar: TessScalar::Uint,
        lanes: 1,
    });
    let control_points = authored
        .control_points
        .iter()
        .map(|input| {
            let ty = TessType::from_format(input.format);
            types.insert(ty);
            (input, ty)
        })
        .collect::<Vec<_>>();
    let patch_inputs = authored
        .patch_inputs
        .iter()
        .map(|input| {
            let ty = TessType::from_format(input.format);
            types.insert(ty);
            (input, ty)
        })
        .collect::<Vec<_>>();
    let system_inputs = [
        ("instance", interface.instance_id.as_ref()),
        ("amplification_id", interface.amplification_id.as_ref()),
        (
            "amplification_count",
            interface.amplification_count.as_ref(),
        ),
    ]
    .into_iter()
    .filter_map(|(name, attribute)| attribute.map(|attribute| (name, attribute)))
    .map(|(name, attribute)| {
        let air_type = attribute
            .type_name
            .as_deref()
            .ok_or_else(|| format!("tessellation system input {name} has no reflected AIR type"))?;
        let ty = TessType::from_air(air_type).ok_or_else(|| {
            format!("tessellation system input {name} has unsupported AIR type {air_type}")
        })?;
        if ty.lanes != 1 {
            return Err(format!("tessellation system input {name} must be scalar"));
        }
        types.insert(ty);
        Ok((name, attribute, ty))
    })
    .collect::<Result<Vec<_>, String>>()?;

    let has_half = types.iter().any(|ty| ty.scalar == TessScalar::Half);
    let has_int16 = types
        .iter()
        .any(|ty| matches!(ty.scalar, TessScalar::Short | TessScalar::Ushort));
    let has_io16 = types.iter().any(|ty| ty.is_16_bit());
    let mut decorations = Vec::new();
    let mut vertex_interfaces = Vec::new();
    let mut control_interfaces = vec![
        "%invocation".to_string(),
        "%primitive".to_string(),
        "%tess_outer".to_string(),
        "%tess_inner".to_string(),
    ];
    for (index, (input, _)) in control_points.iter().enumerate() {
        decorations.push(format!(
            "OpDecorate %vin_{index} Location {}",
            input.location
        ));
        decorations.push(format!(
            "OpDecorate %vout_{index} Location {}",
            input.location
        ));
        decorations.push(format!(
            "OpDecorate %tcin_{index} Location {}",
            input.location
        ));
        decorations.push(format!(
            "OpDecorate %tcout_{index} Location {}",
            input.location
        ));
        vertex_interfaces.extend([format!("%vin_{index}"), format!("%vout_{index}")]);
        control_interfaces.extend([format!("%tcin_{index}"), format!("%tcout_{index}")]);
    }
    for (index, (input, _)) in patch_inputs.iter().enumerate() {
        decorations.push(format!(
            "OpDecorate %patch_{index} Location {}",
            input.location
        ));
        decorations.push(format!("OpDecorate %patch_{index} Patch"));
        control_interfaces.push(format!("%patch_{index}"));
    }
    for (name, attribute, _) in &system_inputs {
        decorations.push(format!(
            "OpDecorate %vs_{name} Location {}",
            attribute.location
        ));
        decorations.push(format!(
            "OpDecorate %tcs_{name}_in Location {}",
            attribute.location
        ));
        decorations.push(format!(
            "OpDecorate %tcs_{name}_out Location {}",
            attribute.location
        ));
        decorations.push(format!("OpDecorate %tcs_{name}_out Patch"));
        vertex_interfaces.push(format!("%vs_{name}"));
        control_interfaces.extend([format!("%tcs_{name}_in"), format!("%tcs_{name}_out")]);
    }
    if !system_inputs.is_empty() {
        vertex_interfaces.push("%instance_index".into());
    }

    let mut declarations = vec![
        "%void = OpTypeVoid".into(),
        "%fn = OpTypeFunction %void".into(),
        "%bool = OpTypeBool".into(),
        "%float = OpTypeFloat 32".into(),
        "%int = OpTypeInt 32 1".into(),
        "%uint = OpTypeInt 32 0".into(),
    ];
    if has_half {
        declarations.push("%half = OpTypeFloat 16".into());
    }
    if has_int16 {
        declarations.extend([
            "%short = OpTypeInt 16 1".into(),
            "%ushort = OpTypeInt 16 0".into(),
        ]);
    }
    for ty in &types {
        if ty.lanes > 1 {
            let scalar = TessType {
                scalar: ty.scalar,
                lanes: 1,
            }
            .id();
            declarations.push(format!(
                "%{} = OpTypeVector %{scalar} {}",
                ty.id(),
                ty.lanes
            ));
        }
    }
    declarations.extend([
        "%uint_0 = OpConstant %uint 0".into(),
        "%uint_1 = OpConstant %uint 1".into(),
        format!("%uint_cp = OpConstant %uint {control_point_count}"),
        "%uint_2 = OpConstant %uint 2".into(),
        "%uint_3 = OpConstant %uint 3".into(),
        "%uint_4 = OpConstant %uint 4".into(),
        "%arr4float = OpTypeArray %float %uint_4".into(),
        "%arr2float = OpTypeArray %float %uint_2".into(),
        "%ptr_output_arr4float = OpTypePointer Output %arr4float".into(),
        "%ptr_output_arr2float = OpTypePointer Output %arr2float".into(),
    ]);
    for ty in &types {
        let id = ty.id();
        declarations.push(format!("%arr_cp_{id} = OpTypeArray %{id} %uint_cp"));
        declarations.push(format!("%ptr_input_{id} = OpTypePointer Input %{id}"));
        declarations.push(format!("%ptr_output_{id} = OpTypePointer Output %{id}"));
        declarations.push(format!(
            "%ptr_input_arr_cp_{id} = OpTypePointer Input %arr_cp_{id}"
        ));
        declarations.push(format!(
            "%ptr_output_arr_cp_{id} = OpTypePointer Output %arr_cp_{id}"
        ));
    }

    let mut constants = Vec::new();
    let mut patch_values = vec![Vec::new(); authored.factors.len()];
    for (input_index, (input, ty)) in patch_inputs.iter().enumerate() {
        for (patch, values) in patch_values.iter_mut().enumerate() {
            let offset = patch * input.stride as usize;
            let end = offset + ty.byte_size();
            let bytes = input.bytes.get(offset..end).ok_or_else(|| {
                format!(
                    "tessellation patch input {} record {patch} is truncated",
                    input.location
                )
            })?;
            let id = format!("patch_{patch}_{input_index}_value");
            emit_tess_constant(&mut constants, &id, *ty, bytes)?;
            values.push((input_index, id));
        }
    }
    let mut factor_ids = Vec::with_capacity(authored.factors.len());
    for (patch, factors) in authored.factors.iter().enumerate() {
        let outer = (0..4)
            .map(|index| {
                let bits = factors.edge_f16.get(index).copied().unwrap_or(0x3c00);
                let id = format!("factor_{patch}_outer_{index}");
                constants.push(format!(
                    "%{id} = OpConstant %float {}",
                    spv_f32_literal(f16_to_f32_bits(bits))?
                ));
                Ok::<_, String>(id)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let inner = (0..2)
            .map(|index| {
                let bits = factors.inside_f16.get(index).copied().unwrap_or(0x3c00);
                let id = format!("factor_{patch}_inner_{index}");
                constants.push(format!(
                    "%{id} = OpConstant %float {}",
                    spv_f32_literal(f16_to_f32_bits(bits))?
                ));
                Ok::<_, String>(id)
            })
            .collect::<Result<Vec<_>, _>>()?;
        factor_ids.push((outer, inner));
    }
    for (name, _, ty) in &system_inputs {
        if *name != "instance" {
            let value = if *name == "amplification_count" {
                authored.amplification_count
            } else {
                0
            };
            emit_tess_integer_constant(&mut constants, &format!("{name}_value"), *ty, value)?;
        }
    }

    let mut variables = vec![
        "%invocation = OpVariable %ptr_input_uint Input".into(),
        "%primitive = OpVariable %ptr_input_uint Input".into(),
        "%tess_outer = OpVariable %ptr_output_arr4float Output".into(),
        "%tess_inner = OpVariable %ptr_output_arr2float Output".into(),
    ];
    if !system_inputs.is_empty() {
        variables.push("%instance_index = OpVariable %ptr_input_uint Input".into());
    }
    for (index, (_, ty)) in control_points.iter().enumerate() {
        let id = ty.id();
        variables.extend([
            format!("%vin_{index} = OpVariable %ptr_input_{id} Input"),
            format!("%vout_{index} = OpVariable %ptr_output_{id} Output"),
            format!("%tcin_{index} = OpVariable %ptr_input_arr_cp_{id} Input"),
            format!("%tcout_{index} = OpVariable %ptr_output_arr_cp_{id} Output"),
        ]);
    }
    for (index, (_, ty)) in patch_inputs.iter().enumerate() {
        variables.push(format!(
            "%patch_{index} = OpVariable %ptr_output_{} Output",
            ty.id()
        ));
    }
    for (name, _, ty) in &system_inputs {
        let id = ty.id();
        variables.extend([
            format!("%vs_{name} = OpVariable %ptr_output_{id} Output"),
            format!("%tcs_{name}_in = OpVariable %ptr_input_arr_cp_{id} Input"),
            format!("%tcs_{name}_out = OpVariable %ptr_output_{id} Output"),
        ]);
    }

    let mut vertex_body = vec![
        "%vertex = OpFunction %void None %fn".into(),
        "%vertex_entry = OpLabel".into(),
    ];
    for (index, (_, ty)) in control_points.iter().enumerate() {
        vertex_body.extend([
            format!("%vertex_value_{index} = OpLoad %{} %vin_{index}", ty.id()),
            format!("OpStore %vout_{index} %vertex_value_{index}"),
        ]);
    }
    if !system_inputs.is_empty() {
        vertex_body.push("%instance_value = OpLoad %uint %instance_index".into());
    }
    for (name, _, ty) in &system_inputs {
        let value = if *name == "instance" {
            convert_uint_value(
                &mut vertex_body,
                "instance_value",
                &format!("vs_{name}_value"),
                *ty,
            )?
        } else {
            format!("%{name}_value")
        };
        vertex_body.push(format!("OpStore %vs_{name} {value}"));
    }
    vertex_body.extend(["OpReturn".into(), "OpFunctionEnd".into()]);

    let mut control_body = vec![
        "%control = OpFunction %void None %fn".into(),
        "%control_entry = OpLabel".into(),
        "%invocation_value = OpLoad %uint %invocation".into(),
    ];
    for (index, (_, ty)) in control_points.iter().enumerate() {
        control_body.extend([
            format!(
                "%tcin_ptr_{index} = OpAccessChain %ptr_input_{} %tcin_{index} %invocation_value",
                ty.id()
            ),
            format!(
                "%tcin_value_{index} = OpLoad %{} %tcin_ptr_{index}",
                ty.id()
            ),
            format!(
                "%tcout_ptr_{index} = OpAccessChain %ptr_output_{} %tcout_{index} %invocation_value",
                ty.id()
            ),
            format!("OpStore %tcout_ptr_{index} %tcin_value_{index}"),
        ]);
    }
    control_body.extend([
        "%is_invocation_zero = OpIEqual %bool %invocation_value %uint_0".into(),
        "OpSelectionMerge %control_merge None".into(),
        "OpBranchConditional %is_invocation_zero %patch_dispatch %control_merge".into(),
        "%patch_dispatch = OpLabel".into(),
        "%primitive_value = OpLoad %uint %primitive".into(),
        "OpSelectionMerge %patch_merge None".into(),
    ]);
    let switch_targets = (1..authored.factors.len())
        .map(|patch| format!("{patch} %patch_{patch}_block"))
        .collect::<Vec<_>>()
        .join(" ");
    control_body.push(format!(
        "OpSwitch %primitive_value %patch_0_block {switch_targets}"
    ));
    for patch in 0..authored.factors.len() {
        control_body.push(format!("%patch_{patch}_block = OpLabel"));
        for (input_index, id) in &patch_values[patch] {
            control_body.push(format!("OpStore %patch_{input_index} %{id}"));
        }
        for (index, id) in factor_ids[patch].0.iter().enumerate() {
            control_body.extend([
                format!(
                    "%outer_{patch}_{index}_ptr = OpAccessChain %ptr_output_float %tess_outer %uint_{index}"
                ),
                format!("OpStore %outer_{patch}_{index}_ptr %{id}"),
            ]);
        }
        for (index, id) in factor_ids[patch].1.iter().enumerate() {
            control_body.extend([
                format!(
                    "%inner_{patch}_{index}_ptr = OpAccessChain %ptr_output_float %tess_inner %uint_{index}"
                ),
                format!("OpStore %inner_{patch}_{index}_ptr %{id}"),
            ]);
        }
        for (name, _, ty) in &system_inputs {
            control_body.extend([
                format!(
                    "%tcs_{name}_{patch}_ptr = OpAccessChain %ptr_input_{} %tcs_{name}_in %uint_0",
                    ty.id()
                ),
                format!(
                    "%tcs_{name}_{patch}_value = OpLoad %{} %tcs_{name}_{patch}_ptr",
                    ty.id()
                ),
                format!("OpStore %tcs_{name}_out %tcs_{name}_{patch}_value"),
            ]);
        }
        control_body.push("OpBranch %patch_merge".into());
    }
    control_body.extend([
        "%patch_merge = OpLabel".into(),
        "OpBranch %control_merge".into(),
        "%control_merge = OpLabel".into(),
        "OpReturn".into(),
        "OpFunctionEnd".into(),
    ]);

    let mut module = Vec::new();
    module.push("OpCapability Shader".into());
    module.push("OpCapability Tessellation".into());
    if has_half {
        module.push("OpCapability Float16".into());
    }
    if has_int16 {
        module.push("OpCapability Int16".into());
    }
    if has_io16 {
        module.push("OpCapability StorageInputOutput16".into());
    }
    module.push("OpMemoryModel Logical GLSL450".into());
    module.push(format!(
        "OpEntryPoint Vertex %vertex \"vertex\" {}",
        vertex_interfaces.join(" ")
    ));
    module.push(format!(
        "OpEntryPoint TessellationControl %control \"control\" {}",
        control_interfaces.join(" ")
    ));
    module.push(format!(
        "OpExecutionMode %control OutputVertices {control_point_count}"
    ));
    if !system_inputs.is_empty() {
        module.push("OpDecorate %instance_index BuiltIn InstanceIndex".into());
    }
    module.extend([
        "OpDecorate %invocation BuiltIn InvocationId".into(),
        "OpDecorate %primitive BuiltIn PrimitiveId".into(),
        "OpDecorate %tess_outer BuiltIn TessLevelOuter".into(),
        "OpDecorate %tess_outer Patch".into(),
        "OpDecorate %tess_inner BuiltIn TessLevelInner".into(),
        "OpDecorate %tess_inner Patch".into(),
    ]);
    module.extend(decorations);
    module.extend(declarations);
    module.extend(constants);
    module.extend(variables);
    module.extend(vertex_body);
    module.extend(control_body);
    Ok(Some(module.join("\n") + "\n"))
}

fn emit_tess_constant(
    constants: &mut Vec<String>,
    id: &str,
    ty: TessType,
    bytes: &[u8],
) -> Result<(), String> {
    let scalar_size = ty.byte_size() / ty.lanes as usize;
    let mut components = Vec::new();
    for lane in 0..ty.lanes as usize {
        let component_id = if ty.lanes == 1 {
            id.to_string()
        } else {
            format!("{id}_{lane}")
        };
        let scalar_bytes = &bytes[lane * scalar_size..(lane + 1) * scalar_size];
        let literal = match ty.scalar {
            TessScalar::Half => {
                let bits = u16::from_le_bytes(scalar_bytes.try_into().expect("two bytes"));
                spv_f32_literal(f16_to_f32_bits(bits))?
            }
            TessScalar::Ushort => {
                u16::from_le_bytes(scalar_bytes.try_into().expect("two bytes")).to_string()
            }
            TessScalar::Short => {
                i16::from_le_bytes(scalar_bytes.try_into().expect("two bytes")).to_string()
            }
            TessScalar::Float => spv_f32_literal(u32::from_le_bytes(
                scalar_bytes.try_into().expect("four bytes"),
            ))?,
            TessScalar::Uint => format!(
                "0x{:08x}",
                u32::from_le_bytes(scalar_bytes.try_into().expect("four bytes"))
            ),
            TessScalar::Int => {
                i32::from_le_bytes(scalar_bytes.try_into().expect("four bytes")).to_string()
            }
        };
        let scalar_ty = TessType {
            scalar: ty.scalar,
            lanes: 1,
        }
        .id();
        constants.push(format!(
            "%{component_id} = OpConstant %{scalar_ty} {literal}"
        ));
        components.push(format!("%{component_id}"));
    }
    if ty.lanes > 1 {
        constants.push(format!(
            "%{id} = OpConstantComposite %{} {}",
            ty.id(),
            components.join(" ")
        ));
    }
    Ok(())
}

fn emit_tess_integer_constant(
    constants: &mut Vec<String>,
    id: &str,
    ty: TessType,
    value: u32,
) -> Result<(), String> {
    if ty.lanes != 1 || matches!(ty.scalar, TessScalar::Half | TessScalar::Float) {
        return Err(format!(
            "tessellation system value {id} must have an integer scalar type"
        ));
    }
    constants.push(format!("%{id} = OpConstant %{} {value}", ty.id()));
    Ok(())
}

fn convert_uint_value(
    body: &mut Vec<String>,
    source: &str,
    destination: &str,
    ty: TessType,
) -> Result<String, String> {
    if ty.lanes != 1 || matches!(ty.scalar, TessScalar::Half | TessScalar::Float) {
        return Err("tessellation instance ID must have an integer scalar type".into());
    }
    if ty.scalar == TessScalar::Uint {
        return Ok(format!("%{source}"));
    }
    let opcode = if matches!(ty.scalar, TessScalar::Int | TessScalar::Short) {
        "OpSConvert"
    } else {
        "OpUConvert"
    };
    body.push(format!("%{destination} = {opcode} %{} %{source}", ty.id()));
    Ok(format!("%{destination}"))
}

fn f16_to_f32_bits(bits: u16) -> u32 {
    let sign = ((bits as u32) & 0x8000) << 16;
    let exponent = ((bits >> 10) & 0x1f) as u32;
    let fraction = (bits & 0x03ff) as u32;
    match exponent {
        0 if fraction == 0 => sign,
        0 => {
            let shift = fraction.leading_zeros() - 21;
            let normalized = fraction << shift;
            sign | ((127 - 15 - shift + 1) << 23) | ((normalized & 0x03ff) << 13)
        }
        0x1f => sign | 0x7f80_0000 | (fraction << 13),
        _ => sign | ((exponent + 127 - 15) << 23) | (fraction << 13),
    }
}

fn spv_f32_literal(bits: u32) -> Result<String, String> {
    let value = f32::from_bits(bits);
    if !value.is_finite() {
        return Err(format!(
            "tessellation floating-point literal {bits:#010x} must be finite"
        ));
    }
    Ok(format!("{value:e}"))
}

fn validate_backend_host(backend: Backend) -> Result<(), String> {
    if backend == Backend::Moltenvk && !cfg!(target_os = "macos") {
        return Err("MoltenVK candidate execution requires macOS".into());
    }
    if backend == Backend::Vulkan && cfg!(target_os = "macos") {
        return Err(
            "native Vulkan candidate execution is not available on macOS; use corpus-moltenvk"
                .into(),
        );
    }
    Ok(())
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
mod platform {
    use super::*;
    use crate::case::OutputSelection;
    use ash::{vk, Device, Entry, Instance};
    use std::cell::RefCell;
    use std::ffi::{CStr, CString};
    use std::path::{Path, PathBuf};

    pub fn execute(
        case: &AuthoredCase,
        resources: &LiteralResources,
        reflection: &ShaderReflection,
        spv: &[u8],
        companion_spv: Option<&[u8]>,
        tessellation_spv: Option<&[u8]>,
        backend: Backend,
    ) -> Result<(Vec<u8>, serde_json::Value), String> {
        debug_assert!(validate_backend_host(backend).is_ok());
        debug_assert!(crate::executor_contract::require_case(case, "candidate executor").is_ok());
        thread_local! {
            static CONTEXT: RefCell<Option<VulkanContext>> = const { RefCell::new(None) };
        }
        CONTEXT.with(|slot| {
            let mut slot = slot.borrow_mut();
            if slot.is_none() {
                *slot = Some(VulkanContext::new()?);
            }
            execute_with_context(
                slot.as_ref().expect("initialized Vulkan context"),
                case,
                resources,
                reflection,
                spv,
                companion_spv,
                tessellation_spv,
            )
        })
    }

    fn execute_with_context(
        context: &VulkanContext,
        case: &AuthoredCase,
        resources: &LiteralResources,
        reflection: &ShaderReflection,
        spv: &[u8],
        companion_spv: Option<&[u8]>,
        tessellation_spv: Option<&[u8]>,
    ) -> Result<(Vec<u8>, serde_json::Value), String> {
        let environment = context.environment();
        if reflection.tessellation.is_some() && !context.tessellation_shader {
            return Err("Vulkan device does not support tessellation shaders".into());
        }
        if !reflection.stencil_members.is_empty() && !context.shader_stencil_export {
            return Err("Vulkan device does not support VK_EXT_shader_stencil_export".into());
        }
        if reflection.fragment_imageblock.is_some() && !context.fragment_shader_pixel_interlock {
            return Err("Vulkan device does not support VK_EXT_fragment_shader_interlock".into());
        }
        let buffers = create_buffers(context, resources, reflection)?;
        let images = create_images(context, resources, reflection)?;
        let texel_buffers = create_texel_buffers(context, resources, reflection)?;
        let samplers = create_samplers(context, case, reflection)?;
        let render_targets = create_render_targets(context, resources)?;
        let depth_stencil = create_depth_stencil(context, resources)?;
        let graphics_attachments = GraphicsAttachments {
            colors: &render_targets,
            depth_stencil: depth_stencil.as_ref(),
        };
        let vertex_inputs = create_vertex_inputs(context, resources)?;
        let framebuffer_fetch = reflection
            .bindings
            .iter()
            .any(|binding| binding.kind == metal2vulkan::reflect::ResourceKind::ColorInput);
        let descriptor_bindings = descriptor_bindings(reflection)?;
        let set_layout_info =
            vk::DescriptorSetLayoutCreateInfo::default().bindings(&descriptor_bindings);
        let set_layout = unsafe {
            context
                .device
                .create_descriptor_set_layout(&set_layout_info, None)
        }
        .map_err(|error| format!("create descriptor-set layout: {error}"))?;
        let mut objects = DeviceObjects::new(&context.device);
        objects.set_layout = set_layout;
        objects.framebuffer_fetch = framebuffer_fetch;

        let set_layouts = [set_layout];
        let pipeline_layout_info =
            vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts);
        objects.pipeline_layout = unsafe {
            context
                .device
                .create_pipeline_layout(&pipeline_layout_info, None)
        }
        .map_err(|error| format!("create pipeline layout: {error}"))?;

        objects.shader = create_shader_module(context, spv, "primary")?;
        match case.stage {
            Stage::Kernel => create_compute_pipeline(context, &mut objects)?,
            Stage::Fragment | Stage::Vertex => {
                if let Some(companion_spv) = companion_spv {
                    objects.companion_shader = create_shader_module(
                        context,
                        companion_spv,
                        "graphics observation companion",
                    )?;
                } else if !case.is_rasterization_disabled_vertex() {
                    return Err("graphics candidate has no observation companion".into());
                }
                if let Some(tessellation_spv) = tessellation_spv {
                    objects.tessellation_shader = create_shader_module(
                        context,
                        tessellation_spv,
                        "tessellation vertex/control companion",
                    )?;
                }
                create_graphics_pipeline(
                    context,
                    case,
                    reflection,
                    graphics_attachments,
                    &vertex_inputs,
                    &mut objects,
                )?;
            }
        }

        let pool_sizes = descriptor_pool_sizes(&descriptor_bindings);
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(1)
            .pool_sizes(&pool_sizes);
        objects.descriptor_pool =
            unsafe { context.device.create_descriptor_pool(&pool_info, None) }
                .map_err(|error| format!("create descriptor pool: {error}"))?;
        let allocation_layouts = [set_layout];
        let allocation_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(objects.descriptor_pool)
            .set_layouts(&allocation_layouts);
        let descriptor_set = unsafe { context.device.allocate_descriptor_sets(&allocation_info) }
            .map_err(|error| format!("allocate descriptor set: {error}"))?[0];
        let descriptor_buffers = buffers
            .iter()
            .filter(|buffer| buffer.descriptor_binding.is_some())
            .collect::<Vec<_>>();
        let buffer_infos = descriptor_buffers
            .iter()
            .map(|buffer| {
                [vk::DescriptorBufferInfo::default()
                    .buffer(buffer.buffer)
                    .offset(0)
                    .range(buffer.len)]
            })
            .collect::<Vec<_>>();
        let mut writes = descriptor_buffers
            .iter()
            .zip(&buffer_infos)
            .map(|(buffer, info)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(descriptor_set)
                    .dst_binding(buffer.descriptor_binding.expect("filtered above"))
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(info)
            })
            .collect::<Vec<_>>();
        let image_targets = images
            .iter()
            .flat_map(|image| {
                image
                    .descriptor_targets()
                    .map(move |target| (image, target))
            })
            .collect::<Vec<_>>();
        let mut image_groups = image_targets
            .iter()
            .filter(|(_, target)| target.count == 1)
            .map(|target| vec![*target])
            .collect::<Vec<_>>();
        let mut array_groups = std::collections::BTreeMap::<
            (u32, i32),
            Vec<(u32, &ImageAllocation, TextureDescriptorTarget)>,
        >::new();
        for image in &images {
            for target in image.descriptor_targets() {
                if target.count > 1 {
                    array_groups
                        .entry((target.binding, target.descriptor_type.as_raw()))
                        .or_default()
                        .push((target.element, image, target));
                }
            }
        }
        for elements in array_groups.values_mut() {
            elements.sort_by_key(|(element, _, _)| *element);
            image_groups.push(
                elements
                    .iter()
                    .map(|(_, image, target)| (*image, *target))
                    .collect(),
            );
        }
        let image_infos = image_groups
            .iter()
            .map(|group| {
                let descriptor_count = group[0].1.count as usize;
                (0..descriptor_count)
                    .map(|index| {
                        let image = group[index.min(group.len() - 1)].0;
                        vk::DescriptorImageInfo::default()
                            .image_view(image.view)
                            .image_layout(vk::ImageLayout::GENERAL)
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        writes.extend(image_groups.iter().zip(&image_infos).map(|(group, info)| {
            let target = group[0].1;
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(target.binding)
                .descriptor_type(target.descriptor_type)
                .image_info(info)
        }));
        let texel_targets = texel_buffers
            .iter()
            .flat_map(|buffer| {
                buffer
                    .descriptor_targets()
                    .map(move |target| (buffer, target))
            })
            .collect::<Vec<_>>();
        let texel_views = texel_targets
            .iter()
            .map(|(buffer, _)| [buffer.view])
            .collect::<Vec<_>>();
        writes.extend(
            texel_targets
                .iter()
                .zip(&texel_views)
                .map(|((_, target), views)| {
                    vk::WriteDescriptorSet::default()
                        .dst_set(descriptor_set)
                        .dst_binding(target.binding)
                        .descriptor_type(target.descriptor_type)
                        .texel_buffer_view(views)
                }),
        );
        let sampler_infos = samplers
            .iter()
            .map(|sampler| [vk::DescriptorImageInfo::default().sampler(sampler.sampler)])
            .collect::<Vec<_>>();
        writes.extend(samplers.iter().zip(&sampler_infos).map(|(sampler, info)| {
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(sampler.descriptor_binding)
                .descriptor_type(vk::DescriptorType::SAMPLER)
                .image_info(info)
        }));
        let color_input_targets = reflection
            .bindings
            .iter()
            .filter(|binding| binding.kind == metal2vulkan::reflect::ResourceKind::ColorInput)
            .map(|binding| {
                let target = render_targets
                    .iter()
                    .find(|target| target.index == binding.metal_index)
                    .ok_or_else(|| {
                        format!(
                            "framebuffer-fetch input {} has no authored render target",
                            binding.metal_index
                        )
                    })?;
                let descriptor = binding.descriptor.ok_or_else(|| {
                    format!(
                        "framebuffer-fetch input {} has no descriptor binding",
                        binding.metal_index
                    )
                })?;
                Ok((descriptor.binding, target))
            })
            .collect::<Result<Vec<_>, String>>()?;
        let color_input_infos = color_input_targets
            .iter()
            .map(|(_, target)| {
                [vk::DescriptorImageInfo::default()
                    .image_view(target.view)
                    .image_layout(vk::ImageLayout::GENERAL)]
            })
            .collect::<Vec<_>>();
        writes.extend(color_input_targets.iter().zip(&color_input_infos).map(
            |((binding, _), info)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(descriptor_set)
                    .dst_binding(*binding)
                    .descriptor_type(vk::DescriptorType::INPUT_ATTACHMENT)
                    .image_info(info)
            },
        ));
        unsafe { context.device.update_descriptor_sets(&writes, &[]) };

        let command_pool_info =
            vk::CommandPoolCreateInfo::default().queue_family_index(context.queue_family);
        objects.command_pool =
            unsafe { context.device.create_command_pool(&command_pool_info, None) }
                .map_err(|error| format!("create command pool: {error}"))?;
        let command_allocation = vk::CommandBufferAllocateInfo::default()
            .command_pool(objects.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let command = unsafe { context.device.allocate_command_buffers(&command_allocation) }
            .map_err(|error| format!("allocate command buffer: {error}"))?[0];
        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe {
            context
                .device
                .begin_command_buffer(command, &begin)
                .map_err(|error| format!("begin command buffer: {error}"))?;
            transition_images_to_general(&context.device, command, &images);
            prepare_texel_buffers(&context.device, command, &texel_buffers);
            match case.stage {
                Stage::Kernel => {
                    encode_compute(&context.device, command, case, descriptor_set, &objects)
                }
                Stage::Fragment | Stage::Vertex => encode_graphics(
                    &context.device,
                    command,
                    case,
                    descriptor_set,
                    graphics_attachments,
                    &vertex_inputs,
                    &objects,
                ),
            }
            make_images_host_readable(&context.device, command, &images);
            make_texel_buffers_host_readable(&context.device, command, &texel_buffers);
            context
                .device
                .end_command_buffer(command)
                .map_err(|error| format!("end command buffer: {error}"))?;
        }
        let command_buffers = [command];
        let submits = [vk::SubmitInfo::default().command_buffers(&command_buffers)];
        objects.fence = unsafe {
            context
                .device
                .create_fence(&vk::FenceCreateInfo::default(), None)
        }
        .map_err(|error| format!("create completion fence: {error}"))?;
        unsafe {
            context
                .device
                .queue_submit(context.queue, &submits, objects.fence)
                .map_err(|error| format!("submit command buffer: {error}"))?;
            context
                .device
                .wait_for_fences(&[objects.fence], true, u64::MAX)
                .map_err(|error| format!("wait for candidate completion: {error}"))?;
        }
        Ok((
            selected_output(
                context,
                case,
                resources,
                &buffers,
                BoundImageResources {
                    images: &images,
                    texel_buffers: &texel_buffers,
                    reflection,
                },
                &render_targets,
                depth_stencil.as_ref(),
            )?,
            environment,
        ))
    }

    fn create_shader_module(
        context: &VulkanContext,
        spv: &[u8],
        label: &str,
    ) -> Result<vk::ShaderModule, String> {
        if !spv.len().is_multiple_of(4) {
            return Err(format!("{label} SPIR-V byte stream is not word-aligned"));
        }
        let words = spv
            .chunks_exact(4)
            .map(|word| u32::from_le_bytes(word.try_into().expect("four-byte chunk")))
            .collect::<Vec<_>>();
        let shader_info = vk::ShaderModuleCreateInfo::default().code(&words);
        unsafe { context.device.create_shader_module(&shader_info, None) }
            .map_err(|error| format!("create {label} shader module: {error}"))
    }

    fn create_compute_pipeline(
        context: &VulkanContext,
        objects: &mut DeviceObjects,
    ) -> Result<(), String> {
        let main = CString::new("main").expect("static entry name");
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(objects.shader)
            .name(&main);
        let pipeline_info = vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(objects.pipeline_layout);
        objects.pipeline = unsafe {
            context.device.create_compute_pipelines(
                vk::PipelineCache::null(),
                &[pipeline_info],
                None,
            )
        }
        .map_err(|(_, error)| format!("create compute pipeline: {error}"))?[0];
        Ok(())
    }

    fn create_graphics_pipeline(
        context: &VulkanContext,
        case: &AuthoredCase,
        reflection: &ShaderReflection,
        attachments: GraphicsAttachments<'_>,
        vertex_inputs: &[VertexInputAllocation],
        objects: &mut DeviceObjects,
    ) -> Result<(), String> {
        let targets = attachments.colors;
        let depth_stencil = attachments.depth_stencil;
        let dimensions = targets
            .first()
            .map(|target| target.dimensions)
            .or_else(|| depth_stencil.map(|attachment| attachment.dimensions))
            .unwrap_or([1, 1]);
        let max_index = targets.iter().map(|target| target.index).max();
        let color_inputs = reflection
            .bindings
            .iter()
            .filter(|binding| binding.kind == metal2vulkan::reflect::ResourceKind::ColorInput)
            .collect::<Vec<_>>();
        let attachment_layout = if color_inputs.is_empty() {
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
        } else {
            vk::ImageLayout::GENERAL
        };
        let mut attachments = targets
            .iter()
            .map(|target| {
                vk::AttachmentDescription::default()
                    .format(vulkan_format(target.format))
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .load_op(vk::AttachmentLoadOp::LOAD)
                    .store_op(vk::AttachmentStoreOp::STORE)
                    .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                    .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                    .initial_layout(attachment_layout)
                    .final_layout(attachment_layout)
            })
            .collect::<Vec<_>>();
        let mut color_references = vec![
            vk::AttachmentReference {
                attachment: vk::ATTACHMENT_UNUSED,
                layout: vk::ImageLayout::UNDEFINED,
            };
            max_index.map_or(0, |index| index as usize + 1)
        ];
        for (attachment, target) in targets.iter().enumerate() {
            color_references[target.index as usize] = vk::AttachmentReference {
                attachment: attachment as u32,
                layout: attachment_layout,
            };
        }
        let mut input_references = color_inputs
            .iter()
            .map(|binding| binding.metal_index)
            .max()
            .map(|index| {
                vec![
                    vk::AttachmentReference {
                        attachment: vk::ATTACHMENT_UNUSED,
                        layout: vk::ImageLayout::UNDEFINED,
                    };
                    index as usize + 1
                ]
            })
            .unwrap_or_default();
        for binding in color_inputs {
            let attachment = targets
                .iter()
                .position(|target| target.index == binding.metal_index)
                .ok_or_else(|| {
                    format!(
                        "framebuffer-fetch input {} has no render-pass attachment",
                        binding.metal_index
                    )
                })?;
            input_references[binding.metal_index as usize] = vk::AttachmentReference {
                attachment: attachment as u32,
                layout: vk::ImageLayout::GENERAL,
            };
        }
        let depth_reference = depth_stencil.map(|_| vk::AttachmentReference {
            attachment: attachments.len() as u32,
            layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        });
        if let Some(attachment) = depth_stencil {
            attachments.push(
                vk::AttachmentDescription::default()
                    .format(if attachment.stencil.is_some() {
                        vk::Format::D32_SFLOAT_S8_UINT
                    } else {
                        vk::Format::D32_SFLOAT
                    })
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .load_op(if attachment.depth.is_some() {
                        vk::AttachmentLoadOp::LOAD
                    } else {
                        vk::AttachmentLoadOp::DONT_CARE
                    })
                    .store_op(if attachment.depth.is_some() {
                        vk::AttachmentStoreOp::STORE
                    } else {
                        vk::AttachmentStoreOp::DONT_CARE
                    })
                    .stencil_load_op(if attachment.stencil.is_some() {
                        vk::AttachmentLoadOp::LOAD
                    } else {
                        vk::AttachmentLoadOp::DONT_CARE
                    })
                    .stencil_store_op(if attachment.stencil.is_some() {
                        vk::AttachmentStoreOp::STORE
                    } else {
                        vk::AttachmentStoreOp::DONT_CARE
                    })
                    .initial_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                    .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL),
            );
        }
        let mut subpass = vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .input_attachments(&input_references)
            .color_attachments(&color_references);
        if let Some(reference) = depth_reference.as_ref() {
            subpass = subpass.depth_stencil_attachment(reference);
        }
        let subpasses = [subpass];
        let dependencies = if input_references.is_empty() {
            Vec::new()
        } else {
            vec![vk::SubpassDependency::default()
                .src_subpass(0)
                .dst_subpass(0)
                .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
                .dst_stage_mask(vk::PipelineStageFlags::FRAGMENT_SHADER)
                .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                .dst_access_mask(vk::AccessFlags::INPUT_ATTACHMENT_READ)
                .dependency_flags(vk::DependencyFlags::BY_REGION)]
        };
        let render_pass_info = vk::RenderPassCreateInfo::default()
            .attachments(&attachments)
            .subpasses(&subpasses)
            .dependencies(&dependencies);
        objects.render_pass = unsafe { context.device.create_render_pass(&render_pass_info, None) }
            .map_err(|error| format!("create fragment render pass: {error}"))?;

        let mut views = targets.iter().map(|target| target.view).collect::<Vec<_>>();
        if let Some(attachment) = depth_stencil {
            views.push(attachment.view);
        }
        let framebuffer_info = vk::FramebufferCreateInfo::default()
            .render_pass(objects.render_pass)
            .attachments(&views)
            .width(dimensions[0])
            .height(dimensions[1])
            .layers(1);
        objects.framebuffer = unsafe { context.device.create_framebuffer(&framebuffer_info, None) }
            .map_err(|error| format!("create fragment framebuffer: {error}"))?;

        let main = CString::new("main").expect("static entry name");
        let vertex_name = CString::new("vertex").expect("static entry name");
        let control_name = CString::new("control").expect("static entry name");
        let stages = if case.is_rasterization_disabled_vertex() {
            vec![vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(objects.shader)
                .name(&main)]
        } else if case.tessellation.is_some() {
            vec![
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::VERTEX)
                    .module(objects.tessellation_shader)
                    .name(&vertex_name),
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::TESSELLATION_CONTROL)
                    .module(objects.tessellation_shader)
                    .name(&control_name),
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::TESSELLATION_EVALUATION)
                    .module(objects.shader)
                    .name(&main),
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::FRAGMENT)
                    .module(objects.companion_shader)
                    .name(&main),
            ]
        } else {
            let (vertex_shader, fragment_shader) = match case.stage {
                Stage::Fragment => (objects.companion_shader, objects.shader),
                Stage::Vertex => (objects.shader, objects.companion_shader),
                Stage::Kernel => unreachable!("graphics pipeline for kernel"),
            };
            vec![
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::VERTEX)
                    .module(vertex_shader)
                    .name(&main),
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::FRAGMENT)
                    .module(fragment_shader)
                    .name(&main),
            ]
        };
        let vertex_bindings = vertex_inputs
            .iter()
            .enumerate()
            .map(|(binding, input)| vk::VertexInputBindingDescription {
                binding: binding as u32,
                stride: input.stride,
                input_rate: vk::VertexInputRate::VERTEX,
            })
            .collect::<Vec<_>>();
        let vertex_attributes = vertex_inputs
            .iter()
            .enumerate()
            .map(|(binding, input)| vk::VertexInputAttributeDescription {
                location: input.location,
                binding: binding as u32,
                format: vulkan_attribute_format(input.format),
                offset: 0,
            })
            .collect::<Vec<_>>();
        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(&vertex_bindings)
            .vertex_attribute_descriptions(&vertex_attributes);
        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default().topology(
            if case.tessellation.is_some() {
                vk::PrimitiveTopology::PATCH_LIST
            } else {
                vulkan_primitive(
                    case.draw
                        .as_ref()
                        .expect("validated graphics draw")
                        .primitive,
                )
            },
        );
        let tessellation_state = reflection.tessellation.as_ref().map(|interface| {
            vk::PipelineTessellationStateCreateInfo::default()
                .patch_control_points(interface.control_point_count)
        });
        objects.patch_control_points = reflection
            .tessellation
            .as_ref()
            .map_or(0, |interface| interface.control_point_count);
        let viewports = [vk::Viewport {
            x: 0.0,
            y: if case.stage == Stage::Vertex {
                dimensions[1] as f32
            } else {
                0.0
            },
            width: dimensions[0] as f32,
            height: if case.stage == Stage::Vertex {
                -(dimensions[1] as f32)
            } else {
                dimensions[1] as f32
            },
            min_depth: 0.0,
            max_depth: 1.0,
        }];
        let scissors = [vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: vk::Extent2D {
                width: dimensions[0],
                height: dimensions[1],
            },
        }];
        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewports(&viewports)
            .scissors(&scissors);
        let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
            .rasterizer_discard_enable(case.is_rasterization_disabled_vertex())
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::NONE)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .line_width(1.0);
        let multisample = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);
        let blend_attachments = (0..color_references.len())
            .map(|_| {
                vk::PipelineColorBlendAttachmentState::default().color_write_mask(
                    vk::ColorComponentFlags::R
                        | vk::ColorComponentFlags::G
                        | vk::ColorComponentFlags::B
                        | vk::ColorComponentFlags::A,
                )
            })
            .collect::<Vec<_>>();
        let color_blend =
            vk::PipelineColorBlendStateCreateInfo::default().attachments(&blend_attachments);
        let depth_enabled = !reflection.depth_members.is_empty();
        let stencil_enabled = !reflection.stencil_members.is_empty();
        let stencil = vk::StencilOpState::default()
            .fail_op(vk::StencilOp::KEEP)
            .pass_op(vk::StencilOp::REPLACE)
            .depth_fail_op(vk::StencilOp::KEEP)
            .compare_op(vk::CompareOp::ALWAYS)
            .compare_mask(u32::MAX)
            .write_mask(u32::MAX);
        let depth_compare = match crate::executor_contract::depth_compare(reflection) {
            crate::executor_contract::DepthCompare::Always => vk::CompareOp::ALWAYS,
            crate::executor_contract::DepthCompare::Less => vk::CompareOp::LESS,
            crate::executor_contract::DepthCompare::Greater => vk::CompareOp::GREATER,
        };
        let depth_stencil_state = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(depth_enabled)
            .depth_write_enable(depth_enabled)
            .depth_compare_op(depth_compare)
            .stencil_test_enable(stencil_enabled)
            .front(stencil)
            .back(stencil);
        let mut pipeline_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterization)
            .multisample_state(&multisample)
            .depth_stencil_state(&depth_stencil_state)
            .color_blend_state(&color_blend)
            .layout(objects.pipeline_layout)
            .render_pass(objects.render_pass)
            .subpass(0);
        if let Some(tessellation_state) = tessellation_state.as_ref() {
            pipeline_info = pipeline_info.tessellation_state(tessellation_state);
        }
        objects.pipeline = unsafe {
            context.device.create_graphics_pipelines(
                vk::PipelineCache::null(),
                &[pipeline_info],
                None,
            )
        }
        .map_err(|(_, error)| format!("create fragment graphics pipeline: {error}"))?[0];
        Ok(())
    }

    unsafe fn encode_compute(
        device: &Device,
        command: vk::CommandBuffer,
        case: &AuthoredCase,
        descriptor_set: vk::DescriptorSet,
        objects: &DeviceObjects,
    ) {
        unsafe {
            device.cmd_bind_pipeline(command, vk::PipelineBindPoint::COMPUTE, objects.pipeline);
            device.cmd_bind_descriptor_sets(
                command,
                vk::PipelineBindPoint::COMPUTE,
                objects.pipeline_layout,
                0,
                &[descriptor_set],
                &[],
            );
            let dispatch = case.dispatch.as_ref().expect("validated kernel dispatch");
            device.cmd_dispatch(
                command,
                div_ceil(dispatch.grid[0], dispatch.threads_per_threadgroup[0]),
                div_ceil(dispatch.grid[1], dispatch.threads_per_threadgroup[1]),
                div_ceil(dispatch.grid[2], dispatch.threads_per_threadgroup[2]),
            );
        }
    }

    unsafe fn encode_graphics(
        device: &Device,
        command: vk::CommandBuffer,
        case: &AuthoredCase,
        descriptor_set: vk::DescriptorSet,
        attachments: GraphicsAttachments<'_>,
        vertex_inputs: &[VertexInputAllocation],
        objects: &DeviceObjects,
    ) {
        let targets = attachments.colors;
        let depth_stencil = attachments.depth_stencil;
        let attachment_layout = if objects.framebuffer_fetch {
            vk::ImageLayout::GENERAL
        } else {
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
        };
        let attachment_access = vk::AccessFlags::COLOR_ATTACHMENT_READ
            | vk::AccessFlags::COLOR_ATTACHMENT_WRITE
            | if objects.framebuffer_fetch {
                vk::AccessFlags::INPUT_ATTACHMENT_READ
            } else {
                vk::AccessFlags::empty()
            };
        let attachment_stages = vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
            | if objects.framebuffer_fetch {
                vk::PipelineStageFlags::FRAGMENT_SHADER
            } else {
                vk::PipelineStageFlags::empty()
            };
        let to_transfer = targets
            .iter()
            .map(|target| {
                render_target_barrier(
                    target,
                    vk::ImageLayout::UNDEFINED,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    vk::AccessFlags::empty(),
                    vk::AccessFlags::TRANSFER_WRITE,
                )
            })
            .collect::<Vec<_>>();
        let depth_to_transfer = depth_stencil.map(|attachment| {
            depth_stencil_barrier(
                attachment,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::AccessFlags::empty(),
                vk::AccessFlags::TRANSFER_WRITE,
            )
        });
        let mut initial_barriers = to_transfer;
        initial_barriers.extend(depth_to_transfer);
        unsafe {
            device.cmd_pipeline_barrier(
                command,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &initial_barriers,
            );
            for target in targets {
                let copy = vk::BufferImageCopy::default()
                    .image_subresource(vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: 0,
                        base_array_layer: 0,
                        layer_count: 1,
                    })
                    .image_extent(vk::Extent3D {
                        width: target.dimensions[0],
                        height: target.dimensions[1],
                        depth: 1,
                    });
                device.cmd_copy_buffer_to_image(
                    command,
                    target.transfer.buffer,
                    target.image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[copy],
                );
            }
            if let Some(attachment) = depth_stencil {
                for (aspect, transfer) in attachment.aspect_transfers() {
                    let copy = vk::BufferImageCopy::default()
                        .image_subresource(vk::ImageSubresourceLayers {
                            aspect_mask: aspect,
                            mip_level: 0,
                            base_array_layer: 0,
                            layer_count: 1,
                        })
                        .image_extent(vk::Extent3D {
                            width: attachment.dimensions[0],
                            height: attachment.dimensions[1],
                            depth: 1,
                        });
                    device.cmd_copy_buffer_to_image(
                        command,
                        transfer.buffer,
                        attachment.image,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        &[copy],
                    );
                }
            }
            let to_color = targets
                .iter()
                .map(|target| {
                    render_target_barrier(
                        target,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        attachment_layout,
                        vk::AccessFlags::TRANSFER_WRITE,
                        attachment_access,
                    )
                })
                .collect::<Vec<_>>();
            let depth_to_attachment = depth_stencil.map(|attachment| {
                depth_stencil_barrier(
                    attachment,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                    vk::AccessFlags::TRANSFER_WRITE,
                    vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                        | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                )
            });
            let mut attachment_barriers = to_color;
            attachment_barriers.extend(depth_to_attachment);
            device.cmd_pipeline_barrier(
                command,
                vk::PipelineStageFlags::TRANSFER,
                attachment_stages
                    | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
                    | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &attachment_barriers,
            );
            let dimensions = targets
                .first()
                .map(|target| target.dimensions)
                .or_else(|| depth_stencil.map(|attachment| attachment.dimensions))
                .unwrap_or([1, 1]);
            let render_pass = vk::RenderPassBeginInfo::default()
                .render_pass(objects.render_pass)
                .framebuffer(objects.framebuffer)
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: vk::Extent2D {
                        width: dimensions[0],
                        height: dimensions[1],
                    },
                });
            device.cmd_begin_render_pass(command, &render_pass, vk::SubpassContents::INLINE);
            device.cmd_bind_pipeline(command, vk::PipelineBindPoint::GRAPHICS, objects.pipeline);
            device.cmd_bind_descriptor_sets(
                command,
                vk::PipelineBindPoint::GRAPHICS,
                objects.pipeline_layout,
                0,
                &[descriptor_set],
                &[],
            );
            if !vertex_inputs.is_empty() {
                let buffers = vertex_inputs
                    .iter()
                    .map(|input| input.buffer.buffer)
                    .collect::<Vec<_>>();
                let offsets = vec![0; buffers.len()];
                device.cmd_bind_vertex_buffers(command, 0, &buffers, &offsets);
            }
            if let Some(tessellation) = &case.tessellation {
                device.cmd_draw(
                    command,
                    tessellation.factors.len() as u32 * objects.patch_control_points,
                    tessellation.instance_count,
                    0,
                    0,
                );
            } else {
                let draw = case.draw.as_ref().expect("validated graphics draw");
                device.cmd_draw(
                    command,
                    draw.vertex_count,
                    draw.instance_count,
                    draw.vertex_start,
                    0,
                );
            }
            device.cmd_end_render_pass(command);
            let to_readback = targets
                .iter()
                .map(|target| {
                    render_target_barrier(
                        target,
                        attachment_layout,
                        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                        vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                        vk::AccessFlags::TRANSFER_READ,
                    )
                })
                .collect::<Vec<_>>();
            let depth_to_readback = depth_stencil.map(|attachment| {
                depth_stencil_barrier(
                    attachment,
                    vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                    vk::AccessFlags::TRANSFER_READ,
                )
            });
            let mut readback_barriers = to_readback;
            readback_barriers.extend(depth_to_readback);
            device.cmd_pipeline_barrier(
                command,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                    | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
                    | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &readback_barriers,
            );
            let buffer_barriers = targets
                .iter()
                .map(|target| {
                    vk::BufferMemoryBarrier::default()
                        .src_access_mask(vk::AccessFlags::TRANSFER_READ)
                        .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .buffer(target.transfer.buffer)
                        .offset(0)
                        .size(target.transfer.len)
                })
                .collect::<Vec<_>>();
            device.cmd_pipeline_barrier(
                command,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &buffer_barriers,
                &[],
            );
            for target in targets {
                let copy = vk::BufferImageCopy::default()
                    .image_subresource(vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: 0,
                        base_array_layer: 0,
                        layer_count: 1,
                    })
                    .image_extent(vk::Extent3D {
                        width: target.dimensions[0],
                        height: target.dimensions[1],
                        depth: 1,
                    });
                device.cmd_copy_image_to_buffer(
                    command,
                    target.image,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    target.transfer.buffer,
                    &[copy],
                );
            }
            if let Some(attachment) = depth_stencil {
                for (aspect, transfer) in attachment.aspect_transfers() {
                    let copy = vk::BufferImageCopy::default()
                        .image_subresource(vk::ImageSubresourceLayers {
                            aspect_mask: aspect,
                            mip_level: 0,
                            base_array_layer: 0,
                            layer_count: 1,
                        })
                        .image_extent(vk::Extent3D {
                            width: attachment.dimensions[0],
                            height: attachment.dimensions[1],
                            depth: 1,
                        });
                    device.cmd_copy_image_to_buffer(
                        command,
                        attachment.image,
                        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                        transfer.buffer,
                        &[copy],
                    );
                }
            }
            let host_barriers = targets
                .iter()
                .map(|target| {
                    vk::BufferMemoryBarrier::default()
                        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                        .dst_access_mask(vk::AccessFlags::HOST_READ)
                        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .buffer(target.transfer.buffer)
                        .offset(0)
                        .size(target.transfer.len)
                })
                .collect::<Vec<_>>();
            let mut host_barriers = host_barriers;
            if let Some(attachment) = depth_stencil {
                host_barriers.extend(attachment.aspect_transfers().map(|(_, transfer)| {
                    vk::BufferMemoryBarrier::default()
                        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                        .dst_access_mask(vk::AccessFlags::HOST_READ)
                        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .buffer(transfer.buffer)
                        .offset(0)
                        .size(transfer.len)
                }));
            }
            device.cmd_pipeline_barrier(
                command,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &[],
                &host_barriers,
                &[],
            );
        }
    }

    fn render_target_barrier(
        target: &RenderTargetAllocation,
        old_layout: vk::ImageLayout,
        new_layout: vk::ImageLayout,
        src_access: vk::AccessFlags,
        dst_access: vk::AccessFlags,
    ) -> vk::ImageMemoryBarrier<'static> {
        vk::ImageMemoryBarrier::default()
            .src_access_mask(src_access)
            .dst_access_mask(dst_access)
            .old_layout(old_layout)
            .new_layout(new_layout)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(target.image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            })
    }

    fn depth_stencil_barrier(
        attachment: &DepthStencilAllocation,
        old_layout: vk::ImageLayout,
        new_layout: vk::ImageLayout,
        src_access: vk::AccessFlags,
        dst_access: vk::AccessFlags,
    ) -> vk::ImageMemoryBarrier<'static> {
        vk::ImageMemoryBarrier::default()
            .src_access_mask(src_access)
            .dst_access_mask(dst_access)
            .old_layout(old_layout)
            .new_layout(new_layout)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(attachment.image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: attachment.aspect,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            })
    }

    fn vulkan_primitive(primitive: crate::case::Primitive) -> vk::PrimitiveTopology {
        match primitive {
            crate::case::Primitive::Point => vk::PrimitiveTopology::POINT_LIST,
            crate::case::Primitive::Line => vk::PrimitiveTopology::LINE_LIST,
            crate::case::Primitive::LineStrip => vk::PrimitiveTopology::LINE_STRIP,
            crate::case::Primitive::Triangle => vk::PrimitiveTopology::TRIANGLE_LIST,
            crate::case::Primitive::TriangleStrip => vk::PrimitiveTopology::TRIANGLE_STRIP,
        }
    }

    struct VulkanContext {
        _entry: Entry,
        instance: Instance,
        device: Device,
        physical: vk::PhysicalDevice,
        queue_family: u32,
        queue: vk::Queue,
        buffer_device_address: bool,
        sampler_anisotropy: bool,
        sample_rate_shading: bool,
        tessellation_shader: bool,
        shader_stencil_export: bool,
        fragment_shader_pixel_interlock: bool,
        max_sampler_anisotropy: f32,
    }

    impl VulkanContext {
        fn new() -> Result<Self, String> {
            let entry = load_entry()?;
            let available = unsafe { entry.enumerate_instance_extension_properties(None) }
                .map_err(|error| format!("enumerate Vulkan instance extensions: {error}"))?;
            let portability_name = ash::khr::portability_enumeration::NAME;
            let portability = available
                .iter()
                .any(|extension| extension_name(&extension.extension_name) == portability_name);
            let extension_names = portability
                .then_some(portability_name.as_ptr())
                .into_iter()
                .collect::<Vec<_>>();
            let app_name = CString::new("metal2vulkan-validation").expect("static app name");
            let application = vk::ApplicationInfo::default()
                .application_name(&app_name)
                .application_version(1)
                .engine_name(&app_name)
                .engine_version(1)
                .api_version(vk::API_VERSION_1_3);
            let create_info = vk::InstanceCreateInfo::default()
                .application_info(&application)
                .enabled_extension_names(&extension_names)
                .flags(if portability {
                    vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR
                } else {
                    vk::InstanceCreateFlags::empty()
                });
            let instance = unsafe { entry.create_instance(&create_info, None) }
                .map_err(|error| format!("create Vulkan instance: {error}"))?;
            let physical_devices = unsafe { instance.enumerate_physical_devices() }
                .map_err(|error| format!("enumerate Vulkan devices: {error}"))?;
            let (physical, queue_family) = physical_devices
                .into_iter()
                .filter_map(|physical| {
                    let families =
                        unsafe { instance.get_physical_device_queue_family_properties(physical) };
                    let family = families
                        .iter()
                        .position(|family| family.queue_flags.contains(vk::QueueFlags::COMPUTE))?;
                    let properties = unsafe { instance.get_physical_device_properties(physical) };
                    let rank = match properties.device_type {
                        vk::PhysicalDeviceType::DISCRETE_GPU => 0,
                        vk::PhysicalDeviceType::INTEGRATED_GPU => 1,
                        vk::PhysicalDeviceType::VIRTUAL_GPU => 2,
                        vk::PhysicalDeviceType::CPU => 3,
                        _ => 4,
                    };
                    Some((rank, physical, family as u32))
                })
                .min_by_key(|(rank, _, _)| *rank)
                .map(|(_, physical, family)| (physical, family))
                .ok_or_else(|| "no Vulkan compute device is available".to_string())?;
            let device_extensions =
                unsafe { instance.enumerate_device_extension_properties(physical) }
                    .map_err(|error| format!("enumerate Vulkan device extensions: {error}"))?;
            let subset_name = ash::khr::portability_subset::NAME;
            let subset = device_extensions
                .iter()
                .any(|extension| extension_name(&extension.extension_name) == subset_name);
            let stencil_export_name = ash::ext::shader_stencil_export::NAME;
            let stencil_export = device_extensions
                .iter()
                .any(|extension| extension_name(&extension.extension_name) == stencil_export_name);
            let interlock_name = ash::ext::fragment_shader_interlock::NAME;
            let interlock_extension = device_extensions
                .iter()
                .any(|extension| extension_name(&extension.extension_name) == interlock_name);
            let priorities = [1.0f32];
            let queue_info = [vk::DeviceQueueCreateInfo::default()
                .queue_family_index(queue_family)
                .queue_priorities(&priorities)];
            let mut supported_bda = vk::PhysicalDeviceBufferDeviceAddressFeatures::default();
            let mut supported_16 = vk::PhysicalDevice16BitStorageFeatures::default();
            let mut supported_float16 = vk::PhysicalDeviceShaderFloat16Int8Features::default();
            let mut supported_interlock =
                vk::PhysicalDeviceFragmentShaderInterlockFeaturesEXT::default();
            let (sampler_anisotropy, sample_rate_shading, tessellation_shader, shader_int16) = {
                let mut features = vk::PhysicalDeviceFeatures2::default()
                    .push_next(&mut supported_bda)
                    .push_next(&mut supported_16)
                    .push_next(&mut supported_float16)
                    .push_next(&mut supported_interlock);
                unsafe { instance.get_physical_device_features2(physical, &mut features) };
                (
                    features.features.sampler_anisotropy == vk::TRUE,
                    features.features.sample_rate_shading == vk::TRUE,
                    features.features.tessellation_shader == vk::TRUE,
                    features.features.shader_int16 == vk::TRUE,
                )
            };
            let buffer_device_address = supported_bda.buffer_device_address == vk::TRUE;
            let fragment_shader_pixel_interlock = interlock_extension
                && supported_interlock.fragment_shader_pixel_interlock == vk::TRUE;
            let enabled_device_extensions = subset
                .then_some(subset_name.as_ptr())
                .into_iter()
                .chain(stencil_export.then_some(stencil_export_name.as_ptr()))
                .chain(fragment_shader_pixel_interlock.then_some(interlock_name.as_ptr()))
                .collect::<Vec<_>>();
            let enabled_features = vk::PhysicalDeviceFeatures::default()
                .sampler_anisotropy(sampler_anisotropy)
                .sample_rate_shading(sample_rate_shading)
                .tessellation_shader(tessellation_shader)
                .shader_int16(shader_int16);
            let mut enabled_bda = vk::PhysicalDeviceBufferDeviceAddressFeatures::default()
                .buffer_device_address(buffer_device_address);
            let mut enabled_16 = vk::PhysicalDevice16BitStorageFeatures::default()
                .storage_input_output16(supported_16.storage_input_output16 == vk::TRUE);
            let mut enabled_float16 = vk::PhysicalDeviceShaderFloat16Int8Features::default()
                .shader_float16(supported_float16.shader_float16 == vk::TRUE)
                .shader_int8(supported_float16.shader_int8 == vk::TRUE);
            let mut enabled_interlock =
                vk::PhysicalDeviceFragmentShaderInterlockFeaturesEXT::default()
                    .fragment_shader_pixel_interlock(fragment_shader_pixel_interlock);
            let mut device_info = vk::DeviceCreateInfo::default()
                .queue_create_infos(&queue_info)
                .enabled_extension_names(&enabled_device_extensions)
                .enabled_features(&enabled_features)
                .push_next(&mut enabled_16)
                .push_next(&mut enabled_float16)
                .push_next(&mut enabled_interlock);
            if buffer_device_address {
                device_info = device_info.push_next(&mut enabled_bda);
            }
            let device = unsafe { instance.create_device(physical, &device_info, None) }
                .map_err(|error| format!("create Vulkan device: {error}"))?;
            let queue = unsafe { device.get_device_queue(queue_family, 0) };
            let max_sampler_anisotropy =
                unsafe { instance.get_physical_device_properties(physical) }
                    .limits
                    .max_sampler_anisotropy;
            Ok(Self {
                _entry: entry,
                instance,
                device,
                physical,
                queue_family,
                queue,
                buffer_device_address,
                sampler_anisotropy,
                sample_rate_shading,
                tessellation_shader,
                shader_stencil_export: stencil_export,
                fragment_shader_pixel_interlock,
                max_sampler_anisotropy,
            })
        }

        fn environment(&self) -> serde_json::Value {
            let properties = unsafe { self.instance.get_physical_device_properties(self.physical) };
            let device = unsafe { CStr::from_ptr(properties.device_name.as_ptr()) }
                .to_string_lossy()
                .into_owned();
            serde_json::json!({
                "device": device,
                "device_id": properties.device_id,
                "vendor_id": properties.vendor_id,
                "driver_version": properties.driver_version,
                "api_version": properties.api_version,
                "buffer_device_address": self.buffer_device_address,
                "fragment_shader_pixel_interlock": self.fragment_shader_pixel_interlock,
                "sampler_anisotropy": self.sampler_anisotropy,
                "sample_rate_shading": self.sample_rate_shading,
                "shader_stencil_export": self.shader_stencil_export,
                "max_sampler_anisotropy": self.max_sampler_anisotropy,
                "architecture": std::env::consts::ARCH,
                "os": std::env::consts::OS,
            })
        }

        fn memory_type(&self, bits: u32, required: vk::MemoryPropertyFlags) -> Result<u32, String> {
            let properties = unsafe {
                self.instance
                    .get_physical_device_memory_properties(self.physical)
            };
            (0..properties.memory_type_count)
                .find(|index| {
                    bits & (1 << index) != 0
                        && properties.memory_types[*index as usize]
                            .property_flags
                            .contains(required)
                })
                .ok_or_else(|| {
                    format!(
                        "no Vulkan memory type satisfies property flags {:#x}",
                        required.as_raw()
                    )
                })
        }
    }

    impl Drop for VulkanContext {
        fn drop(&mut self) {
            unsafe {
                let _ = self.device.device_wait_idle();
                self.device.destroy_device(None);
                self.instance.destroy_instance(None);
            }
        }
    }

    struct BufferAllocation {
        device: Device,
        descriptor_binding: Option<u32>,
        output_binding: Option<u32>,
        argument_buffer_source: Option<(u32, u32)>,
        buffer_address_table: bool,
        device_address_index: Option<u32>,
        buffer: vk::Buffer,
        memory: vk::DeviceMemory,
        len: vk::DeviceSize,
    }

    struct ImageAllocation {
        device: Device,
        identity: ImageIdentity,
        descriptor_binding: u32,
        descriptor_type: vk::DescriptorType,
        descriptor_element: u32,
        descriptor_count: u32,
        descriptor_aliases: Vec<TextureDescriptorTarget>,
        image: vk::Image,
        view: vk::ImageView,
        memory: vk::DeviceMemory,
        aspect: vk::ImageAspectFlags,
        array_layers: u32,
        extent: vk::Extent3D,
        staging: Option<HostBuffer>,
        general_ready: bool,
    }

    struct TexelBufferAllocation {
        device: Device,
        binding: u32,
        descriptor_binding: u32,
        descriptor_type: vk::DescriptorType,
        descriptor_aliases: Vec<TextureDescriptorTarget>,
        format: crate::case::TextureFormat,
        dimensions: [u32; 3],
        buffer: HostBuffer,
        view: vk::BufferView,
    }

    #[derive(Clone, Copy, PartialEq, Eq)]
    pub(super) struct TextureDescriptorTarget {
        pub(super) binding: u32,
        pub(super) descriptor_type: vk::DescriptorType,
        pub(super) element: u32,
        pub(super) count: u32,
    }

    impl ImageAllocation {
        fn descriptor_targets(&self) -> impl Iterator<Item = TextureDescriptorTarget> + '_ {
            std::iter::once(TextureDescriptorTarget {
                binding: self.descriptor_binding,
                descriptor_type: self.descriptor_type,
                element: self.descriptor_element,
                count: self.descriptor_count,
            })
            .chain(self.descriptor_aliases.iter().copied())
        }

        fn uses_descriptor_type(&self, descriptor_type: vk::DescriptorType) -> bool {
            self.descriptor_targets()
                .any(|target| target.descriptor_type == descriptor_type)
        }
    }

    impl TexelBufferAllocation {
        fn descriptor_targets(&self) -> impl Iterator<Item = TextureDescriptorTarget> + '_ {
            std::iter::once(TextureDescriptorTarget {
                binding: self.descriptor_binding,
                descriptor_type: self.descriptor_type,
                element: 0,
                count: 1,
            })
            .chain(self.descriptor_aliases.iter().copied())
        }

        fn uses_descriptor_type(&self, descriptor_type: vk::DescriptorType) -> bool {
            self.descriptor_targets()
                .any(|target| target.descriptor_type == descriptor_type)
        }
    }

    #[derive(Clone, Copy)]
    struct BoundImageResources<'a> {
        images: &'a [ImageAllocation],
        texel_buffers: &'a [TexelBufferAllocation],
        reflection: &'a ShaderReflection,
    }

    struct RenderTargetAllocation {
        device: Device,
        index: u32,
        format: crate::case::TextureFormat,
        dimensions: [u32; 2],
        image: vk::Image,
        view: vk::ImageView,
        memory: vk::DeviceMemory,
        transfer: HostBuffer,
    }

    struct DepthStencilAllocation {
        device: Device,
        dimensions: [u32; 2],
        image: vk::Image,
        view: vk::ImageView,
        memory: vk::DeviceMemory,
        aspect: vk::ImageAspectFlags,
        depth: Option<HostBuffer>,
        stencil: Option<HostBuffer>,
    }

    #[derive(Clone, Copy)]
    struct GraphicsAttachments<'a> {
        colors: &'a [RenderTargetAllocation],
        depth_stencil: Option<&'a DepthStencilAllocation>,
    }

    impl DepthStencilAllocation {
        fn aspect_transfers(&self) -> impl Iterator<Item = (vk::ImageAspectFlags, &HostBuffer)> {
            self.depth
                .iter()
                .map(|buffer| (vk::ImageAspectFlags::DEPTH, buffer))
                .chain(
                    self.stencil
                        .iter()
                        .map(|buffer| (vk::ImageAspectFlags::STENCIL, buffer)),
                )
        }
    }

    struct HostBuffer {
        device: Device,
        buffer: vk::Buffer,
        memory: vk::DeviceMemory,
        len: vk::DeviceSize,
    }

    struct VertexInputAllocation {
        location: u32,
        format: crate::case::AttributeFormat,
        stride: u32,
        buffer: HostBuffer,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum ImageIdentity {
        Texture(u32),
        TextureArrayElement {
            binding: u32,
            element: u32,
        },
        ArgumentBufferTexture {
            buffer_binding: u32,
            field_offset: u32,
        },
        ImplicitImageblock {
            attachment: u32,
            data_rate: u32,
        },
        FragmentImageblock {
            binding: u32,
        },
    }

    struct TextureLiteralRef<'a> {
        identity: ImageIdentity,
        label: String,
        texture_type: crate::case::TextureType,
        format: crate::case::TextureFormat,
        dimensions: [u32; 3],
        sample_count: u32,
        bytes: &'a [u8],
    }

    impl<'a> TextureLiteralRef<'a> {
        fn top_level(resource: &'a crate::literal::LiteralTexture) -> Self {
            Self {
                identity: ImageIdentity::Texture(resource.binding),
                label: format!("texture {}", resource.binding),
                texture_type: resource.texture_type,
                format: resource.format,
                dimensions: resource.dimensions,
                sample_count: resource.sample_count,
                bytes: &resource.bytes,
            }
        }

        fn argument_buffer(resource: &'a crate::literal::LiteralArgumentBufferTexture) -> Self {
            Self {
                identity: ImageIdentity::ArgumentBufferTexture {
                    buffer_binding: resource.buffer_binding,
                    field_offset: resource.field_offset,
                },
                label: resource.label(),
                texture_type: resource.texture_type,
                format: resource.format,
                dimensions: resource.dimensions,
                sample_count: resource.sample_count,
                bytes: &resource.bytes,
            }
        }

        fn array_element(
            binding: u32,
            element: u32,
            resource: &'a crate::literal::LiteralTexture,
        ) -> Self {
            Self {
                identity: ImageIdentity::TextureArrayElement { binding, element },
                label: format!("texture-array {binding} element {element}"),
                texture_type: resource.texture_type,
                format: resource.format,
                dimensions: resource.dimensions,
                sample_count: resource.sample_count,
                bytes: &resource.bytes,
            }
        }

        fn layout(&self) -> Result<crate::literal::TextureLayout, String> {
            crate::literal::texture_layout(self.texture_type, self.dimensions, self.sample_count)
        }
    }

    impl Drop for ImageAllocation {
        fn drop(&mut self) {
            unsafe {
                self.device.destroy_image_view(self.view, None);
                self.device.destroy_image(self.image, None);
                self.device.free_memory(self.memory, None);
            }
        }
    }

    impl Drop for TexelBufferAllocation {
        fn drop(&mut self) {
            unsafe { self.device.destroy_buffer_view(self.view, None) };
        }
    }

    impl Drop for RenderTargetAllocation {
        fn drop(&mut self) {
            unsafe {
                self.device.destroy_image_view(self.view, None);
                self.device.destroy_image(self.image, None);
                self.device.free_memory(self.memory, None);
            }
        }
    }

    impl Drop for DepthStencilAllocation {
        fn drop(&mut self) {
            unsafe {
                self.device.destroy_image_view(self.view, None);
                self.device.destroy_image(self.image, None);
                self.device.free_memory(self.memory, None);
            }
        }
    }

    impl Drop for HostBuffer {
        fn drop(&mut self) {
            unsafe {
                self.device.destroy_buffer(self.buffer, None);
                self.device.free_memory(self.memory, None);
            }
        }
    }

    struct SamplerAllocation {
        device: Device,
        descriptor_binding: u32,
        sampler: vk::Sampler,
    }

    impl Drop for SamplerAllocation {
        fn drop(&mut self) {
            unsafe { self.device.destroy_sampler(self.sampler, None) };
        }
    }

    impl Drop for BufferAllocation {
        fn drop(&mut self) {
            unsafe {
                self.device.destroy_buffer(self.buffer, None);
                self.device.free_memory(self.memory, None);
            }
        }
    }

    struct DeviceObjects {
        device: Device,
        set_layout: vk::DescriptorSetLayout,
        pipeline_layout: vk::PipelineLayout,
        shader: vk::ShaderModule,
        companion_shader: vk::ShaderModule,
        tessellation_shader: vk::ShaderModule,
        pipeline: vk::Pipeline,
        render_pass: vk::RenderPass,
        framebuffer: vk::Framebuffer,
        descriptor_pool: vk::DescriptorPool,
        command_pool: vk::CommandPool,
        fence: vk::Fence,
        framebuffer_fetch: bool,
        patch_control_points: u32,
    }

    impl DeviceObjects {
        fn new(device: &Device) -> Self {
            Self {
                device: device.clone(),
                set_layout: vk::DescriptorSetLayout::null(),
                pipeline_layout: vk::PipelineLayout::null(),
                shader: vk::ShaderModule::null(),
                companion_shader: vk::ShaderModule::null(),
                tessellation_shader: vk::ShaderModule::null(),
                pipeline: vk::Pipeline::null(),
                render_pass: vk::RenderPass::null(),
                framebuffer: vk::Framebuffer::null(),
                descriptor_pool: vk::DescriptorPool::null(),
                command_pool: vk::CommandPool::null(),
                fence: vk::Fence::null(),
                framebuffer_fetch: false,
                patch_control_points: 0,
            }
        }
    }

    impl Drop for DeviceObjects {
        fn drop(&mut self) {
            unsafe {
                if self.fence != vk::Fence::null() {
                    self.device.destroy_fence(self.fence, None);
                }
                if self.command_pool != vk::CommandPool::null() {
                    self.device.destroy_command_pool(self.command_pool, None);
                }
                if self.descriptor_pool != vk::DescriptorPool::null() {
                    self.device
                        .destroy_descriptor_pool(self.descriptor_pool, None);
                }
                if self.pipeline != vk::Pipeline::null() {
                    self.device.destroy_pipeline(self.pipeline, None);
                }
                if self.framebuffer != vk::Framebuffer::null() {
                    self.device.destroy_framebuffer(self.framebuffer, None);
                }
                if self.render_pass != vk::RenderPass::null() {
                    self.device.destroy_render_pass(self.render_pass, None);
                }
                if self.companion_shader != vk::ShaderModule::null() {
                    self.device
                        .destroy_shader_module(self.companion_shader, None);
                }
                if self.tessellation_shader != vk::ShaderModule::null() {
                    self.device
                        .destroy_shader_module(self.tessellation_shader, None);
                }
                if self.shader != vk::ShaderModule::null() {
                    self.device.destroy_shader_module(self.shader, None);
                }
                if self.pipeline_layout != vk::PipelineLayout::null() {
                    self.device
                        .destroy_pipeline_layout(self.pipeline_layout, None);
                }
                if self.set_layout != vk::DescriptorSetLayout::null() {
                    self.device
                        .destroy_descriptor_set_layout(self.set_layout, None);
                }
            }
        }
    }

    fn descriptor_bindings(
        reflection: &ShaderReflection,
    ) -> Result<Vec<vk::DescriptorSetLayoutBinding<'static>>, String> {
        let stage_flags = match reflection.stage {
            metal2vulkan::reflect::ShaderStage::Kernel => vk::ShaderStageFlags::COMPUTE,
            metal2vulkan::reflect::ShaderStage::Fragment => vk::ShaderStageFlags::FRAGMENT,
            metal2vulkan::reflect::ShaderStage::Vertex => vk::ShaderStageFlags::VERTEX,
            metal2vulkan::reflect::ShaderStage::TessellationEvaluation => {
                vk::ShaderStageFlags::TESSELLATION_EVALUATION
            }
        };
        let mut bindings = reflection
            .bindings
            .iter()
            .filter_map(|binding| {
                let descriptor = binding.descriptor?;
                let descriptor_type = match binding.kind {
                    metal2vulkan::reflect::ResourceKind::Buffer
                    | metal2vulkan::reflect::ResourceKind::BufferAddressTable
                    | metal2vulkan::reflect::ResourceKind::AccelerationStructureShadow
                    | metal2vulkan::reflect::ResourceKind::PrimitiveAccelerationStructure
                    | metal2vulkan::reflect::ResourceKind::KernelStageInput => {
                        vk::DescriptorType::STORAGE_BUFFER
                    }
                    metal2vulkan::reflect::ResourceKind::Texture
                    | metal2vulkan::reflect::ResourceKind::StorageImage => {
                        vulkan_texture_descriptor_type(binding)
                    }
                    metal2vulkan::reflect::ResourceKind::TextureArray => {
                        if binding.access == Some(metal2vulkan::reflect::ResourceAccess::Storage) {
                            vk::DescriptorType::STORAGE_IMAGE
                        } else {
                            vk::DescriptorType::SAMPLED_IMAGE
                        }
                    }
                    metal2vulkan::reflect::ResourceKind::EmbeddedArgBufferTexture => {
                        if binding.access == Some(metal2vulkan::reflect::ResourceAccess::Storage) {
                            vk::DescriptorType::STORAGE_IMAGE
                        } else {
                            vk::DescriptorType::SAMPLED_IMAGE
                        }
                    }
                    metal2vulkan::reflect::ResourceKind::Sampler
                    | metal2vulkan::reflect::ResourceKind::StaticSampler => {
                        vk::DescriptorType::SAMPLER
                    }
                    metal2vulkan::reflect::ResourceKind::ColorInput => {
                        vk::DescriptorType::INPUT_ATTACHMENT
                    }
                    metal2vulkan::reflect::ResourceKind::ThreadgroupBuffer => return None,
                    unsupported => {
                        return Some(Err(format!(
                            "candidate descriptor preparation does not yet support {unsupported:?}"
                        )))
                    }
                };
                Some(Ok(vk::DescriptorSetLayoutBinding::default()
                    .binding(descriptor.binding)
                    .descriptor_type(descriptor_type)
                    .descriptor_count(descriptor.count)
                    .stage_flags(stage_flags)))
            })
            .collect::<Result<Vec<_>, _>>()?;
        bindings.extend(
            reflection
                .implicit_imageblock_attachments
                .iter()
                .map(|attachment| {
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(attachment.binding)
                        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                        .descriptor_count(1)
                        .stage_flags(stage_flags)
                }),
        );
        bindings.extend(
            reflection
                .fragment_imageblock
                .iter()
                .flat_map(|imageblock| &imageblock.members)
                .filter_map(|member| member.binding)
                .map(|binding| {
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(binding)
                        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                        .descriptor_count(1)
                        .stage_flags(stage_flags)
                }),
        );
        let mut by_binding =
            std::collections::BTreeMap::<u32, vk::DescriptorSetLayoutBinding<'static>>::new();
        for binding in bindings {
            match by_binding.entry(binding.binding) {
                std::collections::btree_map::Entry::Vacant(entry) => {
                    entry.insert(binding);
                }
                std::collections::btree_map::Entry::Occupied(mut entry) => {
                    let existing = entry.get_mut();
                    if existing.descriptor_type != binding.descriptor_type {
                        return Err(format!(
                            "reflection binding {} aliases descriptor types {} and {}",
                            binding.binding,
                            existing.descriptor_type.as_raw(),
                            binding.descriptor_type.as_raw()
                        ));
                    }
                    existing.descriptor_count =
                        existing.descriptor_count.max(binding.descriptor_count);
                    existing.stage_flags |= binding.stage_flags;
                }
            }
        }
        Ok(by_binding.into_values().collect())
    }

    pub(super) fn vulkan_texture_descriptor_type(
        binding: &metal2vulkan::reflect::ResourceBinding,
    ) -> vk::DescriptorType {
        let buffer = binding
            .texture_shape
            .is_some_and(|shape| shape.dimension == metal2vulkan::meta::TextureDimension::Buffer);
        match (buffer, binding.access) {
            (true, Some(metal2vulkan::reflect::ResourceAccess::Storage)) => {
                vk::DescriptorType::STORAGE_TEXEL_BUFFER
            }
            (true, _) => vk::DescriptorType::UNIFORM_TEXEL_BUFFER,
            (false, Some(metal2vulkan::reflect::ResourceAccess::Storage)) => {
                vk::DescriptorType::STORAGE_IMAGE
            }
            (false, _) => vk::DescriptorType::SAMPLED_IMAGE,
        }
    }

    pub(super) fn top_level_texture_targets(
        reflection: &ShaderReflection,
        metal_index: u32,
        buffer_texture: bool,
    ) -> Result<Vec<TextureDescriptorTarget>, String> {
        let targets = reflection
            .bindings
            .iter()
            .filter(|binding| {
                binding.metal_index == metal_index
                    && matches!(
                        binding.kind,
                        metal2vulkan::reflect::ResourceKind::Texture
                            | metal2vulkan::reflect::ResourceKind::StorageImage
                    )
                    && binding.texture_shape.map_or(!buffer_texture, |shape| {
                        (shape.dimension == metal2vulkan::meta::TextureDimension::Buffer)
                            == buffer_texture
                    })
            })
            .map(|binding| {
                let descriptor = binding
                    .descriptor
                    .ok_or_else(|| format!("texture {metal_index} has no descriptor location"))?;
                Ok(TextureDescriptorTarget {
                    binding: descriptor.binding,
                    descriptor_type: vulkan_texture_descriptor_type(binding),
                    element: 0,
                    count: descriptor.count,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        sorted_texture_targets(targets, &format!("texture {metal_index}"))
    }

    fn sorted_texture_targets(
        mut targets: Vec<TextureDescriptorTarget>,
        label: &str,
    ) -> Result<Vec<TextureDescriptorTarget>, String> {
        targets.sort_by_key(|target| {
            (
                target.descriptor_type != vk::DescriptorType::STORAGE_IMAGE
                    && target.descriptor_type != vk::DescriptorType::STORAGE_TEXEL_BUFFER,
                target.binding,
                target.element,
            )
        });
        targets.dedup();
        if targets.is_empty() {
            return Err(format!("{label} has no executable reflection binding"));
        }
        Ok(targets)
    }

    pub(super) fn texture_array_targets(
        reflection: &ShaderReflection,
        metal_index: u32,
        element: u32,
    ) -> Result<Vec<TextureDescriptorTarget>, String> {
        let targets = reflection
            .bindings
            .iter()
            .filter(|binding| {
                binding.kind == metal2vulkan::reflect::ResourceKind::TextureArray
                    && binding.metal_index == metal_index
            })
            .map(|binding| {
                let descriptor = binding.descriptor.ok_or_else(|| {
                    format!("texture-array {metal_index} has no descriptor location")
                })?;
                Ok(TextureDescriptorTarget {
                    binding: descriptor.binding,
                    descriptor_type: vulkan_texture_descriptor_type(binding),
                    element,
                    count: descriptor.count,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        sorted_texture_targets(targets, &format!("texture-array {metal_index}"))
    }

    fn embedded_texture_targets(
        reflection: &ShaderReflection,
        resource: &crate::literal::LiteralArgumentBufferTexture,
    ) -> Result<Vec<TextureDescriptorTarget>, String> {
        let targets = reflection
            .bindings
            .iter()
            .filter(|binding| {
                binding.kind == metal2vulkan::reflect::ResourceKind::EmbeddedArgBufferTexture
            })
            .filter_map(|binding| {
                let source = binding.embedded_source?;
                let descriptor = binding.descriptor?;
                let delta = resource.field_offset.checked_sub(source.field_offset)?;
                (source.buffer_index == resource.buffer_binding
                    && delta % 8 == 0
                    && delta / 8 < descriptor.count)
                    .then_some(TextureDescriptorTarget {
                        binding: descriptor.binding,
                        descriptor_type: vulkan_texture_descriptor_type(binding),
                        element: delta / 8,
                        count: descriptor.count,
                    })
            })
            .collect();
        sorted_texture_targets(targets, &resource.label())
    }

    fn descriptor_pool_sizes(
        bindings: &[vk::DescriptorSetLayoutBinding<'_>],
    ) -> Vec<vk::DescriptorPoolSize> {
        let mut counts = std::collections::BTreeMap::<i32, u32>::new();
        for binding in bindings {
            *counts.entry(binding.descriptor_type.as_raw()).or_default() +=
                binding.descriptor_count;
        }
        counts
            .into_iter()
            .map(|(raw, descriptor_count)| {
                vk::DescriptorPoolSize::default()
                    .ty(vk::DescriptorType::from_raw(raw))
                    .descriptor_count(descriptor_count)
            })
            .collect()
    }

    fn create_images(
        context: &VulkanContext,
        resources: &LiteralResources,
        reflection: &ShaderReflection,
    ) -> Result<Vec<ImageAllocation>, String> {
        let mut images = resources
            .textures
            .iter()
            .filter(|resource| resource.texture_type != crate::case::TextureType::Buffer)
            .map(|resource| {
                let mut targets = top_level_texture_targets(reflection, resource.binding, false)?;
                let primary = targets.remove(0);
                create_image(
                    context,
                    &TextureLiteralRef::top_level(resource),
                    primary.binding,
                    primary.descriptor_type,
                    primary.element,
                    primary.count,
                    targets,
                )
            })
            .collect::<Result<Vec<_>, String>>()?;
        for array in &resources.texture_arrays {
            for (element, resource) in array.elements.iter().enumerate() {
                let mut targets = texture_array_targets(reflection, array.binding, element as u32)?;
                let primary = targets.remove(0);
                images.push(create_image(
                    context,
                    &TextureLiteralRef::array_element(array.binding, element as u32, resource),
                    primary.binding,
                    primary.descriptor_type,
                    primary.element,
                    primary.count,
                    targets,
                )?);
            }
        }
        for resource in &resources.argument_buffer_textures {
            let mut targets = embedded_texture_targets(reflection, resource)?;
            let primary = targets.remove(0);
            images.push(create_image(
                context,
                &TextureLiteralRef::argument_buffer(resource),
                primary.binding,
                primary.descriptor_type,
                primary.element,
                primary.count,
                targets,
            )?);
        }
        for attachment in &reflection.implicit_imageblock_attachments {
            let resource = resources
                .render_targets
                .iter()
                .find(|resource| resource.index == attachment.attachment)
                .ok_or_else(|| {
                    format!(
                        "implicit imageblock attachment {} has no authored render target",
                        attachment.attachment
                    )
                })?;
            let layers = attachment.max_index.unwrap_or(0).saturating_add(1);
            let mut bytes = Vec::with_capacity(resource.bytes.len() * layers as usize);
            for _ in 0..layers {
                bytes.extend_from_slice(&resource.bytes);
            }
            let literal = crate::literal::LiteralTexture {
                binding: attachment.binding,
                role: crate::case::ResourceRole::InOut,
                texture_type: crate::case::TextureType::D2Array,
                format: resource.format,
                dimensions: [resource.dimensions[0], resource.dimensions[1], layers],
                sample_count: 1,
                bytes,
            };
            let mut input = TextureLiteralRef::top_level(&literal);
            input.identity = ImageIdentity::ImplicitImageblock {
                attachment: attachment.attachment,
                data_rate: attachment.data_rate,
            };
            input.label = format!(
                "implicit imageblock attachment {} rate {}",
                attachment.attachment, attachment.data_rate
            );
            images.push(create_image(
                context,
                &input,
                attachment.binding,
                vk::DescriptorType::STORAGE_IMAGE,
                0,
                1,
                Vec::new(),
            )?);
        }
        if let (Some(authored), Some(reflected)) = (
            resources.fragment_imageblock.as_ref(),
            reflection.fragment_imageblock.as_ref(),
        ) {
            for member in &reflected.members {
                let Some(binding) = member.binding else {
                    continue;
                };
                let resource = authored
                    .members
                    .iter()
                    .find(|resource| resource.semantic == member.semantic)
                    .ok_or_else(|| {
                        format!(
                            "fragment imageblock member {} has no authored plane",
                            member.semantic
                        )
                    })?;
                let literal = crate::literal::LiteralTexture {
                    binding,
                    role: resource.role,
                    texture_type: crate::case::TextureType::D2,
                    format: resource.format.texture_format(),
                    dimensions: [authored.dimensions[0], authored.dimensions[1], 1],
                    sample_count: 1,
                    bytes: resource.bytes.clone(),
                };
                let mut input = TextureLiteralRef::top_level(&literal);
                input.identity = ImageIdentity::FragmentImageblock { binding };
                input.label = format!("fragment imageblock member {}", member.semantic);
                images.push(create_image(
                    context,
                    &input,
                    binding,
                    vk::DescriptorType::STORAGE_IMAGE,
                    0,
                    1,
                    Vec::new(),
                )?);
            }
        }
        Ok(images)
    }

    fn create_texel_buffers(
        context: &VulkanContext,
        resources: &LiteralResources,
        reflection: &ShaderReflection,
    ) -> Result<Vec<TexelBufferAllocation>, String> {
        resources
            .textures
            .iter()
            .filter(|resource| resource.texture_type == crate::case::TextureType::Buffer)
            .map(|resource| {
                let mut targets = top_level_texture_targets(reflection, resource.binding, true)?;
                if let Some(target) = targets.iter().find(|target| target.count != 1) {
                    return Err(format!(
                        "texture buffer {} has unsupported descriptor count {}",
                        resource.binding, target.count
                    ));
                }
                let primary = targets.remove(0);
                let uses_storage = primary.descriptor_type
                    == vk::DescriptorType::STORAGE_TEXEL_BUFFER
                    || targets.iter().any(|target| {
                        target.descriptor_type == vk::DescriptorType::STORAGE_TEXEL_BUFFER
                    });
                let uses_uniform = primary.descriptor_type
                    == vk::DescriptorType::UNIFORM_TEXEL_BUFFER
                    || targets.iter().any(|target| {
                        target.descriptor_type == vk::DescriptorType::UNIFORM_TEXEL_BUFFER
                    });
                let mut required_features = vk::FormatFeatureFlags::empty();
                let mut usage = vk::BufferUsageFlags::empty();
                if uses_storage {
                    required_features |= vk::FormatFeatureFlags::STORAGE_TEXEL_BUFFER;
                    usage |= vk::BufferUsageFlags::STORAGE_TEXEL_BUFFER;
                }
                if uses_uniform {
                    required_features |= vk::FormatFeatureFlags::UNIFORM_TEXEL_BUFFER;
                    usage |= vk::BufferUsageFlags::UNIFORM_TEXEL_BUFFER;
                }
                let format = vulkan_format(resource.format);
                let properties = unsafe {
                    context
                        .instance
                        .get_physical_device_format_properties(context.physical, format)
                };
                if !properties.buffer_features.contains(required_features) {
                    return Err(format!(
                        "Vulkan device lacks texel-buffer feature {:#x} for texture {} format {}",
                        required_features.as_raw(),
                        resource.binding,
                        format.as_raw()
                    ));
                }
                let buffer = create_host_buffer(
                    context,
                    &format!("texture buffer {}", resource.binding),
                    &resource.bytes,
                    usage,
                )?;
                let view_info = vk::BufferViewCreateInfo::default()
                    .buffer(buffer.buffer)
                    .format(format)
                    .offset(0)
                    .range(buffer.len);
                let view = unsafe { context.device.create_buffer_view(&view_info, None) }.map_err(
                    |error| format!("create texture buffer {} view: {error}", resource.binding),
                )?;
                Ok(TexelBufferAllocation {
                    device: context.device.clone(),
                    binding: resource.binding,
                    descriptor_binding: primary.binding,
                    descriptor_type: primary.descriptor_type,
                    descriptor_aliases: targets,
                    format: resource.format,
                    dimensions: resource.dimensions,
                    buffer,
                    view,
                })
            })
            .collect()
    }

    fn create_render_targets(
        context: &VulkanContext,
        resources: &LiteralResources,
    ) -> Result<Vec<RenderTargetAllocation>, String> {
        resources
            .render_targets
            .iter()
            .map(|resource| {
                let label = format!("render target {}", resource.index);
                let format = vulkan_format(resource.format);
                let properties = unsafe {
                    context
                        .instance
                        .get_physical_device_format_properties(context.physical, format)
                };
                if !properties
                    .optimal_tiling_features
                    .contains(vk::FormatFeatureFlags::COLOR_ATTACHMENT)
                {
                    return Err(format!(
                        "Vulkan device lacks color-attachment support for {label} format {}",
                        format.as_raw()
                    ));
                }
                let create_info = vk::ImageCreateInfo::default()
                    .image_type(vk::ImageType::TYPE_2D)
                    .format(format)
                    .extent(vk::Extent3D {
                        width: resource.dimensions[0],
                        height: resource.dimensions[1],
                        depth: 1,
                    })
                    .mip_levels(1)
                    .array_layers(1)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .tiling(vk::ImageTiling::OPTIMAL)
                    .usage(
                        vk::ImageUsageFlags::COLOR_ATTACHMENT
                            | vk::ImageUsageFlags::INPUT_ATTACHMENT
                            | vk::ImageUsageFlags::TRANSFER_SRC
                            | vk::ImageUsageFlags::TRANSFER_DST,
                    )
                    .sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .initial_layout(vk::ImageLayout::UNDEFINED);
                let image = unsafe { context.device.create_image(&create_info, None) }
                    .map_err(|error| format!("create {label}: {error}"))?;
                let requirements = unsafe { context.device.get_image_memory_requirements(image) };
                let memory_type = context.memory_type(
                    requirements.memory_type_bits,
                    vk::MemoryPropertyFlags::DEVICE_LOCAL,
                )?;
                let allocation_info = vk::MemoryAllocateInfo::default()
                    .allocation_size(requirements.size)
                    .memory_type_index(memory_type);
                let memory = match unsafe { context.device.allocate_memory(&allocation_info, None) }
                {
                    Ok(memory) => memory,
                    Err(error) => {
                        unsafe { context.device.destroy_image(image, None) };
                        return Err(format!("allocate {label}: {error}"));
                    }
                };
                if let Err(error) = unsafe { context.device.bind_image_memory(image, memory, 0) } {
                    unsafe {
                        context.device.destroy_image(image, None);
                        context.device.free_memory(memory, None);
                    }
                    return Err(format!("bind {label}: {error}"));
                }
                let view_info = vk::ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(format)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    });
                let view = match unsafe { context.device.create_image_view(&view_info, None) } {
                    Ok(view) => view,
                    Err(error) => {
                        unsafe {
                            context.device.destroy_image(image, None);
                            context.device.free_memory(memory, None);
                        }
                        return Err(format!("create {label} view: {error}"));
                    }
                };
                let transfer = match create_host_buffer(
                    context,
                    &label,
                    &resource.bytes,
                    vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
                ) {
                    Ok(transfer) => transfer,
                    Err(error) => {
                        unsafe {
                            context.device.destroy_image_view(view, None);
                            context.device.destroy_image(image, None);
                            context.device.free_memory(memory, None);
                        }
                        return Err(error);
                    }
                };
                Ok(RenderTargetAllocation {
                    device: context.device.clone(),
                    index: resource.index,
                    format: resource.format,
                    dimensions: resource.dimensions,
                    image,
                    view,
                    memory,
                    transfer,
                })
            })
            .collect()
    }

    fn create_depth_stencil(
        context: &VulkanContext,
        resources: &LiteralResources,
    ) -> Result<Option<DepthStencilAllocation>, String> {
        let Some(resource) = &resources.depth_stencil else {
            return Ok(None);
        };
        let has_depth = resource.depth.is_some();
        let has_stencil = resource.stencil.is_some();
        let format = if has_stencil {
            vk::Format::D32_SFLOAT_S8_UINT
        } else {
            vk::Format::D32_SFLOAT
        };
        let aspect = (if has_depth {
            vk::ImageAspectFlags::DEPTH
        } else {
            vk::ImageAspectFlags::empty()
        }) | if has_stencil {
            vk::ImageAspectFlags::STENCIL
        } else {
            vk::ImageAspectFlags::empty()
        };
        let properties = unsafe {
            context
                .instance
                .get_physical_device_format_properties(context.physical, format)
        };
        if !properties
            .optimal_tiling_features
            .contains(vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT)
        {
            return Err(format!(
                "Vulkan device lacks depth/stencil attachment support for format {}",
                format.as_raw()
            ));
        }
        let create_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(vk::Extent3D {
                width: resource.dimensions[0],
                height: resource.dimensions[1],
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(
                vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
                    | vk::ImageUsageFlags::TRANSFER_SRC
                    | vk::ImageUsageFlags::TRANSFER_DST,
            )
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        let image = unsafe { context.device.create_image(&create_info, None) }
            .map_err(|error| format!("create depth/stencil attachment: {error}"))?;
        let requirements = unsafe { context.device.get_image_memory_requirements(image) };
        let memory_type = context.memory_type(
            requirements.memory_type_bits,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        let allocation_info = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(memory_type);
        let memory = unsafe { context.device.allocate_memory(&allocation_info, None) }
            .map_err(|error| format!("allocate depth/stencil attachment: {error}"))?;
        if let Err(error) = unsafe { context.device.bind_image_memory(image, memory, 0) } {
            unsafe {
                context.device.destroy_image(image, None);
                context.device.free_memory(memory, None);
            }
            return Err(format!("bind depth/stencil attachment: {error}"));
        }
        let view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: aspect,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });
        let view = unsafe { context.device.create_image_view(&view_info, None) }
            .map_err(|error| format!("create depth/stencil attachment view: {error}"))?;
        let depth = resource
            .depth
            .as_deref()
            .map(|bytes| {
                create_host_buffer(
                    context,
                    "depth attachment",
                    bytes,
                    vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
                )
            })
            .transpose()?;
        let stencil = resource
            .stencil
            .as_deref()
            .map(|bytes| {
                create_host_buffer(
                    context,
                    "stencil attachment",
                    bytes,
                    vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
                )
            })
            .transpose()?;
        Ok(Some(DepthStencilAllocation {
            device: context.device.clone(),
            dimensions: resource.dimensions,
            image,
            view,
            memory,
            aspect,
            depth,
            stencil,
        }))
    }

    fn create_host_buffer(
        context: &VulkanContext,
        label: &str,
        bytes: &[u8],
        usage: vk::BufferUsageFlags,
    ) -> Result<HostBuffer, String> {
        let create_info = vk::BufferCreateInfo::default()
            .size(bytes.len() as u64)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buffer = unsafe { context.device.create_buffer(&create_info, None) }
            .map_err(|error| format!("create {label} transfer buffer: {error}"))?;
        let requirements = unsafe { context.device.get_buffer_memory_requirements(buffer) };
        let memory_type = context.memory_type(
            requirements.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        let allocation_info = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(memory_type);
        let memory = match unsafe { context.device.allocate_memory(&allocation_info, None) } {
            Ok(memory) => memory,
            Err(error) => {
                unsafe { context.device.destroy_buffer(buffer, None) };
                return Err(format!("allocate {label} transfer memory: {error}"));
            }
        };
        if let Err(error) = unsafe { context.device.bind_buffer_memory(buffer, memory, 0) } {
            unsafe {
                context.device.destroy_buffer(buffer, None);
                context.device.free_memory(memory, None);
            }
            return Err(format!("bind {label} transfer memory: {error}"));
        }
        let pointer = unsafe {
            context
                .device
                .map_memory(memory, 0, bytes.len() as u64, vk::MemoryMapFlags::empty())
        }
        .map_err(|error| format!("map {label} transfer memory: {error}"))?;
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), pointer.cast::<u8>(), bytes.len());
            context.device.unmap_memory(memory);
        }
        Ok(HostBuffer {
            device: context.device.clone(),
            buffer,
            memory,
            len: bytes.len() as u64,
        })
    }

    fn create_vertex_inputs(
        context: &VulkanContext,
        resources: &LiteralResources,
    ) -> Result<Vec<VertexInputAllocation>, String> {
        resources
            .vertex_inputs
            .iter()
            .chain(
                resources
                    .tessellation
                    .iter()
                    .flat_map(|tessellation| &tessellation.control_points),
            )
            .map(|input| {
                Ok(VertexInputAllocation {
                    location: input.location,
                    format: input.format,
                    stride: input.stride,
                    buffer: create_host_buffer(
                        context,
                        &format!("vertex input {}", input.location),
                        &input.bytes,
                        vk::BufferUsageFlags::VERTEX_BUFFER,
                    )?,
                })
            })
            .collect()
    }

    fn create_image(
        context: &VulkanContext,
        resource: &TextureLiteralRef<'_>,
        descriptor_binding: u32,
        descriptor_type: vk::DescriptorType,
        descriptor_element: u32,
        descriptor_count: u32,
        descriptor_aliases: Vec<TextureDescriptorTarget>,
    ) -> Result<ImageAllocation, String> {
        let layout = resource.layout()?;
        let label = &resource.label;
        if layout.sample_count != 1 {
            if descriptor_type != vk::DescriptorType::SAMPLED_IMAGE
                || descriptor_aliases
                    .iter()
                    .any(|target| target.descriptor_type != vk::DescriptorType::SAMPLED_IMAGE)
            {
                return Err(format!(
                    "multisample {label} requires a sampled-image descriptor"
                ));
            }
            return create_multisample_image(
                context,
                resource,
                TextureDescriptorTarget {
                    binding: descriptor_binding,
                    descriptor_type,
                    element: descriptor_element,
                    count: descriptor_count,
                },
                descriptor_aliases,
                layout,
            );
        }
        let format = vulkan_format(resource.format);
        let uses_storage = descriptor_type == vk::DescriptorType::STORAGE_IMAGE
            || descriptor_aliases
                .iter()
                .any(|target| target.descriptor_type == vk::DescriptorType::STORAGE_IMAGE);
        let uses_sampled = descriptor_type == vk::DescriptorType::SAMPLED_IMAGE
            || descriptor_aliases
                .iter()
                .any(|target| target.descriptor_type == vk::DescriptorType::SAMPLED_IMAGE);
        let optimal_staging = uses_sampled
            && !uses_storage
            && requires_optimal_staging(resource.texture_type, vk::DescriptorType::SAMPLED_IMAGE);
        let mut required_features = vk::FormatFeatureFlags::empty();
        if uses_storage {
            required_features |= vk::FormatFeatureFlags::STORAGE_IMAGE;
        }
        if uses_sampled {
            required_features |= vk::FormatFeatureFlags::SAMPLED_IMAGE;
        }
        let properties = unsafe {
            context
                .instance
                .get_physical_device_format_properties(context.physical, format)
        };
        let available_features = if optimal_staging {
            properties.optimal_tiling_features
        } else {
            properties.linear_tiling_features
        };
        if !available_features.contains(required_features) {
            return Err(format!(
                "Vulkan device lacks {}-tiling feature {:#x} for texture {} format {}",
                if optimal_staging { "optimal" } else { "linear" },
                required_features.as_raw(),
                label,
                format.as_raw(),
            ));
        }
        let image_type = match resource.texture_type {
            crate::case::TextureType::D1 | crate::case::TextureType::D1Array => {
                vk::ImageType::TYPE_1D
            }
            crate::case::TextureType::D3 => vk::ImageType::TYPE_3D,
            _ => vk::ImageType::TYPE_2D,
        };
        let mut flags = vk::ImageCreateFlags::empty();
        if matches!(
            resource.texture_type,
            crate::case::TextureType::Cube | crate::case::TextureType::CubeArray
        ) {
            flags |= vk::ImageCreateFlags::CUBE_COMPATIBLE;
        }
        let mut usage = vk::ImageUsageFlags::empty();
        if uses_storage {
            usage |= vk::ImageUsageFlags::STORAGE;
        }
        if uses_sampled {
            usage |= vk::ImageUsageFlags::SAMPLED;
        }
        usage |= if optimal_staging {
            vk::ImageUsageFlags::TRANSFER_DST
        } else {
            vk::ImageUsageFlags::empty()
        };
        let extent = vk::Extent3D {
            width: layout.width,
            height: layout.height,
            depth: layout.depth,
        };
        let create_info = vk::ImageCreateInfo::default()
            .flags(flags)
            .image_type(image_type)
            .format(format)
            .extent(extent)
            .mip_levels(1)
            .array_layers(layout.array_layers)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(if optimal_staging {
                vk::ImageTiling::OPTIMAL
            } else {
                vk::ImageTiling::LINEAR
            })
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(if optimal_staging {
                vk::ImageLayout::UNDEFINED
            } else {
                vk::ImageLayout::PREINITIALIZED
            });
        let image = unsafe { context.device.create_image(&create_info, None) }
            .map_err(|error| format!("create {label} image: {error}"))?;
        let requirements = unsafe { context.device.get_image_memory_requirements(image) };
        let memory_type = context.memory_type(
            requirements.memory_type_bits,
            if optimal_staging {
                vk::MemoryPropertyFlags::DEVICE_LOCAL
            } else {
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT
            },
        )?;
        let allocation_info = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(memory_type);
        let memory = unsafe { context.device.allocate_memory(&allocation_info, None) }
            .map_err(|error| format!("allocate {label} memory: {error}"))?;
        if let Err(error) = unsafe { context.device.bind_image_memory(image, memory, 0) } {
            unsafe {
                context.device.destroy_image(image, None);
                context.device.free_memory(memory, None);
            }
            return Err(format!("bind {label} memory: {error}"));
        }
        let aspect = if resource.format == crate::case::TextureFormat::Depth32Float {
            vk::ImageAspectFlags::DEPTH
        } else {
            vk::ImageAspectFlags::COLOR
        };
        let staging = if optimal_staging {
            Some(create_host_buffer(
                context,
                &format!("{label} staging"),
                resource.bytes,
                vk::BufferUsageFlags::TRANSFER_SRC,
            )?)
        } else {
            upload_linear_image(context, image, memory, resource, layout, aspect)?;
            None
        };
        let view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vulkan_view_type(resource.texture_type))
            .format(format)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: aspect,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: layout.array_layers,
            });
        let view = unsafe { context.device.create_image_view(&view_info, None) }
            .map_err(|error| format!("create {label} view: {error}"))?;
        Ok(ImageAllocation {
            device: context.device.clone(),
            identity: resource.identity,
            descriptor_binding,
            descriptor_type,
            descriptor_element,
            descriptor_count,
            descriptor_aliases,
            image,
            view,
            memory,
            aspect,
            array_layers: layout.array_layers,
            extent,
            staging,
            general_ready: false,
        })
    }

    pub(super) fn requires_optimal_staging(
        texture_type: crate::case::TextureType,
        descriptor_type: vk::DescriptorType,
    ) -> bool {
        texture_type == crate::case::TextureType::D3
            && descriptor_type == vk::DescriptorType::SAMPLED_IMAGE
    }

    fn vulkan_sample_count(count: u32) -> Result<vk::SampleCountFlags, String> {
        match count {
            1 => Ok(vk::SampleCountFlags::TYPE_1),
            2 => Ok(vk::SampleCountFlags::TYPE_2),
            4 => Ok(vk::SampleCountFlags::TYPE_4),
            8 => Ok(vk::SampleCountFlags::TYPE_8),
            _ => Err(format!("unsupported Vulkan sample count {count}")),
        }
    }

    pub(super) fn assemble_initializer(assembly: &str) -> Result<Vec<u8>, String> {
        let scratch = crate::ScratchDir::new("multisample-initializer")?;
        let asm = scratch.path().join("initializer.spvasm");
        let spv = scratch.path().join("initializer.spv");
        std::fs::write(&asm, assembly)
            .map_err(|error| format!("write {}: {error}", asm.display()))?;
        let asm = asm
            .to_str()
            .ok_or_else(|| "multisample initializer path is not UTF-8".to_string())?;
        let spv_path = spv
            .to_str()
            .ok_or_else(|| "multisample initializer output path is not UTF-8".to_string())?;
        metal2vulkan::tools::run(
            "spirv-as",
            &["--target-env", "vulkan1.3", asm, "-o", spv_path],
        )?;
        std::fs::read(&spv).map_err(|error| format!("read {}: {error}", spv.display()))
    }

    struct MultisampleInitObjects {
        device: Device,
        staging_image: vk::Image,
        staging_memory: vk::DeviceMemory,
        layer_views: Vec<vk::ImageView>,
        framebuffers: Vec<vk::Framebuffer>,
        descriptor_pool: vk::DescriptorPool,
        descriptor_layout: vk::DescriptorSetLayout,
        pipeline_layout: vk::PipelineLayout,
        render_pass: vk::RenderPass,
        pipeline: vk::Pipeline,
        shader: vk::ShaderModule,
        command_pool: vk::CommandPool,
        fence: vk::Fence,
    }

    impl MultisampleInitObjects {
        fn new(device: &Device) -> Self {
            Self {
                device: device.clone(),
                staging_image: vk::Image::null(),
                staging_memory: vk::DeviceMemory::null(),
                layer_views: Vec::new(),
                framebuffers: Vec::new(),
                descriptor_pool: vk::DescriptorPool::null(),
                descriptor_layout: vk::DescriptorSetLayout::null(),
                pipeline_layout: vk::PipelineLayout::null(),
                render_pass: vk::RenderPass::null(),
                pipeline: vk::Pipeline::null(),
                shader: vk::ShaderModule::null(),
                command_pool: vk::CommandPool::null(),
                fence: vk::Fence::null(),
            }
        }
    }

    impl Drop for MultisampleInitObjects {
        fn drop(&mut self) {
            unsafe {
                if self.fence != vk::Fence::null() {
                    self.device.destroy_fence(self.fence, None);
                }
                if self.command_pool != vk::CommandPool::null() {
                    self.device.destroy_command_pool(self.command_pool, None);
                }
                if self.pipeline != vk::Pipeline::null() {
                    self.device.destroy_pipeline(self.pipeline, None);
                }
                for framebuffer in self.framebuffers.drain(..) {
                    self.device.destroy_framebuffer(framebuffer, None);
                }
                if self.render_pass != vk::RenderPass::null() {
                    self.device.destroy_render_pass(self.render_pass, None);
                }
                if self.pipeline_layout != vk::PipelineLayout::null() {
                    self.device
                        .destroy_pipeline_layout(self.pipeline_layout, None);
                }
                if self.descriptor_pool != vk::DescriptorPool::null() {
                    self.device
                        .destroy_descriptor_pool(self.descriptor_pool, None);
                }
                if self.descriptor_layout != vk::DescriptorSetLayout::null() {
                    self.device
                        .destroy_descriptor_set_layout(self.descriptor_layout, None);
                }
                for view in self.layer_views.drain(..) {
                    self.device.destroy_image_view(view, None);
                }
                if self.shader != vk::ShaderModule::null() {
                    self.device.destroy_shader_module(self.shader, None);
                }
                if self.staging_image != vk::Image::null() {
                    self.device.destroy_image(self.staging_image, None);
                }
                if self.staging_memory != vk::DeviceMemory::null() {
                    self.device.free_memory(self.staging_memory, None);
                }
            }
        }
    }

    fn create_multisample_image(
        context: &VulkanContext,
        resource: &TextureLiteralRef<'_>,
        descriptor: TextureDescriptorTarget,
        descriptor_aliases: Vec<TextureDescriptorTarget>,
        layout: crate::literal::TextureLayout,
    ) -> Result<ImageAllocation, String> {
        let label = &resource.label;
        let format = vulkan_format(resource.format);
        let aspect = if resource.format == crate::case::TextureFormat::Depth32Float {
            vk::ImageAspectFlags::DEPTH
        } else {
            vk::ImageAspectFlags::COLOR
        };
        let attachment_feature = if aspect == vk::ImageAspectFlags::DEPTH {
            vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT
        } else {
            vk::FormatFeatureFlags::COLOR_ATTACHMENT
        };
        let properties = unsafe {
            context
                .instance
                .get_physical_device_format_properties(context.physical, format)
        };
        if !properties
            .optimal_tiling_features
            .contains(attachment_feature | vk::FormatFeatureFlags::SAMPLED_IMAGE)
            || !properties
                .linear_tiling_features
                .contains(vk::FormatFeatureFlags::SAMPLED_IMAGE)
        {
            return Err(format!(
                "Vulkan device lacks multisample attachment/staging support for {label} format {}",
                format.as_raw()
            ));
        }
        let samples = vulkan_sample_count(layout.sample_count)?;
        let usage = vk::ImageUsageFlags::SAMPLED
            | if aspect == vk::ImageAspectFlags::DEPTH {
                vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
            } else {
                vk::ImageUsageFlags::COLOR_ATTACHMENT
            };
        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(vk::Extent3D {
                width: layout.width,
                height: layout.height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(layout.array_layers)
            .samples(samples)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        let image = unsafe { context.device.create_image(&image_info, None) }
            .map_err(|error| format!("create multisample {label}: {error}"))?;
        let requirements = unsafe { context.device.get_image_memory_requirements(image) };
        let memory_type = match context.memory_type(
            requirements.memory_type_bits,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        ) {
            Ok(memory_type) => memory_type,
            Err(error) => {
                unsafe { context.device.destroy_image(image, None) };
                return Err(error);
            }
        };
        let allocation = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(memory_type);
        let memory = match unsafe { context.device.allocate_memory(&allocation, None) } {
            Ok(memory) => memory,
            Err(error) => {
                unsafe { context.device.destroy_image(image, None) };
                return Err(format!("allocate multisample {label}: {error}"));
            }
        };
        if let Err(error) = unsafe { context.device.bind_image_memory(image, memory, 0) } {
            unsafe {
                context.device.destroy_image(image, None);
                context.device.free_memory(memory, None);
            }
            return Err(format!("bind multisample {label}: {error}"));
        }
        let view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vulkan_view_type(resource.texture_type))
            .format(format)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: aspect,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: layout.array_layers,
            });
        let view = match unsafe { context.device.create_image_view(&view_info, None) } {
            Ok(view) => view,
            Err(error) => {
                unsafe {
                    context.device.destroy_image(image, None);
                    context.device.free_memory(memory, None);
                }
                return Err(format!("create multisample {label} view: {error}"));
            }
        };
        if let Err(error) = initialize_multisample_image(
            context,
            image,
            format,
            aspect,
            layout,
            resource.format,
            resource.bytes,
        ) {
            unsafe {
                context.device.destroy_image_view(view, None);
                context.device.destroy_image(image, None);
                context.device.free_memory(memory, None);
            }
            return Err(error);
        }
        Ok(ImageAllocation {
            device: context.device.clone(),
            identity: resource.identity,
            descriptor_binding: descriptor.binding,
            descriptor_type: descriptor.descriptor_type,
            descriptor_element: descriptor.element,
            descriptor_count: descriptor.count,
            descriptor_aliases,
            image,
            view,
            memory,
            aspect,
            array_layers: layout.array_layers,
            extent: vk::Extent3D {
                width: layout.width,
                height: layout.height,
                depth: layout.depth,
            },
            staging: None,
            general_ready: true,
        })
    }

    fn initialize_multisample_image(
        context: &VulkanContext,
        destination: vk::Image,
        format: vk::Format,
        aspect: vk::ImageAspectFlags,
        layout: crate::literal::TextureLayout,
        literal_format: crate::case::TextureFormat,
        bytes: &[u8],
    ) -> Result<(), String> {
        if !context.sample_rate_shading {
            return Err("Vulkan device does not support per-sample literal initialization".into());
        }
        let mut objects = MultisampleInitObjects::new(&context.device);
        let staging_layers = layout
            .array_layers
            .checked_mul(layout.sample_count)
            .ok_or_else(|| "multisample staging layer count overflows".to_string())?;
        let staging_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(vk::Extent3D {
                width: layout.width,
                height: layout.height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(staging_layers)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::LINEAR)
            .usage(vk::ImageUsageFlags::SAMPLED)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::PREINITIALIZED);
        objects.staging_image = unsafe { context.device.create_image(&staging_info, None) }
            .map_err(|error| format!("create multisample staging image: {error}"))?;
        let requirements = unsafe {
            context
                .device
                .get_image_memory_requirements(objects.staging_image)
        };
        let memory_type = context.memory_type(
            requirements.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        let allocation = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(memory_type);
        objects.staging_memory = unsafe { context.device.allocate_memory(&allocation, None) }
            .map_err(|error| format!("allocate multisample staging memory: {error}"))?;
        unsafe {
            context
                .device
                .bind_image_memory(objects.staging_image, objects.staging_memory, 0)
        }
        .map_err(|error| format!("bind multisample staging image: {error}"))?;
        let mapped = unsafe {
            context.device.map_memory(
                objects.staging_memory,
                0,
                vk::WHOLE_SIZE,
                vk::MemoryMapFlags::empty(),
            )
        }
        .map_err(|error| format!("map multisample staging image: {error}"))?;
        let pixel_size = literal_format.bytes_per_pixel();
        for layer in 0..layout.array_layers {
            for sample in 0..layout.sample_count {
                let staging_layer = layer * layout.sample_count + sample;
                let host_layout = unsafe {
                    context.device.get_image_subresource_layout(
                        objects.staging_image,
                        vk::ImageSubresource {
                            aspect_mask: aspect,
                            mip_level: 0,
                            array_layer: staging_layer,
                        },
                    )
                };
                for y in 0..layout.height {
                    for x in 0..layout.width {
                        let source_texel = (((layer as usize * layout.height as usize
                            + y as usize)
                            * layout.width as usize
                            + x as usize)
                            * layout.sample_count as usize
                            + sample as usize)
                            * pixel_size;
                        let destination_texel = host_layout.offset as usize
                            + y as usize * host_layout.row_pitch as usize
                            + x as usize * pixel_size;
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                bytes.as_ptr().add(source_texel),
                                mapped.cast::<u8>().add(destination_texel),
                                pixel_size,
                            );
                        }
                    }
                }
            }
        }
        unsafe { context.device.unmap_memory(objects.staging_memory) };

        for layer in 0..layout.array_layers {
            let view_info = vk::ImageViewCreateInfo::default()
                .image(destination)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(format)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: aspect,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: layer,
                    layer_count: 1,
                });
            objects.layer_views.push(
                unsafe { context.device.create_image_view(&view_info, None) }
                    .map_err(|error| format!("create multisample layer {layer} view: {error}"))?,
            );
        }

        let descriptor_binding = [vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)];
        objects.descriptor_layout = unsafe {
            context.device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&descriptor_binding),
                None,
            )
        }
        .map_err(|error| format!("create multisample initializer descriptor layout: {error}"))?;
        let layouts = [objects.descriptor_layout];
        objects.pipeline_layout = unsafe {
            context.device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts),
                None,
            )
        }
        .map_err(|error| format!("create multisample initializer pipeline layout: {error}"))?;

        let samples = vulkan_sample_count(layout.sample_count)?;
        let attachment_layout = if aspect == vk::ImageAspectFlags::DEPTH {
            vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL
        } else {
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
        };
        let attachments = [vk::AttachmentDescription::default()
            .format(format)
            .samples(samples)
            .load_op(vk::AttachmentLoadOp::DONT_CARE)
            .store_op(vk::AttachmentStoreOp::STORE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::GENERAL)];
        let reference = vk::AttachmentReference::default()
            .attachment(0)
            .layout(attachment_layout);
        let color_references = [reference];
        let mut subpass =
            vk::SubpassDescription::default().pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS);
        if aspect == vk::ImageAspectFlags::DEPTH {
            subpass = subpass.depth_stencil_attachment(&reference);
        } else {
            subpass = subpass.color_attachments(&color_references);
        }
        let subpasses = [subpass];
        objects.render_pass = unsafe {
            context.device.create_render_pass(
                &vk::RenderPassCreateInfo::default()
                    .attachments(&attachments)
                    .subpasses(&subpasses),
                None,
            )
        }
        .map_err(|error| format!("create multisample initializer render pass: {error}"))?;
        for view in &objects.layer_views {
            let framebuffer_attachments = [*view];
            let info = vk::FramebufferCreateInfo::default()
                .render_pass(objects.render_pass)
                .attachments(&framebuffer_attachments)
                .width(layout.width)
                .height(layout.height)
                .layers(1);
            objects.framebuffers.push(
                unsafe { context.device.create_framebuffer(&info, None) }.map_err(|error| {
                    format!("create multisample initializer framebuffer: {error}")
                })?,
            );
        }

        let spv = assemble_initializer(&multisample_initializer_spvasm(literal_format))?;
        objects.shader = create_shader_module(context, &spv, "multisample initializer")?;
        let vertex_name = CString::new("vertex").expect("static vertex entry");
        let fragment_name = CString::new("fragment").expect("static fragment entry");
        let stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(objects.shader)
                .name(&vertex_name),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(objects.shader)
                .name(&fragment_name),
        ];
        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
        let viewports = [vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: layout.width as f32,
            height: layout.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        }];
        let scissors = [vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: vk::Extent2D {
                width: layout.width,
                height: layout.height,
            },
        }];
        let viewport = vk::PipelineViewportStateCreateInfo::default()
            .viewports(&viewports)
            .scissors(&scissors);
        let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::NONE)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .line_width(1.0);
        let multisample = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(samples)
            .sample_shading_enable(true)
            .min_sample_shading(1.0);
        let color_attachments = [vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::RGBA)];
        let color_blend = vk::PipelineColorBlendStateCreateInfo::default().attachments(
            if aspect == vk::ImageAspectFlags::DEPTH {
                &[]
            } else {
                &color_attachments
            },
        );
        let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(aspect == vk::ImageAspectFlags::DEPTH)
            .depth_write_enable(aspect == vk::ImageAspectFlags::DEPTH)
            .depth_compare_op(vk::CompareOp::ALWAYS);
        let pipeline_info = [vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport)
            .rasterization_state(&rasterization)
            .multisample_state(&multisample)
            .depth_stencil_state(&depth_stencil)
            .color_blend_state(&color_blend)
            .layout(objects.pipeline_layout)
            .render_pass(objects.render_pass)
            .subpass(0)];
        objects.pipeline = unsafe {
            context.device.create_graphics_pipelines(
                vk::PipelineCache::null(),
                &pipeline_info,
                None,
            )
        }
        .map_err(|(_, error)| format!("create multisample initializer pipeline: {error}"))?[0];

        let pool_sizes = [vk::DescriptorPoolSize {
            ty: vk::DescriptorType::SAMPLED_IMAGE,
            descriptor_count: layout.array_layers,
        }];
        objects.descriptor_pool = unsafe {
            context.device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(layout.array_layers)
                    .pool_sizes(&pool_sizes),
                None,
            )
        }
        .map_err(|error| format!("create multisample initializer descriptor pool: {error}"))?;
        let set_layouts = vec![objects.descriptor_layout; layout.array_layers as usize];
        let descriptor_sets = unsafe {
            context.device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(objects.descriptor_pool)
                    .set_layouts(&set_layouts),
            )
        }
        .map_err(|error| format!("allocate multisample initializer descriptors: {error}"))?;
        let mut source_views = Vec::with_capacity(layout.array_layers as usize);
        for layer in 0..layout.array_layers {
            let info = vk::ImageViewCreateInfo::default()
                .image(objects.staging_image)
                .view_type(vk::ImageViewType::TYPE_2D_ARRAY)
                .format(format)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: aspect,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: layer * layout.sample_count,
                    layer_count: layout.sample_count,
                });
            source_views.push(
                unsafe { context.device.create_image_view(&info, None) }
                    .map_err(|error| format!("create multisample source layer view: {error}"))?,
            );
        }
        objects.layer_views.extend(source_views.iter().copied());
        let image_infos = source_views
            .iter()
            .map(|view| {
                [vk::DescriptorImageInfo::default()
                    .image_view(*view)
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)]
            })
            .collect::<Vec<_>>();
        let writes = descriptor_sets
            .iter()
            .zip(&image_infos)
            .map(|(set, info)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(*set)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                    .image_info(info)
            })
            .collect::<Vec<_>>();
        unsafe { context.device.update_descriptor_sets(&writes, &[]) };

        objects.command_pool = unsafe {
            context.device.create_command_pool(
                &vk::CommandPoolCreateInfo::default().queue_family_index(context.queue_family),
                None,
            )
        }
        .map_err(|error| format!("create multisample initializer command pool: {error}"))?;
        let command = unsafe {
            context.device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(objects.command_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )
        }
        .map_err(|error| format!("allocate multisample initializer command: {error}"))?[0];
        unsafe {
            context.device.begin_command_buffer(
                command,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
        }
        .map_err(|error| format!("begin multisample initializer command: {error}"))?;
        let staging_barrier = [vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::HOST_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ)
            .old_layout(vk::ImageLayout::PREINITIALIZED)
            .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(objects.staging_image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: aspect,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: staging_layers,
            })];
        unsafe {
            context.device.cmd_pipeline_barrier(
                command,
                vk::PipelineStageFlags::HOST,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &staging_barrier,
            );
            context.device.cmd_bind_pipeline(
                command,
                vk::PipelineBindPoint::GRAPHICS,
                objects.pipeline,
            );
            for (framebuffer, descriptor_set) in objects.framebuffers.iter().zip(&descriptor_sets) {
                context.device.cmd_begin_render_pass(
                    command,
                    &vk::RenderPassBeginInfo::default()
                        .render_pass(objects.render_pass)
                        .framebuffer(*framebuffer)
                        .render_area(scissors[0]),
                    vk::SubpassContents::INLINE,
                );
                context.device.cmd_bind_descriptor_sets(
                    command,
                    vk::PipelineBindPoint::GRAPHICS,
                    objects.pipeline_layout,
                    0,
                    &[*descriptor_set],
                    &[],
                );
                context.device.cmd_draw(command, 3, 1, 0, 0);
                context.device.cmd_end_render_pass(command);
            }
            context.device.end_command_buffer(command)
        }
        .map_err(|error| format!("end multisample initializer command: {error}"))?;
        objects.fence = unsafe {
            context
                .device
                .create_fence(&vk::FenceCreateInfo::default(), None)
        }
        .map_err(|error| format!("create multisample initializer fence: {error}"))?;
        let command_buffers = [command];
        let submits = [vk::SubmitInfo::default().command_buffers(&command_buffers)];
        unsafe {
            context
                .device
                .queue_submit(context.queue, &submits, objects.fence)
                .map_err(|error| format!("submit multisample initializer: {error}"))?;
            context
                .device
                .wait_for_fences(&[objects.fence], true, u64::MAX)
                .map_err(|error| format!("wait for multisample initializer: {error}"))?;
        }
        Ok(())
    }

    fn upload_linear_image(
        context: &VulkanContext,
        image: vk::Image,
        memory: vk::DeviceMemory,
        resource: &TextureLiteralRef<'_>,
        layout: crate::literal::TextureLayout,
        aspect: vk::ImageAspectFlags,
    ) -> Result<(), String> {
        let mapped = unsafe {
            context
                .device
                .map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
        }
        .map_err(|error| format!("map {}: {error}", resource.label))?;
        let row_bytes = resource.dimensions[0] as usize * resource.format.bytes_per_pixel();
        let image_bytes = row_bytes * resource.dimensions[1] as usize;
        for layer in 0..layout.array_layers {
            let subresource = vk::ImageSubresource {
                aspect_mask: aspect,
                mip_level: 0,
                array_layer: layer,
            };
            let host_layout = unsafe {
                context
                    .device
                    .get_image_subresource_layout(image, subresource)
            };
            for z in 0..layout.depth {
                for y in 0..layout.height {
                    let source_offset = layer as usize * image_bytes
                        + z as usize * image_bytes
                        + y as usize * row_bytes;
                    let destination_offset = host_layout.offset as usize
                        + z as usize * host_layout.depth_pitch as usize
                        + y as usize * host_layout.row_pitch as usize;
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            resource.bytes.as_ptr().add(source_offset),
                            mapped.cast::<u8>().add(destination_offset),
                            row_bytes,
                        );
                    }
                }
            }
        }
        unsafe { context.device.unmap_memory(memory) };
        Ok(())
    }

    fn create_samplers(
        context: &VulkanContext,
        case: &AuthoredCase,
        reflection: &ShaderReflection,
    ) -> Result<Vec<SamplerAllocation>, String> {
        let mut samplers = Vec::new();
        for binding in &reflection.bindings {
            let descriptor = match binding.descriptor {
                Some(descriptor) => descriptor,
                None => continue,
            };
            let state = match binding.kind {
                metal2vulkan::reflect::ResourceKind::Sampler => {
                    let resource = case
                        .samplers
                        .iter()
                        .find(|resource| resource.binding == binding.metal_index)
                        .ok_or_else(|| format!("missing sampler {}", binding.metal_index))?;
                    let specialization = reflection
                        .runtime_sampler_specializations
                        .iter()
                        .find(|specialization| specialization.metal_index == binding.metal_index)
                        .ok_or_else(|| {
                            format!(
                                "runtime sampler {} has no reflected specialization",
                                binding.metal_index
                            )
                        })?;
                    if specialization.state != resource.runtime_specialization() {
                        return Err(format!(
                            "runtime sampler {} reflection does not match the authored pipeline state",
                            binding.metal_index
                        ));
                    }
                    sampler_info_from_runtime(
                        &specialization.state,
                        context.sampler_anisotropy,
                        context.max_sampler_anisotropy,
                    )?
                }
                metal2vulkan::reflect::ResourceKind::StaticSampler => sampler_info_from_static(
                    binding.static_sampler.as_ref().ok_or_else(|| {
                        format!(
                            "static sampler {} has no reflected state",
                            descriptor.binding
                        )
                    })?,
                    context.sampler_anisotropy,
                    context.max_sampler_anisotropy,
                )?,
                _ => continue,
            };
            let sampler = unsafe { context.device.create_sampler(&state, None) }
                .map_err(|error| format!("create sampler {}: {error}", descriptor.binding))?;
            samplers.push(SamplerAllocation {
                device: context.device.clone(),
                descriptor_binding: descriptor.binding,
                sampler,
            });
        }
        Ok(samplers)
    }

    fn sampler_info_from_runtime(
        state: &metal2vulkan::reflect::RuntimeSamplerState,
        sampler_anisotropy: bool,
        max_sampler_anisotropy: f32,
    ) -> Result<vk::SamplerCreateInfo<'static>, String> {
        sampler_info_from_static(
            &metal2vulkan::reflect::StaticSamplerState {
                min_filter: state.min_filter,
                mag_filter: state.mag_filter,
                mip_filter: state.mip_filter,
                address_mode_s: state.address_mode_s,
                address_mode_t: state.address_mode_t,
                address_mode_r: state.address_mode_r,
                coordinates: state.coordinates,
                compare_function: state.compare_function,
                max_anisotropy: state.max_anisotropy,
                lod_min_clamp: state.lod_min_clamp,
                lod_max_clamp: state.lod_max_clamp,
                border_color: state.border_color,
                reduction: state.reduction,
                lod_bias: state.lod_bias,
                raw_words: [0; 2],
            },
            sampler_anisotropy,
            max_sampler_anisotropy,
        )
    }

    pub(super) fn sampler_info_from_static(
        state: &metal2vulkan::reflect::StaticSamplerState,
        sampler_anisotropy: bool,
        max_sampler_anisotropy: f32,
    ) -> Result<vk::SamplerCreateInfo<'static>, String> {
        use metal2vulkan::reflect::{
            SamplerBorderColor, SamplerFilter, SamplerMipFilter, SamplerReduction,
        };
        if state.max_anisotropy > 1 && !sampler_anisotropy {
            return Err("Vulkan device does not support sampler anisotropy".into());
        }
        if state.max_anisotropy as f32 > max_sampler_anisotropy {
            return Err(format!(
                "Vulkan device sampler anisotropy limit is {max_sampler_anisotropy}, AIR requires {}",
                state.max_anisotropy
            ));
        }
        if state.reduction != SamplerReduction::WeightedAverage {
            return Err("Vulkan sampler min/max reduction extension is not enabled".into());
        }
        let border = match state.border_color {
            SamplerBorderColor::TransparentBlack => vk::BorderColor::FLOAT_TRANSPARENT_BLACK,
            SamplerBorderColor::OpaqueBlack => vk::BorderColor::FLOAT_OPAQUE_BLACK,
            SamplerBorderColor::OpaqueWhite => vk::BorderColor::FLOAT_OPAQUE_WHITE,
        };
        let filter = |filter| match filter {
            SamplerFilter::Nearest => vk::Filter::NEAREST,
            SamplerFilter::Linear | SamplerFilter::Bicubic => vk::Filter::LINEAR,
        };
        let mip = match state.mip_filter {
            SamplerMipFilter::None | SamplerMipFilter::Nearest => vk::SamplerMipmapMode::NEAREST,
            SamplerMipFilter::Linear => vk::SamplerMipmapMode::LINEAR,
        };
        let address = |mode| {
            if state.coordinates == metal2vulkan::reflect::SamplerCoordinates::Pixel
                && !matches!(
                    mode,
                    metal2vulkan::reflect::SamplerAddressMode::ClampToEdge
                        | metal2vulkan::reflect::SamplerAddressMode::ClampToBorder
                )
            {
                // Pixel address behavior is emitted as explicit fetch-coordinate/clamp logic for
                // source shapes admitted by the shared executor contract. The descriptor remains
                // statically present but its address mode is not observed by those instructions.
                vk::SamplerAddressMode::CLAMP_TO_EDGE
            } else {
                vulkan_static_address_mode(mode)
            }
        };
        Ok(vk::SamplerCreateInfo::default()
            .mag_filter(filter(state.mag_filter))
            .min_filter(filter(state.min_filter))
            .mipmap_mode(mip)
            .address_mode_u(address(state.address_mode_s))
            .address_mode_v(address(state.address_mode_t))
            .address_mode_w(address(state.address_mode_r))
            .mip_lod_bias(state.lod_bias)
            .anisotropy_enable(state.max_anisotropy > 1)
            .max_anisotropy(state.max_anisotropy as f32)
            // The product emits ordinary sampling plus an in-shader comparison, never a Dref
            // instruction. Enabling Vulkan comparison here would not implement AIR semantics and
            // is forbidden for unnormalized-coordinate samplers.
            .compare_enable(false)
            .compare_op(vk::CompareOp::NEVER)
            // Literal authored textures have exactly one mip. Vulkan requires both clamps to be
            // zero for unnormalized coordinates; canonicalizing them cannot change the selected
            // mip because level zero is the only level the executor creates.
            .min_lod(
                if state.coordinates == metal2vulkan::reflect::SamplerCoordinates::Pixel {
                    0.0
                } else {
                    state.lod_min_clamp
                },
            )
            .max_lod(
                if state.coordinates == metal2vulkan::reflect::SamplerCoordinates::Pixel {
                    0.0
                } else {
                    state.lod_max_clamp
                },
            )
            .border_color(border)
            .unnormalized_coordinates(matches!(
                state.coordinates,
                metal2vulkan::reflect::SamplerCoordinates::Pixel
            )))
    }

    fn vulkan_static_address_mode(
        mode: metal2vulkan::reflect::SamplerAddressMode,
    ) -> vk::SamplerAddressMode {
        match mode {
            metal2vulkan::reflect::SamplerAddressMode::ClampToZero
            | metal2vulkan::reflect::SamplerAddressMode::ClampToBorder => {
                vk::SamplerAddressMode::CLAMP_TO_BORDER
            }
            metal2vulkan::reflect::SamplerAddressMode::ClampToEdge => {
                vk::SamplerAddressMode::CLAMP_TO_EDGE
            }
            metal2vulkan::reflect::SamplerAddressMode::Repeat => vk::SamplerAddressMode::REPEAT,
            metal2vulkan::reflect::SamplerAddressMode::MirroredRepeat => {
                vk::SamplerAddressMode::MIRRORED_REPEAT
            }
        }
    }

    fn vulkan_format(format: crate::case::TextureFormat) -> vk::Format {
        match format {
            crate::case::TextureFormat::R8Unorm => vk::Format::R8_UNORM,
            crate::case::TextureFormat::Rgba8Unorm => vk::Format::R8G8B8A8_UNORM,
            crate::case::TextureFormat::Rgba8Uint => vk::Format::R8G8B8A8_UINT,
            crate::case::TextureFormat::Rgba8Sint => vk::Format::R8G8B8A8_SINT,
            crate::case::TextureFormat::R16Float => vk::Format::R16_SFLOAT,
            crate::case::TextureFormat::R16Uint => vk::Format::R16_UINT,
            crate::case::TextureFormat::Rg16Float => vk::Format::R16G16_SFLOAT,
            crate::case::TextureFormat::Rgba16Float => vk::Format::R16G16B16A16_SFLOAT,
            crate::case::TextureFormat::Rgba16Uint => vk::Format::R16G16B16A16_UINT,
            crate::case::TextureFormat::R32Uint => vk::Format::R32_UINT,
            crate::case::TextureFormat::R32Sint => vk::Format::R32_SINT,
            crate::case::TextureFormat::R32Float => vk::Format::R32_SFLOAT,
            crate::case::TextureFormat::Rgba32Uint => vk::Format::R32G32B32A32_UINT,
            crate::case::TextureFormat::Rgba32Sint => vk::Format::R32G32B32A32_SINT,
            crate::case::TextureFormat::Rgba32Float => vk::Format::R32G32B32A32_SFLOAT,
            crate::case::TextureFormat::Depth32Float => vk::Format::D32_SFLOAT,
        }
    }

    pub(super) fn multisample_initializer_spvasm(format: crate::case::TextureFormat) -> String {
        use crate::case::TextureFormat as F;
        let (component, four, output_pointer, output) = match format {
            F::Rgba8Uint | F::R16Uint | F::Rgba16Uint | F::R32Uint | F::Rgba32Uint => {
                ("uint", "%v4uint", "%ptr_output_v4uint", "%color")
            }
            F::Rgba8Sint | F::R32Sint | F::Rgba32Sint => {
                ("int", "%v4int", "%ptr_output_v4int", "%color")
            }
            F::R8Unorm
            | F::Rgba8Unorm
            | F::R16Float
            | F::Rg16Float
            | F::Rgba16Float
            | F::R32Float
            | F::Rgba32Float
            | F::Depth32Float => ("float", "%v4float", "%ptr_output_v4float", "%color"),
        };
        let depth = format == F::Depth32Float;
        let depth_execution = if depth {
            "OpExecutionMode %fragment DepthReplacing"
        } else {
            ""
        };
        let fragment_interfaces = if depth {
            "%frag_coord %sample_id %source %frag_depth"
        } else {
            "%frag_coord %sample_id %source %color"
        };
        let output_decorations = if depth {
            "OpDecorate %frag_depth BuiltIn FragDepth".to_string()
        } else {
            format!("OpDecorate {output} Location 0")
        };
        let output_variable = if depth {
            "%frag_depth = OpVariable %ptr_output_float Output".to_string()
        } else {
            format!("{output} = OpVariable {output_pointer} Output")
        };
        let fragment_store = if depth {
            "%red = OpCompositeExtract %float %texel 0\nOpStore %frag_depth %red".to_string()
        } else {
            format!("OpStore {output} %texel")
        };
        format!(
            r#"OpCapability Shader
OpCapability SampleRateShading
OpMemoryModel Logical GLSL450
OpEntryPoint Vertex %vertex "vertex" %vertex_index %position
OpEntryPoint Fragment %fragment "fragment" {fragment_interfaces}
OpExecutionMode %fragment OriginUpperLeft
{depth_execution}
OpDecorate %vertex_index BuiltIn VertexIndex
OpDecorate %position BuiltIn Position
OpDecorate %frag_coord BuiltIn FragCoord
OpDecorate %sample_id BuiltIn SampleId
OpDecorate %sample_id Flat
OpDecorate %source DescriptorSet 0
OpDecorate %source Binding 0
{output_decorations}
%void = OpTypeVoid
%bool = OpTypeBool
%int = OpTypeInt 32 1
%uint = OpTypeInt 32 0
%float = OpTypeFloat 32
%v3int = OpTypeVector %int 3
%v4int = OpTypeVector %int 4
%v4uint = OpTypeVector %uint 4
%v4float = OpTypeVector %float 4
%fn = OpTypeFunction %void
%ptr_input_int = OpTypePointer Input %int
%ptr_input_v4float = OpTypePointer Input %v4float
%ptr_output_v4float = OpTypePointer Output %v4float
%ptr_output_v4int = OpTypePointer Output %v4int
%ptr_output_v4uint = OpTypePointer Output %v4uint
%ptr_output_float = OpTypePointer Output %float
%image = OpTypeImage %{component} 2D 0 1 0 1 Unknown
%ptr_uniform_image = OpTypePointer UniformConstant %image
%vertex_index = OpVariable %ptr_input_int Input
%position = OpVariable %ptr_output_v4float Output
%frag_coord = OpVariable %ptr_input_v4float Input
%sample_id = OpVariable %ptr_input_int Input
%source = OpVariable %ptr_uniform_image UniformConstant
{output_variable}
%int_1 = OpConstant %int 1
%int_2 = OpConstant %int 2
%float_n1 = OpConstant %float -1
%float_0 = OpConstant %float 0
%float_1 = OpConstant %float 1
%float_3 = OpConstant %float 3
%vertex = OpFunction %void None %fn
%vertex_entry = OpLabel
%vertex_id = OpLoad %int %vertex_index
%is_one = OpIEqual %bool %vertex_id %int_1
%is_two = OpIEqual %bool %vertex_id %int_2
%x = OpSelect %float %is_one %float_3 %float_n1
%y = OpSelect %float %is_two %float_3 %float_n1
%vertex_position = OpCompositeConstruct %v4float %x %y %float_0 %float_1
OpStore %position %vertex_position
OpReturn
OpFunctionEnd
%fragment = OpFunction %void None %fn
%fragment_entry = OpLabel
%position_value = OpLoad %v4float %frag_coord
%fx = OpCompositeExtract %float %position_value 0
%fy = OpCompositeExtract %float %position_value 1
%ix = OpConvertFToS %int %fx
%iy = OpConvertFToS %int %fy
%sample = OpLoad %int %sample_id
%coordinate = OpCompositeConstruct %v3int %ix %iy %sample
%source_image = OpLoad %image %source
%texel = OpImageFetch {four} %source_image %coordinate
{fragment_store}
OpReturn
OpFunctionEnd
"#
        )
    }

    fn vulkan_attribute_format(format: crate::case::AttributeFormat) -> vk::Format {
        use crate::case::AttributeFormat as F;
        match format {
            F::Char => vk::Format::R8_SINT,
            F::Char2 => vk::Format::R8G8_SINT,
            F::Char3 => vk::Format::R8G8B8_SINT,
            F::Char4 => vk::Format::R8G8B8A8_SINT,
            F::Uchar => vk::Format::R8_UINT,
            F::Uchar2 => vk::Format::R8G8_UINT,
            F::Uchar3 => vk::Format::R8G8B8_UINT,
            F::Uchar4 => vk::Format::R8G8B8A8_UINT,
            F::Short => vk::Format::R16_SINT,
            F::Short2 => vk::Format::R16G16_SINT,
            F::Short3 => vk::Format::R16G16B16_SINT,
            F::Short4 => vk::Format::R16G16B16A16_SINT,
            F::Ushort => vk::Format::R16_UINT,
            F::Ushort2 => vk::Format::R16G16_UINT,
            F::Ushort3 => vk::Format::R16G16B16_UINT,
            F::Ushort4 => vk::Format::R16G16B16A16_UINT,
            F::Half => vk::Format::R16_SFLOAT,
            F::Half2 => vk::Format::R16G16_SFLOAT,
            F::Half3 => vk::Format::R16G16B16_SFLOAT,
            F::Half4 => vk::Format::R16G16B16A16_SFLOAT,
            F::Float => vk::Format::R32_SFLOAT,
            F::Float2 => vk::Format::R32G32_SFLOAT,
            F::Float3 => vk::Format::R32G32B32_SFLOAT,
            F::Float4 => vk::Format::R32G32B32A32_SFLOAT,
            F::Uint => vk::Format::R32_UINT,
            F::Uint2 => vk::Format::R32G32_UINT,
            F::Uint3 => vk::Format::R32G32B32_UINT,
            F::Uint4 => vk::Format::R32G32B32A32_UINT,
            F::Int => vk::Format::R32_SINT,
            F::Int2 => vk::Format::R32G32_SINT,
            F::Int3 => vk::Format::R32G32B32_SINT,
            F::Int4 => vk::Format::R32G32B32A32_SINT,
        }
    }

    fn vulkan_view_type(texture_type: crate::case::TextureType) -> vk::ImageViewType {
        match texture_type {
            crate::case::TextureType::Buffer => {
                unreachable!("buffer textures use VkBufferView")
            }
            crate::case::TextureType::D1 => vk::ImageViewType::TYPE_1D,
            crate::case::TextureType::D1Array => vk::ImageViewType::TYPE_1D_ARRAY,
            crate::case::TextureType::D2 | crate::case::TextureType::D2Multisample => {
                vk::ImageViewType::TYPE_2D
            }
            crate::case::TextureType::D2Array | crate::case::TextureType::D2MultisampleArray => {
                vk::ImageViewType::TYPE_2D_ARRAY
            }
            crate::case::TextureType::D3 => vk::ImageViewType::TYPE_3D,
            crate::case::TextureType::Cube => vk::ImageViewType::CUBE,
            crate::case::TextureType::CubeArray => vk::ImageViewType::CUBE_ARRAY,
        }
    }

    unsafe fn prepare_texel_buffers(
        device: &Device,
        command: vk::CommandBuffer,
        buffers: &[TexelBufferAllocation],
    ) {
        let barriers = buffers
            .iter()
            .map(|buffer| {
                vk::BufferMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::HOST_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .buffer(buffer.buffer.buffer)
                    .offset(0)
                    .size(buffer.buffer.len)
            })
            .collect::<Vec<_>>();
        if !barriers.is_empty() {
            unsafe {
                device.cmd_pipeline_barrier(
                    command,
                    vk::PipelineStageFlags::HOST,
                    vk::PipelineStageFlags::ALL_COMMANDS,
                    vk::DependencyFlags::empty(),
                    &[],
                    &barriers,
                    &[],
                );
            }
        }
    }

    unsafe fn make_texel_buffers_host_readable(
        device: &Device,
        command: vk::CommandBuffer,
        buffers: &[TexelBufferAllocation],
    ) {
        let barriers = buffers
            .iter()
            .filter(|buffer| buffer.uses_descriptor_type(vk::DescriptorType::STORAGE_TEXEL_BUFFER))
            .map(|buffer| {
                vk::BufferMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags::HOST_READ)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .buffer(buffer.buffer.buffer)
                    .offset(0)
                    .size(buffer.buffer.len)
            })
            .collect::<Vec<_>>();
        if !barriers.is_empty() {
            unsafe {
                device.cmd_pipeline_barrier(
                    command,
                    vk::PipelineStageFlags::ALL_COMMANDS,
                    vk::PipelineStageFlags::HOST,
                    vk::DependencyFlags::empty(),
                    &[],
                    &barriers,
                    &[],
                );
            }
        }
    }

    unsafe fn transition_images_to_general(
        device: &Device,
        command: vk::CommandBuffer,
        images: &[ImageAllocation],
    ) {
        let linear_barriers = images
            .iter()
            .filter(|image| !image.general_ready && image.staging.is_none())
            .map(|image| {
                vk::ImageMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::HOST_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)
                    .old_layout(vk::ImageLayout::PREINITIALIZED)
                    .new_layout(vk::ImageLayout::GENERAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(image.image)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: image.aspect,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: image.array_layers,
                    })
            })
            .collect::<Vec<_>>();
        if !linear_barriers.is_empty() {
            unsafe {
                device.cmd_pipeline_barrier(
                    command,
                    vk::PipelineStageFlags::HOST,
                    vk::PipelineStageFlags::ALL_COMMANDS,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &linear_barriers,
                );
            }
        }
        let staged = images
            .iter()
            .filter_map(|image| image.staging.as_ref().map(|staging| (image, staging)))
            .collect::<Vec<_>>();
        if staged.is_empty() {
            return;
        }
        let transfer_barriers = staged
            .iter()
            .map(|(image, _)| {
                vk::ImageMemoryBarrier::default()
                    .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(image.image)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: image.aspect,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: image.array_layers,
                    })
            })
            .collect::<Vec<_>>();
        unsafe {
            device.cmd_pipeline_barrier(
                command,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &transfer_barriers,
            );
            for (image, staging) in &staged {
                let copy = vk::BufferImageCopy::default()
                    .image_subresource(vk::ImageSubresourceLayers {
                        aspect_mask: image.aspect,
                        mip_level: 0,
                        base_array_layer: 0,
                        layer_count: image.array_layers,
                    })
                    .image_extent(image.extent);
                device.cmd_copy_buffer_to_image(
                    command,
                    staging.buffer,
                    image.image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[copy],
                );
            }
        }
        let general_barriers = staged
            .iter()
            .map(|(image, _)| {
                vk::ImageMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ)
                    .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .new_layout(vk::ImageLayout::GENERAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(image.image)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: image.aspect,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: image.array_layers,
                    })
            })
            .collect::<Vec<_>>();
        unsafe {
            device.cmd_pipeline_barrier(
                command,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &general_barriers,
            );
        }
    }

    unsafe fn make_images_host_readable(
        device: &Device,
        command: vk::CommandBuffer,
        images: &[ImageAllocation],
    ) {
        let barriers = images
            .iter()
            .filter(|image| image.uses_descriptor_type(vk::DescriptorType::STORAGE_IMAGE))
            .map(|image| {
                vk::ImageMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags::HOST_READ)
                    .old_layout(vk::ImageLayout::GENERAL)
                    .new_layout(vk::ImageLayout::GENERAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(image.image)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: image.aspect,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: image.array_layers,
                    })
            })
            .collect::<Vec<_>>();
        if !barriers.is_empty() {
            unsafe {
                device.cmd_pipeline_barrier(
                    command,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::HOST,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &barriers,
                );
            }
        }
    }

    fn create_buffers(
        context: &VulkanContext,
        resources: &LiteralResources,
        reflection: &ShaderReflection,
    ) -> Result<Vec<BufferAllocation>, String> {
        struct Input<'a> {
            descriptor_binding: Option<u32>,
            output_binding: Option<u32>,
            argument_buffer_source: Option<(u32, u32)>,
            label: String,
            buffer_address_table: bool,
            device_address_index: Option<u32>,
            bytes: std::borrow::Cow<'a, [u8]>,
        }
        let mut inputs = Vec::new();
        for resource in &resources.buffers {
            let reflected = reflection
                .bindings
                .iter()
                .find(|binding| {
                    binding.kind == metal2vulkan::reflect::ResourceKind::Buffer
                        && binding.metal_index == resource.binding
                })
                .ok_or_else(|| format!("buffer {} has no reflected binding", resource.binding))?;
            inputs.push(Input {
                descriptor_binding: Some(
                    reflected
                        .descriptor
                        .ok_or_else(|| format!("buffer {} has no descriptor", resource.binding))?
                        .binding,
                ),
                output_binding: Some(resource.binding),
                argument_buffer_source: None,
                buffer_address_table: false,
                device_address_index: Some(resource.binding),
                label: format!("buffer {}", resource.binding),
                bytes: (&resource.bytes[..]).into(),
            });
        }
        for resource in &resources.acceleration_structure_shadows {
            let expected_kind = match resource.kind {
                crate::case::AccelerationStructureKind::Instance => {
                    metal2vulkan::reflect::ResourceKind::AccelerationStructureShadow
                }
                crate::case::AccelerationStructureKind::Primitive => {
                    metal2vulkan::reflect::ResourceKind::PrimitiveAccelerationStructure
                }
            };
            let reflected = reflection
                .bindings
                .iter()
                .find(|binding| {
                    binding.kind == expected_kind && binding.metal_index == resource.binding
                })
                .ok_or_else(|| {
                    format!(
                        "acceleration structure {} has no reflected binding",
                        resource.binding
                    )
                })?;
            let Some(descriptor) = reflected.descriptor else {
                continue;
            };
            inputs.push(Input {
                descriptor_binding: Some(descriptor.binding),
                output_binding: None,
                argument_buffer_source: None,
                buffer_address_table: false,
                device_address_index: Some(resource.binding),
                label: format!("acceleration structure {}", resource.binding),
                bytes: (&resource.bytes[..]).into(),
            });
        }
        for resource in &resources.kernel_stage_inputs {
            let reflected = reflection
                .bindings
                .iter()
                .find(|binding| {
                    binding.kind == metal2vulkan::reflect::ResourceKind::KernelStageInput
                        && binding.stage_input_location == Some(resource.location)
                })
                .ok_or_else(|| {
                    format!(
                        "kernel stage input {} has no reflected binding",
                        resource.location
                    )
                })?;
            inputs.push(Input {
                descriptor_binding: Some(
                    reflected
                        .descriptor
                        .ok_or_else(|| {
                            format!("kernel stage input {} has no descriptor", resource.location)
                        })?
                        .binding,
                ),
                output_binding: None,
                argument_buffer_source: None,
                buffer_address_table: false,
                device_address_index: None,
                label: format!("kernel stage input {}", resource.location),
                bytes: (&resource.bytes[..]).into(),
            });
        }
        for resource in &resources.argument_buffer_buffers {
            let source = (resource.buffer_binding, resource.field_offset);
            let reflected = reflection
                .bindings
                .iter()
                .find(|binding| {
                    binding.kind == metal2vulkan::reflect::ResourceKind::EmbeddedArgBufferBuffer
                        && binding.embedded_source.is_some_and(|embedded| {
                            (embedded.buffer_index, embedded.field_offset) == source
                        })
                })
                .ok_or_else(|| {
                    format!(
                        "argument-buffer buffer {}+{} has no reflected binding",
                        resource.buffer_binding, resource.field_offset
                    )
                })?;
            if reflected.descriptor.is_some() {
                return Err("embedded argument-buffer buffer unexpectedly has a descriptor".into());
            }
            inputs.push(Input {
                descriptor_binding: None,
                output_binding: None,
                argument_buffer_source: Some(source),
                buffer_address_table: false,
                device_address_index: None,
                label: format!(
                    "argument-buffer buffer {}+{}",
                    resource.buffer_binding, resource.field_offset
                ),
                bytes: (&resource.bytes[..]).into(),
            });
        }
        if let Some(binding) = reflection
            .bindings
            .iter()
            .find(|binding| binding.kind == metal2vulkan::reflect::ResourceKind::BufferAddressTable)
        {
            let descriptor = binding
                .descriptor
                .ok_or("buffer-address table has no descriptor")?;
            let slots = inputs
                .iter()
                .filter_map(|buffer| buffer.device_address_index)
                .max()
                .unwrap_or(0)
                .saturating_add(1) as usize;
            inputs.push(Input {
                descriptor_binding: Some(descriptor.binding),
                output_binding: None,
                argument_buffer_source: None,
                buffer_address_table: true,
                device_address_index: None,
                label: "buffer-address table".into(),
                bytes: vec![0; slots.saturating_mul(8)].into(),
            });
        }
        let buffers = inputs
            .into_iter()
            .map(|input| {
                let binding = input.descriptor_binding;
                let bytes = input.bytes;
                let create_info = vk::BufferCreateInfo::default()
                    .size(bytes.len() as u64)
                    .usage(
                        vk::BufferUsageFlags::STORAGE_BUFFER
                            | if context.buffer_device_address {
                                vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                            } else {
                                vk::BufferUsageFlags::empty()
                            },
                    )
                    .sharing_mode(vk::SharingMode::EXCLUSIVE);
                let buffer = unsafe { context.device.create_buffer(&create_info, None) }
                    .map_err(|error| format!("create {}: {error}", input.label))?;
                let requirements = unsafe { context.device.get_buffer_memory_requirements(buffer) };
                let memory_type = context.memory_type(
                    requirements.memory_type_bits,
                    vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                )?;
                let mut address_flags = vk::MemoryAllocateFlagsInfo::default()
                    .flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
                let mut allocation_info = vk::MemoryAllocateInfo::default()
                    .allocation_size(requirements.size)
                    .memory_type_index(memory_type);
                if context.buffer_device_address {
                    allocation_info = allocation_info.push_next(&mut address_flags);
                }
                let memory = match unsafe { context.device.allocate_memory(&allocation_info, None) }
                {
                    Ok(memory) => memory,
                    Err(error) => {
                        unsafe { context.device.destroy_buffer(buffer, None) };
                        return Err(format!("allocate {} memory: {error}", input.label));
                    }
                };
                if let Err(error) = unsafe { context.device.bind_buffer_memory(buffer, memory, 0) }
                {
                    unsafe {
                        context.device.destroy_buffer(buffer, None);
                        context.device.free_memory(memory, None);
                    }
                    return Err(format!("bind {} memory: {error}", input.label));
                }
                let pointer = unsafe {
                    context.device.map_memory(
                        memory,
                        0,
                        bytes.len() as u64,
                        vk::MemoryMapFlags::empty(),
                    )
                }
                .map_err(|error| format!("map {}: {error}", input.label))?;
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        bytes.as_ptr(),
                        pointer.cast::<u8>(),
                        bytes.len(),
                    );
                    context.device.unmap_memory(memory);
                }
                Ok(BufferAllocation {
                    device: context.device.clone(),
                    descriptor_binding: binding,
                    output_binding: input.output_binding,
                    argument_buffer_source: input.argument_buffer_source,
                    buffer_address_table: input.buffer_address_table,
                    device_address_index: input.device_address_index,
                    buffer,
                    memory,
                    len: bytes.len() as u64,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        patch_argument_buffer_device_addresses(context, &buffers, reflection)?;
        patch_buffer_address_table(context, &buffers)?;
        Ok(buffers)
    }

    fn patch_argument_buffer_device_addresses(
        context: &VulkanContext,
        buffers: &[BufferAllocation],
        reflection: &ShaderReflection,
    ) -> Result<(), String> {
        for nested in buffers
            .iter()
            .filter(|buffer| buffer.argument_buffer_source.is_some())
        {
            if !context.buffer_device_address {
                return Err(
                    "Vulkan device does not support bufferDeviceAddress required by an argument-buffer device pointer"
                        .into(),
                );
            }
            let (owner_binding, field_offset) = nested.argument_buffer_source.unwrap();
            let owner = buffers
                .iter()
                .find(|buffer| buffer.output_binding == Some(owner_binding))
                .ok_or_else(|| format!("argument buffer {owner_binding} was not allocated"))?;
            let reflected = reflection
                .bindings
                .iter()
                .find(|binding| {
                    binding.kind
                        == metal2vulkan::reflect::ResourceKind::EmbeddedArgBufferBuffer
                        && binding.embedded_source.is_some_and(|source| {
                            source.buffer_index == owner_binding
                                && source.field_offset == field_offset
                        })
                })
                .and_then(|binding| binding.embedded_source)
                .ok_or_else(|| {
                    format!(
                        "argument-buffer buffer {owner_binding}+{field_offset} has no reflection coordinate"
                    )
                })?;
            if reflected.resource_buffer_index.is_none() {
                return Err("embedded buffer reflection has no nested resource index".into());
            }
            let end = u64::from(field_offset).saturating_add(8);
            if end > owner.len {
                return Err(format!(
                    "argument-buffer device address at {owner_binding}+{field_offset} exceeds owner length {}",
                    owner.len
                ));
            }
            let address = unsafe {
                context.device.get_buffer_device_address(
                    &vk::BufferDeviceAddressInfo::default().buffer(nested.buffer),
                )
            };
            if address == 0 {
                return Err("Vulkan returned a null address for an embedded device buffer".into());
            }
            let pointer = unsafe {
                context.device.map_memory(
                    owner.memory,
                    u64::from(field_offset),
                    8,
                    vk::MemoryMapFlags::empty(),
                )
            }
            .map_err(|error| format!("map argument-buffer device-address field: {error}"))?;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    address.to_le_bytes().as_ptr(),
                    pointer.cast::<u8>(),
                    8,
                );
                context.device.unmap_memory(owner.memory);
            }
        }
        Ok(())
    }

    fn patch_buffer_address_table(
        context: &VulkanContext,
        buffers: &[BufferAllocation],
    ) -> Result<(), String> {
        let Some(table) = buffers.iter().find(|buffer| buffer.buffer_address_table) else {
            return Ok(());
        };
        if !context.buffer_device_address {
            return Err(
                "Vulkan device does not support bufferDeviceAddress required by the translated module"
                    .into(),
            );
        }
        let pointer = unsafe {
            context
                .device
                .map_memory(table.memory, 0, table.len, vk::MemoryMapFlags::empty())
        }
        .map_err(|error| format!("map buffer-address table: {error}"))?;
        let bytes =
            unsafe { std::slice::from_raw_parts_mut(pointer.cast::<u8>(), table.len as usize) };
        for buffer in buffers {
            let Some(index) = buffer.device_address_index else {
                continue;
            };
            let start = index as usize * 8;
            let end = start.saturating_add(8);
            if end > bytes.len() {
                unsafe { context.device.unmap_memory(table.memory) };
                return Err(format!("buffer-address table has no slot {index}"));
            }
            let address = unsafe {
                context.device.get_buffer_device_address(
                    &vk::BufferDeviceAddressInfo::default().buffer(buffer.buffer),
                )
            };
            if address == 0 {
                unsafe { context.device.unmap_memory(table.memory) };
                return Err(format!(
                    "Vulkan returned a null address for buffer slot {index}"
                ));
            }
            bytes[start..end].copy_from_slice(&address.to_le_bytes());
        }
        unsafe { context.device.unmap_memory(table.memory) };
        Ok(())
    }

    fn selected_output(
        context: &VulkanContext,
        case: &AuthoredCase,
        resources: &LiteralResources,
        buffers: &[BufferAllocation],
        bound_images: BoundImageResources<'_>,
        render_targets: &[RenderTargetAllocation],
        depth_stencil: Option<&DepthStencilAllocation>,
    ) -> Result<Vec<u8>, String> {
        let images = bound_images.images;
        let texel_buffers = bound_images.texel_buffers;
        let reflection = bound_images.reflection;
        match case.output {
            OutputSelection::Buffer {
                binding,
                offset,
                length,
            } => {
                let buffer = buffers
                    .iter()
                    .find(|buffer| buffer.output_binding == Some(binding))
                    .ok_or_else(|| format!("output buffer {binding} was not bound"))?;
                if offset.saturating_add(length) > buffer.len {
                    return Err("selected output exceeds Vulkan buffer".into());
                }
                let pointer = unsafe {
                    context.device.map_memory(
                        buffer.memory,
                        offset,
                        length,
                        vk::MemoryMapFlags::empty(),
                    )
                }
                .map_err(|error| format!("map output buffer {binding}: {error}"))?;
                let output = unsafe {
                    std::slice::from_raw_parts(pointer.cast::<u8>(), length as usize).to_vec()
                };
                unsafe { context.device.unmap_memory(buffer.memory) };
                Ok(output)
            }
            OutputSelection::ArgumentBufferBuffer {
                buffer_binding,
                field_offset,
                offset,
                length,
            } => {
                let buffer = buffers
                    .iter()
                    .find(|buffer| {
                        buffer.argument_buffer_source == Some((buffer_binding, field_offset))
                    })
                    .ok_or_else(|| {
                        format!(
                            "output argument-buffer buffer {buffer_binding}+{field_offset} was not bound"
                        )
                    })?;
                if offset.saturating_add(length) > buffer.len {
                    return Err("selected output exceeds Vulkan argument-buffer buffer".into());
                }
                let pointer = unsafe {
                    context.device.map_memory(
                        buffer.memory,
                        offset,
                        length,
                        vk::MemoryMapFlags::empty(),
                    )
                }
                .map_err(|error| format!("map argument-buffer buffer output: {error}"))?;
                let output = unsafe {
                    std::slice::from_raw_parts(pointer.cast::<u8>(), length as usize).to_vec()
                };
                unsafe { context.device.unmap_memory(buffer.memory) };
                Ok(output)
            }
            OutputSelection::Texture {
                binding,
                origin,
                dimensions,
            } => {
                let resource = resources
                    .textures
                    .iter()
                    .find(|resource| resource.binding == binding)
                    .ok_or_else(|| format!("output texture {binding} has no literal resource"))?;
                if resource.texture_type == crate::case::TextureType::Buffer {
                    let allocation = texel_buffers
                        .iter()
                        .find(|buffer| buffer.binding == binding)
                        .ok_or_else(|| format!("output texture buffer {binding} was not bound"))?;
                    let pointer = unsafe {
                        context.device.map_memory(
                            allocation.buffer.memory,
                            0,
                            allocation.buffer.len,
                            vk::MemoryMapFlags::empty(),
                        )
                    }
                    .map_err(|error| format!("map texture buffer {binding}: {error}"))?;
                    let bytes = unsafe {
                        std::slice::from_raw_parts(
                            pointer.cast::<u8>(),
                            allocation.buffer.len as usize,
                        )
                        .to_vec()
                    };
                    unsafe { context.device.unmap_memory(allocation.buffer.memory) };
                    let output = crate::literal::LiteralTexture {
                        binding,
                        role: resource.role,
                        texture_type: crate::case::TextureType::Buffer,
                        format: allocation.format,
                        dimensions: allocation.dimensions,
                        sample_count: 1,
                        bytes,
                    };
                    return output.select(origin, dimensions);
                }
                let allocation = images
                    .iter()
                    .find(|image| image.identity == ImageIdentity::Texture(binding))
                    .ok_or_else(|| format!("output texture {binding} was not bound"))?;
                read_texture_region(
                    context,
                    allocation,
                    &TextureLiteralRef::top_level(resource),
                    origin,
                    dimensions,
                )
            }
            OutputSelection::TextureArrayElement {
                binding,
                element,
                origin,
                dimensions,
            } => {
                let array = resources
                    .texture_arrays
                    .iter()
                    .find(|array| array.binding == binding)
                    .ok_or_else(|| {
                        format!("output texture-array {binding} has no literal resource")
                    })?;
                let resource = array.elements.get(element as usize).ok_or_else(|| {
                    format!("output texture-array {binding} element {element} is not declared")
                })?;
                let allocation = images
                    .iter()
                    .find(|image| {
                        image.identity == ImageIdentity::TextureArrayElement { binding, element }
                    })
                    .ok_or_else(|| {
                        format!("output texture-array {binding} element {element} was not bound")
                    })?;
                read_texture_region(
                    context,
                    allocation,
                    &TextureLiteralRef::array_element(binding, element, resource),
                    origin,
                    dimensions,
                )
            }
            OutputSelection::ArgumentBufferTexture {
                buffer_binding,
                field_offset,
                origin,
                dimensions,
            } => {
                let resource = resources
                    .argument_buffer_textures
                    .iter()
                    .find(|resource| {
                        resource.buffer_binding == buffer_binding
                            && resource.field_offset == field_offset
                    })
                    .ok_or_else(|| {
                        format!(
                            "output argument-buffer texture {buffer_binding}+{field_offset} has no literal resource"
                        )
                    })?;
                let allocation = images
                    .iter()
                    .find(|image| {
                        image.identity
                            == ImageIdentity::ArgumentBufferTexture {
                                buffer_binding,
                                field_offset,
                            }
                    })
                    .ok_or_else(|| format!("{} was not bound", resource.label()))?;
                read_texture_region(
                    context,
                    allocation,
                    &TextureLiteralRef::argument_buffer(resource),
                    origin,
                    dimensions,
                )
            }
            OutputSelection::RenderTarget {
                index,
                origin,
                dimensions,
            } => {
                if case.stage == Stage::Kernel {
                    let resource = resources
                        .render_targets
                        .iter()
                        .find(|resource| resource.index == index)
                        .ok_or_else(|| format!("output render target {index} is not authored"))?;
                    let allocation = images
                        .iter()
                        .find(|image| {
                            image.identity
                                == ImageIdentity::ImplicitImageblock {
                                    attachment: index,
                                    data_rate: 0,
                                }
                        })
                        .ok_or_else(|| {
                            format!("implicit imageblock output attachment {index} was not bound")
                        })?;
                    let literal = crate::literal::LiteralTexture {
                        binding: allocation.descriptor_binding,
                        role: crate::case::ResourceRole::InOut,
                        texture_type: crate::case::TextureType::D2Array,
                        format: resource.format,
                        dimensions: [
                            resource.dimensions[0],
                            resource.dimensions[1],
                            allocation.array_layers,
                        ],
                        sample_count: 1,
                        bytes: resource.bytes.clone(),
                    };
                    let mut input = TextureLiteralRef::top_level(&literal);
                    input.identity = allocation.identity;
                    input.label = format!("implicit imageblock attachment {index} rate 0");
                    read_texture_region(
                        context,
                        allocation,
                        &input,
                        [origin[0], origin[1], 0],
                        [dimensions[0], dimensions[1], 1],
                    )
                } else {
                    read_render_target_region(
                        context,
                        render_targets
                            .iter()
                            .find(|target| target.index == index)
                            .ok_or_else(|| format!("output render target {index} was not bound"))?,
                        origin,
                        dimensions,
                    )
                }
            }
            OutputSelection::Depth { origin, dimensions }
            | OutputSelection::Stencil { origin, dimensions } => {
                let attachment = depth_stencil
                    .ok_or_else(|| "depth/stencil output attachment was not bound".to_string())?;
                let depth = matches!(case.output, OutputSelection::Depth { .. });
                let transfer = if depth {
                    attachment.depth.as_ref()
                } else {
                    attachment.stencil.as_ref()
                }
                .ok_or_else(|| "selected depth/stencil aspect was not allocated".to_string())?;
                let pointer = unsafe {
                    context.device.map_memory(
                        transfer.memory,
                        0,
                        transfer.len,
                        vk::MemoryMapFlags::empty(),
                    )
                }
                .map_err(|error| format!("map depth/stencil readback: {error}"))?;
                let bytes = unsafe {
                    std::slice::from_raw_parts(pointer.cast::<u8>(), transfer.len as usize)
                };
                let output = crate::literal::select_tightly_packed_2d(
                    bytes,
                    attachment.dimensions,
                    origin,
                    dimensions,
                    if depth { 4 } else { 1 },
                );
                unsafe { context.device.unmap_memory(transfer.memory) };
                output
            }
            OutputSelection::FragmentImageblock {
                ref semantic,
                origin,
                dimensions,
            } => {
                let reflected = reflection
                    .fragment_imageblock
                    .as_ref()
                    .and_then(|imageblock| {
                        imageblock
                            .members
                            .iter()
                            .find(|member| member.semantic == *semantic)
                    })
                    .ok_or_else(|| {
                        format!("fragment imageblock member {semantic} is not reflected")
                    })?;
                let binding = reflected.binding.ok_or_else(|| {
                    format!("fragment imageblock member {semantic} has no descriptor")
                })?;
                let authored = resources
                    .fragment_imageblock
                    .as_ref()
                    .ok_or_else(|| "fragment imageblock is not authored".to_string())?;
                let member = authored
                    .members
                    .iter()
                    .find(|member| member.semantic == *semantic)
                    .ok_or_else(|| {
                        format!("fragment imageblock member {semantic} is not authored")
                    })?;
                let allocation = images
                    .iter()
                    .find(|image| image.identity == ImageIdentity::FragmentImageblock { binding })
                    .ok_or_else(|| {
                        format!("fragment imageblock member {semantic} was not bound")
                    })?;
                let literal = crate::literal::LiteralTexture {
                    binding,
                    role: member.role,
                    texture_type: crate::case::TextureType::D2,
                    format: member.format.texture_format(),
                    dimensions: [authored.dimensions[0], authored.dimensions[1], 1],
                    sample_count: 1,
                    bytes: member.bytes.clone(),
                };
                let mut input = TextureLiteralRef::top_level(&literal);
                input.identity = allocation.identity;
                input.label = format!("fragment imageblock member {semantic}");
                read_texture_region(
                    context,
                    allocation,
                    &input,
                    [origin[0], origin[1], 0],
                    [dimensions[0], dimensions[1], 1],
                )
            }
        }
    }

    fn read_render_target_region(
        context: &VulkanContext,
        target: &RenderTargetAllocation,
        origin: [u32; 2],
        dimensions: [u32; 2],
    ) -> Result<Vec<u8>, String> {
        let pixel_size = target.format.bytes_per_pixel();
        let source_row = target.dimensions[0] as usize * pixel_size;
        let selected_row = dimensions[0] as usize * pixel_size;
        let mapped = unsafe {
            context.device.map_memory(
                target.transfer.memory,
                0,
                target.transfer.len,
                vk::MemoryMapFlags::empty(),
            )
        }
        .map_err(|error| format!("map render target {} readback: {error}", target.index))?;
        let mut output = Vec::with_capacity(selected_row * dimensions[1] as usize);
        for y in origin[1]..origin[1] + dimensions[1] {
            let offset = y as usize * source_row + origin[0] as usize * pixel_size;
            let row = unsafe {
                std::slice::from_raw_parts(mapped.cast::<u8>().add(offset), selected_row)
            };
            output.extend_from_slice(row);
        }
        unsafe { context.device.unmap_memory(target.transfer.memory) };
        Ok(output)
    }

    fn read_texture_region(
        context: &VulkanContext,
        allocation: &ImageAllocation,
        resource: &TextureLiteralRef<'_>,
        origin: [u32; 3],
        dimensions: [u32; 3],
    ) -> Result<Vec<u8>, String> {
        let layout = resource.layout()?;
        let pixel_size = resource.format.bytes_per_pixel();
        let selected_row = dimensions[0] as usize * pixel_size;
        let mut output =
            Vec::with_capacity(selected_row * dimensions[1] as usize * dimensions[2] as usize);
        let mapped = unsafe {
            context.device.map_memory(
                allocation.memory,
                0,
                vk::WHOLE_SIZE,
                vk::MemoryMapFlags::empty(),
            )
        }
        .map_err(|error| format!("map output {}: {error}", resource.label))?;
        for selected_z in 0..dimensions[2] {
            let (array_layer, depth_slice) =
                if resource.texture_type == crate::case::TextureType::D3 {
                    (0, origin[2] + selected_z)
                } else {
                    (origin[2] + selected_z, 0)
                };
            if array_layer >= layout.array_layers || depth_slice >= layout.depth {
                unsafe { context.device.unmap_memory(allocation.memory) };
                return Err(format!("selected region exceeds Vulkan {}", resource.label));
            }
            let subresource = vk::ImageSubresource {
                aspect_mask: allocation.aspect,
                mip_level: 0,
                array_layer,
            };
            let host_layout = unsafe {
                context
                    .device
                    .get_image_subresource_layout(allocation.image, subresource)
            };
            for y in 0..dimensions[1] {
                let source_offset = host_layout.offset as usize
                    + depth_slice as usize * host_layout.depth_pitch as usize
                    + (origin[1] + y) as usize * host_layout.row_pitch as usize
                    + origin[0] as usize * pixel_size;
                let row = unsafe {
                    std::slice::from_raw_parts(mapped.cast::<u8>().add(source_offset), selected_row)
                };
                output.extend_from_slice(row);
            }
        }
        unsafe { context.device.unmap_memory(allocation.memory) };
        Ok(output)
    }

    fn load_entry() -> Result<Entry, String> {
        if let Ok(path) = std::env::var("METAL2VULKAN_LIBVULKAN") {
            if !path.trim().is_empty() {
                return unsafe { Entry::load_from(path.trim()) }
                    .map_err(|error| format!("load Vulkan library {path}: {error}"));
            }
        }
        if let Ok(entry) = unsafe { Entry::load() } {
            return Ok(entry);
        }
        for path in vulkan_library_candidates() {
            if !path.is_file() {
                continue;
            }
            if let Ok(entry) = unsafe { Entry::load_from(&path) } {
                return Ok(entry);
            }
        }
        Err(
            "no Vulkan loader found; install vulkan-loader (and MoltenVK on macOS) or set METAL2VULKAN_LIBVULKAN"
                .into(),
        )
    }

    fn vulkan_library_candidates() -> Vec<PathBuf> {
        let mut paths = Vec::new();
        if let Ok(sdk) = std::env::var("VULKAN_SDK") {
            for relative in [
                "lib/libvulkan.dylib",
                "lib/libvulkan.1.dylib",
                "macOS/lib/libvulkan.dylib",
                "lib/libvulkan.so.1",
            ] {
                paths.push(Path::new(&sdk).join(relative));
            }
        }
        for path in [
            "/opt/homebrew/lib/libvulkan.dylib",
            "/opt/homebrew/lib/libvulkan.1.dylib",
            "/usr/local/lib/libvulkan.dylib",
            "/usr/local/lib/libvulkan.1.dylib",
            "/usr/lib/libvulkan.so.1",
            "/usr/local/lib/libvulkan.so.1",
        ] {
            paths.push(PathBuf::from(path));
        }
        paths
    }

    fn extension_name(raw: &[std::ffi::c_char]) -> &CStr {
        unsafe { CStr::from_ptr(raw.as_ptr()) }
    }

    fn div_ceil(numerator: u32, denominator: u32) -> u32 {
        numerator.saturating_add(denominator - 1) / denominator
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_names_cannot_relabel_the_hosts_vulkan_stack() {
        if cfg!(target_os = "macos") {
            assert!(validate_backend_host(Backend::Moltenvk).is_ok());
            assert!(validate_backend_host(Backend::Vulkan).is_err());
        } else if cfg!(target_os = "linux") {
            assert!(validate_backend_host(Backend::Moltenvk).is_err());
            assert!(validate_backend_host(Backend::Vulkan).is_ok());
        }
    }

    #[test]
    fn graphics_pipeline_hash_covers_every_shader_module() {
        let primary = [1, 2, 3, 4];
        assert_eq!(pipeline_spv_sha256(&[&primary]), sha256_bytes(&primary));
        assert_ne!(
            pipeline_spv_sha256(&[&primary, &[5, 6, 7, 8], &[9]]),
            pipeline_spv_sha256(&[&primary, &[5, 6, 7, 9], &[9]])
        );
    }

    #[cfg(any(target_os = "linux", target_os = "macos"))]
    #[test]
    fn sampled_3d_literals_use_portable_optimal_tiling_uploads() {
        assert!(platform::requires_optimal_staging(
            crate::case::TextureType::D3,
            ash::vk::DescriptorType::SAMPLED_IMAGE,
        ));
        assert!(!platform::requires_optimal_staging(
            crate::case::TextureType::D2,
            ash::vk::DescriptorType::SAMPLED_IMAGE,
        ));
        assert!(!platform::requires_optimal_staging(
            crate::case::TextureType::D3,
            ash::vk::DescriptorType::STORAGE_IMAGE,
        ));
    }

    #[cfg(any(target_os = "linux", target_os = "macos"))]
    #[test]
    fn rasterization_disabled_vertex_executes_narrow_attributes_without_a_companion() {
        use crate::case::{
            AttributeFormat, AttributeInput, BufferResource, Comparison, Draw, ExecutionSafety,
            OutputSelection, Primitive, ResourceRole,
        };

        let ll = include_str!("../fixtures/public/vertex_narrow_attributes.ll");
        let case = AuthoredCase {
            air_sha256: sha256_bytes(ll.as_bytes()),
            case_id: "test-vertex-side-effect".into(),
            name: "vertex-narrow-attributes-smoke".into(),
            entry: "vertex_narrow_attributes".into(),
            stage: Stage::Vertex,
            buffers: vec![BufferResource {
                binding: 0,
                role: ResourceRole::Output,
                bytes_b64: None,
                initial_bytes_b64: Some("q6urqw==".into()),
            }],
            argument_buffer_buffers: vec![],
            threadgroup_memory: vec![],
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
            vertex_inputs: vec![
                AttributeInput {
                    location: 0,
                    format: AttributeFormat::Uchar,
                    stride: 1,
                    bytes_b64: "Ag==".into(),
                },
                AttributeInput {
                    location: 1,
                    format: AttributeFormat::Ushort2,
                    stride: 4,
                    bytes_b64: "AwAEAA==".into(),
                },
            ],
            vertex_observation: None,
            kernel_stage_inputs: vec![],
            function_constants: vec![],
            dispatch: None,
            draw: Some(Draw {
                primitive: Primitive::Point,
                vertex_start: 0,
                vertex_count: 1,
                instance_count: 1,
            }),
            tessellation: None,
            output: OutputSelection::Buffer {
                binding: 0,
                offset: 0,
                length: 4,
            },
            compare: Comparison::Exact,
            execution_safety: ExecutionSafety::LoopFree,
            rationale: None,
            authored_by: Some("codex:gpt-5.6-sol".into()),
        };
        let reflection = metal2vulkan::reflect_sanitized(
            ll,
            metal2vulkan::passes::Stage::Vertex,
            metal2vulkan::passes::TransformOptions::default(),
        )
        .unwrap();
        let resources = LiteralResources::prepare(&case).unwrap();
        let scratch = crate::ScratchDir::new("vertex-side-effect-candidate").unwrap();
        let spv = metal2vulkan::translate_sanitized_native_with_options(
            ll,
            metal2vulkan::passes::Stage::Vertex,
            scratch.path(),
            metal2vulkan::passes::TransformOptions::default(),
        )
        .unwrap();
        let backend = if cfg!(target_os = "macos") {
            Backend::Moltenvk
        } else {
            Backend::Vulkan
        };
        let (output, _) =
            platform::execute(&case, &resources, &reflection, &spv, None, None, backend).unwrap();
        assert_eq!(output, 9u32.to_le_bytes());
    }

    #[cfg(any(target_os = "linux", target_os = "macos"))]
    #[test]
    fn combined_depth_stencil_outputs_execute_on_the_candidate_backend() {
        let ll = include_str!("../fixtures/public/fragment_depth_stencil.ll");
        let entry = "fragment_depth_stencil".to_string();
        let case =
            crate::case::combined_depth_stencil_test_case(sha256_bytes(ll.as_bytes()), entry);
        let reflection = metal2vulkan::reflect_sanitized(
            ll,
            metal2vulkan::passes::Stage::Fragment,
            metal2vulkan::passes::TransformOptions::default(),
        )
        .unwrap();
        let resources = LiteralResources::prepare(&case).unwrap();
        let scratch = crate::ScratchDir::new("fragment-depth-stencil-candidate").unwrap();
        let spv = metal2vulkan::translate_sanitized_native_with_options(
            ll,
            metal2vulkan::passes::Stage::Fragment,
            scratch.path(),
            metal2vulkan::passes::TransformOptions::default(),
        )
        .unwrap();
        let companion_ll = scratch.path().join("graphics-companion.ll");
        std::fs::write(&companion_ll, ll).unwrap();
        let companion =
            metal2vulkan::translate_passthrough(companion_ll.to_str().unwrap(), scratch.path())
                .unwrap();
        let backend = if cfg!(target_os = "macos") {
            Backend::Moltenvk
        } else {
            Backend::Vulkan
        };
        let (output, _) = platform::execute(
            &case,
            &resources,
            &reflection,
            &spv,
            Some(&companion),
            None,
            backend,
        )
        .unwrap();
        assert_eq!(output, [7]);
    }

    #[cfg(any(target_os = "linux", target_os = "macos"))]
    #[test]
    fn vector_function_constant_executes_all_authored_lanes_on_candidate() {
        let ll = include_str!("../fixtures/public/kernel_vector_function_constant.ll");
        let case = crate::case::vector_function_constant_test_case(
            sha256_bytes(ll.as_bytes()),
            "kernel_vector_function_constant".into(),
        );
        let reflection = metal2vulkan::reflect_sanitized(
            ll,
            metal2vulkan::passes::Stage::Kernel,
            metal2vulkan::passes::TransformOptions::default(),
        )
        .unwrap();
        let resources = LiteralResources::prepare(&case).unwrap();
        let scratch = crate::ScratchDir::new("vector-function-constant-candidate").unwrap();
        let spv = metal2vulkan::translate_sanitized_native_with_options(
            ll,
            metal2vulkan::passes::Stage::Kernel,
            scratch.path(),
            metal2vulkan::passes::TransformOptions::default(),
        )
        .unwrap();
        let spv = metal2vulkan::specialize_function_constant_bytes(
            &spv,
            &resources.function_constant_values(),
        )
        .unwrap();
        let backend = if cfg!(target_os = "macos") {
            Backend::Moltenvk
        } else {
            Backend::Vulkan
        };
        let (output, _) =
            platform::execute(&case, &resources, &reflection, &spv, None, None, backend).unwrap();
        assert_eq!(output, 10u32.to_le_bytes());
    }

    #[cfg(any(target_os = "linux", target_os = "macos"))]
    #[test]
    fn narrow_implicit_imageblock_executes_on_candidate() {
        let ll = include_str!("../fixtures/public/kernel_implicit_imageblock_half2.ll");
        let case = crate::case::narrow_implicit_imageblock_test_case(
            sha256_bytes(ll.as_bytes()),
            "kernel_implicit_imageblock_half2".into(),
        );
        let options = metal2vulkan::passes::TransformOptions {
            kernel_local_size: [16, 16, 1],
            kernel_threads_per_grid: Some([16, 16, 1]),
            ..metal2vulkan::passes::TransformOptions::default()
        };
        let reflection =
            metal2vulkan::reflect_sanitized(ll, metal2vulkan::passes::Stage::Kernel, options)
                .unwrap();
        assert_eq!(
            reflection.implicit_imageblock_attachments[0].format,
            metal2vulkan::meta::TextureFormat::Rg16f
        );
        let resources = LiteralResources::prepare(&case).unwrap();
        let scratch = crate::ScratchDir::new("narrow-implicit-imageblock-candidate").unwrap();
        let spv = metal2vulkan::translate_sanitized_native_with_options(
            ll,
            metal2vulkan::passes::Stage::Kernel,
            scratch.path(),
            options,
        )
        .unwrap();
        let backend = if cfg!(target_os = "macos") {
            Backend::Moltenvk
        } else {
            Backend::Vulkan
        };
        let (output, _) =
            platform::execute(&case, &resources, &reflection, &spv, None, None, backend).unwrap();
        assert_eq!(output, [0x00, 0x3c, 0x00, 0x40]);
    }

    #[cfg(any(target_os = "linux", target_os = "macos"))]
    #[test]
    fn custom_fragment_imageblock_executes_on_candidate() {
        let ll = include_str!("../fixtures/public/fragment_custom_imageblock.ll");
        let case = crate::case::fragment_imageblock_test_case(
            sha256_bytes(ll.as_bytes()),
            "fragment_custom_imageblock".into(),
        );
        let reflection = metal2vulkan::reflect_sanitized(
            ll,
            metal2vulkan::passes::Stage::Fragment,
            metal2vulkan::passes::TransformOptions::default(),
        )
        .unwrap();
        let resources = LiteralResources::prepare(&case).unwrap();
        let scratch = crate::ScratchDir::new("fragment-imageblock-candidate").unwrap();
        let spv = metal2vulkan::translate_sanitized_native_with_options(
            ll,
            metal2vulkan::passes::Stage::Fragment,
            scratch.path(),
            metal2vulkan::passes::TransformOptions::default(),
        )
        .unwrap();
        let companion_ll = scratch.path().join("graphics-companion.ll");
        std::fs::write(&companion_ll, ll).unwrap();
        let companion =
            metal2vulkan::translate_passthrough(companion_ll.to_str().unwrap(), scratch.path())
                .unwrap();
        let backend = if cfg!(target_os = "macos") {
            Backend::Moltenvk
        } else {
            Backend::Vulkan
        };
        let (output, _) = platform::execute(
            &case,
            &resources,
            &reflection,
            &spv,
            Some(&companion),
            None,
            backend,
        )
        .unwrap();
        assert_eq!(output, [0x00, 0x40]);
    }

    #[test]
    fn tessellation_companion_matches_and_validates_the_reflected_interface() {
        let ll = r#"
!air.vertex = !{!0}
!0 = !{ptr @tes, !1, !2, !8}
!1 = !{!3}
!2 = !{!4, !7, !9, !10, !11}
!3 = !{!"air.position", !"air.arg_type_name", !"float4"}
!4 = !{i32 0, !"air.patch_control_point_input", !5, !6}
!5 = !{!"air.patch_control_point_function", ptr @control.MTL_CONTROL_POINT_FN}
!6 = !{!"air.location_index", i32 1, i32 1, !"air.arg_type_name", !"float3"}
!7 = !{i32 1, !"air.patch_input", !"air.location_index", i32 4, i32 1, !"air.arg_type_name", !"float4"}
!8 = !{!"air.patch", !"quad", !"air.patch_control_point", i32 16}
!9 = !{i32 2, !"air.instance_id", !"air.arg_type_name", !"uint"}
!10 = !{i32 3, !"air.amplification_id", !"air.arg_type_name", !"ushort"}
!11 = !{i32 4, !"air.amplification_count", !"air.arg_type_name", !"ushort"}
"#;
        let meta = metal2vulkan::meta::parse_air_vertex_meta(ll).unwrap();
        let reflection = ShaderReflection::from_vertex(&meta, Some("tes"));
        let resources = LiteralResources {
            tessellation: Some(crate::literal::LiteralTessellation {
                factors: vec![crate::case::TessellationFactors {
                    edge_f16: vec![0x3c00; 4],
                    inside_f16: vec![0x3c00; 2],
                }],
                instance_count: 1,
                amplification_count: 1,
                control_points: vec![crate::literal::LiteralStageInput {
                    location: 1,
                    format: crate::case::AttributeFormat::Float3,
                    stride: 12,
                    bytes: vec![0; 16 * 12],
                }],
                patch_inputs: vec![crate::literal::LiteralStageInput {
                    location: 4,
                    format: crate::case::AttributeFormat::Float4,
                    stride: 16,
                    bytes: vec![0; 16],
                }],
            }),
            ..LiteralResources::default()
        };
        let assembly = tessellation_companion_spvasm(&resources, &reflection)
            .unwrap()
            .unwrap();
        assert!(assembly.contains("OpExecutionMode %control OutputVertices 16"));
        assert!(assembly.contains("OpDecorate %patch_0 Patch"));
        assert!(assembly.contains("OpDecorate %tcs_instance_out Location 5"));
        assert!(assembly.contains("OpCapability StorageInputOutput16"));
        assemble_spvasm(&assembly, "tessellation-companion-test").unwrap();
    }

    #[cfg(any(target_os = "linux", target_os = "macos"))]
    #[test]
    fn pixel_static_sampler_uses_a_legal_non_dref_single_mip_descriptor() {
        let ll = r#"
@__air_sampler_state = internal addrspace(2) constant [2 x i64] [i64 34901797601053330, i64 0], align 8
!air.sampler_states = !{!0}
!0 = !{!"air.sampler_state", ptr addrspace(2) @__air_sampler_state}
"#;
        let reflection = metal2vulkan::reflect_sanitized(
            ll,
            metal2vulkan::passes::Stage::Kernel,
            metal2vulkan::passes::TransformOptions::default(),
        )
        .unwrap();
        let state = reflection
            .bindings
            .iter()
            .find_map(|binding| binding.static_sampler.as_ref())
            .unwrap();
        let info = platform::sampler_info_from_static(state, false, 1.0).unwrap();
        assert_eq!(info.unnormalized_coordinates, ash::vk::TRUE);
        assert_eq!(info.compare_enable, ash::vk::FALSE);
        assert_eq!(info.min_lod, 0.0);
        assert_eq!(info.max_lod, 0.0);
        assert!(info.address_mode_u == ash::vk::SamplerAddressMode::CLAMP_TO_EDGE);
        assert!(info.address_mode_v == ash::vk::SamplerAddressMode::CLAMP_TO_EDGE);
    }

    #[cfg(any(target_os = "linux", target_os = "macos"))]
    #[test]
    fn texture_buffers_use_native_texel_buffer_descriptors() {
        use metal2vulkan::meta::{KernMeta, KernRole};

        let reflection = |type_name: &str| {
            let mut meta = KernMeta {
                roles: vec![(0, KernRole::Texture(0))],
                ..Default::default()
            };
            meta.texture_type_names.insert(0, type_name.into());
            ShaderReflection::from_kernel(&meta, Some("k"), [1, 1, 1])
        };
        let read = reflection("texture_buffer<half, read>");
        assert!(
            platform::vulkan_texture_descriptor_type(&read.bindings[0])
                == ash::vk::DescriptorType::UNIFORM_TEXEL_BUFFER
        );
        let write = reflection("texture_buffer<half, write>");
        assert!(
            platform::vulkan_texture_descriptor_type(&write.bindings[0])
                == ash::vk::DescriptorType::STORAGE_TEXEL_BUFFER
        );
    }

    #[cfg(any(target_os = "linux", target_os = "macos"))]
    #[test]
    fn sampled_and_storage_texture_aliases_populate_both_descriptor_bands() {
        use metal2vulkan::meta::{KernMeta, KernRole};

        for (read_type, write_type, expected_types) in [
            (
                "texture2d<float, sample>",
                "texture2d<float, write>",
                [
                    ash::vk::DescriptorType::STORAGE_IMAGE,
                    ash::vk::DescriptorType::SAMPLED_IMAGE,
                ],
            ),
            (
                "texture_buffer<float, read>",
                "texture_buffer<float, write>",
                [
                    ash::vk::DescriptorType::STORAGE_TEXEL_BUFFER,
                    ash::vk::DescriptorType::UNIFORM_TEXEL_BUFFER,
                ],
            ),
        ] {
            let mut meta = KernMeta {
                roles: vec![(0, KernRole::Texture(40)), (1, KernRole::Texture(40))],
                ..Default::default()
            };
            meta.texture_type_names.insert(0, read_type.into());
            meta.texture_type_names.insert(1, write_type.into());
            let reflection = ShaderReflection::from_kernel(&meta, Some("k"), [1, 1, 1]);
            reflection.validate_descriptor_abi().unwrap();

            let targets = platform::top_level_texture_targets(
                &reflection,
                40,
                read_type.starts_with("texture_buffer"),
            )
            .unwrap();
            assert_eq!(targets.len(), 2);
            assert!(targets[0].descriptor_type == expected_types[0]);
            assert!(targets[1].descriptor_type == expected_types[1]);
            assert_eq!(targets[0].binding, 520);
            assert_eq!(targets[1].binding, 72);

            if !read_type.starts_with("texture_buffer") {
                let mut array_reflection = reflection.clone();
                for binding in &mut array_reflection.bindings {
                    binding.kind = metal2vulkan::reflect::ResourceKind::TextureArray;
                    binding.descriptor.as_mut().unwrap().count = 128;
                }
                let targets = platform::texture_array_targets(&array_reflection, 40, 17).unwrap();
                assert_eq!(targets.len(), 2);
                assert!(targets[0].descriptor_type == expected_types[0]);
                assert!(targets[1].descriptor_type == expected_types[1]);
                assert_eq!(targets[0].element, 17);
                assert_eq!(targets[1].element, 17);
                assert_eq!(targets[0].count, 128);
                assert_eq!(targets[1].count, 128);
            }
        }
    }

    #[cfg(any(target_os = "linux", target_os = "macos"))]
    #[test]
    fn multisample_initializers_are_vulkan_legal_for_each_component_class() {
        use crate::case::TextureFormat;

        for format in [
            TextureFormat::Rgba32Float,
            TextureFormat::Rgba32Uint,
            TextureFormat::Rgba32Sint,
            TextureFormat::Depth32Float,
        ] {
            let bytes =
                platform::assemble_initializer(&platform::multisample_initializer_spvasm(format))
                    .unwrap();
            let scratch = crate::ScratchDir::new("multisample-initializer-test").unwrap();
            metal2vulkan::tools::spirv_val_bytes(&bytes, scratch.path()).unwrap();
        }
    }
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
mod platform {
    use super::*;

    pub fn execute(
        _case: &AuthoredCase,
        _resources: &LiteralResources,
        _reflection: &ShaderReflection,
        _spv: &[u8],
        _companion_spv: Option<&[u8]>,
        _tessellation_spv: Option<&[u8]>,
        _backend: Backend,
    ) -> Result<(Vec<u8>, serde_json::Value), String> {
        Err("candidate execution is supported only on Linux and macOS".into())
    }
}
