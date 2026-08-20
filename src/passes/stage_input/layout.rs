//! Stage-input and resource-block layout decoration.

use super::*;

pub(super) fn include_existing_private_globals(ctx: &mut Ctx) {
    let vars: Vec<Word> = ctx
        .module
        .types_global_values
        .iter()
        .filter(|inst| {
            inst.class.opcode == Op::Variable
                && inst.operands.first() == Some(&Operand::StorageClass(StorageClass::Private))
        })
        .filter_map(|inst| inst.result_id)
        .collect();
    for var in vars {
        ctx.interface_buffer_var(var);
    }
}

impl Ctx {
    /// SPIR-V 1.4 requires all global variables referenced by the entry to be on the interface list,
    /// while 1.3 forbids non-Input/Output variables there. Include buffers and resources when the
    /// module header selects 1.4 or newer.
    pub(in crate::passes) fn interface_buffer_var(&mut self, var: Word) {
        let v = self.module.header.as_ref().map(|h| h.version).unwrap_or(0);
        // version word: high byte major, next minor. 1.4 = 0x00010400.
        if v >= 0x0001_0400 {
            self.interface.push(var);
        }
    }

    /// Lazily create a default sampler resource variable for `air.get_read_sampler()`. Metal's
    /// `texture.read(coord)` is sampler-less, but AIR still threads a sampler pointer (from
    /// `air.get_read_sampler`) into the read intrinsic, where our `lower_read` ignores it. To keep the
    /// SSA value well-typed we materialize one real `OpTypeSampler` UniformConstant variable (binding
    /// past every binding already assigned by the interface pass) and load from it. Memoized so a
    /// shader with many reads shares a single sampler resource.
    pub(in crate::passes) fn default_read_sampler(&mut self) -> Result<Word, String> {
        if let Some(v) = self.default_sampler_var {
            return Ok(v);
        }
        let binding = allocate_static_sampler_binding(&self.module)
            .ok_or_else(|| "no free sampler binding for synthesized read sampler".to_string())?;
        let sty = self.ty_sampler();
        let pptr = self.ty_ptr(StorageClass::UniformConstant, sty);
        let var = self.module.fresh_id();
        self.new_globals.push(Instruction::new(
            Op::Variable,
            Some(pptr),
            Some(var),
            vec![Operand::StorageClass(StorageClass::UniformConstant)],
        ));
        decorate_binding(&mut self.module, var, binding);
        self.interface_buffer_var(var);
        self.default_sampler_var = Some(var);
        Ok(var)
    }

    /// Lazily create a default (null) 2D float texture resource for `air.get_null_texture_2d()`.
    /// Metal's `[[function_constant]]`-gated optional attachments resolve (with our FCs folded off) to
    /// a "null texture" the shader still threads through phis and may sample (yielding 0). We bind one
    /// real `OpTypeImage` 2D-float UniformConstant variable in a free texture-ABI binding and load it;
    /// sampling it is valid (reads as the unbound-descriptor default). Memoized across all uses so the
    /// phi-merged values share one image id and type.
    pub(in crate::passes) fn default_null_image_of(
        &mut self,
        dim: Dim,
        arrayed: bool,
    ) -> Result<Word, String> {
        if let Some(&v) = self.default_null_image_vars.get(&(dim, arrayed)) {
            return Ok(v);
        }
        let binding = allocate_default_texture_binding(&self.module)
            .ok_or_else(|| "no free texture binding for synthesized null image".to_string())?;
        let img_ty = self.ty_image(dim, arrayed, ImageComp::Float);
        let pptr = self.ty_ptr(StorageClass::UniformConstant, img_ty);
        let var = self.module.fresh_id();
        self.new_globals.push(Instruction::new(
            Op::Variable,
            Some(pptr),
            Some(var),
            vec![Operand::StorageClass(StorageClass::UniformConstant)],
        ));
        decorate_binding(&mut self.module, var, binding);
        self.interface_buffer_var(var);
        self.default_null_image_vars.insert((dim, arrayed), var);
        Ok(var)
    }

    /// Lazily create the descriptor-backed plane for one implicit imageblock attachment and AIR
    /// data rate. Array layer is the intrinsic's explicit color/sample index; keeping that operand
    /// prevents indexed imageblock reads from aliasing layer zero.
    pub(in crate::passes) fn implicit_imageblock_var(
        &mut self,
        attachment: u32,
        data_rate: u32,
        format: ImageFormat,
        comp: ImageComp,
    ) -> Result<(Word, Word), String> {
        if data_rate > 2 {
            return Err(format!(
                "implicit imageblock attachment {attachment} has unknown data rate {data_rate}"
            ));
        }
        if let Some(&(var, image_ty, existing_format)) =
            self.implicit_imageblock_vars.get(&(attachment, data_rate))
        {
            if existing_format != format {
                return Err(format!(
                    "implicit imageblock attachment {attachment} rate {data_rate} is used with conflicting formats {existing_format:?} and {format:?}"
                ));
            }
            return Ok((var, image_ty));
        }
        let binding = crate::reflect::imageblock_resource_binding(attachment, data_rate)
            .ok_or_else(|| {
                format!(
                    "implicit imageblock attachment {attachment} rate {data_rate} exceeds the descriptor ABI band"
                )
            })?;
        let image_ty = self.ty_storage_image(Dim::Dim2D, true, format, comp);
        let pointer_ty = self.ty_ptr(StorageClass::UniformConstant, image_ty);
        let var = self.module.fresh_id();
        self.new_globals.push(Instruction::new(
            Op::Variable,
            Some(pointer_ty),
            Some(var),
            vec![Operand::StorageClass(StorageClass::UniformConstant)],
        ));
        decorate_binding(&mut self.module, var, binding);
        self.interface_buffer_var(var);
        self.implicit_imageblock_vars
            .insert((attachment, data_rate), (var, image_ty, format));
        Ok((var, image_ty))
    }

    /// Lazily create one custom fragment-imageblock master plane with the storage format dictated
    /// by the AIR field type. Each supported type has an exact Vulkan storage-image representation;
    /// unknown layouts fail visibly instead of being widened or reinterpreted.
    pub(in crate::passes) fn fragment_imageblock_var(
        &mut self,
        master_member: u32,
        type_name: &str,
    ) -> Result<(Word, Word), String> {
        let format = super::super::fragment_imageblock_format(type_name).ok_or_else(|| {
            format!(
                "fragment imageblock master member {master_member} has unsupported type {type_name}"
            )
        })?;
        if let Some(binding) = self.fragment_imageblock_vars.get(&master_member) {
            return Ok(*binding);
        }
        let capability = spirv::Capability::StorageImageExtendedFormats;
        if !self
            .module
            .capabilities
            .iter()
            .any(|instruction| instruction.operands.as_slice() == [Operand::Capability(capability)])
        {
            self.module.capabilities.push(Instruction::new(
                Op::Capability,
                None,
                None,
                vec![Operand::Capability(capability)],
            ));
        }
        let image_ty =
            self.ty_storage_image(Dim::Dim2D, false, format.image_format, format.component);
        let pointer_ty = self.ty_ptr(StorageClass::UniformConstant, image_ty);
        let var = self.module.fresh_id();
        self.new_globals.push(Instruction::new(
            Op::Variable,
            Some(pointer_ty),
            Some(var),
            vec![Operand::StorageClass(StorageClass::UniformConstant)],
        ));
        let binding = crate::reflect::fragment_imageblock_resource_binding(master_member)
            .ok_or_else(|| {
                format!(
                    "fragment imageblock master member {master_member} exceeds the descriptor ABI band"
                )
            })?;
        decorate_binding(&mut self.module, var, binding);
        self.interface_buffer_var(var);
        self.fragment_imageblock_vars
            .insert(master_member, (var, image_ty));
        Ok((var, image_ty))
    }

    /// A constant/undef of `ty` for unused params (OpUndef is always legal).
    pub(super) fn const_zero(&mut self, ty: Word, _defs: &HashMap<Word, Instruction>) -> Word {
        let id = self.module.fresh_id();
        self.new_globals
            .push(Instruction::new(Op::Undef, Some(ty), Some(id), vec![]));
        id
    }

    /// An OpConstantNull of any type — the canonical zero value (handles aggregates, vectors, scalars).
    fn const_null(&mut self, ty: Word) -> Word {
        let id = self.module.fresh_id();
        self.new_globals.push(Instruction::new(
            Op::ConstantNull,
            Some(ty),
            Some(id),
            vec![],
        ));
        id
    }

    /// Materialize a Private zero-initialized variable of `pointee` for an unmodeled pointer param, and
    /// list it on the entry interface (1.4+). Returns the variable id to splice for the param. The
    /// caller (apply_bindings) re-storage-classes the derived access chains to Private.
    pub(super) fn zero_private_var(&mut self, pointee: Word) -> Word {
        // An absent raw buffer's pointee is a `{ RuntimeArray<uint> }` Block. A RuntimeArray is illegal
        // in Private storage (and in an OpConstantNull), so swap it for a fixed-size array in a fresh,
        // undecorated struct. Access chains off the var only reference the var id and yield element
        // pointers, so the substitution is transparent to them; reads are dead anyway (the resource is
        // function-constant-gated off), and any dynamic index is unbounded-but-valid SPIR-V.
        let pointee = self.private_safe_type(pointee);
        let init = self.const_null(pointee);
        let pptr = self.ty_ptr(StorageClass::Private, pointee);
        let var = self.module.fresh_id();
        self.new_globals.push(Instruction::new(
            Op::Variable,
            Some(pptr),
            Some(var),
            vec![
                Operand::StorageClass(StorageClass::Private),
                Operand::IdRef(init),
            ],
        ));
        self.interface_buffer_var(var);
        var
    }

    /// Return a type safe to use as a Private variable + OpConstantNull: a `RuntimeArray<E>` becomes a
    /// fixed-size `Array<E, N>`, and a struct with any RuntimeArray member becomes a fresh undecorated
    /// struct with those members fixed. Other types pass through unchanged. (Vulkan forbids
    /// RuntimeArray outside StorageBuffer/Workgroup, and OpConstantNull of a RuntimeArray-bearing type.)
    fn private_safe_type(&mut self, ty: Word) -> Word {
        const ABSENT_BUFFER_ARRAY_LEN: u32 = 1024;
        let def = self
            .module
            .types_global_values
            .iter()
            .chain(self.new_globals.iter())
            .find(|inst| inst.result_id == Some(ty))
            .cloned();
        let Some(def) = def else { return ty };
        match def.class.opcode {
            Op::TypeRuntimeArray => match def.operands.first() {
                Some(Operand::IdRef(elem)) => self.ty_array(*elem, ABSENT_BUFFER_ARRAY_LEN),
                _ => ty,
            },
            Op::TypeStruct => {
                let members: Vec<Word> = def
                    .operands
                    .iter()
                    .filter_map(|o| match o {
                        Operand::IdRef(m) => Some(*m),
                        _ => None,
                    })
                    .collect();
                let fixed: Vec<Word> = members.iter().map(|m| self.private_safe_type(*m)).collect();
                if fixed == members {
                    return ty;
                }
                let st = self.module.fresh_id();
                self.new_globals.push(crate::passes::type_inst(
                    Op::TypeStruct,
                    st,
                    fixed.into_iter().map(Operand::IdRef).collect(),
                ));
                st
            }
            _ => ty,
        }
    }
}

/// Decorate a struct type as a Block + member Offset decorations computed from member sizes/aligns
/// (MSL/std140 vec rules). Only handles scalar/vector/array-of-bytes members, which is what AIR
/// fragment-arg structs use.
pub(in crate::passes) fn decorate_block_struct(
    ctx: &mut Ctx,
    struct_ty: Word,
    defs: &HashMap<Word, Instruction>,
) {
    ctx.module.annotations.push(Instruction::new(
        Op::Decorate,
        None,
        None,
        vec![
            Operand::IdRef(struct_ty),
            Operand::Decoration(Decoration::Block),
        ],
    ));
    // Lay out the struct (and everything reachable inside it) exactly once.
    decorate_layout_recursive(ctx, struct_ty, defs);
}

/// Recursively add ArrayStride decorations to array members and Offset decorations to nested structs
/// reachable inside a Block-decorated struct (Vulkan requires every composite in the block to be
/// explicitly laid out). Idempotent enough for our shaders: a given array/struct type is decorated
/// once even if shared, but duplicate identical decorations are harmless to the validator.
fn decorate_layout_recursive(ctx: &mut Ctx, ty: Word, defs: &HashMap<Word, Instruction>) {
    if !ctx.laid_out.insert(ty) {
        return; // already laid out (shared type) — decorating twice is invalid.
    }
    let Some(def) = defs.get(&ty).cloned() else {
        return;
    };
    match def.class.opcode {
        Op::TypeArray | Op::TypeRuntimeArray => {
            let elem = match def.operands.first() {
                Some(Operand::IdRef(e)) => *e,
                _ => return,
            };
            let (es, ea) = layout_ty_size_align(ctx, elem, defs);
            let stride = round_up(es, ea);
            ctx.module.annotations.push(Instruction::new(
                Op::Decorate,
                None,
                None,
                vec![
                    Operand::IdRef(ty),
                    Operand::Decoration(Decoration::ArrayStride),
                    Operand::LiteralBit32(stride),
                ],
            ));
            decorate_layout_recursive(ctx, elem, defs);
        }
        Op::TypeStruct => {
            let mut off = 0u32;
            let explicit_offsets = ctx.air_struct_offsets.get(&ty).cloned();
            for (mi, op) in def.operands.clone().iter().enumerate() {
                let Operand::IdRef(mty) = op else { continue };
                let (s, a) = layout_ty_size_align(ctx, *mty, defs);
                off = explicit_offsets
                    .as_ref()
                    .and_then(|offsets| offsets.get(mi).copied())
                    .unwrap_or_else(|| round_up(off, a));
                ctx.module.annotations.push(Instruction::new(
                    Op::MemberDecorate,
                    None,
                    None,
                    vec![
                        Operand::IdRef(ty),
                        Operand::LiteralBit32(mi as u32),
                        Operand::Decoration(Decoration::Offset),
                        Operand::LiteralBit32(off),
                    ],
                ));
                decorate_layout_recursive(ctx, *mty, defs);
                // Advance by the member's allocation size, not its store size. LLVM's struct
                // layout consumes `alignTo(storeSize, abiAlign)` per member, so a three-lane
                // vector (store 3/6/12 bytes under an AIR `v24:32:32` / `v48:64:64` /
                // `v96:128:128` rule) pushes the next member to its four-lane boundary.
                off += round_up(s, a);
            }
        }
        _ => {}
    }
}

pub(in crate::passes) fn layout_ty_size_align(
    ctx: &Ctx,
    ty: Word,
    defs: &HashMap<Word, Instruction>,
) -> (u32, u32) {
    crate::layout::spirv_size_align(
        ty,
        defs,
        crate::layout::SpirvLayout::air_offsets(
            &ctx.air_struct_offsets,
            ctx.air_data_layout.as_ref(),
        ),
    )
}

pub(in crate::passes) use crate::layout::round_up_u32 as round_up;
