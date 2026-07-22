use crate::spirv_module::Block;
use crate::spirv_module::Operand;
use spirv::Word;

pub(in crate::native) fn block_index_by_label(blocks: &[Block], label_id: Word) -> Option<usize> {
    blocks
        .iter()
        .position(|block| block.label.as_ref().and_then(|label| label.result_id) == Some(label_id))
}

pub(in crate::native) fn id_ref_operand(operand: &Operand) -> Option<Word> {
    let Operand::IdRef(id) = operand else {
        return None;
    };
    Some(*id)
}
