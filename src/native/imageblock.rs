//! AIR imageblock execution-model constants.

/// Maximum number of coordinate-addressable imageblock cells in the current compute contract.
///
/// This matches the translator's existing 512-element threadgroup-memory interface contract and
/// covers every supported `kernel_local_size` shape. Cross-coordinate imageblock lowering allocates
/// one metadata-typed cell per slot and uses the decoded `threads_per_threadgroup.x` as row stride.
pub(super) const CELL_CAPACITY: u32 = 512;
