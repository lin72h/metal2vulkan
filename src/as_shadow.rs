//! Host-populated metadata view for AIR acceleration-structure introspection.
//!
//! Vulkan acceleration-structure descriptors are opaque: shaders cannot ask one for its instance
//! count or enumerate child references. AIR exposes both operations, so the host binds this narrow
//! StorageBuffer shadow at the Metal acceleration-structure resource location whenever a kernel uses
//! those intrinsics. Child references are the serialized 64-bit pointer payloads used by the rest of
//! the AIR BVH ABI.

/// Fixed header preceding the child-reference array.
#[repr(C)]
pub struct AccelerationStructureShadowHeader {
    pub instance_count: u32,
    pub reserved: u32,
}

pub const INSTANCE_COUNT_BYTE_OFFSET: u64 =
    std::mem::offset_of!(AccelerationStructureShadowHeader, instance_count) as u64;
pub const CHILD_REFERENCES_BYTE_OFFSET: u64 =
    std::mem::size_of::<AccelerationStructureShadowHeader>() as u64;
pub const CHILD_REFERENCE_BYTE_STRIDE: u64 = std::mem::size_of::<u64>() as u64;
