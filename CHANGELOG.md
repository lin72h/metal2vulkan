# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Fixed

- A shader that encodes into an indirect command buffer is refused instead of translated into one
  that does not. The `air.*_command` encoder families -- `set_pipeline_state_compute_command`,
  `set_kernel_buffer_compute_command`, `draw_primitives_render_command` and their siblings -- were
  lowered to nothing, on the reasoning that the conformance runner observes buffer bytes only. What
  that produced was a module that validates, binds and reflects exactly like the original while
  performing none of the encoding, which for a kernel whose body is entirely ICB encoding is the
  whole shader. Vulkan's device-generated-command extensions describe a different, driver-defined
  layout that no sequence of SPIR-V instructions can produce from these operands, so the honest
  answer is a `FALLBACK` naming the family. Twelve of 14579 local corpus sources reach it (two of
  the 2880-source A/B sample), all of which previously reported success.

  `AirIntrinsicDisposition::NoVulkanEquivalent` now distinguishes a family the translator has
  modelled and refuses from one nothing has modelled yet, so the refusal names the family rather
  than reporting it as unrecognised. Validation's `IndirectCommandBuffer` tooling requirement is
  derived from the same `is_command_encoder_helper` definition: it previously keyed on
  `!"air.indirect_command_buffer"`, a metadata string no harvested AIR module carries, so it never
  fired for a real encoder. It now fires on an encoder call or an `air.command_buffer` argument.

- One encoder produces every `half` constant the translator mints. The native emitter and the
  passes layer each carried their own `f32` -> binary16 conversion, and they disagreed: the passes
  copy rounded half-away-from-zero rather than to even, and combined the rounded significand into
  the encoding with `|` instead of `+`, so a value whose rounding carries out of the significand
  kept the exponent it started with whenever that exponent was odd. `1.999755859375` encoded as
  `1.0` instead of `2.0`.

  This reached emitted modules through the saturating bounds of `air.convert` from a `half` source
  to a narrow integer: the `32767.0` bound of a 16-bit signed convert encoded as `16384.0`, and the
  `8191.0` bound of a 14-bit one as `4096.0`, each halving the range the conversion could produce.
  A `half` above the halved bound saturated to it instead of to the destination's maximum. Across
  2880 corpus sources six modules change, each replacing the upper `FClamp` bound of a `half` ->
  `short` conversion: `OpConstant %half 0x1p+14` becomes `0x1p+15`, against a lower bound of
  `-0x1p+15` that was already right. (`32767` is not representable as a `half`; `0x1p+15` is the
  nearest one, which is what a correctly rounded bound means. `0x1p+14` was not near it.)

  The correct encoder now lives in `crate::float16` and is exercised at every rounding boundary in
  the format: for each pair of adjacent finite halves, the exact midpoint must land on the even one
  and the two neighbouring `f32` values must land on their own side.

- Every attribute an AIR stage root carries is read, and one the translator has no model for is
  refused instead of dropped. All three roots -- `!air.kernel`, `!air.vertex`, `!air.fragment` --
  state their per-entry attributes as extra operands past `(function, outputs, inputs)`, but not in
  one form: `air.patch` and `air.max_work_group_size` are references to a keyed node, while
  `early_fragment_tests` is a bare string on the root itself. Only the vertex decode looked at that
  tail, and only for `air.patch`, so:
  - `[[early_fragment_tests]]` reaches SPIR-V as `OpExecutionMode ... EarlyFragmentTests`. It was
    dropped on 51 of 14579 local corpus sources, which emitted them with Vulkan's default late
    depth test. Under early tests a fragment the depth test rejects performs none of the body's stores; under
    late tests the same shader performs every buffer, texture and imageblock write and only its
    color output is discarded. A fragment that both declares it and writes `air.depth` or
    `air.stencil` is refused: the test runs before the value it compares exists.
  - `[[max_total_threads_per_threadgroup(N)]]` (`air.max_work_group_size`, 439 of the same sources)
    bounds the requested kernel `LocalSize`. A dispatch wider than the ceiling the entry was
    compiled for is refused rather than emitted as a module that validates and runs a shape the
    source ruled out.

  A/B over a 2880-source sample: no status changes, and the only SPIR-V byte changes are the ten
  fragments that declare `early_fragment_tests`, each gaining exactly the one execution mode and
  still passing `spirv-val`. No source in that sample requests a dispatch past its ceiling.

- `reflect_sanitized` decides whether a kernel needs a buffer-address table with the emitter's own
  device-address predicate rather than a line-prefix scan of the AIR text. The scan disagreed with
  the finished module on 63 of 2880 corpus sources -- 49 by reporting a binding no module declares,
  and 13 by reporting none for a module that does, which leaves a consumer building a descriptor-set
  layout without a binding the shader reads. Asking the predicate the emitter itself asks leaves 8,
  four in each direction, and metadata-only reflection remains about three times cheaper than
  translating. Reflected translation, which reads the table off the module, is unchanged.
  `REFLECTION_VERSION` is now 38.
- `with_runtime_sampler` accepts a pixel-coordinate sampler whose LOD maximum is Metal's default.
  The emulation fetches level zero, so only a *minimum* above zero can exclude the level it reads --
  a maximum never can, since validation already requires it to be at least the minimum. Demanding
  both be zero refused the ordinary case: 531 of the 535 pixel-coordinate static samplers across
  2880 corpus sources carry a maximum of 65504, the half-precision limit AIR encodes "unclamped" as.
  The emulation constraints now live on `StaticSamplerState`, the type the lowering consumes, and
  the AIR constexpr-sampler path applies them too -- it previously applied none, so a static sampler
  the emulation cannot reproduce was lowered anyway.
- A vertex function carrying an `air.patch` node the decoder cannot read is refused instead of being
  emitted as an ordinary vertex shader. The node states a post-tessellation evaluation shader's
  domain and control-point count; dropping it dropped the `Quads`/`Triangles`/`Isolines`,
  `SpacingEqual` and `VertexOrderCcw` execution modes Vulkan requires of that stage along with every
  per-patch input the pipeline wires, and the module that came out validated, bound and reflected
  while drawing the wrong geometry. The node is also located by the `air.patch` marker it carries
  rather than by its position in the vertex root.
- An entry buffer whose declared AIR layout could not be attached to the emitted parameter now says
  which of the two things happened. A byte-addressed buffer is emitted as its raw contents -- a
  pointer to a word, or the block a storage buffer requires wrapping a runtime array of one -- and
  has no members for the declared offsets to land on, which is not the same as two structural
  descriptions disagreeing. Both reported `EmittedShapeMismatch`; over 2880 corpus sources 1733 of
  the 1805 unmapped parameters are the former, and reporting them as mismatches buried the 72 where
  the shapes really do differ.
- A buffer member `air.struct_type_info` names without describing -- a user struct or class
  mentioned by name only -- decodes as `AirType::Opaque { size }` at the byte size the member tuple
  declares, instead of as a 32-bit `Float`. The float leaf named a type AIR never stated and sized
  every such member at four bytes: over 2880 corpus sources, 2513 members across 817 modules are
  opaque, and 2357 of those declare a size other than four. The invented interior also failed to
  match the member the emitter produced, which discarded the declared offsets for the whole buffer
  rather than for the one member that provoked it. `REFLECTION_VERSION` is now 37.
- `validate_descriptor_abi` rejects a reported member layout that reaches past the argument holding
  it. AIR states an argument's size and its member layout as two independent facts and reflection
  reconstructs them independently, so a layout that over-runs means some member's storage was
  mistaken and every member after it names an offset the shader never reads. Nothing else catches
  it: a consumer packs its upload at those offsets, and a buffer whose reconstruction is that far
  off is emitted as raw bytes with no struct type to compare against. Reaching short stays valid,
  since the declared size is a `sizeof` and carries tail padding -- 1340 of the corpus's reported
  layouts legitimately stop short, and none over-run.
- A texture AIR declares write-capable binds as a storage image even when the shader body only
  queries its size. The binding class came from what the body did, and a size query counted as a
  sampled-image use, so such a texture bound in the sampled-texture band while reflection reported
  it in the storage-texture band -- a consumer wrote its descriptor where the shader does not read
  it. Only sampling decides the class now; every other AIR texture operation has a form for either.
- Scratch files under a caller-supplied `tmp` are named per process and per call, so two callers
  sharing one directory no longer overwrite and delete each other's input to `spirv-val` or
  `llvm-dis`. The shared name surfaced as a validation failure in a module that was fine.
- `get_width()`/`get_height()` on a `texture_buffer` translates instead of falling back. SPIR-V
  allows `OpImageQuerySizeLod` only on a 1D/2D/3D/Cube image with `MS` 0 and a `Sampled` operand
  that is not 2; a `Dim Buffer` image must use the LOD-less `OpImageQuerySize`. One
  `image_size_query_op` now states that rule for both size-query lowerings, which had drifted --
  only one of them handled multisample images and neither handled buffer textures.
- A descriptor the translator synthesizes only to type an AIR value is retracted whenever nothing
  consumes that value. This already covered `air.get_read_sampler()`; it now covers
  `air.get_null_texture_*()` too, so a shader that merely asks whether an optional attachment is
  bound no longer demands a texture descriptor it never reads. The rule is stated once, over the
  set of synthesized placeholders, rather than per variable.

### Added

- Reflection schema v36: an argument-buffer member that holds a resource handle is reported at the
  eight bytes it occupies instead of as the type it points at. Metal spells such a member with its
  pointee's name -- `char` for a `device char *`, `float4x3` for a `device float4x3 *`, a
  `texture2d<...>` for a texture -- and reading that name as the member's storage put a 64-byte
  matrix where the buffer holds an address, shifting the meaning of every member after it.
  `MTLGenericBVHData` reported a 120-byte layout for its 72-byte argument; over 2880 corpus sources
  1736 members across 9 named types and 47 opaque ones were mis-sized this way. The member's
  `air.indirect_argument` node decides: every role but `air.indirect_constant` is a reference. The
  emitted SPIR-V follows for the 21 modules whose buffer was typed from this layout, which had
  declared four-byte floats at eight-byte member offsets.
- Reflection schema v35: a descriptor-backed buffer's `access` is widened to cover the loads and
  stores the finished module performs through it. AIR's declared access is not a guarantee about the
  body -- over 2880 corpus sources, 20 buffers reflected `ReadOnly` are stored through, 9 reflected
  `WriteOnly` are loaded from, and 3 reflected `Unused` are both -- and a consumer barriers and
  stages from this field. It only ever widens, since the analysis can miss an access through a
  device address it cannot attribute. `reflect_sanitized` builds no module and keeps the declared
  classification.
- Reflection schema v34: `texture_shape` describes the `OpTypeImage` the module declares at that
  binding, not the shape the AIR type name implies. A `texturecube` that is only texel-read binds as
  a `Dim2D` array, because SPIR-V has no cube texel fetch, and a consumer that created a cube view
  from the old shape would have built a view Vulkan rejects against that image variable. A binding
  whose image variables do not all declare the same type keeps the type-name-derived shape, as does
  `reflect_sanitized`, which builds no module.
- Reflection schema v33: reflected translation reports the descriptors the passes synthesize with no
  Metal argument behind them -- `SynthesizedNullTexture` for a read `air.get_null_texture_*()`
  placeholder and `SynthesizedReadSampler` for a consumed `air.get_read_sampler()` one. Nothing in
  the AIR metadata describes those bindings, so a descriptor-set layout built from reflection alone
  did not cover the module. The list comes from the passes and is filtered to what the finished
  module still declares; the module supplies each resource's class.
- Reflection schema v32: `ShaderReflection::bindings` reports each resource exactly once. A fragment
  entry with an `air.indirect_buffer` argument previously listed every argument-buffer resident
  twice, so a consumer sizing its work from that list allocated and wrote each of those descriptors
  twice. `validate_descriptor_abi`, which both reflection paths run, now rejects two entries that
  agree in every field.
- Reflection schema v31: implicit imageblock render-target planes are reported at every stage whose
  module calls the `air.load/store.implicit_imageblock.*` intrinsics, not only in compute. The
  interface pass materializes the descriptor from the call, so a fragment entry that reads its
  render target back through the imageblock now reflects the binding it declares. Reflected
  translation also reports the buffer-address table the finished module declares, rather than the
  one predicted from an AIR text scan; `reflect_sanitized`, which builds no module, keeps the
  prediction and documents it as an approximation.
- Bounded translation censuses now consume exact authored visible-function-table populations inside
  the isolated worker, using hash-local case lookup and the same checked linkage mapping as Vulkan
  candidate execution; only rows with no authored population remain linkage-required.
- Public `specialize_function_constants_zero` helper for baking discovered Metal function
  constants to their zero/default values, including branch pruning and removal of now-dead entry
  interface globals.
- Public byte-exact function-constant specialization, reflection-only AIR inspection, authored
  linked-function/table specialization, and vertex-observer generation APIs.
- AIR-level scalar/vector function-constant translation APIs that specialize stable
  `air.fc_initializer` globals before metadata, resource-interface, and CFG construction.
- Reflection schema v30: consumer metadata now covers decoded static samplers, texture shape and
  access, argument-buffer resources, kernel stage inputs, tessellation, imageblocks, exact function-
  constant ABI types, buffer extent/access classification, and conservative final-module static and
  invocation-strided buffer footprints, plus runtime sampler/storage-image specialization and the
  effective descriptor layout and kernel dispatch-grid ABI.
- Versioned, caller-selected descriptor layouts through `TransformOptions`, allowing independently
  translated stages to use distinct Vulkan descriptor sets and binding ranges while preserving the
  existing layout as the explicit default.
- Runtime pipeline-state specialization by Metal resource index for dynamically bound samplers and
  writable storage images, including embedded argument-buffer textures, exact reflected sampler and
  image-format state (including two-channel `Rg32Float`), and host storage-image feature checks.
- A typed exact-thread dispatch ABI that decomposes Metal boundary workgroups into at most eight
  Vulkan regions, with specialized local sizes, logical grid bases, public planning helpers, and
  fixed, dynamic, and explicitly proven whole-workgroup forms.
- A task-oriented translation and reflection integration guide plus a compiled serde reflection
  example.
- Additional stage-interface support for fragment `[[point_coord]]`, `[[primitive_id]]`,
  `[[sample_id]]`, and `[[render_target_array_index]]`, flat varyings, framebuffer-fetch color
  inputs, vertex builtins, and fragment outputs with nonzero render-target locations.
- Broader native translation support for texture arrays, storage-image arrays, texture
  gather/sample/read/write variants, half/integer render-target formats, scalar 64-bit integer
  arithmetic, and Workgroup memory patterns used by shared-memory reductions.
- Native lowering and reflection for linked functions, tessellation patch inputs, ray/intersection
  queries and result fields, argument-buffer resources, and implicit, custom, and direct-layout
  imageblocks.
- Native distributed `simdgroup_matrix` 16x16x16 multiply-accumulate lowering for the observed
  f32, f16, bf16, float8, and signed/unsigned i8 AIR element combinations, including dynamic
  transpose operands and 32-lane tile ownership.
- Authored validation contracts and executable Metal/Vulkan cases for tessellation, depth/stencil
  attachments, framebuffer fetch, multisample and buffer textures, narrow vertex attributes,
  vertex side effects, function constants, argument buffers, imageblocks, ray intersections, and
  exact empty observations for entries with no reflected writable output, including rejection of
  vertex positions and varyings as observable outputs.
- Byte-exact Metal/MoltenVK evidence now covers every public synthetic AIR fixture, including vector
  function-constant lanes, barrier-bearing boundary workgroups, narrow vertex attributes, combined
  depth/stencil, and custom and implicit imageblocks, plus indexed local identities for
  unconditional word stores, signed min/max initialization, grid-indexed byte clears, and exact
  grid-index sequence generation, parameter-free scalar and vector fragment returns (including a
  sparse color-location-2 target and scalar half targets), and constant-zero vertex positions
  observed through their exact non-rasterizing framebuffer result, plus two-channel float fragment
  targets, fragment position-depth extraction, and a pixel-coordinate sampled vertical convolution
  with explicit sampler state, weights, bounds, and bias. Constant scalar/vector fragment evidence
  now also covers half, float, and uint targets, while dual-output fragment clears qualify both
  attachments independently for half4 and float4 formats. Authored Vulkan draws now mirror Metal's
  default clockwise front-face winding, with byte-exact coverage for `[[front_facing]]`, primitive
  IDs, generated float/half fragment varyings across additional independently indexed AIR
  identities, scalar float-to-half varying conversion, and constant outputs whose unused stage
  inputs include viewport indices. Additional authored vertex evidence observes generated
  ushort-ID positions and independently captures passthrough position and float2 varying outputs;
  fragment evidence also covers flat uint varyings, scalar depth expansion, and independently
  selected sparse color attachments. Exact vertex observation now additionally covers forced-zero
  clip-space depth, paired position/float2 passthrough outputs, and deterministic position results
  beside explicitly undefined return members; fragment coverage includes alpha-to-half4 conversion,
  aggregate flat-uint returns, and viewport-bearing varying passthrough.
  The indexed evidence set additionally covers float2/float3 clip-position expansion, paired
  float2 vertex varyings, scalar and vector float-to-half fragment conversion, flat-boolean color
  selection, fog-alpha multiplication, empty texture kernels that preserve their selected bytes,
  and a rasterization-disabled vertex side-effect store.
  A complete indexed texture-copy family now has byte-exact coverage for flat and array-layer
  half-texture reads, including explicit function-constant specialization of the flat alternative.
  The complete indexed `backgroundFragment` family now has fresh byte-exact Metal/MoltenVK evidence
  for constant and sampled outputs, including two function-constant-selected gradient textures.
  The complete eight-identity `Clear::clear_fragment{,2,3,4}` suite now has byte-exact evidence for
  constant-buffer float-to-half conversion and aggregate output mapping through four render targets.
  Attachmentless fragment draws now execute on Metal with explicit render-pass dimensions, without
  synthesizing an observable attachment. The two-identity `Clear::clear_depth_stencil_fragment`
  family has exact empty-output Metal/MoltenVK evidence for its structural void-return contract.
  The complete four-identity `Clear::clear_vertex{,_mrt}` family has exact Metal/MoltenVK position
  observations for float2 vertex attributes expanded with buffer-supplied clip-space depth.
  The complete six-identity `FullscreenFragment{DepthStencil,Overlay,Texture}` suite has exact
  Metal/MoltenVK evidence for static-sampler half-texture reads, including red-lane replication,
  uniform-color multiplication, and RGB widening to RGBA32Float outputs.
  The complete six-identity `ARMesh::{mesh_depth_fragment,ar_mesh_shadow_fragment,ar_mesh_fragment}`
  suite has exact Metal/MoltenVK evidence for literal depth, edge-weighted shadow colors, and the
  full 2D/cube sampled-texture lighting path.
  A complete indexed tracking-area family now covers layered textures, fixed texture arrays, and
  function-constant-selected alternatives through eight-SIMD-group shared-memory reductions with
  barrier-synchronized exact uint output. Complete indexed vertex families now also cover
  vertex-ID-generated fullscreen positions across plain, viewport-indexed,
  render-target-layer-indexed, and single-view amplification interfaces.
- A sharded validation workflow with dependency-exact observations, an incremental SQLite source
  index, focused hash/shard selection, explicit full reclassification, capability audits, native
  Metal and Vulkan/MoltenVK A/B execution, and optional OpenRouter-authored case proposals.

### Changed

- Reducible control flow that the structured planner rejects is now nested into real SPIR-V loop
  and selection constructs before the bounded state-machine relooper is considered. Ordinary values
  stay in registers and demoted phis are promoted back at the block where their edges meet, so a
  function no longer arrives as one dispatch loop whose every crossing value is a function-scope
  variable. The nesting is adopted only when the emitted function satisfies the same construct,
  structured-exit, dominance, and phi contract the owned module is held to; anything else stays on
  the state machine.
- Vulkan 1.2 is now the mandatory translation and validation baseline; newer Vulkan features may
  only be exposed as optional performance paths with faithful Vulkan 1.2 fallbacks.
- Structured CFG plans now finalize loop-continue selection ownership before their completeness,
  ordering, and ownership checks; typed emission no longer runs detached continue, selection-arm,
  bypass, or reused-merge normalization after plan admission.
- Nested loop planning now materializes inner multi-exit dispatches before recomputing enclosing
  loop ownership, so newly synthesized exits are validly owned without a construct-tree retry.
- A nested natural loop that exits into its enclosing selection's sibling is now owned by a typed
  regional dispatcher before emission. Bounded scalar-only CFG rejects can use the same ownership
  contract across the whole function, while pointer state still declines rather than being guessed;
  terminal switches retain real dominated reconvergences and conflicting phi ownership rejects
  before instruction emission.
- Loop-local switches that target an enclosing loop role now lower to branch ladders before plan
  admission, preventing source-dominance false positives from emitting invalid case constructs.
- Imageblock scratch layout inference now follows reachable internal calls, preserving complete
  cells when a helper byte-addresses a nonzero field instead of relying on an inlining retry.
- Integer-width phi legalization now happens while pointer-phi transformations construct their
  replacement index phis; retained modules are no longer rescanned or repaired after emission or
  scalar-i64 lowering.
- Vulkan validation uploads every sampled-only literal through a staging buffer into an
  optimal-tiled image, so cube and other sampled shapes do not depend on optional linear-image
  support for their exact create flags.
- Authored output qualification now requires the shared checker to map the selection to a reflected
  shader write, then accepts deterministic byte-identical transformations instead of requiring
  every byte to differ from its initial value.
- The Metal validation executor now derives render-pipeline input topology from each authored draw
  and constructs matching array attachments and active-layer render passes for layered rendering.
- Generated Vulkan fragment companions preserve their authored layer-zero contract through the
  core first-layer rule, without requiring the optional Vulkan 1.2 `shaderOutputLayer` feature.
- Authored Vulkan execution now applies function constants before AIR translation, so nondefault
  values faithfully retain selected resources and CFG arms instead of trying to restore structure
  after default-valued SPIR-V emission.
- Exact Metal `dispatchThreads` kernels no longer round up and cull surplus lanes. The default
  contract now preserves partial-workgroup barriers by dispatching true boundary workgroup sizes;
  consumers must follow the reflected region plan and 48-byte dispatch payload.

- Native emitter wrapper APIs under `tools` now use `emit_vulkan_spirv*` names that match their
  implementation.
- The CLI accepts `--raster-samples` for AIR sample-count queries and derives a default `.vk.spv`
  output path when the output argument is omitted.
- Floating-point lowering is closer to AIR for the covered cases, including f32-to-f16 clamping,
  bf16 narrowing and NaN handling, fast `sin`/`cos`, `pow` zero edges, and exact `mix` endpoints.
- Buffer, pointer, control-flow, and access-chain lowering handles more structural cases, reducing
  fallbacks and invalid SPIR-V for shaders that use dynamic indices, pointer selects, aggregate
  copies, raw subword loads/stores, and local pointer tables.
- Opaque Metal buffers used through multiple scalar element types now retain each typed view as a
  descriptor alias at the same binding, while genuine array element-zero accesses gain their block
  descent and vector-stride pointers retain their vector pointee during interface construction;
  these preserve exact byte strides without late load/store pointer repairs.
- Exact byte-address provenance now composes through resource-wrapper collapse to the final buffer
  descriptor, so raw 32-bit vector loads are constructed from their word lanes before validation.
- Exact raw-word replay now derives synthesized access-chain pointer types from the concrete root's
  storage class, eliminating detached final access-chain storage-class repairs.
- Finalized typed CFG construction now recognizes a phi as an SSA identity only when its block has
  one actual predecessor and every incoming pair names that predecessor with the same structural
  value. All identities, including pointers, are substituted before SPIR-V IDs and representation
  sidecars are constructed. This eliminates the detached phi collapse and its access-chain and
  sampled-image type repairs.
- Buffer-interface discovery now follows transparent pointer aliases, includes pointer-arithmetic
  chains as typed element views, and gives mixed numeric scalar/vector views descriptor aliases at
  the same binding. Each logical pointer is therefore valid by construction instead of being
  retyped away from its byte loads during interface specialization.
- Null-derived access chains are neutralized once in the main access transform; redundant
  whole-module cleanup passes after native pointer rewrites have been removed.
- Exact raw-byte access is replayed at the descriptor-reconstruction boundary, eliminating the
  detached post-native replay and releasing source-only layout type graphs before final cleanup.
- Nondominating-value demotion now explicitly preserves the CFG successor contract, eliminating
  redundant whole-CFG repair passes after register spills.
- Pointer-phi legalization and mixed-storage value lowering now explicitly preserve CFG successors,
  so late structured-CFG repair runs only for an actual edge-producing loop split.
- Multi-entry loop splitting now owns its cloned region, header phis, redirected entry, and selection
  exit as one transaction, eliminating its detached whole-CFG repair pass.
- Multi-exit funnels, deep shared-arm refunnelling, and multi-entry loop splitting now preserve SSA
  dominance at their typed construction boundaries, eliminating the final emitted-module
  nondominating-value demotion pass.
- Construct-tree planning now assigns unreachable merge declarations for fully terminal selections
  alongside every other header, eliminating the late missing-terminal-header completion sweep.
- Dominated-region cloning now carries nested selection ownership through its structural rename map,
  so enclosing-route selections are materialized locally instead of dropped and rediscovered by a
  final missing-selection sweep.
- Construct-tree selection construction now preserves the local live-arm ownership of direct
  terminal guards through generic merge normalization, eliminating its repeated enclosing-escape
  and terminal-live repair sweeps.
- Ordinary selection construction now derives nested exits from the complete immutable source
  ownership map and materializes their private enclosing boundaries innermost-first, eliminating
  the late enclosing-region escape repair fixed point.
- Construct-tree selection construction now retains complete merge ownership through enclosing
  synthesis, eliminating its three post-construction nondominance, pass-through promotion, and
  bypass-refunneling repair stages from the product path.
- Construct-tree source ownership now includes nested terminal selections in its initial
  innermost-first header census, eliminating late post-synthesis terminal-convergence completion.
- Direct terminal ownership now consumes the complete proved linear tail before selection analysis,
  eliminating post-construction direct-guard refunneling from the product path.
- Selection construction now privatizes shared terminal returns when each merge owner is created,
  eliminating the later whole-owner terminal-return fixed-point sweeps.
- Innermost-first terminal-tail ownership now closes enclosing parents directly, removing the late
  parent/nested merge-composition pass from production.
- Switch merge construction now collapses proved terminal case tails when each switch owner is
  recorded, removing the late all-switch terminal finalization sweep from production.
- Loop-free terminal guards whose live convergence enters a later loop now defer a return shared
  with that loop's exit to the terminal owner, which privatizes both boundaries instead of admitting
  overlapping merge ownership.
- Loop-exit switch lowering now includes loop headers and multi-exit loops, producing a conditional
  ladder before planning so one block never needs both loop and selection merge ownership.
- Switch construction now privatizes intermediate continuations shared by a subset of cases even
  when the eventual switch merge is dominated, including loop-local suffixes whose clone does not
  cross a loop header, latch, exit, or nested-loop boundary.
- Construct-tree ownership now runs by default after an ordinary source-CFG planning rejection,
  while forward SSA allocation is scoped only to functions using that reordered plan. This removes
  a redundant emit, finish, validation, and source reparse from structurally owned large functions.
- Switch-tail ownership now preserves SPIR-V's legal fallthrough into the immediately following case,
  and shared-region cloning declines edges whose redirected predecessor crosses a natural-loop
  boundary, preventing non-adjacent case entries and non-dominating loop-exit values by construction.
- Enclosing selection owners now finish only their indexed dependent routes when the owner is
  recorded, removing the late whole-CFG enclosing-route materialization fixed point.
- Indexed translations retain natural emitted-loop ownership through the typed edge-producing
  transforms, allowing the detached stale-loop reclassification adapter to leave production.
- Translation audits now finish the 16-worker ordinary lane before starting the bounded costly-row
  lane, including sub-megabyte CFGs whose serialized AIR reaches 256 KiB. The costly lane is capped
  at two workers, with ≤384-KiB, 370-block/400-call CFGs isolated on one sublane while unrelated
  large and device-address/function-table rows continue on the other, so sustained concurrency does
  not consume the per-attempt 20-second budget. Cached outcomes now fingerprint the audit harness
  as well as the product translator.
- Translation audits can now resume a per-fingerprint retry-tier census in the disposable SQLite
  index, measure exact hash-file selections, and report the complete adoption histogram without
  reopening warm source shards.
- Resumable translation selection now materializes the historical-failure priority set once per
  uncursored batch instead of probing the complete audit history once for every indexed source.
- Function-constant-wrapped visible and intersection tables now retain their shared authored-linkage
  roles, and validation traces visible-table handles through internal helper parameters. A full
  `--reclassify-all` authoring census now reads every indexed source under the current analyzer ABI;
  the ordinary warm pass remains a zero-source-read incremental check. Translation workers classify
  these structurally proven authored-linkage dependencies before spawning the product/tool cascade.
- Direct AIR visible-function references now resolve automatically through an incremental
  same-library symbol index when the retained definition is unique, including transitive linked
  references; exact module byte locations keep warm and targeted lookups shard-local, while missing
  or ambiguous definitions remain explicit authored inputs. Translation audits can resumably replay
  historical linkage rows with `--retry-linkage`.
- Authored visible and intersection table slots now accept exact functions from separately
  harvested Metal libraries, matching the linked-functions API instead of imposing a false
  same-metallib provenance rule; exact module hashes, symbol definitions, and globally unique linked
  names remain mandatory.
- Partial zero-initialization of typed aggregates now lowers recursively into null stores for fully
  covered prefix subobjects, and final SPIR-V construction dependency-orders late synthesized
  module-scope types before existing users. Linked declaration/body pairs also select the bodied
  definition consistently during emitted-helper inlining, removing the associated empty-callee
  panic.
- Same-width scalar-array/vector reinterpret loads now rebuild the vector lane-by-lane with scalar
  bitcasts, preserving logical pointer types instead of requiring the all-buffer raw retry.
- Function-constant buffer alternatives that share one binding and mix scalar families through a
  recurrent pointer carrier now choose byte-addressed storage plus typed construct-tree planning
  during primary construction, instead of relying on the all-buffer raw and relooper retry cascade.
- Function-constant pruning now owns branch folding, reachability, phi, and loop-merge closure, which
  removes the final finish-time whole-module structured-CFG repair adapter.
- Primary construction now checks owned selection entry/exit, loop back-edge declarations,
  conditional merge declarations, and dominator serialization after function-constant CFG pruning,
  choosing bounded relooper form for only the affected functions before the first assembly while
  preserving unrelated functions and the hard downstream-driver state-machine cap. Relooper switch
  lowering now preserves both 32-bit and 64-bit selector literals.
- Function-constant pruning and entry-interface rebuilding now also own specialized Workgroup
  aggregate-stride access lowering instead of invoking a detached pointer repair afterward.
- The inline, SROA, and raw-access retry lowers dynamic typed accesses before constructing its
  relooper module, eliminating its post-relooper access-chain repair adapter.
- Helper inlining now completes address-preserving zero-offset aggregate descent once at its
  self-contained entry-closure boundary, eliminating the emitter's detached whole-module repair scan.
- Resource-select lowering assigns each duplicated `OpSampledImage` its branch image's exact type at
  construction, eliminating its final whole-module sampled-image type repair scan.
- Pointer-phi lowering now assigns synthesized incoming values directly to their predecessor edges,
  eliminating the final post-emission access-chain relocation and phi-order repair scan.
- Finalized typed CFG construction now redirects external entries around loop continue constructs,
  moves their exact phi values onto loop-header edges, and restores dominator serialization before
  emission, eliminating the corresponding numeric SPIR-V repair fixpoint.
- Emitted loop, selection, and switch merges that are reachable from outside their owning construct
  now get phi-aware private boundaries in the finalized typed CFG, eliminating the matching
  post-emission dominance repair.
- Instruction-local control flow is now materialized as real blocks inside the native emitter; when
  it splits a loop header, a dedicated source header retains the loop phis and ownership while any
  nested selection receives a private merge, eliminating the post-emission stale-loop downgrade.
- Finalized typed CFG construction now owns every loop header required by indexed inputs, eliminating
  both post-emission unmarked-loop synthesis and product-wide dominator-order normalization;
  structural loop tests define membership by dominated CFG predecessor edges rather than
  serialization order. Instruction-local control flow now carries each source block's real emitted
  exit into successor phis, eliminating final numeric-SPIR-V phi reconciliation and the product CFG
  repair module altogether.
- Selection boundaries that collide with a loop continue are now resolved in the finalized typed
  CFG, including enclosing selections, direct break/continue branches, and phi-carrying in-loop
  reconvergence; inner construct merges are likewise privatized in the typed plan, and emitted
  helper inlining now preserves the enclosing continue while giving its nested selection a private
  pass-through, eliminating both matching post-emission repairs.
- Structured emission now retains merge markers immediately before their terminators throughout
  lowering, eliminating the permissive pass that reordered already-malformed merge blocks.
- Direct-arm cloning and external loop-entry rewrites now preserve their newly established
  dominator order at the owning structural boundary.
- Loop plans now resolve emission-empty continue pass-through chains from the finalized typed CFG
  before emitting `OpLoopMerge`, eliminating the corresponding post-emission label repair.
- Finalized emission plans now give nested constructs phi-aware private merge targets while
  preserving loop-header backedges, eliminating the retained-SPIR-V shared-merge ownership pass.
- Finalized typed CFG emission now funnels selection/switch bypass edges through their declared
  phi-aware pass-through merges, eliminating that retained-SPIR-V product repair.
- Finalized typed CFG emission now clones shared direct arms only for headers with declared merges,
  then routes each clone through the merge with exact phi synthesis, eliminating the numeric-label
  shared-arm product repair.
- AIR `target datalayout` vector alignment and exact `air.struct_type_info` member offsets now flow
  through the primary emitter, aggregate byte walkers, reflection, and every retry tier instead of
  being reconstructed from generic SPIR-V layout assumptions.
- The default descriptor ABI now uses checked, non-overlapping bands for buffers, sampled textures,
  samplers, color inputs, imageblocks, storage textures, and translator-owned resources. Final
  modules reject descriptors outside their selected class range or set.
- Structured control-flow retries are bounded more tightly and reuse exact ordinary-planner
  rejection facts across the primary and construct-tree attempts, improving behavior on large
  shaders without reusing an accepting plan across retry semantics.
- Corpus translation and classification use bounded parallel workers, a 20-second per-item
  watchdog, a 500 MiB per-item memory ceiling, source-size and structural-cost-aware scheduling, and
  incremental index/cache reuse. Watchdog cleanup polls its own child instead of installing a
  process-global signal handler, and worker panics remain hash-attributed retryable failures instead
  of aborting a resumable census. Warm audits avoid reopening unchanged source shards; forced audits
  remain explicit.
- Large-module translation now streams function bodies into typed blocks, borrows unchanged
  normalization input, shares immutable block carriers, interns repeated instruction opcodes, and
  keeps mutually exclusive instruction and phi facts in one canonical representation. Resolved
  instructions derive def/use edges from those operands instead of allocating parallel name lists, and
  the emitter no longer clones every typed operand into a result-keyed table. Async-copy lowering
  streams into one bounded-growth buffer, skips dead declarations, and lets owned callers release
  superseded source text before typed parsing. Together these changes substantially reduce the
  measured largest local rows without changing their translated classification or skipping SPIR-V
  validation.
- Function-variable hoisting and integer-width normalization now preserve each block's instruction
  allocation, moving only matched variables and snapshotting only the instruction being rewritten.
  This removes full-block clones from universal large-module paths while preserving emitted order.
- Integer conversions now select their opcode from the actual emitted SPIR-V storage types. Legalized
  widths such as LLVM `i24` are masked at their producer, signed extensions restore the logical sign
  bit, and equal-width signed vertex inputs use bitcasts, eliminating the late width-convert repair.
- Validation capability checks, authored schema validation, backend execution gates, and cache
  identities now share one typed contract so a clean audit cannot hide a later executor rejection.
- Corpus capability audits now use the product's canonical AIR-call inventory and report every
  called intrinsic outside a recognized lowering or static-linkage family, including exact symbols
  and counts, instead of interpreting an unrelated clean authoring contract as complete support.
- Full AIR-intrinsic reclassification now updates every cached source in bounded keyset batches,
  independent of `--limit`, without reopening source shards; matrix capability recognition and
  lowering share one exact ABI parser.
- Translation census results now flow through a worker-count-bounded queue. A checkpoint failure
  cancels unclaimed rows and drains only in-flight results instead of allowing workers to retain an
  unbounded completed-row backlog while the scoped parent unwinds.
- Reflection documentation now describes the complete schema v30 descriptor, argument-buffer,
  runtime specialization, stage-interface, and conservative buffer-staging contracts.

### Removed

- Retired the monolithic validation ledgers and superseded mint/remint/run/why utilities in favor of
  sharded authored cases, dependency-exact observations, the source index, and unified corpus
  commands.
- Removed obsolete emitter naming and compatibility terminology; the product and tooling describe
  only the native AIR-to-SPIR-V pipeline.
- Removed the universal post-validation StorageBuffer/Workgroup pointer-phi rewrite, successful-
  module constant-branch reparse, and scalar-i64 module wrapper; same-root pointer merges and literal
  dead arms are now resolved on the typed source graph before emission.

### Fixed

- A buffer parameter whose body reads several distinct types through a one-index access chain is no
  longer wrapped as a flat `{ RuntimeArray<T> }` of whichever type happened to be seen first. A
  genuine `device T*` array has exactly one element view; several views mean the index selects
  members of a record, so the buffer's AIR struct layout is reconstructed instead. Wrapping such a
  buffer flatly kept the first view's stride and left every other access loading the wrong type.
- `air.get_num_samples_texture*` now lowers from the image type instead of an unconditional literal
  `1`: a multisampled 2D image queries `OpImageQuerySamples`, a single-sample image keeps the exact
  constant, and a multisampled non-2D dimensionality is refused. `ImageQuery` capability inference
  covers `OpImageQuerySamples`.
- Opaque buffer sources copied into local aggregates are now inferred as byte-addressed even when
  the destination is not another buffer parameter. AIR aggregate metadata whose explicit member
  offsets overlap Vulkan's naturally aligned block extents also selects the faithful byte view.
- Large CFGs now attempt their structurally valid primary emission before any retry. Missing switch
  merges and phi/predecessor mismatches route by their typed CFG failure class to the whole-CFG
  constructor, replacing the former block-count-triggered raw/relooper and construct-tree pre-route.
- Runtime device pointers now select the Vulkan 1.2 buffer-device-address model from typed AIR
  producer/use structure during primary emission. Opaque resource handles remain in the logical
  resource domain, and the redundant validation-triggered plain-BDA retries have been removed.
- Packed 32-bit reads through Private vector-backed helper views are now completed inside the
  memory-lowering transaction, so finalization no longer needs a repeated post-pass legalization.
- Interface binding now materializes direct loads selected between descriptor-backed and Private
  placeholder arms in the value domain, eliminating the corresponding module-wide finalizer repair.
- Descriptor-backed pointer-select closures, including dynamically indexed local pointer tables,
  are now constructed in the Logical value domain on the primary path instead of relying on
  validation-triggered value-select or physical-address repairs.
- Same-root pointer phis and selects now merge their integer access-chain indices and rematerialize
  one pointer for every storage class. Typed forward-select facts cover loop backedges whose
  advancing arm is a later chain of GEPs, so these recurrences need neither pointer SSA nor
  `VariablePointersStorageBuffer`.
- Scalar and vector function-constant initializer expressions are folded on the owned typed CFG
  before merge planning. Literal edges are pruned independently, surviving phi predecessor sets are
  rebuilt, and single-predecessor phis are substituted without letting an unrelated opaque aggregate
  phi suppress safe pruning. AGX execution-mask lowering now keeps loop ownership on the original
  backedge target and gives the lowered exit test a private structured merge.
- Function-constant-gated texture inputs with real Metal locations now retain their own descriptor
  and exact image shape even when their predicate defaults false, so later SPIR-V specialization
  cannot sample or write an unrelated texture. Only the `-1` location sentinel remains absent.
  Arrayed texture writes take their operand shape from the stable AIR intrinsic symbol even after
  the image handle passes through a local carrier.
- Scalar StorageBuffer remodeling now replays exact source GEP byte offsets as one stride-checked
  runtime-array index, preserving aggregate row/lane addresses after metadata collapses the
  descriptor element type.
- Unsupported image-texel value shapes now bypass buffer/CFG retry tiers that preserve the rejected
  SSA value, returning an honest fallback within the translation budget instead of repeatedly
  emitting the same large module.
- Added structural lowering for one-dimensional linear pixel sampling, logical pointer aliases,
  little-endian byte-aggregate/integer reinterpretation, and the packed signed-i8 AGX3 16x16x16
  matrix-MAC ABI. The matrix adapter validates its complete fixed descriptor contract and reuses
  the common distributed 32-lane matrix implementation.
- Unsupported Metal visible function references now fail fast with an explicit fallback instead of
  being treated as ordinary functions.
- Multiple `OpReturnValue` sites are rewritten consistently to stage outputs, and undefined
  fragment output stores are skipped rather than materialized.
- Kernel local-size options are validated as nonzero and are propagated into AIR local-size queries
  and imageblock lowering.
- SPIR-V generation avoids several invalid Logical-addressing forms by normalizing pointer phis,
  pointer selects, reinterpret loads/stores, access-chain index widths, and cross-binding pointer
  merges before final validation.
- Avoided a large-shader performance and size regression by measuring structured-CFG synthetic
  growth as added blocks rather than total module blocks, keeping large already-structured shaders
  on the primary emit path instead of falling back to the relooper tier.
- Removed inline denorm flush-to-zero emulation and avoided emitting `DenormFlushToZero`
  capability/execution modes on Vulkan devices that do not advertise float-controls support,
  restoring expected output size and compile time for affected shaders.
- Corrected imageblock explicit-slice stores so out-of-bounds oversized writes are discarded while
  zero-area writes still materialize transparent zero.
- Corrected integer `air.calculate_unclamped_lod_texture_2d` lowering so `OpImageQueryLod` uses a
  translator-owned default nearest sampler when the source AIR does not provide one.
- Fixed selected static sampler operands for AIR depth sampling and compare-depth lowering so
  `OpSampledImage` receives a valid SPIR-V sampler value even when AIR selects between sampler
  state globals.
- Fixed private indexed-container GEP lowering so dynamic accesses through palette-array pointer
  phis use ordinary access chains instead of invalid `OpPtrAccessChain` pointer-stride forms.
- Fixed function-constant-gated fragment outputs so the default-zero AIR predicate model does not
  advertise mutually exclusive render-target formats at the same Vulkan location.
- Fixed static-initializer evaluation so a specialized integer vector is retained as a typed vector
  and only a genuinely scalar value can enter the scalar global-fold map.
- Fixed fragment `[[sample_id]]` lowering to use `BuiltIn SampleId` with the required
  `SampleRateShading` capability, and lowered AIR `texture2d_ms` reads to MS `OpTypeImage` fetches
  with a `Sample` operand instead of treating the sample id as a mip LOD.
- Corrected native lowering for tessellation patch inputs; ray-intersection result typing and
  setters; custom, direct, narrow, and integer imageblocks; embedded texture arrays; array gather
  operands; array depth comparison sampling; and integer storage-image atomics.
- Corrected Metal SIMD/quad operations for vector `u16` prefix scans, integer extrema, active
  masks, and exact votes, and preserved signedness for same-width integer conversions and atomic
  subtraction.
- Repaired additional raw byte/word buffer, opaque-pointer, aggregate-pointee, record-layout,
  cross-storage select, and late pointer-typing cases without name-keyed workload exceptions.
- Corrected AIR struct member offsets, aggregate byte strides, and physical-pointer retry layouts
  for three-lane vectors and custom vector alignments, including preservation and conflict checking
  of explicit SPIR-V `ArrayStride` evidence.
- Runtime sampler specialization is now preserved through bounded SSA aliases and rejected when
  pointer selection or mixed joins make the exact state ambiguous; integer-image LOD queries and
  gather paths can no longer bypass the specialized state by substituting a default sampler.
- Runtime storage-image specialization now covers top-level and embedded writable textures across
  reads, writes, imageblock writes, emulated fetches, and atomics, with matching component and host
  capability checks in executable and metadata-only reflection paths.
- Vulkan validation now binds every compatible reflected sampled/storage texture alias to the same
  allocation and validates texture-array and argument-buffer alternatives as complete sets.
- Exact Metal `dispatchThreads` grids share one typed region payload across every logical grid
  builtin, including synthesized stage-input indexing and reflected buffer footprints.
- Module, function, block, instruction, and analysis transformations now express ownership by
  consuming values, returning replacements, mutating existing allocations transactionally, or
  appending directly. Fallible late SPIR-V repairs return their module explicitly, so no path can
  leave caller-visible state replaced by an empty/default ownership placeholder. ID-deleting native
  rewrite adapters now remove debug and annotation records only for the exact result IDs deleted by
  their own transaction, eliminating detached product cleanup sweeps without masking unrelated
  invalid metadata.
- Removed the late module-wide integer arithmetic width repair; packed scalar-slot addressing now
  scales dynamic indices in their declared integer type, so 64-bit access-chain indices remain
  valid and value-preserving without after-the-fact truncation.
- Deep shared-arm refunneling now gives every synthesized value phi an explicit incoming on every
  join predecessor, using an unobservable `undef` only on edges whose route branches to the other
  target.
  Value-domain pointer lowering likewise proves post-merge indices and its complete planned value-phi
  graph satisfy every predecessor dominance edge before commit. These construction contracts remove
  the duplicate post-primary non-dominating-value demotion scan.
- Cross-binding pointer-phi lowering now runs before the whole-function relooper size gate. The
  bounded in-memory address rewrite does not depend on CFG re-emission and remains available to large
  functions whose refunnelled value flow produces a complete cross-binding phi.
- Reduced worst-case translation time and memory growth by caching retry verdicts, pruning dead CFG
  before source re-emission, using linear CFG ordering and candidate scans, bounding generated CFG
  growth, and applying resource limits from worker startup.
- Validation now uploads sampled 3D images through bounded transfer buffers instead of relying on
  non-portable linear-tiling 3D images, restoring exact issue-4 and issue-5 Metal/MoltenVK checks.
- Opaque private-tensor descriptor intrinsics now have an explicit exact static-linkage contract,
  keeping their Apple-defined layout paired with the externally defined tensor-operation helper
  that consumes it instead of inventing a partial native representation.

## v0.1.0

### Added

- First public release of the `metal2vulkan` crate and CLI: native Metal AIR / sanitized LLVM IR →
  Vulkan SPIR-V.
- Library entry points for stage-aware translate, optional reflection metadata, and
  function-constant specialization.
- CLI: `metal2vulkan <in.air|.ll> <out.spv> --stage …`, optional `--emit-meta` JSON with the
  `serde` feature, `PASS` / `FALLBACK` reporting, and FALLBACK repro bundles under
  `$TMPDIR/metal2vulkan-repros` (override with `METAL2VULKAN_REPRO_DIR`).
- Optional `serde` feature for serializable `ShaderReflection` / metadata dumps.
- Consumer docs: architecture overview and reflection binding layout (`docs/`).

### Notes

- The crate is **alpha** (`0.x`): public API, CLI flags, and SPIR-V output may still change.
- This package does not ship third-party captured shaders; coverage is synthetic fixtures and
  unit tests only.
