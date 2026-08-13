# metal2vulkan-validation

This unpublished workspace crate implements authored-case validation. Every GPU resource, initial
byte, function constant, dispatch/draw parameter, output region, and comparison rule comes from a
checked manifest. Runners do not infer, seed, resize, repair, or reinterpret cases.

Acceleration structures are literal resources too. Instance entries carry host-defined child
references for introspection. Primitive entries carry tightly packed little-endian triangle
vertices; those exact bytes build Metal's native BLAS and the Vulkan geometry shadow.

Kernel, fragment, and vertex cases share one literal-resource preparation contract across the checker,
Metal oracle, and Vulkan candidate. It covers buffers, textures, explicit and reflected static
samplers, function-constant bit patterns, acceleration-structure shadows, and render targets;
texture and render-target outputs are compared as tightly packed selected regions rather than
backend-specific row-pitched storage.

Fragment cases use their authored draw and render-target records directly. The Metal oracle and
Vulkan candidate each synthesize the same fullscreen vertex interface from the fragment AIR
varyings, then execute a real render pipeline. Vulkan hashes both SPIR-V stages into the candidate
dependency, so changes to the generated companion invalidate evidence instead of being hidden.

Vertex cases author ordinary per-vertex attributes plus a `vertex_observation`. The generated
fragment observer routes either rasterized position or one typed user varying into render target
zero, padding scalars and short vectors to a four-component float, signed-integer, or
unsigned-integer attachment. Metal and Vulkan consume the same draw and attribute bytes; Vulkan
uses a negative-height viewport for the same screen-space orientation as Metal.

A vertex entry with no raster position may instead omit `vertex_observation` and all attachments,
then select a writable shader resource as its output. Both executors run that draw with
rasterization disabled; Vulkan emits no fragment companion, while Metal uses a private 1x1 encoder
sink that cannot participate in the authored observation. This covers void vertex functions whose
contract is a buffer or storage-texture side effect rather than raster output.

Post-tessellation vertex cases replace `draw` with a `tessellation` record: exact binary16 edge
and inner factors per patch, instance/amplification counts, and typed control-point/per-patch
records. Metal consumes those records through native per-patch vertex stepping and `drawPatches`;
Vulkan generates the matching vertex and tessellation-control stages, including reflected 16-bit
system-value types. All generated SPIR-V modules participate in the candidate dependency hash.

Dynamic Metal `threadgroup` arguments are declared through `threadgroup_memory` binding/length
pairs. Metal binds that exact allocation; Vulkan uses the translator's descriptor-free `Workgroup`
storage. The schema intentionally has no initial bytes because both APIs leave this storage
uninitialized—authored shaders must initialize every location they read.

Compute imageblocks use explicit `imageblock.dimensions`; explicit layouts have no host bytes. The
checker requires those dimensions to match the dispatch threadgroup x/y size. Implicit layouts
reuse authored `render_targets` as their initial attachment values and additionally require
`imageblock.implicit_coverage: full_single_sample`. Both runners therefore begin with the same
fully covered, single-sample pixels: Metal encodes a color-write-disabled coverage draw before its
kernel tile dispatch, while Vulkan initializes the reflected attachment image directly. This is an
authored execution fact, not an executor-supplied default.

Custom fragment imageblocks use the distinct `fragment_imageblock` resource. Each accessed master
member is identified by its AIR `user(...)` semantic and carries an exact tightly packed `half`,
`half4`, `uchar4`, or `ushort` plane; scalar `half` is the backward-compatible default format.
`output.kind: fragment_imageblock` reads a selected plane region. The Metal runner uses same-pass
initializer/resolver fragments, while Vulkan binds the matching reflected storage format under
ordered fragment interlock.

Textures nested in an `air.indirect_buffer` are authored through
`argument_buffer_textures`, keyed by owning buffer binding plus field byte offset. Product
reflection supplies the Metal argument-encoder `[[id(n)]]` and the synthetic Vulkan descriptor,
so manifests do not encode backend-specific binding numbers. A fixed embedded array authors one
texture per handle at `field_offset + 8 * element`; Metal binds consecutive argument IDs and Vulkan
binds those same elements into the single reflected descriptor array.

Top-level `array<texture..., N>` and runtime `array_ref<texture...>` arguments use
`texture_arrays`. Element order is the AIR handle index. Fixed arrays must author exactly `N`
elements; runtime arrays author their valid logical prefix. Vulkan binds that prefix into the
reflected 128-descriptor array and fills the unreachable remainder with the final valid descriptor;
Metal binds the same prefix to consecutive texture slots.

Texture `dimensions` describe spatial extent and array layers. Multisample textures additionally
carry an explicit `sample_count`; literal bytes are tightly packed by layer, row, texel, then sample.
This keeps 2D multisample arrays unambiguous instead of overloading the third dimension.

Kernel `[[stage_in]]` attributes use `kernel_stage_inputs`, keyed by AIR attribute location. Each
literal supplies a typed format, the product's reflected runtime-array stride, and record bytes.
Product reflection owns the collision-free synthetic buffer slot: Metal uses it in an
`MTLStageInputOutputDescriptor` with x-grid stepping, and Vulkan binds the same bytes as the
read-only StorageBuffer array indexed by `GlobalInvocationId.x`.

## Commands

| Command | Responsibility |
|---|---|
| `corpus-harvest` | Extract and sanitize AIR into deterministic private source shards |
| `corpus-index` | Incrementally sync or check the disposable SQLite index (`--rebuild` for recovery) |
| `corpus-next` | Select unplanned AIR or record a durable non-evidence review note |
| `corpus-triage` | Run cached structural, capability, or bounded translation audits over indexed rows |
| `corpus-case-check` | Strictly validate or install an explicit manifest |
| `corpus-status` | Report the exact missing-evidence queue |
| `corpus-ab` | Compare two translator binaries without a GPU |
| `corpus-openrouter-propose` | Record local, untrusted model-proposed cases without installing them |
| `corpus-metal` | Qualify explicit checked cases on Metal |
| `corpus-moltenvk` | Execute explicit cases against exact Metal observations |
| `corpus-vulkan` | Execute explicit cases against exact Metal observations |
| `corpus-refresh` | Refresh stale/missing Metal + MoltenVK slots in one cached macOS process (`--all` forces every case) |

For operational sequences and the evidence each command establishes, use the repository
[validation playbook](../docs/VALIDATION.md). This README describes the validation package and its
resource model; it is not a substitute for the ordered gates in that playbook.

Harvest merges new hashes into only their first-six-bit source shards. It does not load or rewrite
the other source shards, and it publishes rewritten-row locations directly to the disposable
index. The following incremental index sync normally reads zero source bytes; after an interrupted
publication it repairs only shards whose full file stamp changed. Both commands report their
affected/scanned shard counts. llvm-dis's scratch-path `ModuleID` comment is excluded from sanitized
AIR identity, so reharvesting identical bitcode is a duplicate rather than a new row. Non-entry AIR
modules are retained in separate private `local/library-modules` shards, preserving their parent
library memberships for visible/intersection-function dependency resolution instead of dropping
them during stage classification.

Candidate observations record a build-time SHA-256 fingerprint of the product `Cargo.toml` and
`src/` tree. `corpus-status` compares that fingerprint instead of retranslating every authored AIR
row, so status and index refresh do not read source bodies. Any translator-source change makes old
candidate evidence stale until it is refreshed.

MoltenVK execution is macOS-only. Native Vulkan execution is currently Linux-only; the Vulkan
command rejects macOS so one portability-stack run cannot occupy both backend slots.

Run all ordinary validation tests serially:

```sh
cargo test -p metal2vulkan-validation -- --test-threads=1
```

GPU execution is machine-specific. Ordinary CI covers identities, strict manifest rejection,
dependency matching, storage transactions, indexing, and A/B policy using owned fixtures.

## Optional model proposals

`corpus-openrouter-propose` can send fresh private AIR rows to an explicitly selected
OpenRouter model and record untrusted case proposals under `corpus/local/proposals/`. This uploads
private sanitized AIR to a third party. A live run therefore requires both
`OPENROUTER_API_KEY` and `--acknowledge-private-air-upload`; never put the key on the command line
or in the repository.

Start with a dry run and a small explicit limit:

```sh
cargo run -p metal2vulkan-validation --release --bin corpus-openrouter-propose -- \
  --model '~deepseek/deepseek-v4-flash-latest' --limit 10 --dry-run

OPENROUTER_API_KEY=... cargo run -p metal2vulkan-validation --release \
  --bin corpus-openrouter-propose -- \
  --model '~deepseek/deepseek-v4-flash-latest' --limit 100 \
  --concurrency 50 --acknowledge-private-air-upload
```

The Rust tool selects AIR with neither a case nor a review note, skips recorded results on later runs,
and writes each response atomically. Use `--retry-failures` to retry recorded failures. Model
output is only a proposal: inspect it against the exact AIR, replace the zero `case_id` with the
locally computed canonical ID, run `corpus-case-check`, and qualify it on Metal before treating it
as evidence. The script never installs cases or review notes.
Requests set OpenRouter's provider sort to `price`, so compatible endpoints are tried from cheapest
to most expensive instead of using the default price-weighted load balancing. Account-level
provider restrictions still apply. The default `--reasoning-effort low` reserves output budget for
the structured manifest and excludes reasoning text from the recorded response; override it only
when the selected model supports the requested level. The request deliberately omits `max_tokens`,
allowing the selected OpenRouter model/provider to use its native completion limit instead of
truncating a long structured result.
For models that support JSON mode but not provider-enforced structured outputs, `--json-object`
places the generated Rust case schema in the prompt and enforces it through local deserialization;
provider-side strict JSON Schema remains the default.

## Small authoring loop

```sh
cargo run -p metal2vulkan-validation --bin corpus-index
cargo run -p metal2vulkan-validation --bin corpus-next -- --limit 1
# Inspect that exact AIR and write one manifest row.
cargo run -p metal2vulkan-validation --bin corpus-case-check -- \
  --manifest /path/to/case.json --install
cargo run -p metal2vulkan-validation --bin corpus-index
cargo run -p metal2vulkan-validation --bin corpus-status
```

A manifest edit replaces the stable `(air_sha256, name)` slot. Its semantic `case_id` changes and
only observations for the old identity are removed. Sibling cases remain intact.

## GPU-free refactor gate

Build or preserve the old binary before editing, build the new binary, then select an explicit
canary, AIR hash/list, or aligned shard:

```sh
cargo run -p metal2vulkan-validation --release --bin corpus-ab -- \
  --old ./m2v-old --new target/release/metal2vulkan \
  --canary --expect-no-change

cargo run -p metal2vulkan-validation --release --bin corpus-ab -- \
  --old ./m2v-old --new target/release/metal2vulkan \
  --shard 12 --fail-on-unlisted-change \
  --allow-spv-change /path/to/intentional-spv-hashes.txt
```

The cache key includes AIR hash, translator binary hash, translation options, stage, exact
external validator identity, and relevant translator environment. Delta classification is not
semantic evidence.

See [the corpus format](corpus/README.md) and the repository
[validation playbook](../docs/VALIDATION.md).
