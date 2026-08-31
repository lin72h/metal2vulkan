# Validation playbook

Validation has two independent contracts: translator byte stability and authored-case semantic
correctness. Neither SPIR-V validity nor a delta category proves agreement with Metal.

Choose the smallest route that proves the claim being made:

| Goal | Required starting point |
|---|---|
| Ordinary code change | Format, clippy, and parallel product/validation tests |
| Behavior-preserving translator refactor | Ordinary gate plus exact old/new `corpus-ab` over the structurally affected selection |
| Intentional shader behavior change | Synthetic regression, reviewed byte drift, and affected authored Metal/candidate evidence |
| Corpus-wide support or performance claim | Resumable indexed `corpus-triage` audit with its completion/read/time summaries |

## Ordinary gate

Run Rust tests with Cargo's default available parallelism:

```sh
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
cargo test -p metal2vulkan
cargo test -p metal2vulkan-validation
```

CI uses owned synthetic fixtures. Private AIR and machine-specific GPU execution are optional.

## Private corpus and authored evidence

### Harvest and incrementally index

Harvesting only creates private sources; it never creates execution obligations or manifests.

```sh
cargo run -p metal2vulkan-validation --release --bin corpus-harvest
cargo run -p metal2vulkan-validation --bin corpus-index
cargo run -p metal2vulkan-validation --bin corpus-next -- --limit 1
```

Ordinary `corpus-index` runs are incremental: unchanged multi-gigabyte AIR source shards are not
opened. Use `corpus-index -- --rebuild` only for explicit recovery; `--check` verifies a cloned,
incrementally synchronized snapshot without performing a second full rebuild. Upgrading a legacy
index installs shard stamps from file metadata without reading source bodies. Its first lookup of a
row without a byte location scans only that hash's one aligned shard, records every row location in
that shard, and then uses direct seeks.

Harvest is shard-local too. A batch is deduplicated in memory, then merged only into buckets that
received a hash; unrelated source shards are neither opened nor rewritten. The harvest summary
reports `affected_shards`, and index synchronization reports `source_shards_scanned` and
`source_bytes_scanned`, so an accidental corpus-wide traversal is visible immediately. A changed
bucket's compact identities and byte locations are published to SQLite while the merger still has
them in memory; the following index sync therefore reads zero AIR bytes. If publication is
interrupted, the unmatched file stamp makes the next sync repair only that bucket. Sanitization
also removes llvm-dis's input-path `ModuleID` comment, so a scratch-directory name cannot turn an
identical AIR module into a new content identity.

### Audit authoring capabilities

`corpus-triage` stores analyzer-versioned structural facts in the disposable index. Its first audit
reads only selected indexed rows and skips retained blob decoding; repeated audits reuse those facts.
A newly harvested content hash has no cached facts, so only that new row is analyzed.

The authoring-capability audit covers every indexed source, including rows that already have an
authored case. It selects uncached identities in SQLite, reads only their indexed source slices,
and fails when a complete census contains a structural requirement that the authored schema and
executors cannot represent. The shared check consumes reflection—not source names—and covers both
resource kinds and executable shape/state facts such as texture dimensionality and storage format,
static-sampler state, depth/stencil outputs, and tessellation interfaces.

The same census also inventories every called `air.*` ABI symbol through the product crate's
canonical intrinsic-family contract. A call outside that contract adds the typed
`unrecognized-air-intrinsic` product-support gap, with the exact symbol and aggregate call count.
Declarations alone are not uses. Product-support gaps are tracked separately from authored-tooling
requirements, so adding a schema or executor feature cannot make one disappear: the product's
family contract must gain an intentional lowering or exact static-linkage path, and the lowering
remains responsible for validating operand and result shapes.

```sh
cargo run -p metal2vulkan-validation --release --bin corpus-triage -- \
  --audit authoring-capabilities --limit 100000
```

After changing the product or source-derived structural contract, bump the analyzer ABI and add
`--reclassify-all`. It deliberately selects every indexed identity up to `--limit`, reopens the
indexed source slices, and replaces their facts under the new contract. This is the cold closure
proof; follow it with the ordinary command above, which must select no rows and open/read zero source
shards once the index is warm.

The summary exposes classified and remaining counts plus exact shard bytes read, so both an
incomplete support claim and an accidental corpus rescan are visible. Classification defaults to
the host's available logical CPU count (the equivalent of `nproc`). Only one parsed source may wait
outside the workers (individual rows can be tens of MiB), result channels are bounded, and cache
writes are committed in fixed-size batches, so a full recomputation does not retain the corpus in
memory.

### Run the bounded translation census

The translation census is resumable and uses the same indexed source locations. Discovery skips any
hash that has already translated (or has been classified as requiring authored linkage), while
`--current-fingerprint` performs a fresh regression sweep after product or translation-audit harness
changes. Its cache fingerprint covers both contracts. `--retry-failures` re-runs only failures
recorded for that exact fingerprint. `--retry-linkage` resumably revisits every row that an older
fingerprint classified as requiring linkage but the current fingerprint has not attempted.
Uncursored batches still prioritize historical failures, but materialize that compact priority set
once rather than probing the complete audit history independently for every indexed identity;
cursor pages use strict hash order and skip the priority work entirely.

Rows whose exact AIR flow proves that an observed visible-table operation depends on table contents
are classified as `authored_linkage_required` before spawning translation tools. Dead lookups remain
ordinary translation work, while unsupported pointer escapes remain failures; the shortcut applies
only when the shared structural audit proves authored linkage is the sole missing semantic input.
Direct `!air.visible_function_references` are different: their metadata already names the logical
symbol. The audit resolves that symbol recursively when exactly one retained module from the same
library defines it, using the disposable SQLite `(library, symbol)` index and the module's exact
JSONL byte range. When parent-library resolution is missing or ambiguous, an exact authored case
may instead name cross-library reference modules and their recursive dependency closure. Function
tables likewise require exact authored slot population. For either authored linkage form, the
isolated worker reads only that identity's hash-derived case shard, checks every case against the
source row it already owns, translates with the exact linked modules, slots, and function
constants, and validates every resulting module. The row is `translated` only when all of its
authored cases succeed; a row with no complete authored population remains
`authored_linkage_required`.

`--tier-census` is retained as compatibility telemetry. Product translation now reports `default`
for any successfully constructed module and `fallback` for a construction error; representation
selection is deliberately not exposed as validator-gated tier adoption. The disposable index can
resume this measurement without rescanning warm source shards. Inputs that structurally require
authored linkage remain `authored_linkage_required`.

For a targeted regression set, `--hash-file PATH` reads the lowercase SHA-256 in the first
whitespace-delimited field of each non-empty line. It audits exactly those identities through their
indexed byte ranges, so a saved structural manifest never requires a shard or corpus rescan. Combine
it with `--tier-census` to persist the compatibility construction result for exactly that set.

Translation workers default to the host's available logical CPU count (the equivalent of `nproc`).
An explicit `--jobs N` is an operator override, not part of the normal full-corpus command. Each row
is decoded from its indexed shard slice and translated in an independent child with the
per-translation limits below. The summary reports source-read time separately from
translate/validate time so serial decode overhead stays visible. To prevent CPU contention from
turning into false wall-time failures, all requested workers finish sources below 256 KiB first,
then at most two workers translate the costly 256-KiB-and-larger lane. Structurally costly
device-address/function-table rows join that bounded lane regardless of serialized size. Inputs no
larger than 384 KiB with at least 370 source blocks and 400 call instructions use a dedicated
single-worker sublane, so their attempts cannot consume one another's wall-time budget while
unrelated costly rows continue on the other worker.

```sh
cargo run --release -p metal2vulkan-validation --bin corpus-triage -- \
  --audit translation --limit 500 --summary-only
cargo run --release -p metal2vulkan-validation --bin corpus-triage -- \
  --audit translation --retry-failures --limit 500 --summary-only
cargo run --release -p metal2vulkan-validation --bin corpus-triage -- \
  --audit translation --tier-census --limit 500 --summary-only
cargo run --release -p metal2vulkan-validation --bin corpus-triage -- \
  --audit translation --hash-file imageblock-sources.txt --tier-census --summary-only
```

The 500-row checkpoint is the measured default for the current 88,820-row corpus: unlike a
1,000-row checkpoint containing several of its largest modules, it retained comfortable margin
under the repository's 20-second workflow ceiling while still persisting every ten completions.
Jobs default to the host's available logical cores (`nproc`). Explicitly increasing that value
beyond the available cores can starve individual workers and create false timeouts.

The translation audit supplies the validation graphics executor's single-sample pipeline contract
when lowering `air.get_num_samples.i32`; authored candidate execution uses that same value. This is
not a product default: general callers provide their exact raster sample count through
`TransformOptions` or the CLI.

Each translation runs in a killable child with a 20-second / 500-MiB ceiling. The parent owns the
child's scratch subtree, so success, failure, timeout, and memory termination all have the same
deterministic cleanup path; a killed child never strands a PID-named directory. Completed results
cross a queue bounded by the requested worker count. If a ten-row SQLite checkpoint fails, the
parent cancels unclaimed work and drains only the in-flight bounded queue before reporting the
database error.

### Run focused structural audits

Visible-function-table lowering can be audited in deterministic bounded batches without traversing
the corpus. The summary names the first and last hash; pass the last hash back as `--after` to read
the next batch:

```sh
cargo run -p metal2vulkan-validation --bin corpus-triage -- \
  --audit visible-function-tables --limit 200 --summary-only
cargo run -p metal2vulkan-validation --bin corpus-triage -- \
  --audit visible-function-tables --after LAST_SHA256 --limit 200 --summary-only
```

The audit distinguishes direct/cast calls, table-handle and callback-pointer threading through
internal helper parameters, authored slot nullness checks, and unsupported pointer escapes.
Function-constant-wrapped table parameters retain the same authored-linkage role. Dynamic
dispatchers contain only linked functions whose LLVM signature matches the call site, allowing one
authored table to hold heterogeneous visible-function types. AIR's exact `ptrtoint` / truncate /
compare-with-`1` opaque-intersection sentinel probe is folded to false for authored visible tables,
whose slots are strictly linked functions or null; other pointer-integer observations remain
unsupported.

Ray-intersection families use the same indexed cursor workflow:

```sh
cargo run -p metal2vulkan-validation --bin corpus-triage -- \
  --audit ray-intersections --limit 200 --summary-only
cargo run -p metal2vulkan-validation --bin corpus-triage -- \
  --audit ray-intersections --after LAST_SHA256 --limit 200 --summary-only
```

The product crate owns the compositional `air.intersect.*` family descriptor. The audit checks each
callee's exact return aggregate and argument count against that descriptor, so validation and
translation cannot drift into separate whole-symbol allowlists.

Focused audits select cached structural facts, not unsupported-requirement rows. A feature therefore
remains auditable after it becomes fully supported. Device-address captures use the same cursor and
the same killable 20-second / 500-MiB translation worker as the full census:

```sh
cargo run --release -p metal2vulkan-validation --bin corpus-triage -- \
  --audit device-address-hierarchy --limit 200 --summary-only
cargo run --release -p metal2vulkan-validation --bin corpus-triage -- \
  --audit device-address-hierarchy --after LAST_SHA256 --limit 200 --summary-only
```

### Author an executable case

Authored fragment cases are executable, not review-only: `render_targets`, `depth_stencil`, and
`draw` feed real Metal and Vulkan render pipelines. Depth outputs preserve the reflected AIR
`any`/`less`/`greater` qualifier, while stencil outputs use the native fragment stencil-export
contract. Each runner derives the same fullscreen vertex inputs from the
fragment AIR interface, and the candidate dependency hash covers both generated vertex SPIR-V and
translated fragment SPIR-V. Render-target formats retain their exact channel width across both
executors, including scalar `R32Float`, two-channel `Rg32Float`, and four-channel `Rgba32Float`.
Authored Vulkan draws use clockwise front faces, matching a Metal render encoder's default winding;
this keeps `[[front_facing]]` values comparable without rewriting the shader builtin.

Custom fragment `[[imageblock_data]]` is authored separately as `fragment_imageblock`: one
tightly-packed plane per accessed AIR user semantic, with an explicit `half`, `half4`, `uchar4`, or
`ushort` format and the same input/output/inout byte rules as other resources. `half` remains the
omittable default. Metal initializes and resolves those exact types with generated fragment helpers
in the same render pass; Vulkan binds the corresponding storage-image formats under ordered pixel
interlock. `output.kind: fragment_imageblock` selects a semantic and 2D region for comparison.

Framebuffer-fetch inputs reuse the authored render target at the same attachment index. Metal
loads that attachment natively; Vulkan binds the same initialized image as both a color attachment
and subpass input in `GENERAL` layout with a by-region self-dependency, preserving the single-pass
read/modify/write contract. These cases use the ordinary one-instance fullscreen triangle so each
sample has one fragment invocation; manifests that request intra-draw overlapping framebuffer
fetches are rejected instead of assuming an unavailable raster-order attachment extension.

Kernel functions with implicit imageblock attachments execute as Metal tile pipelines and Vulkan
storage-image attachment planes. Their render-target bytes are the shared initial pixel values.
The exact scalar/vector plane format is reflected for `half`, `half2`, `half4`, `float`, and
32-bit integer AIR imageblock calls rather than padding narrow planes to four channels.
`imageblock.implicit_coverage: full_single_sample` explicitly requests the fullscreen,
color-write-disabled Metal coverage prepass needed for ordinary implicit imageblock writes to
persist through the render-pass store action.

Authored vertex cases use the same graphics executor with literal vertex attributes and an explicit
`vertex_observation`. Product reflection carries vertex-output type, field name, and linker semantic;
the generated fragment companion observes that exact interface and writes attachment zero.

Inspect the selected sanitized AIR. Write one meaningful manifest with exact resources, bytes,
function constants, dispatch/draw parameters, output region, exact comparison, and rationale. Then:

For an AIR instance acceleration structure, declare `acceleration_structures` with its Metal
binding and explicit `child_references`. The oracle constructs identity instances of a canonical
triangle; candidates serialize the count and host-defined child payloads using the product shadow
ABI. Do not compare raw child-pointer identities across Metal and Vulkan.

For an AIR primitive acceleration structure, set `kind` to `primitive` and provide
`primitive_triangles_b64`: tightly packed little-endian triangle vertices (nine `f32` values per
triangle). Metal builds a primitive AS from those exact vertices. Vulkan consumes the same bytes
through the reflected geometry shadow when AIR intersection lowering requires it; an unused native
handle correctly has no Vulkan descriptor.

```sh
cargo run -p metal2vulkan-validation --bin corpus-case-check -- \
  --manifest /path/to/case.json --install
cargo run -p metal2vulkan-validation --bin corpus-index
cargo run -p metal2vulkan-validation --bin corpus-index -- --check
cargo run -p metal2vulkan-validation --bin corpus-status
```

If the AIR cannot yet be authored safely, record why without creating a case or evidence row:

```sh
cargo run -p metal2vulkan-validation --bin corpus-next -- \
  --review-air AIR_SHA256 --reason "explicit unsupported requirement" --reviewed-by ID
```

The aligned review note remains an `unplanned` queue annotation and survives index rebuilds. It is
removed when a case for that AIR is installed.

### Optional model proposals

For large private corpora, `corpus-openrouter-propose` may be used as an optional
offline proposal source. It sends sanitized AIR to the configured OpenRouter model, so live use
requires an explicit private-upload acknowledgement and an API key supplied only through
`OPENROUTER_API_KEY`. Responses remain gitignored under `validation/corpus/local/proposals/` and
are not cases or evidence. Every proposed manifest still requires AIR-specific review, mechanical
checking, installation, and Metal qualification through the commands below; the script performs
none of those acceptance steps.

The checker recomputes identities, verifies source hash and AIR metadata, rejects malformed or
duplicate resources, and validates output bounds and product reflection. A cyclic module must use
`execution_safety: authored_bounded` and explain which literal input or function constant gives
every reachable loop a finite bound; `loop_free` remains mechanically checked. It never supplies
missing facts.

Replacing an installed `(air_sha256, name)` slot is atomic. To remove one explicitly, including
only that identity's observations:

```sh
cargo run -p metal2vulkan-validation --bin corpus-case-check -- \
  --delete-air AIR_SHA256 --delete-name CASE_NAME
```

## Contract 1: emitter stability

Preserve the old translator binary, build the new one, and use the smallest useful explicit
selection:

```text
single reproducer
  -> owned synthetic canary
  -> structurally affected private shards
  -> full background corpus
```

```sh
cargo run -p metal2vulkan-validation --release --bin corpus-ab -- \
  --old ./m2v-old --new target/release/metal2vulkan \
  --canary --expect-no-change

cargo run -p metal2vulkan-validation --release --bin corpus-ab -- \
  --old ./m2v-old --new target/release/metal2vulkan \
  --air-list affected.txt --fail-on-unlisted-change \
  --allow-spv-change intentional.txt \
  --allow-fallback-to-success expected-new-support.txt
```

Selections are `--air-sha256`, `--air-list`, `--shard`, and `--canary`. Reports distinguish
unchanged/changed SPIR-V, fallback-to-success, success-to-fallback, valid-to-invalid, and tool or
timeout failure. Cache reuse includes the AIR hash, translator binary hash, exact options, stage,
external validator identity, and relevant `METAL2VULKAN_*` translator environment.

For a behavior-preserving refactor, `--expect-no-change` is the hard gate. For intentional work,
`--fail-on-unlisted-change` makes any drift outside the reviewed hash lists fail.

## Contract 2: semantic correctness

GPU runners accept explicit case IDs only. Metal qualification is first:

```sh
cargo run -p metal2vulkan-validation --release --bin corpus-metal -- \
  --case-id CASE --environment-id METAL_ENV
```

Qualification first requires the shared checker to prove that the selected output maps to a
shader-writable reflected resource, then executes the literal manifest three times. The selected
bytes must be identical on all runs; byte-identical in-place transformations remain valid evidence
because their write contract is established structurally rather than inferred from changed poison
bytes. The accepted row records case/AIR/input/output hashes, output bytes, environment identity,
and exact oracle ABI.

An authored `output` with `kind: "none"` records the exact empty byte sequence after normal GPU
execution. The shared checker accepts it only when product reflection exposes no writable buffer,
image, attachment, depth/stencil aspect, imageblock member, vertex position, or vertex varying; it
cannot be used to ignore an observable shader result.

Candidates require a matching Metal slot and run independently:

```sh
cargo run -p metal2vulkan-validation --release --bin corpus-moltenvk -- \
  --case-id CASE --metal-environment-id METAL_ENV --environment-id MOLTENVK_ENV
cargo run -p metal2vulkan-validation --release --bin corpus-vulkan -- \
  --case-id CASE --metal-environment-id METAL_ENV --environment-id VULKAN_ENV
```

Each candidate row records the exact input, Metal output, emitted SPIR-V, backend environment, and
executor ABI dependencies. The runner compares selected candidate bytes only with the recorded
Metal bytes for the same case. `corpus-moltenvk` is a macOS runner; `corpus-vulkan` requires a
native Vulkan host (currently Linux) and rejects macOS rather than relabeling MoltenVK evidence.

Direct visible-function references and function-table cases use the same commands. A direct
reference under `visible_function_references` names the logical symbol from AIR's
`!air.visible_function_references` metadata and the exact harvested module that defines it. Direct
dependencies may come from a separate explicitly harvested Metal library, matching Metal's linked
functions API. The Vulkan path recursively closes their dependency metadata, replaces every
`.MTL_VISIBLE_FN_REF` stub with a direct call, and rejects an unresolved symbol. The checker also
resolves each authored table slot from the hash-derived private library-module shard and proves that
the exact module defines the exact authored symbol before either runner starts. Table entries may
come from a separate explicitly harvested Metal library, matching `MTLLinkedFunctions`' object-based
linking contract. Metal receives all referenced `MTLFunction` handles through one descriptor and
creates native function tables. Intersection
entries may instead author Metal's explicit opaque-triangle sentinel and its sorted signature
flags; that entry uses the dedicated native opaque-triangle API and needs no library module.
Vulkan translation specializes visible-table calls to the exact linked AIR definitions; constant
slots become direct calls and dynamic slots become a switch over the authored population. A
callback-bearing ray query becomes callback-free only when every possible table slot is explicitly
opaque triangle and has the exact compositional AIR-family signature. Null slots, linked callbacks,
and mismatches remain unsupported rather than silently losing callback semantics.

An intersection table nested in an AIR argument buffer is authored separately under
`argument_buffer_intersection_function_tables` by owner buffer binding and field byte offset. The
checker resolves that pair through product reflection; Metal writes the native table with the
entry function's `MTLArgumentEncoder`, while Vulkan traces the corresponding constant struct-member
load from the owner entry parameter before applying the same opaque-table specialization.

A device buffer nested in an AIR argument buffer is authored under `argument_buffer_buffers` by
the same owner-buffer binding and field byte offset, with ordinary input/output literal bytes. Metal
encodes the native buffer at the reflected argument index. Vulkan allocates it with device-address
support and writes that address into the reflected owner field; the BDA lowering then dereferences
the identical logical resource. `argument_buffer_buffer` output selection observes writable nested
buffers directly.

A top-level AIR `array_ref<void>` is authored under `device_buffer_arrays`. `binding` is the
literal AIR/Metal base buffer index, `length` is the address-table extent, and each sorted sparse
element owns ordinary input/output literal bytes; undeclared slots are null. Metal flattens the
elements onto consecutive native buffer indices. Vulkan binds an address-table buffer at the base,
allocates every element separately with device-address support, and writes their addresses into the
matching slots. `device_buffer_array_element` output selection observes one writable element.

Every function table declares `size` separately from `entries`. The size is the native table
capacity; omitted indices are authored null slots, so `entries: []` honestly represents an all-null
table without inventing a linked function. Entry indices must be unique, sorted, and smaller than
the declared size. This distinction is semantic: AIR table-size and null-slot queries observe it.

Dependency consequences are deliberately narrow:

- unchanged SPIR-V and matching dependencies reuse candidate evidence;
- changed SPIR-V reuses matching Metal evidence and reruns affected candidates;
- replacing a named case deletes only the old identity's observations, then requires Metal again;
- changed Metal ABI removes only incompatible Metal slots and their downstream usability;
- changed candidate ABI or backend environment reruns only that backend/environment slot.

## Product bug loop

1. State whether SPIR-V should change and identify the structural feature.
2. Reproduce with one AIR or owned synthetic fixture.
3. Add a structural synthetic regression when practical.
4. Fix product behavior without corpus names or environment gates.
5. Run format, clippy, and serial product tests.
6. Run narrow A/B with unlisted drift failing, then expand to related canaries/shards.
7. For intentional changed SPIR-V with authored cases, reuse matching Metal and rerun only affected
   candidates.
8. Report exactly which scope was executed and which remains unaudited.

The live corpus contains only current cases and current experiment slots. Git history, not active
superseded rows or mutable append-only rows, preserves replacements.

## Package and privacy boundary

The published product crate excludes `validation/`, docs, scripts, captures, and private sources.
Committed validation artifacts are owned synthetic fixtures, authored manifests, and
hash-identified observations. Never commit `validation/corpus/local/`, `.index.sqlite`, metallibs,
AIR blobs, SPIR-V dumps, or third-party shader bodies.
