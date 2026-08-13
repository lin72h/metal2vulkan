# Authored validation corpus

Canonical state is plain, deterministically ordered JSONL. The generated `.index.sqlite` is
gitignored and may be deleted at any time.

```text
validation/corpus/
  local/sources/shard_000.jsonl ... shard_063.jsonl  # private, gitignored
  local/library-modules/shard_000.jsonl ...          # non-entry AIR dependencies, private
  reviews/shard_000.jsonl ...                       # non-evidence queue notes
  cases/shard_000.jsonl ...                         # authored, no AIR bodies
  observations/metal/shard_000.jsonl ...
  observations/moltenvk/shard_000.jsonl ...
  observations/vulkan/shard_000.jsonl ...
  .index.sqlite                                     # generated, gitignored
```

All paths with the same shard number cover the same bucket. The bucket is the first six bits of
lowercase `air_sha256`. This is a permanent storage contract: do not change the bucket count or
mapping to rebalance the corpus.

The 64-bucket choice is based on a 2026-08-10 measurement of 43,656 private source rows. The
measured 32-, 64-, and 128-bucket layouts averaged 1,364.25, 682.12, and 341.06 rows per bucket,
respectively; their largest buckets held 1,451, 740, and 401 rows. Sixty-four was the smallest
layout with an average below 1,000 rows while retaining reasonably large sequential JSONL shards.

## Sources

Private source rows contain the SHA-256 of the exact sanitized text, stable AIR stage/entry
metadata, sanitized `air_ll`, optional `blob_b64`, source-library hash, and a non-semantic label.
Harvest output is sorted by AIR hash and byte-identical for the same extracted inputs. Metallibs,
AIR bodies, and source shards must never be committed.

Harvest retains non-entry AIR modules separately instead of discarding them. These modules carry
visible/intersection function implementations and other linkable helpers, along with the set of
parent metallib hashes in which each sanitized module appeared. They are not shader queue rows and
cannot be authored as standalone cases; later function-table linking resolves them as dependencies
of a stage entry from the same library provenance.

## Cases

One AIR can have several named cases. A case explicitly records buffers, threadgroup memory,
acceleration structures, top-level/array/argument-buffer textures, samplers, color render targets,
an optional depth/stencil attachment, vertex
inputs, kernel stage inputs, function constants, dispatch or draw parameters, selected output,
comparison, and execution-safety contract. Missing bytes, sizes, formats, dimensions, constants, or
output choices are errors. Cyclic AIR uses `execution_safety: authored_bounded`; its rationale must
identify the authored input or function-constant value that makes every reachable loop finite.

Fragment cases declare every reflected color output in `render_targets`, every reflected depth or
stencil aspect in `depth_stencil`, and provide a `draw`.
Both executors derive a matching fullscreen vertex interface from AIR metadata, preserve the
authored initial attachment bytes with load semantics, execute the draw, and compare the selected
color, depth, or stencil region. The AIR depth qualifier (`any`, `less`, or `greater`) determines
the pipeline depth comparison on both APIs. This is the same observation/evidence path used by
kernel cases.

Custom `air.imageblock_master` fragments additionally declare `fragment_imageblock` planes keyed
by exact AIR user semantic and product-supported member format; a fragment-imageblock output
selects that semantic and a 2D region.

Vertex cases declare `vertex_inputs`, exactly one render target at index zero, and a
`vertex_observation` selecting position or a reflected user-varying location. The generated
fragment observer makes that vertex result an exact render-target observation on both APIs.

An instance `acceleration_structures` entry represents a literal instance acceleration structure. Its
`child_references` length is the instance count. The Metal oracle builds that many identity
instances of a canonical triangle and binds the resulting opaque object. Candidate runners bind
the translator's documented shadow ABI at the same Metal buffer index: little-endian `u32`
instance count, reserved zero `u32`, then the listed little-endian `u64` host pointer payloads.
Those pointer payloads are deliberately explicit; raw child-pointer identities are not portable
Metal-versus-Vulkan comparison values.

A primitive entry sets `kind: "primitive"` and supplies `primitive_triangles_b64` as tightly packed
little-endian `float3` vertices (36 bytes per triangle). The Metal oracle builds the BLAS from those
vertices; the Vulkan candidate uses the identical bytes in the primitive geometry shadow.

Direct Metal visible-function references are authored under `visible_function_references` as the
logical metadata function name plus the SHA-256 of its retained AIR module. The module may belong
to a separately harvested library because Metal's linked-functions contract is explicitly
cross-library; both the Metal oracle and Vulkan candidate consume that same exact module. Visible
and intersection function tables explicitly list their Metal buffer binding and populated slots.
A visible slot names an exact function plus the SHA-256 of a retained non-entry AIR module from the
same parent metallib. An intersection slot is tagged either `linked` with that same exact
dependency contract, or `opaque_triangle` with the sorted Metal intersection-signature flags.
Opaque triangle is a populated native sentinel, not a null slot and not a synthetic AIR function.
Case checking resolves linked dependencies once. Metal binds the corresponding native table entry;
the Vulkan candidate statically replaces visible calls and fully opaque-triangle ray queries only
when their authored signatures match. Unresolved modules, cross-metallib functions, missing
symbols, unauthored slots, and signature mismatches are errors or honest translation fallback.
Tables carried inside an `air.indirect_buffer` use
`argument_buffer_intersection_function_tables`; their identity is the owning buffer binding plus
field byte offset. Reflection supplies the owner parameter, field ordinal, and Metal argument index,
so Metal encoding and Vulkan static specialization consume the same AIR-derived coordinate.

`case_id` is SHA-256 over a canonical semantic JSON encoding. It includes everything capable of
changing execution or comparison and excludes the name, rationale, author, JSON key order, shard,
and storage details. `input_sha256` independently binds observations to the exact literal inputs.

An AIR that is not ready for a case may have one aligned review note containing an explicit reason
and reviewer identity. Review notes remain `unplanned`, are indexed for the queue, and are never
semantic or execution evidence. Authoring the first case for that AIR removes its review note.

Within an AIR, `name` is a unique conceptual slot. Replacing it removes the old case identity and
only its Metal/MoltenVK/Vulkan observations through a recoverable aligned-shard transaction.
Explicit deletion uses `corpus-case-check --delete-air HASH --delete-name NAME` and performs the
same targeted cascade without installing a replacement.

## Observations

A Metal slot is `(case_id, environment_id)`. A candidate slot is
`(case_id, backend, environment_id)`. Requalification replaces the exact slot; duplicate active
slots are rejected.

Metal evidence carries case, AIR, input, output, environment, and oracle-ABI hashes. Candidate
evidence additionally carries the exact Metal output hash, SPIR-V hash, backend environment, and
executor ABI. Reuse requires every dependency to match.

Metal qualification runs a checked literal case three times. All selected outputs must be
identical, and the result must differ from the exact initial bytes across the entire selected
region. A returned poison region is a qualification failure, not a golden. Candidates require an
exact usable Metal observation and compare only against that observation's selected bytes.

## Safety and privacy

The checker accepts acyclic AIR as `loop_free` and cyclic AIR only as `authored_bounded` with an
explicit finite-bound rationale. The latter is an auditable semantic assertion about the literal
inputs, not a claim that a CPU timeout can cancel already committed GPU work.

Committed cases and observations may contain authored input/output bytes but no AIR bodies or
Apple-owned binaries. Git history is the audit trail for replaced cases; active shards retain no
superseded case identities.
