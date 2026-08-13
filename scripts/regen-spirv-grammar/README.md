# regen-spirv-grammar

Regenerates the checked-in SPIR-V decoder / grammar tables from the **Khronos SPIR-V
core grammar JSON**, using the public [rspirv autogen](https://github.com/gfx-rs/rspirv/tree/master/autogen)
tool as a generator. The generator is not a runtime dependency of `metal2vulkan`.

## What gets rewritten

| Output | Role |
|---|---|
| `src/spirv_binary/decode_generated.rs` | Enum bit-decoder methods |
| `src/spirv_binary/error_generated.rs` | Decoder error variants |
| `src/spirv_binary/parse_generated.rs` | Operand parse dispatch |
| `src/spirv_binary/grammar_generated.rs` | Core instruction table |
| `src/spirv_disassemble_generated.rs` | Operand disassembly |
| `src/spirv_operand_display_generated.rs` | `Display` for `Operand` |

Hand-written crate code (`parser.rs`, `decoder.rs`, `grammar.rs`, …) is not touched.

## Defaults (locked to the `spirv` crate pin)

- rspirv tag: `rspirv-0.13.0` (ships `spirv` 0.4.0+sdk-1.4.341.0)
- SPIRV-Headers tag: `vulkan-sdk-1.4.341.0`

Override with `RSPIRV_REF` / `HEADERS_REF` if you deliberately bump both together with
`Cargo.toml`'s `spirv` dependency.

The script caches both repositories under `.cache/rspirv-autogen`. Set
`METAL2VULKAN_AUTOGEN_CACHE` to use a different cache directory. It requires `git`, `cargo`,
`rustfmt`, and Python 3, and requires network access when fetching missing revisions.

## Run

```sh
./scripts/regen-spirv-grammar/regen-spirv-grammar.sh
cargo test -p metal2vulkan --lib spirv_module -- --test-threads=1
```

The script runs `cargo fmt --all`. Review every generated diff and update the `spirv` dependency and
locked revisions together.

## Provenance

- **Source of truth:** Khronos `spirv.core.grammar.json` (SPIRV-Headers).
- **Generator:** `rspirv-autogen` (Apache-2.0 tool; not linked into this crate).
- **Checked-in artifacts:** mechanical output of that generation + small renames for this
  crate's API (`WORD_BYTES`, `Operand`, core-only `OperandKind`). They are not a vendored
  copy of the rspirv library sources.
