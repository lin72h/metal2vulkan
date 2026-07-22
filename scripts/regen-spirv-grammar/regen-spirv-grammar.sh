#!/usr/bin/env bash
# Regenerate metal2vulkan SPIR-V grammar/decoder tables from Khronos SPIRV-Headers
# using the public rspirv autogen tool, then adapt outputs for this crate.
#
# Usage:
#   scripts/regen-spirv-grammar/regen-spirv-grammar.sh
#   RSPIRV_REF=rspirv-0.13.0 HEADERS_REF=vulkan-sdk-1.4.341.0 \
#     scripts/regen-spirv-grammar/regen-spirv-grammar.sh
#
# Requires: git, cargo, rustfmt, python3. Network access for the first clone.
set -euo pipefail

CRATE_DIR="$(cd -- "$(dirname -- "$0")/../.." && pwd)"
# Project-local cache by default (gitignored). Override with METAL2VULKAN_AUTOGEN_CACHE.
CACHE_DIR="${METAL2VULKAN_AUTOGEN_CACHE:-$CRATE_DIR/.cache/rspirv-autogen}"
RSPIRV_REF="${RSPIRV_REF:-rspirv-0.13.0}"
HEADERS_REF="${HEADERS_REF:-vulkan-sdk-1.4.341.0}"

echo "# crate:   $CRATE_DIR"
echo "# cache:   $CACHE_DIR"
echo "# rspirv:  $RSPIRV_REF"
echo "# headers: $HEADERS_REF"

mkdir -p "$CACHE_DIR"
if [ ! -d "$CACHE_DIR/rspirv/.git" ]; then
  git clone --filter=blob:none https://github.com/gfx-rs/rspirv.git "$CACHE_DIR/rspirv"
fi
cd "$CACHE_DIR/rspirv"
git fetch --tags --force origin
git checkout -f "$RSPIRV_REF"

if [ ! -d autogen/external/SPIRV-Headers/.git ]; then
  git clone --filter=blob:none \
    https://github.com/KhronosGroup/SPIRV-Headers.git \
    autogen/external/SPIRV-Headers
fi
(
  cd autogen/external/SPIRV-Headers
  git fetch --tags --force origin
  git checkout -f "$HEADERS_REF"
)

echo "# running rspirv-autogen..."
( cd autogen && cargo run --release )

echo "# adapting outputs into metal2vulkan..."
python3 - "$CACHE_DIR/rspirv" "$CRATE_DIR" "$HEADERS_REF" <<'PY'
import re
import sys
from pathlib import Path

rspirv_root = Path(sys.argv[1])
crate_root = Path(sys.argv[2])
headers_ref = sys.argv[3]

header = f"""// AUTOMATICALLY GENERATED from the Khronos SPIR-V core grammar JSON
// (SPIRV-Headers tag {headers_ref}: include/spirv/unified1/spirv.core.grammar.json)
// via https://github.com/gfx-rs/rspirv autogen, then adapted for this crate.
// DO NOT HAND-EDIT — regenerate with scripts/regen-spirv-grammar/regen-spirv-grammar.sh
"""


def strip_rspirv_header(text: str) -> str:
    return re.sub(
        r"^// AUTOMATICALLY GENERATED from the SPIR-V JSON grammar:\n"
        r"//   external/spirv\.core\.grammar\.json\.\n"
        r"// DO NOT MODIFY!\n\n?",
        "",
        text,
        count=1,
    )


def write_out(rel: str, body: str) -> None:
    path = crate_root / rel
    path.write_text(header + "\n" + body.lstrip("\n"))
    print(f"  wrote {rel} ({path.stat().st_size} bytes)")


r = rspirv_root

dec = strip_rspirv_header((r / "rspirv/binary/autogen_decode_operand.rs").read_text())
dec = dec.replace("WORD_NUM_BYTES", "WORD_BYTES")
write_out("src/spirv_binary/decode_generated.rs", dec)

err = strip_rspirv_header((r / "rspirv/binary/autogen_error.rs").read_text())
write_out("src/spirv_binary/error_generated.rs", err)

parse = strip_rspirv_header((r / "rspirv/binary/autogen_parse_operand.rs").read_text())
parse = parse.replace("impl Parser<'_, '_>", "impl Parser<'_>")
parse = parse.replace("dr::Operand", "Operand")
# Core-only: drop extended-instruction operand kinds that need rspirv ext modules.
parse = re.sub(
    r"\n\s*GOpKind::(?:Debuginfo|NonsemanticClspvreflection|"
    r"NonsemanticShaderDebuginfo100|OpenclDebuginfo100)\(_\) => \{\n"
    r"(?:.*\n)*?\s*\},?",
    "",
    parse,
)
write_out("src/spirv_binary/parse_generated.rs", parse)

dis = strip_rspirv_header((r / "rspirv/binary/autogen_disas_operand.rs").read_text())
write_out("src/spirv_disassemble_generated.rs", dis)

dr = (r / "rspirv/dr/autogen_operand.rs").read_text()
i = dr.find("impl fmt::Display for Operand")
if i < 0:
    raise SystemExit("Display impl for Operand not found in dr/autogen_operand.rs")
j = dr.find("\nimpl ", i + 1)
disp = dr[i : j if j > i else None]
write_out("src/spirv_operand_display_generated.rs", disp)

gram = strip_rspirv_header((r / "rspirv/grammar/autogen_table.rs").read_text())
for variant in (
    "    Debuginfo(debuginfo::ExtOperandKind),\n",
    "    NonsemanticClspvreflection(nonsemantic_clspvreflection::ExtOperandKind),\n",
    "    NonsemanticShaderDebuginfo100(nonsemantic_shader_debuginfo_100::ExtOperandKind),\n",
    "    OpenclDebuginfo100(opencl_debuginfo_100::ExtOperandKind),\n",
):
    if variant not in gram:
        raise SystemExit(f"expected OperandKind variant missing: {variant!r}")
    gram = gram.replace(variant, "")
# Drop extended-instruction opcode wrapper (core instruction table only).
gram = re.sub(
    r'\n#\[doc = "Wrapper enum for all extended instruction set opcodes\."\]'
    r"[\s\S]*?impl From<ExtInstOp> for spirv::Word \{[\s\S]*?\n\}\n",
    "\n",
    gram,
    count=1,
)
gram = gram.replace(
    "static INSTRUCTIONS: &[Instruction<'static>] = &[",
    "static INSTRUCTIONS: &[InstructionGrammar] = &[",
)
gram = gram.replace(
    "pub static INSTRUCTION_TABLE: InstructionTable<spirv::Op> =",
    "pub static INSTRUCTION_TABLE: InstructionTable =",
)
write_out("src/spirv_binary/grammar_generated.rs", gram)
print("done")
PY

echo "# formatting..."
( cd "$CRATE_DIR" && cargo fmt --all )

echo "# RESULT: regenerated SPIR-V tables from Khronos grammar via rspirv-autogen"
