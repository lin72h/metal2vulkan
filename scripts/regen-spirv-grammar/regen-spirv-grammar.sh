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
import json
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

core_grammar = json.loads(
    (
        r
        / "autogen/external/SPIRV-Headers/include/spirv/unified1/spirv.core.grammar.json"
    ).read_text()
)
enumerant_requirement_groups = []
operand_arms = []
for kind in core_grammar["operand_kinds"]:
    category = kind.get("category")
    if category not in ("BitEnum", "ValueEnum"):
        continue
    kind_name = kind["kind"]
    value = "value.bits()" if category == "BitEnum" else "*value as u32"
    kind_requirements = []
    seen_values = set()
    for enumerant in kind["enumerants"]:
        raw_value = enumerant["value"]
        numeric_value = (
            int(raw_value, 0) if isinstance(raw_value, str) else raw_value
        )
        if numeric_value in seen_values:
            continue
        seen_values.add(numeric_value)
        capabilities = enumerant.get("capabilities", [])
        extensions = enumerant.get("extensions", [])
        version = enumerant.get("version")
        if not capabilities and not extensions and version in (None, "None", "1.0"):
            continue
        kind_requirements.append(
            (kind_name, numeric_value, capabilities, extensions, version)
        )
    group_index = len(enumerant_requirement_groups)
    enumerant_requirement_groups.append(kind_requirements)
    operand_arms.append(
        f"        Operand::{kind_name}(value) => EnumerantRequirements::new("
        f"ENUMERANT_REQUIREMENTS_{group_index}, {value}, "
        f"{str(category == 'BitEnum').lower()}),"
    )

requirements = [
    "#[derive(Clone, Copy, Debug)]",
    "pub(crate) struct EnumerantRequirement {",
    "    pub(super) value: u32,",
    "    pub(crate) capabilities: &'static [spirv::Capability],",
    "    pub(crate) extensions: &'static [&'static str],",
    "    pub(crate) min_core_version: Option<(u8, u8)>,",
    "}",
    "",
    "pub(super) struct EnumerantRequirements {",
    "    remaining: &'static [EnumerantRequirement],",
    "    value: u32,",
    "    bit_enum: bool,",
    "}",
    "",
    "impl EnumerantRequirements {",
    "    fn new(",
    "        remaining: &'static [EnumerantRequirement],",
    "        value: u32,",
    "        bit_enum: bool,",
    "    ) -> Self {",
    "        Self { remaining, value, bit_enum }",
    "    }",
    "}",
    "",
    "impl Iterator for EnumerantRequirements {",
    "    type Item = &'static EnumerantRequirement;",
    "",
    "    fn next(&mut self) -> Option<Self::Item> {",
    "        while let Some((requirement, remaining)) = self.remaining.split_first() {",
    "            self.remaining = remaining;",
    "            let matches = if self.bit_enum {",
    "                if requirement.value == 0 {",
    "                    self.value == 0",
    "                } else {",
    "                    self.value & requirement.value == requirement.value",
    "                }",
    "            } else {",
    "                self.value == requirement.value",
    "            };",
    "            if matches {",
    "                return Some(requirement);",
    "            }",
    "        }",
    "        None",
    "    }",
    "}",
    "",
]
for group_index, group in enumerate(enumerant_requirement_groups):
    requirements.append(
        f"static ENUMERANT_REQUIREMENTS_{group_index}: &[EnumerantRequirement] = &["
    )
    for kind, value, capabilities, extensions, version in group:
        caps = ", ".join(f"spirv::Capability::{capability}" for capability in capabilities)
        exts = ", ".join(f'\"{extension}\"' for extension in extensions)
        core_version = (
            f"Some(({version[0]}, {version[2]}))"
            if version not in (None, "None")
            else "None"
        )
        requirements.extend(
            [
                "    EnumerantRequirement {",
                f"        value: {value},",
                f"        capabilities: &[{caps}],",
                f"        extensions: &[{exts}],",
                f"        min_core_version: {core_version},",
                "    },",
            ]
        )
    requirements.extend([
        "];",
        "",
    ])
requirements.extend(
    [
        "pub(super) fn operand_declaration_requirements_generated(",
        "    operand: &Operand,",
        ") -> EnumerantRequirements {",
        "    match operand {",
        *operand_arms,
        "        _ => EnumerantRequirements::new(&[], 0, false),",
        "    }",
        "}",
        "",
    ]
)
write_out(
    "src/spirv_binary/enumerant_requirements_generated.rs",
    "\n".join(requirements),
)
print("done")
PY

echo "# formatting..."
( cd "$CRATE_DIR" && cargo fmt --all )

echo "# RESULT: regenerated SPIR-V tables from Khronos grammar via rspirv-autogen"
