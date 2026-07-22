#!/usr/bin/env python3
# mtlb_extract.py — extract AIR function blobs from a Metal library (MTLB) or embedded dump.
#
# AIR lives inside MTLB as LLVM BitcodeWrapperHeader records (magic 0x0b17c0de), packed at BYTE
# granularity (not always 4-byte aligned). A 4-byte-stride scan can miss them; this walks byte-by-byte
# and carves each complete wrapper. metal2vulkan can then translate each .air to SPIR-V.
#
# Usage: mtlb_extract.py <mtlb-or-resource.bin> [out_dir]
#   Writes blob_<n>_off<offset>.air for each 0x0b17c0de record found.
import sys, struct, os

AIR_WRAP = b"\xde\xc0\x17\x0b"  # 0x0b17c0de little-endian — LLVM BitcodeWrapperHeader magic

def extract(path, out_dir):
    data = open(path, "rb").read()
    os.makedirs(out_dir, exist_ok=True)
    i = n = 0
    found = []
    while True:
        j = data.find(AIR_WRAP, i)
        if j < 0:
            break
        # BitcodeWrapperHeader: magic@0, version@4, BitcodeOffset@8, BitcodeSize@0xc, CPUType@0x10.
        # The wrapped module spans [0, BitcodeOffset + BitcodeSize).
        if j + 0x14 <= len(data):
            bc_off, bc_size = struct.unpack_from("<II", data, j + 8)
            blen = bc_off + bc_size
            if 0x14 <= blen <= len(data) - j and blen < 0x80000:
                out = os.path.join(out_dir, f"blob_{n:02d}_off{j}.air")
                open(out, "wb").write(data[j:j + blen])
                found.append((n, j, blen, out))
                n += 1
        i = j + 1
    for n, j, blen, out in found:
        print(f"blob {n}: off={j} len={blen} -> {out}")
    print(f"TOTAL {len(found)} AIR blobs")
    return found

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: mtlb_extract.py <mtlb-or-resource.bin> [out_dir]", file=sys.stderr)
        print("  out_dir defaults to ./mtlb_blobs under the current working directory", file=sys.stderr)
        sys.exit(2)
    # Default next to the caller's cwd — never a fixed /tmp path.
    out_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.getcwd(), "mtlb_blobs")
    extract(sys.argv[1], out_dir)
