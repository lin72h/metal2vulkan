# mtlb-extract

Extract LLVM BitcodeWrapper (AIR) blobs from a Metal library (`MTLB`) container or any blob that
embeds them.

AIR wrappers use magic `0x0b17c0de` and may sit at **byte** offsets (not 4-byte aligned). This tool
scans byte-by-byte and carves each complete wrapper `[magic … BitcodeOffset + BitcodeSize)`.

## Run

```sh
python3 scripts/mtlb-extract/mtlb_extract.py path/to/library.metallib
# optional explicit output directory (default: ./mtlb_blobs)
python3 scripts/mtlb-extract/mtlb_extract.py path/to/library.metallib ./out/air
```

Writes `blob_<n>_off<offset>.air` for each record found.

Translate with metal2vulkan (stage auto-detect works when AIR metadata is present):

```sh
cargo build --release --bin metal2vulkan
mkdir -p ./out/spv
for b in ./mtlb_blobs/*.air; do
  ./target/release/metal2vulkan "$b" "./out/spv/$(basename "$b" .air).spv"
done
```

## Notes

- This is a **structural carver**, not a full MTLB directory/format parser (no function-name table).
- Default size guard skips wrapper lengths ≥ 512 KiB (`blen < 0x80000`). The harvest
  script also drops any carved `.air` above 512 KiB as a second line of defense.
- Output `.air` files are gitignored at the repo root (`*.air`).
