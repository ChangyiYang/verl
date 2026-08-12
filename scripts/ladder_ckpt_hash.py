#!/usr/bin/env python3
"""Ladder stage-0 file anchor: hash every tensor in a safetensors checkpoint.

WHAT THIS PINS AND WHAT IT DOES NOT
-----------------------------------
This is the ladder's absolute anchor for the CHECKPOINT IDENTITY: it proves both
runs (nccl / delta_sharded) loaded byte-identical input files, and if they did
not, names exactly which tensors differ.

It is NOT directly comparable to the ladder_*.json snapshots the rollout probe
writes. Those hash SGLang's live state, which is TP-sharded, fused (qkv/gate_up/
w13), re-laid-out (DeepGEMM scale layout) and renamed (module paths vs HF names).
The in-memory anchor is instead "both runs' load-stage snapshots are equal":
sglang's disk load is deterministic given identical files + identical topology,
and identical files is what THIS script certifies. sha256 rather than
torch.hash_tensor on purpose — this runs on the CPU head node, and file identity
does not need the same hash function as the live probes, only a stable one.

Usage:
    python scripts/ladder_ckpt_hash.py /path/to/ckpt_dir -o ckpt_anchor.json
"""

import argparse
import hashlib
import json
import struct
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

CHUNK = 64 << 20


def hash_file_tensors(path: Path) -> dict:
    """Per-tensor sha256 from one safetensors file, streamed by byte range."""
    out = {}
    with open(path, "rb") as f:
        (hlen,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(hlen))
        base = 8 + hlen
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            start, end = meta["data_offsets"]
            f.seek(base + start)
            h = hashlib.sha256()
            remaining = end - start
            while remaining:
                buf = f.read(min(CHUNK, remaining))
                if not buf:
                    raise IOError(f"{path}: unexpected EOF in {name}")
                h.update(buf)
                remaining -= len(buf)
            out[name] = {"sha256": h.hexdigest(), "dtype": meta["dtype"], "shape": meta["shape"]}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt_dir", type=Path)
    ap.add_argument("-o", "--output", type=Path, required=True)
    ap.add_argument("-j", "--jobs", type=int, default=8)
    args = ap.parse_args()

    files = sorted(args.ckpt_dir.glob("*.safetensors"))
    if not files:
        print(f"no *.safetensors under {args.ckpt_dir}", file=sys.stderr)
        return 1

    merged: dict = {}
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        for path, tensors in zip(files, ex.map(hash_file_tensors, files)):
            dup = merged.keys() & tensors.keys()
            if dup:  # same tensor in two files = malformed ckpt; refuse to pick one silently
                print(f"duplicate tensor names across files (first: {sorted(dup)[:3]})", file=sys.stderr)
                return 1
            merged.update(tensors)
            print(f"{path.name}: {len(tensors)} tensors", file=sys.stderr)

    scales = sum(1 for n in merged if "scale" in n)
    args.output.write_text(json.dumps(merged, indent=0, sort_keys=True))
    print(f"wrote {len(merged)} tensor hashes ({scales} scale tensors) -> {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
