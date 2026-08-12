#!/usr/bin/env python3
"""Compare ladder snapshots: which tensors' hashes differ between two stages.

Two modes:
  * two .json files (same rank, any runs):
        python scripts/ladder_compare.py a.json b.json
  * two stage prefixes in one dir — pairs files rank-by-rank:
        python scripts/ladder_compare.py --dir /ladder --a nccl_load --b nccl_sync1
    (prefix matches ladder_<prefix>_<counter>_rank<r>.json)

Exit code 0 = all compared pairs identical, 1 = any difference, 2 = usage/pairing
error. A missing rank on one side is an ERROR, not a skip: a silently unpaired
rank looks exactly like a pass.
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def diff_one(a: dict, b: dict, label: str) -> bool:
    only_a = sorted(set(a) - set(b))
    only_b = sorted(set(b) - set(a))
    changed = sorted(k for k in a.keys() & b.keys() if a[k] != b[k])
    same = len(a.keys() & b.keys()) - len(changed)
    print(f"[{label}] common={same + len(changed)} identical={same} DIFFER={len(changed)} "
          f"only_a={len(only_a)} only_b={len(only_b)}")
    if changed:
        by_suffix = Counter(k.rsplit(".", 1)[-1] for k in changed)
        print(f"  differing by suffix: {by_suffix.most_common()}")
        for k in changed[:20]:
            print(f"    {k}")
        if len(changed) > 20:
            print(f"    ... {len(changed) - 20} more")
    for name, lst in (("only_a", only_a), ("only_b", only_b)):
        if lst:
            print(f"  {name} first: {lst[:5]}")
    return not changed and not only_a and not only_b


def pair_by_rank(d: Path, prefix: str) -> dict[str, Path]:
    """rank -> file for ladder_<prefix>_<counter>_rank<r>.json; counter must be unique per rank."""
    out: dict[str, Path] = {}
    pat = re.compile(rf"^ladder_{re.escape(prefix)}_\d+_rank(\w+)\.json$")
    for f in sorted(d.iterdir()):
        m = pat.match(f.name)
        if not m:
            continue
        if m.group(1) in out:
            print(f"ambiguous: two files for rank {m.group(1)} with prefix {prefix} "
                  f"({out[m.group(1)].name}, {f.name}) — pass a more specific prefix", file=sys.stderr)
            sys.exit(2)
        out[m.group(1)] = f
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*", type=Path)
    ap.add_argument("--dir", type=Path)
    ap.add_argument("--a", help="stage prefix, e.g. nccl_load")
    ap.add_argument("--b", help="stage prefix, e.g. nccl_sync1")
    args = ap.parse_args()

    if args.dir:
        if not (args.a and args.b):
            ap.error("--dir needs --a and --b")
        fa, fb = pair_by_rank(args.dir, args.a), pair_by_rank(args.dir, args.b)
        if not fa or not fb:
            print(f"no files for prefix a={args.a}({len(fa)}) b={args.b}({len(fb)}) in {args.dir}", file=sys.stderr)
            return 2
        if fa.keys() != fb.keys():
            print(f"rank sets differ: a={sorted(fa)} b={sorted(fb)}", file=sys.stderr)
            return 2
        # list first: `all` on a generator would stop at the first differing rank,
        # and the per-rank report is the point
        ok = all([diff_one(load(fa[r]), load(fb[r]), f"rank{r}") for r in sorted(fa)])
    elif len(args.files) == 2:
        ok = diff_one(load(args.files[0]), load(args.files[1]), f"{args.files[0].name} vs {args.files[1].name}")
    else:
        ap.error("pass two json files, or --dir with --a/--b")
        return 2
    print("RESULT: IDENTICAL" if ok else "RESULT: DIFFER")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
