#!/usr/bin/env python3
"""Offline: does the delta actually describe the change? (no SGLang involved)

Bit-level correctness on DSv4 is still unverified: both in-line attempts took the
SGLang server down mid-sync, so the verdict never arrived while the performance
numbers piled up. This answers it from the sender's own dumps
(``VERL_DELTA_SELFCHECK_DIR``), with the server nowhere in the picture.

For each sampled parameter present in two consecutive dumps, decode that sync's
encoded wire exactly as the receiver does and run two independent checks:

  A. APPLY  -- ``dense[N-1]`` with the delta applied equals ``dense[N]``, byte for
     byte. Compared as bytes on purpose: bf16/fp8 payloads contain NaN, and
     ``nan != nan`` would make a value check silently pass.
  B. COMPLETE -- every element that really moved between N-1 and N is covered by
     the delta's positions.

A and B overlap more than the split suggests, and a mutation test is what showed
it: sabotaging B made an under-reported delta fail check A anyway, because an
unreported element leaves ``dense[N-1]``'s old value in the applied result. What
B uniquely buys is (i) the case where a parameter produces NO wire entry at all,
where there is no dtype to build an applied result from and A therefore never
runs, and (ii) a diagnosis -- "3 moved elements uncovered" points at dropped data
(a replica, half a fused pair), where A only says "4 bytes differ".

Usage:
    VERL_DELTA_SELFCHECK_DIR=/path/to/dumps  # on the trainer, for >= 2 steps past the seed
    python -m verl.checkpoint_engine.delta_sync.offline_verify /path/to/dumps [--verbose]
"""

import glob
import os
import re
import sys

import torch

WIDTH_DTYPE = {1: torch.uint8, 2: torch.int16, 4: torch.int32}


def decode_positions(piece):
    """Inverse of the sender's gap encoding -- same arithmetic as the receiver.

    gap is ``idx - prev`` with ``idx[-1] := -1``, so a repeated position rides as
    gap 0 and duplicates are legal (applied last-writer-wins).
    """
    width = int(piece["pos_width"])
    view_dtype = WIDTH_DTYPE.get(width)
    if view_dtype is None:
        raise ValueError(f"unsupported pos_width={width}")
    gaps = piece["gaps"].view(torch.uint8).view(view_dtype).to(torch.int64)
    if gaps.numel() and int(gaps.min()) < 0:
        raise ValueError(f"negative gap at pos_width={width}: signed-carrier overflow")
    return torch.cumsum(gaps, dim=0) - 1


def check_pair(prev, cur, verbose=False):
    """Return (n_ok, failures) for one consecutive dump pair."""
    dense_p, dense_c, pieces = prev["dense"], cur["dense"], cur["pieces"]
    names = sorted(set(dense_p) & set(dense_c))
    failures = []
    n_ok = 0
    total_delta_elems = 0
    total_moved_bytes = 0

    for name in names:
        pb, cb = dense_p[name], dense_c[name]
        if pb.numel() != cb.numel():
            failures.append(f"{name}: dense length changed {pb.numel()} -> {cb.numel()}")
            continue

        plist = pieces.get(name)
        if plist is None:
            # No wire entry at all. Legal ONLY if nothing moved -- otherwise the
            # sync silently dropped a parameter, which is precisely check B.
            if int((pb != cb).sum()):
                failures.append(
                    f"{name}: dense moved {int((pb != cb).sum())} bytes but the delta "
                    f"carried NO entry for it (dropped parameter)"
                )
            else:
                n_ok += 1
            continue

        dtype = getattr(torch, plist[0]["dtype_str"])
        esize = torch.empty(0, dtype=dtype).element_size()
        expected = pb.clone().view(dtype)
        covered = torch.zeros(expected.numel(), dtype=torch.bool)

        for piece in plist:
            idx = decode_positions(piece)
            if idx.numel() == 0:
                continue
            if int(idx.min()) < 0 or int(idx.max()) >= expected.numel():
                failures.append(
                    f"{name}: decoded position out of range "
                    f"[{int(idx.min())}, {int(idx.max())}] vs n_elem={expected.numel()}"
                )
                break
            val = piece["val"].view(dtype)
            if val.numel() != idx.numel():
                failures.append(f"{name}: {idx.numel()} positions but {val.numel()} values")
                break
            expected[idx] = val
            covered[idx] = True
            total_delta_elems += idx.numel()
        else:
            # --- check A: the applied result matches, byte for byte
            exp_b = expected.view(torch.uint8)
            bad_apply = int((exp_b != cb).sum())
            # --- check B: everything that moved is covered by the delta.
            # Largely redundant with A (see the module docstring) -- kept for the
            # diagnosis it gives and for the no-entry case A cannot reach.
            moved_elem = (pb.view(dtype).view(torch.uint8).view(-1, esize) != cb.view(-1, esize)).any(dim=1)
            total_moved_bytes += int((pb != cb).sum())
            uncovered = int((moved_elem & ~covered).sum())
            if bad_apply or uncovered:
                failures.append(
                    f"{name}: apply mismatch {bad_apply} bytes, "
                    f"UNCOVERED moved elements {uncovered} (delta is incomplete)"
                    if uncovered
                    else f"{name}: apply mismatch {bad_apply} bytes"
                )
            else:
                n_ok += 1
                if verbose:
                    print(f"    ok {name}: {int(moved_elem.sum())} elems moved, all covered")

    print(
        f"  {len(names)} sampled params | delta elems {total_delta_elems} | "
        f"dense moved {total_moved_bytes} bytes | passed {n_ok} | failed {len(failures)}"
    )
    return n_ok, failures


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    verbose = "--verbose" in argv
    args = [a for a in argv if not a.startswith("--")]
    if not args:
        print(__doc__)
        return 2
    d = args[0]
    files = sorted(
        glob.glob(os.path.join(d, "selfcheck_step*.pt")),
        key=lambda f: int(re.search(r"step(\d+)", f).group(1)),
    )
    if len(files) < 2:
        print(
            f"need at least two consecutive dumps in {d}, found {len(files)}. "
            f"Run with VERL_DELTA_SELFCHECK_DIR set for >= 2 steps past the seed."
        )
        return 2

    all_failures = []
    prev = torch.load(files[0], map_location="cpu", weights_only=False)
    for f in files[1:]:
        cur = torch.load(f, map_location="cpu", weights_only=False)
        print(f"{os.path.basename(files[files.index(f) - 1])} -> {os.path.basename(f)}")
        _, failures = check_pair(prev, cur, verbose)
        all_failures += [f"step{cur['step']}: {x}" for x in failures]
        prev = cur

    print()
    if all_failures:
        print(f"❌ {len(all_failures)} 处不一致:")
        for x in all_failures[:40]:
            print(f"   - {x}")
        if len(all_failures) > 40:
            print(f"   ... 另外 {len(all_failures) - 40} 处")
        return 1
    print("✅ 抽样参数上 delta 精确描述了 dense 的变化,且没有漏报的字节。")
    print("   注意这只覆盖了抽到的参数(VERL_DELTA_SELFCHECK_FRACTION),不是全量证明。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
