#!/usr/bin/env python3
"""DeepScaleR-Preview-Dataset -> verl parquet, under this project's own prompt contract.

The point of this file is that the prompt contract is NOT verl's default. Every
DeepScaleR number the verl run has to be compared against (rounds 1-4, recovery
49.15 / 58.47 / 77.97 / 66.95%) was produced with the INSTRUCTION template below
in a single user turn. verl's own examples use their own instruction strings
('Let's think step by step and output the final answer after "####".' for gsm8k),
and a prompt-format change is exactly the failure that cost the predecessor 15
points on this program once already. So the template is transcribed verbatim from
scripts/prepare_quick_eval_data.py rather than retyped.

Validation split: DeepScaleR ships train only. Rather than invent a split, this
writes AIME 2024 as the val file when --aime is given, which is what recovery is
actually measured on. Without it, val is a small held-out slice of train and is
only there because verl requires a val file to exist.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import datasets

# Transcribed from continual-learning-with-soft-prompt/scripts/prepare_quick_eval_data.py.
# Do not paraphrase: the whole program's recovery numbers are conditioned on it.
INSTRUCTION = (
    "Solve the following math problem step by step. The last line of your "
    "response should be of the form Answer: \\boxed{{$Answer}} where $Answer is "
    "the answer to the problem.\n\n{problem}\n\nRemember to put your answer "
    'on its own line after "Answer:".'
)

# NOT the HF dataset id. verl dispatches its reward function on `data_source`
# (verl/utils/reward_score/__init__.py:59) and raises NotImplementedError for any
# name it does not recognise -- including the real dataset id. "math_dapo" routes
# to verl/utils/reward_score/math_dapo.py, the boxed-answer grader, which is the
# same grader family this program has been scoring with. The true provenance is
# kept in extra_info.dataset so nothing is lost.
DATA_SOURCE = "math_dapo"
DATASET_ID = "agentica-org/DeepScaleR-Preview-Dataset"


def _row(problem: str, answer: str, split: str, idx: int) -> dict:
    return {
        "data_source": DATA_SOURCE,
        "prompt": [{"role": "user", "content": INSTRUCTION.format(problem=problem)}],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": str(answer)},
        "extra_info": {"split": split, "index": idx, "question": problem, "dataset": DATASET_ID},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--aime", type=Path, default=None,
                    help="AIME jsonl to use as the val file (what recovery is measured on)")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    ds = datasets.load_dataset(DATASET_ID, split="train")
    print(f"loaded {len(ds)} rows; columns {ds.column_names}")

    rows = []
    for i, ex in enumerate(ds):
        if args.limit is not None and i >= args.limit:
            break
        # DeepScaleR ships 'problem' / 'answer'; fail loudly rather than guess.
        rows.append(_row(ex["problem"], ex["answer"], "train", i))

    args.out.mkdir(parents=True, exist_ok=True)
    datasets.Dataset.from_list(rows).to_parquet(str(args.out / "train.parquet"))
    print(f"wrote {len(rows)} train rows -> {args.out / 'train.parquet'}")

    if args.aime is not None:
        val = []
        for i, line in enumerate(args.aime.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            r = json.loads(line)
            problem = r["prompt"][0]["content"]
            # Same guard prepare_aime_data.py uses: some of these jsonls are raw
            # problems and some already carry INSTRUCTION. Wrapping twice would
            # silently change the contract.
            if "Remember to put your answer" in problem:
                messages = r["prompt"]
            else:
                messages = [{"role": "user", "content": INSTRUCTION.format(problem=problem)}]
            val.append({
                "data_source": DATA_SOURCE,
                "prompt": messages,
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": str(r["label"])},
                "extra_info": {"split": "test", "index": i, "dataset": "aime-2024"},
            })
        src = args.aime
    else:
        val = [dict(r, extra_info=dict(r["extra_info"], split="test")) for r in rows[:64]]
        src = "first 64 train rows (placeholder -- verl requires a val file)"
    datasets.Dataset.from_list(val).to_parquet(str(args.out / "test.parquet"))
    print(f"wrote {len(val)} val rows from {src} -> {args.out / 'test.parquet'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
