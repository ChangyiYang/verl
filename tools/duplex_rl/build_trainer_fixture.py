# Copyright 2026 Liquid AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Build the minimal real-PCM dataset used by the duplex RayPPOTrainer gate."""

from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True, help="Path to a real PCM/WAV utterance")
    parser.add_argument("--output", required=True, help="Output parquet path")
    parser.add_argument("--num-samples", type=int, default=2)
    args = parser.parse_args()

    audio_path = Path(args.audio).expanduser().resolve()
    if not audio_path.is_file():
        raise FileNotFoundError(audio_path)
    if args.num_samples < 2:
        raise ValueError("The two-step trainer gate requires at least two dataset rows")

    rows = []
    for index in range(args.num_samples):
        rows.append(
            {
                "data_source": "duplex_window",
                "prompt": [
                    {
                        "role": "system",
                        "content": "Listen to the user and interrupt only when speaking is useful.",
                    },
                    {"role": "user", "content": "<audio>"},
                ],
                "audios": [str(audio_path)],
                # The rollout manager produces a controlled G=2 pair whose
                # first actions are LISTEN/SPEAK. These overlapping windows
                # therefore produce rewards [-1, +1] at the same causal state.
                "reward_windows": [
                    {"t_start": 0.0, "t_end": 0.0, "value": -1.0, "only_on": "listen"},
                    {"t_start": 0.0, "t_end": 0.0, "value": 1.0, "only_on": "speak"},
                ],
                "reward_model": {"style": "rule", "ground_truth": ""},
                "extra_info": {"index": index},
            }
        )

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), str(output_path))
    print(f"Wrote {len(rows)} duplex samples to {output_path}")


if __name__ == "__main__":
    main()
