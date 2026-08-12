# Copyright 2026 Liquid AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""窗口化 reward manager —— 把 wall-clock 时间窗映射成 token 级 reward。

核心依据：**全双工模型是帧同步的**，时间与 token 位置是固定映射
（`t = frame_idx × chunk_seconds`，动作 token 位置由 rollout 直接给出）。
因此「在时间窗 [t_s, t_e] 上给 reward」可以无损地表达成「在若干动作 token 位上给 reward」，
而 verl 的 `token_level_scores` 本来就是与 responses 同形的张量，支持逐 token 赋值
（`reward_manager/naive.py` 只是把标量塞在最后一位而已）。
⇒ 窗口化 credit assignment **不需要改动 verl 核心**。

评分口径对齐 MIB（Model Interruption Bench）：
    error window = [t_s, t_e + Δ]，Δ 默认 10s，t_s/t_e 为「错误步」被说出的起止时间
    latency(t)   = max(0, t − t_e)，仅当 t ∈ [t_s, t_user_end] 时有定义
    TP 在窗口内打断 / TN 全程沉默 / FN 该断没断或断错 / FP 无错却打断
判定所需的时间锚（error window、用户说完时刻）由数据集给出，本类只负责
「把时间区间换算成 token 位置并填分」这一步，不做语义判断。
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import torch

from verl import DataProto
from verl.workers.reward_manager import register

__all__ = ["RewardWindow", "WindowRewardManager"]


@dataclass
class RewardWindow:
    """一个带时间范围的 reward 项。

    t_start / t_end : wall-clock 秒，相对该条 rollout 的起点（左闭右闭）
    value           : 落在窗口内的动作 token 各得的分值（可正可负）
    term            : reward 项名（'trigger'/'timing'/'cost'/…），仅用于分解统计
    only_on         : 限定只对某类动作生效——'listen' / 'speak' / None(不限)
    """

    t_start: float
    t_end: float
    value: float
    term: str = "reward"
    only_on: str | None = None


@register("duplex_window")
class WindowRewardManager:
    """把每条样本的 RewardWindow 列表展开成与 responses 同形的 token 级张量。

    依赖 `DuplexRollout` 产出的三个字段：
        duplex_action_pos  [B,F]  每帧动作 token 的绝对位置（-1 为 padding）
        duplex_is_listen   [B,F]  该帧动作是否为 listen
        duplex_frame_time  [B,F]  该帧起点的 wall-clock 秒

    窗口来源（按优先级）：
        1. `data.non_tensor_batch["reward_windows"]` —— 每条样本一个 RewardWindow 列表
        2. `compute_windows(data_item, i)` 回调 —— 由调用方按语义规则生成

    只做「时间 → token 位置 → 填分」，语义判定（TP/FP/…）留给上游。
    """

    def __init__(self, tokenizer=None, num_examine: int = 0, compute_windows=None,
                 reward_fn_key: str = "data_source", **kwargs) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_windows = compute_windows
        self.reward_fn_key = reward_fn_key

    # ------------------------------------------------------------------
    @staticmethod
    def _windows_for(item, idx: int, data: DataProto, compute_windows) -> list[RewardWindow]:
        ntb = data.non_tensor_batch
        if "reward_windows" in ntb:
            w = ntb["reward_windows"][idx]
            return list(w) if w is not None else []
        if compute_windows is not None:
            return list(compute_windows(item, idx) or [])
        return []

    # ------------------------------------------------------------------
    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        for k in ("duplex_action_pos", "duplex_is_listen", "duplex_frame_time"):
            if k not in data.batch:
                raise KeyError(
                    f"WindowRewardManager 需要 DuplexRollout 产出的 `{k}`；"
                    f"当前 batch 只有 {sorted(data.batch.keys())}"
                )

        action_pos = data.batch["duplex_action_pos"]      # [B,F]
        is_listen = data.batch["duplex_is_listen"]        # [B,F]
        frame_time = data.batch["duplex_frame_time"]      # [B,F]

        # reward 张量与 responses 同形；duplex 下没有独立 responses 时退回按序列长度
        if "responses" in data.batch:
            reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        else:
            ref = data.batch["attention_mask"]
            reward_tensor = torch.zeros(ref.shape, dtype=torch.float32, device=ref.device)

        B, T = reward_tensor.shape
        extra: dict[str, list] = defaultdict(list)
        n_printed = 0

        for i in range(B):
            windows = self._windows_for(data[i], i, data, self.compute_windows)
            valid = action_pos[i] >= 0
            if not bool(valid.any()):
                extra["n_windows"].append(len(windows))
                extra["n_hit_frames"].append(0)
                continue

            pos_i = action_pos[i][valid]
            t_i = frame_time[i][valid]
            listen_i = is_listen[i][valid].bool()

            per_term: dict[str, float] = defaultdict(float)
            hit_total = 0
            for w in windows:
                # 时间窗 → 命中的动作帧（左闭右闭）
                sel = (t_i >= w.t_start) & (t_i <= w.t_end)
                if w.only_on == "listen":
                    sel &= listen_i
                elif w.only_on == "speak":
                    sel &= ~listen_i
                if not bool(sel.any()):
                    continue
                tgt = pos_i[sel]
                tgt = tgt[(tgt >= 0) & (tgt < T)]          # 防越界
                if tgt.numel() == 0:
                    continue
                reward_tensor[i, tgt] += w.value
                per_term[w.term] += float(w.value) * int(tgt.numel())
                hit_total += int(tgt.numel())

            extra["n_windows"].append(len(windows))
            extra["n_hit_frames"].append(hit_total)
            for term, v in per_term.items():
                extra[f"reward/{term}"].append(v)

            if n_printed < self.num_examine:
                n_printed += 1
                print(f"[window-reward] sample {i}: {len(windows)} 个窗口，命中 {hit_total} 个动作位，"
                      f"分解 {dict(per_term)}")

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": dict(extra)}
        return reward_tensor
