#!/usr/bin/env python3
"""WindowRewardManager 自检（纯 CPU，不需要模型）。

验收 2.2 要求：给定合成窗口，**填中的 token index 集合必须与手算完全一致，窗口外恒为 0**。
本测试构造已知的动作位/时间轴，逐条核对：

  W1 单窗口精确命中     —— 只有落在 [t_s,t_e] 内的动作位被填，且值正确
  W2 窗口外恒为 0       —— 非命中位必须严格为 0
  W3 多窗口叠加         —— 重叠窗口的值应累加
  W4 only_on 过滤       —— 只对 listen / 只对 speak 生效
  W5 padding 不被误填   —— action_pos = -1 的帧必须跳过
  W6 分解统计正确       —— reward_extra_info 的分项和与实际填入一致
  W7 越界防护           —— 超出序列长度的动作位不写入、不报错
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from tensordict import TensorDict

from verl import DataProto
from verl.workers.reward_manager import get_reward_manager_cls
from verl.workers.reward_manager.window import RewardWindow


def make_data(action_pos, is_listen, frame_time, T=20, B=1):
    ap = torch.tensor(action_pos, dtype=torch.long).reshape(B, -1)
    il = torch.tensor(is_listen, dtype=torch.int32).reshape(B, -1)
    ft = torch.tensor(frame_time, dtype=torch.float32).reshape(B, -1)
    batch = TensorDict({
        "attention_mask": torch.ones(B, T, dtype=torch.int32),
        "duplex_action_pos": ap,
        "duplex_is_listen": il,
        "duplex_frame_time": ft,
    }, batch_size=B)
    return batch


def run() -> int:
    cls = get_reward_manager_cls("duplex_window")
    print(f"[registry] duplex_window -> {cls.__module__}.{cls.__name__}")
    fails = []

    # 5 帧，动作位 2/5/8/11/14，时间 0..4s，交替 listen/speak
    AP = [2, 5, 8, 11, 14]
    IL = [1, 1, 0, 1, 0]
    FT = [0.0, 1.0, 2.0, 3.0, 4.0]
    T = 20

    # ---- W1 / W2 单窗口精确命中 + 窗口外为 0 ----
    rm = cls(compute_windows=lambda item, i: [RewardWindow(1.0, 3.0, 2.5, "trigger")])
    out = rm(DataProto(batch=make_data(AP, IL, FT, T), non_tensor_batch={}, meta_info={}),
             return_dict=True)
    rt = out["reward_tensor"][0]
    expect_pos = {5, 8, 11}                        # t=1,2,3 落在 [1,3]
    got_pos = set((rt != 0).nonzero().flatten().tolist())
    ok = got_pos == expect_pos and all(abs(float(rt[p]) - 2.5) < 1e-6 for p in expect_pos)
    print(f"  W1 单窗口精确命中   : {'PASS' if ok else f'FAIL got={sorted(got_pos)} want={sorted(expect_pos)}'}")
    fails += [] if ok else ["W1"]

    ok = all(float(rt[p]) == 0.0 for p in range(T) if p not in expect_pos)
    print(f"  W2 窗口外恒为 0     : {'PASS' if ok else 'FAIL'}")
    fails += [] if ok else ["W2"]

    # ---- W3 多窗口叠加 ----
    rm = cls(compute_windows=lambda item, i: [
        RewardWindow(0.0, 2.0, 1.0, "trigger"),
        RewardWindow(1.0, 4.0, 0.5, "timing"),
    ])
    rt = rm(DataProto(batch=make_data(AP, IL, FT, T), non_tensor_batch={}, meta_info={}))[0]
    want = {2: 1.0, 5: 1.5, 8: 1.5, 11: 0.5, 14: 0.5}
    ok = all(abs(float(rt[p]) - v) < 1e-6 for p, v in want.items())
    print(f"  W3 多窗口叠加       : {'PASS' if ok else f'FAIL got={[round(float(rt[p]),2) for p in AP]}'}")
    fails += [] if ok else ["W3"]

    # ---- W4 only_on 过滤 ----
    rm = cls(compute_windows=lambda item, i: [RewardWindow(0.0, 4.0, -1.0, "cost", only_on="speak")])
    rt = rm(DataProto(batch=make_data(AP, IL, FT, T), non_tensor_batch={}, meta_info={}))[0]
    expect_pos = {8, 14}                            # 只有 is_listen=0 的两帧
    got_pos = set((rt != 0).nonzero().flatten().tolist())
    ok = got_pos == expect_pos and all(abs(float(rt[p]) + 1.0) < 1e-6 for p in expect_pos)
    print(f"  W4 only_on 过滤     : {'PASS' if ok else f'FAIL got={sorted(got_pos)}'}")
    fails += [] if ok else ["W4"]

    # ---- W5 padding 帧被跳过 ----
    rm = cls(compute_windows=lambda item, i: [RewardWindow(0.0, 10.0, 1.0)])
    rt = rm(DataProto(batch=make_data([2, 5, -1, -1, -1], IL, [0.0, 1.0, 0.0, 0.0, 0.0], T),
                      non_tensor_batch={}, meta_info={}))[0]
    ok = set((rt != 0).nonzero().flatten().tolist()) == {2, 5}
    print(f"  W5 padding 帧跳过   : {'PASS' if ok else 'FAIL'}")
    fails += [] if ok else ["W5"]

    # ---- W6 分解统计 ----
    rm = cls(compute_windows=lambda item, i: [
        RewardWindow(0.0, 1.0, 2.0, "trigger"),
        RewardWindow(2.0, 4.0, -1.0, "cost"),
    ])
    out = rm(DataProto(batch=make_data(AP, IL, FT, T), non_tensor_batch={}, meta_info={}),
             return_dict=True)
    info = out["reward_extra_info"]
    ok = (abs(info["reward/trigger"][0] - 4.0) < 1e-6      # 2 帧 × 2.0
          and abs(info["reward/cost"][0] + 3.0) < 1e-6     # 3 帧 × -1.0
          and info["n_hit_frames"][0] == 5)
    print(f"  W6 分解统计正确     : {'PASS' if ok else f'FAIL {dict(info)}'}")
    fails += [] if ok else ["W6"]

    # ---- W7 越界防护 ----
    rm = cls(compute_windows=lambda item, i: [RewardWindow(0.0, 10.0, 1.0)])
    try:
        rt = rm(DataProto(batch=make_data([2, 999, 5, -1, -1], IL, FT, T),
                          non_tensor_batch={}, meta_info={}))[0]
        ok = set((rt != 0).nonzero().flatten().tolist()) == {2, 5}
    except Exception as e:
        ok = False
        print(f"      越界抛异常: {e}")
    print(f"  W7 越界防护         : {'PASS' if ok else 'FAIL'}")
    fails += [] if ok else ["W7"]

    print("\n" + "=" * 60)
    if fails:
        print(f"❌ 失败项: {fails}")
        return 1
    print("✅ 全部通过 —— 时间窗到 token 位的映射与手算完全一致")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
