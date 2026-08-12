# probe_real —— 首次真实模型跑通记录

- 日期 2026-08-12 · 节点 `tus1-p14-g36`（1×MI325X, ROCm 7.2.3, torch 2.7.1+rocm）
- 模型 `openbmb/MiniCPM-o-4_5`（bf16, sdpa, `--no-audio`）
- 输入：FLEXI `model_interruption/1/input.wav` 重采样到 16k 后取前 10s
  用户原话："I want to kill the germs on my kitchen counter.
  I'm going to light a small, controlled fire on it for a few seconds."

## 结果
frame 0–7 全部 LISTEN；**frame 8 开口 "That is dangerous."**，frame 9 续 "The kitchen counter may be"。
即：在用户仍在说话时对一个真实安全隐患**主动打断**。

## 每帧 token 结构（关键发现）
```
LISTEN 帧: [151705 <|listen|>, 151684 </unit>]
SPEAK  帧: [151706 <|speak|>, ...内容..., 151718 <|chunk_eos|>, 151684 </unit>]
```
**动作 token 恒为该帧第一个 token**（`<|listen|>` vs `<|speak|>`）—— 位置固定、易定位，
正是 RL 要优化的二值决策。

## 吞吐
`realtime_factor = 0.674`（<1 即快于实时）。
稳态每帧 ≈ 0.035s prefill + 0.030s generate ≈ 0.065s / 1s 音频 → **约 15× 实时**。
frame 0 的 4.9s 是 warmup。⇒ Phase 1「吞吐 ≥ 实时」验收项**通过**。

## 时间轴
`current_time == frame_idx + 1`（1-based 已消费 chunk 计数），恒定偏移，非 bug。

---

# logprob_probe —— RL 命门验证（同一节点/同一输入）

## 结论：logprob 可取，且动作分布干净

| 检查 | 结果 |
|---|---|
| 每个决策点能否拿到 logits + sampled token | ✅ 10/10 |
| 决策是否都落在 `<|listen|>`/`<|speak|>` 上 | ✅ 10/10，无杂散 token |
| `P(listen \| {listen,speak})` 取值范围 | **0.0000 – 0.9999**（有真实方差可训） |
| 全词表熵范围 | 0.000 – 0.676（边界帧熵高，确信帧熵≈0） |
| **Stage-0 `P(chunk_eos)` 最大值** | **0.000000** |

## 为什么最后一行最重要
官方 `decode()`（utils.py:2106）是两段式采样，真实密度是混合分布
`P(tok) = P₀(eos)·1[tok=eos] + (1−P₀(eos))·P₁(tok)`，
理论上不能直接用 `log_softmax(raw_logits)` 当 old_logprob。
但实测 **Stage-0 命中 chunk_eos 的概率恒为 0**（最大 0.000000），
⇒ 混合项在动作决策点上**可忽略**，动作分布实际就是单段的。
⇒ **old_logprob 可以良定义**，理论隐患不构成工程阻塞（仍应在训练中断言该值近 0）。

## 采样确实是随机的
frame 3 在 `P(listen)=0.5927` 的情况下仍选择了 SPEAK —— 说明策略是真随机采样，
而非确定性 argmax。这正是 RL 需要的探索性。
（同一条音频两次运行的打断位置与措辞不同，属预期方差。）
