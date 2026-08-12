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

---

# train_equiv_probe —— 训练前向能否复现 rollout 前向

PPO/GRPO 要求初始 `new_logprob == old_logprob`（ratio ≡ 1）。
rollout 走**流式**路径（逐帧 prefill/generate + KV cache），训练要走**整段批量 teacher-forced 前向**。
本探针录下 rollout 期间每次 `StreamDecoder.feed` 的 embeds，拼成完整序列后**一次性**重放，
再逐决策点比对 logits。

| dtype | 决策点 | argmax 一致 | logits 最大绝对差 | log P(listen) 最大差 | 判定 |
|---|---|---|---|---|---|
| bfloat16 | 29 | **29/29** | 1.000000 | 0.187498 | ❌ |
| **float32** | 31 | **31/31** | **0.000210** | **0.000046** | ✅ **等价** |

## 结论
两条路径**数学上等价**；bf16 下的差异纯粹是数值精度（argmax 始终一致即为佐证）。
⇒ **训练侧可用「录 embeds → 单次批量前向」重算 logprob**，不必用流式路径带梯度重放。

## 但 bf16 的差异在工程上不能忽略
bf16 下 `log P(listen)` 偏差可达 0.19 ⇒ ratio ≈ e^0.19 ≈ **1.21**，第 0 步就偏离 1 达 21%。
**解法正是 verl 现成的做法**：verl 不直接采信 rollout 的 logprob，而是用 actor 的训练前向
重算 `old_log_probs`（`ray_trainer.py:1306/1333/1347` 的 `compute_log_prob`；:1631 亦注明
"Decoupled mode: Recomputes old_log_probs as proximal anchor"）。
只要我们的 `compute_log_prob` 与训练走同一条批量前向，ratio 在第 0 步严格为 1，与 rollout 精度无关。

## ⚠️ 本测试未覆盖的一点
重放用的是**录下来的 embeds**，因此验证的是 **LLM 主干**的流式↔批量等价性，
**未验证音频编码器**在「流式分块编码」与「整段编码」下是否一致。
若训练侧打算**重新计算**音频 embeds（而非复用录下的），需另做一次等价性检查。
