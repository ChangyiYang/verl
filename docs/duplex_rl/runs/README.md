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

---

# DuplexRollout —— 端到端自检通过（真模型，1×MI325X）

`verl/workers/rollout/duplex_rollout.py` + 注册表一行（`("duplex","sync")`）。
`tools/duplex_rl/test_duplex_rollout.py` 不经 Ray/FSDP worker，直接构造 DataProto 调用：

```
[registry] duplex/sync -> verl.workers.rollout.duplex_rollout.DuplexRollout
[shapes]   embeds(1,150,4096) mask(1,150) action_pos(1,10) frames=10
  T1 形状一致              PASS
  T2 动作位可定位          PASS
  T3 动作与 is_listen 一致  PASS
  T3b 采样与 argmax 不同    1/10 帧（随机策略的正常表现）
  T4 response_mask 合理    PASS (生成位 22/150)
  T5 时间轴单调等距        PASS
  T6 可从 embeds 复算 logprob PASS  (P(listen) 范围 [0.0000, 0.9998])
```

## 一处被测试抓出来的真实缺陷
初版 T3 失败 2/10。排查后发现是**测试写错了**：它拿 `argmax(logits)` 去核对动作，
但动作是**采样**出来的——在决策边界上（P(listen)≈0.59）两者本来就会不同。
但这暴露了 rollout 的一个真实缺口：**没有输出 token id 序列**，
导致无法直接核对"某位置到底落了哪个 token"。
⇒ 已给 DuplexRollout 补上 `duplex_token_ids [B,T]`（条件位填 -1），
   verl 侧也本来就需要 token 序列。T3b 现在把"采样≠argmax"作为观测量单独报出。

## 输出契约
| 字段 | 形状 | 含义 |
|---|---|---|
| `duplex_embeds` | [B,T,H] | 完整输入嵌入序列，训练侧据此单次批量前向重算 logprob |
| `duplex_token_ids` | [B,T] | 每位置 token id；音频等条件位为 -1 |
| `response_mask` | [B,T] | 1=模型生成位（可训练），0=条件输入 |
| `duplex_action_pos` | [B,F] | 每帧动作 token 的绝对位置 |
| `duplex_is_listen` | [B,F] | 每帧动作（1=listen） |
| `duplex_frame_time` | [B,F] | 每帧起点秒 —— 窗口 reward 的时间锚 |

---

# WindowRewardManager —— 时间窗 → token 级 reward（纯 CPU 自检全通过）

`verl/workers/reward_manager/window.py`，用 verl 自带的装饰器注册为 `duplex_window`。

依据：帧同步 ⇒ 时间与 token 位置固定映射 ⇒「时间窗给 reward」≡「若干动作 token 位给 reward」。
verl 的 `token_level_scores` 本就与 responses 同形、支持逐 token 赋值
（`naive.py` 只是把标量塞在最后一位），**故窗口化 credit assignment 不改 verl 核心**。

```
[registry] duplex_window -> verl.workers.reward_manager.window.WindowRewardManager
  W1 单窗口精确命中   PASS      W5 padding 帧跳过   PASS
  W2 窗口外恒为 0     PASS      W6 分解统计正确     PASS
  W3 多窗口叠加       PASS      W7 越界防护         PASS
  W4 only_on 过滤     PASS
```
对应 PLAN.md 验收 2.2「填中的 token index 集合与手算完全一致；窗口外全 0」。

## 接口
```python
RewardWindow(t_start, t_end, value, term="trigger", only_on=None)   # only_on: 'listen'/'speak'/None
```
窗口来源二选一：`non_tensor_batch["reward_windows"]`（每样本一个列表），
或 `compute_windows(item, i)` 回调。本类只做「时间→token 位→填分」，
**不做语义判定**（TP/FP/TOO_EARLY… 由上游按 MIB 口径产出后转成窗口）。
