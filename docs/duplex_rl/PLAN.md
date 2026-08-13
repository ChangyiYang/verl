# Audio Full-Duplex RL on verl — 方案与验收标准

分支：`duplex-rl` · 目标：在 verl 上支持**全双工语音模型的主动打断 RL**（InterruptPO 的第一阶段）

---

## 0. 一句话方案

**不重写框架，给 verl 加三个模块**：一个逐帧推进的 `DuplexRollout`、一个把时间窗映射成 token 区间的 `WindowRewardManager`、一个帧↔秒↔token 的对齐工具。verl 其余部分（GRPO / KL / FSDP / Ray / checkpoint）全部复用。

底座：**MiniCPM-o 4.5**（9B，真全双工，原生 Listen-Speak 二值 control token）。备选 Moshi。

---

## 1. 为什么这么做（三条支撑）

**① 全双工模型是帧同步的 ⇒ duplex rollout 本质就是 token 序列。**
每帧 = 一组 token，`时间 = token_index × 帧长`。所以"在 wall-clock 窗口 `[t_s, t_e+Δ]` 上给 reward" ≡ "在 token index 区间上给 reward"。
verl 的 `reward_manager/naive.py:54` 已经构造 `torch.zeros_like(responses)` 形状的 reward 张量，`ray_trainer.py:103` 消费 `token_level_scores` —— **逐 token 赋值的基础设施已经存在**，naive 只是把标量塞在最后一位。**窗口化 credit assignment 不需要改 verl 核心。**

**② 训练时用户音频是预录的 ⇒ 不需要真正的双向实时流。**
```
用户流：teacher-force（预录音频 token，逐帧喂入，已知）
模型流：sample（逐帧采样，这才是被训练的策略）
```
退化成"带外部条件输入的逐帧自回归采样循环"。真 live duplex 只有**部署**才需要。这是"重写流式框架"与"给 verl 加个 rollout"的分水岭。ASPIRin 用 8×V100 就做完了。

**③ 只训 timing，不训 content。**
MIB 实测：模型在 turn-based 下能检测 59–67% 的错误，但被迫实时打断时 TP% 跌到 <10%——"**瓶颈不是理解错误，而是在说话中途产生 content-aware 的打断**"。
ASPIRin 实测：直接对 raw token 做 GRPO 会**生成塌缩 + 重复**，投影成粗粒度状态后 duplicate n-gram 降 >50%。
⇒ **我们只对 MiniCPM-o 的 Listen-Speak control token 做 RL，不动内容生成。** MiniCPM-o 原生就把 whether-to-speak 和 what-to-say 分开了（论文消融证明 LS ≫ LT）。

---

## 2. 要写的代码

```
verl/workers/rollout/duplex_rollout.py       # 新增 · 核心 · ~300–500 行
    class DuplexRollout(BaseRollout):
        generate_sequences(prompts: DataProto) -> DataProto
        # 逐帧循环：喂入预录用户 token → 前向 → 采样 LS control token
        #   （+ 说话时采样内容 token）→ 推进一帧
        # 输出：response_ids（交错流）、response_mask（1=模型发声，0=倾听）
        #       frame_index → wall-clock 秒 的映射表

verl/workers/reward_manager/window.py        # 新增 · ~150 行
    class WindowRewardManager:
        # 输入：[(t_start, t_end, value), ...] 时间窗列表
        # 输出：形状 = responses 的 token_level_scores 张量
        # 负责把秒 → frame_index → token_index 区间，按项填值

~~verl/utils/duplex/frame_align.py~~          # ❌ 实际不需要（2026-08-12 实测后删除）
    帧↔秒↔token 的换算已内联：rollout 直接输出 duplex_frame_time / duplex_action_pos，
    reward manager 直接按秒比较，无需单独的换算模块。

verl/workers/rollout/base.py                 # 改 1 行：_ROLLOUT_REGISTRY 加条目
```
**verl 其余代码不动。** 全部新增在独立文件里，便于 rebase 上游。

---

## 3. 分阶段方案 + 验收标准

> 验收原则：每个 Phase 的 gate 都必须是**可自动执行、有明确阈值、能判 pass/fail** 的。
> 不接受"看起来работает"这类判断。

### Phase 0 — 离线评测器（CPU only，不占卡）

**做什么**：把 MIB 的判定协议实现出来，套到 FLEXI 的 200 条 model_interrupt 上；并合成负例补上 FLEXI 缺的 no-error 对照集。

产出：
- `tools/duplex_eval/mib_protocol.py` —— error window `[t_s, t_e+Δ]`(Δ=10s)、`latency(t)=max(0,t−t_e)`、TP/TN/FP/FN 四分类、6 个 FN + 4 个 FP 子类型
- `tools/duplex_eval/build_negatives.py` —— 合成 no-error 对照音频

**✅ 验收（全部必须通过）**
| # | 检查项 | 阈值 |
|---|---|---|
| 0.1 | 单元测试：构造合成样本，覆盖 **全部 10 个子类型**（TOO_EARLY / WRONG_ERROR / TOO_LATE / PH_CORR / PH_OTHER / MISSED / FALSE_CLAIM / UNRELATED / AFFIRM / PH_CLAIM） | 10/10 分类正确 |
| 0.2 | **判别力对照**：跑两个退化策略——"永不出声"和"逢音必打断" | 前者 MISSED=100%；后者在负例上 FP=100% 且 TP% 不高于随机。**若两者得分接近，说明指标坏了** |
| 0.3 | 确定性：同一输入跑 3 次 | 标签完全一致（LLM judge 部分固定 temperature=0 且缓存） |
| 0.4 | 负例集规模与质量 | ≥200 条 no-error 样本；随机抽 20 条人工/LLM 复核，**≥95% 确认确实无错误** |
| 0.5 | 端到端跑通 FLEXI 200 条 | 输出完整报告，无崩溃，未判定样本 <2% |

**Gate**：0.1–0.5 全绿才进 Phase 1。

---

### Phase 1 — 底座在 ROCm 上跑通（1–2 卡）

**做什么**：验证 MiniCPM-o 4.5 能在 MI325X 上加载、前向、采样，且 **Listen-Speak control token 可读可控**。

**✅ 验收**
| # | 检查项 | 阈值 |
|---|---|---|
| 1.1 | 模型在 1 张 MI325X 上加载并对 30s 音频完成前向 | 无报错，logits 无 NaN |
| 1.2 | **能逐帧读出 LS control token** | 能打印出每个决策点的 listen/speak 及其 logit |
| 1.3 | 能连续采样 ≥60s 交互 | 不崩、不退化成静音或复读 |
| 1.4 | **吞吐 ≥ 实时** | 处理 1s 音频耗时 < 1s（否则 rollout 不可行，需先优化） |
| 1.5 | **复现已发表数字**（防止环境搭错） | 在 FLEXI model_interrupt 上跑出的 TP%/打断率，与论文报告的量级一致（MiniCPM-o Instruct-FD interrupt = 1.7%，即应落在 <10% 区间）。**若跑出个 40% 说明环境或协议接错了** |
| 1.6 | License 确认 | 明确可用于我们的用途，书面记录在本文档 |

**Gate**：1.1–1.6 全绿。**1.4 不达标则先做性能优化，不进 Phase 2**（rollout 会成为瓶颈）。
**1.5 是最关键的防呆项**——先复现别人的数，再谈自己的数。

---

### Phase 2 — DuplexRollout + 窗口 reward 打通（8 卡）

**做什么**：实现三个新模块，跑通一次端到端 GRPO。配置照 ASPIRin：LoRA、G=2、KL β=1e-3。

**✅ 验收 — 分三层**

**(a) 正确性（单元级）**
| # | 检查项 | 阈值 |
|---|---|---|
| 2.1 | `frame_align` 往返转换 | `frame → sec → frame` 与 `token → frame → token` **完全相等**，随机 10k 次 |
| 2.2 | `WindowRewardManager` 填值 | 给定合成窗口，**填中的 token index 集合与手算完全一致**；窗口外全 0 |
| 2.3 | `DuplexRollout` 输出形状 | `response_ids` / `response_mask` / `token_level_scores` 三者形状一致；mask 中 speak 段与实际采样的发声帧对齐 |

**(b) 训练能跑（系统级）**
| # | 检查项 | 阈值 |
|---|---|---|
| 2.4 | 端到端 1 个 GRPO step | ✅ 1×MI325X G=2 gate 通过：ratio=1、loss/grad finite、参数变化且策略方向正确 |
| 2.5 | checkpoint 存取 | 存后重载，前向输出与存前**逐比特一致** |
| 2.6 | 连续 50 step 稳定性 | 不 OOM、不 hang（日志 20s 内必须有增长）、entropy 不塌到 0 |

**(c) 有效性（这才是真验收）**
| # | 检查项 | 阈值 |
|---|---|---|
| 2.7 | **TP% 提升** | 在 held-out 集上，TP% 比 base **绝对提升 ≥5 个点**，**≥3 个随机种子**上都成立 |
| 2.8 | **FP% 不失控** ⭐ | FP% 绝对增幅 **≤5 个点**。<br>*理由：MIB 实测 FP% 可以从 2.0 飙到 98.7 而 TP% 毫无提升——"多打断"换不来"打得准"。只看 TP% 一定会被 reward hacking。* |
| 2.9 | **抗退化对照** ⭐ | 用我们的 reward 评"逢音必打断"退化策略，**其得分必须低于训练后的策略**。若退化策略得分更高，说明 reward 设计漏了，回炉 |
| 2.10 | **语义不退化** | held-out 上 duplicate 2-gram / 3-gram 比例**增幅 ≤10%**（ASPIRin 口径）；LLM judge 回复质量分不低于 base −0.2（5 分制） |
| 2.11 | 延迟指标 | 打断的平均 latency（MIB 定义）**不劣于 base** |

**Gate**：(a)(b) 全绿 **且** 2.7–2.11 全部满足才算 Phase 2 成功。
**2.8 + 2.9 是防 reward hacking 的双保险，不可省略。**

---

### Phase 3 — 完整动作空间 + 完整 reward

扩到 7 动作（listen / backchannel / soft_interrupt / hard_interrupt / clarify / yield / continue）+ `R_trigger + R_timing + R_content + R_cost + R_recovery + R_KL`。

**✅ 验收**
| # | 检查项 | 阈值 |
|---|---|---|
| 3.1 | 跨 benchmark 泛化 | 在**训练时未用过**的 benchmark 上（如只训 FLEXI 则测 MIB 协议 / ProVoice CFC）TP% 提升 ≥3 点 |
| 3.2 | 分桶不退化 | 按 disfluency 类型分桶，**没有任何一桶显著变差**（尤其 SELF_CORRECTION——不能靠抢跑刷分） |
| 3.3 | state-rollback 负例 | 在 FDB v3 的 21 条 self-correction 场景上，**抢跑率不升高**（这是 `R_cost` 的直接检验） |
| 3.4 | 与已发表方法对照 | 在 FLEXI 上与 ASPIRin/Multi-Faceted 报告的量级可比 |

---

## 4. 关键风险与对策

| 风险 | 影响 | 对策 |
|---|---|---|
| **开源底座检测能力弱** | MIB 实测 Moshi/PersonaPlex/Freeze-Omni **在 turn-based 下也检测不出错误**；59–67% 那三个全是闭源。若 MiniCPM-o 也如此，"只训 timing"的前提不成立 | **Phase 1.5 就要测出来**：先在 turn-based 条件下量 MiniCPM-o 的检测率。若 <20%，则必须把 detection 也纳入训练目标，方案要改 |
| 1Hz 决策粒度偏粗 | 打断时机精度受限 | MIB 窗口容差 Δ=10s，1Hz 够用；不够就试 0.2s（论文消融过） |
| rollout 太慢 | 训练不可行 | Phase 1.4 先卡吞吐；不达标先优化，必要时降到更小底座 |
| reward hacking | 刷 TP% 但乱打断 | 2.8 / 2.9 双保险；FP% 与退化策略对照同时纳入 gate |
| MiniCPM-o license 不明 | 阻断 | Phase 1.6 先确认，不确认不投入 |

---

## 5. 里程碑与产出

| Phase | 算力 | 产出 | 主 Gate |
|---|---|---|---|
| 0 | CPU | 离线评测器 + 负例集 | 10 个子类型全对 + 退化策略可区分 |
| 1 | 1–2 卡 | ROCm 上可跑的底座 | **复现已发表数字** + 吞吐 ≥ 实时 |
| 2 | 8 卡 | DuplexRollout + 窗口 reward + 首个训练结果 | **TP% +≥5 且 FP% +≤5 且 退化策略得分更低** |
| 3 | 8–16 卡 | 完整动作空间 | 跨 benchmark 泛化 + 分桶不退化 |

---

## 5.5 训练侧构造输入（实测得出，必须遵守）

HF 模型**二选一**：`Qwen3Model.forward` 明确
`raise ValueError("You must specify exactly one of input_ids or inputs_embeds")`。

⚠️ 但**不可**把 `duplex_embeds` 原样当常量喂 —— 文本 token 位的 embeds 也是 rollout 时
由 `embed_tokens` 查表得到的，原样喂会让 **embedding 矩阵静默失去梯度**（不报错）。

正确做法是用 rollout 的两个字段**拼**出唯一的 `inputs_embeds`：

```python
emb = duplex_embeds.clone()                              # 音频位：常量
m   = duplex_token_ids >= 0
emb[m] = model.model.embed_tokens(duplex_token_ids[m])   # 文本位：可导
logits = model(inputs_embeds=emb, position_ids=...)
```
断言（建议常开）：第 0 步重建的文本位 embeds 与录下的应逐比特相同。

## 6. 待定 / 需确认

1. **MiniCPM-o 4.5 的 license**（论文只标了 OpenBMB 版权）—— Phase 1.6 前必须确认。
2. **这个分支最终落到哪个 repo**：目前开在 `ChangyiYang/verl`。若要接 liquid_rl 主线，应迁到 `Liquid4All/verl`。
3. **Instruct-FD 数据**尚未开源；Phase 0/2 先用 FLEXI + 自建负例，等开源后再加。
4. 是否需要保留 Moshi 作为并行对照（有 ASPIRin/Multi-Faceted 两套现成配方可比）。

---

### 附：本方案依据的实测结论
- verl `reward_manager/naive.py:54` + `ray_trainer.py:103` → **per-token reward 基础设施已存在**
- verl `workers/rollout/base.py` → `BaseRollout` 是干净 ABC，`HFRollout` 证明可绕开 vLLM/SGLang；`_ROLLOUT_REGISTRY` 加一行即可注册
- MIB（Anonymous ACL submission）→ detection–interruption gap：66.5%→8.3%；FP% 2.0→98.7 与 TP% 无关
- ASPIRin（arXiv 2604.10065）→ raw-token GRPO 会生成塌缩；8×V100 + LoRA r=256 + G=2 + KL β=1e-3
- MiniCPM-o 4.5（arXiv 2604.27393）→ Listen-Speak 二值 control token，`g_k=[v^k;a^k;o^k]` 帧同步，1Hz 决策
- FLEXI（arXiv 2509.22243）→ 200 条 model_interrupt，**全部含错误、无负例**
