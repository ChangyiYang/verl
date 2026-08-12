"""Duplex RL — 接口契约参考（design reference ONLY）

⚠️ 本文件不被任何代码 import，也不会被执行。它的唯一作用是把 PLAN.md 里的
   三个新模块的**输入/输出契约**钉死，方便 review 时对齐认知。
   真正的实现落在 PLAN.md §2 指定的路径下。

核心不变量（整个设计建立在这条上）：
    全双工模型是帧同步的 ⇒ 时间与 token 位置是**固定映射**：
        t_seconds = frame_index / frame_rate
        frame_index = token_index // tokens_per_frame
    因此 "在时间窗 [t_s, t_e] 上给 reward" 可以无损地表达成
    "在 token index 区间 [i_s, i_e] 上给 reward"，
    而 verl 的 token_level_scores 张量本来就支持逐 token 赋值。
"""

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# 1) frame_align —— 帧 / 秒 / token index 三者互转
#    落地路径: verl/utils/duplex/frame_align.py
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FrameSpec:
    """描述一个全双工模型的时间栅格。

    frame_rate:       每秒多少帧。MiniCPM-o 4.5 决策粒度 1Hz；Moshi 为 12.5Hz。
    tokens_per_frame: 每帧产生多少个 token（含 control token + 内容/音频 token）。
    """
    frame_rate: float
    tokens_per_frame: int


def sec_to_frame(t: float, spec: FrameSpec) -> int:
    """秒 → 帧索引（向下取整）。"""
    raise NotImplementedError


def frame_to_sec(frame: int, spec: FrameSpec) -> float:
    """帧索引 → 该帧起点的秒。

    验收 2.1 要求 frame -> sec -> frame 往返完全相等。
    """
    raise NotImplementedError


def frame_to_token_span(frame: int, spec: FrameSpec) -> tuple[int, int]:
    """帧索引 → 该帧对应的 token index 半开区间 [start, end)。"""
    raise NotImplementedError


def asr_words_to_frames(word_timestamps: list[tuple[str, float, float]],
                        spec: FrameSpec) -> list[tuple[str, int, int]]:
    """ASR 词级时间戳 → 帧区间。

    用于把 MIB 的 error window 对齐到帧栅格：
    error window = [t_s, t_e + Δ]，其中 t_s/t_e 是"错误步"被说出的起止时间，
    Δ = 10s 容差。
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# 2) WindowRewardManager —— 时间窗 → token_level_scores
#    落地路径: verl/workers/reward_manager/window.py
# ---------------------------------------------------------------------------

@dataclass
class RewardWindow:
    """一个带时间范围的 reward 项。

    t_start / t_end: wall-clock 秒，相对该条 rollout 的起点。
    value:           填入该区间内每个 token 的分值（可正可负）。
    term:            reward 项名（'trigger' / 'timing' / 'cost' / ...），仅用于日志与分解统计。
    """
    t_start: float
    t_end: float
    value: float
    term: str


class WindowRewardManagerContract:
    """把一组时间窗展开成形状 == responses 的 token 级张量。

    与 verl 现有机制的衔接：
      - verl/workers/reward_manager/naive.py:54 已经在构造
            reward_tensor = torch.zeros_like(data.batch["responses"], dtype=float32)
        naive 只在最后一个有效位置填标量；我们改成按 token index 区间填。
      - verl/trainer/ppo/ray_trainer.py:103 消费 data.batch["token_level_scores"]，
        随后减去 KL：token_level_rewards = token_level_scores - beta * kld
      ⇒ 因此**不需要改动 verl 核心**，只是换一个 reward manager。

    验收 2.2：给定合成窗口，填中的 token index 集合必须与手算完全一致，且窗口外恒为 0。
    """

    def __call__(self, data, windows_per_sample: list[list[RewardWindow]]):
        """返回 token_level_scores，形状与 data.batch['responses'] 相同。"""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# 3) DuplexRollout —— 逐帧推进的 rollout
#    落地路径: verl/workers/rollout/duplex_rollout.py
#    注册: verl/workers/rollout/base.py 的 _ROLLOUT_REGISTRY 加一行
# ---------------------------------------------------------------------------

class DuplexRolloutContract:
    """继承 verl 的 BaseRollout（workers/rollout/base.py）。

    关键简化 —— 训练时用户音频是**预录**的，所以不需要真正的双向实时流：
        用户流：teacher-force（把预录音频的 token 逐帧喂进去，已知）
        模型流：sample（逐帧采样 Listen-Speak control token；决定说话时再采内容 token）
    这退化成"带外部条件输入的逐帧自回归采样循环"，纯 PyTorch 即可。
    真正的 live duplex 只有部署阶段需要。

    generate_sequences(prompts: DataProto) -> DataProto，其中输出至少包含：
        response_ids   : 交错后的 token 流
        response_mask  : 1 = 模型发声帧，0 = 倾听帧
        frame_times    : 每帧起点的 wall-clock 秒（供 WindowRewardManager 用）
        ls_decisions   : 每个决策点的 listen/speak 取值（便于诊断与做 MIB 子类型归因）

    验收 2.3：三者形状一致；response_mask 的 speak 段必须与实际采样出的发声帧对齐。
    """

    def generate_sequences(self, prompts):
        raise NotImplementedError
