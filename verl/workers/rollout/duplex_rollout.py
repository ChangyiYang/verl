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
"""Full-duplex rollout for frame-synchronous speech models (MiniCPM-o 4.5).

设计依据（均已在 `docs/duplex_rl/runs/` 实测验证，不是推演）：

1. **全双工模型是帧同步的** —— 每帧 = 一组 token，时间 = frame_idx × 帧长（1s）。
   所以一次 duplex rollout 本质就是一条 token 序列，和普通 LLM rollout 同构。

2. **动作 token 位置固定** —— 每帧第一个 token 就是二值动作：
       LISTEN 帧: [<|listen|> 151705, </unit> 151684]
       SPEAK  帧: [<|speak|>  151706, ...内容..., <|chunk_eos|> 151718, </unit> 151684]
   ⇒ 要做 credit assignment 的位置一抓一个准。

3. **训练时用户音频是预录的** ⇒ 不需要真正的双向实时流：
       用户流 teacher-force（逐帧喂已知音频），模型流 sample。
   真 live duplex 只有部署才需要。

4. **流式 rollout 与整段批量前向数值等价**（fp32 实测：logits 差 2.1e-4，
   log P(listen) 差 4.6e-5，argmax 31/31 一致）⇒ 训练侧可用「录下的 embeds →
   单次批量前向」重算 logprob。因此本 rollout **把每次 feed 的 embeds 一并输出**。

5. `StreamDecoder.feed` 未加 `@torch.no_grad()`（只有 `decode` 加了）⇒ 梯度可通。

⚠️ **训练侧构造输入的正确方式（容易踩的坑）**

HF 模型二选一：`Qwen3Model.forward` 里明确
`raise ValueError("You must specify exactly one of input_ids or inputs_embeds")`，
**不能同时传 input_ids 和 inputs_embeds**。

但**不能把 `duplex_embeds` 原样当常量喂进去** —— 序列里文本 token 位的 embeds
也是 rollout 时由 `embed_token()`（即 `model.embed_tokens` 查表）算好录下来的，
原样喂 ⇒ **embedding 矩阵拿不到梯度，且是静默冻结、不会报错**。

正确做法是用本类产出的两个字段**拼**出唯一的 `inputs_embeds`：

    emb = duplex_embeds.clone()                   # 音频位：常量（免去重跑音频编码器）
    m   = duplex_token_ids >= 0                   # 文本位有 id，条件位为 -1
    emb[m] = model.model.embed_tokens(duplex_token_ids[m])   # 文本位：可导
    logits = model(inputs_embeds=emb, position_ids=...)

⇒ 这就是本类**必须同时输出 token_ids 与 embeds** 的原因：不是都传给模型，
   而是用来**拼**模型的输入。
建议常开一条断言：第 0 步重建出的文本位 embeds 应与录下的逐比特相同；
若不同，说明 rollout 与训练的权重/精度已经不一致。

注意：rollout 期间记录的 logprob **仅供诊断**，不作为 `old_log_probs`。
verl 会用 actor 的训练前向重算（`ray_trainer.py` 的 `compute_log_prob`），
这样 ratio 在第 0 步严格为 1，与 rollout 精度无关（bf16 下二者可差 0.19 nats）。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from tensordict import TensorDict

from verl import DataProto
from verl.utils.device import get_device_id, get_device_name
from verl.workers.rollout.base import BaseRollout
from verl.workers.rollout.replica import TokenOutput

__all__ = ["DuplexRollout", "DuplexFrame", "DuplexTrajectory", "pack_duplex_payloads"]

logger = logging.getLogger(__name__)

# 输入音频采样率，见 modeling_minicpmo.py 的 SAMPLE_RATE
INPUT_SR = 16000
# 序列中「非 token 位」（音频嵌入、系统提示嵌入等条件输入）的占位 id
PAD_ID = -1


@dataclass
class DuplexFrame:
    """一个 1s chunk 的记录 —— rollout 的一个时间步。"""

    frame_idx: int
    t_start: float
    t_end: float
    is_listen: bool
    action_token_id: int
    action_seq_pos: int          # 动作 token 在整条序列中的绝对位置
    token_ids: list[int] = field(default_factory=list)
    text: str = ""
    end_of_turn: bool = False


@dataclass
class DuplexTrajectory:
    frames: list[DuplexFrame]
    token_ids: list[int]
    embeds: torch.Tensor          # [T, H] 完整输入嵌入序列（含音频与文本）
    seq_token_ids: torch.Tensor   # [T] 每个位置的 token id；音频等条件位为 PAD_ID
    response_mask: torch.Tensor   # [T] 1 = 模型生成的 token（可训练），0 = 条件输入
    action_positions: torch.Tensor  # [n_frames] 每帧动作 token 的绝对位置


def pack_duplex_payloads(payloads: list[dict[str, Any]]) -> TensorDict:
    """Pad serialized trajectories into the actor/reward DataProto schema."""
    if not payloads:
        raise ValueError("Cannot pack an empty duplex rollout batch")
    batch_size = len(payloads)
    max_length = max(int(payload["embeds"].shape[0]) for payload in payloads)
    hidden_size = int(payloads[0]["embeds"].shape[-1])
    max_frames = max(len(payload["frames"]) for payload in payloads)
    dtype = payloads[0]["embeds"].dtype

    embeds = torch.zeros(batch_size, max_length, hidden_size, dtype=dtype)
    attention_mask = torch.zeros(batch_size, max_length, dtype=torch.int32)
    position_ids = torch.zeros(batch_size, max_length, dtype=torch.long)
    sequence_action_mask = torch.zeros(batch_size, max_length, dtype=torch.int32)
    token_ids = torch.full((batch_size, max_length), PAD_ID, dtype=torch.long)
    action_pos = torch.full((batch_size, max_frames), -1, dtype=torch.long)
    is_listen = torch.zeros(batch_size, max_frames, dtype=torch.int32)
    frame_time = torch.zeros(batch_size, max_frames, dtype=torch.float32)

    for batch_index, payload in enumerate(payloads):
        sample_embeds = payload["embeds"]
        sample_token_ids = payload["seq_token_ids"]
        sample_action_positions = payload["action_positions"]
        length = int(sample_embeds.shape[0])
        embeds[batch_index, :length] = sample_embeds
        attention_mask[batch_index, :length] = 1
        position_ids[batch_index, :length] = torch.arange(length)
        token_ids[batch_index, :length] = sample_token_ids
        if sample_action_positions.numel():
            action_pos[batch_index, : sample_action_positions.numel()] = sample_action_positions
            sequence_action_mask[batch_index, sample_action_positions] = 1
        frames = payload["frames"]
        if frames:
            is_listen[batch_index, : len(frames)] = torch.tensor(
                [int(frame["is_listen"]) for frame in frames], dtype=torch.int32
            )
            frame_time[batch_index, : len(frames)] = torch.tensor(
                [float(frame["t_start"]) for frame in frames], dtype=torch.float32
            )

    safe_input_ids = token_ids.masked_fill(token_ids < 0, 0)
    action_response_pos = action_pos - 1
    action_response_pos.masked_fill_(action_pos <= 0, -1)
    return TensorDict(
        {
            "prompts": safe_input_ids[:, :1],
            "responses": safe_input_ids[:, 1:],
            "input_ids": safe_input_ids,
            "attention_mask": attention_mask,
            "response_mask": sequence_action_mask[:, 1:],
            "position_ids": position_ids,
            "duplex_embeds": embeds,
            "duplex_token_ids": token_ids,
            "duplex_action_pos": action_pos,
            "duplex_action_response_pos": action_response_pos,
            "duplex_is_listen": is_listen,
            "duplex_frame_time": frame_time,
        },
        batch_size=batch_size,
    )


class DuplexRollout(BaseRollout):
    """把预录用户音频逐帧喂进全双工模型，采样模型的 listen/speak 行为。

    与普通 rollout 的本质差别：
      - 普通：`generate(prompt) -> response` 一次
      - 本类：每秒一轮 `streaming_prefill` + `streaming_generate`，N 秒音频 = N 轮

    产出的 DataProto 除 verl 标准字段外，另带 duplex 专用字段（见 `generate_sequences`）。
    """

    def __init__(self, module=None, config=None, tokenizer=None, **kwargs):
        model_config = kwargs.pop("model_config", None)
        device_mesh = kwargs.pop("device_mesh", None)
        super().__init__(config=config, model_config=model_config, device_mesh=device_mesh)
        if module is None:
            module = self._load_duplex_module(model_config)
        self.module = module              # MiniCPMODuplex（由 MiniCPMO.as_duplex() 得到）
        self.config = config
        self.tokenizer = tokenizer or getattr(module, "tokenizer", None)

        self.listen_id = int(module.listen_token_id)
        self.speak_id = int(self.tokenizer.convert_tokens_to_ids("<|speak|>"))
        duplex_config = getattr(config, "duplex", {}) or {}
        self.chunk_seconds = float(duplex_config.get("chunk_seconds", getattr(config, "chunk_seconds", 1.0)))
        self.global_steps = 0
        self._update_in_progress = False
        self._weights_valid = True
        self._released = False
        self._last_audit_payload: dict[str, Any] | None = None

    def _load_duplex_module(self, model_config):
        """Load an independent official-HF model for rollout workers."""
        if model_config is None:
            raise RuntimeError("Distributed DuplexRollout construction requires model_config")

        from copy import deepcopy

        from transformers import AutoModel

        hf_config = deepcopy(model_config.hf_config)
        # The RL policy uses the audio encoder and thinker. Vision and TTS do
        # not contribute to LS/text policy logits and needlessly duplicate
        # their weights in the colocated rollout copy.
        hf_config.init_vision = bool(self.config.duplex.get("init_vision", False))
        hf_config.init_tts = bool(self.config.duplex.get("init_tts", False))
        dtype_name = str(getattr(self.config, "dtype", "bfloat16"))
        dtype = getattr(torch, dtype_name)
        device = torch.device(get_device_name(), get_device_id())

        model = AutoModel.from_pretrained(
            model_config.local_path,
            config=hf_config,
            trust_remote_code=model_config.trust_remote_code,
            torch_dtype=dtype,
        ).eval().to(device)

        # MiniCPMODuplex.from_existing_model currently initializes TTS even
        # when generate_audio=False. Suppress that optional branch without
        # modifying the remote-code package.
        original_init_tts = model.init_tts
        model.init_tts = lambda **_kwargs: None
        try:
            return model.as_duplex(
                device=str(device),
                generate_audio=False,
                ls_mode="explicit",
                **dict(self.config.duplex.get("runtime_kwargs", {})),
            )
        finally:
            model.init_tts = original_init_tts

    # ------------------------------------------------------------------
    # BaseRollout lifecycle.
    # ------------------------------------------------------------------
    async def resume(self, tags: list[str]):
        self._released = False

    @staticmethod
    def _normalize_weight_name(name: str) -> str:
        prefixes = ("module.", "_fsdp_wrapped_module.")
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if name.startswith(prefix):
                    name = name[len(prefix) :]
                    changed = True
        return name

    @torch.no_grad()
    async def update_weights(self, weights, global_steps: int | None = None, wire_format="named_tensors", **kwargs):
        """Copy a complete HF-keyed state stream into the rollout model.

        Failure is intentionally non-transactional at the tensor-storage
        level, but it is transactional at the policy level: generation remains
        disabled until a later complete synchronization succeeds.
        """
        if wire_format != "named_tensors":
            raise ValueError(f"Duplex rollout requires named_tensors, got {wire_format!r}")
        if self._update_in_progress:
            raise RuntimeError("A duplex weight update is already in progress")

        self._update_in_progress = True
        self._weights_valid = False
        target = self.module.model.state_dict(keep_vars=True)
        has_transformer_weights = any(".layers." in name and name.endswith(".weight") for name in target)
        seen: set[str] = set()
        changed_transformer_weight = None
        try:
            for source_name, source in weights:
                name = self._normalize_weight_name(source_name)
                if name in seen:
                    raise RuntimeError(f"Duplicate duplex rollout weight: {name}")
                if name not in target:
                    raise RuntimeError(f"Unexpected duplex rollout weight: {source_name} -> {name}")
                destination = target[name]
                if tuple(destination.shape) != tuple(source.shape):
                    raise RuntimeError(
                        f"Shape mismatch for {name}: rollout={tuple(destination.shape)}, actor={tuple(source.shape)}"
                    )
                if destination.dtype != source.dtype:
                    raise RuntimeError(
                        f"Dtype mismatch for {name}: rollout={destination.dtype}, actor={source.dtype}"
                    )
                if (
                    changed_transformer_weight is None
                    and global_steps is not None
                    and int(global_steps) > 0
                    and ".layers." in name
                    and name.endswith(".weight")
                    and not torch.equal(destination, source)
                ):
                    changed_transformer_weight = name
                with torch.no_grad():
                    destination.copy_(source, non_blocking=True)
                if not torch.equal(destination, source):
                    raise RuntimeError(f"Post-copy duplex rollout weight mismatch: {name}")
                seen.add(name)

            missing = sorted(set(target) - seen)
            if missing:
                preview = ", ".join(missing[:8])
                raise RuntimeError(f"Missing {len(missing)} duplex rollout weights; first: {preview}")
            if (
                global_steps is not None
                and int(global_steps) > 0
                and has_transformer_weights
                and changed_transformer_weight is None
            ):
                raise RuntimeError("Actor update changed no ordinary transformer weight before duplex synchronization")

            self.clear_kv_cache()
            if global_steps is not None:
                self.global_steps = int(global_steps)
            self._weights_valid = True
            print(
                f"[duplex] full_weight_sync version={self.global_steps} tensors={len(seen)} "
                f"changed_transformer={changed_transformer_weight or 'initial_sync'}"
            )
        finally:
            self._update_in_progress = False

    async def release(self):
        self.clear_kv_cache()
        self._released = True

    def clear_kv_cache(self):
        model = getattr(self.module, "model", None)
        if model is not None:
            if hasattr(model, "reset_session"):
                model.reset_session(reset_token2wav_cache=True)
            if hasattr(model, "audio_past_key_values"):
                model.audio_past_key_values = None
        decoder = getattr(self.module, "decoder", None)
        if decoder is not None and hasattr(decoder, "reset"):
            decoder.reset()
        processor = getattr(self.module, "processor", None)
        if processor is not None and hasattr(processor, "reset_streaming"):
            processor.reset_streaming()

    @staticmethod
    def _input_embeddings(model):
        unwrapped = getattr(model, "_fsdp_wrapped_module", model)
        llm = getattr(unwrapped, "llm", None)
        if llm is not None:
            return llm.get_input_embeddings()
        return unwrapped.get_input_embeddings()

    @torch.no_grad()
    def audit_against_actor(self, actor_module) -> dict[str, float] | None:
        """Compare fixed-prefix LS logits after an actor-to-rollout sync."""
        payload = self._last_audit_payload
        if payload is None:
            return None
        action_positions = payload["action_positions"]
        if not action_positions.numel():
            raise RuntimeError("Cannot audit duplex sync without an LS action position")
        prefix_length = int(action_positions[0])
        if prefix_length <= 0:
            raise RuntimeError(f"Invalid duplex audit prefix length: {prefix_length}")

        device = next(actor_module.parameters()).device
        recorded = payload["embeds"][:prefix_length].to(device=device)
        token_ids = payload["seq_token_ids"][:prefix_length].to(device=device)
        token_mask = token_ids >= 0
        attention_mask = torch.ones(1, prefix_length, dtype=torch.long, device=device)
        position_ids = torch.arange(prefix_length, device=device).unsqueeze(0)

        def _ls_logits(model, call):
            embeds = recorded.clone()
            if token_mask.any():
                embeds[token_mask] = self._input_embeddings(model)(token_ids[token_mask])
            output = call(
                inputs_embeds=embeds.unsqueeze(0),
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
            )
            return output.logits[0, -1, [self.listen_id, self.speak_id]].float().cpu()

        was_training = actor_module.training
        actor_module.eval()
        try:
            actor_logits = _ls_logits(actor_module, actor_module)
        finally:
            actor_module.train(was_training)
        rollout_model = self.module.model
        rollout_logits = _ls_logits(rollout_model, rollout_model.llm)
        max_abs_diff = float((actor_logits - rollout_logits).abs().max())
        # The FSDP mixed-precision wrapper and the plain HF module may round
        # one bf16 ULP differently even with bit-identical weights. At the
        # observed LS logit scale one ULP is 0.03125, so allow two ULPs.
        if max_abs_diff > 0.0625:
            raise RuntimeError(
                f"Post-sync actor/rollout LS logits diverged: max_abs_diff={max_abs_diff}, "
                f"actor={actor_logits.tolist()}, rollout={rollout_logits.tolist()}"
            )
        result = {"max_abs_diff": max_abs_diff, "prefix_length": float(prefix_length)}
        print(
            f"[duplex] sync_logit_audit version={self.global_steps} "
            f"prefix={prefix_length} max_abs_diff={max_abs_diff:.6g}"
        )
        return result

    @staticmethod
    def trajectory_to_payload(trajectory: DuplexTrajectory) -> dict[str, Any]:
        return {
            "embeds": trajectory.embeds.detach().cpu(),
            "seq_token_ids": trajectory.seq_token_ids.detach().cpu(),
            "response_mask": trajectory.response_mask.detach().cpu(),
            "action_positions": trajectory.action_positions.detach().cpu(),
            "frames": [
                {
                    "frame_idx": frame.frame_idx,
                    "t_start": frame.t_start,
                    "t_end": frame.t_end,
                    "is_listen": frame.is_listen,
                    "action_token_id": frame.action_token_id,
                    "action_seq_pos": frame.action_seq_pos,
                    "token_ids": frame.token_ids,
                    "text": frame.text,
                    "end_of_turn": frame.end_of_turn,
                }
                for frame in trajectory.frames
            ],
        }

    @torch.no_grad()
    async def generate(
        self,
        request_id: str,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data=None,
        video_data=None,
        audio_data=None,
        mm_processor_kwargs=None,
        system_prompt: str | None = None,
        **kwargs,
    ) -> TokenOutput:
        """Run one real-audio streaming trajectory for the async server path."""
        del request_id, prompt_ids, sampling_params
        if self._update_in_progress or not self._weights_valid:
            raise RuntimeError("Duplex rollout weights are updating or invalid")
        if self._released:
            await self.resume(tags=["weights", "kv_cache"])
        if audio_data is None or len(audio_data) != 1:
            raise ValueError("Duplex generate expects exactly one mono waveform in audio_data")
        if image_data is not None or video_data is not None:
            raise ValueError("Duplex generate does not accept image or video inputs")
        if mm_processor_kwargs:
            raise ValueError("Duplex generate does not accept multimodal processor kwargs")
        force_first_action = kwargs.pop("force_first_action", None)
        if kwargs:
            raise TypeError(f"Unexpected duplex generate kwargs: {sorted(kwargs)}")
        if force_first_action == "listen":
            force_first_action_id = self.listen_id
        elif force_first_action == "speak":
            force_first_action_id = self.speak_id
        elif force_first_action is None:
            force_first_action_id = None
        else:
            raise ValueError(f"Unknown force_first_action={force_first_action!r}")
        with torch.no_grad():
            trajectory = self._rollout_one(
                np.asarray(audio_data[0], dtype=np.float32),
                system_prompt,
                force_first_action_id=force_first_action_id,
            )
        payload = self.trajectory_to_payload(trajectory)
        self._last_audit_payload = payload
        return TokenOutput(
            token_ids=trajectory.token_ids,
            stop_reason="completed",
            extra_fields={
                "duplex_trajectory": payload,
                "global_steps": self.global_steps,
            },
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _split_chunks(wav: np.ndarray, chunk_seconds: float) -> list[np.ndarray]:
        """切成等长块；末块补零以保持节拍稳定。"""
        n = int(round(INPUT_SR * chunk_seconds))
        out = []
        for s in range(0, len(wav), n):
            c = wav[s:s + n]
            if len(c) < n:
                c = np.pad(c, (0, n - len(c)))
            out.append(c)
        return out

    def _rollout_one(
        self,
        wav: np.ndarray,
        system_prompt: str | None = None,
        force_first_action_id: int | None = None,
    ) -> DuplexTrajectory:
        """对一条音频跑完整的逐帧循环，并把嵌入序列一并录下。"""
        duplex = self.module
        dec = duplex.decoder

        fed: list[torch.Tensor] = []
        cursor = {"len": 0}
        # 记录每个 decode 决策：(该 token 的落位, 采样出的 token id)
        decision_marks: list[int] = []
        decision_tokens: list[int] = []

        orig_feed, orig_decode = dec.feed, dec.decode

        def traced_feed(embeds, return_logits=False):
            fed.append(embeds.detach())
            cursor["len"] += int(embeds.size(0))
            return orig_feed(embeds, return_logits=return_logits)

        forced = {"pending": force_first_action_id is not None}

        def traced_decode(logits, *a, **kw):
            decision_marks.append(cursor["len"])
            if forced["pending"]:
                forced["pending"] = False
                tok = torch.tensor([force_first_action_id], device=logits.device, dtype=torch.long)
            else:
                tok = orig_decode(logits, *a, **kw)
            decision_tokens.append(int(tok.item()) if hasattr(tok, "item") else int(tok))
            return tok

        dec.feed, dec.decode = traced_feed, traced_decode
        try:
            # MiniCPMODuplex.prepare() resets the streaming processor and LLM
            # decoder, but the current official implementation does not clear
            # the audio encoder KV cache. Without this reset, sample b+1 in a
            # serial batch is conditioned on sample b's audio.
            model = getattr(duplex, "model", None)
            if model is not None and hasattr(model, "audio_past_key_values"):
                model.audio_past_key_values = None
            prep = {"prefix_system_prompt": system_prompt} if system_prompt else {}
            duplex.prepare(**prep)
            prompt_len = cursor["len"]      # 系统提示等条件输入的长度

            frames: list[DuplexFrame] = []
            prev_n_tok = 0
            for i, chunk in enumerate(self._split_chunks(wav, self.chunk_seconds)):
                n_marks_before = len(decision_marks)
                duplex.streaming_prefill(audio_waveform=chunk)
                out = duplex.streaming_generate()

                all_tok = list(getattr(duplex, "total_ids", []))
                new_tok = all_tok[prev_n_tok:]
                prev_n_tok = len(all_tok)

                # 本帧第一个决策点 = 动作 token；它被 feed 进去的位置即该 mark
                act_pos = decision_marks[n_marks_before] if len(decision_marks) > n_marks_before else -1
                frames.append(DuplexFrame(
                    frame_idx=i,
                    t_start=i * self.chunk_seconds,
                    t_end=(i + 1) * self.chunk_seconds,
                    is_listen=bool(out.get("is_listen", True)),
                    action_token_id=new_tok[0] if new_tok else -1,
                    action_seq_pos=act_pos,
                    token_ids=new_tok,
                    text=out.get("text", "") or "",
                    end_of_turn=bool(out.get("end_of_turn", False)),
                ))
        finally:
            dec.feed, dec.decode = orig_feed, orig_decode

        embeds = torch.cat(fed, dim=0) if fed else torch.empty(0)
        T = int(embeds.size(0))

        # response_mask：模型生成的 token 位置为 1，音频/系统提示等条件输入为 0。
        # 每个 decode 决策产生的 token 会在下一次 feed 中进入序列，
        # 因此 mark 处（0-based 即 mark）就是该 token 的落位。
        response_mask = torch.zeros(T, dtype=torch.int32)
        # 位置 → token id；音频等条件位没有 token，填 PAD_ID(-1)
        seq_token_ids = torch.full((T,), PAD_ID, dtype=torch.long)
        for m, tid in zip(decision_marks, decision_tokens):
            if 0 <= m < T:
                response_mask[m] = 1
                seq_token_ids[m] = tid

        action_positions = torch.tensor(
            [f.action_seq_pos for f in frames if f.action_seq_pos >= 0], dtype=torch.long
        )
        return DuplexTrajectory(
            frames=frames,
            token_ids=list(getattr(duplex, "total_ids", [])),
            embeds=embeds,
            seq_token_ids=seq_token_ids,
            response_mask=response_mask,
            action_positions=action_positions,
        )

    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """对 batch 内每条音频独立跑一次 duplex rollout。

        输入：`prompts.non_tensor_batch["audios"]` —— 每条为 16kHz 单声道 float32 波形。
        （duplex 实例带流式状态，一个实例同时只能跑一条 session，故此处按样本串行。
          实测稳态约 15× 实时，串行不构成瓶颈。）

        输出 DataProto，除 verl 标准字段外另含：
          duplex_embeds        [B, T, H]  完整输入嵌入序列（训练侧据此重算 logprob）
          duplex_token_ids     [B, T]     每位置的 token id；条件位（音频等）为 -1
          duplex_action_pos    [B, F]     每帧动作 token 的绝对位置（-1 为 padding）
          duplex_action_response_pos [B,F] 动作在 verl response/reward 张量中的位置
          duplex_is_listen     [B, F]     每帧动作（1=listen, 0=speak）
          duplex_frame_time    [B, F]     每帧起点的 wall-clock 秒 —— 供窗口 reward 用
        """
        audios = prompts.non_tensor_batch["audios"]
        system_prompt = prompts.meta_info.get("system_prompt")

        trajs = [self._rollout_one(np.asarray(a, dtype=np.float32), system_prompt) for a in audios]

        B = len(trajs)
        T = max(int(t.embeds.size(0)) for t in trajs)
        H = int(trajs[0].embeds.size(-1))
        F = max(len(t.frames) for t in trajs)
        dev, dtype = trajs[0].embeds.device, trajs[0].embeds.dtype

        embeds = torch.zeros(B, T, H, device=dev, dtype=dtype)
        attention_mask = torch.zeros(B, T, dtype=torch.int32, device=dev)
        sequence_action_mask = torch.zeros(B, T, dtype=torch.int32, device=dev)
        position_ids = torch.zeros(B, T, dtype=torch.long, device=dev)
        seq_tokens = torch.full((B, T), PAD_ID, dtype=torch.long, device=dev)
        action_pos = torch.full((B, F), -1, dtype=torch.long, device=dev)
        is_listen = torch.zeros(B, F, dtype=torch.int32, device=dev)
        frame_time = torch.zeros(B, F, dtype=torch.float32, device=dev)

        for b, t in enumerate(trajs):
            L = int(t.embeds.size(0))
            embeds[b, :L] = t.embeds
            attention_mask[b, :L] = 1
            if t.action_positions.numel():
                sequence_action_mask[b, t.action_positions.to(dev)] = 1
            position_ids[b, :L] = torch.arange(L, device=dev)
            seq_tokens[b, :L] = t.seq_token_ids.to(dev)
            n = len(t.frames)
            if t.action_positions.numel():
                action_pos[b, :t.action_positions.numel()] = t.action_positions.to(dev)
            is_listen[b, :n] = torch.tensor([int(f.is_listen) for f in t.frames],
                                            dtype=torch.int32, device=dev)
            frame_time[b, :n] = torch.tensor([f.t_start for f in t.frames],
                                             dtype=torch.float32, device=dev)

        # Duplex has no natural prompt/response boundary, but the PPO stack
        # requires the standard fields. Treat position 0 as a one-token causal
        # seed and positions 1..T-1 as the response. Conditioning positions use
        # a harmless placeholder id; the actor takes their embeddings from
        # duplex_embeds instead of looking this id up.
        safe_input_ids = seq_tokens.masked_fill(seq_tokens < 0, 0)
        action_response_pos = action_pos - 1
        action_response_pos.masked_fill_(action_pos <= 0, -1)

        batch = TensorDict(
            {
                "prompts": safe_input_ids[:, :1],
                "responses": safe_input_ids[:, 1:],
                "input_ids": safe_input_ids,
                "attention_mask": attention_mask,
                "response_mask": sequence_action_mask[:, 1:],
                "position_ids": position_ids,
                "duplex_embeds": embeds,
                "duplex_token_ids": seq_tokens,
                "duplex_action_pos": action_pos,
                "duplex_action_response_pos": action_response_pos,
                "duplex_is_listen": is_listen,
                "duplex_frame_time": frame_time,
            },
            batch_size=B,
        )
        meta = dict(prompts.meta_info)
        meta.update({
            "duplex": True,
            "listen_token_id": self.listen_id,
            "speak_token_id": self.speak_id,
            "chunk_seconds": self.chunk_seconds,
        })
        return DataProto(batch=batch, meta_info=meta)
