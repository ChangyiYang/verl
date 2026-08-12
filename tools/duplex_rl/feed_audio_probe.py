#!/usr/bin/env python3
"""Step 1 探针：把一条用户音频按全双工方式喂进 MiniCPM-o 4.5，记录产出的 token 与音频。

这是 duplex RL 的第一块地基。它只做一件事、做对一件事：
    一条 wav → 按 1s 切块 → 逐块 prefill/generate → 落盘 (每帧的 token / listen 决策 / 音频)

为什么必须走 PyTorch 而不是 serving 引擎：
    RL 需要每一步动作的 logprob。vLLM-Omni 虽然已经有 MiniCPM-o 4.5 的
    experimental full-duplex runtime（/v1/duplex、/v1/realtime?duplex=1），
    但其 API 只返回文本与 base64 音频，**不暴露 token id / logits / listen-speak 状态**。
    而 MiniCPMODuplex 的采样点是
        last_id = decoder.decode(logits=...)          # logits 在手
        is_listen = (last_id == listen_token_id)      # 动作就是一个词表 token
        logits, hidden = decoder.feed(..., return_logits=True)
    所以 logprob 可直接取到。

用法：
    # 真跑（需要权重 + 1 张卡）
    python feed_audio_probe.py --audio in.wav --model openbmb/MiniCPM-o-4_5 --out runs/probe1

    # 不加载模型，仅验证切块/记录/落盘逻辑
    python feed_audio_probe.py --audio in.wav --dry-run --out runs/dry
"""

from __future__ import annotations

import argparse
import json
import time
import wave
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np

# MiniCPM-o 侧的固定量（见 modeling_minicpmo.py: self.SAMPLE_RATE = 16000）
INPUT_SR = 16000
# duplex 循环的节拍：每次 prefill 喂 1 秒音频（论文消融 0.1/0.2/1.0s，取 1.0s）
CHUNK_SECONDS = 1.0


# ---------------------------------------------------------------------------
# 记录结构
# ---------------------------------------------------------------------------

@dataclass
class FrameRecord:
    """一个 1s chunk 的完整记录 —— 这就是未来 DuplexRollout 的一个时间步。"""
    frame_idx: int
    t_start: float                      # 该 chunk 在音频中的起点（秒）
    t_end: float
    is_listen: bool                     # 模型这一帧选择"听"还是"说"
    end_of_turn: bool
    model_current_time: int | None      # 模型自报的 audio_chunk_idx，用于校验时间轴
    new_token_ids: list[int] = field(default_factory=list)   # 本帧新采样出的 token
    text: str = ""
    n_audio_samples: int = 0
    cost_prefill_s: float = 0.0
    cost_generate_s: float = 0.0


# ---------------------------------------------------------------------------
# 音频 I/O
# ---------------------------------------------------------------------------

def load_wav_mono(path: str, target_sr: int = INPUT_SR) -> np.ndarray:
    """读 wav → float32 单声道 @ target_sr，取值范围 [-1, 1]。"""
    try:
        import librosa
        wav, _ = librosa.load(path, sr=target_sr, mono=True)
        return wav.astype(np.float32)
    except ImportError:
        pass

    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        n_ch = wf.getnchannels()
        raw = wf.readframes(wf.getnframes())
    data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if n_ch > 1:
        data = data.reshape(-1, n_ch).mean(axis=1)
    if sr != target_sr:
        raise RuntimeError(
            f"{path} 采样率为 {sr}，需要 {target_sr}。请安装 librosa，或先用 ffmpeg 转换。"
        )
    return data


def write_wav(path: str, wav: np.ndarray, sr: int) -> None:
    wav = np.clip(wav, -1.0, 1.0)
    pcm = (wav * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def split_chunks(wav: np.ndarray, sr: int, chunk_seconds: float) -> list[np.ndarray]:
    """切成固定长度的块；最后不足一块的部分补零，保证节拍稳定。"""
    n = int(round(sr * chunk_seconds))
    out = []
    for start in range(0, len(wav), n):
        c = wav[start:start + n]
        if len(c) < n:
            c = np.pad(c, (0, n - len(c)))
        out.append(c)
    return out


# ---------------------------------------------------------------------------
# 模型侧
# ---------------------------------------------------------------------------

class DryRunDuplex:
    """不加载权重的替身，用来验证切块/记录/落盘全链路。

    行为：前 3 帧 listen，之后交替 speak/listen，speak 时吐几个假 token 和一段静音。
    """

    SAMPLE_RATE_OUT = 24000

    def __init__(self) -> None:
        self.total_ids: list[int] = []
        self._i = 0

    def prepare(self, **_) -> None:
        self.total_ids.clear()
        self._i = 0

    def streaming_prefill(self, audio_waveform=None, **_):
        return {"success": True}

    def streaming_generate(self, **_):
        i, self._i = self._i, self._i + 1
        is_listen = i < 3 or i % 2 == 1
        if is_listen:
            self.total_ids.append(-1)                       # 占位：listen token
            return {"is_listen": True, "text": "",
                    "audio_waveform": np.zeros(0, dtype=np.float32),
                    "end_of_turn": False, "current_time": i}
        self.total_ids.extend([1000 + i, 1001 + i, 1002 + i])
        return {"is_listen": False, "text": f"[dry-run speak @ frame {i}] ",
                "audio_waveform": np.zeros(self.SAMPLE_RATE_OUT // 2, dtype=np.float32),
                "end_of_turn": True, "current_time": i}


def build_model(args):
    """加载真实的 MiniCPMODuplex。

    注意：duplex 与单工是两套实现，不能在同一实例上切换
    （见 MiniCPM-o-Demo/core/schemas/duplex.py 的说明）。
    """
    import sys
    sys.path.insert(0, args.minicpm_repo)
    from MiniCPMO45.modeling_minicpmo import MiniCPMODuplex  # type: ignore

    model = MiniCPMODuplex.from_pretrained(
        args.model,
        device=args.device,
        ls_mode="explicit",              # 论文的 Listen-Speak 显式控制
        generate_audio=not args.no_audio,
    )
    return model


# ---------------------------------------------------------------------------
# 主循环
# ---------------------------------------------------------------------------

def run(args) -> int:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    wav = load_wav_mono(args.audio, INPUT_SR)
    chunks = split_chunks(wav, INPUT_SR, CHUNK_SECONDS)
    duration = len(wav) / INPUT_SR
    print(f"[input] {args.audio}  {duration:.2f}s  → {len(chunks)} chunks × {CHUNK_SECONDS}s")

    model = DryRunDuplex() if args.dry_run else build_model(args)
    if args.dry_run:
        print("[mode] DRY RUN —— 不加载模型，仅验证链路")

    model.prepare(
        **({} if args.dry_run else dict(
            prefix_system_prompt=args.system_prompt,
            prompt_wav_path=args.ref_wav,
        ))
    )

    records: list[FrameRecord] = []
    audio_out: list[np.ndarray] = []
    prev_n_ids = 0
    t0 = time.time()

    for i, chunk in enumerate(chunks):
        ts = time.time()
        model.streaming_prefill(audio_waveform=chunk)
        cost_prefill = time.time() - ts

        ts = time.time()
        out = model.streaming_generate(
            **({} if args.dry_run else dict(
                max_new_speak_tokens_per_chunk=args.max_speak_tokens,
                decode_mode="sampling",
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
            ))
        )
        cost_generate = time.time() - ts

        # 取出本帧新增的 token —— total_ids 是模型侧累积的完整采样序列
        all_ids = list(getattr(model, "total_ids", []))
        new_ids = all_ids[prev_n_ids:]
        prev_n_ids = len(all_ids)

        aw = out.get("audio_waveform")
        aw = np.zeros(0, dtype=np.float32) if aw is None else np.asarray(aw, dtype=np.float32)
        if aw.size:
            audio_out.append(aw)

        rec = FrameRecord(
            frame_idx=i,
            t_start=i * CHUNK_SECONDS,
            t_end=(i + 1) * CHUNK_SECONDS,
            is_listen=bool(out.get("is_listen", True)),
            end_of_turn=bool(out.get("end_of_turn", False)),
            model_current_time=out.get("current_time"),
            new_token_ids=new_ids,
            text=out.get("text", "") or "",
            n_audio_samples=int(aw.size),
            cost_prefill_s=round(cost_prefill, 4),
            cost_generate_s=round(cost_generate, 4),
        )
        records.append(rec)

        flag = "LISTEN" if rec.is_listen else "SPEAK "
        print(f"  frame {i:3d} [{rec.t_start:5.1f}s] {flag} "
              f"tok={len(new_ids):3d} audio={aw.size:6d} "
              f"prefill={cost_prefill:.3f}s gen={cost_generate:.3f}s"
              + (f'  "{rec.text.strip()[:60]}"' if rec.text.strip() else ""))

    wall = time.time() - t0

    # ---- 落盘 ----
    trace_path = out_dir / "trace.jsonl"
    with open(trace_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

    all_ids = list(getattr(model, "total_ids", []))
    (out_dir / "tokens.json").write_text(
        json.dumps({"total_ids": all_ids, "n_tokens": len(all_ids)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    out_sr = getattr(model, "SAMPLE_RATE_OUT", 24000)
    n_audio = 0
    if audio_out:
        cat = np.concatenate(audio_out)
        write_wav(str(out_dir / "output.wav"), cat, out_sr)
        n_audio = cat.size

    n_speak = sum(1 for r in records if not r.is_listen)
    rtf = wall / duration if duration > 0 else float("nan")
    summary = {
        "audio": args.audio,
        "input_duration_s": round(duration, 3),
        "n_frames": len(records),
        "n_speak_frames": n_speak,
        "n_listen_frames": len(records) - n_speak,
        "n_tokens_total": len(all_ids),
        "output_audio_samples": n_audio,
        "output_audio_seconds": round(n_audio / out_sr, 3) if n_audio else 0.0,
        "wall_seconds": round(wall, 3),
        "realtime_factor": round(rtf, 3),   # <1.0 表示快于实时
        "dry_run": args.dry_run,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n[summary] {json.dumps(summary, ensure_ascii=False)}")
    print(f"[written] {trace_path}")
    print(f"[written] {out_dir / 'tokens.json'}")
    if n_audio:
        print(f"[written] {out_dir / 'output.wav'}")

    # 时间轴自检：模型自报的 current_time 应与我们的 frame_idx 对齐
    bad = [r.frame_idx for r in records
           if r.model_current_time is not None and r.model_current_time != r.frame_idx]
    if bad:
        print(f"[WARN] current_time 与 frame_idx 不一致的帧: {bad[:10]}"
              f"{' …' if len(bad) > 10 else ''} —— 时间轴对齐需要复核")
    else:
        print("[ok] 时间轴自检通过：model.current_time == frame_idx")

    if rtf > 1.0 and not args.dry_run:
        print(f"[WARN] 实时率 {rtf:.2f} > 1.0，慢于实时；rollout 吞吐会成为瓶颈")

    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--audio", required=True, help="输入 wav（任意采样率，内部转 16k 单声道）")
    p.add_argument("--out", default="runs/probe", help="输出目录")
    p.add_argument("--model", default="openbmb/MiniCPM-o-4_5")
    p.add_argument("--minicpm-repo", default="", help="含 MiniCPMO45/ 的仓库路径")
    p.add_argument("--device", default="cuda")
    p.add_argument("--system-prompt", default=None)
    p.add_argument("--ref-wav", default=None, help="音色参考 wav")
    p.add_argument("--no-audio", action="store_true", help="只出 token 不出音频（更快）")
    p.add_argument("--max-speak-tokens", type=int, default=20,
                   help="每个 chunk 内最多采样多少 token（对齐官方默认）")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--top-p", type=float, default=0.8)
    p.add_argument("--dry-run", action="store_true", help="不加载模型，仅验证链路")
    args = p.parse_args()

    if not args.dry_run and not args.minicpm_repo:
        p.error("--minicpm-repo 必填（除非 --dry-run）：需指向含 MiniCPMO45/ 的仓库")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
