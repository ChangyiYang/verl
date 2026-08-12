#!/usr/bin/env python3
"""Step 2 探针：验证能否取到 listen/speak 决策的 logprob —— RL 的命门。

背景：官方 `decoder.decode()`（utils.py:2106）**不是**一个普通的 categorical 采样，
而是两段式的：

    Stage 0: 用 **未经温度缩放** 的 softmax(logits) 采一次，只为判断是否 chunk_eos；
             命中 chunk_eos 就直接返回。
    Stage 1: 否则 → 屏蔽 forbidden → 重复惩罚 → `logits[listen_id] *= listen_prob_scale`
             （注意是乘在 **logit** 上，不是概率上，与 docstring 措辞不符）
             → 若 listen_top_k 且 listen 排名进前 k，**确定性**返回 listen
             → 否则 logits/temperature → top-k/top-p 过滤 → softmax → multinomial

所以"实际行为策略"的密度是个混合分布：
    P(tok) = P₀(eos)·1[tok=eos] + (1−P₀(eos))·P₁(tok)
直接对原始 logits 做 log_softmax 得到的 **不是** 真实采样分布——拿它当 old_logprob，
importance ratio 就是错的。这正是本脚本要量化的问题。

本脚本做三件事：
  1. 用 monkeypatch 截获每次 decode 的入参 logits 与返回 token（不改官方实现）
  2. 对每帧的**首个决策点**记录多种口径的概率，横向对比
  3. 输出一份可直接看的报告，回答"logprob 到底能不能拿、该怎么定义"

用法（需 1 张卡）：
    python logprob_probe.py --audio user10s.wav --model openbmb/MiniCPM-o-4_5 --out runs/lp
"""

from __future__ import annotations

import argparse
import json
import wave
from pathlib import Path

import numpy as np

INPUT_SR = 16000
CHUNK_SECONDS = 1.0


def load_wav_mono(path: str) -> np.ndarray:
    with wave.open(path, "rb") as wf:
        sr, n_ch = wf.getframerate(), wf.getnchannels()
        x = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    if n_ch > 1:
        x = x.reshape(-1, n_ch).mean(axis=1)
    if sr != INPUT_SR:
        raise RuntimeError(f"需要 {INPUT_SR}Hz，实际 {sr}Hz —— 请先重采样")
    return x


def split_chunks(x: np.ndarray) -> list[np.ndarray]:
    n = int(round(INPUT_SR * CHUNK_SECONDS))
    out = []
    for s in range(0, len(x), n):
        c = x[s:s + n]
        if len(c) < n:
            c = np.pad(c, (0, n - len(c)))
        out.append(c)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--model", default="openbmb/MiniCPM-o-4_5")
    ap.add_argument("--out", default="runs/logprob")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--temperature", type=float, default=0.7)
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F
    from transformers import AutoModel

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    chunks = split_chunks(load_wav_mono(args.audio))
    print(f"[input] {args.audio} → {len(chunks)} chunks")

    print(f"[load] {args.model} …")
    model = AutoModel.from_pretrained(
        args.model, trust_remote_code=True,
        torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    ).eval().to(args.device)
    duplex = model.as_duplex(device=args.device, generate_audio=False, ls_mode="explicit")

    dec = duplex.decoder
    listen_id = duplex.listen_token_id
    speak_id = duplex.tokenizer.convert_tokens_to_ids("<|speak|>")
    eos_id = getattr(dec, "chunk_eos_id", None)
    print(f"[ids] listen={listen_id} speak={speak_id} chunk_eos={eos_id}")

    captured: list[dict] = []
    orig_decode = dec.decode

    def traced_decode(logits, *a, **kw):
        """截获真实决策点：记下入参 logits，再让官方实现照常跑。"""
        lg = logits.detach().float()[0]            # [vocab]
        tok = orig_decode(logits, *a, **kw)
        tid = int(tok.item()) if hasattr(tok, "item") else int(tok)

        # 口径 A：原始 logits 的全词表 log_softmax（= Stage 0 用的分布）
        logp_raw_full = F.log_softmax(lg, dim=-1)
        # 口径 B：只在 {listen, speak} 两个动作上归一化 —— 我们真正要训的二值策略
        pair = torch.stack([lg[listen_id], lg[speak_id]])
        logp_pair = F.log_softmax(pair, dim=-1)
        # 口径 C：加温度后再在两个动作上归一化
        logp_pair_T = F.log_softmax(pair / args.temperature, dim=-1)

        captured.append({
            "sampled_id": tid,
            "is_listen": tid == listen_id,
            "is_speak": tid == speak_id,
            "logit_listen": float(lg[listen_id]),
            "logit_speak": float(lg[speak_id]),
            "logp_raw_listen": float(logp_raw_full[listen_id]),
            "logp_raw_speak": float(logp_raw_full[speak_id]),
            "logp_raw_sampled": float(logp_raw_full[tid]),
            "p_pair_listen": float(logp_pair[0].exp()),
            "p_pair_listen_T": float(logp_pair_T[0].exp()),
            "logp_pair_sampled": (float(logp_pair[0]) if tid == listen_id
                                  else float(logp_pair[1]) if tid == speak_id else None),
            "p0_eos": float(F.softmax(lg, dim=-1)[eos_id]) if eos_id is not None else None,
            "entropy_full": float(-(logp_raw_full.exp() * logp_raw_full).sum()),
        })
        return tok

    dec.decode = traced_decode
    duplex.prepare()

    frames = []
    for i, ch in enumerate(chunks):
        before = len(captured)
        duplex.streaming_prefill(audio_waveform=ch)
        out = duplex.streaming_generate()
        steps = captured[before:]
        first = steps[0] if steps else None          # 每帧第一个 decode = 动作决策点
        frames.append({
            "frame_idx": i,
            "is_listen": bool(out.get("is_listen", True)),
            "text": out.get("text", "") or "",
            "n_decode_steps": len(steps),
            "action_step": first,
        })
        if first:
            act = "LISTEN" if first["is_listen"] else ("SPEAK" if first["is_speak"] else f"other({first['sampled_id']})")
            print(f"  frame {i:2d}  {act:7s} "
                  f"P(listen|pair)={first['p_pair_listen']:.4f} "
                  f"logp_raw(sampled)={first['logp_raw_sampled']:+.4f} "
                  f"H={first['entropy_full']:.3f} steps={len(steps)}"
                  + (f'  "{frames[-1]["text"].strip()[:40]}"' if frames[-1]["text"].strip() else ""))

    dec.decode = orig_decode

    (out_dir / "logprob_trace.json").write_text(
        json.dumps({"frames": frames, "all_decode_steps": captured}, ensure_ascii=False, indent=2),
        encoding="utf-8")

    # ---- 结论 ----
    acts = [f["action_step"] for f in frames if f["action_step"]]
    n_ls = sum(1 for a in acts if a["is_listen"] or a["is_speak"])
    print(f"\n{'='*66}")
    print(f"决策点总数 {len(acts)}，其中落在 listen/speak 上的 {n_ls}")
    if acts:
        print(f"logprob 可取：是 —— 每个决策点都拿到了 logits 与 sampled token")
        print(f"P(listen|{{listen,speak}}) 范围 "
              f"[{min(a['p_pair_listen'] for a in acts):.4f}, {max(a['p_pair_listen'] for a in acts):.4f}]")
        print(f"全词表熵 范围 [{min(a['entropy_full'] for a in acts):.3f}, "
              f"{max(a['entropy_full'] for a in acts):.3f}]")
        mx = max(a["p0_eos"] for a in acts if a["p0_eos"] is not None)
        print(f"Stage-0 P(chunk_eos) 最大 {mx:.6f} "
              f"—— 越小说明两段式采样对动作分布的污染越轻")
    print(f"\n[written] {out_dir/'logprob_trace.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
