#!/usr/bin/env python3
"""Step 3 探针：验证「训练前向」能否复现「rollout 前向」的 logits —— 训练可行性的关键。

为什么这是关键：
    PPO/GRPO 要求初始时 new_logprob == old_logprob（importance ratio ≡ 1）。
    rollout 走的是**流式**路径（逐帧 prefill/generate，带 KV cache 与因果卷积状态），
    训练要走的是**整段 teacher-forced 批量前向**。
    两者若数值不等，ratio 在第 0 步就不是 1，PPO 直接坏掉。

做法（不改官方实现）：
    1. monkeypatch `StreamDecoder.feed`，按顺序录下每一次喂进去的 embeds [L,H]
       —— 把它们首尾相接，就是本次 rollout 的**完整输入嵌入序列**
    2. monkeypatch `decode`，记录每个决策点发生时的累计序列长度 L_k
       该决策用的 logits，来自序列中第 (L_k − 1) 个位置
    3. rollout 结束后，把完整嵌入序列**一次性**喂进一个全新的 decoder，
       对所有位置算 lm_head，取第 (L_k − 1) 个位置的 logits
    4. 与 rollout 当时的 logits 逐点比较

结论若一致 ⇒ 训练侧可以用「录下 embeds → 单次批量前向」来重算 logprob，
verl 的训练路径基本可以照常接。
"""

from __future__ import annotations

import argparse
import json
import wave
from pathlib import Path

import numpy as np

INPUT_SR = 16000


def load_wav_mono(path: str) -> np.ndarray:
    with wave.open(path, "rb") as wf:
        sr, ch = wf.getframerate(), wf.getnchannels()
        x = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    assert sr == INPUT_SR, f"需要 {INPUT_SR}Hz，实际 {sr}"
    return x


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--model", default="openbmb/MiniCPM-o-4_5")
    ap.add_argument("--out", default="runs/train_equiv")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-frames", type=int, default=10)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16","float32"])
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F
    from transformers import AutoModel

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    x = load_wav_mono(args.audio)
    n = INPUT_SR
    chunks = [np.pad(x[i:i + n], (0, max(0, n - len(x[i:i + n])))) for i in range(0, len(x), n)][:args.max_frames]
    print(f"[input] {args.audio} → {len(chunks)} frames")

    print(f"[load] {args.model} …")
    model = AutoModel.from_pretrained(
        args.model, trust_remote_code=True,
        torch_dtype=getattr(torch, args.dtype), attn_implementation="sdpa",
    ).eval().to(args.device)
    duplex = model.as_duplex(device=args.device, generate_audio=False, ls_mode="explicit")

    dec = duplex.decoder
    listen_id, speak_id = duplex.listen_token_id, duplex.tokenizer.convert_tokens_to_ids("<|speak|>")

    fed: list[torch.Tensor] = []       # 按序录下每次 feed 的 embeds
    cum = {"len": 0}
    decisions: list[dict] = []         # 每个决策点：位置 + 当时的 logits

    orig_feed, orig_decode = dec.feed, dec.decode

    def traced_feed(embeds, return_logits=False):
        fed.append(embeds.detach().clone())
        cum["len"] += int(embeds.size(0))
        return orig_feed(embeds, return_logits=return_logits)

    def traced_decode(logits, *a, **kw):
        tok = orig_decode(logits, *a, **kw)
        tid = int(tok.item()) if hasattr(tok, "item") else int(tok)
        decisions.append({
            "pos": cum["len"] - 1,                       # 产生该 logits 的位置
            "sampled_id": tid,
            "logits_row": logits.detach().float()[0].cpu(),
        })
        return tok

    dec.feed, dec.decode = traced_feed, traced_decode
    duplex.prepare()

    frame_first_decision = []
    for i, c in enumerate(chunks):
        before = len(decisions)
        duplex.streaming_prefill(audio_waveform=c)
        out = duplex.streaming_generate()
        if len(decisions) > before:
            frame_first_decision.append(before)
        print(f"  frame {i:2d} {'LISTEN' if out.get('is_listen') else 'SPEAK '} "
              f"decisions={len(decisions)-before} cumlen={cum['len']}")

    dec.feed, dec.decode = orig_feed, orig_decode

    full = torch.cat(fed, dim=0)                          # [T, H] 完整输入嵌入序列
    print(f"\n[replay] 完整序列 {tuple(full.shape)}，决策点 {len(decisions)} 个")

    # ---- 单次批量 teacher-forced 前向 ----
    dec.reset()
    with torch.no_grad():
        o = dec.m(
            inputs_embeds=full.unsqueeze(0),
            position_ids=torch.arange(full.size(0), device=full.device).unsqueeze(0),
            return_dict=True, output_hidden_states=True,
        )
        all_logits = dec.m.lm_head(o.hidden_states[-1])[0].float()   # [T, vocab]
    print(f"[replay] 批量前向 logits {tuple(all_logits.shape)}")

    # ---- 逐决策点比对 ----
    rows = []
    for k, d in enumerate(decisions):
        a = d["logits_row"].to(all_logits.device)
        b = all_logits[d["pos"]]
        pa = F.log_softmax(torch.stack([a[listen_id], a[speak_id]]), dim=-1)
        pb = F.log_softmax(torch.stack([b[listen_id], b[speak_id]]), dim=-1)
        rows.append({
            "k": k, "pos": d["pos"], "sampled_id": d["sampled_id"],
            "max_abs_logit_diff": float((a - b).abs().max()),
            "argmax_same": bool(a.argmax() == b.argmax()),
            "logp_listen_rollout": float(pa[0]), "logp_listen_replay": float(pb[0]),
            "logp_pair_absdiff": float((pa - pb).abs().max()),
        })

    worst = max(rows, key=lambda r: r["max_abs_logit_diff"])
    worst_lp = max(rows, key=lambda r: r["logp_pair_absdiff"])
    same = sum(r["argmax_same"] for r in rows)

    print(f"\n{'='*70}")
    print(f"决策点数            : {len(rows)}")
    print(f"argmax 一致          : {same}/{len(rows)}")
    print(f"logits 最大绝对差     : {worst['max_abs_logit_diff']:.6f}  (决策点 k={worst['k']}, pos={worst['pos']})")
    print(f"log P(listen) 最大差  : {worst_lp['logp_pair_absdiff']:.6f}  (k={worst_lp['k']})")
    ok = same == len(rows) and worst_lp["logp_pair_absdiff"] < 1e-2
    print(f"\n{'✅ 等价：训练可用「录 embeds → 单次批量前向」重算 logprob' if ok else '❌ 不等价：训练需改用流式 teacher-forced 前向'}")

    (out_dir / "equiv_report.json").write_text(json.dumps({
        "n_decisions": len(rows), "argmax_same": same,
        "max_abs_logit_diff": worst["max_abs_logit_diff"],
        "max_logp_pair_diff": worst_lp["logp_pair_absdiff"],
        "seq_len": int(full.size(0)), "equivalent": ok, "rows": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[written] {out_dir/'equiv_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
