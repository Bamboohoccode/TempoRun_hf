"""Stage 3 — text->keyframe retrieval -> submission.json.

Loads all per-video shards into one keyframe index, embeds each task description
with CLIP text encoder, finds the most similar keyframes (chunked top-k on GPU),
dedups to distinct videos, and emits up to 10 (video_id, frame_ms) predictions
per task — one frame (the matched keyframe's timestamp) per distinct video.
"""
from __future__ import annotations
import argparse, glob, json, os, time
from pathlib import Path
import numpy as np


def load_index(shard_dir):
    embs, vids, ts = [], [], []
    files = sorted(glob.glob(os.path.join(shard_dir, "*.npz")))
    for f in files:
        d = np.load(f)
        e = d["emb"]
        if e.shape[0] == 0:
            continue
        embs.append(e)
        vid = Path(f).stem
        vids.extend([vid] * e.shape[0])
        ts.append(d["ts_ms"])
    if not embs:
        raise SystemExit(f"no shards in {shard_dir}")
    emb = np.concatenate(embs, 0)            # [N, D] fp16
    ts = np.concatenate(ts, 0).astype(np.int32)
    vids = np.array(vids)
    print(f"[index] {emb.shape[0]} keyframes from {len(files)} videos, dim={emb.shape[1]}", flush=True)
    return emb, vids, ts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shards", required=True)
    p.add_argument("--tasks", required=True, help="a round's task file, e.g. public_round_tasks.jsonl")
    p.add_argument("--out", required=True, help="submission.json path")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--model", default="ViT-B-32")
    p.add_argument("--pretrained", default="laion2b_s34b_b79k")
    p.add_argument("--top-videos", type=int, default=10)
    p.add_argument("--cand-keyframes", type=int, default=400)
    p.add_argument("--pth_dir",default = "")
    args = p.parse_args()

    import torch
    emb, vids, ts = load_index(args.shards)
    tasks = [json.loads(l) for l in open(args.tasks)]
    print(f"[tasks] {len(tasks)}", flush=True)

    from clip_model import ClipModel
    from LoRA import     assign_LoRA,Apply_weights
    model = ClipModel(args.model, args.pretrained, device=args.device)
    assign_LoRA(model,lora_r= 8,lora_alpha=16)

    if(args.pretrained == "None"):
        # Load pth file
        Apply_weights(model,args.device,args.pth_dir)
    
    Q = model.encode_texts([t["description"] for t in tasks])      # [T, D] fp32

    dev = args.device
    idx = torch.from_numpy(emb).to(dev).half()                    # [N, D]
    Qt = torch.from_numpy(Q).to(dev).half()                        # [T, D]
    T, N = Qt.shape[0], idx.shape[0]
    K = min(args.cand_keyframes, N)

    # chunked top-K keyframes per query
    CH = 200_000
    top_val = torch.full((T, K), float("-inf"), device=dev, dtype=torch.float16)
    top_idx = torch.zeros((T, K), device=dev, dtype=torch.long)
    t0 = time.time()
    for s in range(0, N, CH):
        e = min(s + CH, N)
        sims = Qt @ idx[s:e].T                                     # [T, chunk]
        cat_v = torch.cat([top_val, sims], 1)
        cat_i = torch.cat([top_idx, torch.arange(s, e, device=dev).expand(T, e - s)], 1)
        top_val, sel = cat_v.topk(K, dim=1)
        top_idx = torch.gather(cat_i, 1, sel)
    print(f"[retrieve] scored {N} keyframes x {T} queries in {time.time()-t0:.0f}s", flush=True)

    top_idx = top_idx.cpu().numpy()
    top_val = top_val.float().cpu().numpy()

    preds = []
    for ti, task in enumerate(tasks):
        rows = top_idx[ti]
        sims = top_val[ti]
        seen = {}
        for r, sim in zip(rows, sims):
            v = str(vids[r])
            if v in seen:
                continue
            center = int(ts[r])
            seen[v] = (center, float(sim))
            if len(seen) >= args.top_videos:
                break
        results = []
        for rank, (v, (center, sim)) in enumerate(seen.items(), 1):
            results.append({
                "rank": rank, "video_id": v,
                "frame_ms": int(center),
            })
        preds.append({"task_id": task["task_id"], "results": results})

    sub = {"predictions": preds}
    json.dump(sub, open(args.out, "w"))
    print(f"[done] wrote {args.out} ({len(preds)} tasks)", flush=True)


if __name__ == "__main__":
    main()
