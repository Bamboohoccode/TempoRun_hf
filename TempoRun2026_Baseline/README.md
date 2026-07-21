# TempoRun 2026 — CLIP I-frame Baseline

A simple, reproducible baseline for the **TempoRun temporal video-retrieval** task.
Given a natural-language description, return up to 10 ranked **frame** predictions
`(video_id, frame_ms)`. A prediction is correct if your **submitted frame falls
inside the correct clip's labeled interval** — see [Scoring](#scoring).

- **Corpus:** 5006 short clips (≤180 s each; V3C-format
  `<coll>/videos/<id>/<id>.mp4`).
- **Index built by this baseline:** ~223k I-frame embeddings (≈45 keyframes/clip).
- **Best baseline:** PE-Core-bigG **0.387** vs ViT-B-32 **0.227** on public round — see [Results](#results).

---

## How it works

The pipeline is four independent stages. **Keyframe extraction and embedding are
separate steps** — extract the frames once (slow, ffmpeg/I-O bound), then embed them
with any encoder (GPU bound) without re-decoding video.

1. **Keyframe extraction** (`extract_keyframes.py`) — for every clip, ffmpeg emits only
   the **I-frames (keyframes)** via `-skip_frame nokey`, and `showinfo` records each
   keyframe's `pts_time` (→ clip-relative timestamp). The keyframe JPGs and their
   timestamps are written to disk (`<kf>/<video_id>/k_*.jpg` + `ts_ms.npy`).
   No model involved. Content-driven count. Resumable.
2. **Embedding** (`extract_embed.py`) — each saved keyframe is encoded with open_clip.
   Default **ViT-B-32 / laion2b_s34b_b79k** (512-d); swap in a stronger encoder with
   `--model/--pretrained` — we ship **PE-Core-bigG-14-448 / meta** (1280-d), which
   roughly doubles the score. Per-clip `<out>/shards/<video_id>.npz`
   (`emb[K,D]` fp16 + `ts_ms[K]`). Resumable. Because Stage 1 already saved the frames,
   you can re-embed the **same** keyframes with a different encoder for a fair
   comparison without re-running ffmpeg.
3. **Retrieval** (`retrieve.py`) — each task description is encoded with the text
   encoder; cosine top-k over all keyframes (chunked on GPU), deduped to distinct
   clips. **Each prediction is the matched keyframe itself — a single `frame_ms`.**
4. **Scoring** (`score.py`) — frame-in-interval × reciprocal rank.

---

## Setup

```bash
pip install -r requirements.txt      # torch, open_clip_torch, opencv, pillow, numpy, tqdm
# system dependency: ffmpeg / ffprobe on PATH (for extract_keyframes.py)
```

**Models download automatically.** open_clip fetches the pretrained weights from the
HuggingFace hub on first use and caches them under `$HF_HOME` (default
`~/.cache/huggingface`) — there is no manual download step:
- `ViT-B-32 / laion2b_s34b_b79k` ≈ 0.6 GB
- `PE-Core-bigG-14-448 / meta` ≈ 9 GB

The first run needs internet. For **offline / air-gapped** nodes, pre-download once with
network on (optionally point `HF_HOME` at a shared cache), then run with `HF_HUB_OFFLINE=1`.

GPU notes:
- cuDNN is **disabled** in `clip_model.py` — the ViT patch-embed conv otherwise needs
  a cuDNN workspace that fails when processes share a busy GPU; native conv needs none.
- **Match your torch CUDA build to the driver.** `cu130` needs driver ≥ CUDA 13 and
  `sm_75+` (not Pascal). On a CUDA-12.x driver (e.g. L40) install a `cu128` build:
  `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128`.
- Large models (PE-Core-bigG ≈ 9 GB): use **fp16/bf16** (`--precision`) to halve VRAM.

---

## Reproduce

```bash
ROOT=/path/to/clips_trim; DEV=cuda:0
KF=keyframes                 # Stage-1 output (fast local disk recommended)

# 1. Extract I-frames once (no GPU; shard across processes; resumable).
for i in 0 1 2 3 4 5; do
  python extract_keyframes.py \
    --dataset-root $ROOT/V3C1 --dataset-root $ROOT/V3C2 \
    --out $KF --shard-index $i --shard-count 6 &
done; wait

# 2. Embed the saved keyframes (GPU; resumable). Re-run with any encoder — no re-decode.
for i in 0 1 2 3 4 5; do
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<gpu> \
  python extract_embed.py \
    --keyframes $KF --out artifacts/index --device $DEV \
    --model ViT-B-32 --pretrained laion2b_s34b_b79k \
    --shard-index $i --shard-count 6 &
done; wait
#   Stronger encoder:  --model PE-Core-bigG-14-448 --pretrained meta  (add --precision bf16)

# 3. Retrieve -> submission.json (one frame_ms per prediction)
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<gpu> \
python retrieve.py \
  --shards artifacts/index/shards \
  --tasks  public_round_tasks.jsonl \
  --out    submission.json \
  --model ViT-B-32 --pretrained laion2b_s34b_b79k --device $DEV

# 4. Score against ground truth
python score.py --submission submission.json \
                --ground-truth gt_public_round.jsonl
```

---

## Scoring

**Frame submission.** Each prediction is a single frame `(video_id, frame_ms)`.
For a prediction at rank `r`:

```
wrong video_id     -> 0
correct video_id   -> hit * (1 / r)

  hit = 1.0 if  gt_start_ms <= frame_ms <= gt_end_ms  else 0.0
```

- **task_score** = max over that task's ≤10 predictions = `1 / (rank of best in-interval frame)`.
- **final score** = mean of task_score over all tasks (this is **MRR** over frame-in-interval hits).
- Reciprocal-rank credit: r1→1.00, r2→0.50, r3→0.33, r5→0.20, r10→0.10; **not in top 10 → 0**.
- There is no interval to submit and no IoU — submit your single best frame; it either
  lands inside the labeled segment or it doesn't.
- Also reported: `frame_hit_rate` (tasks with any in-top-10 hit), `Recall@1/5/10`, `MRR`.

---

## Submission

Each line of a round's task file (e.g. `public_round_tasks.jsonl`) is one JSON object:
`{"task_id": "T0001", "description": "..."}`. Produce a `submission.json`
with one prediction block per `task_id`:

```json
{
  "predictions": [
    {
      "task_id": "T0001",
      "results": [
        {"rank": 1, "video_id": "v3c1_00123", "frame_ms": 363636},
        {"rank": 2, "video_id": "v3c1_04567", "frame_ms": 676767}
      ]
    }
  ]
}
```

Rules:
- ≤ **10** results per task; `rank` is 1..N (1 = best); cover every task in that round's task file.
- `video_id` must exist in the corpus; `frame_ms >= 0`.
- Each result is **one frame** — the moment you believe matches the query.

---

## Results

open_clip zero-shot, searching all 5006 clips (~223k keyframes), **frame-in-interval
× reciprocal-rank** metric:

| Model | full (1000) | public round (300) | private round (700) |
|---|---|---|---|
| ViT-B-32 (default) | **secret** | 0.227 | **secret** |
| **PE-Core-bigG-14-448** | **secret** | **0.387** | **secret** |

PE-Core roughly **doubles** ViT-B-32. PE-Core lands a frame in the right clip's
segment within top-10 for ~42% of tasks (`frame_hit_rate`), at rank-1 for ~35%.
