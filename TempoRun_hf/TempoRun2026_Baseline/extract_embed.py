"""Stage 2 — embed already-extracted keyframes with CLIP. NO ffmpeg, NO video decode.

Consumes the keyframes produced by `extract_keyframes.py` (Stage 1) and encodes them
with open_clip. Because it reads saved frames instead of decoding video, you can
re-embed the SAME keyframes with a different `--model` / `--pretrained` (a fair,
apples-to-apples encoder comparison) without re-running keyframe extraction.

Input layout (from extract_keyframes.py):
  <keyframes>/<video_id>/k_*.jpg + ts_ms.npy
Output:
  <out>/shards/<video_id>.npz   with emb[K,D] (fp16) and ts_ms[K] (int32)

Resumable: a video whose shard already exists is skipped.
Shardable across processes/GPUs: --shard-index / --shard-count.
"""

from __future__ import annotations
import argparse, glob, os, sys, time
import torch
from pathlib import Path
import numpy as np
REPO_ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(REPO_ROOT))


def list_keyframe_dirs(kf_root):
    """Every video folder that finished Stage 1 (has a ts_ms.npy marker)."""
    return [Path(p).parent for p in sorted(glob.glob(os.path.join(kf_root, "*", "ts_ms.npy")))]


def load_frames(vdir):
    """Load keyframe JPGs (in order) + aligned timestamps for one video."""
    from PIL import Image
    files = sorted(glob.glob(os.path.join(vdir, "k_*.jpg")))
    ts = np.load(os.path.join(vdir, "ts_ms.npy"))
    n = min(len(files), len(ts))
    imgs, kept_ts = [], []
    for i in range(n):
        try:
            imgs.append(Image.open(files[i]).convert("RGB"))
            kept_ts.append(int(ts[i]))
        except Exception:
            continue
    return imgs, kept_ts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keyframes", required=True, help="dir produced by extract_keyframes.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="ViT-B-32")
    ap.add_argument("--pretrained", default="laion2b_s34b_b79k")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="debug: cap #videos")
    ap.add_argument("--pth_dir",default = "")
    args = ap.parse_args()

    shard_dir = Path(args.out) / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    fail_log = Path(args.out) / f"failed_embed_shard{args.shard_index}.txt"

    vdirs = list_keyframe_dirs(args.keyframes)
    if not vdirs:
        raise SystemExit(f"no keyframes in {args.keyframes} — run extract_keyframes.py first")
    mine = [d for i, d in enumerate(vdirs) if i % args.shard_count == args.shard_index]
    if args.limit:
        mine = mine[:args.limit]
    print(f"[shard {args.shard_index}/{args.shard_count}] {len(mine)}/{len(vdirs)} videos", flush=True)

    from clip_model import ClipModel
    from LoRA import assign_LoRA,Apply_weights
    model = ClipModel(args.model, args.pretrained, device=args.device) # Sửa đoạn này nếu lấy file .pth
    print(f"[clip] {args.model}/{args.pretrained} on {args.device} ", flush=True)
    assign_LoRA(model,lora_r= 8,lora_alpha=16)

    if(args.pretrained == "None"):
        # Load pth file
        Apply_weights(model,args.device,args.pth_dir)

    t0 = time.time(); done = nframes = failed = 0
    for vdir in mine:
        vid = vdir.name
        out_npz = shard_dir / f"{vid}.npz"
        if out_npz.exists():
            done += 1
            continue
        try:
            imgs, ts = load_frames(vdir)
            if not imgs:
                raise RuntimeError("no frames")
            emb = model.encode_images(imgs, batch_size=args.batch_size)
            np.savez(out_npz, emb=emb.astype(np.float16), ts_ms=np.asarray(ts, dtype=np.int32))
            nframes += len(imgs)
        except Exception as ex:
            failed += 1
            with open(fail_log, "a") as f:
                f.write(f"{vid}\t{ex}\n")
        done += 1
        if done % 50 == 0:
            el = time.time() - t0
            print(f"[shard {args.shard_index}] {done}/{len(mine)} | {nframes} frames | "
                  f"fail={failed} | {done/el*60:.0f} vids/min | ETA {(len(mine)-done)/max(done/el,1e-9)/60:.0f}min",
                  flush=True)
    print(f"[shard {args.shard_index}] DONE {done} videos, {nframes} frames, {failed} failed, "
          f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
