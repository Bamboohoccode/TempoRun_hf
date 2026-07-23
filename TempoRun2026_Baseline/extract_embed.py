"""Embed extracted video keyframes into one NPZ shard per video."""
from __future__ import annotations

import argparse
import glob
import os
import time
from pathlib import Path

import numpy as np


def list_keyframe_dirs(root: str) -> list[Path]:
    return [
        Path(path).parent
        for path in sorted(glob.glob(os.path.join(root, "*", "ts_ms.npy")))
    ]


def load_frames(video_dir: Path):
    from PIL import Image

    files = sorted(video_dir.glob("k_*.jpg"))
    timestamps = np.load(video_dir / "ts_ms.npy")
    images, kept_timestamps = [], []
    for path, timestamp in zip(files, timestamps):
        try:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
            kept_timestamps.append(int(timestamp))
        except (OSError, ValueError):
            continue
    return images, kept_timestamps


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--keyframes", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="ViT-gopt-16-SigLIP2-384")
    parser.add_argument("--pretrained", default="webli")
    parser.add_argument("--precision", default="fp16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max_length", type=int, default=64)
    return parser.parse_args()


def main():
    args = parse_args()
    shard_dir = Path(args.out) / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    fail_log = Path(args.out) / f"failed_embed_shard{args.shard_index}.txt"

    video_dirs = list_keyframe_dirs(args.keyframes)
    mine = [
        directory
        for index, directory in enumerate(video_dirs)
        if index % args.shard_count == args.shard_index
    ]
    if args.limit:
        mine = mine[: args.limit]
    if not mine:
        raise SystemExit(f"no keyframes in {args.keyframes}")

    from clip_model import ClipModel

    model = ClipModel(
        model_name=args.model,
        pretrained=args.pretrained,
        precision=args.precision,
        device=args.device,
        max_text_length=args.max_length,
    )
    print(
        f"[encoder] model={args.model} pretrained={args.pretrained} "
        f"dim={model.dim} device={args.device}",
        flush=True,
    )

    started = time.time()
    done = frames = failed = 0
    for video_dir in mine:
        output = shard_dir / f"{video_dir.name}.npz"
        if output.exists():
            done += 1
            continue
        try:
            images, timestamps = load_frames(video_dir)
            if not images:
                raise RuntimeError("no readable frames")
            embeddings = model.encode_images(images, batch_size=args.batch_size)
            np.savez(
                output,
                emb=embeddings.astype(np.float16),
                ts_ms=np.asarray(timestamps, dtype=np.int32),
            )
            frames += len(images)
        except Exception as error:
            failed += 1
            with fail_log.open("a", encoding="utf-8") as stream:
                stream.write(f"{video_dir.name}\t{error}\n")
        done += 1
        if done % 50 == 0:
            elapsed = max(time.time() - started, 1e-9)
            print(
                f"[embed] {done}/{len(mine)} videos | {frames} frames | "
                f"failed={failed} | {done / elapsed * 60:.1f} videos/min",
                flush=True,
            )

    print(
        f"[done] {done} videos, {frames} frames, {failed} failed",
        flush=True,
    )


if __name__ == "__main__":
    main()
