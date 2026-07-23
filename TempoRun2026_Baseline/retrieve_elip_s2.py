"""Stage 3: SigLIP2 retrieval + ELIP-S-2 reranking -> submission.json.

Stage 1 computes exact chunked top-K against the precomputed keyframe index.
Stage 2 opens only those candidate images, injects query-conditioned ELIP
prompts into SigLIP2's vision transformer, reranks them, deduplicates videos,
and writes one timestamp per distinct video.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np


def load_index(shard_dir: str):
    embeddings, video_ids, timestamps, local_indices = [], [], [], []
    files = sorted(glob.glob(os.path.join(shard_dir, "*.npz")))
    nonempty_videos = 0
    expected_dim = None

    for filename in files:
        with np.load(filename, allow_pickle=False) as data:
            if "emb" not in data or "ts_ms" not in data:
                raise ValueError(f"{filename} must contain emb and ts_ms")
            emb = np.asarray(data["emb"])
            ts_ms = np.asarray(data["ts_ms"])

        if emb.ndim != 2:
            raise ValueError(f"{filename}: emb must be [K, D], got {emb.shape}")
        if len(ts_ms) != len(emb):
            raise ValueError(
                f"{filename}: {len(emb)} embeddings but {len(ts_ms)} timestamps"
            )
        if emb.shape[0] == 0:
            continue
        if expected_dim is None:
            expected_dim = emb.shape[1]
        elif emb.shape[1] != expected_dim:
            raise ValueError(
                f"{filename}: dim={emb.shape[1]}, expected {expected_dim}"
            )

        video_id = Path(filename).stem
        embeddings.append(emb)
        video_ids.extend([video_id] * emb.shape[0])
        timestamps.append(ts_ms)
        local_indices.extend(range(emb.shape[0]))
        nonempty_videos += 1

    if not embeddings:
        raise SystemExit(f"no non-empty shards in {shard_dir}")

    emb = np.concatenate(embeddings, axis=0)
    vids = np.asarray(video_ids)
    ts = np.concatenate(timestamps, axis=0).astype(np.int64)
    local_idx = np.asarray(local_indices, dtype=np.int32)
    print(
        f"[index] {len(emb)} keyframes from {nonempty_videos} videos, "
        f"dim={emb.shape[1]}, dtype={emb.dtype}",
        flush=True,
    )
    return emb, vids, ts, local_idx


def read_tasks(filename: str):
    with open(filename, encoding="utf-8") as stream:
        tasks = [json.loads(line) for line in stream if line.strip()]
    for task in tasks:
        if "task_id" not in task or "description" not in task:
            raise ValueError("Every task needs task_id and description")
    return tasks


def chunked_topk(
    embeddings: np.ndarray,
    queries: np.ndarray,
    device: str,
    k: int,
    index_chunk_size: int,
    query_batch_size: int,
    score_dtype: str,
    renormalize_index: bool,
):
    """Exact top-K while keeping the keyframe index in CPU memory."""
    import torch

    if embeddings.shape[1] != queries.shape[1]:
        raise ValueError(
            f"Index dim {embeddings.shape[1]} != query dim {queries.shape[1]}"
        )

    torch_dtype = torch.float16 if score_dtype == "fp16" else torch.float32
    total_queries, total_frames = len(queries), len(embeddings)
    k = min(k, total_frames)
    result_values = np.empty((total_queries, k), dtype=np.float32)
    result_indices = np.empty((total_queries, k), dtype=np.int64)

    for q_start in range(0, total_queries, query_batch_size):
        q_end = min(q_start + query_batch_size, total_queries)
        query_tensor = torch.as_tensor(
            queries[q_start:q_end], device=device, dtype=torch.float32
        )
        query_tensor = torch.nn.functional.normalize(query_tensor, dim=-1)
        query_tensor = query_tensor.to(dtype=torch_dtype)
        batch_queries = q_end - q_start

        top_values = torch.full(
            (batch_queries, k),
            float("-inf"),
            device=device,
            dtype=torch.float32,
        )
        top_indices = torch.zeros(
            (batch_queries, k), device=device, dtype=torch.long
        )

        for start in range(0, total_frames, index_chunk_size):
            end = min(start + index_chunk_size, total_frames)
            chunk_np = embeddings[start:end]
            chunk = torch.as_tensor(chunk_np, device=device, dtype=torch.float32)
            if renormalize_index:
                chunk = torch.nn.functional.normalize(chunk, dim=-1)
            chunk = chunk.to(dtype=torch_dtype)

            # Convert scores to fp32 before top-k to reduce near-tie instability.
            similarities = (query_tensor @ chunk.T).float()
            chunk_indices = torch.arange(start, end, device=device).expand(
                batch_queries, end - start
            )
            candidate_values = torch.cat([top_values, similarities], dim=1)
            candidate_indices = torch.cat(
                [top_indices, chunk_indices], dim=1
            )
            top_values, selected = candidate_values.topk(k, dim=1)
            top_indices = torch.gather(candidate_indices, 1, selected)

        result_values[q_start:q_end] = top_values.cpu().numpy()
        result_indices[q_start:q_end] = top_indices.cpu().numpy()

    return result_values, result_indices


class KeyframeResolver:
    IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}

    def __init__(self, root: str):
        self.root = Path(root)

    @lru_cache(maxsize=1024)
    def frames(self, video_id: str) -> tuple[Path, ...]:
        directory = self.root / video_id
        if not directory.is_dir():
            return ()
        return tuple(
            sorted(
                path
                for path in directory.iterdir()
                if path.is_file() and path.suffix.lower() in self.IMAGE_SUFFIXES
            )
        )

    def resolve(self, video_id: str, local_index: int) -> Path | None:
        frames = self.frames(video_id)
        if 0 <= local_index < len(frames):
            return frames[local_index]
        return None


def choose_diverse_candidates(
    rows: np.ndarray,
    scores: np.ndarray,
    video_ids: np.ndarray,
    max_per_video: int,
):
    counts: dict[str, int] = defaultdict(int)
    selected = []
    for row, score in zip(rows, scores):
        video_id = str(video_ids[row])
        if counts[video_id] >= max_per_video:
            continue
        selected.append((int(row), float(score)))
        counts[video_id] += 1
    return selected


def rerank_task(
    query_feature: np.ndarray,
    candidates,
    reranker,
    resolver: KeyframeResolver,
    video_ids: np.ndarray,
    timestamps: np.ndarray,
    local_indices: np.ndarray,
    top_videos: int,
    rerank_batch_size: int,
    stage1_weight: float,
):
    # No mapper checkpoint: run the pretrained SigLIP2 Stage 1 only.  Do not
    # instantiate or score with an untrained random prompt mapper.
    if reranker is None:
        results = []
        used_videos = set()
        for row, _ in candidates:
            video_id = str(video_ids[row])
            if video_id in used_videos:
                continue
            results.append((video_id, int(timestamps[row])))
            used_videos.add(video_id)
            if len(results) >= top_videos:
                break
        return [
            {"rank": rank, "video_id": video_id, "frame_ms": frame_ms}
            for rank, (video_id, frame_ms) in enumerate(results, start=1)
        ]

    from PIL import Image

    if resolver is None:
        raise ValueError("Keyframe resolver is required when ELIP is enabled")

    images, valid_candidates = [], []
    for row, stage1_score in candidates:
        video_id = str(video_ids[row])
        image_path = resolver.resolve(video_id, int(local_indices[row]))
        if image_path is None:
            continue
        try:
            with Image.open(image_path) as image:
                images.append(image.convert("RGB"))
            valid_candidates.append((row, stage1_score))
        except (OSError, ValueError):
            continue

    best_by_video = {}
    if images:
        elip_scores = reranker.rerank(
            query_feature=query_feature,
            pil_images=images,
            batch_size=rerank_batch_size,
        )
        for (row, stage1_score), elip_score in zip(
            valid_candidates, elip_scores
        ):
            final_score = (
                stage1_weight * stage1_score
                + (1.0 - stage1_weight) * float(elip_score)
            )
            video_id = str(video_ids[row])
            previous = best_by_video.get(video_id)
            if previous is None or final_score > previous[0]:
                best_by_video[video_id] = (
                    final_score,
                    row,
                    float(elip_score),
                )

    ranked = sorted(best_by_video.items(), key=lambda item: -item[1][0])
    results = []
    used_videos = set()
    for video_id, (_, row, _) in ranked[:top_videos]:
        results.append((video_id, int(timestamps[row])))
        used_videos.add(video_id)

    # Missing/corrupt images should not make a structurally incomplete result.
    # Backfill distinct videos according to their original Stage-1 order.
    if len(results) < top_videos:
        for row, _ in candidates:
            video_id = str(video_ids[row])
            if video_id in used_videos:
                continue
            results.append((video_id, int(timestamps[row])))
            used_videos.add(video_id)
            if len(results) >= top_videos:
                break

    return [
        {"rank": rank, "video_id": video_id, "frame_ms": frame_ms}
        for rank, (video_id, frame_ms) in enumerate(results, start=1)
    ]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", required=True)
    parser.add_argument(
        "--keyframes",
        default="",
        help="raw keyframe root; required only when --elip-checkpoint is used",
    )
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--elip-checkpoint",
        default="",
        help="official ELIP-S-2 .pt; omit for pretrained SigLIP2 Stage 1 only",
    )
    parser.add_argument(
        "--elip-repo",
        default="",
        help="path to the cloned official ypliubit/ELIP repository",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--model", default="ViT-gopt-16-SigLIP2-384"
    )
    parser.add_argument(
        "--pretrained",
        default="webli",
        help="OpenCLIP pretrained tag; use None only for random weights",
    )
    parser.add_argument("--precision", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--top-videos", type=int, default=10)
    parser.add_argument("--cand-keyframes", type=int, default=400)
    parser.add_argument("--max-keyframes-per-video", type=int, default=3)
    parser.add_argument("--index-chunk-size", type=int, default=200_000)
    parser.add_argument("--query-batch-size", type=int, default=32)
    parser.add_argument("--rerank-batch-size", type=int, default=2)
    parser.add_argument("--num-prompts", type=int, default=10)
    parser.add_argument(
        "--score-dtype", choices=["fp16", "fp32"], default="fp16"
    )
    parser.add_argument("--renormalize-index", action="store_true")
    parser.add_argument(
        "--stage1-weight",
        type=float,
        default=0.0,
        help="0=pure ELIP; values such as 0.15 blend the Stage-1 score",
    )
    parser.add_argument(
        "--lora-checkpoint",
        default="",
        help="optional project LoRA .pth applied before ELIP",
    )
    args = parser.parse_args()
    if not 0.0 <= args.stage1_weight <= 1.0:
        parser.error("--stage1-weight must be in [0, 1]")
    if args.cand_keyframes < args.top_videos:
        parser.error("--cand-keyframes must be >= --top-videos")
    if args.elip_checkpoint and (not args.keyframes or not args.elip_repo):
        parser.error(
            "--keyframes and --elip-repo are required with --elip-checkpoint"
        )
    if args.elip_checkpoint and args.lora_checkpoint:
        parser.error("LoRA and the official ELIP checkpoint cannot be combined")
    return args


def main():
    args = parse_args()
    embeddings, video_ids, timestamps, local_indices = load_index(args.shards)
    tasks = read_tasks(args.tasks)
    print(f"[tasks] {len(tasks)}", flush=True)

    descriptions = [task["description"] for task in tasks]
    if args.elip_checkpoint:
        # Importing official ELIP and pip open_clip in one process would bind
        # the same module name to two implementations. Use only ELIP's encoder
        # in this branch; it shares the SigLIP2 embedding space with Stage 1.
        from elip_s2 import ELIPS2Reranker

        reranker = ELIPS2Reranker(
            checkpoint=args.elip_checkpoint,
            elip_repo=args.elip_repo,
            model_name=args.model,
            pretrained=args.pretrained,
            precision=args.precision,
            device=args.device,
            max_text_length=args.max_length,
        )
        queries = reranker.encode_texts(descriptions)
        resolver = KeyframeResolver(args.keyframes)
        print(f"[stage2] ELIP enabled: {args.elip_checkpoint}", flush=True)
    else:
        from clip_model_siglip2 import ClipModel

        model = ClipModel(
            model_name=args.model,
            pretrained=args.pretrained,
            precision=args.precision,
            device=args.device,
            max_text_length=args.max_length,
        )
        if args.lora_checkpoint:
            from LoRA import Apply_weights, assign_LoRA

            assign_LoRA(model, lora_r=8, lora_alpha=16)
            Apply_weights(model, args.device, args.lora_checkpoint)
            model.model.eval()
        queries = model.encode_texts(descriptions)
        reranker = None
        resolver = None
        print(
            "[stage2] ELIP disabled; using pretrained SigLIP2 Stage 1 only",
            flush=True,
        )

    started = time.time()
    top_values, top_indices = chunked_topk(
        embeddings=embeddings,
        queries=queries,
        device=args.device,
        k=args.cand_keyframes,
        index_chunk_size=args.index_chunk_size,
        query_batch_size=args.query_batch_size,
        score_dtype=args.score_dtype,
        renormalize_index=args.renormalize_index,
    )
    print(
        f"[stage1] scored {len(embeddings)} keyframes x {len(tasks)} "
        f"queries in {time.time() - started:.1f}s",
        flush=True,
    )

    predictions = []
    started = time.time()
    for task_index, task in enumerate(tasks):
        candidates = choose_diverse_candidates(
            rows=top_indices[task_index],
            scores=top_values[task_index],
            video_ids=video_ids,
            max_per_video=args.max_keyframes_per_video,
        )
        results = rerank_task(
            query_feature=queries[task_index],
            candidates=candidates,
            reranker=reranker,
            resolver=resolver,
            video_ids=video_ids,
            timestamps=timestamps,
            local_indices=local_indices,
            top_videos=args.top_videos,
            rerank_batch_size=args.rerank_batch_size,
            stage1_weight=args.stage1_weight,
        )
        predictions.append({"task_id": task["task_id"], "results": results})
        print(
            f"[stage2] {task_index + 1}/{len(tasks)} task_id={task['task_id']} "
            f"candidates={len(candidates)} results={len(results)}",
            flush=True,
        )

    print(f"[stage2] completed in {time.time() - started:.1f}s", flush=True)
    output = {"predictions": predictions}
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(output, stream, ensure_ascii=False, indent=2)
    print(f"[done] wrote {output_path} ({len(predictions)} tasks)", flush=True)


if __name__ == "__main__":
    main()
