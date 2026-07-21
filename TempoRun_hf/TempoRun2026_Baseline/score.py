"""Stage 4 — score a FRAME submission.json against a ground-truth jsonl (self-contained).

A prediction is a single frame (video_id, frame_ms). It is CORRECT if the frame lies
inside the GT interval [start_ms, end_ms] AND the video_id matches.

  score = mean over tasks of  max_pred( hit * 1/rank )     hit in {0,1}
        = MRR over frame-in-interval hits.

A correct rank-1 prediction scores 1.0; not-in-top-10 scores 0.
"""
from __future__ import annotations
import argparse, json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--submission", required=True)
    p.add_argument("--ground-truth", required=True)
    p.add_argument("--max-predictions", type=int, default=10)
    args = p.parse_args()

    sub = json.load(open(args.submission))
    pred_by_task = {x["task_id"]: x for x in sub["predictions"]}
    gts = [json.loads(l) for l in open(args.ground_truth)]

    n = 0; sum_score = 0.0; hits = r1 = r5 = r10 = 0; mrr = 0.0
    for gt in gts:
        n += 1
        pr = pred_by_task.get(gt["task_id"])
        best = 0.0; best_rank = None; hit_any = False
        if pr:
            for r in sorted(pr["results"], key=lambda x: x["rank"])[:args.max_predictions]:
                if r["video_id"] != gt["video_id"]:
                    continue
                hit = gt["start_ms"] <= r["frame_ms"] <= gt["end_ms"]
                rs = (1.0 if hit else 0.0) / r["rank"]
                if hit: hit_any = True
                if rs > best:
                    best, best_rank = rs, r["rank"]
        sum_score += best
        if hit_any:
            hits += 1
            if best_rank == 1: r1 += 1
            if best_rank is not None and best_rank <= 5: r5 += 1
            if best_rank is not None and best_rank <= 10: r10 += 1
            if best_rank: mrr += 1.0 / best_rank

    n = max(1, n)
    print(json.dumps({
        "metric": "frame_in_interval / reciprocal_rank",
        "num_tasks": n,
        "score": round(sum_score / n, 4),
        "frame_hit_rate": round(hits / n, 4),
        "recall@1": round(r1 / n, 4),
        "recall@5": round(r5 / n, 4),
        "recall@10": round(r10 / n, 4),
        "mrr": round(mrr / n, 4),
    }, indent=2))


if __name__ == "__main__":
    main()
