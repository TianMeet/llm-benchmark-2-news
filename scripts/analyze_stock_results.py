"""分析股票联想评测的真实 LLM 结果。"""
import json
from pathlib import Path

RUN_DIR = "runs/20260228_101134_472144"
DATASET = "datasets/stock_association_eval.jsonl"

# Load ground truth
gt_map = {}
with open(DATASET) as f:
    for line in f:
        s = json.loads(line)
        gt_map[s["sample_id"]] = s

for model in ["kimi-personal", "gpt-4o-mini"]:
    print(f"\n{'='*60}")
    print(f"Model: {model}")
    print(f"{'='*60}")
    results = []
    path = Path(RUN_DIR) / "by_model" / "stock_association" / f"{model}.jsonl"
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            if row.get("success") and row.get("parsed"):
                sid = row["sample_id"]
                pred = set(row["parsed"].get("stocks", []))
                gt = set(gt_map[sid]["gt_stocks"])
                inter = pred & gt
                p = len(inter) / len(pred) if pred else 0
                r = len(inter) / len(gt) if gt else 0
                f1 = 2 * p * r / (p + r) if (p + r) else 0
                results.append({
                    "sid": sid,
                    "name": gt_map[sid]["plate_name"],
                    "pred_count": len(pred),
                    "gt_count": len(gt),
                    "match_count": len(inter),
                    "precision": p,
                    "recall": r,
                    "f1": f1,
                })

    results.sort(key=lambda x: -x["f1"])
    print(f"\nTop 10 best F1:")
    for r in results[:10]:
        print(f"  {r['name']:20s} pred={r['pred_count']:3d} gt={r['gt_count']:3d} "
              f"match={r['match_count']:3d} P={r['precision']:.1%} R={r['recall']:.1%} F1={r['f1']:.1%}")

    print(f"\nBottom 5 worst F1:")
    for r in results[-5:]:
        print(f"  {r['name']:20s} pred={r['pred_count']:3d} gt={r['gt_count']:3d} "
              f"match={r['match_count']:3d} P={r['precision']:.1%} R={r['recall']:.1%} F1={r['f1']:.1%}")

    f1s = [x["f1"] for x in results]
    ps = [x["precision"] for x in results]
    rs = [x["recall"] for x in results]
    print(f"\nOverall ({len(results)} samples):")
    print(f"  F1:        avg={sum(f1s)/len(f1s):.1%}  median={sorted(f1s)[len(f1s)//2]:.1%}  max={max(f1s):.1%}  min={min(f1s):.1%}")
    print(f"  Precision: avg={sum(ps)/len(ps):.1%}")
    print(f"  Recall:    avg={sum(rs)/len(rs):.1%}")
