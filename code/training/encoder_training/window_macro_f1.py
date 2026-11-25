import json
import argparse
from pathlib import Path

import numpy as np


def _prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return {"precision": p, "recall": r, "f1": f1}


def pairwise_window_f1(y_gold, y_pred):
    """
    Directed pairwise micro-averaged F1 between gold and pred binary window labels.
    """
    y_gold = np.asarray(y_gold).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    tp_gold_pred = int(((y_gold == 1) & (y_pred == 1)).sum())
    fp_gold_pred = int(((y_gold == 0) & (y_pred == 1)).sum())
    fn_gold_pred = int(((y_gold == 1) & (y_pred == 0)).sum())

    tp_pred_gold = tp_gold_pred
    fp_pred_gold = int(((y_pred == 0) & (y_gold == 1)).sum())
    fn_pred_gold = int(((y_pred == 1) & (y_gold == 0)).sum())

    total_tp = tp_gold_pred + tp_pred_gold
    total_fp = fp_gold_pred + fp_pred_gold
    total_fn = fn_gold_pred + fn_pred_gold

    return {
        "pairwise_micro": _prf(total_tp, total_fp, total_fn),
        "per_pair": {
            "gold→pred": _prf(tp_gold_pred, fp_gold_pred, fn_gold_pred),
            "pred→gold": _prf(tp_pred_gold, fp_pred_gold, fn_pred_gold),
        },
        "num_windows": int(len(y_gold)),
        "num_positive_gold": int((y_gold == 1).sum()),
        "num_positive_pred": int((y_pred == 1).sum()),
    }

def pairwise_window_macro_f1(data):
    """
    Macro-averaged F1 across samples (sample_index).
    """
    from collections import defaultdict

    grouped = defaultdict(list)

    # group windows by sample_index
    for ex in data:
        grouped[ex["sample_index"]].append(ex)

    f1s = []
    precisions = []
    recalls = []

    for sample_id, items in grouped.items():
        y_gold = [int(x["gold"]) for x in items]
        y_pred = [int(x["pred"]) for x in items]

        res = pairwise_window_f1(y_gold, y_pred)["pairwise_micro"]

        f1s.append(res["f1"])
        precisions.append(res["precision"])
        recalls.append(res["recall"])

    return {
        "pairwise_macro": {
            "precision": float(np.mean(precisions)) if precisions else 0.0,
            "recall": float(np.mean(recalls)) if recalls else 0.0,
            "f1": float(np.mean(f1s)) if f1s else 0.0,
        },
        "num_samples": len(grouped),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--eval_file", required=True)
    parser.add_argument("--out_path", required=True)
    args = parser.parse_args()

    with open(args.eval_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"{args.eval_file} must be a JSON list")

    y_gold = [int(ex["gold"]) for ex in data]
    y_pred = [int(ex["pred"]) for ex in data]

    results = {
        "model_name": args.model_name,
        "eval_file": args.eval_file,
        "micro": pairwise_window_f1(y_gold, y_pred),
        "macro": pairwise_window_macro_f1(data),
    }

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()