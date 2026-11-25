import json
import argparse
from pathlib import Path
import numpy as np


def _prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return {"precision": p, "recall": r, "f1": f1}


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

    # Per-example metrics for macro
    macro_precisions = []
    macro_recalls = []
    macro_f1s = []
    kappas = []

    # Global counts for micro
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for ex in data:
        ex_f1 = ex["pairwise_f1_binary_FINAL_example"]
        macro_precisions.append(float(ex_f1["precision"]))
        macro_recalls.append(float(ex_f1["recall"]))
        macro_f1s.append(float(ex_f1["f1"]))

        total_tp += int(ex_f1["tp"])
        total_fp += int(ex_f1["fp"])
        total_fn += int(ex_f1["fn"])

        if "pairwise_cohens_kappa_binary_FINAL_example" in ex:
            kappas.append(float(ex["pairwise_cohens_kappa_binary_FINAL_example"]))

    results = {
        "model_name": args.model_name,
        "eval_file": args.eval_file,
        "micro": {
            "pairwise_micro": _prf(total_tp, total_fp, total_fn),
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "num_examples": len(data),
        },
        "macro": {
            "pairwise_macro": {
                "precision": float(np.mean(macro_precisions)) if macro_precisions else 0.0,
                "recall": float(np.mean(macro_recalls)) if macro_recalls else 0.0,
                "f1": float(np.mean(macro_f1s)) if macro_f1s else 0.0,
            },
            "num_examples": len(data),
        },
    }

    if kappas:
        results["kappa_macro"] = {
            "mean": float(np.mean(kappas)),
            "num_examples": len(kappas),
        }

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()