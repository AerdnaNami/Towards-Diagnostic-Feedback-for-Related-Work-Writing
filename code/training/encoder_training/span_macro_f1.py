import re
import json
import argparse
from pathlib import Path
from itertools import permutations


_TOKEN_RE_EVAL = re.compile(r"[A-Za-z0-9]+", re.UNICODE)


def _norm_span_eval(s: str):
    return " ".join(str(s).strip().strip('"').strip("'").split())


def _tokenize_span_eval(s: str):
    return _TOKEN_RE_EVAL.findall(_norm_span_eval(s).lower())


def _span_iou(a: str, b: str):
    ta = set(_tokenize_span_eval(a))
    tb = set(_tokenize_span_eval(b))
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def match_spans_iou(gold_spans, pred_spans, iou_threshold=0.5):
    gold = [_norm_span_eval(s) for s in (gold_spans or []) if _norm_span_eval(s)]
    pred = [_norm_span_eval(s) for s in (pred_spans or []) if _norm_span_eval(s)]

    if not gold and not pred:
        return 0, 0, 0
    if not gold:
        return 0, len(pred), 0
    if not pred:
        return 0, 0, len(gold)

    candidates = []
    for gi, g in enumerate(gold):
        for pi, p in enumerate(pred):
            iou = _span_iou(g, p)
            if iou >= iou_threshold:
                candidates.append((iou, gi, pi))

    candidates.sort(reverse=True, key=lambda x: x[0])
    matched_gold, matched_pred = set(), set()
    tp = 0

    for iou, gi, pi in candidates:
        if gi in matched_gold or pi in matched_pred:
            continue
        matched_gold.add(gi)
        matched_pred.add(pi)
        tp += 1

    return tp, len(pred) - tp, len(gold) - tp


def _prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return {"precision": p, "recall": r, "f1": f1}


def pairwise_span_f1_micro(gold_spans_per_example, pred_spans_per_example, iou_threshold=0.5):
    """
    Micro-averaged pairwise F1: pools TP/FP/FN across all examples before computing P/R/F1.
    """
    annotators = {"gold": gold_spans_per_example, "pred": pred_spans_per_example}
    n = min(len(gold_spans_per_example), len(pred_spans_per_example))

    total_tp = total_fp = total_fn = 0
    per_pair = {}

    for a, b in permutations(["gold", "pred"], 2):
        tp = fp = fn = 0
        for i in range(n):
            tpi, fpi, fni = match_spans_iou(
                annotators[a][i], annotators[b][i], iou_threshold=iou_threshold
            )
            tp += tpi
            fp += fpi
            fn += fni
        per_pair[f"{a}→{b}"] = _prf(tp, fp, fn)
        total_tp += tp
        total_fp += fp
        total_fn += fn

    return {
        "pairwise_micro": _prf(total_tp, total_fp, total_fn),
        "per_pair": per_pair,
        "num_examples": n,
        "iou_threshold": iou_threshold,
    }


def pairwise_span_f1_macro(gold_spans_per_example, pred_spans_per_example, iou_threshold=0.5):
    """
    Macro-averaged pairwise F1: computes P/R/F1 per example, then averages across examples.
    Examples where both gold and pred are empty are excluded from the average.
    """
    n = min(len(gold_spans_per_example), len(pred_spans_per_example))

    per_example_results = []
    precisions, recalls, f1s = [], [], []

    for i in range(n):
        gold = gold_spans_per_example[i]
        pred = pred_spans_per_example[i]

        # Skip examples where both are empty — no signal either way
        if not gold and not pred:
            per_example_results.append(None)
            continue

        # Compute bidirectional and average
        total_tp = total_fp = total_fn = 0
        for a_spans, b_spans in [(gold, pred), (pred, gold)]:
            tpi, fpi, fni = match_spans_iou(a_spans, b_spans, iou_threshold=iou_threshold)
            total_tp += tpi
            total_fp += fpi
            total_fn += fni

        prf = _prf(total_tp, total_fp, total_fn)
        per_example_results.append(prf)
        precisions.append(prf["precision"])
        recalls.append(prf["recall"])
        f1s.append(prf["f1"])

    n_scored = len(f1s)
    macro = {
        "precision": sum(precisions) / n_scored if n_scored else 0.0,
        "recall": sum(recalls) / n_scored if n_scored else 0.0,
        "f1": sum(f1s) / n_scored if n_scored else 0.0,
    }

    return {
        "pairwise_macro": macro,
        "num_examples_scored": n_scored,
        "num_examples_total": n,
        "iou_threshold": iou_threshold,
    }


def load_eval_file(path):
    """
    Load eval file. Expects a JSON list of objects with 'gold_spans' and 'pred_spans',
    each being a list of span dicts with a 'text' field, or a list of strings.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    gold_spans_per_example = []
    pred_spans_per_example = []

    for ex in data:
        gold = ex.get("gold_spans", [])
        pred = ex.get("pred_spans", [])

        # Support both list of dicts (with 'text' key) and list of strings
        gold = [s["text"] if isinstance(s, dict) else s for s in gold]
        pred = [s["text"] if isinstance(s, dict) else s for s in pred]

        gold_spans_per_example.append(gold)
        pred_spans_per_example.append(pred)

    return gold_spans_per_example, pred_spans_per_example


def main():
    parser = argparse.ArgumentParser(description="Evaluate pairwise span F1 (micro and macro).")
    parser.add_argument("--model_name", required=True, help="Name or identifier of the model being evaluated")
    parser.add_argument("--eval_file", required=True, help="Path to the eval JSON file with gold and pred spans")
    parser.add_argument("--out_path", required=True, help="Path to save the output results JSON")
    parser.add_argument("--iou_threshold", type=float, default=0.5, help="IoU threshold for span matching (default: 0.5)")

    args = parser.parse_args()

    print(f"Model:      {args.model_name}")
    print(f"Eval file:  {args.eval_file}")
    print(f"IoU threshold: {args.iou_threshold}")

    gold_spans_per_example, pred_spans_per_example = load_eval_file(args.eval_file)
    print(f"Loaded {len(gold_spans_per_example)} examples.")

    micro_results = pairwise_span_f1_micro(gold_spans_per_example, pred_spans_per_example, iou_threshold=args.iou_threshold)
    macro_results = pairwise_span_f1_macro(gold_spans_per_example, pred_spans_per_example, iou_threshold=args.iou_threshold)

    results = {
        "model_name": args.model_name,
        "eval_file": args.eval_file,
        "iou_threshold": args.iou_threshold,
        "micro": micro_results,
        "macro": macro_results,
    }

    print("\n----- RESULTS -----")
    print(json.dumps({k: v for k, v in results.items() if k != "micro"}, indent=2))
    print(json.dumps({"micro": results["micro"]}, indent=2))

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
