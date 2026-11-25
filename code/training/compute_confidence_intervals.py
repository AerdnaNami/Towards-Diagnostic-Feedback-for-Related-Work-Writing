#!/usr/bin/env python3
"""Bootstrap confidence intervals for every model under ``local_eval_runs``.

The script resamples evaluation examples for sentence/short-span tasks and whole
documents (including all their windows) for paragraph tasks.  This preserves the
evaluation unit used by the macro metrics and avoids treating overlapping windows
from the same document as independent observations.

Example:
    python compute_confidence_intervals.py
    python compute_confidence_intervals.py --bootstrap-samples 10000 --seed 42
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


SPAN_TOKEN_RE = re.compile(r"[A-Za-z0-9]+", re.UNICODE)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def prf(tp: float, fp: float, fn: float) -> Tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return float(precision), float(recall), float(f1)


def kappa_from_confusion(matrix: np.ndarray) -> float:
    total = float(matrix.sum())
    if total <= 0:
        return 0.0
    observed = float(np.trace(matrix)) / total
    expected = float(np.dot(matrix.sum(axis=1), matrix.sum(axis=0))) / (total * total)
    if expected >= 1.0:
        return 0.0
    value = (observed - expected) / (1.0 - expected)
    return 0.0 if not np.isfinite(value) else float(value)


def classification_metrics(matrix: np.ndarray, positive_ids: Sequence[int]) -> Dict[str, float]:
    positive = np.asarray(list(positive_ids), dtype=int)
    negative = np.asarray([i for i in range(matrix.shape[0]) if i not in set(positive)], dtype=int)
    tp = float(matrix[np.ix_(positive, positive)].sum()) if positive.size else 0.0
    fp = float(matrix[np.ix_(negative, positive)].sum()) if negative.size and positive.size else 0.0
    fn = float(matrix[np.ix_(positive, negative)].sum()) if positive.size and negative.size else 0.0
    precision, recall, f1 = prf(tp, fp, fn)
    total = float(matrix.sum())
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": float(np.trace(matrix) / total) if total else 0.0,
        "cohen_kappa": kappa_from_confusion(matrix),
    }


def binary_confusion(gold: Sequence[int], pred: Sequence[int]) -> np.ndarray:
    matrix = np.zeros((2, 2), dtype=np.int64)
    for gold_label, pred_label in zip(gold, pred):
        matrix[int(gold_label), int(pred_label)] += 1
    return matrix


def norm_span(text: str) -> str:
    return " ".join(str(text).strip().strip('"').strip("'").split())


def span_iou(left: str, right: str) -> float:
    left_tokens = set(SPAN_TOKEN_RE.findall(norm_span(left).lower()))
    right_tokens = set(SPAN_TOKEN_RE.findall(norm_span(right).lower()))
    if not left_tokens and not right_tokens:
        return 1.0
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def span_texts(spans: Iterable[Any]) -> List[str]:
    return [str(span.get("text", "")) if isinstance(span, dict) else str(span) for span in spans]


def match_spans_iou(gold_spans: Sequence[str], pred_spans: Sequence[str], threshold: float) -> Tuple[int, int, int]:
    gold = [norm_span(span) for span in gold_spans if norm_span(span)]
    pred = [norm_span(span) for span in pred_spans if norm_span(span)]
    candidates = []
    for gold_index, gold_span in enumerate(gold):
        for pred_index, pred_span in enumerate(pred):
            overlap = span_iou(gold_span, pred_span)
            if overlap >= threshold:
                candidates.append((overlap, gold_index, pred_index))
    candidates.sort(reverse=True)
    matched_gold = set()
    matched_pred = set()
    true_positive = 0
    for _, gold_index, pred_index in candidates:
        if gold_index in matched_gold or pred_index in matched_pred:
            continue
        matched_gold.add(gold_index)
        matched_pred.add(pred_index)
        true_positive += 1
    return true_positive, len(pred) - true_positive, len(gold) - true_positive


def exact_span_counts(gold_spans: Sequence[Mapping[str, Any]], pred_spans: Sequence[Mapping[str, Any]]) -> Tuple[int, int, int]:
    def keys(spans: Sequence[Mapping[str, Any]]) -> set:
        return {
            (str(span.get("label", "")), int(span.get("start_token", -1)), int(span.get("end_token", -1)))
            for span in spans
        }

    gold = keys(gold_spans)
    pred = keys(pred_spans)
    return len(gold & pred), len(pred - gold), len(gold - pred)


def labels_from_spans(example: Mapping[str, Any], key: str) -> List[str]:
    spans = example.get(key, []) or []
    token_count = len(str(example.get("text", "")).split())
    for span in spans:
        if isinstance(span, dict):
            token_count = max(token_count, int(span.get("end_token", 0)))
    labels = ["O"] * token_count
    for span in sorted(spans, key=lambda item: (int(item.get("start_token", 0)), int(item.get("end_token", 0)))):
        if not isinstance(span, dict):
            continue
        start = max(0, min(token_count, int(span.get("start_token", 0))))
        end = max(start, min(token_count, int(span.get("end_token", start))))
        entity = str(span.get("label", "ENTITY"))
        for index in range(start, end):
            labels[index] = f"B-{entity}" if index == start else f"I-{entity}"
    return labels


def record_word_label_sequences(example: Mapping[str, Any]) -> Tuple[List[str], List[str]]:
    if isinstance(example.get("gold_labels"), list) and isinstance(example.get("pred_labels"), list):
        gold = [str(label) for label in example["gold_labels"]]
        pred = [str(label) for label in example["pred_labels"]]
    else:
        gold = labels_from_spans(example, "gold_spans")
        pred = labels_from_spans(example, "pred_spans")
    size = min(len(gold), len(pred))
    return gold[:size], pred[:size]


def record_label_sequences(example: Mapping[str, Any]) -> Tuple[List[str], List[str]]:
    gold_binary = example.get("gold_binary")
    pred_binary = example.get("pred_binary", example.get("final_pred_binary"))
    if isinstance(gold_binary, list) and isinstance(pred_binary, list):
        size = min(len(gold_binary), len(pred_binary))
        return (
            ["ENTITY" if int(value) == 1 else "O" for value in gold_binary[:size]],
            ["ENTITY" if int(value) == 1 else "O" for value in pred_binary[:size]],
        )
    if example.get("postprocessing") and isinstance(example.get("gold_labels"), list) and isinstance(example.get("pred_labels"), list):
        return record_word_label_sequences(example)
    if isinstance(example.get("aligned_gold_labels"), list) and isinstance(example.get("aligned_pred_labels"), list):
        gold = [str(label) for label in example["aligned_gold_labels"]]
        pred = [str(label) for label in example["aligned_pred_labels"]]
        size = min(len(gold), len(pred))
        return gold[:size], pred[:size]
    return record_word_label_sequences(example)


def bootstrap_intervals(
    unit_count: int,
    metric_function: Callable[[np.ndarray], Dict[str, float]],
    *,
    samples: int,
    confidence: float,
    rng: np.random.Generator,
) -> Dict[str, Dict[str, float]]:
    if unit_count <= 0:
        raise ValueError("Cannot bootstrap an empty evaluation set")
    all_indices = np.arange(unit_count, dtype=int)
    estimates = metric_function(all_indices)
    draws = {name: np.empty(samples, dtype=float) for name in estimates}
    for sample_index in range(samples):
        indices = rng.integers(0, unit_count, size=unit_count)
        values = metric_function(indices)
        for name, value in values.items():
            draws[name][sample_index] = value

    tail = (1.0 - confidence) / 2.0
    output = {}
    for name, estimate in estimates.items():
        lower, upper = np.quantile(draws[name], [tail, 1.0 - tail])
        output[name] = {
            "estimate": float(estimate),
            "lower": float(lower),
            "upper": float(upper),
        }
    return output


def token_span_unit_arrays(
    records: Sequence[Mapping[str, Any]],
    *,
    iou_threshold: float,
    coverage_threshold: float,
) -> Dict[str, np.ndarray]:
    sequences = [record_label_sequences(record) for record in records]
    word_sequences = [record_word_label_sequences(record) for record in records]
    label_names = sorted({label for pair in sequences for sequence in pair for label in sequence}, key=lambda label: (label != "O", label))
    if "O" not in label_names:
        label_names.insert(0, "O")
    label_to_id = {label: index for index, label in enumerate(label_names)}
    matrices = np.zeros((len(records), len(label_names), len(label_names)), dtype=np.int64)
    kappas = np.zeros(len(records), dtype=float)
    span_counts = np.zeros((len(records), 3), dtype=np.int64)
    exact_counts = np.zeros((len(records), 3), dtype=np.int64)
    span_macro = np.zeros((len(records), 3), dtype=float)
    span_macro_valid = np.zeros(len(records), dtype=bool)
    coverage_labels = np.zeros((len(records), 2), dtype=np.int64)
    sequence_exact = np.zeros(len(records), dtype=float)

    for index, (record, (gold_labels, pred_labels), (gold_word_labels, pred_word_labels)) in enumerate(
        zip(records, sequences, word_sequences)
    ):
        for gold_label, pred_label in zip(gold_labels, pred_labels):
            matrices[index, label_to_id[gold_label], label_to_id[pred_label]] += 1
        kappas[index] = kappa_from_confusion(matrices[index])
        sequence_exact[index] = float(gold_labels == pred_labels)

        gold_spans = record.get("gold_spans", []) or []
        pred_spans = record.get("pred_spans", []) or []
        tp, fp, fn = match_spans_iou(span_texts(gold_spans), span_texts(pred_spans), iou_threshold)
        span_counts[index] = (2 * tp, fp + fn, fn + fp)
        exact_counts[index] = exact_span_counts(gold_spans, pred_spans)
        if gold_spans or pred_spans:
            span_macro_valid[index] = True
            span_macro[index] = prf(*span_counts[index])

        token_count = len(gold_word_labels)
        gold_coverage = sum(label != "O" for label in gold_word_labels) / token_count if token_count else 0.0
        pred_coverage = sum(label != "O" for label in pred_word_labels) / token_count if token_count else 0.0
        coverage_labels[index] = (gold_coverage >= coverage_threshold, pred_coverage >= coverage_threshold)

    return {
        "matrices": matrices,
        "kappas": kappas,
        "span_counts": span_counts,
        "exact_counts": exact_counts,
        "span_macro": span_macro,
        "span_macro_valid": span_macro_valid,
        "coverage_labels": coverage_labels,
        "sequence_exact": sequence_exact,
        "positive_ids": np.asarray([index for label, index in label_to_id.items() if label != "O"], dtype=int),
    }


def token_span_metric_function(
    arrays: Mapping[str, np.ndarray],
    *,
    include_coverage: bool,
    token_metric_prefix: str,
) -> Callable[[np.ndarray], Dict[str, float]]:
    def metrics(indices: np.ndarray) -> Dict[str, float]:
        matrix = arrays["matrices"][indices].sum(axis=0)
        token = classification_metrics(matrix, arrays["positive_ids"])
        span_precision, span_recall, span_f1 = prf(*arrays["span_counts"][indices].sum(axis=0))
        exact_span_f1 = prf(*arrays["exact_counts"][indices].sum(axis=0))[2]
        valid = arrays["span_macro_valid"][indices]
        macro_values = arrays["span_macro"][indices][valid]
        result = {
            f"{token_metric_prefix}token_overlap_f1_entity_vs_O.precision": token["precision"],
            f"{token_metric_prefix}token_overlap_f1_entity_vs_O.recall": token["recall"],
            f"{token_metric_prefix}token_overlap_f1_entity_vs_O.f1": token["f1"],
            f"{token_metric_prefix}token_accuracy": token["accuracy"],
            f"{token_metric_prefix}cohen_kappa_token_level": token["cohen_kappa"],
            f"{token_metric_prefix}cohen_kappa_token_level_macro": float(arrays["kappas"][indices].mean()),
            "pairwise_span_f1.pairwise_micro.precision": span_precision,
            "pairwise_span_f1.pairwise_micro.recall": span_recall,
            "pairwise_span_f1.pairwise_micro.f1": span_f1,
            "pairwise_span_f1.pairwise_macro.precision": float(macro_values[:, 0].mean()) if len(macro_values) else 0.0,
            "pairwise_span_f1.pairwise_macro.recall": float(macro_values[:, 1].mean()) if len(macro_values) else 0.0,
            "pairwise_span_f1.pairwise_macro.f1": float(macro_values[:, 2].mean()) if len(macro_values) else 0.0,
            "word_level.exact_span_f1": exact_span_f1,
            f"{token_metric_prefix}sequence_exact_match": float(arrays["sequence_exact"][indices].mean()),
        }
        if include_coverage:
            coverage = arrays["coverage_labels"][indices]
            coverage_metrics = classification_metrics(binary_confusion(coverage[:, 0], coverage[:, 1]), [1])
            for name, value in coverage_metrics.items():
                result[f"sentence_label_from_bio_coverage.{name}"] = value
        return result

    return metrics


def paragraph_units(records: Sequence[Mapping[str, Any]]) -> Tuple[List[Any], np.ndarray]:
    grouped: Dict[Any, List[Mapping[str, Any]]] = defaultdict(list)
    for index, record in enumerate(records):
        doc_index = record.get("doc_index")
        grouped[("row", index) if doc_index is None else ("doc", doc_index)].append(record)
    group_ids = list(grouped)
    matrices = np.zeros((len(group_ids), 2, 2), dtype=np.int64)
    for index, group_id in enumerate(group_ids):
        gold = [int(record["gold"]) for record in grouped[group_id]]
        pred = [int(record["pred"]) for record in grouped[group_id]]
        matrices[index] = binary_confusion(gold, pred)
    return group_ids, matrices


def paragraph_metric_function(matrices: np.ndarray) -> Callable[[np.ndarray], Dict[str, float]]:
    document_kappas = np.asarray([kappa_from_confusion(matrix) for matrix in matrices])
    document_pairwise = np.asarray([prf(2 * matrix[1, 1], matrix[0, 1] + matrix[1, 0], matrix[1, 0] + matrix[0, 1]) for matrix in matrices])

    def metrics(indices: np.ndarray) -> Dict[str, float]:
        matrix = matrices[indices].sum(axis=0)
        pooled = classification_metrics(matrix, [1])
        pairwise = prf(2 * matrix[1, 1], matrix[0, 1] + matrix[1, 0], matrix[1, 0] + matrix[0, 1])
        macro_pairwise = document_pairwise[indices].mean(axis=0)
        return {
            "final_eval_metrics.precision": pooled["precision"],
            "final_eval_metrics.recall": pooled["recall"],
            "final_eval_metrics.f1": pooled["f1"],
            "final_eval_metrics.accuracy": pooled["accuracy"],
            "final_eval_metrics.cohen_kappa": pooled["cohen_kappa"],
            "final_eval_metrics.cohen_kappa_macro_by_document": float(document_kappas[indices].mean()),
            "pairwise_window_f1.pairwise_micro.precision": pairwise[0],
            "pairwise_window_f1.pairwise_micro.recall": pairwise[1],
            "pairwise_window_f1.pairwise_micro.f1": pairwise[2],
            "pairwise_window_f1.pairwise_macro.precision": float(macro_pairwise[0]),
            "pairwise_window_f1.pairwise_macro.recall": float(macro_pairwise[1]),
            "pairwise_window_f1.pairwise_macro.f1": float(macro_pairwise[2]),
        }

    return metrics


def prediction_path(run_dir: Path, task: str) -> Path:
    paths = {
        "paragraph": run_dir / "paragraph" / "paragraph_eval_window_predictions.json",
        "sentence": run_dir / "sentence" / "sentence_eval_predicted_spans.json",
        "short": run_dir / "short" / "short_unsupp_eval_predicted_spans.json",
    }
    if task not in paths:
        raise ValueError(f"Unsupported evaluation task: {task!r}")
    return paths[task]


def evaluate_run(
    summary_path: Path,
    *,
    bootstrap_samples: int,
    confidence: float,
    seed: int,
) -> Dict[str, Any]:
    summary = read_json(summary_path)
    task = str(summary["task"])
    run_name = str(summary.get("run_name", summary_path.parent.name))
    predictions_path = prediction_path(summary_path.parent, task)
    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions not found: {predictions_path}")
    records = read_json(predictions_path)
    if not isinstance(records, list) or not records:
        raise ValueError(f"{predictions_path} must contain a non-empty JSON list")

    rng = np.random.default_rng(seed)
    task_summary = summary.get(task, {})
    if task == "paragraph":
        group_ids, matrices = paragraph_units(records)
        metric_function = paragraph_metric_function(matrices)
        unit_count = len(group_ids)
        unit_name = "documents"
        notes = []
    else:
        iou_threshold = float(task_summary.get("pairwise_span_f1", {}).get("iou_threshold", 0.5))
        coverage_threshold = float(task_summary.get("coverage_threshold", 0.5))
        arrays = token_span_unit_arrays(
            records,
            iou_threshold=iou_threshold,
            coverage_threshold=coverage_threshold,
        )
        has_aligned_labels = all(
            isinstance(record.get("aligned_gold_labels"), list)
            and isinstance(record.get("aligned_pred_labels"), list)
            for record in records
        )
        has_postprocessed_labels = all(
            bool(record.get("postprocessing"))
            and isinstance(record.get("gold_labels"), list)
            and isinstance(record.get("pred_labels"), list)
            for record in records
        )
        has_binary_labels = all(
            isinstance(record.get("gold_binary"), list)
            and isinstance(record.get("pred_binary", record.get("final_pred_binary")), list)
            for record in records
        )
        has_exact_labels = has_binary_labels or has_aligned_labels or has_postprocessed_labels
        token_metric_prefix = "" if has_exact_labels else "reconstructed_word_level."
        metric_function = token_span_metric_function(
            arrays,
            include_coverage=task == "sentence",
            token_metric_prefix=token_metric_prefix,
        )
        unit_count = len(records)
        unit_name = "examples"
        notes = [] if has_exact_labels else [
            "The saved predictions predate aligned_gold_labels/aligned_pred_labels. "
            "Token and exact-span metrics are therefore reconstructed at word level and use the "
            "reconstructed_word_level prefix; pairwise span metrics remain exact. Rerun evaluation "
            "with the updated evaluate_models.py to obtain aligned-subtoken confidence intervals."
        ]

    intervals = bootstrap_intervals(
        unit_count,
        metric_function,
        samples=bootstrap_samples,
        confidence=confidence,
        rng=rng,
    )
    return {
        "run_name": run_name,
        "task": task,
        "category_name": summary.get("category_name"),
        "model_path": summary.get("model_path"),
        "predictions_path": str(predictions_path),
        "bootstrap_unit": unit_name,
        "num_bootstrap_units": unit_count,
        "notes": notes,
        "metrics": intervals,
    }


def write_csv(path: Path, runs: Sequence[Mapping[str, Any]], confidence: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "run_name",
                "task",
                "category_name",
                "metric",
                "estimate",
                "ci_lower",
                "ci_upper",
                "confidence_level",
                "bootstrap_unit",
                "num_bootstrap_units",
            ],
        )
        writer.writeheader()
        for run in runs:
            for metric, interval in run["metrics"].items():
                writer.writerow(
                    {
                        "run_name": run["run_name"],
                        "task": run["task"],
                        "category_name": run["category_name"],
                        "metric": metric,
                        "estimate": interval["estimate"],
                        "ci_lower": interval["lower"],
                        "ci_upper": interval["upper"],
                        "confidence_level": confidence,
                        "bootstrap_unit": run["bootstrap_unit"],
                        "num_bootstrap_units": run["num_bootstrap_units"],
                    }
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=Path(__file__).resolve().parent / "local_eval_runs")
    parser.add_argument("--output", type=Path, default=None, help="JSON output path (default: RUNS_DIR/confidence_intervals.json)")
    parser.add_argument("--csv-output", type=Path, default=None, help="CSV output path (default: RUNS_DIR/confidence_intervals.csv)")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.bootstrap_samples < 1:
        raise ValueError("--bootstrap-samples must be at least 1")
    if not 0.0 < args.confidence < 1.0:
        raise ValueError("--confidence must be between 0 and 1")

    runs_dir = args.runs_dir.resolve()
    summary_paths = sorted(path for path in runs_dir.glob("*/summary.json") if path.is_file())
    if not summary_paths:
        raise FileNotFoundError(f"No model summary files found under {runs_dir}")

    runs = []
    errors = []
    for run_index, summary_path in enumerate(summary_paths):
        try:
            run = evaluate_run(
                summary_path,
                bootstrap_samples=args.bootstrap_samples,
                confidence=args.confidence,
                seed=args.seed + run_index,
            )
            runs.append(run)
            print(f"Computed intervals for {run['run_name']} ({run['num_bootstrap_units']} {run['bootstrap_unit']})")
        except Exception as error:
            errors.append({"summary_path": str(summary_path), "error": f"{type(error).__name__}: {error}"})
            print(f"Skipped {summary_path.parent.name}: {error}")

    payload = {
        "method": "nonparametric percentile bootstrap",
        "confidence_level": args.confidence,
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "resampling": {
            "sentence": "evaluation examples with replacement",
            "short": "evaluation examples with replacement",
            "paragraph": "documents with replacement; all windows retained within each sampled document",
        },
        "runs": runs,
        "errors": errors,
    }
    output_path = (args.output or runs_dir / "confidence_intervals.json").resolve()
    csv_path = (args.csv_output or runs_dir / "confidence_intervals.csv").resolve()
    write_json(output_path, payload)
    write_csv(csv_path, runs, args.confidence)
    print(f"Saved JSON: {output_path}")
    print(f"Saved CSV:  {csv_path}")
    if errors:
        raise SystemExit(f"Completed with {len(errors)} skipped run(s); inspect the JSON errors field")


if __name__ == "__main__":
    main()
