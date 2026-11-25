#!/usr/bin/env python3
"""Create point-range plots from confidence intervals in ``local_eval_runs``.

The script discovers each ``local_eval_runs/*/confidence_intervals.csv`` file,
combines model families, and writes one horizontal point-range PNG per task plus
a multipage PDF containing the same figures.

Examples:
    python plot_confidence_intervals.py
    python plot_confidence_intervals.py --view table
    python plot_confidence_intervals.py --runs-dir local_eval_runs --dpi 300
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RUNS_DIR = SCRIPT_DIR / "local_eval_runs"
# Matplotlib otherwise tries the user's home cache, which may be read-only on
# managed/cluster environments. Keep this cache alongside the generated data.
os.environ.setdefault("MPLCONFIGDIR", str(DEFAULT_RUNS_DIR / ".matplotlib_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D


MODEL_ORDER = ["XLM-R", "SciBERT", "ModernBERT", "NeoBERT"]
MODEL_COLORS = {
    "XLM-R": "#0072B2",
    "SciBERT": "#E69F00",
    "ModernBERT": "#009E73",
    "NeoBERT": "#CC79A7",
}
GROUP_ORDER = ["format", "unsupported_short", "unsupported_long", "coherence", "lacks_synthesis"]
GROUP_TITLES = {
    "format": "Format",
    "unsupported_short": "Unsupported Claim — Short",
    "unsupported_long": "Unsupported Claim — Long",
    "coherence": "Coherence",
    "lacks_synthesis": "Lacks Synthesis",
}

TABLE_METRICS = {
    "format": {
        "pairwise_span_f1.pairwise_micro.f1",
        "pairwise_span_f1.pairwise_macro.f1",
        "cohen_kappa_token_level",
    },
    "unsupported_short": {
        "pairwise_span_f1.pairwise_micro.f1",
        "pairwise_span_f1.pairwise_macro.f1",
        "cohen_kappa_token_level",
    },
    "unsupported_long": {
        "pairwise_span_f1.pairwise_micro.f1",
        "pairwise_span_f1.pairwise_macro.f1",
        "cohen_kappa_token_level",
    },
    "coherence": {
        "pairwise_window_f1.pairwise_micro.f1",
        "pairwise_window_f1.pairwise_macro.f1",
        "final_eval_metrics.cohen_kappa",
    },
    "lacks_synthesis": {
        "pairwise_window_f1.pairwise_micro.f1",
        "pairwise_window_f1.pairwise_macro.f1",
        "final_eval_metrics.cohen_kappa",
    },
}
MICRO_METRICS = {
    group: next(metric for metric in metrics if "pairwise_micro.f1" in metric)
    for group, metrics in TABLE_METRICS.items()
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: RUNS_DIR/confidence_interval_plots)",
    )
    parser.add_argument(
        "--view",
        choices=("all", "table", "micro"),
        default="all",
        help="Plot every metric, the table metrics, or only pairwise micro F1.",
    )
    parser.add_argument("--dpi", type=int, default=250)
    return parser.parse_args()


def model_family(path: Path, run_name: str) -> str:
    text = f"{path.parent.name} {run_name}".lower()
    if "xlmr" in text or "xlm-r" in text or "xlm_roberta" in text:
        return "XLM-R"
    if "scibert" in text:
        return "SciBERT"
    if "modernbert" in text:
        return "ModernBERT"
    if "neobert" in text:
        return "NeoBERT"
    return path.parent.name


def evaluation_group(row: Mapping[str, str]) -> str:
    category = row["category_name"].strip().lower()
    run_name = row["run_name"].lower()
    if category == "unsupported_claim":
        if "short" in run_name:
            return "unsupported_short"
        if "long" in run_name:
            return "unsupported_long"
        raise ValueError(f"Cannot determine short/long unsupported-claim task from {row['run_name']!r}")
    return category


def read_rows(runs_dir: Path) -> List[Dict[str, object]]:
    csv_paths = sorted(runs_dir.glob("*/confidence_intervals.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No */confidence_intervals.csv files found under {runs_dir}")

    rows: List[Dict[str, object]] = []
    metric_order: Dict[str, int] = {}
    required = {"run_name", "category_name", "metric", "estimate", "ci_lower", "ci_upper"}
    for csv_path in csv_paths:
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")
            for raw in reader:
                group = evaluation_group(raw)
                metric = raw["metric"]
                metric_order.setdefault(metric, len(metric_order))
                estimate = float(raw["estimate"])
                lower = float(raw["ci_lower"])
                upper = float(raw["ci_upper"])
                if lower > estimate or estimate > upper:
                    raise ValueError(
                        f"Invalid interval in {csv_path}: {lower} <= {estimate} <= {upper} is false"
                    )
                rows.append(
                    {
                        **raw,
                        "source": str(csv_path),
                        "model": model_family(csv_path, raw["run_name"]),
                        "group": group,
                        "metric_order": metric_order[metric],
                        "estimate": estimate,
                        "lower": lower,
                        "upper": upper,
                    }
                )
    return rows


def friendly_metric(metric: str) -> str:
    exact = {
        "cohen_kappa_token_level": "Cohen's kappa (pooled)",
        "cohen_kappa_token_level_macro": "Cohen's kappa (macro)",
        "final_eval_metrics.cohen_kappa": "Cohen's kappa (pooled)",
        "final_eval_metrics.cohen_kappa_macro_by_document": "Cohen's kappa (macro)",
        "pairwise_span_f1.pairwise_micro.f1": "Pairwise span F1 (micro)",
        "pairwise_span_f1.pairwise_macro.f1": "Pairwise span F1 (macro)",
        "pairwise_window_f1.pairwise_micro.f1": "Pairwise window F1 (micro)",
        "pairwise_window_f1.pairwise_macro.f1": "Pairwise window F1 (macro)",
    }
    if metric in exact:
        return exact[metric]
    label = metric.replace("final_eval_metrics.", "")
    label = label.replace("pairwise_span_f1.", "span ")
    label = label.replace("pairwise_window_f1.", "window ")
    label = label.replace("sentence_label_from_bio_coverage.", "sentence coverage ")
    label = label.replace("token_overlap_f1_entity_vs_O.", "token overlap ")
    label = label.replace("word_level.", "word ")
    label = label.replace("pairwise_micro.", "micro ")
    label = label.replace("pairwise_macro.", "macro ")
    label = re.sub(r"[._]+", " ", label)
    return label.strip().capitalize()


def group_rows(
    rows: Sequence[Mapping[str, object]],
    group: str,
    view: str,
) -> List[Mapping[str, object]]:
    selected = [row for row in rows if row["group"] == group]
    if view == "table":
        selected = [row for row in selected if row["metric"] in TABLE_METRICS[group]]
    elif view == "micro":
        selected = [row for row in selected if row["metric"] == MICRO_METRICS[group]]
    return selected


def metric_names(rows: Sequence[Mapping[str, object]]) -> List[str]:
    order = {}
    for row in rows:
        metric = str(row["metric"])
        order.setdefault(metric, int(row["metric_order"]))
    return sorted(order, key=order.get)


def models_present(rows: Iterable[Mapping[str, object]]) -> List[str]:
    found = {str(row["model"]) for row in rows}
    ordered = [model for model in MODEL_ORDER if model in found]
    return ordered + sorted(found - set(ordered))


def make_figure(
    rows: Sequence[Mapping[str, object]],
    group: str,
    view: str,
):
    selected = group_rows(rows, group, view)
    metrics = metric_names(selected)
    models = models_present(selected)
    if not selected or not metrics:
        raise ValueError(f"No rows available for group={group!r}, view={view!r}")

    height = max(3.0, 0.48 * len(metrics) + 1.8)
    fig, ax = plt.subplots(figsize=(10.5, height), constrained_layout=True)
    y_positions = {metric: len(metrics) - 1 - index for index, metric in enumerate(metrics)}
    offsets = {
        model: (index - (len(models) - 1) / 2.0) * min(0.16, 0.62 / max(1, len(models)))
        for index, model in enumerate(models)
    }

    seen = set()
    for row in selected:
        key = (row["model"], row["metric"])
        if key in seen:
            raise ValueError(f"Duplicate model/metric row for {group}: {key}")
        seen.add(key)
        model = str(row["model"])
        y = y_positions[str(row["metric"])] + offsets[model]
        estimate = float(row["estimate"])
        lower = float(row["lower"])
        upper = float(row["upper"])
        ax.errorbar(
            estimate,
            y,
            xerr=[[estimate - lower], [upper - estimate]],
            fmt="o",
            markersize=5.2,
            capsize=2.5,
            elinewidth=1.35,
            color=MODEL_COLORS.get(model, "#555555"),
            markeredgecolor="white",
            markeredgewidth=0.45,
            zorder=3,
        )

    ax.axvline(0.0, color="#777777", linewidth=0.8, linestyle="--", zorder=1)
    ax.set_xlim(-1.02, 1.02)
    ax.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax.set_yticks([y_positions[metric] for metric in metrics])
    ax.set_yticklabels([friendly_metric(metric) for metric in metrics], fontsize=8.5)
    ax.set_xlabel("Point estimate with 95% confidence interval")
    ax.set_title(GROUP_TITLES[group], loc="left", fontweight="bold")
    ax.grid(axis="x", color="#dddddd", linewidth=0.7)
    ax.grid(axis="y", color="#eeeeee", linewidth=0.5)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            label=model,
            markerfacecolor=MODEL_COLORS.get(model, "#555555"),
            markeredgecolor="white",
            markersize=7,
        )
        for model in models
    ]
    legend_options = (
        {"loc": "upper right", "ncol": min(2, len(models))}
        if view == "micro"
        else {
            "loc": "upper center",
            "bbox_to_anchor": (0.5, -0.055),
            "ncol": min(4, len(models)),
        }
    )
    ax.legend(handles=handles, title="Model", frameon=False, **legend_options)
    titles = {
        "all": "Model confidence intervals",
        "table": "Table metrics: Pairwise F1 and Cohen's kappa",
        "micro": "Pairwise micro-F1 confidence intervals",
    }
    fig.suptitle(
        titles[view],
        fontsize=13,
        fontweight="bold",
    )
    return fig


def main() -> None:
    args = parse_args()
    if args.dpi < 72:
        raise ValueError("--dpi must be at least 72")
    runs_dir = args.runs_dir.resolve()
    output_dir = (args.output_dir or runs_dir / "confidence_interval_plots").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_rows(runs_dir)

    pdf_path = output_dir / f"{args.view}_confidence_intervals.pdf"
    written = []
    with PdfPages(pdf_path) as pdf:
        for group in GROUP_ORDER:
            fig = make_figure(rows, group, args.view)
            png_path = output_dir / f"{args.view}_{group}_point_range.png"
            fig.savefig(png_path, dpi=args.dpi, bbox_inches="tight", facecolor="white")
            pdf.savefig(fig, bbox_inches="tight", facecolor="white")
            plt.close(fig)
            written.append(png_path)

    print(f"Loaded {len(rows)} confidence-interval rows from {runs_dir}")
    for path in written:
        print(f"Saved PNG: {path}")
    print(f"Saved PDF: {pdf_path}")


if __name__ == "__main__":
    main()
