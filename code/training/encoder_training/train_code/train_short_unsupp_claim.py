import json
import argparse
from pathlib import Path
from typing import Dict, List, Any
import re
from itertools import permutations
import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, cohen_kappa_score

class DataCollatorForTokenClassificationWithWordStarts(DataCollatorForTokenClassification):
    def torch_call(self, features):
        word_starts = [f.pop("word_starts") for f in features]
        batch = super().torch_call(features)

        max_len = batch["input_ids"].shape[1]
        padded_ws = []
        for ws in word_starts:
            ws = ws[:max_len]
            ws = ws + [0] * (max_len - len(ws))
            padded_ws.append(ws)

        batch["word_starts"] = torch.tensor(padded_ws, dtype=torch.long)
        return batch

# ----------------------------
# Config + Data
# ----------------------------
def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_json_dataset(path: str) -> Dataset:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"{path} must be a JSON list of examples")

    cleaned = []
    for i, ex in enumerate(data):
        if not isinstance(ex, dict):
            continue
        toks = ex.get("tokens")
        labs = ex.get("labels")
        if not isinstance(toks, list) or not isinstance(labs, list):
            raise ValueError(f"{path} example {i}: missing tokens/labels lists")
        if len(toks) != len(labs):
            raise ValueError(
                f"{path} example {i}: len(tokens)={len(toks)} != len(labels)={len(labs)}"
            )

        toks = [str(t) for t in toks]
        labs = [str(l) for l in labs]

        cleaned.append(
            {
                "tokens": toks,
                "labels": labs,
                "text": str(ex.get("text", " ".join(toks))),
                "filename": str(ex.get("filename", "")),
            }
        )

    return Dataset.from_list(cleaned)


# ----------------------------
# BIO helpers
# ----------------------------
def _ent_type(lbl: str) -> str:
    lbl = (lbl or "O").strip()
    if "-" in lbl:
        return lbl.split("-", 1)[1].strip()
    return ""


def bio_fix_sequence(labels: List[str]) -> List[str]:
    fixed = []
    prev = "O"
    for lbl in labels:
        lbl = (lbl or "O").strip()

        if lbl.startswith("I-"):
            x = _ent_type(lbl)
            if not (prev == f"B-{x}" or prev == f"I-{x}"):
                lbl = f"B-{x}"

        if lbl.startswith("I-") and prev.startswith(("B-", "I-")):
            x = _ent_type(lbl)
            if _ent_type(prev) != x:
                lbl = f"B-{x}"

        fixed.append(lbl)
        prev = lbl
    return fixed


def to_binary_entity_labels(word_labels: List[str]) -> List[str]:
    out = []
    for l in word_labels:
        s = (l or "O").strip()
        out.append("O" if s == "O" else "ENTITY")
    return out


# ----------------------------
# Tokenization + alignment
# ----------------------------
def tokenize_and_align(
    example: Dict[str, Any],
    tokenizer,
    label2id: Dict[str, int],
    label_all_subtokens: bool = True,
) -> Dict[str, Any]:
    max_len = getattr(tokenizer, "model_max_length", 512) or 512
    max_len = min(int(max_len), 512)

    tokenized = tokenizer(
        example["tokens"],
        is_split_into_words=True,
        truncation=True,
        max_length=max_len,
        padding=False,
    )

    word_ids = tokenized.word_ids()
    aligned = []
    word_starts = []
    prev_word = None

    for widx in word_ids:
        if widx is None:
            aligned.append(-100)
            word_starts.append(0)
            continue

        wlab = example["labels"][widx]
        if widx != prev_word:
            aligned.append(label2id[wlab])
            word_starts.append(1)
        else:
            if label_all_subtokens:
                aligned.append(label2id[wlab])
            else:
                aligned.append(-100)
            word_starts.append(0)

        prev_word = widx

    tokenized["labels"] = aligned
    tokenized["word_starts"] = word_starts
    return tokenized


# ----------------------------
# Eval helpers (binary)
# ----------------------------
def flatten_valid(pred_ids: np.ndarray, label_ids: np.ndarray):
    flat_p, flat_l = [], []
    for p_seq, l_seq in zip(pred_ids, label_ids):
        for p, l in zip(p_seq, l_seq):
            if l == -100:
                continue
            flat_p.append(int(p))
            flat_l.append(int(l))
    return np.array(flat_p, dtype=np.int64), np.array(flat_l, dtype=np.int64)


def exact_match_seq(pred_ids: np.ndarray, label_ids: np.ndarray) -> float:
    exact = 0
    total = 0
    for p_seq, l_seq in zip(pred_ids, label_ids):
        mask = (l_seq != -100)
        if int(mask.sum()) == 0:
            continue
        total += 1
        if np.all(p_seq[mask] == l_seq[mask]):
            exact += 1
    return float(exact / total) if total else 0.0


def token_overlap_f1_binary_from_ids(flat_pred: np.ndarray, flat_gold: np.ndarray, entity_id: int) -> Dict[str, Any]:
    if flat_gold.size == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0, "fp": 0, "fn": 0}

    gold_pos = (flat_gold == entity_id).astype(np.int64)
    pred_pos = (flat_pred == entity_id).astype(np.int64)

    tp = int(((pred_pos == 1) & (gold_pos == 1)).sum())
    fp = int(((pred_pos == 1) & (gold_pos == 0)).sum())
    fn = int(((pred_pos == 0) & (gold_pos == 1)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {"precision": float(precision), "recall": float(recall), "f1": float(f1), "tp": tp, "fp": fp, "fn": fn}


def compute_metrics_fn_binary(label2id: Dict[str, int]):
    entity_id = int(label2id["ENTITY"])

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)

        flat_p, flat_l = flatten_valid(preds, labels)
        if flat_l.size == 0:
            return {"entity_precision": 0.0, "entity_recall": 0.0, "entity_f1": 0.0, "token_accuracy": 0.0}

        gold_pos = (flat_l == entity_id).astype(np.int64)
        pred_pos = (flat_p == entity_id).astype(np.int64)
        p, r, f1, _ = precision_recall_fscore_support(gold_pos, pred_pos, average="binary", zero_division=0)

        acc = float((flat_p == flat_l).mean())

        return {
            "entity_precision": float(p),
            "entity_recall": float(r),
            "entity_f1": float(f1),
            "token_accuracy": float(acc),
        }

    return compute_metrics


_SPAN_TOKEN_RE = re.compile(r"[A-Za-z0-9]+", re.UNICODE)


def _norm_span(s: str) -> str:
    return " ".join(str(s).strip().strip('"').strip("'").split())


def _span_iou(a: str, b: str) -> float:
    ta = set(_SPAN_TOKEN_RE.findall(_norm_span(a).lower()))
    tb = set(_SPAN_TOKEN_RE.findall(_norm_span(b).lower()))
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def match_spans_iou(
    gold_spans: List[str],
    pred_spans: List[str],
    iou_threshold: float = 0.5,
):
    gold = [_norm_span(s) for s in (gold_spans or []) if _norm_span(s)]
    pred = [_norm_span(s) for s in (pred_spans or []) if _norm_span(s)]

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


def _prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return {"precision": p, "recall": r, "f1": f1}


def binary_spans_from_word_labels(tokens: List[str], word_labels: List[str]) -> List[Dict[str, Any]]:
    spans = []
    i = 0
    n = min(len(tokens), len(word_labels))

    while i < n:
        lab = (word_labels[i] or "O").strip()
        if lab != "ENTITY":
            i += 1
            continue

        start = i
        i += 1
        while i < n and (word_labels[i] or "O").strip() == "ENTITY":
            i += 1

        end = i
        spans.append({
            "label": "ENTITY",
            "start_token": start,
            "end_token": end,
            "text": " ".join(tokens[start:end]),
        })

    return spans


def pairwise_span_f1(
    gold_spans_per_example: List[List[str]],
    pred_spans_per_example: List[List[str]],
    iou_threshold: float = 0.5,
) -> Dict[str, Any]:
    n = min(len(gold_spans_per_example), len(pred_spans_per_example))
    ann = {"gold": gold_spans_per_example[:n], "pred": pred_spans_per_example[:n]}

    total_tp = total_fp = total_fn = 0
    per_pair = {}

    for a, b in permutations(["gold", "pred"], 2):
        tp = fp = fn = 0
        for i in range(n):
            tpi, fpi, fni = match_spans_iou(ann[a][i], ann[b][i], iou_threshold=iou_threshold)
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


# ----------------------------
# Weighted trainer
# ----------------------------
class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        inputs = dict(inputs)
        inputs.pop("word_starts", None)

        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        weight = self.class_weights
        if weight is not None and weight.device != logits.device:
            weight = weight.to(logits.device)

        loss_fct = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=-100)
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        return (loss, outputs) if return_outputs else loss


# ----------------------------
# Word-level prediction reconstruction
# ----------------------------
def word_level_pred_labels_from_word_starts(pred_ids_seq, word_starts_seq, id2label):
    L = min(len(pred_ids_seq), len(word_starts_seq))
    idxs = [i for i in range(L) if int(word_starts_seq[i]) == 1]
    return [id2label[int(pred_ids_seq[i])] for i in idxs]


# ----------------------------
# Train + eval helpers
# ----------------------------
def train_one_model(
    base_model: str,
    label_list: List[str],
    label2id: Dict[str, int],
    id2label: Dict[int, str],
    train_ds: Dataset,
    dev_ds: Dataset,
    tokenizer,
    defaults: Dict[str, Any],
    output_dir: Path,
    entity_weight: float,
):
    model = AutoModelForTokenClassification.from_pretrained(
        base_model,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
        trust_remote_code=True,
        ignore_mismatched_sizes=True,
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        eval_strategy=defaults.get("evaluation_strategy", "epoch"),
        save_strategy=defaults.get("save_strategy", "epoch"),
        logging_steps=int(defaults.get("logging_steps", 200)),
        learning_rate=float(defaults.get("learning_rate", 3e-5)),
        per_device_train_batch_size=int(defaults.get("per_device_train_batch_size", 16)),
        per_device_eval_batch_size=int(defaults.get("per_device_eval_batch_size", 16)),
        num_train_epochs=float(defaults.get("num_train_epochs", 3)),
        weight_decay=float(defaults.get("weight_decay", 0.01)),
        warmup_ratio=float(defaults.get("warmup_ratio", 0.1)),
        report_to="none",
        remove_unused_columns=False,
        load_best_model_at_end=False,
    )

    collator = DataCollatorForTokenClassificationWithWordStarts(tokenizer, padding=True)
    class_weights = torch.tensor([1.0, float(entity_weight)], dtype=torch.float32)

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics_fn_binary(label2id),
        class_weights=class_weights,
    )

    trainer.train()
    dev_metrics = trainer.evaluate(dev_ds)
    return trainer, dev_metrics


def evaluate_final(
    trainer,
    eval_ds: Dataset,
    eval_raw: Dataset,
    id2label: Dict[int, str],
    label2id: Dict[str, int],
    output_dir: Path,
    iou_threshold: float,
):
    pred = trainer.predict(eval_ds)
    logits = pred.predictions
    labels = pred.label_ids
    pred_ids = np.argmax(logits, axis=-1)

    gold_spans_list = []
    pred_spans_list = []
    predicted_spans_payload = []

    for i, (raw_ex, pred_seq, gold_seq) in enumerate(zip(eval_raw, pred_ids, labels)):
        tokens = raw_ex["tokens"]
        gold_word_labels = raw_ex["labels"]
        word_starts_seq = eval_ds[i]["word_starts"]

        pred_word_labels = word_level_pred_labels_from_word_starts(pred_seq, word_starts_seq, id2label)

        gold_spans = binary_spans_from_word_labels(tokens, gold_word_labels)
        pred_spans = binary_spans_from_word_labels(tokens, pred_word_labels)

        gold_spans_list.append([s["text"] for s in gold_spans])
        pred_spans_list.append([s["text"] for s in pred_spans])

        predicted_spans_payload.append({
            "filename": raw_ex.get("filename", ""),
            "text": raw_ex.get("text", " ".join(tokens)),
            "tokens": tokens,
            "gold_labels": gold_word_labels,
            "pred_labels": pred_word_labels,
            "gold_spans": gold_spans,
            "pred_spans": pred_spans,
        })

    pairwise = pairwise_span_f1(gold_spans_list, pred_spans_list, iou_threshold=iou_threshold)

    flat_p, flat_l = flatten_valid(pred_ids, labels)

    if flat_l.size:
        cm = confusion_matrix(flat_l, flat_p, labels=list(range(len(label2id))))
        cm_payload = {"labels": ["O", "ENTITY"], "matrix": cm.tolist()}
    else:
        cm_payload = {"labels": ["O", "ENTITY"], "matrix": []}

    cm_path = output_dir / "confusion_matrix.json"
    with open(cm_path, "w", encoding="utf-8") as f:
        json.dump(cm_payload, f, indent=2, ensure_ascii=False)

    pred_spans_path = output_dir / "eval_predicted_spans.json"
    with open(pred_spans_path, "w", encoding="utf-8") as f:
        json.dump(predicted_spans_payload, f, indent=2, ensure_ascii=False)

    entity_id = int(label2id["ENTITY"])
    overlap = token_overlap_f1_binary_from_ids(flat_p, flat_l, entity_id=entity_id)

    token_exact_match = float((flat_p == flat_l).mean()) if flat_l.size else 0.0
    seq_exact_match = exact_match_seq(pred_ids, labels)
    kappa = float(cohen_kappa_score(flat_l, flat_p, labels=list(range(len(label2id))))) if flat_l.size else 0.0

    return {
        "num_eval_sequences": int(labels.shape[0]),
        "num_eval_tokens_scored": int(flat_l.size),
        "token_overlap_f1_entity_vs_O": overlap,
        "exact_match": {
            "token_accuracy": float(token_exact_match),
            "sequence_exact_match": float(seq_exact_match),
        },
        "cohens_kappa_token_level": float(kappa),
        "pairwise_span_f1": pairwise,
        "confusion_matrix_path": str(cm_path),
        "eval_predicted_spans_path": str(pred_spans_path),
    }


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--category", required=True)
    ap.add_argument("--run_name", required=True)
    ap.add_argument("--entity_weight_grid", type=str, default="1.0,2.0,3.0,4.0,5.0")
    args = ap.parse_args()

    cfg = load_config(args.config)
    defaults = cfg.get("defaults", {})
    base_model = cfg.get("base_model")

    cat_cfg = next(
        c
        for c in cfg["categories"]
        if c["name"] == args.category and c.get("run_name") == args.run_name
    )

    train_path = cat_cfg["train_path"]
    eval_path = cat_cfg["eval_path"]
    dev_path = cat_cfg.get("dev_path") or cat_cfg.get("eval_path")

    output_root = Path(cfg.get("output_root", "./runs"))
    output_dir = output_root / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    train_raw = load_json_dataset(train_path)
    dev_raw = load_json_dataset(dev_path)
    eval_raw = load_json_dataset(eval_path)

    def fix_and_binary(ex):
        fixed = bio_fix_sequence(ex["labels"])
        return {**ex, "labels": to_binary_entity_labels(fixed)}

    train_raw = train_raw.map(fix_and_binary)
    dev_raw = dev_raw.map(fix_and_binary)
    eval_raw = eval_raw.map(fix_and_binary)

    label_list = ["O", "ENTITY"]
    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {i: l for l, i in label2id.items()}

    print("Base model:", base_model)
    print("Output dir:", str(output_dir))
    print("Labels:", label_list)

    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True, trust_remote_code=True)
    label_all_subtokens = True

    train_ds = train_raw.map(lambda ex: tokenize_and_align(ex, tokenizer, label2id, label_all_subtokens), batched=False)
    dev_ds = dev_raw.map(lambda ex: tokenize_and_align(ex, tokenizer, label2id, label_all_subtokens), batched=False)
    eval_ds = eval_raw.map(lambda ex: tokenize_and_align(ex, tokenizer, label2id, label_all_subtokens), batched=False)

    keep = {"input_ids", "attention_mask", "labels", "word_starts"}
    if "token_type_ids" in train_ds.column_names:
        keep.add("token_type_ids")

    def prune(ds: Dataset, name: str) -> Dataset:
        drop = [c for c in ds.column_names if c not in keep]
        if drop:
            print(f"[{name}] removing columns:", drop)
            ds = ds.remove_columns(drop)
        return ds

    train_ds = prune(train_ds, "train")
    dev_ds = prune(dev_ds, "dev")
    eval_ds = prune(eval_ds, "eval")

    entity_weights = [float(x.strip()) for x in args.entity_weight_grid.split(",")]
    print("Entity weight sweep:", entity_weights)

    best_entity_f1 = -1.0
    best_weight = None
    best_trainer = None
    sweep_results = []

    for entity_weight in entity_weights:
        run_dir = output_dir / f"_temp_entity_weight_{entity_weight}"
        run_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nTraining with ENTITY weight = {entity_weight}")
        trainer, dev_metrics = train_one_model(
            base_model=base_model,
            label_list=label_list,
            label2id=label2id,
            id2label=id2label,
            train_ds=train_ds,
            dev_ds=dev_ds,
            tokenizer=tokenizer,
            defaults=defaults,
            output_dir=run_dir,
            entity_weight=entity_weight,
        )

        dev_f1 = float(dev_metrics.get("eval_entity_f1", 0.0))
        sweep_results.append({
            "entity_weight": entity_weight,
            "dev_metrics": dev_metrics,
        })

        print(f"  dev ENTITY F1 = {dev_f1:.4f}")

        if dev_f1 > best_entity_f1:
            best_entity_f1 = dev_f1
            best_weight = entity_weight
            best_trainer = trainer

    print(f"\nBest ENTITY weight: {best_weight} (dev ENTITY F1 = {best_entity_f1:.4f})")

    iou_threshold = float(defaults.get("iou_threshold", 0.5))
    final_eval = evaluate_final(
        trainer=best_trainer,
        eval_ds=eval_ds,
        eval_raw=eval_raw,
        id2label=id2label,
        label2id=label2id,
        output_dir=output_dir,
        iou_threshold=iou_threshold,
    )

    best_model_dir = output_dir / "best_model"
    best_model_dir.mkdir(parents=True, exist_ok=True)
    best_trainer.save_model(str(best_model_dir))
    tokenizer.save_pretrained(str(best_model_dir))

    sweep_path = output_dir / "entity_weight_sweep_results.json"
    with open(sweep_path, "w", encoding="utf-8") as f:
        json.dump(sweep_results, f, indent=2, ensure_ascii=False)

    final_report = {
        "run_name": args.run_name,
        "category": args.category,
        "base_model": base_model,
        "best_entity_weight": best_weight,
        "best_metric": "dev_entity_f1",
        "best_metric_value": best_entity_f1,
        "entity_weight_sweep_results_path": str(sweep_path),
        "best_model_path": str(best_model_dir),
        **final_eval,
    }

    report_path = output_dir / "final_eval_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2, ensure_ascii=False)

    print("\n-----FINAL EVAL REPORT (EVAL SET)-----")
    print(json.dumps(final_report, indent=2))
    print("\nDone. Saved to:", str(output_dir))


if __name__ == "__main__":
    main()