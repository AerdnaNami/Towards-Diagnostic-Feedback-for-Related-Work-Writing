import json
import argparse
import shutil
from pathlib import Path
from itertools import product

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification,
)

from seqeval.metrics import f1_score
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, cohen_kappa_score
import re
import time
import random
from transformers import set_seed

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
set_seed(SEED)

# ----------------------------
# I/O (BIO JSON: [{"tokens":[...],"labels":[...]}])
# ----------------------------
def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_ner_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"{path} must be a JSON list")

    cleaned = []
    for i, ex in enumerate(data):
        if not isinstance(ex, dict):
            continue

        toks = ex.get("tokens")
        labs = ex.get("labels")
        if not isinstance(toks, list) or not isinstance(labs, list):
            raise ValueError(f"{path} example {i} missing tokens/labels lists")
        if len(toks) != len(labs):
            raise ValueError(f"{path} example {i}: len(tokens)={len(toks)} != len(labels)={len(labs)}")

        ex = dict(ex)
        ex["tokens"] = [str(t) for t in toks]
        ex["labels"] = [str(l) for l in labs]
        cleaned.append(ex)

    return Dataset.from_list(cleaned)


# ----------------------------
# BIO repair (force span starts to be B-*)
# ----------------------------
def _ent_type(lbl):
    lbl = (lbl or "O").strip()
    if "-" in lbl:
        return lbl.split("-", 1)[1].strip()
    return ""


def bio_fix_sequence(labels):
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


def apply_bio_fix_to_aligned_pred_ids(pred_ids, aligned_gold, id2label, label2id):
    fixed = pred_ids.copy()

    for i in range(pred_ids.shape[0]):
        idxs = np.where(aligned_gold[i] != -100)[0]
        if idxs.size == 0:
            continue

        word_pred_labels = [id2label[int(pred_ids[i, j])] for j in idxs]
        word_pred_labels = bio_fix_sequence(word_pred_labels)

        for j, lbl in zip(idxs, word_pred_labels):
            fixed[i, j] = int(label2id[lbl])

    return fixed


# ----------------------------
# Tokenization + alignment
# ----------------------------
def _continuation_label(word_label):
    word_label = (word_label or "O").strip()
    if word_label.startswith("B-"):
        return "I-" + word_label[2:]
    if word_label.startswith("I-"):
        return word_label
    return "O"


def tokenize_and_align_labels(example, tokenizer, label2id, label_all_subtokens=True):
    max_len = getattr(tokenizer, "model_max_length", 512) or 512
    max_len = min(max_len, 512)

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
    prev_word_idx = None

    for word_idx in word_ids:
        if word_idx is None:
            aligned.append(-100)
            word_starts.append(0)
            continue

        word_label = example["labels"][word_idx]

        if word_idx != prev_word_idx:
            aligned.append(label2id[word_label])
            word_starts.append(1)
        else:
            word_starts.append(0)
            if label_all_subtokens:
                cont = _continuation_label(word_label)
                aligned.append(label2id[cont])
            else:
                aligned.append(-100)

        prev_word_idx = word_idx

    tokenized["labels"] = aligned
    tokenized["word_starts"] = word_starts
    return tokenized


def flatten_valid_tokens(predictions, labels):
    flat_preds, flat_labels = [], []
    for pred_seq, label_seq in zip(predictions, labels):
        for p, l in zip(pred_seq, label_seq):
            if l != -100:
                flat_preds.append(int(p))
                flat_labels.append(int(l))
    return np.array(flat_preds), np.array(flat_labels)


def filter_out_gold_O(flat_preds, flat_labels, id2label):
    if flat_labels.size == 0:
        return flat_preds, flat_labels
    keep = np.array([id2label[int(l)] != "O" for l in flat_labels], dtype=bool)
    return flat_preds[keep], flat_labels[keep]


# ----------------------------
# Metrics
# ----------------------------
def compute_seqeval_f1(pred_ids, label_ids, id2label):
    true_preds, true_labels = [], []

    for p_seq, l_seq in zip(pred_ids, label_ids):
        preds, labs = [], []
        for p, l in zip(p_seq, l_seq):
            if l != -100:
                preds.append(id2label[int(p)])
                labs.append(id2label[int(l)])
        true_preds.append(preds)
        true_labels.append(labs)

    return float(f1_score(true_labels, true_preds))


def token_overlap_f1(pred_ids, label_ids, id2label):
    tp = fp = fn = 0

    for p_seq, l_seq in zip(pred_ids, label_ids):
        for p, l in zip(p_seq, l_seq):
            if l == -100:
                continue

            gold_ent = id2label[int(l)] != "O"
            pred_ent = id2label[int(p)] != "O"

            if pred_ent and gold_ent:
                tp += 1
            elif pred_ent and not gold_ent:
                fp += 1
            elif (not pred_ent) and gold_ent:
                fn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
    }


def krippendorff_alpha_nominal(rater1, rater2, num_labels):
    if rater1.shape[0] == 0:
        return 0.0

    o = np.zeros((num_labels, num_labels), dtype=np.float64)
    for a, b in zip(rater1, rater2):
        if a == b:
            o[a, a] += 2.0
        else:
            o[a, b] += 1.0
            o[b, a] += 1.0

    N = o.sum()
    if N <= 1:
        return 0.0

    Do = (o.sum() - np.trace(o)) / N
    n = o.sum(axis=0)
    De = (N * N - np.sum(n * n)) / (N * (N - 1))
    if De <= 0:
        return 1.0 if Do == 0 else 0.0

    return float(1.0 - (Do / De))


def sentence_label_from_bio_coverage(word_labels, threshold=0.5):
    if not word_labels:
        return 0
    n = len(word_labels)
    ent = sum(1 for l in word_labels if (l or "O").strip() != "O")
    coverage = ent / n if n else 0.0
    return 1 if coverage >= threshold else 0


def word_level_pred_ids_from_aligned(pred_ids_seq, aligned_label_ids_seq):
    return [int(p) for p, l in zip(pred_ids_seq.tolist(), aligned_label_ids_seq.tolist()) if l != -100]


def bio_spans_from_word_labels(tokens, word_labels):
    spans = []
    i = 0
    n = min(len(tokens), len(word_labels))

    while i < n:
        lab = (word_labels[i] or "O").strip()
        if lab == "O":
            i += 1
            continue

        if lab.startswith("B-"):
            ent = lab[2:]
            start = i
            i += 1
            while i < n and (word_labels[i] or "O").strip() == f"I-{ent}":
                i += 1
            end = i
            spans.append({"label": ent, "start_token": start, "end_token": end, "text": " ".join(tokens[start:end])})
            continue

        i += 1

    return spans


def remove_short_predicted_spans_word_labels(word_labels, min_span_len=4):
    out = [(l or "O").strip() for l in word_labels]
    n = len(out)
    i = 0

    while i < n:
        lab = out[i]
        if lab == "O":
            i += 1
            continue

        if lab.startswith("B-"):
            ent = lab[2:]
            start = i
            i += 1
            while i < n and out[i] == f"I-{ent}":
                i += 1
            end = i
            if (end - start) < int(min_span_len):
                for k in range(start, end):
                    out[k] = "O"
            continue

        i += 1

    return out


def apply_remove_short_spans_to_aligned_pred_ids(pred_ids, aligned_gold, id2label, label2id, min_span_len=4):
    fixed = pred_ids.copy()

    for i in range(pred_ids.shape[0]):
        idxs = np.where(aligned_gold[i] != -100)[0]
        if idxs.size == 0:
            continue

        word_pred_labels = [id2label[int(pred_ids[i, j])] for j in idxs]
        word_pred_labels = bio_fix_sequence(word_pred_labels)
        word_pred_labels = remove_short_predicted_spans_word_labels(word_pred_labels, min_span_len=min_span_len)

        for j, lbl in zip(idxs, word_pred_labels):
            fixed[i, j] = int(label2id[lbl])

    return fixed


def build_label_weights(label_list, o_weight=1.0, b_weight=1.5, i_weight=4.0):
    weights = []
    for lab in label_list:
        if lab == "O":
            weights.append(float(o_weight))
        elif lab.startswith("B-"):
            weights.append(float(b_weight))
        elif lab.startswith("I-"):
            weights.append(float(i_weight))
        else:
            weights.append(1.0)
    return weights


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


class WeightedTokenTrainer(Trainer):
    def __init__(self, *args, label_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._label_weights = label_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        inputs = dict(inputs)
        inputs.pop("word_starts", None)

        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        weight = self._label_weights
        if weight is not None and weight.device != logits.device:
            weight = weight.to(logits.device)

        loss_fct = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=-100)
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = dict(inputs)
        inputs.pop("word_starts", None)
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)


def word_level_pred_labels_from_word_starts(pred_ids_seq, word_starts_seq, id2label):
    L = min(len(pred_ids_seq), len(word_starts_seq))
    idxs = [i for i in range(L) if int(word_starts_seq[i]) == 1]
    return [id2label[int(pred_ids_seq[i])] for i in idxs]


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


def pairwise_span_f1(gold_spans_per_example, pred_spans_per_example, iou_threshold=0.5):
    from itertools import permutations

    annotators = {"gold": gold_spans_per_example, "pred": pred_spans_per_example}
    n = min(len(gold_spans_per_example), len(pred_spans_per_example))

    total_tp = total_fp = total_fn = 0
    per_pair = {}

    for a, b in permutations(["gold", "pred"], 2):
        tp = fp = fn = 0
        for i in range(n):
            tpi, fpi, fni = match_spans_iou(
                annotators[a][i], annotators[b][i], iou_threshold=iou_threshold,
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


# ----------------------------
# Evaluate at a given coverage threshold
# ----------------------------
def evaluate_at_threshold(
    thr, eval_raw, pred_ids, aligned_gold,
    id2label, label2id, label_list, eval_ds,
    min_pred_span_len, iou_threshold=0.5
):
    gold_sent_y, pred_sent_y = [], []

    for raw_ex, pred_seq, gold_seq in zip(eval_raw, pred_ids, aligned_gold):
        gold_word_labels = raw_ex["labels"]
        pred_word_ids = word_level_pred_ids_from_aligned(pred_seq, gold_seq)
        pred_word_labels = [id2label[i] for i in pred_word_ids]
        pred_word_labels = bio_fix_sequence(pred_word_labels)
        pred_word_labels = remove_short_predicted_spans_word_labels(
            pred_word_labels, min_span_len=int(min_pred_span_len)
        )
        gold_sent_y.append(sentence_label_from_bio_coverage(gold_word_labels, threshold=thr))
        pred_sent_y.append(sentence_label_from_bio_coverage(pred_word_labels, threshold=thr))

    gold_sent_y = np.array(gold_sent_y, dtype=int)
    pred_sent_y = np.array(pred_sent_y, dtype=int)

    sent_cm = confusion_matrix(gold_sent_y, pred_sent_y, labels=[0, 1]).tolist()
    sent_p, sent_r, sent_f1, _ = precision_recall_fscore_support(
        gold_sent_y, pred_sent_y, average="binary", zero_division=0
    )
    sent_acc = float((gold_sent_y == pred_sent_y).mean()) if gold_sent_y.size else 0.0

    flat_preds, flat_labels = flatten_valid_tokens(pred_ids, aligned_gold)

    cm_all = confusion_matrix(flat_labels, flat_preds, labels=list(range(len(label_list))))
    cm_all_dict = {"labels": label_list, "matrix": cm_all.tolist()}
    alpha_all = krippendorff_alpha_nominal(flat_labels, flat_preds, num_labels=len(label_list))
    kappa_all = (
        float(cohen_kappa_score(flat_labels, flat_preds, labels=list(range(len(label_list)))))
        if flat_labels.size else 0.0
    )
    p_micro_all, r_micro_all, f1_micro_all, _ = precision_recall_fscore_support(
        flat_labels, flat_preds, average="micro", zero_division=0
    )
    p_macro_all, r_macro_all, f1_macro_all, _ = precision_recall_fscore_support(
        flat_labels, flat_preds, average="macro", zero_division=0
    )

    flat_preds_ent, flat_labels_ent = filter_out_gold_O(flat_preds, flat_labels, id2label)
    cm_ent = confusion_matrix(flat_labels_ent, flat_preds_ent, labels=list(range(len(label_list))))
    cm_ent_dict = {"labels": label_list, "matrix": cm_ent.tolist()}
    alpha_ent = krippendorff_alpha_nominal(flat_labels_ent, flat_preds_ent, num_labels=len(label_list))
    kappa_ent = (
        float(cohen_kappa_score(flat_labels_ent, flat_preds_ent, labels=list(range(len(label_list)))))
        if flat_labels_ent.size else 0.0
    )
    p_micro_ent, r_micro_ent, f1_micro_ent, _ = precision_recall_fscore_support(
        flat_labels_ent, flat_preds_ent, average="micro", zero_division=0
    )
    p_macro_ent, r_macro_ent, f1_macro_ent, _ = precision_recall_fscore_support(
        flat_labels_ent, flat_preds_ent, average="macro", zero_division=0
    )

    pred_spans_by_example = []
    for i, (raw_ex, pred_seq, gold_seq) in enumerate(zip(eval_raw, pred_ids, aligned_gold)):
        tokens_words = raw_ex["tokens"]
        word_starts_seq = eval_ds[i]["word_starts"]
        pred_word_labels = word_level_pred_labels_from_word_starts(pred_seq, word_starts_seq, id2label)
        pred_word_labels = bio_fix_sequence(pred_word_labels)
        pred_word_labels = remove_short_predicted_spans_word_labels(
            pred_word_labels, min_span_len=int(min_pred_span_len)
        )
        pred_spans = bio_spans_from_word_labels(tokens_words, pred_word_labels)
        gold_spans = bio_spans_from_word_labels(tokens_words, raw_ex["labels"])
        pred_spans_by_example.append({
            "doc_index": raw_ex.get("doc_index"),
            "sample_index": raw_ex.get("sample_index"),
            "text": raw_ex.get("text", " ".join(tokens_words)),
            "gold_spans": gold_spans,
            "pred_spans": pred_spans,
        })

    gold_spans_list = [[s["text"] for s in ex["gold_spans"]] for ex in pred_spans_by_example]
    pred_spans_list = [[s["text"] for s in ex["pred_spans"]] for ex in pred_spans_by_example]

    seqeval_span_f1 = compute_seqeval_f1(pred_ids, aligned_gold, id2label)
    overlap = token_overlap_f1(pred_ids, aligned_gold, id2label)
    pairwise = pairwise_span_f1(gold_spans_list, pred_spans_list, iou_threshold=iou_threshold)

    return {
        "coverage_threshold": thr,
        "seqeval_span_f1": float(seqeval_span_f1),
        "token_overlap_f1_entity_vs_O": overlap,
        "pairwise_span_f1": pairwise,
        "sentence_label_from_bio_coverage": {
            "confusion_matrix_labels": [0, 1],
            "confusion_matrix": sent_cm,
            "precision": float(sent_p),
            "recall": float(sent_r),
            "f1": float(sent_f1),
            "accuracy": float(sent_acc),
            "num_sentences": int(gold_sent_y.size),
        },
        "token_metrics_all_tokens": {
            "krippendorff_alpha_nominal": float(alpha_all),
            "cohen_kappa_token_level": float(kappa_all),
            "token_prf": {
                "micro": {"precision": float(p_micro_all), "recall": float(r_micro_all), "f1": float(f1_micro_all)},
                "macro": {"precision": float(p_macro_all), "recall": float(r_macro_all), "f1": float(f1_macro_all)},
            },
            "confusion_matrix": cm_all_dict,
            "num_tokens_scored": int(flat_labels.size),
        },
        "token_metrics_entity_only": {
            "krippendorff_alpha_nominal": float(alpha_ent),
            "cohen_kappa_token_level": float(kappa_ent),
            "token_prf": {
                "micro": {"precision": float(p_micro_ent), "recall": float(r_micro_ent), "f1": float(f1_micro_ent)},
                "macro": {"precision": float(p_macro_ent), "recall": float(r_macro_ent), "f1": float(f1_macro_ent)},
            },
            "confusion_matrix": cm_ent_dict,
            "num_tokens_scored": int(flat_labels_ent.size),
        },
        "pred_spans_by_example": pred_spans_by_example,
    }


# ----------------------------
# Train one model, return trainer + predictions
# ----------------------------
def train_model(
    o_weight, b_weight, i_weight,
    base_model, label_list, label2id, id2label,
    train_ds, dev_ds, eval_ds,
    tokenizer, defaults, temp_dir, min_pred_span_len
):
    model = AutoModelForTokenClassification.from_pretrained(
        base_model,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
        trust_remote_code=True,
    )

    training_args = TrainingArguments(
        output_dir=str(temp_dir),
        eval_strategy=defaults.get("evaluation_strategy", "epoch"),
        save_strategy=defaults.get("save_strategy", "epoch"),
        logging_steps=int(defaults.get("logging_steps", 50)),
        learning_rate=float(defaults.get("learning_rate", 3e-5)),
        per_device_train_batch_size=int(defaults.get("per_device_train_batch_size", 16)),
        per_device_eval_batch_size=int(defaults.get("per_device_eval_batch_size", 16)),
        num_train_epochs=float(defaults.get("num_train_epochs", 3)),
        weight_decay=float(defaults.get("weight_decay", 0.01)),
        warmup_ratio=float(defaults.get("warmup_ratio", 0.1)),
        load_best_model_at_end=False,
        report_to="none",
        remove_unused_columns=False,
    )

    data_collator = DataCollatorForTokenClassificationWithWordStarts(tokenizer, padding=True)
    weights = build_label_weights(label_list, o_weight=o_weight, b_weight=b_weight, i_weight=i_weight)
    label_weights = torch.tensor(weights, dtype=torch.float32)

    trainer = WeightedTokenTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        label_weights=label_weights,
    )

    trainer.train()

    logits, aligned_gold, _ = trainer.predict(eval_ds)
    pred_ids = np.argmax(logits, axis=2)

    pred_ids = apply_bio_fix_to_aligned_pred_ids(pred_ids, aligned_gold, id2label, label2id)

    if int(min_pred_span_len) > 1:
        pred_ids = apply_remove_short_spans_to_aligned_pred_ids(
            pred_ids, aligned_gold, id2label, label2id,
            min_span_len=int(min_pred_span_len),
        )

    return trainer, pred_ids, aligned_gold

def safe_rmtree(path, retries=3, delay=1.0):
    for attempt in range(retries):
        try:
            if path.exists():
                shutil.rmtree(path)
            return
        except OSError as e:
            if attempt == retries - 1:
                print(f"Warning: failed to remove {path}: {e}")
                return
            time.sleep(delay)

# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--category", required=True)
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--coverage_threshold_grid", type=str, default="0.3,0.4,0.5")
    parser.add_argument("--iou_threshold", type=float, default=0.5)

    parser.add_argument("--o_weight_grid", type=str, default="0.5,0.75,1.0")
    parser.add_argument("--b_weight_grid", type=str, default="2.0,3.0,4.0")
    parser.add_argument("--i_weight_grid", type=str, default="5.0,6.0,7.0,8.0")

    parser.add_argument("--min_pred_span_len", type=int, default=1)

    args = parser.parse_args()

    label_all_subtokens = True

    cfg = load_config(args.config)
    defaults = cfg.get("defaults", {})
    base_model = cfg.get("base_model")

    cat_cfg = next(
        c for c in cfg["categories"]
        if c["name"] == args.category and c.get("run_name") == args.run_name
    )

    category_name = cat_cfg["name"]
    train_path = cat_cfg["train_path"]
    dev_path = cat_cfg.get("dev_path", cat_cfg.get("eval_path"))
    eval_path = cat_cfg.get("eval_path", cat_cfg.get("dev_path"))

    output_root = Path(cfg.get("output_root", "./runs"))
    run_output_dir = output_root / category_name / args.run_name
    print(f"CATEGORY: {category_name}")
    print(f"RUN NAME: {args.run_name}")
    print(f"OUTPUT DIRECTORY: {run_output_dir}")
    run_output_dir.mkdir(parents=True, exist_ok=True)

    train_raw = load_ner_json(train_path)
    dev_raw = load_ner_json(dev_path)
    eval_raw = load_ner_json(eval_path)

    print("Applying BIO-fix to GOLD labels (train/dev/eval).")
    train_raw = train_raw.map(lambda ex: {**ex, "labels": bio_fix_sequence(ex["labels"])}, batched=False)
    dev_raw = dev_raw.map(lambda ex: {**ex, "labels": bio_fix_sequence(ex["labels"])}, batched=False)
    eval_raw = eval_raw.map(lambda ex: {**ex, "labels": bio_fix_sequence(ex["labels"])}, batched=False)

    label_set = {lab for ex in train_raw for lab in ex["labels"]}
    label_set.add("O")
    label_list = sorted(label_set)

    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {i: l for l, i in label2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    train_ds = train_raw.map(
        lambda x: tokenize_and_align_labels(x, tokenizer, label2id, label_all_subtokens), batched=False
    )
    dev_ds = dev_raw.map(
        lambda x: tokenize_and_align_labels(x, tokenizer, label2id, label_all_subtokens), batched=False
    )
    eval_ds = eval_raw.map(
        lambda x: tokenize_and_align_labels(x, tokenizer, label2id, label_all_subtokens), batched=False
    )

    def prune(ds, name):
        keep = {"input_ids", "attention_mask", "labels", "word_starts"}
        if "token_type_ids" in ds.column_names:
            keep.add("token_type_ids")
        cols_to_remove = [c for c in ds.column_names if c not in keep]
        if cols_to_remove:
            print(f"[{name}] removing columns: {cols_to_remove}")
            ds = ds.remove_columns(cols_to_remove)
        return ds

    train_ds = prune(train_ds, "train")
    dev_ds = prune(dev_ds, "dev")
    eval_ds = prune(eval_ds, "eval")

    # ----------------------------
    # Build sweep grids
    # ----------------------------
    o_weights = [float(x.strip()) for x in args.o_weight_grid.split(",")]
    b_weights = [float(x.strip()) for x in args.b_weight_grid.split(",")]
    i_weights = [float(x.strip()) for x in args.i_weight_grid.split(",")]
    coverage_thresholds = [float(t.strip()) for t in args.coverage_threshold_grid.split(",")]
    weight_combos = list(product(o_weights, b_weights, i_weights))

    total_runs = len(weight_combos) * len(coverage_thresholds)
    print(f"\nLabel list: {label_list}")
    print(f"Entity weight combos to sweep: {weight_combos}")
    print(f"Coverage thresholds to sweep: {coverage_thresholds}")
    print(f"Total configurations: {total_runs} "
          f"({len(weight_combos)} weight combos x {len(coverage_thresholds)} thresholds)")

    # ----------------------------
    # Outer loop: entity weights (each requires a full retrain)
    # Inner loop: coverage threshold (post-processing only, no retraining)
    # ----------------------------
    all_sweep_results = []
    best_pairwise_f1 = -1.0
    best_thr = None
    best_weights = None
    best_results = None
    best_trainer = None

    for o_w, b_w, i_w in weight_combos:
        print(f"\n{'='*60}")
        print(f"Training with o_weight={o_w}, b_weight={b_w}, i_weight={i_w}")
        print(f"{'='*60}")

        temp_dir = run_output_dir / f"_temp_o{o_w}_b{b_w}_i{i_w}"
        temp_dir.mkdir(parents=True, exist_ok=True)

        trainer, pred_ids, aligned_gold = train_model(
            o_weight=o_w,
            b_weight=b_w,
            i_weight=i_w,
            base_model=base_model,
            label_list=label_list,
            label2id=label2id,
            id2label=id2label,
            train_ds=train_ds,
            dev_ds=dev_ds,
            eval_ds=eval_ds,
            tokenizer=tokenizer,
            defaults=defaults,
            temp_dir=temp_dir,
            min_pred_span_len=args.min_pred_span_len,
        )

        for thr in coverage_thresholds:
            print(f"  Evaluating coverage_threshold={thr} ...")
            result = evaluate_at_threshold(
                thr=thr,
                eval_raw=eval_raw,
                pred_ids=pred_ids,
                aligned_gold=aligned_gold,
                id2label=id2label,
                label2id=label2id,
                label_list=label_list,
                eval_ds=eval_ds,
                min_pred_span_len=args.min_pred_span_len,
                iou_threshold=args.iou_threshold,
            )
            pairwise_f1 = result["pairwise_span_f1"]["pairwise_micro"]["f1"]
            print(f"    pairwise_span_f1={pairwise_f1:.4f}")

            all_sweep_results.append({
                "o_weight": o_w,
                "b_weight": b_w,
                "i_weight": i_w,
                "coverage_threshold": thr,
                "pairwise_span_f1": pairwise_f1,
            })

            if pairwise_f1 > best_pairwise_f1:
                best_pairwise_f1 = pairwise_f1
                best_thr = thr
                best_weights = {"o_weight": o_w, "b_weight": b_w, "i_weight": i_w}
                best_results = result
                best_trainer = trainer

        safe_rmtree(temp_dir)

    print(f"\nBest configuration: weights={best_weights}, coverage_threshold={best_thr} "
        f"(pairwise_span_f1={best_pairwise_f1:.4f})")

    # Save best model only
    best_model_dir = run_output_dir / "best_model"
    safe_rmtree(best_model_dir)
    best_model_dir.mkdir(parents=True, exist_ok=True)

    best_trainer.save_model(str(best_model_dir))
    tokenizer.save_pretrained(str(best_model_dir))
    print(f"Best model saved to: {best_model_dir}")

    # ----------------------------
    # Save predicted spans for best configuration
    # ----------------------------
    pred_spans_path = run_output_dir / "eval_predicted_spans.json"
    with open(pred_spans_path, "w", encoding="utf-8") as f:
        json.dump(best_results["pred_spans_by_example"], f, indent=2, ensure_ascii=False)

    # ----------------------------
    # Compile and save final results
    # ----------------------------
    final_results = {
        "category": args.category,
        "run_name": args.run_name,
        "base_model": base_model,
        "best_coverage_threshold": best_thr,
        "best_loss_weights": best_weights,
        "best_metric": "pairwise_f1",
        "best_metric_value": best_pairwise_f1,
        "full_sweep_results": all_sweep_results,
        "min_pred_span_len": int(args.min_pred_span_len),
        "iou_threshold": args.iou_threshold,
        "label_all_subtokens": bool(label_all_subtokens),
        "seqeval_span_f1": best_results["seqeval_span_f1"],
        "token_overlap_f1_entity_vs_O": best_results["token_overlap_f1_entity_vs_O"],
        "pairwise_span_f1": best_results["pairwise_span_f1"],
        "sentence_label_from_bio_coverage": best_results["sentence_label_from_bio_coverage"],
        "token_metrics_all_tokens": best_results["token_metrics_all_tokens"],
        "token_metrics_entity_only": best_results["token_metrics_entity_only"],
        "artifacts": {
            "eval_predicted_spans_path": str(pred_spans_path),
            "best_model_path": str(best_model_dir),
        },
    }

    print("\n-----FINAL EVAL RESULTS (best configuration)-----")
    print(json.dumps({k: v for k, v in final_results.items() if k != "full_sweep_results"}, indent=2))

    with open(run_output_dir / "final_eval_metrics.json", "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
