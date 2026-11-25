import json
import argparse
import shutil
from pathlib import Path
from tqdm import tqdm 
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, cohen_kappa_score


# ----------------------------
# I/O (BIO JSON: [{"tokens":[...],"labels":[...]}])
# ----------------------------
def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_bio_json(path):
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
# BIO repair (kept for coverage stability)
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


# ----------------------------
# Label from BIO coverage
# ----------------------------
def coverage_label_from_bio(word_labels, threshold=0.5):
    if not word_labels:
        return 0
    n = len(word_labels)
    ent = sum(1 for l in word_labels if (l or "O").strip() != "O")
    cov = ent / n if n else 0.0
    return 1 if cov >= threshold else 0


# ----------------------------
# "Sentences together" as token windows over paragraph tokens
# ----------------------------
def build_token_windows(tokens, labels, window_tokens=160, stride_tokens=80):
    out = []
    n = len(tokens)
    if n == 0:
        return out

    w = int(window_tokens)
    s = int(stride_tokens)
    if w <= 0:
        w = n
    if s <= 0:
        s = w

    for start in range(0, n, s):
        end = min(n, start + w)
        win_toks = tokens[start:end]
        win_labs = labels[start:end]
        if not win_toks:
            continue
        out.append((start, end, win_toks, win_labs))
        if end == n:
            break

    return out


def make_classification_dataset(raw_ds, threshold=0.5, window_tokens=160, stride_tokens=80):
    rows = []

    for ex in raw_ds:
        toks = ex["tokens"]
        labs = bio_fix_sequence(ex["labels"])

        windows = build_token_windows(
            toks,
            labs,
            window_tokens=window_tokens,
            stride_tokens=stride_tokens,
        )

        for w_idx, (s, e, win_toks, win_labs) in enumerate(windows):
            y = coverage_label_from_bio(win_labs, threshold=threshold)

            row = {
                "text": " ".join(win_toks),
                "labels": int(y),
                "win_start_token": int(s),
                "win_end_token": int(e),
                "window_index": int(w_idx),
            }

            if "doc_index" in ex:
                row["doc_index"] = ex["doc_index"]
            if "sample_index" in ex:
                row["sample_index"] = ex["sample_index"]

            rows.append(row)

    return Dataset.from_list(rows)


# ----------------------------
# Tokenization (sequence classification)
# ----------------------------
def tokenize_cls(example, tokenizer):
    max_len = getattr(tokenizer, "model_max_length", 512) or 512
    max_len = min(max_len, 512)
    return tokenizer(example["text"], truncation=True, max_length=max_len)


# ----------------------------
# Metrics (binary)
# ----------------------------
def safe_cohen_kappa(labels, preds):
    labels = np.asarray(labels).astype(int)
    preds = np.asarray(preds).astype(int)

    if labels.size == 0:
        return 0.0
    if np.unique(labels).size < 2 or np.unique(preds).size < 2:
        return 0.0

    kappa = cohen_kappa_score(labels, preds, labels=[0, 1])
    if np.isnan(kappa):
        return 0.0
    return float(kappa)


def compute_binary_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1).astype(int)
    labels = labels.astype(int)

    cm = confusion_matrix(labels, preds, labels=[0, 1]).tolist()
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    acc = float((labels == preds).mean()) if labels.size else 0.0
    kappa = safe_cohen_kappa(labels, preds)

    return {
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "accuracy": float(acc),
        "cohen_kappa": float(kappa),
        "confusion_matrix_labels": [0, 1],
        "confusion_matrix": cm,
        "num_examples": int(labels.size),
    }


def drop_non_tensor_columns(ds):
    """
    IMPORTANT: remove any string/object columns (e.g. 'text') so the data collator
    doesn't try to convert them into tensors.
    """
    keep = {"input_ids", "attention_mask", "labels"}
    if "token_type_ids" in ds.column_names:
        keep.add("token_type_ids")

    cols_to_remove = [c for c in ds.column_names if c not in keep]
    if cols_to_remove:
        ds = ds.remove_columns(cols_to_remove)
    return ds


def _prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return {"precision": p, "recall": r, "f1": f1}


def pairwise_window_f1(y_gold, y_pred):
    """
    Directed pairwise micro-averaged F1 between gold and pred binary window labels.
    Treats each window as one unit of annotation.
    Computes gold→pred and pred→gold, then micro-averages across both directions.
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


def parse_float_grid(value, default):
    if value is None or value.strip() == "":
        return list(default)
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def parse_int_grid(value, default):
    if value is None or value.strip() == "":
        return list(default)
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def sanitize_float_for_name(x):
    return str(x).replace(".", "p")


def prepare_datasets(train_raw, dev_raw, eval_raw, threshold, window_tokens, stride_tokens):
    train_cls = make_classification_dataset(
        train_raw,
        threshold=threshold,
        window_tokens=window_tokens,
        stride_tokens=stride_tokens,
    )
    dev_cls = make_classification_dataset(
        dev_raw,
        threshold=threshold,
        window_tokens=window_tokens,
        stride_tokens=stride_tokens,
    )
    eval_cls = make_classification_dataset(
        eval_raw,
        threshold=threshold,
        window_tokens=window_tokens,
        stride_tokens=stride_tokens,
    )
    return train_cls, dev_cls, eval_cls


def tokenize_datasets(train_cls, dev_cls, eval_cls, tokenizer):
    train_tok = train_cls.map(lambda x: tokenize_cls(x, tokenizer), batched=False)
    dev_tok = dev_cls.map(lambda x: tokenize_cls(x, tokenizer), batched=False)
    eval_tok = eval_cls.map(lambda x: tokenize_cls(x, tokenizer), batched=False)
    return train_tok, dev_tok, eval_tok


def run_one_configuration(
    *,
    cfg,
    defaults,
    base_model,
    category,
    run_name,
    output_root,
    tokenizer,
    train_raw,
    dev_raw,
    eval_raw,
    coverage_threshold,
    window_tokens,
    stride_tokens,
):
    run_suffix = (
        f"cov_{sanitize_float_for_name(coverage_threshold)}"
        f"__win_{window_tokens}"
        f"__stride_{stride_tokens}"
    )
    run_output_dir = output_root / f"{run_name}__{run_suffix}"
    run_output_dir.mkdir(parents=True, exist_ok=True)

    train_cls, dev_cls, eval_cls = prepare_datasets(
        train_raw,
        dev_raw,
        eval_raw,
        threshold=coverage_threshold,
        window_tokens=window_tokens,
        stride_tokens=stride_tokens,
    )

    train_tok, dev_tok, eval_tok = tokenize_datasets(train_cls, dev_cls, eval_cls, tokenizer)
    eval_for_inspect = eval_tok

    train_ds = drop_non_tensor_columns(train_tok)
    dev_ds = drop_non_tensor_columns(dev_tok)
    eval_ds = drop_non_tensor_columns(eval_tok)

    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=2,
        id2label={0: "NOT_CATEGORY", 1: category.upper()},
        label2id={"NOT_CATEGORY": 0, category.upper(): 1},
        trust_remote_code=True,
    )

    training_args = TrainingArguments(
        output_dir=str(run_output_dir),
        eval_strategy=defaults.get("evaluation_strategy", "epoch"),
        save_strategy="no",
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
    data_collator = DataCollatorWithPadding(tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_binary_metrics,
    )

    print(
        f"{category} WINDOWED classification training"
        f" | coverage_threshold={coverage_threshold}"
        f" | window_tokens={window_tokens}"
        f" | stride_tokens={stride_tokens}"
    )
    trainer.train()

    dev_pred = trainer.predict(dev_ds)
    dev_logits = dev_pred.predictions
    dev_y_pred = np.argmax(dev_logits, axis=1).astype(int)
    dev_y_true = np.array(dev_ds["labels"], dtype=int)

    eval_pred = trainer.predict(eval_ds)
    eval_logits = eval_pred.predictions
    eval_y_pred = np.argmax(eval_logits, axis=1).astype(int)
    eval_y_true = np.array(eval_ds["labels"], dtype=int)

    inspect = []
    for ex, yp in zip(eval_for_inspect, eval_y_pred.tolist()):
        inspect.append(
            {
                "doc_index": ex.get("doc_index"),
                "sample_index": ex.get("sample_index"),
                "window_index": ex.get("window_index"),
                "win_start_token": ex.get("win_start_token"),
                "win_end_token": ex.get("win_end_token"),
                "text": ex.get("text"),
                "gold": int(ex["labels"]),
                "pred": int(yp),
            }
        )

    inspect_path = run_output_dir / "eval_window_predictions.json"
    with open(inspect_path, "w", encoding="utf-8") as f:
        json.dump(inspect, f, indent=2, ensure_ascii=False)

    dev_metrics = compute_binary_metrics((dev_logits, dev_y_true))
    eval_metrics = compute_binary_metrics((eval_logits, eval_y_true))
    eval_pairwise = pairwise_window_f1(eval_y_true, eval_y_pred)

    result = {
        "task": "windowed_sequence_classification",
        "category": category,
        "run_name": run_name,
        "base_model": base_model,
        "coverage_threshold": float(coverage_threshold),
        "window_tokens": int(window_tokens),
        "stride_tokens": int(stride_tokens),
        "train_num_windows": int(len(train_ds)),
        "dev_num_windows": int(len(dev_ds)),
        "eval_num_windows": int(len(eval_ds)),
        "dev_metrics": dev_metrics,
        "final_eval_metrics": eval_metrics,
        "pairwise_window_f1": eval_pairwise,
        "artifacts": {
            "run_output_dir": str(run_output_dir),
            "eval_window_predictions_path": str(inspect_path),
        },
    }

    with open(run_output_dir / "final_eval_metrics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    trainer.save_model(run_output_dir)
    tokenizer.save_pretrained(run_output_dir)

    return result, run_output_dir

import json
import argparse
import shutil
from pathlib import Path

import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, cohen_kappa_score


# ----------------------------
# I/O (BIO JSON: [{"tokens":[...],"labels":[...]}])
# ----------------------------
def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_bio_json(path):
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
# BIO repair (kept for coverage stability)
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


# ----------------------------
# Label from BIO coverage
# ----------------------------
def coverage_label_from_bio(word_labels, threshold=0.5):
    if not word_labels:
        return 0
    n = len(word_labels)
    ent = sum(1 for l in word_labels if (l or "O").strip() != "O")
    cov = ent / n if n else 0.0
    return 1 if cov >= threshold else 0


# ----------------------------
# "Sentences together" as token windows over paragraph tokens
# ----------------------------
def build_token_windows(tokens, labels, window_tokens=160, stride_tokens=80):
    out = []
    n = len(tokens)
    if n == 0:
        return out

    w = int(window_tokens)
    s = int(stride_tokens)
    if w <= 0:
        w = n
    if s <= 0:
        s = w

    for start in range(0, n, s):
        end = min(n, start + w)
        win_toks = tokens[start:end]
        win_labs = labels[start:end]
        if not win_toks:
            continue
        out.append((start, end, win_toks, win_labs))
        if end == n:
            break

    return out


def make_classification_dataset(raw_ds, threshold=0.5, window_tokens=160, stride_tokens=80):
    rows = []

    for ex in raw_ds:
        toks = ex["tokens"]
        labs = bio_fix_sequence(ex["labels"])

        windows = build_token_windows(
            toks,
            labs,
            window_tokens=window_tokens,
            stride_tokens=stride_tokens,
        )

        for w_idx, (s, e, win_toks, win_labs) in enumerate(windows):
            y = coverage_label_from_bio(win_labs, threshold=threshold)

            row = {
                "text": " ".join(win_toks),
                "labels": int(y),
                "win_start_token": int(s),
                "win_end_token": int(e),
                "window_index": int(w_idx),
            }

            if "doc_index" in ex:
                row["doc_index"] = ex["doc_index"]
            if "sample_index" in ex:
                row["sample_index"] = ex["sample_index"]

            rows.append(row)

    return Dataset.from_list(rows)


# ----------------------------
# Tokenization (sequence classification)
# ----------------------------
def tokenize_cls(example, tokenizer):
    max_len = getattr(tokenizer, "model_max_length", 512) or 512
    max_len = min(max_len, 512)
    return tokenizer(example["text"], truncation=True, max_length=max_len)


# ----------------------------
# Metrics (binary)
# ----------------------------
def safe_cohen_kappa(labels, preds):
    labels = np.asarray(labels).astype(int)
    preds = np.asarray(preds).astype(int)

    if labels.size == 0:
        return 0.0
    if np.unique(labels).size < 2 or np.unique(preds).size < 2:
        return 0.0

    kappa = cohen_kappa_score(labels, preds, labels=[0, 1])
    if np.isnan(kappa):
        return 0.0
    return float(kappa)


def compute_binary_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1).astype(int)
    labels = labels.astype(int)

    cm = confusion_matrix(labels, preds, labels=[0, 1]).tolist()
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    acc = float((labels == preds).mean()) if labels.size else 0.0
    kappa = safe_cohen_kappa(labels, preds)

    return {
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "accuracy": float(acc),
        "cohen_kappa": float(kappa),
        "confusion_matrix_labels": [0, 1],
        "confusion_matrix": cm,
        "num_examples": int(labels.size),
    }


def drop_non_tensor_columns(ds):
    keep = {"input_ids", "attention_mask", "labels"}
    if "token_type_ids" in ds.column_names:
        keep.add("token_type_ids")

    cols_to_remove = [c for c in ds.column_names if c not in keep]
    if cols_to_remove:
        ds = ds.remove_columns(cols_to_remove)
    return ds


def _prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return {"precision": p, "recall": r, "f1": f1}


def pairwise_window_f1(y_gold, y_pred):
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


def parse_float_grid(value, default):
    if value is None or value.strip() == "":
        return list(default)
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def parse_int_grid(value, default):
    if value is None or value.strip() == "":
        return list(default)
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def sanitize_float_for_name(x):
    return str(x).replace(".", "p")


def sanitize_name(text):
    return (
        str(text)
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace(":", "_")
    )


def prepare_datasets(train_raw, dev_raw, eval_raw, threshold, window_tokens, stride_tokens):
    train_cls = make_classification_dataset(
        train_raw,
        threshold=threshold,
        window_tokens=window_tokens,
        stride_tokens=stride_tokens,
    )
    dev_cls = make_classification_dataset(
        dev_raw,
        threshold=threshold,
        window_tokens=window_tokens,
        stride_tokens=stride_tokens,
    )
    eval_cls = make_classification_dataset(
        eval_raw,
        threshold=threshold,
        window_tokens=window_tokens,
        stride_tokens=stride_tokens,
    )
    return train_cls, dev_cls, eval_cls


def tokenize_datasets(train_cls, dev_cls, eval_cls, tokenizer):
    train_tok = train_cls.map(lambda x: tokenize_cls(x, tokenizer), batched=False)
    dev_tok = dev_cls.map(lambda x: tokenize_cls(x, tokenizer), batched=False)
    eval_tok = eval_cls.map(lambda x: tokenize_cls(x, tokenizer), batched=False)
    return train_tok, dev_tok, eval_tok


def run_one_configuration(
    *,
    defaults,
    base_model,
    category,
    run_name,
    category_root,
    tokenizer,
    train_raw,
    dev_raw,
    eval_raw,
    coverage_threshold,
    window_tokens,
    stride_tokens,
):
    run_suffix = (
        f"cov_{sanitize_float_for_name(coverage_threshold)}"
        f"__win_{window_tokens}"
        f"__stride_{stride_tokens}"
    )
    run_output_dir = category_root / run_suffix
    run_output_dir.mkdir(parents=True, exist_ok=True)

    train_cls, dev_cls, eval_cls = prepare_datasets(
        train_raw,
        dev_raw,
        eval_raw,
        threshold=coverage_threshold,
        window_tokens=window_tokens,
        stride_tokens=stride_tokens,
    )

    train_tok, dev_tok, eval_tok = tokenize_datasets(train_cls, dev_cls, eval_cls, tokenizer)
    eval_for_inspect = eval_tok

    train_ds = drop_non_tensor_columns(train_tok)
    dev_ds = drop_non_tensor_columns(dev_tok)
    eval_ds = drop_non_tensor_columns(eval_tok)

    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=2,
        id2label={0: "NOT_CATEGORY", 1: category.upper()},
        label2id={"NOT_CATEGORY": 0, category.upper(): 1},
        trust_remote_code=True,
    )

    training_args = TrainingArguments(
        output_dir=str(run_output_dir),
        eval_strategy=defaults.get("evaluation_strategy", "epoch"),
        save_strategy=defaults.get("save_strategy", "epoch"),
        logging_steps=int(defaults.get("logging_steps", 50)),
        learning_rate=float(defaults.get("learning_rate", 3e-5)),
        per_device_train_batch_size=int(defaults.get("per_device_train_batch_size", 16)),
        per_device_eval_batch_size=int(defaults.get("per_device_eval_batch_size", 16)),
        num_train_epochs=float(defaults.get("num_train_epochs", 3)),
        weight_decay=float(defaults.get("weight_decay", 0.01)),
        warmup_ratio=float(defaults.get("warmup_ratio", 0.1)),
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="none",
        remove_unused_columns=False,
        save_total_limit=1,
    )

    data_collator = DataCollatorWithPadding(tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_binary_metrics,
    )

    print(
        f"{category} WINDOWED classification training"
        f" | coverage_threshold={coverage_threshold}"
        f" | window_tokens={window_tokens}"
        f" | stride_tokens={stride_tokens}"
    )
    trainer.train()

    dev_pred = trainer.predict(dev_ds)
    dev_logits = dev_pred.predictions
    dev_y_pred = np.argmax(dev_logits, axis=1).astype(int)
    dev_y_true = np.array(dev_ds["labels"], dtype=int)

    eval_pred = trainer.predict(eval_ds)
    eval_logits = eval_pred.predictions
    eval_y_pred = np.argmax(eval_logits, axis=1).astype(int)
    eval_y_true = np.array(eval_ds["labels"], dtype=int)

    inspect = []
    for ex, yp in zip(eval_for_inspect, eval_y_pred.tolist()):
        inspect.append(
            {
                "doc_index": ex.get("doc_index"),
                "sample_index": ex.get("sample_index"),
                "window_index": ex.get("window_index"),
                "win_start_token": ex.get("win_start_token"),
                "win_end_token": ex.get("win_end_token"),
                "text": ex.get("text"),
                "gold": int(ex["labels"]),
                "pred": int(yp),
            }
        )

    inspect_path = run_output_dir / "eval_window_predictions.json"
    with open(inspect_path, "w", encoding="utf-8") as f:
        json.dump(inspect, f, indent=2, ensure_ascii=False)

    dev_metrics = compute_binary_metrics((dev_logits, dev_y_true))
    eval_metrics = compute_binary_metrics((eval_logits, eval_y_true))
    eval_pairwise = pairwise_window_f1(eval_y_true, eval_y_pred)

    result = {
        "task": "windowed_sequence_classification",
        "category": category,
        "run_name": run_name,
        "base_model": base_model,
        "coverage_threshold": float(coverage_threshold),
        "window_tokens": int(window_tokens),
        "stride_tokens": int(stride_tokens),
        "train_num_windows": int(len(train_ds)),
        "dev_num_windows": int(len(dev_ds)),
        "eval_num_windows": int(len(eval_ds)),
        "dev_metrics": dev_metrics,
        "final_eval_metrics": eval_metrics,
        "pairwise_window_f1": eval_pairwise,
        "artifacts": {
            "run_output_dir": str(run_output_dir),
            "eval_window_predictions_path": str(inspect_path),
        },
    }

    with open(run_output_dir / "final_eval_metrics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    trainer.save_model(run_output_dir)
    tokenizer.save_pretrained(run_output_dir)

    return result, run_output_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--category", required=True)
    parser.add_argument("--run_name", required=True)

    parser.add_argument("--coverage_threshold", type=float, default=0.5)
    parser.add_argument("--window_tokens", type=int, default=160)
    parser.add_argument("--stride_tokens", type=int, default=80)

    parser.add_argument("--coverage_threshold_grid", type=str, default="")
    parser.add_argument("--window_tokens_grid", type=str, default="")
    parser.add_argument("--stride_tokens_grid", type=str, default="")

    parser.add_argument("--select_by", type=str, default="dev_f1", choices=["dev_f1", "dev_kappa", "eval_f1", "eval_kappa"])

    args = parser.parse_args()

    cfg = load_config(args.config)
    defaults = cfg.get("defaults", {})
    base_model = cfg.get("base_model")

    cat_cfg = next(
        c for c in cfg["categories"]
        if c["name"] == args.category and c.get("run_name") == args.run_name
    )
    train_path = cat_cfg["train_path"]
    dev_path = cat_cfg.get("dev_path", cat_cfg.get("eval_path"))
    eval_path = cat_cfg.get("eval_path", cat_cfg.get("dev_path"))

    output_root = Path(cfg.get("output_root", "./runs"))
    output_root.mkdir(parents=True, exist_ok=True)

    model_name = sanitize_name(Path(base_model).name if "/" in str(base_model) else base_model)
    category_name = sanitize_name(args.category)

    category_root = output_root / category_name
    category_root.mkdir(parents=True, exist_ok=True)

    coverage_threshold_grid = [0.3, 0.4, 0.5, 0.6]
    window_tokens_grid = [128, 160, 192]
    stride_tokens_grid = [64, 80, 96]

    print("Base model:", base_model)
    print("Model name dir:", model_name)
    print("Category:", args.category)
    print("Category dir:", category_name)
    print("Run name:", args.run_name)
    print("coverage_threshold_grid:", coverage_threshold_grid)
    print("window_tokens_grid:", window_tokens_grid)
    print("stride_tokens_grid:", stride_tokens_grid)
    print("select_by:", args.select_by)

    train_raw = load_bio_json(train_path)
    dev_raw = load_bio_json(dev_path)
    eval_raw = load_bio_json(eval_path)

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    sweep_results = []
    best_result = None
    best_run_dir = None
    best_score = None

    total_runs = len(coverage_threshold_grid) * len(window_tokens_grid) * len(stride_tokens_grid)
    progress = tqdm(total=total_runs, desc="Sweep")

    for coverage_threshold in coverage_threshold_grid:
        for window_tokens in window_tokens_grid:
            for stride_tokens in stride_tokens_grid:
                result, run_dir = run_one_configuration(
                    defaults=defaults,
                    base_model=base_model,
                    category=args.category,
                    run_name=args.run_name,
                    category_root=category_root,
                    tokenizer=tokenizer,
                    train_raw=train_raw,
                    dev_raw=dev_raw,
                    eval_raw=eval_raw,
                    coverage_threshold=coverage_threshold,
                    window_tokens=window_tokens,
                    stride_tokens=stride_tokens,
                )

                if args.select_by == "dev_f1":
                    score = result["dev_metrics"]["f1"]
                    tie_break = result["dev_metrics"]["cohen_kappa"]
                elif args.select_by == "dev_kappa":
                    score = result["dev_metrics"]["cohen_kappa"]
                    tie_break = result["dev_metrics"]["f1"]
                elif args.select_by == "eval_f1":
                    score = result["final_eval_metrics"]["f1"]
                    tie_break = result["final_eval_metrics"]["cohen_kappa"]
                else:
                    score = result["final_eval_metrics"]["cohen_kappa"]
                    tie_break = result["final_eval_metrics"]["f1"]

                sweep_entry = dict(result)
                sweep_entry["selection_score"] = float(score)
                sweep_entry["selection_tie_break"] = float(tie_break)
                sweep_results.append(sweep_entry)

                if best_result is None:
                    best_result = result
                    best_run_dir = run_dir
                    best_score = (float(score), float(tie_break))
                else:
                    candidate = (float(score), float(tie_break))
                    if candidate > best_score:
                        # Delete previous best run dir
                        if best_run_dir is not None and best_run_dir.exists():
                            shutil.rmtree(best_run_dir)
                        best_result = result
                        best_run_dir = run_dir
                        best_score = candidate
                    else:
                        # Delete this run dir since it's not the best
                        if run_dir.exists():
                            shutil.rmtree(run_dir)

                progress.update(1)

    progress.close()

    best_model_dir = category_root / "best_model"
    if best_model_dir.exists():
        shutil.rmtree(best_model_dir)
    shutil.copytree(best_run_dir, best_model_dir)

    all_results = {
        "task": "windowed_sequence_classification_sweep",
        "category": args.category,
        "run_name": args.run_name,
        "base_model": base_model,
        "model_name_dir": model_name,
        "category_dir": category_name,
        "select_by": args.select_by,
        "grid": {
            "coverage_threshold": coverage_threshold_grid,
            "window_tokens": window_tokens_grid,
            "stride_tokens": stride_tokens_grid,
        },
        "num_runs": int(len(sweep_results)),
        "best_run": {
            "coverage_threshold": best_result["coverage_threshold"],
            "window_tokens": best_result["window_tokens"],
            "stride_tokens": best_result["stride_tokens"],
            "dev_metrics": best_result["dev_metrics"],
            "final_eval_metrics": best_result["final_eval_metrics"],
            "pairwise_window_f1": best_result["pairwise_window_f1"],
            "source_run_dir": str(best_run_dir),
            "copied_best_model_dir": str(best_model_dir),
        },
        "all_runs": sweep_results,
    }

    sweep_results_path = category_root / "sweep_results.json"
    with open(sweep_results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print("\n-----BEST SWEEP RESULT-----")
    print(json.dumps(all_results["best_run"], indent=2))
    print(f"\nSaved:\n  - {sweep_results_path}\n  - best model copied to {best_model_dir}")


if __name__ == "__main__":
    main()