import json
import os
import random
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# Keep transformers on the PyTorch path only for this evaluation module.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, cohen_kappa_score
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    DataCollatorWithPadding,
)
from transformers.modeling_outputs import TokenClassifierOutput


def _log_stage(message: str) -> None:
    print(f"[eval] {message}", flush=True)


def _get_trainer_components():
    from transformers import Trainer, TrainingArguments

    return Trainer, TrainingArguments


DEFAULT_COMBINED_REAL_DATA = (
    Path(__file__).resolve().parent.parent
    / "synthetic_sampling"
    / "samples"
    / "combined_real_data.json"
)
DEFAULT_TRAINING_DATA_DIR = Path(__file__).resolve().parent / "data"

_TOKEN_RE = re.compile(r"\S+")
_SENT_BOUNDARY_RE = re.compile(r"(?<=[.!?])(?:[\"')\]]+)?\s+(?=[A-Z0-9\"(\[])")
_SPAN_TOKEN_RE = re.compile(r"[A-Za-z0-9]+", re.UNICODE)
_SHORT_CITATION_PATTERN = re.compile(
    r"""
    (
        \(
            [A-Z][A-Za-z\-]+
            (?:(?:\s+et\ al\.)|(?:\s+(?:&|and)\s+[A-Z][A-Za-z\-]+))?
            \,?\s+\d{4}[a-z]?
        \)
        |
        [A-Z][A-Za-z\-]+
        (?:(?:\s+et\ al\.)|(?:\s+(?:&|and)\s+[A-Z][A-Za-z\-]+))?
        \s+\(\d{4}[a-z]?\)
        |
        \[(\d+([,\-–]\s*\d+)*)\]
    )
    """,
    re.VERBOSE,
)


def _read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _clean_ner_examples(payload: Any, source_name: str) -> List[Dict[str, Any]]:
    if not isinstance(payload, list):
        raise ValueError(f"{source_name} must be a JSON list when used as a prepared NER dataset.")

    cleaned = []
    for i, ex in enumerate(payload):
        if not isinstance(ex, dict):
            continue
        toks = ex.get("tokens")
        labs = ex.get("labels")
        if not isinstance(toks, list) or not isinstance(labs, list):
            raise ValueError(f"{source_name} example {i} missing tokens/labels lists")
        if len(toks) != len(labs):
            raise ValueError(
                f"{source_name} example {i}: len(tokens)={len(toks)} != len(labels)={len(labs)}"
            )

        item = dict(ex)
        item["tokens"] = [str(t) for t in toks]
        item["labels"] = [str(l) for l in labs]
        if "text" not in item:
            item["text"] = " ".join(item["tokens"])
        cleaned.append(item)

    return cleaned


def _load_prepared_ner_dataset(data_path: str | Path) -> Optional[Dataset]:
    payload = _read_json(data_path)
    if not isinstance(payload, list):
        return None
    if not payload:
        return Dataset.from_list([])
    first = payload[0]
    if not isinstance(first, dict) or "tokens" not in first or "labels" not in first:
        return None
    return Dataset.from_list(_clean_ner_examples(payload, str(data_path)))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _canonical_category_key(category_name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(category_name).strip().lower()).strip("_")


def _display_category_name(category_name: str) -> str:
    key = _canonical_category_key(category_name)
    return {
        "unsupported_claim": "Unsupported claim",
        "lacks_synthesis": "Lacks synthesis",
        "coherence": "Coherence",
        "format": "Format",
    }.get(key, category_name)


def _dataset_to_jsonable_rows(ds: Dataset) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for ex in ds:
        row: Dict[str, Any] = {}
        for key, value in ex.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                row[key] = value
            elif isinstance(value, list):
                row[key] = list(value)
            else:
                row[key] = value
        rows.append(row)
    return rows


def _trim_fields(rows: Sequence[Dict[str, Any]], keep_fields: Sequence[str]) -> List[Dict[str, Any]]:
    keep = set(keep_fields)
    return [{k: row[k] for k in keep_fields if k in row} for row in rows if isinstance(row, dict)]


def _ent_type(lbl: str) -> str:
    lbl = (lbl or "O").strip()
    if "-" in lbl:
        return lbl.split("-", 1)[1].strip()
    return ""


def bio_fix_sequence(labels: Sequence[str]) -> List[str]:
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


def _continuation_label(word_label: str) -> str:
    word_label = (word_label or "O").strip()
    if word_label.startswith("B-"):
        return "I-" + word_label[2:]
    if word_label.startswith("I-"):
        return word_label
    return "O"


def _binary_entity_labels(word_labels: Sequence[str]) -> List[str]:
    return ["O" if (x or "O").strip() == "O" else "ENTITY" for x in word_labels]


def _norm_entity_name(category_name: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "_", category_name.strip().upper()).strip("_")


def _load_category_examples(data_path: str | Path, category_name: str) -> List[Dict[str, Any]]:
    payload = _read_json(data_path)
    if not isinstance(payload, dict):
        raise ValueError(f"{data_path} must be a JSON object keyed by category.")

    resolved_key = None
    if category_name in payload:
        resolved_key = category_name
    else:
        requested_key = _canonical_category_key(category_name)
        for raw_key in payload:
            if _canonical_category_key(raw_key) == requested_key:
                resolved_key = raw_key
                break

    if resolved_key is None:
        raise KeyError(f"Category {category_name!r} not found in {data_path}.")

    examples = payload[resolved_key]
    if not isinstance(examples, list):
        raise ValueError(f"Category {resolved_key!r} must map to a list.")
    return examples


def _tokenize_with_offsets(text: str) -> List[Tuple[str, int, int]]:
    return [(m.group(0), m.start(), m.end()) for m in _TOKEN_RE.finditer(text)]


def _split_sentences_with_offsets(text: str) -> List[Tuple[int, int, str]]:
    pieces: List[Tuple[int, int, str]] = []
    start = 0
    for m in _SENT_BOUNDARY_RE.finditer(text):
        end = m.start()
        segment = text[start:end].strip()
        if segment:
            seg_start = start + text[start:end].find(segment)
            seg_end = seg_start + len(segment)
            pieces.append((seg_start, seg_end, text[seg_start:seg_end]))
        start = m.end()
    tail = text[start:].strip()
    if tail:
        tail_start = start + text[start:].find(tail)
        tail_end = tail_start + len(tail)
        pieces.append((tail_start, tail_end, text[tail_start:tail_end]))
    if not pieces and text.strip():
        stripped = text.strip()
        seg_start = text.find(stripped)
        pieces.append((seg_start, seg_start + len(stripped), stripped))
    return pieces


def _span_overlaps(token_start: int, token_end: int, span_start: int, span_end: int) -> bool:
    return max(token_start, span_start) < min(token_end, span_end)


def _label_tokens_from_spans(
    token_offsets: Sequence[Tuple[str, int, int]],
    spans: Sequence[Tuple[int, int]],
    entity_name: str,
) -> List[str]:
    labels = ["O"] * len(token_offsets)
    if not token_offsets or not spans:
        return labels

    covered: List[List[int]] = []
    for span_start, span_end in spans:
        idxs = [
            i
            for i, (_, tok_start, tok_end) in enumerate(token_offsets)
            if _span_overlaps(tok_start, tok_end, span_start, span_end)
        ]
        if idxs:
            covered.append(idxs)

    for idxs in covered:
        first = True
        for idx in idxs:
            labels[idx] = f"B-{entity_name}" if first else f"I-{entity_name}"
            first = False

    return bio_fix_sequence(labels)


def _count_tokens_in_span(text: str) -> int:
    return len(_TOKEN_RE.findall(text or ""))


def _extract_char_spans(example: Dict[str, Any], *, min_words: Optional[int] = None, max_words: Optional[int] = None) -> List[Tuple[int, int]]:
    spans = []
    for span in example.get("spans", []):
        span_text = example.get("document", "")[int(span["start"]):int(span["end"])]
        n_words = _count_tokens_in_span(span_text)
        if min_words is not None and n_words < min_words:
            continue
        if max_words is not None and n_words > max_words:
            continue
        spans.append((int(span["start"]), int(span["end"])))
    return spans


def _build_document_level_bio_dataset(
    data_path: str | Path,
    category_name: str,
    *,
    min_words: Optional[int] = None,
    max_words: Optional[int] = None,
) -> Dataset:
    entity_name = _norm_entity_name(category_name)
    rows = []
    for doc_index, ex in enumerate(_load_category_examples(data_path, category_name)):
        text = str(ex.get("document", ""))
        token_offsets = _tokenize_with_offsets(text)
        spans = _extract_char_spans(ex, min_words=min_words, max_words=max_words)
        labels = _label_tokens_from_spans(token_offsets, spans, entity_name)
        rows.append(
            {
                "doc_index": doc_index,
                "text": text,
                "tokens": [tok for tok, _, _ in token_offsets],
                "labels": labels,
            }
        )
    return Dataset.from_list(rows)


def _build_sentence_level_bio_dataset(
    data_path: str | Path,
    category_name: str,
    *,
    min_span_words: Optional[int] = None,
    max_span_words: Optional[int] = None,
    require_positive: bool = False,
) -> Dataset:
    entity_name = _norm_entity_name(category_name)
    rows = []
    for doc_index, ex in enumerate(_load_category_examples(data_path, category_name)):
        text = str(ex.get("document", ""))
        kept_spans = _extract_char_spans(ex, min_words=min_span_words, max_words=max_span_words)
        for sent_index, (sent_start, sent_end, sent_text) in enumerate(_split_sentences_with_offsets(text)):
            sentence_spans = []
            for span_start, span_end in kept_spans:
                if max(sent_start, span_start) < min(sent_end, span_end):
                    sentence_spans.append((max(sent_start, span_start) - sent_start, min(sent_end, span_end) - sent_start))

            token_offsets = _tokenize_with_offsets(sent_text)
            labels = _label_tokens_from_spans(token_offsets, sentence_spans, entity_name)
            if require_positive and all(lbl == "O" for lbl in labels):
                continue
            if not token_offsets:
                continue
            rows.append(
                {
                    "doc_index": doc_index,
                    "sent_index": sent_index,
                    "text": sent_text,
                    "tokens": [tok for tok, _, _ in token_offsets],
                    "labels": labels,
                }
            )
    return Dataset.from_list(rows)


def _span_word_len(span_text: str) -> int:
    return len(_TOKEN_RE.findall(span_text or ""))


def _split_items_by_span_length(
    items: Sequence[Dict[str, Any]],
    *,
    min_words: int = 1,
    max_words: int = 5,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    short_items: List[Dict[str, Any]] = []
    other_items: List[Dict[str, Any]] = []

    for sample in items:
        doc_text = str(sample.get("document", ""))
        spans = sample.get("spans", [])

        short_spans = []
        other_spans = []
        for sp in spans:
            span_start = int(sp["start"])
            span_end = int(sp["end"])
            span_text = sp.get("span") or sp.get("text") or doc_text[span_start:span_end]
            wl = _span_word_len(str(span_text))
            if min_words <= wl <= max_words:
                short_spans.append(sp)
            else:
                other_spans.append(sp)

        if short_spans:
            short_items.append({"document": doc_text, "spans": short_spans})
        if other_spans:
            other_items.append({"document": doc_text, "spans": other_spans})

    return short_items, other_items


def _build_unsupported_claim_sentence_dataset(
    data_path: str | Path,
    *,
    min_span_words: int,
    max_span_words: Optional[int],
    neg_ratio: float,
    seed: int = 42,
) -> Dataset:
    category_name = "Unsupported claim"
    label_name = _norm_entity_name(category_name)
    items = _load_category_examples(data_path, category_name)

    if max_span_words is not None:
        selected_items, other_items = _split_items_by_span_length(
            items,
            min_words=min_span_words,
            max_words=max_span_words,
        )
        chosen_items = selected_items
    else:
        _, other_items = _split_items_by_span_length(
            items,
            min_words=1,
            max_words=max(min_span_words - 1, 0),
        )
        chosen_items = other_items

    rng = random.Random(seed)
    rows: List[Dict[str, Any]] = []
    for doc_index, sample in enumerate(chosen_items):
        doc_text = str(sample.get("document", ""))
        spans = sample.get("spans", [])
        sentence_rows = []
        for sent_index, (sent_start, sent_end, sent_text) in enumerate(_split_sentences_with_offsets(doc_text)):
            token_offsets = _tokenize_with_offsets(sent_text)
            sentence_rows.append(
                {
                    "doc_index": doc_index,
                    "sent_index": sent_index,
                    "text": sent_text,
                    "start": sent_start,
                    "end": sent_end,
                    "tokens": token_offsets,
                    "labels": ["O"] * len(token_offsets),
                    "has_span": False,
                }
            )

        for span in spans:
            span_start = int(span["start"])
            span_end = int(span["end"])
            for sent_row in sentence_rows:
                if span_end <= sent_row["start"] or span_start >= sent_row["end"]:
                    continue
                local_start = max(0, span_start - sent_row["start"])
                local_end = min(len(sent_row["text"]), span_end - sent_row["start"])
                new_labels = _label_tokens_from_spans(
                    sent_row["tokens"],
                    [(local_start, local_end)],
                    label_name,
                )
                for i, label in enumerate(new_labels):
                    if label != "O":
                        sent_row["labels"][i] = label
                sent_row["has_span"] = True

        positives = []
        negatives = []
        for sent_row in sentence_rows:
            ex = {
                "tokens": [tok for tok, _, _ in sent_row["tokens"]],
                "labels": sent_row["labels"],
            }
            if sent_row["has_span"]:
                positives.append(ex)
            else:
                negatives.append(ex)

        rows.extend(positives)
        target_neg = int(round(len(positives) * float(neg_ratio)))
        if target_neg > 0 and negatives:
            if target_neg >= len(negatives):
                rows.extend(negatives)
            else:
                rows.extend(rng.sample(negatives, target_neg))

    return Dataset.from_list(rows)


def _load_sentence_eval_dataset(data_path: str | Path, category_name: str) -> Dataset:
    prepared = _load_prepared_ner_dataset(data_path)
    if prepared is not None:
        return prepared.map(
            lambda ex: {**ex, "labels": bio_fix_sequence(ex["labels"])},
            batched=False,
        )
    return _build_sentence_level_bio_dataset(data_path, category_name)


def _load_paragraph_eval_dataset(data_path: str | Path, category_name: str) -> Dataset:
    prepared = _load_prepared_ner_dataset(data_path)
    if prepared is not None:
        return prepared.map(
            lambda ex: {**ex, "labels": bio_fix_sequence(ex["labels"])},
            batched=False,
        )
    return _build_document_level_bio_dataset(data_path, category_name)


def _load_short_eval_dataset(
    data_path: str | Path,
    category_name: str,
    *,
    max_span_words: int,
) -> Dataset:
    prepared = _load_prepared_ner_dataset(data_path)
    if prepared is not None:
        prepared = prepared.map(
            lambda ex: {**ex, "labels": _binary_entity_labels(bio_fix_sequence(ex["labels"]))},
            batched=False,
        )
        return prepared.filter(lambda ex: any(lbl != "O" for lbl in ex["labels"]))

    raw_ds = _build_sentence_level_bio_dataset(
        data_path,
        category_name,
        min_span_words=1,
        max_span_words=max_span_words,
        require_positive=True,
    )
    return raw_ds.map(
        lambda ex: {**ex, "labels": _binary_entity_labels(bio_fix_sequence(ex["labels"]))},
        batched=False,
    )


def export_preprocessed_real_eval_dataset(
    category_name: str,
    task_name: str,
    *,
    source_path: str | Path = DEFAULT_COMBINED_REAL_DATA,
    output_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    display_name = _display_category_name(category_name)
    category_key = _canonical_category_key(category_name)
    task_key = str(task_name).strip().lower()

    if output_path is None:
        default_name_map = {
            ("coherence", "sentence"): DEFAULT_TRAINING_DATA_DIR / "coherence" / "coherence_ner_eval.json",
            ("lacks_synthesis", "sentence"): DEFAULT_TRAINING_DATA_DIR / "lacks_synthesis" / "lacks_synthesis_ner_eval.json",
            ("format", "sentence"): DEFAULT_TRAINING_DATA_DIR / "format" / "format_ner_eval.json",
            ("unsupported_claim", "sentence"): DEFAULT_TRAINING_DATA_DIR / "unsupported_claim" / "unsupported_claim_ner_eval.json",
            ("unsupported_claim", "short"): DEFAULT_TRAINING_DATA_DIR / "unsupported_claim" / "unsupported_claim_ner_eval_short_1_5w.json",
            ("unsupported_claim", "long"): DEFAULT_TRAINING_DATA_DIR / "unsupported_claim" / "unsupported_claim_ner_eval_long_6plus.json",
        }
        try:
            output_path = default_name_map[(category_key, task_key)]
        except KeyError as exc:
            raise ValueError(
                f"No default export path defined for category={category_key!r} task={task_key!r}"
            ) from exc

    if task_key == "sentence":
        if category_key == "unsupported_claim":
            ds = _build_unsupported_claim_sentence_dataset(
                source_path,
                min_span_words=6,
                max_span_words=None,
                neg_ratio=0.8,
                seed=42,
            )
        else:
            require_positive = category_key in {"coherence", "lacks_synthesis", "format"}
            ds = _build_sentence_level_bio_dataset(
                source_path,
                display_name,
                require_positive=require_positive,
            )
    elif task_key == "short":
        if category_key != "unsupported_claim":
            raise ValueError("The short export task is only supported for unsupported_claim.")
        ds = _build_unsupported_claim_sentence_dataset(
            source_path,
            min_span_words=1,
            max_span_words=5,
            neg_ratio=0.1,
            seed=42,
        )
    elif task_key == "long":
        if category_key != "unsupported_claim":
            raise ValueError("The long export task is only supported for unsupported_claim.")
        ds = _build_unsupported_claim_sentence_dataset(
            source_path,
            min_span_words=6,
            max_span_words=None,
            neg_ratio=0.8,
            seed=42,
        )
    else:
        raise ValueError(f"Unsupported export task: {task_name!r}")

    rows = _dataset_to_jsonable_rows(ds)
    if category_key in {"format", "unsupported_claim"}:
        rows = _trim_fields(rows, ("tokens", "labels"))
    else:
        rows = _trim_fields(rows, ("doc_index", "sent_index", "text", "tokens", "labels"))

    output_path = Path(output_path)
    _write_json(output_path, rows)
    return {
        "category_name": display_name,
        "category_key": category_key,
        "task_name": task_key,
        "source_path": str(source_path),
        "output_path": str(output_path),
        "num_examples": len(rows),
    }


def _load_safetensors_state_dict(weights_path: Path) -> Dict[str, torch.Tensor]:
    try:
        from safetensors.torch import load_file
    except ImportError as e:
        raise ImportError(
            "Loading a local .safetensors checkpoint requires the `safetensors` package."
        ) from e
    return load_file(str(weights_path))


def _load_checkpoint_state_dict(checkpoint_path: Path) -> Dict[str, torch.Tensor]:
    safetensors_path = checkpoint_path / "model.safetensors"
    if safetensors_path.exists():
        return _load_safetensors_state_dict(safetensors_path)

    pytorch_bin_path = checkpoint_path / "pytorch_model.bin"
    if pytorch_bin_path.exists():
        return torch.load(str(pytorch_bin_path), map_location="cpu")

    raise FileNotFoundError(
        f"No supported checkpoint weights found in {checkpoint_path}. "
        "Expected model.safetensors or pytorch_model.bin."
    )


def _configure_safe_model_config(config: AutoConfig) -> AutoConfig:
    model_type = getattr(config, "model_type", None)
    if model_type == "modernbert":
        # ModernBERT defaults to more aggressive attention/compile paths that are fragile
        # across torch/flash-attn/torchao combinations. Keep evaluation on the safest path.
        setattr(config, "reference_compile", False)
        setattr(config, "_attn_implementation_internal", "eager")
        setattr(config, "_attn_implementation", "eager")
    return config


def _directory_has_hf_weights(model_dir: Path) -> bool:
    return any(
        (model_dir / candidate).exists()
        for candidate in (
            "model.safetensors",
            "model.safetensors.index.json",
            "pytorch_model.bin",
            "pytorch_model.bin.index.json",
        )
    )


def _validate_model_source(
    model_name_or_path: str | Path,
    *,
    tokenizer_name_or_path: Optional[str | Path] = None,
    config_name_or_path: Optional[str | Path] = None,
) -> None:
    model_path = Path(model_name_or_path)
    source = str(model_name_or_path)

    if model_path.exists():
        if model_path.is_dir():
            if not (model_path / "config.json").exists():
                raise FileNotFoundError(
                    f"Model directory {model_path} is missing config.json."
                )
            if not _directory_has_hf_weights(model_path):
                raise FileNotFoundError(
                    f"Model directory {model_path} does not contain model weights "
                    "(expected model.safetensors, model.safetensors.index.json, "
                    "pytorch_model.bin, or pytorch_model.bin.index.json)."
                )
        elif model_path.suffix == ".safetensors":
            if not model_path.is_file():
                raise FileNotFoundError(f"Model file not found: {model_path}")
        else:
            raise FileNotFoundError(
                f"Unsupported model path {model_path}. Expected a checkpoint directory "
                "or a .safetensors file."
            )
    else:
        if source.startswith("/") or source.startswith("."):
            raise FileNotFoundError(f"Model path not found: {model_path}")

    for label, raw_path in (
        ("Tokenizer", tokenizer_name_or_path),
        ("Config", config_name_or_path),
    ):
        if raw_path is None:
            continue
        resolved = Path(raw_path)
        if resolved.exists():
            continue
        raw_text = str(raw_path)
        if raw_text.startswith("/") or raw_text.startswith("."):
            raise FileNotFoundError(f"{label} path not found: {resolved}")


def _load_tokenizer_and_model(
    model_name_or_path: str | Path,
    model_cls,
    *,
    tokenizer_name_or_path: Optional[str | Path] = None,
    config_name_or_path: Optional[str | Path] = None,
):
    _log_stage(
        f"Validating model source: model={model_name_or_path} tokenizer={tokenizer_name_or_path} config={config_name_or_path}"
    )
    _validate_model_source(
        model_name_or_path,
        tokenizer_name_or_path=tokenizer_name_or_path,
        config_name_or_path=config_name_or_path,
    )
    model_path = Path(model_name_or_path)
    source = str(model_name_or_path)

    if model_path.is_file() and model_path.suffix == ".safetensors":
        tokenizer_source = str(tokenizer_name_or_path or model_path.parent)
        config_source = str(config_name_or_path or model_path.parent)

        _log_stage(f"Loading tokenizer from {tokenizer_source}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True, use_fast=True)

        _log_stage(f"Loading config from {config_source}")
        config = _configure_safe_model_config(
            AutoConfig.from_pretrained(config_source, trust_remote_code=True)
        )
        _log_stage("Instantiating model from config")
        model = model_cls.from_config(config, trust_remote_code=True)

        _log_stage(f"Loading state_dict from {model_path}")
        state_dict = _load_safetensors_state_dict(model_path)
        _log_stage("Applying state_dict to model")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            raise ValueError(
                f"Failed to load {model_path}: missing keys in checkpoint: {missing[:20]}"
            )
        if unexpected:
            raise ValueError(
                f"Failed to load {model_path}: unexpected keys in checkpoint: {unexpected[:20]}"
            )
        _log_stage(f"Model loaded; initial parameter device={next(model.parameters()).device}")
        return tokenizer, model

    config_source = str(config_name_or_path or source)
    config = _configure_safe_model_config(
        AutoConfig.from_pretrained(config_source, trust_remote_code=True)
    )

    _log_stage(f"Loading tokenizer from {tokenizer_name_or_path or source}")
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_name_or_path or source),
        trust_remote_code=True,
        use_fast=True,
    )
    if model_path.is_dir() and getattr(config, "model_type", None) == "modernbert":
        _log_stage(
            f"Loading ModernBERT from checkpoint directory {source} via config+state_dict "
            "(reference_compile=False, attn_implementation=eager)"
        )
        model = model_cls.from_config(config, trust_remote_code=True)
        _log_stage(f"Loading state_dict from checkpoint directory {model_path}")
        state_dict = _load_checkpoint_state_dict(model_path)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            raise ValueError(
                f"Failed to load {model_path}: missing keys in checkpoint: {missing[:20]}"
            )
        if unexpected:
            raise ValueError(
                f"Failed to load {model_path}: unexpected keys in checkpoint: {unexpected[:20]}"
            )
    else:
        _log_stage(f"Loading model from {source} with low_cpu_mem_usage=True")
        model = model_cls.from_pretrained(
            source,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            config=config,
            attn_implementation=getattr(config, "_attn_implementation", None),
        )
    _log_stage(f"Model loaded; initial parameter device={next(model.parameters()).device}")
    return tokenizer, model


def _neobert_token_label_maps(model_path: Path, num_labels: int) -> Tuple[Dict[int, str], Dict[str, int]]:
    path_text = str(model_path).lower()
    if num_labels == 2 and "unsupported_short" in path_text:
        labels = ["O", "ENTITY"]
    elif num_labels == 3 and "unsupported_long" in path_text:
        labels = ["O", "B-UNSUPPORTED_CLAIM", "I-UNSUPPORTED_CLAIM"]
    elif num_labels == 3 and "format" in path_text:
        labels = ["O", "B-FORMAT", "I-FORMAT"]
    else:
        raise ValueError(
            f"Cannot infer NeoBERT token labels from {model_path} with classifier size {num_labels}."
        )
    id2label = {index: label for index, label in enumerate(labels)}
    return id2label, {label: index for index, label in id2label.items()}


def _load_custom_neobert_token_model(
    model_path: Path,
    *,
    tokenizer_name_or_path: Optional[str | Path] = None,
    config_name_or_path: Optional[str | Path] = None,
):
    """Load checkpoints produced by the local NeoBERT token-training wrapper."""
    _validate_model_source(
        model_path,
        tokenizer_name_or_path=tokenizer_name_or_path,
        config_name_or_path=config_name_or_path,
    )
    tokenizer_source = str(tokenizer_name_or_path or model_path.parent)
    config_source = str(config_name_or_path or "chandar-lab/NeoBERT")
    _log_stage(f"Loading tokenizer from {tokenizer_source}")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        trust_remote_code=True,
        use_fast=True,
    )
    _log_stage(f"Loading NeoBERT base config from {config_source}")
    config = AutoConfig.from_pretrained(config_source, trust_remote_code=True)
    state_dict = _load_safetensors_state_dict(model_path)
    classifier_weight = state_dict.get("classifier.weight")
    if classifier_weight is None or classifier_weight.ndim != 2:
        raise ValueError(f"NeoBERT checkpoint {model_path} is missing classifier.weight.")
    num_labels = int(classifier_weight.shape[0])
    id2label, label2id = _neobert_token_label_maps(model_path, num_labels)
    config.num_labels = num_labels
    config.id2label = id2label
    config.label2id = label2id

    class NeoBERTForTokenClassification(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = config
            self.num_labels = num_labels
            self.id2label = id2label
            self.label2id = label2id
            self.bert = AutoModel.from_config(config, trust_remote_code=True)
            self.dropout = torch.nn.Dropout(0.1)
            self.classifier = torch.nn.Linear(int(config.hidden_size), num_labels)

        def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            labels=None,
            **kwargs,
        ):
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **kwargs,
            )
            logits = self.classifier(self.dropout(outputs.last_hidden_state))
            loss = None
            if labels is not None:
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, self.num_labels),
                    labels.view(-1),
                    ignore_index=-100,
                )
            return TokenClassifierOutput(loss=loss, logits=logits)

    _log_stage(f"Instantiating custom NeoBERT token classifier with {num_labels} labels")
    model = NeoBERTForTokenClassification()
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        raise ValueError(
            f"Failed to load {model_path}: missing keys in checkpoint: {missing[:20]}"
        )
    if unexpected:
        raise ValueError(
            f"Failed to load {model_path}: unexpected keys in checkpoint: {unexpected[:20]}"
        )
    _log_stage(f"Model loaded; initial parameter device={next(model.parameters()).device}")
    return tokenizer, model


def _load_token_model(
    model_name_or_path: str | Path,
    *,
    tokenizer_name_or_path: Optional[str | Path] = None,
    config_name_or_path: Optional[str | Path] = None,
):
    model_path = Path(model_name_or_path)
    if model_path.is_file() and model_path.suffix == ".safetensors" and "neobert" in str(model_path).lower():
        return _load_custom_neobert_token_model(
            model_path,
            tokenizer_name_or_path=tokenizer_name_or_path,
            config_name_or_path=config_name_or_path,
        )
    return _load_tokenizer_and_model(
        model_name_or_path,
        AutoModelForTokenClassification,
        tokenizer_name_or_path=tokenizer_name_or_path,
        config_name_or_path=config_name_or_path,
    )


def _load_sequence_model(
    model_name_or_path: str | Path,
    *,
    tokenizer_name_or_path: Optional[str | Path] = None,
    config_name_or_path: Optional[str | Path] = None,
):
    return _load_tokenizer_and_model(
        model_name_or_path,
        AutoModelForSequenceClassification,
        tokenizer_name_or_path=tokenizer_name_or_path,
        config_name_or_path=config_name_or_path,
    )


def _model_label_maps(model) -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    label_list = [id2label[i] for i in sorted(id2label)]
    label2id = {label: idx for idx, label in enumerate(label_list)}
    return label_list, label2id, {v: k for k, v in label2id.items()}


def _ensure_labels_exist(label2id: Dict[str, int], required_labels: Iterable[str]) -> None:
    missing = [lbl for lbl in required_labels if lbl not in label2id]
    if missing:
        raise ValueError(f"Model is missing expected labels: {missing}. Available labels: {sorted(label2id)}")


def tokenize_and_align_labels(example, tokenizer, label2id, label_all_subtokens=True):
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
            aligned.append(label2id[_continuation_label(word_label)] if label_all_subtokens else -100)
        prev_word_idx = word_idx
    tokenized["labels"] = aligned
    tokenized["word_starts"] = word_starts
    return tokenized


def tokenize_and_align_binary(example, tokenizer, label2id, label_all_subtokens=True):
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
            aligned.append(label2id[word_label] if label_all_subtokens else -100)
            word_starts.append(0)
        prev_word_idx = word_idx
    tokenized["labels"] = aligned
    tokenized["word_starts"] = word_starts
    return tokenized


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


def _trainer_output_dir(output_dir: Optional[str | Path], name: str) -> str:
    if output_dir is not None:
        path = Path(output_dir) / name
        path.mkdir(parents=True, exist_ok=True)
        return str(path)
    return tempfile.mkdtemp(prefix=f"{name}_")


def _build_predict_trainer(
    model,
    tokenizer,
    data_collator,
    output_dir: Optional[str | Path],
    name: str,
    *,
    eval_batch_size: int = 4,
) -> Any:
    Trainer, TrainingArguments = _get_trainer_components()

    class PredictOnlyTrainer(Trainer):
        def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
            inputs = dict(inputs)
            inputs.pop("word_starts", None)
            return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)

    _log_stage(
        f"Building Trainer(name={name}, eval_batch_size={max(1, int(eval_batch_size))}, cuda_available={torch.cuda.is_available()})"
    )
    args = TrainingArguments(
        output_dir=_trainer_output_dir(output_dir, name),
        per_device_eval_batch_size=max(1, int(eval_batch_size)),
        report_to="none",
        remove_unused_columns=False,
    )
    trainer = PredictOnlyTrainer(model=model, args=args, tokenizer=tokenizer, data_collator=data_collator)
    _log_stage(f"Trainer ready; model parameter device={next(trainer.model.parameters()).device}")
    return trainer


def flatten_valid_tokens(predictions, labels):
    flat_preds, flat_labels = [], []
    for pred_seq, label_seq in zip(predictions, labels):
        for p, l in zip(pred_seq, label_seq):
            if l != -100:
                flat_preds.append(int(p))
                flat_labels.append(int(l))
    return np.array(flat_preds), np.array(flat_labels)


def macro_cohen_kappa_from_sequences(
    label_sequences,
    pred_sequences,
    label_ids,
    *,
    ignore_index: Optional[int] = -100,
    include_gold_ids: Optional[Sequence[int]] = None,
):
    """Average Cohen's kappa across non-empty examples instead of pooling tokens."""
    kappas = []
    allowed_gold = None if include_gold_ids is None else np.asarray(include_gold_ids, dtype=int)
    for labels, preds in zip(label_sequences, pred_sequences):
        labels = np.asarray(labels, dtype=int).reshape(-1)
        preds = np.asarray(preds, dtype=int).reshape(-1)
        size = min(labels.size, preds.size)
        labels = labels[:size]
        preds = preds[:size]
        mask = np.ones(size, dtype=bool)
        if ignore_index is not None:
            mask &= labels != int(ignore_index)
        if allowed_gold is not None:
            mask &= np.isin(labels, allowed_gold)
        labels = labels[mask]
        preds = preds[mask]
        if labels.size == 0:
            continue
        if np.unique(np.concatenate((labels, preds))).size < 2:
            kappa = 0.0
        else:
            kappa = cohen_kappa_score(labels, preds, labels=list(label_ids))
            kappa = 0.0 if np.isnan(kappa) else float(kappa)
        kappas.append(float(kappa))
    return {
        "mean": float(np.mean(kappas)) if kappas else 0.0,
        "num_examples_scored": len(kappas),
    }


def filter_out_gold_O(flat_preds, flat_labels, id2label):
    if flat_labels.size == 0:
        return flat_preds, flat_labels
    keep = np.array([id2label[int(l)] != "O" for l in flat_labels], dtype=bool)
    return flat_preds[keep], flat_labels[keep]


def compute_seqeval_f1(pred_ids, label_ids, id2label):
    tp = fp = fn = 0
    for p_seq, l_seq in zip(pred_ids, label_ids):
        pred_labels = []
        gold_labels = []
        for p, l in zip(p_seq, l_seq):
            if l != -100:
                pred_labels.append(id2label[int(p)])
                gold_labels.append(id2label[int(l)])
        dummy_pred_tokens = ["_"] * len(pred_labels)
        dummy_gold_tokens = ["_"] * len(gold_labels)
        pred_spans = {(s["start_token"], s["end_token"], s["label"]) for s in bio_spans_from_word_labels(dummy_pred_tokens, pred_labels)}
        gold_spans = {(s["start_token"], s["end_token"], s["label"]) for s in bio_spans_from_word_labels(dummy_gold_tokens, gold_labels)}
        tp += len(pred_spans & gold_spans)
        fp += len(pred_spans - gold_spans)
        fn += len(gold_spans - pred_spans)
    return float((2 * tp) / ((2 * tp) + fp + fn)) if ((2 * tp) + fp + fn) else 0.0


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
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1), "tp": int(tp), "fp": int(fp), "fn": int(fn)}


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
    ent = sum(1 for l in word_labels if (l or "O").strip() != "O")
    coverage = ent / len(word_labels)
    return 1 if coverage >= threshold else 0


def word_level_pred_ids_from_aligned(pred_ids_seq, aligned_label_ids_seq):
    return [int(p) for p, l in zip(pred_ids_seq.tolist(), aligned_label_ids_seq.tolist()) if l != -100]


def word_level_pred_labels_from_word_starts(pred_ids_seq, word_starts_seq, id2label):
    L = min(len(pred_ids_seq), len(word_starts_seq))
    idxs = [i for i in range(L) if int(word_starts_seq[i]) == 1]
    return [id2label[int(pred_ids_seq[i])] for i in idxs]


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
            spans.append({"label": ent, "start_token": start, "end_token": i, "text": " ".join(tokens[start:i])})
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
            if (i - start) < int(min_span_len):
                for k in range(start, i):
                    out[k] = "O"
            continue
        i += 1
    return out


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


def match_spans_iou(gold_spans, pred_spans, iou_threshold=0.5):
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
    for _, gi, pi in candidates:
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
    n = min(len(gold_spans_per_example), len(pred_spans_per_example))
    annotators = {"gold": gold_spans_per_example[:n], "pred": pred_spans_per_example[:n]}
    total_tp = total_fp = total_fn = 0
    per_pair = {}
    macro_precisions = []
    macro_recalls = []
    macro_f1s = []
    for i in range(n):
        gold = annotators["gold"][i]
        pred = annotators["pred"][i]
        if not gold and not pred:
            continue
        example_tp = example_fp = example_fn = 0
        for a_spans, b_spans in ((gold, pred), (pred, gold)):
            tp, fp, fn = match_spans_iou(a_spans, b_spans, iou_threshold=iou_threshold)
            example_tp += tp
            example_fp += fp
            example_fn += fn
        example_prf = _prf(example_tp, example_fp, example_fn)
        macro_precisions.append(example_prf["precision"])
        macro_recalls.append(example_prf["recall"])
        macro_f1s.append(example_prf["f1"])

    for a, b in [("gold", "pred"), ("pred", "gold")]:
        tp = fp = fn = 0
        for i in range(n):
            tpi, fpi, fni = match_spans_iou(annotators[a][i], annotators[b][i], iou_threshold=iou_threshold)
            tp += tpi
            fp += fpi
            fn += fni
        per_pair[f"{a}→{b}"] = _prf(tp, fp, fn)
        total_tp += tp
        total_fp += fp
        total_fn += fn
    return {
        "pairwise_micro": _prf(total_tp, total_fp, total_fn),
        "pairwise_macro": {
            "precision": float(np.mean(macro_precisions)) if macro_precisions else 0.0,
            "recall": float(np.mean(macro_recalls)) if macro_recalls else 0.0,
            "f1": float(np.mean(macro_f1s)) if macro_f1s else 0.0,
        },
        "per_pair": per_pair,
        "num_examples": n,
        "num_examples_macro_scored": len(macro_f1s),
        "iou_threshold": iou_threshold,
    }


def _prepare_token_eval_dataset(raw_ds: Dataset, tokenizer, label2id, *, binary: bool = False) -> Dataset:
    mapper = tokenize_and_align_binary if binary else tokenize_and_align_labels
    eval_ds = raw_ds.map(lambda x: mapper(x, tokenizer, label2id, True), batched=False)
    keep = {"input_ids", "attention_mask", "labels", "word_starts"}
    if "token_type_ids" in eval_ds.column_names:
        keep.add("token_type_ids")
    drop = [c for c in eval_ds.column_names if c not in keep]
    if drop:
        eval_ds = eval_ds.remove_columns(drop)
    return eval_ds


def evaluate_sentence_model(
    model_name_or_path: str | Path,
    data_path: str | Path = DEFAULT_COMBINED_REAL_DATA,
    category_name: str = "Unsupported claim",
    *,
    tokenizer_name_or_path: Optional[str | Path] = None,
    config_name_or_path: Optional[str | Path] = None,
    coverage_threshold: float = 0.5,
    min_pred_span_len: int = 1,
    iou_threshold: float = 0.5,
    output_dir: Optional[str | Path] = None,
    eval_batch_size: int = 4,
) -> Dict[str, Any]:
    _log_stage(
        f"Starting sentence evaluation for model={model_name_or_path} data={data_path} category={category_name}"
    )
    raw_ds = _load_sentence_eval_dataset(data_path, category_name)
    _log_stage(f"Built sentence dataset with {len(raw_ds)} examples")
    tokenizer, model = _load_token_model(
        model_name_or_path,
        tokenizer_name_or_path=tokenizer_name_or_path,
        config_name_or_path=config_name_or_path,
    )
    label_list, label2id, id2label = _model_label_maps(model)

    entity_name = _norm_entity_name(category_name)
    _ensure_labels_exist(label2id, ["O", f"B-{entity_name}", f"I-{entity_name}"])

    eval_ds = _prepare_token_eval_dataset(raw_ds, tokenizer, label2id, binary=False)
    _log_stage(f"Tokenized sentence dataset with {len(eval_ds)} examples")
    trainer = _build_predict_trainer(
        model,
        tokenizer,
        DataCollatorForTokenClassificationWithWordStarts(tokenizer, padding=True),
        output_dir,
        "sentence_eval",
        eval_batch_size=eval_batch_size,
    )

    _log_stage("Running sentence trainer.predict()")
    logits, aligned_gold, _ = trainer.predict(eval_ds)
    pred_ids = np.argmax(logits, axis=2)
    pred_ids = apply_bio_fix_to_aligned_pred_ids(pred_ids, aligned_gold, id2label, label2id)
    if int(min_pred_span_len) > 1:
        pred_ids = apply_remove_short_spans_to_aligned_pred_ids(pred_ids, aligned_gold, id2label, label2id, min_span_len=int(min_pred_span_len))

    gold_sent_y, pred_sent_y = [], []
    gold_word_binary_sequences = []
    pred_word_binary_sequences = []
    pred_spans_by_example = []
    for i, (raw_ex, pred_seq, gold_seq) in enumerate(zip(raw_ds, pred_ids, aligned_gold)):
        gold_word_labels = raw_ex["labels"]
        pred_word_labels = word_level_pred_labels_from_word_starts(
            pred_seq,
            eval_ds[i]["word_starts"],
            id2label,
        )
        pred_word_labels = bio_fix_sequence(pred_word_labels)
        if int(min_pred_span_len) > 1:
            pred_word_labels = remove_short_predicted_spans_word_labels(pred_word_labels, min_span_len=int(min_pred_span_len))
        gold_word_binary = [0 if str(label).strip() == "O" else 1 for label in gold_word_labels]
        pred_word_binary = [0 if str(label).strip() == "O" else 1 for label in pred_word_labels]
        gold_word_binary_sequences.append(gold_word_binary)
        pred_word_binary_sequences.append(pred_word_binary)
        gold_sent_y.append(sentence_label_from_bio_coverage(gold_word_labels, threshold=coverage_threshold))
        pred_sent_y.append(sentence_label_from_bio_coverage(pred_word_labels, threshold=coverage_threshold))
        gold_spans = bio_spans_from_word_labels(raw_ex["tokens"], gold_word_labels)
        pred_spans = bio_spans_from_word_labels(raw_ex["tokens"], pred_word_labels)
        pred_spans_by_example.append(
            {
                "doc_index": raw_ex.get("doc_index"),
                "sent_index": raw_ex.get("sent_index"),
                "text": raw_ex.get("text"),
                "tokens": list(raw_ex["tokens"]),
                "gold_labels": list(gold_word_labels),
                "pred_labels": list(pred_word_labels),
                "gold_binary": gold_word_binary,
                "pred_binary": pred_word_binary,
                "gold_spans": gold_spans,
                "pred_spans": pred_spans,
                "aligned_gold_labels": [id2label[int(label)] for label in gold_seq if int(label) != -100],
                "aligned_pred_labels": [
                    id2label[int(prediction)]
                    for prediction, label in zip(pred_seq, gold_seq)
                    if int(label) != -100
                ],
            }
        )

    gold_sent_y = np.array(gold_sent_y, dtype=int)
    pred_sent_y = np.array(pred_sent_y, dtype=int)
    sent_cm = confusion_matrix(gold_sent_y, pred_sent_y, labels=[0, 1]).tolist()
    sent_p, sent_r, sent_f1, _ = precision_recall_fscore_support(gold_sent_y, pred_sent_y, average="binary", zero_division=0)
    sent_acc = float((gold_sent_y == pred_sent_y).mean()) if gold_sent_y.size else 0.0

    flat_gold_word_binary = np.asarray(
        [label for sequence in gold_word_binary_sequences for label in sequence],
        dtype=np.int64,
    )
    flat_pred_word_binary = np.asarray(
        [label for sequence in pred_word_binary_sequences for label in sequence],
        dtype=np.int64,
    )
    binary_word_metrics = compute_binary_metrics_from_arrays(
        flat_gold_word_binary,
        flat_pred_word_binary,
    )
    binary_word_macro_kappa = macro_cohen_kappa_from_sequences(
        gold_word_binary_sequences,
        pred_word_binary_sequences,
        [0, 1],
        ignore_index=None,
    )

    flat_preds, flat_labels = flatten_valid_tokens(pred_ids, aligned_gold)
    cm_all = confusion_matrix(flat_labels, flat_preds, labels=list(range(len(label_list))))
    alpha_all = krippendorff_alpha_nominal(flat_labels, flat_preds, num_labels=len(label_list))
    kappa_all = float(cohen_kappa_score(flat_labels, flat_preds, labels=list(range(len(label_list))))) if flat_labels.size else 0.0
    p_micro_all, r_micro_all, f1_micro_all, _ = precision_recall_fscore_support(flat_labels, flat_preds, average="micro", zero_division=0)
    p_macro_all, r_macro_all, f1_macro_all, _ = precision_recall_fscore_support(flat_labels, flat_preds, average="macro", zero_division=0)
    label_ids = list(range(len(label_list)))
    macro_kappa_all = macro_cohen_kappa_from_sequences(
        aligned_gold,
        pred_ids,
        label_ids,
    )

    flat_preds_ent, flat_labels_ent = filter_out_gold_O(flat_preds, flat_labels, id2label)
    cm_ent = confusion_matrix(flat_labels_ent, flat_preds_ent, labels=list(range(len(label_list))))
    alpha_ent = krippendorff_alpha_nominal(flat_labels_ent, flat_preds_ent, num_labels=len(label_list))
    kappa_ent = float(cohen_kappa_score(flat_labels_ent, flat_preds_ent, labels=list(range(len(label_list))))) if flat_labels_ent.size else 0.0
    p_micro_ent, r_micro_ent, f1_micro_ent, _ = precision_recall_fscore_support(flat_labels_ent, flat_preds_ent, average="micro", zero_division=0)
    p_macro_ent, r_macro_ent, f1_macro_ent, _ = precision_recall_fscore_support(flat_labels_ent, flat_preds_ent, average="macro", zero_division=0)
    entity_label_ids = [label_id for label_id in label_ids if id2label[label_id] != "O"]
    macro_kappa_ent = macro_cohen_kappa_from_sequences(
        aligned_gold,
        pred_ids,
        label_ids,
        include_gold_ids=entity_label_ids,
    )

    gold_spans_list = [[s["text"] for s in ex["gold_spans"]] for ex in pred_spans_by_example]
    pred_spans_list = [[s["text"] for s in ex["pred_spans"]] for ex in pred_spans_by_example]

    result = {
        "task": "sentence_token_classification",
        "model_name_or_path": model_name_or_path,
        "data_path": str(data_path),
        "category_name": category_name,
        "num_examples": int(len(raw_ds)),
        "coverage_threshold": float(coverage_threshold),
        "seqeval_span_f1": float(compute_seqeval_f1(pred_ids, aligned_gold, id2label)),
        "token_overlap_f1_entity_vs_O": token_overlap_f1(pred_ids, aligned_gold, id2label),
        "pairwise_span_f1": pairwise_span_f1(gold_spans_list, pred_spans_list, iou_threshold=iou_threshold),
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
            "label_space": "binary_word_inside_vs_outside_span",
            "cohen_kappa_token_level": binary_word_metrics["cohen_kappa"],
            "cohen_kappa_token_level_macro": binary_word_macro_kappa,
            "token_prf": {
                "binary": {
                    "precision": binary_word_metrics["precision"],
                    "recall": binary_word_metrics["recall"],
                    "f1": binary_word_metrics["f1"],
                },
            },
            "accuracy": binary_word_metrics["accuracy"],
            "confusion_matrix": {
                "labels": ["O", "ENTITY"],
                "matrix": binary_word_metrics["confusion_matrix"],
            },
            "num_tokens_scored": binary_word_metrics["num_examples"],
        },
        "bio_subtoken_metrics_all_tokens": {
            "krippendorff_alpha_nominal": float(alpha_all),
            "cohen_kappa_token_level": float(kappa_all),
            "cohen_kappa_token_level_macro": macro_kappa_all,
            "token_prf": {
                "micro": {"precision": float(p_micro_all), "recall": float(r_micro_all), "f1": float(f1_micro_all)},
                "macro": {"precision": float(p_macro_all), "recall": float(r_macro_all), "f1": float(f1_macro_all)},
            },
            "confusion_matrix": {"labels": label_list, "matrix": cm_all.tolist()},
            "num_tokens_scored": int(flat_labels.size),
        },
        "bio_subtoken_metrics_entity_only": {
            "krippendorff_alpha_nominal": float(alpha_ent),
            "cohen_kappa_token_level": float(kappa_ent),
            "cohen_kappa_token_level_macro": macro_kappa_ent,
            "token_prf": {
                "micro": {"precision": float(p_micro_ent), "recall": float(r_micro_ent), "f1": float(f1_micro_ent)},
                "macro": {"precision": float(p_macro_ent), "recall": float(r_macro_ent), "f1": float(f1_macro_ent)},
            },
            "confusion_matrix": {"labels": label_list, "matrix": cm_ent.tolist()},
            "num_tokens_scored": int(flat_labels_ent.size),
        },
    }

    if output_dir is not None:
        out = Path(output_dir)
        _write_json(out / "sentence_eval_predicted_spans.json", pred_spans_by_example)
        _write_json(out / "sentence_eval_metrics.json", result)

    return result


def coverage_label_from_bio(word_labels, threshold=0.5):
    if not word_labels:
        return 0
    ent = sum(1 for l in word_labels if (l or "O").strip() != "O")
    cov = ent / len(word_labels)
    return 1 if cov >= threshold else 0


def build_token_windows(tokens, labels, window_tokens=160, stride_tokens=80):
    out = []
    n = len(tokens)
    if n == 0:
        return out
    w = int(window_tokens) if int(window_tokens) > 0 else n
    s = int(stride_tokens) if int(stride_tokens) > 0 else w
    for start in range(0, n, s):
        end = min(n, start + w)
        win_toks = tokens[start:end]
        win_labs = labels[start:end]
        if win_toks:
            out.append((start, end, win_toks, win_labs))
        if end == n:
            break
    return out


def make_classification_dataset(raw_ds, threshold=0.5, window_tokens=160, stride_tokens=80):
    rows = []
    for dataset_index, ex in enumerate(raw_ds):
        toks = ex["tokens"]
        labs = bio_fix_sequence(ex["labels"])
        doc_index = ex.get("doc_index")
        if doc_index is None:
            doc_index = dataset_index
        windows = build_token_windows(toks, labs, window_tokens=window_tokens, stride_tokens=stride_tokens)
        for w_idx, (s, e, win_toks, win_labs) in enumerate(windows):
            rows.append(
                {
                    # Keep token lists here to avoid duplicating large window strings
                    # before tokenization. The tokenizer can consume pre-split tokens.
                    "tokens": win_toks,
                    "labels": int(coverage_label_from_bio(win_labs, threshold=threshold)),
                    "win_start_token": int(s),
                    "win_end_token": int(e),
                    "window_index": int(w_idx),
                    "doc_index": int(doc_index),
                }
            )
    return Dataset.from_list(rows)


def tokenize_cls(example, tokenizer):
    max_len = getattr(tokenizer, "model_max_length", 512) or 512
    max_len = min(int(max_len), 512)
    return tokenizer(
        example["tokens"],
        is_split_into_words=True,
        truncation=True,
        max_length=max_len,
    )


def safe_cohen_kappa(labels, preds):
    labels = np.asarray(labels).astype(int)
    preds = np.asarray(preds).astype(int)
    if labels.size == 0:
        return 0.0
    if np.unique(labels).size < 2 or np.unique(preds).size < 2:
        return 0.0
    kappa = cohen_kappa_score(labels, preds, labels=[0, 1])
    return 0.0 if np.isnan(kappa) else float(kappa)


def compute_binary_metrics_from_arrays(labels, preds):
    labels = np.asarray(labels).astype(int)
    preds = np.asarray(preds).astype(int)
    cm = confusion_matrix(labels, preds, labels=[0, 1]).tolist()
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    acc = float((labels == preds).mean()) if labels.size else 0.0
    return {
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "accuracy": float(acc),
        "cohen_kappa": float(safe_cohen_kappa(labels, preds)),
        "confusion_matrix_labels": [0, 1],
        "confusion_matrix": cm,
        "num_examples": int(labels.size),
    }


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


def macro_window_metrics(y_gold, y_pred, group_ids):
    """Average window pairwise F1 and Cohen's kappa across documents."""
    grouped = {}
    for gold, pred, group_id in zip(y_gold, y_pred, group_ids):
        group = grouped.setdefault(group_id, {"gold": [], "pred": []})
        group["gold"].append(int(gold))
        group["pred"].append(int(pred))

    precisions = []
    recalls = []
    f1s = []
    kappas = []
    for group in grouped.values():
        pairwise = pairwise_window_f1(group["gold"], group["pred"])["pairwise_micro"]
        precisions.append(pairwise["precision"])
        recalls.append(pairwise["recall"])
        f1s.append(pairwise["f1"])
        kappas.append(safe_cohen_kappa(group["gold"], group["pred"]))

    return {
        "pairwise_macro": {
            "precision": float(np.mean(precisions)) if precisions else 0.0,
            "recall": float(np.mean(recalls)) if recalls else 0.0,
            "f1": float(np.mean(f1s)) if f1s else 0.0,
        },
        "cohen_kappa_macro": {
            "mean": float(np.mean(kappas)) if kappas else 0.0,
            "num_documents_scored": len(kappas),
        },
        "num_documents": len(grouped),
    }


def evaluate_paragraph_model(
    model_name_or_path: str | Path,
    data_path: str | Path = DEFAULT_COMBINED_REAL_DATA,
    category_name: str = "Unsupported claim",
    *,
    tokenizer_name_or_path: Optional[str | Path] = None,
    config_name_or_path: Optional[str | Path] = None,
    coverage_threshold: float = 0.5,
    window_tokens: int = 160,
    stride_tokens: int = 80,
    output_dir: Optional[str | Path] = None,
    eval_batch_size: int = 4,
) -> Dict[str, Any]:
    _log_stage(
        f"Starting paragraph evaluation for model={model_name_or_path} data={data_path} category={category_name}"
    )
    raw_ds = _load_paragraph_eval_dataset(data_path, category_name)
    _log_stage(f"Built paragraph document dataset with {len(raw_ds)} examples")
    cls_ds = make_classification_dataset(raw_ds, threshold=coverage_threshold, window_tokens=window_tokens, stride_tokens=stride_tokens)
    _log_stage(f"Built paragraph window dataset with {len(cls_ds)} windows")
    tokenizer, model = _load_sequence_model(
        model_name_or_path,
        tokenizer_name_or_path=tokenizer_name_or_path,
        config_name_or_path=config_name_or_path,
    )
    tok_ds = cls_ds.map(lambda x: tokenize_cls(x, tokenizer), batched=False)
    _log_stage(f"Tokenized paragraph dataset with {len(tok_ds)} windows")

    inspect_ds = tok_ds
    keep = {"input_ids", "attention_mask", "labels"}
    if "token_type_ids" in tok_ds.column_names:
        keep.add("token_type_ids")
    drop = [c for c in tok_ds.column_names if c not in keep]
    if drop:
        tok_ds = tok_ds.remove_columns(drop)

    trainer = _build_predict_trainer(
        model,
        tokenizer,
        DataCollatorWithPadding(tokenizer),
        output_dir,
        "paragraph_eval",
        eval_batch_size=eval_batch_size,
    )
    _log_stage("Running paragraph trainer.predict()")
    pred = trainer.predict(tok_ds)
    logits = pred.predictions
    y_pred = np.argmax(logits, axis=1).astype(int)
    y_true = np.array(tok_ds["labels"], dtype=int)
    window_macro = macro_window_metrics(y_true, y_pred, inspect_ds["doc_index"])
    final_eval_metrics = compute_binary_metrics_from_arrays(y_true, y_pred)
    final_eval_metrics["cohen_kappa_macro_by_document"] = window_macro["cohen_kappa_macro"]
    pairwise_metrics = pairwise_window_f1(y_true, y_pred)
    pairwise_metrics["pairwise_macro"] = window_macro["pairwise_macro"]
    pairwise_metrics["num_documents"] = window_macro["num_documents"]

    inspect = []
    for ex, yp in zip(inspect_ds, y_pred.tolist()):
        inspect.append(
            {
                "doc_index": ex.get("doc_index"),
                "window_index": ex.get("window_index"),
                "win_start_token": ex.get("win_start_token"),
                "win_end_token": ex.get("win_end_token"),
                "text": ex.get("text"),
                "gold": int(ex["labels"]),
                "pred": int(yp),
            }
        )

    result = {
        "task": "windowed_sequence_classification",
        "model_name_or_path": model_name_or_path,
        "data_path": str(data_path),
        "category_name": category_name,
        "coverage_threshold": float(coverage_threshold),
        "window_tokens": int(window_tokens),
        "stride_tokens": int(stride_tokens),
        "eval_num_windows": int(len(tok_ds)),
        "final_eval_metrics": final_eval_metrics,
        "pairwise_window_f1": pairwise_metrics,
    }

    if output_dir is not None:
        out = Path(output_dir)
        _write_json(out / "paragraph_eval_window_predictions.json", inspect)
        _write_json(out / "paragraph_eval_metrics.json", result)

    return result


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
        mask = l_seq != -100
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
    return {"precision": tp / (tp + fp) if (tp + fp) else 0.0, "recall": tp / (tp + fn) if (tp + fn) else 0.0, "f1": (2 * tp) / ((2 * tp) + fp + fn) if ((2 * tp) + fp + fn) else 0.0, "tp": tp, "fp": fp, "fn": fn}


def binary_spans_from_word_labels(tokens: List[str], word_labels: List[str]) -> List[Dict[str, Any]]:
    spans = []
    i = 0
    n = min(len(tokens), len(word_labels))
    while i < n:
        if (word_labels[i] or "O").strip() != "ENTITY":
            i += 1
            continue
        start = i
        i += 1
        while i < n and (word_labels[i] or "O").strip() == "ENTITY":
            i += 1
        spans.append({"label": "ENTITY", "start_token": start, "end_token": i, "text": " ".join(tokens[start:i])})
    return spans


def _short_word_entity_probabilities(logits_seq, word_starts_seq, entity_id: int, num_words: int) -> List[float]:
    logits = np.asarray(logits_seq, dtype=np.float64)
    logits = logits - logits.max(axis=-1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    starts = [i for i, flag in enumerate(word_starts_seq[: len(probabilities)]) if int(flag) == 1]
    result = [0.0] * int(num_words)
    for word_index, token_index in enumerate(starts[:num_words]):
        result[word_index] = float(probabilities[token_index, entity_id])
    return result


def _prune_short_singletons(
    binary_labels: Sequence[int],
    entity_probabilities: Sequence[float],
    *,
    singleton_keep_threshold: float,
) -> List[int]:
    output = [int(value) for value in binary_labels]
    for i, value in enumerate(output):
        if value != 1:
            continue
        has_left = i > 0 and output[i - 1] == 1
        has_right = i + 1 < len(output) and output[i + 1] == 1
        if not (has_left or has_right) and float(entity_probabilities[i]) < float(singleton_keep_threshold):
            output[i] = 0
    return output


def _dynamic_citation_windows(sentence: str) -> Tuple[int, int]:
    length = len(sentence)
    after = max(40, min(int(length * 0.25), 140))
    before = max(20, min(int(length * 0.12), 80))
    return after, before


def _citation_near_short_span(
    sentence: str,
    span_text: str,
    *,
    after_chars: int,
    before_chars: int,
) -> Optional[Dict[str, Any]]:
    if not span_text:
        return None
    mention = re.search(re.escape(span_text), sentence, re.IGNORECASE)
    if mention is None:
        return None
    start, end = mention.span()
    after = sentence[end : end + max(0, int(after_chars))]
    citation_after = _SHORT_CITATION_PATTERN.search(after)
    if citation_after is not None:
        return {
            "citation_found": True,
            "citation_where": "immediately_after" if citation_after.start() == 0 else "after",
            "citation_match": citation_after.group(0),
            "mention_start": start,
            "mention_end": end,
        }
    before = sentence[max(0, start - max(0, int(before_chars))) : start]
    citation_before = None
    for candidate in _SHORT_CITATION_PATTERN.finditer(before):
        citation_before = candidate
    return {
        "citation_found": citation_before is not None,
        "citation_where": "before" if citation_before is not None else None,
        "citation_match": citation_before.group(0) if citation_before is not None else None,
        "mention_start": start,
        "mention_end": end,
    }


def _apply_short_citation_filter(
    tokens: Sequence[str],
    binary_labels: Sequence[int],
) -> Tuple[List[int], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    sentence = " ".join(tokens)
    raw_spans = binary_spans_from_word_labels(
        list(tokens),
        ["ENTITY" if int(value) == 1 else "O" for value in binary_labels],
    )
    after_chars, before_chars = _dynamic_citation_windows(sentence)
    kept = []
    removed = []
    for span in raw_spans:
        citation = _citation_near_short_span(
            sentence,
            span["text"],
            after_chars=after_chars,
            before_chars=before_chars,
        )
        if citation is None:
            removed.append(
                {
                    **span,
                    "citation_found": None,
                    "citation_where": "span_not_found_in_sentence_text",
                    "citation_match": None,
                }
            )
        elif citation["citation_found"]:
            removed.append({**span, **citation})
        else:
            kept.append({**span, **citation})
    final_binary = [0] * len(binary_labels)
    for span in kept:
        for index in range(max(0, int(span["start_token"])), min(len(final_binary), int(span["end_token"]))):
            final_binary[index] = 1
    return final_binary, raw_spans, kept, removed


def _pairwise_binary_labels(binary_labels: Sequence[int]) -> List[int]:
    return [
        1 if int(binary_labels[i]) == int(binary_labels[j]) else 0
        for i in range(len(binary_labels))
        for j in range(i + 1, len(binary_labels))
    ]


def _binary_prf_from_arrays(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, Any]:
    gold = np.asarray(y_true, dtype=np.int64)
    pred = np.asarray(y_pred, dtype=np.int64)
    tp = int(((pred == 1) & (gold == 1)).sum())
    fp = int(((pred == 1) & (gold == 0)).sum())
    fn = int(((pred == 0) & (gold == 1)).sum())
    return {**_prf(tp, fp, fn), "tp": tp, "fp": fp, "fn": fn}


def evaluate_short_unsupp_claim_model(
    model_name_or_path: str | Path,
    data_path: str | Path = DEFAULT_COMBINED_REAL_DATA,
    category_name: str = "Unsupported claim",
    *,
    tokenizer_name_or_path: Optional[str | Path] = None,
    config_name_or_path: Optional[str | Path] = None,
    max_span_words: int = 5,
    output_dir: Optional[str | Path] = None,
    iou_threshold: float = 0.5,
    entity_probability_threshold: float = 0.35,
    singleton_keep_threshold: float = 0.85,
    apply_citation_filter: bool = True,
    eval_batch_size: int = 4,
) -> Dict[str, Any]:
    _log_stage(
        f"Starting short-span evaluation for model={model_name_or_path} data={data_path} category={category_name}"
    )
    raw_ds = _load_short_eval_dataset(
        data_path,
        category_name,
        max_span_words=max_span_words,
    )
    _log_stage(f"Built short-span dataset with {len(raw_ds)} examples")

    tokenizer, model = _load_token_model(
        model_name_or_path,
        tokenizer_name_or_path=tokenizer_name_or_path,
        config_name_or_path=config_name_or_path,
    )
    label_list, label2id, id2label = _model_label_maps(model)
    _ensure_labels_exist(label2id, ["O", "ENTITY"])

    eval_ds = _prepare_token_eval_dataset(raw_ds, tokenizer, label2id, binary=True)
    _log_stage(f"Tokenized short-span dataset with {len(eval_ds)} examples")
    trainer = _build_predict_trainer(
        model,
        tokenizer,
        DataCollatorForTokenClassificationWithWordStarts(tokenizer, padding=True),
        output_dir,
        "short_unsupp_eval",
        eval_batch_size=eval_batch_size,
    )
    _log_stage("Running short-span trainer.predict()")
    pred = trainer.predict(eval_ds)
    logits = pred.predictions
    labels = pred.label_ids
    pred_ids = np.argmax(logits, axis=-1)

    gold_spans_list = []
    pred_spans_list = []
    all_gold_binary = []
    all_final_binary = []
    all_gold_pairwise = []
    all_pred_pairwise = []
    gold_binary_sequences = []
    pred_binary_sequences = []
    pairwise_example_metrics = []
    predicted_spans_payload = []
    for i, (raw_ex, pred_seq) in enumerate(zip(raw_ds, pred_ids)):
        tokens = list(raw_ex["tokens"])
        gold_word_labels = raw_ex["labels"]
        gold_binary = [0 if str(label).strip() == "O" else 1 for label in gold_word_labels]
        entity_probabilities = _short_word_entity_probabilities(
            logits[i],
            eval_ds[i]["word_starts"],
            entity_id=int(label2id["ENTITY"]),
            num_words=len(tokens),
        )
        raw_binary = [
            1 if probability >= float(entity_probability_threshold) else 0
            for probability in entity_probabilities
        ]
        pruned_binary = _prune_short_singletons(
            raw_binary,
            entity_probabilities,
            singleton_keep_threshold=singleton_keep_threshold,
        )
        if apply_citation_filter:
            final_binary, raw_pred_spans, kept_spans, removed_spans = _apply_short_citation_filter(
                tokens,
                pruned_binary,
            )
        else:
            final_binary = pruned_binary
            raw_pred_spans = binary_spans_from_word_labels(
                tokens,
                ["ENTITY" if value == 1 else "O" for value in pruned_binary],
            )
            kept_spans = raw_pred_spans
            removed_spans = []

        pred_word_labels = ["ENTITY" if value == 1 else "O" for value in final_binary]
        gold_spans = binary_spans_from_word_labels(tokens, gold_word_labels)
        pred_spans = binary_spans_from_word_labels(tokens, pred_word_labels)
        gold_spans_list.append([s["text"] for s in gold_spans])
        pred_spans_list.append([s["text"] for s in pred_spans])

        gold_pairs = _pairwise_binary_labels(gold_binary)
        pred_pairs = _pairwise_binary_labels(final_binary)
        all_gold_binary.extend(gold_binary)
        all_final_binary.extend(final_binary)
        all_gold_pairwise.extend(gold_pairs)
        all_pred_pairwise.extend(pred_pairs)
        gold_binary_sequences.append(gold_binary)
        pred_binary_sequences.append(final_binary)
        example_pairwise = _binary_prf_from_arrays(gold_pairs, pred_pairs)
        pairwise_example_metrics.append(example_pairwise)

        predicted_spans_payload.append(
            {
                "doc_index": raw_ex.get("doc_index"),
                "sent_index": raw_ex.get("sent_index"),
                "text": raw_ex.get("text"),
                "tokens": tokens,
                "gold_labels": gold_word_labels,
                "pred_labels": pred_word_labels,
                "gold_binary": gold_binary,
                "entity_probabilities": entity_probabilities,
                "raw_pred_binary_thresholded": raw_binary,
                "pred_binary_after_singleton_prune": pruned_binary,
                "final_pred_binary": final_binary,
                "gold_spans": gold_spans,
                "pred_spans": pred_spans,
                "raw_pred_spans_after_prune": raw_pred_spans,
                "kept_spans_no_citation": kept_spans,
                "removed_spans_with_citation": removed_spans,
                "pairwise_f1_binary_final_example": example_pairwise,
                "pairwise_cohen_kappa_binary_final_example": safe_cohen_kappa(gold_pairs, pred_pairs),
                "postprocessing": {
                    "entity_probability_threshold": float(entity_probability_threshold),
                    "singleton_keep_threshold": float(singleton_keep_threshold),
                    "citation_filter": bool(apply_citation_filter),
                },
                "aligned_gold_labels": [
                    id2label[int(label)] for label in labels[i] if int(label) != -100
                ],
                "aligned_pred_labels": [
                    id2label[int(prediction)]
                    for prediction, label in zip(pred_seq, labels[i])
                    if int(label) != -100
                ],
            }
        )

    flat_p, flat_l = flatten_valid(pred_ids, labels)
    entity_id = int(label2id["ENTITY"])
    gold_binary_array = np.asarray(all_gold_binary, dtype=np.int64)
    final_binary_array = np.asarray(all_final_binary, dtype=np.int64)
    gold_pairwise_array = np.asarray(all_gold_pairwise, dtype=np.int64)
    pred_pairwise_array = np.asarray(all_pred_pairwise, dtype=np.int64)
    macro_kappa = macro_cohen_kappa_from_sequences(
        gold_binary_sequences,
        pred_binary_sequences,
        [0, 1],
        ignore_index=None,
    )
    pairwise_macro = {
        name: float(np.mean([metric[name] for metric in pairwise_example_metrics]))
        if pairwise_example_metrics else 0.0
        for name in ("precision", "recall", "f1")
    }
    raw_argmax_kappa = safe_cohen_kappa(flat_l, flat_p)

    result = {
        "task": "short_unsupported_claim_token_classification",
        "model_name_or_path": model_name_or_path,
        "data_path": str(data_path),
        "category_name": category_name,
        "max_span_words": int(max_span_words),
        "num_eval_sequences": int(labels.shape[0]),
        "num_eval_tokens_scored": int(gold_binary_array.size),
        "postprocessing": {
            "entity_probability_threshold": float(entity_probability_threshold),
            "singleton_keep_threshold": float(singleton_keep_threshold),
            "citation_filter": bool(apply_citation_filter),
        },
        "token_overlap_f1_entity_vs_O": _binary_prf_from_arrays(
            gold_binary_array,
            final_binary_array,
        ),
        "exact_match": {
            "token_accuracy": float((final_binary_array == gold_binary_array).mean())
            if gold_binary_array.size else 0.0,
            "sequence_exact_match": float(np.mean([
                gold == pred for gold, pred in zip(gold_binary_sequences, pred_binary_sequences)
            ])) if gold_binary_sequences else 0.0,
        },
        "cohens_kappa_token_level": safe_cohen_kappa(gold_binary_array, final_binary_array),
        "cohens_kappa_token_level_macro": macro_kappa,
        "pairwise_span_f1": pairwise_span_f1(gold_spans_list, pred_spans_list, iou_threshold=iou_threshold),
        "pairwise_binary_agreement": {
            "pairwise_micro": _binary_prf_from_arrays(gold_pairwise_array, pred_pairwise_array),
            "pairwise_macro": pairwise_macro,
            "cohen_kappa": safe_cohen_kappa(gold_pairwise_array, pred_pairwise_array),
            "num_pairwise_items_scored": int(gold_pairwise_array.size),
        },
        "confusion_matrix": {
            "labels": ["O", "ENTITY"],
            "matrix": confusion_matrix(gold_binary_array, final_binary_array, labels=[0, 1]).tolist()
            if gold_binary_array.size else [],
        },
        "raw_argmax_subtoken_metrics": {
            "num_tokens_scored": int(flat_l.size),
            "token_overlap_f1_entity_vs_O": token_overlap_f1_binary_from_ids(
                flat_p,
                flat_l,
                entity_id=entity_id,
            ),
            "cohen_kappa_token_level": raw_argmax_kappa,
            "confusion_matrix": {
                "labels": ["O", "ENTITY"],
                "matrix": confusion_matrix(flat_l, flat_p, labels=[0, 1]).tolist()
                if flat_l.size else [],
            },
        },
    }

    if output_dir is not None:
        out = Path(output_dir)
        _write_json(out / "short_unsupp_eval_predicted_spans.json", predicted_spans_payload)
        _write_json(out / "short_unsupp_eval_metrics.json", result)

    return result
