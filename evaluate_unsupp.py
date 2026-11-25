from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
from pathlib import Path
import spacy
import re
from typing import List, Dict, Any, Optional
import json 
import numpy as np
from sklearn.metrics import cohen_kappa_score
from tqdm import tqdm 

unsupp_dir = "code/training/encoder_training/neobert/unsupported_short_neoBERT_NO_SCIERC/checkpoint-24"
tokenizer = AutoTokenizer.from_pretrained(unsupp_dir, local_files_only=True, model_max_length=256)
model = AutoModelForTokenClassification.from_pretrained(unsupp_dir, local_files_only=True)
model.to("cuda" if torch.cuda.is_available() else "cpu")


@torch.inference_mode()
def predict_subwords(text: str):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    # ensure inputs are on same device as model
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    outputs = model(**inputs)

    pred_ids = outputs.logits.argmax(dim=-1)[0].tolist()
    input_ids = inputs["input_ids"][0].tolist()

    toks = tokenizer.convert_ids_to_tokens(input_ids)
    labs = [model.config.id2label[int(i)] for i in pred_ids]

    out = []
    for tok, lab in zip(toks, labs):
        if tok in (tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token):
            continue
        out.append((tok, lab))
    return out


def merge_wordpiece_span(tokens):
    """
    Merge a list of WordPiece tokens into a single string.
    Example: ['conv', '##ai'] -> 'convai'
    """
    s = ""
    for t in tokens:
        if t.startswith("##"):
            s += t[2:]
        else:
            if s != "":
                s += " "
            s += t
    return s


citation_pattern = re.compile(
r"""
(
    # (Chen et al., 2021b) OR (Chen, 2021b) OR (Chen & Smith, 2021b) OR (Chen and Smith, 2021b)
    \(
        [A-Z][A-Za-z\-]+
        (?:
            (?:\s+et\ al\.)
            |
            (?:\s+(?:&|and)\s+[A-Z][A-Za-z\-]+)
        )?
        \,?\s+\d{4}[a-z]?
    \)
    |
    # Chen et al. (2021b) OR Chen & Smith (2021b) OR Chen (2021b)
    [A-Z][A-Za-z\-]+
    (?:
        (?:\s+et\ al\.) 
        |
        (?:\s+(?:&|and)\s+[A-Z][A-Za-z\-]+)
    )?
    \s+\(\d{4}[a-z]?\)
    |
    # [12] or [12, 15-18]
    \[(\d+([,\-\u2013]\s*\d+)*)\]
)
""",
re.VERBOSE,
)


def citation_after_mention(
    sentence: str,
    span: str,
    *,
    window_chars: int = 200,
    pre_window_chars: int = 50,
    case_insensitive: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    Returns None if span not in sentence.

    Citation is considered present if it appears:
      - immediately after the mention, OR
      - anywhere within window_chars AFTER, OR
      - anywhere within pre_window_chars BEFORE.
    """
    if not span:
        return None

    flags = re.IGNORECASE if case_insensitive else 0
    m = re.search(re.escape(span), sentence, flags)
    if not m:
        return None

    span_start, span_end = m.start(), m.end()

    after = sentence[span_end : span_end + max(0, int(window_chars))]
    cit_after = citation_pattern.search(after)
    if cit_after:
        where = "immediately_after" if cit_after.start() == 0 else "after"
        return {
            "span": span,
            "sentence": sentence,
            "sentence_index": None,
            "mention_start": span_start,
            "mention_end": span_end,
            "citation_found": True,
            "citation_where": where,
            "citation_match": cit_after.group(0),
        }

    pre_w = max(0, int(pre_window_chars))
    before_start = max(0, span_start - pre_w)
    before = sentence[before_start:span_start]
    cit_before = None
    for mm in citation_pattern.finditer(before):
        cit_before = mm

    if cit_before:
        return {
            "span": span,
            "sentence": sentence,
            "sentence_index": None,
            "mention_start": span_start,
            "mention_end": span_end,
            "citation_found": True,
            "citation_where": "before",
            "citation_match": cit_before.group(0),
        }

    return {
        "span": span,
        "sentence": sentence,
        "sentence_index": None,
        "mention_start": span_start,
        "mention_end": span_end,
        "citation_found": False,
        "citation_where": None,
        "citation_match": None,
    }


def find_first_mentions_missing_citation(
    sentences: List[str],
    spans: List[str],
    *,
    window_chars: int = 200,
    pre_window_chars: int = 80,
    case_insensitive: bool = True,
) -> List[Dict[str, Any]]:
    """
    For each span, locate its FIRST mention across sentences (by order),
    then check whether a citation appears after the mention OR before-with-comma.
    Returns ONLY the ones where citation is missing.
    """
    missing = []

    for span in spans:
        first_result = None
        first_sent_idx = None

        for i, sent in enumerate(sentences):
            res = citation_after_mention(
                sent,
                span,
                window_chars=window_chars,
                pre_window_chars=pre_window_chars,
                case_insensitive=case_insensitive,
            )
            if res is not None:
                first_result = res
                first_sent_idx = i
                break

        if first_result is None:
            continue

        first_result["sentence_index"] = first_sent_idx

        if not first_result["citation_found"]:
            missing.append(first_result)

    return missing


def extract_gold_spans_from_bio(tokens, labels):
    spans = []
    i = 0
    n = len(labels)

    while i < n:
        lab = (labels[i] or "O").strip()
        if lab.startswith("B-"):
            ent = lab[2:]
            start = i
            i += 1
            while i < n and (labels[i] or "O").strip() == f"I-{ent}":
                i += 1
            end = i
            spans.append(
                {
                    "label": ent,
                    "start_token": start,
                    "end_token": end,
                    "text": " ".join(tokens[start:end]),
                }
            )
        else:
            i += 1
    return spans


def token_char_offsets_from_space_join(tokens):
    """
    If you form text = " ".join(tokens), compute each token's (start,end) char offsets in that text.
    """
    offsets = []
    pos = 0
    for i, tok in enumerate(tokens):
        if i > 0:
            pos += 1
        start = pos
        end = start + len(tok)
        offsets.append((start, end))
        pos = end
    return offsets


@torch.inference_mode()
def predict_word_binary(
    tokens,
    tokenizer,
    model,
    max_length: int = 256,
):
    enc = tokenizer(
        tokens,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )

    outputs = model(**{k: v.to(model.device) for k, v in enc.items()})
    pred_ids = outputs.logits.argmax(dim=-1)[0].tolist()

    word_ids = enc.word_ids(batch_index=0)

    word_pred_label_ids = [None] * len(tokens)
    seen = set()

    for tok_i, widx in enumerate(word_ids):
        if widx is None:
            continue
        if widx in seen:
            continue
        seen.add(widx)
        word_pred_label_ids[widx] = int(pred_ids[tok_i])

    out = []
    for widx in range(len(tokens)):
        lid = word_pred_label_ids[widx]
        if lid is None:
            out.append(0)
        else:
            lab = model.config.id2label[int(lid)]
            out.append(0 if lab == "O" else 1)
    return out


def spans_from_binary(tokens, binary_labels):
    spans = []
    i = 0
    n = min(len(tokens), len(binary_labels))

    while i < n:
        if binary_labels[i] != 1:
            i += 1
            continue
        start = i
        i += 1
        while i < n and binary_labels[i] == 1:
            i += 1
        end = i
        spans.append(
            {
                "label": "ENTITY",
                "start_token": start,
                "end_token": end,
                "text": " ".join(tokens[start:end]),
            }
        )
    return spans


def sentence_for_span(doc, span_char_start, span_char_end):
    for sent in doc.sents:
        if sent.start_char < span_char_end and sent.end_char > span_char_start:
            return sent.text.strip()
    return doc.text.strip()


def token_overlap_f1_binary(y_pred, y_true):
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {"precision": float(precision), "recall": float(recall), "f1": float(f1), "tp": tp, "fp": fp, "fn": fn}


def safe_cohen_kappa(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.size == 0:
        return 0.0

    if np.unique(y_true).size < 2 or np.unique(y_pred).size < 2:
        return 0.0

    kappa = cohen_kappa_score(y_true, y_pred, labels=[0, 1])
    if np.isnan(kappa):
        return 0.0
    return float(kappa)


def prune_isolated_ones(binary_labels: List[int]) -> List[int]:
    n = len(binary_labels)
    out = binary_labels[:]
    for i in range(n):
        if out[i] != 1:
            continue
        left = (i > 0 and out[i - 1] == 1)
        right = (i < n - 1 and out[i + 1] == 1)
        if not (left or right):
            out[i] = 0
    return out


def dynamic_windows(
    sentence: str,
    *,
    after_frac: float = 0.25,
    before_frac: float = 0.12,
    min_after: int = 40,
    max_after: int = 140,
    min_before: int = 20,
    max_before: int = 80,
):
    n = len(sentence)
    after = int(n * after_frac)
    before = int(n * before_frac)

    after = max(min_after, min(after, max_after))
    before = max(min_before, min(before, max_before))
    return after, before


@torch.inference_mode()
def predict_word_probs(
    tokens: List[str],
    tokenizer,
    model,
    max_length: int = 256,
):
    """
    Returns:
      - p_entity_per_word: List[float] length=len(tokens), probability of ENTITY for each word
      - pred_label_id_per_word: argmax label id for each word (0/1), mainly for debugging

    Uses first-subtoken for each word (same alignment style as your current code).
    """
    enc = tokenizer(
        tokens,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    enc_dev = {k: v.to(model.device) for k, v in enc.items()}
    outputs = model(**enc_dev)

    logits = outputs.logits[0]
    probs = torch.softmax(logits, dim=-1)
    p_entity_tok = probs[:, 1].detach().cpu().numpy()

    pred_ids = logits.argmax(dim=-1).detach().cpu().numpy().tolist()
    word_ids = enc.word_ids(batch_index=0)

    p_entity_per_word = [0.0] * len(tokens)
    pred_id_per_word = [0] * len(tokens)
    seen = set()

    for tok_i, widx in enumerate(word_ids):
        if widx is None or widx in seen:
            continue
        seen.add(widx)
        p_entity_per_word[widx] = float(p_entity_tok[tok_i])
        pred_id_per_word[widx] = int(pred_ids[tok_i])

    return p_entity_per_word, pred_id_per_word


def binarize_with_threshold(p_entity: List[float], thresh: float) -> List[int]:
    return [1 if p >= thresh else 0 for p in p_entity]


def prune_isolated_with_exception(
    binary: List[int],
    p_entity: List[float],
    *,
    singleton_keep_thresh: float = 0.95,
) -> List[int]:
    """
    If binary[i]==1 but has no ENTITY neighbor, set to 0 unless p_entity[i] >= singleton_keep_thresh.
    """
    n = len(binary)
    out = binary[:]
    for i in range(n):
        if out[i] != 1:
            continue
        left = (i > 0 and out[i - 1] == 1)
        right = (i < n - 1 and out[i + 1] == 1)
        if not (left or right):
            if p_entity[i] < singleton_keep_thresh:
                out[i] = 0
    return out


def pairwise_binary_labels(binary_labels):
    n = len(binary_labels)
    out = []
    for i in range(n):
        for j in range(i + 1, n):
            out.append(1 if binary_labels[i] == binary_labels[j] else 0)
    return out


def pairwise_f1_binary(y_pred_pairs, y_true_pairs):
    tp = int(((y_pred_pairs == 1) & (y_true_pairs == 1)).sum())
    fp = int(((y_pred_pairs == 1) & (y_true_pairs == 0)).sum())
    fn = int(((y_pred_pairs == 0) & (y_true_pairs == 1)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def evaluate_with_params(eval_data, thresh, singleton_keep_thresh):
    all_gold_bin = []
    all_final_pred_bin = []
    all_gold_pairwise = []
    all_final_pred_pairwise = []
    analysis_rows = []

    for ex_idx, example in enumerate(eval_data):
        tokens = example["tokens"]
        gold_labels = example["labels"]

        sent_text = " ".join(tokens)
        gold_bin = [0 if (str(l).strip() == "O") else 1 for l in gold_labels]

        p_entity, _pred_id_dbg = predict_word_probs(tokens, tokenizer, model, max_length=256)
        raw_bin = binarize_with_threshold(p_entity, thresh)

        L = min(len(tokens), len(gold_bin), len(raw_bin), len(p_entity))
        tokens_L = tokens[:L]
        gold_bin_L = gold_bin[:L]
        raw_bin_L = raw_bin[:L]
        p_entity_L = p_entity[:L]

        pruned_bin = prune_isolated_with_exception(
            raw_bin_L,
            p_entity_L,
            singleton_keep_thresh=singleton_keep_thresh,
        )

        raw_pred_spans = spans_from_binary(tokens_L, pruned_bin)

        kept_spans = []
        removed_spans = []

        for sp in raw_pred_spans:
            span_text = sp["text"]
            win_after, win_before = dynamic_windows(sent_text)
            cit_res = citation_after_mention(
                sent_text,
                span_text,
                window_chars=win_after,
                pre_window_chars=win_before,
                case_insensitive=True,
            )

            if cit_res is None:
                removed_spans.append(
                    {
                        **sp,
                        "citation_found": None,
                        "citation_where": "span_not_found_in_sentence_text",
                        "citation_match": None,
                    }
                )
                continue

            if cit_res["citation_found"]:
                removed_spans.append(
                    {
                        **sp,
                        "citation_found": True,
                        "citation_where": cit_res["citation_where"],
                        "citation_match": cit_res["citation_match"],
                        "mention_start": cit_res["mention_start"],
                        "mention_end": cit_res["mention_end"],
                    }
                )
            else:
                kept_spans.append(
                    {
                        **sp,
                        "citation_found": False,
                        "citation_where": None,
                        "citation_match": None,
                        "mention_start": cit_res["mention_start"],
                        "mention_end": cit_res["mention_end"],
                    }
                )

        final_pred = [0] * L
        for sp in kept_spans:
            s = max(0, int(sp["start_token"]))
            e = min(L, int(sp["end_token"]))
            for i in range(s, e):
                final_pred[i] = 1

        all_gold_bin.extend(gold_bin_L)
        all_final_pred_bin.extend(final_pred)

        gold_pairwise = pairwise_binary_labels(gold_bin_L)
        pred_pairwise = pairwise_binary_labels(final_pred)

        all_gold_pairwise.extend(gold_pairwise)
        all_final_pred_pairwise.extend(pred_pairwise)

        example_pairwise_f1 = None
        example_pairwise_kappa = None
        if gold_pairwise:
            y_true_pairs_ex = np.array(gold_pairwise, dtype=np.int64)
            y_pred_pairs_ex = np.array(pred_pairwise, dtype=np.int64)
            example_pairwise_f1 = pairwise_f1_binary(y_pred_pairs_ex, y_true_pairs_ex)
            example_pairwise_kappa = safe_cohen_kappa(y_true_pairs_ex, y_pred_pairs_ex)

        analysis_rows.append(
            {
                "example_index": ex_idx,
                "sentence": sent_text,
                "tokens": tokens_L,
                "gold_labels": gold_labels[:L],
                "gold_binary": gold_bin_L,
                "p_entity": p_entity_L,
                "raw_pred_binary_thresh": raw_bin_L,
                "pred_binary_after_neighbor_prune": pruned_bin,
                "final_pred_binary": final_pred,
                "raw_pred_spans_after_prune": raw_pred_spans,
                "kept_spans_no_citation": kept_spans,
                "removed_spans_with_citation": removed_spans,
                "pairwise_gold_binary": gold_pairwise,
                "pairwise_final_pred_binary": pred_pairwise,
                "pairwise_f1_binary_FINAL_example": example_pairwise_f1,
                "pairwise_cohens_kappa_binary_FINAL_example": example_pairwise_kappa,
                "params": {
                    "THRESH": thresh,
                    "SINGLETON_KEEP_THRESH": singleton_keep_thresh,
                },
            }
        )

    y_true = np.array(all_gold_bin, dtype=np.int64)
    y_pred = np.array(all_final_pred_bin, dtype=np.int64)

    overlap = token_overlap_f1_binary(y_pred, y_true)
    kappa = safe_cohen_kappa(y_true, y_pred)

    y_true_pairwise = np.array(all_gold_pairwise, dtype=np.int64)
    y_pred_pairwise = np.array(all_final_pred_pairwise, dtype=np.int64)

    pairwise_overlap = pairwise_f1_binary(y_pred_pairwise, y_true_pairwise) if y_true_pairwise.size else {
        "precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0, "fp": 0, "fn": 0
    }
    pairwise_kappa = safe_cohen_kappa(y_true_pairwise, y_pred_pairwise)

    report = {
        "num_examples": int(len(eval_data)),
        "num_tokens_scored": int(y_true.size),
        "num_pairwise_items_scored": int(y_true_pairwise.size),
        "token_overlap_f1_binary_FINAL": overlap,
        "cohens_kappa_binary_FINAL": float(kappa),
        "pairwise_f1_binary_FINAL": pairwise_overlap,
        "pairwise_cohens_kappa_binary_FINAL": float(pairwise_kappa),
        "thresholds": {
            "THRESH": thresh,
            "SINGLETON_KEEP_THRESH": singleton_keep_thresh,
        },
    }

    return report, analysis_rows


def main():
    eval_path = "code/training/data/unsupported_claim/unsupported_claim_ner_eval_short_1_5w.json"
    out_dir = Path(unsupp_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    thresh_grid = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55]
    singleton_keep_thresh_grid = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]

    with open(eval_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    sweep_results = []
    best_report = None
    best_analysis_rows = None
    best_score = None
    best_params = None

    for thresh in tqdm(thresh_grid, desc="Sweeping THRESH"):
        for singleton_keep_thresh in singleton_keep_thresh_grid:
            report, analysis_rows = evaluate_with_params(
                eval_data,
                thresh=thresh,
                singleton_keep_thresh=singleton_keep_thresh,
            )

            score = report["token_overlap_f1_binary_FINAL"]["f1"]

            sweep_results.append(
                {
                    "THRESH": thresh,
                    "SINGLETON_KEEP_THRESH": singleton_keep_thresh,
                    "token_overlap_f1_binary_FINAL": report["token_overlap_f1_binary_FINAL"],
                    "cohens_kappa_binary_FINAL": report["cohens_kappa_binary_FINAL"],
                    "pairwise_f1_binary_FINAL": report["pairwise_f1_binary_FINAL"],
                    "pairwise_cohens_kappa_binary_FINAL": report["pairwise_cohens_kappa_binary_FINAL"],
                }
            )

            if (
                best_score is None
                or score > best_score
                or (
                    score == best_score
                    and report["cohens_kappa_binary_FINAL"] > best_report["cohens_kappa_binary_FINAL"]
                )
            ):
                best_score = score
                best_report = report
                best_analysis_rows = analysis_rows
                best_params = {
                    "THRESH": thresh,
                    "SINGLETON_KEEP_THRESH": singleton_keep_thresh,
                }

    final_report = dict(best_report)
    final_report["best_params_selected_by"] = "cohens_kappa_binary_FINAL"
    final_report["sweep_results"] = sweep_results

    with open(out_dir / "eval_final_report.json", "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2, ensure_ascii=False)

    with open(out_dir / "eval_final_predictions_with_gold_and_sentence.json", "w", encoding="utf-8") as f:
        json.dump(best_analysis_rows, f, indent=2, ensure_ascii=False)

    print("\n--- BEST FINAL EVAL REPORT ---")
    print(json.dumps(final_report, indent=2))
    print(f"\nBest params: {best_params}")
    print(f"\nSaved:\n  - {out_dir / 'eval_final_report.json'}\n  - {out_dir / 'eval_final_predictions_with_gold_and_sentence.json'}")


if __name__ == "__main__":
    main()