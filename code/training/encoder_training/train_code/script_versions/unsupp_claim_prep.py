import spacy
import json
import re
import random

nlp = spacy.load("en_core_web_sm")


def split_sentences_with_offsets(text):
    doc = nlp(text)
    out = []
    for sent in doc.sents:
        sent_text = text[sent.start_char:sent.end_char]
        if sent_text.strip() == "":
            continue
        out.append({"text": sent_text, "start": sent.start_char, "end": sent.end_char})
    return out


def iter_token_spans(text):
    for m in re.finditer(r"\S+", text):
        yield m.group(0), m.start(), m.end()


def tag_tokens(tokens_with_offsets, span_start, span_end, label_name):
    labels = ["O"] * len(tokens_with_offsets)

    hit_indices = [
        i for i, (_, tok_start, tok_end) in enumerate(tokens_with_offsets)
        if not (tok_end <= span_start or tok_start >= span_end)
    ]

    if not hit_indices:
        return labels

    labels[hit_indices[0]] = f"B-{label_name}"
    for i in hit_indices[1:]:
        labels[i] = f"I-{label_name}"

    return labels


def span_word_len(span_text):
    return len(re.findall(r"\S+", span_text or ""))


def split_items_by_span_length(items, min_words=1, max_words=5):
    """
    Returns: (short_span_items, other_items)
    - short_span_items: items where spans are filtered to ONLY those with word_len in [min_words, max_words]
    - other_items: items where spans are filtered to ONLY those with word_len > max_words (or < min_words)
    Items with no remaining spans are dropped.
    """
    short_items = []
    other_items = []

    for sample in items:
        doc_text = sample["document"]
        spans = sample.get("spans", [])

        short_spans = []
        other_spans = []

        for sp in spans:
            txt = sp.get("span") or sp.get("text") or doc_text[sp["start"]:sp["end"]]
            wl = span_word_len(txt)

            if min_words <= wl <= max_words:
                short_spans.append(sp)
            else:
                other_spans.append(sp)

        if short_spans:
            short_items.append({"document": doc_text, "spans": short_spans})
        if other_spans:
            other_items.append({"document": doc_text, "spans": other_spans})

    return short_items, other_items


def process_items(items, label_name="UNSUPPORTED_CLAIM", neg_ratio=1.0, seed=13):
    """
    Returns positives (sentences overlapping spans) + sampled negatives from the same document
    (sentences with NO overlap with any span).
    """
    rng = random.Random(seed)
    output = []

    for sample in items:
        doc_text = sample["document"]
        sentences = split_sentences_with_offsets(doc_text)

        sent_objs = []
        for sent in sentences:
            sent_text = sent["text"]
            sent_start = sent["start"]
            sent_end = sent["end"]

            tokens = list(iter_token_spans(sent_text))

            sent_objs.append({
                "text": sent_text,
                "start": sent_start,
                "end": sent_end,
                "tokens": tokens,
                "labels": ["O"] * len(tokens),
                "has_span": False,
            })

        # tag positives
        for span in sample["spans"]:
            span_start_doc = span["start"]
            span_end_doc = span["end"]

            for sent_obj in sent_objs:
                sent_start_doc = sent_obj["start"]
                sent_end_doc = sent_obj["end"]

                if span_end_doc <= sent_start_doc or span_start_doc >= sent_end_doc:
                    continue

                local_start = max(0, span_start_doc - sent_start_doc)
                local_end = min(len(sent_obj["text"]), span_end_doc - sent_start_doc)

                new_labels = tag_tokens(
                    sent_obj["tokens"],
                    local_start,
                    local_end,
                    label_name
                )

                for i, label in enumerate(new_labels):
                    if label != "O":
                        sent_obj["labels"][i] = label

                sent_obj["has_span"] = True

        positives = []
        negatives = []

        for sent_obj in sent_objs:
            ex = {
                "tokens": [t[0] for t in sent_obj["tokens"]],
                "labels": sent_obj["labels"],
            }
            if sent_obj["has_span"]:
                positives.append(ex)
            else:
                negatives.append(ex)

        output.extend(positives)

        target_neg = int(round(len(positives) * float(neg_ratio)))
        if target_neg > 0 and negatives:
            if target_neg >= len(negatives):
                output.extend(negatives)
            else:
                output.extend(rng.sample(negatives, target_neg))

    return output

synth_path = "../../../../synthetic_sampling/samples/synthetic_samples_complete.json"
real_path = "../../../../synthetic_sampling/samples/combined_real_data.json"

# split synth into two train sets:
#   - short spans (1-5 words)
#   - long spans (6+ words)
synth_out_short = "../../../data/unsupported_claim/unsupported_claim_ner_train_short_1_5w.json"
synth_out_long = "../../../data/unsupported_claim/unsupported_claim_ner_train_long_6plus.json"

real_out_short = "../../../data/unsupported_claim/unsupported_claim_ner_eval_short_1_5w.json"
real_out_long = "../../../data/unsupported_claim/unsupported_claim_ner_eval_long_6plus.json"

with open(synth_path, "r", encoding="utf-8") as f:
    span_data_all = json.load(f)

with open(real_path, "r", encoding="utf-8") as f:
    real_data_all = json.load(f)

synth_items = span_data_all["Unsupported claim"]
real_items = real_data_all["Unsupported claim"]

LABEL_NAME = "UNSUPPORTED_CLAIM"

# split items by span word length ---
synth_items_short, synth_items_long = split_items_by_span_length(synth_items, min_words=1, max_words=5)
real_items_short, real_items_long = split_items_by_span_length(real_items, min_words=1, max_words=5)

# Build datasets (includes negatives per doc)
synth_sents_short = process_items(synth_items_short, label_name=LABEL_NAME, neg_ratio=0.1, seed=42)
synth_sents_long = process_items(synth_items_long, label_name=LABEL_NAME, neg_ratio=0.8, seed=42)

real_sents_short = process_items(real_items_short, label_name=LABEL_NAME, neg_ratio=0.1, seed=42)
real_sents_long = process_items(real_items_long, label_name=LABEL_NAME, neg_ratio=0.8, seed=42)

with open(synth_out_short, "w", encoding="utf-8") as f:
    json.dump(synth_sents_short, f, ensure_ascii=False, indent=2)

with open(synth_out_long, "w", encoding="utf-8") as f:
    json.dump(synth_sents_long, f, ensure_ascii=False, indent=2)

with open(real_out_long, "w", encoding="utf-8") as f:
    json.dump(real_sents_long, f, ensure_ascii=False, indent=2)

with open(real_out_short, "w", encoding="utf-8") as f:
    json.dump(real_sents_short, f, ensure_ascii=False, indent=2)

print("Short-span synth examples:", len(synth_items_short), "-> examples:", len(synth_sents_short))
print("Long-span synth examples:", len(synth_items_long), "-> examples:", len(synth_sents_long))
print("Long-span Eval examples   :", len(real_sents_long))
print("Short-span Eval examples   :", len(real_sents_short))