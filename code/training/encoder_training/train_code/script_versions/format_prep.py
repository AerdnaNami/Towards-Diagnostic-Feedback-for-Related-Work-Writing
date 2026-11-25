import re
import spacy
from tqdm import tqdm

CITATION_PATTERN = re.compile(
    r"""
    (
        \[\s*\d+(?:\s*[-–]\s*\d+)?(?:\s*,\s*\d+)*\s*\]      # [12], [3,7], [4-8]
      |
        \((?:[A-Z][A-Za-z]+(?:\s+et al\.)?(?:\s*&\s*[A-Z][A-Za-z]+)?,\s*\d{4}
            (?:;\s*[A-Z][A-Za-z]+(?:\s+et al\.)?(?:\s*&\s*[A-Z][A-Za-z]+)?,\s*\d{4})*
        )\)                                                 # (Smith, 2020; Brown, 2019)
      |
        \([A-Z][A-Za-z]+\s+and\s+[A-Z][A-Za-z]+,\s*\d{4}\)  # (Smith and Jones, 2019)
    )
    """,
    re.VERBOSE,
)

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

def has_citation(sentence):
    return CITATION_PATTERN.search(sentence) is not None

def overlaps(a_start, a_end, b_start, b_end):
    return not (a_end <= b_start or b_end <= a_start)

def clip_span_to_sentence(span_start, span_end, sent_start, sent_end):
    return max(span_start, sent_start), min(span_end, sent_end)

def build_format_sentence_dataset(format_items, category_name = "Format") :
    out_rows = []

    for doc_idx, item in enumerate(tqdm(format_items, desc=f"Processing {category_name}")):
        doc_text = item.get("document", "") or ""
        issue_spans = item.get("spans", []) or []

        sentences = split_sentences_with_offsets(doc_text)

        for sent_idx, sent in enumerate(sentences):
            sent_text = sent["text"]
            sent_start = sent["start"]
            sent_end = sent["end"]

            overlapping_issue_spans = []
            for sp in issue_spans:
                sp_start = int(sp["start"])
                sp_end = int(sp["end"])

                if overlaps(sp_start, sp_end, sent_start, sent_end):
                    clipped_start, clipped_end = clip_span_to_sentence(sp_start, sp_end, sent_start, sent_end)
                    local_start = clipped_start - sent_start
                    local_end = clipped_end - sent_start

                    overlapping_issue_spans.append({
                        "doc_start": sp_start,
                        "doc_end": sp_end,
                        "sent_start": local_start,
                        "sent_end": local_end,
                        "text": sent_text[local_start:local_end],
                        "span_text_from_source": sp.get("span", ""),
                    })

            if overlapping_issue_spans:
                label = 1 
            else:
                if not has_citation(sent_text):
                    continue
                label = 0

            out_rows.append({
                "category": category_name,
                "doc_index": doc_idx,
                "sent_index": sent_idx,
                "label": label,
                "sentence": sent_text,
                "sentence_char_start_in_doc": sent_start,
                "sentence_char_end_in_doc": sent_end,
                "issue_spans_in_sentence": overlapping_issue_spans,
            })

    return out_rows

import json

synth_path = "../../../synthetic_sampling/samples/synthetic_samples_complete.json"
real_path = "../../../synthetic_sampling/samples/combined_real_data.json"
synth_out_path = "../../data/format/format_sentence_synth.json"
real_out_path = "../../data/format/format_sentence_real.json"

with open(synth_path, "r", encoding="utf-8") as f:
    span_data_all = json.load(f)

with open(real_path, "r", encoding="utf-8") as f:
    real_data_all = json.load(f)

synth_format = span_data_all["Format"]
real_format = real_data_all['Format']

synth_rows = build_format_sentence_dataset(synth_format, category_name="Format")
real_rows = build_format_sentence_dataset(real_format, category_name="Format")

print(f"SYNTH Total sentence samples kept (citation-only): {len(synth_rows)}")
print(f"Positives: {sum(r['label'] for r in synth_rows)} | Negatives: {sum(1-r['label'] for r in synth_rows)}")

print(f"REAL Total sentence samples kept (citation-only): {len(synth_rows)}")
print(f"Positives: {sum(r['label'] for r in real_rows)} | Negatives: {sum(1-r['label'] for r in real_rows)}")


with open(synth_out_path, "w", encoding="utf-8") as f:
    json.dump(synth_rows, f, indent=2, ensure_ascii=False)

with open(real_out_path, "w", encoding="utf-8") as f:
    json.dump(real_rows, f, indent=2, ensure_ascii=False)

print(f"Wrote: {synth_out_path} and {real_out_path}")