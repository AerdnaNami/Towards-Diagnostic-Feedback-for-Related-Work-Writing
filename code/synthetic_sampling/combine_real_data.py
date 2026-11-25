"""Combine total_real_data.json with nlpeer_v2 citation annotations.

Merges synthetic_sampling/samples/total_real_data.json with the nlpeer_v2
annotation spans found under
annotation_eval/annotated_data/citation_annotations/annotations/*, grouping
spans by (source file, label) and pulling the full document text from
data_collection/to_annotate/<file>. Output is written alongside
total_real_data.json in the same {category: [{"document", "spans"}]} format.
"""

import glob
import json
import os
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(BASE_DIR)

TOTAL_REAL_DATA_PATH = os.path.join(BASE_DIR, "samples", "total_real_data.json")
ANNOTATIONS_DIR = os.path.join(
    REPO_ROOT, "annotation_eval", "annotated_data", "citation_annotations", "annotations"
)
SOURCE_DOCS_DIR = os.path.join(REPO_ROOT, "data_collection", "to_annotate")
OUTPUT_PATH = os.path.join(BASE_DIR, "samples", "combined_real_data.json")


def load_nlpeer_spans():
    """Group nlpeer_v2 annotation spans by (source file, label)."""
    grouped = defaultdict(list)
    pattern = os.path.join(ANNOTATIONS_DIR, "*", "nlpeer_v2_*.txt.json")
    for annotation_path in sorted(glob.glob(pattern)):
        with open(annotation_path) as f:
            spans = json.load(f)
        for span in spans:
            key = (span["file"], span["label"])
            grouped[key].append({
                "span": span["text"],
                "start": span["start"],
                "end": span["end"],
            })
    return grouped


def build_real_entries(grouped_spans):
    """Turn grouped spans into {category: [{"document", "spans"}]} entries."""
    entries_by_category = defaultdict(list)
    doc_cache = {}
    for (source_file, label), spans in grouped_spans.items():
        if source_file not in doc_cache:
            doc_path = os.path.join(SOURCE_DOCS_DIR, source_file)
            with open(doc_path) as f:
                doc_cache[source_file] = f.read()
        entries_by_category[label].append({
            "document": doc_cache[source_file],
            "spans": spans,
        })
    return entries_by_category


def main():
    with open(TOTAL_REAL_DATA_PATH) as f:
        combined = json.load(f)

    grouped_spans = load_nlpeer_spans()
    new_entries = build_real_entries(grouped_spans)

    for category, entries in new_entries.items():
        combined.setdefault(category, []).extend(entries)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(combined, f, indent=2)

    for category, entries in combined.items():
        print(f"{category}: {len(entries)} samples")
    print(f"Wrote combined data to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
