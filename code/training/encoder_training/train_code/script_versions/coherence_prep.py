import json
from tqdm import tqdm


def overlaps(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return not (a_end <= b_start or b_end <= a_start)


def split_paragraphs_with_offsets(text: str):
    paragraphs = []
    parts = text.split("\n\n") # seperate document into paragraphs as input to the model
    cursor = 0
    for part in parts:
        part = part.strip()
        if not part:
            continue

        start = text.find(part, cursor)
        end = start + len(part)

        paragraphs.append({
            "text": part,
            "start": start,
            "end": end
        })

        cursor = end

    return paragraphs

synth_path = "../../../../synthetic_sampling/samples/synthetic_samples_complete.json"
real_path = "../../../../synthetic_sampling/samples/combined_real_data.json"
synth_out_path = "../../../data/coherence/coherence_ner_train.json"
real_out_path = "../../../data/coherence/coherence_ner_eval.json"

with open(synth_path, "r", encoding="utf-8") as f:
    span_data_all = json.load(f)

with open(real_path, "r", encoding="utf-8") as f:
    real_data_all = json.load(f)

synth_coherence_paragraphs = []
real_coherence_paragraphs = []

for doc_idx, item in enumerate(tqdm(span_data_all["Coherence"], desc="Processing Coherence")):
    doc_text = item["document"]
    spans = item.get("spans", [])

    paragraphs = split_paragraphs_with_offsets(doc_text)

    for p_idx, para in enumerate(paragraphs):
        p_start = para["start"]
        p_end = para["end"]
        para_text = para["text"]

        for sp in spans:
            sp_start = int(sp["start"])
            sp_end = int(sp["end"])

            if overlaps(sp_start, sp_end, p_start, p_end):

                # Clip span to paragraph
                clipped_start = max(sp_start, p_start)
                clipped_end = min(sp_end, p_end)

                # Convert to paragraph-local indices
                local_start = clipped_start - p_start
                local_end = clipped_end - p_start

                synth_coherence_paragraphs.append({
                    "doc_index": doc_idx,
                    "paragraph": para_text,
                    "span": para_text[local_start:local_end],
                    "start": local_start,
                    "end": local_end,
                })



for doc_idx, item in enumerate(tqdm(real_data_all["Coherence"], desc="Processing Coherence")):
    doc_text = item["document"]
    spans = item.get("spans", [])

    paragraphs = split_paragraphs_with_offsets(doc_text)

    for p_idx, para in enumerate(paragraphs):
        p_start = para["start"]
        p_end = para["end"]
        para_text = para["text"]

        for sp in spans:
            sp_start = int(sp["start"])
            sp_end = int(sp["end"])

            if overlaps(sp_start, sp_end, p_start, p_end):

                # Clip span to paragraph
                clipped_start = max(sp_start, p_start)
                clipped_end = min(sp_end, p_end)

                # Convert to paragraph-local indices
                local_start = clipped_start - p_start
                local_end = clipped_end - p_start

                real_coherence_paragraphs.append({
                    "doc_index": doc_idx,
                    "paragraph": para_text,
                    "span": para_text[local_start:local_end],
                    "start": local_start,
                    "end": local_end,
                })

print("Total synthetic coherence paragraph-span pairs:", len(synth_coherence_paragraphs))
print("Total real coherence paragraph-span pairs:", len(real_coherence_paragraphs))
