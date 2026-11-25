# citation_evaluation

Code and data for Towards Diagnostic Feedback for Related Work Writing: A Span-Level Benchmark, EACL 2027.

<!-- The Related Work section is central to academic writing, yet prior work has largely focused on generation and citation classification rather than the analysis of writing issues within the text. To support model evaluation in this setting, we formulate span-level error detection for Related Work sections and introduce a taxonomy of four categories: \textit{Format}, \textit{Unsupported Claim}, \textit{Coherence}, and \textit{Lacks Synthesis}. We construct an annotated dataset from {\tt NLPeer} with 137 documents and 395 spans, and generate 1,100 synthetic training spans using {\tt GPT-5}. Using this dataset, we benchmark prompting-based baselines and encoder models for detecting writing issue spans, alongside a citation-aware post-processing pipeline for short \textit{Unsupported Claim} spans. Encoder models outperform prompting approaches in span localization, while prompting methods tend to identify broad regions without precise boundaries. We also find that synthetic spans match real spans structurally but show reduced lexical diversity, affecting boundary alignment. Our work provides a benchmark dataset and empirical analysis for automated evaluation of Related Work writing quality, which we make publicly available. -->

We detect citation-related issues in the related-work sections of scientific papers. Each issue is framed as a span-tagging (BIO) task. There are four categories:

| Category | Granularity | Data folder |
|---|---|---|
| Coherence | paragraph | `code/training/data/coherence/` |
| Format | sentence | `code/training/data/format/` |
| Lacks synthesis | paragraph | `code/training/data/lacks_synthesis/` |
| Unsupported claim (short spans: 1–5 words; long spans: 6+ words) | sentence | `code/training/data/unsupported_claim/` |

Models are trained on a mix of real annotated examples and LLM-generated synthetic examples, and evaluated on real data.

## Repository structure

```
code/
├── data_collection/      # Extracting related-work sections from NLPeer (needs NLPeer, see below)
├── annotation_eval/      # Inter-annotator agreement (Krippendorff's alpha, pairwise F1)
├── synthetic_sampling/   # Synthetic data generation (generator–critic), audit, real-vs-synthetic analysis
│   └── samples/          # Final real and synthetic samples
├── baselines/            # Zero-shot and few-shot LLM baselines
├── correlation/          # Correlation analysis
└── training/
    ├── configs/          # One config per model (encoders and decoders)
    ├── data/             # Train/dev/eval splits per category (BIO format)
    ├── encoder_training/ # Data prep notebooks, training scripts, macro-F1 scripts and results
    ├── local_eval_runs/  # Evaluation outputs: metrics, predictions, confidence intervals and plots
    ├── run_encoder_training.sh
    ├── run_decoder_training.sh
    └── run_local_evaluation.sh
```

## Setup

```bash
pip install -r requirements.txt
```

The training scripts also use `jq`.

The data generation and baseline notebooks call the OpenAI API. Copy the example key files and fill them in. The real `keys.env` files are git-ignored.

```bash
cp code/keys.env.example code/keys.env                                        # API_KEY
cp code/synthetic_sampling/keys.env.example code/synthetic_sampling/keys.env  # API_KEY_2
```

## Data

### Included
- `code/training/data/`: train/dev/eval splits for each category.
- `code/synthetic_sampling/samples/`:
  - `synthetic_samples_complete.json`: the final synthetic samples.
  - `total_real_data.json` and `combined_real_data.json`: the real annotated examples.
  - `for_synth_sampling.json`, `for_eval.json` and `grouped_eval_data.json`: subsets used for generation and analysis.

The real examples are short excerpts from related-work sections of papers in the NLPeer corpus. If you use them, please also cite NLPeer (see below).

### Not included (download separately)
These datasets are not redistributed here. Get them from their original sources and follow their licenses.

- **NLPeer** ([Dycke et al., 2023](https://aclanthology.org/2023.acl-long.277)). It is only needed to re-run `code/data_collection/`. Download it from https://github.com/UKPLab/nlpeer and set `nlpeer_path` in `code/data_collection/nlpeer_data_collection.ipynb`.
- **SciERC** ([Luan et al., 2018](https://aclanthology.org/D18-1360)). It is used as extra training data for the unsupported-claim (short) models. Download it from http://nlp.cs.washington.edu/sciIE/ and place the raw `.txt`/`.ann` files in `code/training/data/sciERC/`. Then run `code/training/encoder_training/train_code/sciERC_prep.ipynb` to produce `sciERC_train.json` and `sciERC_eval.json` in `code/training/data/unsupported_claim/`.

## Models

| Type | Models |
|---|---|
| Encoders | `answerdotai/ModernBERT-base`, `chandar-lab/NeoBERT`, `allenai/scibert_scivocab_uncased`, `xlm-roberta-base` |
| Decoders | `meta-llama/Llama-3.1-8B`, `meta-llama/Llama-3.2-3B-Instruct`, `allenai/scitulu-7b`, `Uni-SMART/SciLitLLM` |
| LLM baselines and synthetic data generation | OpenAI GPT-5 (see notebooks) |

## Citation

If you use this code or data, please cite:

```bibtex
@inproceedings{TODO,
  title     = {TODO},
  author    = {TODO},
  booktitle = {TODO},
  year      = {TODO},
}
```

Please also cite the datasets this work builds on:

```bibtex
@inproceedings{dycke-etal-2023-nlpeer,
  title     = "{NLP}eer: A Unified Resource for the Computational Study of Peer Review",
  author    = "Dycke, Nils and Kuznetsov, Ilia and Gurevych, Iryna",
  booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
  year      = "2023",
  address   = "Toronto, Canada",
  publisher = "Association for Computational Linguistics",
  url       = "https://aclanthology.org/2023.acl-long.277",
  pages     = "5049--5073",
}

@inproceedings{luan-etal-2018-multi,
  title     = "Multi-Task Identification of Entities, Relations, and Coreference for Scientific Knowledge Graph Construction",
  author    = "Luan, Yi and He, Luheng and Ostendorf, Mari and Hajishirzi, Hannaneh",
  booktitle = "Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing",
  year      = "2018",
  address   = "Brussels, Belgium",
  publisher = "Association for Computational Linguistics",
  url       = "https://aclanthology.org/D18-1360",
  pages     = "3219--3232",
}
```

## License

<!-- TODO: choose a license, e.g. MIT/Apache-2.0 for code, CC BY 4.0 for the synthetic data. The real excerpts follow the NLPeer source licenses. -->
TODO
