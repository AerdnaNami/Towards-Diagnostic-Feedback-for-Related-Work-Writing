#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATA_PATH=""
CATEGORY_NAME=""
TASK="all"
MODEL_SPECS_FILE=""
OUTPUT_ROOT="${SCRIPT_DIR}/local_eval_runs"
COVERAGE_THRESHOLD="0.5"
WINDOW_TOKENS="160"
STRIDE_TOKENS="80"
MAX_SPAN_WORDS="5"
IOU_THRESHOLD="0.5"
MIN_PRED_SPAN_LEN="1"
SHORT_ENTITY_THRESHOLD="0.35"
SHORT_SINGLETON_KEEP_THRESHOLD="0.85"
SHORT_CITATION_FILTER="1"
EVAL_BATCH_SIZE="4"
CATEGORY_EXPLICIT="0"
DATA_PATH_EXPLICIT="0"
COVERAGE_THRESHOLD_EXPLICIT="0"
WINDOW_TOKENS_EXPLICIT="0"
STRIDE_TOKENS_EXPLICIT="0"
IOU_THRESHOLD_EXPLICIT="0"
MIN_PRED_SPAN_LEN_EXPLICIT="0"
SHORT_ENTITY_THRESHOLD_EXPLICIT="0"
SHORT_SINGLETON_KEEP_THRESHOLD_EXPLICIT="0"

usage() {
  cat <<'EOF'
Usage:
  bash training/run_local_evaluation.sh --models-file /path/to/models.tsv [options]

Required model spec file format:
  Tab-separated with either:
    run_name<TAB>model_path<TAB>task<TAB>tokenizer_path<TAB>config_path
  or the legacy format:
    run_name<TAB>model_path<TAB>tokenizer_path<TAB>config_path

  In the new format, `task` is one of: sentence, paragraph, short, all.
  Only the first two columns are required in the legacy format.
  Use "-" for tokenizer_path or config_path to fall back to the model parent directory.
  Use "-" for task to fall back to the global --task value.

Example:
  short_xlmr	/home/me/checkpoints/short/model.safetensors	short	/home/me/checkpoints/short	/home/me/checkpoints/short
  para_modernbert	/home/me/checkpoints/paragraph_model	paragraph	-	-
  legacy_sentence	/home/me/checkpoints/sentence_model	/home/me/checkpoints/sentence_model	-

Options:
  --task sentence|paragraph|short|all
  --category unsupported_claim|coherence|format|lacks_synthesis
  --data-path /abs/path/to/prepared_eval.json
  --output-root /abs/path/to/output_dir
  --coverage-threshold 0.5
  --window-tokens 160
  --stride-tokens 80
  --max-span-words 5
  --iou-threshold 0.5
  --min-pred-span-len 1
  --short-entity-threshold 0.35
  --short-singleton-keep-threshold 0.85
  --disable-short-citation-filter
  --eval-batch-size 4

For each model, compatible defaults are read from final_eval_metrics.json in the
model directory. Passing the corresponding option explicitly overrides the report.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --models-file)
      MODEL_SPECS_FILE="$2"
      shift 2
      ;;
    --task)
      TASK="$2"
      shift 2
      ;;
    --category)
      CATEGORY_NAME="$2"
      CATEGORY_EXPLICIT="1"
      shift 2
      ;;
    --data-path)
      DATA_PATH="$2"
      DATA_PATH_EXPLICIT="1"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --coverage-threshold)
      COVERAGE_THRESHOLD="$2"
      COVERAGE_THRESHOLD_EXPLICIT="1"
      shift 2
      ;;
    --window-tokens)
      WINDOW_TOKENS="$2"
      WINDOW_TOKENS_EXPLICIT="1"
      shift 2
      ;;
    --stride-tokens)
      STRIDE_TOKENS="$2"
      STRIDE_TOKENS_EXPLICIT="1"
      shift 2
      ;;
    --max-span-words)
      MAX_SPAN_WORDS="$2"
      shift 2
      ;;
    --iou-threshold)
      IOU_THRESHOLD="$2"
      IOU_THRESHOLD_EXPLICIT="1"
      shift 2
      ;;
    --min-pred-span-len)
      MIN_PRED_SPAN_LEN="$2"
      MIN_PRED_SPAN_LEN_EXPLICIT="1"
      shift 2
      ;;
    --short-entity-threshold)
      SHORT_ENTITY_THRESHOLD="$2"
      SHORT_ENTITY_THRESHOLD_EXPLICIT="1"
      shift 2
      ;;
    --short-singleton-keep-threshold)
      SHORT_SINGLETON_KEEP_THRESHOLD="$2"
      SHORT_SINGLETON_KEEP_THRESHOLD_EXPLICIT="1"
      shift 2
      ;;
    --disable-short-citation-filter)
      SHORT_CITATION_FILTER="0"
      shift
      ;;
    --eval-batch-size)
      EVAL_BATCH_SIZE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

is_valid_task() {
  case "$1" in
    sentence|paragraph|short|all)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

resolve_spec_path() {
  local raw_path="$1"
  if [[ -z "${raw_path}" || "${raw_path}" == "-" ]]; then
    printf '%s\n' "${raw_path}"
    return 0
  fi

  if [[ "${raw_path}" = /* ]]; then
    printf '%s\n' "${raw_path}"
    return 0
  fi

  printf '%s\n' "${SCRIPT_DIR}/${raw_path}"
}

validate_optional_path() {
  local label="$1"
  local path_value="$2"
  local run_name="$3"

  if [[ -z "${path_value}" || "${path_value}" == "-" ]]; then
    return 0
  fi

  if [[ ! -e "${path_value}" ]]; then
    echo "${label} path not found for ${run_name}: ${path_value}" >&2
    exit 1
  fi
}

infer_category_key() {
  local run_name_lower="$1"
  local model_path_lower="$2"
  local haystack="${run_name_lower} ${model_path_lower}"

  case "${haystack}" in
    *coherence*)
      printf '%s\n' "coherence"
      ;;
    *lacks_synth*|*lacks_synthesis*)
      printf '%s\n' "lacks_synthesis"
      ;;
    *format*)
      printf '%s\n' "format"
      ;;
    *unsupp*|*unsupported*)
      printf '%s\n' "unsupported_claim"
      ;;
    *)
      printf '%s\n' ""
      ;;
  esac
}

default_eval_path_for() {
  local category_key="$1"
  local run_name_lower="$2"
  local task_name="$3"

  case "${category_key}" in
    coherence)
      printf '%s\n' "${SCRIPT_DIR}/data/coherence/coherence_ner_eval.json"
      ;;
    lacks_synthesis)
      printf '%s\n' "${SCRIPT_DIR}/data/lacks_synthesis/lacks_synthesis_ner_eval.json"
      ;;
    format)
      printf '%s\n' "${SCRIPT_DIR}/data/format/format_ner_eval.json"
      ;;
    unsupported_claim)
      if [[ "${task_name}" == "short" || "${run_name_lower}" == *short* ]]; then
        printf '%s\n' "${SCRIPT_DIR}/data/unsupported_claim/unsupported_claim_ner_eval_short_1_5w.json"
      elif [[ "${task_name}" == "sentence" || "${run_name_lower}" == *long* ]]; then
        printf '%s\n' "${SCRIPT_DIR}/data/unsupported_claim/unsupported_claim_ner_eval_long_6plus.json"
      else
        printf '%s\n' "${SCRIPT_DIR}/data/unsupported_claim/unsupported_claim_ner_eval.json"
      fi
      ;;
    *)
      printf '%s\n' ""
      ;;
  esac
}

if [[ -z "${MODEL_SPECS_FILE}" ]]; then
  echo "--models-file is required." >&2
  usage
  exit 1
fi

MODEL_SPECS_FILE="$(resolve_spec_path "${MODEL_SPECS_FILE}")"

if [[ ! -f "${MODEL_SPECS_FILE}" ]]; then
  echo "Models file not found: ${MODEL_SPECS_FILE}" >&2
  exit 1
fi

if ! [[ "${EVAL_BATCH_SIZE}" =~ ^[0-9]+$ ]] || [[ "${EVAL_BATCH_SIZE}" -lt 1 ]]; then
  echo "--eval-batch-size must be a positive integer. Got: ${EVAL_BATCH_SIZE}" >&2
  exit 1
fi

if [[ "${OUTPUT_ROOT}" != /* ]]; then
  OUTPUT_ROOT="${SCRIPT_DIR}/${OUTPUT_ROOT}"
fi

mkdir -p "${OUTPUT_ROOT}"

cd "${REPO_ROOT}"

while IFS=$'\t' read -r RUN_NAME MODEL_PATH COL3 COL4 COL5 EXTRA; do
  [[ -z "${RUN_NAME:-}" ]] && continue
  [[ "${RUN_NAME}" =~ ^# ]] && continue

  if [[ -n "${EXTRA:-}" ]]; then
    echo "Invalid line in ${MODEL_SPECS_FILE}: expected at most 5 tab-separated columns." >&2
    exit 1
  fi

  if is_valid_task "${COL3:-}"; then
    RUN_TASK="${COL3}"
    TOKENIZER_PATH="${COL4:--}"
    CONFIG_PATH="${COL5:--}"
  else
    RUN_TASK="${TASK}"
    TOKENIZER_PATH="${COL3:--}"
    CONFIG_PATH="${COL4:--}"
    if [[ -n "${COL5:-}" ]]; then
      echo "Invalid line in ${MODEL_SPECS_FILE}: 5-column rows must use run_name<TAB>model_path<TAB>task<TAB>tokenizer_path<TAB>config_path." >&2
      exit 1
    fi
  fi

  if [[ "${RUN_TASK}" == "-" || -z "${RUN_TASK}" ]]; then
    RUN_TASK="${TASK}"
  fi

  if ! is_valid_task "${RUN_TASK}"; then
    echo "Invalid task for ${RUN_NAME}: ${RUN_TASK}. Expected one of sentence, paragraph, short, all." >&2
    exit 1
  fi

  MODEL_PATH="$(resolve_spec_path "${MODEL_PATH}")"
  TOKENIZER_PATH="$(resolve_spec_path "${TOKENIZER_PATH}")"
  CONFIG_PATH="$(resolve_spec_path "${CONFIG_PATH}")"

  if [[ ! -e "${MODEL_PATH}" ]]; then
    echo "Model path not found for ${RUN_NAME}: ${MODEL_PATH}" >&2
    if [[ "${MODEL_PATH}" == *.safetensors ]]; then
      local_parent_dir="$(dirname "${MODEL_PATH}")"
      if [[ -d "${local_parent_dir}" ]]; then
        echo "A directory exists at ${local_parent_dir}, but the requested weights file is missing." >&2
        echo "If this is a Hugging Face checkpoint directory, point the models file at the directory instead of a missing .safetensors file." >&2
      fi
    fi
    exit 1
  fi

  validate_optional_path "Tokenizer" "${TOKENIZER_PATH}" "${RUN_NAME}"
  validate_optional_path "Config" "${CONFIG_PATH}" "${RUN_NAME}"

  RUN_NAME_LOWER="$(printf '%s' "${RUN_NAME}" | tr '[:upper:]' '[:lower:]')"
  MODEL_PATH_LOWER="$(printf '%s' "${MODEL_PATH}" | tr '[:upper:]' '[:lower:]')"

  RUN_CATEGORY="${CATEGORY_NAME}"
  if [[ "${CATEGORY_EXPLICIT}" != "1" ]]; then
    RUN_CATEGORY="$(infer_category_key "${RUN_NAME_LOWER}" "${MODEL_PATH_LOWER}")"
    if [[ -z "${RUN_CATEGORY}" ]]; then
      echo "Could not infer category for ${RUN_NAME}. Pass --category explicitly." >&2
      exit 1
    fi
  fi

  RUN_DATA_PATH="${DATA_PATH}"
  if [[ "${DATA_PATH_EXPLICIT}" != "1" ]]; then
    RUN_DATA_PATH="$(default_eval_path_for "${RUN_CATEGORY}" "${RUN_NAME_LOWER}" "${RUN_TASK}")"
    if [[ -z "${RUN_DATA_PATH}" ]]; then
      echo "Could not determine default eval data path for ${RUN_NAME} (category=${RUN_CATEGORY}, task=${RUN_TASK}). Pass --data-path explicitly." >&2
      exit 1
    fi
  else
    RUN_DATA_PATH="$(resolve_spec_path "${RUN_DATA_PATH}")"
  fi

  if [[ ! -f "${RUN_DATA_PATH}" ]]; then
    echo "Eval data path not found for ${RUN_NAME}: ${RUN_DATA_PATH}" >&2
    exit 1
  fi

  RUN_OUTPUT_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
  mkdir -p "${RUN_OUTPUT_DIR}"

  echo "Running ${RUN_NAME}"
  echo "  model_path=${MODEL_PATH}"
  echo "  tokenizer_path=${TOKENIZER_PATH}"
  echo "  config_path=${CONFIG_PATH}"
  echo "  task=${RUN_TASK}"
  echo "  category=${RUN_CATEGORY}"
  echo "  data_path=${RUN_DATA_PATH}"

  python - "$RUN_NAME" "$MODEL_PATH" "$TOKENIZER_PATH" "$CONFIG_PATH" "$RUN_TASK" "$RUN_DATA_PATH" "$RUN_CATEGORY" "$RUN_OUTPUT_DIR" "$COVERAGE_THRESHOLD" "$WINDOW_TOKENS" "$STRIDE_TOKENS" "$MAX_SPAN_WORDS" "$IOU_THRESHOLD" "$MIN_PRED_SPAN_LEN" "$EVAL_BATCH_SIZE" "$COVERAGE_THRESHOLD_EXPLICIT" "$WINDOW_TOKENS_EXPLICIT" "$STRIDE_TOKENS_EXPLICIT" "$IOU_THRESHOLD_EXPLICIT" "$MIN_PRED_SPAN_LEN_EXPLICIT" "$SHORT_ENTITY_THRESHOLD" "$SHORT_SINGLETON_KEEP_THRESHOLD" "$SHORT_CITATION_FILTER" "$SHORT_ENTITY_THRESHOLD_EXPLICIT" "$SHORT_SINGLETON_KEEP_THRESHOLD_EXPLICIT" <<'PY'
import json
import sys
from pathlib import Path

print("[runner] Python entrypoint started", flush=True)
try:
    import torch
    print(
        f"[runner] torch imported: version={torch.__version__} cuda_available={torch.cuda.is_available()} device_count={torch.cuda.device_count()}",
        flush=True,
    )
except Exception as exc:
    print(f"[runner] torch import failed: {exc!r}", flush=True)
    raise

print("[runner] Importing training.evaluate_models", flush=True)
from training.evaluate_models import (
    evaluate_paragraph_model,
    evaluate_sentence_model,
    evaluate_short_unsupp_claim_model,
)
print("[runner] Imported training.evaluate_models", flush=True)

(
    run_name,
    model_path,
    tokenizer_path,
    config_path,
    task,
    data_path,
    category_name,
    run_output_dir,
    coverage_threshold,
    window_tokens,
    stride_tokens,
    max_span_words,
    iou_threshold,
    min_pred_span_len,
    eval_batch_size,
    coverage_threshold_explicit,
    window_tokens_explicit,
    stride_tokens_explicit,
    iou_threshold_explicit,
    min_pred_span_len_explicit,
    short_entity_threshold,
    short_singleton_keep_threshold,
    short_citation_filter,
    short_entity_threshold_explicit,
    short_singleton_keep_threshold_explicit,
) = sys.argv[1:]

tokenizer_arg = None if tokenizer_path == "-" else tokenizer_path
config_arg = None if config_path == "-" else config_path

run_dir = Path(run_output_dir)
run_dir.mkdir(parents=True, exist_ok=True)


def find_training_report(path_value):
    model_source = Path(path_value)
    report_dir = model_source.parent if model_source.is_file() else model_source
    if report_dir.name.startswith("checkpoint-") or report_dir.name == "best_model":
        report_dir = report_dir.parent
    for report_name in ("final_eval_metrics.json", "final_eval_report.json"):
        candidate = report_dir / report_name
        if candidate.is_file():
            return candidate
    return None


training_report_path = find_training_report(model_path)
training_report = {}
if training_report_path is not None:
    with open(training_report_path, "r", encoding="utf-8") as f:
        training_report = json.load(f)

short_postprocess_report_path = None
short_postprocess_report = {}
model_source = Path(model_path)
model_report_dir = model_source.parent if model_source.is_file() else model_source
short_report_candidates = [model_report_dir / "eval_final_report.json"]
if training_report_path is not None:
    short_report_candidates.append(training_report_path.parent / "eval_final_report.json")
for candidate in short_report_candidates:
    if candidate.is_file():
        short_postprocess_report_path = candidate
        with open(candidate, "r", encoding="utf-8") as f:
            short_postprocess_report = json.load(f)
        break

if coverage_threshold_explicit != "1":
    report_coverage = training_report.get(
        "best_coverage_threshold",
        training_report.get("coverage_threshold"),
    )
    if report_coverage is not None:
        coverage_threshold = str(report_coverage)
if window_tokens_explicit != "1" and training_report.get("window_tokens") is not None:
    window_tokens = str(training_report["window_tokens"])
if stride_tokens_explicit != "1" and training_report.get("stride_tokens") is not None:
    stride_tokens = str(training_report["stride_tokens"])
if iou_threshold_explicit != "1" and training_report.get("iou_threshold") is not None:
    iou_threshold = str(training_report["iou_threshold"])
if min_pred_span_len_explicit != "1" and training_report.get("min_pred_span_len") is not None:
    min_pred_span_len = str(training_report["min_pred_span_len"])
short_thresholds = short_postprocess_report.get("thresholds", {})
if short_entity_threshold_explicit != "1" and short_thresholds.get("THRESH") is not None:
    short_entity_threshold = str(short_thresholds["THRESH"])
if (
    short_singleton_keep_threshold_explicit != "1"
    and short_thresholds.get("SINGLETON_KEEP_THRESH") is not None
):
    short_singleton_keep_threshold = str(short_thresholds["SINGLETON_KEEP_THRESH"])

print(
    "[runner] Effective settings: "
    f"coverage_threshold={coverage_threshold} window_tokens={window_tokens} "
    f"stride_tokens={stride_tokens} min_pred_span_len={min_pred_span_len} "
    f"iou_threshold={iou_threshold} "
    f"short_entity_threshold={short_entity_threshold} "
    f"short_singleton_keep_threshold={short_singleton_keep_threshold} "
    f"short_citation_filter={short_citation_filter} "
    f"training_report={training_report_path} "
    f"short_postprocess_report={short_postprocess_report_path}",
    flush=True,
)

results = {
    "run_name": run_name,
    "model_path": model_path,
    "tokenizer_path": tokenizer_arg,
    "config_path": config_arg,
    "task": task,
    "category_name": category_name,
    "data_path": data_path,
    "training_report_path": str(training_report_path) if training_report_path else None,
    "evaluation_settings": {
        "coverage_threshold": float(coverage_threshold),
        "window_tokens": int(window_tokens),
        "stride_tokens": int(stride_tokens),
        "max_span_words": int(max_span_words),
        "iou_threshold": float(iou_threshold),
        "min_pred_span_len": int(min_pred_span_len),
        "short_entity_threshold": float(short_entity_threshold),
        "short_singleton_keep_threshold": float(short_singleton_keep_threshold),
        "short_citation_filter": short_citation_filter == "1",
    },
}

if task in ("sentence", "all"):
    sentence_dir = run_dir / "sentence"
    results["sentence"] = evaluate_sentence_model(
        model_name_or_path=model_path,
        tokenizer_name_or_path=tokenizer_arg,
        config_name_or_path=config_arg,
        data_path=data_path,
        category_name=category_name,
        coverage_threshold=float(coverage_threshold),
        min_pred_span_len=int(min_pred_span_len),
        iou_threshold=float(iou_threshold),
        output_dir=sentence_dir,
        eval_batch_size=int(eval_batch_size),
    )

if task in ("paragraph", "all"):
    paragraph_dir = run_dir / "paragraph"
    results["paragraph"] = evaluate_paragraph_model(
        model_name_or_path=model_path,
        tokenizer_name_or_path=tokenizer_arg,
        config_name_or_path=config_arg,
        data_path=data_path,
        category_name=category_name,
        coverage_threshold=float(coverage_threshold),
        window_tokens=int(window_tokens),
        stride_tokens=int(stride_tokens),
        output_dir=paragraph_dir,
        eval_batch_size=int(eval_batch_size),
    )

if task in ("short", "all"):
    short_dir = run_dir / "short"
    results["short"] = evaluate_short_unsupp_claim_model(
        model_name_or_path=model_path,
        tokenizer_name_or_path=tokenizer_arg,
        config_name_or_path=config_arg,
        data_path=data_path,
        category_name=category_name,
        max_span_words=int(max_span_words),
        iou_threshold=float(iou_threshold),
        entity_probability_threshold=float(short_entity_threshold),
        singleton_keep_threshold=float(short_singleton_keep_threshold),
        apply_citation_filter=short_citation_filter == "1",
        output_dir=short_dir,
        eval_batch_size=int(eval_batch_size),
    )

summary_path = run_dir / "summary.json"
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"Saved summary to {summary_path}")
PY
done < "${MODEL_SPECS_FILE}"

echo "All evaluations finished. Output root: ${OUTPUT_ROOT}"
