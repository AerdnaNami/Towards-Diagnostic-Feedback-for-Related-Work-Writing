#!/bin/bash

# ----------------------------
# Configuration
# ----------------------------
SPAN_SCRIPT="span_macro_f1.py"
WINDOW_SCRIPT="window_macro_f1.py"
IOU_THRESHOLD=0.5
OUT_DIR="macro_f1_results"

# ----------------------------
# Categories to evaluate
# ----------------------------
CATEGORIES=(
    "unsupported_short"
)

# ----------------------------
# Models
# ----------------------------
MODELS=(
    "xlm_roberta"
    "scibert"
    "modernbert"
    "neobert"
)

# ----------------------------
# Run evaluation
# ----------------------------
for CATEGORY in "${CATEGORIES[@]}"; do

    echo "========================================="
    echo "Processing category: $CATEGORY"
    echo "========================================="

    CATEGORY_OUT_DIR="${OUT_DIR}/${CATEGORY}"
    mkdir -p "$CATEGORY_OUT_DIR"

    for MODEL_NAME in "${MODELS[@]}"; do

        # Build eval file path
        case "$MODEL_NAME" in
            scibert)
                    EVAL_FILE="scibert/4_${CATEGORY}_scibert/eval_predicted_spans.json"
                ;;
            xlm_roberta)
                    EVAL_FILE="xlm_roberta/4_${CATEGORY}_xlm-roberta-base/eval_predicted_spans.json"
                ;;
            modernbert)
                    EVAL_FILE="modernbert/4_${CATEGORY}_modernBERT/eval_predicted_spans.json"
                ;;
            neobert)
                    EVAL_FILE="neobert/4_${CATEGORY}_neoBERT/eval_predicted_spans.json"
                ;;
        esac

        if [ ! -f "$EVAL_FILE" ]; then
            echo "WARNING: eval file not found for $MODEL_NAME: $EVAL_FILE — skipping."
            continue
        fi

        OUT_PATH="${CATEGORY_OUT_DIR}/${MODEL_NAME}_pairwise_f1.json"

        echo "Evaluating: $MODEL_NAME"
        echo "  Category:   $CATEGORY"
        echo "  Eval file:  $EVAL_FILE"
        echo "  Output:     $OUT_PATH"

        if [[ "$CATEGORY" == "coherence" || "$CATEGORY" == "lacks_synthesis" ]]; then
            python "$WINDOW_SCRIPT" \
                --model_name "$MODEL_NAME" \
                --eval_file "$EVAL_FILE" \
                --out_path "$OUT_PATH"
        else
            python "$SPAN_SCRIPT" \
                --model_name "$MODEL_NAME" \
                --eval_file "$EVAL_FILE" \
                --out_path "$OUT_PATH" 
        fi

        echo "Done: $MODEL_NAME"
        echo "---"
    done

done

echo "All evaluations complete. Results saved to $OUT_DIR/"