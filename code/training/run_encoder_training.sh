#!/usr/bin/env bash
set -uo pipefail

CONFIGS=(
  "configs/config_xlmr.json"
  "configs/config_scibert.json"
  "configs/config_modernbert.json"
)

for CONFIG in "${CONFIGS[@]}"; do

  if [ ! -f "$CONFIG" ]; then
    echo "Config file not found: $CONFIG — skipping"
    continue
  fi

  echo
  echo "========================================="
  echo "Processing config: $CONFIG"
  echo "========================================="

  mapfile -t CAT_OBJS < <(jq -c '.categories[]' "$CONFIG")

  if [ "${#CAT_OBJS[@]}" -eq 0 ]; then
    echo "Warning: no categories found in $CONFIG — skipping"
    continue
  fi

  for CAT_OBJ in "${CAT_OBJS[@]}"; do
    CAT_NAME=$(jq -r '.name' <<<"$CAT_OBJ")
    TRAIN_SCRIPT=$(jq -r '.train_script' <<<"$CAT_OBJ")
    RUN_NAME=$(jq -r '.run_name' <<<"$CAT_OBJ")

    I_WEIGHT_GRID=$(jq -r '.i_weight_grid // "5.0,6.0,7.0,8.0"' <<<"$CAT_OBJ")
    B_WEIGHT_GRID=$(jq -r '.b_weight_grid // "3.0"' <<<"$CAT_OBJ")
    O_WEIGHT_GRID=$(jq -r '.o_weight_grid // "0.5,0.75,1.0"' <<<"$CAT_OBJ")
    MIN_SPAN_LEN=$(jq -r '.min_pred_span_len // "5"' <<<"$CAT_OBJ")
    COVERAGE_GRID=$(jq -r '.coverage_threshold_grid // "0.3"' <<<"$CAT_OBJ")
    WINDOW_TOKENS=$(jq -r '.window_tokens // "160"' <<<"$CAT_OBJ")
    STRIDE_TOKENS=$(jq -r '.stride_tokens // "80"' <<<"$CAT_OBJ")

    if [ -z "$CAT_NAME" ] || [ "$CAT_NAME" = "null" ]; then
      echo "ERROR: category missing .name in $CONFIG"
      exit 1
    fi
    
    if [ -z "$TRAIN_SCRIPT" ] || [ "$TRAIN_SCRIPT" = "null" ]; then
      echo "ERROR: category '$CAT_NAME' missing .train_script in $CONFIG"
      exit 1
    fi

    echo "TRAIN_SCRIPT: $TRAIN_SCRIPT"
    if [ ! -f "$TRAIN_SCRIPT" ]; then
      echo "Train script not found for category '$CAT_NAME': $TRAIN_SCRIPT"
      exit 1
    fi

    echo "========================================="
    echo "Training category: $CAT_NAME"
    echo "  config: $CONFIG"
    echo "  script: $TRAIN_SCRIPT"
    echo "  run_name: $RUN_NAME"
    echo "========================================="

    if [[ "$CAT_NAME" == "coherence" ]] || [[ "$CAT_NAME" == "lacks_synthesis" ]]; then
      echo "  (using window/stride args)"
      python "$TRAIN_SCRIPT" \
        --config "$CONFIG" \
        --category "$CAT_NAME" \
        --run_name "$RUN_NAME" \
        --coverage_threshold "$COVERAGE" \
        --window_tokens "$WINDOW_TOKENS" \
        --stride_tokens "$STRIDE_TOKENS"

    elif [[ "$RUN_NAME" == *"unsupported_short"* ]]; then
      echo "  (skipping weights/span_len args for short run)"
      python "$TRAIN_SCRIPT" \
        --config "$CONFIG" \
        --category "$CAT_NAME" \
        --run_name "$RUN_NAME"

    else
      echo "  (using weights/span_len args)"
      python "$TRAIN_SCRIPT" \
        --config "$CONFIG" \
        --category "$CAT_NAME" \
        --run_name "$RUN_NAME" \
        --o_weight_grid "$O_WEIGHT_GRID" \
        --i_weight_grid "$I_WEIGHT_GRID" \
        --b_weight_grid "$B_WEIGHT_GRID" \
        --coverage_threshold_grid "$COVERAGE_GRID" \
        --min_pred_span_len "$MIN_SPAN_LEN"
    fi

  done

done

echo
echo "All configs and categories finished."