#!/usr/bin/env bash
set -uo pipefail

PAIRS=(
  "decoder_training/train_code/train_llama-3.1-8b.py configs/config_llama-3.1-8b.json"
)

run_pair () {
  local TRAIN_SCRIPT="$1"
  local CONFIG="$2"

  if [ ! -f "$TRAIN_SCRIPT" ]; then
    echo "Train script not found: $TRAIN_SCRIPT"
    exit 1
  fi

  if [ ! -f "$CONFIG" ]; then
    echo "Config file not found: $CONFIG"
    exit 1
  fi

  mapfile -t RUNS < <(jq -r '.categories[] | select(.run_name != null) | "\(.name)|\(.run_name)"' "$CONFIG")

  if [ ${#RUNS[@]} -eq 0 ]; then
    echo "No valid category entries with run_name found in $CONFIG"
    exit 1
  fi

  for ENTRY in "${RUNS[@]}"; do
    IFS="|" read -r CAT RUN_NAME <<< "$ENTRY"
    echo "Training ($TRAIN_SCRIPT | $CONFIG): category=$CAT run_name=$RUN_NAME"

    if ! python "$TRAIN_SCRIPT" \
        --config "$CONFIG" \
        --category "$CAT" \
        --run_name "$RUN_NAME"; then
      echo "WARNING: run failed for category=$CAT run_name=$RUN_NAME — continuing"
    fi

    echo "Finished: category=$CAT run_name=$RUN_NAME"
    echo "---"
  done
}

for PAIR in "${PAIRS[@]}"; do
  read -r SCRIPT CFG <<< "$PAIR"
  run_pair "$SCRIPT" "$CFG"
done