#!/usr/bin/env bash

ENV_PATH=/media/home/hspark/venvs/OMNIDC/bin/activate
OUTPUT_ROOT="${1:-/media/home/zxguo/thin-obj-data-setup/data/omnidc_output_with_eval}"

PY_SCRIPT="/media/home/zxguo/thin-obj-data-setup/external/vision-lab-wrappers-datasets/src/depth_completion/run_depth_completion.py"

source "$ENV_PATH"
mkdir -p "$OUTPUT_ROOT"

shift || true 

if [ "$#" -eq 0 ]; then
  set -- /media/home/zxguo/thin-obj-data-setup/data/data_collection_realsense2
fi

for DATA_ROOT in "$@"; do
  echo "=========================================="
  echo "[DATA_ROOT] $DATA_ROOT"
  echo "=========================================="

  for SCENE_DIR in "$DATA_ROOT"/*; do
    [ -d "$SCENE_DIR" ] || continue
    SCENE_NAME="$(basename "$SCENE_DIR")"
    OUT_DIR="$OUTPUT_ROOT/$SCENE_NAME"

    DONE_MARKER="$OUT_DIR/depths"

    echo "------------------------------------------"
    echo "[SCENE] $SCENE_NAME"
    echo "  IN : $SCENE_DIR"
    echo "  OUT: $OUT_DIR"
    echo "------------------------------------------"

    # If there are no depths, run the model. If depths exist, skip model and go to eval logic.
    if [ -d "$DONE_MARKER" ] && [ "$(ls -A "$DONE_MARKER" 2>/dev/null | wc -l)" -gt 0 ]; then
      echo "[SKIP] $SCENE_NAME (found outputs in $DONE_MARKER)"
    else
      python "$PY_SCRIPT" \
        --rgb_dir "$SCENE_DIR/data/image" \
        --sparse_depth_dir "$SCENE_DIR/data/sparse_depth_1500" \
        --intrinsics_path "$SCENE_DIR/data/intrinsics.npy" \
        --out_dir "$OUT_DIR" \
        --device cuda \
        --run_model
    fi

    RESULTS_FILE="$OUT_DIR/eval/results.txt"
    if [ -s "$RESULTS_FILE" ]; then
      echo "[SKIP EVAL] Found existing $RESULTS_FILE"
    else
      echo "[EVAL] Running evaluation"
      python "$PY_SCRIPT" \
        --eval \
        --pred_depth_path "$OUT_DIR/depths" \
        --ground_truth_path "$SCENE_DIR/data/aligned_depth" \
        --dataset_name "$DATASET_NAME" \
        --method_name eval \
        --evaluation_protocol "$EVAL_PROTOCOL" \
        --eval_output_dirpath "$OUT_DIR"
    fi
  done
done
