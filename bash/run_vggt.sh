#!/usr/bin/env bash
set -euo pipefail

# ===============================================================
# VGGT batch runner + eval
# ===============================================================

ENV_PATH="${ENV_PATH:-$HOME/venv/vggt/bin/activate}"
DATA_ROOT="${1:-/media/home/zxguo/thin-obj-data-setup/data/data_collection_COLMAP}"
OUTPUT_ROOT="${2:-/media/home/zxguo/thin-obj-data-setup/data/vggt_output_with_eval}"
PY_SCRIPT="${PY_SCRIPT:-/media/home/zxguo/thin-obj-data-setup/external/vision-lab-wrappers-datasets/src/multiview/run_multi_view_model.py}"

DEVICE="${DEVICE:-gpu}"
BATCH_SIZE="${BATCH_SIZE:-30}"
STRIDE="${STRIDE:-30}"

DATASET_NAME="${DATASET_NAME:-thin-obj}"
EVAL_PROTOCOL="${EVAL_PROTOCOL:-default}"

echo "[INFO] Activating venv: $ENV_PATH"
source "$ENV_PATH"
mkdir -p "$OUTPUT_ROOT"

for SCENE_DIR in "$DATA_ROOT"/*; do
  [ -d "$SCENE_DIR" ] || continue

  SCENE_NAME="$(basename "$SCENE_DIR")"
  OUT_DIR="$OUTPUT_ROOT/$SCENE_NAME"

  echo "=========================================="
  echo "[SCENE] $SCENE_NAME"
  echo "  IN : $SCENE_DIR"
  echo "  OUT: $OUT_DIR"
  echo "------------------------------------------"


  DEPTH_DIR="$OUT_DIR/depths"
  HAS_DEPTHS=0
  if [ -d "$DEPTH_DIR" ] && [ "$(ls -1 "$DEPTH_DIR"/*.png 2>/dev/null | wc -l)" -gt 0 ]; then
    HAS_DEPTHS=1
  fi

  if [ "$HAS_DEPTHS" -eq 1 ]; then
    echo "[SKIP RUN] Found existing depths in $DEPTH_DIR"
  else
    echo "[RUN] VGGT inference"
    python "$PY_SCRIPT" \
      --scene_dir "$SCENE_DIR" \
      --output_dirpath "$OUTPUT_ROOT" \
      --batch_size "$BATCH_SIZE" \
      --stride "$STRIDE" \
      --device "$DEVICE" \
      --run_model
  fi


  RESULTS_FILE="$OUT_DIR/eval/results.txt"
  if [ -s "$RESULTS_FILE" ]; then
    echo "[SKIP EVAL] Found existing $RESULTS_FILE"
  else
    echo "[EVAL] Running evaluation"
    python "$PY_SCRIPT" \
      --scene_dir "$SCENE_DIR" \
      --output_dirpath "$OUTPUT_ROOT" \
      --device "$DEVICE" \
      --eval \
      --pred_depth_path "$OUT_DIR/depths" \
      --ground_truth_path "$SCENE_DIR/gt_depth" \
      --dataset_name "$DATASET_NAME" \
      --method_name eval \
      --evaluation_protocol "$EVAL_PROTOCOL" \
      --eval_output_dirpath "$OUT_DIR"
  fi

  echo "[DONE] $SCENE_NAME"
  echo
done

echo "=========================================="
echo "[ALL SCENES COMPLETE]"
echo "=========================================="
