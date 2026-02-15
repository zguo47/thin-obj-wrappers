#!/usr/bin/env bash
set -uo pipefail

# ===============================================================
 # 2DGS
# ===============================================================

# ---- Configurable paths ----
ENV_PATH=~/venv/surfel_splatting/bin/activate
DATA_ROOT="${1:-/media/home/zxguo/thin-obj-data-setup/data/data_collection_GS}"   # root containing scene folders
OUTPUT_ROOT="${2:-/media/home/zxguo/thin-obj-data-setup/data/gaussian_2d_output_with_eval}" # root output folder
PY_SCRIPT="/media/home/zxguo/thin-obj-data-setup/external/vision-lab-wrappers-datasets/src/novel_view_synthesis/run_novel_view_synthesis_model.py"

# Run settings
MODEL_NAME="gaussian_splatting_2d"
DATASET_NAME="thin-obj"
ITERATION=30000
DEVICE="gpu"     

# ---- Activate environment ----
echo "[INFO] Activating venv: $ENV_PATH"
source "$ENV_PATH"

mkdir -p "$OUTPUT_ROOT"

# ---- Loop through each subdirectory ----
for SCENE_DIR in "$DATA_ROOT"/*; do
  if [ ! -d "$SCENE_DIR" ]; then
    continue
  fi

  SCENE_NAME="$(basename "$SCENE_DIR")"
  OUT_DIR="$OUTPUT_ROOT/$SCENE_NAME"

  mkdir -p "$OUT_DIR"

  # -------------------------------------------------------------
  # Skip conditions
  # -------------------------------------------------------------
  RENDERS_DIR="$OUT_DIR/test/renders"
  RESULTS_FILE_TEST="$OUT_DIR/eval_test/results.txt"
  RESULTS_FILE_TRAIN="$OUT_DIR/eval_train/results.txt"

  HAS_TEST_RENDERS=0
  if [ -d "$RENDERS_DIR" ] && [ "$(ls -A "$RENDERS_DIR" 2>/dev/null | wc -l)" -gt 0 ]; then
    HAS_TEST_RENDERS=1
  fi

  HAS_RESULTS=0
  if [ -s "$RESULTS_FILE_TEST" ] && [ -s "$RESULTS_FILE_TRAIN" ]; then
    HAS_RESULTS=1
  fi

  # -------------------------------------------------------------
  # Train + render (only if renders missing)
  # -------------------------------------------------------------
  if [ "$HAS_TEST_RENDERS" -eq 1 ]; then
    echo "[SKIP RUN] Found existing test renders in $RENDERS_DIR"
  else
    echo "[RUN] Training + rendering (no test renders found)"
    python "$PY_SCRIPT" \
      --model_name "$MODEL_NAME" \
      --dataset_name "$DATASET_NAME" \
      --output_dirpath "$OUT_DIR" \
      --gs_source_path "$SCENE_DIR" \
      --do_train \
      --do_render \
      --split all \
      --iteration "$ITERATION" \
      --device "$DEVICE" \
      --run_model
  fi

  # -------------------------------------------------------------
  # Evaluation (only if results.txt missing)
  # -------------------------------------------------------------
  if [ "$HAS_RESULTS" -eq 1 ]; then
  echo "[SKIP EVAL] Found existing $RESULTS_FILE_TEST and $RESULTS_FILE_TRAIN"
  else
    echo "Evaluating..."
    echo

    python "$PY_SCRIPT" \
      --pred_depth_path "$OUT_DIR/test/depths" \
      --ground_truth_path "$SCENE_DIR/gt_depth" \
      --dataset_name thin-obj \
      --method_name eval_test \
      --evaluation_protocol default \
      --eval_output_dirpath "$OUT_DIR" \
      --eval

    python "$PY_SCRIPT" \
      --pred_depth_path "$OUT_DIR/train/depths" \
      --ground_truth_path "$SCENE_DIR/gt_depth" \
      --dataset_name thin-obj \
      --method_name eval_train \
      --evaluation_protocol default \
      --eval_output_dirpath "$OUT_DIR" \
      --eval
  fi
  
  echo "[DONE] $SCENE_NAME"
  echo
done

echo "=========================================="
echo "[ALL SCENES COMPLETE]"
echo "=========================================="
