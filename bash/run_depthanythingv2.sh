#!/usr/bin/env bash

set -e

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <input_root> <output_root>"
    exit 1
fi

INPUT_ROOT="$1"
OUTPUT_ROOT="$2"

export CUDA_VISIBLE_DEVICES=0

echo "=============================================="
echo "[INFO] Running monocular depth on dataset"
echo "[INFO] (masked unweighted, unmasked weighted, masked weighted)"
echo "[INFO] Input:  $INPUT_ROOT"
echo "[INFO] Output: $OUTPUT_ROOT"
echo "=============================================="

mkdir -p "$OUTPUT_ROOT"

MODEL_NAME="depthanything-v2"

# --------------------------------------------------------------
# Helper: dir has ≥1 file?
# --------------------------------------------------------------
has_files() {
    local d="$1"
    [ -d "$d" ] && [ "$(find "$d" -type f | wc -l)" -gt 0 ]
}

# --------------------------------------------------------------
# LOOP OVER SCENES
# --------------------------------------------------------------
for SCENE_DIR in "$INPUT_ROOT"/*; do
    [ -d "$SCENE_DIR" ] || continue

    SCENE_NAME=$(basename "$SCENE_DIR")
    SCENE_OUT="$OUTPUT_ROOT/$SCENE_NAME"
    echo "----------------------------------------------"
    echo "[SCENE] $SCENE_NAME"
    echo "----------------------------------------------"

    IMAGE_TXT="$SCENE_DIR/image.txt"
    DEPTH_TXT="$SCENE_DIR/aligned_depth.txt"
    MASK_DIR="$SCENE_DIR/mask"
    WEIGHTS_TXT="$SCENE_DIR/weights.txt"

    if [ ! -f "$IMAGE_TXT" ]; then
        echo "[WARN] Missing: $IMAGE_TXT – skipping scene."
        continue
    fi
    if [ ! -f "$DEPTH_TXT" ]; then
        echo "[WARN] Missing: $DEPTH_TXT – skipping scene."
        continue
    fi

    mkdir -p "$SCENE_OUT"

    # ==========================================================
    # 1) MASKED UNWEIGHTED
    # ==========================================================
    if [ -d "$MASK_DIR" ]; then
        BASE_MASKED_UW="$SCENE_OUT/masked_unweighted"
        OUT_MU_DEFAULT="$BASE_MASKED_UW/default/$MODEL_NAME"
        OUT_MU_LINEAR="$BASE_MASKED_UW/linear_fit/$MODEL_NAME"
        OUT_MU_MEDIAN="$BASE_MASKED_UW/median_scale/$MODEL_NAME"

        if has_files "$OUT_MU_DEFAULT" && has_files "$OUT_MU_LINEAR" && has_files "$OUT_MU_MEDIAN"; then
            echo "[SKIP] Masked unweighted already done for $SCENE_NAME."
        else
            echo "[RUN] Masked UNWEIGHTED for $SCENE_NAME"

            # default
            python src/monocular_depth/run_external_mde_model.py \
                --image_path         "$IMAGE_TXT" \
                --ground_truth_path  "$DEPTH_TXT" \
                --mask_dir           "$MASK_DIR" \
                --model_name         "$MODEL_NAME" \
                --dataset_name       thin-obj \
                --min_predict_depth  0.1 \
                --max_predict_depth  100.0 \
                --min_evaluate_depth 1e-3 \
                --max_evaluate_depth 10.0 \
                --output_dirpath     "$BASE_MASKED_UW/default" \
                --save_outputs \
                --keep_input_filenames \
                --device gpu

            # linear_fit
            python src/monocular_depth/run_external_mde_model.py \
                --image_path         "$IMAGE_TXT" \
                --ground_truth_path  "$DEPTH_TXT" \
                --mask_dir           "$MASK_DIR" \
                --model_name         "$MODEL_NAME" \
                --dataset_name       thin-obj \
                --min_predict_depth  0.1 \
                --max_predict_depth  100.0 \
                --evaluation_protocol linear_fit \
                --min_evaluate_depth 1e-3 \
                --max_evaluate_depth 10.0 \
                --output_dirpath     "$BASE_MASKED_UW/linear_fit" \
                --save_outputs \
                --keep_input_filenames \
                --device gpu

            # median_scale
            python src/monocular_depth/run_external_mde_model.py \
                --image_path         "$IMAGE_TXT" \
                --ground_truth_path  "$DEPTH_TXT" \
                --mask_dir           "$MASK_DIR" \
                --model_name         "$MODEL_NAME" \
                --dataset_name       thin-obj \
                --min_predict_depth  0.1 \
                --max_predict_depth  100.0 \
                --evaluation_protocol median_scale \
                --min_evaluate_depth 1e-3 \
                --max_evaluate_depth 10.0 \
                --output_dirpath     "$BASE_MASKED_UW/median_scale" \
                --save_outputs \
                --keep_input_filenames \
                --device gpu
        fi
    else
        echo "[SKIP] No mask dir for $SCENE_NAME → masked_unweighted not run."
    fi

    # ==========================================================
    # 2) UNMASKED WEIGHTED
    # ==========================================================
    if [ -f "$WEIGHTS_TXT" ]; then
        BASE_UNMASKED_W="$SCENE_OUT/unmasked_weighted"
        OUT_UW_DEFAULT="$BASE_UNMASKED_W/default/$MODEL_NAME"
        OUT_UW_LINEAR="$BASE_UNMASKED_W/linear_fit/$MODEL_NAME"
        OUT_UW_MEDIAN="$BASE_UNMASKED_W/median_scale/$MODEL_NAME"

        if has_files "$OUT_UW_DEFAULT" && has_files "$OUT_UW_LINEAR" && has_files "$OUT_UW_MEDIAN"; then
            echo "[SKIP] Unmasked weighted already done for $SCENE_NAME."
        else
            echo "[RUN] Unmasked WEIGHTED for $SCENE_NAME"

            # default
            python src/monocular_depth/run_external_mde_model.py \
                --image_path         "$IMAGE_TXT" \
                --ground_truth_path  "$DEPTH_TXT" \
                --weight_path        "$WEIGHTS_TXT" \
                --model_name         "$MODEL_NAME" \
                --dataset_name       thin-obj \
                --min_predict_depth  0.1 \
                --max_predict_depth  100.0 \
                --min_evaluate_depth 1e-3 \
                --max_evaluate_depth 10.0 \
                --output_dirpath     "$BASE_UNMASKED_W/default" \
                --save_outputs \
                --keep_input_filenames \
                --device gpu

            # linear_fit
            python src/monocular_depth/run_external_mde_model.py \
                --image_path         "$IMAGE_TXT" \
                --ground_truth_path  "$DEPTH_TXT" \
                --weight_path        "$WEIGHTS_TXT" \
                --model_name         "$MODEL_NAME" \
                --dataset_name       thin-obj \
                --evaluation_protocol linear_fit \
                --min_predict_depth  0.1 \
                --max_predict_depth  100.0 \
                --min_evaluate_depth 1e-3 \
                --max_evaluate_depth 10.0 \
                --output_dirpath     "$BASE_UNMASKED_W/linear_fit" \
                --save_outputs \
                --keep_input_filenames \
                --device gpu

            # median_scale
            python src/monocular_depth/run_external_mde_model.py \
                --image_path         "$IMAGE_TXT" \
                --ground_truth_path  "$DEPTH_TXT" \
                --weight_path        "$WEIGHTS_TXT" \
                --model_name         "$MODEL_NAME" \
                --dataset_name       thin-obj \
                --evaluation_protocol median_scale \
                --min_predict_depth  0.1 \
                --max_predict_depth  100.0 \
                --min_evaluate_depth 1e-3 \
                --max_evaluate_depth 10.0 \
                --output_dirpath     "$BASE_UNMASKED_W/median_scale" \
                --save_outputs \
                --keep_input_filenames \
                --device gpu
        fi
    else
        echo "[SKIP] No weights.txt for $SCENE_NAME → unmasked_weighted not run."
    fi


    # ==========================================================
    # 3) MASKED WEIGHTED
    # ==========================================================
    if [ -d "$MASK_DIR" ] && [ -f "$WEIGHTS_TXT" ]; then
        BASE_MASKED_W="$SCENE_OUT/masked_weighted"
        OUT_MW_DEFAULT="$BASE_MASKED_W/default/$MODEL_NAME"
        OUT_MW_LINEAR="$BASE_MASKED_W/linear_fit/$MODEL_NAME"
        OUT_MW_MEDIAN="$BASE_MASKED_W/median_scale/$MODEL_NAME"

        if has_files "$OUT_MW_DEFAULT" && has_files "$OUT_MW_LINEAR" && has_files "$OUT_MW_MEDIAN"; then
            echo "[SKIP] Masked weighted already done for $SCENE_NAME."
        else
            echo "[RUN] Masked WEIGHTED for $SCENE_NAME"

            # default
            python src/monocular_depth/run_external_mde_model.py \
                --image_path         "$IMAGE_TXT" \
                --ground_truth_path  "$DEPTH_TXT" \
                --weight_path        "$WEIGHTS_TXT" \
                --mask_dir           "$MASK_DIR" \
                --model_name         "$MODEL_NAME" \
                --dataset_name       thin-obj \
                --min_predict_depth  0.1 \
                --max_predict_depth  100.0 \
                --min_evaluate_depth 1e-3 \
                --max_evaluate_depth 10.0 \
                --output_dirpath     "$BASE_MASKED_W/default" \
                --save_outputs \
                --keep_input_filenames \
                --device gpu

            # linear_fit
            python src/monocular_depth/run_external_mde_model.py \
                --image_path         "$IMAGE_TXT" \
                --ground_truth_path  "$DEPTH_TXT" \
                --weight_path        "$WEIGHTS_TXT" \
                --mask_dir           "$MASK_DIR" \
                --model_name         "$MODEL_NAME" \
                --dataset_name       thin-obj \
                --min_predict_depth  0.1 \
                --max_predict_depth  100.0 \
                --evaluation_protocol linear_fit \
                --min_evaluate_depth 1e-3 \
                --max_evaluate_depth 10.0 \
                --output_dirpath     "$BASE_MASKED_W/linear_fit" \
                --save_outputs \
                --keep_input_filenames \
                --device gpu

            # median_scale
            python src/monocular_depth/run_external_mde_model.py \
                --image_path         "$IMAGE_TXT" \
                --ground_truth_path  "$DEPTH_TXT" \
                --weight_path        "$WEIGHTS_TXT" \
                --mask_dir           "$MASK_DIR" \
                --model_name         "$MODEL_NAME" \
                --dataset_name       thin-obj \
                --min_predict_depth  0.1 \
                --max_predict_depth  100.0 \
                --evaluation_protocol median_scale \
                --min_evaluate_depth 1e-3 \
                --max_evaluate_depth 10.0 \
                --output_dirpath     "$BASE_MASKED_W/median_scale" \
                --save_outputs \
                --keep_input_filenames \
                --device gpu
        fi
    else
        if [ ! -d "$MASK_DIR" ]; then
            echo "[SKIP] No mask dir for $SCENE_NAME → masked_weighted not run."
        elif [ ! -f "$WEIGHTS_TXT" ]; then
            echo "[SKIP] No weights.txt for $SCENE_NAME → masked_weighted not run."
        fi
    fi

    echo "[DONE] $SCENE_NAME"
done

echo "=============================================="
echo "[ALL SCENES COMPLETED]"
echo "=============================================="
