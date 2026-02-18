#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python src/monocular_depth/run_external_mde_model.py \
--image_path \
    /media/common/datasets/thin_object_data_subset/markerholder_hallway/image.txt \
--ground_truth_path \
    /media/common/datasets/thin_object_data_subset/markerholder_hallway/aligned_depth.txt \
--model_name depthanything-v2 \
--dataset_name thin-obj \
--min_predict_depth 0.1 \
--max_predict_depth 10.0 \
--evaluation_protocol default \
--min_evaluate_depth 1e-3 \
--max_evaluate_depth 10.0 \
--output_dirpath \
    /media/home/anduong/thin-obj-result/markerholder_hallway \
--save_outputs \
--keep_input_filenames \
--device gpu

# python src/monocular_depth/run_external_mde_model.py \
# --image_path \
#     /media/home/zxguo/thin-obj-data-setup/data/data_collection_realsense2/bee_hallway/image.txt \
# --ground_truth_path \
#     /media/home/zxguo/thin-obj-data-setup/data/data_collection_realsense2/bee_hallway/aligned_depth.txt \
# --model_name depthanything-v2 \
# --dataset_name thin-obj \
# --min_predict_depth 0.1 \
# --max_predict_depth 100.0 \
# --evaluation_protocol kitti linear_fit \
# --min_evaluate_depth 1e-3 \
# --max_evaluate_depth 10.0 \
# --output_dirpath \
#     outputs_thinobj_bee/linear_fit \
# --save_outputs \
# --keep_input_filenames \
# --device gpu

# python src/monocular_depth/run_external_mde_model.py \
# --image_path \
#     /media/home/zxguo/thin-obj-data-setup/data/data_collection_realsense2/bee_hallway/image.txt \
# --ground_truth_path \
#     /media/home/zxguo/thin-obj-data-setup/data/data_collection_realsense2/bee_hallway/aligned_depth.txt \
# --model_name depthanything-v2 \
# --dataset thin-obj \
# --min_predict_depth 0.1 \
# --max_predict_depth 100.0 \
# --evaluation_protocol kitti median_scale \
# --min_evaluate_depth 1e-3 \
# --max_evaluate_depth 10.0 \
# --output_dirpath \
#     outputs_thinobj_bee/median_scale \
# --save_outputs \
# --keep_input_filenames \
# --device gpu
