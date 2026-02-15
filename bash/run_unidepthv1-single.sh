#!/bin/bash

python src/monocular_depth/run_external_mde_model.py \
--image_path \
    /media/home/zxguo/thin-obj-data-setup/thin_object_data/data_collection_realsense2/bee_hallway/image.txt \
--ground_truth_path \
    /media/home/zxguo/thin-obj-data-setup/thin_object_data/data_collection_realsense2/bee_hallway/aligned_depth.txt \
--model_name unidepth-v1-vitl14 \
--dataset_name thin_obj \
--min_predict_depth 0.1 \
--max_predict_depth 100.0 \
--evaluation_protocol default \
--min_evaluate_depth 1e-3 \
--max_evaluate_depth 10.0 \
--output_dirpath \
    /media/home/zxguo/thin-obj-data-setup/thin_object_data/unidepthv1_output/default \
--save_outputs \
--keep_input_filenames \
--device gpu
