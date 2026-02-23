python src/monocular_depth/run_external_mde_model.py \
--image_path \
    /media/common/datasets/thin_object_data_subset/markerholder_hallway/image.txt \
--ground_truth_path \
    /media/common/datasets/thin_object_data_subset/markerholder_hallway/aligned_depth.txt \
--model_name depthanything-v2 \
--dataset_name thin-obj \
--evaluation_protocol default \
--output_dirpath \
    /media/home/anduong/thin-obj-result/marigold/markerholder_hallway \
--save_outputs \
--keep_input_filenames \
--device gpu