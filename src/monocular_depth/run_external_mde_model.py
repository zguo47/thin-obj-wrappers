import argparse
import torch
from external_mde_main import run_mde


parser = argparse.ArgumentParser()

# Input filepaths
parser.add_argument('--image_path',
    type=str, required=True, help='Path to list of image paths')
parser.add_argument('--intrinsics_path',
    type=str, default=None, help='Path to list of intrinsics paths')
parser.add_argument('--ground_truth_path',
    type=str, default=None, help='Path to list of ground truth paths')
parser.add_argument('--dataset_name',
    type=str, default='', help='Dataset name e.g. kitti or nyuv2')
parser.add_argument('--weight_path',
    type=str, default=None,
    help='Optional path to list of per-frame weights (npy paths)')

# mask settings
parser.add_argument('--mask_dir',
    type=str, default=None, help='Path to directory of mask .png files')

# Network settings
parser.add_argument('--model_name',
    type=str, required=True, help='Monocular depth estimation model to instantiate')
parser.add_argument('--restore_paths_model',
    nargs='+', type=str, default=None, help='Paths to restore depth model from checkpoint')
parser.add_argument('--min_predict_depth',
    type=float, default=1.5, help='Minimum depth prediction value')
parser.add_argument('--max_predict_depth',
    type=float, default=100.0, help='Maximum depth prediction value')

# Evaluation settings
parser.add_argument('--min_evaluate_depth',
    type=float, default=0, help='Minimum value of depth to evaluate')
parser.add_argument('--max_evaluate_depth',
    type=float, default=100.0, help='Maximum value of depth to evaluate')
parser.add_argument('--evaluation_protocol',
    nargs='+', type=str, default=['default'], help='Protocol for evaluation')

# Output settings
parser.add_argument('--output_dirpath',
    type=str, required=True, help='Path to directory to log results')
parser.add_argument('--save_outputs',
    action='store_true', help='If set then strore inputs and outputs into output directory')
parser.add_argument('--keep_input_filenames',
    action='store_true', help='If set then keep input filenames')

# Hardware settings
parser.add_argument('--device', type=str, default='gpu', help='Device to use: gpu, cpu')

args = parser.parse_args()

if __name__ == '__main__':

    # Hardware settings
    args.device = args.device.lower()
    if args.device not in ['cpu', 'gpu', 'cuda']:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    args.device = 'cuda' if args.device == 'gpu' else args.device

    run_mde(image_path=args.image_path,
            ground_truth_path=args.ground_truth_path,
            weight_path=args.weight_path,
            mask_dir=args.mask_dir,
            intrinsics_path=args.intrinsics_path,
            model_name=args.model_name.lower(),
            # Model settings
            restore_paths_model=args.restore_paths_model,
            dataset_name=args.dataset_name.lower(),
            min_predict_depth=args.min_predict_depth,
            max_predict_depth=args.max_predict_depth,
            # Evaluation settings
            min_evaluate_depth=args.min_evaluate_depth,
            max_evaluate_depth=args.max_evaluate_depth,
            evaluation_protocol=args.evaluation_protocol,
            # Output settings
            output_dirpath=args.output_dirpath,
            save_outputs=args.save_outputs,
            keep_input_filenames=args.keep_input_filenames,
            # Hardware settings
            device=args.device)
