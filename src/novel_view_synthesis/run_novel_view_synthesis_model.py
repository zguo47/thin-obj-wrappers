import argparse
import torch
from external_novel_view_synthesis_main import run_model, run_depth_eval

parser = argparse.ArgumentParser()


parser.add_argument('--intrinsics_path',
    type=str, default=None, help='Path to list of intrinsics paths')
parser.add_argument('--ground_truth_path',
    type=str, default=None, help='Path to list of ground truth paths')
parser.add_argument('--pred_depth_path',
    type=str, default=None, help='Path to list of predicted depth paths')
parser.add_argument('--dataset_name',
    type=str, default='', help='Dataset name e.g. kitti or nyuv2')
parser.add_argument('--method_name',
    type=str, default='eval',
    help='Name of the eval method (used for logging & subfolder)')
parser.add_argument('--weight_path',
    type=str, default=None,
    help='Optional path to list of per-frame weights (npy paths)')
parser.add_argument('--mask_path',
    type=str, default=None,
    help='Path to list of per-frame mask paths (npy or image)')
parser.add_argument('--model_name',
    type=str, 
    help='One of: gaussian_splatting_2d, gaussian_splatting_3d, nerfacto, depth_nerfacto')

# ------------------------
# Run modes
# ------------------------
parser.add_argument('--do_train', action='store_true', help='If set, train/optimize the model')
parser.add_argument('--do_render', action='store_true', help='If set, render after training (or from existing state)')
parser.add_argument('--split', type=str, default='test', choices=['train', 'test', 'all'], help='Which split to render')
parser.add_argument('--iteration', type=int, default=-1, help='Iteration to render (GS disk render uses this)')

# ------------------------
# Gaussian Splatting paths
# ------------------------
parser.add_argument('--gs_source_path', type=str, default=None,
    help='GS: dataset source_path (required if using gaussian_splatting_2d/3d)')
parser.add_argument('--gs_checkpoint_path', type=str, default=None,
    help='GS: optional checkpoint to restore before training')

# ------------------------
# Nerfstudio paths
# ------------------------
parser.add_argument('--ns_scene_dir', type=str, default=None,
    help='Nerfstudio: scene directory (required to train nerfacto/depth_nerfacto)')
parser.add_argument('--ns_model_type', type=str, default=None,
    help='Nerfstudio: method config key, e.g. "nerfacto" or "depth-nerfacto"')
parser.add_argument('--ns_load_config_path', type=str, default=None,
    help='Nerfstudio: path to config.yml for rendering if pipeline not already in memory')

# ------------------------
# Evaluation settings 
# ------------------------
parser.add_argument('--min_predict_depth',
    type=float, default=1.5, help='Minimum depth prediction value')
parser.add_argument('--max_predict_depth',
    type=float, default=100.0, help='Maximum depth prediction value')
parser.add_argument('--min_evaluate_depth',
    type=float, default=0, help='Minimum value of depth to evaluate')
parser.add_argument('--max_evaluate_depth',
    type=float, default=100.0, help='Maximum value of depth to evaluate')
parser.add_argument('--evaluation_protocol',
    nargs='+', type=str, default=['default'], help='Protocol for evaluation')

# ------------------------
# Output settings
# ------------------------
parser.add_argument('--output_dirpath',
    type=str, help='Path to directory to log results')
parser.add_argument('--eval_output_dirpath',
    type=str, help='Path to directory to log eval results')
parser.add_argument('--use_mask',
    action='store_true', help='If set then use mask for inputs and outputs')

# ------------------------
# Hardware settings
# ------------------------
parser.add_argument('--device', 
    type=str, default='gpu', help='Device to use: gpu, cpu')

# ------------------------
# Modes
# ------------------------
parser.add_argument('--run_model', action='store_true', help='If set, run model training and rendering')
parser.add_argument('--eval', action='store_true', help='If set, run depth evaluation')


args = parser.parse_args()

if __name__ == "__main__":
    args.device = (args.device or "gpu").lower()
    if args.device not in ['cpu', 'gpu', 'cuda']:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.device = 'cuda' if args.device == 'gpu' else args.device

    do_train = args.do_train
    do_render = args.do_render
    if (not do_train) and (not do_render):
        do_train, do_render = True, True

    if args.run_model:
        model, renderings, depths = run_model(
            model_name=args.model_name,
            dataset_name=args.dataset_name.lower(),
            output_dirpath=args.output_dirpath,

            do_train=do_train,
            do_render=do_render,
            split=args.split,
            iteration=args.iteration,

            device=args.device,

            # -------- GS
            gs_source_path=args.gs_source_path,
            gs_checkpoint_path=args.gs_checkpoint_path,

            # -------- Nerfstudio
            ns_scene_dir=args.ns_scene_dir,
            ns_model_type=args.ns_model_type,
            ns_load_config_path=args.ns_load_config_path,
        )

    if args.eval:
        run_depth_eval(
            pred_depth_path=args.pred_depth_path,
            ground_truth_path=args.ground_truth_path,
            weight_path=args.weight_path,
            mask_path=args.mask_path,
            dataset_name=args.dataset_name.lower(),
            method_name=args.method_name,
            # Method-ish settings
            min_predict_depth=args.min_predict_depth,
            max_predict_depth=args.max_predict_depth,
            # Evaluation settings
            min_evaluate_depth=args.min_evaluate_depth,
            max_evaluate_depth=args.max_evaluate_depth,
            evaluation_protocol=args.evaluation_protocol,
            # Output settings
            use_mask=args.use_mask,
            output_dirpath=args.eval_output_dirpath,
        )

