import argparse
import torch

from external_depth_completion_main import run_omnidc, run_depth_eval


parser = argparse.ArgumentParser("Run OMNI-DC on a scene folder.")

parser.add_argument(
    "--rgb_dir",
    type=str,
    help="Folder containing RGB images (e.g. .../images or .../data/image)",
)
parser.add_argument(
    "--sparse_depth_dir",
    type=str,
    help="Folder containing sparse depth 16-bit PNGs, filenames match RGB stems",
)
parser.add_argument(
    "--out_dir",
    type=str,
    help="Output folder for predictions",
)

parser.add_argument(
    "--intrinsics_path",
    type=str,
    default=None,
    help="Path to universal 3x3 intrinsics .npy (recommended). If omitted, uses identity.",
)
parser.add_argument(
    "--rgb_ext",
    type=str,
    default="png",
    help="RGB extension to match (png/jpg/jpeg)",
)
parser.add_argument("--device", type=str, default="cuda", help="Device to use: gpu/cuda/cpu")
parser.add_argument(
    "--depth_multiplier",
    type=float,
    default=256.0,
    help="Multiplier used by your load_depth() (default 256)",
)
parser.add_argument(
    "--save_npy",
    action="store_true",
    help="Also save float32 depth maps as .npy",
)
parser.add_argument(
    "--overwrite",
    action="store_true",
    help="Overwrite existing outputs",
)

parser.add_argument('--run_model', 
action='store_true', help='If set, run model training and rendering')
parser.add_argument('--eval', 
action='store_true', help='If set, run depth evaluation')

parser.add_argument("--pred_depth_path", 
    type=str, default=None)
parser.add_argument("--ground_truth_path", 
    type=str, default=None)
parser.add_argument('--mask_path',
    type=str, default=None,
    help='Path to list of per-frame mask paths (npy or image)')
parser.add_argument("--weight_path", 
    type=str, default=None)

parser.add_argument("--dataset_name", 
    type=str, default="thin-obj")
parser.add_argument("--method_name", 
    type=str, default="eval")

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

parser.add_argument("--use_mask", action="store_true")
parser.add_argument("--eval_output_dirpath", type=str, default=None)


args = parser.parse_args()


if __name__ == "__main__":
    device = (args.device or "gpu").lower()
    if device not in ["cpu", "gpu", "cuda"]:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cuda" if device == "gpu" else device

    if args.run_model:
        run_omnidc(
            rgb_dir=args.rgb_dir,
            sparse_depth_dir=args.sparse_depth_dir,
            out_dir=args.out_dir,
            intrinsics_path=args.intrinsics_path,
            rgb_ext=args.rgb_ext,
            device=device,
            depth_multiplier=args.depth_multiplier,
            save_npy=args.save_npy,
            overwrite=args.overwrite,
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
