import argparse
import torch

from external_multi_view_main import run_vggt, run_depth_eval


parser = argparse.ArgumentParser()

parser.add_argument("--scene_dir",
    type=str, required=True,
    help="Path to a scene directory (expects data/image/*.png inside).",
)
parser.add_argument("--output_dirpath",
    type=str, required=True,
    help="Root output directory (VGGT will write into output_dirpath/<scene_name>/...).",
)

parser.add_argument("--batch_size", type=int, default=30, help="Batch size for VGGT inference.")
parser.add_argument("--stride", type=int, default=30, help="Stride for batching (step between batches).")

parser.add_argument("--device", type=str, default="gpu", help="Device to use: gpu/cuda/cpu")

parser.add_argument('--run_model', action='store_true', help='If set, run model training and rendering')
parser.add_argument('--eval', action='store_true', help='If set, run depth evaluation')

# ------------------------
# Eval args 
# ------------------------
parser.add_argument("--pred_depth_path", type=str, default=None)
parser.add_argument("--ground_truth_path", type=str, default=None)
parser.add_argument('--mask_path',
    type=str, default=None,
    help='Path to list of per-frame mask paths (npy or image)')
parser.add_argument("--weight_path", type=str, default=None)

parser.add_argument("--dataset_name", type=str, default="thin-obj")
parser.add_argument("--method_name", type=str, default="eval")

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
        model, out_dir = run_vggt(
            scene_dir=args.scene_dir,
            output_dirpath=args.output_dirpath,
            device=device,
            batch_size=args.batch_size,
            stride=args.stride,
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


