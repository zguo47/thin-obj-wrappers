import os
import torch
import data_utils
from log_utils import log
import numpy as np
from tqdm import tqdm
import time
import eval_utils

def run_model(
    model_name: str,
    dataset_name: str,
    output_dirpath: str,
    # mode
    do_train: bool = True,
    do_render: bool = True,
    split: str = "test",
    iteration: int = -1,
    # device
    device: str = "cuda",

    # -------- GS
    gs_source_path: str = None,        # scene folder 
    gs_checkpoint_path: str = None,    # optional checkpoint restore

    # -------- Nerfstudio
    ns_scene_dir: str = None,
    ns_model_type: str = None,         # "nerfacto" or "depth-nerfacto" 
    ns_load_config_path: str = None,   
):
    """
    Models:
      - gaussian_splatting_2d
      - gaussian_splatting_3d
      - nerfacto
      - depth_nerfacto

    Unified behavior after your updates:
      - `output_dirpath` is the single place where:
          * GS checkpoints / model files / renders are written (via dataset.model_path == output_dirpath)
          * Nerfstudio outputs/checkpoints/renders are written (via cfg.output_dir == output_dirpath)
      - If do_train=True and do_render=True: train first, then render using the in-memory trained model.
      - If do_train=False and do_render=True: render from disk/config (if supported by the model).
    """

    model_name = model_name.lower()
    split = split.lower()
    os.makedirs(output_dirpath, exist_ok=True)

    dev = torch.device(device if device is not None else "cuda")

    renderings, depths = None, None

    # -------------------------
    # Gaussian Splatting 2D
    # -------------------------
    if model_name == "gaussian_splatting_2d":

        from gaussian_splatting_2d_model import GaussianSplatting2DModel

        model = GaussianSplatting2DModel(dataset_name=dataset_name, device=dev)

        if do_train or do_render:
            if gs_source_path is None:
                raise ValueError("2DGS requires gs_source_path when do_train or do_render is True.")
            # IMPORTANT: output_dirpath is where checkpoints/renders go
            model.set_paths(source_path=gs_source_path, model_path=output_dirpath)

        if do_train:
            model.optimize(None, None, None, checkpoint_path=gs_checkpoint_path)

        if do_render:
            # render should use in-memory (self.scene/self.gaussians) if optimize() ran
            if split == "train":
                model.render(dataset=None, pipe=None, iteration=iteration, skip_train=False, skip_test=True)
                renderings, depths = model.get_train_outputs()
            elif split == "test":
                model.render(dataset=None, pipe=None, iteration=iteration, skip_train=True, skip_test=False)
                renderings, depths = model.get_test_outputs()
            elif split == "all":
                model.render(dataset=None, pipe=None, iteration=iteration, skip_train=False, skip_test=True)
                train_r, train_d = model.get_train_outputs()
                model.render(dataset=None, pipe=None, iteration=iteration, skip_train=True, skip_test=False)
                test_r, test_d = model.get_test_outputs()
                renderings, depths = {"train": train_r, "test": test_r}, {"train": train_d, "test": test_d}
            else:
                raise ValueError(f"Unsupported split={split}. Use 'train', 'test', or 'all'.")

        return model, renderings, depths

    # -------------------------
    # Gaussian Splatting 3D
    # -------------------------
    if model_name == "gaussian_splatting_3d":

        from gaussian_splatting_3d_model import GaussianSplatting3DModel

        model = GaussianSplatting3DModel(dataset_name=dataset_name, device=dev)

        if do_train or do_render:
            if gs_source_path is None:
                raise ValueError("3DGS requires gs_source_path when do_train or do_render is True.")
            # IMPORTANT: output_dirpath is where checkpoints/renders go
            model.set_paths(source_path=gs_source_path, model_path=output_dirpath)

        if do_train:
            model.optimize(None, None, None, checkpoint_path=gs_checkpoint_path)

        if do_render:
            # render_sets uses in-memory self.scene/self.gaussians after optimize()
            if split == "train":
                model.render_sets(dataset=None, pipeline=None, iteration=iteration, skip_train=False, skip_test=True)
                renderings, depths = model.get_train_outputs()
            elif split == "test":
                model.render_sets(dataset=None, pipeline=None, iteration=iteration, skip_train=True, skip_test=False)
                renderings, depths = model.get_test_outputs()
            elif split == "all":
                model.render_sets(dataset=None, pipeline=None, iteration=iteration, skip_train=False, skip_test=True)
                train_r, train_d = model.get_train_outputs()
                model.render_sets(dataset=None, pipeline=None, iteration=iteration, skip_train=True, skip_test=False)
                test_r, test_d = model.get_test_outputs()
                renderings, depths = {"train": train_r, "test": test_r}, {"train": train_d, "test": test_d}
            else:
                raise ValueError(f"Unsupported split={split}. Use 'train', 'test', or 'all'.")

        return model, renderings, depths

    # -------------------------
    # Nerfacto (Nerfstudio)
    # -------------------------
    if model_name == "nerfacto":

        from nerfacto_model import NerfactoModel

        model = NerfactoModel(dataset_name=dataset_name, device=dev)

        # you said you updated nerfacto to also know output_dir so render can save there
        # (store output_dir + config path inside the model during optimize)
        if do_train:
            if ns_scene_dir is None:
                raise ValueError("Nerfacto train requires ns_scene_dir.")
            mtype = ns_model_type if ns_model_type is not None else "nerfacto"
            model.optimize(scene_dir=ns_scene_dir, output_root=output_dirpath, model_type=mtype)

        # render either from trained pipeline or from config path
        if do_render:
            # If you didn’t train in this run, you need a config.yml path.
            if (not do_train) and (ns_load_config_path is None):
                raise ValueError("Nerfacto render without training requires ns_load_config_path (config.yml).")
            model.set_render_inputs(load_config_path=ns_load_config_path)

            if split in ["train", "test"]:
                renderings, depths = model.run_render(split=split)
            elif split == "all":
                train_r, train_d = model.run_render(split="train")
                test_r, test_d = model.run_render(split="test")
                renderings, depths = {"train": train_r, "test": test_r}, {"train": train_d, "test": test_d}
            else:
                raise ValueError(f"Unsupported split={split}. Use 'train', 'test', or 'all'.")

        return model, renderings, depths

    # -------------------------
    # Depth-Nerfacto (Nerfstudio)
    # -------------------------
    if model_name == "depth_nerfacto":

        from depth_nerfacto_model import DepthNerfactoModel

        model = DepthNerfactoModel(dataset_name=dataset_name, device=dev)

        if do_train:
            if ns_scene_dir is None:
                raise ValueError("Depth-Nerfacto train requires ns_scene_dir.")
            mtype = ns_model_type if ns_model_type is not None else "depth-nerfacto"
            model.optimize(scene_dir=ns_scene_dir, output_root=output_dirpath, model_type=mtype)

        if do_render:
            if (not do_train) and (ns_load_config_path is None):
                raise ValueError("Depth-Nerfacto render without training requires ns_load_config_path (config.yml).")
            model.set_render_inputs(load_config_path=ns_load_config_path)

            if split in ["train", "test"]:
                renderings, depths = model.run_render(split=split)
            elif split == "all":
                train_r, train_d = model.run_render(split="train")
                test_r, test_d = model.run_render(split="test")
                renderings, depths = {"train": train_r, "test": test_r}, {"train": train_d, "test": test_d}
            else:
                raise ValueError(f"Unsupported split={split}. Use 'train', 'test', or 'all'.")

        return model, renderings, depths

    raise ValueError(
        f"Unknown model_name={model_name}. Expected one of: "
        f"gaussian_splatting_2d, gaussian_splatting_3d, nerfacto, depth_nerfacto"
    )


def run_depth_eval(pred_depth_path,
                          ground_truth_path,
                          weight_path,
                          mask_path,
                          dataset_name,
                          # "method" settings (just for logging / clipping)
                          method_name,
                          min_predict_depth,
                          max_predict_depth,
                          # Evaluation settings
                          min_evaluate_depth,
                          max_evaluate_depth,
                          evaluation_protocol,
                          # Output settings
                          use_mask,
                          output_dirpath):
    """
    Evaluate precomputed depth maps vs ground-truth,
    but only within the region indicated by masks in mask_dir.
    """

    # -----------------------------
    # Setup output & logging
    # -----------------------------
    output_dirpath = os.path.join(output_dirpath, method_name)

    if not os.path.exists(output_dirpath):
        os.makedirs(output_dirpath)

    log_path = os.path.join(output_dirpath, 'results.txt')

    # -----------------------------
    # Read input paths
    # -----------------------------
    pred_paths = data_utils.return_paths(pred_depth_path)
    n_sample = len(pred_paths)

    gt_paths = data_utils.return_paths(ground_truth_path)

    weights = None
    if weight_path is not None:
        weight_paths = data_utils.return_paths(weight_path)
        if len(weight_paths) != n_sample:
            raise ValueError(f"#weights ({len(weight_paths)}) != #pred depth ({n_sample})")
        weights = weight_paths

    if use_mask:
        mask_paths = data_utils.return_paths(mask_path)

    # Log input paths
    log('Input paths:', log_path)
    log(pred_depth_path, log_path)
    log(ground_truth_path, log_path)
    if use_mask:
        log(f"mask_dir={mask_dir}", log_path)
    log('', log_path)

    # -----------------------------
    # Log "network" / method settings
    # -----------------------------
    log_network_settings(
        log_path,
        model_name=method_name,
        min_predict_depth=min_predict_depth,
        max_predict_depth=max_predict_depth,
        # No actual model parameters here
        parameters_model=[],
        restore_paths=[]
    )

    log_evaluation_settings(
        log_path,
        min_evaluate_depth=min_evaluate_depth,
        max_evaluate_depth=max_evaluate_depth,
        evaluation_protocol=evaluation_protocol
    )

    log_system_settings(
        log_path,
        checkpoint_path=output_dirpath,
        device='cpu',
        n_thread=1
    )

    # -----------------------------
    # Prepare metrics buffers
    # -----------------------------
    a1 = np.zeros(n_sample)
    a2 = np.zeros(n_sample)
    a3 = np.zeros(n_sample)
    abs_rel = np.zeros(n_sample)
    log_10_mae = np.zeros(n_sample)
    log_rmse = np.zeros(n_sample)
    mae = np.zeros(n_sample)
    rmse = np.zeros(n_sample)

    time_elapse = 0.0

    # -----------------------------
    # Main loop
    # -----------------------------
    for idx in tqdm(range(n_sample), desc='Evaluating', total=n_sample):
        pred_path = pred_paths[idx]
        gt_path = gt_paths[idx] 

        t0 = time.time()
        pred = data_utils.load_depth(pred_path)
        time_elapse += (time.time() - t0)

        gt = data_utils.load_depth(gt_path)

        if use_mask:
            mask_path = mask_paths[idx]
            mask_bool = data_utils.load_mask(mask_path, gt.shape)

        # Optionally clamp prediction range
        pred = np.clip(pred, min_predict_depth, max_predict_depth)

        if weights is not None:
            w = np.load(weights[idx]).astype(np.float32)
            # match valid region
            w = np.nan_to_num(w)
        else:
            w = None


        # -------------------------
        # Metrics
        # -------------------------

        ground_truth = gt
        output_depth = pred

        # Apply optional scaling strategies
        if 'median_scale' in evaluation_protocol:
            mask_eval = (ground_truth > min_evaluate_depth) & (ground_truth <= max_evaluate_depth)
            if np.any(mask_eval):
                scale = np.median(ground_truth[mask_eval]) / np.median(output_depth[mask_eval])
                output_depth = output_depth * scale

        if 'linear_fit' in evaluation_protocol:
            ground_truth_valid = (ground_truth > min_evaluate_depth) & (ground_truth <= max_evaluate_depth)
            ground_truth_to_fit = ground_truth[ground_truth_valid]
            output_depth_to_fit = output_depth[ground_truth_valid]
            if len(output_depth_to_fit) > 0:
                A = np.vstack([output_depth_to_fit, np.ones(len(output_depth_to_fit))]).T
                m, c = np.linalg.lstsq(A, ground_truth_to_fit, rcond=None)[0]
                output_depth = m * output_depth + c

        # Clamp to evaluation range and clean up NaNs/Infs
        output_depth[output_depth > max_evaluate_depth] = max_evaluate_depth
        output_depth[output_depth < min_evaluate_depth] = min_evaluate_depth
        output_depth[np.isinf(output_depth)] = max_evaluate_depth
        output_depth[np.isnan(output_depth)] = min_evaluate_depth

        # Validity map where GT > 0
        validity_map_ground_truth = (ground_truth > 0)

        # Mask within [min, max]
        min_max_mask = np.logical_and(
            ground_truth > min_evaluate_depth,
            ground_truth < max_evaluate_depth)

        # Combine with mask_bool
        final_mask_bool = np.logical_and(validity_map_ground_truth, min_max_mask)
        if use_mask:
            final_mask_bool = np.logical_and(final_mask_bool, mask_bool)

        mask = np.where(final_mask_bool)

        if w is not None:
            w_masked = w[mask]
        else:
            w_masked = None

        # Compute metrics
        a1[idx] = eval_utils.a1_err(output_depth[mask], ground_truth[mask], threshold=1.05, weights=w_masked)
        a2[idx] = eval_utils.a2_err(output_depth[mask], ground_truth[mask], threshold=1.05, weights=w_masked)
        a3[idx] = eval_utils.a3_err(output_depth[mask], ground_truth[mask], threshold=1.05, weights=w_masked)
        abs_rel[idx] = eval_utils.abs_rel_err(output_depth[mask], ground_truth[mask], weights=w_masked)
        log_10_mae[idx] = eval_utils.log_10_mean_abs_err(output_depth[mask], ground_truth[mask], weights=w_masked)
        log_rmse[idx] = eval_utils.log_root_mean_sq_err(output_depth[mask], ground_truth[mask], weights=w_masked)
        mae[idx] = eval_utils.mean_abs_err(output_depth[mask], ground_truth[mask], weights=w_masked)
        rmse[idx] = eval_utils.root_mean_sq_err(output_depth[mask], ground_truth[mask], weights=w_masked)

    # -----------------------------
    # Aggregate & log stats
    # -----------------------------
    time_elapse_ms = time_elapse * 1000.0

    valid_idx = ~np.isnan(a1)

    a1_mean = np.mean(a1[valid_idx])
    a2_mean = np.mean(a2[valid_idx])
    a3_mean = np.mean(a3[valid_idx])
    abs_rel_mean = np.mean(abs_rel[valid_idx])
    log_10_mae_mean = np.mean(log_10_mae[valid_idx])
    log_rmse_mean = np.mean(log_rmse[valid_idx])
    mae_mean = np.mean(mae[valid_idx])
    rmse_mean = np.mean(rmse[valid_idx])

    a1_std = np.std(a1[valid_idx])
    a2_std = np.std(a2[valid_idx])
    a3_std = np.std(a3[valid_idx])
    abs_rel_std = np.std(abs_rel[valid_idx])
    log_10_mae_std = np.std(log_10_mae[valid_idx])
    log_rmse_std = np.std(log_rmse[valid_idx])
    mae_std = np.std(mae[valid_idx])
    rmse_std = np.std(rmse[valid_idx])

    # Print evaluation results to console and file
    log('\nEvaluation results (masked)', log_path)
    log('{:>8}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}'.format(
        'Step', 'A1', 'A2', 'A3', 'Abs_Rel', 'log10 MAE', 'log_RMSE', 'MAE', 'RMSE'),
        log_path)
    log('{:8}  {:8.3f}  {:8.3f}  {:8.3f}  {:8.3f}  {:8.3f}  {:8.3f}  {:8.3f}  {:8.3f}'.format(
        0,
        a1_mean,
        a2_mean,
        a3_mean,
        abs_rel_mean,
        log_10_mae_mean,
        log_rmse_mean,
        mae_mean,
        rmse_mean),
        log_path)

    log('{:>8}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}'.format(
        '', '+/-', '+/-', '+/-', '+/-', '+/-', '+/-', '+/-', '+/-'),
        log_path)
    log('{:8}  {:8.3f}  {:8.3f}  {:8.3f}  {:8.3f}  {:8.3f}  {:8.3f}  {:8.3f} {:8.3f}'.format(
        '',
        a1_std,
        a2_std,
        a3_std,
        abs_rel_std,
        log_10_mae_std,
        log_rmse_std,
        mae_std,
        rmse_std),
        log_path)

    log('Completed processing {} samples (masked)'.format(n_sample), log_path)
    log('Total time: {:.2f} ms  Average time per sample: {:.2f} ms'.format(
        time_elapse_ms, time_elapse_ms / float(n_sample)), log_path)


def log_network_settings(log_path,
                         # Depth network / method settings
                         model_name,
                         min_predict_depth=-1.0,
                         max_predict_depth=-1.0,
                         # Weight settings
                         parameters_model=[],
                         restore_paths=[]):

    # Compute number of parameters (0 for this script)
    n_parameter_model = sum(getattr(p, 'numel', lambda: 0)() for p in parameters_model)
    n_parameter = n_parameter_model

    log('Network settings:', log_path)
    log('model_name={}'.format(model_name),
        log_path)
    log('n_parameter={}'.format(n_parameter),
        log_path)
    log('min_predict_depth={:.2f}  max_predict_depth={:.2f}'.format(
        min_predict_depth, max_predict_depth),
        log_path)

    if len(restore_paths) > 0 and restore_paths[0] != '':
        for idx, path in enumerate(restore_paths):
            if idx == 0:
                log('restore_paths={}'.format(path), log_path)
            else:
                log('{:14}{}'.format('', path), log_path)

    log('', log_path)


def log_evaluation_settings(log_path,
                            min_evaluate_depth,
                            max_evaluate_depth,
                            evaluation_protocol=['default']):

    log('Evaluation settings:', log_path)
    log('evaluation_protocol={}'.format(
        evaluation_protocol),
        log_path)
    log('min_evaluate_depth={:.2f}  max_evaluate_depth={:.2f}'.format(
        min_evaluate_depth, max_evaluate_depth),
        log_path)
    log('', log_path)


def log_system_settings(log_path,
                        # Checkpoint settings
                        checkpoint_path,
                        restore_paths=None,
                        # Hardware settings
                        device='cpu',
                        n_thread=8):

    log('Checkpoint settings:', log_path)

    if checkpoint_path is not None:
        log('checkpoint_path={}'.format(checkpoint_path), log_path)
        log('', log_path)

    log('Hardware settings:', log_path)
    log('device={}'.format(device), log_path)
    log('n_thread={}'.format(n_thread), log_path)
    log('', log_path)
