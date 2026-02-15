import os, time, torch
from tqdm import tqdm
import numpy as np
import data_utils, datasets, eval_utils
from external_mde_model import ExternalMonocularDepthEstimationModel
from log_utils import log
from PIL import Image
from fvcore.nn import FlopCountAnalysis


def run_mde(image_path,
            ground_truth_path,
            weight_path,
            mask_dir,
            intrinsics_path,
            dataset_name,
            # Model settings
            model_name,
            restore_paths_model,
            min_predict_depth,
            max_predict_depth,
            # Evaluation settings
            min_evaluate_depth,
            max_evaluate_depth,
            evaluation_protocol,
            # Output settings
            output_dirpath,
            save_outputs,
            keep_input_filenames,
            # Hardware settings
            device='cuda'):

    # Set up Hardware
    if device == 'cuda' or device == 'gpu':
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Set up output paths
    output_dirpath = os.path.join(output_dirpath, model_name)

    if not os.path.exists(output_dirpath):
        os.makedirs(output_dirpath)

    log_path = os.path.join(output_dirpath, 'results.txt')

    # Read input paths
    image_paths = data_utils.read_paths(image_path)

    n_sample = len(image_paths)

    is_available_ground_truth = False

    if ground_truth_path is not None:
        is_available_ground_truth = True
        ground_truth_paths = data_utils.read_paths(ground_truth_path)
    else:
        ground_truth_paths = None
    
    weights = None
    if weight_path is not None:
        weight_paths = data_utils.read_paths(weight_path)
        if len(weight_paths) != n_sample:
            raise ValueError(f"#weights ({len(weight_paths)}) != #pred depth ({n_sample})")
        weights = weight_paths

    is_available_intrinsics = False
    if intrinsics_path is not None:
        is_available_intrinsics = True
        intrinsics_paths = data_utils.read_paths(intrinsics_path)
    else:
        intrinsics_paths = None

    # Load masks if provided
    if mask_dir is not None:

        from pathlib import Path

        def build_mask_paths(mask_dir, ref_paths):
            mask_dir = Path(mask_dir)
            if not mask_dir.exists():
                raise ValueError(f"Mask directory does not exist: {mask_dir}")

            mask_files = [p for p in mask_dir.iterdir()]
            if not mask_files:
                raise ValueError(f"No mask images found in {mask_dir}")

            stem_to_mask = {}
            for p in mask_files:
                raw_stem = p.stem
                if raw_stem.endswith("_mask"):
                    clean_stem = raw_stem[:-5]
                else:
                    clean_stem = raw_stem
                stem_to_mask[clean_stem] = str(p)

            out = []
            for ref in ref_paths:
                ref_stem = os.path.splitext(os.path.basename(ref))[0]
                if ref_stem not in stem_to_mask:
                    raise ValueError(
                        f"Missing mask for '{ref_stem}' in {mask_dir}"
                    )
                out.append(stem_to_mask[ref_stem])
            return out

        # reference = image filenames
        mask_paths = build_mask_paths(mask_dir, image_paths)

    else:
        mask_paths = None

    dataloader = torch.utils.data.DataLoader(
        datasets.MonocularDepthEstimationDataset(
            dataset_name=dataset_name,
            image_paths=image_paths,
            intrinsics_paths=intrinsics_paths,
            ground_truth_paths=ground_truth_paths),
        batch_size=1,
        shuffle=False,
        num_workers=1,
        drop_last=False)

    '''
    Set up the model
    '''
    model = ExternalMonocularDepthEstimationModel(
        model_name=model_name,
        min_predict_depth=min_predict_depth,
        max_predict_depth=max_predict_depth,
        device=device)

    parameters_model = model.parameters()

    # Reload model
    if restore_paths_model is not None and restore_paths_model != '':
        _, _ = model.restore_model(restore_paths_model)

    # Set to evaluation mode
    model.eval()

    '''
    Log input paths
    '''
    log('Input paths:', log_path)
    input_paths = [
        image_path
    ]

    if is_available_intrinsics:
        input_paths.append(intrinsics_path)

    if is_available_ground_truth:
        input_paths.append(ground_truth_path)

    for path in input_paths:
        if path is not None:
            log(path, log_path)
    log('', log_path)

    log_network_settings(
        log_path,
        # Depth network settings
        model_name=model_name,
        min_predict_depth=min_predict_depth,
        max_predict_depth=max_predict_depth,
        # Weight settings
        parameters_model=parameters_model)

    log_evaluation_settings(
        log_path,
        min_evaluate_depth=min_evaluate_depth,
        max_evaluate_depth=max_evaluate_depth,
        evaluation_protocol=evaluation_protocol)

    log_system_settings(
        log_path,
        # Checkpoint settings
        checkpoint_path=output_dirpath,
        # Hardware settings
        device=device,
        n_thread=1)

    '''
    Run model
    '''
    n_sample = len(dataloader)

    # Define evaluation metrics
    a1 = np.zeros(n_sample)
    a2 = np.zeros(n_sample)
    a3 = np.zeros(n_sample)
    abs_rel = np.zeros(n_sample)
    log_10_mae = np.zeros(n_sample)
    log_rmse = np.zeros(n_sample)
    mae = np.zeros(n_sample)
    rmse = np.zeros(n_sample)

    time_elapse = 0.0

    if save_outputs:
        log('Saving outputs to {}'.format(output_dirpath), log_path)

        image_dirpath = os.path.join(output_dirpath, 'image')
        output_depth_dirpath = os.path.join(output_dirpath, 'output_monocular_depth')
        ground_truth_dirpath = os.path.join(output_dirpath, 'ground_truth')

        dirpaths = [
            image_dirpath,
            output_depth_dirpath,
            ground_truth_dirpath
        ]

        for dirpath in dirpaths:
            if not os.path.exists(dirpath):
                os.makedirs(dirpath)

    for idx, inputs in tqdm(enumerate(dataloader), desc='Evaluating', total=n_sample):

        if mask_paths is not None:
            mask_path = mask_paths[idx]
            mask_img = Image.open(mask_path)
            mask_arr = np.array(mask_img)
            if mask_arr.ndim == 3:
                mask_arr = mask_arr[..., 0]
            mask_bool = mask_arr.astype(np.float32) > 0.5
        else:
            mask_bool = None

        # Move inputs to device
        inputs = [
            in_.to(device) for in_ in inputs
        ]

        if is_available_ground_truth:
            image, ground_truth, intrinsics = inputs
        else:
            image, intrinsics = inputs

        time_start = time.time()

        with torch.no_grad():
            # Forward through monocular depth estimation network
            output_depth = model.forward(
                image=image,
                intrinsics=intrinsics)

        time_elapse = time_elapse + (time.time() - time_start)

        if idx == n_sample - 1:
            torch.cuda.empty_cache()

            try:
                flops = FlopCountAnalysis(model, (image))
            except Exception as e:
                print(f"FLOP computation failed: {e}")
        
        if weights is not None:
            w = np.load(weights[idx]).astype(np.float32)
            # match valid region
            w = np.nan_to_num(w)
        else:
            w = None

        # Convert to numpy to validate
        output_depth = np.squeeze(output_depth.cpu().numpy())

        if save_outputs:

            if keep_input_filenames:
                filename = os.path.basename(image_paths[idx])
                image_filename = os.path.splitext(filename)[0] + ".jpg"
                depth_filename = os.path.splitext(filename)[0] + ".png"
            else:
                filename = '{:010d}'.format(idx)
                image_filename = filename + '.jpg'
                depth_filename = filename + '.png'

            # Create image path and write to disk
            image = np.transpose(np.squeeze(image.cpu().numpy()), (1, 2, 0))

            image_path = os.path.join(
                image_dirpath,
                image_filename)
            image = (255 * image).astype(np.uint8)
            Image.fromarray(image).save(image_path)

            # Create output depth path
            output_depth_path = os.path.join(
                output_depth_dirpath,
                depth_filename)
            data_utils.save_depth(output_depth, output_depth_path)

            if is_available_ground_truth:
                # Create ground truth path and write to disk
                ground_truth = np.squeeze(ground_truth.cpu().numpy())

                ground_truth_path = os.path.join(
                    ground_truth_dirpath,
                    depth_filename)
                data_utils.save_depth(ground_truth, ground_truth_path)

        if is_available_ground_truth:

            if torch.is_tensor(ground_truth):
                ground_truth = np.squeeze(ground_truth.cpu().numpy())

            if len(output_depth.shape) > 2:
                output_depth = output_depth[:, :, 0]
            if len(ground_truth.shape) > 2:
                ground_truth = ground_truth[:, :, 0]

            if 'median_scale' in evaluation_protocol:
                mask = (ground_truth > min_evaluate_depth) & (ground_truth <= max_evaluate_depth)
                scale = np.median(ground_truth[mask]) / np.median(output_depth[mask])
                output_depth = output_depth * scale

            if 'linear_fit' in evaluation_protocol:
                ground_truth_valid = (ground_truth > min_evaluate_depth) & (ground_truth <= max_evaluate_depth)
                ground_truth_to_fit = ground_truth[ground_truth_valid]
                output_depth_to_fit = output_depth[ground_truth_valid]
                A = np.vstack([output_depth_to_fit, np.ones(len(output_depth_to_fit))]).T

                m, c = np.linalg.lstsq(A, ground_truth_to_fit, rcond=None)[0]
                output_depth = m * output_depth + c

            output_depth[output_depth > max_evaluate_depth] = max_evaluate_depth
            output_depth[output_depth < min_evaluate_depth] = min_evaluate_depth
            output_depth[np.isinf(output_depth)] = max_evaluate_depth
            output_depth[np.isnan(output_depth)] = min_evaluate_depth

            # Validity map of output -> locations where output is valid
            validity_map_ground_truth = np.where(ground_truth > 0, 1, 0)

            # Select valid regions to evaluate
            min_max_mask = np.logical_and(
                ground_truth > min_evaluate_depth,
                ground_truth < max_evaluate_depth)

            if 'kitti' in evaluation_protocol:
                eval_mask = np.zeros(min_max_mask.shape)
                gt_height, gt_width = ground_truth.shape
                eval_mask[int(0.40810811 * gt_height):int(0.99189189 * gt_height), int(0.03594771 * gt_width):int(0.96405229 * gt_width)] = 1
                min_max_mask = np.logical_and(min_max_mask, eval_mask)
            elif 'nyu_v2' in evaluation_protocol:
                eval_mask = np.zeros(min_max_mask.shape)
                eval_mask[45:472, 43:608] = 1
                min_max_mask = np.logical_and(min_max_mask, eval_mask)

            if mask_bool is not None:
                mask = np.where(np.logical_and(np.logical_and(validity_map_ground_truth, min_max_mask), mask_bool) > 0)
            else:
                mask = np.where(np.logical_and(validity_map_ground_truth, min_max_mask) > 0)
        
            # Apply weights (optional)
            if w is not None:
                w_masked = w[mask]
            else:
                w_masked = None

            # Compute validation metrics
            a1[idx] = eval_utils.a1_err(output_depth[mask], ground_truth[mask], threshold=1.05, weights=w_masked)
            a2[idx] = eval_utils.a2_err(output_depth[mask], ground_truth[mask], threshold=1.05, weights=w_masked)
            a3[idx] = eval_utils.a3_err(output_depth[mask], ground_truth[mask], threshold=1.05, weights=w_masked)
            abs_rel[idx] = eval_utils.abs_rel_err(output_depth[mask], ground_truth[mask], weights=w_masked)
            log_10_mae[idx] = eval_utils.log_10_mean_abs_err(output_depth[mask], ground_truth[mask], weights=w_masked)
            log_rmse[idx] = eval_utils.log_root_mean_sq_err(output_depth[mask], ground_truth[mask], weights=w_masked)
            mae[idx] = eval_utils.mean_abs_err(output_depth[mask], ground_truth[mask], weights=w_masked)
            rmse[idx] = eval_utils.root_mean_sq_err(output_depth[mask], ground_truth[mask], weights=w_masked)

    # Compute total time elapse in ms
    time_elapse = time_elapse * 1000.0

    if is_available_ground_truth:
        a1_mean = np.mean(a1)
        a2_mean = np.mean(a2)
        a3_mean = np.mean(a3)
        abs_rel_mean = np.mean(abs_rel)
        log_10_mae_mean = np.mean(log_10_mae)
        log_rmse_mean = np.mean(log_rmse)
        mae_mean = np.mean(mae)
        rmse_mean = np.mean(rmse)

        a1_std = np.std(a1)
        a2_std = np.std(a2)
        a3_std = np.std(a3)
        abs_rel_std = np.std(abs_rel)
        log_10_mae_std = np.std(log_10_mae)
        log_rmse_std = np.std(log_rmse)
        mae_std = np.std(mae)
        rmse_std = np.std(rmse)

        # Print evaluation results to console and file
        log('\nEvaluation results', log_path)
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

    # Log run time
    log('Completed processing {} samples'.format(n_sample), log_path)
    log('Total time: {:.2f} ms  Average time per sample: {:.2f} ms'.format(
        time_elapse, time_elapse / float(n_sample)), log_path)

def log_network_settings(log_path,
                         # Depth network settings
                         model_name,
                         min_predict_depth=-1.0,
                         max_predict_depth=-1.0,
                         # Weight settings
                         parameters_model=[],
                         restore_paths=[]):

    # Computer number of parameters
    n_parameter_model = sum(p.numel() for p in parameters_model)
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
                log('{:14}{}'.format('', path))

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
                        device=torch.device('cuda'),
                        n_thread=8):

    log('Checkpoint settings:', log_path)

    if checkpoint_path is not None:
        log('checkpoint_path={}'.format(checkpoint_path), log_path)
        log('', log_path)

    log('Hardware settings:', log_path)
    log('device={}'.format(device.type), log_path)
    log('n_thread={}'.format(n_thread), log_path)
    log('', log_path)
