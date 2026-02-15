import torch
from log_utils import log

'''
Helper functions for logging
'''
def log_input_settings(log_path,
                       input_channels_image=None,
                       input_channels_depth=None,
                       normalized_image_range=None,
                       outlier_removal_kernel_size=None,
                       outlier_removal_threshold=None,
                       n_batch=None,
                       n_height=None,
                       n_width=None):

    batch_settings_text = ''
    batch_settings_vars = []

    if n_batch is not None:
        batch_settings_text = batch_settings_text + 'n_batch={}'
        batch_settings_vars.append(n_batch)

    batch_settings_text = \
        batch_settings_text + '  ' if len(batch_settings_text) > 0 else batch_settings_text

    if n_height is not None:
        batch_settings_text = batch_settings_text + 'n_height={}'
        batch_settings_vars.append(n_height)

    batch_settings_text = \
        batch_settings_text + '  ' if len(batch_settings_text) > 0 else batch_settings_text

    if n_width is not None:
        batch_settings_text = batch_settings_text + 'n_width={}'
        batch_settings_vars.append(n_width)

    log('Input settings:', log_path)

    if len(batch_settings_vars) > 0:
        log(batch_settings_text.format(*batch_settings_vars),
            log_path)

    if input_channels_image is not None or input_channels_depth is not None:
        log('input_channels_image={}  input_channels_depth={}'.format(
            input_channels_image, input_channels_depth),
            log_path)

    if normalized_image_range is not None:
        log('normalized_image_range={}'.format(normalized_image_range),
            log_path)

    if outlier_removal_kernel_size is not None and outlier_removal_threshold is not None:
        log('outlier_removal_kernel_size={}  outlier_removal_threshold={:.2f}'.format(
            outlier_removal_kernel_size, outlier_removal_threshold),
            log_path)
    log('', log_path)

def log_network_settings(log_path,
                         # Depth network settings
                         model_name,
                         mde_model_name,
                         min_predict_depth,
                         max_predict_depth,
                         # Pose network settings
                         encoder_type_pose=None,
                         rotation_parameterization_pose=None,
                         # Weight settings
                         parameters_depth_model=[],
                         parameters_pose_model=[],
                         parameters_secondary_model=[],
                         parameters_image_uncertainty_model=[]):

    # Computer number of parameters
    n_parameter_depth = sum(p.numel() for p in parameters_depth_model)
    n_parameter_pose = sum(p.numel() for p in parameters_pose_model)
    n_parameter_secondary = sum(p.numel() for p in parameters_secondary_model)
    n_parameter_image_uncertainty= sum(p.numel() for p in parameters_image_uncertainty_model)

    n_parameter = n_parameter_depth + n_parameter_pose + n_parameter_secondary + n_parameter_image_uncertainty

    n_parameter_text = 'n_parameter={}'.format(n_parameter)
    n_parameter_vars = []

    if n_parameter_depth > 0 :
        n_parameter_text = \
            n_parameter_text + '  ' if len(n_parameter_text) > 0 else n_parameter_text

        n_parameter_text = n_parameter_text + 'n_parameter_depth={}'
        n_parameter_vars.append(n_parameter_depth)

    if n_parameter_secondary > 0 :
        n_parameter_text = \
            n_parameter_text + '  ' if len(n_parameter_text) > 0 else n_parameter_text

        n_parameter_text = n_parameter_text + 'n_parameter_secondary={}'
        n_parameter_vars.append(n_parameter_secondary)

    if n_parameter_image_uncertainty > 0 :
        n_parameter_text = \
            n_parameter_text + '  ' if len(n_parameter_text) > 0 else n_parameter_text

        n_parameter_text = n_parameter_text + 'n_parameter_image_uncertainty={}'
        n_parameter_vars.append(n_parameter_image_uncertainty)

    if n_parameter_pose > 0 :
        n_parameter_text = \
            n_parameter_text + '  ' if len(n_parameter_text) > 0 else n_parameter_text

        n_parameter_text = n_parameter_text + 'n_parameter_pose={}'
        n_parameter_vars.append(n_parameter_pose)

    log('Depth network settings:', log_path)
    log('model_name={}  mde_model_name={}'.format(model_name, mde_model_name),
        log_path)
    log('min_predict_depth={:.2f}  max_predict_depth={:.2f}'.format(
        min_predict_depth, max_predict_depth),
        log_path)
    log('', log_path)

    if encoder_type_pose is not None:
        log('Pose network settings:', log_path)
        log('encoder_type_pose={}'.format(encoder_type_pose),
            log_path)
        log('rotation_parameterization_pose={}'.format(
            rotation_parameterization_pose),
            log_path)
        log('', log_path)

    log('Weight settings:', log_path)
    log(n_parameter_text.format(*n_parameter_vars),
        log_path)
    log('', log_path)

def log_training_settings(log_path,
                          # Training settings
                          n_batch,
                          n_train_sample,
                          n_train_step,
                          learning_rates,
                          learning_schedule,
                          # Augmentation settings
                          augmentation_probabilities,
                          augmentation_schedule,
                          # Photometric data augmentations
                          augmentation_random_brightness,
                          augmentation_random_contrast,
                          augmentation_random_gamma,
                          augmentation_random_hue,
                          augmentation_random_saturation,
                          augmentation_random_noise_type,
                          augmentation_random_noise_spread,
                          # Geometric data augmentations
                          augmentation_random_crop_type,
                          augmentation_random_crop_to_shape,
                          augmentation_random_resize_to_shape,
                          augmentation_random_flip_type,
                          augmentation_random_rotate_max,
                          augmentation_random_crop_and_pad,
                          augmentation_random_resize_and_crop,
                          augmentation_random_resize_and_pad,
                          # Occlusion data augmentations
                          augmentation_random_remove_patch_percent_range,
                          augmentation_random_remove_patch_size):

    log('Training settings:', log_path)
    log('n_sample={}  n_epoch={}  n_step={}'.format(
        n_train_sample, learning_schedule[-1], n_train_step),
        log_path)
    log('learning_schedule=[%s]' %
        ', '.join('{}-{} : {}'.format(
            ls * (n_train_sample // n_batch), le * (n_train_sample // n_batch), v)
            for ls, le, v in zip([0] + learning_schedule[:-1], learning_schedule, learning_rates)),
        log_path)
    log('', log_path)

    log('Augmentation settings:', log_path)
    log('augmentation_schedule=[%s]' %
        ', '.join('{}-{} : {}'.format(
            ls * (n_train_sample // n_batch), le * (n_train_sample // n_batch), v)
            for ls, le, v in zip([0] + augmentation_schedule[:-1], augmentation_schedule, augmentation_probabilities)),
        log_path)

    log('Photometric data augmentations:', log_path)
    log('augmentation_random_brightness={}'.format(augmentation_random_brightness),
        log_path)
    log('augmentation_random_contrast={}'.format(augmentation_random_contrast),
        log_path)
    log('augmentation_random_gamma={}'.format(augmentation_random_gamma),
        log_path)
    log('augmentation_random_hue={}'.format(augmentation_random_hue),
        log_path)
    log('augmentation_random_saturation={}'.format(augmentation_random_saturation),
        log_path)
    log('augmentation_random_noise_type={}  augmentation_random_noise_spread={}'.format(
        augmentation_random_noise_type, augmentation_random_noise_spread),
        log_path)

    log('Geometric data augmentations:', log_path)
    log('augmentation_random_crop_type={}'.format(augmentation_random_crop_type),
        log_path)
    log('augmentation_random_crop_to_shape={}'.format(augmentation_random_crop_to_shape),
        log_path)
    log('augmentation_random_resize_to_shape={}'.format(augmentation_random_resize_to_shape),
        log_path)
    log('augmentation_random_flip_type={}'.format(augmentation_random_flip_type),
        log_path)
    log('augmentation_random_rotate_max={}'.format(augmentation_random_rotate_max),
        log_path)
    log('augmentation_random_crop_and_pad={}'.format(augmentation_random_crop_and_pad),
        log_path)
    log('augmentation_random_crop_and_pad={}'.format(augmentation_random_crop_and_pad),
        log_path)
    log('augmentation_random_resize_and_crop={}'.format(augmentation_random_resize_and_crop),
        log_path)
    log('augmentation_random_resize_and_pad={}'.format(augmentation_random_resize_and_pad),
        log_path)

    log('augmentation_random_remove_patch_percent_range={}  augmentation_random_remove_patch_size={}'.format(
        augmentation_random_remove_patch_percent_range, augmentation_random_remove_patch_size),
        log_path)
    log('', log_path)

def log_loss_func_settings(log_path,
                           # Loss function settings
                           w_losses={},
                           w_weight_decay_depth=None,
                           w_weight_decay_pose=None):

    w_losses_text = ''
    for idx, (key, value) in enumerate(w_losses.items()):

        if idx > 0 and idx % 3 == 0:
            w_losses_text = w_losses_text + '\n'

        w_losses_text = w_losses_text + '{}={:.1e}'.format(key, value) + '  '

    log('Loss function settings:', log_path)
    if len(w_losses_text) > 0:
        log(w_losses_text, log_path)

    if w_weight_decay_depth is not None:
        log('w_weight_decay_depth={:.1e}'.format(
            w_weight_decay_depth),
            log_path)

    if w_weight_decay_pose is not None:
        log('w_weight_decay_pose={:.1e}'.format(
            w_weight_decay_pose),
            log_path)
    log('', log_path)

def log_evaluation_settings(log_path,
                            min_evaluate_depth,
                            max_evaluate_depth,
                            evaluation_protocol='default'):

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
                        n_step_per_checkpoint=None,
                        summary_event_path=None,
                        n_step_per_summary=None,
                        n_image_per_summary=None,
                        validation_start_step=None,
                        restore_paths=None,
                        restore_path_pose_model=None,
                        # Hardware settings
                        device=torch.device('cuda'),
                        n_thread=8):

    log('Checkpoint settings:', log_path)

    if checkpoint_path is not None:
        log('checkpoint_path={}'.format(checkpoint_path), log_path)

        if n_step_per_checkpoint is not None:
            log('checkpoint_save_frequency={}'.format(n_step_per_checkpoint), log_path)

        if validation_start_step is not None:
            log('validation_start_step={}'.format(validation_start_step), log_path)

        log('', log_path)

        summary_settings_text = ''
        summary_settings_vars = []

    if summary_event_path is not None:
        log('Tensorboard settings:', log_path)
        log('event_path={}'.format(summary_event_path), log_path)

    if n_step_per_summary is not None:
        summary_settings_text = summary_settings_text + 'log_summary_frequency={}'
        summary_settings_vars.append(n_step_per_summary)

        summary_settings_text = \
            summary_settings_text + '  ' if len(summary_settings_text) > 0 else summary_settings_text

    if n_image_per_summary is not None:
        summary_settings_text = summary_settings_text + 'n_image_per_summary={}'
        summary_settings_vars.append(n_image_per_summary)

        summary_settings_text = \
            summary_settings_text + '  ' if len(summary_settings_text) > 0 else summary_settings_text

    if len(summary_settings_text) > 0:
        log(summary_settings_text.format(*summary_settings_vars), log_path)

    if restore_paths is not None and restore_paths[0] != '':
        log('restore_paths_scaffnet_model={}'.format(restore_paths[0]),
            log_path)
        if len(restore_paths) == 2 and restore_paths[1] != '':
            log('restore_paths_depth_model={}'.format(restore_paths[1]),
            log_path)
        if len(restore_paths) == 3 and restore_paths[2] != '':
            log('restore_paths_image_uncertainty_model={}'.format(restore_paths[2]),
            log_path)

    if restore_path_pose_model is not None and restore_path_pose_model != '':
        log('restore_path_pose_model={}'.format(restore_path_pose_model),
            log_path)

    log('', log_path)

    log('Hardware settings:', log_path)
    log('device={}'.format(device.type), log_path)
    log('n_thread={}'.format(n_thread), log_path)
    log('', log_path)

def log_fusion_settings(log_path,
                        # Fusion Settings
                        fusion_settings,
                        fusion_parameters):
    log('fusion settings={}'.format(fusion_settings),
        log_path)
    log('fusion parameters={}'.format(fusion_parameters),
        log_path)
    if 'mix_depth' in fusion_settings:
        log("Combining sparse and dense depth for input to Encoder", log_path)
        if 'residual' not in fusion_settings:
            log("Training SpaDe", log_path)

    if 'residual' in fusion_settings:
        log("Training URL w/ alpha ={:.2f} beta ={:.2f}".format(fusion_parameters['alpha'],
                                                                 fusion_parameters['beta']),
                                                                 log_path)
    if 'dense_prop' in fusion_settings:
        log("NLSPN dense propagatiion w/ uncertainty < {:.2f}".format(fusion_parameters['uncertainty_threshold']),
                                                                      log_path)

def log_evaluation_results(title,
                           a1_mean,
                           a2_mean,
                           a3_mean,
                           abs_rel_mean,
                           log_10_mae_mean,
                           log_rmse_mean,
                           mae_mean,
                           rmse_mean,
                           a1_std,
                           a2_std,
                           a3_std,
                           abs_rel_std,
                           log_10_mae_std,
                           log_rmse_std,
                           mae_std,
                           rmse_std,
                           n_valid_points_output,
                           n_valid_points_ground_truth,
                           step=-1,
                           log_path=None):

    # Print evalulation results to console
    log('\n' + title + ':', log_path)

    log('\nEvaluation results', log_path)
    log('{:>8}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}  {:>14}  {:>14}'.format(
        'Step', 'A1', 'A2', 'A3', 'Abs_Rel', 'log10 MAE', 'log_RMSE', 'MAE', 'RMSE', '# Output', '# Ground truth'),
        log_path)
    log('{:8}  {:8.3f}  {:8.3f}  {:8.3f}  {:8.3f}  {:8.3f}  {:8.3f}  {:8.3f}  {:8.3f}  {:14.3f}  {:14.3f}'.format(
        step,
        a1_mean,
        a2_mean,
        a3_mean,
        abs_rel_mean,
        log_10_mae_mean,
        log_rmse_mean,
        mae_mean,
        rmse_mean,
        n_valid_points_output,
        n_valid_points_ground_truth),
        log_path)

    log('{:>8}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}'.format(
        '', '+/-', '+/-', '+/-', '+/-', '+/-', '+/-', '+/-', '+/-'),
        log_path)
    log('{:8}  {:8.3f}  {:8.3f}  {:8.3f}  {:8.3f}  {:8.3f}  {:8.3f}'.format(
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