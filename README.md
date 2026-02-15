# vision-lab-wrappers

run
```
pip install -r requirements.txt
```

if you want to run depth-any-camera
```
python external_src/monocular_depth/depth_any_camera/dac/models/ops/setup.py build
```

OMNI-DC

Please follow the virtualenv setup here:
https://github.com/princeton-vl/OMNI-DC/tree/main

Download the Depth Anything checkpoint (https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true) to 'external_src/depth_completion/OMNI-DC/src/depth_models/depth_anything_v2/checkpoints/' .

Thin-Object Baselines
```
src
|__depth_completion
   |__OMNIDC
|__monocular_depth
   |__DepthAnythingV2
   |__DepthAnythingV1
   |__UniDepthV1
   |__UniDepthV2(to be fixed)
   |__Marigold(TODO)
   |__Metric3D(TODO)
   |__DepthAnythingV3(TODO)
|__multiview
   |__VGGT
   |__MapAnything(TODO)
|__novel_view_synthesis
   |__2D_Gaussian_Splatting
   |__3D_Gaussian_Splatting
   |__Nerfacto
   |__Depth-Nerfacto
```

Usage Examples

Processing single sequence
```
bash bash/run_depthanythingv2-thin_obj.sh
```
Processing batches
```
bash bash/run_gaussian_splatting_2d.sh <root containing scenes> <output root>
```

Creating New Baseline Wrapper Models

Example: Monocular Depth Model
```
# Wrapper Class for all mde models
src/monocular_depth/external_mde_model.py

# Calls all mde models, run estimation and save evaluation results
src/monocular_depth/external_mde_model.py

# Main function and arg parsers
src/monocular_depth/run_external_mde_model.py

# Utility functions, including depth loading and saving
src/monocular_depth/data_utils.py

# Utility functions for evaluation
src/monocular_depth/eval_utils.py
```
