import argparse
import cv2
import glob
import matplotlib
import numpy as np
import os
import torch
import pdb

from depth_anything_v2.dpt import DepthAnythingV2

'''
THIS DOES NOT CREATE ANY DEPTH MAPS, IT JUST WRITES THE FILENAMES TO A TEXT FILE
i.e., the depth maps may not even exist!!
'''


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2')

    parser.add_argument('--img-path', type=str)
    parser.add_argument('--lidar-path', type=str)
    parser.add_argument('--store-path', type=str)
    # parser.add_argument('--input-size', type=int, default=900)
    # parser.add_argument('--outdir', type=str, default='./vis_depth')

    # parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])

    # parser.add_argument('--pred-only', dest='pred_only', action='store_true', help='only display the prediction')
    # parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='do not apply colorful palette')

    args = parser.parse_args()

    # DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    # model_configs = {
    #     'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    #     'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    #     'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    #     'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    # }

    # depth_anything = DepthAnythingV2(**model_configs[args.encoder])
    # depth_anything.load_state_dict(torch.load(f'external_src/Depth-Anything-V2/checkpoints/depth_anything_v2_{args.encoder}.pth', map_location='cpu'))
    # depth_anything = depth_anything.to(DEVICE).eval()

    if os.path.isfile(args.img_path):
        if args.img_path.endswith('txt'):
            with open(args.img_path, 'r') as f:
                filenames = f.read().splitlines()
        else:
            filenames = [args.img_path]
    else:
        filenames = glob.glob(os.path.join(args.img_path, '**/*'), recursive=True)

    with open(args.lidar_path, 'r') as f:
        lidar_filenames = f.read().splitlines()

    cmap = matplotlib.colormaps.get_cmap('Spectral_r')
    save_dirs = []
    for k, (filename, lidar_filename) in enumerate(zip(filenames, lidar_filenames)):
        #save_dirs.append(lidar_filename.replace('lidar', 'depthanythingv2_518').replace('png', 'npy'))
        save_dirs.append(lidar_filename.replace('lidar', 'depthanythingv2_518_patrick').replace('png', 'npy'))

    split = 'train' if 'train' in args.img_path else 'val'
    #txt_file_write = open('nuscenes_{}_mde_prediction_518'.format(split), 'w')
    txt_file_write = open(os.path.join(args.store_path, 'nuscenes_{}_mde_prediction_518_patrick'.format(split)), 'w')

    for txt in save_dirs:
        txt_file_write.write(txt + '\n')
    txt_file_write.close()
    
    

        # pdb.set_trace()

        # os.makedirs(os.path.split(save_dir)[0], exist_ok=True)

        # if args.pred_only:
        #     np.save(save_dir, depth)
        # else:
        #     split_region = np.ones((raw_image.shape[0], 50, 3), dtype=np.uint8) * 255
        #     combined_result = cv2.hconcat([raw_image, split_region, depth])

        #     cv2.imwrite(os.path.join(save_dir, os.path.splitext(os.path.basename(filename))[0] + '.png'), combined_result)