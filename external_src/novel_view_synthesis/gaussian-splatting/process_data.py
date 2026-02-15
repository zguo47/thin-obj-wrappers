#!/usr/bin/env python3
import os, sys, shutil
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R
import cv2

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def load_intrinsics(intr_path, w, h):
    K = np.load(intr_path)
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]
    # COLMAP cameras.txt format:
    # CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]
    return f"1 PINHOLE {w} {h} {fx} {fy} {cx} {cy}\n"

def convert_trajectory(traj_path, n_imgs):
    """Convert ORB-SLAM3 CameraTrajectory.txt (c2w) -> COLMAP images.txt (w2c)."""
    lines = [ln.strip() for ln in open(traj_path) if ln.strip() and not ln.startswith("#")]
    out = []
    for i, line in enumerate(lines):
        vals = list(map(float, line.split()))
        if len(vals) != 8:
            continue
        t, tx, ty, tz, qx, qy, qz, qw = vals

        # ORB-SLAM3 outputs T_c2w -> invert for COLMAP
        q = np.array([qx, qy, qz, qw])
        q /= np.linalg.norm(q)
        R_c2w = R.from_quat([qx, qy, qz, qw]).as_matrix()
        T_c2w = np.eye(4); T_c2w[:3,:3] = R_c2w; T_c2w[:3,3] = [tx,ty,tz]
        T_w2c = np.linalg.inv(T_c2w)

        # back to quaternion
        R_w2c = R.from_matrix(T_w2c[:3,:3]).as_quat()  # [x,y,z,w]
        qx, qy, qz, qw = R_w2c
        tx, ty, tz = T_w2c[:3,3]

        img_name = f"{i+1:06d}.png"
        # COLMAP images.txt requires 2 lines per image
        header_line = f"{i+1} {qw} {qx} {qy} {qz} {tx} {ty} {tz} 1 {img_name}\n"
        pts2d_line = "0 0 0\n"  # dummy 2D-3D matches
        out.append(header_line + pts2d_line)

    return out[:n_imgs]

def process_scene(scene_root, out_root):
    scene_root = Path(scene_root)
    scene_name = scene_root.name
    out_scene = Path(out_root) / scene_name

    print(f"[INFO] Processing scene: {scene_name}")
    ensure_dir(out_scene / "images")
    ensure_dir(out_scene / "depth")
    ensure_dir(out_scene / "sparse/0")

    # copy & rename RGB images
    img_dir = scene_root / "data/image"
    img_files = sorted(list(img_dir.glob("*.png")))
    for i, f in enumerate(img_files):
        out_path = out_scene / "images" / f"{i+1:06d}.png"
        shutil.copy(f, out_path)

    # copy & rename depth maps
    dpt_dir = scene_root / "data/depth"
    dpt_files = sorted(list(dpt_dir.glob("*.png")))
    for i, f in enumerate(dpt_files):
        out_path = out_scene / "depth" / f"{i+1:06d}.png"
        shutil.copy(f, out_path)

    # intrinsics
    h, w = cv2.imread(str(img_files[0])).shape[:2]
    cam_line = load_intrinsics(scene_root / "data/intrinsics.npy", w, h)
    with open(out_scene / "sparse/0/cameras.txt", "w") as f:
        f.write(cam_line)

    # trajectory -> images.txt
    traj_path = scene_root / "CameraTrajectory.txt"
    img_lines = convert_trajectory(traj_path, len(img_files))
    with open(out_scene / "sparse/0/images.txt", "w") as f:
        for ln in img_lines:
            f.write(ln)

    # empty points3D.txt
    open(out_scene / "sparse/0/points3D.txt", "w").close()

    print(f"[INFO] Scene {scene_name} processed → {out_scene}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python process_data.py <original_data_root>")
        sys.exit(1)

    data_root = Path(sys.argv[1])
    out_root = Path(str(data_root) + "_processed")
    ensure_dir(out_root)

    for scene in data_root.iterdir():
        if scene.is_dir() and (scene / "CameraTrajectory.txt").exists():
            process_scene(scene, out_root)

if __name__ == "__main__":
    main()
