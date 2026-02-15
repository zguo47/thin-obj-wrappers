#!/usr/bin/env python3
import os, sys
import numpy as np
from pathlib import Path

def read_noncomment_lines(p):
    lines = []
    with open(p, "r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            if ln.lstrip().startswith("#"):
                continue
            lines.append(ln.rstrip("\n"))
    return lines

def load_cameras_txt(p):
    # pass through payload (COLMAP allows comments)
    payload = [ln for ln in read_noncomment_lines(p) if ln.strip()]
    if not payload:
        raise RuntimeError("cameras.txt has no camera lines.")
    return payload

def load_images_txt(p):
    lines = [ln for ln in open(p, "r", encoding="utf-8", errors="ignore")]
    # drop comments, keep empties (we read in pairs)
    lines = [ln.rstrip("\n") for ln in lines if not ln.lstrip().startswith("#")]
    out = []
    i = 0
    while i < len(lines):
        header = lines[i].strip()
        if not header:
            i += 1
            continue
        parts = header.split()
        if len(parts) < 10:
            # skip malformed
            i += 1
            continue
        img_id = int(parts[0])
        qw, qx, qy, qz = map(float, parts[1:5])
        tx, ty, tz = map(float, parts[5:8])
        cam_id = int(parts[8])
        name   = parts[9]
        out.append((img_id, qw, qx, qy, qz, tx, ty, tz, cam_id, name))
        # skip the following line (2D points); we will rewrite a blank one
        i += 2 if i+1 < len(lines) else 1
    return out

def write_cameras_txt(out_path, camera_lines):
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write("# Number of cameras: {}\n".format(len(camera_lines)))
        for ln in camera_lines:
            f.write(ln.strip() + "\n")

def write_images_txt(out_path, records):
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write("# Number of images: {}\n".format(len(records)))
        # sort by input id and reindex to 1..N
        records = sorted(records, key=lambda x: x[0])
        for new_id, (old_id, qw,qx,qy,qz, tx,ty,tz, cam_id, name) in enumerate(records, start=1):
            q = np.array([qx,qy,qz,qw], dtype=np.float64)
            n = np.linalg.norm(q)
            if n == 0:
                q = np.array([0,0,0,1], dtype=np.float64)
            else:
                q = q / n
            x,y,z,w = q  # note: will write as QW QX QY QZ
            f.write(f"{new_id} {w:.17g} {x:.17g} {y:.17g} {z:.17g} {tx:.17g} {ty:.17g} {tz:.17g} {cam_id} {name}\n")
            f.write("\n")  # empty 2D line

def write_points3D_header(out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write("# Number of points: 0\n")

def main():
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print("Usage: python fix_colmap_txt_no_invert.py <in_dir> <out_dir> [--keep-points]")
        sys.exit(1)
    in_dir = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    keep_points = (len(sys.argv) == 4 and sys.argv[3] == "--keep-points")
    out_dir.mkdir(parents=True, exist_ok=True)

    cams_in = in_dir / "cameras.txt"
    imgs_in = in_dir / "images.txt"
    pts_in  = in_dir / "points3D.txt"

    cams = load_cameras_txt(cams_in)
    imgs = load_images_txt(imgs_in)

    write_cameras_txt(out_dir / "cameras.txt", cams)
    write_images_txt(out_dir / "images.txt", imgs)

    if keep_points and pts_in.exists():
        with open(pts_in, "r", encoding="utf-8", errors="ignore") as rf, \
             open(out_dir / "points3D.txt", "w", encoding="utf-8") as wf:
            wf.write(rf.read())
    else:
        write_points3D_header(out_dir / "points3D.txt")

    print(f"[OK] Wrote strict COLMAP TXT (w2c assumed) → {out_dir}")

if __name__ == "__main__":
    main()
