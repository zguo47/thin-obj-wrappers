#!/usr/bin/env python3
import os
import sys
from pathlib import Path
import imageio.v3 as iio
import numpy as np


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def convert_float_to_uint16(img):
    """Normalize a float image (mode F) to uint16 range [0, 65535]."""
    img_min, img_max = np.nanmin(img), np.nanmax(img)
    if img_max > img_min:
        img_norm = (img - img_min) / (img_max - img_min)
        img_uint16 = (img_norm * 65535).astype(np.uint16)
    else:
        img_uint16 = np.zeros_like(img, dtype=np.uint16)
    return img_uint16


def tiff_to_png(input_dir, output_dir):
    """Convert all .tiff/.tif images in a directory to .png, handling float modes safely."""
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    ensure_dir(output_dir)

    tiff_files = sorted(list(input_dir.glob("*.tif")) + list(input_dir.glob("*.tiff")))
    if not tiff_files:
        print(f"[WARN] No TIFF files found in {input_dir}")
        return

    for f in tiff_files:
        try:
            img = iio.imread(f)
            if np.issubdtype(img.dtype, np.floating):
                img = convert_float_to_uint16(img)
            elif img.dtype == np.uint32:
                img = (img / 256).astype(np.uint16)  # convert 32-bit to 16-bit

            out_path = output_dir / (f.stem + ".png")
            iio.imwrite(out_path, img)
            print(f"[OK] {f.name} → {out_path.name} [{img.dtype}]")

        except Exception as e:
            print(f"[ERROR] Failed to convert {f.name}: {e}")

    print(f"[INFO] Done. Converted {len(tiff_files)} TIFF files to PNG.")


def main():
    if len(sys.argv) != 3:
        print("Usage: python tiff_to_png_safe.py <input_dir> <output_dir>")
        sys.exit(1)

    input_dir, output_dir = sys.argv[1], sys.argv[2]
    tiff_to_png(input_dir, output_dir)


if __name__ == "__main__":
    main()
