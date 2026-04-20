import argparse
import os
import re
import numpy as np
from PIL import Image
from skimage.restoration import richardson_lucy

def get_num(name):
    m = re.match(r"(\d+)", name)
    if m:
        return m.group(1)
    else:
        return None

def open_pic(path):
    img = Image.open(path).convert("L")
    arr = np.array(img).astype(np.float32)
    arr = arr / 255.0
    return arr

def save_pic(arr, path):
    arr = np.clip(arr, 0.0, 1.0)
    arr = arr * 255.0
    arr = arr.astype(np.uint8)
    img = Image.fromarray(arr)

    folder = os.path.dirname(path)
    if folder != "":
        if os.path.exists(folder) == False:
            os.makedirs(folder)

    img.save(path)

def make_map(folder):
    d = {}
    names = os.listdir(folder)

    i = 0
    while i < len(names):
        name = names[i]

        if name.lower().endswith(".png"):
            num = get_num(name)
            if num != None:
                d[num] = name

        i = i + 1

    return d

def do_rl(blur, psf, times):
    s = psf.sum()
    if s != 0:
        psf = psf / s

    out = richardson_lucy(blur, psf, num_iter=times)
    out = np.clip(out, 0.0, 1.0)
    return out

def run_all(blur_dir, psf_dir, out_dir, limit=None, times=10):
    blur_files = make_map(blur_dir)
    psf_files = make_map(psf_dir)

    nums = []

    for num in blur_files:
        if num in psf_files:
            nums.append(num)

    nums.sort()

    if limit != None:
        nums = nums[:limit]

    print("Found", len(nums), "matching blur/psf pairs")

    if len(nums) == 0:
        print("No matching files found")
        return

    i = 0

    while i < len(nums):
        num = nums[i]

        blur_name = blur_files[num]
        psf_name = psf_files[num]

        blur_path = os.path.join(blur_dir, blur_name)
        psf_path = os.path.join(psf_dir, psf_name)
        out_path = os.path.join(out_dir, blur_name)

        try:
            blur = open_pic(blur_path)
            psf = open_pic(psf_path)

            out = do_rl(blur, psf, times)
            save_pic(out, out_path)

            print("[", i + 1, "/", len(nums), "] saved", out_path)

        except Exception as e:
            print("Skipping", blur_name)
            print(e)

        i = i + 1

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--blur-dir", required=True)
    p.add_argument("--psf-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--times", type=int, default=10)

    a = p.parse_args()

    run_all(a.blur_dir, a.psf_dir, a.output_dir, a.limit, a.times)

if __name__ == "__main__":
    main()
