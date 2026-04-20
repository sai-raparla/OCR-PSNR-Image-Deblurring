import argparse
import os
import re
import numpy as np
from PIL import Image

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
    files = os.listdir(folder)

    for name in files:
        low = name.lower()
        if low.endswith(".png"):
            num = get_num(name)
            if num != None:
                d[num] = name

    return d

def psf_to_big(psf, shape):
    h = shape[0]
    w = shape[1]

    ph = psf.shape[0]
    pw = psf.shape[1]

    big = np.zeros((h, w), dtype=np.float32)
    big[0:ph, 0:pw] = psf

    big = np.roll(big, -(ph // 2), axis=0)
    big = np.roll(big, -(pw // 2), axis=1)

    return np.fft.fft2(big)

def do_wiener(blur, psf, k):
    s = psf.sum()
    if s != 0:
        psf = psf / s

    g = np.fft.fft2(blur)
    h = psf_to_big(psf, blur.shape)

    top = np.conj(h)
    bottom = (np.abs(h) ** 2) + k

    f = (top / bottom) * g
    out = np.fft.ifft2(f)
    out = np.real(out)
    out = np.clip(out, 0.0, 1.0)

    return out

def run_all(blur_dir, psf_dir, out_dir, limit=None, k=0.01):
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

            out = do_wiener(blur, psf, k)
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
    p.add_argument("--k", type=float, default=0.01)

    a = p.parse_args()

    run_all(a.blur_dir, a.psf_dir, a.output_dir, a.limit, a.k)

if __name__ == "__main__":
    main()
