import os
import csv
import re
import argparse
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
    return arr

def mse(a, b):
    x = a - b
    x = x * x
    return float(np.mean(x))

def mae(a, b):
    x = np.abs(a - b)
    return float(np.mean(x))

def psnr(a, b):
    m = mse(a, b)
    if m == 0:
        return 999.0
    v = 255.0 / np.sqrt(m)
    ans = 20 * np.log10(v)
    return float(ans)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--orig-dir", required=True)
    p.add_argument("--blur-dir", required=True)
    p.add_argument("--restored-dir", required=True)
    p.add_argument("--out-csv", required=True)
    p.add_argument("--limit", type=int, default=None)
    a = p.parse_args()

    folder = os.path.dirname(a.out_csv)
    if folder != "":
        if os.path.exists(folder) == False:
            os.makedirs(folder)

    names = os.listdir(a.blur_dir)
    names.sort()

    if a.limit != None:
        names = names[:a.limit]

    f = open(a.out_csv, "w", newline="")
    w = csv.writer(f)

    w.writerow([
        "id",
        "blur_mse",
        "wiener_mse",
        "blur_mae",
        "wiener_mae",
        "blur_psnr",
        "wiener_psnr",
        "mse_drop",
        "psnr_gain"
    ])

    count = 0

    total_blur_mse = 0.0
    total_wiener_mse = 0.0
    total_blur_mae = 0.0
    total_wiener_mae = 0.0
    total_blur_psnr = 0.0
    total_wiener_psnr = 0.0

    i = 0
    while i < len(names):
        blur_name = names[i]

        if blur_name.endswith(".png"):
            num = get_num(blur_name)

            if num != None:
                orig_name = num + "_orig.png"

                orig_path = os.path.join(a.orig_dir, orig_name)
                blur_path = os.path.join(a.blur_dir, blur_name)
                restored_path = os.path.join(a.restored_dir, blur_name)

                ok1 = os.path.exists(orig_path)
                ok2 = os.path.exists(blur_path)
                ok3 = os.path.exists(restored_path)

                if ok1 and ok2 and ok3:
                    orig = open_pic(orig_path)
                    blur = open_pic(blur_path)
                    restored = open_pic(restored_path)

                    blur_mse = mse(orig, blur)
                    wiener_mse = mse(orig, restored)

                    blur_mae = mae(orig, blur)
                    wiener_mae = mae(orig, restored)

                    blur_psnr = psnr(orig, blur)
                    wiener_psnr = psnr(orig, restored)

                    mse_drop = blur_mse - wiener_mse
                    psnr_gain = wiener_psnr - blur_psnr

                    w.writerow([
                        num,
                        blur_mse,
                        wiener_mse,
                        blur_mae,
                        wiener_mae,
                        blur_psnr,
                        wiener_psnr,
                        mse_drop,
                        psnr_gain
                    ])

                    total_blur_mse = total_blur_mse + blur_mse
                    total_wiener_mse = total_wiener_mse + wiener_mse
                    total_blur_mae = total_blur_mae + blur_mae
                    total_wiener_mae = total_wiener_mae + wiener_mae
                    total_blur_psnr = total_blur_psnr + blur_psnr
                    total_wiener_psnr = total_wiener_psnr + wiener_psnr

                    count = count + 1

        i = i + 1

    f.close()

    if count > 0:
        print("rows:", count)
        print("avg blur mse:", total_blur_mse / count)
        print("avg wiener mse:", total_wiener_mse / count)
        print("avg blur mae:", total_blur_mae / count)
        print("avg wiener mae:", total_wiener_mae / count)
        print("avg blur psnr:", total_blur_psnr / count)
        print("avg wiener psnr:", total_wiener_psnr / count)
        print("saved:", a.out_csv)
    else:
        print("no rows written")

if __name__ == "__main__":
    main()
