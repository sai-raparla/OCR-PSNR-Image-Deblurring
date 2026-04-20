import os
import csv
import argparse
import matplotlib.pyplot as plt

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv-path", required=True)
    p.add_argument("--out-dir", required=True)
    a = p.parse_args()

    if os.path.exists(a.out_dir) == False:
        os.makedirs(a.out_dir)

    ids = []
    blur_mse = []
    wiener_mse = []
    blur_mae = []
    wiener_mae = []
    blur_psnr = []
    wiener_psnr = []
    mse_drop = []
    psnr_gain = []

    f = open(a.csv_path, "r")
    r = csv.DictReader(f)

    for row in r:
        ids.append(row["id"])
        blur_mse.append(float(row["blur_mse"]))
        wiener_mse.append(float(row["wiener_mse"]))
        blur_mae.append(float(row["blur_mae"]))
        wiener_mae.append(float(row["wiener_mae"]))
        blur_psnr.append(float(row["blur_psnr"]))
        wiener_psnr.append(float(row["wiener_psnr"]))
        mse_drop.append(float(row["mse_drop"]))
        psnr_gain.append(float(row["psnr_gain"]))

    f.close()

    n = len(ids)

    if n == 0:
        print("no data")
        return

    total1 = 0.0
    total2 = 0.0
    total3 = 0.0
    total4 = 0.0
    total5 = 0.0
    total6 = 0.0

    i = 0
    while i < n:
        total1 = total1 + blur_mse[i]
        total2 = total2 + wiener_mse[i]
        total3 = total3 + blur_mae[i]
        total4 = total4 + wiener_mae[i]
        total5 = total5 + blur_psnr[i]
        total6 = total6 + wiener_psnr[i]
        i = i + 1

    avg_blur_mse = total1 / n
    avg_wiener_mse = total2 / n
    avg_blur_mae = total3 / n
    avg_wiener_mae = total4 / n
    avg_blur_psnr = total5 / n
    avg_wiener_psnr = total6 / n

    x = []
    i = 1
    while i <= n:
        x.append(i)
        i = i + 1

    plt.figure(figsize=(6, 4))
    plt.bar(["Blur", "Wiener"], [avg_blur_mse, avg_wiener_mse])
    plt.title("Average MSE")
    plt.ylabel("MSE")
    plt.savefig(os.path.join(a.out_dir, "avg_mse.png"), bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.bar(["Blur", "Wiener"], [avg_blur_mae, avg_wiener_mae])
    plt.title("Average MAE")
    plt.ylabel("MAE")
    plt.savefig(os.path.join(a.out_dir, "avg_mae.png"), bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.bar(["Blur", "Wiener"], [avg_blur_psnr, avg_wiener_psnr])
    plt.title("Average PSNR")
    plt.ylabel("PSNR")
    plt.savefig(os.path.join(a.out_dir, "avg_psnr.png"), bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(x, blur_psnr, label="Blur")
    plt.plot(x, wiener_psnr, label="Wiener")
    plt.title("PSNR Per Image")
    plt.xlabel("Image Number")
    plt.ylabel("PSNR")
    plt.legend()
    plt.savefig(os.path.join(a.out_dir, "psnr_per_image.png"), bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(x, mse_drop)
    plt.title("MSE Drop From Wiener")
    plt.xlabel("Image Number")
    plt.ylabel("MSE Drop")
    plt.savefig(os.path.join(a.out_dir, "mse_drop.png"), bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(x, psnr_gain)
    plt.title("PSNR Gain From Wiener")
    plt.xlabel("Image Number")
    plt.ylabel("PSNR Gain")
    plt.savefig(os.path.join(a.out_dir, "psnr_gain.png"), bbox_inches="tight")
    plt.close()

    print("saved graphs in", a.out_dir)
    print("avg blur mse:", avg_blur_mse)
    print("avg wiener mse:", avg_wiener_mse)
    print("avg blur mae:", avg_blur_mae)
    print("avg wiener mae:", avg_wiener_mae)
    print("avg blur psnr:", avg_blur_psnr)
    print("avg wiener psnr:", avg_wiener_psnr)

if __name__ == "__main__":
    main()
