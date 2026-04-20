import csv
import os
import matplotlib.pyplot as plt

def get_avg(path):
    f = open(path, "r")
    r = csv.DictReader(f)

    total_blur_mse = 0.0
    total_wiener_mse = 0.0
    total_blur_mae = 0.0
    total_wiener_mae = 0.0
    total_blur_psnr = 0.0
    total_wiener_psnr = 0.0
    count = 0

    for row in r:
        total_blur_mse = total_blur_mse + float(row["blur_mse"])
        total_wiener_mse = total_wiener_mse + float(row["wiener_mse"])
        total_blur_mae = total_blur_mae + float(row["blur_mae"])
        total_wiener_mae = total_wiener_mae + float(row["wiener_mae"])
        total_blur_psnr = total_blur_psnr + float(row["blur_psnr"])
        total_wiener_psnr = total_wiener_psnr + float(row["wiener_psnr"])
        count = count + 1

    f.close()

    if count == 0:
        return None

    out = {}
    out["blur_mse"] = total_blur_mse / count
    out["wiener_mse"] = total_wiener_mse / count
    out["blur_mae"] = total_blur_mae / count
    out["wiener_mae"] = total_wiener_mae / count
    out["blur_psnr"] = total_blur_psnr / count
    out["wiener_psnr"] = total_wiener_psnr / count
    return out

def main():
    p1 = "classical/results/wiener_metrics_k_0001.csv"
    p2 = "classical/results/wiener_metrics_k_001.csv"
    p3 = "classical/results/wiener_metrics_k_01.csv"

    a1 = get_avg(p1)
    a2 = get_avg(p2)
    a3 = get_avg(p3)

    if a1 == None or a2 == None or a3 == None:
        print("missing csv data")
        return

    os.makedirs("classical/results/k_graphs", exist_ok=True)

    names = ["0.001", "0.01", "0.1"]

    mse_vals = [a1["wiener_mse"], a2["wiener_mse"], a3["wiener_mse"]]
    mae_vals = [a1["wiener_mae"], a2["wiener_mae"], a3["wiener_mae"]]
    psnr_vals = [a1["wiener_psnr"], a2["wiener_psnr"], a3["wiener_psnr"]]

    plt.figure(figsize=(6,4))
    plt.bar(names, mse_vals)
    plt.title("Wiener MSE for Different k")
    plt.xlabel("k")
    plt.ylabel("MSE")
    plt.savefig("classical/results/k_graphs/k_mse.png", bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(6,4))
    plt.bar(names, mae_vals)
    plt.title("Wiener MAE for Different k")
    plt.xlabel("k")
    plt.ylabel("MAE")
    plt.savefig("classical/results/k_graphs/k_mae.png", bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(6,4))
    plt.bar(names, psnr_vals)
    plt.title("Wiener PSNR for Different k")
    plt.xlabel("k")
    plt.ylabel("PSNR")
    plt.savefig("classical/results/k_graphs/k_psnr.png", bbox_inches="tight")
    plt.close()

    print("k 0.001")
    print("mse:", a1["wiener_mse"])
    print("mae:", a1["wiener_mae"])
    print("psnr:", a1["wiener_psnr"])

    print("k 0.01")
    print("mse:", a2["wiener_mse"])
    print("mae:", a2["wiener_mae"])
    print("psnr:", a2["wiener_psnr"])

    print("k 0.1")
    print("mse:", a3["wiener_mse"])
    print("mae:", a3["wiener_mae"])
    print("psnr:", a3["wiener_psnr"])

    print("saved graphs in classical/results/k_graphs")

if __name__ == "__main__":
    main()
