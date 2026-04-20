import csv
import os
import matplotlib.pyplot as plt

def get_avg(path):
    f = open(path, "r")
    r = csv.DictReader(f)

    total_mse = 0.0
    total_mae = 0.0
    total_psnr = 0.0
    count = 0

    for row in r:
        total_mse = total_mse + float(row["wiener_mse"])
        total_mae = total_mae + float(row["wiener_mae"])
        total_psnr = total_psnr + float(row["wiener_psnr"])
        count = count + 1

    f.close()

    if count == 0:
        return None

    out = {}
    out["mse"] = total_mse / count
    out["mae"] = total_mae / count
    out["psnr"] = total_psnr / count
    return out

def main():
    a1 = get_avg("classical/results/rl_metrics_t1.csv")
    a2 = get_avg("classical/results/rl_metrics_t2.csv")
    a3 = get_avg("classical/results/rl_metrics_t3.csv")
    a5 = get_avg("classical/results/rl_metrics_t5.csv")
    a10 = get_avg("classical/results/rl_metrics_50.csv")

    if a1 == None or a2 == None or a3 == None or a5 == None or a10 == None:
        print("missing csv")
        return

    os.makedirs("classical/results/rl_iter_graphs", exist_ok=True)

    names = ["1", "2", "3", "5", "10"]

    mse_vals = [a1["mse"], a2["mse"], a3["mse"], a5["mse"], a10["mse"]]
    mae_vals = [a1["mae"], a2["mae"], a3["mae"], a5["mae"], a10["mae"]]
    psnr_vals = [a1["psnr"], a2["psnr"], a3["psnr"], a5["psnr"], a10["psnr"]]

    plt.figure(figsize=(6,4))
    plt.bar(names, mse_vals)
    plt.title("RL MSE for Different Iterations")
    plt.xlabel("iterations")
    plt.ylabel("MSE")
    plt.savefig("classical/results/rl_iter_graphs/rl_iter_mse.png", bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(6,4))
    plt.bar(names, mae_vals)
    plt.title("RL MAE for Different Iterations")
    plt.xlabel("iterations")
    plt.ylabel("MAE")
    plt.savefig("classical/results/rl_iter_graphs/rl_iter_mae.png", bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(6,4))
    plt.bar(names, psnr_vals)
    plt.title("RL PSNR for Different Iterations")
    plt.xlabel("iterations")
    plt.ylabel("PSNR")
    plt.savefig("classical/results/rl_iter_graphs/rl_iter_psnr.png", bbox_inches="tight")
    plt.close()

    print("iter 1")
    print("mse:", a1["mse"])
    print("mae:", a1["mae"])
    print("psnr:", a1["psnr"])

    print("iter 2")
    print("mse:", a2["mse"])
    print("mae:", a2["mae"])
    print("psnr:", a2["psnr"])

    print("iter 3")
    print("mse:", a3["mse"])
    print("mae:", a3["mae"])
    print("psnr:", a3["psnr"])

    print("iter 5")
    print("mse:", a5["mse"])
    print("mae:", a5["mae"])
    print("psnr:", a5["psnr"])

    print("iter 10")
    print("mse:", a10["mse"])
    print("mae:", a10["mae"])
    print("psnr:", a10["psnr"])

    print("saved graphs in classical/results/rl_iter_graphs")

if __name__ == "__main__":
    main()
