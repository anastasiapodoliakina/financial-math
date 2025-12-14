import numpy as np
import matplotlib.pyplot as plt

from market import CRASH_Z_WIN, CRASH_Z, FLASH_CRASH_ABS, crash_flags_zscore

def plot_sample_price_paths(price_paths, n_show=5):
    for s in ["short_gamma", "long_gamma", "no_hedge"]:
        plt.figure()
        for S in price_paths[s][:n_show]:
            plt.plot(S, alpha=0.7)
        plt.title(f"Sample price paths: {s} (n={n_show})")
        plt.xlabel("time")
        plt.ylabel("price")
        plt.show()

def plot_boxplots(results_df):
    plt.figure()
    results_df.boxplot(column="realized_vol", by="scenario")
    plt.title("Realized volatility by scenario")
    plt.suptitle("")
    plt.ylabel("Annualized vol")
    plt.show()

    plt.figure()
    results_df.boxplot(column="max_drawdown", by="scenario")
    plt.title("Max drawdown by scenario")
    plt.suptitle("")
    plt.ylabel("Max drawdown")
    plt.show()

def plot_crash_bars(results_df):
    plt.figure()
    order = ["short_gamma", "long_gamma", "no_hedge"]
    mz = results_df.groupby("scenario")["crashes_z"].mean().reindex(order)
    plt.bar(mz.index, mz.values)
    for i, v in enumerate(mz.values):
        plt.text(i, v, f"{v:.2f}", ha="center", va="bottom")
    plt.title(f"Average z-crashes per path (win={CRASH_Z_WIN}, z={CRASH_Z})")
    plt.ylabel("avg # z-crashes")
    plt.show()

    plt.figure()
    ma = results_df.groupby("scenario")["crashes_abs"].mean().reindex(order)
    plt.bar(ma.index, ma.values)
    for i, v in enumerate(ma.values):
        plt.text(i, v, f"{v:.2f}", ha="center", va="bottom")
    plt.title(f"Average abs-crashes per path (threshold={FLASH_CRASH_ABS})")
    plt.ylabel("avg # abs-crashes")
    plt.show()

def plot_return_histograms(return_paths, bins=60):
    for s in ["short_gamma", "long_gamma", "no_hedge"]:
        all_r = np.concatenate(return_paths[s])
        plt.figure()
        plt.hist(all_r, bins=bins, density=True)
        plt.title(f"Return distribution: {s}")
        plt.xlabel("log-return")
        plt.ylabel("density")
        plt.show()

def plot_dealer_gamma_example(gamma_paths):
    plt.figure()
    plt.plot(gamma_paths["short_gamma"][0], label="short_gamma")
    plt.plot(gamma_paths["long_gamma"][0], label="long_gamma")
    plt.title("Dealer net gamma (example path)")
    plt.xlabel("time")
    plt.ylabel("net gamma")
    plt.legend()
    plt.show()

def plot_zcrash_marks(price_paths, return_paths):
    for s in ["short_gamma", "long_gamma", "no_hedge"]:
        S = price_paths[s][0]
        r = return_paths[s][0]
        flags = crash_flags_zscore(r)
        idx = np.where(flags)[0] + 1

        plt.figure()
        plt.plot(S, label="price")
        if len(idx) > 0:
            plt.scatter(idx, S[idx], marker="x", label="z-crash")
        plt.title(f"Example path with z-crashes marked: {s}")
        plt.xlabel("time")
        plt.ylabel("price")
        plt.legend()
        plt.show()
