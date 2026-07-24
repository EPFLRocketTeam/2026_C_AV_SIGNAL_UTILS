import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def main():
    # Load dataset
    df = pd.read_csv("output.csv")
    
    dev_cols  = [c for c in df.columns if c.startswith("dev_")]
    keep_cols = [c for c in df.columns if c.startswith("keep_")]
    times     = df["time"].values

    # Sort the kept values across each timestep row-wise
    kept_matrix = df[keep_cols].values
    kept_sorted = np.sort(kept_matrix, axis=1)

    algo_mean = df["algo_mean"].values
    raw_mean  = df["raw_mean"].values
    true_mean = df["true_mean"].values

    # Calculate errors relative to True Mean (1.0)
    del_raw  = raw_mean - true_mean
    del_algo = algo_mean - true_mean
    err_raw  = np.abs(raw_mean - true_mean)   # With glitched devices
    err_algo = np.abs(algo_mean - true_mean)  # Filtered algorithm error

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1, 1]})

    # -------------------------------------------------------------
    # Top Plot: Signals and Mean Estimators
    # -------------------------------------------------------------
    # 1. Raw signals (Opacity: 0.2)
    for i, col in enumerate(dev_cols):
        ax1.plot(
            times, df[col], 
            color="lightgray", alpha=0.2, 
            label="Raw Signals (25% U[-9.0, 11.0] Glitches)" if i == 0 else ""
        )

    # 2. Sorted kept signals (Opacity: 0.5)
    for k in range(kept_sorted.shape[1]):
        ax1.plot(
            times, kept_sorted[:, k], 
            color="#1f77b4", alpha=0.5, 
            label="Kept Signals (Sorted)" if k == 0 else ""
        )

    # 3. Raw Mean (With Outliers)
    ax1.plot(
        times, raw_mean, 
        color="#ff7f0e", alpha=0.8, linewidth=1.5, linestyle="--",
        label="Raw Mean (Includes Glitches)"
    )

    # 4. Algorithm Mean (Filtered)
    ax1.plot(
        times, algo_mean, 
        color="#d62728", alpha=1.0, linewidth=2.0, 
        label="Algorithm Mean (Variance Filtered)"
    )

    # 5. True Ground Truth Mean
    ax1.axhline(1.0, color="black", linestyle=":", linewidth=1.5, label="True Mean (1.0)")

    ax1.set_title("Signal Comparison: 25% Uniform[-9.0, 11.0] Glitches vs. Variance-Filtered Estimator")
    ax1.set_ylabel("Signal Value")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right")

    # -------------------------------------------------------------
    # Bottom Plot: Absolute Estimation Errors
    # -------------------------------------------------------------
    ax2.plot(
        times, err_raw, 
        color="#ff7f0e", alpha=0.7, linewidth=1.5, 
        label=f"Error With Outliers (Mean: {np.mean(err_raw):.4f})"
    )

    ax2.plot(
        times, err_algo, 
        color="#d62728", alpha=0.9, linewidth=1.8, 
        label=f"Error Without Outliers (Mean: {np.mean(err_algo):.4f})"
    )

    ax2.set_title(r"Absolute Estimation Error $| \hat{\mu} - \mu_{\text{true}} |$")
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("Absolute Error")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right")

    ax3.plot(
        times, del_raw, 
        color="#ff7f0e", alpha=0.7, linewidth=1.5, 
        label=f"Delta With Outliers"
    )

    ax3.plot(
        times, del_algo, 
        color="#d62728", alpha=0.9, linewidth=1.8, 
        label=f"Delta Without Outliers"
    )

    ax3.set_title(r"Estimation Delta $\hat{\mu} - \mu_{\text{true}}$")
    ax3.set_xlabel("Time Step")
    ax3.set_ylabel("Absolute Error")
    ax3.grid(True, linestyle="--", alpha=0.5)
    ax3.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig("outlier_asymmetric_glitch_comparison.png", dpi=150)
    plt.show()

if __name__ == "__main__":
    main()