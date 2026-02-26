"""
Experiment: Verify Online Pen Testing Lower Bounds

Simulates the Online Pen Testing problem to verify competitive ratio lower bounds
for sub-linear (0 < c < 1) and super-linear (c > 1) cost regimes.

Algorithms:
  1. Single-threshold algorithm (θ = τ_{1/n})
  2. Best-margin-gain algorithm (geometric quantile grid)
  Combined: fair coin flip between Algo 1 and Algo 2 each trial.

Usage:
    python experiment.py
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

matplotlib.use("Agg")
import sys
import time
import warnings

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_VALUES = [10, 50, 100, 500, 1000, 5000, 10_000]
C_SUB = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]
C_SUPER = [1.01, 1.05, 1.1, 1.5, 2, 3, 5, 10]
M = 50_000  # trials per (dist, n, c) combo

DISTRIBUTIONS = {
    "Uniform(0,1)": stats.uniform(loc=0, scale=1),
    "Exponential(1)": stats.expon(scale=1),
    "Pareto(α=2)": stats.pareto(b=2, scale=1),
    "LogNormal(0,1)": stats.lognorm(s=1, scale=np.exp(0)),
}

RNG = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def quantile(dist, alpha):
    """Return τ_α = F^{-1}(1 - α)."""
    return dist.ppf(1 - alpha)


def prophet_value(samples):
    """E[X^max] = mean of per-trial max over (M, n) sample array."""
    return samples.max(axis=1).mean()


def draw_samples(dist, n, m):
    """Draw (m, n) array of samples from *dist* using the global RNG."""
    return dist.rvs(size=(m, n), random_state=RNG)


# ---------------------------------------------------------------------------
# Algorithm 1: single threshold
# ---------------------------------------------------------------------------


def algo1_scores(samples, dist, n, c):
    """Vectorised single-threshold algorithm.

    θ = τ_{1/n}.  Accept first pen i with X_i > θ.
    Score = max(X_i - c·θ, 0) if accepted, else 0.
    """
    theta = quantile(dist, 1.0 / n)
    # (M, n) boolean mask of passing pens
    passing = samples > theta
    # index of first passing pen per trial (n if none)
    first_idx = np.argmax(passing, axis=1)
    any_pass = passing.any(axis=1)
    accepted_vals = samples[np.arange(len(samples)), first_idx]
    scores = np.where(any_pass, np.maximum(accepted_vals - c * theta, 0.0), 0.0)
    return scores


# ---------------------------------------------------------------------------
# Algorithm 2: best margin gain
# ---------------------------------------------------------------------------


def algo2_scores(samples, dist, n, c):
    """Vectorised best-margin-gain algorithm."""
    if c < 1:
        beta = 2.0
        k = max(int(np.ceil(np.log2(n))), 1)
    else:
        ln_n = np.log(n) if n > 1 else 1.0
        ln_c = np.log(c) if c > 1 else 1.0
        k = max(int(np.ceil(np.sqrt(ln_n / ln_c))), 1)
        beta = n ** (1.0 / k) if k > 0 else n

    # quantile levels α_j = β^{-j}, j = 0 … k
    js = np.arange(k + 1)
    alphas = beta ** (-js.astype(float))
    alphas = np.clip(alphas, 1e-15, 1.0 - 1e-15)
    a = np.array([quantile(dist, al) for al in alphas])  # thresholds a_j

    # best margin: j* = argmax_{j>=1} (a_j - c * a_{j-1})
    margins = a[1:] - c * a[:-1]  # length k
    j_star = np.argmax(margins) + 1  # 1-indexed

    theta = a[j_star - 1]  # threshold to cross
    target = a[j_star]  # target quantile value

    passing = samples > theta
    first_idx = np.argmax(passing, axis=1)
    any_pass = passing.any(axis=1)
    accepted_vals = samples[np.arange(len(samples)), first_idx]
    scores = np.where(any_pass, np.maximum(accepted_vals - c * theta, 0.0), 0.0)
    return scores


# ---------------------------------------------------------------------------
# Combined algorithm (fair coin flip)
# ---------------------------------------------------------------------------


def combined_scores(samples, dist, n, c):
    s1 = algo1_scores(samples, dist, n, c)
    s2 = algo2_scores(samples, dist, n, c)
    coin = RNG.integers(0, 2, size=len(s1))  # 0 or 1
    return np.where(coin == 0, s1, s2)


# ---------------------------------------------------------------------------
# Theoretical bounds
# ---------------------------------------------------------------------------


def theoretical_bound_sub(c, n):
    """Lower bound on competitive ratio for 0 < c < 1."""
    # Simplified Ω(min((1-c)/c, 1)) with leading constants
    ratio = (1 - c) / c
    return 0.5 * (1 - 1 / np.e) * min(ratio, 1.0)


def theoretical_bound_super(c, n):
    """Lower bound on competitive ratio for c > 1."""
    ln_n = np.log(n) if n > 1 else 1.0
    ln_c = np.log(c)
    return 1.0 / ((c / (c - 1)) * np.exp(2 * np.sqrt(ln_n * ln_c)))


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------


def run_experiments():
    results = []  # list of dicts

    total_combos = len(DISTRIBUTIONS) * len(N_VALUES) * (len(C_SUB) + len(C_SUPER))
    done = 0
    t0 = time.time()

    for dist_name, dist in DISTRIBUTIONS.items():
        for n in N_VALUES:
            # Draw samples once per (dist, n); reuse across c values
            samples = draw_samples(dist, n, M)
            prophet = prophet_value(samples)

            for c in C_SUB + C_SUPER:
                done += 1
                regime = "sub" if c < 1 else "super"

                comb = combined_scores(samples, dist, n, c)
                emp_val = comb.mean()
                emp_ratio = emp_val / prophet if prophet > 0 else 0.0

                if regime == "sub":
                    bound = theoretical_bound_sub(c, n)
                else:
                    bound = theoretical_bound_super(c, n)

                passed = emp_ratio >= bound * 0.95  # 5% tolerance for simulation noise

                results.append(
                    {
                        "dist": dist_name,
                        "n": n,
                        "c": c,
                        "regime": regime,
                        "prophet": prophet,
                        "emp_val": emp_val,
                        "emp_ratio": emp_ratio,
                        "bound": bound,
                        "pass": passed,
                    }
                )

                if done % 20 == 0 or done == total_combos:
                    elapsed = time.time() - t0
                    print(
                        f"  [{done}/{total_combos}] {elapsed:.1f}s  "
                        f"{dist_name} n={n} c={c} ratio={emp_ratio:.4f} "
                        f"bound={bound:.4f} {'PASS' if passed else 'FAIL'}"
                    )

    return results


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------


def print_table(results):
    print("\n" + "=" * 100)
    print(
        f"{'Distribution':<18} {'n':>7} {'c':>6} {'Regime':<6} "
        f"{'Prophet':>10} {'EmpVal':>10} {'EmpRatio':>10} {'Bound':>10} {'Status':>6}"
    )
    print("-" * 100)
    n_pass = n_fail = 0
    for r in results:
        status = "PASS" if r["pass"] else "FAIL"
        if r["pass"]:
            n_pass += 1
        else:
            n_fail += 1
        print(
            f"{r['dist']:<18} {r['n']:>7} {r['c']:>6.2f} {r['regime']:<6} "
            f"{r['prophet']:>10.4f} {r['emp_val']:>10.4f} {r['emp_ratio']:>10.6f} "
            f"{r['bound']:>10.6f} {status:>6}"
        )
    print("=" * 100)
    print(f"Total: {n_pass} PASS, {n_fail} FAIL out of {len(results)}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_ratio_vs_c(results):
    """Plot 1: Competitive ratio vs c for fixed n=1000, per distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for regime, c_vals, ax, bound_fn in [
        ("sub", C_SUB, axes[0], theoretical_bound_sub),
        ("super", C_SUPER, axes[1], theoretical_bound_super),
    ]:
        ax.set_title(
            f"{'Sub-linear (c < 1)' if regime == 'sub' else 'Super-linear (c > 1)'}"
        )
        ax.set_xlabel("c")
        ax.set_ylabel("Competitive Ratio")

        bounds = [bound_fn(c, 1000) for c in c_vals]
        ax.plot(c_vals, bounds, "k--", linewidth=2, label="Theoretical LB")

        for dist_name in DISTRIBUTIONS:
            ratios = []
            for c in c_vals:
                match = [
                    r
                    for r in results
                    if r["dist"] == dist_name and r["n"] == 1000 and r["c"] == c
                ]
                ratios.append(match[0]["emp_ratio"] if match else 0)
            ax.plot(c_vals, ratios, "o-", markersize=4, label=dist_name)

        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig("plot_ratio_vs_c.png", dpi=150)
    print("Saved plot_ratio_vs_c.png")
    plt.close(fig)


def plot_ratio_vs_n(results):
    """Plot 2: Competitive ratio vs n for representative c values."""
    rep_c_sub = [0.3, 0.5, 0.9]
    rep_c_super = [1.1, 2, 5]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for regime, c_rep, ax, bound_fn in [
        ("sub", rep_c_sub, axes[0], theoretical_bound_sub),
        ("super", rep_c_super, axes[1], theoretical_bound_super),
    ]:
        ax.set_title(
            f"{'Sub-linear (c < 1)' if regime == 'sub' else 'Super-linear (c > 1)'}"
        )
        ax.set_xlabel("n")
        ax.set_ylabel("Competitive Ratio")
        ax.set_xscale("log")

        for c in c_rep:
            # average across distributions
            ratios = []
            bounds = []
            for n in N_VALUES:
                matches = [r for r in results if r["n"] == n and r["c"] == c]
                ratios.append(
                    np.mean([m["emp_ratio"] for m in matches]) if matches else 0
                )
                bounds.append(bound_fn(c, n))
            ax.plot(N_VALUES, ratios, "o-", markersize=4, label=f"Empirical c={c}")
            ax.plot(N_VALUES, bounds, "--", linewidth=1.5, label=f"Bound c={c}")

        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig("plot_ratio_vs_n.png", dpi=150)
    print("Saved plot_ratio_vs_n.png")
    plt.close(fig)


def plot_heatmap(results):
    """Plot 3: Heatmap of pass/fail across distributions and c values."""
    all_c = C_SUB + C_SUPER
    dist_names = list(DISTRIBUTIONS.keys())
    # Fix n=1000 for the heatmap
    grid = np.zeros((len(dist_names), len(all_c)))

    for i, d in enumerate(dist_names):
        for j, c in enumerate(all_c):
            match = [
                r for r in results if r["dist"] == d and r["n"] == 1000 and r["c"] == c
            ]
            if match:
                grid[i, j] = 1.0 if match[0]["pass"] else 0.0

    fig, ax = plt.subplots(figsize=(12, 4))
    cmap = matplotlib.colors.ListedColormap(["#e74c3c", "#2ecc71"])
    ax.imshow(grid, cmap=cmap, aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(all_c)))
    ax.set_xticklabels([f"{c}" for c in all_c], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(dist_names)))
    ax.set_yticklabels(dist_names, fontsize=9)
    ax.set_xlabel("c")
    ax.set_title("Pass (green) / Fail (red) — n=1000, 50k trials")

    # annotate cells
    for i in range(len(dist_names)):
        for j in range(len(all_c)):
            ax.text(
                j,
                i,
                "P" if grid[i, j] else "F",
                ha="center",
                va="center",
                fontsize=7,
                fontweight="bold",
                color="white",
            )

    fig.tight_layout()
    fig.savefig("plot_heatmap.png", dpi=150)
    print("Saved plot_heatmap.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    print("=" * 60)
    print("Online Pen Testing — Lower Bound Verification")
    print(f"Trials per combo: {M}")
    print(f"n values: {N_VALUES}")
    print(f"c (sub-linear): {C_SUB}")
    print(f"c (super-linear): {C_SUPER}")
    print(f"Distributions: {list(DISTRIBUTIONS.keys())}")
    print("=" * 60)

    t_start = time.time()
    results = run_experiments()
    elapsed = time.time() - t_start
    print(f"\nExperiments completed in {elapsed:.1f}s")

    print_table(results)

    print("\nGenerating plots...")
    plot_ratio_vs_c(results)
    plot_ratio_vs_n(results)
    plot_heatmap(results)

    n_fail = sum(1 for r in results if not r["pass"])
    if n_fail == 0:
        print("\nAll experiments PASSED.")
    else:
        print(f"\n{n_fail} experiment(s) FAILED — inspect table above.")

    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
