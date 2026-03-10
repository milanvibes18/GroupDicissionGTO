import pandas as pd
import numpy as np
from scipy import stats
import pingouin as pg
from statsmodels.stats.multitest import multipletests


def run_stats():
    try:
        df = pd.read_csv("outputs/logs/real_benchmark_iterations.csv")
    except FileNotFoundError:
        print("Error: Could not find 'outputs/logs/real_benchmark_iterations.csv'.")
        return

    smart_iters = df['Smart_GTO'].values
    pso_iters = df['PSO'].values
    gwo_iters = df['GWO'].values
    static_iters = df['Static_GTO'].values

    def get_med_iqr(data):
        return f"{np.median(data):.1f} ({np.percentile(data, 25):.0f}-{np.percentile(data, 75):.0f})"

    print("\n" + "=" * 50)
    print(" 📊 TABLE 3 DATA (Medians & IQRs)")
    print("=" * 50)
    print(f"Smart GTO : {get_med_iqr(smart_iters)}")
    print(f"PSO       : {get_med_iqr(pso_iters)}")
    print(f"GWO       : {get_med_iqr(gwo_iters)}")
    print(f"Static GTO: {get_med_iqr(static_iters)}")

    # Wilcoxon Tests
    res_pso = pg.wilcoxon(smart_iters, pso_iters, alternative='two-sided')
    res_gwo = pg.wilcoxon(smart_iters, gwo_iters, alternative='two-sided')
    res_static = pg.wilcoxon(smart_iters, static_iters, alternative='two-sided')

    p_raw = [res_pso['p-val'].iloc[0], res_gwo['p-val'].iloc[0], res_static['p-val'].iloc[0]]
    _, p_holm, _, _ = multipletests(p_raw, alpha=0.05, method='holm')

    # Corrected Manual Bootstrap for Effect Size CI (r_rb)
    def get_rb_ci_manual(x, y):
        np.random.seed(42)
        rb_vals = []
        for _ in range(1000):
            idx = np.random.choice(len(x), len(x), replace=True)
            try:
                res = pg.wilcoxon(x[idx], y[idx], alternative='two-sided')
                rb_vals.append(abs(res['RBC'].iloc[0]))
            except:
                pass
        if not rb_vals: return 0.0, 0.0
        return np.percentile(rb_vals, 2.5), np.percentile(rb_vals, 97.5)

    ci_pso = get_rb_ci_manual(smart_iters, pso_iters)
    ci_gwo = get_rb_ci_manual(smart_iters, gwo_iters)
    ci_static = get_rb_ci_manual(smart_iters, static_iters)

    print("\n" + "=" * 50)
    print(" 📈 TABLE 4 DATA (Stats, p-values, Effect Sizes)")
    print("=" * 50)
    print(
        f"Smart vs PSO    | W={res_pso['W-val'].iloc[0]:.1f} | uncorr_p={p_raw[0]:.2e} | Holm_p={p_holm[0]:.2e} | r_rb={abs(res_pso['RBC'].iloc[0]):.2f} | 95% CI=[{ci_pso[0]:.2f}, {ci_pso[1]:.2f}]")
    print(
        f"Smart vs GWO    | W={res_gwo['W-val'].iloc[0]:.1f} | uncorr_p={p_raw[1]:.2e} | Holm_p={p_holm[1]:.2e} | r_rb={abs(res_gwo['RBC'].iloc[0]):.2f} | 95% CI=[{ci_gwo[0]:.2f}, {ci_gwo[1]:.2f}]")
    print(
        f"Smart vs Static | W={res_static['W-val'].iloc[0]:.1f} | uncorr_p={p_raw[2]:.2e} | Holm_p={p_holm[2]:.2e} | r_rb={abs(res_static['RBC'].iloc[0]):.2f} | 95% CI=[{ci_static[0]:.2f}, {ci_static[1]:.2f}]")

    # Median Difference CI for Abstract
    diff_pso = smart_iters - pso_iters
    res_boot = stats.bootstrap((diff_pso,), np.median, confidence_level=0.95, random_state=42, method='percentile')

    print("\n" + "=" * 50)
    print(" 📝 ABSTRACT DATA")
    print("=" * 50)
    med_diff = np.median(smart_iters) - np.median(pso_iters)
    reduction_pct = (abs(med_diff) / np.median(pso_iters)) * 100
    print(f"Reduction %       : {reduction_pct:.1f}%")
    print(f"Median Diff 95% CI: [{res_boot.confidence_interval.low:.1f}, {res_boot.confidence_interval.high:.1f}]")
    print("=" * 50)


if __name__ == "__main__":
    run_stats()