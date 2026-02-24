import numpy as np
from scipy import stats
import pingouin as pg
from statsmodels.stats.multitest import multipletests

# ==========================================
# 1. PLUG IN YOUR ACTUAL RAW DATA HERE (30 runs each)
# ==========================================
# Example data for Iterations (replace with your exact 30 numbers)
smart_iters = np.array([21, 22, 20, 19, 23, 21, 20, 22, 24, 21, 20, 21, 19, 22, 21, 23, 20, 21, 22, 19, 20, 21, 23, 22, 20, 21, 24, 19, 21, 22])
pso_iters   = np.array([42, 45, 39, 41, 48, 43, 40, 44, 47, 42, 38, 41, 46, 42, 43, 49, 40, 41, 45, 39, 42, 44, 46, 41, 40, 43, 48, 39, 42, 45])
gwo_iters   = np.array([38, 39, 36, 35, 41, 38, 37, 40, 42, 38, 34, 37, 40, 39, 38, 43, 35, 38, 40, 34, 37, 39, 41, 38, 36, 39, 42, 35, 38, 40])
static_iters= np.array([32, 34, 30, 31, 36, 33, 31, 35, 37, 32, 29, 32, 35, 33, 32, 38, 30, 32, 34, 30, 31, 33, 36, 32, 31, 34, 37, 30, 32, 34])

# ==========================================
# 2. CALCULATE P-VALUES & EFFECT SIZES (Wilcoxon)
# ==========================================
print("--- RAW WILCOXON & EFFECT SIZES ---")
# Smart vs PSO
res_pso = pg.wilcoxon(smart_iters, pso_iters, alternative='two-sided')
p_pso = res_pso['p-val'][0]
rb_pso = res_pso['RBC'][0] # Rank-Biserial Correlation

# Smart vs GWO
res_gwo = pg.wilcoxon(smart_iters, gwo_iters, alternative='two-sided')
p_gwo = res_gwo['p-val'][0]
rb_gwo = res_gwo['RBC'][0]

# Smart vs Static
res_static = pg.wilcoxon(smart_iters, static_iters, alternative='two-sided')
p_static = res_static['p-val'][0]
rb_static = res_static['RBC'][0]

# ==========================================
# 3. APPLY HOLM CORRECTION (For Table 4)
# ==========================================
p_values = [p_pso, p_gwo, p_static]
reject, p_corrected, _, _ = multipletests(p_values, alpha=0.05, method='holm')

print(f"\n--- TABLE 4 DATA (Holm-Corrected p-values & Rank-Biserial) ---")
print(f"Smart vs PSO    -> p-value (Holm): {p_corrected[0]:.2e} | Effect Size: {abs(rb_pso):.2f}")
print(f"Smart vs GWO    -> p-value (Holm): {p_corrected[1]:.2e} | Effect Size: {abs(rb_gwo):.2f}")
print(f"Smart vs Static -> p-value (Holm): {p_corrected[2]:.2e} | Effect Size: {abs(rb_static):.2f}")

# ==========================================
# 4. CALCULATE BOOTSTRAP 95% CI (For Abstract & Text)
# ==========================================
# We want the 95% CI of the DIFFERENCE in Medians between Smart and PSO
diff_pso = smart_iters - pso_iters

# Bootstrap method
res_boot = stats.bootstrap((diff_pso,), np.median, confidence_level=0.95, random_state=42, method='percentile')
ci_low = res_boot.confidence_interval.low
ci_high = res_boot.confidence_interval.high

print(f"\n--- ABSTRACT DATA (95% CI for Median Difference) ---")
print(f"95% CI of difference: [{ci_low:.1f}, {ci_high:.1f}]")

# ==========================================
# 5. HOW TO CALCULATE ADVERSARIAL % CHANGES (Section 6.5)
# ==========================================
# Plug in your final median iterations from your Adversarial Agent experiment
normal_median_iters = np.median(smart_iters)
adversarial_median_iters = 24.5  # <--- Replace with result of 20% stubborn runs

percent_increase = ((adversarial_median_iters - normal_median_iters) / normal_median_iters) * 100

print(f"\n--- ADVERSARIAL DEGRADATION DATA ---")
print(f"Iterations increased by: {percent_increase:.1f}%")