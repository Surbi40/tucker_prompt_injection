import json
from scipy.stats import wilcoxon
from src.utils.config import get_project_root

ROOT = get_project_root()

with open(ROOT / "results" / "tucker_vs_cp_ds1.json") as f:
    results = json.load(f)

tucker_f1 = [r["f1"] for r in results["tucker"]]
cp_f1 = [r["f1"] for r in results["cp"]]

print(f"Tucker per-fold F1: {tucker_f1}")
print(f"CP per-fold F1:     {cp_f1}")

diffs = [t - c for t, c in zip(tucker_f1, cp_f1)]
print(f"Per-fold differences (Tucker - CP): {diffs}")

stat, p_value = wilcoxon(tucker_f1, cp_f1)
print(f"\nWilcoxon signed-rank statistic: {stat}")
print(f"p-value: {p_value:.4f}")

if p_value < 0.05:
    print("Result: statistically significant difference (p < 0.05)")
else:
    print("Result: NOT statistically significant at p < 0.05")
    print("With only 5 folds, this is expected — Wilcoxon has very low power at n=5.")