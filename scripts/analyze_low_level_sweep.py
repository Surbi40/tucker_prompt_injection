import json
import numpy as np
from scipy.stats import wilcoxon
from src.utils.config import get_project_root

ROOT = get_project_root()

with open(ROOT / "results" / "low_label_sweep_ds1.json") as f:
    results = json.load(f)

FRACTIONS = ["0.2", "0.1", "0.05", "0.02", "0.01", "0.005"]
LOW_LABEL_FOCUS = ["0.02", "0.01", "0.005"]

print(f"{'Frac%':<8}{'Method':<8}{'Mean':<8}{'Std':<8}{'Min':<8}{'Max':<8}")
print("-" * 48)

summary = {}

for frac in FRACTIONS:
    frac_data = results[frac]
    summary[frac] = {}
    for method in ["tucker", "svm", "rf"]:
        vals = np.array(frac_data[method])
        summary[frac][method] = {
            "mean": float(vals.mean()),
            "std": float(vals.std()),
            "min": float(vals.min()),
            "max": float(vals.max()),
            "n": len(vals),
        }
        print(f"{float(frac)*100:<8.1f}{method:<8}{vals.mean():<8.3f}{vals.std():<8.3f}{vals.min():<8.3f}{vals.max():<8.3f}")
    print()

print("\n" + "=" * 60)
print("SIGNIFICANCE TESTS AT LOW-LABEL FRACTIONS (2%, 1%, 0.5%)")
print("=" * 60)

sig_results = {}

for frac in LOW_LABEL_FOCUS:
    frac_data = results[frac]
    tucker = np.array(frac_data["tucker"])
    svm = np.array(frac_data["svm"])
    rf = np.array(frac_data["rf"])

    n = min(len(tucker), len(svm), len(rf))
    tucker, svm, rf = tucker[:n], svm[:n], rf[:n]

    print(f"\n--- Fraction {float(frac)*100:.1f}% (n={n} paired folds) ---")

    stat_ts, p_ts = wilcoxon(tucker, svm)
    print(f"Tucker vs SVM: mean diff={tucker.mean()-svm.mean():.3f}, p={p_ts:.4f}, "
          f"{'SIGNIFICANT' if p_ts < 0.05 else 'not significant'}")

    stat_tr, p_tr = wilcoxon(tucker, rf)
    print(f"Tucker vs RF:  mean diff={tucker.mean()-rf.mean():.3f}, p={p_tr:.4f}, "
          f"{'SIGNIFICANT' if p_tr < 0.05 else 'not significant'}")

    tucker_wins_svm = int((tucker > svm).sum())
    tucker_wins_rf = int((tucker > rf).sum())
    print(f"Tucker beats SVM in {tucker_wins_svm}/{n} folds")
    print(f"Tucker beats RF in {tucker_wins_rf}/{n} folds")

    sig_results[frac] = {
        "n": n,
        "tucker_vs_svm": {"p_value": float(p_ts), "mean_diff": float(tucker.mean() - svm.mean()), "tucker_wins": tucker_wins_svm},
        "tucker_vs_rf": {"p_value": float(p_tr), "mean_diff": float(tucker.mean() - rf.mean()), "tucker_wins": tucker_wins_rf},
    }

out = {"per_fraction_stats": summary, "low_label_significance": sig_results}
with open(ROOT / "results" / "low_label_sweep_analysis.json", "w") as f:
    json.dump(out, f, indent=2)

print(f"\n\nSaved full analysis to results/low_label_sweep_analysis.json")