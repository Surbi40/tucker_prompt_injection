import sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))
from src.utils.config import get_project_root

ROOT = get_project_root()
with open(ROOT / "results" / "run_02" / "sensitivity_analysis_ds1.json") as f:
    results = json.load(f)

results_sorted = sorted(results, key=lambda r: -r["mean_f1"])
print("Top 5 configs by F1:")
for r in results_sorted[:5]:
    print(f"  window={r['window_size']:2d} rank={r['rank']:2d}  F1={r['mean_f1']:.3f}±{r['std_f1']:.3f}  "
          f"decomp={r['mean_decomp_time_sec']:.2f}s/fold")

best = results_sorted[0]
print(f"\nBest config: window={best['window_size']}, rank={best['rank']}")
print("Use this (window, rank) as the standardized config for the full experiment matrix.")