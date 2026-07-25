import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import json
import re
from src.utils.config import get_project_root

ROOT = get_project_root()

with open(ROOT / "datasets" / "prompt_injection" / "ds2_wambosec_clean.json", "r", encoding="utf-8") as f:
    data = json.load(f)

for label, name in [(0, "benign"), (1, "injection")]:
    subset = [row["text"] for row in data if row["label"] == label]
    special_char_ratio = sum(
        len(re.sub(r"[a-zA-Z0-9\s']", "", t)) / max(len(t), 1) for t in subset
    ) / len(subset)
    print(f"{name}: avg special-char ratio = {special_char_ratio:.4f}")