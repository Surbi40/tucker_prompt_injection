import sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import numpy as np
import re
from collections import Counter
from langdetect import detect, LangDetectException
from src.utils.config import get_project_root

ROOT = get_project_root()
out_dir = ROOT / "datasets" / "prompt_injection"

datasets = {
    "DS1": out_dir / "ds1_deepset_clean.json",
    
}

def is_english(text):
    """Return True if langdetect says English, False if another language,
    None if detection fails (too short/ambiguous)."""
    try:
        return detect(text) == 'en'
    except LangDetectException:
        return None

def obfuscation_ratio(text):
    """Fraction of characters that are neither alphanumeric, whitespace,
    nor common punctuation. Flags %XX encoding, homoglyphs, emoji, etc."""
    if len(text) == 0:
        return 0.0
    normal = sum(1 for c in text if c.isalnum() or c.isspace() or c in ".,!?'\"-:;()")
    return 1 - (normal / len(text))

for name, path in datasets.items():
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    texts = [d["text"] for d in data]
    labels = np.array([d["label"] for d in data])

    lengths = np.array([len(t) for t in texts])
    obf = np.array([obfuscation_ratio(t) for t in texts])
    lang_flags = [is_english(t) for t in texts]

    en_mask = np.array([f is True for f in lang_flags])
    non_en_mask = np.array([f is False for f in lang_flags])
    unknown_mask = np.array([f is None for f in lang_flags])

    print(f"\n{'='*60}\n{name} (n={len(texts)})\n{'='*60}")
    print(f"Language: English={en_mask.sum()} ({en_mask.mean()*100:.1f}%)  "
          f"Non-English={non_en_mask.sum()} ({non_en_mask.mean()*100:.1f}%)  "
          f"Undetected={unknown_mask.sum()}")
    if non_en_mask.sum() > 0:
        print(f"  Injection rate: English={labels[en_mask].mean():.2f}  "
              f"Non-English={labels[non_en_mask].mean():.2f}")

    print(f"Length: mean={lengths.mean():.0f} chars, median={np.median(lengths):.0f}")
    long_mask = lengths > np.percentile(lengths, 75)
    print(f"  Injection rate: top-quartile-length={labels[long_mask].mean():.2f}  "
          f"rest={labels[~long_mask].mean():.2f}")

    high_obf_mask = obf > 0.15  # threshold: >15% non-standard chars
    print(f"Obfuscation: high-obfuscation rows={high_obf_mask.sum()} ({high_obf_mask.mean()*100:.1f}%)")
    if high_obf_mask.sum() > 0:
        print(f"  Injection rate: high-obfuscation={labels[high_obf_mask].mean():.2f}  "
              f"normal={labels[~high_obf_mask].mean():.2f}")
