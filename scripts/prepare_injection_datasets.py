from datasets import load_dataset
import pandas as pd
import numpy as np
import re
import hashlib
import json
from src.utils.config import get_project_root

ROOT = get_project_root()

def audit_dedup(df, name, n=10):
    """Show which raw texts collapsed onto the same dedup key, for spot-checking."""
    df2 = add_dedup_hash(df)
    dupe_keys = df2["_dedup_key"].value_counts()
    dupe_keys = dupe_keys[dupe_keys > 1].head(n)
    for key in dupe_keys.index:
        texts = df2[df2["_dedup_key"] == key]["text"].tolist()
        print(f"\n--- {name} dedup group ({len(texts)} rows) ---")
        for t in texts[:5]:
            print(" •", repr(t[:120]))



def clean_text(text):
    text = str(text).strip()
    text = re.sub(r'\s+', ' ', text)
    return text


def normalize_for_dedup(text):
    t = text.lower().strip()
    t = re.sub(r'[^\w\s]', '', t)
    t = re.sub(r'\s+', ' ', t)
    return t


def add_dedup_hash(df):
    df = df.copy()
    df["_dedup_key"] = df["text"].apply(lambda t: hashlib.md5(normalize_for_dedup(t).encode()).hexdigest())
    return df


def load_and_clean(dataset_name, text_col, label_col, extra_cols=None, min_len=5, max_len=2000):
    ds = load_dataset(dataset_name)
    frames = []
    for split in ds.keys():
        df = pd.DataFrame(ds[split])
        df["split"] = split
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    df = df.rename(columns={text_col: "text", label_col: "label"})
    df["text"] = df["text"].apply(clean_text)

    keep = ["text", "label", "split"] + (extra_cols or [])
    df = df[[c for c in keep if c in df.columns]]

    df = df[df["text"].str.len().between(min_len, max_len)]

    df = add_dedup_hash(df)
    before = len(df)
    df = df.drop_duplicates(subset="_dedup_key")
    after = len(df)

    df["label"] = df["label"].astype(int)

    n_benign = (df["label"] == 0).sum()
    n_inject = (df["label"] == 1).sum()
    ratio = n_benign / max(n_inject, 1)

    print(f"\n{dataset_name}")
    print(f"  Rows after cleaning: {after} (dropped {before - after} near-duplicates)")
    print(f"  Label balance:\n{df['label'].value_counts().to_string()}")
    print(f"  Benign:Injection ratio = {ratio:.2f}:1")

    # Save metadata summary
    stats = {
        "rows": int(after),
        "benign": int(n_benign),
        "injection": int(n_inject),
        "duplicates_removed": int(before - after),
        "ratio": f"{ratio:.2f}:1"
    }
    safe_name = dataset_name.replace("/","_")

    out_dir = ROOT / "datasets" / "prompt_injection"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{safe_name}_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    return df


def check_label_consistency(df, name):
    """DS2-specific: verify label (int) agrees with is_malicious (bool)."""
    if "is_malicious" not in df.columns:
        return
    mismatch = (df["label"] == 1) != (df["is_malicious"] == True)
    n_mismatch = mismatch.sum()
    print(f"  [{name}] label vs is_malicious mismatches: {n_mismatch} / {len(df)}")
    if n_mismatch > 0:
        print(f"  WARNING: {n_mismatch} rows disagree — inspect before trusting label column")
    # Drop redundant column after check
    df.drop(columns=["is_malicious"], inplace=True)


def check_cross_overlap(name_a, df_a, name_b, df_b):
    keys_a = set(add_dedup_hash(df_a)["_dedup_key"])
    keys_b = set(add_dedup_hash(df_b)["_dedup_key"])
    overlap = keys_a & keys_b
    pct = len(overlap) / len(keys_a) * 100 if len(keys_a) else 0
    print(f"  {name_a} vs {name_b}: {len(overlap)} overlapping rows ({pct:.1f}% of {name_a})")
    return overlap


def build_combined(datasets: dict, target_per_class=1000, seed=42, balance=True):
    """
    Quota-based: each source contributes an equal share per class (capped by
    its own availability), so Combined is genuinely multi-source and not
    dominated by whichever dataset happens to be largest (DS4).
    """
    pool = pd.concat(
        [df.assign(source=name) for name, df in datasets.items()],
        ignore_index=True
    )
    pool = add_dedup_hash(pool)
    before = len(pool)
    pool = pool.drop_duplicates(subset="_dedup_key")
    print(f"\nCombined: cross-dataset dedup {before} -> {len(pool)}")

    if not balance:
        combined = pool.copy()
    else:
        def allocate_quota(class_label):
            class_pool = pool[pool["label"] == class_label]
            avail = {name: len(class_pool[class_pool["source"] == name]) for name in datasets}
            n_sources = len(datasets)
            base = target_per_class // n_sources
            allocated = {name: min(base, avail[name]) for name in datasets}
            leftover = target_per_class - sum(allocated.values())
            while leftover > 0:
                spare = {n: avail[n] - allocated[n] for n in datasets if avail[n] > allocated[n]}
                if not spare:
                    break
                share = max(1, leftover // len(spare))
                for name in spare:
                    add = min(share, spare[name], leftover)
                    allocated[name] += add
                    leftover -= add
                    if leftover <= 0:
                        break
            return allocated, class_pool

        samples = []
        for label in [0, 1]:
            allocated, class_pool = allocate_quota(label)
            print(f"  class {label} quota: {allocated}")
            for name, n in allocated.items():
                src_rows = class_pool[class_pool["source"] == name]
                samples.append(src_rows.sample(n=min(n, len(src_rows)), random_state=seed))
        combined = pd.concat(samples, ignore_index=True)

    combined = combined.sample(frac=1, random_state=seed).reset_index(drop=True)
    combined = combined.drop(columns=["_dedup_key"])

    print(f"Combined total: {len(combined)} rows")
    print(combined["label"].value_counts().to_string())
    print("\nSource composition (source x label):")
    print(combined.groupby(["source", "label"]).size().unstack(fill_value=0))

    return combined

def make_experimental_subsample(df, name, n=2000, seed=42):
    """
    Fixed-size stratified subsample for datasets too large to run repeated
    tensor decomposition + 5-fold CV + hyperparameter grids on directly.
    Mirrors the original paper's DS4 standardization to 2000 prompts.
    """
    if len(df) <= n:
        print(f"{name}: {len(df)} rows <= {n}, no subsampling needed")
        return df
    frac = n / len(df)
    sub = df.groupby("label", group_keys=False).apply(
        lambda g: g.sample(frac=frac, random_state=seed)
    )
    print(f"{name}: subsampled {len(df)} -> {len(sub)} for experiments (full file saved separately)")
    print(sub["label"].value_counts().to_string())
    return sub



if __name__ == "__main__":
    ds1 = load_and_clean("deepset/prompt-injections", text_col="text", label_col="label")

    ds2 = load_and_clean(
        "wambosec/prompt-injections",
        text_col="prompt", label_col="label",
        extra_cols=["category", "goal", "length_type", "is_malicious"],
    )
    check_label_consistency(ds2, "DS2")

    ds3 = load_and_clean("S-Labs/prompt-injection-dataset", text_col="text", label_col="label")

    # DS4 raw audit block (optional, if you want to inspect duplicates)
    ds4_raw = load_dataset("geekyrakshit/prompt-injection-dataset")
    frames = []
    for split in ds4_raw.keys():
        df = pd.DataFrame(ds4_raw[split])
        df["split"] = split
        frames.append(df)
    ds4_raw = pd.concat(frames, ignore_index=True)
    ds4_raw = ds4_raw.rename(columns={"prompt": "text", "label": "label"})
    ds4_raw["text"] = ds4_raw["text"].apply(clean_text)
    audit_dedup(ds4_raw, "DS4", n=10)

    # Normal DS4 cleaning
    ds4 = load_and_clean("geekyrakshit/prompt-injection-dataset", text_col="prompt", label_col="label")

    print("\n=== Cross-dataset overlap checks ===")
    check_cross_overlap("DS1", ds1, "DS4", ds4)
    check_cross_overlap("DS1", ds1, "DS2", ds2)
    check_cross_overlap("DS3", ds3, "DS4", ds4)
    check_cross_overlap("DS2", ds2, "DS4", ds4)

    #  Build Combined with quota logic (~4000 rows total, balanced across sources)
    combined = build_combined(
        {"ds1": ds1, "ds2": ds2, "ds3": ds3, "ds4": ds4},
        target_per_class=1000, seed=42, balance=True
    )

    #  Subsample DS3 and DS4 for experiments (2000 rows each)
    ds4_exp = make_experimental_subsample(ds4, "DS4", n=2000, seed=42)
    ds3_exp = make_experimental_subsample(ds3, "DS3", n=2000, seed=42)

    #  Save outputs
    out_dir = ROOT / "datasets" / "prompt_injection"
    out_dir.mkdir(parents=True, exist_ok=True)

    ds1.drop(columns=["_dedup_key"]).to_json(out_dir / "ds1_deepset_clean.json", orient="records", indent=2)
    ds2.drop(columns=["_dedup_key"]).to_json(out_dir / "ds2_wambosec_clean.json", orient="records", indent=2)
    ds3.drop(columns=["_dedup_key"]).to_json(out_dir / "ds3_slabs_clean.json", orient="records", indent=2)
    ds4.drop(columns=["_dedup_key"]).to_json(out_dir / "ds4_geekyrakshit_full.json", orient="records", lines=True)
    ds3_exp.to_json(out_dir / "ds3_slabs_experiment.json", orient="records", indent=2)
    ds4_exp.to_json(out_dir / "ds4_geekyrakshit_experiment.json", orient="records", indent=2)
    combined.to_json(out_dir / "combined_clean.json", orient="records", indent=2)

    print(f"\nSaved to {out_dir}")
