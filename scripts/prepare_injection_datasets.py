from datasets import load_dataset
import pandas as pd
import re
from src.utils.config import get_project_root

ROOT = get_project_root()


def clean_text(text):
    text = str(text).strip()
    text = re.sub(r'\s+', ' ', text)
    return text


def load_and_clean(dataset_name, text_col, label_col, extra_cols=None):
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

    df = df[df["text"].str.len() >= 5]

    before = len(df)
    df = df.drop_duplicates(subset="text")
    after = len(df)

    df["label"] = df["label"].astype(int)

    print(f"\n{dataset_name}")
    print(f"  Rows after cleaning: {after} (dropped {before - after} duplicates)")
    print(f"  Label balance:\n{df['label'].value_counts().to_string()}")

    return df


if __name__ == "__main__":
    ds1 = load_and_clean(
        "deepset/prompt-injections",
        text_col="text",
        label_col="label",
    )

    ds2 = load_and_clean(
        "wambosec/prompt-injections",
        text_col="prompt",
        label_col="label",
        extra_cols=["category", "goal", "length_type"],
    )

    out_dir = ROOT / "datasets" / "prompt_injection"
    out_dir.mkdir(parents=True, exist_ok=True)

    ds1.to_json(out_dir / "ds1_deepset_clean.json", orient="records", indent=2)
    ds2.to_json(out_dir / "ds2_wambosec_clean.json", orient="records", indent=2)

    print(f"\nSaved to {out_dir}")