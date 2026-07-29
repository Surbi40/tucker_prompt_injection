from datasets import load_dataset
import pandas as pd 



DATASETS = {
    "DS1": "deepset/prompt-injections",
   

}

def inspect_dataset(name, hf_id):
    print(f"\n---{name} ({hf_id})---")
    ds = load_dataset(hf_id)

    for split in ds.keys():
        print(f"Split: {split}")
        print(f"Column: {ds[split].column_names}")
        print(f"Rows: {len(ds[split])}")

        df = ds[split].to_pandas()

        print(" Samle rows: ")
        print(df.head(2))

        if "label" in df.columns:
            print("label distribution: ")
            print(df["label"].value_counts())

inspect_dataset("DS1", DATASETS["DS1"])

