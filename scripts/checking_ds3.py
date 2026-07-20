from datasets import load_dataset

for name in ["S-Labs/prompt-injection-dataset", "geekyrakshit/prompt-injection-dataset"]:
    ds = load_dataset(name)
    print(name)
    for split in ds.keys():
        print(" ", split, ds[split].column_names, len(ds[split]))
    print(" sample:", ds[list(ds.keys())[0]][0])
    print()
