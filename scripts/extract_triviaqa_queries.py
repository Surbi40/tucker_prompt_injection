import json 
from pathlib import Path
from datasets import load_from_disk


from src.utils.config import load_config, get_project_root

config = load_config()
ROOT = get_project_root()

RAW_PATH = ROOT / "datasets" / "triviaqa" / "raw"
OUT_PATH = ROOT / "datasets" / "triviaqa" / "queries.json"

def main():
    print(f"loading TriviaQA datasets from {RAW_PATH}...")
    dataset = load_from_disk(str(RAW_PATH))
    print(f"loaded {len(dataset)} rows")
    print(f"columns:{dataset.column_names}")

    print("\n Sample Row: ")
    print(dataset[0])

    if "question" not in dataset.column_names:
        raise ValueError(
            f"Expected a 'question' column but found : {dataset.column_names}."
            "Check the sample row above and adjust the column name below."
        )
    needed = config["clean_samples"] + config["injected_samples"]

    buffer_size = needed + 50

    questions = dataset["question"][:buffer_size]


    #basic cleaning
    seen = set()
    cleaned = []
    for q in questions:
        q = q.strip()
        if len(q) <5:
            continue
        if q in seen:
            continue
        seen.add(q)
        cleaned.append(q)
    
    if len(cleaned) < needed:
        raise ValueError(
            f"Only {len(cleaned)} usable question after cleaning, " f"but {needed} are required. Increase buffer_size or lower"
            f"clean_samples/injected_samples in config."
        )
    final_queries = cleaned[:needed]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(final_queries,f,indent=2)
    
    print(f"\nSaved{len(final_queries)} quesries to {OUT_PATH}")

if __name__ =="__main__":
    main()