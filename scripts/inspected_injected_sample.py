import json
from src.utils.config import get_project_root

ROOT = get_project_root()


with open(ROOT/"data"/"processed"/"dataset_200.json") as f:
    dataset= json.load(f)

injected = [s for s in dataset if s["label"]==1]

print(f"Total Injected samples: {len(injected)}\n")


for i, sample in enumerate(injected[:10]):  
    print(f"{'='*70}")
    print(f"Sample {i+1}")
    print(f"Query: {sample['query']}")
    print(f"{'-'*70}")
    print(f"Response:\n{sample['response']}")
    print(f"{'='*70}\n")