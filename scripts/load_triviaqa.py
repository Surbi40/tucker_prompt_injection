from datasets import load_from_disk

ds= load_from_disk("datasets/triviaqa/raw")
print(ds[0])