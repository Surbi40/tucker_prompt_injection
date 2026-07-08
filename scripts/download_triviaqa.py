from datasets import load_dataset
from src.utils.config import get_project_root
from datasets import Dataset

ROOT = get_project_root()

def main():
    # Stream TriviaQA instead of full download
    streamed = load_dataset("mandarjoshi/trivia_qa", "rc", split="train", streaming=True)

    # Take only 400 samples
    subset = list(streamed.take(400))

    triviaqa = Dataset.from_list(subset)
    out_path = ROOT / "datasets" / "triviaqa" / "raw"
    triviaqa.save_to_disk(str(out_path))

    print(f"Saved {len(triviaqa)} rows to {out_path}")

if __name__ == "__main__":
    main()
