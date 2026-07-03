from datasets import load_dataset, Dataset

# Stream TriviaQA instead of full download
streamed = load_dataset("mandarjoshi/trivia_qa", "rc", split="train", streaming=True)

# Take only 150 samples
subset = list(streamed.take(150))

# Convert to a Dataset and save
small_dataset = Dataset.from_list(subset)
small_dataset.save_to_disk("datasets/triviaqa/raw")
