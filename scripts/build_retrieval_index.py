from sentence_transformers import SentenceTransformer
import faiss, numpy as np, pickle, os
from datasets import load_dataset
wiki = load_dataset("kaitchup/wikipedia-20220301-en-sample-10k",split="train[:2000]" )

#chunk articles (around 200 words each)
def chunk_text(text, chunk_size=200):
    words=text.split()
    for i in range (0, len(words), chunk_size):
        yield " ".join(words[i:i+chunk_size])

chunks = []
for article in wiki:
    for chunk in chunk_text(article["text"]):
        chunks.append(chunk)
#Limit to around 1000 chunks 
chunks = chunks[:1000]

#embed chunks

embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embeddings = embedder.encode(chunks, convert_to_numpy=True, show_progress_bar=True)

#build FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

#save index and chunks
os.makedirs("datasets/wikipedia", exist_ok=True)
faiss.write_index(index, "datasets/wikipedia/index.faiss")
with open("datasets/wikipedia/chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)

print("Retrieval index built from first 2000 articles and saved to datasets/wikipedia/")