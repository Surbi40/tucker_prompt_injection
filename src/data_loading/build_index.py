import faiss, pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from src.utils.config import load_config, set_seed, get_project_root

ROOT = get_project_root()
config = load_config()
set_seed(config["random_seed"])

embedder = SentenceTransformer(config["embedding_model"])

# Load chunks from wikipedia
with open(ROOT / "datasets" / "wikipedia" / "chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

# Encode chunks
vectors = embedder.encode(chunks)

# Build FAISS index
index = faiss.IndexFlatL2(vectors.shape[1])
index.add(np.array(vectors))

# Save index
faiss.write_index(index, str(ROOT / "datasets" / "wikipedia" / "index.faiss"))

print("FAISS index built and saved to datasets/wikipedia/index.faiss")
