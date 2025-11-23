#!/usr/bin/env python3
"""
chunk_and_embed.py
Loads the JSONL from ingest.py, computes embeddings, stores FAISS index with metadata.
"""
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
from tqdm import tqdm

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"  # compact and effective
OUT_DIR = Path("./vectorstore")

def load_chunks(jsonl_path):
    with open(jsonl_path, "r", encoding="utf8") as f:
        for line in f:
            yield json.loads(line)

def build_faiss_index(chunks, model_name=EMBED_MODEL_NAME, out_dir=OUT_DIR):
    model = SentenceTransformer(model_name)
    vectors = []
    metas = []
    ids = []
    for chunk in tqdm(list(chunks), desc="Encoding chunks"):
        text = chunk.get("text", "")
        emb = model.encode(text, show_progress_bar=False)
        vectors.append(emb)
        metas.append({
            "id": chunk.get("id"),
            "source": chunk.get("source"),
            "chapter": chunk.get("chapter"),
            "page": chunk.get("page"),
            "chunk_index": chunk.get("chunk_index"),
            "text_snippet": text[:400]
        })
        ids.append(chunk.get("id"))
    if not vectors:
        raise ValueError("No chunks found to index.")
    vectors = np.vstack(vectors).astype("float32")
    d = vectors.shape[1]

    # create index: normalized inner product works as cosine after L2-normalizing
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(d)

    index.add(vectors)
    out_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out_dir / "faiss.index"))

    with open(out_dir / "metas.pkl", "wb") as f:
        pickle.dump({"metas": metas, "ids": ids, "model": model_name}, f)

    # also save raw vectors (optional)
    with open(out_dir / "vectors.npy", "wb") as f:
        np.save(f, vectors)

    print("✔ FAISS index and metadata saved to:", out_dir)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks_jsonl", default="./Data/chunks.jsonl")
    parser.add_argument("--out_dir", default="./vectorstore")
    parser.add_argument("--embed_model", default=EMBED_MODEL_NAME)
    args = parser.parse_args()

    chunks = list(load_chunks(args.chunks_jsonl))
    build_faiss_index(chunks, model_name=args.embed_model, out_dir=Path(args.out_dir))
