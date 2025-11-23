# debug_search.py
from sentence_transformers import SentenceTransformer
import faiss, pickle, numpy as np, time, json
from pathlib import Path

VECTOR_DIR = Path("vectorstore")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index(str(VECTOR_DIR / "faiss.index"))
with open(VECTOR_DIR / "metas.pkl","rb") as f:
    store = pickle.load(f)
metas = store["metas"]

def raw_search(query, top_k=10):
    t0 = time.perf_counter()
    q_emb = embed_model.encode(query, convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q_emb.reshape(1,-1))
    D,I = index.search(q_emb.reshape(1,-1), top_k)
    elapsed = time.perf_counter()-t0
    results = []
    for score, idx in zip(D[0], I[0]):
        if idx < 0: continue
        meta = metas[idx]
        results.append({
            "score": float(score),
            "id": meta.get("id"),
            "chapter": meta.get("chapter"),
            "page": meta.get("page"),
            "snippet": meta.get("text_snippet")[:300].replace("\n"," ")
        })
    print("Search time:", elapsed)
    print(json.dumps(results, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    q = input("Query: ").strip()
    raw_search(q, top_k=10)
