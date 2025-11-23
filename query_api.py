# query_api.py  (drop-in replacement)
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List, Dict, Any, Tuple
import time
import os
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import requests
from pathlib import Path
from rich import print as rprint
import difflib
import json
from collections import OrderedDict
import re

APP = FastAPI(title="Hybrid Support Bot")

# ---- Config ----
VECTOR_DIR = Path("./vectorstore")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "all-MiniLM-L6-v2")
TOP_K = int(os.environ.get("TOP_K", 3))
RETRIEVAL_THRESHOLD = float(os.environ.get("RETRIEVAL_THRESHOLD", 0.30))

# Ollama (local fast model)
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "phi3")
OLLAMA_MAX_TOKENS = int(os.environ.get("OLLAMA_MAX_TOKENS", 120))
OLLAMA_TIMEOUT = int(os.environ.get("OLLAMA_TIMEOUT", 60))  # increased default timeout

# Local internship PDF path (provided file)
CHALLENGE_PDF_PATH = "/mnt/data/AI & AUTOMATION INTERNSHIP SELECTION CHALLENGE.pdf"

# ---- API models ----
class QueryIn(BaseModel):
    query: str
    chapter_hint: Optional[str] = None
    top_k: Optional[int] = TOP_K

class QueryOut(BaseModel):
    answer: str
    retrieved_chunks: List[Dict[str, Any]]
    retrieval_latency_s: float
    generation_latency_s: float
    retrieval_scores: List[float]

# ---- Load vectorstore ----
if not (VECTOR_DIR / "faiss.index").exists():
    raise SystemExit("Missing FAISS index at vectorstore/ - run chunk_and_embed.py first")

index = faiss.read_index(str(VECTOR_DIR / "faiss.index"))
with open(VECTOR_DIR / "metas.pkl", "rb") as f:
    store = pickle.load(f)
metas = store["metas"]
rprint(f"[green]Loaded vectorstore with {len(metas)} chunks.[/green]")

# ---- Embedder ----
embed_model = SentenceTransformer(EMBED_MODEL)

# ---- Chapter list (for fuzzy matching) ----
_unique_chapters = sorted({(m.get("chapter") or "Unknown").strip() for m in metas})
rprint(f"[yellow]Loaded chapters (sample):[/yellow] {_unique_chapters[:10]}")

def map_chapter_hint(hint: Optional[str]) -> Optional[str]:
    if not hint:
        return None
    hint = hint.strip().lower()
    # exact partial match
    for ch in _unique_chapters:
        if hint in ch.lower():
            return ch
    # fuzzy match
    matches = difflib.get_close_matches(hint, _unique_chapters, n=1, cutoff=0.5)
    return matches[0] if matches else None

# ---- Semantic search ----
def semantic_search(query: str, top_k: int = TOP_K, chapter_hint: Optional[str] = None) -> Tuple[List[dict], List[float], float]:
    t0 = time.perf_counter()
    q_emb = embed_model.encode(query, convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q_emb.reshape(1, -1))
    D, I = index.search(q_emb.reshape(1, -1), top_k * 3)  # oversample then filter
    rlat = time.perf_counter() - t0

    results, scores = [], []
    for score, idx in zip(D[0], I[0]):
        if idx < 0:
            continue
        meta = metas[idx]
        if chapter_hint:
            if chapter_hint.lower() not in (meta.get("chapter") or "").lower():
                continue
        results.append(meta)
        scores.append(float(score))
        if len(results) >= top_k:
            break
    return results, scores, rlat

# ---- Small in-memory LRU cache for recent queries ----
class SimpleLRUCache:
    def __init__(self, max_size=128):
        self.store = OrderedDict()
        self.max_size = max_size

    def get(self, key):
        if key in self.store:
            self.store.move_to_end(key)
            return self.store[key]
        return None

    def set(self, key, value):
        self.store[key] = value
        self.store.move_to_end(key)
        if len(self.store) > self.max_size:
            self.store.popitem(last=False)

cache = SimpleLRUCache(max_size=128)

# ---- Extractive shortcut ----
def extractive_answer(query: str, chunks: List[dict], score_threshold: float = 0.40) -> Optional[str]:
    """
    If the top chunk is sufficiently similar (score > threshold) OR
    contains many query tokens, return an extractive answer immediately
    to avoid calling the LLM.
    """
    if not chunks:
        return None
    top = chunks[0]
    text = (top.get("text_snippet") or "").replace("\n", " ").strip()
    # Token overlap heuristic
    q_tokens = set(re.findall(r"\w+", query.lower()))
    txt_tokens = set(re.findall(r"\w+", text.lower()))
    if not q_tokens:
        return None
    overlap = q_tokens & txt_tokens
    overlap_ratio = len(overlap) / max(1, len(q_tokens))

    # prefer extractive if overlap high
    if overlap_ratio >= 0.6:
        # return snippet with reference
        return f"{text[:800].rstrip()} ({top.get('chapter')}, p.{top.get('page')})"

    # else no extractive answer
    return None

# ---- Build compact prompt ----
def build_prompt(query: str, chunks: List[dict], max_chars_per_chunk=400, max_chunks=2) -> str:
    top = chunks[:max_chunks]
    snippet_texts = []
    for c in top:
        txt = (c.get("text_snippet") or "").replace("\n", " ").strip()
        if len(txt) > max_chars_per_chunk:
            txt = txt[:max_chars_per_chunk].rsplit(" ", 1)[0] + "..."
        snippet_texts.append(f"[{c.get('chapter')}, p.{c.get('page')}] {txt}")
    context = "\n\n".join(snippet_texts)
    prompt = (
        "You are a technical assistant that answers questions using ONLY the provided manual context.\n"
        "If the answer is not present in the context, reply exactly: \"I don't know\".\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {query}\n\n"
        "Answer concisely (1-3 sentences). Add a short reference like (chapter,page) if present."
    )
    return prompt

# ---- Robust Ollama caller (NDJSON-safe) ----
def call_local_generator(prompt: str) -> Tuple[str, float]:
    t0 = time.perf_counter()
    url = OLLAMA_URL
    model = os.environ.get("OLLAMA_MODEL", OLLAMA_MODEL)
    payload = {"model": model, "prompt": prompt, "max_tokens": OLLAMA_MAX_TOKENS, "stream": False}

    try:
        resp = requests.post(url, json=payload, timeout=OLLAMA_TIMEOUT)
        raw = resp.text or ""
    except Exception as e:
        return f"MODEL_UNAVAILABLE: {e}", time.perf_counter() - t0

    gen_latency = time.perf_counter() - t0
    if not raw:
        return "", gen_latency

    # NDJSON streaming lines -> collect 'response' or 'text' fragments
    lines = [ln for ln in raw.splitlines() if ln.strip()]
    fragments = []
    for ln in lines:
        try:
            obj = json.loads(ln)
        except Exception:
            continue
        if isinstance(obj, dict):
            if "response" in obj and isinstance(obj["response"], str):
                if obj["response"]:
                    fragments.append(obj["response"])
                continue
            if "text" in obj and isinstance(obj["text"], str) and obj["text"]:
                fragments.append(obj["text"])
                continue

    if fragments:
        short_count = sum(1 for s in fragments if len(s) <= 3)
        if short_count > len(fragments) * 0.6:
            full = "".join(fragments).strip()
        else:
            full = " ".join(s.strip() for s in fragments).replace("  ", " ").strip()
        return full, gen_latency

    # Try single JSON body with common keys
    try:
        obj = resp.json()
        if isinstance(obj, dict):
            for key in ("response", "text", "output", "generated_text", "result"):
                v = obj.get(key)
                if isinstance(v, str) and v.strip():
                    return v.strip(), gen_latency

            if "choices" in obj and isinstance(obj["choices"], list) and obj["choices"]:
                first = obj["choices"][0]
                if isinstance(first, dict):
                    if "message" in first and isinstance(first["message"], dict) and "content" in first["message"]:
                        c = first["message"]["content"]
                        if isinstance(c, str) and c.strip():
                            return c.strip(), gen_latency
                    for ck in ("text", "content"):
                        if ck in first and isinstance(first[ck], str) and first[ck].strip():
                            return first[ck].strip(), gen_latency

            for arrk in ("results", "outputs"):
                if arrk in obj and isinstance(obj[arrk], list) and obj[arrk]:
                    parts = []
                    for item in obj[arrk]:
                        if isinstance(item, dict):
                            for kk in ("response", "text", "output", "generated_text"):
                                if kk in item and isinstance(item[kk], str) and item[kk].strip():
                                    parts.append(item[kk].strip())
                    if parts:
                        return " ".join(parts).strip(), gen_latency
    except Exception:
        pass

    # Fallback regex
    try:
        matches = re.findall(r'"response"\s*:\s*"([^"]+)"', raw)
        if matches:
            joined = "".join(matches).strip()
            if joined:
                return joined, gen_latency
    except Exception:
        pass

    try:
        rprint("[yellow]Ollama preview (first 800 chars):[/yellow]")
        rprint(raw[:800])
    except Exception:
        pass

    return "", gen_latency

# ---- Main endpoint ----
@APP.post("/query", response_model=QueryOut)
def query(q: QueryIn):
    mapped_hint = map_chapter_hint(q.chapter_hint)
    retrieved, scores, rlat = semantic_search(q.query, top_k=(q.top_k or TOP_K), chapter_hint=mapped_hint)
    rprint(f"[green]Retrieval latency:[/green] {rlat:.4f}s (found {len(retrieved)})")

    # low confidence guard
    if not scores or max(scores) < RETRIEVAL_THRESHOLD:
        return QueryOut(
            answer="I don't know based on the provided manual (low retrieval confidence).",
            retrieved_chunks=retrieved,
            retrieval_latency_s=rlat,
            generation_latency_s=0.0,
            retrieval_scores=scores
        )

    # Try cache
    cache_key = f"{q.query}||{mapped_hint}||{','.join([str(c.get('id')) for c in retrieved])}"
    cached = cache.get(cache_key)
    if cached:
        rprint("[cyan]Cache hit[/cyan]")
        return QueryOut(
            answer=cached["answer"],
            retrieved_chunks=retrieved,
            retrieval_latency_s=rlat,
            generation_latency_s=cached["gen_latency"],
            retrieval_scores=scores
        )

    # Extractive shortcut: attempt to produce immediate snippet if overlap is high
    ext = extractive_answer(q.query, retrieved, score_threshold=0.40)
    if ext:
        rprint("[magenta]Returning extractive answer (no LLM call)[/magenta]")
        # store to cache with gen_latency 0
        cache.set(cache_key, {"answer": ext, "gen_latency": 0.0})
        return QueryOut(
            answer=ext,
            retrieved_chunks=retrieved,
            retrieval_latency_s=rlat,
            generation_latency_s=0.0,
            retrieval_scores=scores
        )

    # Build short prompt (small context => faster generation)
    prompt = build_prompt(q.query, retrieved, max_chars_per_chunk=400, max_chunks=2)

    answer, glat = call_local_generator(prompt)
    rprint(f"[blue]Generation latency:[/blue] {glat:.4f}s (model={os.environ.get('OLLAMA_MODEL', OLLAMA_MODEL)})")

    # store to cache
    cache.set(cache_key, {"answer": answer, "gen_latency": glat})

    return QueryOut(
        answer=answer,
        retrieved_chunks=retrieved,
        retrieval_latency_s=rlat,
        generation_latency_s=glat,
        retrieval_scores=scores
    )

# ---- Run the app ----
if __name__ == "__main__":
    import uvicorn
    rprint(f"[green]Starting app with model={os.environ.get('OLLAMA_MODEL', OLLAMA_MODEL)} and url={OLLAMA_URL}[/green]")
    uvicorn.run("query_api:APP", host="0.0.0.0", port=8000, reload=True)
