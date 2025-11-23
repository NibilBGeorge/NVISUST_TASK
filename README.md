Hybrid Support Bot — RAG (FAISS + Local LLM via Ollama)

One-line summary: A Retrieval-Augmented Generation system that answers manual/technical questions by searching a FAISS vectorstore built from PDF manuals and generating concise answers using a local Ollama model.

Table of contents

What is included

Prerequisites

Project layout

Setup & installation (step-by-step)

Ingest PDF → vectorstore (how to run ingest.py)

Start the API server (query_api.py)

Example queries / smoke tests

Environment variables & tuning for latency

How to run tests (suggested smoke + pytest)

What to submit (per internship instructions)

Troubleshooting & tips

Notes about internship PDF used (local path)

What is included

ingest.py — Convert a PDF manual into text chunks, embed them and create a FAISS index and metadata file.

query_api.py — FastAPI service that performs semantic retrieval + local LLM generation (Ollama).

requirements.txt — Python dependencies.

vectorstore/ — (output) faiss.index and metas.pkl after running ingestion.

README.md — this file (instructions).

demo/ (optional) — place screenshots or demo video link here.

Prerequisites

Python 3.10+ (recommended)

Git (for repo)

Enough disk and memory for embeddings & model usage

Ollama installed locally (for local LLM)

download: https://ollama.com

recommended model: phi3 (fast) or llama3 (higher quality)

Project layout (example)
.
├── ingest.py
├── query_api.py
├── requirements.txt
├── README.md
├── vectorstore/                # created by ingest (faiss.index, metas.pkl)
└── demo/                       # screenshots or video link file (put here)

Setup & installation (step-by-step)
1. Clone repo
git clone ttps://github.com/NibilBGeorge/NVISUST_TASK.git
cd <NVISUST_TASK>

2. Create and activate Python virtual environment

Windows (PowerShell)

python -m venv .venv
.\.venv\Scripts\Activate.ps1      # or Activate.bat


Linux / macOS

python -m venv .venv
source .venv/bin/activate

3. Install dependencies
pip install -r requirements.txt


If sentence-transformers or torch installation fails, follow the error instructions or install a CPU-only torch wheel for your Python version.

4. Start Ollama and pull a model (local)

Install Ollama then in a terminal:

# pull recommended fast model
ollama pull phi3

# or for higher quality
ollama pull llama3


Test:

# test local server is working
curl -X POST "http://localhost:11434/api/generate" -H "Content-Type: application/json" \
  -d '{"model":"phi3","prompt":"hello","max_tokens":10}'


You should get NDJSON lines or JSON with a non-empty response / text.

Ingest PDF → vectorstore (how to run ingest.py)

Purpose: Convert your PDF into chunks + embeddings and write FAISS index + metas.

Required argument: --pdf_path — path to your manual PDF.

Example (PowerShell / bash):

python ingest.py --pdf_path "service-manual-w11187658-whirlpool.pdf"


If you want to use the internship instruction PDF used earlier (local path):

/mnt/data/AI & AUTOMATION INTERNSHIP SELECTION CHALLENGE.pdf


So you could run:

python ingest.py --pdf_path "/mnt/data/AI & AUTOMATION INTERNSHIP SELECTION CHALLENGE.pdf"


Output files (created in vectorstore/):

vectorstore/faiss.index

vectorstore/metas.pkl

Notes:

The ingest script usually accepts optional parameters for chunk size, overlap. See script header comments.

If the script errors that --pdf_path is required, pass the path as shown above.

Start the API server (query_api.py)

Once vectorstore/ is present and Ollama is running:

# simple run (development)
python query_api.py


Or using uvicorn (recommended for production-like):

uvicorn query_api:APP --host 0.0.0.0 --port 8000 --reload


Open the interactive docs:

http://127.0.0.1:8000/docs

Example request JSON (body)
{
  "query": "How do I run service mode?",
  "chapter_hint": "TESTING",
  "top_k": 3
}


If you do not want to send chapter_hint or top_k, you can omit them:

{ "query": "How do I run service mode?" }

Example queries / smoke tests

PowerShell:

$body = @{ query = "How do I run service mode?"; chapter_hint = "TESTING"; top_k = 3 } | ConvertTo-Json
Invoke-RestMethod -Uri "http://127.0.0.1:8000/query" -Method Post -Body $body -ContentType "application/json"


curl (bash):

curl -X POST "http://127.0.0.1:8000/query" -H "Content-Type: application/json" \
  -d '{"query":"How do I run service mode?","chapter_hint":"TESTING","top_k":3}'


Swagger UI: http://127.0.0.1:8000/docs
 → try /query endpoint.

Environment variables & tuning for latency

You can tweak via environment variables:

EMBED_MODEL — embedding model (default: all-MiniLM-L6-v2). Use smaller/CPU-friendly models for lower memory.

TOP_K — number of results to return (default in code: 3). Lower TOP_K → less retrieval & smaller prompt → lower latency.

RETRIEVAL_THRESHOLD — float threshold to skip LLM generation when retrieval confidence is low.

OLLAMA_URL — http://localhost:11434/api/generate by default.

OLLAMA_MODEL — phi3 (fast) or llama3.

OLLAMA_MAX_TOKENS — max tokens for generation.

Example (Linux/macOS):

export OLLAMA_MODEL=phi3
export TOP_K=2
export EMBED_MODEL=all-MiniLM-L6-v2
python query_api.py


Example (Windows PowerShell):

$env:OLLAMA_MODEL="phi3"
$env:TOP_K="2"
python query_api.py


Latency tips

Use phi3 (faster) instead of larger models.

Reduce TOP_K and restrict to 1-2 chunks.

Shorten chunk size during ingestion (smaller snippets).

Use extractive answer heuristic: if top chunk has high token overlap with query, return snippet directly (this avoids LLM call).

Use the in-memory LRU cache in query_api.py (already included in optimized code).

How to run tests (suggested smoke + pytest)

You can add tests/ with simple unit tests. Example smoke tests you can run manually:

Smoke test: Is server up?

curl -s -X POST "http://127.0.0.1:8000/query" -H "Content-Type: application/json" -d '{"query":"ping"}'


Expect a JSON response (maybe "I don't know..." or something).

Automated via pytest
Create a tests/test_api.py with requests using httpx or requests to hit /query. Then:

pip install pytest httpx
pytest -q


(Include tests in repo under tests/ before running pytest.)

What to submit (per internship instructions)

Place the following in your GitHub repository:

Complete source code (ingest.py, query_api.py, any helper scripts).

vectorstore/ — optionally include the generated faiss.index and metas.pkl or instructions to build them.

README.md — this file (must include the notes below).

Demo — either:

a short Loom/YouTube video link (put link in README), or

a demo/screenshots/ folder with screenshots showing: server start, swagger UI request, sample query results, Ollama terminal.

One-line submission: Provide the repo link as the single task deliverable (respond to the form/email with repo link).

Troubleshooting & tips

ingest.py says --pdf_path required
Provide the argument exactly, e.g. python ingest.py --pdf_path "/full/path/to/file.pdf".

Ollama “model not found”
ollama list or ollama pull phi3 to download the model. Confirm OLLAMA_URL and model name match.

Timeouts while calling Ollama
Increase request timeout in query_api.py requests.post(..., timeout=60) or use smaller prompts / fewer tokens.

Large memory usage
Use a smaller embedding model or run on machine with more RAM; lower TOP_K and chunk size.

