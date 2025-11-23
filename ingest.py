import argparse
import json
import re
from pathlib import Path
import pdfplumber
from tqdm import tqdm

CHUNK_SIZE = 500  # approx characters per chunk


def extract_text_and_headings(pdf_path: Path):
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            pages.append({"page_no": i, "text": text})
    return pages


def detect_chapter_title(page_text: str):
    lines = [ln.strip() for ln in page_text.splitlines() if ln.strip()]
    if not lines:
        return "Unknown"

    first = lines[0]

    if re.match(r'^(CHAPTER|Chapter|SECTION|Section)\b', first, re.I):
        return first

    if len(first) < 100 and first.upper() == first:
        return first

    if len(first.split()) <= 6 and first[0].isupper():
        return first

    return "Unknown"


def chunk_text(text: str, chunk_size=CHUNK_SIZE):
    words = text.split()
    chunks = []
    current = []
    cur_len = 0
    for w in words:
        current.append(w)
        cur_len += len(w) + 1
        if cur_len >= chunk_size:
            chunks.append(" ".join(current))
            current = []
            cur_len = 0

    if current:
        chunks.append(" ".join(current))

    return chunks


def main(pdf_path: str, out_jsonl: str):
    pdf_path = Path(pdf_path)

    pages = extract_text_and_headings(pdf_path)
    out_path = Path(out_jsonl)

    idx = 0
    with out_path.open("w", encoding="utf8") as f:
        for p in tqdm(pages, desc="Processing PDF Pages"):
            page_no = p["page_no"]
            text = p["text"]
            chapter = detect_chapter_title(text)
            chunks = chunk_text(text)

            for i, chunk in enumerate(chunks):
                idx += 1
                meta = {
                    "id": f"chunk_{idx}",
                    "source": pdf_path.name,
                    "chapter": chapter,
                    "page": page_no,
                    "chunk_index": i,
                    "text": chunk
                }
                f.write(json.dumps(meta, ensure_ascii=False) + "\n")

    print(f"✔ Ingestion finished. Output written to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf_path", required=True)
    parser.add_argument("--out_jsonl", default="./data/chunks.jsonl")
    args = parser.parse_args()

    main(args.pdf_path, args.out_jsonl)



