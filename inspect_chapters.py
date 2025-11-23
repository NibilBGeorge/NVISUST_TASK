# inspect_chapters.py
import pickle, collections, json
m = pickle.load(open("vectorstore/metas.pkl","rb"))
chapters = [ (meta.get("chapter") or "Unknown") for meta in m["metas"] ]
counts = collections.Counter(chapters)
print(json.dumps(counts.most_common(50), ensure_ascii=False, indent=2))
# show first 10 items to inspect sample meta
print("\nSample metas (first 10):")
for meta in m["metas"][:10]:
    print(json.dumps({
        "id": meta.get("id"),
        "chapter": meta.get("chapter"),
        "page": meta.get("page"),
        "snippet": meta.get("text_snippet")[:140].replace("\n"," ")
    }, ensure_ascii=False))
