"""
Step 3b: ask a question -> find the best matching parts of the video
         (hybrid search: BM25 keywords + BGE-M3 vectors, merged with RRF)

Usage:
    python search.py                  # mode from settings.json (default: hybrid)
    python search.py --mode vector    # only vector search (the old way)
    python search.py --mode bm25      # only keyword search
    python search.py --mode compare   # show vector / bm25 / hybrid / hybrid+rerank side by side
    python search.py --db results/lecture2/chroma_db --video lecture2
    (default database: library/chroma_db if it exists, else output/chroma_db)
"""
import argparse
import sys

COLLECTION = "lecture_chunks"
MODEL_NAME = "BAAI/bge-m3"
TOP_K = 3  # how many results to show

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["hybrid", "vector", "bm25", "compare"])
parser.add_argument("--db")
parser.add_argument("--video", help="only these videos, comma-separated")
parser.add_argument("--no-rerank", action="store_true")
args = parser.parse_args()

import chromadb
from sentence_transformers import SentenceTransformer

from hybrid import HybridRetriever, describe, load_reranker, load_settings
from rag import default_db

DB_DIR = args.db or str(default_db())
try:
    collection = chromadb.PersistentClient(path=DB_DIR).get_collection(COLLECTION)
except Exception:
    sys.exit("Database not found. Run build_db.py first.")

print("Loading BGE-M3...")
import torch

# GPU 版 PyTorch 可用時用 CUDA；否則保持 CPU 模式。
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Search embedding device: {DEVICE}")
model = SentenceTransformer(MODEL_NAME, device=DEVICE)

settings = load_settings()
if args.no_rerank:
    settings["rerank"] = False
retriever = HybridRetriever(collection, model, settings, load_reranker(settings))
videos = args.video.split(",") if args.video else None
mode = args.mode or settings["search_mode"]
print(f"Database: {DB_DIR}  videos: {', '.join(retriever.videos)}")
print(f"Search mode: {mode}  re-rank: {'on' if retriever.reranker else 'off'}  "
      f"({len(retriever.ids)} chunks indexed for BM25)")


def mmss(seconds):
    seconds = int(seconds)
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def show(results, title=None):
    if title:
        print(f"\n===== {title} =====")
    if not results:
        print("  (no result)")
    for rank, r in enumerate(results, start=1):
        meta = r["meta"]
        print(f"\n#{rank}  {meta.get('video', '')}  {mmss(meta['start'])} -> {mmss(meta['end'])}   {describe(r)}")
        print(f"   slide : {meta['slide_text'][:80] or '(none)'}")
        print(f"   image : {meta['image']}")
        print(f"   text  : {r['text'][:200]}...")


while True:
    question = input("\nQuestion (empty to quit): ").strip()
    if not question:
        break

    if mode == "compare":
        for m in ("vector", "bm25", "hybrid"):
            show(retriever.search(question, TOP_K, mode=m, rerank=False, videos=videos), title=m.upper())
        if retriever.reranker:
            show(retriever.search(question, TOP_K, mode="hybrid", rerank=True, videos=videos),
                 title="HYBRID + RERANK")
    else:
        show(retriever.search(question, TOP_K, mode=mode, videos=videos))
