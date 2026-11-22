"""
Hybrid search = keyword search (BM25) + meaning search (BGE-M3 vectors)

Why:
    Vector search understands meaning ("repeat code" ~ "loop"), but it can miss
    exact words such as function names, numbers or rare terms ("range", "1,5").
    BM25 is the opposite: it is very good at exact words, but knows nothing
    about meaning. Using both and merging the two rankings gives better results.

How the two rankings are merged: Reciprocal Rank Fusion (RRF)
    score(chunk) = vector_weight / (rrf_k + vector_rank)
                 + bm25_weight   / (rrf_k + bm25_rank)
    Only the rank (1st, 2nd, ...) is used, not the raw scores, so we do not
    need to compare a "distance" with a "BM25 score" directly.

Optional 3rd stage: re-ranking (rerank)
    BM25 and vectors each look at the question and a chunk SEPARATELY (the
    chunk vector was made before the question existed). A cross-encoder
    re-ranker (BAAI/bge-reranker-v2-m3) reads the question and one chunk
    TOGETHER and gives a relevance score, which is much more accurate but too
    slow for every chunk. So: hybrid search picks ~20 candidates, the
    re-ranker sorts only those.

Used by search.py, rag.py / answer.py and the evaluation scripts.
Settings are read from settings.json.
"""
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np

# ------------------------------------------------------------------
# Settings (can be changed in settings.json)
# ------------------------------------------------------------------
DEFAULTS = {
    "search_mode": "hybrid",   # "hybrid", "vector" or "bm25"
    "candidates": 10,          # how many results each method gives before merging
    "rrf_k": 60,               # RRF constant; 60 is the value from the original paper
    "vector_weight": 1.0,      # importance of the meaning ranking
    "bm25_weight": 1.0,        # importance of the keyword ranking
    "bm25_k1": 1.5,            # BM25: how fast repeated words stop adding score
    "bm25_b": 0.75,            # BM25: how much long chunks are penalized
    "min_bm25": 1.0,           # answer.py: a keyword score this high counts as "related"
    "rerank": False,           # 3rd stage: re-rank with a cross-encoder (tested: no gain, +1-3 s on GTX 1060)
    "rerank_model": "BAAI/bge-reranker-v2-m3",
    "rerank_max_length": 512,  # tokens per (question, chunk) pair
    "rerank_batch": 8,
    "to_traditional": True,    # BM25: treat simplified and traditional Chinese as the same (needs opencc)
    # ---- rag.py / answer.py ----
    "gemini_model": "gemini-3.5-flash-lite",
    "answer_top_k": 5,         # how many parts of the video Gemini reads
    "min_rerank": 0.02,        # re-rank score below this = "not related" (0..1)
    "max_distance": 0.95,      # without re-ranker: vector distance above this = "not related"
}


def load_settings(path="settings.json"):
    settings = dict(DEFAULTS)
    p = Path(path)
    if p.exists():
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        settings.update({k: data[k] for k in DEFAULTS if k in data})
    return settings


# ------------------------------------------------------------------
# 1. Tokenizer: text -> list of words for BM25
# ------------------------------------------------------------------
STOPWORDS = set("""
a an and are as at be but by can do does did for from has have how i if in into is it
its it's me my of on or our so that the their them then there these this those to was
we what when where which who why will with you your
slide speech visual title body
""".split())
# "slide/speech/visual/title/body" are the labels build_chunks.py adds to
# every chunk, so they carry no information.

WORD_RE = re.compile(r"[a-z0-9]+|[一-鿿]")


# OCR often outputs simplified characters while questions are in traditional
# Chinese, so "機率" would not match "机率". OpenCC converts everything to
# traditional before counting words. Optional: without it, nothing changes.
#   pip install opencc-python-reimplemented
try:
    from opencc import OpenCC
    _S2T = OpenCC("s2t")
except Exception:
    _S2T = None
USE_S2T = True


def tokenize(text):
    """Lowercase English words / numbers, and Chinese characters as 1- and 2-char pieces."""
    if _S2T is not None and USE_S2T:
        text = _S2T.convert(text)
    raw = WORD_RE.findall(text.lower())
    tokens = [t for t in raw if t not in STOPWORDS]
    # Chinese has no spaces: also add pairs of neighbouring characters (bigrams)
    for a, b in zip(raw, raw[1:]):
        if len(a) == 1 and len(b) == 1 and "一" <= a <= "鿿" and "一" <= b <= "鿿":
            tokens.append(a + b)
    return tokens


# ------------------------------------------------------------------
# 2. BM25 (Okapi BM25), written out so no extra package is needed
# ------------------------------------------------------------------
class BM25:
    def __init__(self, documents, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(d)) for d in documents]
        self.lengths = [sum(c.values()) for c in self.docs]
        self.avg_len = (sum(self.lengths) / len(self.lengths)) if self.docs else 0
        n = len(self.docs)
        df = Counter()                      # in how many chunks each word appears
        for c in self.docs:
            df.update(c.keys())
        # IDF: rare words are worth more than common words
        self.idf = {w: math.log((n - f + 0.5) / (f + 0.5) + 1) for w, f in df.items()}

    def scores(self, query):
        q_words = tokenize(query)
        result = []
        for counts, length in zip(self.docs, self.lengths):
            s = 0.0
            for w in q_words:
                tf = counts.get(w, 0)
                if tf:
                    norm = self.k1 * (1 - self.b + self.b * length / self.avg_len)
                    s += self.idf[w] * tf * (self.k1 + 1) / (tf + norm)
            result.append(s)
        return result


# ------------------------------------------------------------------
# 3. Re-ranker (cross-encoder), loaded only when needed
# ------------------------------------------------------------------
class Reranker:
    def __init__(self, model_name, device=None, max_length=512, batch_size=8):
        import torch
        from sentence_transformers import CrossEncoder
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.model = CrossEncoder(model_name, device=self.device, max_length=max_length)
        # Half precision halves GPU memory and is faster on newer cards (RTX 20xx and up).
        # Older cards such as the GTX 10xx compute half precision very slowly, so keep full.
        if self.device == "cuda":
            try:
                if torch.cuda.get_device_capability()[0] >= 7:
                    self.model.model.half()
            except Exception:
                pass

    def score(self, question, texts):
        """Relevance 0..1 for each text (higher = more related)."""
        if not texts:
            return []
        pairs = [(question, t) for t in texts]
        # a 1-output cross-encoder applies a sigmoid by default -> 0..1
        scores = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        return [float(x) for x in scores]


def load_reranker(settings, device=None):
    """Return a Reranker, or None (with a message) if it cannot be loaded."""
    if not settings.get("rerank"):
        return None
    try:
        print(f"Loading re-ranker {settings['rerank_model']} (first time: downloading ~2 GB)...")
        return Reranker(settings["rerank_model"], device, settings["rerank_max_length"],
                        settings["rerank_batch"])
    except Exception as e:
        print(f"  re-ranker not available ({e}); continuing without it")
        return None


# ------------------------------------------------------------------
# 4. Hybrid retriever
# ------------------------------------------------------------------
class HybridRetriever:
    def __init__(self, collection, embedder, settings=None, reranker=None):
        global USE_S2T
        self.collection = collection
        self.embedder = embedder
        self.s = settings or load_settings()
        self.reranker = reranker
        USE_S2T = bool(self.s.get("to_traditional", True))

        # BM25 needs every chunk's text. Read it from the Chroma database that
        # build_db.py already made, so nothing has to be rebuilt.
        data = collection.get(include=["documents", "metadatas"])
        self.ids = data["ids"]
        self.docs = data["documents"]
        self.metas = data["metadatas"]
        self.index = {cid: i for i, cid in enumerate(self.ids)}
        self.bm25 = BM25(self.docs, k1=self.s["bm25_k1"], b=self.s["bm25_b"])
        self.videos = sorted({m.get("video", "") for m in self.metas})

    def _vector_ranking(self, question, n, videos=None):
        vector = self.embedder.encode([question], normalize_embeddings=True)[0]
        where = None
        if videos:
            where = {"video": videos[0]} if len(videos) == 1 else {"video": {"$in": list(videos)}}
        res = self.collection.query(query_embeddings=[vector.tolist()], n_results=n, where=where)
        ids = res["ids"][0]
        distances = dict(zip(ids, res["distances"][0]))
        return ids, distances, vector

    def _bm25_ranking(self, question, n, videos=None):
        scores = self.bm25.scores(question)
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        order = [i for i in order if scores[i] > 0]       # 0 = no shared word
        if videos:
            order = [i for i in order if self.metas[i].get("video") in videos]
        order = order[:n]
        return [self.ids[i] for i in order], {self.ids[i]: scores[i] for i in order}

    def _fill_distances(self, ids, vector, distances):
        """Chunks found only by BM25 have no vector distance yet: compute it
        the same way Chroma does (squared L2 on normalized vectors)."""
        missing = [i for i in ids if i not in distances]
        if not missing:
            return
        got = self.collection.get(ids=missing, include=["embeddings"])
        for cid, emb in zip(got["ids"], got["embeddings"]):
            diff = np.asarray(emb, dtype=float) - vector
            distances[cid] = float(np.dot(diff, diff))

    def search(self, question, top_k=3, mode=None, rerank=None, videos=None):
        """mode: hybrid / vector / bm25.  rerank: None = follow settings.
        videos: only search these videos (list of names), None = all."""
        mode = mode or self.s["search_mode"]
        if rerank is None:
            rerank = bool(self.s.get("rerank")) and self.reranker is not None
        pool_size = sum(1 for m in self.metas if not videos or m.get("video") in videos)
        n = min(self.s["candidates"], pool_size)
        k = self.s["rrf_k"]
        if n == 0:
            return []

        vec_ids, distances, vector = ([], {}, None)
        bm_ids, bm_scores = ([], {})
        if mode in ("hybrid", "vector"):
            vec_ids, distances, vector = self._vector_ranking(question, n, videos)
        if mode in ("hybrid", "bm25"):
            bm_ids, bm_scores = self._bm25_ranking(question, n, videos)

        vec_rank = {cid: r for r, cid in enumerate(vec_ids, start=1)}
        bm_rank = {cid: r for r, cid in enumerate(bm_ids, start=1)}

        # Reciprocal Rank Fusion
        fused = {}
        for cid, r in vec_rank.items():
            fused[cid] = fused.get(cid, 0) + self.s["vector_weight"] / (k + r)
        for cid, r in bm_rank.items():
            fused[cid] = fused.get(cid, 0) + self.s["bm25_weight"] / (k + r)

        ranked = sorted(fused, key=fused.get, reverse=True)
        rerank_scores = {}
        if rerank and self.reranker is not None and ranked:
            # every candidate from the two lists is re-scored, then sorted again
            texts = [self.docs[self.index[cid]] for cid in ranked]
            rerank_scores = dict(zip(ranked, self.reranker.score(question, texts)))
            ranked = sorted(ranked, key=lambda c: rerank_scores[c], reverse=True)
        best = ranked[:top_k]

        if vector is None:  # bm25 mode: still compute distances for display/filtering
            vector = self.embedder.encode([question], normalize_embeddings=True)[0]
        self._fill_distances(best, np.asarray(vector, dtype=float), distances)

        results = []
        for cid in best:
            i = self.index[cid]
            results.append({
                "id": cid,
                "meta": self.metas[i],
                "text": self.docs[i],
                "rrf": fused[cid],
                "distance": distances.get(cid),
                "vector_rank": vec_rank.get(cid),   # None = not in vector top list
                "bm25_rank": bm_rank.get(cid),      # None = no keyword match
                "bm25_score": bm_scores.get(cid, 0.0),
                "rerank_score": rerank_scores.get(cid),  # None = not re-ranked
            })
        return results


def describe(r):
    """One short line explaining where a result came from (for printing)."""
    v = f"vector #{r['vector_rank']}" if r["vector_rank"] else "vector -"
    b = f"BM25 #{r['bm25_rank']} ({r['bm25_score']:.2f})" if r["bm25_rank"] else "BM25 -"
    d = f"distance {r['distance']:.3f}" if r["distance"] is not None else ""
    rr = f" | rerank {r['rerank_score']:.3f}" if r.get("rerank_score") is not None else ""
    return f"RRF {r['rrf']:.4f} | {v} | {b} | {d}{rr}"
