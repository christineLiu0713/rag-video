"""
問答核心（answer.py、評估程式、之後的網站都用這支）

    問題 ─► 檢索：BM25 + 向量（RRF 合併）─► 重新排序（cross-encoder）
         ─► 過濾不相關的片段（都不相關就直接回答「影片裡找不到」，不呼叫 Gemini）
         ─► Gemini 只讀這些片段，回傳 JSON：{found, answer, used_parts}
         ─► 程式把 used_parts 換成「影片名稱 + 時間」當出處（時間由程式產生，不會被模型亂編）

用法（程式內）：
    from rag import VideoRAG
    rag = VideoRAG()                      # 預設用 library/chroma_db（全部影片），沒有就用 output/chroma_db
    result = rag.ask("什麼是 union bound？")
    print(result["answer"], result["sources"])

Gemini 金鑰：環境變數 GEMINI_API_KEY，或放在專案資料夾的 gemini_key.txt（已在 .gitignore）。
"""
import json
import os
import re
import time
from pathlib import Path

from hybrid import HybridRetriever, load_reranker, load_settings

ROOT = Path(__file__).resolve().parent
COLLECTION = "lecture_chunks"
EMBED_MODEL = "BAAI/bge-m3"

SYSTEM_PROMPT = """You are a teaching assistant. You answer questions about lecture videos using ONLY the numbered video parts given by the user.

Return JSON with:
- "found": true if the parts contain the answer, false if they do not.
- "answer": a short, direct answer (1 to 4 sentences) in your own words.
  If found is false, say briefly that the video does not cover this.
- "used_parts": the numbers of the parts you used, e.g. [2, 3]. Empty if found is false.

Rules:
- Never use outside knowledge to fill in facts that are not in the parts. Never guess.
- Do not write times or part numbers inside "answer"; the program adds the sources.
- Write the answer in the language of the question. For Chinese questions use Traditional Chinese (Taiwan).
- "Speech" is automatic speech recognition: names and rare terms may be misheard.
  "Slide" text is OCR: it may contain mistakes or simplified characters.
  Prefer the speech text for facts, and quietly correct obvious recognition errors."""

ANSWER_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "found": {"type": "BOOLEAN"},
        "answer": {"type": "STRING"},
        "used_parts": {"type": "ARRAY", "items": {"type": "INTEGER"}},
    },
    "required": ["found", "answer", "used_parts"],
}

NOT_FOUND_TEXT = {"zh": "影片裡找不到和這個問題相關的內容。",
                  "en": "I could not find this in the video."}


def mmss(seconds):
    seconds = int(seconds)
    h, m, s = seconds // 3600, seconds % 3600 // 60, seconds % 60
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def is_chinese(text):
    return bool(re.search(r"[一-鿿]", text))


def load_api_key():
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key and (ROOT / "gemini_key.txt").exists():
        key = (ROOT / "gemini_key.txt").read_text(encoding="utf-8").strip().splitlines()[0].strip()
    return key


def default_db():
    for p in (ROOT / "library" / "chroma_db", ROOT / "output" / "chroma_db"):
        if p.is_dir():
            return p
    return ROOT / "output" / "chroma_db"


def parse_json_reply(text):
    """Gemini 的回覆轉成 dict；萬一不是合法 JSON，盡量救回來。"""
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    parts = [int(x) for x in re.findall(r"\[(\d+)\]", text)]
    return {"found": True, "answer": text, "used_parts": parts}


class Gemini:
    """呼叫 Gemini 並處理暫時性錯誤（429 次數上限、500/503 伺服器忙）。"""

    def __init__(self, model, api_key=None):
        from google import genai
        from google.genai import types
        api_key = api_key or load_api_key()
        if not api_key:
            raise RuntimeError("Gemini key not found: set GEMINI_API_KEY, "
                               "or put the key in gemini_key.txt in the project folder.")
        self.types = types
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.use_schema = True

    def json(self, system, user, schema=None, retries=4):
        wait = 5
        for attempt in range(retries + 1):
            try:
                cfg = dict(system_instruction=system, temperature=0.2,
                           response_mime_type="application/json")
                if schema and self.use_schema:
                    cfg["response_schema"] = schema
                resp = self.client.models.generate_content(
                    model=self.model, contents=user,
                    config=self.types.GenerateContentConfig(**cfg))
                return parse_json_reply(resp.text)
            except Exception as e:
                msg = str(e)
                if "schema" in msg.lower() and self.use_schema:
                    self.use_schema = False          # 舊版套件不支援 schema：改成只要求 JSON
                    continue
                temporary = any(c in msg for c in ("429", "500", "503", "RESOURCE_EXHAUSTED",
                                                   "UNAVAILABLE", "DEADLINE"))
                if not temporary or attempt == retries:
                    raise
                time.sleep(wait)
                wait = min(wait * 2, 60)


class VideoRAG:
    def __init__(self, db_dir=None, settings=None, use_llm=True, reranker="auto", embedder=None):
        import chromadb

        self.s = settings or load_settings(ROOT / "settings.json")
        self.db_dir = Path(db_dir) if db_dir else default_db()
        try:
            collection = chromadb.PersistentClient(path=str(self.db_dir)).get_collection(COLLECTION)
        except Exception:
            raise RuntimeError(f"Database not found: {self.db_dir}. "
                               f"Run pipeline.py (or build_db.py / build_library.py) first.")
        if embedder is None:
            import torch
            from sentence_transformers import SentenceTransformer
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Loading BGE-M3 on {device}...")
            embedder = SentenceTransformer(EMBED_MODEL, device=device)
        if reranker == "auto":
            reranker = load_reranker(self.s)
        self.retriever = HybridRetriever(collection, embedder, self.s, reranker)
        self.videos = self.retriever.videos
        self.llm = Gemini(self.s["gemini_model"]) if use_llm else None

    # ---------------- 檢索 ----------------
    def related(self, r):
        """這個片段算不算「和問題有關」。"""
        if r.get("rerank_score") is not None:
            return r["rerank_score"] >= self.s["min_rerank"]
        return (r["distance"] is not None and r["distance"] <= self.s["max_distance"]) \
            or r["bm25_score"] >= self.s["min_bm25"]

    def retrieve(self, question, videos=None, top_k=None):
        results = self.retriever.search(question, top_k or self.s["answer_top_k"], videos=videos)
        return [r for r in results if self.related(r)], results

    # ---------------- 問答 ----------------
    def ask(self, question, videos=None):
        lang = "zh" if is_chinese(question) else "en"
        t0 = time.time()
        parts, all_results = self.retrieve(question, videos)
        retrieve_ms = (time.time() - t0) * 1000
        result = {"question": question, "status": "ok", "found": False, "answer": "",
                  "sources": [], "parts": [self._part_info(n, p) for n, p in enumerate(parts, 1)],
                  "candidates": [self._part_info(n, p) for n, p in enumerate(all_results, 1)],
                  "retrieve_ms": round(retrieve_ms), "llm_ms": 0, "error": None}

        if not parts:
            result.update(status="not_found", answer=NOT_FOUND_TEXT[lang])
            return result
        if self.llm is None:
            result.update(status="no_llm", answer="(Gemini not loaded)")
            return result

        many_videos = len({p["meta"].get("video") for p in parts}) > 1 or len(self.videos) > 1
        context = "\n\n".join(
            f"[Part {n} | " + (f"video: {p['meta'].get('video')} | " if many_videos else "")
            + f"{mmss(p['meta']['start'])}-{mmss(p['meta']['end'])}]\n{p['text']}"
            for n, p in enumerate(parts, 1))
        user = f"Video parts:\n\n{context}\n\nQuestion: {question}"

        t1 = time.time()
        try:
            reply = self.llm.json(SYSTEM_PROMPT, user, ANSWER_SCHEMA)
        except Exception as e:
            result.update(status="error", error=str(e),
                          answer="Gemini error (404 = wrong model name; 400/401/403 = wrong key; "
                                 "429 = free limit reached, wait a bit)")
            return result
        result["llm_ms"] = round((time.time() - t1) * 1000)

        found = bool(reply.get("found", True))
        used = []
        for x in reply.get("used_parts") or []:
            try:
                x = int(x)
            except (TypeError, ValueError):
                continue
            if 1 <= x <= len(parts) and x not in used:   # 不存在的編號直接丟掉
                used.append(x)
        result.update(found=found, answer=str(reply.get("answer", "")).strip(),
                      status="ok" if found else "not_found",
                      cited_valid=len(used) == len(reply.get("used_parts") or []))
        if found:
            result["sources"] = [result["parts"][n - 1] for n in sorted(used)]
        return result

    @staticmethod
    def _part_info(n, p):
        m = p["meta"]
        return {"n": n, "id": p["id"], "video": m.get("video"), "start": m["start"], "end": m["end"],
                "time": f"{mmss(m['start'])}-{mmss(m['end'])}", "image": m.get("image", ""),
                "result_dir": m.get("result_dir", ""),
                "rerank": None if p.get("rerank_score") is None else round(p["rerank_score"], 4),
                "distance": None if p.get("distance") is None else round(p["distance"], 4),
                "bm25": round(p.get("bm25_score") or 0, 3)}


def format_result(r, show_video=True):
    """給命令列看的文字版。"""
    lines = [r["answer"]]
    if r["sources"]:
        lines.append("\n出處 / Sources:")
        for s in r["sources"]:
            v = f"{s['video']}  " if show_video and s["video"] else ""
            lines.append(f"  {v}{s['time']}" + (f"   {s['image']}" if s["image"] else ""))
    return "\n".join(lines)
