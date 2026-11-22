"""
比較三種檢索方法：純向量 (vector)、純關鍵字 (bm25)、混合 (hybrid)

Usage（在專案最外層資料夾執行）:
    python evaluation/eval_search.py                                    # 小影片題目 + results/test_video
    python evaluation/eval_search.py evaluation/questions/lecture2.json results/lecture2/chroma_db
    或直接雙擊 tools/run_eval.bat

題目檔格式：每題有問題 q、類型 type，以及下面其中一種「正確答案」：
    anchors  : 答案在影片中出現的時間點（秒）。自動對應到包含這些時間點的片段，
               重新切片段後題目仍然有效（建議用這種）
    relevant : 直接指定片段 id（舊格式，重新切片段後就會失效）

評估指標：
    Hit@1 : 第 1 名就是正確片段的比例
    Hit@3 : 前 3 名內有正確片段的比例（search.py 取前 3 名）
    Hit@5 : 前 5 名內有正確片段的比例（answer.py 讓 Gemini 讀前 5 名）
    MRR   : 第一個正確片段名次的倒數平均（第 1 名=1，第 2 名=0.5，第 3 名=0.33…，沒找到=0）

「+ rerank」= 混合檢索選出候選後，再用 cross-encoder（bge-reranker-v2-m3）重新排序。
用影片庫（library/chroma_db，裡面有很多支影片）評估時，會在全部影片裡搜尋（比較難，也比較接近實際），
加 --within-video 則只在題目那支影片裡搜尋。

結果會印在畫面上，並存成 <資料庫上一層>/eval_result.md、eval_detail.csv、eval_summary.json
"""
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

# 讓這個放在 evaluation/ 裡的程式也能 import 最外層的 hybrid.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

WITHIN_VIDEO = "--within-video" in sys.argv
NO_RERANK = "--no-rerank" in sys.argv
ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
QUESTIONS_FILE = ARGS[0] if len(ARGS) > 0 else "evaluation/questions/test_video.json"
DB_DIR = ARGS[1] if len(ARGS) > 1 else "results/test_video/chroma_db"
RESULT_DIR = Path(DB_DIR).parent
COLLECTION = "lecture_chunks"
MODEL_NAME = "BAAI/bge-m3"
# (名稱, 檢索模式, 覆寫的設定)。candidates=None 代表每種方法都排完所有片段
CONFIGS = [
    ("vector", "vector", {"candidates": None}),
    ("bm25", "bm25", {"candidates": None}),
    ("hybrid (all, w=1)", "hybrid", {"candidates": None, "bm25_weight": 1.0}),
    ("hybrid (top10, w=1)", "hybrid", {"candidates": 10, "bm25_weight": 1.0}),
    ("hybrid (top10, w=0.5)", "hybrid", {"candidates": 10, "bm25_weight": 0.5}),
    ("hybrid (top10, w=0.3)", "hybrid", {"candidates": 10, "bm25_weight": 0.3}),
    ("vector + rerank", "vector", {"candidates": 20, "rerank": True}),
    ("hybrid + rerank", "hybrid", {"candidates": 10, "bm25_weight": 1.0, "rerank": True}),
    ("hybrid20 + rerank", "hybrid", {"candidates": 20, "bm25_weight": 1.0, "rerank": True}),
]

import chromadb
import torch
from sentence_transformers import SentenceTransformer

from hybrid import HybridRetriever, load_reranker, load_settings

with open(QUESTIONS_FILE, encoding="utf-8") as f:
    qdata = json.load(f)
questions = qdata["questions"]

collection = chromadb.PersistentClient(path=DB_DIR).get_collection(COLLECTION)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading BGE-M3 on {DEVICE}...")
model = SentenceTransformer(MODEL_NAME, device=DEVICE)

settings = load_settings()
reranker = None if NO_RERANK else load_reranker({**settings, "rerank": True}, DEVICE)
if reranker is None:
    CONFIGS = [c for c in CONFIGS if not c[2].get("rerank")]
MODES = [c[0] for c in CONFIGS]
retriever = HybridRetriever(collection, model, settings, reranker)
base = dict(retriever.s)

# 影片庫裡有很多支影片：答案只能是題目那支影片的片段
video = qdata.get("video")
multi = len(retriever.videos) > 1
if multi:
    if not video or video not in retriever.videos:
        sys.exit(f"Question file has video={video!r}, not in this database: {retriever.videos}")
    RESULT_DIR = Path(DB_DIR).parent / f"eval_{video}{'_within' if WITHIN_VIDEO else ''}"
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
search_videos = [video] if (multi and WITHIN_VIDEO) else None
n_chunks = sum(1 for m in retriever.metas if not search_videos or m.get("video") in search_videos)

# 把 anchors（時間點）換成這個資料庫裡的片段 id
spans = {cid: (m["start"], m["end"]) for cid, m in zip(retriever.ids, retriever.metas)
         if not multi or m.get("video") == video}
for item in questions:
    if "anchors" in item:
        item["relevant"] = [cid for cid, (a, b) in spans.items()
                            if any(a - 1 <= t <= b + 1 for t in item["anchors"])]
        if not item["relevant"]:
            sys.exit(f"No chunk covers the time {item['anchors']} for question: {item['q']}")

# 檢查題目裡的片段 id 是否存在（舊格式）
missing = {r for q in questions for r in q["relevant"]} - set(retriever.ids)
if missing:
    sys.exit(f"These chunk ids are not in the database: {sorted(missing)}\n"
             f"Check the question file and the database folder.")

print(f"{len(questions)} questions, {n_chunks} chunks in database\n")

rows = []
for item in questions:
    relevant = set(item["relevant"])
    for name, mode, overrides in CONFIGS:
        retriever.s = {**base, **overrides}
        if retriever.s["candidates"] is None:
            retriever.s["candidates"] = n_chunks
        t = time.time()
        results = retriever.search(item["q"], top_k=n_chunks, mode=mode,
                                   rerank=bool(overrides.get("rerank")), videos=search_videos)
        ms = (time.time() - t) * 1000
        ranked = [r["id"] for r in results]
        rank = next((i + 1 for i, cid in enumerate(ranked) if cid in relevant), None)
        rows.append({
            "question": item["q"], "type": item["type"], "mode": name,
            "rank": rank, "top3": " ".join(ranked[:3]), "ms": round(ms, 1),
        })

# ------------------------------------------------------------------
# 統計
# ------------------------------------------------------------------
def metrics(subset):
    n = len(subset)
    if not n:
        return None
    hit1 = sum(1 for r in subset if r["rank"] == 1) / n
    hit3 = sum(1 for r in subset if r["rank"] and r["rank"] <= 3) / n
    hit5 = sum(1 for r in subset if r["rank"] and r["rank"] <= 5) / n
    mrr = sum(1 / r["rank"] for r in subset if r["rank"]) / n
    ms = sum(r["ms"] for r in subset) / n
    return hit1, hit3, mrr, ms, hit5


types = list(dict.fromkeys(q["type"] for q in questions))  # 依題目檔出現的順序
groups = [("ALL", None)] + [(t, t) for t in types]
lines = ["# 檢索方法比較（vector / bm25 / hybrid）", "",
         f"- 資料庫：{DB_DIR}　題目檔：{QUESTIONS_FILE}",
         "- all = 每種方法排完所有片段再合併；top10 = 各取前 10 名再合併（系統實際設定）；w = bm25_weight",
         f"- 題目數：{len(questions)}　片段數：{n_chunks}　向量模型：{MODEL_NAME}（{DEVICE}）",
         f"- RRF：k={settings['rrf_k']}、vector_weight={settings['vector_weight']}",
         f"- 重新排序模型：{settings['rerank_model'] if reranker else '（沒有載入）'}"
         + ("　搜尋範圍：影片庫全部影片" if multi and not WITHIN_VIDEO else ""), ""]

for title, t in groups:
    count = len([q for q in questions if t is None or q["type"] == t])
    lines += [f"## {title}（{count} 題）", "",
              "| 方法 | Hit@1 | Hit@3 | Hit@5 | MRR | 平均時間 (ms) |", "|---|---|---|---|---|---|"]
    for mode in MODES:
        m = metrics([r for r in rows if r["mode"] == mode and (t is None or r["type"] == t)])
        if m:
            lines.append(f"| {mode} | {m[0]:.0%} | {m[1]:.0%} | {m[4]:.0%} | {m[2]:.3f} | {m[3]:.0f} |")
    lines.append("")

# 每題名次對照：最能看出差別
lines += ["## 每題正確片段的名次（越小越好，- = 沒找到）", "",
          "| 類型 | 問題 | " + " | ".join(MODES) + " |", "|---|---|" + "---|" * len(MODES)]
by_q = defaultdict(dict)
for r in rows:
    by_q[(r["type"], r["question"])][r["mode"]] = r["rank"]
for (t, q), ranks in by_q.items():
    cells = [str(ranks[m]) if ranks[m] else "-" for m in MODES]
    lines.append(f"| {t} | {q} | " + " | ".join(cells) + " |")

report = "\n".join(lines)
print(report)

Path(RESULT_DIR / "eval_result.md").write_text(report + "\n", encoding="utf-8")
with open(RESULT_DIR / "eval_detail.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
summary = {"questions": len(questions), "chunks": n_chunks, "methods": {}}
for name in MODES:
    summary["methods"][name] = {}
    for title, t in groups:
        m = metrics([r for r in rows if r["mode"] == name and (t is None or r["type"] == t)])
        if m:
            summary["methods"][name][title] = {"hit1": round(m[0], 3), "hit3": round(m[1], 3),
                                               "hit5": round(m[4], 3),
                                               "mrr": round(m[2], 3), "ms": round(m[3], 1)}
json.dump(summary, open(RESULT_DIR / "eval_summary.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
print(f"\nSaved: {RESULT_DIR / 'eval_result.md'}, {RESULT_DIR / 'eval_detail.csv'}, "
      f"{RESULT_DIR / 'eval_summary.json'}")
