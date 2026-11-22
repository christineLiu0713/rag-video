"""
評估「最後的回答」，不只評估檢索：問題 -> rag.py 回答 -> 打分數

每一題會記錄：
    檢索命中   : Gemini 讀到的片段裡，有沒有包含答案的片段（anchors 時間點）
    引用正確   : 回答附上的出處，有沒有指到答案所在的片段
    回答評分   : 另外請 Gemini 當評審，拿「答案所在片段的逐字稿」當標準答案，
                 判定 correct（正確）/ partial（部分正確）/ wrong（錯誤），
                 並檢查有沒有講出片段裡沒有的內容（unsupported = 疑似亂編）
    錯誤拒答   : 影片裡明明有答案，系統卻說找不到
另外用 evaluation/unanswerable.json（影片裡沒有的問題）測「該拒答時有沒有拒答」。

Usage（在專案最外層資料夾執行）:
    python evaluation/eval_answer.py                       # 測試集全部影片 + 拒答題，搜尋整個影片庫
    python evaluation/eval_answer.py --only mit_prob_l01   # 只測某幾支
    python evaluation/eval_answer.py --within-video        # 只在題目那支影片裡搜尋
    python evaluation/eval_answer.py --limit 5             # 每支影片只測 5 題（先試跑）
    python evaluation/eval_answer.py --fresh               # 不沿用上次的結果，全部重問

每題的結果一做完就寫進 results/answer_eval/answers.jsonl，中斷（例如 Gemini 免費額度用完）
後再執行會接著做。最後產生 results/answer_eval/answer_eval.md 與 answers.csv（可人工複查）。

注意：評審也是 Gemini，會有偏差；報告時建議人工抽查 answers.csv 裡的一部分。
"""
import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag import VideoRAG, default_db  # noqa: E402

OUT_DIR = ROOT / "results" / "answer_eval"

JUDGE_PROMPT = """You grade answers produced by a question-answering system for lecture videos.
You get: the question, REFERENCE passages (the part of the lecture transcript where the answer is),
the CONTEXT passages the system actually read, and the system's ANSWER.

Return JSON:
- "verdict": "correct" if the answer answers the question and agrees with the reference;
             "partial" if it is partly right, incomplete, or too vague to be useful;
             "wrong" if it is wrong, contradicts the reference, or answers a different question.
- "unsupported": true if the answer states facts that appear in neither the reference nor the context
                 (made-up details), otherwise false.
- "reason": one short sentence.
The transcript comes from speech recognition and may contain misheard words; do not punish an answer
for correcting obvious recognition errors. Answers may be in Chinese or English: judge the meaning."""

JUDGE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "verdict": {"type": "STRING", "enum": ["correct", "partial", "wrong"]},
        "unsupported": {"type": "BOOLEAN"},
        "reason": {"type": "STRING"},
    },
    "required": ["verdict", "unsupported", "reason"],
}


def covers(start, end, anchors):
    return any(start - 1 <= t <= end + 1 for t in anchors)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--testset", default="evaluation/testset.json")
    ap.add_argument("--only", help="comma-separated video names")
    ap.add_argument("--db", help="default: library/chroma_db")
    ap.add_argument("--within-video", action="store_true")
    ap.add_argument("--limit", type=int, help="questions per video")
    ap.add_argument("--delay", type=float, default=4.0, help="seconds between Gemini calls (free limit)")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--no-unanswerable", action="store_true")
    args = ap.parse_args()

    testset = json.load(open(ROOT / args.testset, encoding="utf-8"))["videos"]
    if args.only:
        wanted = set(args.only.split(","))
        testset = [v for v in testset if v["name"] in wanted]

    rag = VideoRAG(args.db or default_db())
    print(f"Database: {rag.db_dir}  videos: {', '.join(rag.videos)}")
    print(f"Re-ranker: {'on' if rag.retriever.reranker else 'off'}   model: {rag.s['gemini_model']}")
    r_index = rag.retriever

    # ---------------- 題目 ----------------
    jobs = []   # (video_or_None, question dict)
    for item in testset:
        if item["name"] not in rag.videos:
            print(f"  {item['name']}: not in the database, skipped")
            continue
        qfile = ROOT / item["questions"]
        if not qfile.exists():
            continue
        qs = [q for q in json.load(open(qfile, encoding="utf-8"))["questions"] if "anchors" in q]
        if args.limit:
            qs = qs[:args.limit]
        jobs += [(item["name"], q) for q in qs]
    if not args.no_unanswerable:
        for q in json.load(open(ROOT / "evaluation" / "unanswerable.json", encoding="utf-8"))["questions"]:
            jobs.append((None, q))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = OUT_DIR / "answers.jsonl"
    done = {}
    if cache_file.exists() and not args.fresh:
        for line in open(cache_file, encoding="utf-8"):
            d = json.loads(line)
            done[(d["video"], d["question"])] = d
    if args.fresh and cache_file.exists():
        cache_file.unlink()
    todo = [(v, q) for v, q in jobs if (v, q["q"]) not in done]
    print(f"{len(jobs)} questions ({len(jobs) - len(todo)} already done)\n")

    # ---------------- 逐題回答＋評分 ----------------
    with open(cache_file, "a", encoding="utf-8") as cache:
        for n, (video, q) in enumerate(todo, 1):
            videos = [video] if (video and args.within_video) else None
            res = rag.ask(q["q"], videos)
            time.sleep(args.delay if res["llm_ms"] else 0)
            row = {"video": video, "question": q["q"], "type": q.get("type", ""),
                   "status": res["status"], "found": res["found"], "answer": res["answer"],
                   "sources": [f"{s['video']} {s['time']}" for s in res["sources"]],
                   "retrieve_ms": res["retrieve_ms"], "llm_ms": res["llm_ms"], "error": res["error"],
                   "cited_valid": res.get("cited_valid")}

            if video is None:                                   # 影片裡沒有答案的題目
                row["verdict"] = "refused_ok" if not res["found"] else "answered_wrongly"
            elif res["status"] == "error":
                row["verdict"] = "error"
            else:
                anchors = q["anchors"]
                gold = [i for i, m in enumerate(r_index.metas)
                        if m.get("video") == video and covers(m["start"], m["end"], anchors)]
                row["retrieval_hit"] = any(p["video"] == video and covers(p["start"], p["end"], anchors)
                                           for p in res["parts"])
                row["cited_correct"] = any(s["video"] == video and covers(s["start"], s["end"], anchors)
                                           for s in res["sources"])
                if not res["found"]:
                    row["verdict"] = "false_refusal"
                else:
                    reference = "\n\n".join(r_index.docs[i] for i in gold)
                    context = "\n\n".join(r_index.docs[r_index.index[p["id"]]] for p in res["parts"])
                    user = (f"Question: {q['q']}\n\nREFERENCE:\n{reference}\n\nCONTEXT:\n{context}"
                            f"\n\nANSWER:\n{res['answer']}")
                    try:
                        j = rag.llm.json(JUDGE_PROMPT, user, JUDGE_SCHEMA)
                        row.update(verdict=j.get("verdict", "wrong"),
                                   unsupported=bool(j.get("unsupported")), judge_reason=j.get("reason", ""))
                    except Exception as e:
                        row.update(verdict="judge_error", judge_reason=str(e)[:200])
                    time.sleep(args.delay)

            cache.write(json.dumps(row, ensure_ascii=False) + "\n")
            cache.flush()
            done[(video, q["q"])] = row
            print(f"[{n}/{len(todo)}] {video or '(unanswerable)'} | {row['verdict']:16s} | {q['q'][:50]}",
                  flush=True)
            if row["verdict"] == "error" and "429" in (row["error"] or ""):
                print("\nGemini free limit reached. Run this again later: it continues from here.")
                break

    rows = [done[(v, q["q"])] for v, q in jobs if (v, q["q"]) in done]
    write_report(rows, rag)


def write_report(rows, rag):
    def pct(a, b):
        return f"{a / b:.0%}" if b else "-"

    answerable = [r for r in rows if r["video"] and r["verdict"] not in ("error", "judge_error")]
    unans = [r for r in rows if not r["video"]]
    groups = defaultdict(list)
    for r in answerable:
        groups[r["video"]].append(r)

    lines = [f"# 回答品質評估（{datetime.now():%Y-%m-%d %H:%M}）", "",
             f"- 資料庫：{rag.db_dir.relative_to(ROOT) if rag.db_dir.is_relative_to(ROOT) else rag.db_dir}"
             f"（{len(rag.videos)} 支影片）　生成模型：{rag.s['gemini_model']}　評審：同一個模型",
             f"- 重新排序：{'開' if rag.retriever.reranker else '關'}　Gemini 讀前 {rag.s['answer_top_k']} 段"
             f"　不相關門檻：rerank < {rag.s['min_rerank']}", "",
             "## 有答案的題目", "",
             "| 影片 | 題數 | 正確 | 部分正確 | 錯誤 | 錯誤拒答 | 檢索命中 | 引用正確 | 疑似亂編 |",
             "|---|---|---|---|---|---|---|---|---|"]

    def line(name, rs):
        n = len(rs)
        c = sum(r["verdict"] == "correct" for r in rs)
        p = sum(r["verdict"] == "partial" for r in rs)
        w = sum(r["verdict"] == "wrong" for r in rs)
        fr = sum(r["verdict"] == "false_refusal" for r in rs)
        hit = sum(bool(r.get("retrieval_hit")) for r in rs)
        found = [r for r in rs if r["found"]]
        cited = sum(bool(r.get("cited_correct")) for r in found)
        uns = sum(bool(r.get("unsupported")) for r in found)
        return (f"| {name} | {n} | {pct(c, n)} | {pct(p, n)} | {pct(w, n)} | {pct(fr, n)} | "
                f"{pct(hit, n)} | {pct(cited, len(found))} | {pct(uns, len(found))} |")

    for name, rs in groups.items():
        lines.append(line(name, rs))
    if len(groups) > 1:
        lines.append(line("**全部**", answerable))

    refused = sum(r["verdict"] == "refused_ok" for r in unans)
    lines += ["", "## 影片裡沒有答案的題目（應該拒答）", "",
              f"正確拒答：{refused}/{len(unans)}（{pct(refused, len(unans))}）", ""]
    for r in unans:
        if r["verdict"] != "refused_ok":
            lines.append(f"- 沒拒答：{r['question']} → {r['answer'][:80]}")

    ms = [r for r in rows if r.get("llm_ms")]
    if ms:
        lines += ["", f"平均時間：檢索 {sum(r['retrieve_ms'] for r in ms) / len(ms):.0f} ms，"
                      f"Gemini {sum(r['llm_ms'] for r in ms) / len(ms):.0f} ms", ""]

    bad = [r for r in answerable if r["verdict"] in ("wrong", "partial", "false_refusal")]
    if bad:
        lines += ["## 沒答好的題目（請人工確認）", "",
                  "| 影片 | 問題 | 評分 | 檢索命中 | 回答 | 評審理由 |", "|---|---|---|---|---|---|"]
        for r in bad:
            ans = r["answer"].replace("|", "/").replace("\n", " ")[:120]
            lines.append(f"| {r['video']} | {r['question']} | {r['verdict']} | "
                         f"{'是' if r.get('retrieval_hit') else '否'} | {ans} | "
                         f"{(r.get('judge_reason') or '').replace('|', '/')[:100]} |")

    lines += ["", "說明：",
              "- 正確／部分正確／錯誤：由 Gemini 評審對照「答案所在片段的逐字稿」判定。",
              "- 錯誤拒答：影片裡有答案，系統卻說找不到。",
              "- 檢索命中：Gemini 讀到的片段中包含答案所在片段。",
              "- 引用正確：在有回答的題目中，附上的出處指到答案所在片段的比例。",
              "- 疑似亂編：回答中有參考片段和讀到的片段都沒有的內容。"]
    report = "\n".join(lines) + "\n"
    (OUT_DIR / "answer_eval.md").write_text(report, encoding="utf-8")
    keys = ["video", "type", "question", "verdict", "found", "retrieval_hit", "cited_correct",
            "unsupported", "answer", "sources", "judge_reason", "retrieve_ms", "llm_ms", "error"]
    with open(OUT_DIR / "answers.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({**r, "sources": "; ".join(r.get("sources") or [])})
    print("\n" + report)
    print(f"Saved: {OUT_DIR.relative_to(ROOT)}/answer_eval.md, answers.csv, answers.jsonl")


if __name__ == "__main__":
    main()
