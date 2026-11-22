"""
把處理好的影片匯出成一個「課程資料檔」，給其他功能直接使用（出考卷、做摘要、做字卡…）。
不需要 GPU、Chroma 或任何模型，只讀 results/<影片>/ 裡已經產生的 JSON。

Usage:
    python export_dataset.py                      # 匯出 results/ 裡所有影片
    python export_dataset.py --only mit_prob_l01  # 只匯出某幾支（逗號分隔）
    python export_dataset.py --out exports/my_course.json

輸出：exports/lecture_dataset.json（格式說明見 exports/README.md）
"""
import argparse
import json
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
SKIP = {"_old", "logs", "answer_eval"}

# 影片來源（testset.json 裡有 title / source / license 欄位時，以那邊為準）
SOURCES = {
    "test_video": {"title": "Python 入門短片（測試用）", "license": "未記錄"},
    "mit_prob_l01": {"title": "MIT RES.6-012 Introduction to Probability, Lecture 1: Probability models and axioms",
                     "license": "CC BY-NC-SA 4.0 (MIT OpenCourseWare)"},
    "mit_1806_l01": {"title": "MIT 18.06 Linear Algebra (Gilbert Strang), Lecture 1: The geometry of linear equations",
                     "license": "CC BY-NC-SA 4.0 (MIT OpenCourseWare)"},
    "lecture2": {"title": "MIT 6.0001 Introduction to Computer Science and Programming in Python, Lecture 2: Branching and Iteration",
                 "license": "CC BY-NC-SA 4.0 (MIT OpenCourseWare)"},
    "ntu_cs_12": {"title": "臺大開放式課程 計算機概論 第 12 講（程式語言）",
                  "license": "CC BY-NC-SA（臺大開放式課程）"},
    "ntu_zhuangzi_01": {"title": "臺大開放式課程 正是時候讀莊子 第 1 講（逍遙遊）",
                        "license": "CC BY-NC-SA（臺大開放式課程）"},
}


def load(path, default=None):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def r1(x):
    return round(float(x), 1)


def export_video(name, folder, info):
    meta = load(folder / "meta.json", {})
    timings = load(folder / "timings.json", {})
    whisper = load(folder / "whisper.json", [])
    ocr = load(folder / "ocr.json", [])
    chunks = []
    if (folder / "chunks.jsonl").exists():
        for line in open(folder / "chunks.jsonl", encoding="utf-8"):
            if line.strip():
                chunks.append(json.loads(line))
    if not chunks:
        return None

    src = {**SOURCES.get(name, {}), **{k: info[k] for k in ("title", "source", "license") if k in info}}
    return {
        "name": name,
        "title": src.get("title", name),
        "source": src.get("source", ""),
        "license": src.get("license", ""),
        "type": info.get("type", ""),
        "language": meta.get("language") or info.get("lang", ""),
        "duration_sec": r1(timings.get("video", {}).get("duration_sec") or chunks[-1]["end"]),
        "counts": {"chunks": len(chunks), "transcript_segments": len(whisper), "slides": len(ocr)},
        # 語意完整的段落（約 45 秒），出題時一段可以出 1～3 題
        "chunks": [{"id": c["id"], "start": r1(c["start"]), "end": r1(c["end"]),
                    "speech": c.get("speech_text", "").strip(),
                    "slide": c.get("slide_text", "").strip()} for c in chunks],
        # 逐句逐字稿（Whisper），需要精確時間時用
        "transcript": [{"start": r1(s["start"]), "end": r1(s["end"]), "text": s["text"].strip()}
                       for s in whisper],
        # 每一頁投影片／畫面文字（OCR）出現的時間區間
        "slides": [{"start": r1(s["start"]), "end": r1(s["end"]), "text": s["text"].strip()}
                   for s in ocr if s.get("text", "").strip()],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="comma-separated video names")
    ap.add_argument("--out", default="exports/lecture_dataset.json")
    args = ap.parse_args()

    testset = {v["name"]: v for v in load(ROOT / "evaluation" / "testset.json", {}).get("videos", [])}
    wanted = set(args.only.split(",")) if args.only else None

    videos = []
    for folder in sorted(RESULTS.iterdir()) if RESULTS.exists() else []:
        if folder.name in SKIP or not folder.is_dir():
            continue
        name_file = folder / "video_name.txt"
        name = name_file.read_text(encoding="utf-8").strip() if name_file.exists() else folder.name
        if wanted and name not in wanted:
            continue
        v = export_video(name, folder, testset.get(name, {}))
        if v:
            videos.append(v)
            print(f"  + {name:18s} {v['counts']['chunks']:4d} chunks, "
                  f"{v['counts']['transcript_segments']:5d} sentences, {v['counts']['slides']:4d} slides")

    order = list(testset)                       # 依測試集順序排列
    videos.sort(key=lambda v: order.index(v["name"]) if v["name"] in order else len(order))
    data = {
        "schema_version": 1,
        "generated": date.today().isoformat(),
        "description": "Lecture video dataset produced by rag-video: transcript (Whisper), slide text (OCR) "
                       "and ~45-second chunks with timestamps. See exports/README.md.",
        "notes": ["Speech text comes from automatic speech recognition and may contain misheard words.",
                  "Slide text comes from OCR and may contain typos or simplified Chinese characters.",
                  "Times are in seconds from the start of each video."],
        "videos": videos,
    }
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n{len(videos)} videos -> {out.relative_to(ROOT)} ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
