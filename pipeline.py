"""
一鍵處理影片：截圖+OCR、Whisper、切片段、建資料庫，並記錄每個步驟的時間與記憶體。
（取代 tools/*.bat 裡的流程，Windows / Linux 都能用，之後架到伺服器也是用這支。）

處理一支影片（結果留在 output/，search.py / answer.py 直接可用）：
    python pipeline.py videos/lecture.mp4 --name lecture --lang en

處理整個測試集，並評估語音辨識錯誤率與檢索準確度：
    python pipeline.py --testset evaluation/testset.json
    python pipeline.py --testset evaluation/testset.json --only mit_prob_l01,mit_1806_l01
  每支影片處理完會移到 results/<名稱>/，最後產生 results/testset_report.md 與 .csv

output/ 裡原本的影片會先移到 results/<它的名稱>/，不會被覆蓋。
"""
import argparse
import csv
import json
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output"
RESULTS = ROOT / "results"
LOGS = RESULTS / "logs"
PY = sys.executable


# ------------------------------------------------------------------
# 記憶體監控：每 2 秒記錄一次整個程序樹的 RAM 與顯示卡記憶體，保留最大值
# ------------------------------------------------------------------
class PeakMonitor:
    def __init__(self, pid):
        self.pid, self.ram, self.vram = pid, 0, 0
        self.gpu_pid = False            # 這個步驟的程序有出現在顯示卡的使用清單裡
        self.vram_start = self._vram_now()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    @staticmethod
    def _vram_now():
        try:
            r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=5)
            return int(r.stdout.strip().splitlines()[0]) * 1024 * 1024
        except Exception:
            return 0

    def _sample(self):
        pids = {self.pid}
        try:
            import psutil
            p = psutil.Process(self.pid)
            children = p.children(recursive=True)
            pids |= {c.pid for c in children}
            rss = p.memory_info().rss + sum(c.memory_info().rss for c in children)
            self.ram = max(self.ram, rss)
        except Exception:
            pass
        self.vram = max(self.vram, self._vram_now())
        try:
            r = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                               capture_output=True, text=True, timeout=5)
            if pids & {int(x) for x in r.stdout.split() if x.strip().isdigit()}:
                self.gpu_pid = True
        except Exception:
            pass

    @property
    def gpu_used(self):
        """有沒有用到 GPU：程序出現在 nvidia-smi 清單，或顯示卡記憶體多用了 300 MB 以上。"""
        return self.gpu_pid or (self.vram - self.vram_start) > 300 * 2**20

    def _run(self):
        while not self._stop.wait(2):
            self._sample()

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self._thread.join()


mon_gpu = {}   # 每個步驟有沒有用到 GPU（寫進 timings.json）


def run_step(name, args, log):
    """執行一個步驟，輸出寫進 log，回傳 (秒數, 是否成功, RAM 峰值, VRAM 峰值)。"""
    print(f"  [{datetime.now():%H:%M:%S}] {name} ...", flush=True)
    log.write(f"\n===== [{datetime.now():%H:%M:%S}] {name}: {' '.join(map(str, args))}\n")
    log.flush()
    t = time.time()
    env = {**__import__("os").environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
    proc = subprocess.Popen(args, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env)
    with PeakMonitor(proc.pid) as mon:
        code = proc.wait()
    sec = time.time() - t
    gpu = (f"GPU: used (peak {mon.vram / 2**20:.0f} MB)" if mon.gpu_used else "GPU: not used") \
        if mon.vram_start or mon.vram else "GPU: none detected"
    print(f"           {'done' if code == 0 else 'FAILED'} in {sec:.0f}s   {gpu}", flush=True)
    log.write(f"\n===== {name}: {gpu}\n")
    mon_gpu[name] = mon.gpu_used
    return sec, code == 0, mon.ram, mon.vram


def archive_output():
    """把 output/ 裡上一支影片移到 results/<名稱>/。"""
    if not OUT.exists() or not any(OUT.iterdir()):
        return
    name_file = OUT / "video_name.txt"
    name = name_file.read_text(encoding="utf-8").strip() if name_file.exists() else "previous"
    dest = RESULTS / name
    if dest.exists():
        dest = RESULTS / f"{name}_{datetime.now():%Y%m%d_%H%M%S}"
    RESULTS.mkdir(exist_ok=True)
    shutil.move(str(OUT), str(dest))
    print(f"  moved previous output/ -> {dest.relative_to(ROOT)}")


def process(video, name, lang, log):
    """處理一支影片，回傳每個步驟的紀錄；失敗時 ok = False。"""
    archive_output()
    steps, ok = {}, True
    for step, args in [
        ("extract", [PY, "extract.py", str(video), "--lang", lang]),
        ("build_chunks", [PY, "build_chunks.py", name]),
        ("build_db", [PY, "build_db.py"]),
    ]:
        sec, ok, ram, vram = run_step(step, args, log)
        steps[step] = {"seconds": round(sec, 1), "peak_ram_mb": round(ram / 2**20),
                       "peak_vram_mb": round(vram / 2**20), "gpu_used": mon_gpu.get(step)}
        if not ok:
            break
    if OUT.exists():
        (OUT / "video_name.txt").write_text(name, encoding="utf-8")
        timings = OUT / "timings.json"
        data = json.load(open(timings, encoding="utf-8")) if timings.exists() else {}
        data["steps"] = steps
        data["ok"] = ok
        json.dump(data, open(timings, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    return ok


def evaluate(item, log):
    """評估語音辨識錯誤率（有官方字幕時）與檢索準確度（有題目時）。"""
    if item.get("reference_srt"):
        run_step("eval_transcript",
                 [PY, "evaluation/eval_transcript.py", "output/whisper.json", *item["reference_srt"],
                  "--lang", item["lang"], "--out", "output/eval_transcript.json"], log)
    if item.get("questions") and (ROOT / item["questions"]).exists():
        run_step("eval_search",
                 [PY, "evaluation/eval_search.py", item["questions"], "output/chroma_db"], log)


def update_library(include_output=False):
    """把所有處理好的影片合併成影片庫 library/（rag.py / answer.py 預設使用）。"""
    args = [PY, "build_library.py"] + (["--include-output"] if include_output else [])
    with open(LOGS / "build_library.txt", "w", encoding="utf-8") as log:
        run_step("build_library", args, log)


def load(path):
    try:
        return json.load(open(path, encoding="utf-8"))
    except Exception:
        return {}


def report_row(item, folder):
    t = load(folder / "timings.json")
    steps, ocr, wh = t.get("steps", {}), t.get("ocr", {}), t.get("whisper", {})
    dur = (t.get("video", {}).get("duration_sec") or 0) or (item.get("duration_min") or 0) * 60
    total = sum(s.get("seconds", 0) for s in steps.values())
    chunks = sum(1 for _ in open(folder / "chunks.jsonl", encoding="utf-8")) \
        if (folder / "chunks.jsonl").exists() else None
    tr = load(folder / "eval_transcript.json")
    ev = load(folder / "eval_summary.json").get("methods", {})

    def mrr(m):
        return ev.get(m, {}).get("ALL", {}).get("mrr")

    def hit3(m):
        return ev.get(m, {}).get("ALL", {}).get("hit3")

    return {
        "name": item["name"], "type": item.get("type", ""), "lang": item["lang"],
        "minutes": round(dur / 60, 1) if dur else None,
        "ok": t.get("ok"),
        "total_min": round(total / 60, 1),
        "speed_ratio": round(total / dur, 2) if dur else None,
        "screenshots_s": t.get("screenshots", {}).get("seconds"),
        "whisper_s": wh.get("seconds"), "whisper_batch": wh.get("batch_size"),
        "ocr_s": ocr.get("total_sec_incl_model_load", ocr.get("seconds")),
        "ocr_calls": ocr.get("ocr_calls"), "screenshots": ocr.get("screenshots"),
        "overlays_removed": ocr.get("overlays_removed"),
        "build_chunks_s": steps.get("build_chunks", {}).get("seconds"),
        "build_db_s": steps.get("build_db", {}).get("seconds"),
        "chunks": chunks,
        "peak_ram_mb": max((s.get("peak_ram_mb", 0) for s in steps.values()), default=0) or None,
        "peak_vram_mb": max((s.get("peak_vram_mb", 0) for s in steps.values()), default=0) or None,
        "gpu_steps": "、".join(k for k, v in steps.items() if v.get("gpu_used")) or
                     ("無" if any("gpu_used" in v for v in steps.values()) else "-"),
        "asr_error": f"{tr['metric']} {tr['rate']:.1%}" if tr else "",
        "vector_mrr": mrr("vector"), "hybrid_mrr": mrr("hybrid (top10, w=1)"),
        "vector_hit3": hit3("vector"), "hybrid_hit3": hit3("hybrid (top10, w=1)"),
        "rerank_mrr": mrr("hybrid + rerank"), "rerank_hit3": hit3("hybrid + rerank"),
    }


def write_report(items):
    rows = [report_row(it, RESULTS / it["name"]) for it in items if (RESULTS / it["name"]).exists()]
    if not rows:
        return
    with open(RESULTS / "testset_report.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    def fmt(v, pct=False):
        if v is None or v == "":
            return "-"
        return f"{v:.0%}" if pct else str(v)

    lines = [f"# 測試集結果（{datetime.now():%Y-%m-%d %H:%M}）", "",
             "## 處理時間", "",
             "| 影片 | 類型 | 長度 (分) | 總處理 (分) | 處理/影片 | 截圖 (秒) | Whisper (秒) | OCR (秒) | OCR 次數/截圖數 | 建資料庫 (秒) | 片段數 | RAM 峰值 (MB) | VRAM 峰值 (MB) | 用到 GPU 的步驟 |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['name']} | {r['type']} | {fmt(r['minutes'])} | {fmt(r['total_min'])} | "
                     f"{fmt(r['speed_ratio'])} | {fmt(r['screenshots_s'])} | {fmt(r['whisper_s'])} | "
                     f"{fmt(r['ocr_s'])} | {fmt(r['ocr_calls'])}/{fmt(r['screenshots'])} | "
                     f"{fmt(r['build_db_s'])} | {fmt(r['chunks'])} | {fmt(r['peak_ram_mb'])} | "
                     f"{fmt(r['peak_vram_mb'])} | {r['gpu_steps']} |")
    lines += ["", "## 準確度", "",
              "| 影片 | 語音辨識錯誤率 | 向量 MRR | 混合 MRR | 混合+重排 MRR | 向量 Hit@3 | 混合 Hit@3 | 混合+重排 Hit@3 |",
              "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['name']} | {r['asr_error'] or '-'} | {fmt(r['vector_mrr'])} | "
                     f"{fmt(r['hybrid_mrr'])} | {fmt(r['rerank_mrr'])} | {fmt(r['vector_hit3'], True)} | "
                     f"{fmt(r['hybrid_hit3'], True)} | {fmt(r['rerank_hit3'], True)} |")
    def avg(key):
        vals = [r[key] for r in rows if r[key] is not None]
        return sum(vals) / len(vals) if vals else None
    if len(rows) > 1:
        lines.append(f"| **平均** | | {fmt(round(avg('vector_mrr') or 0, 3))} | {fmt(round(avg('hybrid_mrr') or 0, 3))} | "
                     f"{fmt(round(avg('rerank_mrr') or 0, 3)) if avg('rerank_mrr') else '-'} | "
                     f"{fmt(avg('vector_hit3'), True)} | {fmt(avg('hybrid_hit3'), True)} | "
                     f"{fmt(avg('rerank_hit3'), True)} |")
    lines += ["", "處理/影片 = 總處理時間 ÷ 影片長度（0.5 代表 1 小時的影片要 30 分鐘）。",
              "語音辨識錯誤率：英文 WER、中文 CER，與官方字幕比較；沒有官方字幕的影片不計算。",
              "詳細：results/<影片>/eval_result.md、timings.json；執行紀錄：results/logs/。"]
    (RESULTS / "testset_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nReport: {(RESULTS / 'testset_report.md').relative_to(ROOT)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("video", nargs="?")
    parser.add_argument("--name")
    parser.add_argument("--lang", choices=["en", "zh"], default="en")
    parser.add_argument("--testset")
    parser.add_argument("--only", help="comma-separated names from the test set")
    parser.add_argument("--report-only", action="store_true", help="just rebuild the report")
    args = parser.parse_args()
    LOGS.mkdir(parents=True, exist_ok=True)

    if args.testset:
        items = load(ROOT / args.testset).get("videos", [])
        if args.only:
            wanted = set(args.only.split(","))
            items = [it for it in items if it["name"] in wanted]
        if not args.report_only:
            start = time.time()
            for n, item in enumerate(items, 1):
                video = ROOT / item["video"]
                print(f"\n[{n}/{len(items)}] {item['name']} ({item.get('type', '')}, "
                      f"{item.get('duration_min')} min)", flush=True)
                if not video.exists():
                    print(f"  video not found: {item['video']} (skipped)")
                    continue
                old = RESULTS / item["name"]
                if old.exists():           # 之前的結果移到 results/_old/，報表才會用這次的新結果
                    (RESULTS / "_old").mkdir(exist_ok=True)
                    shutil.move(str(old), str(RESULTS / "_old" / f"{item['name']}_{datetime.now():%Y%m%d_%H%M%S}"))
                with open(LOGS / f"pipeline_{item['name']}.txt", "w", encoding="utf-8") as log:
                    try:
                        if process(video, item["name"], item["lang"], log):
                            evaluate(item, log)
                    except Exception as e:
                        print(f"  ERROR: {e}")
                        log.write(f"\nERROR: {e}\n")
                archive_output()          # 移到 results/<名稱>/，下一支從乾淨的 output/ 開始
            print(f"\nAll done in {(time.time() - start) / 60:.0f} min")
            update_library()
        write_report(items)
        return

    if not args.video:
        parser.error("give a video file, or --testset")
    name = args.name or Path(args.video).stem
    with open(LOGS / f"pipeline_{name}.txt", "w", encoding="utf-8") as log:
        ok = process(Path(args.video), name, args.lang, log)
    if ok:
        update_library(include_output=True)
    print("\nFinished." if ok else "\nFailed, see the log:", LOGS / f"pipeline_{name}.txt")
    print("Times:", json.dumps(load(OUT / "timings.json").get("steps", {}), ensure_ascii=False))


if __name__ == "__main__":
    main()
