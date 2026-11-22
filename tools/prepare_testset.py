"""
一次性：把下載資料夾裡的測試影片整理進專案，並建立測試集清單。

    python tools/prepare_testset.py

做的事（只搬移、合併，不刪除任何檔案）：
  1. MIT 機率課第 1 講的 10 個片段 → 用 FFmpeg 接成一支 videos/mit_prob_l01.mp4（不重新編碼）
     10 個官方字幕 → 依各片段長度調整時間後接成 evaluation/references/mit_prob_l01.srt
  2. MIT 18.06 第 1 講 → videos/mit_1806_l01.mp4，字幕 → evaluation/references/mit_1806_l01.srt
  3. 台大莊子第 1 講 → videos/ntu_zhuangzi_01.mp4
  4. 依片段長度把題目的時間點換成整支影片的秒數 → evaluation/questions/mit_prob_l01.json
  5. 寫出 evaluation/testset.json（pipeline.py --testset 會用）
用過的片段檔搬到 _to_delete/downloads_used/，確認沒問題後可以刪掉。
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOWNLOADS = Path.home() / "Downloads"
VIDEOS = ROOT / "videos"
REFS = ROOT / "evaluation" / "references"
QUESTIONS = ROOT / "evaluation" / "questions"
USED = ROOT / "_to_delete" / "downloads_used"
for d in (VIDEOS, REFS, QUESTIONS, USED):
    d.mkdir(parents=True, exist_ok=True)


def duration(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "csv=p=0", str(path)], capture_output=True, text=True, check=True)
    return float(out.stdout.strip())


def srt_time(sec):
    ms = int(round(sec * 1000))
    h, ms = divmod(ms, 3600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def parse_time(t):
    h, m, s = t.strip().replace(",", ".").split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def shifted_srt(path, offset, start_index):
    out, idx = [], start_index
    text = path.read_text(encoding="utf-8", errors="replace").replace("\r", "")
    for block in text.strip().split("\n\n"):
        rows = block.split("\n")
        t = [i for i, r in enumerate(rows) if "-->" in r]
        if not t:
            continue
        a, b = rows[t[0]].split("-->")
        out.append(f"{idx}\n{srt_time(parse_time(a) + offset)} --> {srt_time(parse_time(b) + offset)}\n"
                   + "\n".join(rows[t[0] + 1:]))
        idx += 1
    return out, idx


def move(src, dst):
    if src.exists() and not dst.exists():
        shutil.move(str(src), str(dst))
        print(f"  moved {src.name} -> {dst.relative_to(ROOT)}")
    elif dst.exists():
        print(f"  already there: {dst.relative_to(ROOT)}")
    else:
        print(f"  NOT FOUND: {src}")


# ------------------------------------------------------------------
# 1. MIT 機率課：10 段接成一支
# ------------------------------------------------------------------
print("[1] MIT probability lecture 1")
segs = [DOWNLOADS / f"prob_MITRES6_012S18_L01-{i:02d}_300k.mp4" for i in range(1, 11)]
subs = [DOWNLOADS / f"prob_MITRES6_012S18_L01-{i:02d}_300k.srt" for i in range(1, 11)]
target = VIDEOS / "mit_prob_l01.mp4"
offsets = []
if all(p.exists() for p in segs):
    t = 0.0
    for p in segs:
        offsets.append(t)
        t += duration(p)
    if not target.exists():
        lst = ROOT / "_to_delete" / "concat_list.txt"
        lst.write_text("".join(f"file '{p.as_posix()}'\n" for p in segs), encoding="utf-8")
        # -map 0:V:0 / 0:a:0：只取影片和聲音，排除 MP4 內的封面圖片
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
                        "-i", str(lst), "-map", "0:V:0", "-map", "0:a:0", "-c", "copy",
                        str(target)], check=True)
        print(f"  joined 10 segments -> videos/mit_prob_l01.mp4 ({t / 60:.1f} min)")
    blocks, idx = [], 1
    for p, off in zip(subs, offsets):
        b, idx = shifted_srt(p, off, idx)
        blocks += b
    (REFS / "mit_prob_l01.srt").write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
    print("  joined 10 subtitle files -> evaluation/references/mit_prob_l01.srt")
    json.dump({"segments": [p.name for p in segs], "offsets_sec": offsets},
              open(REFS / "mit_prob_l01_offsets.json", "w"), indent=1)
    for p in segs + subs:
        move(p, USED / p.name)
elif target.exists() and (REFS / "mit_prob_l01_offsets.json").exists():
    offsets = json.load(open(REFS / "mit_prob_l01_offsets.json"))["offsets_sec"]
    print("  already prepared")
else:
    print("  segments not found in Downloads, skipped")

# 題目：(片段編號, 片段內秒數) → 整支影片的秒數
PROB_Q = [
    ("mutually exclusive and collectively exhaustive", [(2, 85)], "keyword"),
    ("tetrahedral die with 16 possible outcomes", [(7, 5)], "keyword"),
    ("what is the union bound", [(6, 365)], "keyword"),
    ("countable additivity axiom", [(9, 375)], "keyword"),
    ("nonnegativity axiom", [(4, 245)], "keyword"),
    ("discrete uniform probability law", [(7, 215)], "keyword"),
    ("why is the unit square an uncountable set", [(9, 605)], "keyword"),
    ("probability of the empty set", [(5, 45)], "keyword"),
    ("does the weather outside matter when listing outcomes of a coin flip", [(2, 205)], "paraphrase"),
    ("why can't each point of a continuous target get its own positive probability", [(4, 85)], "paraphrase"),
    ("how do we draw an experiment that happens in stages", [(3, 165)], "paraphrase"),
    ("if one event is contained in another which one is more likely", [(6, 45)], "paraphrase"),
    ("how to get a probability by counting when all outcomes are equally likely", [(7, 285)], "paraphrase"),
    ("what are the general steps for solving a probability problem", [(8, 205)], "paraphrase"),
    ("how do we check that the given probabilities of the integers really add to one", [(9, 85)], "paraphrase"),
    ("what is the chance that the dart hits one exact point", [(8, 165)], "paraphrase"),
    ("樣本空間要怎麼選才適當", [(2, 285)], "chinese"),
    ("機率的公理有哪些", [(4, 245)], "chinese"),
    ("兩個事件聯集的機率怎麼算", [(6, 165)], "chinese"),
    ("連續樣本空間的例子", [(3, 245)], "chinese"),
]
if offsets:
    json.dump({"video": "mit_prob_l01",
               "note": "MIT RES.6-012 Introduction to Probability, Lecture 1（10 段接成一支）。"
                       "anchors = 答案出現的時間點（秒），依官方字幕標定。",
               "questions": [{"q": q, "type": t,
                              "anchors": [round(offsets[s - 1] + x, 1) for s, x in a]}
                             for q, a, t in PROB_Q]},
              open(QUESTIONS / "mit_prob_l01.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("  wrote evaluation/questions/mit_prob_l01.json")

# ------------------------------------------------------------------
# 2. MIT 18.06、3. 台大莊子
# ------------------------------------------------------------------
print("[2] MIT 18.06 lecture 1")
move(DOWNLOADS / "strang_1806_lec01.mp4", VIDEOS / "mit_1806_l01.mp4")
if (DOWNLOADS / "strang_1806_lec01.srt").exists() and not (REFS / "mit_1806_l01.srt").exists():
    shutil.copy(DOWNLOADS / "strang_1806_lec01.srt", REFS / "mit_1806_l01.srt")
    move(DOWNLOADS / "strang_1806_lec01.srt", USED / "strang_1806_lec01.srt")
print("[3] NTU Zhuangzi lecture 1")
move(DOWNLOADS / "1_逍遙遊 (一) ─ 北冥有魚.mp4", VIDEOS / "ntu_zhuangzi_01.mp4")

# ------------------------------------------------------------------
# 4. 測試集清單
# ------------------------------------------------------------------
TESTSET = [
    {"name": "test_video", "video": "videos/test_video.mp4", "lang": "en",
     "type": "英文・短片", "questions": "evaluation/questions/test_video.json"},
    {"name": "mit_prob_l01", "video": "videos/mit_prob_l01.mp4", "lang": "en",
     "type": "英文・投影片螢幕錄影", "questions": "evaluation/questions/mit_prob_l01.json",
     "reference_srt": ["evaluation/references/mit_prob_l01.srt"]},
    {"name": "mit_1806_l01", "video": "videos/mit_1806_l01.mp4", "lang": "en",
     "type": "英文・黑板課", "questions": "evaluation/questions/mit_1806_l01.json",
     "reference_srt": ["evaluation/references/mit_1806_l01.srt"]},
    {"name": "lecture2", "video": "videos/lecture2.mp4", "lang": "en",
     "type": "英文・教室錄影", "questions": "evaluation/questions/lecture2.json"},
    {"name": "ntu_cs_12", "video": "videos/ntu_cs_12_full.mp4", "lang": "zh",
     "type": "中文・教室錄影（英文投影片）・長片", "questions": "evaluation/questions/ntu_cs_12.json"},
    {"name": "ntu_zhuangzi_01", "video": "videos/ntu_zhuangzi_01.mp4", "lang": "zh",
     "type": "中文・人文課程", "questions": "evaluation/questions/ntu_zhuangzi_01.json"},
]
for item in TESTSET:
    p = ROOT / item["video"]
    item["duration_min"] = round(duration(p) / 60, 1) if p.exists() else None
json.dump({"videos": TESTSET}, open(ROOT / "evaluation" / "testset.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print("[4] wrote evaluation/testset.json")
for item in TESTSET:
    ok = "OK " if item["duration_min"] else "MISSING"
    q = "questions OK" if (ROOT / item["questions"]).exists() else "no questions yet"
    print(f"  {ok} {item['name']:16s} {str(item['duration_min']):>6} min  {item['type']}  ({q})")
print("PREPARE_FINISHED")
