"""
比較 Whisper 逐字稿與官方字幕（人工字幕），計算語音辨識錯誤率。

    英文：WER（word error rate，字錯誤率）＝ (替換 + 刪除 + 插入) / 官方字幕字數
    中文：CER（character error rate，字元錯誤率），每個中文字算一個單位

Usage:
    python evaluation/eval_transcript.py output/whisper.json evaluation/references/x.srt --lang en

說明：對齊使用 Python difflib（不是嚴格的最小編輯距離），數值可能比標準 WER 略高一點，
但同一支影片前後比較（例如換模型、開批次處理）是公平的。
"""
import argparse
import difflib
import json
import re
import sys
from pathlib import Path

NUM_WORDS = {"0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five",
             "6": "six", "7": "seven", "8": "eight", "9": "nine", "10": "ten"}


def read_srt(path):
    text = Path(path).read_text(encoding="utf-8", errors="replace").replace("\r", "")
    lines = []
    for block in text.strip().split("\n\n"):
        rows = block.split("\n")
        times = [i for i, r in enumerate(rows) if "-->" in r]
        if times:
            lines.extend(rows[times[0] + 1:])
    return " ".join(lines)


def tokens(text, lang):
    text = text.lower()
    text = re.sub(r"\[[^\]]*\]|\([^)]*\)", " ", text)      # [LAUGHTER]、(inaudible) 之類的標記
    text = re.sub(r"<[^>]+>", " ", text)                     # 字幕裡的 HTML 標籤
    text = text.replace("'", "")                              # don't -> dont
    words = re.findall(r"[a-z0-9]+|[一-鿿]", text)
    words = [NUM_WORDS.get(w, w) for w in words]              # 2 -> two，避免寫法不同算成錯
    return words


def error_rate(ref, hyp):
    sm = difflib.SequenceMatcher(None, ref, hyp, autojunk=False)
    sub = dele = ins = 0
    for op, a1, a2, b1, b2 in sm.get_opcodes():
        if op == "replace":
            n, m = a2 - a1, b2 - b1
            sub += min(n, m)
            dele += max(0, n - m)
            ins += max(0, m - n)
        elif op == "delete":
            dele += a2 - a1
        elif op == "insert":
            ins += b2 - b1
    n = max(1, len(ref))
    return {"rate": (sub + dele + ins) / n, "substitutions": sub, "deletions": dele,
            "insertions": ins, "reference_tokens": len(ref), "hypothesis_tokens": len(hyp)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("whisper_json")
    parser.add_argument("reference_srt", nargs="+")
    parser.add_argument("--lang", choices=["en", "zh"], default="en")
    parser.add_argument("--out", help="save result as JSON")
    args = parser.parse_args()

    hyp_text = " ".join(s["text"] for s in json.load(open(args.whisper_json, encoding="utf-8")))
    ref_text = " ".join(read_srt(p) for p in args.reference_srt)
    result = error_rate(tokens(ref_text, args.lang), tokens(hyp_text, args.lang))
    result["metric"] = "WER" if args.lang == "en" else "CER"

    print(f"{result['metric']}: {result['rate']:.1%}  "
          f"(substitutions {result['substitutions']}, deletions {result['deletions']}, "
          f"insertions {result['insertions']}, reference {result['reference_tokens']} tokens)")
    if args.out:
        json.dump(result, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    sys.exit(main())
