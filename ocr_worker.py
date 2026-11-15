"""在獨立程序執行 PaddleOCR，避免與 PyTorch/CTranslate2 的 CUDA DLL 衝突。"""
import difflib
import json
import re
import sys
from pathlib import Path

# Windows 終端機可能使用 CP950，辨識到其他語系字元時用替代符號輸出，避免中斷 OCR。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

from PIL import Image, ImageChops, ImageEnhance, ImageFilter, ImageStat

OUT = Path("output")
FRAMES = OUT / "frames"
with open("settings.json", encoding="utf-8") as f:
    SETTINGS = json.load(f)
with open(OUT / "meta.json", encoding="utf-8") as f:
    INTERVAL = json.load(f)["interval"]

MIN_CONFIDENCE = SETTINGS["min_confidence"]
MIN_LINE_CHARS = SETTINGS["min_line_chars"]
FOOTER_Y, HEADER_Y = SETTINGS["footer_y"], SETTINGS["header_y"]
BAD_WORDS = SETTINGS["bad_words"]
SKIP_SIMILAR = SETTINGS["skip_similar"]
SIMILAR_THRESHOLD = SETTINGS["similar_threshold"]

# 先載入此 OCR 環境的 CPU PyTorch，避免 Paddle/PyTorch 的 Windows DLL 載入順序衝突。
import torch
import paddle
from paddleocr import PaddleOCR

# GPU 版 PaddlePaddle 可用時使用 CUDA；否則使用 CPU。
USE_GPU = SETTINGS["use_gpu"] and paddle.device.is_compiled_with_cuda()
OCR_DEVICE = "gpu:0" if USE_GPU else "cpu"
print(f"      PaddleOCR device: {'CUDA' if USE_GPU else 'CPU'}", flush=True)
ocr = PaddleOCR(
    # 明確指定輕量模型；PP-OCRv4 mobile 在 CPU 上比預設較大型模型快。
    text_detection_model_name=SETTINGS.get("ocr_detection_model", "PP-OCRv4_mobile_det"),
    text_recognition_model_name=SETTINGS.get("ocr_recognition_model", "en_PP-OCRv4_mobile_rec"),
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    enable_mkldnn=False,
    device=OCR_DEVICE,
)


def center_y(res, i):
    """取得第 i 個文字框的垂直中心座標。"""
    try:
        box = res["rec_boxes"][i]
        return (float(box[1]) + float(box[3])) / 2
    except Exception:
        try:
            poly = res["rec_polys"][i]
            return sum(float(p[1]) for p in poly) / len(poly)
        except Exception:
            return None


def read_text(image_path):
    """預處理畫面、執行 OCR，並回傳清理後的標題與內文。"""
    source = Image.open(image_path).convert("RGB")
    height = source.size[1]
    if SETTINGS["preprocess_image"]:
        source = ImageEnhance.Contrast(source.convert("L")).enhance(1.35)
        source = source.filter(ImageFilter.UnsharpMask(radius=1, percent=140, threshold=3)).convert("RGB")

    import numpy as np
    lines = []
    for res in ocr.predict(np.asarray(source)):
        for i, (text, score) in enumerate(zip(res["rec_texts"], res["rec_scores"])):
            text = text.strip()
            if score < MIN_CONFIDENCE or len(text) < MIN_LINE_CHARS:
                continue
            if SETTINGS["filter_gibberish"]:
                alnum = sum(ch.isalnum() for ch in text)
                if alnum / max(len(text), 1) < 0.55:
                    continue
            if any(word in text.lower() for word in BAD_WORDS):
                continue
            y = center_y(res, i)
            if y is not None and (y / height > FOOTER_Y or y / height < HEADER_Y):
                continue
            try:
                box = res["rec_boxes"][i]
                box_height = float(box[3]) - float(box[1])
            except Exception:
                box_height = 0
            lines.append((text, box_height, y if y is not None else 0))

    if not lines:
        return ""
    typical_height = sorted(line[1] for line in lines)[len(lines) // 2]
    titles = ([line[0] for line in lines
               if line[1] >= typical_height * 1.35 and line[2] < height * 0.55]
              if typical_height > 0 else [])
    bodies = [line[0] for line in lines if line[0] not in titles]
    return f"Title: {' '.join(titles)} Body: {' '.join(bodies)}" if titles else " ".join(line[0] for line in lines)


def thumbnail(path):
    return Image.open(path).convert("L").resize((64, 36))


def difference(a, b):
    return ImageStat.Stat(ImageChops.difference(a, b)).mean[0]


def same_slide(a, b):
    # 明確標題改變時即視為換頁；否則用文字相似度合併逐步出現的文字。
    def title_of(text):
        match = re.search(r"Title:\s*(.*?)(?:\s+Body:|$)", text, re.IGNORECASE)
        return match.group(1).strip().lower() if match else None

    title_a, title_b = title_of(a), title_of(b)
    if title_a and title_b and title_a != title_b:
        return False
    if a in b or b in a:
        return True
    return difflib.SequenceMatcher(None, a, b).ratio() >= 0.85


ocr_result = []
frame_files = sorted(FRAMES.glob("frame_*.jpg"))
last_thumb, last_text = None, ""
ocr_runs = 0
for i, path in enumerate(frame_files):
    timestamp = i * INTERVAL
    thumb = thumbnail(path)
    if SKIP_SIMILAR and last_thumb is not None and difference(thumb, last_thumb) < SIMILAR_THRESHOLD:
        text = last_text
    else:
        text = read_text(path)
        last_thumb, last_text = thumb, text
        ocr_runs += 1
        print(f"   second {timestamp:5d}: {text[:70]}", flush=True)

    if len(text) < 3:
        continue
    if ocr_result and same_slide(ocr_result[-1]["text"], text):
        ocr_result[-1]["end"] = float(timestamp + INTERVAL)
        if len(text) > len(ocr_result[-1]["text"]):
            ocr_result[-1]["text"] = text
    else:
        ocr_result.append({"start": float(timestamp), "end": float(timestamp + INTERVAL), "text": text})

with open(OUT / "ocr.json", "w", encoding="utf-8") as f:
    json.dump(ocr_result, f, ensure_ascii=False, indent=2)
print(f"   OCR was run on {ocr_runs} of {len(frame_files)} screenshots (the rest were unchanged).")
print(f"   Done: {len(ocr_result)} text blocks. Saved to output/ocr.json")
