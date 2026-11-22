"""
Step 1: video -> speech text (Whisper) + on-screen text (PaddleOCR)

Usage:
    python extract.py video.mp4                  # run everything
    python extract.py video.mp4 --skip-whisper   # new screenshots + OCR, keep the old whisper.json
                                                 # (use this after changing INTERVAL)
    python extract.py video.mp4 --ocr-only       # re-run only OCR on the existing screenshots
    python extract.py video.mp4 --lang zh        # 中文影片（預設看 settings.json 的 language）

Results are saved in the "output" folder:
    audio.wav       : audio extracted from the video
    frames/         : screenshots (one every INTERVAL seconds)
    meta.json       : remembers INTERVAL for the next steps
    whisper.json    : speech converted to text (with timestamps)
    ocr.json        : cleaned on-screen text (with timestamps)
"""
import os
import sys
import json
import shutil
import argparse
import subprocess
import time
from pathlib import Path

# Skip PaddleOCR's connectivity check on every run (faster startup)
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

# ------------------------------------------------------------------
# Settings you can edit
# ------------------------------------------------------------------
DEFAULT_SETTINGS = {
    "interval": 1, "min_confidence": 0.7, "min_line_chars": 4,
    "footer_y": 0.94, "header_y": 0.0,
    "bad_words": ["simplilearn", "rights reserved", "©"],
    "skip_similar": True, "similar_threshold": 3.0,
    # 影片投影片多為英文，使用較輕量的模型縮短 CPU 辨識時間。
    "ocr_detection_model": "PP-OCRv4_mobile_det",
    "ocr_recognition_model": "en_PP-OCRv4_mobile_rec",
    "use_gpu": True, "preprocess_image": True,
    "filter_gibberish": True,
    "target_sec": 45, "max_words": 180, "overlap": 1,
    "min_words": 2, "min_overlap_sec": 3,
    "slide_boundary_window_sec": 15,
    # 影片語言：en = 英文，zh = 中文。決定 Whisper 模型與 OCR 辨識模型
    "language": "en",
    "whisper_model": "auto"   # auto：英文用 small.en，中文用 medium
}
SETTINGS_PATH = Path("settings.json")
if SETTINGS_PATH.exists():
    with open(SETTINGS_PATH, encoding="utf-8") as f:
        SETTINGS = {**DEFAULT_SETTINGS, **json.load(f)}
else:
    SETTINGS = DEFAULT_SETTINGS.copy()
# 設定檔首次不存在時自動產生，之後可直接修改參數而不必改程式。
if not SETTINGS_PATH.exists():
    with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(DEFAULT_SETTINGS, f, ensure_ascii=False, indent=2)
INTERVAL = SETTINGS["interval"]
MIN_CONFIDENCE = SETTINGS["min_confidence"]
MIN_LINE_CHARS = SETTINGS["min_line_chars"]
FOOTER_Y, HEADER_Y = SETTINGS["footer_y"], SETTINGS["header_y"]
BAD_WORDS = SETTINGS["bad_words"]
SKIP_SIMILAR = SETTINGS["skip_similar"]
SIMILAR_THRESHOLD = SETTINGS["similar_threshold"]
# ------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("video", nargs="?", default="video.mp4")
parser.add_argument("--ocr-only", action="store_true",
                    help="reuse existing screenshots, only redo OCR")
parser.add_argument("--skip-whisper", action="store_true",
                    help="take new screenshots and run OCR, but skip audio/Whisper")
parser.add_argument("--lang", choices=["en", "zh"],
                    help="video language (default: settings.json language)")
args = parser.parse_args()
LANG = args.lang or SETTINGS.get("language", "en")
WHISPER_MODEL = SETTINGS.get("whisper_model", "auto")
if WHISPER_MODEL == "auto":
    # 英文專用模型較快；中文需要多語言模型，medium 的中文辨識比 small 準很多
    WHISPER_MODEL = "small.en" if LANG == "en" else "medium"
# 引導 Whisper 輸出繁體中文（不加的話常輸出簡體）
WHISPER_PROMPT = "以下是台灣大學的課程錄影，使用繁體中文。" if LANG == "zh" else None
print(f"Language: {LANG}   Whisper model: {WHISPER_MODEL}")

OUT = Path("output")
FRAMES = OUT / "frames"
AUDIO = OUT / "audio.wav"
META = OUT / "meta.json"

take_frames = not args.ocr_only
run_whisper = not (args.ocr_only or args.skip_whisper)


def run(cmd):
    subprocess.run(cmd, check=True)


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


TIMINGS = OUT / "timings.json"


def record_time(step, seconds, **extra):
    """把每個步驟花的秒數記到 output/timings.json（pipeline.py 會彙整成報表）。"""
    data = json.load(open(TIMINGS, encoding="utf-8")) if TIMINGS.exists() else {}
    data[step] = {"seconds": round(seconds, 1), **extra}
    save_json(data, TIMINGS)


def video_duration(path):
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                              "-of", "csv=p=0", str(path)], capture_output=True, text=True)
        return round(float(out.stdout.strip()), 1)
    except Exception:
        return None


# ------------------------------------------------------------------
# 1. Prepare folders
# ------------------------------------------------------------------
OUT.mkdir(exist_ok=True)

if args.ocr_only:
    if not list(FRAMES.glob("frame_*.jpg")):
        sys.exit("No screenshots found in output/frames. "
                 "Run once without --ocr-only first.")
    if not META.exists():
        sys.exit("output/meta.json not found, so the screenshot interval is unknown. "
                 "Run: python extract.py video.mp4 --skip-whisper")
    saved_meta = json.load(open(META))
    INTERVAL = saved_meta["interval"]  # use the interval of the saved screenshots
    save_json({**saved_meta, "language": LANG}, META)  # OCR 依這裡的語言選模型
    print(f"Using the saved screenshots (one every {INTERVAL} second(s))")
else:
    if not Path(args.video).exists():
        sys.exit(f"Video file not found: {args.video}")
    if FRAMES.exists():
        shutil.rmtree(FRAMES)  # remove old screenshots when re-running
    FRAMES.mkdir()
    if TIMINGS.exists():
        TIMINGS.unlink()      # 新影片：重新記錄時間
    record_time("video", 0, duration_sec=video_duration(args.video), file=str(args.video),
                language=LANG)

# ------------------------------------------------------------------
# 2. Extract audio and screenshots with FFmpeg
# ------------------------------------------------------------------
if take_frames:
    print(f"[1/4] Extracting screenshots (one every {INTERVAL} second(s))...")
    t = time.time()
    # -map 0:V:0：只取真正的影片畫面。有些 MP4 內含封面圖片（attached picture），
    # 沒指定的話 FFmpeg 可能會挑解析度較高的封面，結果每張截圖都是同一張封面
    run(["ffmpeg", "-y", "-loglevel", "error", "-i", args.video, "-map", "0:V:0",
         "-vf", f"fps=1/{INTERVAL}", str(FRAMES / "frame_%04d.jpg")])
    save_json({"interval": INTERVAL, "language": LANG}, META)
    record_time("screenshots", time.time() - t, count=len(list(FRAMES.glob("frame_*.jpg"))))
if run_whisper:
    print("      Extracting audio...")
    t = time.time()
    run(["ffmpeg", "-y", "-loglevel", "error", "-i", args.video, "-map", "0:a:0",
         "-vn", "-ac", "1", "-ar", "16000", str(AUDIO)])
    record_time("audio", time.time() - t)

# ------------------------------------------------------------------
# 3. Whisper: speech -> text
# ------------------------------------------------------------------
if run_whisper:
    print("[2/4] Whisper is listening...")
    from faster_whisper import WhisperModel
    import ctranslate2
    import torch

    # 有可用 CUDA 且設定允許時使用 GPU，否則自動退回 CPU。
    use_gpu = SETTINGS["use_gpu"] and torch.cuda.is_available()
    print(f"      Whisper device: {'CUDA' if use_gpu else 'CPU'}")
    cuda_types = ctranslate2.get_supported_compute_types("cuda") if use_gpu else set()
    gpu_compute_type = ("float16" if "float16" in cuda_types else
                        "int8_float32" if "int8_float32" in cuda_types else "float32")

    # 批次處理：一次把好幾段音訊一起送進模型，GPU 利用率較高（0 = 關閉）
    BATCH = int(SETTINGS.get("whisper_batch_size", 8))
    whisper_info = {"model": WHISPER_MODEL, "batch_size": 0}

    def to_list(segments):
        # faster-whisper yields results lazily, so consume them here to catch GPU runtime errors.
        return [{"start": round(seg.start, 1), "end": round(seg.end, 1),
                 "text": seg.text.strip()} for seg in segments]

    def transcribe(device, compute_type, batch):
        t_load = time.time()
        model = WhisperModel(WHISPER_MODEL, device=device, compute_type=compute_type)
        whisper_info["load_sec"] = round(time.time() - t_load, 1)
        options = dict(language=LANG, initial_prompt=WHISPER_PROMPT)
        if batch > 0:
            try:
                from faster_whisper import BatchedInferencePipeline
                pipeline = BatchedInferencePipeline(model=model)
                # without_timestamps=False：保留句子層級的時間點（切片段需要）
                segments, _ = pipeline.transcribe(str(AUDIO), batch_size=batch,
                                                  without_timestamps=False, **options)
                result = to_list(segments)
                whisper_info.update(batch_size=batch, device=device, compute_type=compute_type)
                return result
            except Exception as e:
                # 舊版 faster-whisper 沒有批次功能，或顯示卡記憶體不足：改回一般模式
                print(f"      Batched Whisper failed ({type(e).__name__}: {str(e)[:100]}); "
                      f"using normal mode")
                if device == "cuda":
                    try:
                        import torch
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
        segments, _ = model.transcribe(str(AUDIO), vad_filter=True, **options)
        result = to_list(segments)
        whisper_info.update(batch_size=0, device=device, compute_type=compute_type)
        return result

    t = time.time()
    try:
        whisper_result = (transcribe("cuda", gpu_compute_type, BATCH) if use_gpu
                          else transcribe("cpu", "int8", BATCH))
    except Exception as e:
        if not use_gpu:
            raise
        # CUDA device is visible but its runtime libraries may be missing; retry safely on CPU.
        print(f"      CUDA inference failed; retrying on CPU: {e}")
        whisper_result = transcribe("cpu", "int8", BATCH)
    record_time("whisper", time.time() - t, sentences=len(whisper_result), **whisper_info)
    print(f"      Whisper: {whisper_info}")

    for item in whisper_result:
        print(f"   {item['start']:7.1f} -> {item['end']:7.1f} | {item['text']}")

    save_json(whisper_result, OUT / "whisper.json")
    print(f"   Done: {len(whisper_result)} sentences. Saved to output/whisper.json")
else:
    print("[2/4] Skipped Whisper (keeping the existing output/whisper.json)")

# ------------------------------------------------------------------
# 4. PaddleOCR: on-screen text -> text
# ------------------------------------------------------------------
print("[3/4] PaddleOCR is reading text from the screenshots...")
# Windows 上 Paddle 與 PyTorch/CTranslate2 的 CUDA DLL 會衝突，OCR 改由獨立程序執行。
ocr_python = Path(sys.base_prefix) / "python.exe"
t = time.time()
run([str(ocr_python if ocr_python.exists() else Path(sys.executable)),
     str(Path(__file__).with_name("ocr_worker.py"))])
# ocr_worker.py 也會寫入 ocr 的細節（OCR 次數等），這裡補上含載入模型的總時間
data = json.load(open(TIMINGS, encoding="utf-8")) if TIMINGS.exists() else {}
data.setdefault("ocr", {})["total_sec_incl_model_load"] = round(time.time() - t, 1)
save_json(data, TIMINGS)

# ------------------------------------------------------------------
# 5. Summary
# ------------------------------------------------------------------
print("[4/4] Finished!")
print("Open these files in the output folder to see the results:")
print("  - output/whisper.json  (speech)")
print("  - output/ocr.json      (on-screen text)")
