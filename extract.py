"""
Step 1: video -> speech text (Whisper) + on-screen text (PaddleOCR)

Usage:
    python extract.py video.mp4                  # run everything
    python extract.py video.mp4 --skip-whisper   # new screenshots + OCR, keep the old whisper.json
                                                 # (use this after changing INTERVAL)
    python extract.py video.mp4 --ocr-only       # re-run only OCR on the existing screenshots

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
    "slide_boundary_window_sec": 15
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
args = parser.parse_args()

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
    INTERVAL = json.load(open(META))["interval"]  # use the interval of the saved screenshots
    print(f"Using the saved screenshots (one every {INTERVAL} second(s))")
else:
    if not Path(args.video).exists():
        sys.exit(f"Video file not found: {args.video}")
    if FRAMES.exists():
        shutil.rmtree(FRAMES)  # remove old screenshots when re-running
    FRAMES.mkdir()

# ------------------------------------------------------------------
# 2. Extract audio and screenshots with FFmpeg
# ------------------------------------------------------------------
if take_frames:
    print(f"[1/4] Extracting screenshots (one every {INTERVAL} second(s))...")
    run(["ffmpeg", "-y", "-loglevel", "error", "-i", args.video,
         "-vf", f"fps=1/{INTERVAL}", str(FRAMES / "frame_%04d.jpg")])
    save_json({"interval": INTERVAL}, META)
if run_whisper:
    print("      Extracting audio...")
    run(["ffmpeg", "-y", "-loglevel", "error", "-i", args.video,
         "-vn", "-ac", "1", "-ar", "16000", str(AUDIO)])

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

    def transcribe(device, compute_type):
        model = WhisperModel("small.en", device=device, compute_type=compute_type)
        segments, _ = model.transcribe(str(AUDIO), language="en", vad_filter=True)
        # faster-whisper yields results lazily, so consume them here to catch GPU runtime errors.
        return [{"start": round(seg.start, 1), "end": round(seg.end, 1),
                 "text": seg.text.strip()} for seg in segments]

    try:
        whisper_result = transcribe("cuda", gpu_compute_type) if use_gpu else transcribe("cpu", "int8")
    except Exception as e:
        if not use_gpu:
            raise
        # CUDA device is visible but its runtime libraries may be missing; retry safely on CPU.
        print(f"      CUDA inference failed; retrying on CPU: {e}")
        whisper_result = transcribe("cpu", "int8")

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
run([str(ocr_python if ocr_python.exists() else Path(sys.executable)),
     str(Path(__file__).with_name("ocr_worker.py"))])

# ------------------------------------------------------------------
# 5. Summary
# ------------------------------------------------------------------
print("[4/4] Finished!")
print("Open these files in the output folder to see the results:")
print("  - output/whisper.json  (speech)")
print("  - output/ocr.json      (on-screen text)")
