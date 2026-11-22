"""在獨立程序執行 PaddleOCR，避免與 PyTorch/CTranslate2 的 CUDA DLL 衝突。

加速方式（都可在 settings.json 調整）：
  1. 先比對縮圖，只把「畫面有變」的截圖送去 OCR（skip_similar）
  2. 多個 OCR 程序同時處理不同截圖（ocr_workers）
  3. CPU 加速指令 MKL-DNN / oneDNN（ocr_mkldnn），失敗時自動關閉
  4. 偵測前把圖片長邊限制在 ocr_max_side 像素以內，減少運算量

用法：
  python ocr_worker.py                  # 由 extract.py 呼叫，一般不必手動執行
  python ocr_worker.py --benchmark 8    # 用 8 張截圖比較不同設定的速度與結果
"""
import difflib
import json
import os
import re
import sys
import time
from pathlib import Path

# Windows 終端機可能使用 CP950，辨識到其他語系字元時用替代符號輸出，避免中斷 OCR。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

from PIL import Image, ImageChops, ImageEnhance, ImageFilter, ImageStat

OUT = Path("output")
FRAMES = OUT / "frames"

DEFAULTS = {
    "min_confidence": 0.7, "min_line_chars": 4, "footer_y": 0.94, "header_y": 0.0,
    "bad_words": [], "skip_similar": True, "similar_threshold": 3.0,
    "ocr_detection_model": "PP-OCRv4_mobile_det",
    "ocr_recognition_model": "en_PP-OCRv4_mobile_rec",
    "use_gpu": False, "preprocess_image": True, "filter_gibberish": True,
    # 新增的加速設定
    "ocr_workers": 0,        # 同時執行的 OCR 程序數；0 = 依 CPU 核心數自動決定
    "ocr_mkldnn": True,      # 使用 MKL-DNN/oneDNN 加速 CPU 推論；出錯會自動改回 False
    "ocr_max_side": 960,     # 偵測時圖片長邊上限（像素）；0 = 不限制
    "ocr_gpu": False,        # PaddleOCR 是否用 GPU（GTX 1060 上結果不正確，預設關閉）
    # 判斷兩張截圖是不是同一頁投影片
    "slide_match": "tokens",         # tokens = 比共用的字（抗遮擋、抗辨識錯字）；strict = 舊方法（整串相似度 >= 85%）
    "slide_match_threshold": 0.5,    # tokens：共用字數 / 較少那邊的字數 >= 這個值就算同一頁
    # 自動放慢取樣：連續幾張都是同一頁時，跳著做 OCR；文字變了就回頭找出換頁的位置
    "ocr_adaptive": True,
    "adaptive_after": 2,             # 連續幾次判斷為同一頁後開始加大間隔
    "adaptive_max_stride": 4,        # 最多每幾張截圖才做一次 OCR
    # 自動移除浮水印 / 固定 logo / 頻道名稱：同一位置、相似文字出現在這個比例以上的畫面就移除
    "overlay_filter": True,
    "overlay_min_ratio": 0.3,
}


def load_settings():
    s = dict(DEFAULTS)
    with open("settings.json", encoding="utf-8") as f:
        s.update(json.load(f))
    return s


SETTINGS = load_settings()


def video_language():
    """extract.py 把影片語言記在 output/meta.json；沒有的話看 settings.json。"""
    try:
        with open(OUT / "meta.json", encoding="utf-8") as f:
            return json.load(f).get("language") or SETTINGS.get("language", "en")
    except Exception:
        return SETTINGS.get("language", "en")


LANG = video_language()


def recognition_model():
    model = SETTINGS["ocr_recognition_model"]
    # en_ 開頭的是英文專用模型，認不得中文字；中文影片改用中英文通用模型
    if LANG == "zh" and model.startswith("en_"):
        model = "PP-OCRv4_mobile_rec"
    return model


# ------------------------------------------------------------------
# OCR 引擎（每個 OCR 程序各自建立一份）
# ------------------------------------------------------------------
_ocr = None


def make_ocr(mkldnn, threads):
    # 先載入此 OCR 環境的 CPU PyTorch，避免 Paddle/PyTorch 的 Windows DLL 載入順序衝突。
    try:
        import torch  # noqa: F401
    except Exception:
        pass
    import paddle
    from paddleocr import PaddleOCR

    use_gpu = SETTINGS["ocr_gpu"] and paddle.device.is_compiled_with_cuda()
    kwargs = dict(
        # 明確指定輕量模型；PP-OCRv4 mobile 在 CPU 上比預設較大型模型快。
        text_detection_model_name=SETTINGS["ocr_detection_model"],
        text_recognition_model_name=recognition_model(),
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        device="gpu:0" if use_gpu else "cpu",
    )
    if not use_gpu:
        kwargs.update(enable_mkldnn=mkldnn, cpu_threads=threads)
    if SETTINGS["ocr_max_side"]:
        kwargs.update(text_det_limit_side_len=SETTINGS["ocr_max_side"],
                      text_det_limit_type="max")
    return PaddleOCR(**kwargs), ("CUDA" if use_gpu else "CPU")


def init_ocr(mkldnn, threads, verbose=False):
    """建立 OCR；開 MKL-DNN 時先試跑一張小圖，出錯就關掉 MKL-DNN 重建。"""
    global _ocr
    import numpy as np
    if mkldnn:
        try:
            _ocr, device = make_ocr(True, threads)
            test = Image.new("RGB", (320, 80), "white")
            list(_ocr.predict(np.asarray(test)))
            if verbose:
                print(f"      PaddleOCR device: {device} (MKL-DNN on, {threads} threads), "
                      f"language {LANG}, model {recognition_model()}", flush=True)
            return True
        except Exception as e:
            if verbose:
                print(f"      MKL-DNN failed ({type(e).__name__}: {str(e)[:80]}); "
                      f"falling back to MKL-DNN off", flush=True)
    _ocr, device = make_ocr(False, threads)
    if verbose:
        print(f"      PaddleOCR device: {device} (MKL-DNN off, {threads} threads)", flush=True)
    return False


def _pool_init(mkldnn, threads, max_side):
    SETTINGS["ocr_max_side"] = max_side
    init_ocr(mkldnn, threads, verbose=False)


# ------------------------------------------------------------------
# 單張截圖 → 文字（與原本相同的清理規則）
# ------------------------------------------------------------------
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


RAW = {}   # 截圖路徑 -> (每行文字資料, 圖高, 圖寬)，給浮水印過濾用


def ocr_lines(image_path):
    """預處理畫面、執行 OCR，回傳清理後的每一行：(文字, 字高, 中心 y, 中心 x)。"""
    import numpy as np
    source = Image.open(image_path).convert("RGB")
    width, height = source.size
    if SETTINGS["preprocess_image"]:
        source = ImageEnhance.Contrast(source.convert("L")).enhance(1.35)
        source = source.filter(ImageFilter.UnsharpMask(radius=1, percent=140, threshold=3)).convert("RGB")

    lines = []
    for res in _ocr.predict(np.asarray(source)):
        for i, (text, score) in enumerate(zip(res["rec_texts"], res["rec_scores"])):
            text = text.strip()
            if score < SETTINGS["min_confidence"] or len(text) < SETTINGS["min_line_chars"]:
                continue
            if SETTINGS["filter_gibberish"]:
                alnum = sum(ch.isalnum() for ch in text)
                if alnum / max(len(text), 1) < 0.55:
                    continue
            if any(word in text.lower() for word in SETTINGS["bad_words"]):
                continue
            y = center_y(res, i)
            if y is not None and (y / height > SETTINGS["footer_y"] or y / height < SETTINGS["header_y"]):
                continue
            try:
                box = res["rec_boxes"][i]
                box_height = float(box[3]) - float(box[1])
                x = (float(box[0]) + float(box[2])) / 2
            except Exception:
                box_height, x = 0, width / 2
            lines.append((text, box_height, y if y is not None else 0, x))
    return lines, height, width


def format_lines(lines, height):
    """把每一行組成 "Title: ... Body: ..."（字特別大、位置偏上的當標題）。"""
    if not lines:
        return ""
    typical_height = sorted(line[1] for line in lines)[len(lines) // 2]
    titles = ([line[0] for line in lines
               if line[1] >= typical_height * 1.35 and line[2] < height * 0.55]
              if typical_height > 0 else [])
    bodies = [line[0] for line in lines if line[0] not in titles]
    return f"Title: {' '.join(titles)} Body: {' '.join(bodies)}" if titles else " ".join(line[0] for line in lines)


def read_text(image_path):
    lines, height, width = ocr_lines(image_path)
    RAW[str(image_path)] = (lines, height, width)
    return format_lines(lines, height)


# ------------------------------------------------------------------
# 浮水印 / 固定 logo 過濾
# ------------------------------------------------------------------
def _norm(text):
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]", "", text.lower())


def find_overlays(raw, order):
    """找出疊加在影片上的固定文字（浮水印、頻道名稱、固定頁尾），三個條件都要符合：

    1. 位置在畫面四個角落（10x10 格中各角落 2x2 格）：浮水印、logo 幾乎都在角落。
       只看角落是因為螢幕錄影裡的程式碼、選單列、標題列也會整支影片都在邊緣，那些不能刪
    2. 同一格、相似文字出現在 overlay_min_ratio 以上有字的截圖
    3. 分散在整支影片：影片切成 5 段，至少 4 段都有（投影片只會出現一段時間）
    回傳 {格子: 代表文字}。
    """
    frames = [(order[p], v) for p, v in raw.items() if v[0] and p in order]
    n = len(frames)
    if n < 20:                                # 有字的截圖太少，統計不可靠
        return {}
    total = max(order.values()) + 1
    cells = {}
    for idx, (lines, h, w) in frames:
        seen = set()
        for text, _, y, x in lines:
            key = (min(9, int(10 * x / max(w, 1))), min(9, int(10 * y / max(h, 1))))
            if not (key[0] in (0, 1, 8, 9) and key[1] in (0, 1, 8, 9)):
                continue                      # 不在四個角落（程式碼、標題列常常整支影片都在邊緣，不能刪）
            t = _norm(text)
            if t and (key, t) not in seen:
                seen.add((key, t))
                cells.setdefault(key, []).append((t, idx))
    overlays = {}
    for key, items in cells.items():
        if len(items) < SETTINGS["overlay_min_ratio"] * n:
            continue
        texts = [t for t, _ in items]
        # 以最常見的文字當代表，再數有多少張的文字跟它相似（OCR 常把浮水印讀成不同錯字）
        canonical = max(set(texts), key=texts.count)
        similar = [idx for t, idx in items
                   if difflib.SequenceMatcher(None, t, canonical).ratio() >= 0.6]
        spread = len({min(4, 5 * idx // total) for idx in similar})
        if len(similar) >= SETTINGS["overlay_min_ratio"] * n and spread >= 4:
            overlays[key] = canonical
    return overlays


def is_overlay(line, h, w, overlays):
    text, _, y, x = line
    key = (min(9, int(10 * x / max(w, 1))), min(9, int(10 * y / max(h, 1))))
    if key not in overlays:
        return False
    return difflib.SequenceMatcher(None, _norm(text), overlays[key]).ratio() >= 0.6


def cleaned_text(path, overlays):
    lines, h, w = RAW[str(path)]
    return format_lines([l for l in lines if not is_overlay(l, h, w, overlays)], h)


def _read_one(path):
    return read_text(path)


# ------------------------------------------------------------------
# 步驟 1：比對縮圖，決定哪些截圖需要 OCR（很快，不用 OCR 模型）
# ------------------------------------------------------------------
def thumbnail(path):
    return Image.open(path).convert("L").resize((64, 36))


def difference(a, b):
    return ImageStat.Stat(ImageChops.difference(a, b)).mean[0]


def choose_frames(frame_files):
    """回傳 source[i] = 第 i 張截圖要沿用哪一張截圖的 OCR 結果。"""
    source, last_thumb, last_idx = [], None, None
    for i, path in enumerate(frame_files):
        thumb = thumbnail(path)
        if (SETTINGS["skip_similar"] and last_thumb is not None
                and difference(thumb, last_thumb) < SETTINGS["similar_threshold"]):
            source.append(last_idx)
        else:
            source.append(i)
            last_thumb, last_idx = thumb, i
    return source


# ------------------------------------------------------------------
# 步驟 2：OCR（可多個程序同時跑）
# ------------------------------------------------------------------
def worker_plan():
    cores = os.cpu_count() or 4
    workers = SETTINGS["ocr_workers"] or max(1, min(4, cores // 4))
    if SETTINGS["ocr_gpu"]:
        workers = 1  # 只有一張顯示卡，多程序沒有好處
    threads = max(1, cores // workers)
    return workers, threads


def run_ocr(paths, workers, threads, mkldnn, show=None):
    """依序回傳每張圖的文字。workers > 1 時用多個程序平行處理。"""
    if workers <= 1:
        init_ocr(mkldnn, threads, verbose=True)
        for p in paths:
            text = read_text(p)
            if show:
                show(p, text)
            yield text
        return

    # 先在主程序確認 MKL-DNN 能不能用，避免每個子程序都失敗一次
    ok = init_ocr(mkldnn, threads, verbose=True)
    global _ocr
    _ocr = None  # 主程序不需要保留模型，節省記憶體
    print(f"      Running {workers} OCR processes in parallel", flush=True)
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(workers, initializer=_pool_init,
                                      initargs=(ok, threads, SETTINGS["ocr_max_side"])) as pool:
        for p, text in zip(paths, pool.imap(_read_one, paths, chunksize=1)):
            if show:
                show(p, text)
            yield text


def run_adaptive(frame_files, show=None):
    """自動放慢取樣（一個 OCR 程序）。回傳 (每張截圖的文字, OCR 次數, 回頭搜尋次數)。

    1. 畫面像素幾乎沒變 → 沿用上一張的文字（Pillow 比對，跟原本一樣）
    2. 連續 adaptive_after 次 OCR 都是同一頁 → 間隔加倍（最多 adaptive_max_stride 張）
    3. 文字變了 → 間隔恢復成 1，並在跳過的範圍內用二分搜尋找出真正換頁的那張
    跳過的截圖沿用它前面最近一張有做 OCR 的文字。
    """
    n = len(frame_files)
    results = {}
    calls = {"ocr": 0, "search": 0}

    def ocr(k, kind="ocr"):
        calls[kind] += 1
        text = read_text(frame_files[k])
        results[k] = text
        if show:
            show(frame_files[k], text)
        return text

    after, max_stride = SETTINGS["adaptive_after"], SETTINGS["adaptive_max_stride"]
    i, stride, run = 0, 1, 0
    prev_i, prev_t, last_thumb = None, None, None
    while i < n:
        thumb = thumbnail(frame_files[i])
        if (SETTINGS["skip_similar"] and last_thumb is not None
                and difference(thumb, last_thumb) < SETTINGS["similar_threshold"]):
            i += 1                    # 畫面沒變：不必 OCR，之後填入前一張的文字
            continue
        last_thumb = thumb
        text = ocr(i)
        if prev_t is not None and same_slide(prev_t, text):
            run += 1
            if run >= after:
                stride = min(stride * 2, max_stride)
        else:
            if prev_i is not None and i - prev_i > 1:
                lo, hi = prev_i, i    # 換頁發生在 lo 之後、hi 之前（含 hi）
                lo_thumb = thumbnail(frame_files[lo])
                while hi - lo > 1:
                    mid = (lo + hi) // 2
                    mid_thumb = thumbnail(frame_files[mid])
                    # 畫面跟 lo 幾乎一樣 → 一定還是同一頁，不必 OCR
                    if (SETTINGS["skip_similar"] and
                            difference(mid_thumb, lo_thumb) < SETTINGS["similar_threshold"]):
                        lo, lo_thumb = mid, mid_thumb
                    elif same_slide(prev_t, ocr(mid, "search")):
                        lo, lo_thumb = mid, mid_thumb
                    else:
                        hi = mid
            run, stride = 0, 1
        prev_i, prev_t = i, text
        i += stride

    texts, sources, current, src = [], [], "", 0
    for k in range(n):
        if k in results:
            current, src = results[k], k
        texts.append(current)
        sources.append(src)
    return texts, sources, calls["ocr"], calls["search"]


# ------------------------------------------------------------------
# 步驟 3：合併成投影片區塊（與原本相同）
# ------------------------------------------------------------------
TOKEN_RE = re.compile(r"[a-z0-9]{2,}|[\u4e00-\u9fff]")


def slide_tokens(text):
    text = re.sub(r"\b(title|body):", " ", text.lower())
    return set(TOKEN_RE.findall(text))


def title_of(text):
    match = re.search(r"Title:\s*(.*?)(?:\s+Body:|$)", text, re.IGNORECASE)
    return match.group(1).strip().lower() if match else None


def same_slide(a, b):
    """兩張截圖的文字是不是同一頁投影片。"""
    if SETTINGS["slide_match"] == "strict":
        # 舊方法：標題不同就換頁；否則用整串文字相似度
        title_a, title_b = title_of(a), title_of(b)
        if title_a and title_b and title_a != title_b:
            return False
        if a in b or b in a:
            return True
        return difflib.SequenceMatcher(None, a, b).ratio() >= 0.85

    # tokens：老師擋住一部分、浮水印被讀錯時，整串文字會差很多，
    # 但大部分的字還是一樣，所以改看「共用的字」佔多少
    A, B = slide_tokens(a), slide_tokens(b)
    if len(A) < 3 and len(B) < 3:      # 兩張都幾乎沒有字（例如只拍到老師）
        return True
    if len(A) < 3 or len(B) < 3:       # 一張有投影片、一張沒有
        return False
    ta, tb = title_of(a), title_of(b)
    if ta and tb:
        TA, TB = slide_tokens(ta), slide_tokens(tb)
        if len(TA) >= 2 and len(TB) >= 2 and not (TA & TB):
            return False               # 兩邊都有明確標題而且完全不同
    return len(A & B) / min(len(A), len(B)) >= SETTINGS["slide_match_threshold"]


def build_blocks(texts, interval):
    ocr_result = []
    for i, text in enumerate(texts):
        timestamp = i * interval
        if len(text) < 3:
            continue
        if ocr_result and same_slide(ocr_result[-1]["text"], text):
            ocr_result[-1]["end"] = float(timestamp + interval)
            if len(text) > len(ocr_result[-1]["text"]):
                ocr_result[-1]["text"] = text
        else:
            ocr_result.append({"start": float(timestamp), "end": float(timestamp + interval), "text": text})
    return ocr_result


def record_ocr_timing(**info):
    path = OUT / "timings.json"
    try:
        data = json.load(open(path, encoding="utf-8")) if path.exists() else {}
    except Exception:
        data = {}
    data["ocr"] = {**data.get("ocr", {}), **info}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    with open(OUT / "meta.json", encoding="utf-8") as f:
        interval = json.load(f)["interval"]

    frame_files = sorted(FRAMES.glob("frame_*.jpg"))
    # 截圖比 settings.json 的 interval 還密時，只取每 step 張做 OCR（不必重新截圖）
    step = max(1, round(SETTINGS.get("interval", interval) / interval))
    if step > 1:
        frame_files = frame_files[::step]
        interval = interval * step
        print(f"      Using every {step}th screenshot (one every {interval} second(s))", flush=True)
    workers, threads = worker_plan()

    def show(p, text):
        sec = frame_files.index(p) * interval if p in frame_files else 0
        print(f"   second {sec:5d}: {text[:70]}", flush=True)

    if SETTINGS["ocr_adaptive"] and workers == 1:
        init_ocr(SETTINGS["ocr_mkldnn"], threads, verbose=True)
        print(f"      Adaptive sampling on (after {SETTINGS['adaptive_after']} same slides, "
              f"up to every {SETTINGS['adaptive_max_stride']} screenshots; "
              f"slide match: {SETTINGS['slide_match']})", flush=True)
        t1 = time.time()
        texts, source, n_ocr, n_search = run_adaptive(frame_files, show)
        ocr_time = time.time() - t1
        n_calls = n_ocr + n_search
        print(f"   OCR was run {n_calls} times for {len(frame_files)} screenshots "
              f"({n_ocr} regular + {n_search} to locate slide changes).")
    else:
        t0 = time.time()
        source = choose_frames(frame_files)
        needed = sorted(set(source))
        print(f"      {len(needed)} of {len(frame_files)} screenshots changed and need OCR "
              f"(compared in {time.time() - t0:.1f}s)", flush=True)
        workers = min(workers, max(1, len(needed)))
        t1 = time.time()
        paths = [frame_files[i] for i in needed]
        texts_needed = dict(zip(needed, run_ocr(paths, workers, threads, SETTINGS["ocr_mkldnn"], show)))
        ocr_time = time.time() - t1
        texts = [texts_needed[source[i]] for i in range(len(frame_files))]
        n_calls = len(needed)
        print(f"   OCR was run on {len(needed)} of {len(frame_files)} screenshots (the rest were unchanged).")

    # 浮水印過濾（需要每一行的位置；多程序模式拿不到，就跳過）
    overlays = {}
    if SETTINGS["overlay_filter"] and RAW:
        overlays = find_overlays(RAW, {str(p): i for i, p in enumerate(frame_files)})
        if overlays:
            print("      Removed overlay text (watermark / logo) found in many frames:")
            for key, text in sorted(overlays.items()):
                print(f"         position {key}: {text[:40]}")
            cache = {}
            for k in range(len(frame_files)):
                p = str(frame_files[source[k]])
                if p in RAW:
                    if p not in cache:
                        cache[p] = cleaned_text(p, overlays)
                    texts[k] = cache[p]

    ocr_result = build_blocks(texts, interval)

    with open(OUT / "ocr.json", "w", encoding="utf-8") as f:
        json.dump(ocr_result, f, ensure_ascii=False, indent=2)
    record_ocr_timing(seconds=round(ocr_time, 1), screenshots=len(frame_files), ocr_calls=n_calls,
                      adaptive=bool(SETTINGS["ocr_adaptive"] and workers == 1),
                      slide_match=SETTINGS["slide_match"], text_blocks=len(ocr_result),
                      overlays_removed=len(overlays), workers=workers, threads=threads)
    per = ocr_time / max(1, n_calls)
    print(f"   OCR time: {ocr_time:.1f}s total, {per:.2f}s per OCR call "
          f"({workers} process(es) x {threads} thread(s))")
    print(f"   Done: {len(ocr_result)} text blocks. Saved to output/ocr.json")


# ------------------------------------------------------------------
# --benchmark：比較不同設定的速度，並檢查結果是否和最慢的原始設定一樣
# ------------------------------------------------------------------
def benchmark(n):
    frame_files = sorted(FRAMES.glob("frame_*.jpg"))
    source = choose_frames(frame_files)
    paths = [frame_files[i] for i in sorted(set(source))][:n]
    if not paths:
        sys.exit("No screenshots in output/frames. Run extract.py first.")
    cores = os.cpu_count() or 4
    print(f"Benchmark on {len(paths)} screenshots, {cores} logical CPU cores\n")

    configs = [
        ("original (1 proc, MKL-DNN off, no resize)", 1, cores, False, 0),
        ("+ resize to 960", 1, cores, False, 960),
        ("+ MKL-DNN", 1, cores, True, 960),
    ]
    w, t = worker_plan()
    if w > 1:
        configs.append((f"+ {w} processes", w, t, True, 960))

    baseline = None
    rows = []
    for name, workers, threads, mkldnn, max_side in configs:
        SETTINGS["ocr_max_side"] = max_side
        start = time.time()
        texts = list(run_ocr(paths, workers, threads, mkldnn))
        sec = time.time() - start
        if baseline is None:
            baseline = texts
        sim = sum(difflib.SequenceMatcher(None, a, b).ratio()
                  for a, b in zip(baseline, texts)) / len(texts)
        rows.append((name, sec, sim))
        print(f"  {name:45s} {sec:7.1f}s   text same as original: {sim:6.1%}\n", flush=True)

    print("\nSummary (time includes loading the model):")
    for name, sec, sim in rows:
        print(f"  {name:45s} {sec:7.1f}s  x{rows[0][1] / sec:4.1f}  same text {sim:6.1%}")


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--benchmark":
        benchmark(int(sys.argv[2]))
    elif len(sys.argv) == 2 and sys.argv[1] == "--benchmark":
        benchmark(8)
    else:
        main()
