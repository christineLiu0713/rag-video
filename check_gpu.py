"""
檢查這台電腦能不能用 GPU（NVIDIA 顯示卡）跑本專案，並實際在 GPU 上算一次確認。

Usage（在專案資料夾，用 venv 的 Python）:
    python check_gpu.py
    或雙擊 tools/check_gpu.bat

檢查項目：
    1. 顯示卡與驅動（nvidia-smi）
    2. PyTorch 是否為 CUDA 版、能不能用 GPU        -> BGE-M3 向量、重新排序用
    3. 實際在 GPU 上做一次矩陣乘法，和 CPU 比速度
    4. faster-whisper（CTranslate2）能不能用 GPU    -> 語音辨識用
    5. PaddleOCR（系統 Python）的狀態              -> OCR 刻意用 CPU + MKL-DNN

結果會印在畫面上，並存成 results/logs/gpu_check.txt，可以直接傳給開發者。
"""
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
lines = []
summary = []


def say(text=""):
    print(text, flush=True)
    lines.append(text)


def result(name, ok, detail):
    mark = "OK " if ok else ("-- " if ok is None else "NO ")
    summary.append(f"[{mark}] {name}: {detail}")
    say(f"  -> [{mark}] {detail}")


def run(cmd, timeout=30):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                           encoding="utf-8", errors="replace")
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    except Exception as e:
        return -1, str(e)


say(f"GPU check  {datetime.now():%Y-%m-%d %H:%M}")
say(f"Python {sys.version.split()[0]} ({sys.executable})")
say(f"OS     {platform.platform()}")

# ------------------------------------------------------------------
say("\n1. 顯示卡與驅動 (nvidia-smi)")
if shutil.which("nvidia-smi"):
    code, out = run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used",
                     "--format=csv,noheader"])
    if code == 0 and out.strip():
        for gpu in out.strip().splitlines():
            say(f"  {gpu}")
        result("顯示卡", True, out.strip().splitlines()[0])
    else:
        result("顯示卡", False, f"nvidia-smi 執行失敗：{out.strip()[:200]}")
else:
    result("顯示卡", False, "找不到 nvidia-smi：沒有 NVIDIA 顯示卡，或沒有安裝驅動")

# ------------------------------------------------------------------
say("\n2. PyTorch (BGE-M3 向量、重新排序)")
torch = None
try:
    import torch
    say(f"  torch {torch.__version__}  CUDA build: {torch.version.cuda}")
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        total = torch.cuda.get_device_properties(0).total_memory / 2**30
        say(f"  device 0: {name}  compute capability {cap[0]}.{cap[1]}  {total:.1f} GB")
        result("PyTorch GPU", True, f"可以用 GPU（{name}, {total:.1f} GB）")
    elif torch.version.cuda is None:
        result("PyTorch GPU", False, "裝的是 CPU 版 PyTorch，請改裝 CUDA 版（見 requirements.txt）")
    else:
        result("PyTorch GPU", False, "CUDA 版 PyTorch，但偵測不到可用的顯示卡（驅動太舊？）")
except Exception as e:
    result("PyTorch GPU", False, f"無法載入 torch：{e}")

# ------------------------------------------------------------------
say("\n3. 實際在 GPU 上計算（4096x4096 矩陣乘法）")
if torch is not None and torch.cuda.is_available():
    try:
        a = torch.randn(4096, 4096)
        t = time.time()
        for _ in range(3):
            a @ a
        cpu = (time.time() - t) / 3
        g = a.cuda()
        g @ g
        torch.cuda.synchronize()
        t = time.time()
        for _ in range(3):
            g @ g
        torch.cuda.synchronize()
        gpu = (time.time() - t) / 3
        used = torch.cuda.max_memory_allocated() / 2**20
        say(f"  CPU {cpu * 1000:.0f} ms   GPU {gpu * 1000:.0f} ms   (GPU 用了 {used:.0f} MB)")
        result("GPU 實算", True, f"GPU 計算正常，比 CPU 快 {cpu / gpu:.1f} 倍")
    except Exception as e:
        result("GPU 實算", False, f"GPU 計算失敗：{e}")
else:
    result("GPU 實算", None, "跳過（PyTorch 沒有可用的 GPU）")

# ------------------------------------------------------------------
say("\n4. faster-whisper / CTranslate2 (語音辨識)")
try:
    import ctranslate2
    n = ctranslate2.get_cuda_device_count()
    say(f"  ctranslate2 {ctranslate2.__version__}  CUDA devices: {n}")
    if n > 0:
        types = sorted(ctranslate2.get_supported_compute_types("cuda"))
        say(f"  supported on GPU: {', '.join(types)}")
        result("Whisper GPU", True, "語音辨識會用 GPU")
    else:
        result("Whisper GPU", False, "語音辨識會用 CPU（較慢）：CTranslate2 偵測不到 CUDA")
except Exception as e:
    result("Whisper GPU", False, f"無法載入 faster-whisper/ctranslate2：{e}")

# ------------------------------------------------------------------
say("\n5. PaddleOCR (系統 Python，設計上用 CPU + MKL-DNN)")
py_launcher = shutil.which("py")
if py_launcher:
    code, out = run([py_launcher, "-3.11", "-c",
                     "import paddle; print('PADDLE_INFO', paddle.__version__, paddle.device.is_compiled_with_cuda())"], 120)
    info = [l for l in out.splitlines() if l.startswith("PADDLE_INFO")]
    out = info[0] if info else out.strip()[-150:]
    if code == 0 and info:
        ver, cuda = out.split()[1:3]
        say(f"  paddlepaddle {ver}  GPU build: {cuda}")
        good = ver.startswith("3.2")
        result("PaddleOCR", True if good else None,
               f"paddlepaddle {ver}，OCR 用 CPU（正常）" + ("" if good else "；建議用 3.2.2，3.3.x 的 MKL-DNN 會出錯"))
    else:
        result("PaddleOCR", False, f"系統 Python 3.11 找不到 paddlepaddle：{out[:150]}")
else:
    result("PaddleOCR", None, "找不到 py 啟動器，跳過（非 Windows 或沒有系統 Python）")

# ------------------------------------------------------------------
say("\n========== 結果 ==========")
for s in summary:
    say(s)
gpu_ok = any(s.startswith("[OK ] PyTorch GPU") for s in summary)
whisper_ok = any(s.startswith("[OK ] Whisper GPU") for s in summary)
if gpu_ok and whisper_ok:
    say("\n有在用 GPU：語音辨識、向量化都會用顯示卡。")
elif gpu_ok or whisper_ok:
    say("\n部分使用 GPU：" + ("向量化" if gpu_ok else "語音辨識") + " 用顯示卡，其餘用 CPU。")
else:
    say("\n沒有用到 GPU：全部用 CPU 執行（可以跑，但處理影片會慢很多）。")

log = ROOT / "results" / "logs" / "gpu_check.txt"
log.parent.mkdir(parents=True, exist_ok=True)
log.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"\nSaved: {log.relative_to(ROOT)}")
