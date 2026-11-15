# RAG 教學影片智慧搜尋（Lecture Video RAG）

從一支英文教學影片出發：用 Whisper 取得語音文字、用 PaddleOCR 取得投影片文字，
切分並對齊後轉成向量存入 Chroma，最後可以用問題搜尋影片片段，並由 Gemini 產生簡短回答。

## 檔案說明

| 檔案 | 功能 | 輸入 → 輸出 |
|---|---|---|
| `extract.py` | 每秒截圖 1 張、Whisper 語音轉文字、PaddleOCR 辨識畫面文字（含過濾 logo、亂碼） | `video.mp4` → `whisper.json`、`ocr.json`、`frames/`、`meta.json` |
| `settings.json` | OCR 過濾、GPU、預處理與分段參數，供 `extract.py` 和 `build_chunks.py` 共用 | — |
| `build_chunks.py` | 清理、分段（約 45 秒）、對齊（語音與投影片依時間配對） | `whisper.json`、`ocr.json` → `chunks.jsonl` |
| `add_captions.py` | （選用）用 BLIP 為每個片段的截圖產生描述，補充 OCR 看不到的圖片內容 | `chunks.jsonl` → 更新後的 `chunks.jsonl` |
| `build_db.py` | 用 BGE-M3 轉成 1024 維向量，存入 Chroma | `chunks.jsonl` → `output/chroma_db/` |
| `search.py` | 輸入問題，顯示最接近的 3 個影片片段（時間、距離、截圖） | 問題 → 片段 |
| `answer.py` | 先檢索，再把片段交給 Gemini，產生附時間點的簡短回答 | 問題 → 回答與來源 |
| `.gitignore` | 排除 `venv/`、`output/`、影片與金鑰檔，避免上傳 | — |

`extract.py` 的選項：

- `--skip-whisper`：重新截圖並做 OCR，保留舊的 `whisper.json`
- `--ocr-only`：只對現有截圖重做 OCR

可直接編輯 `settings.json` 調整截圖間隔、OCR 信心度、相似畫面跳過門檻、GPU 開關、影像預處理，以及分段長度和換頁邊界搜尋範圍。OCR 偵測到標題改變或投影片文字明顯不同時會建立新區塊，分段會優先在該轉換附近的語音句界切開。GPU 需要安裝支援 CUDA 的 PyTorch/Paddle；未偵測到 CUDA 時會使用 CPU。

## 使用的技術

| 任務 | 技術 | 說明 |
|---|---|---|
| 分離音訊、截圖 | FFmpeg | 每秒 1 張截圖 |
| 語音轉文字 | faster-whisper（`small.en`） | int8，輸出短句與時間 |
| 畫面文字辨識 | PaddleOCR PP-OCRv4 mobile | 英文輕量模型；信心度 ≥ 0.7、至少 4 字元、排除底部 6% 的 logo 區域 |
| 減少 OCR 次數 | Pillow | 畫面幾乎沒變就沿用上一次結果 |
| 分段 | Python | 約 45 秒（上限 180 字）、不切斷句子、重疊 1 句 |
| 對齊 | Python | 時間重疊 ≥ 3 秒的投影片文字附加到片段 |
| 圖片描述（選用） | BLIP（`blip-image-captioning-base`） | 補充 OCR 看不到的插圖內容 |
| 向量化 | BAAI/bge-m3 | 1024 維、多語言、最長 8192 tokens |
| 向量資料庫 | Chroma | 內部以 SQLite 儲存，另含 id、文字、metadata |
| 回答生成 | Gemini Flash-Lite（`google-genai`） | 只閱讀檢索到的片段，找不到就回答找不到 |

向量化用的文字格式：`Slide: … Visual: … Speech: …`

## 環境

- 作業系統：Windows
- 有 NVIDIA GPU 時可讓 Whisper 與 BGE-M3 使用 CUDA；需安裝相容的 GPU 套件。未偵測到 CUDA 時會退回 CPU
- Python 3.11（建議）。太新的版本可能裝不到 PaddlePaddle（我們在 3.15 遇到此問題）
- 使用虛擬環境 `venv`
- FFmpeg：`winget install Gyan.FFmpeg`

## 安裝

先在系統 Python 安裝 OCR 的 CPU 套件，讓 OCR 工作程序可以和 CUDA PyTorch 分開執行：

```powershell
py -3.11 -m pip install paddlepaddle==3.3.1 paddleocr==3.7.0
```

建立主環境時，使用 `--system-site-packages` 讓它能呼叫系統環境裡的 OCR 套件；GTX 1060 使用支援 Pascal 架構的 PyTorch CUDA 12.6 版本：

```powershell
py -3.11 -m venv --system-site-packages venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install faster-whisper sentence-transformers chromadb google-genai
```

## 啟用已設定的 GPU 環境（Windows PowerShell）

在專案資料夾開啟 PowerShell，執行：

```powershell
cd "D:\7\413410654\專題\rag-video-main"
.\venv\Scripts\Activate.ps1
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

啟用成功時，命令列開頭會顯示 `(venv)`，檢查結果應包含 `True` 和 `NVIDIA GeForce GTX 1060 6GB`。如果 PowerShell 阻擋啟用腳本，先在同一個視窗執行 `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`，再啟用一次。完成後輸入 `deactivate` 離開環境。

在這台 Windows 電腦上，Whisper 與 BGE-M3 會在 `venv` 使用 CUDA；PaddleOCR 由系統 Python 的獨立程序執行，避免 Paddle 與 PyTorch 的 CUDA DLL 載入衝突。Paddle GPU 在 GTX 1060 上目前辨識文字不正確，因此 OCR 保持 CPU 執行。可用 `nvidia-smi` 確認顯示卡與驅動，並檢查 `torch.cuda.is_available()` 是否為 `True`。

## 執行順序

把影片命名為 `video.mp4`，放在專案資料夾，然後依序執行：

```
python extract.py video.mp4
python build_chunks.py
python add_captions.py      （選用，需要時才執行）
python build_db.py
python search.py
python answer.py
```

第一次執行時會下載模型（Whisper、PaddleOCR、BGE-M3 約 2 GB、BLIP 約 1 GB），需要網路。

## 使用 answer.py 前

到 https://aistudio.google.com/apikey 取得 API key，並設定環境變數：

```
set GEMINI_API_KEY=你的金鑰
```

**不要把金鑰寫進程式碼或上傳到 GitHub。**

## 注意事項

- 影片與 `output/` 資料夾（截圖、JSON、資料庫）不在這個儲存庫裡，需要自己執行程式產生。
- 修改 `extract.py` 或 `build_chunks.py` 之後，後面的步驟都要重新執行。

## 目前限制

- OCR 與 BLIP 都無法判斷「哪個數字對應哪個圖示」。
- 只用一支約 6 分鐘的影片測試，分段長度與距離門檻尚未系統性比較。
- 測驗題生成尚未實作。
