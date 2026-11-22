# RAG 教學影片智慧搜尋（Lecture Video RAG）

把中英文教學影片變成可以發問的資料庫：用 Whisper 取得語音文字、用 PaddleOCR 取得投影片文字，
切分並對齊後轉成向量存入 Chroma；發問時用混合檢索找出片段，再由 Gemini 產生附「影片名稱 + 時間」出處的簡短回答。

- 6 支測試影片（約 5 小時，中英文）處理共 63 分鐘，正確片段在前 5 名的比例平均 95.7%。
- 所有改動與測試數據：[docs/CHANGELOG.md](docs/CHANGELOG.md)
- 給其他功能（出考卷、摘要…）直接使用的課程資料檔：[exports/](exports/README.md)

## 資料夾結構

```
rag-video-main/
├─ README.md、requirements.txt、settings.json、.gitignore
├─ extract.py、ocr_worker.py、build_chunks.py、add_captions.py、build_db.py   處理影片
├─ check_gpu.py                檢查有沒有用到 GPU（給測試者用）
├─ pipeline.py                 一鍵處理影片／整個測試集，記錄時間與記憶體，自動更新影片庫
├─ build_library.py            把所有影片合併成影片庫 library/
├─ hybrid.py、rag.py           檢索（BM25 + 向量）與問答核心
├─ search.py、answer.py        命令列搜尋與問答
├─ export_dataset.py           匯出課程資料檔 exports/lecture_dataset.json
├─ evaluation/     eval_search.py（檢索）、eval_answer.py（回答品質）、eval_transcript.py（語音錯字率）、
│                  testset.json（測試集）、questions/（題目）、references/（官方字幕）、unanswerable.json（拒答題）
├─ exports/        lecture_dataset.json（課程資料檔）與格式說明
├─ docs/           CHANGELOG.md（改動紀錄與測試數據）
├─ tools/          雙擊執行的 .bat；tools/old/ 是測試時用過的一次性檔案
├─ videos/         影片檔（不上傳 GitHub）
├─ output/         正在處理的影片（處理完會移到 results/）
├─ results/        每支影片的結果、testset_report.md、logs/（截圖、資料庫、音訊不上傳）
└─ library/        影片庫（不上傳，pipeline.py 自動產生）
```

## 檔案說明

| 檔案 | 功能 | 輸入 → 輸出 |
|---|---|---|
| `extract.py` | 每隔 `interval` 秒截圖、Whisper 語音轉文字、PaddleOCR 辨識畫面文字（含過濾 logo、亂碼）；`--lang zh` 處理中文影片 | `video.mp4` → `whisper.json`、`ocr.json`、`frames/`、`meta.json` |
| `settings.json` | OCR 過濾、GPU、預處理與分段參數，供 `extract.py` 和 `build_chunks.py` 共用 | — |
| `build_chunks.py` | 清理、分段（約 45 秒）、對齊（語音與投影片依時間配對） | `whisper.json`、`ocr.json` → `chunks.jsonl` |
| `add_captions.py` | （選用）用 BLIP 為每個片段的截圖產生描述，補充 OCR 看不到的圖片內容 | `chunks.jsonl` → 更新後的 `chunks.jsonl` |
| `build_db.py` | 用 BGE-M3 轉成 1024 維向量，存入 Chroma | `chunks.jsonl` → `output/chroma_db/` |
| `hybrid.py` | 混合檢索：BM25 關鍵字 + BGE-M3 向量，用 RRF 合併排名（供 `search.py`、`answer.py` 使用） | — |
| `search.py` | 輸入問題，用混合檢索顯示最相關的 3 個影片片段；`--mode vector/bm25/hybrid/compare` 可切換或並排比較 | 問題 → 片段 |
| `rag.py` | 問答核心 `VideoRAG.ask()`：檢索 → 過濾不相關片段 → Gemini 回 JSON → 程式產生出處 | 問題 → 回答、出處、片段 |
| `answer.py` | 命令列問答（呼叫 `rag.py`）；`--video` 只問某支影片、`-q` 問一題 | 問題 → 回答與來源 |
| `build_library.py` | 合併 `results/*/chroma_db` 成 `library/chroma_db`，直接複製向量 | → `library/` |
| `export_dataset.py` | 匯出逐字稿、投影片文字、片段（含時間）給其他功能使用 | `results/` → `exports/lecture_dataset.json` |
| `evaluation/eval_answer.py` | 回答品質：正確率、引用正確率、疑似亂編、拒答（評審為 Gemini） | → `results/answer_eval/` |
| `evaluation/eval_search.py` | 用標好答案的題目比較 vector / bm25 / hybrid 的 Hit@1、Hit@3、MRR | 題目檔 + 資料庫 → `eval_result.md`、`eval_detail.csv` |
| `.gitignore` | 排除 `venv/`、`output/`、影片、截圖、資料庫與金鑰檔，避免上傳 | — |

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
| 關鍵字檢索 | BM25（自行實作，無額外套件） | 英文單字、數字；中文用單字＋相鄰兩字 |
| 混合排名 | Reciprocal Rank Fusion（RRF, k=60） | 只用名次合併兩種結果，不需換算分數 |
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
py -3.11 -m pip install paddlepaddle==3.2.2 paddleocr==3.7.0
```

建立主環境時，使用 `--system-site-packages` 讓它能呼叫系統環境裡的 OCR 套件；GTX 1060 使用支援 Pascal 架構的 PyTorch CUDA 12.6 版本：

```powershell
py -3.11 -m venv --system-site-packages venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
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

## 檢查有沒有用到 GPU

雙擊 `tools/check_gpu.bat`（或 `python check_gpu.py`），會檢查並實際在顯示卡上算一次：

| 項目 | 用途 | 正常結果 |
|---|---|---|
| 顯示卡與驅動（nvidia-smi） | — | 顯示卡名稱與記憶體 |
| PyTorch GPU + 實際矩陣乘法 | BGE-M3 向量、重新排序 | `[OK]`，並顯示比 CPU 快幾倍 |
| faster-whisper（CTranslate2） | 語音辨識 | `[OK]` |
| PaddleOCR（系統 Python） | OCR | 版本 3.2.x，用 CPU（設計如此） |

最後一行會總結「有在用 GPU／部分使用／沒有用到 GPU」，結果存在 `results/logs/gpu_check.txt`，可以直接傳給開發者。
另外 `pipeline.py` 處理影片時，每個步驟結束都會印出 `GPU: used (peak … MB)` 或 `GPU: not used`，
並寫進 `timings.json` 和 `results/testset_report.md` 的「用到 GPU 的步驟」欄。

## 執行順序

**處理一支影片（最簡單）：** 把影片放進 `videos/`，打開 `tools/run_pipeline.bat` 修改最上面三行（影片路徑、名稱、語言 `en`/`zh`），再雙擊執行。之前 `output/` 裡的結果會先自動移到 `results/<名稱>/`，紀錄寫在 `results/logs/`。

**用指令：**

```
python pipeline.py videos/lecture.mp4 --name lecture --lang en
python search.py
python answer.py
```

`pipeline.py` 會依序執行下面三個程式，並把每個步驟的時間、RAM／顯示卡記憶體峰值記到 `output/timings.json`。也可以手動一個一個執行：

```
python extract.py videos/lecture.mp4 --lang en     （中文影片用 --lang zh）
python build_chunks.py lecture                     （片段 id 會是 lecture_001、…）
python add_captions.py                             （選用，需要時才執行）
python build_db.py
```

## 使用 answer.py 前

到 https://aistudio.google.com/apikey 取得 API key，在專案資料夾建立 `gemini_key.txt`，裡面只放金鑰一行（已列在 `.gitignore`，不會上傳）。也可以改用環境變數：

```
set GEMINI_API_KEY=你的金鑰
```

```
python answer.py                         # 問影片庫裡全部影片
python answer.py --video mit_prob_l01    # 只問一支
python answer.py -q "what is the union bound"
```

**不要把金鑰寫進程式碼或上傳到 GitHub。**

## 注意事項

- 影片、`output/` 與 `results/` 裡的截圖和資料庫不在這個儲存庫裡，需要自己執行程式產生。
- 修改 `extract.py` 或 `build_chunks.py` 之後，後面的步驟都要重新執行。

## OCR 加速

`ocr_worker.py` 的加速方式（在 `settings.json` 調整）：

| 設定 | 預設 | 作用 |
|---|---|---|
| `ocr_workers` | `1` | 同時執行的 OCR 程序數；`0` = 自動。開啟 MKL-DNN 時 1 個程序最快（實測 2 個反而較慢） |
| `ocr_mkldnn` | `true` | 開啟 CPU 加速（MKL-DNN/oneDNN）；若出錯會自動改回關閉。**paddlepaddle 3.3.x 有 bug 無法開啟，請用 3.2.2** |
| `ocr_max_side` | `960` | 偵測文字前把圖片長邊縮到 960 像素以內；`0` = 不縮 |
| `ocr_gpu` | `false` | PaddleOCR 用 GPU；GTX 1060 上辨識結果不正確，所以預設關閉 |
| `interval` | `3` | 每幾秒截一張圖 |
| `ocr_adaptive` | `true` | 自動放慢取樣：連續幾張都是同一頁時跳著做 OCR，文字變了就用二分搜尋找出換頁位置 |
| `adaptive_after`、`adaptive_max_stride` | `2`、`4` | 連續 2 次同一頁後開始加大間隔，最多每 4 張做一次 OCR |
| `slide_match` | `tokens` | 判斷是不是同一頁：`tokens` 看共用的字（抗老師遮擋、浮水印錯字）；`strict` 為舊方法（整串文字相似度 ≥ 85%） |
| `slide_match_threshold` | `0.5` | `tokens`：共用字數 ÷ 字數較少那張 ≥ 這個值就算同一頁 |
| `overlay_filter`、`overlay_min_ratio` | `true`、`0.3` | 自動移除浮水印／頻道名稱／固定頁尾：在畫面邊緣、同位置、相似文字出現在 30% 以上有字的截圖，且分散在影片 5 段中的至少 4 段 |

測試哪種設定在自己的電腦最快（需先有 `output/frames/`）：

```powershell
py -3.11 ocr_worker.py --benchmark 8
```

會列出每種設定的時間、加速倍數，以及辨識文字和原始設定的相似度。

## 中文影片

`settings.json` 的 `language`（或 `extract.py --lang`）設成 `zh` 時：

| 項目 | 英文（en） | 中文（zh） |
|---|---|---|
| Whisper 模型 | `small.en`（英文專用，較快） | `medium`（多語言，中文較準；第一次會下載約 1.5 GB） |
| Whisper 提示 | 無 | 「以下是台灣大學的課程錄影，使用繁體中文。」讓輸出偏向繁體 |
| OCR 辨識模型 | `en_PP-OCRv4_mobile_rec` | `PP-OCRv4_mobile_rec`（中英文通用） |
| 分段字數 | 以空白分隔的單字 | 每個中文字算一個字（中文沒有空格） |

`whisper_model` 可在 `settings.json` 指定（`auto` = 依語言自動選）。`whisper_batch_size`（預設 8）開啟批次處理加速；舊版 faster-whisper 或顯示卡記憶體不足時會自動改回一般模式，`0` = 關閉。

## 測試集與評估

測試集清單在 `evaluation/testset.json`，涵蓋不同語言與影片類型：

| 名稱 | 類型 | 來源（皆為 CC BY-NC-SA） |
|---|---|---|
| `test_video` | 英文・短片 | — |
| `mit_prob_l01` | 英文・投影片螢幕錄影（附官方字幕） | MIT RES.6-012 Introduction to Probability, Lecture 1 |
| `mit_1806_l01` | 英文・黑板課（附官方字幕） | MIT 18.06 Linear Algebra (Strang), Lecture 1 |
| `lecture2` | 英文・教室錄影 | MIT 6.0001, Lecture 2 |
| `ntu_cs_12` | 中文・教室錄影（英文投影片）・82 分鐘 | 臺大開放式課程 計算機概論 第 12 講 |
| `ntu_zhuangzi_01` | 中文・人文課程 | 臺大開放式課程 正是時候讀莊子 第 1 講 |

- 第一次使用：雙擊 `tools/prepare_testset.bat`，把下載的影片整理進 `videos/` 並建立清單。
- 執行全部：雙擊 `tools/run_testset.bat`（約數小時）。結果在 `results/testset_report.md` 與 `.csv`：每支影片每個步驟的時間、處理時間 ÷ 影片長度、OCR 次數、記憶體峰值、語音辨識錯誤率、檢索準確度。
- 題目格式用 `anchors`（答案在影片中的秒數），評估時自動對應到片段，所以改了切片段的方式題目仍然有效。
- 語音辨識錯誤率：`evaluation/eval_transcript.py` 與官方字幕比較（英文 WER、中文 CER）。

## 混合檢索（Hybrid Search）

只用向量檢索時，系統懂「意思」，但對確切的字（函式名稱、數字、少見術語）不夠敏感；BM25 剛好相反。`hybrid.py` 同時做兩種檢索，各取前 `candidates` 名，再用 RRF 合併：

```
RRF 分數 = vector_weight / (rrf_k + 向量名次) + bm25_weight / (rrf_k + BM25 名次)
```

- BM25 的文字直接從 Chroma 資料庫讀出，**不需要重新執行 `build_db.py`**。
- `answer.py` 保留片段的條件改為：向量距離 ≤ `MAX_DISTANCE` **或** BM25 分數 ≥ `min_bm25`。
- 可在 `settings.json` 調整：`search_mode`（`hybrid` / `vector` / `bm25`）、`candidates`、`rrf_k`、`vector_weight`、`bm25_weight`、`bm25_k1`、`bm25_b`、`min_bm25`。
- 比較三種方法：`python search.py --mode compare`
- 系統化評估：`python evaluation/eval_search.py <題目檔> <資料庫>`，或雙擊 `tools/run_rerank_eval.bat` 評估全部測試影片。
- **繁簡統一**：安裝 `opencc-python-reimplemented` 後，BM25 會把簡體、繁體當成同一個字（OCR 常輸出簡體）。
- **重新排序（選用，預設關閉）**：`rerank: true` 時，用 `BAAI/bge-reranker-v2-m3` 把問題和候選片段一起讀再重新排名。實測 6 支影片沒有變好（前 3 名 93.3% → 90.8%），在 GTX 1060 上每題多 1～3 秒，所以預設關閉。
- **影片庫**：`pipeline.py` 處理完會自動執行 `build_library.py`。`search.py`、`answer.py` 預設讀 `library/chroma_db`，可用 `--video` 只搜尋某幾支，或 `--db results/<影片>/chroma_db` 指定單支影片。

## 問答流程（rag.py）

1. 混合檢索前 5 個片段（`answer_top_k`）。
2. 過濾不相關的片段（向量距離 ≤ `max_distance` 或 BM25 ≥ `min_bm25`）；全部不相關就直接回答「影片裡找不到」，不呼叫 Gemini。
3. Gemini 只讀這些片段，回傳 JSON：`found`（有沒有答案）、`answer`、`used_parts`（用了第幾段）。
4. 程式把 `used_parts` 換成「影片名稱 + 時間」當出處，模型不會編出不存在的時間。
5. Gemini 遇到 429／500／503 會自動等待重試。

回答品質評估：放好金鑰後雙擊 `tools/run_answer_eval.bat`，結果在 `results/answer_eval/answer_eval.md` 與 `answers.csv`。

## 課程資料檔（給出考卷等功能）

`exports/lecture_dataset.json` 收錄 6 支影片的片段、逐句逐字稿、投影片文字與時間，不需要安裝本專案就能使用。格式與出題範例見 [exports/README.md](exports/README.md)；處理新影片後雙擊 `tools/export_dataset.bat` 重新產生。

## 目前限制

- OCR 與 BLIP 都無法判斷「哪個數字對應哪個圖示」。
- 分段長度與距離門檻尚未系統性比較。
- 測驗題生成不在本專案實作；提供 `exports/lecture_dataset.json` 給負責的人使用。
- 回答品質評估（`eval_answer.py`）程式已完成，尚待執行；評審也是 Gemini，需人工抽查。
- 多支影片放在同一個影片庫時，主題相近的影片會互搶答案（前 5 名命中約 85%）；網站上建議先選課程。
- BM25 沒有做詞形還原（stemming），`indentation` 和 `indented` 會被當成不同的字。
- 測試題目由開發者撰寫，可能偏向系統；之後應請其他人出題。
- 浮水印過濾需要至少 20 張有字的截圖才會啟用，很短的影片不會過濾。
