# 課程資料檔 `lecture_dataset.json`

rag-video 處理好的 6 支課程影片（約 5 小時，中英文各有），整理成一個 JSON 檔。
**不用安裝本專案、不用 GPU**，任何程式語言讀 JSON 就能用，例如拿來出考卷、做摘要、做字卡、做課程索引。

重新產生（處理新影片後）：

```
python export_dataset.py                       # 全部影片
python export_dataset.py --only mit_prob_l01   # 只匯出某幾支
```

## 內容

| name | 課程 | 語言 | 長度 | 片段 |
|---|---|---|---|---|
| `test_video` | Python 入門短片（測試用） | 英文 | 3 分 | 5 |
| `mit_prob_l01` | MIT RES.6-012 機率導論 第 1 講：機率模型與公理 | 英文 | 68 分 | 95 |
| `mit_1806_l01` | MIT 18.06 線性代數 第 1 講：線性方程組的幾何 | 英文 | 40 分 | 65 |
| `lecture2` | MIT 6.0001 Python 程式設計 第 2 講：分支與迴圈 | 英文 | 44 分 | 65 |
| `ntu_cs_12` | 臺大 計算機概論 第 12 講：程式語言 | 中文（英文投影片） | 82 分 | 126 |
| `ntu_zhuangzi_01` | 臺大 正是時候讀莊子 第 1 講：逍遙遊 | 中文 | 69 分 | 123 |

## 格式

```json
{
  "schema_version": 1,
  "generated": "2026-11-22",
  "videos": [
    {
      "name": "mit_prob_l01",
      "title": "MIT RES.6-012 Introduction to Probability, Lecture 1: ...",
      "license": "CC BY-NC-SA 4.0 (MIT OpenCourseWare)",
      "language": "en",
      "duration_sec": 4059.6,
      "counts": {"chunks": 95, "transcript_segments": 1026, "slides": 14},
      "chunks": [
        {"id": "mit_prob_l01_004", "start": 110.5, "end": 155.5,
         "speech": "Putting together a probabilistic model ... involves two steps. ...",
         "slide": "Sample space"}
      ],
      "transcript": [{"start": 0.7, "end": 4.2, "text": "Welcome to the first lecture of this class."}],
      "slides": [{"start": 294.0, "end": 450.0, "text": "Sample space List set of possible outcomes ..."}]
    }
  ]
}
```

| 欄位 | 說明 | 適合用來 |
|---|---|---|
| `chunks` | 約 45 秒一段，語音文字 `speech` + 當時畫面上的投影片文字 `slide` | **出題的主要材料**：一段可出 1～3 題，題目附上 `start` 當「答案在影片幾分幾秒」 |
| `transcript` | Whisper 逐句逐字稿，每句有開始／結束秒數 | 需要精確時間、做字幕、找原句 |
| `slides` | 每一頁投影片（或畫面文字）出現的時間區間 | 依投影片分章節、出「這頁在講什麼」的題目 |

時間單位都是秒，從影片開頭算起；`12:34` = `754` 秒。

## 使用注意

- `speech` 是自動語音辨識，人名和少見名詞可能聽錯（例如莊子影片：「鯤」→「昆」、「列子」→「烈子」）。
- `slide` 是 OCR，可能有錯字或簡體字；中文影片的投影片文字比較不可靠，出題建議以 `speech` 為主。
- 中文逐字稿有些句子很長（Whisper 沒有斷句），用 `chunks` 會比 `transcript` 好用。
- 出完的題目請人工檢查：答案應該能在該片段的 `speech` 裡找到。

## 範例：用 Gemini 出選擇題

```python
import json
from google import genai

data = json.load(open("exports/lecture_dataset.json", encoding="utf-8"))
video = next(v for v in data["videos"] if v["name"] == "mit_prob_l01")
chunks = video["chunks"][:10]                       # 前 10 段（約 7 分鐘）

material = "\n\n".join(f"[{c['id']} | {int(c['start'])//60:02d}:{int(c['start'])%60:02d}]\n{c['speech']}"
                       for c in chunks)
prompt = f"""根據下面的課程片段出 5 題四選一選擇題，用繁體中文。
每題輸出 JSON：{{"question", "options": [4 個], "answer": 正確選項的索引, "explanation", "chunk_id"}}。
只能根據片段內容出題，答案必須能在片段中找到。

{material}"""

client = genai.Client(api_key="你的金鑰")
resp = client.models.generate_content(model="gemini-3.5-flash-lite", contents=prompt)
print(resp.text)
```

每題的 `chunk_id` 對回 `chunks` 就能拿到 `start`，考卷上可以寫「詳見影片 01:50」。

## 授權

影片內容來自 MIT OpenCourseWare 與臺大開放式課程，皆為 **CC BY-NC-SA**：
使用這份資料要標示來源（`title`、`license` 欄位）、**不可商業使用**、改作後也要用相同授權分享。
