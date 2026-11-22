# 檢索方法比較（vector / bm25 / hybrid）

- 資料庫：results\test_video\chroma_db　題目檔：evaluation\questions\test_video.json
- all = 每種方法排完所有片段再合併；top10 = 各取前 10 名再合併（系統實際設定）；w = bm25_weight
- 題目數：20　片段數：5　向量模型：BAAI/bge-m3（cuda）
- RRF：k=60、vector_weight=1.0
- 重新排序模型：BAAI/bge-reranker-v2-m3

## ALL（20 題）

| 方法 | Hit@1 | Hit@3 | Hit@5 | MRR | 平均時間 (ms) |
|---|---|---|---|---|---|
| vector | 80% | 100% | 100% | 0.892 | 50 |
| bm25 | 55% | 75% | 75% | 0.625 | 38 |
| hybrid (all, w=1) | 75% | 100% | 100% | 0.875 | 40 |
| hybrid (top10, w=1) | 75% | 100% | 100% | 0.875 | 40 |
| hybrid (top10, w=0.5) | 85% | 100% | 100% | 0.917 | 40 |
| hybrid (top10, w=0.3) | 85% | 100% | 100% | 0.917 | 39 |
| vector + rerank | 85% | 100% | 100% | 0.925 | 503 |
| hybrid + rerank | 85% | 100% | 100% | 0.925 | 477 |
| hybrid20 + rerank | 85% | 100% | 100% | 0.925 | 479 |

## keyword（8 題）

| 方法 | Hit@1 | Hit@3 | Hit@5 | MRR | 平均時間 (ms) |
|---|---|---|---|---|---|
| vector | 88% | 100% | 100% | 0.917 | 63 |
| bm25 | 88% | 100% | 100% | 0.917 | 38 |
| hybrid (all, w=1) | 75% | 100% | 100% | 0.875 | 40 |
| hybrid (top10, w=1) | 75% | 100% | 100% | 0.875 | 40 |
| hybrid (top10, w=0.5) | 88% | 100% | 100% | 0.917 | 40 |
| hybrid (top10, w=0.3) | 88% | 100% | 100% | 0.917 | 39 |
| vector + rerank | 100% | 100% | 100% | 1.000 | 543 |
| hybrid + rerank | 100% | 100% | 100% | 1.000 | 476 |
| hybrid20 + rerank | 100% | 100% | 100% | 1.000 | 478 |

## paraphrase（8 題）

| 方法 | Hit@1 | Hit@3 | Hit@5 | MRR | 平均時間 (ms) |
|---|---|---|---|---|---|
| vector | 75% | 100% | 100% | 0.875 | 43 |
| bm25 | 50% | 88% | 88% | 0.646 | 37 |
| hybrid (all, w=1) | 75% | 100% | 100% | 0.875 | 41 |
| hybrid (top10, w=1) | 75% | 100% | 100% | 0.875 | 41 |
| hybrid (top10, w=0.5) | 88% | 100% | 100% | 0.938 | 42 |
| hybrid (top10, w=0.3) | 88% | 100% | 100% | 0.938 | 40 |
| vector + rerank | 75% | 100% | 100% | 0.875 | 476 |
| hybrid + rerank | 75% | 100% | 100% | 0.875 | 478 |
| hybrid20 + rerank | 75% | 100% | 100% | 0.875 | 480 |

## chinese（4 題）

| 方法 | Hit@1 | Hit@3 | Hit@5 | MRR | 平均時間 (ms) |
|---|---|---|---|---|---|
| vector | 75% | 100% | 100% | 0.875 | 40 |
| bm25 | 0% | 0% | 0% | 0.000 | 38 |
| hybrid (all, w=1) | 75% | 100% | 100% | 0.875 | 38 |
| hybrid (top10, w=1) | 75% | 100% | 100% | 0.875 | 40 |
| hybrid (top10, w=0.5) | 75% | 100% | 100% | 0.875 | 40 |
| hybrid (top10, w=0.3) | 75% | 100% | 100% | 0.875 | 38 |
| vector + rerank | 75% | 100% | 100% | 0.875 | 479 |
| hybrid + rerank | 75% | 100% | 100% | 0.875 | 476 |
| hybrid20 + rerank | 75% | 100% | 100% | 0.875 | 477 |

## 每題正確片段的名次（越小越好，- = 沒找到）

| 類型 | 問題 | vector | bm25 | hybrid (all, w=1) | hybrid (top10, w=1) | hybrid (top10, w=0.5) | hybrid (top10, w=0.3) | vector + rerank | hybrid + rerank | hybrid20 + rerank |
|---|---|---|---|---|---|---|---|---|---|---|
| keyword | what does i stand for in for i in range | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| keyword | why does range(1,5) print only four times | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| keyword | Line 9 is not indented | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| keyword | what does print(counter) output | 3 | 1 | 2 | 2 | 3 | 3 | 1 | 1 | 1 |
| keyword | which keyword starts the for loop on line five | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| keyword | 101computing tutorial introduction | 1 | 3 | 2 | 2 | 1 | 1 | 1 | 1 | 1 |
| keyword | increment | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| keyword | five take away one | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| paraphrase | how can I make the computer repeat some instructions many times | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| paraphrase | why did the message appear fewer times than I expected | 1 | 2 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| paraphrase | what happens to a statement that is not shifted to the right | 1 | 1 | 1 | 1 | 1 | 1 | 2 | 2 | 2 |
| paraphrase | does renaming the loop variable change the result | 1 | 3 | 1 | 1 | 1 | 1 | 2 | 2 | 2 |
| paraphrase | how do I put several statements inside the loop body | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| paraphrase | what is displayed when the program is executed | 2 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| paraphrase | how does the loop variable change on each pass | 1 | 3 | 2 | 2 | 1 | 1 | 1 | 1 | 1 |
| paraphrase | how does the lesson finish | 2 | - | 2 | 2 | 2 | 2 | 1 | 1 | 1 |
| chinese | for 迴圈的用途是什麼 | 1 | - | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| chinese | 為什麼只印出四次而不是五次 | 1 | - | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| chinese | 沒有縮排的程式碼會執行幾次 | 2 | - | 2 | 2 | 2 | 2 | 2 | 2 | 2 |
| chinese | 變數名稱 i 代表什麼意思 | 1 | - | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
