# 檢索方法比較（vector / bm25 / hybrid）

- 資料庫：results\lecture2\chroma_db　題目檔：evaluation\questions\lecture2.json
- all = 每種方法排完所有片段再合併；top10 = 各取前 10 名再合併（系統實際設定）；w = bm25_weight
- 題目數：30　片段數：65　向量模型：BAAI/bge-m3（cuda）
- RRF：k=60、vector_weight=1.0
- 重新排序模型：BAAI/bge-reranker-v2-m3

## ALL（30 題）

| 方法 | Hit@1 | Hit@3 | Hit@5 | MRR | 平均時間 (ms) |
|---|---|---|---|---|---|
| vector | 77% | 90% | 100% | 0.846 | 50 |
| bm25 | 50% | 63% | 73% | 0.589 | 41 |
| hybrid (all, w=1) | 63% | 83% | 93% | 0.762 | 41 |
| hybrid (top10, w=1) | 63% | 83% | 93% | 0.769 | 41 |
| hybrid (top10, w=0.5) | 70% | 83% | 93% | 0.797 | 41 |
| hybrid (top10, w=0.3) | 70% | 80% | 93% | 0.794 | 41 |
| vector + rerank | 67% | 93% | 97% | 0.796 | 2124 |
| hybrid + rerank | 67% | 93% | 97% | 0.791 | 1533 |
| hybrid20 + rerank | 67% | 93% | 97% | 0.790 | 2653 |

## keyword（12 題）

| 方法 | Hit@1 | Hit@3 | Hit@5 | MRR | 平均時間 (ms) |
|---|---|---|---|---|---|
| vector | 100% | 100% | 100% | 1.000 | 59 |
| bm25 | 83% | 83% | 92% | 0.868 | 38 |
| hybrid (all, w=1) | 83% | 100% | 100% | 0.917 | 38 |
| hybrid (top10, w=1) | 83% | 100% | 100% | 0.917 | 38 |
| hybrid (top10, w=0.5) | 100% | 100% | 100% | 1.000 | 39 |
| hybrid (top10, w=0.3) | 100% | 100% | 100% | 1.000 | 38 |
| vector + rerank | 83% | 100% | 100% | 0.903 | 2257 |
| hybrid + rerank | 83% | 100% | 100% | 0.903 | 1443 |
| hybrid20 + rerank | 83% | 100% | 100% | 0.903 | 2629 |

## paraphrase（12 題）

| 方法 | Hit@1 | Hit@3 | Hit@5 | MRR | 平均時間 (ms) |
|---|---|---|---|---|---|
| vector | 50% | 83% | 100% | 0.676 | 46 |
| bm25 | 25% | 58% | 75% | 0.424 | 47 |
| hybrid (all, w=1) | 33% | 67% | 83% | 0.551 | 45 |
| hybrid (top10, w=1) | 33% | 67% | 83% | 0.567 | 44 |
| hybrid (top10, w=0.5) | 33% | 67% | 83% | 0.556 | 45 |
| hybrid (top10, w=0.3) | 33% | 58% | 83% | 0.549 | 45 |
| vector + rerank | 42% | 83% | 92% | 0.628 | 2059 |
| hybrid + rerank | 42% | 83% | 92% | 0.616 | 1768 |
| hybrid20 + rerank | 42% | 83% | 92% | 0.615 | 2932 |

## chinese（6 題）

| 方法 | Hit@1 | Hit@3 | Hit@5 | MRR | 平均時間 (ms) |
|---|---|---|---|---|---|
| vector | 83% | 83% | 100% | 0.875 | 41 |
| bm25 | 33% | 33% | 33% | 0.361 | 37 |
| hybrid (all, w=1) | 83% | 83% | 100% | 0.875 | 39 |
| hybrid (top10, w=1) | 83% | 83% | 100% | 0.875 | 40 |
| hybrid (top10, w=0.5) | 83% | 83% | 100% | 0.875 | 39 |
| hybrid (top10, w=0.3) | 83% | 83% | 100% | 0.875 | 40 |
| vector + rerank | 83% | 100% | 100% | 0.917 | 1985 |
| hybrid + rerank | 83% | 100% | 100% | 0.917 | 1241 |
| hybrid20 + rerank | 83% | 100% | 100% | 0.917 | 2142 |

## 每題正確片段的名次（越小越好，- = 沒找到）

| 類型 | 問題 | vector | bm25 | hybrid (all, w=1) | hybrid (top10, w=1) | hybrid (top10, w=0.5) | hybrid (top10, w=0.3) | vector + rerank | hybrid + rerank | hybrid20 + rerank |
|---|---|---|---|---|---|---|---|---|---|---|
| keyword | what does elif stand for | 1 | 6 | 2 | 2 | 1 | 1 | 2 | 2 | 2 |
| keyword | Control-C stops an infinite loop | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| keyword | range(start, stop, step) default values | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| keyword | SyntaxError when using one equal sign | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| keyword | Legend of Zelda lost woods | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| keyword | break statement | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| keyword | concatenation operator plus does not add spaces | 1 | 4 | 2 | 2 | 1 | 1 | 3 | 3 | 3 |
| keyword | compare strings lexicographically | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| keyword | free food mailing list | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| keyword | mysum is 7 plus 8 plus 9 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| keyword | star operator repeats a string | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| keyword | input function prompt | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| paraphrase | how do I glue two pieces of text together | 1 | 3 | 2 | 2 | 2 | 2 | 1 | 1 | 1 |
| paraphrase | what happens if the loop condition never becomes false | 5 | 3 | 2 | 2 | 3 | 4 | 8 | 7 | 8 |
| paraphrase | how can a program choose between several alternatives | 3 | 1 | 2 | 2 | 2 | 2 | 3 | 3 | 3 |
| paraphrase | what kind of data does whatever the user types become | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| paraphrase | how do I turn the user's text into a number | 4 | 3 | 4 | 4 | 4 | 4 | 1 | 1 | 1 |
| paraphrase | can I check whether a word is bigger than a number | 2 | 4 | 1 | 1 | 1 | 1 | 3 | 3 | 3 |
| paraphrase | how do I leave a loop early | 1 | 1 | 1 | 1 | 1 | 1 | 2 | 2 | 2 |
| paraphrase | when should I repeat a fixed number of times versus until something happens | 1 | 23 | 5 | 4 | 4 | 4 | 2 | 3 | 3 |
| paraphrase | how are groups of statements marked in Python | 2 | 20 | 8 | 7 | 6 | 6 | 4 | 4 | 4 |
| paraphrase | difference between giving a variable a value and checking if two values are the same | 1 | 4 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| paraphrase | a strategy to get through a maze by keeping one hand on the wall | 3 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 |
| paraphrase | can the counting function use decimal numbers | 1 | - | 27 | 6 | 6 | 6 | 1 | 1 | 1 |
| chinese | elif 是什麼意思 | 1 | 6 | 1 | 1 | 1 | 1 | 2 | 2 | 2 |
| chinese | 無窮迴圈要怎麼停止 | 4 | - | 4 | 4 | 4 | 4 | 1 | 1 | 1 |
| chinese | 字串可以和數字比較大小嗎 | 1 | - | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| chinese | for 迴圈和 while 迴圈有什麼差別 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| chinese | break 會跳出哪一個迴圈 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| chinese | 怎麼讓使用者輸入資料 | 1 | - | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
