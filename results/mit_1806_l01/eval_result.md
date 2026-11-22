# 檢索方法比較（vector / bm25 / hybrid）

- 資料庫：results\mit_1806_l01\chroma_db　題目檔：evaluation\questions\mit_1806_l01.json
- all = 每種方法排完所有片段再合併；top10 = 各取前 10 名再合併（系統實際設定）；w = bm25_weight
- 題目數：18　片段數：65　向量模型：BAAI/bge-m3（cuda）
- RRF：k=60、vector_weight=1.0
- 重新排序模型：BAAI/bge-reranker-v2-m3

## ALL（18 題）

| 方法 | Hit@1 | Hit@3 | Hit@5 | MRR | 平均時間 (ms) |
|---|---|---|---|---|---|
| vector | 72% | 89% | 89% | 0.799 | 51 |
| bm25 | 50% | 56% | 67% | 0.569 | 39 |
| hybrid (all, w=1) | 67% | 94% | 94% | 0.783 | 39 |
| hybrid (top10, w=1) | 67% | 94% | 94% | 0.782 | 39 |
| hybrid (top10, w=0.5) | 67% | 89% | 94% | 0.786 | 38 |
| hybrid (top10, w=0.3) | 67% | 89% | 94% | 0.784 | 39 |
| vector + rerank | 72% | 89% | 89% | 0.800 | 1224 |
| hybrid + rerank | 72% | 89% | 89% | 0.800 | 835 |
| hybrid20 + rerank | 72% | 89% | 89% | 0.799 | 1515 |

## keyword（7 題）

| 方法 | Hit@1 | Hit@3 | Hit@5 | MRR | 平均時間 (ms) |
|---|---|---|---|---|---|
| vector | 71% | 86% | 86% | 0.780 | 65 |
| bm25 | 57% | 71% | 100% | 0.714 | 36 |
| hybrid (all, w=1) | 71% | 100% | 100% | 0.810 | 36 |
| hybrid (top10, w=1) | 71% | 100% | 100% | 0.810 | 35 |
| hybrid (top10, w=0.5) | 71% | 86% | 100% | 0.798 | 36 |
| hybrid (top10, w=0.3) | 71% | 86% | 100% | 0.790 | 36 |
| vector + rerank | 71% | 86% | 86% | 0.782 | 1239 |
| hybrid + rerank | 71% | 86% | 86% | 0.782 | 856 |
| hybrid20 + rerank | 71% | 86% | 86% | 0.782 | 1527 |

## paraphrase（7 題）

| 方法 | Hit@1 | Hit@3 | Hit@5 | MRR | 平均時間 (ms) |
|---|---|---|---|---|---|
| vector | 71% | 86% | 86% | 0.774 | 44 |
| bm25 | 71% | 71% | 71% | 0.749 | 43 |
| hybrid (all, w=1) | 57% | 86% | 86% | 0.705 | 44 |
| hybrid (top10, w=1) | 57% | 86% | 86% | 0.701 | 43 |
| hybrid (top10, w=0.5) | 57% | 86% | 86% | 0.724 | 41 |
| hybrid (top10, w=0.3) | 57% | 86% | 86% | 0.724 | 42 |
| vector + rerank | 71% | 86% | 86% | 0.798 | 1180 |
| hybrid + rerank | 71% | 86% | 86% | 0.799 | 926 |
| hybrid20 + rerank | 71% | 86% | 86% | 0.795 | 1645 |

## chinese（4 題）

| 方法 | Hit@1 | Hit@3 | Hit@5 | MRR | 平均時間 (ms) |
|---|---|---|---|---|---|
| vector | 75% | 100% | 100% | 0.875 | 38 |
| bm25 | 0% | 0% | 0% | 0.000 | 37 |
| hybrid (all, w=1) | 75% | 100% | 100% | 0.875 | 37 |
| hybrid (top10, w=1) | 75% | 100% | 100% | 0.875 | 37 |
| hybrid (top10, w=0.5) | 75% | 100% | 100% | 0.875 | 37 |
| hybrid (top10, w=0.3) | 75% | 100% | 100% | 0.875 | 37 |
| vector + rerank | 75% | 100% | 100% | 0.833 | 1275 |
| hybrid + rerank | 75% | 100% | 100% | 0.833 | 639 |
| hybrid20 + rerank | 75% | 100% | 100% | 0.833 | 1265 |

## 每題正確片段的名次（越小越好，- = 沒找到）

| 類型 | 問題 | vector | bm25 | hybrid (all, w=1) | hybrid (top10, w=1) | hybrid (top10, w=0.5) | hybrid (top10, w=0.3) | vector + rerank | hybrid + rerank | hybrid20 + rerank |
|---|---|---|---|---|---|---|---|---|---|---|
| keyword | fundamental problem of linear algebra | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| keyword | row picture where two lines meet | 3 | 4 | 3 | 3 | 3 | 3 | 3 | 3 | 3 |
| keyword | linear combination of the columns | 1 | 2 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| keyword | column picture in three dimensions | 8 | 4 | 3 | 3 | 4 | 5 | 7 | 7 | 7 |
| keyword | non-singular invertible matrix | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| keyword | dot product of a row with x | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| keyword | nine vectors in nine-dimensional space | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| paraphrase | what shape is the set of solutions of one equation in three unknowns | 12 | 10 | 10 | 14 | 14 | 14 | 12 | 11 | 15 |
| paraphrase | when is it impossible to reach every right-hand side | 1 | 7 | 3 | 3 | 2 | 2 | 1 | 1 | 1 |
| paraphrase | how do you multiply a matrix by a vector | 1 | 1 | 1 | 1 | 1 | 1 | 2 | 2 | 2 |
| paraphrase | which point solves both equations in the two by two example | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| paraphrase | why is it hard to see three planes meeting | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| paraphrase | what happens if we keep the matrix but change the right-hand side | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| paraphrase | what do all combinations of the two columns fill | 3 | 1 | 2 | 2 | 2 | 2 | 1 | 1 | 1 |
| chinese | 線性代數的基本問題是什麼 | 1 | - | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| chinese | 矩陣乘以向量怎麼算 | 1 | - | 1 | 1 | 1 | 1 | 3 | 3 | 3 |
| chinese | 奇異矩陣是什麼情況 | 2 | - | 2 | 2 | 2 | 2 | 1 | 1 | 1 |
| chinese | 行向量的線性組合 | 1 | - | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
