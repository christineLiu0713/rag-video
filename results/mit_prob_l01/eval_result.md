# 檢索方法比較（vector / bm25 / hybrid）

- 資料庫：results\mit_prob_l01\chroma_db　題目檔：evaluation\questions\mit_prob_l01.json
- all = 每種方法排完所有片段再合併；top10 = 各取前 10 名再合併（系統實際設定）；w = bm25_weight
- 題目數：20　片段數：95　向量模型：BAAI/bge-m3（cuda）
- RRF：k=60、vector_weight=1.0
- 重新排序模型：BAAI/bge-reranker-v2-m3

## ALL（20 題）

| 方法 | Hit@1 | Hit@3 | Hit@5 | MRR | 平均時間 (ms) |
|---|---|---|---|---|---|
| vector | 75% | 95% | 95% | 0.849 | 57 |
| bm25 | 60% | 70% | 70% | 0.651 | 42 |
| hybrid (all, w=1) | 70% | 95% | 100% | 0.818 | 42 |
| hybrid (top10, w=1) | 70% | 100% | 100% | 0.833 | 41 |
| hybrid (top10, w=0.5) | 70% | 100% | 100% | 0.842 | 40 |
| hybrid (top10, w=0.3) | 75% | 100% | 100% | 0.867 | 40 |
| vector + rerank | 60% | 95% | 95% | 0.775 | 1347 |
| hybrid + rerank | 65% | 90% | 95% | 0.796 | 945 |
| hybrid20 + rerank | 60% | 95% | 95% | 0.772 | 1766 |

## keyword（8 題）

| 方法 | Hit@1 | Hit@3 | Hit@5 | MRR | 平均時間 (ms) |
|---|---|---|---|---|---|
| vector | 88% | 100% | 100% | 0.938 | 75 |
| bm25 | 75% | 100% | 100% | 0.854 | 38 |
| hybrid (all, w=1) | 88% | 100% | 100% | 0.938 | 39 |
| hybrid (top10, w=1) | 88% | 100% | 100% | 0.938 | 37 |
| hybrid (top10, w=0.5) | 88% | 100% | 100% | 0.938 | 38 |
| hybrid (top10, w=0.3) | 88% | 100% | 100% | 0.938 | 36 |
| vector + rerank | 62% | 100% | 100% | 0.812 | 1343 |
| hybrid + rerank | 62% | 100% | 100% | 0.812 | 895 |
| hybrid20 + rerank | 62% | 100% | 100% | 0.812 | 1678 |

## paraphrase（8 題）

| 方法 | Hit@1 | Hit@3 | Hit@5 | MRR | 平均時間 (ms) |
|---|---|---|---|---|---|
| vector | 75% | 88% | 88% | 0.830 | 46 |
| bm25 | 75% | 75% | 75% | 0.775 | 47 |
| hybrid (all, w=1) | 62% | 88% | 100% | 0.754 | 46 |
| hybrid (top10, w=1) | 62% | 100% | 100% | 0.792 | 43 |
| hybrid (top10, w=0.5) | 62% | 100% | 100% | 0.812 | 43 |
| hybrid (top10, w=0.3) | 75% | 100% | 100% | 0.875 | 45 |
| vector + rerank | 62% | 88% | 88% | 0.771 | 1371 |
| hybrid + rerank | 62% | 88% | 88% | 0.771 | 1117 |
| hybrid20 + rerank | 62% | 88% | 88% | 0.764 | 2087 |

## chinese（4 題）

| 方法 | Hit@1 | Hit@3 | Hit@5 | MRR | 平均時間 (ms) |
|---|---|---|---|---|---|
| vector | 50% | 100% | 100% | 0.708 | 42 |
| bm25 | 0% | 0% | 0% | 0.000 | 38 |
| hybrid (all, w=1) | 50% | 100% | 100% | 0.708 | 42 |
| hybrid (top10, w=1) | 50% | 100% | 100% | 0.708 | 43 |
| hybrid (top10, w=0.5) | 50% | 100% | 100% | 0.708 | 37 |
| hybrid (top10, w=0.3) | 50% | 100% | 100% | 0.708 | 40 |
| vector + rerank | 50% | 100% | 100% | 0.708 | 1305 |
| hybrid + rerank | 75% | 75% | 100% | 0.812 | 700 |
| hybrid20 + rerank | 50% | 100% | 100% | 0.708 | 1299 |

## 每題正確片段的名次（越小越好，- = 沒找到）

| 類型 | 問題 | vector | bm25 | hybrid (all, w=1) | hybrid (top10, w=1) | hybrid (top10, w=0.5) | hybrid (top10, w=0.3) | vector + rerank | hybrid + rerank | hybrid20 + rerank |
|---|---|---|---|---|---|---|---|---|---|---|
| keyword | mutually exclusive and collectively exhaustive | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| keyword | tetrahedral die with 16 possible outcomes | 1 | 1 | 1 | 1 | 1 | 1 | 2 | 2 | 2 |
| keyword | what is the union bound | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| keyword | countable additivity axiom | 2 | 1 | 2 | 2 | 2 | 2 | 2 | 2 | 2 |
| keyword | nonnegativity axiom | 1 | 3 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| keyword | discrete uniform probability law | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| keyword | why is the unit square an uncountable set | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| keyword | probability of the empty set | 1 | 2 | 1 | 1 | 1 | 1 | 2 | 2 | 2 |
| paraphrase | does the weather outside matter when listing outcomes of a coin flip | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| paraphrase | why can't each point of a continuous target get its own positive probability | 1 | 6 | 3 | 3 | 2 | 1 | 2 | 2 | 2 |
| paraphrase | how do we draw an experiment that happens in stages | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| paraphrase | if one event is contained in another which one is more likely | 1 | 34 | 5 | 2 | 2 | 2 | 6 | 6 | 9 |
| paraphrase | how to get a probability by counting when all outcomes are equally likely | 7 | 1 | 1 | 1 | 1 | 1 | 2 | 2 | 2 |
| paraphrase | what are the general steps for solving a probability problem | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| paraphrase | how do we check that the given probabilities of the integers really add to one | 2 | 1 | 2 | 2 | 2 | 2 | 1 | 1 | 1 |
| paraphrase | what is the chance that the dart hits one exact point | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| chinese | 樣本空間要怎麼選才適當 | 1 | - | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| chinese | 機率的公理有哪些 | 3 | - | 3 | 3 | 3 | 3 | 3 | 4 | 3 |
| chinese | 兩個事件聯集的機率怎麼算 | 1 | - | 1 | 1 | 1 | 1 | 2 | 1 | 2 |
| chinese | 連續樣本空間的例子 | 2 | - | 2 | 2 | 2 | 2 | 1 | 1 | 1 |
