# 檢索方法比較（vector / bm25 / hybrid）

- 資料庫：results\ntu_zhuangzi_01\chroma_db　題目檔：evaluation\questions\ntu_zhuangzi_01.json
- all = 每種方法排完所有片段再合併；top10 = 各取前 10 名再合併（系統實際設定）；w = bm25_weight
- 題目數：22　片段數：123　向量模型：BAAI/bge-m3（cuda）
- RRF：k=60、vector_weight=1.0
- 重新排序模型：BAAI/bge-reranker-v2-m3

## ALL（22 題）

| 方法 | Hit@1 | Hit@3 | Hit@5 | MRR | 平均時間 (ms) |
|---|---|---|---|---|---|
| vector | 59% | 86% | 91% | 0.730 | 54 |
| bm25 | 45% | 68% | 77% | 0.577 | 49 |
| hybrid (all, w=1) | 59% | 86% | 91% | 0.733 | 44 |
| hybrid (top10, w=1) | 59% | 91% | 91% | 0.742 | 41 |
| hybrid (top10, w=0.5) | 64% | 86% | 91% | 0.754 | 40 |
| hybrid (top10, w=0.3) | 68% | 86% | 91% | 0.777 | 40 |
| vector + rerank | 55% | 73% | 91% | 0.672 | 1484 |
| hybrid + rerank | 55% | 73% | 91% | 0.670 | 1152 |
| hybrid20 + rerank | 55% | 73% | 91% | 0.670 | 2046 |

## keyword（10 題）

| 方法 | Hit@1 | Hit@3 | Hit@5 | MRR | 平均時間 (ms) |
|---|---|---|---|---|---|
| vector | 60% | 80% | 90% | 0.706 | 64 |
| bm25 | 60% | 90% | 90% | 0.719 | 49 |
| hybrid (all, w=1) | 50% | 90% | 90% | 0.686 | 41 |
| hybrid (top10, w=1) | 50% | 90% | 90% | 0.683 | 38 |
| hybrid (top10, w=0.5) | 60% | 80% | 90% | 0.708 | 40 |
| hybrid (top10, w=0.3) | 60% | 80% | 90% | 0.708 | 39 |
| vector + rerank | 50% | 70% | 90% | 0.645 | 1495 |
| hybrid + rerank | 50% | 70% | 90% | 0.645 | 1122 |
| hybrid20 + rerank | 50% | 70% | 90% | 0.645 | 2126 |

## paraphrase（8 題）

| 方法 | Hit@1 | Hit@3 | Hit@5 | MRR | 平均時間 (ms) |
|---|---|---|---|---|---|
| vector | 50% | 100% | 100% | 0.750 | 43 |
| bm25 | 50% | 75% | 100% | 0.688 | 52 |
| hybrid (all, w=1) | 75% | 100% | 100% | 0.875 | 45 |
| hybrid (top10, w=1) | 75% | 100% | 100% | 0.875 | 43 |
| hybrid (top10, w=0.5) | 75% | 100% | 100% | 0.875 | 40 |
| hybrid (top10, w=0.3) | 88% | 100% | 100% | 0.938 | 41 |
| vector + rerank | 75% | 88% | 100% | 0.823 | 1429 |
| hybrid + rerank | 75% | 88% | 100% | 0.823 | 1287 |
| hybrid20 + rerank | 75% | 88% | 100% | 0.823 | 2178 |

## english（4 題）

| 方法 | Hit@1 | Hit@3 | Hit@5 | MRR | 平均時間 (ms) |
|---|---|---|---|---|---|
| vector | 75% | 75% | 75% | 0.753 | 49 |
| bm25 | 0% | 0% | 0% | 0.000 | 41 |
| hybrid (all, w=1) | 50% | 50% | 75% | 0.565 | 48 |
| hybrid (top10, w=1) | 50% | 75% | 75% | 0.625 | 46 |
| hybrid (top10, w=0.5) | 50% | 75% | 75% | 0.625 | 43 |
| hybrid (top10, w=0.3) | 50% | 75% | 75% | 0.625 | 42 |
| vector + rerank | 25% | 50% | 75% | 0.438 | 1564 |
| hybrid + rerank | 25% | 50% | 75% | 0.425 | 954 |
| hybrid20 + rerank | 25% | 50% | 75% | 0.425 | 1583 |

## 每題正確片段的名次（越小越好，- = 沒找到）

| 類型 | 問題 | vector | bm25 | hybrid (all, w=1) | hybrid (top10, w=1) | hybrid (top10, w=0.5) | hybrid (top10, w=0.3) | vector + rerank | hybrid + rerank | hybrid20 + rerank |
|---|---|---|---|---|---|---|---|---|---|---|
| keyword | 北冥有魚，其名為鯤 | 2 | 3 | 2 | 2 | 2 | 2 | 2 | 2 | 2 |
| keyword | 鯤是什麼意思 | 40 | 47 | 42 | - | - | - | - | - | - |
| keyword | 大鵬要飛上九萬里 | 3 | 2 | 2 | 2 | 3 | 3 | 2 | 2 | 2 |
| keyword | 蜩與學鳩 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| keyword | 朝菌不知晦朔 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| keyword | 彭祖活了七百八十歲 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| keyword | 宋榮子猶然笑之 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| keyword | 列子御風而行 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| keyword | 至人無己，神人無功，聖人無名 | 5 | 1 | 3 | 3 | 4 | 4 | 5 | 5 | 5 |
| keyword | 乘天地之正，御六氣之辯 | 1 | 3 | 2 | 2 | 1 | 1 | 4 | 4 | 4 |
| paraphrase | 魚變成鳥代表什麼樣的改變 | 1 | 4 | 1 | 1 | 1 | 1 | 3 | 3 | 3 |
| paraphrase | 為什麼大鳥起飛需要很強的風 | 2 | 1 | 1 | 1 | 1 | 1 | 4 | 4 | 4 |
| paraphrase | 要去越遠的地方就要準備越多東西 | 1 | 4 | 2 | 2 | 2 | 1 | 1 | 1 | 1 |
| paraphrase | 壽命很短的生物沒辦法理解時間 | 2 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| paraphrase | 全世界的人都批評他也不在意 | 2 | 2 | 2 | 2 | 2 | 2 | 1 | 1 | 1 |
| paraphrase | 還要等待外在條件才能行動，所以不是最高境界 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| paraphrase | 別人得到你想要的東西，你也替他高興 | 2 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| paraphrase | 冥這個字的意思 | 1 | 2 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| english | why does the giant bird need such a strong wind to fly | 1 | - | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| english | what is the highest state described by Zhuangzi | 97 | - | 97 | - | - | - | - | - | - |
| english | the story of a huge fish that turns into a bird | 1 | - | 1 | 1 | 1 | 1 | 2 | 2 | 2 |
| english | does Zhuangzi think small birds are as good as the big one | 1 | - | 4 | 2 | 2 | 2 | 4 | 5 | 5 |
