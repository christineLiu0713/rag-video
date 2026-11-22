# 檢索方法比較（vector / bm25 / hybrid）

- 資料庫：results\ntu_cs_12\chroma_db　題目檔：evaluation\questions\ntu_cs_12.json
- all = 每種方法排完所有片段再合併；top10 = 各取前 10 名再合併（系統實際設定）；w = bm25_weight
- 題目數：25　片段數：126　向量模型：BAAI/bge-m3（cuda）
- RRF：k=60、vector_weight=1.0
- 重新排序模型：BAAI/bge-reranker-v2-m3

## ALL（25 題）

| 方法 | Hit@1 | Hit@3 | Hit@5 | MRR | 平均時間 (ms) |
|---|---|---|---|---|---|
| vector | 60% | 92% | 96% | 0.765 | 51 |
| bm25 | 52% | 76% | 84% | 0.660 | 46 |
| hybrid (all, w=1) | 60% | 92% | 100% | 0.758 | 43 |
| hybrid (top10, w=1) | 60% | 92% | 96% | 0.748 | 39 |
| hybrid (top10, w=0.5) | 60% | 92% | 96% | 0.760 | 38 |
| hybrid (top10, w=0.3) | 60% | 92% | 96% | 0.760 | 39 |
| vector + rerank | 60% | 100% | 100% | 0.780 | 1807 |
| hybrid + rerank | 56% | 100% | 100% | 0.760 | 1532 |
| hybrid20 + rerank | 56% | 100% | 100% | 0.760 | 2939 |

## keyword（10 題）

| 方法 | Hit@1 | Hit@3 | Hit@5 | MRR | 平均時間 (ms) |
|---|---|---|---|---|---|
| vector | 80% | 100% | 100% | 0.883 | 62 |
| bm25 | 70% | 90% | 90% | 0.796 | 44 |
| hybrid (all, w=1) | 80% | 100% | 100% | 0.883 | 42 |
| hybrid (top10, w=1) | 80% | 100% | 100% | 0.883 | 39 |
| hybrid (top10, w=0.5) | 70% | 100% | 100% | 0.850 | 36 |
| hybrid (top10, w=0.3) | 70% | 100% | 100% | 0.833 | 37 |
| vector + rerank | 80% | 100% | 100% | 0.900 | 1709 |
| hybrid + rerank | 80% | 100% | 100% | 0.900 | 1445 |
| hybrid20 + rerank | 80% | 100% | 100% | 0.900 | 2704 |

## paraphrase（10 題）

| 方法 | Hit@1 | Hit@3 | Hit@5 | MRR | 平均時間 (ms) |
|---|---|---|---|---|---|
| vector | 40% | 80% | 90% | 0.628 | 44 |
| bm25 | 60% | 70% | 80% | 0.680 | 50 |
| hybrid (all, w=1) | 40% | 90% | 100% | 0.637 | 46 |
| hybrid (top10, w=1) | 40% | 80% | 90% | 0.603 | 41 |
| hybrid (top10, w=0.5) | 50% | 80% | 90% | 0.651 | 41 |
| hybrid (top10, w=0.3) | 50% | 80% | 90% | 0.667 | 41 |
| vector + rerank | 50% | 100% | 100% | 0.700 | 1924 |
| hybrid + rerank | 40% | 100% | 100% | 0.650 | 1647 |
| hybrid20 + rerank | 40% | 100% | 100% | 0.650 | 3149 |

## english（5 題）

| 方法 | Hit@1 | Hit@3 | Hit@5 | MRR | 平均時間 (ms) |
|---|---|---|---|---|---|
| vector | 60% | 100% | 100% | 0.800 | 42 |
| bm25 | 0% | 60% | 80% | 0.351 | 40 |
| hybrid (all, w=1) | 60% | 80% | 100% | 0.750 | 41 |
| hybrid (top10, w=1) | 60% | 100% | 100% | 0.767 | 37 |
| hybrid (top10, w=0.5) | 60% | 100% | 100% | 0.800 | 37 |
| hybrid (top10, w=0.3) | 60% | 100% | 100% | 0.800 | 39 |
| vector + rerank | 40% | 100% | 100% | 0.700 | 1771 |
| hybrid + rerank | 40% | 100% | 100% | 0.700 | 1478 |
| hybrid20 + rerank | 40% | 100% | 100% | 0.700 | 2990 |

## 每題正確片段的名次（越小越好，- = 沒找到）

| 類型 | 問題 | vector | bm25 | hybrid (all, w=1) | hybrid (top10, w=1) | hybrid (top10, w=0.5) | hybrid (top10, w=0.3) | vector + rerank | hybrid + rerank | hybrid20 + rerank |
|---|---|---|---|---|---|---|---|---|---|---|
| keyword | assembler 是做什麼的 | 1 | 3 | 1 | 1 | 1 | 1 | 2 | 2 | 2 |
| keyword | compiler 和 interpreter 有什麼不同 | 1 | 8 | 3 | 3 | 2 | 2 | 2 | 2 | 2 |
| keyword | 什麼是 portability 可攜性 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| keyword | 第四代程式語言 SQL | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| keyword | 第一代 Machine Instruction | 3 | 1 | 2 | 2 | 2 | 3 | 1 | 1 | 1 |
| keyword | declarative 和 prologue | 1 | 2 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| keyword | Lisp 的括號 match | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| keyword | encapsulation 封裝 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| keyword | 什麼是 l value | 2 | 1 | 1 | 1 | 2 | 2 | 1 | 1 | 1 |
| keyword | go to 指令 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| paraphrase | 為什麼要發明組合語言 | 2 | 3 | 3 | 3 | 3 | 2 | 3 | 3 | 3 |
| paraphrase | 翻譯一次之後就可以一直重複執行的方式 | 2 | 1 | 2 | 2 | 2 | 2 | 1 | 1 | 1 |
| paraphrase | 為什麼驅動程式很難直接搬到別台電腦 | 2 | 1 | 1 | 1 | 1 | 1 | 2 | 2 | 2 |
| paraphrase | 把整個程式看成一個函數的寫法 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| paraphrase | 只要把問題描述清楚電腦就會自己找答案 | 5 | 1 | 3 | 3 | 3 | 3 | 2 | 2 | 2 |
| paraphrase | 用 n 乘上前一項來定義階乘 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| paraphrase | 不准別人直接修改我類別裡面的資料 | 1 | 4 | 1 | 1 | 1 | 1 | 3 | 3 | 3 |
| paraphrase | 同一個函式名稱在不同物件會執行不同的版本 | 12 | 1 | 5 | 6 | 11 | 11 | 1 | 1 | 1 |
| paraphrase | 為什麼老師不建議用跳躍指令 | 1 | 7 | 2 | 2 | 1 | 1 | 3 | 3 | 3 |
| paraphrase | 用很小的空間存機率值的例子 | 2 | 14 | 2 | 5 | 4 | 4 | 1 | 2 | 2 |
| english | what is the difference between a compiler and an interpreter | 1 | 2 | 1 | 1 | 1 | 1 | 2 | 2 | 2 |
| english | why is Java so portable | 1 | 2 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| english | what are the four programming paradigms | 2 | 19 | 4 | 3 | 2 | 2 | 2 | 2 | 2 |
| english | how to compute an average in functional programming | 1 | 5 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| english | why should we avoid goto statements | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 |
