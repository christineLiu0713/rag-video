import json, sys
chunks=[json.loads(l) for l in open(sys.argv[1],encoding='utf-8')]
# (question, anchor seconds where the answer is said, type)
Q=[
("assembler 是做什麼的", [422], "keyword"),
("compiler 和 interpreter 有什麼不同", [542], "keyword"),
("什麼是 portability 可攜性", [797, 812], "keyword"),
("第四代程式語言 SQL", [120], "keyword"),
("第一代 Machine Instruction", [90], "keyword"),
("declarative 和 prologue", [1326, 1345], "keyword"),
("Lisp 的括號 match", [1843], "keyword"),
("encapsulation 封裝", [2185], "keyword"),
("什麼是 l value", [2617, 2643], "keyword"),
("go to 指令", [2778, 2828], "keyword"),

("為什麼要發明組合語言", [100], "paraphrase"),
("翻譯一次之後就可以一直重複執行的方式", [542], "paraphrase"),
("為什麼驅動程式很難直接搬到別台電腦", [906], "paraphrase"),
("把整個程式看成一個函數的寫法", [1281, 1301], "paraphrase"),
("只要把問題描述清楚電腦就會自己找答案", [1345], "paraphrase"),
("用 n 乘上前一項來定義階乘", [1421], "paraphrase"),
("不准別人直接修改我類別裡面的資料", [2153, 2179], "paraphrase"),
("同一個函式名稱在不同物件會執行不同的版本", [2232], "paraphrase"),
("為什麼老師不建議用跳躍指令", [2828], "paraphrase"),
("用很小的空間存機率值的例子", [2054], "paraphrase"),

("what is the difference between a compiler and an interpreter", [542], "english"),
("why is Java so portable", [630, 995], "english"),
("what are the four programming paradigms", [1122, 1502], "english"),
("how to compute an average in functional programming", [1779], "english"),
("why should we avoid goto statements", [2828], "english"),
]
out=[]
for q,times,t in Q:
    rel=[c["id"] for c in chunks if any(c["start"]-1<=x<=c["end"]+1 for x in times)]
    assert rel, (q,times)
    out.append({"q":q,"relevant":rel,"type":t})
    print(t, q, rel)
json.dump({"video":"ntu_cs_12","note":"台大計算機概論第12講前50分鐘。relevant 由答案所在的逐字稿時間點對應到片段。type: keyword=用影片裡出現的詞；paraphrase=換句話說；english=用英文問中文影片",
 "questions":out}, open(sys.argv[2],'w',encoding='utf-8'), ensure_ascii=False, indent=1)
