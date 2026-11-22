"""
Step 4: 問問題 -> 找出影片片段 -> Gemini 寫出簡短回答，附上影片名稱與時間

完整流程在 rag.py：
    檢索    : BM25 關鍵字 + BGE-M3 向量（RRF 合併）
    重新排序 : bge-reranker-v2-m3 把問題和片段一起讀，排出最相關的 5 段
    生成    : Gemini 只讀這些片段，找不到答案時會說找不到（不亂猜）

準備（只要一次）:
    pip install google-genai
    金鑰：在專案資料夾建立 gemini_key.txt，裡面貼上金鑰（https://aistudio.google.com/apikey）
          或在 CMD 先執行  set GEMINI_API_KEY=你的金鑰

Usage:
    python answer.py                          # 搜尋影片庫裡全部影片（library/），沒有就用 output/
    python answer.py --video mit_prob_l01     # 只問某支影片
    python answer.py --db results/lecture2/chroma_db
    python answer.py -q "what is the union bound"   # 只問一題就結束
"""
import argparse
import json
import sys

from rag import VideoRAG, format_result

parser = argparse.ArgumentParser()
parser.add_argument("--db", help="Chroma folder (default: library/chroma_db, else output/chroma_db)")
parser.add_argument("--video", help="only these videos, comma-separated")
parser.add_argument("-q", "--question", help="ask one question and exit")
parser.add_argument("--json", action="store_true", help="print the full result as JSON")
args = parser.parse_args()

try:
    rag = VideoRAG(args.db)
except RuntimeError as e:
    sys.exit(str(e))

videos = args.video.split(",") if args.video else None
print(f"Database: {rag.db_dir}  ({len(rag.videos)} videos: {', '.join(rag.videos)})")
print(f"Gemini model: {rag.s['gemini_model']}   re-ranker: "
      f"{'on' if rag.retriever.reranker else 'off'}")


def run(question):
    r = rag.ask(question, videos)
    if args.json:
        print(json.dumps(r, ensure_ascii=False, indent=1))
        return
    print("\n" + format_result(r, show_video=len(rag.videos) > 1))
    if r["error"]:
        print(f"\n({r['error']})")
    print(f"\n[檢索 {r['retrieve_ms']} ms, Gemini {r['llm_ms']} ms, 讀了 {len(r['parts'])} 段]")


if args.question:
    run(args.question)
    sys.exit()

while True:
    try:
        question = input("\nQuestion (empty to quit): ").strip()
    except EOFError:
        break
    if not question:
        break
    run(question)
