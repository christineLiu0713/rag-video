"""
把每支影片自己的資料庫合併成一個「影片庫」，讓問答可以一次搜尋所有影片。

    results/<影片>/chroma_db  (每支影片處理完的結果)
        ↓ 直接複製向量，不用重新算（幾秒鐘就好）
    library/chroma_db         (rag.py / answer.py / search.py 預設使用)
    library/videos.json       (影片清單：名稱、語言、片段數、資料夾)

Usage:
    python build_library.py                  # 合併 results/ 底下全部影片
    python build_library.py --only a,b       # 只放這幾支
    python build_library.py --include-output # 連 output/ 裡剛處理完的影片也放進去

pipeline.py 每處理完一支影片會自動執行這支。
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
LIBRARY = ROOT / "library"
COLLECTION = "lecture_chunks"
SKIP = {"_old", "logs"}


def find_sources(only=None, include_output=False):
    """回傳 [(影片名稱, 結果資料夾)]。"""
    sources = []
    if RESULTS.exists():
        for folder in sorted(RESULTS.iterdir()):
            if folder.name in SKIP or not (folder / "chroma_db").is_dir():
                continue
            name_file = folder / "video_name.txt"
            name = name_file.read_text(encoding="utf-8").strip() if name_file.exists() else folder.name
            sources.append((name, folder))
    out = ROOT / "output"
    if include_output and (out / "chroma_db").is_dir():
        name_file = out / "video_name.txt"
        name = name_file.read_text(encoding="utf-8").strip() if name_file.exists() else "output"
        sources = [s for s in sources if s[0] != name] + [(name, out)]
    if only:
        sources = [s for s in sources if s[0] in only]
    # 同名只留最後一個（例如 results/x 與 results/x_20261122_...）
    seen = {}
    for name, folder in sources:
        seen[name] = folder
    return list(seen.items())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", help="comma-separated video names")
    parser.add_argument("--include-output", action="store_true")
    args = parser.parse_args()

    import chromadb

    sources = find_sources(set(args.only.split(",")) if args.only else None, args.include_output)
    if not sources:
        sys.exit("No processed video found in results/. Run pipeline.py first.")

    final = LIBRARY / "chroma_db"
    if final.exists():
        try:
            shutil.rmtree(final)                     # 每次都重建：只是複製，很快
        except PermissionError:
            sys.exit("library/chroma_db is in use: close answer.py / search.py and run again.")
    final.parent.mkdir(exist_ok=True)
    target = chromadb.PersistentClient(path=str(final)).create_collection(COLLECTION)

    videos = []
    for name, folder in sources:
        try:
            col = chromadb.PersistentClient(path=str(folder / "chroma_db")).get_collection(COLLECTION)
        except Exception as e:
            print(f"  skip {name}: {e}")
            continue
        data = col.get(include=["embeddings", "documents", "metadatas"])
        metas = []
        for m in data["metadatas"]:
            m = dict(m)
            m["video"] = name
            m["result_dir"] = folder.relative_to(ROOT).as_posix()
            img = m.get("image") or ""
            if img.startswith("output/"):            # 處理完後 output/ 已經移到 results/<影片>/
                m["image"] = m["result_dir"] + img[len("output"):]
            metas.append(m)
        ids = [cid if cid.startswith(name) else f"{name}_{cid}" for cid in data["ids"]]
        embeddings = [list(map(float, e)) for e in data["embeddings"]]
        for i in range(0, len(ids), 500):            # Chroma 一次加入的數量有上限
            target.add(ids=ids[i:i + 500], embeddings=embeddings[i:i + 500],
                       documents=data["documents"][i:i + 500], metadatas=metas[i:i + 500])
        meta_file = folder / "meta.json"
        lang = json.load(open(meta_file, encoding="utf-8")).get("language") if meta_file.exists() else None
        videos.append({"name": name, "language": lang, "chunks": len(ids),
                       "result_dir": folder.relative_to(ROOT).as_posix()})
        print(f"  + {name:20s} {len(ids):4d} chunks  ({folder.relative_to(ROOT).as_posix()})")

    total = target.count()
    json.dump({"videos": videos}, open(LIBRARY / "videos.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"\nLibrary: {len(videos)} videos, {total} chunks -> library/chroma_db")


if __name__ == "__main__":
    main()
