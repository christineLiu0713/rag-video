"""一次性：重新接 MIT 機率課影片（上一版接成只有封面圖片＋聲音，沒有影片畫面）。"""
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SEGS = ROOT / "_to_delete" / "downloads_used"
target = ROOT / "videos" / "mit_prob_l01.mp4"

segs = [SEGS / f"prob_MITRES6_012S18_L01-{i:02d}_300k.mp4" for i in range(1, 11)]
missing = [p.name for p in segs if not p.exists()]
if missing:
    raise SystemExit(f"segments not found: {missing}")
if target.exists():
    shutil.move(str(target), str(ROOT / "_to_delete" / "mit_prob_l01_without_video.mp4"))
lst = ROOT / "_to_delete" / "concat_list.txt"
lst.write_text("".join(f"file '{p.as_posix()}'\n" for p in segs), encoding="utf-8")
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
                "-map", "0:V:0", "-map", "0:a:0", "-c", "copy", str(target)], check=True)
out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_name,width,height",
                      "-of", "compact", str(target)], capture_output=True, text=True).stdout
print(out)
if "h264" not in out:
    raise SystemExit("joined file still has no video stream")
print(f"OK: {target.name} {target.stat().st_size / 2**20:.0f} MB")
