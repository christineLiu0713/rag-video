@echo off
rem One-off: compare old chunks (551) vs new chunks (min_chunk_sec) on the Chinese video
cd /d "%~dp0.."
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set LOG=results\logs\ntu_rechunk.txt
echo START %time% > %LOG%
echo [1] eval old chunks >> %LOG%
venv\Scripts\python.exe evaluation\eval_search.py evaluation\questions\ntu_cs_12_before.json output\chroma_db > results\logs\eval_ntu_before.txt 2>&1
copy /Y output\eval_result.md results\logs\eval_ntu_before.md >> %LOG%
echo [2] rebuild chunks %time% >> %LOG%
venv\Scripts\python.exe build_chunks.py ntu_cs_12 >> %LOG% 2>&1
venv\Scripts\python.exe build_db.py >> %LOG% 2>&1
echo [3] remap questions %time% >> %LOG%
venv\Scripts\python.exe evaluation\make_questions_ntu_cs_12.py output\chunks.jsonl evaluation\questions\ntu_cs_12.json >> %LOG% 2>&1
echo [4] eval new chunks %time% >> %LOG%
venv\Scripts\python.exe evaluation\eval_search.py evaluation\questions\ntu_cs_12.json output\chroma_db > results\logs\eval_ntu_after.txt 2>&1
echo RECHUNK_FINISHED %time% >> %LOG%
