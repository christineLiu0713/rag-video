@echo off
rem One-off: re-run OCR on the Chinese video with adaptive sampling + token slide matching,
rem then rebuild chunks / database and evaluate. Old results are backed up first.
cd /d "%~dp0.."
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set LOG=results\logs\ntu_adaptive.txt
echo START %time% > %LOG%
copy /Y output\ocr.json results\logs\ntu_ocr_before_adaptive.json >nul
copy /Y output\chunks.jsonl results\logs\ntu_chunks_before_adaptive.jsonl >nul
echo [STEP ocr] %time% >> %LOG%
venv\Scripts\python.exe extract.py videos\ntu_cs_12_first50min.mp4 --ocr-only --lang zh >> %LOG% 2>&1
echo [STEP build_chunks] %time% >> %LOG%
venv\Scripts\python.exe build_chunks.py ntu_cs_12 >> %LOG% 2>&1
echo [STEP build_db] %time% >> %LOG%
venv\Scripts\python.exe build_db.py >> %LOG% 2>&1
echo [STEP eval] %time% >> %LOG%
venv\Scripts\python.exe evaluation\make_questions_ntu_cs_12.py output\chunks.jsonl evaluation\questions\ntu_cs_12.json >> %LOG% 2>&1
venv\Scripts\python.exe evaluation\eval_search.py evaluation\questions\ntu_cs_12.json output\chroma_db > results\logs\eval_ntu_adaptive.txt 2>&1
echo [END] %time% >> %LOG%
echo ADAPTIVE_FINISHED >> %LOG%
