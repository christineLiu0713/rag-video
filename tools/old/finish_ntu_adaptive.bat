@echo off
rem Finish the adaptive test: build database + evaluation (OCR and chunks are already done)
cd /d "%~dp0.."
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set LOG=results\logs\ntu_adaptive_finish.txt
echo [STEP build_db] %time% > %LOG%
venv\Scripts\python.exe build_db.py >> %LOG% 2>&1
echo [STEP eval] %time% >> %LOG%
venv\Scripts\python.exe evaluation\make_questions_ntu_cs_12.py output\chunks.jsonl evaluation\questions\ntu_cs_12.json >> %LOG% 2>&1
venv\Scripts\python.exe evaluation\eval_search.py evaluation\questions\ntu_cs_12.json output\chroma_db > results\logs\eval_ntu_adaptive.txt 2>&1
echo [END] %time% >> %LOG%
echo FINISHED - you can close this window >> %LOG%
echo.
echo FINISHED - you can close this window.
pause
