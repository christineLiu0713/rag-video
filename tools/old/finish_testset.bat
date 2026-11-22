@echo off
rem 1) re-run test_video with the corrected watermark filter
rem 2) evaluate ntu_zhuangzi_01 with its new questions
rem 3) rebuild the full report
cd /d "%~dp0.."
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set LOG=results\logs\finish_testset.txt
venv\Scripts\python.exe pipeline.py --testset evaluation\testset.json --only test_video > %LOG% 2>&1
venv\Scripts\python.exe evaluation\eval_search.py evaluation\questions\ntu_zhuangzi_01.json results\ntu_zhuangzi_01\chroma_db >> %LOG% 2>&1
venv\Scripts\python.exe pipeline.py --testset evaluation\testset.json --report-only >> %LOG% 2>&1
echo FINISH_TESTSET_DONE >> %LOG%
type %LOG%
echo.
echo FINISHED - you can close this window.
pause
