@echo off
rem re-evaluate the two MIT videos with corrected anchors, then rebuild the report
cd /d "%~dp0.."
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set LOG=results\logs\reeval_mit.txt
venv\Scripts\python.exe evaluation\eval_search.py evaluation\questions\mit_prob_l01.json results\mit_prob_l01\chroma_db > %LOG% 2>&1
venv\Scripts\python.exe evaluation\eval_search.py evaluation\questions\mit_1806_l01.json results\mit_1806_l01\chroma_db >> %LOG% 2>&1
venv\Scripts\python.exe pipeline.py --testset evaluation\testset.json --report-only >> %LOG% 2>&1
echo REEVAL_DONE >> %LOG%
type %LOG%
echo.
echo FINISHED - you can close this window.
pause
