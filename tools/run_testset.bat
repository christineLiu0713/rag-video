@echo off
rem ============================================================
rem  Process ALL videos in evaluation\testset.json and evaluate them.
rem  Takes a few hours: run it when you do not need the computer.
rem  Result: results\testset_report.md (and .csv)
rem  To run only some videos, add e.g.  --only mit_prob_l01,mit_1806_l01
rem ============================================================
cd /d "%~dp0.."
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
venv\Scripts\python.exe pipeline.py --testset evaluation\testset.json %*
echo.
echo FINISHED - you can close this window.
pause
