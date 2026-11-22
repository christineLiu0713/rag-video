@echo off
rem 1) re-join the MIT probability video correctly  2) run the whole test set
cd /d "%~dp0.."
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
if not exist results\logs mkdir results\logs
venv\Scripts\python.exe tools\fix_prob_video.py > results\logs\fix_prob_video.txt 2>&1
type results\logs\fix_prob_video.txt
findstr /C:"OK:" results\logs\fix_prob_video.txt >nul || (echo FIX FAILED - stopping. & pause & exit /b 1)
venv\Scripts\python.exe pipeline.py --testset evaluation\testset.json > results\logs\testset_run.txt 2>&1
type results\logs\testset_run.txt
echo TESTSET_FINISHED >> results\logs\testset_run.txt
echo.
echo FINISHED - you can close this window.
pause
