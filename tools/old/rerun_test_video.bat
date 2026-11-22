@echo off
rem Re-run test_video with the corrected (corner-only) watermark filter, then rebuild the full report
cd /d "%~dp0.."
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
venv\Scripts\python.exe pipeline.py --testset evaluation\testset.json --only test_video > results\logs\rerun_test_video.txt 2>&1
venv\Scripts\python.exe pipeline.py --testset evaluation\testset.json --report-only >> results\logs\rerun_test_video.txt 2>&1
echo RERUN_FINISHED >> results\logs\rerun_test_video.txt
type results\logs\rerun_test_video.txt
echo.
echo FINISHED - you can close this window.
pause
