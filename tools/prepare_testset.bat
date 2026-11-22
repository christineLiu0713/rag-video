@echo off
rem One-time: move the downloaded test videos into the project and build evaluation\testset.json
cd /d "%~dp0.."
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
venv\Scripts\python.exe tools\prepare_testset.py > results\logs\prepare_testset.txt 2>&1
type results\logs\prepare_testset.txt
echo.
echo FINISHED - you can close this window.
pause
