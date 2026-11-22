@echo off
rem Check whether this computer can use the NVIDIA GPU for this project.
rem The result is also saved to results\logs\gpu_check.txt (send this file to the developer).
cd /d "%~dp0.."
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
if exist venv\Scripts\python.exe (
  venv\Scripts\python.exe check_gpu.py
) else (
  python check_gpu.py
)
echo.
pause
