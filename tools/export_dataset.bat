@echo off
rem Export all processed videos to exports\lecture_dataset.json (for exam generation and other features)
cd /d "%~dp0.."
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
venv\Scripts\python.exe export_dataset.py
echo.
pause
