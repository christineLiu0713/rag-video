@echo off
rem ============================================================
rem  Process ONE video. Edit the three lines below, then double-click.
rem    VIDEO = video path (relative to the project folder)
rem    NAME  = chunk id prefix / result folder name
rem    LANG  = en or zh
rem  The result stays in output\ (used by search.py / answer.py).
rem  The previous output\ is moved to results\<its name>\ first.
rem ============================================================
set VIDEO=videos\ntu_zhuangzi_01.mp4
set NAME=ntu_zhuangzi_01
set LANG=zh

cd /d "%~dp0.."
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
venv\Scripts\python.exe pipeline.py %VIDEO% --name %NAME% --lang %LANG%
echo.
echo FINISHED - you can close this window.
pause
