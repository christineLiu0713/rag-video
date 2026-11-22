@echo off
rem Compare vector / bm25 / hybrid retrieval. One line = question file + database.
cd /d "%~dp0.."
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
if not exist results\logs mkdir results\logs
set LOG=results\logs\eval.txt
echo Running search evaluation, please wait...
echo EVAL %date% %time% > %LOG%
call :one test_video
call :one lecture2
call :one ntu_cs_12
echo EVAL_FINISHED >> %LOG%
exit /b

:one
rem use results\<name> if it was archived, otherwise output\ (the latest video)
if not exist evaluation\questions\%1.json exit /b
set DB=output\chroma_db
if exist results\%1\chroma_db set DB=results\%1\chroma_db
venv\Scripts\python.exe evaluation\eval_search.py evaluation\questions\%1.json %DB% >> %LOG% 2>&1
exit /b
