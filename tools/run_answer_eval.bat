@echo off
rem Evaluate the final answers (needs a Gemini key in gemini_key.txt or GEMINI_API_KEY).
rem Results: results\answer_eval\answer_eval.md and answers.csv
rem If the free Gemini limit is reached, just run this again later: it continues.
cd /d "%~dp0.."
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
if not exist gemini_key.txt if "%GEMINI_API_KEY%"=="" (
  echo Put your Gemini key in a file named gemini_key.txt in the project folder first.
  pause
  exit /b
)
set LOG=results\logs\answer_eval.txt
venv\Scripts\python.exe evaluation\eval_answer.py > %LOG% 2>&1
echo ANSWER_EVAL_DONE >> %LOG%
type results\answer_eval\answer_eval.md
echo.
echo FINISHED - you can close this window. Log: %LOG%
pause
