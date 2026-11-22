@echo off
rem OCR speed test (needs screenshots in output\frames\)
cd /d "%~dp0.."
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
if not exist results\logs mkdir results\logs
echo Running OCR benchmark, please wait...
py -3.11 ocr_worker.py --benchmark 8 > results\logs\ocr_benchmark.txt 2>&1
echo BENCHMARK_FINISHED >> results\logs\ocr_benchmark.txt
