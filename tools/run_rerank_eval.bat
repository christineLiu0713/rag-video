@echo off
rem 1) install OpenCC (simplified -> traditional for keyword search)
rem 2) build the video library (library/chroma_db) from results/*
rem 3) retrieval evaluation with the re-ranker for every test video
rem    (first run downloads BAAI/bge-reranker-v2-m3, about 2.3 GB)
rem 4) the same questions searched over the WHOLE library (harder)
rem 5) rebuild results/testset_report.md
cd /d "%~dp0.."
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set LOG=results\logs\rerank_eval.txt
echo ===== pip install opencc > %LOG%
venv\Scripts\python.exe -m pip install opencc-python-reimplemented >> %LOG% 2>&1
echo ===== build_library >> %LOG%
venv\Scripts\python.exe build_library.py >> %LOG% 2>&1
for %%V in (test_video mit_prob_l01 mit_1806_l01 lecture2 ntu_cs_12 ntu_zhuangzi_01) do (
  echo ===== eval %%V >> %LOG%
  venv\Scripts\python.exe evaluation\eval_search.py evaluation\questions\%%V.json results\%%V\chroma_db >> %LOG% 2>&1
)
for %%V in (test_video mit_prob_l01 mit_1806_l01 lecture2 ntu_cs_12 ntu_zhuangzi_01) do (
  echo ===== library eval %%V >> %LOG%
  venv\Scripts\python.exe evaluation\eval_search.py evaluation\questions\%%V.json library\chroma_db >> %LOG% 2>&1
)
venv\Scripts\python.exe pipeline.py --testset evaluation\testset.json --report-only >> %LOG% 2>&1
echo RERANK_EVAL_DONE >> %LOG%
echo.
echo FINISHED - you can close this window. Log: %LOG%
pause
