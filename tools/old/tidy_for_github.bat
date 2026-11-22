@echo off
rem Move one-off batch files used during testing into tools\old\ so tools\ only keeps the main ones.
cd /d "%~dp0.."
for %%F in (finish_ntu_adaptive.bat finish_testset.bat fix_and_run_testset.bat fix_prob_video.py rechunk_and_eval_ntu.bat reeval_mit.bat rerun_ocr_ntu_adaptive.bat rerun_test_video.bat run_eval.bat) do (
  if exist %%F move /y %%F old\ >nul && echo moved %%F
)
echo.
echo Done. tools\ now keeps: prepare_testset, run_pipeline, run_testset, run_ocr_benchmark, run_rerank_eval, run_answer_eval, export_dataset
pause
