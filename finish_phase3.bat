@echo off
echo ===================================================
echo     ALGAI PHASE 3: THE JUDGE (GPU ACCELERATED)
echo ===================================================
echo.
echo NOTE: Ensure Phase 2 is finished before running this!
echo.
echo 1. Generating Judge Data (Mistakes Dataset) on GPU...
venv_gpu\Scripts\python.exe backend\scripts\generate_judge_data.py
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Data Generation Failed!
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo 2. Training The Judge (XGBoost Meta-Model)...
venv_gpu\Scripts\python.exe backend\scripts\train_judge_advanced.py
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Judge Training Failed!
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo ===================================================
echo         PHASE 3 COMPLETE. SYSTEM READY.
echo ===================================================
echo Now ping the Overlord to begin Phase 4.
pause
