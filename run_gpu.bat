@echo off
REM Algai Training Helper (GPU FORCED)
cd /d "%~dp0"

echo ==========================================
echo      Algai Training (GPU ONLY)
echo ==========================================
echo.
echo NOTE: This script forces the use of venv_gpu.
echo.

set "GPU_PYTHON=venv_gpu\Scripts\python.exe"

if not exist "%GPU_PYTHON%" (
    echo CRITICAL ERROR: Could not find %GPU_PYTHON%
    echo Please ensure you created the environment correctly.
    pause
    exit /b
)

echo Using Python: %GPU_PYTHON%
echo.
echo 0. ** DOWNLOAD DATA ** (Run this first!)
echo 1. Train Global Base Model (All Data)
echo 2. Fine-Tune Specialist Model (Single Stock)
echo 3. Start THE OVERLORD (Infinite Ensemble Loop)
echo 4. Start THE JUDGE (Meta-Model Stacking)
echo 5. Run PREDICTION TEST (Ensemble Verdict)
echo 6. Run PRECISION AUDIT (10,000 Sample Test)
echo 7. Exit
echo.
set /p choice="Enter selection (0-7): "

if "%choice%"=="0" goto download_mode
if "%choice%"=="1" goto global_mode
if "%choice%"=="2" goto finetune_ask
if "%choice%"=="3" goto overlord_mode
if "%choice%"=="4" goto judge_mode
if "%choice%"=="5" goto predict_mode
if "%choice%"=="6" goto audit_mode
if "%choice%"=="7" exit /b

:download_mode
echo.
echo Starting Data Download (10 Years)...
set PYTHONPATH=%PYTHONPATH%;%CD%\backend
"%GPU_PYTHON%" backend\scripts\download_alpaca.py
echo.
pause
exit /b

:global_mode
echo Starting Global Base Training...
set PYTHONPATH=%PYTHONPATH%;%CD%\backend
"%GPU_PYTHON%" backend\scripts\train_global.py
echo.
pause
exit /b

:finetune_ask
echo.
set /p ticker="Enter Stock Ticker (e.g. AAPL): "
echo Starting Fine-Tuning for %ticker%...
set PYTHONPATH=%PYTHONPATH%;%CD%\backend
"%GPU_PYTHON%" backend\scripts\train_finetune.py %ticker%
echo.
pause
exit /b

:overlord_mode
echo.
echo Waking the Overlord...
echo (Press Ctrl+C to stop the infinite loop)
set PYTHONPATH=%PYTHONPATH%;%CD%\backend
"%GPU_PYTHON%" backend\scripts\train_ensemble.py
echo.
pause
exit /b

:judge_mode
echo.
echo Summoning the Judge...
set PYTHONPATH=%PYTHONPATH%;%CD%\backend
"%GPU_PYTHON%" backend\scripts\train_stacking.py
echo.
pause
exit /b

:predict_mode
echo.
set /p ticker="Enter Ticker to Predict (e.g. NVDA): "
echo Querying the Ensemble for %ticker%...
set PYTHONPATH=%PYTHONPATH%;%CD%\backend
"%GPU_PYTHON%" backend\scripts\predict_ensemble.py %ticker%
echo.
pause
exit /b

:audit_mode
echo.
echo Starting The Precision Audit (Stress Test)...
set PYTHONPATH=%PYTHONPATH%;%CD%\backend
"%GPU_PYTHON%" backend\scripts\backtest_ensemble.py 10000
echo.
pause
exit /b
