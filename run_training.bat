@echo off
REM Algai Training Helper

set "PROJECT_ROOT=%~dp0"
set "GPU_PYTHON=%PROJECT_ROOT%venv_gpu\Scripts\python.exe"

echo DEBUG: PROJECT_ROOT=%PROJECT_ROOT%
echo DEBUG: Checking for %GPU_PYTHON%
if exist "%GPU_PYTHON%" (
    set "PY_EXE=%GPU_PYTHON%"
    echo Using GPU Environment (venv_gpu)
    echo Found at: %GPU_PYTHON%
) else (
    set "PY_EXE=%PROJECT_ROOT%backend\venv\Scripts\python.exe"
    echo Using Default Environment
    echo Could NOT find: %GPU_PYTHON%
)

if "%1"=="global" (
    echo Starting Global Base Training...
    set PYTHONPATH=%PYTHONPATH%;%PROJECT_ROOT%backend
    "%PY_EXE%" "%PROJECT_ROOT%backend\scripts\train_global.py"
    pause
    exit /b
)

if "%1"=="finetune" (
    if "%2"=="" (
        echo Please provide a symbol. Usage: run_training finetune AAPL
        exit /b
    )
    echo Starting Fine-Tuning for %2...
    set PYTHONPATH=%PYTHONPATH%;%PROJECT_ROOT%backend
    "%PY_EXE%" "%PROJECT_ROOT%backend\scripts\train_finetune.py" %2
    pause
    exit /b
)

:menu
REM cls
echo ==========================================
echo           Algai Training Helper
echo ==========================================
echo.
echo 1. Train Global Base Model (All Data)
echo 2. Fine-Tune Specialist Model (Single Stock)
echo 3. Exit
echo.
set /p choice="Enter selection (1-3): "

if "%choice%"=="1" goto global_mode
if "%choice%"=="2" goto finetune_ask
if "%choice%"=="3" exit /b

goto menu

:global_mode
echo Starting Global Base Training...
set PYTHONPATH=%PYTHONPATH%;%PROJECT_ROOT%backend
echo DEBUG: Executing with %PY_EXE%
"%PY_EXE%" "%PROJECT_ROOT%backend\scripts\train_global.py"
echo.
pause
exit /b

:finetune_ask
echo.
set /p ticker="Enter Stock Ticker (e.g. AAPL): "
echo Starting Fine-Tuning for %ticker%...
set PYTHONPATH=%PYTHONPATH%;%PROJECT_ROOT%backend
"%PY_EXE%" "%PROJECT_ROOT%backend\scripts\train_finetune.py" %ticker%
echo.
pause
exit /b
