@echo off
setlocal EnableExtensions
cd /d "%~dp0"

title GPU Cloth Simulation Lab - One-click CUDA launcher
set "VENV_PYTHON=%CD%\.venv\Scripts\python.exe"
set "BUNDLED_PYTHON=%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"

echo ============================================================
echo   GPU Cloth Simulation Lab - Python / Taichi CUDA
echo ============================================================
echo.

if not exist "%VENV_PYTHON%" (
    echo [1/4] Creating the Python virtual environment...
    if exist "%BUNDLED_PYTHON%" (
        "%BUNDLED_PYTHON%" -m venv ".venv"
    ) else (
        py -3.12 -m venv ".venv" 2>nul
        if errorlevel 1 python -m venv ".venv"
    )
    if errorlevel 1 goto :python_error
) else (
    echo [1/4] Python virtual environment is ready.
)

echo [2/4] Checking Taichi and NumPy dependencies...
"%VENV_PYTHON%" -c "import taichi, numpy; assert taichi.__version__ == (1, 7, 4); assert numpy.__version__ == '1.26.4'" >nul 2>nul
if errorlevel 1 (
    echo       Installing dependencies from requirements.txt...
    "%VENV_PYTHON%" -m pip install --disable-pip-version-check -r requirements.txt
    if errorlevel 1 goto :dependency_error
) else (
    echo       Dependency versions are correct.
)

echo [3/4] Checking the NVIDIA GPU and CUDA driver...
nvidia-smi >nul 2>nul
if errorlevel 1 goto :gpu_error
for /f "tokens=*" %%G in ('nvidia-smi --query-gpu=name --format=csv^,noheader 2^>nul') do echo       GPU: %%G

echo [4/4] Running the 30-step CUDA collision smoke test...
"%VENV_PYTHON%" "gpu_cloth_demo.py" --arch cuda --headless --steps 30 --cloth cape --algorithm pbd
if errorlevel 1 goto :test_error

echo.
echo ============================================================
echo   PASS: CUDA, cloth solver, and Bunny collision are ready.
echo ============================================================
echo.

if /I "%~1"=="--test-only" goto :success

echo Starting the real-time GPU demo...
echo Tip: hold the right mouse button and drag to orbit around the Bunny.
echo.
"%VENV_PYTHON%" "gpu_cloth_demo.py" --arch cuda
if errorlevel 1 goto :runtime_error
goto :success

:python_error
echo.
echo [ERROR] Python 3.12 was not found, or .venv could not be created.
echo Install 64-bit Python 3.12 and run this script again.
goto :failed

:dependency_error
echo.
echo [ERROR] GPU dependencies could not be installed. Check the network.
goto :failed

:gpu_error
echo.
echo [ERROR] NVIDIA driver was not detected. Check nvidia-smi.
goto :failed

:test_error
echo.
echo [ERROR] CUDA smoke test failed. The real-time demo was not started.
goto :failed

:runtime_error
echo.
echo [ERROR] The real-time GPU demo exited unexpectedly.
goto :failed

:failed
echo.
pause
exit /b 1

:success
echo.
echo Done.
if /I not "%~1"=="--test-only" pause
exit /b 0
