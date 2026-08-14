@echo off
setlocal EnableExtensions
cd /d "%~dp0"
set "WEB_PYTHON=%CD%\.venv\Scripts\python.exe"
set "WEB_URL=http://localhost:8080"

if not exist "%WEB_PYTHON%" (
    where py >nul 2>nul
    if not errorlevel 1 set "WEB_PYTHON=py"
)
if not exist "%WEB_PYTHON%" if /I not "%WEB_PYTHON%"=="py" goto :python_error

echo ============================================================
echo   Python GPU Cloth - synchronized WebGL preview
echo   %WEB_URL%
echo ============================================================

powershell.exe -NoProfile -Command "try { $r = Invoke-WebRequest -UseBasicParsing '%WEB_URL%/index.html' -TimeoutSec 2; if ($r.StatusCode -eq 200) { exit 0 } } catch {}; exit 1" >nul 2>nul
if not errorlevel 1 (
    echo Web simulation is already running.
    if /I "%~1"=="--test-only" exit /b 0
    echo Opening the existing page...
    start "" "%WEB_URL%"
    exit /b 0
)

if /I "%~1"=="--test-only" (
    echo Web environment and files are ready.
    exit /b 0
)

echo Close this window or press Ctrl+C to stop the web server.
echo.

start "" powershell.exe -NoProfile -WindowStyle Hidden -Command "Start-Sleep -Milliseconds 900; Start-Process '%WEB_URL%'"
"%WEB_PYTHON%" -m http.server 8080 --bind 127.0.0.1
exit /b %ERRORLEVEL%

:python_error
echo [ERROR] Python was not found. Run start_gpu.bat once first.
pause
exit /b 1
