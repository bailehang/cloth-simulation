@echo off
cd /d "%~dp0"
echo Starting Cloth Simulation Lab...
echo Open http://localhost:8080 in your browser
echo.
python -m http.server 8080
pause
