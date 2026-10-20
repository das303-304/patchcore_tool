@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Python venv not found: .venv\Scripts\python.exe
    echo         Please run setup_env.sh first.
    pause
    exit /b 1
)

echo Starting PatchCore model builder ...
".venv\Scripts\python.exe" build_tool.py --port 8766 --open
pause
