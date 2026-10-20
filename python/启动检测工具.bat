@echo off
chcp 65001 >nul
cd /d "%~dp0"
title PatchCore Inspection Tool

echo ============================================================
echo   PatchCore Inspection Tool
echo ============================================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Python environment not found:
    echo         .venv\Scripts\python.exe
    echo.
    echo         Please run setup_env.sh to rebuild it.
    echo.
    pause
    exit /b 1
)

if not exist "inspect_tool.py" (
    echo [ERROR] inspect_tool.py not found in this folder.
    echo.
    pause
    exit /b 1
)

rem Server binds the port first, then opens the browser.
rem This avoids the browser loading before the server is ready.
".venv\Scripts\python.exe" -u inspect_tool.py --port 8765 --open

echo.
echo ============================================================
echo   Service stopped.
echo ============================================================
pause
