@echo off
title HandFlux Pro Launcher
cd /d "%~dp0"
echo Launching HandFlux Pro...
".venv\Scripts\python.exe" HandFlux.py
if errorlevel 1 (
    echo.
    echo Press any key to exit...
    pause >nul
)
