@echo off
title Naruto Jutsu Camera Launcher
cd /d "%~dp0"
echo Launching Naruto Jutsu Camera...
".venv\Scripts\python.exe" naruto_jutsu.py
if errorlevel 1 (
    echo.
    echo Press any key to exit...
    pause >nul
)
