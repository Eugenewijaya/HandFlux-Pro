@echo off
title Foto Kita Blur Camera Launcher
cd /d "%~dp0"
echo Launching Foto Kita Blur Camera...
".venv\Scripts\python.exe" foto_kita_blur.py
if errorlevel 1 (
    echo.
    echo Press any key to exit...
    pause >nul
)
