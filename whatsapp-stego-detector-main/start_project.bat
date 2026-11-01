@echo off
title WhatsApp Stego Detector Launcher
echo ======================================================
echo    Starting WhatsApp Steganography Detector...
echo ======================================================
echo.

echo [1/2] Starting Backend Server (Port 5000)...
start "Stego Backend (Flask)" cmd /k "cd /d %~dp0backend && python app.py"

timeout /t 2 /nobreak >nul

echo [2/2] Starting Frontend Server (Port 8000)...
start "Stego Frontend (HTTP)" cmd /k "cd /d %~dp0 && python -m http.server 8000 --directory demo_chatbox"

timeout /t 1 /nobreak >nul

echo [3/3] Opening WhatsApp Web Interface in your default browser...
start http://localhost:8000/

echo.
echo ======================================================
echo  Project is UP and RUNNING!
echo  - Frontend: http://localhost:8000/
echo  - Backend:  http://127.0.0.1:5000/
echo ======================================================
echo You can keep this window open or close it.
pause
