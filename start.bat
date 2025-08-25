@echo off
title AutoClean Pipeline Startup
color 0A

echo ========================================
echo    AutoClean Data Pipeline Startup
echo ========================================
echo.

:: Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH!
    echo Please install Python and try again.
    pause
    exit /b 1
)

:: Show Python version
echo Python version:
python --version
echo.

:: Install required packages
echo Installing/updating required packages...
pip install fastapi uvicorn pandas numpy scipy scikit-learn python-multipart
if errorlevel 1 (
    echo Warning: Some packages might not have installed correctly.
    echo You may need to install them manually.
)
echo.

:: Start backend server in a new window
echo Starting backend server...
start "AutoClean Backend" cmd /k "echo Starting FastAPI backend server... && python main.py"

:: Wait a moment for backend to start
timeout /t 3 /nobreak >nul

:: Start frontend server in a new window
echo Starting frontend server...
start "AutoClean Frontend" cmd /k "echo Starting frontend server... && python   frontend\start_frontend.py"

:: Wait a moment for frontend to start
timeout /t 2 /nobreak >nul

echo.
echo ========================================
echo  Servers are starting up!
echo ========================================
echo.
echo Backend API: http://localhost:8000
echo Frontend UI: http://localhost:3000
echo.
echo Your browser should open automatically.
echo If not, navigate to: http://localhost:3000
echo.
echo Press any key to close this window...
echo (The servers will continue running)
pause >nul