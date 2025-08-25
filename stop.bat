@echo off
title Stop AutoClean Servers
color 0C

echo ========================================
echo    Stopping AutoClean Servers
echo ========================================
echo.

echo Stopping backend server (FastAPI/uvicorn)...
taskkill /f /im python.exe /fi "WINDOWTITLE eq AutoClean Backend*" >nul 2>&1
taskkill /f /im uvicorn.exe >nul 2>&1

echo Stopping frontend server...
taskkill /f /im python.exe /fi "WINDOWTITLE eq AutoClean Frontend*" >nul 2>&1

:: More aggressive cleanup - kill Python processes on ports 8000 and 3000
for /f "tokens=5" %%a in ('netstat -aon ^| find ":8000" ^| find "LISTENING"') do (
    echo Killing process %%a on port 8000...
    taskkill /f /pid %%a >nul 2>&1
)

for /f "tokens=5" %%a in ('netstat -aon ^| find ":3000" ^| find "LISTENING"') do (
    echo Killing process %%a on port 3000...
    taskkill /f /pid %%a >nul 2>&1
)

echo.
echo Servers stopped!
echo.
pause