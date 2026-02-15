@echo off
setlocal

echo ============================================
echo   Project Setup and Launch
echo ============================================
echo.

:: Check for Python
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH.
    echo Download it from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during install.
    pause
    exit /b 1
)

:: Check for Node.js / npm
where npm >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Node.js/npm is not installed or not in PATH.
    echo Download it from https://nodejs.org/
    echo Restart your terminal after installing.
    pause
    exit /b 1
)

:: ---- Python API Setup ----
echo [1/4] Setting up Python virtual environment...
cd hand_tracking

if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

call venv\Scripts\activate

echo [2/4] Installing Python dependencies...
echo        (This may take a while on first run due to large packages like torch)
if not exist requirements.txt (
    echo [ERROR] requirements.txt not found in hand_tracking folder!
    pause
    exit /b 1
)

:: Remove the broken 'dotenv' package line and install from cleaned file
findstr /v /b "dotenv==" requirements.txt > requirements_clean.txt
pip install -r requirements_clean.txt
del requirements_clean.txt

if %errorlevel% neq 0 (
    echo [ERROR] Failed to install Python dependencies.
    pause
    exit /b 1
)
echo Python dependencies installed successfully.

echo [3/4] Starting Python API...
start "Python API" cmd /k "cd /d %cd% && call venv\Scripts\activate && python api.py"

:: ---- Next.js App Setup ----
cd ..\app

echo [4/4] Installing npm dependencies and starting Next.js app...
call npm install

start "Next.js Dev" cmd /k "cd /d %cd% && npx next dev"

:: Wait for the dev server to start, then open browser
echo.
echo Waiting for dev server to start...
timeout /t 5 /nobreak >nul
start http://localhost:3000

echo.
echo ============================================
echo   Both services started!
echo   Browser opened to http://localhost:3000
echo   Close the opened windows to stop them.
echo ============================================
pause
