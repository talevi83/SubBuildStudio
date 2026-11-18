@echo off
setlocal enabledelayedexpansion
title SubBuild Studio - Launcher

echo ===================================================
echo             SubBuild Studio Launcher
echo ===================================================
echo.

cd /d "%~dp0"

set NO_UPDATE=0
if /i "%~1"=="--no-update" set NO_UPDATE=1
if /i "%~2"=="--no-update" set NO_UPDATE=1
if "%SUBBUILD_NO_UPDATE%"=="1" set NO_UPDATE=1

:: 1. Ensure backend virtual environment exists
if not exist "backend\.venv\Scripts\python.exe" (
    echo [1/5] Setting up Python virtual environment...
    where uv >nul 2>nul
    if %ERRORLEVEL% equ 0 (
        uv venv --python 3.12 backend\.venv
        uv pip install -r backend\requirements.txt --python backend\.venv\Scripts\python.exe
    ) else (
        python -m venv backend\.venv
        backend\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
    )
    if not exist "backend\.venv\Scripts\python.exe" (
        echo [ERROR] Failed to set up Python virtual environment!
        pause
        exit /b 1
    )
) else (
    echo [1/5] Python virtual environment ready.
)

:: 2. Check for Git / Release updates before launching
if "%NO_UPDATE%"=="1" (
    echo [2/5] Auto-update check skipped.
) else (
    echo [2/5] Checking for updates from Git...
    set "PYTHONPATH=%~dp0backend"
    "%~dp0backend\.venv\Scripts\python.exe" -m app.core.updater --auto --no-frontend-build
)

:: 3. Ensure .env exists
if not exist "backend\.env" (
    echo [3/5] Initializing backend\.env from template...
    copy "backend\.env.example" "backend\.env" >nul
) else (
    echo [3/5] Configuration file backend\.env found.
)

:: 4. Build the frontend if it is missing or older than its sources
set NEED_BUILD=1
if not exist "frontend\dist\index.html" goto :build_frontend
for /f %%i in ('powershell -NoProfile -Command "$b=(Get-Item 'frontend\dist\index.html').LastWriteTime; $s=(Get-ChildItem 'frontend\src','frontend\index.html','frontend\package.json' -Recurse -File | Sort-Object LastWriteTime -Descending | Select-Object -First 1).LastWriteTime; if ($s -gt $b) {1} else {0}"') do set NEED_BUILD=%%i
:build_frontend
if not "%NEED_BUILD%"=="1" goto :frontend_ready
echo [4/5] Building frontend studio...
pushd frontend
if not exist "node_modules" call npm install
call npm run build
set BUILD_RESULT=%ERRORLEVEL%
popd
if not "%BUILD_RESULT%"=="0" (
    echo [ERROR] Frontend build failed!
    pause
    exit /b 1
)
goto :frontend_done
:frontend_ready
echo [4/5] Frontend studio build is up to date.
:frontend_done

:: 5. Make sure the window packages (pywebview, pystray, pillow) are installed
if /i "%~1"=="--browser" goto :browser_mode
if /i "%~2"=="--browser" goto :browser_mode
"%~dp0backend\.venv\Scripts\python.exe" -c "import webview, pystray, PIL" >nul 2>nul
if errorlevel 1 (
    echo [5/5] Installing the desktop window packages ^(one time^)...
    where uv >nul 2>nul
    if !ERRORLEVEL! equ 0 (
        uv pip install -r packaging\requirements-desktop.txt --python backend\.venv\Scripts\python.exe
    ) else (
        backend\.venv\Scripts\python.exe -m pip install -q --disable-pip-version-check -r packaging\requirements-desktop.txt
    )
    "%~dp0backend\.venv\Scripts\python.exe" -c "import webview, pystray, PIL" >nul 2>nul
    if errorlevel 1 (
        echo [WARNING] Could not install the window packages - opening in the browser instead.
        goto :browser_mode
    )
)

:: Start SubBuild Studio in its own window (no browser, no console). The API key and the
:: projects stay in backend\ (backend\.env and backend\storage), like in the browser mode.
echo [5/5] Starting SubBuild Studio in its own window...
set "SUBBUILD_HOME=%~dp0backend"
start "" "%~dp0backend\.venv\Scripts\pythonw.exe" "%~dp0packaging\launcher.py"
exit /b 0

:browser_mode
:: Per-launch session token (backend\app\core\auth.py): the link below signs the browser in.
for /f "delims=" %%t in ('backend\.venv\Scripts\python.exe -c "import secrets; print(secrets.token_urlsafe(32))"') do set "SUBBUILD_TOKEN=%%t"
echo [5/5] Starting SubBuild Studio on http://localhost:8000 ...
echo       Open: http://localhost:8000/?token=%SUBBUILD_TOKEN%
echo.
echo ===================================================
echo Server is running! Press Ctrl+C in this window to stop.
echo ===================================================
echo.

:: Launch browser in background after 2 seconds
start "" cmd /c "timeout /t 2 /nobreak >nul & start "" "http://localhost:8000/?token=%SUBBUILD_TOKEN%""

:: Start Uvicorn server
backend\.venv\Scripts\python.exe -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000

if %ERRORLEVEL% neq 0 (
    echo.
    echo Server stopped with error.
    pause
)
