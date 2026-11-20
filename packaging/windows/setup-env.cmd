@echo off
:: SubBuild Studio - one-time setup of the Python environment, run by the Windows installer
:: (and safe to run again: it only installs what is missing or changed).
::
:: Everything stays inside the install folder, so uninstalling removes it all:
::   tools\uv.exe       the package manager that ships with the installer
::   python\            Python 3.12 from python.org, also shipped with the installer (signed; no
::                      system-wide Python needed)
::
:: The environment is made by Python's own venv (its launchers are signed too); uv only installs
:: the packages into it. uv's own Python and venv launchers are unsigned, and behaviour-based
:: antivirus blocked them.
::   backend\.venv\     the app's packages (faster-whisper, the CUDA libraries, the window...)
setlocal
title SubBuild Studio - Setup
cd /d "%~dp0..\.."

set "UV=%CD%\tools\uv.exe"
set "PY=%CD%\python\python.exe"
set "UV_PYTHON_DOWNLOADS=never"
set "UV_CACHE_DIR=%CD%\.uv-cache"
set "UV_LINK_MODE=copy"
set "VENV_PY=%CD%\backend\.venv\Scripts\python.exe"

echo ===================================================
echo   SubBuild Studio - installing the AI engine
echo   This happens once and can take several minutes.
echo ===================================================
echo.

if not exist "%UV%" (
    echo [ERROR] tools\uv.exe is missing - please download the installer again.
    goto :fail
)
if not exist "%PY%" (
    echo [ERROR] python\python.exe is missing - please download the installer again.
    goto :fail
)

:: An environment made by an earlier build (another Python) is rebuilt from the shipped one
if exist "backend\.venv\pyvenv.cfg" (
    findstr /i /c:"home = %CD%\python" "backend\.venv\pyvenv.cfg" >nul 2>nul || (
        echo Rebuilding the Python environment for the new Python...
        rmdir /s /q "backend\.venv"
    )
)

if not exist "%VENV_PY%" (
    echo [1/2] Creating the Python environment...
    "%PY%" -m venv --without-pip "backend\.venv" || goto :fail
) else (
    echo [1/2] Python environment found.
)

echo [2/2] Installing packages...
"%UV%" pip install --python "%VENV_PY%" -r "backend\requirements.txt" -r "packaging\requirements-desktop.txt" || goto :fail

:: The download cache holds a second copy of every package (several GB with the CUDA libraries)
"%UV%" cache clean >nul 2>nul
if exist "%UV_CACHE_DIR%" rmdir /s /q "%UV_CACHE_DIR%" >nul 2>nul

echo.
echo Done.
exit /b 0

:fail
echo.
echo [ERROR] Setup did not finish. Check the internet connection and run the installer again.
pause
exit /b 1
