@echo off
:: SubBuild Studio - one-time setup of the Python environment, run by the Windows installer
:: (and safe to run again: it only installs what is missing or changed).
::
:: Everything stays inside the install folder, so uninstalling removes it all:
::   tools\uv.exe       the package manager that ships with the installer
::   python\            Python 3.12 from python.org, shipped with the installer (signed; no
::                      system-wide Python needed). It is private to the app, so the app's
::                      packages (faster-whisper, the CUDA libraries, the window...) are installed
::                      straight into it - no venv, and the shortcut starts the signed
::                      python\pythonw.exe itself. (uv's own Python and launchers are unsigned,
::                      and behaviour-based antivirus blocked them.)
setlocal
title SubBuild Studio - Setup
cd /d "%~dp0..\.."

set "UV=%CD%\tools\uv.exe"
set "PY=%CD%\python\python.exe"
set "UV_PYTHON_DOWNLOADS=never"
set "UV_CACHE_DIR=%CD%\.uv-cache"
set "UV_LINK_MODE=copy"

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

:: Test builds of the installer used a venv; it is not used any more
if exist "backend\.venv" rmdir /s /q "backend\.venv" >nul 2>nul

set "WITH_OPTIONAL=0"
if /I "%~1"=="--with-optional-audio" set "WITH_OPTIONAL=1"

echo Installing packages...
:: --break-system-packages: this Python belongs to the app alone, so installing into it is intended.
:: (Without it, an EXTERNALLY-MANAGED marker left by an older build's uv-managed Python stops uv.)
"%UV%" pip install --python "%PY%" --break-system-packages -r "backend\requirements.txt" -r "packaging\requirements-desktop.txt" || goto :fail

if "%WITH_OPTIONAL%"=="1" (
    echo.
    echo Installing optional Vocal Isolation ^(Demucs^) and WhisperX packages...
    nvidia-smi >nul 2>nul
    if not errorlevel 1 (
        echo NVIDIA GPU detected - installing CUDA PyTorch + optional audio engines...
        "%UV%" pip install --python "%PY%" --break-system-packages --extra-index-url https://download.pytorch.org/whl/cu128 torch torchaudio -r "backend\requirements-optional.txt" || goto :fail
    ) else (
        echo No NVIDIA GPU detected - installing CPU PyTorch + optional audio engines...
        "%UV%" pip install --python "%PY%" --break-system-packages torch torchaudio -r "backend\requirements-optional.txt" || goto :fail
    )
)

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
