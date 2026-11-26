@echo off
setlocal
cd /d "%~dp0"
echo ===== ClipForge setup =====

where python >nul 2>nul
if errorlevel 1 (
  echo [X] Python was not found. Install Python 3.10 or newer from https://www.python.org/downloads/
  echo     Tick "Add python.exe to PATH" during install, then run this script again.
  pause & exit /b 1
)

for /f %%v in ('python -c "import sys;print(1 if (3,10)<=sys.version_info[:2]<=(3,13) else 0)"') do set PYOK=%%v
if not "%PYOK%"=="1" (
  echo [!] Your Python version is outside 3.10 to 3.13, which is what ClipForge was built for.
  echo     If the package install below fails, install Python 3.12 from python.org and run setup again.
)

where ffmpeg >nul 2>nul
if errorlevel 1 (
  echo [X] FFmpeg was not found on PATH.
  echo     Easiest fix: open a terminal and run:   winget install Gyan.FFmpeg
  echo     Then close and reopen this window and run setup again.
  pause & exit /b 1
)

if not exist venv (
  echo Creating virtual environment...
  python -m venv venv
  if errorlevel 1 ( echo [X] Could not create venv & pause & exit /b 1 )
)

echo Installing packages (first time takes a few minutes)...
call venv\Scripts\python.exe -m pip install --upgrade pip
call venv\Scripts\python.exe -m pip install -r backend\requirements.txt
if errorlevel 1 ( echo [X] Package install failed & pause & exit /b 1 )

echo.
echo NVIDIA GPU transcription needs the CUDA libraries. Installing them now...
call venv\Scripts\python.exe -m pip install nvidia-cublas-cu12 nvidia-cudnn-cu12==9.*
if errorlevel 1 echo [!] CUDA libraries failed to install. Transcription will use the CPU instead.

if not exist .env copy .env.example .env >nul

echo.
echo [OK] Setup finished. Double-click start.bat to launch ClipForge.
pause
