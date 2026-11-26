@echo off
cd /d "%~dp0"
if not exist venv\Scripts\python.exe (
  echo Run setup.bat first.
  pause & exit /b 1
)

where ffmpeg >nul 2>nul
if errorlevel 1 (
  echo [X] FFmpeg was not found on PATH. Install it with:   winget install Gyan.FFmpeg
  echo     then close this window, open a new one and run start.bat again.
  pause & exit /b 1
)

rem Make the pip-installed CUDA libraries visible to faster-whisper
for /f "delims=" %%i in ('venv\Scripts\python.exe -c "import site,os;p=site.getsitepackages()[-1];print(os.path.join(p,'nvidia','cublas','bin')+';'+os.path.join(p,'nvidia','cudnn','bin'))"') do set "PATH=%%i;%PATH%"

echo Starting ClipForge at http://127.0.0.1:8000  (close this window to stop it)
rem Open the browser a few seconds from now, once the server is ready to answer
start "" /min cmd /c "timeout /t 4 /nobreak >nul & start http://127.0.0.1:8000"
cd backend
..\venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
pause
