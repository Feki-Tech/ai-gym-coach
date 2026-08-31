@echo off
rem One-click coach server for the iPhone app (docs/IOS.md section 8).
rem Starts Ollama, makes sure the model is there, then runs coach_server.py.
cd /d "%~dp0"

where ollama >nul 2>nul
if errorlevel 1 (
  echo Ollama is not installed. Install it first:  winget install Ollama.Ollama
  pause & exit /b 1
)

rem start the Ollama service if it is not already answering
curl -s -m 2 http://localhost:11434/api/tags >nul 2>nul
if errorlevel 1 (
  echo Starting Ollama...
  start "" /min ollama serve
  timeout /t 4 /nobreak >nul
)

echo Making sure the model is downloaded (first time: ~2 GB)...
ollama pull llama3.2:3b

echo.
echo Starting the coach server. When Windows asks about the firewall,
echo click "Allow" (private networks) - the iPhone needs to reach this PC.
echo Enter the URL and pairing code below in the app: home - chat icon - Coach settings.
echo.
py coach_server.py
pause
