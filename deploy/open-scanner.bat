@echo off
rem Opens the arbitrage scanner UI: SSH tunnel to the server + browser.
rem Keep this window open while you use the scanner; close it to disconnect.
title Arbitrage scanner

rem Tunnel already open (port 8501 busy) - just open the browser.
netstat -ano | findstr "127.0.0.1:8501" | findstr LISTENING >nul
if %errorlevel%==0 (
    start "" http://127.0.0.1:8501
    exit /b
)

rem Open the browser in 3 seconds, when the tunnel is up.
start "" /min cmd /c "timeout /t 3 >nul & start http://127.0.0.1:8501"

echo Scanner: http://127.0.0.1:8501
echo Tunnel is open. Close this window to disconnect.
ssh -N -o ServerAliveInterval=30 -o ExitOnForwardFailure=yes -L 8501:127.0.0.1:8501 arbitrage

echo.
echo Connection closed.
pause
