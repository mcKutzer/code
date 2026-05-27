@echo off
REM Launch the MPT Streamlit app and open it in the default browser.
REM Double-click to run. Close this window to stop the server.

setlocal

REM Anchor to the directory this .bat lives in, regardless of where it's launched from
cd /d "%~dp0"

REM Suppress Streamlit's first-run email prompt by pre-creating an empty credentials file
if not exist "%USERPROFILE%\.streamlit" mkdir "%USERPROFILE%\.streamlit"
if not exist "%USERPROFILE%\.streamlit\credentials.toml" (
    > "%USERPROFILE%\.streamlit\credentials.toml" echo [general]
    >> "%USERPROFILE%\.streamlit\credentials.toml" echo email = ""
)

REM Run Streamlit in the foreground; it auto-opens the browser once the server is ready.
python -m streamlit run app\main.py ^
    --server.port 8501 ^
    --browser.gatherUsageStats false

REM If Streamlit exits (crash or Ctrl-C), keep the window open so the error stays visible
echo.
echo Streamlit has stopped. Press any key to close this window.
pause >nul

endlocal
