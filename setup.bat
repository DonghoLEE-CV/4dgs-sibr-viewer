@echo off
REM SIBR-4D setup: checks prerequisites, creates env\, downloads and builds everything inside this folder.
REM Options: setup.bat -Clean (rebuild the viewer)   setup.bat -Yes (answer yes to install prompts)
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\setup.ps1" %*
if errorlevel 1 (
  echo.
  echo Setup did not finish. See the messages above and the logs folder.
)
pause
