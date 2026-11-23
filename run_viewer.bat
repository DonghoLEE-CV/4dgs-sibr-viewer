@echo off
REM SIBR-4D viewer.  usage: run_viewer.bat [model folder] [model folder] ... [--mode 1-8] [--cmp 0-4]
REM You can also drag model folders onto this file or onto the viewer window.
setlocal
set "HERE=%~dp0"
set "BIN=%HERE%third_party\sibr_core\install\bin"
if not exist "%BIN%\SIBR_gaussian4D_app_rwdi.exe" (
  echo The viewer is not built yet. Run setup.bat first.
  pause
  exit /b 1
)
set "PATH=%BIN%;%HERE%env\bin;%HERE%env\Library\bin;%PATH%"
cd /d "%BIN%"
start "" SIBR_gaussian4D_app_rwdi.exe %*
endlocal
