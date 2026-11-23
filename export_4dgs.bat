@echo off
REM Convert a 4DGaussians checkpoint into a SIBR-4D model folder.
REM usage: export_4dgs.bat --ckpt <4DGaussians output folder> --data <dataset scene folder> --out <new model folder> [--frames N]
setlocal
set "HERE=%~dp0"
if not exist "%HERE%env\python.exe" (
  echo Run setup.bat first.
  exit /b 1
)
"%HERE%env\python.exe" "%HERE%tools\export_g4d.py" %*
endlocal
