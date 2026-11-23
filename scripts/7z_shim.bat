@echo off
REM Stand-in for "7z x <archive> -y" used by SIBR's cmake/windows/downloadAndExtractZipFile.cmake,
REM so 7-Zip does not need to be installed. "cmake -E tar" (libarchive) reads .7z and .zip.
cmake -E tar xf %2
