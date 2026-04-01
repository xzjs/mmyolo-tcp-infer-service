@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "PS_SCRIPT=%SCRIPT_DIR%build_and_push_image.ps1"

if not exist "%PS_SCRIPT%" (
  echo [ERROR] Script not found: "%PS_SCRIPT%"
  exit /b 1
)

echo [INFO] Build image only (no push)...
powershell -NoProfile -ExecutionPolicy Bypass -File "%PS_SCRIPT%" -Push:$false %*
if errorlevel 1 (
  echo [ERROR] Build failed.
  exit /b 1
)

echo [DONE] Build completed (no push).
exit /b 0
