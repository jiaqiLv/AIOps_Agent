@echo off
REM Export requirements.txt for the project

echo Exporting requirements.txt...

REM Method 1: pip freeze (all packages in current environment)
echo === Method 1: pip freeze ===
pip freeze > requirements_all.txt
echo Exported all packages to requirements_all.txt

echo.
echo === Method 2: pipreqs (project dependencies only) ===

REM Check if pipreqs is installed
pip show pipreqs >nul 2>&1
if %errorlevel% neq 0 (
    echo pipreqs not found. Installing...
    pip install pipreqs
)

REM Run pipreqs from project root
pushd "%~dp0.."
pipreqs --force .
popd

echo Exported project dependencies to requirements.txt
echo.
echo Done!
echo   - requirements_all.txt: All packages in current environment
echo   - requirements.txt: Only project dependencies (recommended for deployment)