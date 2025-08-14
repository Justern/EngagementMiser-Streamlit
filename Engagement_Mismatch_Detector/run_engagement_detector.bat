@echo off
REM Engagement Mismatch Detector Launcher
REM This batch file allows you to run the script from anywhere

set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%"
python score_engagement_mismatch_standalone.py %*

REM Return to original directory
cd /d "%CD%"
