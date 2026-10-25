@echo off
rem Prophet launcher: runs prophet.py next to this file with whichever Python is available.
where python >nul 2>nul
if %errorlevel%==0 (
  python -B "%~dp0prophet.py" %*
) else (
  py -3 -B "%~dp0prophet.py" %*
)
exit /b %errorlevel%
