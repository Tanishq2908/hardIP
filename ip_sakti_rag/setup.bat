@echo off
cd /d "%~dp0"
call .venv\Scripts\activate
python setup_all.py
pause
