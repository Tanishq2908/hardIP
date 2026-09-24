@echo off
cd /d "%~dp0"
call .venv\Scripts\activate
echo Starting IP-SAKTI Sahayak at http://localhost:8000
start "" http://localhost:8000
uvicorn api:app --port 8000
