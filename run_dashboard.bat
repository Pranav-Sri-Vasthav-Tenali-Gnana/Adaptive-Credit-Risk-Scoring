@echo off
REM Launches the Streamlit dashboard. Double-click this file, or run it from
REM the project root.
cd /d "%~dp0"
echo Starting Streamlit dashboard at http://localhost:8501 ...
"%~dp0venv\Scripts\python.exe" -m streamlit run dashboard\app.py --server.port 8501
pause
