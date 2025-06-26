@echo off
echo Starting AI Mini Chat Services...
echo.

echo Starting Backend (FastAPI) on http://localhost:8000
start "Backend" cmd /k "cd backend && python main.py"

echo.
echo Starting Frontend (Flask) on http://localhost:5000
start "Frontend" cmd /k "cd frontend && python simple_app.py"

echo.
echo Services are starting...
echo Backend: http://localhost:8000
echo Frontend: http://localhost:5000
echo.
echo Press any key to exit this launcher...
pause > nul 