@echo off
REM Quick start script for Customer Churn Prediction System (Windows)

cls
echo ========================================
echo Customer Churn Prediction System
echo Quick Start Setup
echo ========================================
echo.

REM Check Python version
echo Checking Python version...
python --version

REM Create virtual environment
echo.
echo Creating virtual environment...
python -m venv venv

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies
echo.
echo Installing dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt

REM Initialize database
echo.
echo Initializing database...
python -c "from database import init_database; init_database(); print('Database initialized successfully!')"

REM Create necessary directories
echo.
echo Creating required directories...
if not exist database mkdir database
if not exist datasets mkdir datasets
if not exist models mkdir models
if not exist logs mkdir logs

REM Run application
echo.
echo ========================================
echo Starting application...
echo Open browser at: http://localhost:8501
echo ========================================
echo.

streamlit run app.py

pause
