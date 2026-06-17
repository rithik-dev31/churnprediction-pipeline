#!/bin/bash
# Quick start script for Customer Churn Prediction System

echo "========================================"
echo "Customer Churn Prediction System"
echo "Quick Start Setup"
echo "========================================"
echo ""

# Check Python version
echo "Checking Python version..."
python3 --version

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Initialize database
echo ""
echo "Initializing database..."
python3 -c "from database import init_database; init_database(); print('Database initialized successfully!')"

# Create necessary directories
echo ""
echo "Creating required directories..."
mkdir -p database datasets models logs

# Run application
echo ""
echo "========================================"
echo "Starting application..."
echo "Open browser at: http://localhost:8501"
echo "========================================"
echo ""

streamlit run app.py
