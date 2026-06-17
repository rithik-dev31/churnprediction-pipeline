"""
Database module for SQLite operations.
Handles user management, dataset storage, and prediction logging.
"""

import sqlite3
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import hashlib
import bcrypt

DB_PATH = "database/churn_prediction.db"

def init_database():
    """Initialize SQLite database with required tables."""
    os.makedirs("database", exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP
        )
    ''')
    
    # Datasets table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS datasets (
            dataset_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            dataset_name TEXT NOT NULL,
            file_path TEXT NOT NULL,
            file_size INTEGER,
            rows INTEGER,
            columns INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(user_id)
        )
    ''')
    
    # Models table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS models (
            model_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            model_name TEXT NOT NULL,
            model_type TEXT NOT NULL,
            model_path TEXT NOT NULL,
            accuracy REAL,
            precision REAL,
            recall REAL,
            f1_score REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(user_id)
        )
    ''')
    
    # Predictions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            model_id INTEGER,
            input_data TEXT,
            prediction INTEGER,
            confidence REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(user_id),
            FOREIGN KEY(model_id) REFERENCES models(model_id)
        )
    ''')
    
    conn.commit()
    conn.close()
    print(f"Database initialized at {DB_PATH}")


def hash_password(password: str) -> str:
    """Hash password using bcrypt."""
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')


def verify_password(password: str, password_hash: str) -> bool:
    """Verify password against hash."""
    return bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8'))


# ==================== USER OPERATIONS ====================

def create_user(username: str, email: str, password: str) -> Tuple[bool, str]:
    """Create a new user account."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        password_hash = hash_password(password)
        cursor.execute('''
            INSERT INTO users (username, email, password_hash)
            VALUES (?, ?, ?)
        ''', (username, email, password_hash))
        
        conn.commit()
        conn.close()
        return True, "User created successfully"
    except sqlite3.IntegrityError as e:
        if "username" in str(e):
            return False, "Username already exists"
        else:
            return False, "Email already exists"
    except Exception as e:
        return False, f"Error: {str(e)}"


def authenticate_user(username: str, password: str) -> Tuple[bool, Optional[int], str]:
    """Authenticate user and return user_id if successful."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('SELECT user_id, password_hash FROM users WHERE username = ?', (username,))
        result = cursor.fetchone()
        conn.close()
        
        if not result:
            return False, None, "User not found"
        
        user_id, password_hash = result
        if verify_password(password, password_hash):
            # Update last login
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute('UPDATE users SET last_login = ? WHERE user_id = ?', 
                         (datetime.now(), user_id))
            conn.commit()
            conn.close()
            return True, user_id, "Authentication successful"
        else:
            return False, None, "Invalid password"
    except Exception as e:
        return False, None, f"Error: {str(e)}"


def get_user(user_id: int) -> Optional[Dict]:
    """Get user information by user_id."""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('SELECT user_id, username, email, created_at FROM users WHERE user_id = ?', 
                      (user_id,))
        result = cursor.fetchone()
        conn.close()
        
        return dict(result) if result else None
    except Exception as e:
        print(f"Error: {str(e)}")
        return None


# ==================== DATASET OPERATIONS ====================

def save_dataset_info(user_id: int, dataset_name: str, file_path: str, 
                     file_size: int, rows: int, columns: int) -> Tuple[bool, str]:
    """Save dataset information to database."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO datasets (user_id, dataset_name, file_path, file_size, rows, columns)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (user_id, dataset_name, file_path, file_size, rows, columns))
        
        conn.commit()
        conn.close()
        return True, "Dataset saved successfully"
    except Exception as e:
        return False, f"Error: {str(e)}"


def get_user_datasets(user_id: int) -> List[Dict]:
    """Get all datasets for a user."""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM datasets WHERE user_id = ? ORDER BY created_at DESC', 
                      (user_id,))
        results = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in results]
    except Exception as e:
        print(f"Error: {str(e)}")
        return []


# ==================== MODEL OPERATIONS ====================

def save_model_info(user_id: int, model_name: str, model_type: str, 
                   model_path: str, accuracy: float, precision: float, 
                   recall: float, f1_score: float) -> Tuple[bool, int, str]:
    """Save trained model information to database."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO models (user_id, model_name, model_type, model_path, accuracy, precision, recall, f1_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (user_id, model_name, model_type, model_path, accuracy, precision, recall, f1_score))
        
        model_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return True, model_id, "Model saved successfully"
    except Exception as e:
        return False, -1, f"Error: {str(e)}"


def get_user_models(user_id: int) -> List[Dict]:
    """Get all trained models for a user."""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM models WHERE user_id = ? ORDER BY created_at DESC', 
                      (user_id,))
        results = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in results]
    except Exception as e:
        print(f"Error: {str(e)}")
        return []


# ==================== PREDICTION OPERATIONS ====================

def save_prediction(user_id: int, model_id: int, input_data: str, 
                   prediction: int, confidence: float) -> Tuple[bool, str]:
    """Save prediction to database."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO predictions (user_id, model_id, input_data, prediction, confidence)
            VALUES (?, ?, ?, ?, ?)
        ''', (user_id, model_id, input_data, prediction, confidence))
        
        conn.commit()
        conn.close()
        return True, "Prediction saved successfully"
    except Exception as e:
        return False, f"Error: {str(e)}"


def get_user_predictions(user_id: int, limit: int = 50) -> List[Dict]:
    """Get recent predictions for a user."""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM predictions WHERE user_id = ? 
            ORDER BY created_at DESC LIMIT ?
        ''', (user_id, limit))
        results = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in results]
    except Exception as e:
        print(f"Error: {str(e)}")
        return []


if __name__ == "__main__":
    init_database()
