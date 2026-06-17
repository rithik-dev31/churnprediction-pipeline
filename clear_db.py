#!/usr/bin/env python3
import sqlite3
import os

DB_PATH = "database/churn_prediction.db"

if os.path.exists(DB_PATH):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Get all table names
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    
    # Delete all data from each table
    for table in tables:
        table_name = table[0]
        cursor.execute(f'DELETE FROM {table_name}')
        print(f'Cleared table: {table_name}')
    
    conn.commit()
    conn.close()
    print('\n✅ All data deleted successfully from the database!')
else:
    print(f"❌ Database file not found at {DB_PATH}")
