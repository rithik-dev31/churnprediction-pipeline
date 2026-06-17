#!/usr/bin/env python3
import sqlite3
import os

DB_PATH = "database/churn_prediction.db"

# Remove existing database file
if os.path.exists(DB_PATH):
    os.remove(DB_PATH)
    print(f"✅ Database file deleted: {DB_PATH}")

# Reinitialize with empty tables
from database import init_database
init_database()
print("✅ Database reinitialized with empty tables")
