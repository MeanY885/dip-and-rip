#!/usr/bin/env python3
"""
Database migration script to add new tables without affecting existing data.
Run this script on the production server to create SalaryRecord and TradingRecord tables.
"""

import sqlite3
import os

def run_migration():
    # Path to your production database
    db_path = '/app/data/finance_tracker.db'
    
    # Check if database exists
    if not os.path.exists(db_path):
        print(f"Database not found at {db_path}")
        return False
        
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if SalaryRecord table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='salary_record';")
        salary_exists = cursor.fetchone()
        
        if not salary_exists:
            print("Creating SalaryRecord table...")
            cursor.execute('''
                CREATE TABLE salary_record (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    year INTEGER NOT NULL,
                    month INTEGER NOT NULL,
                    before_tax REAL,
                    commission REAL,
                    after_tax REAL,
                    notes TEXT,
                    date DATE NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(year, month)
                );
            ''')
            print("✅ SalaryRecord table created successfully")
        else:
            print("ℹ️  SalaryRecord table already exists")
            
        # Check if TradingRecord table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='trading_record';")
        trading_exists = cursor.fetchone()
        
        if not trading_exists:
            print("Creating TradingRecord table...")
            cursor.execute('''
                CREATE TABLE trading_record (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy VARCHAR(100) NOT NULL,
                    year INTEGER NOT NULL,
                    month INTEGER NOT NULL,
                    performance_percentage REAL NOT NULL,
                    trade_count INTEGER DEFAULT 0,
                    profitable_trades INTEGER DEFAULT 0,
                    notes TEXT,
                    date DATE NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(strategy, year, month)
                );
            ''')
            print("✅ TradingRecord table created successfully")
        else:
            print("ℹ️  TradingRecord table already exists")
            
        conn.commit()
        conn.close()
        
        print("\n🎉 Migration completed successfully!")
        print("Your existing data has been preserved.")
        return True
        
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting database migration...")
    success = run_migration()
    if success:
        print("✅ Ready to deploy new application version!")
    else:
        print("❌ Migration failed. Please check the error above.")