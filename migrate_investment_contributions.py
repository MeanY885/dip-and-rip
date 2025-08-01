#!/usr/bin/env python3
"""
Migration script to create InvestmentContribution table and populate initial contributions
Run this once to add the new contribution tracking functionality
"""

import os
import sqlite3
from datetime import datetime

# Get database path
database_url = os.environ.get('SQLALCHEMY_DATABASE_URI', 'sqlite:///finance_tracker.db')
if database_url.startswith('sqlite:////'):
    db_path = database_url[10:]
else:
    db_path = database_url[10:]

print(f"Migrating database: {db_path}")

# Connect to database
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

try:
    # Check if investment_contribution table already exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='investment_contribution'")
    if cursor.fetchone():
        print("InvestmentContribution table already exists. Nothing to do.")
        exit(0)
    
    print("Creating InvestmentContribution table...")
    
    # Create the new table
    cursor.execute("""
        CREATE TABLE investment_contribution (
            id INTEGER PRIMARY KEY,
            investment_id INTEGER NOT NULL,
            amount FLOAT NOT NULL,
            date DATE NOT NULL,
            type VARCHAR(20) NOT NULL DEFAULT 'additional',
            notes TEXT,
            created_at DATETIME,
            FOREIGN KEY (investment_id) REFERENCES investment (id)
        )
    """)
    
    print("Populating initial contributions from existing investments...")
    
    # Get all existing investments and create initial contributions
    cursor.execute("SELECT id, start_investment, start_date FROM investment")
    investments = cursor.fetchall()
    
    for investment_id, start_investment, start_date in investments:
        cursor.execute("""
            INSERT INTO investment_contribution (
                investment_id, amount, date, type, notes, created_at
            ) VALUES (?, ?, ?, 'initial', 'Initial investment migrated from existing data', ?)
        """, (investment_id, start_investment, start_date, datetime.utcnow()))
    
    # Commit changes
    conn.commit()
    print(f"Migration completed successfully! Created {len(investments)} initial contribution records.")
    
    # Show summary
    cursor.execute("SELECT COUNT(*) FROM investment_contribution")
    count = cursor.fetchone()[0]
    print(f"Total contribution records: {count}")
    
except Exception as e:
    print(f"Migration failed: {e}")
    conn.rollback()
    
finally:
    conn.close()