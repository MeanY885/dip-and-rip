#!/usr/bin/env python3
"""
Migration script to update BitcoinTrade table schema
Run this once to migrate from old column names to new ones
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
    # Check if bitcoin_trade table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='bitcoin_trade'")
    if not cursor.fetchone():
        print("No bitcoin_trade table found. Nothing to migrate.")
        exit(0)
    
    # Check current schema
    cursor.execute("PRAGMA table_info(bitcoin_trade)")
    columns = cursor.fetchall()
    column_names = [col[1] for col in columns]
    
    if 'initial_investment_gbp' in column_names:
        print("Table already migrated. Nothing to do.")
        exit(0)
    
    print("Starting migration...")
    
    # Create new table with updated schema
    cursor.execute("""
        CREATE TABLE bitcoin_trade_new (
            id INTEGER PRIMARY KEY,
            status VARCHAR(20) NOT NULL,
            date DATE NOT NULL,
            type VARCHAR(10) NOT NULL DEFAULT 'BTC',
            initial_investment_gbp FLOAT NOT NULL,
            btc_buy_price FLOAT NOT NULL,
            btc_sell_price FLOAT,
            profit FLOAT,
            fee FLOAT,
            btc_amount FLOAT,
            created_at DATETIME,
            updated_at DATETIME
        )
    """)
    
    # Copy data from old table to new table
    cursor.execute("""
        INSERT INTO bitcoin_trade_new (
            id, status, date, type, 
            initial_investment_gbp, btc_buy_price, btc_sell_price, 
            profit, fee, btc_amount, created_at, updated_at
        )
        SELECT 
            id,
            CASE 
                WHEN crypto_value IS NOT NULL AND fee IS NOT NULL THEN 'Closed'
                ELSE 'Open'
            END as status,
            date,
            type,
            gbp_investment as initial_investment_gbp,
            btc_value as btc_buy_price,
            crypto_value as btc_sell_price,
            profit,
            fee,
            gbp_investment / btc_value as btc_amount,
            created_at,
            updated_at
        FROM bitcoin_trade
    """)
    
    # Drop old table and rename new table
    cursor.execute("DROP TABLE bitcoin_trade")
    cursor.execute("ALTER TABLE bitcoin_trade_new RENAME TO bitcoin_trade")
    
    # Commit changes
    conn.commit()
    print("Migration completed successfully!")
    
    # Show migrated data
    cursor.execute("SELECT COUNT(*) FROM bitcoin_trade")
    count = cursor.fetchone()[0]
    print(f"Migrated {count} records")
    
except Exception as e:
    print(f"Migration failed: {e}")
    conn.rollback()
    
finally:
    conn.close() 