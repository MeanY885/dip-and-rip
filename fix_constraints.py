#!/usr/bin/env python3
"""
Fix NOT NULL constraints in bitcoin_trade table
"""

import os
import sqlite3

# Get database path
database_url = os.environ.get('SQLALCHEMY_DATABASE_URI', 'sqlite:///finance_tracker.db')
if database_url.startswith('sqlite:////'):
    db_path = database_url[10:]
else:
    db_path = database_url[10:]

print(f"Fixing constraints in database: {db_path}")

# Connect to database
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

try:
    print("Starting constraint fix...")
    
    # Begin transaction
    cursor.execute("BEGIN TRANSACTION")
    
    # Create new table with correct constraints
    cursor.execute("""
        CREATE TABLE bitcoin_trade_fixed (
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
    
    # Copy data from original table
    cursor.execute("INSERT INTO bitcoin_trade_fixed SELECT * FROM bitcoin_trade")
    
    # Drop original table
    cursor.execute("DROP TABLE bitcoin_trade")
    
    # Rename new table
    cursor.execute("ALTER TABLE bitcoin_trade_fixed RENAME TO bitcoin_trade")
    
    # Commit transaction
    cursor.execute("COMMIT")
    
    print("Constraints fixed successfully!")
    
    # Verify the fix
    cursor.execute("PRAGMA table_info(bitcoin_trade)")
    columns = cursor.fetchall()
    
    print("\nUpdated table schema:")
    for col in columns:
        nullable = "NULL" if col[3] == 0 else "NOT NULL"
        print(f"  {col[1]}: {col[2]} {nullable}")
    
except Exception as e:
    print(f"Fix failed: {e}")
    cursor.execute("ROLLBACK")
    
finally:
    conn.close() 