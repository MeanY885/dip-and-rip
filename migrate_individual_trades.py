#!/usr/bin/env python3
"""
Safe migration script to add individual_trade table without affecting existing data.
This script only creates the new table if it doesn't exist.
"""

import sqlite3
import os
import sys

def migrate_database(db_path):
    """Add individual_trade table if it doesn't exist"""
    try:
        # Connect to database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if individual_trade table already exists
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='individual_trade'
        """)
        
        if cursor.fetchone():
            print("✅ individual_trade table already exists - no migration needed")
            return True
            
        # Create the individual_trade table
        print("Creating individual_trade table...")
        cursor.execute("""
            CREATE TABLE individual_trade (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy VARCHAR(100) NOT NULL,
                trade_date DATE NOT NULL,
                profit_loss_amount DECIMAL(10, 2) NOT NULL,
                profit_loss_percentage DECIMAL(5, 2),
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create index for better performance
        cursor.execute("""
            CREATE INDEX idx_individual_trade_strategy_date 
            ON individual_trade(strategy, trade_date)
        """)
        
        # Commit changes
        conn.commit()
        print("✅ Successfully created individual_trade table and index")
        
        # Verify table was created
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='individual_trade'
        """)
        
        if cursor.fetchone():
            print("✅ Table creation verified")
            return True
        else:
            print("❌ Table creation failed - verification failed")
            return False
            
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        return False
    finally:
        if conn:
            conn.close()

def main():
    # Default database path - adjust if needed
    db_path = "/app/data/finance_tracker.db"
    
    # Check if custom path provided
    if len(sys.argv) > 1:
        db_path = sys.argv[1]
    
    # Check if database file exists
    if not os.path.exists(db_path):
        print(f"❌ Database file not found: {db_path}")
        print("Usage: python migrate_individual_trades.py [database_path]")
        sys.exit(1)
    
    print(f"🔄 Starting migration for: {db_path}")
    
    # Create backup
    backup_path = f"{db_path}.backup.migration"
    try:
        import shutil
        shutil.copy2(db_path, backup_path)
        print(f"✅ Backup created: {backup_path}")
    except Exception as e:
        print(f"⚠️  Could not create backup: {e}")
        print("Continuing anyway...")
    
    # Run migration
    success = migrate_database(db_path)
    
    if success:
        print("\n✅ Migration completed successfully!")
        print("The individual trades feature is now available.")
        print("All existing data remains unchanged.")
    else:
        print("\n❌ Migration failed!")
        if os.path.exists(backup_path):
            print(f"You can restore from backup: {backup_path}")
        sys.exit(1)

if __name__ == "__main__":
    main()