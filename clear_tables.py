#!/usr/bin/env python3
"""
Clear Individual Trades and Trading Summary tables.

This script safely clears the individual_trade and trading_record tables
while keeping all other data (salary, investments, etc.) intact.

Usage:
    python clear_tables.py [database_path]

Default database path: /app/data/finance_tracker.db
"""

import sqlite3
import os
import sys
from datetime import datetime

def clear_tables(db_path):
    """Clear individual_trade and trading_record tables"""
    try:
        # Connect to database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        print(f"🔄 Clearing tables in: {db_path}")
        print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Clear individual trades table
        print("\n📊 Clearing individual_trade table...")
        cursor.execute('DELETE FROM individual_trade')
        deleted_individual = cursor.rowcount
        print(f"   ✅ Deleted {deleted_individual} individual trade records")
        
        # Clear trading records (summary table)
        print("📈 Clearing trading_record table...")
        cursor.execute('DELETE FROM trading_record')
        deleted_trading = cursor.rowcount
        print(f"   ✅ Deleted {deleted_trading} trading summary records")
        
        # Commit changes
        conn.commit()
        print(f"\n💾 Changes committed to database")
        
        # Verify tables are empty
        print("🔍 Verifying tables are empty...")
        cursor.execute('SELECT COUNT(*) FROM individual_trade')
        individual_count = cursor.fetchone()[0]
        cursor.execute('SELECT COUNT(*) FROM trading_record')
        trading_count = cursor.fetchone()[0]
        
        print(f"   Individual trades table: {individual_count} records")
        print(f"   Trading summary table: {trading_count} records")
        
        if individual_count == 0 and trading_count == 0:
            print("\n✅ SUCCESS: Both tables cleared successfully!")
            print("🎯 You can now start fresh with individual trades tracking")
        else:
            print("\n⚠️  WARNING: Tables may not be completely empty")
            return False
            
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: Failed to clear tables: {e}")
        return False
    finally:
        if conn:
            conn.close()

def create_backup(db_path):
    """Create a backup of the database before clearing"""
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = f"{db_path}.backup.clear_tables.{timestamp}"
        
        import shutil
        shutil.copy2(db_path, backup_path)
        print(f"💾 Backup created: {backup_path}")
        return backup_path
    except Exception as e:
        print(f"⚠️  Could not create backup: {e}")
        return None

def main():
    # Default database path
    db_path = "/app/data/finance_tracker.db"
    
    # Check if custom path provided
    if len(sys.argv) > 1:
        db_path = sys.argv[1]
    
    # Check if database file exists
    if not os.path.exists(db_path):
        print(f"❌ Database file not found: {db_path}")
        print("Usage: python clear_tables.py [database_path]")
        sys.exit(1)
    
    print("🧹 CLEARING INDIVIDUAL TRADES & TRADING SUMMARY TABLES")
    print("=" * 60)
    print(f"Database: {db_path}")
    
    # Create backup
    backup_path = create_backup(db_path)
    
    # Confirm action
    print(f"\n⚠️  This will DELETE ALL DATA from:")
    print(f"   - individual_trade table (all individual trades)")
    print(f"   - trading_record table (all monthly summaries)")
    print(f"\n🛡️  Other data (salary, investments, etc.) will NOT be affected")
    
    if backup_path:
        print(f"📦 Backup available at: {backup_path}")
    
    confirm = input("\nAre you sure you want to proceed? (yes/no): ").lower()
    if confirm not in ['yes', 'y']:
        print("❌ Operation cancelled")
        sys.exit(0)
    
    # Clear tables
    success = clear_tables(db_path)
    
    if success:
        print("\n🎉 OPERATION COMPLETED SUCCESSFULLY!")
        print("📱 You can now refresh your web application")
        print("🔄 Both Individual Trades and Trading Summary will be empty")
    else:
        print("\n💥 OPERATION FAILED!")
        if backup_path:
            print(f"🔄 You can restore from backup: {backup_path}")
        sys.exit(1)

if __name__ == "__main__":
    main()