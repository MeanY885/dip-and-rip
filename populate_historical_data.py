#!/usr/bin/env python3
"""
Script to populate historical BTC price data
Run this to fetch and store initial historical data in the database
"""

import os
import sys
import requests
from datetime import datetime, timedelta

# Add the current directory to Python path to import app modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    """Main function to populate historical data"""
    try:
        # Import after setting up path
        from app import app, store_historical_price_data
        
        with app.app_context():
            print("Fetching and storing historical BTC price data...")
            
            # Fetch 30 days of historical data (720 hours)
            result = store_historical_price_data(hours_back=720)
            
            if result['success']:
                print(f"✅ Success! Stored {result['stored_count']} historical price records")
                print(f"   Currency: {result['currency']}")
                print(f"   Time period: {result['hours_back']} hours back")
            else:
                print(f"❌ Error: {result['error']}")
                return 1
                
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)