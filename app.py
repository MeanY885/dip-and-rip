# app.py (with data viewer and price monitor)
from flask import Flask, render_template, request, jsonify, redirect
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import desc
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
import plotly.utils
from plotly.subplots import make_subplots
import json
import os
from datetime import datetime, timedelta
from collections import defaultdict
import logging
import itertools
import shutil
from flask import send_file
import tempfile
import hashlib
import re
import csv
import io
from werkzeug.utils import secure_filename
# Removed PDF and Excel imports - now using TXT parsing only
from difflib import SequenceMatcher
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import atexit

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# File upload configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'temp_uploads'
ALLOWED_EXTENSIONS = {'txt'}

# Database configuration
database_url = os.environ.get('SQLALCHEMY_DATABASE_URI', 'sqlite:///finance_tracker.db')
app.config['SQLALCHEMY_DATABASE_URI'] = database_url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Handle database directory creation for file-based SQLite
if database_url.startswith('sqlite:///'):
    # Extract the path (handle both relative and absolute paths)
    if database_url.startswith('sqlite:////'):
        # Absolute path (4 slashes)
        db_path = database_url[10:]  # Remove 'sqlite:///'
    else:
        # Relative path (3 slashes)
        db_path = database_url[10:]  # Remove 'sqlite:///'
    
    db_dir = os.path.dirname(db_path)
    
    if db_dir:
        try:
            os.makedirs(db_dir, mode=0o755, exist_ok=True)
            logger.info(f"Database directory ensured: {db_dir}")
            
            # Verify we can write to the directory
            if os.access(db_dir, os.W_OK):
                logger.info(f"Write access confirmed for: {db_dir}")
            else:
                logger.warning(f"No write access to: {db_dir}")
                
        except Exception as e:
            logger.error(f"Failed to create database directory {db_dir}: {e}")
            # Fallback to current directory
            app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///finance_tracker.db'
            logger.info("Falling back to current directory for database")

db = SQLAlchemy(app)

# Helper function for safe date formatting
def format_date_safe(date_obj):
    """Safely format date object to ISO format string"""
    if date_obj is None:
        return None
    
    # If it's already a string, return as is
    if isinstance(date_obj, str):
        return date_obj
    
    # If it has isoformat method (datetime.date or datetime.datetime)
    if hasattr(date_obj, 'isoformat'):
        return date_obj.isoformat()
    
    # Fallback to string conversion
    return str(date_obj)

def check_duration_column_exists():
    """Check if the duration_minutes column exists in the bitcoin_trade table"""
    try:
        from sqlalchemy import text
        db.session.execute(text("SELECT duration_minutes FROM bitcoin_trade LIMIT 1"))
        return True
    except:
        return False

# Database Models
class Investment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    start_date = db.Column(db.Date, nullable=False)
    start_investment = db.Column(db.Float, nullable=False)
    current_value = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    yearly_records = db.relationship('YearlyRecord', backref='investment_ref', lazy=True, cascade='all, delete-orphan')
    monthly_records = db.relationship('MonthlyRecord', backref='investment_ref', lazy=True, cascade='all, delete-orphan')

class YearlyRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    investment_id = db.Column(db.Integer, db.ForeignKey('investment.id'), nullable=False)
    year = db.Column(db.Integer, nullable=False)
    value = db.Column(db.Float, nullable=False)
    notes = db.Column(db.Text)
    date = db.Column(db.Date, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    __table_args__ = (db.UniqueConstraint('investment_id', 'year', name='unique_investment_year'),)

class MonthlyRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    investment_id = db.Column(db.Integer, db.ForeignKey('investment.id'), nullable=False)
    year = db.Column(db.Integer, nullable=False)
    month = db.Column(db.Integer, nullable=False)
    value = db.Column(db.Float, nullable=False)
    notes = db.Column(db.Text)
    date = db.Column(db.Date, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    __table_args__ = (db.UniqueConstraint('investment_id', 'year', 'month', name='unique_investment_year_month'),)

class UserPreferences(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    widget_order = db.Column(db.Text)  # JSON string of widget order
    investment_order = db.Column(db.Text)  # JSON string of investment order
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class BitcoinTrade(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    status = db.Column(db.String(20), nullable=False)  # 'Open', 'Closed'
    date = db.Column(db.Date, nullable=False)
    type = db.Column(db.String(10), nullable=False, default='BTC')  # 'BTC'
    initial_investment_gbp = db.Column(db.Float, nullable=False)  # Renamed from gbp_investment
    btc_buy_price = db.Column(db.Float, nullable=False)  # Renamed from btc_value
    btc_sell_price = db.Column(db.Float, nullable=True)  # Renamed from crypto_value, nullable for open trades
    profit = db.Column(db.Float, nullable=True)  # Auto-calculated, nullable for open trades
    fee = db.Column(db.Float, nullable=True)  # Nullable for open trades
    btc_amount = db.Column(db.Float, nullable=True)  # Store the amount of BTC purchased
    final_value_gbp = db.Column(db.Float, nullable=True)  # Final value received from trade
    # duration_minutes = db.Column(db.Integer, nullable=True)  # Duration in minutes for closed trades - COMMENTED OUT UNTIL DB MIGRATION
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def set_duration_minutes(self, value):
        """Safely set duration_minutes if the attribute exists"""
        if hasattr(self, 'duration_minutes'):
            self.duration_minutes = value
    
    def get_duration_minutes(self):
        """Safely get duration_minutes, return None if attribute doesn't exist"""
        return getattr(self, 'duration_minutes', None)

class BitcoinPriceHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, nullable=False, unique=True)
    price_gbp = db.Column(db.Float, nullable=False)
    price_usd = db.Column(db.Float, nullable=True)  # Optional USD price for reference
    volume = db.Column(db.Float, nullable=True)  # Trading volume if available
    source = db.Column(db.String(20), default='kraken')  # Data source
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    __table_args__ = (db.Index('idx_timestamp', 'timestamp'),)

class BitcoinPriceHistoryMinute(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, nullable=False, unique=True)
    price_gbp = db.Column(db.Float, nullable=False)
    price_usd = db.Column(db.Float, nullable=True)  # Optional USD price for reference
    volume = db.Column(db.Float, nullable=True)  # Trading volume if available
    source = db.Column(db.String(20), default='kraken')  # Data source
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Additional fields for minute-level analysis
    high_price_gbp = db.Column(db.Float, nullable=True)  # High price within the minute
    low_price_gbp = db.Column(db.Float, nullable=True)   # Low price within the minute
    open_price_gbp = db.Column(db.Float, nullable=True)  # Opening price of the minute
    close_price_gbp = db.Column(db.Float, nullable=True) # Closing price of the minute
    
    __table_args__ = (
        db.Index('idx_minute_timestamp', 'timestamp'),
        db.Index('idx_minute_created_at', 'created_at'),
    )

class FinanceCategory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False, unique=True)
    color = db.Column(db.String(7), default='#007bff')  # Hex color code
    description = db.Column(db.Text, nullable=True)
    budget = db.Column(db.Float, nullable=True)  # Monthly budget for this category
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    transactions = db.relationship('FinanceTransaction', backref='category_ref', lazy=True)

class FinanceTransaction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.Date, nullable=False)
    amount = db.Column(db.Float, nullable=False)
    description = db.Column(db.Text, nullable=False)
    category_id = db.Column(db.Integer, db.ForeignKey('finance_category.id'), nullable=True)
    month = db.Column(db.Integer, nullable=False)  # 1-12
    year = db.Column(db.Integer, nullable=False)
    source_file = db.Column(db.String(255), nullable=True)  # Track which file imported from
    source_row = db.Column(db.Text, nullable=True)  # Original row data for duplicate detection
    hash_value = db.Column(db.String(64), nullable=True, index=True)  # For duplicate detection
    is_duplicate = db.Column(db.Boolean, default=False)
    confidence_score = db.Column(db.Float, nullable=True)  # Auto-categorization confidence
    is_recurring = db.Column(db.Boolean, default=False)  # Mark if this is a recurring transaction
    needs_validation = db.Column(db.Boolean, default=False)  # Highlight for validation after import
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    __table_args__ = (db.Index('idx_date_amount', 'date', 'amount'),
                      db.Index('idx_year_month', 'year', 'month'))

class AccountBalance(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    balance = db.Column(db.Float, nullable=False)
    last_updated = db.Column(db.DateTime, default=datetime.utcnow)
    source_file = db.Column(db.String(255), nullable=True)

class RecurringTracker(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    description_pattern = db.Column(db.Text, nullable=False)  # Normalized description for matching
    expected_amount = db.Column(db.Float, nullable=False)  # Last known amount
    amount_history = db.Column(db.Text, nullable=True)  # JSON string of price changes over time
    frequency_days = db.Column(db.Integer, nullable=False)  # Expected days between occurrences
    last_occurrence = db.Column(db.Date, nullable=False)  # Last seen date
    status = db.Column(db.String(20), default='active')  # 'active', 'price_changed', 'missing', 'stopped'
    variance_threshold = db.Column(db.Float, default=0.05)  # 5% default threshold for price change alerts
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class FinanceCategoryLearning(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    description_pattern = db.Column(db.Text, nullable=False)
    category_id = db.Column(db.Integer, db.ForeignKey('finance_category.id'), nullable=False)
    frequency = db.Column(db.Integer, default=1)  # How often this pattern matches this category
    confidence = db.Column(db.Float, default=1.0)  # 0-1 confidence score
    last_used = db.Column(db.DateTime, default=datetime.utcnow)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    category = db.relationship('FinanceCategory', backref='learning_patterns', lazy=True)

# Create tables
with app.app_context():
    db.create_all()
    
    # Handle migration for BitcoinTrade table if needed
    try:
        # Check if we need to migrate old column names to new ones
        from sqlalchemy import inspect, text
        
        inspector = inspect(db.engine)
        if 'bitcoin_trade' in inspector.get_table_names():
            columns = [col['name'] for col in inspector.get_columns('bitcoin_trade')]
            
            # Check if old columns exist
            if 'gbp_investment' in columns and 'initial_investment_gbp' not in columns:
                logger.info("Migrating BitcoinTrade table to new schema...")
                
                # Create migration SQL
                migration_sql = """
                    ALTER TABLE bitcoin_trade ADD COLUMN initial_investment_gbp FLOAT;
                    ALTER TABLE bitcoin_trade ADD COLUMN btc_buy_price FLOAT;
                    ALTER TABLE bitcoin_trade ADD COLUMN btc_sell_price FLOAT;
                    ALTER TABLE bitcoin_trade ADD COLUMN btc_amount FLOAT;
                    
                    UPDATE bitcoin_trade 
                    SET initial_investment_gbp = gbp_investment,
                        btc_buy_price = btc_value,
                        btc_sell_price = crypto_value,
                        btc_amount = gbp_investment / btc_value;
                    
                    UPDATE bitcoin_trade
                    SET status = 'Closed'
                    WHERE btc_sell_price IS NOT NULL AND fee IS NOT NULL;
                    
                    UPDATE bitcoin_trade
                    SET status = 'Open'
                    WHERE btc_sell_price IS NULL OR fee IS NULL;
                """
                
                # Execute migration
                with db.engine.connect() as conn:
                    for statement in migration_sql.strip().split(';'):
                        if statement.strip():
                            conn.execute(text(statement))
                    conn.commit()
                
                logger.info("Migration completed successfully")
                
                # Drop old columns after successful migration
                drop_sql = """
                    ALTER TABLE bitcoin_trade DROP COLUMN gbp_investment;
                    ALTER TABLE bitcoin_trade DROP COLUMN btc_value;
                    ALTER TABLE bitcoin_trade DROP COLUMN crypto_value;
                """
                
                with db.engine.connect() as conn:
                    for statement in drop_sql.strip().split(';'):
                        if statement.strip():
                            try:
                                conn.execute(text(statement))
                            except Exception as e:
                                logger.warning(f"Could not drop old column: {e}")
                    conn.commit()
                    
    except Exception as e:
        logger.warning(f"Migration check failed: {e}")
        # If migration fails, just continue - the table might be new
    
    # Handle migration for FinanceTransaction table - add needs_validation column if missing
    try:
        inspector = inspect(db.engine)
        if 'finance_transaction' in inspector.get_table_names():
            columns = [col['name'] for col in inspector.get_columns('finance_transaction')]
            
            if 'needs_validation' not in columns:
                logger.info("Adding needs_validation column to finance_transaction table...")
                
                migration_sql = "ALTER TABLE finance_transaction ADD COLUMN needs_validation BOOLEAN DEFAULT 0;"
                
                with db.engine.connect() as conn:
                    conn.execute(text(migration_sql))
                    conn.commit()
                
                logger.info("needs_validation column added successfully")
                
    except Exception as e:
        logger.warning(f"FinanceTransaction migration check failed: {e}")
    
    # Optimize database for minute-level data collection
    try:
        inspector = inspect(db.engine)
        tables = inspector.get_table_names()
        
        # Check if the minute table exists and add additional indexes
        if 'bitcoin_price_history_minute' in tables:
            logger.info("Optimizing BitcoinPriceHistoryMinute table...")
            
            # Additional performance indexes for minute data
            optimization_sql = """
                CREATE INDEX IF NOT EXISTS idx_minute_price_timestamp_gbp ON bitcoin_price_history_minute(timestamp, price_gbp);
                CREATE INDEX IF NOT EXISTS idx_minute_timestamp_desc ON bitcoin_price_history_minute(timestamp DESC);
                CREATE INDEX IF NOT EXISTS idx_minute_created_timestamp ON bitcoin_price_history_minute(created_at, timestamp);
                PRAGMA optimize;
            """
            
            with db.engine.connect() as conn:
                for statement in optimization_sql.strip().split(';'):
                    if statement.strip():
                        try:
                            conn.execute(text(statement))
                        except Exception as e:
                            logger.warning(f"Could not create index: {e}")
                conn.commit()
            
            logger.info("Database optimization completed")
            
            # Data retention check - log current data status
            current_count = BitcoinPriceHistoryMinute.query.count()
            if current_count > 0:
                oldest = BitcoinPriceHistoryMinute.query.order_by(BitcoinPriceHistoryMinute.timestamp.asc()).first()
                newest = BitcoinPriceHistoryMinute.query.order_by(BitcoinPriceHistoryMinute.timestamp.desc()).first()
                logger.info(f"Minute data: {current_count} records from {oldest.timestamp} to {newest.timestamp}")
            else:
                logger.info("No minute-level data found - will start collecting on scheduler activation")
    
    except Exception as e:
        logger.warning(f"Database optimization failed: {e}")

class BTCBacktester:
    def __init__(self, lookback_days=365, investment_value=1000, buy_dip_percent=5, sell_gain_percent=10, transaction_fee_percent=0.1):
        self.lookback_days = lookback_days
        self.investment_value = investment_value
        self.buy_dip_percent = buy_dip_percent / 100
        self.sell_gain_percent = sell_gain_percent / 100
        self.transaction_fee_percent = transaction_fee_percent / 100
        self.data = None
        self.trades = []
        
    def fetch_data(self):
        """Fetch BTC data from Kraken API"""
        try:
            url = "https://api.kraken.com/0/public/OHLC"
            end_date = datetime.now()
            
            # For short timeframes, ensure we fetch enough data for analysis
            # Minimum 60 days total to ensure sufficient data points
            buffer_days = max(30, 60 - self.lookback_days) if self.lookback_days < 30 else 30
            total_days = self.lookback_days + buffer_days
            
            start_date = end_date - timedelta(days=total_days)
            since = int(start_date.timestamp())
            
            logger.info(f"Fetching Kraken data from {start_date} to {end_date}")
            
            # GBP only - Kraken only for backtester (simpler for now)
            kraken_pair = "XXBTZGBP"
            display_name = "BTC-GBP"
            url = "https://api.kraken.com/0/public/OHLC"
            
            try:
                logger.info(f"Trying Kraken pair: {kraken_pair} ({display_name})")
                
                params = {
                    'pair': kraken_pair,
                    'interval': 1440,
                    'since': since
                }
                
                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()
                
                data = response.json()
                
                if 'error' in data and data['error']:
                    logger.error(f"Kraken API error for {kraken_pair}: {data['error']}")
                    return None
                
                if 'result' not in data or not data['result']:
                    logger.warning(f"No result data for {kraken_pair}")
                    return None
                
                pair_data = None
                for key in data['result'].keys():
                    if key != 'last':
                        pair_data = data['result'][key]
                        break
                
                # Adjust minimum data requirement based on lookback period
                min_data_points = max(5, min(10, self.lookback_days))
                if not pair_data or len(pair_data) < min_data_points:
                    logger.warning(f"Insufficient data for {kraken_pair}: got {len(pair_data) if pair_data else 0}, need {min_data_points}")
                    return None
                
                df_data = []
                logger.info(f"Processing {len(pair_data)} rows for {kraken_pair}")
                logger.info(f"Sample raw data: {pair_data[0] if pair_data else 'No data'}")
                
                for row in pair_data:
                    df_data.append({
                        'timestamp': int(row[0]),
                        'Open': float(row[1]),
                        'High': float(row[2]),
                        'Low': float(row[3]),
                        'Close': float(row[4]),
                        'Volume': float(row[6])
                    })
                
                df = pd.DataFrame(df_data)
                df['Date'] = pd.to_datetime(df['timestamp'], unit='s')
                df.set_index('Date', inplace=True)
                df.drop('timestamp', axis=1, inplace=True)
                df = df.sort_index()
                
                logger.info(f"Successfully fetched {len(df)} rows of data for {display_name}")
                logger.info(f"Price range: £{df['Close'].min():.2f} to £{df['Close'].max():.2f}")
                logger.info(f"First few prices: {df['Close'].head(3).tolist()}")
                self.data = df
                return True
                
            except Exception as e:
                logger.error(f"Error processing {kraken_pair}: {str(e)}")
                return False
            
        except Exception as e:
            logger.error(f"General error fetching Kraken data: {str(e)}")
            return False
    
    def run_backtest_with_params(self, buy_dip_percent, sell_gain_percent):
        """Run backtest with specific parameters (for optimization)"""
        if self.data is None or self.data.empty:
            return None
        
        # Filter data to the requested lookback period (keeping most recent data)
        filtered_data = self.data.tail(self.lookback_days) if len(self.data) > self.lookback_days else self.data
            
        # Calculate signals with given parameters
        buy_dip_decimal = buy_dip_percent / 100
        sell_gain_decimal = sell_gain_percent / 100
        
        returns = filtered_data['Close'].pct_change()
        buy_signals = returns <= -buy_dip_decimal
        sell_signals = returns >= sell_gain_decimal
        
        # Run simulation
        cash = self.investment_value
        btc_holdings = 0
        holding_btc = False
        total_fees_paid = 0
        num_trades = 0
        portfolio_values = []
        
        for i, (date, row) in enumerate(filtered_data.iterrows()):
            price = row['Close']
            
            # Buy signal
            if buy_signals.iloc[i] and not holding_btc and cash > 0:
                transaction_fee = cash * self.transaction_fee_percent
                cash_after_fees = cash - transaction_fee
                btc_holdings = cash_after_fees / price
                total_fees_paid += transaction_fee
                cash = 0
                holding_btc = True
                num_trades += 1
            
            # Sell signal
            elif sell_signals.iloc[i] and holding_btc and btc_holdings > 0:
                cash_before_fees = btc_holdings * price
                transaction_fee = cash_before_fees * self.transaction_fee_percent
                cash = cash_before_fees - transaction_fee
                total_fees_paid += transaction_fee
                btc_holdings = 0
                holding_btc = False
                num_trades += 1
            
            current_value = cash + (btc_holdings * price)
            portfolio_values.append(current_value)
        
        final_value = portfolio_values[-1]
        total_return = (final_value - self.investment_value) / self.investment_value * 100
        
        # Calculate max drawdown
        peaks = pd.Series(portfolio_values).cummax()
        drawdowns = (pd.Series(portfolio_values) - peaks) / peaks * 100
        max_drawdown = drawdowns.min()
        
        return {
            'buy_dip_percent': buy_dip_percent,
            'sell_gain_percent': sell_gain_percent,
            'final_value': final_value,
            'total_return': total_return,
            'num_trades': num_trades,
            'total_fees_paid': total_fees_paid,
            'max_drawdown': max_drawdown
        }
    
    def optimize_parameters(self, buy_range=(1, 15), sell_range=(1, 25), step=0.5):
        """Find optimal buy/sell percentages"""
        logger.info("Starting parameter optimization...")
        
        if not self.fetch_data():
            return None
        
        # Generate parameter combinations
        buy_percentages = np.arange(buy_range[0], buy_range[1] + step, step)
        sell_percentages = np.arange(sell_range[0], sell_range[1] + step, step)
        
        results = []
        total_combinations = len(buy_percentages) * len(sell_percentages)
        
        logger.info(f"Testing {total_combinations} parameter combinations...")
        
        for i, (buy_pct, sell_pct) in enumerate(itertools.product(buy_percentages, sell_percentages)):
            # FIXED: Allow sell % to equal buy % (now includes 1%/1%)
            if sell_pct < buy_pct:  # Only skip if sell is LESS than buy
                continue
                
            result = self.run_backtest_with_params(buy_pct, sell_pct)
            if result:
                results.append(result)
            
            if (i + 1) % 50 == 0:
                logger.info(f"Processed {i + 1}/{total_combinations} combinations")
        
        # Calculate buy & hold benchmark
        initial_price = self.data['Close'].iloc[0]
        final_price = self.data['Close'].iloc[-1]
        buy_hold_return = (final_price - initial_price) / initial_price * 100
        
        # Sort by total return (descending)
        results.sort(key=lambda x: x['total_return'], reverse=True)
        
        logger.info(f"Optimization complete. Best return: {results[0]['total_return']:.2f}%")
        
        return {
            'results': results,
            'buy_hold_return': buy_hold_return,
            'total_tested': len(results),
            'data_period': f"{self.data.index[0].strftime('%Y-%m-%d')} to {self.data.index[-1].strftime('%Y-%m-%d')}"
        }
    
    def optimize_parameters_multi_timeframe(self, buy_range=(1, 15), sell_range=(1, 25), step=0.5):
        """Find optimal buy/sell percentages across multiple timeframes"""
        logger.info("Starting multi-timeframe parameter optimization...")
        
        # Define timeframes in days
        timeframes = {
            '1d': 1,
            '3d': 3,
            '7d': 7,
            '14d': 14,
            '30d': 30,
            '60d': 60,
            '90d': 90,
            '6m': 180,
            '1y': 365
        }
        
        # Store original lookback_days
        original_lookback = self.lookback_days
        
        multi_results = {}
        
        # Generate parameter combinations
        buy_percentages = np.arange(buy_range[0], buy_range[1] + step, step)
        sell_percentages = np.arange(sell_range[0], sell_range[1] + step, step)
        
        for timeframe_name, days in timeframes.items():
            logger.info(f"Optimizing for {timeframe_name} ({days} days)...")
            
            # Set lookback for this timeframe
            self.lookback_days = days
            
            # Fetch data for this timeframe
            if not self.fetch_data():
                logger.warning(f"Failed to fetch data for {timeframe_name}")
                continue
                
            results = []
            total_combinations = len(buy_percentages) * len(sell_percentages)
            
            for i, (buy_pct, sell_pct) in enumerate(itertools.product(buy_percentages, sell_percentages)):
                if sell_pct < buy_pct:
                    continue
                    
                result = self.run_backtest_with_params(buy_pct, sell_pct)
                if result:
                    results.append(result)
            
            # Calculate buy & hold benchmark for this timeframe
            if not self.data.empty:
                initial_price = self.data['Close'].iloc[0]
                final_price = self.data['Close'].iloc[-1]
                buy_hold_return = (final_price - initial_price) / initial_price * 100
            else:
                buy_hold_return = 0
            
            # Sort by total return (descending)
            results.sort(key=lambda x: x['total_return'], reverse=True)
            
            multi_results[timeframe_name] = {
                'results': results,
                'buy_hold_return': buy_hold_return,
                'total_tested': len(results),
                'data_period': f"{self.data.index[0].strftime('%Y-%m-%d')} to {self.data.index[-1].strftime('%Y-%m-%d')}" if not self.data.empty else "No data",
                'timeframe_days': days
            }
            
            logger.info(f"Completed {timeframe_name}: {len(results)} combinations tested")
        
        # Restore original lookback_days
        self.lookback_days = original_lookback
        
        # Calculate parameter consistency across timeframes
        consistency_analysis = self._calculate_parameter_consistency(multi_results)
        
        logger.info("Multi-timeframe optimization complete")
        
        return {
            'timeframes': multi_results,
            'consistency': consistency_analysis,
            'total_timeframes': len(multi_results)
        }
    
    def _calculate_parameter_consistency(self, multi_results):
        """Calculate consistency scores for parameter combinations across timeframes"""
        param_performance = {}
        
        # Collect all parameter combinations and their performance
        for timeframe, data in multi_results.items():
            if not data['results']:
                continue
                
            for result in data['results']:
                param_key = f"{result['buy_dip_percent']:.1f}_{result['sell_gain_percent']:.1f}"
                
                if param_key not in param_performance:
                    param_performance[param_key] = {
                        'buy_pct': result['buy_dip_percent'],
                        'sell_pct': result['sell_gain_percent'],
                        'timeframes': {},
                        'avg_return': 0,
                        'consistency_score': 0,
                        'positive_timeframes': 0
                    }
                
                param_performance[param_key]['timeframes'][timeframe] = {
                    'return': result['total_return'],
                    'rank': data['results'].index(result) + 1,
                    'total_tested': len(data['results'])
                }
        
        # Calculate consistency metrics
        for param_key, data in param_performance.items():
            returns = [tf['return'] for tf in data['timeframes'].values()]
            ranks = [tf['rank'] for tf in data['timeframes'].values()]
            
            if returns:
                data['avg_return'] = np.mean(returns)
                data['std_return'] = np.std(returns)
                data['avg_rank'] = np.mean(ranks)
                data['positive_timeframes'] = sum(1 for r in returns if r > 0)
                
                # Consistency score (lower is better for ranks, higher is better for returns)
                data['consistency_score'] = (data['avg_return'] / (data['std_return'] + 1)) * (data['positive_timeframes'] / len(returns))
        
        # Sort by consistency score
        sorted_params = sorted(param_performance.items(), key=lambda x: x[1]['consistency_score'], reverse=True)
        
        return {
            'top_consistent_params': sorted_params[:10],
            'analysis_summary': {
                'total_params_tested': len(param_performance),
                'timeframes_analyzed': len(multi_results)
            }
        }
    
    def get_automated_optimal_strategies(self):
        """Get automated optimal strategies for all timeframes - lightweight calculation"""
        logger.info("Calculating automated optimal strategies...")
        
        # Define timeframes in days
        timeframes = {
            '1d': 1,
            '3d': 3,
            '7d': 7,
            '14d': 14,
            '30d': 30,
            '60d': 60,
            '90d': 90,
            '6m': 180,
            '1y': 365
        }
        
        # Store original settings
        original_lookback = self.lookback_days
        original_investment = self.investment_value
        
        # Use smaller ranges for automated analysis (performance optimization)
        buy_range = (1, 10)
        sell_range = (1, 15)
        step = 1.0  # Larger step for faster calculation
        
        strategies = {}
        all_strategies = []
        
        for timeframe_name, days in timeframes.items():
            logger.info(f"Analyzing {timeframe_name} ({days} days)...")
            
            # Set lookback for this timeframe
            self.lookback_days = days
            self.investment_value = 1000  # Standard investment for comparison
            
            # Fetch data for this timeframe
            if not self.fetch_data():
                logger.warning(f"Failed to fetch data for {timeframe_name}")
                continue
            
            # Quick optimization with limited parameter range
            buy_percentages = np.arange(buy_range[0], buy_range[1] + step, step)
            sell_percentages = np.arange(sell_range[0], sell_range[1] + step, step)
            
            best_result = None
            best_return = -float('inf')
            results = []
            
            for buy_pct in buy_percentages:
                for sell_pct in sell_percentages:
                    if sell_pct < buy_pct:
                        continue
                    
                    result = self.run_backtest_with_params(buy_pct, sell_pct)
                    if result:
                        results.append(result)
                        if result['total_return'] > best_return:
                            best_return = result['total_return']
                            best_result = result
            
            if best_result and results:
                # Calculate confidence score based on consistency and performance
                returns = [r['total_return'] for r in results]
                avg_return = np.mean(returns)
                std_return = np.std(returns) if len(returns) > 1 else 0
                positive_results = sum(1 for r in returns if r > 0)
                
                # Confidence score: higher is better, considers consistency and positive performance
                confidence = min(100, max(0, 
                    (positive_results / len(returns)) * 100 * 
                    (1 + (avg_return / (std_return + 1)) * 0.1)
                ))
                
                # Buy & hold benchmark
                if not self.data.empty:
                    initial_price = self.data['Close'].iloc[0]
                    final_price = self.data['Close'].iloc[-1]
                    buy_hold_return = (final_price - initial_price) / initial_price * 100
                else:
                    buy_hold_return = 0
                
                strategy = {
                    'timeframe': timeframe_name,
                    'timeframe_days': days,
                    'buy_percent': best_result['buy_dip_percent'],
                    'sell_percent': best_result['sell_gain_percent'],
                    'expected_return': best_result['total_return'],
                    'confidence_score': confidence,
                    'num_trades': best_result['num_trades'],
                    'max_drawdown': best_result['max_drawdown'],
                    'buy_hold_return': buy_hold_return,
                    'outperformance': best_result['total_return'] - buy_hold_return,
                    'last_updated': datetime.now().isoformat()
                }
                
                strategies[timeframe_name] = strategy
                all_strategies.append(strategy)
        
        # Calculate overall best strategy
        overall_best = self._calculate_overall_best_strategy(all_strategies)
        
        # Restore original settings
        self.lookback_days = original_lookback
        self.investment_value = original_investment
        
        logger.info("Automated strategy calculation complete")
        
        return {
            'timeframe_strategies': strategies,
            'overall_best': overall_best,
            'calculation_time': datetime.now().isoformat(),
            'total_timeframes': len(strategies)
        }
    
    def _calculate_overall_best_strategy(self, all_strategies):
        """Calculate the overall best strategy across all timeframes"""
        if not all_strategies:
            return None
        
        # Weight strategies by timeframe importance (longer timeframes get higher weight)
        timeframe_weights = {
            '1d': 0.5, '3d': 0.7, '7d': 1.0, '14d': 1.2, '30d': 1.5,
            '60d': 1.8, '90d': 2.0, '6m': 2.2, '1y': 2.5
        }
        
        # Calculate weighted scores for each strategy combination
        param_scores = {}
        
        for strategy in all_strategies:
            timeframe = strategy['timeframe']
            param_key = f"{strategy['buy_percent']:.0f}_{strategy['sell_percent']:.0f}"
            weight = timeframe_weights.get(timeframe, 1.0)
            
            # Score based on return, confidence, and outperformance
            score = (
                strategy['expected_return'] * 0.4 +
                strategy['confidence_score'] * 0.3 +
                strategy['outperformance'] * 0.3
            ) * weight
            
            if param_key not in param_scores:
                param_scores[param_key] = {
                    'buy_percent': strategy['buy_percent'],
                    'sell_percent': strategy['sell_percent'],
                    'total_score': 0,
                    'weighted_return': 0,
                    'weighted_confidence': 0,
                    'timeframe_count': 0,
                    'strategies': []
                }
            
            param_scores[param_key]['total_score'] += score
            param_scores[param_key]['weighted_return'] += strategy['expected_return'] * weight
            param_scores[param_key]['weighted_confidence'] += strategy['confidence_score'] * weight
            param_scores[param_key]['timeframe_count'] += 1
            param_scores[param_key]['strategies'].append(strategy)
        
        # Find best overall strategy
        best_param = max(param_scores.items(), key=lambda x: x[1]['total_score'])
        best_data = best_param[1]
        
        # Calculate averages
        total_weight = sum(timeframe_weights[s['timeframe']] for s in best_data['strategies'])
        avg_return = best_data['weighted_return'] / total_weight
        avg_confidence = best_data['weighted_confidence'] / total_weight
        
        return {
            'buy_percent': best_data['buy_percent'],
            'sell_percent': best_data['sell_percent'],
            'average_return': avg_return,
            'average_confidence': avg_confidence,
            'timeframe_count': best_data['timeframe_count'],
            'total_score': best_data['total_score'],
            'strategies': best_data['strategies']
        }
    
    def calculate_signals(self):
        """Calculate buy/sell signals based on percentage moves"""
        if self.data is None or self.data.empty:
            return
            
        self.data['Returns'] = self.data['Close'].pct_change()
        self.data['Buy_Signal'] = self.data['Returns'] <= -self.buy_dip_percent
        self.data['Sell_Signal'] = self.data['Returns'] >= self.sell_gain_percent
        
        logger.info(f"Generated {self.data['Buy_Signal'].sum()} buy signals and {self.data['Sell_Signal'].sum()} sell signals")
        
    def backtest(self):
        """Run the backtest simulation with transaction costs"""
        logger.info("Starting backtest...")
        
        if not self.fetch_data():
            logger.error("Failed to fetch data for backtest")
            return None
            
        # Filter data to the requested lookback period (keeping most recent data)
        if len(self.data) > self.lookback_days:
            self.data = self.data.tail(self.lookback_days)
            logger.info(f"Filtered data to most recent {self.lookback_days} days ({len(self.data)} rows)")
            logger.info(f"Filtered price range: £{self.data['Close'].min():.2f} to £{self.data['Close'].max():.2f}")
        else:
            logger.info(f"Using all available data: {len(self.data)} rows")
            logger.info(f"Available price range: £{self.data['Close'].min():.2f} to £{self.data['Close'].max():.2f}")
        
        self.calculate_signals()
        
        cash = self.investment_value
        btc_holdings = 0
        holding_btc = False
        
        portfolio_values = []
        buy_hold_values = []
        total_fees_paid = 0
        self.trades = []
        
        initial_price = self.data['Close'].iloc[0]
        btc_amount_if_bought_and_held = self.investment_value / initial_price
        
        for i, (date, row) in enumerate(self.data.iterrows()):
            price = row['Close']
            
            if row['Buy_Signal'] and not holding_btc and cash > 0:
                transaction_fee = cash * self.transaction_fee_percent
                cash_after_fees = cash - transaction_fee
                btc_bought = cash_after_fees / price
                
                btc_holdings = btc_bought
                total_fees_paid += transaction_fee
                
                self.trades.append({
                    'date': date.isoformat(),
                    'action': 'BUY',
                    'price': float(price),
                    'amount': float(btc_bought),
                    'cash_used': float(cash),
                    'fee': float(transaction_fee),
                    'cash_after_fees': float(cash_after_fees)
                })
                cash = 0
                holding_btc = True
                logger.info(f"BUY: {btc_bought:.6f} BTC at {price:.2f} (fee: £{transaction_fee:.2f}) on {date.strftime('%Y-%m-%d')}")
            
            elif row['Sell_Signal'] and holding_btc and btc_holdings > 0:
                cash_before_fees = btc_holdings * price
                transaction_fee = cash_before_fees * self.transaction_fee_percent
                cash_received = cash_before_fees - transaction_fee
                
                total_fees_paid += transaction_fee
                
                self.trades.append({
                    'date': date.isoformat(),
                    'action': 'SELL',
                    'price': float(price),
                    'amount': float(btc_holdings),
                    'cash_before_fees': float(cash_before_fees),
                    'fee': float(transaction_fee),
                    'cash_received': float(cash_received)
                })
                cash = cash_received
                btc_holdings = 0
                holding_btc = False
                logger.info(f"SELL: £{cash_received:.2f} cash at {price:.2f} (fee: £{transaction_fee:.2f}) on {date.strftime('%Y-%m-%d')}")
            
            current_strategy_value = cash + (btc_holdings * price)
            portfolio_values.append(current_strategy_value)
            
            current_buy_hold_value = btc_amount_if_bought_and_held * price
            buy_hold_values.append(current_buy_hold_value)
        
        self.data['Strategy_Value'] = portfolio_values
        self.data['Buy_Hold_Value'] = buy_hold_values
        
        final_strategy_value = portfolio_values[-1]
        final_buy_hold_value = buy_hold_values[-1]
        
        strategy_return = (final_strategy_value - self.investment_value) / self.investment_value * 100
        buy_hold_return = (final_buy_hold_value - self.investment_value) / self.investment_value * 100
        
        strategy_peaks = pd.Series(portfolio_values).cummax()
        strategy_drawdowns = (pd.Series(portfolio_values) - strategy_peaks) / strategy_peaks * 100
        max_drawdown = strategy_drawdowns.min()
        
        buy_hold_peaks = pd.Series(buy_hold_values).cummax()
        buy_hold_drawdowns = (pd.Series(buy_hold_values) - buy_hold_peaks) / buy_hold_peaks * 100
        buy_hold_max_drawdown = buy_hold_drawdowns.min()
        
        logger.info(f"Backtest complete.")
        logger.info(f"Strategy: Final value: £{final_strategy_value:.2f}, Return: {strategy_return:.2f}%")
        logger.info(f"Buy & Hold: Final value: £{final_buy_hold_value:.2f}, Return: {buy_hold_return:.2f}%")
        
        return {
            'final_value': final_strategy_value,
            'total_return': strategy_return,
            'buy_hold_return': buy_hold_return,
            'buy_hold_final_value': final_buy_hold_value,
            'num_trades': len(self.trades),
            'trades': self.trades[-10:],
            'final_position': 'BTC' if holding_btc else 'Cash',
            'total_fees_paid': total_fees_paid,
            'max_drawdown': max_drawdown,
            'buy_hold_max_drawdown': buy_hold_max_drawdown,
            'data': self.data
        }
    
    def create_chart(self, results):
        """Create enhanced interactive chart with subplots"""
        logger.info(f"Creating chart with data shape: {self.data.shape}")
        logger.info(f"Chart data date range: {self.data.index.min()} to {self.data.index.max()}")
        logger.info(f"Chart price range: £{self.data['Close'].min():.2f} to £{self.data['Close'].max():.2f}")
        logger.info(f"Chart data sample: {self.data[['Close']].head(3).to_dict()}")
        
        # Enhanced debugging for chart data issues
        logger.info(f"Data index type: {type(self.data.index)}")
        logger.info(f"Close column dtype: {self.data['Close'].dtype}")
        logger.info(f"Data index sample: {self.data.index[:3].tolist()}")
        logger.info(f"Close values sample: {self.data['Close'][:3].tolist()}")
        logger.info(f"Data contains NaN: Close={self.data['Close'].isna().sum()}, Index={self.data.index.isna().sum() if hasattr(self.data.index, 'isna') else 'N/A'}")
        logger.info(f"Data contains Inf: Close={np.isinf(self.data['Close']).sum()}")
        
        # Check for zero values specifically
        zero_count = (self.data['Close'] == 0).sum()
        logger.info(f"Zero values in Close: {zero_count}")
        if zero_count > 0:
            logger.warning(f"Found {zero_count} zero values in Close data!")
            logger.warning(f"Zero value indices: {self.data[self.data['Close'] == 0].index.tolist()}")
            
        # Convert data to native Python types to avoid numpy serialization issues
        logger.info("Converting data to native Python types for Plotly compatibility...")
        chart_x_data = [x.isoformat() if hasattr(x, 'isoformat') else str(x) for x in self.data.index]
        chart_y_data = [float(y) for y in self.data['Close'].astype(float)]
        logger.info(f"Converted x data sample: {chart_x_data[:3]}")
        logger.info(f"Converted y data sample: {chart_y_data[:3]}")
        logger.info(f"Converted y data types: {[type(y) for y in chart_y_data[:3]]}")
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('BTC Price with Trading Signals', 'Portfolio Value Comparison'),
            vertical_spacing=0.1,
            row_heights=[0.6, 0.4]
        )
        
        # Price chart using converted data
        price_trace = go.Scatter(
            x=chart_x_data,
            y=chart_y_data,
            name='BTC Price',
            line=dict(color='orange', width=2),
            hovertemplate='<b>Price:</b> £%{y:,.2f}<br><b>Date:</b> %{x}<extra></extra>'
        )
        
        # Debug the actual trace data
        logger.info(f"Price trace x data length: {len(price_trace.x)}")
        logger.info(f"Price trace y data length: {len(price_trace.y)}")
        logger.info(f"Price trace y sample: {list(price_trace.y[:3])}")
        logger.info(f"Price trace y min/max: {min(price_trace.y):.2f} / {max(price_trace.y):.2f}")
        
        fig.add_trace(price_trace, row=1, col=1)
        
        # Buy signals (potential)
        buy_signals = self.data[self.data['Buy_Signal']]
        if not buy_signals.empty:
            fig.add_trace(go.Scatter(
                x=[x.isoformat() if hasattr(x, 'isoformat') else str(x) for x in buy_signals.index],
                y=[float(y) for y in buy_signals['Close'].astype(float)],
                mode='markers',
                name='Buy Signals',
                marker=dict(color='lightgreen', size=8, symbol='triangle-up'),
                hovertemplate='<b>Buy Signal</b><br>Price: £%{y:,.2f}<br>Date: %{x}<extra></extra>'
            ), row=1, col=1)
        
        # Sell signals (potential)
        sell_signals = self.data[self.data['Sell_Signal']]
        if not sell_signals.empty:
            fig.add_trace(go.Scatter(
                x=[x.isoformat() if hasattr(x, 'isoformat') else str(x) for x in sell_signals.index],
                y=[float(y) for y in sell_signals['Close'].astype(float)],
                mode='markers',
                name='Sell Signals',
                marker=dict(color='lightcoral', size=8, symbol='triangle-down'),
                hovertemplate='<b>Sell Signal</b><br>Price: £%{y:,.2f}<br>Date: %{x}<extra></extra>'
            ), row=1, col=1)
        
        # Executed trades
        if self.trades:
            buy_trades = [t for t in self.trades if t['action'] == 'BUY']
            sell_trades = [t for t in self.trades if t['action'] == 'SELL']
            
            if buy_trades:
                fig.add_trace(go.Scatter(
                    x=[pd.to_datetime(t['date']) for t in buy_trades],
                    y=[t['price'] for t in buy_trades],
                    mode='markers',
                    name='Executed Buys',
                    marker=dict(color='darkgreen', size=15, symbol='triangle-up', 
                              line=dict(color='white', width=2)),
                    hovertemplate='<b>EXECUTED BUY</b><br>Price: £%{y:,.2f}<br>Amount: %{customdata:.6f} BTC<br>Fee: £%{meta:.2f}<extra></extra>',
                    customdata=[t['amount'] for t in buy_trades],
                    meta=[t['fee'] for t in buy_trades]
                ), row=1, col=1)
            
            if sell_trades:
                fig.add_trace(go.Scatter(
                    x=[pd.to_datetime(t['date']) for t in sell_trades],
                    y=[t['price'] for t in sell_trades],
                    mode='markers',
                    name='Executed Sells',
                    marker=dict(color='darkred', size=15, symbol='triangle-down', 
                              line=dict(color='white', width=2)),
                    hovertemplate='<b>EXECUTED SELL</b><br>Price: £%{y:,.2f}<br>Amount: %{customdata:.6f} BTC<br>Fee: £%{meta:.2f}<extra></extra>',
                    customdata=[t['amount'] for t in sell_trades],
                    meta=[t['fee'] for t in sell_trades]
                ), row=1, col=1)
        
        # Portfolio values using converted data
        fig.add_trace(go.Scatter(
            x=chart_x_data,
            y=[float(y) for y in self.data['Strategy_Value'].astype(float)],
            name='Dip & Rip Strategy',
            line=dict(color='blue', width=3),
            hovertemplate='<b>Strategy Value:</b> £%{y:,.2f}<br><b>Date:</b> %{x}<extra></extra>'
        ), row=2, col=1)
        
        fig.add_trace(go.Scatter(
            x=chart_x_data,
            y=[float(y) for y in self.data['Buy_Hold_Value'].astype(float)],
            name='Buy & Hold',
            line=dict(color='gray', width=2, dash='dash'),
            hovertemplate='<b>Buy & Hold Value:</b> £%{y:,.2f}<br><b>Date:</b> %{x}<extra></extra>'
        ), row=2, col=1)
        
        fig.add_hline(
            y=self.investment_value, 
            line_dash="dot", 
            line_color="black",
            annotation_text=f"Initial Investment: £{self.investment_value:,}",
            annotation_position="top right",
            row=2, col=1
        )
        
        fig.update_layout(
            title='BTC Dip & Rip Strategy Analysis',
            height=800,
            showlegend=True,
            hovermode='x unified'
        )
        
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Price (£)", row=1, col=1)
        fig.update_yaxes(title_text="Portfolio Value (£)", row=2, col=1)
        
        # Debug figure structure before JSON conversion
        logger.info(f"Figure has {len(fig.data)} traces")
        for i, trace in enumerate(fig.data):
            logger.info(f"Trace {i}: name='{trace.name}', type='{trace.type}', y_data_length={len(trace.y) if hasattr(trace, 'y') and trace.y is not None else 'None'}")
            if hasattr(trace, 'y') and trace.y is not None and len(trace.y) > 0:
                logger.info(f"  Y data sample: {list(trace.y[:3])}")
                if hasattr(trace.y, '__iter__') and len([y for y in trace.y if y != 0]) > 0:
                    non_zero_y = [y for y in trace.y if y != 0]
                    logger.info(f"  Non-zero Y range: {min(non_zero_y):.2f} to {max(non_zero_y):.2f}")
        
        # Log chart JSON structure before returning
        chart_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        logger.info(f"Chart JSON length: {len(chart_json)} characters")
        
        # Check if price data exists in JSON with more details
        if '"name":"BTC Price"' in chart_json:
            logger.info("BTC Price trace found in chart JSON")
            # Look for actual y values in the JSON
            import re
            btc_price_section = re.search(r'"name":"BTC Price".*?"y":\[(.*?)\]', chart_json, re.DOTALL)
            if btc_price_section:
                y_values_str = btc_price_section.group(1)
                logger.info(f"BTC Price y values preview: {y_values_str[:100]}...")
            else:
                logger.warning("BTC Price trace found but no y values detected in JSON")
        else:
            logger.warning("BTC Price trace NOT found in chart JSON")
            # Debug: Let's see what trace names ARE in the JSON
            import re
            trace_names = re.findall(r'"name":"([^"]+)"', chart_json)
            logger.warning(f"Traces found in JSON: {trace_names}")
            
            # Let's also see a sample of the JSON structure around traces
            logger.warning(f"JSON preview (first 500 chars): {chart_json[:500]}")
            
            # Check if there are any traces at all
            if '"data":[' in chart_json:
                logger.info("JSON contains data array")
            else:
                logger.error("JSON does not contain data array!")
                
            # Check for specific pandas/numpy serialization issues
            if 'np.float64' in chart_json:
                logger.error("Found np.float64 in JSON - serialization issue!")
            if 'NaN' in chart_json:
                logger.error("Found NaN in JSON!")
            if 'Infinity' in chart_json:
                logger.error("Found Infinity in JSON!")
            
        return chart_json

# Helper function to update monthly investment records
def update_current_monthly_record(investment_id, current_value):
    """Update or create monthly record for current month when investment value changes"""
    try:
        from datetime import datetime
        
        # Get current date
        now = datetime.now()
        current_year = now.year
        current_month = now.month
        
        # Check if monthly record already exists for this investment in current month
        existing_record = MonthlyRecord.query.filter_by(
            investment_id=investment_id,
            year=current_year,
            month=current_month
        ).first()
        
        if existing_record:
            # Update existing record
            existing_record.value = current_value
            existing_record.date = now.date()
            logger.info(f"Updated monthly record for investment {investment_id}: {current_value}")
        else:
            # Create new monthly record
            new_record = MonthlyRecord(
                investment_id=investment_id,
                year=current_year,
                month=current_month,
                value=current_value,
                date=now.date(),
                notes=f"Auto-updated from current value"
            )
            db.session.add(new_record)
            logger.info(f"Created new monthly record for investment {investment_id}: {current_value}")
        
        # Don't commit here - let the calling function handle the transaction
        return True
        
    except Exception as e:
        logger.error(f"Error updating monthly record for investment {investment_id}: {str(e)}")
        return False

def update_current_yearly_record(investment_id, current_value):
    """Update or create yearly record for current year when investment value changes"""
    try:
        from datetime import datetime
        
        # Get current date
        now = datetime.now()
        current_year = now.year
        
        # Check if yearly record already exists for this investment in current year
        existing_record = YearlyRecord.query.filter_by(
            investment_id=investment_id,
            year=current_year
        ).first()
        
        if existing_record:
            # Update existing record
            existing_record.value = current_value
            existing_record.date = now.date()
            logger.info(f"Updated yearly record for investment {investment_id}: {current_value}")
        else:
            # Create new yearly record
            new_record = YearlyRecord(
                investment_id=investment_id,
                year=current_year,
                value=current_value,
                date=now.date(),
                notes=f"Auto-updated from current value"
            )
            db.session.add(new_record)
            logger.info(f"Created new yearly record for investment {investment_id}: {current_value}")
        
        # Don't commit here - let the calling function handle the transaction
        return True
        
    except Exception as e:
        logger.error(f"Error updating yearly record for investment {investment_id}: {str(e)}")
        return False

# Function to handle month advancement (called at midnight on 1st of each month)
def advance_to_next_month():
    """Called at midnight on 1st of each month - create monthly snapshots for all investments"""
    with app.app_context():
        try:
            from datetime import datetime
            now = datetime.now()
            logger.info(f"Month advanced to: {now.strftime('%B %Y')} at {now.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Get all investments
            investments = Investment.query.all()
        
            if not investments:
                logger.info("No investments found - skipping monthly snapshot creation")
                return True
                
            snapshots_created = 0
            snapshots_updated = 0
            
            for investment in investments:
                try:
                    # Check if monthly record already exists for this investment in current month
                    existing_record = MonthlyRecord.query.filter_by(
                        investment_id=investment.id,
                        year=now.year,
                        month=now.month
                    ).first()
                    
                    if existing_record:
                        # Update existing record with current value
                        existing_record.value = investment.current_value
                        existing_record.date = now.date()
                        existing_record.notes = f"Auto-updated monthly snapshot"
                        snapshots_updated += 1
                        logger.info(f"Updated monthly snapshot for '{investment.name}': £{investment.current_value}")
                    else:
                        # Create new monthly record
                        new_record = MonthlyRecord(
                            investment_id=investment.id,
                            year=now.year,
                            month=now.month,
                            value=investment.current_value,
                            date=now.date(),
                            notes=f"Auto-created monthly snapshot"
                        )
                        db.session.add(new_record)
                        snapshots_created += 1
                        logger.info(f"Created monthly snapshot for '{investment.name}': £{investment.current_value}")
                        
                except Exception as e:
                    logger.error(f"Error creating monthly snapshot for investment '{investment.name}': {str(e)}")
                    continue
            
            # Commit all changes
            db.session.commit()
            
            logger.info(f"Monthly snapshot process completed: {snapshots_created} created, {snapshots_updated} updated")
            return True
            
        except Exception as e:
            logger.error(f"Error in month advancement: {str(e)}")
            db.session.rollback()
            return False

def advance_to_next_year():
    """Called at midnight on 1st of January - create yearly snapshots for all investments"""
    with app.app_context():
        try:
            from datetime import datetime
            now = datetime.now()
            logger.info(f"Year advanced to: {now.year} at {now.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Get all investments
            investments = Investment.query.all()
        
            if not investments:
                logger.info("No investments found - skipping yearly snapshot creation")
                return True
                
            snapshots_created = 0
            snapshots_updated = 0
            
            for investment in investments:
                try:
                    # Check if yearly record already exists for this investment in current year
                    existing_record = YearlyRecord.query.filter_by(
                        investment_id=investment.id,
                        year=now.year
                    ).first()
                    
                    if existing_record:
                        # Update existing record with current value
                        existing_record.value = investment.current_value
                        existing_record.date = now.date()
                        existing_record.notes = f"Auto-updated yearly snapshot"
                        snapshots_updated += 1
                        logger.info(f"Updated yearly snapshot for '{investment.name}': £{investment.current_value}")
                    else:
                        # Create new yearly record
                        new_record = YearlyRecord(
                            investment_id=investment.id,
                            year=now.year,
                            value=investment.current_value,
                            date=now.date(),
                            notes=f"Auto-created yearly snapshot"
                        )
                        db.session.add(new_record)
                        snapshots_created += 1
                        logger.info(f"Created yearly snapshot for '{investment.name}': £{investment.current_value}")
                        
                except Exception as e:
                    logger.error(f"Error creating yearly snapshot for investment '{investment.name}': {str(e)}")
                    continue
            
            # Commit all changes
            db.session.commit()
            
            logger.info(f"Yearly snapshot process completed: {snapshots_created} created, {snapshots_updated} updated")
            return True
            
        except Exception as e:
            logger.error(f"Error in year advancement: {str(e)}")
            db.session.rollback()
            return False

# Initialize APScheduler
scheduler = BackgroundScheduler()

# Schedule month advancement job to run at midnight on 1st of every month
scheduler.add_job(
    func=advance_to_next_month,
    trigger=CronTrigger(day=1, hour=0, minute=0),
    id='advance_month',
    name='Advance to next month',
    replace_existing=True
)

# Schedule year advancement job to run at midnight on 1st of January
scheduler.add_job(
    func=advance_to_next_year,
    trigger=CronTrigger(month=1, day=1, hour=0, minute=0),
    id='advance_year',
    name='Advance to next year',
    replace_existing=True
)

# Note: Additional scheduler jobs will be added after function definitions

# Start the scheduler first
try:
    scheduler.start()
    logger.info("APScheduler started successfully - monthly advancement scheduled")
    logger.info(f"Scheduler state: {scheduler.state}")
    
    # Log scheduler jobs after starting
    logger.info("Scheduled jobs:")
    for job in scheduler.get_jobs():
        next_run = job.next_run_time.isoformat() if job.next_run_time else "Not scheduled"
        logger.info(f"  - {job.id}: {job.name} - Next run: {next_run}")
    
    # Log the next scheduled run for our specific job
    job = scheduler.get_job('advance_month')
    if job:
        logger.info(f"Next monthly advancement scheduled for: {job.next_run_time}")
    else:
        logger.error("Monthly advancement job not found!")
        
except Exception as e:
    logger.error(f"Failed to start APScheduler: {str(e)}")

# Shut down scheduler when app exits
atexit.register(lambda: scheduler.shutdown())

# Utility function to fetch current BTC price
def get_current_btc_price():
    """Get current BTC/GBP price from Kraken and store as historical data"""
    try:
        url = "https://api.kraken.com/0/public/Ticker"
        params = {'pair': 'XXBTZGBP'}
        
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        if 'error' in data and data['error']:
            # Try USD if GBP fails
            params = {'pair': 'XXBTZGBP'}
            response = requests.get(url, params=params, timeout=5)
            data = response.json()
        
        if 'result' in data and data['result']:
            for pair_name, pair_data in data['result'].items():
                if 'c' in pair_data:  # 'c' is current price
                    price = float(pair_data['c'][0])
                    currency = 'GBP' if 'GBP' in pair_name else 'USD'
                    current_time = datetime.now()
                    
                    # Store this price as historical data
                    try:
                        # Round timestamp to nearest minute to avoid too many entries
                        rounded_time = current_time.replace(second=0, microsecond=0)
                        
                        # Check if we already have data for this timestamp
                        existing = BitcoinPriceHistory.query.filter_by(timestamp=rounded_time).first()
                        
                        if not existing:
                            # Store new historical price data
                            price_record = BitcoinPriceHistory(
                                timestamp=rounded_time,
                                price_gbp=price if currency == 'GBP' else None,
                                price_usd=price if currency == 'USD' else None,
                                volume=float(pair_data.get('v', [0, 0])[1]) if 'v' in pair_data else None,  # 24h volume
                                source='kraken'
                            )
                            db.session.add(price_record)
                            db.session.commit()
                            logger.info(f"Stored historical price: {price} {currency} at {rounded_time}")
                        
                    except Exception as e:
                        logger.error(f"Failed to store historical price data: {e}")
                        db.session.rollback()
                        # Continue with returning current price even if storage fails
                    
                    return {
                        'success': True,
                        'price': price,
                        'pair': currency,
                        'timestamp': current_time.isoformat()
                    }
        
        return {'success': False, 'error': 'No price data available'}
        
    except Exception as e:
        return {'success': False, 'error': str(e)}

def price_to_percentage(target_price, current_price, is_sell=False):
    """Convert absolute price to percentage change"""
    try:
        if current_price <= 0 or target_price <= 0:
            return 0
        
        if is_sell:
            # For sell: percentage gain from buy price
            # Assuming current_price is the buy price reference
            percentage = ((target_price - current_price) / current_price) * 100
        else:
            # For buy: percentage dip from current price  
            percentage = ((current_price - target_price) / current_price) * 100
        
        return round(percentage, 2)
    except:
        return 0

def percentage_to_price(percentage, current_price, is_sell=False):
    """Convert percentage to absolute price"""
    try:
        if current_price <= 0 or percentage < 0:
            return 0
        
        if is_sell:
            # For sell: current_price + percentage gain
            target_price = current_price * (1 + percentage / 100)
        else:
            # For buy: current_price - percentage dip
            target_price = current_price * (1 - percentage / 100)
        
        return round(target_price, 2)
    except:
        return 0

def calculate_profit_preview(buy_price, sell_price, investment_amount, fee_percent):
    """Calculate potential profit for price pair"""
    try:
        if buy_price <= 0 or sell_price <= 0 or investment_amount <= 0:
            return {
                'success': False,
                'error': 'Invalid input values'
            }
        
        if sell_price <= buy_price:
            return {
                'success': False,
                'error': 'Sell price must be higher than buy price'
            }
        
        # Calculate transaction costs
        buy_fee = investment_amount * (fee_percent / 100)
        cash_after_buy_fee = investment_amount - buy_fee
        btc_amount = cash_after_buy_fee / buy_price
        
        # Calculate sell transaction
        gross_sell_value = btc_amount * sell_price
        sell_fee = gross_sell_value * (fee_percent / 100)
        net_sell_value = gross_sell_value - sell_fee
        
        # Calculate profit
        total_fees = buy_fee + sell_fee
        profit = net_sell_value - investment_amount
        profit_percentage = (profit / investment_amount) * 100
        
        # Calculate stop loss scenarios
        stop_loss_scenarios = []
        stop_loss_percentages = [0.2, 0.4, 0.6, 0.8, 1.0]
        
        for stop_loss_percent in stop_loss_percentages:
            # Calculate stop loss price (percentage below buy price)
            stop_loss_price = buy_price * (1 - stop_loss_percent / 100)
            
            # Calculate loss if stop loss is triggered
            gross_stop_loss_value = btc_amount * stop_loss_price
            stop_loss_sell_fee = gross_stop_loss_value * (fee_percent / 100)
            net_stop_loss_value = gross_stop_loss_value - stop_loss_sell_fee
            
            # Calculate total loss
            stop_loss_total_fees = buy_fee + stop_loss_sell_fee
            stop_loss_loss = investment_amount - net_stop_loss_value
            stop_loss_loss_percentage = (stop_loss_loss / investment_amount) * 100
            
            stop_loss_scenarios.append({
                'percentage': stop_loss_percent,
                'price': round(stop_loss_price, 2),
                'loss_amount': round(stop_loss_loss, 2),
                'loss_percentage': round(stop_loss_loss_percentage, 2),
                'total_fees': round(stop_loss_total_fees, 2),
                'final_value': round(net_stop_loss_value, 2)
            })
        
        return {
            'success': True,
            'investment': investment_amount,
            'buy_price': buy_price,
            'sell_price': sell_price,
            'btc_amount': round(btc_amount, 8),
            'total_fees': round(total_fees, 2),
            'gross_profit': round(gross_sell_value - cash_after_buy_fee, 2),
            'net_profit': round(profit, 2),
            'profit_percentage': round(profit_percentage, 2),
            'final_value': round(net_sell_value, 2),
            'stop_loss_scenarios': stop_loss_scenarios
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

# Utility function to fetch historical data for data viewer
def fetch_historical_data(days=365):
    """Fetch historical BTC data for data viewer"""
    try:
        url = "https://api.kraken.com/0/public/OHLC"
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days + 30)
        since = int(start_date.timestamp())
        
        # GBP only
        kraken_pair = "XXBTZGBP"
        display_name = "BTC-GBP"
        
        params = {
            'pair': kraken_pair,
            'interval': 1440,
            'since': since
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if 'error' in data and data['error']:
            return {'success': False, 'error': str(data['error'])}
        
        if 'result' not in data or not data['result']:
            return {'success': False, 'error': 'No result data'}
        
        pair_data = None
        for key in data['result'].keys():
            if key != 'last':
                pair_data = data['result'][key]
                break
        
        if not pair_data or len(pair_data) < 10:
            return {'success': False, 'error': 'Insufficient data'}
        
        df_data = []
        for row in pair_data:
            df_data.append({
                'Date': datetime.fromtimestamp(int(row[0])).strftime('%Y-%m-%d'),
                'Open': float(row[1]),
                'High': float(row[2]),
                'Low': float(row[3]),
                'Close': float(row[4]),
                'Volume': float(row[6])
            })
        
        df = pd.DataFrame(df_data)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        
        return {
            'success': True,
            'data': df.to_dict('records'),
            'source': display_name,
            'total_rows': len(df)
        }
        
    except Exception as e:
        return {'success': False, 'error': str(e)}

def store_historical_price_data(hours_back=168):  # Default 7 days (168 hours)
    """Fetch and store historical BTC price data"""
    try:
        url = "https://api.kraken.com/0/public/OHLC"
        end_date = datetime.now()
        start_date = end_date - timedelta(hours=hours_back)
        since = int(start_date.timestamp())
        
        # GBP only
        kraken_pair = "XXBTZGBP"
        
        params = {
            'pair': kraken_pair,
            'interval': 60,  # 1 hour intervals
            'since': since
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if 'error' in data and data['error']:
            return {'success': False, 'error': str(data['error'])}
        
        if 'result' not in data or not data['result']:
            return {'success': False, 'error': 'No result data'}
        
        pair_data = None
        for key in data['result'].keys():
            if key != 'last':
                pair_data = data['result'][key]
                break
        
        if not pair_data:
            return {'success': False, 'error': 'No pair data found'}
        
        # Store the data
        stored_count = 0
        for row in pair_data:
            timestamp = datetime.fromtimestamp(int(row[0]))
            close_price = float(row[4])
            volume = float(row[6])
            
            # Check if this timestamp already exists
            existing = BitcoinPriceHistory.query.filter_by(timestamp=timestamp).first()
            if not existing:
                price_record = BitcoinPriceHistory(
                    timestamp=timestamp,
                    price_gbp=close_price,  # GBP only now
                    price_usd=None,
                    volume=volume,
                    source='kraken'
                )
                db.session.add(price_record)
                stored_count += 1
        
        db.session.commit()
        return {
            'success': True,
            'stored_count': stored_count,
            'hours_back': hours_back
        }
        
    except Exception as e:
        db.session.rollback()
        return {'success': False, 'error': str(e)}

def store_minute_price_data(minutes_back=1440):  # Default 24 hours (1440 minutes)
    """Fetch and store 1-minute interval BTC price data - GBP only"""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(minutes=minutes_back)
        
        # Try Kraken GBP first, then Binance GBP as fallback
        sources_to_try = [
            ("kraken", "XXBTZGBP", "https://api.kraken.com/0/public/OHLC"),
            ("binance", "BTCGBP", "https://data-api.binance.vision/api/v3/klines")
        ]
        
        for source, pair, url in sources_to_try:
            try:
                if source == 'kraken':
                    since = int(start_date.timestamp())
                    params = {
                        'pair': pair,
                        'interval': 1,  # 1 minute intervals
                        'since': since
                    }
                    
                    response = requests.get(url, params=params, timeout=10)
                    response.raise_for_status()
                    data = response.json()
                    
                    if 'error' in data and data['error']:
                        logger.warning(f"Kraken API error for {pair}: {data['error']}")
                        continue
                    
                    if 'result' not in data or not data['result']:
                        continue
                    
                    pair_data = None
                    for key in data['result'].keys():
                        if key != 'last':
                            pair_data = data['result'][key]
                            break
                    
                    if not pair_data:
                        continue
                    
                    # Store Kraken data
                    stored_count = 0
                    for row in pair_data:
                        timestamp = datetime.fromtimestamp(int(row[0]))
                        open_price = float(row[1])
                        high_price = float(row[2])
                        low_price = float(row[3])
                        close_price = float(row[4])
                        volume = float(row[6])
                        
                        # Check if this timestamp already exists
                        existing = BitcoinPriceHistoryMinute.query.filter_by(timestamp=timestamp).first()
                        if not existing:
                            price_record = BitcoinPriceHistoryMinute(
                                timestamp=timestamp,
                                price_gbp=close_price,
                                price_usd=None,
                                open_price_gbp=open_price,
                                high_price_gbp=high_price,
                                low_price_gbp=low_price,
                                close_price_gbp=close_price,
                                volume=volume,
                                source='kraken'
                            )
                            db.session.add(price_record)
                            stored_count += 1
                    
                elif source == 'binance':
                    # Binance API uses different format
                    start_time = int(start_date.timestamp() * 1000)  # Binance uses milliseconds
                    end_time = int(end_date.timestamp() * 1000)
                    
                    params = {
                        'symbol': pair,
                        'interval': '1m',  # 1 minute intervals
                        'startTime': start_time,
                        'endTime': end_time,
                        'limit': 1000
                    }
                    
                    response = requests.get(url, params=params, timeout=10)
                    response.raise_for_status()
                    data = response.json()
                    
                    if not data:
                        continue
                    
                    # Store Binance data
                    stored_count = 0
                    for row in data:
                        timestamp = datetime.fromtimestamp(int(row[0]) / 1000)  # Convert from milliseconds
                        open_price = float(row[1])
                        high_price = float(row[2])
                        low_price = float(row[3])
                        close_price = float(row[4])
                        volume = float(row[5])
                        
                        # Check if this timestamp already exists
                        existing = BitcoinPriceHistoryMinute.query.filter_by(timestamp=timestamp).first()
                        if not existing:
                            price_record = BitcoinPriceHistoryMinute(
                                timestamp=timestamp,
                                price_gbp=close_price,
                                price_usd=None,
                                open_price_gbp=open_price,
                                high_price_gbp=high_price,
                                low_price_gbp=low_price,
                                close_price_gbp=close_price,
                                volume=volume,
                                source='binance'
                            )
                            db.session.add(price_record)
                            stored_count += 1
                
                db.session.commit()
                logger.info(f"Stored {stored_count} minute-level GBP price records from {source}")
                return {
                    'success': True,
                    'stored_count': stored_count,
                    'source': source,
                    'minutes_back': minutes_back
                }
                
            except Exception as e:
                logger.error(f"Error storing minute data from {source} for {pair}: {e}")
                db.session.rollback()
                continue
        
        return {'success': False, 'error': 'All data sources failed'}
        
    except Exception as e:
        logger.error(f"Error in store_minute_price_data: {e}")
        db.session.rollback()
        return {'success': False, 'error': str(e)}

def initialize_historical_minute_data(days_back=30):
    """Initialize historical minute-level data on startup"""
    try:
        logger.info(f"Starting historical minute data initialization for {days_back} days")
        
        # Check if we already have recent data
        latest_record = BitcoinPriceHistoryMinute.query.order_by(
            BitcoinPriceHistoryMinute.timestamp.desc()
        ).first()
        
        # If we have data from within the last 10 minutes, skip initialization
        # Changed from 1 hour to 10 minutes to allow backfilling recent gaps
        if latest_record:
            minutes_since_latest = (datetime.now() - latest_record.timestamp).total_seconds() / 60
            if minutes_since_latest < 10:
                logger.info("Very recent minute data exists, skipping initialization")
                return {'success': True, 'message': 'Recent data exists, skipping initialization', 'stored_count': 0}
            elif minutes_since_latest > 10:
                logger.info(f"Data gap detected: {minutes_since_latest:.1f} minutes since last record. Proceeding with backfill.")
        else:
            logger.info("No existing minute data found. Proceeding with full initialization.")
        
        # Calculate total minutes to fetch
        total_minutes = days_back * 24 * 60
        
        # Kraken API limits - fetch in chunks to avoid hitting rate limits
        chunk_size = 1440  # 24 hours per chunk
        total_stored = 0
        chunks_processed = 0
        
        # Process in reverse chronological order (most recent first)
        for chunk_start in range(0, total_minutes, chunk_size):
            chunk_minutes = min(chunk_size, total_minutes - chunk_start)
            
            logger.info(f"Fetching chunk {chunks_processed + 1}: {chunk_minutes} minutes back from {chunk_start} minutes ago")
            
            # Fetch this chunk
            result = store_minute_price_data(minutes_back=chunk_start + chunk_minutes)
            
            if result['success']:
                total_stored += result['stored_count']
                chunks_processed += 1
                logger.info(f"Chunk {chunks_processed} completed: {result['stored_count']} records stored")
            else:
                logger.warning(f"Chunk {chunks_processed + 1} failed: {result.get('error', 'Unknown error')}")
            
            # Rate limiting - wait between chunks to avoid API limits
            if chunk_start + chunk_size < total_minutes:  # Don't wait after the last chunk
                import time
                time.sleep(2)  # 2 second delay between chunks
        
        logger.info(f"Historical minute data initialization completed: {total_stored} records stored across {chunks_processed} chunks")
        
        return {
            'success': True,
            'message': f'Initialized {total_stored} minute-level records for {days_back} days',
            'stored_count': total_stored,
            'chunks_processed': chunks_processed,
            'days_back': days_back
        }
        
    except Exception as e:
        logger.error(f"Error in initialize_historical_minute_data: {e}")
        return {'success': False, 'error': str(e)}

def detect_and_fill_data_gaps(max_gap_hours=2):
    """Detect gaps in minute-level data and attempt to fill them"""
    try:
        logger.info("Starting data gap detection and backfill process")
        
        # Find all records from the last 7 days, ordered by timestamp
        seven_days_ago = datetime.now() - timedelta(days=7)
        records = BitcoinPriceHistoryMinute.query.filter(
            BitcoinPriceHistoryMinute.timestamp >= seven_days_ago
        ).order_by(BitcoinPriceHistoryMinute.timestamp).all()
        
        if len(records) < 2:
            return {'success': True, 'message': 'Insufficient data for gap detection', 'gaps_filled': 0}
        
        gaps_found = []
        gaps_filled = 0
        
        # Check for gaps between consecutive records
        for i in range(1, len(records)):
            prev_time = records[i-1].timestamp
            curr_time = records[i].timestamp
            
            # Calculate expected time difference (should be 1 minute)
            time_diff = (curr_time - prev_time).total_seconds() / 60  # in minutes
            
            # If gap is larger than expected and within our fill threshold
            if time_diff > 5 and time_diff <= (max_gap_hours * 60):  # 5+ minutes missing, up to max_gap_hours
                gap_start = prev_time + timedelta(minutes=1)
                gap_end = curr_time - timedelta(minutes=1)
                
                gaps_found.append({
                    'start': gap_start,
                    'end': gap_end,
                    'duration_minutes': int(time_diff - 1)
                })
                
                logger.info(f"Found gap: {gap_start} to {gap_end} ({int(time_diff-1)} minutes)")
        
        # Attempt to fill detected gaps
        for gap in gaps_found:
            try:
                # Calculate minutes back from now to the gap start
                minutes_back = int((datetime.now() - gap['start']).total_seconds() / 60)
                
                # Attempt to fetch data for this period
                result = store_minute_price_data(minutes_back=minutes_back + gap['duration_minutes'])
                
                if result['success'] and result['stored_count'] > 0:
                    gaps_filled += 1
                    logger.info(f"Successfully filled gap from {gap['start']} to {gap['end']}")
                
            except Exception as e:
                logger.warning(f"Failed to fill gap from {gap['start']} to {gap['end']}: {e}")
                continue
        
        return {
            'success': True,
            'gaps_found': len(gaps_found),
            'gaps_filled': gaps_filled,
            'message': f'Found {len(gaps_found)} gaps, successfully filled {gaps_filled}'
        }
        
    except Exception as e:
        logger.error(f"Error in detect_and_fill_data_gaps: {e}")
        return {'success': False, 'error': str(e)}

def collect_current_minute_price():
    """Collect current BTC price and store in minute table - for scheduler - GBP only"""
    with app.app_context():  # Add Flask app context for scheduler
        # Try Kraken GBP first, then Binance GBP as fallback
        sources_to_try = [
            ("kraken", "XXBTZGBP", "https://api.kraken.com/0/public/Ticker"),
            ("binance", "BTCGBP", "https://data-api.binance.vision/api/v3/ticker/price")
        ]
        
        for source, pair, url in sources_to_try:
            try:
                if source == "kraken":
                    params = {'pair': pair}
                    response = requests.get(url, params=params, timeout=10)
                    response.raise_for_status()
                    data = response.json()
                    
                    if 'error' in data and data['error']:
                        logger.warning(f"Kraken ticker API error: {data['error']}")
                        continue
                    
                    if 'result' not in data:
                        continue
                    
                    # Get the pair data
                    pair_data = None
                    for key, value in data['result'].items():
                        if key != 'last':
                            pair_data = value
                            break
                    
                    if not pair_data:
                        continue
                    
                    # Extract current price
                    current_price = float(pair_data['c'][0])  # Last trade closed price
                    volume = float(pair_data.get('v', [0, 0])[1]) if 'v' in pair_data else None
                    
                elif source == "binance":
                    params = {'symbol': pair}
                    response = requests.get(url, params=params, timeout=10)
                    response.raise_for_status()
                    data = response.json()
                    
                    if 'price' not in data:
                        continue
                    
                    # Extract current price
                    current_price = float(data['price'])
                    volume = None  # Volume not available in this endpoint
                
                current_time = datetime.now()
                
                # Round timestamp to nearest minute to avoid too many entries
                rounded_time = current_time.replace(second=0, microsecond=0)
                
                # Check if we already have data for this minute
                existing = BitcoinPriceHistoryMinute.query.filter_by(timestamp=rounded_time).first()
                
                if not existing:
                    # Store new minute-level price data
                    price_record = BitcoinPriceHistoryMinute(
                        timestamp=rounded_time,
                        price_gbp=current_price,
                        close_price_gbp=current_price,  # Use current price as close for real-time data
                        volume=volume,
                        source=source
                    )
                    db.session.add(price_record)
                    db.session.commit()
                    
                    logger.info(f"Stored current minute price: £{current_price:.2f} at {rounded_time} from {source}")
                    return {'success': True, 'price': current_price, 'timestamp': rounded_time.isoformat(), 'source': source}
                else:
                    # Update existing record if needed (e.g., to set high/low within the minute)
                    if existing.high_price_gbp is None or current_price > existing.high_price_gbp:
                        existing.high_price_gbp = current_price
                    if existing.low_price_gbp is None or current_price < existing.low_price_gbp:
                        existing.low_price_gbp = current_price
                    existing.close_price_gbp = current_price  # Always update close to latest
                    
                    db.session.commit()
                    return {'success': True, 'price': current_price, 'timestamp': rounded_time.isoformat(), 'updated': True, 'source': source}
                    
            except Exception as e:
                logger.warning(f"Error collecting current minute price from {source}: {e}")
                continue
        
        # If we get here, all sources failed
        logger.error("All price data sources failed")
        db.session.rollback()
        return {'success': False, 'error': 'All price data sources failed'}

def calculate_swing_analysis(period_hours, days_back=7):
    """Calculate swing analysis for specified time periods using backward-looking windows from current time"""
    try:
        current_time = datetime.now()
        
        # For single period analysis, we look at multiple windows over the days_back period
        # But for multi-period, we want the most recent period_hours from now
        analysis_start = current_time - timedelta(days=days_back)
        
        # Query minute-level data for the analysis period
        minute_data = BitcoinPriceHistoryMinute.query.filter(
            BitcoinPriceHistoryMinute.timestamp >= analysis_start,
            BitcoinPriceHistoryMinute.timestamp <= current_time,
            BitcoinPriceHistoryMinute.price_gbp.isnot(None)
        ).order_by(BitcoinPriceHistoryMinute.timestamp.desc()).all()  # Most recent first
        
        if not minute_data:
            return {'success': False, 'error': 'No minute-level data available'}
        
        # Get current price (most recent data point)
        current_price = minute_data[0].price_gbp or minute_data[0].close_price_gbp
        if not current_price:
            return {'success': False, 'error': 'No current price available'}
        
        # For backward-looking analysis, we analyze multiple period_hours windows
        # within the days_back timeframe to get statistics
        swing_results = []
        
        # Calculate how many complete windows we can fit in the analysis period
        total_minutes = days_back * 24 * 60
        window_minutes = period_hours * 60
        num_windows = total_minutes // window_minutes
        
        # Analyze each complete window
        for window_num in range(min(num_windows, 50)):  # Limit to 50 windows for performance
            window_start_time = current_time - timedelta(hours=(window_num + 1) * period_hours)
            window_end_time = current_time - timedelta(hours=window_num * period_hours)
            
            # Get data for this specific window
            window_data = [
                record for record in minute_data 
                if window_start_time <= record.timestamp <= window_end_time
            ]
            
            if len(window_data) < 2:
                continue
            
            # Find highest and lowest prices in this window
            window_prices = []
            for record in window_data:
                price = record.price_gbp or record.close_price_gbp
                if price:
                    window_prices.append(price)
                # Also include high/low if available for more accuracy
                if record.high_price_gbp:
                    window_prices.append(record.high_price_gbp)
                if record.low_price_gbp:
                    window_prices.append(record.low_price_gbp)
            
            if not window_prices:
                continue
            
            window_high = max(window_prices)
            window_low = min(window_prices)
            window_start_price = window_data[-1].price_gbp or window_data[-1].close_price_gbp  # Oldest in window
            
            if window_start_price and window_start_price > 0:
                # Calculate swings from the window start price
                lowest_drop_pct = ((window_low - window_start_price) / window_start_price) * 100
                highest_increase_pct = ((window_high - window_start_price) / window_start_price) * 100
                volatility_pct = ((window_high - window_low) / window_start_price) * 100
                
                swing_results.append({
                    'window_start': window_start_time.isoformat(),
                    'window_end': window_end_time.isoformat(),
                    'period_hours': period_hours,
                    'window_high': round(window_high, 2),
                    'window_low': round(window_low, 2),
                    'window_start_price': round(window_start_price, 2),
                    'lowest_drop_pct': round(lowest_drop_pct, 2),
                    'highest_increase_pct': round(highest_increase_pct, 2),
                    'volatility_pct': round(volatility_pct, 2),
                    'data_points': len(window_data)
                })
        
        # Calculate summary statistics from all windows
        if swing_results:
            drops = [result['lowest_drop_pct'] for result in swing_results]
            increases = [result['highest_increase_pct'] for result in swing_results]
            volatilities = [result['volatility_pct'] for result in swing_results]
            
            summary = {
                'period_hours': period_hours,
                'days_analyzed': days_back,
                'total_windows': len(swing_results),
                'avg_lowest_drop_pct': round(sum(drops) / len(drops), 2),
                'max_lowest_drop_pct': round(min(drops), 2),  # min because drops are negative
                'avg_highest_increase_pct': round(sum(increases) / len(increases), 2),
                'max_highest_increase_pct': round(max(increases), 2),
                'avg_volatility_pct': round(sum(volatilities) / len(volatilities), 2),
                'max_volatility_pct': round(max(volatilities), 2),
                'current_price': round(current_price, 2)
            }
        else:
            summary = {
                'period_hours': period_hours,
                'days_analyzed': days_back,
                'total_windows': 0,
                'current_price': round(current_price, 2)
            }
        
        return {
            'success': True,
            'summary': summary,
            'swing_data': swing_results[:20] if len(swing_results) > 20 else swing_results  # Return first 20 for performance
        }
        
    except Exception as e:
        logger.error(f"Error in swing analysis: {e}")
        return {'success': False, 'error': str(e)}

def calculate_recent_period_swing(period_hours):
    """Calculate swing for a specific recent period from current time backward"""
    try:
        current_time = datetime.now()
        period_start = current_time - timedelta(hours=period_hours)
        
        # Query minute-level data for this specific period
        minute_data = BitcoinPriceHistoryMinute.query.filter(
            BitcoinPriceHistoryMinute.timestamp >= period_start,
            BitcoinPriceHistoryMinute.timestamp <= current_time,
            BitcoinPriceHistoryMinute.price_gbp.isnot(None)
        ).order_by(BitcoinPriceHistoryMinute.timestamp.asc()).all()
        
        if not minute_data:
            return {'success': False, 'error': f'No data available for {period_hours}h period'}
        
        # Get all prices in this period
        all_prices = []
        for record in minute_data:
            price = record.price_gbp or record.close_price_gbp
            if price:
                all_prices.append(price)
            # Include high/low for more accurate extremes
            if record.high_price_gbp:
                all_prices.append(record.high_price_gbp)
            if record.low_price_gbp:
                all_prices.append(record.low_price_gbp)
        
        if not all_prices:
            return {'success': False, 'error': f'No valid prices for {period_hours}h period'}
        
        # Get period start price and current price
        period_start_price = minute_data[0].price_gbp or minute_data[0].close_price_gbp
        current_price = minute_data[-1].price_gbp or minute_data[-1].close_price_gbp
        
        # Find absolute highest and lowest in the period
        period_high = max(all_prices)
        period_low = min(all_prices)
        
        if not period_start_price or period_start_price <= 0:
            return {'success': False, 'error': f'Invalid start price for {period_hours}h period'}
        
        # Calculate swings from period start price
        max_rise_pct = ((period_high - period_start_price) / period_start_price) * 100
        max_drop_pct = ((period_low - period_start_price) / period_start_price) * 100
        volatility_pct = ((period_high - period_low) / period_start_price) * 100
        
        # Also calculate from current price perspective
        rise_from_current = ((period_high - current_price) / current_price) * 100
        drop_from_current = ((period_low - current_price) / current_price) * 100
        
        return {
            'success': True,
            'period_hours': period_hours,
            'data_points': len(minute_data),
            'period_start_price': round(period_start_price, 2),
            'current_price': round(current_price, 2),
            'period_high': round(period_high, 2),
            'period_low': round(period_low, 2),
            'max_rise_pct': round(max_rise_pct, 2),
            'max_drop_pct': round(max_drop_pct, 2),
            'volatility_pct': round(volatility_pct, 2),
            'rise_from_current': round(rise_from_current, 2),
            'drop_from_current': round(drop_from_current, 2),
            'period_start': period_start.isoformat(),
            'period_end': current_time.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error calculating recent period swing for {period_hours}h: {e}")
        return {'success': False, 'error': str(e)}

def get_multi_period_swing_analysis(days_back=7):
    """Get swing analysis for multiple time periods ensuring mathematical consistency"""
    try:
        current_time = datetime.now()
        
        # Get all minute data for the longest period (24 hours)
        period_start = current_time - timedelta(hours=24)
        minute_data = BitcoinPriceHistoryMinute.query.filter(
            BitcoinPriceHistoryMinute.timestamp >= period_start,
            BitcoinPriceHistoryMinute.timestamp <= current_time,
            BitcoinPriceHistoryMinute.price_gbp.isnot(None)
        ).order_by(BitcoinPriceHistoryMinute.timestamp.asc()).all()
        
        if not minute_data:
            return {'success': False, 'error': 'No minute-level data available for analysis'}
        
        periods = [1, 3, 6, 9, 12, 24]  # hours
        results = {}
        cumulative_max_rise = 0
        cumulative_max_drop = 0
        
        for period in periods:
            period_start_time = current_time - timedelta(hours=period)
            
            # Filter data for this period
            period_data = [
                record for record in minute_data 
                if record.timestamp >= period_start_time
            ]
            
            if not period_data:
                period_label = '1d' if period == 24 else f'{period}h'
                results[period_label] = {'error': f'No data available for {period}h period'}
                continue
            
            # Get all prices in this period
            all_prices = []
            for record in period_data:
                price = record.price_gbp or record.close_price_gbp
                if price:
                    all_prices.append(price)
                # Include high/low for more accurate extremes
                if record.high_price_gbp:
                    all_prices.append(record.high_price_gbp)
                if record.low_price_gbp:
                    all_prices.append(record.low_price_gbp)
            
            if not all_prices:
                period_label = '1d' if period == 24 else f'{period}h'
                results[period_label] = {'error': f'No valid prices for {period}h period'}
                continue
            
            # Get period start price and current price
            period_start_price = period_data[0].price_gbp or period_data[0].close_price_gbp
            current_price = period_data[-1].price_gbp or period_data[-1].close_price_gbp
            
            # Find absolute highest and lowest in the period
            period_high = max(all_prices)
            period_low = min(all_prices)
            
            if not period_start_price or period_start_price <= 0:
                period_label = '1d' if period == 24 else f'{period}h'
                results[period_label] = {'error': f'Invalid start price for {period}h period'}
                continue
            
            # Calculate swings from period start price
            max_rise_pct = ((period_high - period_start_price) / period_start_price) * 100
            max_drop_pct = ((period_low - period_start_price) / period_start_price) * 100
            volatility_pct = ((period_high - period_low) / period_start_price) * 100
            
            # Ensure cumulative consistency - longer periods must have >= swings
            max_rise_pct = max(max_rise_pct, cumulative_max_rise)
            max_drop_pct = min(max_drop_pct, cumulative_max_drop)  # min because drops are negative
            
            # Update cumulative maximums for next iteration
            cumulative_max_rise = max(cumulative_max_rise, max_rise_pct)
            cumulative_max_drop = min(cumulative_max_drop, max_drop_pct)
            
            period_label = '1d' if period == 24 else f'{period}h'
            results[period_label] = {
                'period_hours': period,
                'total_windows': 1,
                'avg_lowest_drop_pct': round(max_drop_pct, 2),
                'max_lowest_drop_pct': round(max_drop_pct, 2),
                'avg_highest_increase_pct': round(max_rise_pct, 2),
                'max_highest_increase_pct': round(max_rise_pct, 2),
                'avg_volatility_pct': round(volatility_pct, 2),
                'current_price': round(current_price, 2),
                'data_points': len(period_data),
                'period_high': round(period_high, 2),
                'period_low': round(period_low, 2),
                'period_start_price': round(period_start_price, 2)
            }
        
        return {
            'success': True,
            'days_analyzed': days_back,
            'periods': results,
            'analysis_time': current_time.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in multi-period swing analysis: {e}")
        return {'success': False, 'error': str(e)}

def fetch_minute_data_for_viewer(days=7, limit=None):
    """Fetch minute-level data for the data viewer"""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        query = BitcoinPriceHistoryMinute.query.filter(
            BitcoinPriceHistoryMinute.timestamp >= start_date,
            BitcoinPriceHistoryMinute.timestamp <= end_date,
            BitcoinPriceHistoryMinute.price_gbp.isnot(None)
        ).order_by(BitcoinPriceHistoryMinute.timestamp.desc())
        
        if limit:
            query = query.limit(limit)
        
        minute_data = query.all()
        
        if not minute_data:
            return {'success': False, 'error': 'No minute-level data available'}
        
        # Convert to data viewer format
        data_points = []
        for record in minute_data:
            price = record.price_gbp or record.close_price_gbp
            open_price = record.open_price_gbp or price
            high_price = record.high_price_gbp or price
            low_price = record.low_price_gbp or price
            close_price = record.close_price_gbp or price
            
            data_points.append({
                'Date': record.timestamp.strftime('%Y-%m-%d %H:%M'),
                'Open': round(open_price, 2),
                'High': round(high_price, 2),
                'Low': round(low_price, 2),
                'Close': round(close_price, 2),
                'Volume': record.volume or 0
            })
        
        # Reverse to get chronological order
        data_points.reverse()
        
        return {
            'success': True,
            'data': data_points,
            'source': 'minute-level kraken data',
            'total_rows': len(data_points),
            'period': f'Last {days} days (1-minute intervals)'
        }
        
    except Exception as e:
        logger.error(f"Error fetching minute data for viewer: {e}")
        return {'success': False, 'error': str(e)}

def cleanup_old_minute_data(days_to_keep=30):
    """Clean up old minute-level data to manage database size"""
    try:
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        # Count records before cleanup
        total_count = BitcoinPriceHistoryMinute.query.count()
        old_count = BitcoinPriceHistoryMinute.query.filter(
            BitcoinPriceHistoryMinute.timestamp < cutoff_date
        ).count()
        
        if old_count == 0:
            logger.info(f"No minute data older than {days_to_keep} days found - no cleanup needed")
            return {'success': True, 'deleted_count': 0, 'total_count': total_count}
        
        logger.info(f"Cleaning up {old_count} minute records older than {cutoff_date}")
        
        # Delete old records in batches to avoid locking the database
        batch_size = 1000
        deleted_total = 0
        
        while True:
            # Get a batch of old records
            old_records = BitcoinPriceHistoryMinute.query.filter(
                BitcoinPriceHistoryMinute.timestamp < cutoff_date
            ).limit(batch_size).all()
            
            if not old_records:
                break
            
            # Delete the batch
            for record in old_records:
                db.session.delete(record)
            
            db.session.commit()
            deleted_total += len(old_records)
            
            logger.info(f"Deleted {deleted_total}/{old_count} old minute records...")
            
            # Small delay to prevent overwhelming the database
            import time
            time.sleep(0.1)
        
        # Optimize database after cleanup
        try:
            with db.engine.connect() as conn:
                conn.execute(text("VACUUM"))
                conn.execute(text("PRAGMA optimize"))
                conn.commit()
        except Exception as e:
            logger.warning(f"Database optimization after cleanup failed: {e}")
        
        remaining_count = BitcoinPriceHistoryMinute.query.count()
        logger.info(f"Cleanup completed: deleted {deleted_total} records, {remaining_count} remaining")
        
        return {
            'success': True,
            'deleted_count': deleted_total,
            'total_count': total_count,
            'remaining_count': remaining_count,
            'days_kept': days_to_keep
        }
        
    except Exception as e:
        logger.error(f"Error in cleanup_old_minute_data: {e}")
        db.session.rollback()
        return {'success': False, 'error': str(e)}

def archive_old_price_data(days_to_keep=365):
    """Archive old hourly price data (BitcoinPriceHistory) to manage database size"""
    try:
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        # Count records before cleanup
        total_count = BitcoinPriceHistory.query.count()
        old_count = BitcoinPriceHistory.query.filter(
            BitcoinPriceHistory.timestamp < cutoff_date
        ).count()
        
        if old_count == 0:
            logger.info(f"No hourly data older than {days_to_keep} days found - no archiving needed")
            return {'success': True, 'archived_count': 0, 'total_count': total_count}
        
        logger.info(f"Archiving {old_count} hourly records older than {cutoff_date}")
        
        # For now, just delete old records (in a real system, you might export to files first)
        batch_size = 500
        archived_total = 0
        
        while True:
            # Get a batch of old records
            old_records = BitcoinPriceHistory.query.filter(
                BitcoinPriceHistory.timestamp < cutoff_date
            ).limit(batch_size).all()
            
            if not old_records:
                break
            
            # Delete the batch
            for record in old_records:
                db.session.delete(record)
            
            db.session.commit()
            archived_total += len(old_records)
            
            logger.info(f"Archived {archived_total}/{old_count} old hourly records...")
            
            # Small delay to prevent overwhelming the database
            import time
            time.sleep(0.1)
        
        remaining_count = BitcoinPriceHistory.query.count()
        logger.info(f"Archiving completed: archived {archived_total} records, {remaining_count} remaining")
        
        return {
            'success': True,
            'archived_count': archived_total,
            'total_count': total_count,
            'remaining_count': remaining_count,
            'days_kept': days_to_keep
        }
        
    except Exception as e:
        logger.error(f"Error in archive_old_price_data: {e}")
        db.session.rollback()
        return {'success': False, 'error': str(e)}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def calculate_transaction_hash(transaction_data):
    """Calculate hash for duplicate detection"""
    hash_string = f"{transaction_data['date']}_{transaction_data['amount']}_{transaction_data['description']}"
    return hashlib.sha256(hash_string.encode()).hexdigest()

def extract_description_keywords(description):
    """Extract meaningful keywords from transaction description"""
    keywords = re.findall(r'\b[A-Za-z]{3,}\b', description.lower())
    return ' '.join(keywords[:5])  # Take first 5 meaningful words

def predict_category(description, existing_patterns):
    """Predict category based on learned patterns"""
    best_match = None
    best_score = 0.0
    
    description_keywords = extract_description_keywords(description)
    
    for pattern in existing_patterns:
        similarity = SequenceMatcher(None, description_keywords, pattern.description_pattern).ratio()
        score = similarity * pattern.confidence * (pattern.frequency / 10.0)  # Weight by frequency
        
        if score > best_score and score > 0.2:  # Minimum threshold (lowered for better auto-categorization)
            best_score = score
            best_match = pattern
    
    return best_match, best_score

def detect_recurring_transactions():
    """Analyze transactions to detect recurring patterns and update RecurringTracker"""
    try:
        # Get all transactions from the last 6 months
        six_months_ago = datetime.now().date() - timedelta(days=180)
        transactions = FinanceTransaction.query.filter(
            FinanceTransaction.date >= six_months_ago
        ).order_by(FinanceTransaction.date.asc()).all()
        
        # Group transactions by normalized description pattern
        pattern_groups = defaultdict(list)
        for transaction in transactions:
            pattern = extract_description_keywords(transaction.description)
            if pattern and len(pattern) > 5:  # Only consider meaningful patterns
                pattern_groups[pattern].append(transaction)
        
        # Analyze each group for recurring behavior
        for pattern, group_transactions in pattern_groups.items():
            if len(group_transactions) >= 3:  # Need at least 3 occurrences to detect pattern
                analyze_recurring_pattern(pattern, group_transactions)
                
        logger.info(f"Recurring detection completed for {len(pattern_groups)} transaction patterns")
        
    except Exception as e:
        logger.error(f"Error in recurring detection: {e}")

def analyze_recurring_pattern(pattern, transactions):
    """Analyze a group of similar transactions for recurring behavior"""
    try:
        # Sort by date
        transactions.sort(key=lambda t: t.date)
        
        # Calculate intervals between transactions
        intervals = []
        for i in range(1, len(transactions)):
            days_diff = (transactions[i].date - transactions[i-1].date).days
            intervals.append(days_diff)
        
        # Determine if pattern is recurring (consistent intervals)
        if intervals and len(set(intervals)) <= 2:  # Allow some variance in intervals
            avg_frequency = sum(intervals) / len(intervals)
            
            # Check for existing tracker
            existing_tracker = RecurringTracker.query.filter_by(
                description_pattern=pattern
            ).first()
            
            latest_transaction = transactions[-1]
            
            if existing_tracker:
                update_recurring_tracker(existing_tracker, latest_transaction, avg_frequency)
            else:
                create_recurring_tracker(pattern, transactions, avg_frequency)
                
    except Exception as e:
        logger.error(f"Error analyzing pattern {pattern}: {e}")

def create_recurring_tracker(pattern, transactions, frequency_days):
    """Create a new recurring tracker"""
    try:
        latest_transaction = transactions[-1]
        
        # Create amount history
        amount_history = []
        for transaction in transactions[-5:]:  # Keep last 5 amounts
            amount_history.append({
                'date': transaction.date.isoformat(),
                'amount': transaction.amount
            })
        
        tracker = RecurringTracker(
            description_pattern=pattern,
            expected_amount=latest_transaction.amount,
            amount_history=json.dumps(amount_history),
            frequency_days=int(frequency_days),
            last_occurrence=latest_transaction.date,
            status='active'
        )
        
        db.session.add(tracker)
        db.session.commit()
        
        logger.info(f"Created recurring tracker for: {pattern}")
        
    except Exception as e:
        logger.error(f"Error creating recurring tracker: {e}")
        db.session.rollback()

def update_recurring_tracker(tracker, transaction, frequency_days):
    """Update existing recurring tracker with new transaction"""
    try:
        # Check for price changes
        amount_change_percent = 0
        if tracker.expected_amount != 0:
            amount_change_percent = abs(transaction.amount - tracker.expected_amount) / abs(tracker.expected_amount)
        
        # Update amount history
        history = json.loads(tracker.amount_history or "[]")
        history.append({
            'date': transaction.date.isoformat(),
            'amount': transaction.amount
        })
        
        # Keep only last 10 entries
        tracker.amount_history = json.dumps(history[-10:])
        
        # Update tracker fields
        tracker.last_occurrence = transaction.date
        tracker.frequency_days = int(frequency_days)
        tracker.updated_at = datetime.utcnow()
        
        # Check for price change
        if amount_change_percent > tracker.variance_threshold:
            tracker.status = 'price_changed'
            tracker.expected_amount = transaction.amount
            logger.info(f"Price change detected for {tracker.description_pattern}: {amount_change_percent:.1%}")
        else:
            tracker.status = 'active'
            tracker.expected_amount = transaction.amount
        
        db.session.commit()
        
    except Exception as e:
        logger.error(f"Error updating recurring tracker: {e}")
        db.session.rollback()

def check_missing_recurring_payments():
    """Check for missing recurring payments and update status"""
    try:
        active_trackers = RecurringTracker.query.filter_by(status='active').all()
        today = datetime.now().date()
        
        for tracker in active_trackers:
            days_since_last = (today - tracker.last_occurrence).days
            expected_days = tracker.frequency_days
            
            # Consider missing if 1.5x the expected frequency has passed
            if days_since_last > (expected_days * 1.5):
                if days_since_last > (expected_days * 3):
                    tracker.status = 'stopped'
                    logger.info(f"Marked as stopped: {tracker.description_pattern}")
                else:
                    tracker.status = 'missing'
                    logger.info(f"Marked as missing: {tracker.description_pattern}")
                
                tracker.updated_at = datetime.utcnow()
                db.session.commit()
                
    except Exception as e:
        logger.error(f"Error checking missing payments: {e}")
        db.session.rollback()

def parse_csv_file(file_content):
    """Parse CSV file and extract transactions"""
    transactions = []
    lines = file_content.decode('utf-8').strip().split('\n')
    
    # Try to detect CSV format
    sniffer = csv.Sniffer()
    delimiter = sniffer.sniff(lines[0]).delimiter
    
    reader = csv.DictReader(lines, delimiter=delimiter)
    
    for row in reader:
        # Try common column names
        date_col = None
        amount_col = None
        desc_col = None
        
        for key in row.keys():
            key_lower = key.lower()
            if any(word in key_lower for word in ['date', 'transaction date', 'posting date']):
                date_col = key
            elif any(word in key_lower for word in ['amount', 'value', 'debit', 'credit']):
                amount_col = key
            elif any(word in key_lower for word in ['description', 'memo', 'details', 'merchant']):
                desc_col = key
        
        if date_col and amount_col and desc_col:
            try:
                # Parse date
                date_str = row[date_col]
                date_obj = pd.to_datetime(date_str).date()
                
                # Parse amount
                amount_str = str(row[amount_col]).replace(',', '').replace('£', '').replace('$', '')
                amount = float(amount_str)
                
                transactions.append({
                    'date': date_obj,
                    'amount': amount,
                    'description': row[desc_col],
                    'source_row': str(row)
                })
            except (ValueError, TypeError) as e:
                logger.warning(f"Skipping invalid row: {row} - {e}")
                continue
    
    return transactions

def parse_txt_statement(content):
    """Parse .txt statement with clear Date, Description, Amount, Balance format"""
    transactions = []
    current_transaction = {}
    last_balance = None
    
    for line_num, line in enumerate(content.split('\n'), 1):
        line = line.strip()
        if not line:
            continue
            
        # Skip header lines
        if line.startswith('From:') or line.startswith('Account:') or 'XXXX' in line:
            continue
            
        # Parse each field
        if line.startswith('Date:'):
            # If we have a complete transaction, save it
            if current_transaction.get('date') and current_transaction.get('description') and current_transaction.get('amount') is not None:
                transactions.append(current_transaction)
            
            # Start new transaction
            date_str = line.replace('Date:', '').strip()
            try:
                date_obj = datetime.strptime(date_str, '%d/%m/%Y').date()
                current_transaction = {'date': date_obj}
            except ValueError:
                logger.warning(f"Invalid date format: {date_str}")
                current_transaction = {}
                
        elif line.startswith('Description:'):
            if current_transaction:
                description = line.replace('Description:', '').strip()
                current_transaction['description'] = description
                
        elif line.startswith('Amount:'):
            if current_transaction:
                amount_str = line.replace('Amount:', '').strip()
                try:
                    amount = float(amount_str)
                    current_transaction['amount'] = amount
                except ValueError:
                    logger.warning(f"Invalid amount format: {amount_str}")
                    
        elif line.startswith('Balance:'):
            if current_transaction:
                balance_str = line.replace('Balance:', '').strip()
                try:
                    balance = float(balance_str)
                    current_transaction['balance'] = balance
                    current_transaction['source_row'] = f"Line {line_num}"
                    last_balance = balance  # Track the last balance
                except ValueError:
                    logger.warning(f"Invalid balance format: {balance_str}")
    
    # Add the last transaction if complete
    if current_transaction.get('date') and current_transaction.get('description') and current_transaction.get('amount') is not None:
        transactions.append(current_transaction)
    
    # Sort transactions by date to find the most recent one
    if transactions:
        transactions.sort(key=lambda tx: tx['date'])
        most_recent_balance = transactions[-1].get('balance')  # Get balance from most recent transaction
        logger.info(f"Parsed {len(transactions)} transactions from .txt file")
        logger.info(f"Current account balance (from most recent transaction): £{most_recent_balance:.2f}")
        return transactions, most_recent_balance
    else:
        logger.info(f"Parsed {len(transactions)} transactions from .txt file")
        return transactions, None

def parse_generic_statement(lines):
    """Fallback parser for non-Santander formats"""
    # This contains the original parsing logic as backup
    # For now, return empty list - can be enhanced later if needed
    return []

def parse_txt_file(file_content):
    """Parse .txt file with clear Date, Description, Amount, Balance format"""
    try:
        # Decode content if it's bytes
        if isinstance(file_content, bytes):
            content = file_content.decode('utf-8', errors='ignore')
        else:
            content = file_content
            
        transactions, last_balance = parse_txt_statement(content)
        logger.info(f"Successfully parsed {len(transactions)} transactions from .txt file")
        return transactions, last_balance
        
    except Exception as e:
        logger.error(f"Error parsing .txt file: {e}")
        return [], None
# Removed Excel parsing - now using TXT files only

def get_historical_price_data(hours_back=24):
    """Get historical price data from database"""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(hours=hours_back)
        
        price_data = BitcoinPriceHistory.query.filter(
            BitcoinPriceHistory.timestamp >= start_date,
            BitcoinPriceHistory.timestamp <= end_date
        ).order_by(BitcoinPriceHistory.timestamp.asc()).all()
        
        if not price_data:
            return {'success': False, 'error': 'No historical data available'}
        
        data_points = []
        for record in price_data:
            data_points.append({
                'timestamp': record.timestamp.isoformat(),
                'price_gbp': record.price_gbp,
                'price_usd': record.price_usd,
                'volume': record.volume if hasattr(record, 'volume') else None
            })
        
        return {
            'success': True,
            'data': data_points,
            'hours_back': hours_back,
            'total_points': len(data_points)
        }
        
    except Exception as e:
        logger.error(f"Error fetching historical data: {e}")
        return {'success': False, 'error': str(e)}

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/personal-finance')
def personal_finance():
    return render_template('personal_finance.html')

@app.route('/bitcoin-tracker')
def bitcoin_tracker():
    return render_template('bitcoin_tracker.html')

@app.route('/finance-tracker')
def finance_tracker():
    return render_template('finance_tracker.html')

@app.route('/data-viewer')
def data_viewer():
    # Redirect to Bitcoin Tracker
    return redirect('/bitcoin-tracker')

@app.route('/price-monitor')
def price_monitor():
    # Redirect to Bitcoin Tracker
    return redirect('/bitcoin-tracker')

@app.route('/debug')
def debug():
    return render_template('debug.html')

@app.route('/api/current-price')
def current_price():
    """API endpoint for current BTC price"""
    price_data = get_current_btc_price()
    return jsonify(price_data)

@app.route('/api/historical-data', methods=['POST'])
def historical_data():
    """API endpoint for historical data"""
    try:
        data = request.json
        days = int(data.get('days', 365))
        
        historical = fetch_historical_data(days)
        return jsonify(historical)
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/dip-analysis', methods=['POST'])
def dip_analysis():
    """API endpoint for dip analysis"""
    try:
        data = request.json
        last_sell_price = float(data.get('last_sell_price', 0))
        target_dip_percent = float(data.get('target_dip_percent', 5))
        
        # Get current price
        current_price_data = get_current_btc_price()
        if not current_price_data['success']:
            return jsonify(current_price_data)
        
        current_price = current_price_data['price']
        
        # Calculate dip analysis
        if last_sell_price > 0:
            # Calculate current change from last sell price
            current_change = ((current_price - last_sell_price) / last_sell_price) * 100
            
            # Calculate target buy price (last sell price minus target dip)
            target_buy_price = last_sell_price * (1 - (target_dip_percent / 100))
            
            # Calculate how much more dip is needed
            additional_dip_needed = ((current_price - target_buy_price) / current_price) * 100
            
            # Status determination
            if current_price <= target_buy_price:
                status = "BUY_SIGNAL"
                status_text = "🟢 Target dip reached!"
            elif additional_dip_needed <= 1:
                status = "CLOSE_TO_TARGET"
                status_text = f"🟡 Very close to target (need {additional_dip_needed:.1f}% more dip)"
            elif additional_dip_needed <= 3:
                status = "APPROACHING_TARGET"
                status_text = f"🟠 Approaching target (need {additional_dip_needed:.1f}% more dip)"
            else:
                status = "FAR_FROM_TARGET"
                status_text = f"🔴 Still far from target (need {additional_dip_needed:.1f}% more dip)"
            
            return jsonify({
                'success': True,
                'current_price': current_price,
                'last_sell_price': last_sell_price,
                'target_buy_price': target_buy_price,
                'current_change_percent': current_change,
                'target_dip_percent': target_dip_percent,
                'additional_dip_needed': max(0, additional_dip_needed),
                'status': status,
                'status_text': status_text,
                'pair': current_price_data['pair'],
                'timestamp': current_price_data['timestamp']
            })
        else:
            return jsonify({
                'success': True,
                'current_price': current_price,
                'pair': current_price_data['pair'],
                'timestamp': current_price_data['timestamp'],
                'message': 'Enter your last sell price to see dip analysis'
            })
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/optimal-strategy', methods=['GET'])
def get_optimal_strategy():
    """API endpoint for automated optimal strategy recommendations"""
    try:
        logger.info("Received optimal strategy request")
        
        # Create BTCBacktester instance with default settings
        backtester = BTCBacktester(
            lookback_days=365,  # Will be overridden for each timeframe
            investment_value=1000,
            transaction_fee_percent=0.1
        )
        
        # Get automated optimal strategies
        strategies = backtester.get_automated_optimal_strategies()
        
        if strategies is None:
            return jsonify({'error': 'Failed to calculate optimal strategies. Could not fetch market data.'}), 500
        
        # Format response for frontend
        formatted_response = {
            'success': True,
            'strategies': strategies,
            'last_updated': datetime.now().isoformat(),
            'cache_duration': 300  # 5 minutes cache recommendation
        }
        
        return jsonify(formatted_response)
        
    except Exception as e:
        logger.error(f"Error in optimal strategy route: {str(e)}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/backtest', methods=['POST'])
def run_backtest():
    try:
        data = request.json
        logger.info(f"Received backtest request: {data}")
        
        # Input validation
        lookback_days = int(data.get('lookback_days', 365))
        investment_value = float(data.get('investment_value', 1000))
        transaction_fee_percent = float(data.get('transaction_fee_percent', 0.1))
        
        # Handle both percentage and price input modes
        input_mode = data.get('input_mode', 'percentage')
        
        if input_mode == 'price':
            # Convert price inputs to percentages
            buy_price = float(data.get('buy_price_gbp', 0))
            sell_price = float(data.get('sell_price_gbp', 0))
            current_price = float(data.get('current_btc_price', 0))
            
            if buy_price <= 0 or sell_price <= 0 or current_price <= 0:
                return jsonify({'error': 'All price values must be positive when using price mode.'}), 400
            
            if sell_price <= buy_price:
                return jsonify({'error': 'Sell price must be higher than buy price.'}), 400
            
            # Convert to percentages
            buy_dip_percent = price_to_percentage(buy_price, current_price, False)
            sell_gain_percent = price_to_percentage(sell_price, buy_price, True)
            
            logger.info(f"Price mode: £{buy_price} -> £{sell_price} = {buy_dip_percent}% dip, {sell_gain_percent}% gain")
        else:
            # Use percentage inputs directly
            buy_dip_percent = float(data.get('buy_dip_percent', 5))
            sell_gain_percent = float(data.get('sell_gain_percent', 10))
        
        # Validate input ranges
        if lookback_days < 1:
            return jsonify({'error': 'Lookback days must be at least 1 day.'}), 400
        if lookback_days > 1825:  # 5 years
            return jsonify({'error': 'Lookback days cannot exceed 5 years (1825 days).'}), 400
        if investment_value <= 0:
            return jsonify({'error': 'Investment value must be positive.'}), 400
        if buy_dip_percent <= 0 or sell_gain_percent <= 0:
            return jsonify({'error': 'Buy and sell percentages must be positive.'}), 400
        if sell_gain_percent < buy_dip_percent:
            return jsonify({'error': 'Sell percentage must be greater than or equal to buy percentage.'}), 400
        
        backtester = BTCBacktester(
            lookback_days=lookback_days,
            investment_value=investment_value,
            buy_dip_percent=buy_dip_percent,
            sell_gain_percent=sell_gain_percent,
            transaction_fee_percent=transaction_fee_percent
        )
        
        results = backtester.backtest()
        
        if results is None:
            # Enhanced error message for short timeframes
            if lookback_days < 7:
                logger.error(f"Backtest failed for short timeframe: {lookback_days} days")
                return jsonify({'error': f'Insufficient market data for {lookback_days} day analysis. Very short timeframes may not have enough trading activity for meaningful results. Try 7+ days.'}), 500
            else:
                logger.error("Backtest returned None")
                return jsonify({'error': 'Failed to fetch market data from Kraken. Please try again in a few minutes.'}), 500
        
        chart_json = backtester.create_chart(results)
        
        return jsonify({
            'success': True,
            'results': {
                'final_value': round(results['final_value'], 2),
                'total_return': round(results['total_return'], 2),
                'buy_hold_return': round(results['buy_hold_return'], 2),
                'buy_hold_final_value': round(results['buy_hold_final_value'], 2),
                'num_trades': results['num_trades'],
                'recent_trades': results['trades'],
                'final_position': results['final_position'],
                'total_fees_paid': round(results['total_fees_paid'], 2),
                'max_drawdown': round(results['max_drawdown'], 2),
                'buy_hold_max_drawdown': round(results['buy_hold_max_drawdown'], 2)
            },
            'chart': chart_json
        })
        
    except Exception as e:
        logger.error(f"Error in backtest route: {str(e)}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/optimize', methods=['POST'])
def optimize_parameters():
    try:
        data = request.json
        logger.info(f"Received optimization request: {data}")
        
        # Input validation
        lookback_days = int(data.get('lookback_days', 365))
        investment_value = float(data.get('investment_value', 1000))
        transaction_fee_percent = float(data.get('transaction_fee_percent', 0.1))
        
        # Validate input ranges
        if lookback_days < 1:
            return jsonify({'error': 'Lookback days must be at least 1 day.'}), 400
        if lookback_days > 1825:  # 5 years
            return jsonify({'error': 'Lookback days cannot exceed 5 years (1825 days).'}), 400
        if investment_value <= 0:
            return jsonify({'error': 'Investment value must be positive.'}), 400
        
        backtester = BTCBacktester(
            lookback_days=lookback_days,
            investment_value=investment_value,
            transaction_fee_percent=transaction_fee_percent
        )
        
        # Handle both percentage and price input modes for optimization ranges
        input_mode = data.get('opt_input_mode', 'percentage')
        
        if input_mode == 'price':
            # Convert price ranges to percentage ranges
            buy_price_min = float(data.get('buy_price_min', 0))
            buy_price_max = float(data.get('buy_price_max', 0))
            sell_price_min = float(data.get('sell_price_min', 0))
            sell_price_max = float(data.get('sell_price_max', 0))
            current_price = float(data.get('current_btc_price', 0))
            
            if buy_price_min <= 0 or buy_price_max <= 0 or sell_price_min <= 0 or sell_price_max <= 0 or current_price <= 0:
                return jsonify({'error': 'All price values must be positive when using price mode.'}), 400
            
            if buy_price_min >= buy_price_max or sell_price_min >= sell_price_max:
                return jsonify({'error': 'Price range maximums must be greater than minimums.'}), 400
            
            # Convert price ranges to percentage ranges
            buy_max = price_to_percentage(buy_price_min, current_price, False)  # Lower price = higher dip %
            buy_min = price_to_percentage(buy_price_max, current_price, False)  # Higher price = lower dip %
            
            # For sell prices, use buy_price_min as reference (assuming worst-case buy scenario)
            sell_min = price_to_percentage(sell_price_min, buy_price_max, True)
            sell_max = price_to_percentage(sell_price_max, buy_price_min, True)
            
            logger.info(f"Price mode optimization: Buy £{buy_price_min}-£{buy_price_max} = {buy_min}%-{buy_max}% dip")
            logger.info(f"Price mode optimization: Sell £{sell_price_min}-£{sell_price_max} = {sell_min}%-{sell_max}% gain")
        else:
            # Use percentage inputs directly
            buy_min = float(data.get('buy_min', 1))
            buy_max = float(data.get('buy_max', 15))
            sell_min = float(data.get('sell_min', 1))
            sell_max = float(data.get('sell_max', 25))
        
        step = float(data.get('step', 0.5))
        
        optimization_results = backtester.optimize_parameters(
            buy_range=(buy_min, buy_max),
            sell_range=(sell_min, sell_max),
            step=step
        )
        
        if optimization_results is None:
            # Enhanced error message for short timeframes
            if lookback_days < 7:
                return jsonify({'error': f'Insufficient market data for {lookback_days} day optimization. Very short timeframes may not have enough trading activity for meaningful optimization. Try 7+ days.'}), 500
            else:
                return jsonify({'error': 'Failed to optimize parameters. Could not fetch market data.'}), 500
        
        return jsonify({
            'success': True,
            'optimization': optimization_results
        })
        
    except Exception as e:
        logger.error(f"Error in optimization route: {str(e)}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/optimize_multi_timeframe', methods=['POST'])
def optimize_parameters_multi_timeframe():
    try:
        data = request.json
        logger.info(f"Received multi-timeframe optimization request: {data}")
        
        backtester = BTCBacktester(
            lookback_days=int(data.get('lookback_days', 365)),  # This will be overridden for each timeframe
            investment_value=float(data.get('investment_value', 1000)),
            transaction_fee_percent=float(data.get('transaction_fee_percent', 0.1))
        )
        
        # Get optimization ranges
        buy_min = float(data.get('buy_min', 1))
        buy_max = float(data.get('buy_max', 15))
        sell_min = float(data.get('sell_min', 1))
        sell_max = float(data.get('sell_max', 25))
        step = float(data.get('step', 0.5))
        
        optimization_results = backtester.optimize_parameters_multi_timeframe(
            buy_range=(buy_min, buy_max),
            sell_range=(sell_min, sell_max),
            step=step
        )
        
        if optimization_results is None:
            return jsonify({'error': 'Failed to optimize parameters across timeframes. Could not fetch market data.'}), 500
        
        return jsonify({
            'success': True,
            'multi_optimization': optimization_results
        })
        
    except Exception as e:
        logger.error(f"Error in multi-timeframe optimization route: {str(e)}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/test-data')
def test_data():
    try:
        url = "https://api.kraken.com/0/public/OHLC"
        params = {
            'pair': 'XXBTZGBP',
            'interval': 1440,
            'count': 5
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if 'error' in data and data['error']:
            return jsonify({'success': False, 'error': f'Kraken API error: {data["error"]}'})
        
        pair_data = None
        for key in data['result'].keys():
            if key != 'last':
                pair_data = data['result'][key]
                break
        
        if pair_data:
            latest_price = float(pair_data[-1][4])
            return jsonify({
                'success': True,
                'rows': len(pair_data),
                'latest_price': latest_price,
                'source': 'Kraken API'
            })
        else:
            return jsonify({'success': False, 'error': 'No price data found'})
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# Personal Finance API Endpoints
@app.route('/api/finance/investments', methods=['GET', 'POST'])
def finance_investments():
    if request.method == 'GET':
        # Get all investments
        investments = Investment.query.order_by(Investment.created_at.desc()).all()
        
        # Get user preferences for ordering
        prefs = UserPreferences.query.first()
        investment_order = []
        if prefs and prefs.investment_order:
            investment_order = json.loads(prefs.investment_order)
        
        # Apply custom order if exists
        if investment_order:
            ordered_investments = []
            unordered_investments = []
            
            for inv_id in investment_order:
                inv = next((i for i in investments if i.id == inv_id), None)
                if inv:
                    ordered_investments.append(inv)
            
            # Add any investments not in the order
            for inv in investments:
                if inv.id not in investment_order:
                    unordered_investments.append(inv)
            
            investments = ordered_investments + unordered_investments
        
        return jsonify({
            'success': True,
            'data': [{
                'id': inv.id,
                'name': inv.name,
                'startDate': inv.start_date.isoformat(),
                'startInvestment': inv.start_investment,
                'currentValue': inv.current_value
            } for inv in investments]
        })
    
    elif request.method == 'POST':
        # Create new investment
        try:
            data = request.json
            
            investment = Investment(
                name=data['name'],
                start_date=datetime.strptime(data['startDate'], '%Y-%m-%d').date(),
                start_investment=float(data['startInvestment']),
                current_value=float(data['currentValue'])
            )
            
            db.session.add(investment)
            db.session.commit()
            
            return jsonify({
                'success': True,
                'data': {
                    'id': investment.id,
                    'name': investment.name,
                    'startDate': investment.start_date.isoformat(),
                    'startInvestment': investment.start_investment,
                    'currentValue': investment.current_value
                }
            })
            
        except Exception as e:
            db.session.rollback()
            return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/finance/investments/<int:investment_id>', methods=['PUT', 'DELETE'])
def finance_investment_detail(investment_id):
    investment = Investment.query.get_or_404(investment_id)
    
    if request.method == 'PUT':
        # Update investment
        try:
            data = request.json
            
            if 'currentValue' in data:
                new_value = float(data['currentValue'])
                investment.current_value = new_value
                # Automatically update monthly and yearly records with new current value
                update_current_monthly_record(investment.id, new_value)
                update_current_yearly_record(investment.id, new_value)
            if 'name' in data:
                investment.name = data['name']
            if 'startInvestment' in data:
                investment.start_investment = float(data['startInvestment'])
            if 'startDate' in data:
                investment.start_date = datetime.strptime(data['startDate'], '%Y-%m-%d').date()
            
            db.session.commit()
            
            return jsonify({
                'success': True,
                'data': {
                    'id': investment.id,
                    'name': investment.name,
                    'startDate': investment.start_date.isoformat(),
                    'startInvestment': investment.start_investment,
                    'currentValue': investment.current_value
                }
            })
            
        except Exception as e:
            db.session.rollback()
            return jsonify({'success': False, 'error': str(e)}), 400
    
    elif request.method == 'DELETE':
        # Delete investment and all related records
        try:
            db.session.delete(investment)
            db.session.commit()
            
            return jsonify({'success': True})
            
        except Exception as e:
            db.session.rollback()
            return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/finance/monthly-records', methods=['GET', 'POST'])
def finance_monthly_records():
    if request.method == 'GET':
        # Get all monthly records
        records = MonthlyRecord.query.join(Investment).order_by(MonthlyRecord.year.desc(), MonthlyRecord.month.desc()).all()
        
        # Get current month and year for indicator
        now = datetime.now()
        current_year = now.year
        current_month = now.month
        
        return jsonify({
            'success': True,
            'data': [{
                'id': record.id,
                'investmentId': record.investment_id,
                'investmentName': record.investment_ref.name,
                'year': record.year,
                'month': record.month,
                'value': record.value,
                'valueDisplay': f"{record.value}*" if (record.year == current_year and record.month == current_month) else str(record.value),
                'isCurrent': record.year == current_year and record.month == current_month,
                'notes': record.notes or '',
                'date': record.date.isoformat()
            } for record in records]
        })
    
    elif request.method == 'POST':
        # Create or update monthly record
        try:
            data = request.json
            
            # Check if record already exists
            existing_record = MonthlyRecord.query.filter_by(
                investment_id=data['investmentId'],
                year=data['year'],
                month=data['month']
            ).first()
            
            if existing_record:
                # Update existing record
                existing_record.value = float(data['value'])
                existing_record.notes = data.get('notes', '')
                existing_record.date = datetime.strptime(f"{data['year']}-{data['month']:02d}-01", '%Y-%m-%d').date()
                record = existing_record
            else:
                # Create new record
                record = MonthlyRecord(
                    investment_id=data['investmentId'],
                    year=data['year'],
                    month=data['month'],
                    value=float(data['value']),
                    notes=data.get('notes', ''),
                    date=datetime.strptime(f"{data['year']}-{data['month']:02d}-01", '%Y-%m-%d').date()
                )
                db.session.add(record)
            
            db.session.commit()
            
            return jsonify({
                'success': True,
                'data': {
                    'id': record.id,
                    'investmentId': record.investment_id,
                    'investmentName': record.investment_ref.name,
                    'year': record.year,
                    'month': record.month,
                    'value': record.value,
                    'notes': record.notes or '',
                    'date': record.date.isoformat()
                }
            })
            
        except Exception as e:
            db.session.rollback()
            return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/finance/monthly-records/<int:record_id>', methods=['DELETE'])
def finance_monthly_record_detail(record_id):
    record = MonthlyRecord.query.get_or_404(record_id)
    
    try:
        db.session.delete(record)
        db.session.commit()
        
        return jsonify({'success': True})
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/finance/yearly-records', methods=['GET', 'POST'])
def finance_yearly_records():
    if request.method == 'GET':
        # Get all yearly records
        records = YearlyRecord.query.join(Investment).order_by(YearlyRecord.year.desc()).all()
        
        # Get current year for indicator
        now = datetime.now()
        current_year = now.year
        
        return jsonify({
            'success': True,
            'data': [{
                'id': record.id,
                'investmentId': record.investment_id,
                'investmentName': record.investment_ref.name,
                'year': record.year,
                'value': record.value,
                'valueDisplay': f"{record.value}*" if record.year == current_year else str(record.value),
                'isCurrent': record.year == current_year,
                'notes': record.notes or '',
                'date': record.date.isoformat()
            } for record in records]
        })
    
    elif request.method == 'POST':
        # Create or update yearly record
        try:
            data = request.json
            
            # Check if record already exists
            existing_record = YearlyRecord.query.filter_by(
                investment_id=data['investmentId'],
                year=data['year']
            ).first()
            
            if existing_record:
                # Update existing record
                existing_record.value = float(data['value'])
                existing_record.notes = data.get('notes', '')
                existing_record.date = datetime.strptime(f"{data['year']}-01-01", '%Y-%m-%d').date()
                record = existing_record
            else:
                # Create new record
                record = YearlyRecord(
                    investment_id=data['investmentId'],
                    year=data['year'],
                    value=float(data['value']),
                    notes=data.get('notes', ''),
                    date=datetime.strptime(f"{data['year']}-01-01", '%Y-%m-%d').date()
                )
                db.session.add(record)
            
            db.session.commit()
            
            return jsonify({
                'success': True,
                'data': {
                    'id': record.id,
                    'investmentId': record.investment_id,
                    'investmentName': record.investment_ref.name,
                    'year': record.year,
                    'value': record.value,
                    'notes': record.notes or '',
                    'date': record.date.isoformat()
                }
            })
            
        except Exception as e:
            db.session.rollback()
            return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/finance/yearly-records/<int:record_id>', methods=['DELETE'])
def finance_yearly_record_detail(record_id):
    record = YearlyRecord.query.get_or_404(record_id)
    
    try:
        db.session.delete(record)
        db.session.commit()
        
        return jsonify({'success': True})
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/finance/trigger-monthly-snapshot', methods=['POST'])
def trigger_monthly_snapshot():
    """Manual endpoint to test monthly snapshot creation"""
    try:
        result = advance_to_next_month()
        
        if result:
            return jsonify({
                'success': True,
                'message': 'Monthly snapshot process completed successfully'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Monthly snapshot process failed - check logs for details'
            }), 500
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Error triggering monthly snapshot: {str(e)}'
        }), 500

@app.route('/api/finance/trigger-yearly-snapshot', methods=['POST'])
def trigger_yearly_snapshot():
    """Manual endpoint to test yearly snapshot creation"""
    try:
        result = advance_to_next_year()
        
        if result:
            return jsonify({
                'success': True,
                'message': 'Yearly snapshot process completed successfully'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Yearly snapshot process failed - check logs for details'
            }), 500
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Error triggering yearly snapshot: {str(e)}'
        }), 500

@app.route('/api/finance/scheduler-status', methods=['GET'])
def scheduler_status():
    """Get scheduler status and next job run times"""
    try:
        jobs = []
        for job in scheduler.get_jobs():
            jobs.append({
                'id': job.id,
                'name': job.name,
                'next_run_time': job.next_run_time.isoformat() if job.next_run_time else None,
                'func': str(job.func)
            })
        
        return jsonify({
            'success': True,
            'scheduler_state': str(scheduler.state),
            'jobs': jobs
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Error getting scheduler status: {str(e)}'
        }), 500

@app.route('/api/finance/preferences', methods=['GET', 'POST'])
def finance_preferences():
    if request.method == 'GET':
        # Get user preferences
        prefs = UserPreferences.query.first()
        
        if prefs:
            return jsonify({
                'success': True,
                'data': {
                    'widgetOrder': json.loads(prefs.widget_order) if prefs.widget_order else ["investment", "yearly", "monthly"],
                    'investmentOrder': json.loads(prefs.investment_order) if prefs.investment_order else []
                }
            })
        else:
            return jsonify({
                'success': True,
                'data': {
                    'widgetOrder': ["investment", "yearly", "monthly"],
                    'investmentOrder': []
                }
            })
    
    elif request.method == 'POST':
        # Update user preferences
        try:
            data = request.json
            
            prefs = UserPreferences.query.first()
            if not prefs:
                prefs = UserPreferences()
                db.session.add(prefs)
            
            if 'widgetOrder' in data:
                prefs.widget_order = json.dumps(data['widgetOrder'])
            if 'investmentOrder' in data:
                prefs.investment_order = json.dumps(data['investmentOrder'])
            
            prefs.updated_at = datetime.utcnow()
            db.session.commit()
            
            return jsonify({'success': True})
            
        except Exception as e:
            db.session.rollback()
            return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/finance/backup', methods=['GET'])
def finance_backup():
    """Download database backup"""
    try:
        # Get the current database file path
        db_uri = app.config['SQLALCHEMY_DATABASE_URI']
        if not db_uri.startswith('sqlite:///'):
            return jsonify({'success': False, 'error': 'Backup only supported for SQLite databases'})
        
        # Extract database file path
        if db_uri.startswith('sqlite:////'):
            # Absolute path
            db_path = db_uri[10:]
        else:
            # Relative path
            db_path = db_uri[10:]
            
        if not os.path.exists(db_path):
            return jsonify({'success': False, 'error': 'Database file not found'})
        
        # Create timestamped filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_filename = f'finance_backup_{timestamp}.db'
        
        # Create a temporary copy for download
        temp_dir = tempfile.gettempdir()
        temp_backup_path = os.path.join(temp_dir, backup_filename)
        shutil.copy2(db_path, temp_backup_path)
        
        # Send file for download
        return send_file(
            temp_backup_path,
            as_attachment=True,
            download_name=backup_filename,
            mimetype='application/octet-stream'
        )
        
    except Exception as e:
        logger.error(f"Error creating backup: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/finance/restore', methods=['POST'])
def finance_restore():
    """Restore database from uploaded file"""
    try:
        if 'backup_file' not in request.files:
            return jsonify({'success': False, 'error': 'No backup file provided'})
        
        file = request.files['backup_file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'})
        
        if not file.filename.endswith('.db'):
            return jsonify({'success': False, 'error': 'Invalid file type. Please upload a .db file'})
        
        # Get current database path
        db_uri = app.config['SQLALCHEMY_DATABASE_URI']
        if not db_uri.startswith('sqlite:///'):
            return jsonify({'success': False, 'error': 'Restore only supported for SQLite databases'})
        
        if db_uri.startswith('sqlite:////'):
            db_path = db_uri[10:]
        else:
            db_path = db_uri[10:]
        
        # Create backup of current database
        if os.path.exists(db_path):
            backup_current = f"{db_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            shutil.copy2(db_path, backup_current)
            logger.info(f"Current database backed up to: {backup_current}")
        
        # Save uploaded file as new database
        file.save(db_path)
        
        # Reinitialize database connection
        db.engine.dispose()
        
        return jsonify({
            'success': True, 
            'message': 'Database restored successfully. Please refresh the page to see updated data.'
        })
        
    except Exception as e:
        logger.error(f"Error restoring backup: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

# Bitcoin Trading Tracker API Endpoints
@app.route('/api/bitcoin/trades', methods=['GET', 'POST'])
def bitcoin_trades():
    if request.method == 'GET':
        # Get all bitcoin trades
        try:
            # Use raw SQL to avoid date parsing issues
            from sqlalchemy import text
            
            # Check if duration_minutes column exists
            has_duration_column = check_duration_column_exists()
            
            # Check if final_value_gbp column exists
            try:
                # Try with final_value_gbp column first
                result = db.session.execute(text("""
                    SELECT id, status, date, type, initial_investment_gbp, 
                           btc_buy_price, btc_sell_price, profit, fee, btc_amount,
                           created_at, updated_at, final_value_gbp
                    FROM bitcoin_trade
                    ORDER BY date DESC
                """))
                
                trades_data = []
                for row in result:
                    trades_data.append({
                        'id': row[0],
                        'status': row[1],
                        'date': row[2],
                        'type': row[3],
                        'initial_investment_gbp': row[4],
                        'btc_buy_price': row[5],
                        'btc_sell_price': row[6],
                        'profit': row[7],
                        'fee': row[8],
                        'btc_amount': row[9],
                        'created_at': (str(row[10]) + 'Z') if row[10] else None,
                        'updated_at': (str(row[11]) + 'Z') if row[11] else None,
                        'final_value_gbp': row[12],
                        'duration_minutes': None  # Will be calculated on frontend
                    })
                    
            except Exception as e:
                # Fallback to query without final_value_gbp column if it doesn't exist
                if 'no such column' in str(e).lower():
                    result = db.session.execute(text("""
                        SELECT id, status, date, type, initial_investment_gbp, 
                               btc_buy_price, btc_sell_price, profit, fee, btc_amount,
                               created_at, updated_at
                        FROM bitcoin_trade
                        ORDER BY date DESC
                    """))
                    
                    trades_data = []
                    for row in result:
                        trades_data.append({
                            'id': row[0],
                            'status': row[1],
                            'date': row[2],
                            'type': row[3],
                            'initial_investment_gbp': row[4],
                            'btc_buy_price': row[5],
                            'btc_sell_price': row[6],
                            'profit': row[7],
                            'fee': row[8],
                            'btc_amount': row[9],
                            'created_at': (str(row[10]) + 'Z') if row[10] else None,
                            'updated_at': (str(row[11]) + 'Z') if row[11] else None,
                            'final_value_gbp': None,  # Default value for missing column
                            'duration_minutes': None  # Will be calculated on frontend
                        })
                else:
                    raise e
            
            return jsonify({
                'success': True,
                'data': trades_data
            })
            
        except Exception as e:
            logger.error(f"Error loading bitcoin trades: {str(e)}")
            return jsonify({'success': False, 'error': str(e)}), 500
    
    elif request.method == 'POST':
        # Create new bitcoin trade
        try:
            data = request.json
            
            # Check if duration_minutes column exists
            has_duration_column = check_duration_column_exists()
            
            # Handle both old and new field names for backward compatibility
            initial_investment = float(data.get('initial_investment_gbp', data.get('gbp_investment', 0)))
            btc_buy_price = float(data.get('btc_buy_price', data.get('btc_value', 0)))
            btc_amount = initial_investment / btc_buy_price
            
            # Check if all fields are provided to determine status and calculate profit
            btc_sell_price = data.get('btc_sell_price', data.get('crypto_value'))
            fee = data.get('fee')
            final_value_gbp = data.get('final_value_gbp')
            
            # Priority 1: Use final_value_gbp if provided
            if final_value_gbp:
                final_value_gbp = float(final_value_gbp)
                fee = float(fee) if fee is not None else 0
                profit = final_value_gbp - initial_investment - fee
                status = 'Closed'
            # Priority 2: Use traditional calculation
            elif btc_sell_price and fee is not None:
                # All fields provided - calculate profit and set status to Closed
                btc_sell_price = float(btc_sell_price)
                fee = float(fee)
                
                # Calculate profit: (BTC amount * sell price) - initial investment - fee
                gross_return = btc_amount * btc_sell_price
                profit = gross_return - initial_investment - fee
                status = 'Closed'
                final_value_gbp = None
                
                # Calculate duration in minutes from trade date to now
                trade_date = datetime.strptime(data['date'], '%Y-%m-%d')
                current_time = datetime.utcnow()
                duration_delta = current_time - trade_date
                duration_minutes = int(duration_delta.total_seconds() / 60)
            else:
                # Missing fields - set status to Open
                btc_sell_price = None
                fee = None
                profit = None
                status = 'Open'
                duration_minutes = None
                final_value_gbp = None
            
            # Create trade object with or without duration_minutes based on column existence
            trade_data = {
                'status': status,
                'date': datetime.strptime(data['date'], '%Y-%m-%d').date(),
                'type': data.get('type', 'BTC'),
                'initial_investment_gbp': initial_investment,
                'btc_buy_price': btc_buy_price,
                'btc_sell_price': btc_sell_price,
                'profit': profit,
                'fee': fee,
                'btc_amount': btc_amount,
                'final_value_gbp': final_value_gbp
            }
            
            trade = BitcoinTrade(**trade_data)
            
            # Set duration_minutes using safe method if needed
            if has_duration_column and duration_minutes is not None:
                trade.set_duration_minutes(duration_minutes)
            
            db.session.add(trade)
            db.session.commit()
            
            return jsonify({
                'success': True,
                'data': {
                    'id': trade.id,
                    'status': trade.status,
                    'date': format_date_safe(trade.date),
                    'type': trade.type,
                    'initial_investment_gbp': trade.initial_investment_gbp,
                    'btc_buy_price': trade.btc_buy_price,
                    'btc_sell_price': trade.btc_sell_price,
                    'profit': trade.profit,
                    'fee': trade.fee,
                    'btc_amount': trade.btc_amount,
                    'created_at': (trade.created_at.isoformat() + 'Z') if trade.created_at else None,
                    'updated_at': (trade.updated_at.isoformat() + 'Z') if trade.updated_at else None,
                    'duration_minutes': None  # Will be calculated on frontend
                }
            })
            
        except Exception as e:
            db.session.rollback()
            return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/bitcoin/trades/<int:trade_id>', methods=['PUT', 'DELETE'])
def bitcoin_trade_detail(trade_id):
    trade = BitcoinTrade.query.get_or_404(trade_id)
    
    if request.method == 'PUT':
        # Update bitcoin trade
        try:
            data = request.json
            
            # Update fields
            if 'date' in data:
                trade.date = datetime.strptime(data['date'], '%Y-%m-%d').date()
            if 'type' in data:
                trade.type = data['type']
            if 'initial_investment_gbp' in data:
                trade.initial_investment_gbp = float(data['initial_investment_gbp'])
            if 'btc_buy_price' in data:
                trade.btc_buy_price = float(data['btc_buy_price'])
            if 'btc_sell_price' in data:
                trade.btc_sell_price = float(data['btc_sell_price']) if data['btc_sell_price'] else None
            if 'fee' in data:
                trade.fee = float(data['fee']) if data['fee'] is not None else None
            if 'final_value_gbp' in data:
                # Only update if the column exists in the database
                if hasattr(trade, 'final_value_gbp'):
                    trade.final_value_gbp = float(data['final_value_gbp']) if data['final_value_gbp'] else None
            
            # Recalculate BTC amount
            if trade.initial_investment_gbp and trade.btc_buy_price:
                trade.btc_amount = trade.initial_investment_gbp / trade.btc_buy_price
            
            # Check if all fields are complete to calculate profit and update status
            # Priority 1: Use final_value_gbp if provided and column exists
            if hasattr(trade, 'final_value_gbp') and trade.final_value_gbp is not None and trade.initial_investment_gbp is not None:
                fee = trade.fee or 0
                trade.profit = trade.final_value_gbp - trade.initial_investment_gbp - fee
                trade.status = 'Closed'
            # Priority 2: Use traditional calculation
            elif trade.btc_sell_price is not None and trade.fee is not None and trade.btc_amount is not None:
                # Calculate profit
                gross_return = trade.btc_amount * trade.btc_sell_price
                trade.profit = gross_return - trade.initial_investment_gbp - trade.fee
                trade.status = 'Closed'
                
                # Calculate duration in minutes from created_at to now
                if trade.created_at:
                    current_time = datetime.utcnow()
                    duration_delta = current_time - trade.created_at
                    # Set duration using safe method
                    trade.set_duration_minutes(int(duration_delta.total_seconds() / 60))
            else:
                # Not all fields complete
                trade.profit = None
                trade.status = 'Open'
                # Set duration to None using safe method
                trade.set_duration_minutes(None)
            
            trade.updated_at = datetime.utcnow()
            db.session.commit()
            
            return jsonify({
                'success': True,
                'data': {
                    'id': trade.id,
                    'status': trade.status,
                    'date': format_date_safe(trade.date),
                    'type': trade.type,
                    'initial_investment_gbp': trade.initial_investment_gbp,
                    'btc_buy_price': trade.btc_buy_price,
                    'btc_sell_price': trade.btc_sell_price,
                    'profit': trade.profit,
                    'fee': trade.fee,
                    'btc_amount': trade.btc_amount,
                    'created_at': (trade.created_at.isoformat() + 'Z') if trade.created_at else None,
                    'updated_at': (trade.updated_at.isoformat() + 'Z') if trade.updated_at else None,
                    'duration_minutes': None  # Will be calculated on frontend
                }
            })
            
        except Exception as e:
            db.session.rollback()
            return jsonify({'success': False, 'error': str(e)}), 400
    
    elif request.method == 'DELETE':
        # Delete bitcoin trade
        try:
            db.session.delete(trade)
            db.session.commit()
            
            return jsonify({'success': True})
            
        except Exception as e:
            db.session.rollback()
            return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/bitcoin/current-price')
def bitcoin_current_price():
    """Get current BTC price for calculations"""
    try:
        price_data = get_current_btc_price()
        if price_data:
            return jsonify({
                'success': True,
                'price': price_data['price'],
                'currency': price_data['pair']
            })
        else:
            return jsonify({'success': False, 'error': 'Could not fetch current price'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/bitcoin/price-to-percentage', methods=['POST'])
def bitcoin_price_to_percentage():
    """Convert absolute price to percentage change"""
    try:
        data = request.json
        target_price = float(data.get('target_price', 0))
        current_price = float(data.get('current_price', 0))
        is_sell = data.get('is_sell', False)
        
        percentage = price_to_percentage(target_price, current_price, is_sell)
        
        return jsonify({
            'success': True,
            'percentage': percentage,
            'target_price': target_price,
            'current_price': current_price,
            'is_sell': is_sell
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/bitcoin/percentage-to-price', methods=['POST'])
def bitcoin_percentage_to_price():
    """Convert percentage to absolute price"""
    try:
        data = request.json
        percentage = float(data.get('percentage', 0))
        current_price = float(data.get('current_price', 0))
        is_sell = data.get('is_sell', False)
        
        target_price = percentage_to_price(percentage, current_price, is_sell)
        
        return jsonify({
            'success': True,
            'target_price': target_price,
            'percentage': percentage,
            'current_price': current_price,
            'is_sell': is_sell
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/bitcoin/profit-preview', methods=['POST'])
def bitcoin_profit_preview():
    """Calculate potential profit for price pair"""
    try:
        data = request.json
        buy_price = float(data.get('buy_price', 0))
        sell_price = float(data.get('sell_price', 0))
        investment_amount = float(data.get('investment_amount', 1000))
        fee_percent = float(data.get('fee_percent', 0.1))
        
        result = calculate_profit_preview(buy_price, sell_price, investment_amount, fee_percent)
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/bitcoin/historical-data')
def bitcoin_historical_data():
    """Get historical BTC price data from database"""
    try:
        hours_back = request.args.get('hours', 24, type=int)
        hours_back = min(hours_back, 8760)  # Maximum 1 year
        
        result = get_historical_price_data(hours_back)
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/bitcoin/fetch-historical-data', methods=['POST'])
def bitcoin_fetch_historical_data():
    """Fetch and store historical BTC price data"""
    try:
        hours_back = request.json.get('hours', 168) if request.json else 168  # Default 7 days
        hours_back = min(hours_back, 8760)  # Maximum 1 year
        
        result = store_historical_price_data(hours_back)
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/bitcoin/minute-data', methods=['POST'])
def bitcoin_minute_data():
    """Get minute-level BTC price data"""
    try:
        data = request.json or {}
        days = int(data.get('days', 7))
        limit = data.get('limit', None)
        
        # Limit days to prevent excessive data transfer
        days = min(days, 30)  # Maximum 30 days of minute data
        
        if limit:
            limit = min(limit, 10000)  # Maximum 10,000 records
        
        result = fetch_minute_data_for_viewer(days, limit)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error in minute data API: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/bitcoin/swing-analysis', methods=['POST'])
def bitcoin_swing_analysis():
    """Get swing analysis for specified periods"""
    try:
        data = request.json or {}
        period_hours = int(data.get('period_hours', 1))
        days_back = int(data.get('days_back', 7))
        
        # Validate parameters
        if period_hours not in [1, 3, 6, 9, 12, 24]:
            return jsonify({'success': False, 'error': 'period_hours must be 1, 3, 6, 9, 12, or 24'})
        
        days_back = min(days_back, 30)  # Maximum 30 days
        
        result = calculate_swing_analysis(period_hours, days_back)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error in swing analysis API: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/bitcoin/multi-swing-analysis', methods=['POST'])
def bitcoin_multi_swing_analysis():
    """Get swing analysis for all periods (1h, 3h, 6h, 9h)"""
    try:
        data = request.json or {}
        days_back = int(data.get('days_back', 7))
        
        days_back = min(days_back, 30)  # Maximum 30 days
        
        result = get_multi_period_swing_analysis(days_back)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error in multi-swing analysis API: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/bitcoin/store-minute-data', methods=['POST'])
def bitcoin_store_minute_data():
    """Manually fetch and store minute-level BTC price data"""
    try:
        data = request.json or {}
        minutes_back = int(data.get('minutes_back', 1440))  # Default 24 hours
        
        # Limit to prevent excessive API calls
        minutes_back = min(minutes_back, 10080)  # Maximum 7 days
        
        result = store_minute_price_data(minutes_back)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error storing minute data: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/bitcoin/minute-collection-status')
def bitcoin_minute_collection_status():
    """Get status of minute-level data collection"""
    try:
        # Get latest minute data record
        latest_record = BitcoinPriceHistoryMinute.query.order_by(
            BitcoinPriceHistoryMinute.timestamp.desc()
        ).first()
        
        # Get count of records in last 24 hours
        twenty_four_hours_ago = datetime.now() - timedelta(hours=24)
        recent_count = BitcoinPriceHistoryMinute.query.filter(
            BitcoinPriceHistoryMinute.timestamp >= twenty_four_hours_ago
        ).count()
        
        # Get total record count
        total_count = BitcoinPriceHistoryMinute.query.count()
        
        status = {
            'success': True,
            'latest_record': {
                'timestamp': latest_record.timestamp.isoformat() if latest_record else None,
                'price_gbp': latest_record.price_gbp if latest_record else None
            } if latest_record else None,
            'records_last_24h': recent_count,
            'total_records': total_count,
            'collection_active': recent_count > 0,
            'expected_records_24h': 1440  # 24 hours * 60 minutes
        }
        
        return jsonify(status)
    except Exception as e:
        logger.error(f"Error getting collection status: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/bitcoin/initialize-minute-data', methods=['POST'])
def bitcoin_initialize_minute_data():
    """Manually trigger initialization of historical minute-level data"""
    try:
        data = request.json or {}
        days_back = int(data.get('days_back', 30))
        
        # Validate days_back
        if days_back < 1:
            return jsonify({'success': False, 'error': 'Must specify at least 1 day'})
        if days_back > 30:
            return jsonify({'success': False, 'error': 'Cannot initialize more than 30 days due to API limits'})
        
        logger.info(f"Manual initialization requested for {days_back} days")
        result = initialize_historical_minute_data(days_back)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error in manual initialization: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/bitcoin/fill-data-gaps', methods=['POST'])
def bitcoin_fill_data_gaps():
    """Manually trigger data gap detection and filling"""
    try:
        data = request.json or {}
        max_gap_hours = int(data.get('max_gap_hours', 2))
        
        # Validate max_gap_hours
        if max_gap_hours < 1:
            return jsonify({'success': False, 'error': 'max_gap_hours must be at least 1'})
        if max_gap_hours > 24:
            return jsonify({'success': False, 'error': 'max_gap_hours cannot exceed 24'})
        
        result = detect_and_fill_data_gaps(max_gap_hours)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error in gap filling: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/bitcoin/cleanup-minute-data', methods=['POST'])
def bitcoin_cleanup_minute_data():
    """Manually trigger cleanup of old minute-level data"""
    try:
        data = request.json or {}
        days_to_keep = int(data.get('days_to_keep', 30))
        
        # Validate days_to_keep
        if days_to_keep < 7:
            return jsonify({'success': False, 'error': 'Must keep at least 7 days of data'})
        if days_to_keep > 90:
            return jsonify({'success': False, 'error': 'Cannot keep more than 90 days of minute data'})
        
        result = cleanup_old_minute_data(days_to_keep)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error in manual cleanup: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/bitcoin/archive-hourly-data', methods=['POST'])
def bitcoin_archive_hourly_data():
    """Manually trigger archiving of old hourly data"""
    try:
        data = request.json or {}
        days_to_keep = int(data.get('days_to_keep', 365))
        
        # Validate days_to_keep
        if days_to_keep < 30:
            return jsonify({'success': False, 'error': 'Must keep at least 30 days of hourly data'})
        if days_to_keep > 1095:  # 3 years
            return jsonify({'success': False, 'error': 'Cannot keep more than 3 years of hourly data'})
        
        result = archive_old_price_data(days_to_keep)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error in manual archiving: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/dashboard/summary')
def dashboard_summary():
    """Get unified dashboard summary combining personal finance and bitcoin data"""
    try:
        # Get personal finance data
        investments = Investment.query.all()
        
        # Calculate personal finance metrics
        total_invested = sum(inv.start_investment for inv in investments)
        current_value = sum(inv.current_value for inv in investments)
        total_profit = current_value - total_invested
        profit_percentage = (total_profit / total_invested * 100) if total_invested > 0 else 0.0
        
        # Find best and worst performers
        best_performer = None
        worst_performer = None
        if investments:
            performers = []
            for inv in investments:
                profit = inv.current_value - inv.start_investment
                profit_pct = (profit / inv.start_investment * 100) if inv.start_investment > 0 else 0.0
                performers.append({'name': inv.name, 'profit_pct': profit_pct})
            
            performers.sort(key=lambda x: x['profit_pct'])
            if performers:
                worst_performer = performers[0]
                best_performer = performers[-1]
        
        # Calculate average annual return
        avg_annual_return = 0.0
        if investments:
            total_return = 0.0
            total_weight = 0.0
            for inv in investments:
                profit = inv.current_value - inv.start_investment
                years_invested = (datetime.now().date() - inv.start_date).days / 365.25
                if years_invested > 0 and inv.start_investment > 0:
                    annual_return = (profit / inv.start_investment) / years_invested * 100
                    weight = inv.start_investment
                    total_return += annual_return * weight
                    total_weight += weight
            if total_weight > 0:
                avg_annual_return = total_return / total_weight
        
        # Get Bitcoin trading data
        btc_trades = BitcoinTrade.query.all()
        
        # Calculate Bitcoin metrics
        open_positions = len([t for t in btc_trades if t.status == 'Open'])
        btc_total_profit = sum(t.profit for t in btc_trades if t.profit is not None)
        total_fees = sum(t.fee for t in btc_trades if t.fee is not None)
        
        # Get current BTC price for open position estimates
        current_btc_price = None
        price_data = get_current_btc_price()
        if price_data:
            current_btc_price = price_data['price']
            
            # Add estimated profits for open positions
            for trade in btc_trades:
                if trade.status == 'Open' and trade.btc_amount and current_btc_price:
                    estimated_value = trade.btc_amount * current_btc_price
                    estimated_profit = estimated_value - trade.initial_investment_gbp
                    btc_total_profit += estimated_profit
        
        # Find best and worst Bitcoin trades
        btc_best_trade = None
        btc_worst_trade = None
        if btc_trades:
            completed_trades = [t for t in btc_trades if t.profit is not None]
            if completed_trades:
                best_trade = max(completed_trades, key=lambda x: x.profit)
                worst_trade = min(completed_trades, key=lambda x: x.profit)
                btc_best_trade = {'profit': best_trade.profit, 'date': best_trade.date.strftime('%Y-%m-%d')}
                btc_worst_trade = {'profit': worst_trade.profit, 'date': worst_trade.date.strftime('%Y-%m-%d')}
        
        # Calculate total time invested (earliest investment)
        time_invested_months = 0
        if investments:
            earliest_date = min(inv.start_date for inv in investments)
            time_invested_months = (datetime.now().date() - earliest_date).days / 30.44
        
        # Combined portfolio value
        combined_portfolio_value = current_value + btc_total_profit
        
        # Get yearly and monthly data for historical performance
        yearly_records = YearlyRecord.query.all()
        monthly_records = MonthlyRecord.query.all()
        
        # Calculate yearly profits for historical performance
        yearly_profits = {}
        for record in yearly_records:
            year = record.year
            investment = next((inv for inv in investments if inv.id == record.investment_id), None)
            if investment:
                if year not in yearly_profits:
                    yearly_profits[year] = 0
                
                # Calculate profit for this investment in this year
                profit = record.value - investment.start_investment
                yearly_profits[year] += profit
        
        # Calculate S&P 500 comparison using actual investment period
        sp500_comparison = None
        if investments and time_invested_months > 0:
            investment_period_years = time_invested_months / 12
            
            # Use different S&P 500 rates based on time period
            if investment_period_years <= 1:
                sp500_annual_rate = 8.5  # Short term average
            elif investment_period_years <= 3:
                sp500_annual_rate = 9.2  # Medium term average
            else:
                sp500_annual_rate = 10.0  # Long term average
            
            # Calculate total returns for comparison
            my_total_return = profit_percentage
            sp500_total_return = (pow(1 + (sp500_annual_rate / 100), investment_period_years) - 1) * 100
            
            sp500_comparison = {
                'my_return': my_total_return,
                'sp500_return': sp500_total_return,
                'outperforming': my_total_return >= sp500_total_return,
                'difference': my_total_return - sp500_total_return
            }
        
        # Analyze Bitcoin trading patterns for best strategy
        best_strategy = None
        if btc_trades:
            # Analyze completed trades to find best dip/gain patterns
            completed_trades = [t for t in btc_trades if t.profit is not None and t.btc_buy_price and t.btc_sell_price]
            
            if len(completed_trades) >= 3:  # Need at least 3 trades for analysis
                strategies = []
                
                # Test different buy/sell strategies
                for buy_dip in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
                    for sell_gain in [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.5, 10.0]:
                        # Calculate hypothetical performance for this strategy
                        total_profit = 0
                        trade_count = 0
                        
                        for trade in completed_trades:
                            # Calculate actual dip and gain percentages
                            if trade.btc_buy_price and trade.btc_sell_price:
                                gain_pct = ((trade.btc_sell_price - trade.btc_buy_price) / trade.btc_buy_price) * 100
                                
                                # If this trade fits our strategy criteria, count it
                                if gain_pct >= sell_gain * 0.8:  # Allow some tolerance
                                    total_profit += trade.profit or 0
                                    trade_count += 1
                        
                        if trade_count > 0:
                            avg_profit = total_profit / trade_count
                            strategies.append({
                                'buy_dip': buy_dip,
                                'sell_gain': sell_gain,
                                'avg_profit': avg_profit,
                                'trade_count': trade_count,
                                'total_profit': total_profit
                            })
                
                # Find the best strategy by total profit
                if strategies:
                    best_strategy = max(strategies, key=lambda x: x['total_profit'])
        
        return jsonify({
            'success': True,
            'personal_finance': {
                'total_invested': total_invested,
                'current_value': current_value,
                'total_profit': total_profit,
                'profit_percentage': profit_percentage,
                'avg_annual_return': avg_annual_return,
                'time_invested_months': time_invested_months,
                'best_performer': best_performer,
                'worst_performer': worst_performer,
                'investment_count': len(investments),
                'sp500_comparison': sp500_comparison,
                'yearly_profits': yearly_profits
            },
            'bitcoin_trading': {
                'current_price': current_btc_price,
                'open_positions': open_positions,
                'total_profit': btc_total_profit,
                'total_fees': total_fees,
                'best_trade': btc_best_trade,
                'worst_trade': btc_worst_trade,
                'total_trades': len(btc_trades),
                'best_strategy': best_strategy
            },
            'combined': {
                'total_portfolio_value': combined_portfolio_value,
                'analysis_tools': 8  # Personal Finance (3 tabs) + Bitcoin Tracker (5 tabs)
            }
        })
        
    except Exception as e:
        logger.error(f"Dashboard summary error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

# Finance Tracker API Endpoints
@app.route('/api/finance-tracker/categories', methods=['GET', 'POST'])
def finance_categories():
    if request.method == 'GET':
        categories = FinanceCategory.query.order_by(FinanceCategory.name).all()
        return jsonify({
            'success': True,
            'data': [{
                'id': cat.id,
                'name': cat.name,
                'color': cat.color,
                'description': cat.description or '',
                'budget': cat.budget,
                'transaction_count': len(cat.transactions)
            } for cat in categories]
        })
    
    elif request.method == 'POST':
        try:
            data = request.json
            category = FinanceCategory(
                name=data['name'],
                color=data.get('color', '#007bff'),
                description=data.get('description', ''),
                budget=data.get('budget')
            )
            db.session.add(category)
            db.session.commit()
            
            return jsonify({
                'success': True,
                'data': {
                    'id': category.id,
                    'name': category.name,
                    'color': category.color,
                    'description': category.description or '',
                    'budget': category.budget,
                    'transaction_count': 0
                }
            })
        except Exception as e:
            db.session.rollback()
            return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/finance-tracker/categories/<int:category_id>', methods=['PUT', 'DELETE'])
def finance_category_detail(category_id):
    category = FinanceCategory.query.get_or_404(category_id)
    
    if request.method == 'PUT':
        try:
            data = request.json
            category.name = data.get('name', category.name)
            category.color = data.get('color', category.color)
            category.description = data.get('description', category.description)
            category.budget = data.get('budget', category.budget)
            db.session.commit()
            
            return jsonify({
                'success': True,
                'data': {
                    'id': category.id,
                    'name': category.name,
                    'color': category.color,
                    'description': category.description or '',
                    'transaction_count': len(category.transactions)
                }
            })
        except Exception as e:
            db.session.rollback()
            return jsonify({'success': False, 'error': str(e)}), 400
    
    elif request.method == 'DELETE':
        try:
            db.session.delete(category)
            db.session.commit()
            return jsonify({'success': True})
        except Exception as e:
            db.session.rollback()
            return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/finance-tracker/upload', methods=['POST'])
def finance_upload():
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'success': False, 'error': 'Invalid file type. Please upload PDF files only.'}), 400
        
        filename = secure_filename(file.filename)
        file_content = file.read()
        
        # Parse TXT file
        transactions = []
        last_balance = None
        if filename.lower().endswith('.txt'):
            transactions, last_balance = parse_txt_file(file_content)
        else:
            return jsonify({'success': False, 'error': 'Only TXT files are supported'}), 400
        
        if not transactions:
            return jsonify({'success': False, 'error': 'No valid transactions found in file'}), 400
        
        # Get existing learning patterns for auto-categorization
        patterns = FinanceCategoryLearning.query.all()
        
        # Process transactions
        processed_transactions = []
        duplicates_found = 0
        
        for trans_data in transactions:
            # Calculate hash for duplicate detection
            hash_value = calculate_transaction_hash(trans_data)
            
            # Check for existing transaction
            existing = FinanceTransaction.query.filter_by(hash_value=hash_value).first()
            if existing:
                duplicates_found += 1
                continue
            
            # Predict category
            predicted_category = None
            confidence_score = 0.0
            if patterns:
                best_pattern, score = predict_category(trans_data['description'], patterns)
                if best_pattern:
                    predicted_category = best_pattern.category_id
                    confidence_score = score
            
            # Determine if validation is needed for this transaction
            needs_validation = (
                predicted_category is None or  # No category predicted
                confidence_score < 0.6 or      # Low confidence in prediction
                abs(trans_data['amount']) > 500  # Large amount (configurable threshold)
            )
            
            # Create transaction
            transaction = FinanceTransaction(
                date=trans_data['date'],
                amount=trans_data['amount'],
                description=trans_data['description'],
                month=trans_data['date'].month,
                year=trans_data['date'].year,
                source_file=filename,
                source_row=trans_data['source_row'],
                hash_value=hash_value,
                category_id=predicted_category,
                confidence_score=confidence_score,
                needs_validation=needs_validation
            )
            
            db.session.add(transaction)
            processed_transactions.append({
                'date': trans_data['date'].isoformat(),
                'amount': trans_data['amount'],
                'description': trans_data['description'],
                'predicted_category': predicted_category,
                'confidence': confidence_score
            })
        
        # Save current account balance if available
        if last_balance is not None:
            # Update or create account balance record
            balance_record = AccountBalance.query.first()
            if balance_record:
                balance_record.balance = last_balance
                balance_record.last_updated = datetime.utcnow()
                balance_record.source_file = filename
            else:
                balance_record = AccountBalance(
                    balance=last_balance,
                    source_file=filename
                )
                db.session.add(balance_record)
        
        db.session.commit()
        
        # Count Money In vs Money Out
        money_in_count = sum(1 for tx in processed_transactions if tx['amount'] > 0)
        money_out_count = sum(1 for tx in processed_transactions if tx['amount'] < 0)
        
        return jsonify({
            'success': True,
            'imported': len(processed_transactions),
            'duplicates_skipped': duplicates_found,
            'money_in_count': money_in_count,
            'money_out_count': money_out_count,
            'current_balance': last_balance,
            'transactions': processed_transactions[:10]  # Show first 10 for preview
        })
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"Upload error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/finance-tracker/transactions')
def finance_transactions():
    try:
        # Get query parameters
        year = request.args.get('year', type=int)
        month = request.args.get('month', type=int)
        category_id = request.args.get('category_id', type=int)
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 50, type=int)
        
        # Build query
        query = FinanceTransaction.query
        
        if year:
            query = query.filter(FinanceTransaction.year == year)
        if month:
            query = query.filter(FinanceTransaction.month == month)
        if category_id:
            query = query.filter(FinanceTransaction.category_id == category_id)
        
        # Get paginated results
        transactions = query.order_by(FinanceTransaction.date.desc()).paginate(
            page=page, per_page=per_page, error_out=False
        )
        
        return jsonify({
            'success': True,
            'data': [{
                'id': t.id,
                'date': t.date.isoformat(),
                'amount': t.amount,
                'description': t.description,
                'category_id': t.category_id,
                'category_name': t.category_ref.name if t.category_ref else None,
                'category_color': t.category_ref.color if t.category_ref else '#6c757d',
                'confidence_score': t.confidence_score,
                'is_recurring': t.is_recurring,
                'needs_validation': t.needs_validation,
                'month': t.month,
                'year': t.year
            } for t in transactions.items],
            'pagination': {
                'page': transactions.page,
                'per_page': transactions.per_page,
                'total': transactions.total,
                'pages': transactions.pages
            }
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/finance-tracker/transactions/<int:transaction_id>', methods=['PUT', 'DELETE'])
def finance_transaction_operations(transaction_id):
    try:
        transaction = FinanceTransaction.query.get_or_404(transaction_id)
        
        if request.method == 'PUT':
            # Update transaction
            data = request.json
            
            # Update fields if provided
            if 'date' in data:
                try:
                    new_date = datetime.strptime(data['date'], '%Y-%m-%d').date()
                    transaction.date = new_date
                    transaction.year = new_date.year
                    transaction.month = new_date.month
                except ValueError:
                    return jsonify({'success': False, 'error': 'Invalid date format. Use YYYY-MM-DD'}), 400
            
            if 'amount' in data:
                try:
                    transaction.amount = float(data['amount'])
                except ValueError:
                    return jsonify({'success': False, 'error': 'Invalid amount format'}), 400
            
            if 'description' in data:
                if len(data['description'].strip()) == 0:
                    return jsonify({'success': False, 'error': 'Description cannot be empty'}), 400
                transaction.description = data['description'].strip()
                
                # Update hash for duplicate detection
                transaction.transaction_hash = calculate_transaction_hash({
                    'date': transaction.date,
                    'amount': transaction.amount,
                    'description': transaction.description
                })
            
            db.session.commit()
            
            return jsonify({
                'success': True,
                'data': {
                    'id': transaction.id,
                    'date': transaction.date.isoformat(),
                    'amount': transaction.amount,
                    'description': transaction.description,
                    'category_id': transaction.category_id,
                    'category_name': transaction.category_ref.name if transaction.category_ref else None,
                    'category_color': transaction.category_ref.color if transaction.category_ref else '#6c757d',
                    'confidence_score': transaction.confidence_score,
                    'month': transaction.month,
                    'year': transaction.year
                }
            })
        
        elif request.method == 'DELETE':
            # Delete transaction
            db.session.delete(transaction)
            db.session.commit()
            
            return jsonify({
                'success': True,
                'message': f'Transaction {transaction_id} deleted successfully'
            })
            
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/finance-tracker/transactions/delete-all', methods=['DELETE'])
def delete_all_transactions():
    try:
        # Count transactions before deletion
        transaction_count = FinanceTransaction.query.count()
        
        # Delete all transactions
        FinanceTransaction.query.delete()
        db.session.commit()
        
        return jsonify({
            'success': True,
            'deleted_count': transaction_count,
            'message': f'Successfully deleted {transaction_count} transactions'
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/finance-tracker/transactions/<int:transaction_id>/categorize', methods=['POST'])
def categorize_transaction(transaction_id):
    try:
        transaction = FinanceTransaction.query.get_or_404(transaction_id)
        data = request.json
        category_id = data.get('category_id')
        
        # Update transaction category
        transaction.category_id = category_id
        transaction.confidence_score = 1.0  # Manual categorization is 100% confident
        transaction.needs_validation = False  # Manual categorization validates the transaction
        
        # Learn from this categorization
        if category_id:
            keywords = extract_description_keywords(transaction.description)
            
            # Check if pattern already exists
            existing_pattern = FinanceCategoryLearning.query.filter_by(
                description_pattern=keywords,
                category_id=category_id
            ).first()
            
            if existing_pattern:
                existing_pattern.frequency += 1
                existing_pattern.last_used = datetime.utcnow()
                existing_pattern.confidence = min(1.0, existing_pattern.confidence + 0.1)
            else:
                new_pattern = FinanceCategoryLearning(
                    description_pattern=keywords,
                    category_id=category_id,
                    frequency=1,
                    confidence=0.8
                )
                db.session.add(new_pattern)
        
        db.session.commit()
        
        return jsonify({
            'success': True,
            'data': {
                'id': transaction.id,
                'category_id': transaction.category_id,
                'confidence_score': transaction.confidence_score
            }
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/finance-tracker/transactions/<int:transaction_id>/validate', methods=['POST'])
def validate_transaction(transaction_id):
    """Mark a transaction as validated"""
    try:
        transaction = FinanceTransaction.query.get_or_404(transaction_id)
        
        # Mark as validated
        transaction.needs_validation = False
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': 'Transaction marked as validated'
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/finance-tracker/summary')
def finance_summary():
    try:
        year = request.args.get('year', datetime.now().year, type=int)
        month = request.args.get('month', type=int)
        
        # Build base query
        query = FinanceTransaction.query.filter(FinanceTransaction.year == year)
        if month:
            query = query.filter(FinanceTransaction.month == month)
        
        transactions = query.all()
        
        # For monthly overview, we need ALL transactions from the past 6 months
        current_date = datetime.now()
        six_months_ago = current_date - timedelta(days=180)  # Roughly 6 months
        
        all_transactions_query = FinanceTransaction.query.filter(
            FinanceTransaction.date >= six_months_ago.date()
        )
        all_transactions = all_transactions_query.all()
        
        # Calculate summary metrics
        total_transactions = len(transactions)
        total_amount = sum(t.amount for t in transactions)
        income = sum(t.amount for t in transactions if t.amount > 0)
        expenses = sum(t.amount for t in transactions if t.amount < 0)
        
        # Calculate total budget from all categories
        all_categories = FinanceCategory.query.all()
        total_budget = sum(cat.budget for cat in all_categories if cat.budget and cat.budget > 0)
        
        # Category breakdown
        categories = {}
        for t in transactions:
            if t.category_ref:
                cat_name = t.category_ref.name
                if cat_name not in categories:
                    categories[cat_name] = {
                        'name': cat_name,
                        'color': t.category_ref.color,
                        'amount': 0,
                        'count': 0
                    }
                categories[cat_name]['amount'] += t.amount
                categories[cat_name]['count'] += 1
        
        # Monthly breakdown for the last 6 months
        try:
            current_date = datetime.now()
            monthly_data = {}
            
            # Generate last 6 months from current month backwards
            for i in range(6):
                target_year = current_date.year
                target_month = current_date.month - i
                
                # Handle year rollover
                while target_month <= 0:
                    target_month += 12
                    target_year -= 1
                
                try:
                    
                    # Filter transactions for this specific month and year
                    month_transactions = []
                    for t in all_transactions:
                        try:
                            if hasattr(t, 'date') and t.date and t.date.month == target_month and t.date.year == target_year:
                                month_transactions.append(t)
                        except Exception as e:
                            print(f"Error filtering transaction {t.id}: {e}")
                            continue
                    
                    # Calculate categories for this month
                    month_categories = {}
                    for t in month_transactions:
                        try:
                            if t.amount < 0:  # Only include expenses
                                cat_name = 'Other'  # Default fallback
                                if hasattr(t, 'category') and t.category and hasattr(t.category, 'name'):
                                    cat_name = t.category.name
                                elif hasattr(t, 'category_id') and t.category_id:
                                    # Try to get category name from relationship
                                    try:
                                        category = FinanceCategory.query.get(t.category_id)
                                        if category:
                                            cat_name = category.name
                                    except:
                                        pass
                                
                                if cat_name not in month_categories:
                                    month_categories[cat_name] = 0
                                month_categories[cat_name] += abs(t.amount)
                        except Exception as e:
                            print(f"Error processing transaction category {t.id}: {e}")
                            continue
                    
                    monthly_data[f"{target_year}-{target_month:02d}"] = {
                        'month': target_month,
                        'year': target_year,
                        'month_name': datetime(target_year, target_month, 1).strftime('%b %Y'),
                        'total': sum(t.amount for t in month_transactions) if month_transactions else 0,
                        'income': sum(t.amount for t in month_transactions if t.amount > 0) if month_transactions else 0,
                        'expenses': sum(t.amount for t in month_transactions if t.amount < 0) if month_transactions else 0,
                        'count': len(month_transactions),
                        'categories': month_categories
                    }
                except Exception as e:
                    print(f"Error processing month {target_month}/{target_year}: {e}")
                
                # Always add month data (whether successful or not) to ensure all 6 months appear
                if f"{target_year}-{target_month:02d}" not in monthly_data:
                    monthly_data[f"{target_year}-{target_month:02d}"] = {
                        'month': target_month,
                        'year': target_year,
                        'month_name': datetime(target_year, target_month, 1).strftime('%b %Y'),
                        'total': 0,
                        'income': 0,
                        'expenses': 0,
                        'count': 0,
                        'categories': {}
                    }
        except Exception as e:
            print(f"Error generating monthly data: {e}")
            # Fallback to empty data
            monthly_data = {}
        
        return jsonify({
            'success': True,
            'summary': {
                'total_transactions': total_transactions,
                'total_amount': total_amount,
                'income': income,
                'expenses': abs(expenses),
                'budget': total_budget,
                'net': total_budget + expenses,  # Remaining Budget - Expenses (expenses are negative)
                'categories': list(categories.values()),
                'monthly_data': list(monthly_data.values())
            }
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/finance-tracker/transactions/recurring')
def finance_recurring_transactions():
    try:
        # Get all transactions marked as recurring
        recurring_transactions = FinanceTransaction.query.filter(
            FinanceTransaction.is_recurring == True
        ).all()
        
        # Group by description pattern to avoid stacking/duplication
        pattern_groups = defaultdict(list)
        for transaction in recurring_transactions:
            pattern = extract_description_keywords(transaction.description)
            if pattern:
                pattern_groups[pattern].append(transaction)
        
        # Calculate monthly total using latest amount per pattern (no stacking)
        recurring_total = 0
        for pattern, transactions in pattern_groups.items():
            transactions.sort(key=lambda t: t.date)
            latest_amount = abs(transactions[-1].amount)  # Take latest amount for this pattern
            recurring_total += latest_amount
        
        return jsonify({
            'success': True,
            'recurring_total': recurring_total,
            'count': len(pattern_groups),  # Count unique patterns, not total transactions
            'unique_patterns': len(pattern_groups)
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/finance-tracker/transactions/recurring-detailed')
def finance_recurring_transactions_detailed():
    try:
        # Get all transactions marked as recurring with detailed information
        recurring_transactions = FinanceTransaction.query.filter(
            FinanceTransaction.is_recurring == True
        ).order_by(FinanceTransaction.date.desc()).all()
        
        # Convert to JSON format
        transactions_data = []
        for transaction in recurring_transactions:
            transactions_data.append({
                'id': transaction.id,
                'date': format_date_safe(transaction.date),
                'amount': transaction.amount,
                'description': transaction.description,
                'category_id': transaction.category_id,
                'is_recurring': transaction.is_recurring,
                'confidence_score': transaction.confidence_score,
                'created_at': format_date_safe(transaction.created_at)
            })
        
        # Calculate monthly total from recurring transactions
        recurring_total = sum(abs(t.amount) for t in recurring_transactions)
        
        return jsonify({
            'success': True,
            'transactions': transactions_data,
            'recurring_total': recurring_total,
            'count': len(transactions_data)
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/finance-tracker/transactions/<int:transaction_id>/recurring', methods=['PUT'])
def update_transaction_recurring(transaction_id):
    try:
        transaction = FinanceTransaction.query.get_or_404(transaction_id)
        data = request.json
        
        transaction.is_recurring = data.get('is_recurring', False)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': 'Recurring status updated successfully'
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/finance-tracker/recurring-insights')
def recurring_insights():
    """Get recurring transaction insights and alerts for manually marked recurring items only"""
    try:
        # Get manually marked recurring transactions (is_recurring = True)
        manually_recurring = FinanceTransaction.query.filter_by(is_recurring=True).all()
        
        insights = {
            'total_recurring': 0,
            'price_changes': [],
            'missing_payments': [],
            'stopped_payments': [],
            'active_count': 0
        }
        
        # Group manually marked recurring transactions by description pattern
        pattern_groups = defaultdict(list)
        for transaction in manually_recurring:
            pattern = extract_description_keywords(transaction.description)
            if pattern:
                pattern_groups[pattern].append(transaction)
        
        # Analyze each group for price changes and missing payments
        for pattern, transactions in pattern_groups.items():
            if len(transactions) >= 2:  # Need at least 2 to detect changes
                transactions.sort(key=lambda t: t.date)
                latest = transactions[-1]
                previous = transactions[-2]
                
                # Check for price changes (5% threshold)
                if abs(latest.amount) != abs(previous.amount):
                    change_percent = ((abs(latest.amount) - abs(previous.amount)) / abs(previous.amount)) * 100
                    if abs(change_percent) >= 5:  # 5% threshold
                        insights['price_changes'].append({
                            'description': latest.description,
                            'old_amount': previous.amount,
                            'new_amount': latest.amount,
                            'change_percent': change_percent,
                            'last_occurrence': latest.date.isoformat()
                        })
                
                # Check for missing payments (simple frequency check)
                intervals = []
                for i in range(1, len(transactions)):
                    days_diff = (transactions[i].date - transactions[i-1].date).days
                    intervals.append(days_diff)
                
                if intervals:
                    avg_frequency = sum(intervals) / len(intervals)
                    days_since_last = (datetime.now().date() - latest.date).days
                    
                    # Consider missing if 1.5x the expected frequency has passed
                    if days_since_last > (avg_frequency * 1.5):
                        days_overdue = days_since_last - int(avg_frequency)
                        if days_since_last > (avg_frequency * 3):
                            insights['stopped_payments'].append({
                                'description': latest.description,
                                'expected_amount': latest.amount,
                                'last_occurrence': latest.date.isoformat()
                            })
                        else:
                            insights['missing_payments'].append({
                                'description': latest.description,
                                'expected_amount': latest.amount,
                                'days_overdue': days_overdue,
                                'last_occurrence': latest.date.isoformat()
                            })
            
            # Count active recurring (all manually marked ones)
            insights['active_count'] += 1
            insights['total_recurring'] += abs(transactions[-1].amount)
        
        return jsonify({
            'success': True,
            'insights': insights
        })
        
    except Exception as e:
        logger.error(f"Error getting recurring insights: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/finance-tracker/recurring-monthly-prediction')
def recurring_monthly_prediction():
    """Get monthly recurring cost tracking and next month prediction"""
    try:
        # Get manually marked recurring transactions
        recurring_transactions = FinanceTransaction.query.filter_by(is_recurring=True).all()
        
        if not recurring_transactions:
            return jsonify({
                'success': True,
                'monthly_tracking': {},
                'next_month_prediction': 0,
                'prediction_confidence': 0,
                'total_recurring_items': 0
            })
        
        # Group by description pattern
        pattern_groups = defaultdict(list)
        for transaction in recurring_transactions:
            pattern = extract_description_keywords(transaction.description)
            if pattern:
                pattern_groups[pattern].append(transaction)
        
        monthly_tracking = {}
        total_prediction = 0
        
        # Analyze each recurring pattern
        for pattern, transactions in pattern_groups.items():
            transactions.sort(key=lambda t: t.date)
            
            # Track monthly costs for this pattern
            monthly_costs = defaultdict(list)
            for transaction in transactions:
                month_key = f"{transaction.date.year}-{transaction.date.month:02d}"
                monthly_costs[month_key].append(abs(transaction.amount))
            
            # Calculate monthly averages (in case multiple transactions in same month)
            monthly_averages = {}
            for month, amounts in monthly_costs.items():
                monthly_averages[month] = sum(amounts) / len(amounts)
            
            # Get recent trend for prediction
            recent_months = sorted(monthly_averages.keys())[-3:]  # Last 3 months
            if recent_months:
                recent_amounts = [monthly_averages[month] for month in recent_months]
                predicted_amount = sum(recent_amounts) / len(recent_amounts)  # Average of recent months
                
                # Detect trend (increasing/decreasing)
                if len(recent_amounts) >= 2:
                    trend = recent_amounts[-1] - recent_amounts[0]
                    trend_percent = (trend / recent_amounts[0]) * 100 if recent_amounts[0] > 0 else 0
                else:
                    trend = 0
                    trend_percent = 0
                
                total_prediction += predicted_amount
                
                # Store tracking data for this pattern
                monthly_tracking[pattern] = {
                    'description': transactions[-1].description,
                    'monthly_costs': monthly_averages,
                    'predicted_next_month': predicted_amount,
                    'trend': trend,
                    'trend_percent': trend_percent,
                    'last_amount': abs(transactions[-1].amount),
                    'frequency_months': len(recent_months)
                }
        
        # Calculate prediction confidence based on data consistency
        confidence = min(100, (len(pattern_groups) * 20))  # Higher confidence with more recurring items
        
        return jsonify({
            'success': True,
            'monthly_tracking': monthly_tracking,
            'next_month_prediction': round(total_prediction, 2),
            'prediction_confidence': confidence,
            'total_recurring_items': len(pattern_groups)
        })
        
    except Exception as e:
        logger.error(f"Error getting monthly recurring prediction: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/finance-tracker/auto-categorize', methods=['POST'])
def auto_categorize_transactions():
    """Apply auto-categorization to existing uncategorized transactions"""
    try:
        # Get all uncategorized transactions
        uncategorized = FinanceTransaction.query.filter(
            FinanceTransaction.category_id.is_(None)
        ).all()
        
        if not uncategorized:
            return jsonify({
                'success': True,
                'message': 'No uncategorized transactions found',
                'categorized_count': 0
            })
        
        # Get existing learning patterns
        patterns = FinanceCategoryLearning.query.all()
        
        if not patterns:
            return jsonify({
                'success': True, 
                'message': 'No learning patterns available. Categorize some transactions manually first.',
                'categorized_count': 0
            })
        
        categorized_count = 0
        
        for transaction in uncategorized:
            # Predict category
            best_pattern, score = predict_category(transaction.description, patterns)
            
            # Use a lower threshold for auto-categorization (0.2 instead of 0.3)
            if best_pattern and score > 0.2:
                transaction.category_id = best_pattern.category_id
                transaction.confidence_score = score
                categorized_count += 1
                
                # Update pattern usage statistics
                best_pattern.frequency += 1
                best_pattern.last_used = datetime.utcnow()
        
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': f'Successfully auto-categorized {categorized_count} transactions',
            'categorized_count': categorized_count,
            'total_uncategorized': len(uncategorized)
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/finance-tracker/debug-pdf', methods=['POST'])
def debug_pdf():
    """Debug endpoint to examine PDF structure"""
    if not PDF_SUPPORT:
        return jsonify({'success': False, 'error': 'PDF support not available. Install pdfplumber to enable PDF parsing.'}), 400
    
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if not file.filename.lower().endswith('.pdf'):
            return jsonify({'success': False, 'error': 'Please upload a PDF file'}), 400
        
        file_content = file.read()
        
        # Extract text from PDF
        with pdfplumber.open(io.BytesIO(file_content)) as pdf:
            full_text = ""
            for page_num, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                if page_text:
                    full_text += f"=== PAGE {page_num + 1} ===\n{page_text}\n\n"
        
        # Get lines and analyze them
        lines = full_text.split('\n')
        non_empty_lines = [line.strip() for line in lines if line.strip()]
        
        # Look for date patterns
        date_lines = []
        amount_lines = []
        
        for i, line in enumerate(non_empty_lines[:50]):  # First 50 lines
            # Check for dates
            if re.search(r'\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}', line):
                date_lines.append(f"Line {i+1}: {line}")
            
            # Check for amounts
            if re.search(r'£?[\d,]+\.?\d*', line):
                amount_lines.append(f"Line {i+1}: {line}")
        
        return jsonify({
            'success': True,
            'debug_info': {
                'total_text_length': len(full_text),
                'total_lines': len(lines),
                'non_empty_lines': len(non_empty_lines),
                'first_20_lines': non_empty_lines[:20],
                'lines_with_dates': date_lines[:10],
                'lines_with_amounts': amount_lines[:10],
                'sample_text': full_text[:2000]  # First 2000 characters
            }
        })
        
    except Exception as e:
        logger.error(f"PDF debug error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

# Temporary admin endpoint for database migration - REMOVE AFTER USE
@app.route('/admin/migrate-final-value', methods=['POST'])
def migrate_final_value_column():
    try:
        # Check if column already exists by trying to select from it
        try:
            db.session.execute(text('SELECT final_value_gbp FROM bitcoin_trade LIMIT 1'))
            return jsonify({'success': True, 'message': 'Column already exists'})
        except Exception:
            # Column doesn't exist, add it
            db.session.execute(text('ALTER TABLE bitcoin_trade ADD COLUMN final_value_gbp FLOAT'))
            db.session.commit()
            return jsonify({'success': True, 'message': 'Successfully added final_value_gbp column'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# Add scheduler jobs after all functions are defined
# Schedule minute-level BTC price collection every minute
scheduler.add_job(
    func=collect_current_minute_price,
    trigger=CronTrigger(second=0),  # Run at the start of every minute
    id='collect_minute_price',
    name='Collect minute-level BTC price',
    replace_existing=True,
    max_instances=1  # Prevent overlapping executions
)

# Schedule data gap detection and filling (every 6 hours)
scheduler.add_job(
    func=detect_and_fill_data_gaps,
    trigger=CronTrigger(hour='*/6', minute=30),  # Run every 6 hours at 30 minutes past the hour
    id='fill_data_gaps',
    name='Detect and fill data gaps',
    replace_existing=True,
    max_instances=1
)

# Schedule cleanup of old minute data (daily at 2 AM)
scheduler.add_job(
    func=cleanup_old_minute_data,
    trigger=CronTrigger(hour=2, minute=0),
    args=[30],  # Keep 30 days of minute data
    id='cleanup_minute_data',
    name='Cleanup old minute-level data',
    replace_existing=True
)

# Schedule archiving of old hourly data (weekly on Sunday at 3 AM)
scheduler.add_job(
    func=archive_old_price_data,
    trigger=CronTrigger(day_of_week=6, hour=3, minute=0),  # Sunday = 6
    args=[365],  # Keep 1 year of hourly data
    id='archive_hourly_data',
    name='Archive old hourly data',
    replace_existing=True
)

@app.route('/api/debug/template-deployment')
def debug_template_deployment():
    """Debug endpoint to verify template deployment in container"""
    import os
    import hashlib
    
    try:
        template_path = os.path.join(app.template_folder, 'data_viewer.html')
        
        result = {
            'template_folder': app.template_folder,
            'template_path': template_path,
            'file_exists': os.path.exists(template_path),
            'timestamp': datetime.now().isoformat()
        }
        
        if os.path.exists(template_path):
            # Get file hash
            with open(template_path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            result['file_hash'] = file_hash
            
            # Check for key features
            with open(template_path, 'r') as f:
                content = f.read()
            
            result['features'] = {
                'has_data_type_dropdown': 'id="dataType"' in content,
                'has_swing_button': 'loadSwingAnalysis' in content,
                'has_swing_container': 'swingAnalysisContainer' in content,
                'file_size': len(content)
            }
            
            # Get file stats
            stat = os.stat(template_path)
            result['file_stats'] = {
                'size': stat.st_size,
                'modified_time': stat.st_mtime,
                'modified_readable': datetime.fromtimestamp(stat.st_mtime).isoformat()
            }
        
        return jsonify({
            'success': True,
            'debug_info': result
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    # Initialize database tables
    with app.app_context():
        db.create_all()
        
        # Initialize historical minute data on startup
        logger.info("Starting application initialization...")
        try:
            initialization_result = initialize_historical_minute_data(days_back=30)
            if initialization_result['success']:
                logger.info(f"Startup initialization completed: {initialization_result['message']}")
            else:
                logger.warning(f"Startup initialization failed: {initialization_result.get('error', 'Unknown error')}")
        except Exception as e:
            logger.error(f"Error during startup initialization: {e}")
    
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port, debug=True)