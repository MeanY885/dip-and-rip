-- Manual migration SQL commands
-- Run these on your production database to add the new tables

-- Create SalaryRecord table
CREATE TABLE IF NOT EXISTS salary_record (
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

-- Create TradingRecord table  
CREATE TABLE IF NOT EXISTS trading_record (
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

-- Verify tables were created
.tables