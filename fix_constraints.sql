-- Fix NOT NULL constraints for bitcoin_trade table
-- Run these commands in your SQLite database

-- SQLite doesn't support ALTER COLUMN directly, so we need to recreate the table
-- First, let's backup the current data and recreate the table with proper constraints

BEGIN TRANSACTION;

-- Create the new table with correct constraints
CREATE TABLE bitcoin_trade_fixed (
    id INTEGER PRIMARY KEY,
    status VARCHAR(20) NOT NULL,
    date DATE NOT NULL,
    type VARCHAR(10) NOT NULL DEFAULT 'BTC',
    initial_investment_gbp FLOAT NOT NULL,
    btc_buy_price FLOAT NOT NULL,
    btc_sell_price FLOAT,  -- NULLABLE
    profit FLOAT,          -- NULLABLE
    fee FLOAT,             -- NULLABLE
    btc_amount FLOAT,      -- NULLABLE
    created_at DATETIME,
    updated_at DATETIME
);

-- Copy data from the original table
INSERT INTO bitcoin_trade_fixed 
SELECT * FROM bitcoin_trade;

-- Drop the original table
DROP TABLE bitcoin_trade;

-- Rename the new table
ALTER TABLE bitcoin_trade_fixed RENAME TO bitcoin_trade;

COMMIT; 