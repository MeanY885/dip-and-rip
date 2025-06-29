#!/usr/bin/env python3

import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

def test_txt_parser():
    try:
        with open('/Users/chriseddisford/Downloads/Statements09012897612628.txt', 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            
        transactions, last_balance = parse_txt_statement(content)
        
        print(f"\n=== TXT PARSER TEST RESULTS ===")
        print(f"Total transactions: {len(transactions)}")
        print(f"Current account balance: £{last_balance:.2f}")
        
        money_in = [tx for tx in transactions if tx['amount'] > 0]
        money_out = [tx for tx in transactions if tx['amount'] < 0]
        
        print(f"Money In transactions: {len(money_in)}")
        print(f"Money Out transactions: {len(money_out)}")
        
        print(f"\nFirst 5 transactions:")
        for i, tx in enumerate(transactions[:5]):
            tx_type = "Money In" if tx['amount'] > 0 else "Money Out"
            print(f"{i+1}. {tx['date']} £{tx['amount']:.2f} ({tx_type}) - {tx['description'][:50]}")
            
        print(f"\nLast 5 transactions:")
        for i, tx in enumerate(transactions[-5:], len(transactions)-4):
            tx_type = "Money In" if tx['amount'] > 0 else "Money Out"
            print(f"{i}. {tx['date']} £{tx['amount']:.2f} ({tx_type}) - {tx['description'][:50]}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_txt_parser()