#!/usr/bin/env python3

import io
import re
import logging
from datetime import datetime

try:
    import pdfplumber
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    print("pdfplumber not installed")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_single_transaction(line_num, line):
    """Parse a single transaction line and extract date, description, balance"""
    date_match = re.match(r'(\d{1,2}\/\d{1,2}\/\d{4})', line)
    if not date_match:
        return None
        
    try:
        date_str = date_match.group(1)
        date_obj = datetime.strptime(date_str, '%d/%m/%Y').date()
        
        # Find all £ amounts in the line
        pound_matches = []
        for match in re.finditer(r'£\s*([\d,]+\.?\d*)', line):
            amount = match.group(1)
            position = match.start()
            pound_matches.append((amount, position))
        
        if len(pound_matches) < 2:
            return None
        
        # The last amount is always the balance
        balance_amount = float(pound_matches[-1][0].replace(',', ''))
        
        # Extract description
        date_end = date_match.end()
        first_amount_pos = pound_matches[0][1]
        base_description = line[date_end:first_amount_pos].strip()
        
        # Check for additional merchant info after the balance
        if len(pound_matches) >= 2:
            balance_pos = pound_matches[-1][1]
            balance_end = line.find(pound_matches[-1][0], balance_pos) + len(pound_matches[-1][0])
            remaining_text = line[balance_end:].strip()
            
            if remaining_text and len(remaining_text) > 3:
                # Clean up common patterns
                remaining_text = re.sub(r'^,?\s*', '', remaining_text)
                remaining_text = re.sub(r'\s*ON\s+\d{1,2}-\d{1,2}-\d{4}.*$', '', remaining_text)
                if remaining_text:
                    description = base_description + " " + remaining_text
                else:
                    description = base_description
            else:
                description = base_description
        else:
            description = base_description
        
        # Clean description
        description = re.sub(r'\s+', ' ', description).strip()
        
        return {
            'date': date_obj,
            'description': description,
            'balance': balance_amount,
            'source_row': f"Line {line_num + 1}: {line}"
        }
        
    except (ValueError, IndexError) as e:
        logger.debug(f"Error parsing line: {line[:50]}... Error: {e}")
        return None

def should_exclude_transaction(description):
    """Check if transaction should be excluded (internal transfers)"""
    return (description == 'TRANSFER TO Shared Account' or 
            description == 'TRANSFER TO Extra Monthly Savings' or
            description == 'TRANSFER TO Savings Account' or
            description.startswith('TRANSFER TO Extra Monthly'))

def parse_santander_statement(lines):
    """Parse Santander statement using balance-based Money In/Out detection"""
    
    # Pre-process lines to handle multi-line transactions and extract all transaction data
    all_transactions = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or len(line) < 10:
            i += 1
            continue
            
        # Skip header lines
        if any(header in line.lower() for header in ['date description', 'money in', 'money out', 'balance', 'transaction date']):
            i += 1
            continue
            
        # Check if this line starts with a date (transaction line)
        if re.match(r'\d{1,2}\/\d{1,2}\/\d{4}', line):
            full_line = line
            j = i + 1
            
            # Look ahead for continuation lines
            while j < len(lines):
                next_line = lines[j].strip()
                if not next_line:
                    j += 1
                    continue
                    
                # Stop if we hit another transaction date
                if re.match(r'\d{1,2}\/\d{1,2}\/\d{4}', next_line):
                    break
                    
                # Stop if we hit a header line
                if any(header in next_line.lower() for header in ['date description', 'money in', 'money out', 'balance']):
                    break
                    
                # Add continuation line if it doesn't look like just amounts
                if not re.match(r'^£\s*[\d,]+\.?\d*\s*$', next_line):
                    full_line += " " + next_line
                    logger.debug(f"Added continuation: '{next_line}' to transaction")
                
                j += 1
                
            # Parse this transaction
            parsed_tx = parse_single_transaction(i, full_line)
            if parsed_tx:
                all_transactions.append(parsed_tx)
                
            i = j
        else:
            i += 1
    
    # Sort transactions by date to ensure correct order for balance comparison
    all_transactions.sort(key=lambda tx: tx['date'])
    
    # Use balance changes to determine Money In vs Money Out
    money_out_transactions = []
    money_in_count = 0
    internal_transfer_count = 0
    credit_refund_count = 0
    previous_balance = None
    
    for tx in all_transactions:
        current_balance = tx['balance']
        
        if previous_balance is not None:
            balance_change = current_balance - previous_balance
            
            # If balance increased, this was Money In - skip it
            if balance_change > 0:
                money_in_count += 1
                logger.debug(f"Skipping Money In (balance increased by £{balance_change:.2f}): {tx['description'][:30]}...")
                previous_balance = current_balance
                continue
            
            # If balance decreased, this was Money Out - include it
            # But still filter out internal transfers
            if should_exclude_transaction(tx['description']):
                internal_transfer_count += 1
                logger.debug(f"Skipping internal transfer: {tx['description']}")
                previous_balance = current_balance
                continue
            
            # Additional check: Skip transactions that look like credits/refunds
            desc_lower = tx['description'].lower()
            if (desc_lower.startswith('credit from') or
                desc_lower.startswith('refund from') or
                desc_lower.startswith('bank giro credit') or
                desc_lower.startswith('faster payments receipt')):
                credit_refund_count += 1
                logger.debug(f"Skipping credit/refund: {tx['description']}")
                previous_balance = current_balance
                continue
            
            # This is a legitimate Money Out transaction
            money_out_transactions.append({
                'date': tx['date'],
                'description': tx['description'],
                'amount': balance_change,  # Use actual balance change (already negative)
                'source_row': tx['source_row']
            })
            
        previous_balance = current_balance
    
    print(f"Summary:")
    print(f"  Total transactions parsed: {len(all_transactions)}")
    print(f"  Money In (balance increased): {money_in_count}")
    print(f"  Internal transfers: {internal_transfer_count}")
    print(f"  Credits/refunds: {credit_refund_count}")
    print(f"  Money Out transactions: {len(money_out_transactions)}")
    
    logger.info(f"Extracted {len(money_out_transactions)} Money Out transactions using balance-based detection")
    return money_out_transactions

def test_parser():
    if not PDF_SUPPORT:
        print("Cannot test - pdfplumber not available")
        return
        
    try:
        with open('/Users/chriseddisford/Downloads/Statements09012897612628 (1).pdf', 'rb') as f:
            content = f.read()
            
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            all_lines = []
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    all_lines.extend(page_text.split('\n'))
        
        transactions = parse_santander_statement(all_lines)
        print(f'Found {len(transactions)} Money Out transactions using balance-based detection')
        print(f'Target was 52 Money Out transactions. Current count: {len(transactions)}')
        
        for i, tx in enumerate(transactions[:15]):
            print(f'{i+1}. {tx["date"]} £{tx["amount"]:.2f} {tx["description"][:60]}')
            
        print(f"\nSample of last 5 transactions:")
        for i, tx in enumerate(transactions[-5:], len(transactions)-4):
            print(f'{i}. {tx["date"]} £{tx["amount"]:.2f} {tx["description"][:60]}')
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_parser()