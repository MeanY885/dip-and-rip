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
    previous_balance = None
    
    for tx in all_transactions:
        current_balance = tx['balance']
        
        if previous_balance is not None:
            balance_change = current_balance - previous_balance
            
            # If balance increased, this was Money In - skip it
            if balance_change > 0:
                logger.debug(f"Skipping Money In (balance increased by £{balance_change:.2f}): {tx['description'][:30]}...")
                previous_balance = current_balance
                continue
            
            # If balance decreased, this was Money Out - include it
            # But still filter out internal transfers
            if should_exclude_transaction(tx['description']):
                logger.debug(f"Skipping internal transfer: {tx['description']}")
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
    
    logger.info(f"Extracted {len(money_out_transactions)} Money Out transactions using balance-based detection")
    return money_out_transactions

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