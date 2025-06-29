#!/usr/bin/env python3

# Script to fix the mixed up function
with open('/Users/chriseddisford/Documents/dip-and-rip/app.py', 'r') as f:
    content = f.read()

# Replace the mixed up content
old_mixed_up = '''def get_historical_price_data(hours_back=24):
    """Get historical price data from database"""
    try:
        if filename.endswith('.xls'):'''

new_correct = '''def get_historical_price_data(hours_back=24):
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

# Routes'''

# Find where the Excel parsing mess starts and ends
start_marker = "def get_historical_price_data(hours_back=24):"
end_marker = "# Routes"

start_index = content.find(start_marker)
if start_index != -1:
    # Find the end (look for next function or routes)
    remaining = content[start_index:]
    end_index = remaining.find("@app.route")
    if end_index != -1:
        # Replace the entire section
        before = content[:start_index]
        after = content[start_index + end_index:]
        new_content = before + new_correct + after
        
        with open('/Users/chriseddisford/Documents/dip-and-rip/app.py', 'w') as f:
            f.write(new_content)
        
        print("Fixed the mixed up function!")
    else:
        print("Could not find end marker")
else:
    print("Could not find start marker")