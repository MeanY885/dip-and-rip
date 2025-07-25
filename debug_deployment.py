#!/usr/bin/env python3
"""
Debug deployment script to verify template files in container
"""

import os
import hashlib

def get_file_hash(filepath):
    """Get MD5 hash of a file"""
    try:
        with open(filepath, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except FileNotFoundError:
        return "FILE_NOT_FOUND"

def check_template_file():
    """Check the data_viewer.html template file"""
    template_path = "/app/templates/data_viewer.html"
    
    print("=== DEPLOYMENT DEBUG INFO ===")
    print(f"Template path: {template_path}")
    print(f"File exists: {os.path.exists(template_path)}")
    
    if os.path.exists(template_path):
        print(f"File hash: {get_file_hash(template_path)}")
        
        # Check for key elements
        with open(template_path, 'r') as f:
            content = f.read()
            
        has_data_type = 'id="dataType"' in content
        has_swing_button = 'loadSwingAnalysis' in content
        has_swing_container = 'swingAnalysisContainer' in content
        
        print(f"Has Data Type dropdown: {has_data_type}")
        print(f"Has Swing Analysis button: {has_swing_button}")
        print(f"Has Swing Analysis container: {has_swing_container}")
        
        # Check template modification time
        stat = os.stat(template_path)
        print(f"Last modified: {stat.st_mtime}")
        
        return {
            'exists': True,
            'hash': get_file_hash(template_path),
            'has_features': has_data_type and has_swing_button and has_swing_container
        }
    else:
        return {'exists': False}

if __name__ == "__main__":
    result = check_template_file()
    print(f"\nResult: {result}")