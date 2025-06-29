#!/usr/bin/env python3

# Test script to clean up the old PDF parsing code
with open('/Users/chriseddisford/Documents/dip-and-rip/app.py', 'r') as f:
    lines = f.readlines()

# Find the line with "return [], None" and remove everything until the next function
output_lines = []
skip_mode = False

for i, line in enumerate(lines):
    line_num = i + 1
    
    # Start skipping after the "return [], None" line
    if line.strip() == "return [], None":
        output_lines.append(line)
        skip_mode = True
        continue
    
    # Stop skipping when we hit the next function definition
    if skip_mode and line.strip().startswith("def "):
        skip_mode = False
    
    # Add line if we're not in skip mode
    if not skip_mode:
        output_lines.append(line)

# Write the cleaned version
with open('/Users/chriseddisford/Documents/dip-and-rip/app_clean.py', 'w') as f:
    f.writelines(output_lines)

print("Cleaned version created as app_clean.py")