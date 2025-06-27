#!/usr/bin/env python
"""Initialize the database with all required tables"""

from app import app, db

with app.app_context():
    # Create all tables
    db.create_all()
    print("Database tables created successfully!")
    
    # List all tables
    from sqlalchemy import inspect
    inspector = inspect(db.engine)
    tables = inspector.get_table_names()
    print(f"Tables in database: {tables}") 