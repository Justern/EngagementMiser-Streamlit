#!/usr/bin/env python3
"""
Check database schema for TwitterUsers table
"""

import pandas as pd
from sqlalchemy import create_engine, text as sql_text

def check_users_schema():
    """Check the actual schema of TwitterUsers table."""
    try:
        # Database connection
        SQL_SERVER = "localhost"
        SQL_DB = "EngagementMiser"
        SQL_DRIVER = "ODBC Driver 18 for SQL Server"
        
        CONN_STR = (
            f"mssql+pyodbc://@{SQL_SERVER}/{SQL_DB}"
            f"?driver={SQL_DRIVER.replace(' ', '+')}"
            "&Trusted_Connection=yes"
            "&TrustServerCertificate=yes"
        )
        
        engine = create_engine(CONN_STR)
        
        # Get table schema
        query = sql_text("""
            SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_NAME = 'TwitterUsers'
            ORDER BY ORDINAL_POSITION
        """)
        
        with engine.connect() as conn:
            result = conn.execute(query)
            columns = result.fetchall()
        
        print("TwitterUsers table schema:")
        print("=" * 50)
        for col in columns:
            print(f"{col[0]:<25} {col[1]:<15} {col[2]}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_users_schema()
