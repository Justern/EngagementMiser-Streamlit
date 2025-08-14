#!/usr/bin/env python3
"""
Check database schema for Tweets_Sample_4M table
"""

import pandas as pd
from sqlalchemy import create_engine, text as sql_text

def check_schema():
    """Check the actual schema of Tweets_Sample_4M table."""
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
            WHERE TABLE_NAME = 'Tweets_Sample_4M'
            ORDER BY ORDINAL_POSITION
        """)
        
        with engine.connect() as conn:
            result = conn.execute(query)
            columns = result.fetchall()
        
        print("Tweets_Sample_4M table schema:")
        print("=" * 50)
        for col in columns:
            print(f"{col[0]:<25} {col[1]:<15} {col[2]}")
            
        # Also check a sample row
        print("\nSample row data:")
        print("=" * 50)
        sample_query = sql_text("""
            SELECT TOP 1 *
            FROM [EngagementMiser].[dbo].[Tweets_Sample_4M]
        """)
        
        with engine.connect() as conn:
            sample = conn.execute(sample_query).fetchone()
            if sample:
                # Get column names
                col_names = [desc[0] for desc in result.description] if result.description else []
                for i, value in enumerate(sample):
                    if i < len(col_names):
                        print(f"{col_names[i]:<25}: {str(value)[:50]}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_schema()
