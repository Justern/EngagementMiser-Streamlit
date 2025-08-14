#!/usr/bin/env python3
"""
Check Azure database table schemas to see actual column names
"""

import pyodbc
import pandas as pd

def check_azure_schemas():
    """Check the actual column names in Azure database tables."""
    
    # Azure connection details
    server = "ecs-sql-server-engagementmiser.database.windows.net"
    database = "ecs_tweets_db"
    username = "ecsadmin"
    password = "EngagementMiser!"
    
    try:
        # Connect to Azure
        conn_str = f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password};TrustServerCertificate=yes"
        conn = pyodbc.connect(conn_str)
        
        print("✅ Connected to Azure SQL Database!")
        
        # Check Tweets_Sample_4M table
        print("\n📋 Tweets_Sample_4M table columns:")
        query = "SELECT COLUMN_NAME, DATA_TYPE, CHARACTER_MAXIMUM_LENGTH FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = 'Tweets_Sample_4M' ORDER BY ORDINAL_POSITION"
        df = pd.read_sql(query, conn)
        print(df.to_string(index=False))
        
        # Check TwitterUsers table
        print("\n📋 TwitterUsers table columns:")
        query = "SELECT COLUMN_NAME, DATA_TYPE, CHARACTER_MAXIMUM_LENGTH FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = 'TwitterUsers' ORDER BY ORDINAL_POSITION"
        df = pd.read_sql(query, conn)
        print(df.to_string(index=False))
        
        # Check if tables exist
        print("\n📋 All tables in database:")
        query = "SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE = 'BASE TABLE' ORDER BY TABLE_NAME"
        df = pd.read_sql(query, conn)
        print(df.to_string(index=False))
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    check_azure_schemas()
