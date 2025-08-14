#!/usr/bin/env python3
"""
Test script to check the actual database schema.
"""

import pyodbc
import pandas as pd

def test_schema():
    """Test the actual database schema."""
    try:
        # Connection string with ODBC Driver 18
        conn_str = 'DRIVER={ODBC Driver 18 for SQL Server};SERVER=localhost;DATABASE=EngagementMiser;Trusted_Connection=yes;Encrypt=no;TrustServerCertificate=yes;'
        
        print("🔌 Testing database connection...")
        conn = pyodbc.connect(conn_str)
        print("✅ Database connection successful!")
        
        # Test Tweets_Sample_4M table
        print("\n📊 Testing Tweets_Sample_4M table schema...")
        schema_query = """
        SELECT TOP 1 *
        FROM [EngagementMiser].[dbo].[Tweets_Sample_4M]
        """
        
        schema_data = pd.read_sql(schema_query, conn)
        print(f"✅ Retrieved {len(schema_data.columns)} columns:")
        for col in schema_data.columns:
            print(f"   • {col}")
        
        # Test TwitterUsers table
        print("\n👥 Testing TwitterUsers table schema...")
        users_schema_query = """
        SELECT TOP 1 *
        FROM [EngagementMiser].[dbo].[TwitterUsers]
        """
        
        users_schema_data = pd.read_sql(users_schema_query, conn)
        print(f"✅ Retrieved {len(users_schema_data.columns)} columns:")
        for col in users_schema_data.columns:
            print(f"   • {col}")
        
        conn.close()
        print("\n🎉 Schema test completed successfully!")
        
    except Exception as e:
        print(f"❌ Schema test failed: {e}")

if __name__ == "__main__":
    test_schema()
