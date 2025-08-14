#!/usr/bin/env python3
"""
Update database connections in all model files to use Azure instead of localhost.
"""

import os
import glob

def update_database_connections():
    """Update all simple_score.py files to use Azure database connections."""
    
    # Azure connection details
    azure_server = "ecs-sql-server-engagementmiser.database.windows.net"
    azure_database = "ecs_tweets_db"
    azure_username = "ecsadmin"
    azure_password = "EngagementMiser!"
    
    # Find all simple_score.py files
    simple_score_files = glob.glob("**/simple_score.py", recursive=True)
    
    print(f"Found {len(simple_score_files)} simple_score.py files:")
    
    for file_path in simple_score_files:
        print(f"\nProcessing: {file_path}")
        
        try:
            # Read the file
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Track changes
            changes_made = []
            
            # Replace localhost with Azure
            if "localhost" in content:
                content = content.replace("localhost", azure_server)
                changes_made.append("localhost → Azure server")
            
            if "EngagementMiser" in content and "database" in content.lower():
                content = content.replace("EngagementMiser", azure_database)
                changes_made.append("Database name → Azure database")
            
            # Replace Windows Authentication with SQL Authentication
            if "Trusted_Connection=yes" in content:
                content = content.replace("Trusted_Connection=yes", f"UID={azure_username};PWD={azure_password}")
                changes_made.append("Windows Auth → SQL Auth")
            
            # Update connection string format for Azure
            if "mssql+pyodbc://@" in content:
                content = content.replace("mssql+pyodbc://@", f"mssql+pyodbc://{azure_username}:{azure_password}@")
                changes_made.append("Connection string → Azure format")
            
            # Add TrustServerCertificate for Azure
            if "TrustServerCertificate=yes" not in content:
                content = content.replace("&TrustServerCertificate=yes", "&TrustServerCertificate=yes")
            
            # Write the updated content back
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            if changes_made:
                print(f"  ✅ Updated: {', '.join(changes_made)}")
            else:
                print(f"  ℹ️  No changes needed")
                
        except Exception as e:
            print(f"  ❌ Error processing {file_path}: {e}")
    
    print("\nDatabase connection updates completed!")

if __name__ == "__main__":
    update_database_connections()
