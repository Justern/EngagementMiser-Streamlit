#!/usr/bin/env python3
"""
Database configuration for Generic Comment Detector
Modify these settings to connect to your SQL Server instance
"""

# Database connection settings
DATABASE_CONFIG = {
    # SQL Server connection details
    'server': 'localhost',  # Change to your SQL Server instance
    'database': 'EngagementMiser',
    'driver': 'ODBC Driver 17 for SQL Server',  # Change if using different driver
    
    # Authentication options
    'trusted_connection': True,  # Use Windows Authentication
    # 'username': 'your_username',  # Uncomment for SQL Authentication
    # 'password': 'your_password',  # Uncomment for SQL Authentication
    
    # Connection timeout settings
    'timeout': 30,
    'connection_timeout': 30,
    
    # Additional options
    'autocommit': True,
    'multiple_active_result_sets': True
}

def get_connection_string():
    """
    Generate connection string from configuration.
    
    Returns:
        str: Formatted connection string
    """
    config = DATABASE_CONFIG
    
    # Base connection string
    conn_str = f"DRIVER={{{config['driver']}}};"
    conn_str += f"SERVER={config['server']};"
    conn_str += f"DATABASE={config['database']};"
    
    # Authentication
    if config.get('trusted_connection'):
        conn_str += "Trusted_Connection=yes;"
    else:
        conn_str += f"UID={config['username']};"
        conn_str += f"PWD={config['password']};"
    
    # Additional options
    conn_str += f"Timeout={config['timeout']};"
    conn_str += f"Connection Timeout={config['connection_timeout']};"
    
    return conn_str

def get_test_connection_string():
    """
    Get a simple test connection string.
    
    Returns:
        str: Basic connection string for testing
    """
    return (
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=localhost;"
        "DATABASE=EngagementMiser;"
        "Trusted_Connection=yes;"
    )

# Example connection strings for different scenarios
EXAMPLE_CONNECTIONS = {
    'local_windows_auth': (
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=localhost;"
        "DATABASE=EngagementMiser;"
        "Trusted_Connection=yes;"
    ),
    
    'local_sql_auth': (
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=localhost;"
        "DATABASE=EngagementMiser;"
        "UID=your_username;"
        "PWD=your_password;"
    ),
    
    'remote_server': (
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=your_server_name_or_ip;"
        "DATABASE=EngagementMiser;"
        "Trusted_Connection=yes;"
    ),
    
    'named_instance': (
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=localhost\\SQLEXPRESS;"
        "DATABASE=EngagementMiser;"
        "Trusted_Connection=yes;"
    )
}

if __name__ == "__main__":
    print("Database Configuration for Generic Comment Detector")
    print("=" * 50)
    
    print(f"Current configuration:")
    print(f"  Server: {DATABASE_CONFIG['server']}")
    print(f"  Database: {DATABASE_CONFIG['database']}")
    print(f"  Driver: {DATABASE_CONFIG['driver']}")
    print(f"  Authentication: {'Windows' if DATABASE_CONFIG['trusted_connection'] else 'SQL'}")
    
    print(f"\nGenerated connection string:")
    print(get_connection_string())
    
    print(f"\nAvailable example connections:")
    for name, conn_str in EXAMPLE_CONNECTIONS.items():
        print(f"  {name}: {conn_str}")
    
    print(f"\nTo use a different connection, modify DATABASE_CONFIG in this file.")
