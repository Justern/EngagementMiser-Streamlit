#!/usr/bin/env python3
"""
Fix SQL parameter binding issues in streamlit_app_integrated_models.py
"""

def fix_sql_parameters():
    """Fix all SQL parameter binding from ? to :tweet_id format."""
    
    # Read the file
    with open('streamlit_app_integrated_models.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix all SQL parameter bindings
    # Replace ? with :tweet_id in WHERE clauses
    content = content.replace("WHERE tweet_id = ?", "WHERE tweet_id = :tweet_id")
    
    # Replace list parameter binding with dictionary
    content = content.replace("result = conn.execute(text(query), [str(tweet_id)]).fetchone()", 
                            "result = conn.execute(text(query), {\"tweet_id\": str(tweet_id)}).fetchone()")
    
    # Write the fixed content back
    with open('streamlit_app_integrated_models.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Fixed all SQL parameter binding issues!")

if __name__ == "__main__":
    fix_sql_parameters()
