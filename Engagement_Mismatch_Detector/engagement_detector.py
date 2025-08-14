#!/usr/bin/env python3
"""
Engagement Mismatch Detector Launcher
====================================

This script can be run from anywhere and will automatically find and execute
the main engagement detector script.

Usage:
    python engagement_detector.py <tweet_id>
    python engagement_detector.py --interactive
"""

import os
import sys
import subprocess

def main():
    # Get the directory where this launcher script is located
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Path to the main script
    main_script = os.path.join(SCRIPT_DIR, "score_engagement_mismatch_standalone.py")
    
    # Check if main script exists
    if not os.path.exists(main_script):
        print(f"❌ Error: Main script not found at {main_script}")
        return 1
    
    # Get all command line arguments
    args = sys.argv[1:] if len(sys.argv) > 1 else []
    
    # Run the main script with all arguments
    try:
        result = subprocess.run([sys.executable, main_script] + args, 
                              cwd=SCRIPT_DIR, 
                              capture_output=False)
        return result.returncode
    except Exception as e:
        print(f"❌ Error running main script: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
