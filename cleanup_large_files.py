#!/usr/bin/env python3
"""
Cleanup script to remove large model.safetensors files
while keeping other essential model files.
"""

import os
import glob

def cleanup_large_files():
    """Remove large model.safetensors files from all model checkpoints."""
    
    # Find all model.safetensors files
    safetensors_files = glob.glob("**/model.safetensors", recursive=True)
    
    print(f"Found {len(safetensors_files)} model.safetensors files:")
    
    for file_path in safetensors_files:
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
        print(f"  {file_path} ({file_size:.1f} MB)")
        
        # Remove the large file
        try:
            os.remove(file_path)
            print(f"    ✅ Removed {file_path}")
        except Exception as e:
            print(f"    ❌ Error removing {file_path}: {e}")
    
    print("\nCleanup completed!")

if __name__ == "__main__":
    cleanup_large_files()
