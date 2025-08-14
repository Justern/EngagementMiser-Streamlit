#!/usr/bin/env python3
"""
Update model loading references to use Hugging Face Hub instead of local checkpoints.
"""

import os
import glob

def update_model_references():
    """Update all model loading references to use Hugging Face Hub."""
    
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
            
            # Replace local checkpoint loading with Hugging Face Hub
            if "checkpoint_path = os.path.join(os.path.dirname(__file__), \"checkpoints\"" in content:
                # This is a model that needs Hugging Face Hub replacement
                model_name = os.path.basename(os.path.dirname(file_path))
                
                # Create Hugging Face Hub reference
                hf_reference = f"MidlAnalytics/{model_name.lower().replace('_', '-')}"
                
                # Replace the checkpoint path logic
                old_code = """    checkpoint_path = os.path.join(os.path.dirname(__file__), "checkpoints", "clickbait_classifier")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)"""
                
                new_code = f"""    # Load from Hugging Face Hub
    hf_repo = "{hf_reference}"
    
    try:
        # Load tokenizer and model from Hugging Face Hub
        tokenizer = AutoTokenizer.from_pretrained(hf_repo)
        model = AutoModelForSequenceClassification.from_pretrained(hf_repo)
    except Exception as e:
        print(f"Error loading from Hugging Face Hub: {{e}}")
        # Fallback to generic model
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")"""
                
                if old_code in content:
                    content = content.replace(old_code, new_code)
                    changes_made.append("Local checkpoint → Hugging Face Hub")
                
                # Also update any other checkpoint references
                if "checkpoints/" in content:
                    content = content.replace("checkpoints/", "hf_hub/")
                    changes_made.append("Checkpoint path → HF Hub path")
            
            # Write the updated content back
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            if changes_made:
                print(f"  ✅ Updated: {', '.join(changes_made)}")
            else:
                print(f"  ℹ️  No changes needed")
                
        except Exception as e:
            print(f"  ❌ Error processing {file_path}: {e}")
    
    print("\nModel reference updates completed!")

if __name__ == "__main__":
    update_model_references()
