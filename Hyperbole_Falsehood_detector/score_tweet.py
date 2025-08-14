"""
Simple script to score a tweet by tweet_id using the trained Hyperbole/Falsehood detector.
Make sure you've run the main training script first!

Now includes Media Catchphrases similarity boost functionality!
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sqlalchemy import create_engine, text as sql_text

# Connection details (same as main script)
SQL_SERVER = "localhost"
SQL_DB = "EngagementMiser"
SQL_DRIVER = "ODBC Driver 18 for SQL Server"

# Connection string with Windows Authentication
CONN_STR = (
    f"mssql+pyodbc://@{SQL_SERVER}/{SQL_DB}"
    f"?driver={SQL_DRIVER.replace(' ', '+')}"
    "&Trusted_Connection=yes"
    "&TrustServerCertificate=yes"
)

engine = create_engine(CONN_STR)

# Media Catchphrases Integration
CATCHPHRASES_QUERY = """
SELECT phrase, context
FROM [EngagementMiser].[dbo].[Media_Catchphrases_Corpus]
WHERE phrase IS NOT NULL AND phrase != ''
"""

# Load catchphrases during initialization
try:
    import pandas as pd
    catchphrases_df = pd.read_sql_query(sql_text(CATCHPHRASES_QUERY), engine)
    CATCHPHRASES = list(catchphrases_df['phrase'].str.lower())
    CATCHPHRASE_CONTEXTS = dict(zip(catchphrases_df['phrase'].str.lower(), catchphrases_df['context']))
    print(f"✅ Loaded {len(CATCHPHRASES)} catchphrases from Media_Catchphrases_Corpus")
except Exception as e:
    print(f"⚠️ Warning: Could not load catchphrases corpus: {e}")
    CATCHPHRASES = []
    CATCHPHRASE_CONTEXTS = {}

# Similarity scoring configuration
SIMILARITY_THRESHOLD = 0.6  # Minimum similarity to trigger boost
SIMILARITY_BOOST = 0.15     # Amount to boost score when similarity detected
SIMILARITY_METHOD = "partial"  # "partial" for substring matching

def calculate_text_similarity(text: str, catchphrases: list) -> tuple[float, str, str]:
    """
    Calculate similarity between input text and catchphrases.
    Returns (similarity_score, best_match_phrase, context).
    """
    if not catchphrases:
        return 0.0, "", ""
    
    text_lower = text.lower()
    best_score = 0.0
    best_phrase = ""
    best_context = ""
    
    for phrase in catchphrases:
        phrase_lower = phrase.lower()
        
        if SIMILARITY_METHOD == "partial":
            # Check if phrase is contained in text or vice versa
            if phrase_lower in text_lower or text_lower in phrase_lower:
                # Calculate overlap ratio
                overlap = len(set(phrase_lower.split()) & set(text_lower.split()))
                total = len(set(phrase_lower.split()) | set(text_lower.split()))
                if total > 0:
                    score = overlap / total
                    if score > best_score:
                        best_score = score
                        best_phrase = phrase
                        best_context = CATCHPHRASE_CONTEXTS.get(phrase_lower, "")
    
    return best_score, best_phrase, best_context

def apply_similarity_boost(base_score: float, similarity_score: float, phrase: str, context: str) -> tuple[float, dict]:
    """
    Apply similarity boost to the base score if similarity threshold is met.
    Returns (adjusted_score, boost_info).
    """
    boost_info = {
        "similarity_detected": False,
        "similarity_score": similarity_score,
        "matched_phrase": phrase,
        "context": context,
        "boost_applied": 0.0,
        "original_score": base_score
    }
    
    if similarity_score >= SIMILARITY_THRESHOLD and phrase:
        boost_amount = SIMILARITY_BOOST * similarity_score
        adjusted_score = min(1.0, base_score + boost_amount)
        boost_info.update({
            "similarity_detected": True,
            "boost_applied": boost_amount,
            "final_score": adjusted_score
        })
        return adjusted_score, boost_info
    
    return base_score, boost_info

def load_trained_model():
    """Load the trained model and tokenizer from checkpoints."""
    checkpoint_path = "checkpoints/text_softlabel_roberta"
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Model checkpoint not found at {checkpoint_path}. "
            "Please run the main training script first!"
        )
    
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    return model, tokenizer, device

def score_tweet_id(tweet_id: str, model, tokenizer, device, max_len=160, apply_similarity_boost: bool = True):
    """Score a tweet by looking up its text in the database."""
    
    # Get tweet text from database
    query = sql_text("""
        SELECT TOP 1 text
        FROM [EngagementMiser].[dbo].[Hyperbole_Falsehood_tweets_annot]
        WHERE CAST(tweet_id AS VARCHAR(32)) = :tid
    """)
    
    with engine.connect() as conn:
        row = conn.execute(query, {"tid": str(tweet_id)}).fetchone()
    
    if not row or not row[0]:
        raise ValueError(f"tweet_id {tweet_id} not found or has no text.")
    
    tweet_text = str(row[0])
    
    # Tokenize and predict
    with torch.no_grad():
        enc = tokenizer(
            tweet_text,
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        base_score = torch.sigmoid(logits.squeeze()).item()
        
        if apply_similarity_boost and CATCHPHRASES:
            # Calculate similarity with catchphrases
            similarity_score, matched_phrase, context = calculate_text_similarity(tweet_text, CATCHPHRASES)
            
            # Apply similarity boost if threshold is met
            final_score, boost_info = apply_similarity_boost(base_score, similarity_score, matched_phrase, context)
            return float(final_score), tweet_text, boost_info
        else:
            # Return base score with no boost info
            boost_info = {
                "similarity_detected": False,
                "similarity_score": 0.0,
                "matched_phrase": "",
                "context": "",
                "boost_applied": 0.0,
                "original_score": base_score,
                "final_score": base_score
            }
            return float(base_score), tweet_text, boost_info

def main():
    """Main function to score a tweet by ID."""
    
    # Load the trained model
    try:
        model, tokenizer, device = load_trained_model()
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Get tweet_id from user
    tweet_id = input("Enter tweet_id to score: ").strip()
    
    if not tweet_id:
        print("Please provide a tweet_id.")
        return
    
    try:
        # Score the tweet
        score, text, boost_info = score_tweet_id(tweet_id, model, tokenizer, device)
        
        print(f"\n📱 Tweet ID: {tweet_id}")
        print(f"📝 Text: {text[:100]}{'...' if len(text) > 100 else ''}")
        print(f"🎯 Final Score: {score:.4f}")
        print(f"   (0 = not hyperbolic/false, 1 = very hyperbolic/false)")
        
        # Show similarity boost information if applicable
        if boost_info["similarity_detected"]:
            print(f"🚀 Similarity Boost Applied!")
            print(f"   Original Score: {boost_info['original_score']:.4f}")
            print(f"   Boost Amount: +{boost_info['boost_applied']:.4f}")
            print(f"   Matched Phrase: '{boost_info['matched_phrase']}'")
            print(f"   Context: {boost_info['context']}")
            print(f"   Similarity Score: {boost_info['similarity_score']:.3f}")
        else:
            print(f"ℹ️ No catchphrase similarity detected")
        
        # Add interpretation
        if score < 0.3:
            interpretation = "Likely truthful"
        elif score < 0.7:
            interpretation = "Uncertain - moderate risk"
        else:
            interpretation = "High risk of hyperbole/falsehood"
        
        print(f"💡 Interpretation: {interpretation}")
        
    except Exception as e:
        print(f"❌ Error scoring tweet: {e}")

if __name__ == "__main__":
    main()
