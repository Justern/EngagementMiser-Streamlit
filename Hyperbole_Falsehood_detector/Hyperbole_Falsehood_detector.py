"""
Train a text-only transformer with soft labels pulled from SQL Server,
then score any tweet_id on demand (0–1 probability-like score).

Table used:
  [EngagementMiser].[dbo].[Hyperbole_Falsehood_tweets_annot]
Columns (as provided):
  tweet_id (str/int), text (str), author_id, source, label_confidence (float in [0,1])

Notes:
- We treat label_confidence as a *soft* target and optimize BCEWithLogitsLoss.
- Model outputs a single logit; we apply sigmoid at inference to get [0,1].
- Only tweet `text` is used as input features.
"""

import os
import math
import random
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Union, Tuple

# Torch & HF
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW

# Progress bar
from tqdm.auto import tqdm

# --- SQL connectivity (via SQLAlchemy + pyodbc) ---
# Updated connection details for local SQL Server
SQL_SERVER   = os.getenv("SQL_SERVER",   "localhost")
SQL_DB       = os.getenv("SQL_DB",       "EngagementMiser")
SQL_DRIVER   = os.getenv("SQL_DRIVER",   "ODBC Driver 18 for SQL Server")

from sqlalchemy import create_engine, text as sql_text
# Connection string format for SQL Server over ODBC with Windows Authentication
CONN_STR = (
    f"mssql+pyodbc://@{SQL_SERVER}/{SQL_DB}"
    f"?driver={SQL_DRIVER.replace(' ', '+')}"
    "&Trusted_Connection=yes"
    "&TrustServerCertificate=yes"
)
engine = create_engine(CONN_STR, fast_executemany=True)

# ----------------------
# Media Catchphrases Integration
# ----------------------
# Load catchphrases corpus for similarity-based scoring boost
CATCHPHRASES_QUERY = """
SELECT phrase, context
FROM [EngagementMiser].[dbo].[Media_Catchphrases_Corpus]
WHERE phrase IS NOT NULL AND phrase != ''
"""

# Load catchphrases during initialization
try:
    with engine.connect() as conn:
        catchphrases_df = pd.read_sql(sql_text(CATCHPHRASES_QUERY), conn)
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
SIMILARITY_METHOD = "partial"  # "partial" for substring matching, "fuzzy" for fuzzy matching

def calculate_text_similarity(text: str, catchphrases: list) -> Tuple[float, str, str]:
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
        
        elif SIMILARITY_METHOD == "fuzzy":
            # Simple word overlap scoring
            text_words = set(text_lower.split())
            phrase_words = set(phrase_lower.split())
            
            if len(phrase_words) > 0:
                overlap = len(text_words & phrase_words)
                score = overlap / len(phrase_words)
                if score > best_score:
                    best_score = score
                    best_phrase = phrase
                    best_context = CATCHPHRASE_CONTEXTS.get(phrase_lower, "")
    
    return best_score, best_phrase, best_context

def apply_similarity_boost(base_score: float, similarity_score: float, phrase: str, context: str) -> Tuple[float, dict]:
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

# ----------------------
# Reproducibility helpers
# ----------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

# ----------------------
# Config (tweak as needed)
# ----------------------
MODEL_NAME = "roberta-base"     # solid default; swap for "bert-base-uncased", etc.
MAX_LEN    = 220                # typical tweet length cap (after URLs/handles)
BATCH_SIZE = 16
EPOCHS     = 3
LR         = 2e-5               # standard fine-tune LR
WARMUP_PCT = 0.1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

# ----------------------
# 1) Load from SQL Server
# ----------------------
QUERY = """
SELECT
    CAST(tweet_id AS VARCHAR(32)) AS tweet_id,   -- keep id as string to be safe
    text,
    CAST(label_confidence AS FLOAT) AS label_confidence
FROM [EngagementMiser].[dbo].[Hyperbole_Falsehood_tweets_annot]
WHERE text IS NOT NULL
  AND label_confidence IS NOT NULL
"""

with engine.connect() as conn:
    df = pd.read_sql(sql_text(QUERY), conn)

# Basic sanity checks / cleaning
df["text"] = df["text"].astype(str).str.strip()
df = df[(df["text"].str.len() > 0)]
# Ensure labels are in [0,1]; if your table guarantees this, the clip is harmless:
df["label_confidence"] = df["label_confidence"].astype(float).clip(0.0, 1.0)

print(f"Loaded {len(df):,} rows from SQL.")

# ----------------------
# 2) Train / Val split
# ----------------------
from sklearn.model_selection import train_test_split
train_df, val_df = train_test_split(
    df, test_size=0.1, random_state=42, stratify=None  # regression/soft labels -> no stratify
)
print(f"Train: {len(train_df):,} | Val: {len(val_df):,}")

# ----------------------
# 3) Tokenizer & Dataset
# ----------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

class TweetDataset(Dataset):
    """Wraps text + soft label for the dataloader."""
    def __init__(self, texts, labels, tokenizer, max_len=160):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        t = self.texts[idx]
        y = float(self.labels[idx])  # soft label in [0,1]

        enc = self.tokenizer(
            t,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )

        # Flatten batch dimension (since return_tensors='pt' adds it)
        item = {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            # No token_type_ids for RoBERTa; for BERT you could include it if present
            "labels":         torch.tensor(y, dtype=torch.float),
        }
        return item

train_ds = TweetDataset(train_df["text"], train_df["label_confidence"], tokenizer, MAX_LEN)
val_ds   = TweetDataset(val_df["text"],   val_df["label_confidence"],   tokenizer, MAX_LEN)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

# ----------------------
# 4) Model (single-logit head)
# ----------------------
# We’ll still use AutoModelForSequenceClassification for convenience,
# but we’ll compute BCEWithLogitsLoss *manually* so it behaves like a soft-label classifier.
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=1  # single regression-style logit
)
model.to(DEVICE)

# Soft-label objective: binary cross-entropy with logits
criterion = nn.BCEWithLogitsLoss()

# Optimizer / Scheduler
optimizer = AdamW(model.parameters(), lr=LR)
num_training_steps = EPOCHS * len(train_loader)
num_warmup_steps = int(WARMUP_PCT * num_training_steps)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
)

# Optional: grad clipping value
MAX_GRAD_NORM = 1.0

# ----------------------
# 5) Training / Evaluation loops
# ----------------------
def train_one_epoch(epoch: int):
    model.train()
    total_loss = 0.0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [train]")
    for batch in pbar:
        optimizer.zero_grad(set_to_none=True)

        input_ids      = batch["input_ids"].to(DEVICE, non_blocking=True)
        attention_mask = batch["attention_mask"].to(DEVICE, non_blocking=True)
        labels         = batch["labels"].to(DEVICE, non_blocking=True)

        # Forward: we don't pass labels to the model so we can control the loss function
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits  = outputs.logits.squeeze(-1)  # shape: (B,)

        loss = criterion(logits, labels)  # soft labels in [0,1]
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / max(1, len(train_loader))

@torch.no_grad()
def evaluate():
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []

    pbar = tqdm(val_loader, desc="Validation")
    for batch in pbar:
        input_ids      = batch["input_ids"].to(DEVICE, non_blocking=True)
        attention_mask = batch["attention_mask"].to(DEVICE, non_blocking=True)
        labels         = batch["labels"].to(DEVICE, non_blocking=True)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits  = outputs.logits.squeeze(-1)

        loss = criterion(logits, labels)
        total_loss += loss.item()

        # Convert logits -> probabilities with sigmoid for reporting
        probs = torch.sigmoid(logits)
        all_preds.append(probs.cpu().numpy())
        all_targets.append(labels.cpu().numpy())

    all_preds   = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    # Report a few friendly metrics for soft labels
    mae = float(np.mean(np.abs(all_preds - all_targets)))
    mse = float(np.mean((all_preds - all_targets) ** 2))

    return total_loss / max(1, len(val_loader)), mae, mse

best_val = math.inf
for epoch in range(EPOCHS):
    train_loss = train_one_epoch(epoch)
    val_loss, val_mae, val_mse = evaluate()
    print(
        f"Epoch {epoch+1}: train_loss={train_loss:.4f} | "
        f"val_loss={val_loss:.4f} | val_MAE={val_mae:.4f} | val_MSE={val_mse:.4f}"
    )

    # (Optional) Simple checkpointing on validation loss
    if val_loss < best_val:
        best_val = val_loss
        os.makedirs("checkpoints", exist_ok=True)
        model.save_pretrained("checkpoints/text_softlabel_roberta")
        tokenizer.save_pretrained("checkpoints/text_softlabel_roberta")
        print("✅ Saved checkpoint -> checkpoints/text_softlabel_roberta")

# ----------------------
# 6) Single-text inference helper
# ----------------------
@torch.no_grad()
def predict_single_text(text: str, apply_similarity_boost: bool = True) -> Tuple[float, dict]:
    """
    Returns a tuple of (score, boost_info) where score is a float in [0,1].
    If apply_similarity_boost is True, incorporates catchphrase similarity boost.
    """
    model.eval()
    enc = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt",
    )
    input_ids      = enc["input_ids"].to(DEVICE)
    attention_mask = enc["attention_mask"].to(DEVICE)

    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    # shape: (1, 1) -> squeeze to scalar
    base_score = torch.sigmoid(logits.squeeze()).item()
    
    if apply_similarity_boost and CATCHPHRASES:
        # Calculate similarity with catchphrases
        similarity_score, matched_phrase, context = calculate_text_similarity(text, CATCHPHRASES)
        
        # Apply similarity boost if threshold is met
        final_score, boost_info = apply_similarity_boost(base_score, similarity_score, matched_phrase, context)
        return float(final_score), boost_info
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
        return float(base_score), boost_info

# ----------------------
# 7) Score a tweet by tweet_id (pulled from SQL)
# ----------------------
def score_tweet_id(tweet_id: Union[str, int], apply_similarity_boost: bool = True) -> Tuple[float, str, dict]:
    """
    Look up the tweet text by tweet_id in the SQL table and return (score, text, boost_info).
    Raises ValueError if tweet_id not found.
    """
    query = sql_text(
        """
        SELECT TOP 1 text
        FROM [EngagementMiser].[dbo].[Hyperbole_Falsehood_tweets_annot]
        WHERE CAST(tweet_id AS VARCHAR(32)) = :tid
        """
    )
    with engine.connect() as conn:
        row = conn.execute(query, {"tid": str(tweet_id)}).fetchone()
    if not row or not row[0]:
        raise ValueError(f"tweet_id {tweet_id} not found or has no text.")
    
    tweet_text = str(row[0])
    score, boost_info = predict_single_text(tweet_text, apply_similarity_boost)
    return score, tweet_text, boost_info

# ----------------------
# 8) Example usage
# ----------------------
if __name__ == "__main__":
    # Example: score one of your provided ids
    try:
        example_id = "1486656887567355907"
        score, text, boost_info = score_tweet_id(example_id)
        
        print(f"\n📱 Tweet ID: {example_id}")
        print(f"📝 Text: {text[:100]}{'...' if len(text) > 100 else ''}")
        print(f"🎯 Final Score: {score:.4f} (0=not hyperbolic/false, 1=very)")
        
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
        print("Scoring error:", e)
