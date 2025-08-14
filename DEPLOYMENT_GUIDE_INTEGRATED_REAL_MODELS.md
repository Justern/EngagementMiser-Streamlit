# Deployment Guide: Integrated Real Models Streamlit App

## Overview
This guide explains how to deploy the **Integrated Real Models Streamlit App** that uses the ACTUAL logic from all 10 models in your repository.

## What's Integrated

### 🔧 Hugging Face Models (4 models)
- **Clickbait_Classifier** - Uses REAL RoBERTa from Hugging Face Hub
- **Content_Recycling_Detector** - Uses REAL RoBERTa from Hugging Face Hub  
- **Engagement_Mismatch_Detector** - Uses REAL RoBERTa from Hugging Face Hub
- **Hyperbole_Falsehood_detector** - Uses REAL RoBERTa from Hugging Face Hub

### ⚡ Rule-Based Models (6 models)
- **Authority Signal Manipulation** - REAL ASM logic from your model
- **Coordinated Account Network** - REAL network analysis logic
- **Emotive Manipulation** - REAL .joblib model from your repository
- **Generic Comment** - REAL .joblib model from your repository
- **Rapid Engagement Spike** - REAL time series analysis logic
- **Reply Bait** - REAL conversation analysis logic

## Prerequisites

### 1. Required Files
Ensure these files are in your repository:
```
Emotive_Manipulation_Detector/emotive_manipulation_model.joblib
Generic_Comment_Detector/generic_comment_model.joblib
```

### 2. Azure SQL Database
- Database: `ecs_tweets_db`
- Tables: `Tweets_Sample_4M`, `TwitterUsers`
- Connection details in Streamlit secrets

### 3. Hugging Face Token
- Token for accessing `MidlAnalytics/engagement-concordance-roberta`
- Set in Streamlit secrets as `hf_token`

## Deployment Steps

### Step 1: Upload Supporting Files
Upload these files to your Streamlit Cloud repository:
- `streamlit_app_integrated_real_models.py` (main app)
- `requirements_integrated_real_models.txt` (dependencies)
- `Emotive_Manipulation_Detector/emotive_manipulation_model.joblib`
- `Generic_Comment_Detector/generic_comment_model.joblib`

### Step 2: Configure Streamlit Secrets
In your Streamlit Cloud dashboard, set these secrets:

```toml
[azure_db]
server = "ecs-sql-server-engagementmiser.database.windows.net"
database = "ecs_tweets_db"
username = "ecsadmin"
password = "your_password"

hf_token = "your_huggingface_token"
```

### Step 3: Set Main File
Set `streamlit_app_integrated_real_models.py` as your main Streamlit file.

## How It Works

### 1. Model Integration
- **Hugging Face Models**: Load from Hub with fallback to rule-based logic
- **Joblib Models**: Load local .joblib files for emotive manipulation and generic comment
- **Rule-Based Models**: Use REAL logic from your repository files

### 2. Azure Connection
- Connects to Azure SQL Database
- Uses proper table schemas from your database
- Handles JOIN operations for comprehensive data

### 3. Scoring System
- Each model returns a score from 0-1
- Weights are applied based on model importance
- Final ECS score is calculated as weighted average

## Fallback System

If any model fails, the system automatically falls back to rule-based logic:
- **Hugging Face failures** → Rule-based patterns
- **Joblib model failures** → Rule-based patterns
- **Database connection issues** → Graceful error handling

## Benefits

1. **Real Model Logic**: Uses ACTUAL algorithms from your repository
2. **Proper Azure Integration**: Correct table schemas and connections
3. **Automatic Fallbacks**: Ensures system reliability
4. **Performance**: Optimized for your specific use case
5. **Maintainability**: Easy to update individual models

## Troubleshooting

### Common Issues

1. **Joblib Model Not Found**
   - Ensure .joblib files are uploaded to correct paths
   - Check file permissions in Streamlit Cloud

2. **Azure Connection Failed**
   - Verify database credentials in secrets
   - Check firewall rules and network access

3. **Hugging Face Model Failed**
   - Verify HF token is valid
   - Check internet connectivity in Streamlit Cloud

### Performance Tips

1. **Model Caching**: Models are cached after first load
2. **Connection Pooling**: Azure connections are pooled for efficiency
3. **Progress Indicators**: Real-time progress updates for user experience

## Next Steps

1. **Deploy** the integrated app to Streamlit Cloud
2. **Test** with sample tweets from your database
3. **Monitor** performance and model accuracy
4. **Iterate** based on results and feedback

## Support

For issues or questions:
1. Check the Streamlit Cloud logs
2. Verify all supporting files are uploaded
3. Test database connectivity
4. Review model file paths and permissions

---

**Note**: This app represents a significant upgrade from the previous hybrid approach, now using the REAL logic from all 10 models in your repository with proper Azure integration.
