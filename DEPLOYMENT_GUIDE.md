# 🚀 **Streamlit Cloud Deployment Guide**

## **Overview**
This guide will help you deploy your Engagement Concordance Score (ECS) application to Streamlit Cloud for online access.

## **Prerequisites**
- ✅ GitHub repository set up (`EngagementMiser-Streamlit`)
- ✅ All specialized models working locally
- ✅ ngrok tunnel configured for local database access
- ✅ Streamlit Cloud account

## **Step 1: Streamlit Cloud Setup**

### **1.1 Access Streamlit Cloud**
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Click **"New app"**

### **1.2 Connect Repository**
- **Repository**: `Justern/EngagementMiser-Streamlit`
- **Branch**: `main`
- **Main file path**: `streamlit_app_local_db.py` ⚠️ **IMPORTANT: Use this file, not streamlit_app_fixed.py**

### **1.3 Advanced Settings**
- **Python version**: 3.9 or 3.10
- **Requirements file**: `requirements.txt`

## **Step 2: Configure Secrets**

### **2.1 Local Database (via ngrok)**
```toml
[local_db]
server = "0.tcp.ngrok.io:15165"
database = "EngagementMiser"
username = ""
password = ""
```

### **2.2 Azure Database (backup)**
```toml
[azure_db]
server = "ecs-sql-server-engagementmiser.database.windows.net"
database = "ecs_tweets_db"
username = "ecsadmin"
password = "EngagementMiser!"
```

## **Step 3: Deploy**

1. Click **"Deploy!"**
2. Wait for build to complete
3. Your app will be available at: `https://your-app-name.streamlit.app`

## **Step 4: Test Deployment**

### **4.1 Verify Database Connection**
- Check if the app connects to your local MSSQL via ngrok
- Verify tweet selection dropdown works

### **4.2 Test Model Functionality**
- Select a tweet from the dropdown
- Run ECS analysis
- Verify all 10 specialized models produce different scores

## **Troubleshooting**

### **Common Issues**

#### **❌ Build Fails**
- Check Python version compatibility
- Verify all imports in `requirements.txt`
- Check file paths in `deployment_config.py`

#### **❌ Database Connection Fails**
- Ensure ngrok is running: `ngrok http 1433`
- Update ngrok URL in Streamlit Cloud secrets
- Check Windows Firewall settings

#### **❌ Models Not Working**
- Verify all 10 specialized model folders exist
- Check `simple_score.py` files in each model folder
- Ensure proper file permissions

### **Debug Commands**

#### **Check ngrok Status**
```bash
ngrok http 1433
```

#### **Test Local Database**
```bash
python test_deployment.py
```

#### **Verify Model Paths**
```bash
python -c "import deployment_config; print('Models loaded successfully')"
```

## **File Structure for Deployment**

```
EngagementMiser-Streamlit/
├── streamlit_app_local_db.py      # Main Streamlit app
├── deployment_config.py           # Model configuration
├── requirements.txt               # Python dependencies
├── .streamlit/
│   └── secrets.toml             # Local secrets (not deployed)
└── DEPLOYMENT_GUIDE.md           # This file
```

## **Maintenance**

### **Updating ngrok URL**
1. Get new ngrok URL: `ngrok http 1433`
2. Update Streamlit Cloud secrets
3. Redeploy app

### **Adding New Models**
1. Add model folder to your local Models directory
2. Update `deployment_config.py` with new method
3. Commit and push to GitHub
4. Redeploy on Streamlit Cloud

## **Security Notes**

- ⚠️ **Never commit secrets to GitHub**
- 🔒 **Use Streamlit Cloud secrets for sensitive data**
- 🌐 **ngrok exposes your local database - use only for development/testing**
- 🚀 **For production, consider migrating to Azure SQL Database**

## **Support**

If you encounter issues:
1. Check Streamlit Cloud logs
2. Verify ngrok tunnel is active
3. Test models locally first
4. Check GitHub repository for latest updates

---

**🎯 Your app should now be accessible online with all 10 specialized models working!**
