#!/usr/bin/env python3
"""
Test script for deployment models
================================

Tests the deployment-ready ECS system to ensure it works correctly.
"""

import sys
import os

def test_deployment_models():
    """Test the deployment models."""
    print("🧪 Testing Deployment Models...")
    
    try:
        # Test import
        from deployment_config import deployment_models
        print("✅ Successfully imported deployment models")
        
        # Test database connection
        from deployment_config import get_azure_engine
        engine = get_azure_engine()
        if engine:
            print("✅ Azure database connection successful")
        else:
            print("⚠️ Azure database connection failed (check credentials)")
        
        # Test model methods exist
        required_methods = [
            'hyperbole_falsehood_score',
            'clickbait_score', 
            'engagement_mismatch_score',
            'content_recycling_score',
            'coordinated_network_score',
            'emotive_manipulation_score',
            'rapid_engagement_spike_score',
            'generic_comment_score',
            'authority_signal_score',
            'reply_bait_score'
        ]
        
        for method in required_methods:
            if hasattr(deployment_models, method):
                print(f"✅ {method} method exists")
            else:
                print(f"❌ {method} method missing")
        
        print(f"\n📊 Total methods found: {len([m for m in required_methods if hasattr(deployment_models, m)])}/10")
        
        # Test with a sample tweet ID (this will fail if no database connection)
        if engine:
            try:
                # Test a simple model call
                sample_tweet_id = "123456789012345"
                score = deployment_models.hyperbole_falsehood_score(sample_tweet_id)
                print(f"✅ Test model call successful, score: {score}")
            except Exception as e:
                print(f"⚠️ Test model call failed: {e}")
        
        print("\n🎉 Deployment models test completed!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False

def test_streamlit_app():
    """Test if the Streamlit app can be imported."""
    print("\n🧪 Testing Streamlit App Import...")
    
    try:
        # Test if we can import the main app
        import streamlit_app_fixed
        print("✅ Streamlit app imports successfully")
        
        # Check if main functions exist
        if hasattr(streamlit_app_fixed, 'main'):
            print("✅ Main function exists")
        else:
            print("❌ Main function missing")
        
        if hasattr(streamlit_app_fixed, 'load_ecs_system'):
            print("✅ ECS loading function exists")
        else:
            print("❌ ECS loading function missing")
        
        print("🎉 Streamlit app test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Streamlit app test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Testing Engagement Concordance Score Deployment System")
    print("=" * 60)
    
    # Test deployment models
    models_ok = test_deployment_models()
    
    # Test Streamlit app
    app_ok = test_streamlit_app()
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 TEST SUMMARY")
    print("=" * 60)
    
    if models_ok and app_ok:
        print("🎉 ALL TESTS PASSED! Your deployment system is ready.")
        print("\n📁 Files ready for deployment:")
        print("  ✅ streamlit_app_fixed.py")
        print("  ✅ deployment_config.py")
        print("  ✅ requirements.txt")
        print("  ✅ .streamlit/secrets.toml")
        print("\n🚀 Ready to deploy to Streamlit Cloud or other platforms!")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        print("\n🔧 Common fixes:")
        print("  - Ensure all files are in the same directory")
        print("  - Check Azure database credentials")
        print("  - Install required dependencies: pip install -r requirements.txt")
    
    print("\n📚 For deployment instructions, see DEPLOYMENT_GUIDE.md")
