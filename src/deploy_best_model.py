# deploy_best_model.py

import os
import requests
from huggingface_hub import login, upload_file

# Environment variables
HF_TOKEN = os.getenv("HF_TOKEN")
HF_REPO_ID = os.getenv("HF_REPO_ID")
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")

# Files to upload
FILES_TO_UPLOAD = [
    ("best_model.pth", "best_model.pth"),
    ("best_prediction.png", "best_prediction.png")
]

# Step 1: Upload files to Hugging Face
if HF_TOKEN and HF_REPO_ID:
    try:
        print("🔐 Logging into Hugging Face...")
        login(token=HF_TOKEN)
        
        for local_path, repo_path in FILES_TO_UPLOAD:
            if os.path.exists(local_path):
                print(f"📤 Uploading {local_path} to {HF_REPO_ID}...")
                upload_file(
                    path_or_fileobj=local_path,
                    path_in_repo=repo_path,
                    repo_id=HF_REPO_ID,
                    token=HF_TOKEN,
                )
            else:
                print(f"⚠️ File not found: {local_path}")
    except Exception as e:
        print(f"❌ Hugging Face upload failed: {e}")
else:
    print("❗ Hugging Face credentials not found in environment variables.")

# Step 2: Send Slack notification
if SLACK_WEBHOOK_URL:
    try:
        print("📣 Sending Slack notification...")
        message = {
            "text": "✅ *MRI Model Training Complete!*\nNew best model and prediction have been uploaded to 🤗 Hugging Face."
        }
        response = requests.post(SLACK_WEBHOOK_URL, json=message)
        if response.status_code != 200:
            raise Exception(f"Slack API error: {response.status_code}, {response.text}")
    except Exception as e:
        print(f"❌ Slack notification failed: {e}")
else:
    print("❗ Slack webhook URL not found in environment variables.")

