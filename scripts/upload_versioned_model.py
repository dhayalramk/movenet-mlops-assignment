# ✅ scripts/upload_versioned_model.py
import os
import zipfile
import boto3
import subprocess
from datetime import datetime
from dotenv import load_dotenv
import requests

load_dotenv()

# ---------- Configuration ----------
KAGGLE_USERNAME = os.getenv("KAGGLE_USERNAME")
KAGGLE_KEY = os.getenv("KAGGLE_KEY")
KAGGLE_MODEL_URL = "google/movenet"
MODEL_FILE = "movenet_singlepose_lightning.tflite"  # Change as needed

ENV = os.getenv("ENV", "prod")
REGION = os.getenv("AWS_REGION", "ap-south-1")
ACCOUNT_ID = os.getenv("AWS_ACCOUNT_ID")
VERSION = os.getenv("MODEL_VERSION", datetime.utcnow().strftime("%Y%m%d"))

BUCKET_NAME = f"{ACCOUNT_ID}-{ENV}-movenet-models"
MODEL_DIR = f"/tmp/movenet_model_{VERSION}/"
S3_PREFIX = f"models/{VERSION}/"

os.makedirs(MODEL_DIR, exist_ok=True)

# ---------- Download model from Kaggle ----------
# ---------- Download model from TensorFlow Hub ----------

def download_model():
    print("▶️ Downloading model from TensorFlow Hub...")

    model_url = "https://storage.googleapis.com/tfhub-lite-models/google/lite-model/movenet/singlepose/lightning/tflite/float16/4.tflite"
    model_path = os.path.join(MODEL_DIR, MODEL_FILE)

    response = requests.get(model_url)
    if response.status_code != 200:
        raise Exception(f"❌ Failed to download model. HTTP {response.status_code}")

    with open(model_path, "wb") as f:
        f.write(response.content)

    print(f"✅ Model downloaded to {model_path}")


# ---------- Upload to S3 ----------
def upload_to_s3():
    print("⬆️ Uploading to S3...")

    s3 = boto3.client("s3", region_name=REGION)

    for root, dirs, files in os.walk(MODEL_DIR):
        for file in files:
            local_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_path, MODEL_DIR)
            s3_key = os.path.join(S3_PREFIX, relative_path)

            print(f"→ Uploading: {s3_key}")
            s3.upload_file(local_path, BUCKET_NAME, s3_key)

    print(f"✅ Upload complete to s3://{BUCKET_NAME}/{S3_PREFIX}")

if __name__ == "__main__":
    download_model()
    upload_to_s3()
