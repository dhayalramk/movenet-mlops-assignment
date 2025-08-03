import os
import boto3
import tarfile
import requests
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# ---------- Configuration ----------
ENV = os.getenv("ENV", "prod")
REGION = os.getenv("AWS_REGION", "ap-south-1")
ACCOUNT_ID = os.getenv("AWS_ACCOUNT_ID")
VERSION = os.getenv("MODEL_VERSION", datetime.utcnow().strftime("%Y%m%d"))

BUCKET_NAME = f"{ACCOUNT_ID}-{ENV}-movenet-models"
MODEL_DIR = f"/tmp/movenet_model_{VERSION}/"
S3_PREFIX = f"models/{VERSION}/"

# Define model tar.gz download URLs
MODELS = {
    "singlepose-lightning": "https://storage.googleapis.com/tfhub-modules/google/movenet/singlepose/lightning/4.tar.gz",
    "singlepose-thunder": "https://storage.googleapis.com/tfhub-modules/google/movenet/singlepose/thunder/4.tar.gz",
    "multipose-lightning": "https://storage.googleapis.com/tfhub-modules/google/movenet/multipose/lightning/1.tar.gz"
}

# ---------- Download and extract SavedModels ----------
def download_and_extract_models():
    print("▶️ Downloading and extracting models...")

    os.makedirs(MODEL_DIR, exist_ok=True)

    for name, url in MODELS.items():
        print(f"⏬ Downloading {name}...")

        tar_path = os.path.join(MODEL_DIR, f"{name}.tar.gz")
        model_path = os.path.join(MODEL_DIR, name)

        try:
            # Download .tar.gz file
            r = requests.get(url, stream=True)
            r.raise_for_status()
            with open(tar_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"✅ Downloaded: {tar_path}")

            # Extract .tar.gz
            os.makedirs(model_path, exist_ok=True)
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(path=model_path)
            print(f"📦 Extracted to: {model_path}")

        except Exception as e:
            print(f"❌ Error downloading/extracting {name}: {e}")
            continue

# ---------- Upload to S3 ----------
def upload_to_s3():
    print("⬆️ Uploading to S3...")

    s3 = boto3.client("s3", region_name=REGION)

    for root, dirs, files in os.walk(MODEL_DIR):
        for file in files:
            local_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_path, MODEL_DIR)

            # Upload to versioned path
            versioned_key = os.path.join(S3_PREFIX, relative_path).replace("\\", "/")
            print(f"→ Uploading versioned: s3://{BUCKET_NAME}/{versioned_key}")
            s3.upload_file(local_path, BUCKET_NAME, versioned_key)

            # Upload to stable/latest path
            if "/" in relative_path:
                latest_key = os.path.join("models", relative_path).replace("\\", "/")
                print(f"→ Uploading latest: s3://{BUCKET_NAME}/{latest_key}")
                s3.upload_file(local_path, BUCKET_NAME, latest_key)

    print(f"✅ Upload complete: versioned + stable paths uploaded.")

# ---------- Run ----------
if __name__ == "__main__":
    try:
        download_and_extract_models()
        upload_to_s3()
    except Exception as err:
        print(f"💥 Error: {err}")
