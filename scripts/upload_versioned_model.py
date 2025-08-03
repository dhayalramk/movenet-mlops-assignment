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
MODEL_DIR = f"/tmp/movenet_tfjs_models_{VERSION}/"
S3_PREFIX = f"models/{VERSION}/"

MODELS = {
    "singlepose-lightning": "https://tfhub.dev/google/tfjs-model/movenet/singlepose/lightning/4?tfjs-format=file",
    "singlepose-thunder": "https://tfhub.dev/google/tfjs-model/movenet/singlepose/thunder/4?tfjs-format=file",
    "multipose-lightning": "https://tfhub.dev/google/tfjs-model/movenet/multipose/lightning/1?tfjs-format=file",
}

def download_and_extract_tfjs_models():
    print("▶️ Downloading and extracting TFJS models...")

    os.makedirs(MODEL_DIR, exist_ok=True)

    for name, url in MODELS.items():
        print(f"⏬ Downloading {name}...")
        model_path = os.path.join(MODEL_DIR, name)
        os.makedirs(model_path, exist_ok=True)

        try:
            # Manually resolve the .tar.gz URL by following redirect
            head = requests.head(url, allow_redirects=True)
            final_url = head.url
            filename = final_url.split("/")[-1]

            # Download
            tar_path = os.path.join(MODEL_DIR, f"{name}.tar.gz")
            r = requests.get(final_url, stream=True)
            r.raise_for_status()
            with open(tar_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"✅ Downloaded: {filename}")

            # Extract
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(path=model_path)
            print(f"📦 Extracted to: {model_path}")

        except Exception as e:
            print(f"❌ Error downloading/extracting {name}: {e}")
            continue

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

            # Upload to latest path
            if "/" in relative_path:
                latest_key = os.path.join("models", relative_path).replace("\\", "/")
                print(f"→ Uploading stable: s3://{BUCKET_NAME}/{latest_key}")
                s3.upload_file(local_path, BUCKET_NAME, latest_key)

    print("✅ Upload complete.")

if __name__ == "__main__":
    try:
        download_and_extract_tfjs_models()
        upload_to_s3()
    except Exception as err:
        print(f"💥 Error: {err}")
