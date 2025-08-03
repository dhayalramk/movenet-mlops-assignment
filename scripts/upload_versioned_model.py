import os
import boto3
import subprocess
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

# Define model subdirectories and source URLs
MODELS = {
    "singlepose-lightning": "https://storage.googleapis.com/tfhub-lite-models/google/lite-model/movenet/singlepose/lightning/tflite/float16/4.tflite",
    "singlepose-thunder": "https://storage.googleapis.com/tfhub-lite-models/google/lite-model/movenet/singlepose/thunder/tflite/float16/4.tflite",
    "multipose-lightning": "https://storage.googleapis.com/tfhub-lite-models/google/lite-model/movenet/multipose/lightning/tflite/float16/1.tflite"
}

# ---------- Download all models ----------
def download_models():
    print("▶️ Downloading models...")

    for name, url in MODELS.items():
        model_path = os.path.join(MODEL_DIR, name)
        os.makedirs(model_path, exist_ok=True)
        output_file = os.path.join(model_path, "model.json")  # renamed for tfjs compatibility

        print(f"⏬ {name} → model.json")
        try:
            subprocess.run([
                "curl", "-L", url,
                "-o", output_file,
                "-H", "User-Agent: Mozilla/5.0"
            ], check=True)
        except subprocess.CalledProcessError as e:
            raise Exception(f"❌ Failed to download {name}. {e}")
        print(f"✅ Downloaded: {output_file}")

# ---------- Upload to S3 ----------
def upload_to_s3():
    print("⬆️ Uploading to S3...")

    s3 = boto3.client("s3", region_name=REGION)

    for root, dirs, files in os.walk(MODEL_DIR):
        for file in files:
            local_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_path, MODEL_DIR)
            s3_key = os.path.join(S3_PREFIX, relative_path).replace("\\", "/")

            print(f"→ Uploading: s3://{BUCKET_NAME}/{s3_key}")
            s3.upload_file(local_path, BUCKET_NAME, s3_key)

    print(f"✅ Upload complete: s3://{BUCKET_NAME}/{S3_PREFIX}")

# ---------- Run ----------
if __name__ == "__main__":
    download_models()
    upload_to_s3()
