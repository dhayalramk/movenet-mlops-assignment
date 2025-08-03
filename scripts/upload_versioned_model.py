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

# Define model subdirectories and source URLs (TFHub redirectors)
MODELS = {
    "singlepose-lightning": "https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/float16/4?lite-format=tflite",
    "singlepose-thunder": "https://tfhub.dev/google/lite-model/movenet/singlepose/thunder/tflite/float16/4?lite-format=tflite",
    "multipose-lightning": "https://tfhub.dev/google/lite-model/movenet/multipose/lightning/tflite/float16/1?lite-format=tflite"
}

# ---------- Check if file is valid binary ----------
def is_valid_tflite(file_path):
    try:
        with open(file_path, "rb") as f:
            header = f.read(4)
            return header != b"<?xm"
    except Exception:
        return False

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

            if not is_valid_tflite(output_file):
                raise Exception("Invalid file downloaded — likely an HTML/XML error page.")

        except Exception as e:
            print(f"❌ Skipping {name}: {e}")
            continue

        print(f"✅ Downloaded: {output_file}")

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
        download_models()
        upload_to_s3()
    except Exception as err:
        print(f"💥 Error: {err}")
