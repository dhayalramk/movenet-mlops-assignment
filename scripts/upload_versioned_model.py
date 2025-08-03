import os
import boto3
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

# TFJS-compatible model URLs
MODELS = {
    "singlepose-lightning": "https://storage.googleapis.com/tfjs-models/savedmodel/pose/singlepose/lightning/model.json",
    "singlepose-thunder": "https://storage.googleapis.com/tfjs-models/savedmodel/pose/singlepose/thunder/model.json",
    "multipose-lightning": "https://storage.googleapis.com/tfjs-models/savedmodel/pose/multipose/lightning/model.json"
}


def download_tfjs_models():
    print("▶️ Downloading TFJS models...")

    os.makedirs(MODEL_DIR, exist_ok=True)

    for name, url in MODELS.items():
        model_path = os.path.join(MODEL_DIR, name)
        os.makedirs(model_path, exist_ok=True)

        try:
            print(f"⏬ Downloading {name} model.json...")
            model_json = requests.get(url)
            model_json.raise_for_status()

            model_json_path = os.path.join(model_path, "model.json")
            with open(model_json_path, "wb") as f:
                f.write(model_json.content)

            # Parse JSON to get .bin files
            bin_files = model_json.json().get("weightsManifest", [])[0].get("paths", [])

            for bin_file in bin_files:
                bin_url = url.replace("model.json", bin_file)
                print(f"⏬ Downloading {bin_file}...")
                bin_data = requests.get(bin_url)
                bin_data.raise_for_status()
                with open(os.path.join(model_path, bin_file), "wb") as f:
                    f.write(bin_data.content)

            print(f"✅ Downloaded: {name} with model.json and weights")

        except Exception as e:
            print(f"❌ Error downloading {name}: {e}")


def upload_to_s3():
    print("⬆️ Uploading to S3...")

    s3 = boto3.client("s3", region_name=REGION)

    for root, dirs, files in os.walk(MODEL_DIR):
        for file in files:
            local_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_path, MODEL_DIR)

            versioned_key = os.path.join(S3_PREFIX, relative_path).replace("\\", "/")
            print(f"→ Uploading versioned: s3://{BUCKET_NAME}/{versioned_key}")
            s3.upload_file(local_path, BUCKET_NAME, versioned_key)

            if "/" in relative_path:
                latest_key = os.path.join("models", relative_path).replace("\\", "/")
                print(f"→ Uploading stable: s3://{BUCKET_NAME}/{latest_key}")
                s3.upload_file(local_path, BUCKET_NAME, latest_key)

    print("✅ Upload complete.")


if __name__ == "__main__":
    try:
        download_tfjs_models()
        upload_to_s3()
    except Exception as err:
        print(f"💥 Error: {err}")
