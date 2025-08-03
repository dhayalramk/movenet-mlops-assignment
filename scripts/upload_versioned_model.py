import os
import zipfile
import boto3
import subprocess
from datetime import datetime
from dotenv import load_dotenv
import json

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

def get_cloudfront_info(stack_name: str):
    print(f"🔍 Fetching CloudFront info from stack: {stack_name}")
    result = subprocess.run([
        "aws", "cloudformation", "describe-stacks",
        "--stack-name", stack_name,
        "--region", REGION,
        "--query", "Stacks[0].Outputs",
        "--output", "json"
    ], stdout=subprocess.PIPE, check=True)

    outputs = json.loads(result.stdout)
    dist_id = None
    dist_url = None

    for item in outputs:
        if item["OutputKey"] == "CloudFrontURL":
            dist_url = item["OutputValue"]
        elif item["OutputKey"] == "CloudFrontDistributionId":
            dist_id = item["OutputValue"]

    if not dist_url or not dist_id:
        raise ValueError("❌ CloudFront values not found in stack outputs")

    print(f"✅ CloudFront URL: {dist_url}")
    print(f"✅ CloudFront Distribution ID: {dist_id}")
    return dist_id, dist_url


STACK_NAME = os.getenv("STACK_NAME", "frontend-static-site")
CLOUDFRONT_DIST_ID, CLOUDFRONT_URL = get_cloudfront_info(STACK_NAME)


# ---------- Download model from Kaggle ----------
def download_model():
    print("▶️ Downloading model from Kaggle...")

    os.environ["KAGGLE_USERNAME"] = KAGGLE_USERNAME
    os.environ["KAGGLE_KEY"] = KAGGLE_KEY

    # Download model files
    subprocess.run([
        "kaggle", "datasets", "download",
        "-d", KAGGLE_MODEL_URL,
        "-p", MODEL_DIR,
        "--unzip"
    ], check=True)

    print(f"✅ Model downloaded to {MODEL_DIR}")

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
