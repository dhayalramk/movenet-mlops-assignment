#!/usr/bin/env python3
import os
import shutil
import subprocess
import tarfile
import requests
import boto3
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# ---------- Configuration ----------
ENV       = os.getenv("ENV", "prod")
REGION    = os.getenv("AWS_REGION", "ap-south-1")
ACCOUNT_ID= os.getenv("AWS_ACCOUNT_ID")
VERSION   = os.getenv("MODEL_VERSION", datetime.utcnow().strftime("%Y%m%d"))

BUCKET    = f"{ACCOUNT_ID}-{ENV}-movenet-models"
BASE_TMP  = f"/tmp/movenet_tfjs_models_{VERSION}"
RAW_DIR   = os.path.join(BASE_TMP, "raw")
OUT_DIR   = BASE_TMP
S3_PREFIX = f"models/{VERSION}/"

# TF-Hub TF2 SavedModel compressed URLs
MODELS = {
    "singlepose-lightning":   "https://tfhub.dev/google/movenet/singlepose/lightning/4?tf-hub-format=compressed",
    "singlepose-thunder":     "https://tfhub.dev/google/movenet/singlepose/thunder/4?tf-hub-format=compressed",
    "multipose-lightning":    "https://tfhub.dev/google/movenet/multipose/lightning/1?tf-hub-format=compressed",
}

def download_and_extract():
    os.makedirs(RAW_DIR, exist_ok=True)
    for name, url in MODELS.items():
        print(f"▶️  Downloading & extracting {name}...")
        r = requests.get(url, stream=True)
        r.raise_for_status()
        tar_path = os.path.join(RAW_DIR, f"{name}.tar.gz")
        with open(tar_path, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)

        extract_to = os.path.join(RAW_DIR, name)
        temp_extract = os.path.join(extract_to, "_temp")
        os.makedirs(temp_extract, exist_ok=True)

        # Extract into temp folder
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=temp_extract)

        # Move contents to `saved_model` inside model dir
        final_path = os.path.join(extract_to, "saved_model")
        os.makedirs(final_path, exist_ok=True)

        for item in os.listdir(temp_extract):
            shutil.move(os.path.join(temp_extract, item), final_path)

        shutil.rmtree(temp_extract)
        print(f"   ✅  Extracted and organized saved_model for {name}")


def convert_to_tfjs():
    for name in MODELS:
        src = os.path.join(RAW_DIR, name, "saved_model")
        dst = os.path.join(OUT_DIR, name)
        os.makedirs(dst, exist_ok=True)
        print(f"🔄  Converting {name} → TF.js format…")
        subprocess.run([
            "tensorflowjs_converter",
            "--input_format", "tf_saved_model",
            "--output_format", "tfjs_graph_model",
            "--signature_name", "serving_default",
            "--saved_model_tags", "serve",
            src, dst
        ], check=True)
        print(f"   ✅  {name} converted at {dst}")

def upload_to_s3():
    s3 = boto3.client("s3", region_name=REGION)
    for root, _, files in os.walk(OUT_DIR):
        for fn in files:
            if fn.endswith((".json",".bin")):
                local = os.path.join(root, fn)
                rel   = os.path.relpath(local, OUT_DIR).replace("\\","/")
                for prefix in (S3_PREFIX, "models/"):
                    key = prefix + rel
                    print(f"⬆️  Uploading s3://{BUCKET}/{key}")
                    s3.upload_file(local, BUCKET, key)
    print("✅  Upload complete!")

if __name__ == "__main__":
    # Clean out any prior state
    if os.path.isdir(BASE_TMP):
        shutil.rmtree(BASE_TMP)

    download_and_extract()
    convert_to_tfjs()
    upload_to_s3()
