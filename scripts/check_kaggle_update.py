import os
import hashlib
import boto3
import zipfile
import requests
import datetime
import json
import subprocess

KAGGLE_DATASET = "google/movenet"
LOCAL_MODEL_PATH = "/tmp/movenet_model"
MODEL_VARIANTS = ["singlepose-lightning", "singlepose-thunder", "multipose-lightning"]

def download_model(model_type):
    print(f"🔽 Downloading model: {model_type}")
    subprocess.run([
        "kaggle", "datasets", "download",
        "--dataset", KAGGLE_DATASET,
        "--unzip", "-p", f"{LOCAL_MODEL_PATH}/{model_type}"
    ], check=True)

def calculate_hash(folder_path):
    sha256 = hashlib.sha256()
    for root, _, files in os.walk(folder_path):
        for file in sorted(files):
            with open(os.path.join(root, file), 'rb') as f:
                while chunk := f.read(8192):
                    sha256.update(chunk)
    return sha256.hexdigest()

def model_changed(s3_client, bucket, model_type, local_hash):
    key = f"{model_type}/model_hash.txt"
    try:
        obj = s3_client.get_object(Bucket=bucket, Key=key)
        remote_hash = obj['Body'].read().decode('utf-8').strip()
        return local_hash != remote_hash
    except s3_client.exceptions.NoSuchKey:
        return True

def upload_model(s3_client, bucket, model_type, local_path, local_hash):
    for root, _, files in os.walk(local_path):
        for file in files:
            local_file = os.path.join(root, file)
            s3_key = f"{model_type}/{os.path.relpath(local_file, local_path)}"
            print(f"📤 Uploading {s3_key}")
            s3_client.upload_file(local_file, bucket, s3_key, ExtraArgs={'ACL': 'public-read'})

    # Store hash
    s3_client.put_object(
        Bucket=bucket,
        Key=f"{model_type}/model_hash.txt",
        Body=local_hash.encode('utf-8'),
        ACL='public-read'
    )

def main():
    bucket = os.environ.get("MODEL_BUCKET")
    region = os.environ.get("AWS_REGION")
    os.makedirs(LOCAL_MODEL_PATH, exist_ok=True)
    s3 = boto3.client("s3", region_name=region)

    for model in MODEL_VARIANTS:
        model_dir = os.path.join(LOCAL_MODEL_PATH, model)
        if os.path.exists(model_dir):
            subprocess.run(["rm", "-rf", model_dir])
        download_model(model)

        model_hash = calculate_hash(model_dir)
        print(f"🔍 Hash for {model}: {model_hash}")

        if model_changed(s3, bucket, model, model_hash):
            print(f"🚨 Model {model} has changed. Uploading new version.")
            upload_model(s3, bucket, model, model_dir, model_hash)
        else:
            print(f"✅ No changes for {model}. Skipping upload.")

if __name__ == "__main__":
    main()
