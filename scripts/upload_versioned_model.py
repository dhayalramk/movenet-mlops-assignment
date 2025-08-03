import os
import boto3
import tarfile
import shutil
import subprocess
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
RAW_DIR = f"/tmp/movenet_tfjs_models_{VERSION}/raw/"
MODEL_DIR = f"/tmp/movenet_tfjs_models_{VERSION}/"
S3_PREFIX = f"models/{VERSION}/"

MODELS = {
    "singlepose-lightning": "https://tfhub.dev/google/movenet/singlepose/lightning/4?tf-hub-format=compressed",
    "singlepose-thunder": "https://tfhub.dev/google/movenet/singlepose/thunder/4?tf-hub-format=compressed",
    "multipose-lightning": "https://tfhub.dev/google/movenet/multipose/lightning/1?tf-hub-format=compressed"
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

        # Move contents into `saved_model/`
        final_path = os.path.join(extract_to, "saved_model")
        os.makedirs(final_path, exist_ok=True)

        for item in os.listdir(temp_extract):
            shutil.move(os.path.join(temp_extract, item), final_path)

        shutil.rmtree(temp_extract)
        print(f"   ✅  Extracted and organized saved_model for {name}")


def convert_to_tfjs():
    for name in MODELS:
        saved_model_dir = os.path.join(RAW_DIR, name, "saved_model")
        output_dir = os.path.join(MODEL_DIR, name)
        os.makedirs(output_dir, exist_ok=True)

        print(f"🔄  Converting {name} → TF.js format…")
        subprocess.run([
            "tensorflowjs_converter",
            "--input_format", "tf_saved_model",
            "--output_format", "tfjs_graph_model",
            "--signature_name", "serving_default",
            "--saved_model_tags", "serve",
            saved_model_dir,
            output_dir
        ], check=True)
        print(f"   ✅  Converted: {name}")


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
        download_and_extract()
        convert_to_tfjs()
        upload_to_s3()
    except subprocess.CalledProcessError as e:
        print(f"💥 Conversion failed: {e}")
    except Exception as err:
        print(f"💥 Error: {err}")
