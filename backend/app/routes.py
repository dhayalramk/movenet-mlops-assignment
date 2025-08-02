from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, Any
import boto3
import os
import json
import base64
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

router = APIRouter()

class InferenceResult(BaseModel):
    session_id: str
    model_version: str
    keypoints: list
    metadata: Dict[str, Any]
    image_base64: str  # Expect full string with header

@router.post("/upload", tags=["Inference"])
async def upload_result(payload: InferenceResult):
    try:
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        session_id = payload.session_id
        env = os.getenv("ENV", "dev")
        region = os.getenv("AWS_REGION", "ap-south-1")
        account_id = os.getenv("AWS_ACCOUNT_ID")

        logs_bucket = f"{account_id}-{env}-movenet-logs"
        s3 = boto3.client("s3", region_name=region)

        # Save pose data JSON
        pose_key = f"inference/{session_id}_{timestamp}.json"
        s3.put_object(
            Bucket=logs_bucket,
            Key=pose_key,
            Body=json.dumps(payload.dict(), indent=2),
            ContentType="application/json"
        )

        # Save image
        image_data = payload.image_base64.split(",")[1]  # Remove header
        image_bytes = base64.b64decode(image_data)
        image_key = f"images/{session_id}_{timestamp}.jpg"
        s3.put_object(
            Bucket=logs_bucket,
            Key=image_key,
            Body=image_bytes,
            ContentType="image/jpeg"
        )

        return JSONResponse(status_code=200, content={
            "message": "Upload successful",
            "pose_key": pose_key,
            "image_key": image_key
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
