from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, Any, Optional
import boto3
import os
import json
import base64
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

router = APIRouter()

# ---------- Pydantic Model ----------
class InferenceResult(BaseModel):
    session_id: str
    model_version: str
    keypoints: list
    metadata: Dict[str, Any]
    image_base64: Optional[str] = None
    video_base64: Optional[str] = None


# ---------- Upload Inference ----------
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

        # Save pose inference JSON
        pose_key = f"inference/{session_id}_{timestamp}.json"
        s3.put_object(
            Bucket=logs_bucket,
            Key=pose_key,
            Body=json.dumps(payload.dict(), indent=2),
            ContentType="application/json"
        )

        image_key, video_key = None, None

        # Optional: Save image
        if payload.image_base64:
            image_data = payload.image_base64.split(",")[1]
            image_bytes = base64.b64decode(image_data)
            image_key = f"images/{session_id}_{timestamp}.jpg"
            s3.put_object(
                Bucket=logs_bucket,
                Key=image_key,
                Body=image_bytes,
                ContentType="image/jpeg"
            )

        # Optional: Save video
        if payload.video_base64:
            video_data = payload.video_base64.split(",")[1]
            video_bytes = base64.b64decode(video_data)
            video_key = f"videos/{session_id}_{timestamp}.mp4"
            s3.put_object(
                Bucket=logs_bucket,
                Key=video_key,
                Body=video_bytes,
                ContentType="video/mp4"
            )

        return JSONResponse(status_code=200, content={
            "message": "Upload successful",
            "pose_key": pose_key,
            "image_key": image_key,
            "video_key": video_key
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# ---------- Monitoring ----------
@router.get("/metrics", tags=["Monitoring"])
async def get_metrics():
    return {
        "status": "ok",
        "model_version": "v1",
        "avg_inference_time_ms": 75,
        "timestamp": datetime.utcnow().isoformat()
    }
