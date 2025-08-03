from fastapi import FastAPI
from app.routes import router as api_router
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI(
    title="MoveNet Backend API",
    version="1.0.0",
    description="Backend for MoveNet pose detection MLOps project"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or replace "*" with your CloudFront domain to restrict
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check route
@app.get("/ping", tags=["Health"])
def ping():
    return {"message": "success"}

# Mount all API routes from routes.py
app.include_router(api_router)

# Optional: run locally with uvicorn if this file is executed directly
if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True) 
