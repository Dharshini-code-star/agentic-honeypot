"""
Agentic Honey-Pot – AI Scam Engagement System
GUVI-Compatible FastAPI Application
"""

from dotenv import load_dotenv
import os
from pathlib import Path
import logging

from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.security import APIKeyHeader
from fastapi.responses import JSONResponse

# -------------------------------------------------
# Environment setup
# -------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=BASE_DIR / ".env")

API_KEY = os.getenv("API_KEY")

# -------------------------------------------------
# Logging
# -------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# -------------------------------------------------
# FastAPI app
# -------------------------------------------------

app = FastAPI(
    title="Agentic Honey-Pot API",
    description="GUVI Hackathon Compatible API",
    version="1.0.0"
)

# -------------------------------------------------
# API Key Security
# -------------------------------------------------

api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)

if not API_KEY:
    logger.warning("API_KEY environment variable not set! Authentication will fail.")

def verify_api_key(api_key: str = Depends(api_key_header)) -> bool:
    if not API_KEY:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server API key not configured"
        )

    if api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authenticated"
        )

    return True

# -------------------------------------------------
# Health Check
# -------------------------------------------------

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "agentic-honeypot"
    }

# -------------------------------------------------
# GUVI-Compatible Endpoint
# -------------------------------------------------

@app.post("/api/agentic-honeypot")
async def agentic_honeypot(
    request: Request,
    authenticated: bool = Depends(verify_api_key)
):
    body = await request.json()

    logger.info("GUVI request received")

    return JSONResponse(
        status_code=200,
        content={
            "status": "success",
            "message": "Request processed successfully",
            "data": body
        }
    )

# -------------------------------------------------
# Global Exception Handler
# -------------------------------------------------

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)

    return JSONResponse(
        status_code=200,
        content={
            "status": "error",
            "message": "Safe fallback response",
            "data": {}
        }
    )

# -------------------------------------------------
# Local run support
# -------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
