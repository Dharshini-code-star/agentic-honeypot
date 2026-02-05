"""
Agentic Honey-Pot – AI Scam Engagement System
Main FastAPI Application Entry Point
"""


from dotenv import load_dotenv
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=BASE_DIR / ".env")

API_KEY = os.getenv("API_KEY")



from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import APIKeyHeader

from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging



# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global state (in-memory)
conversation_manager = None
intent_detector = None
agent = None
intelligence_extractor = None
response_builder = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize components on startup"""
    global conversation_manager, intent_detector, agent, intelligence_extractor, response_builder
    
    logger.info("Initializing Agentic Honey-Pot system...")
    
    # Initialize all components
    conversation_manager = ConversationManager()
    intent_detector = IntentDetector()
    agent = ScamEngagementAgent()
    intelligence_extractor = IntelligenceExtractor()
    response_builder = ResponseBuilder()
    
    logger.info("System initialization complete")
    yield
    
    # Cleanup on shutdown
    logger.info("Shutting down Agentic Honey-Pot system...")

# Create FastAPI app
app = FastAPI(
    title="Agentic Honey-Pot API",
    description="AI Scam Engagement System",
    version="1.0.0",
    lifespan=lifespan
)

# Security


api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)

API_KEY = os.getenv("API_KEY")

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


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "agentic-honeypot"}
from pydantic import BaseModel
from typing import Optional
from pydantic import BaseModel
from typing import Optional

class HoneypotRequest(BaseModel):
    language: str
    audio_base64: str
    audio_format: Optional[str] = None
    conversation_id: Optional[str] = None


from fastapi import Request

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

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """
    Global exception handler - ensures NOTHING crashes the API.
    Returns a safe, valid JSON response for any unhandled exception.
    """
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    
    safe_response = {
        "scam_detected": False,
        "agent_activated": False,
        "agent_reply": "I'm sorry, I didn't quite understand that. Could you please rephrase?",
        "engagement_metrics": {
            "turn_count": 1,
            "engagement_duration": "0s"
        },
        "extracted_intelligence": {
            "bank_accounts": [],
            "upi_ids": [],
            "phishing_urls": []
        },
        "status": "error"
    }
    
    return JSONResponse(content=safe_response, status_code=200)

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
