"""
FastAPI server for fraud call detection using Gemini
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict
from contextlib import asynccontextmanager
import logging
from pathlib import Path
import sys
import os

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.fraud_analyzer import FraudAnalyzer
from api.audio_transcriber import AudioTranscriber

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global services
fraud_analyzer = None
audio_transcriber = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown"""
    global fraud_analyzer, audio_transcriber
    
    # Startup
    logger.info("Starting Fraud Detection API...")
    
    try:
        # Check for API key
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            # Temporary hardcoded key - REPLACE WITH YOUR ACTUAL KEY
            api_key = "AIzaSyDytan6HeRy8MBJRrfN3mLFLWvl1gOYL4U"  # ⚠️ PUT YOUR KEY HERE
            logger.warning("⚠️ Using hardcoded API key (temporary)")
        else:
            logger.info("✓ Gemini API key found from .env")
        
        # Initialize fraud analyzer with Gemini
        fraud_analyzer = FraudAnalyzer(api_key=api_key)
        logger.info("✓ Fraud analyzer initialized with Gemini AI")
        
        # Initialize audio transcriber
        audio_transcriber = AudioTranscriber()
        logger.info("✓ Audio transcriber initialized")
        
        logger.info("✅ API ready to accept requests!")
        
    except Exception as e:
        logger.error(f"❌ Error during startup: {str(e)}")
        raise
    
    yield  # Server is running
    
    # Shutdown
    logger.info("Shutting down Fraud Detection API...")


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Fraud Call Detection API",
    description="Real-time audio fraud detection system powered by Gemini AI",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Fraud Call Detection API",
        "version": "1.0.0",
        "status": "running",
        "powered_by": "Gemini AI",
        "endpoints": {
            "health": "/health",
            "analyze_text": "/api/analyze/text",
            "analyze_audio": "/api/analyze/audio",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": fraud_analyzer is not None,
        "transcriber_loaded": audio_transcriber is not None,
        "ai_backend": "Gemini AI"
    }


class TextAnalysisRequest(BaseModel):
    """Request model for text analysis"""
    text: str


class TextAnalysisResponse(BaseModel):
    """Response model for text analysis"""
    is_fraud: bool
    fraud_probability: float
    legitimate_probability: float
    risk_level: str
    detected_patterns: Dict[str, bool]
    recommendations: list
    reasoning: str
    key_phrases: list


@app.post("/api/analyze/text", response_model=TextAnalysisResponse)
async def analyze_text(request: TextAnalysisRequest):
    """
    Analyze text for fraud indicators
    
    Args:
        request: Text to analyze
        
    Returns:
        Fraud analysis results
    """
    if fraud_analyzer is None:
        raise HTTPException(status_code=503, detail="Fraud analyzer not initialized")
    
    try:
        logger.info(f"Analyzing text: {request.text[:50]}...")
        
        # Analyze text
        result = fraud_analyzer.analyze_text(request.text)
        
        return TextAnalysisResponse(**result)
        
    except Exception as e:
        logger.error(f"Error analyzing text: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/analyze/audio")
async def analyze_audio(file: UploadFile = File(...)):
    """
    Analyze audio file for fraud indicators
    
    Args:
        file: Audio file (wav, mp3, m4a, etc.)
        
    Returns:
        Fraud analysis results including transcription
    """
    if fraud_analyzer is None or audio_transcriber is None:
        raise HTTPException(status_code=503, detail="Services not initialized")
    
    temp_path = None
    
    try:
        logger.info(f"Received audio file: {file.filename}")
        
        # Save uploaded file temporarily
        temp_path = Path(f"temp_{file.filename}")
        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Transcribe audio
        logger.info("Transcribing audio...")
        transcription = audio_transcriber.transcribe(str(temp_path))
        
        if not transcription or transcription.strip() == "":
            # Clean up temp file
            if temp_path and temp_path.exists():
                temp_path.unlink()
            raise HTTPException(status_code=400, detail="Could not transcribe audio")
        
        logger.info(f"Transcription: {transcription[:100]}...")
        
        # Analyze transcribed text with Gemini
        analysis = fraud_analyzer.analyze_text(transcription)
        
        # Clean up temp file
        if temp_path and temp_path.exists():
            temp_path.unlink()
        
        # Return results with transcription
        return {
            "transcription": transcription,
            "analysis": analysis,
            "filename": file.filename
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing audio: {str(e)}")
        # Clean up temp file if it exists
        if temp_path and temp_path.exists():
            temp_path.unlink()
        raise HTTPException(status_code=500, detail=str(e))


class BatchAnalysisRequest(BaseModel):
    """Request model for batch analysis"""
    texts: list[str]


@app.post("/api/analyze/batch")
async def analyze_batch(request: BatchAnalysisRequest):
    """
    Analyze multiple texts in batch
    
    Args:
        request: List of texts to analyze
        
    Returns:
        List of fraud analysis results
    """
    if fraud_analyzer is None:
        raise HTTPException(status_code=503, detail="Fraud analyzer not initialized")
    
    try:
        results = []
        
        for text in request.texts:
            result = fraud_analyzer.analyze_text(text)
            results.append(result)
        
        return {
            "total": len(results),
            "fraud_detected": sum(1 for r in results if r['is_fraud']),
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Error in batch analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)