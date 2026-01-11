from fastapi import FastAPI, UploadFile, File, Depends
from sqlalchemy.orm import Session
from database.db import get_db, engine
import database.models as models
from audio_analysis.audio_processor import process_recorded_call
from audio_analysis.fusion_engine import generate_multimodal_score

# Create tables if they don't exist
models.Base.metadata.create_all(bind=engine)

app = FastAPI()

@app.post("/analyze-call")
async def analyze_call(file: UploadFile = File(...), db: Session = Depends(get_db)):
    # 1. Process Audio (Transcription + Acoustic Features)
    raw_data = process_recorded_call(file.file)
    
    # 2. Get XLNet Prediction (Semantic Risk)
    # Placeholder: In production, this calls your trained XLNet model
    xlnet_prob = 0.85 
    
    # 3. Fuse Results into one 0-1 Score
    final_result = generate_multimodal_score(xlnet_prob, raw_data['vocal_features'])
    
    # 4. Save to Database for History
    report = models.FraudReport(
        filename=file.filename,
        final_score=final_result['final_score'],
        status=final_result['status'],
        transcription=raw_data['text'],
        breakdown=final_result['contributions']
    )
    db.add(report)
    db.commit()
    
    return final_result