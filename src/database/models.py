from sqlalchemy import Column, Integer, String, Float, DateTime, JSON
from datetime import datetime
from .db import Base

class FraudReport(Base):
    __tablename__ = "fraud_reports"
    
    id = Column(Integer, primary_key=True, index=True)
    caller_id = Column(String) # Phone number or identifier
    final_score = Column(Float) # 0 to 1 result
    status = Column(String) # e.g., "Critical", "Suspicious", "Safe"
    
    # Store the breakdown for detailed UI views
    analysis_details = Column(JSON) # Stores text_contribution and vocal_risk
    
    transcription = Column(String) # The raw text from Whisper
    timestamp = Column(DateTime, default=datetime.utcnow)