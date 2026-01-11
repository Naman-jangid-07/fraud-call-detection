import torch
import numpy as np

def calculate_final_fraud_score(text_prob, vocal_features):
    """
    Fuses textual and acoustic evidence into one score.
    
    text_prob: Probability from XLNet (0 to 1)
    vocal_features: Dictionary containing pitch, urgency, etc.
    """
    
    # 1. Normalize Audio Features (e.g., high pitch/urgency increases risk)
    # In a real system, these thresholds come from your training data.
    audio_risk_score = 0.0
    if vocal_features['avg_pitch'] > 250:  # Example threshold for stress
        audio_risk_score += 0.4
    if vocal_features['urgency_score'] > 0.1: # Example for rapid speech
        audio_risk_score += 0.3
    
    audio_risk_score = min(audio_risk_score, 1.0)

    # 2. Weighted Fusion Strategy
    # Text is usually a stronger indicator, so we give it more weight.
    TEXT_WEIGHT = 0.7
    AUDIO_WEIGHT = 0.3
    
    final_score = (text_prob * TEXT_WEIGHT) + (audio_risk_score * AUDIO_WEIGHT)
    
    # 3. Decision Logic
    status = "SAFE"
    if final_score > 0.7:
        status = "CRITICAL FRAUD"
    elif final_score > 0.4:
        status = "SUSPICIOUS"
        
    return {
        "final_score": round(final_score, 4),
        "status": status,
        "breakdown": {
            "text_contribution": text_prob,
            "audio_contribution": audio_risk_score
        }
    }