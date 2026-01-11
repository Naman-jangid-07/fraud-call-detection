from audio_analysis.speech_to_text import transcribe_call
from models.fraud_detector_model import get_model
import torch

class RealTimeDetector:
    def __init__(self, model_path):
        self.model = get_model()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def analyze_segment(self, audio_segment):
        # 1. Real-time STT using Whisper
        text = transcribe_call(audio_segment) # Passing segment directly
        
        # 2. Text-based probability from XLNet
        # (Assuming your inference logic from Phase 2)
        text_prob = self.get_xlnet_prediction(text)
        
        return text, text_prob

    def get_xlnet_prediction(self, text):
        # Your inference logic here
        return 0.85 # Placeholder for a detected fraud phrase