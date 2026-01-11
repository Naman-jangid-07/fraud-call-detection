from .speech_to_text import transcribe_call
from .audio_features import extract_vocal_signals

def process_recorded_call(file_path):
    print(f"Processing: {file_path}")
    
    # Step 1: Get Transcription
    transcription = transcribe_call(file_path)
    
    # Step 2: Get Vocal Features
    vocal_metrics = extract_vocal_signals(file_path)
    
    return {
        "text": transcription,
        "vocal_features": vocal_metrics
    }