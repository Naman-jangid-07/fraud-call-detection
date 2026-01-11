import librosa # pip install librosa
import numpy as np

def extract_vocal_signals(audio_path):
    y, sr = librosa.load(audio_path)
    
    # 1. Pitch (Heightened pitch often indicates stress)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    avg_pitch = np.mean(pitches[pitches > 0])
    
    # 2. Speech Rate (Urgency)
    # Simple estimate: zero-crossing rate helps detect high-energy speech
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    
    # 3. Spectral Centroid (Can help identify "tinny" synthetic voices)
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    
    return {
        "avg_pitch": avg_pitch,
        "urgency_score": zcr,
        "spectral_quality": spectral_centroid
    }