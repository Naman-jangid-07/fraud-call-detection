import whisper # pip install openai-whisper

def transcribe_call(audio_path):
    # Load the base model (approx. 7x faster than the 'large' model)
    model = whisper.load_model("base")
    
    # Transcribe the audio file
    result = model.transcribe(audio_path)
    
    # Return the raw text for your XLNet fraud detector
    return result["text"]

if __name__ == "__main__":
    text = transcribe_call("data/sample_call.wav")
    print(f"Transcription: {text}")