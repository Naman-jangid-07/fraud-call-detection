"""
Audio transcription service using multiple providers
"""

import whisper
import logging
from pathlib import Path
import os

logger = logging.getLogger(__name__)


class AudioTranscriber:
    """Transcribe audio files to text with multiple backend options"""
    
    def __init__(self, provider: str = "whisper", model_size: str = "base"):
        """
        Initialize audio transcriber
        
        Args:
            provider: 'whisper', 'google', or 'assembly'
            model_size: Whisper model size (tiny, base, small, medium, large)
        """
        self.provider = provider.lower()
        
        if self.provider == "whisper":
            logger.info(f"Loading Whisper {model_size} model...")
            self.model = whisper.load_model(model_size)
            logger.info("Whisper model loaded!")
        else:
            self.model = None
            logger.info(f"Using {provider} transcription service")
        
    def transcribe(self, audio_path: str) -> str:
        """
        Transcribe audio file to text
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Transcribed text
        """
        if self.provider == "whisper":
            return self._transcribe_whisper(audio_path)
        elif self.provider == "google":
            return self._transcribe_google(audio_path)
        elif self.provider == "assembly":
            return self._transcribe_assemblyai(audio_path)
        else:
            return self._transcribe_whisper(audio_path)
    
    def _transcribe_whisper(self, audio_path: str) -> str:
        """Transcribe using OpenAI Whisper"""
        try:
            logger.info(f"Transcribing with Whisper: {audio_path}...")
            
            result = self.model.transcribe(
                audio_path,
                language="en",
                verbose=False,
                fp16=False  # Use CPU-compatible mode
            )
            
            text = result["text"].strip()
            logger.info(f"Whisper transcription complete: {len(text)} characters")
            
            return text
            
        except Exception as e:
            logger.error(f"Whisper transcription error: {str(e)}")
            raise
    
    def _transcribe_google(self, audio_path: str) -> str:
        """Transcribe using Google Cloud Speech-to-Text"""
        try:
            from google.cloud import speech
            
            client = speech.SpeechClient()
            
            with open(audio_path, "rb") as audio_file:
                content = audio_file.read()
            
            audio = speech.RecognitionAudio(content=content)
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                language_code="en-US",
                enable_automatic_punctuation=True,
            )
            
            response = client.recognize(config=config, audio=audio)
            
            transcript = ""
            for result in response.results:
                transcript += result.alternatives[0].transcript + " "
            
            return transcript.strip()
            
        except Exception as e:
            logger.error(f"Google STT error: {str(e)}")
            # Fallback to Whisper
            return self._transcribe_whisper(audio_path)
    
    def _transcribe_assemblyai(self, audio_path: str) -> str:
        """Transcribe using AssemblyAI"""
        try:
           import assemblyai as aai

           aai.settings.api_key = "AIzaSyDytan6HeRy8MBJRrfN3mLFLWvl1gOYL4U"
           transcriber = aai.Transcriber()
           transcript = transcriber.transcribe(audio_path)
           print(transcript.text)
            
        except Exception as e:
            logger.error(f"AssemblyAI error: {str(e)}")
            # Fallback to Whisper
            return self._transcribe_whisper(audio_path)