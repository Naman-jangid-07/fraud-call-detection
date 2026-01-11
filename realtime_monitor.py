"""
Real-time call monitoring system
"""

import pyaudio
import wave
import threading
import queue
import time
from pathlib import Path
import requests
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RealTimeMonitor:
    """Monitor and analyze calls in real-time"""
    
    def __init__(self, api_url="http://localhost:8000", chunk_duration=10):
        """
        Initialize real-time monitor
        
        Args:
            api_url: Backend API URL
            chunk_duration: Duration of audio chunks to analyze (seconds)
        """
        self.api_url = api_url
        self.chunk_duration = chunk_duration
        self.is_monitoring = False
        self.audio_queue = queue.Queue()
        
        # Audio parameters
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000
        
        self.audio = pyaudio.PyAudio()
        
    def start_monitoring(self):
        """Start monitoring audio"""
        self.is_monitoring = True
        
        # Start recording thread
        record_thread = threading.Thread(target=self._record_audio)
        record_thread.start()
        
        # Start analysis thread
        analysis_thread = threading.Thread(target=self._analyze_audio)
        analysis_thread.start()
        
        logger.info("Real-time monitoring started!")
        
    def stop_monitoring(self):
        """Stop monitoring"""
        self.is_monitoring = False
        logger.info("Real-time monitoring stopped!")
        
    def _record_audio(self):
        """Record audio from microphone"""
        stream = self.audio.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK
        )
        
        frames = []
        frame_count = 0
        frames_per_chunk = int(self.RATE / self.CHUNK * self.chunk_duration)
        
        while self.is_monitoring:
            data = stream.read(self.CHUNK)
            frames.append(data)
            frame_count += 1
            
            # Save chunk when duration reached
            if frame_count >= frames_per_chunk:
                self.audio_queue.put(frames.copy())
                frames = []
                frame_count = 0
        
        stream.stop_stream()
        stream.close()
        
    def _analyze_audio(self):
        """Analyze audio chunks"""
        chunk_number = 0
        
        while self.is_monitoring:
            if not self.audio_queue.empty():
                frames = self.audio_queue.get()
                chunk_number += 1
                
                # Save temporary audio file
                temp_file = f"temp_chunk_{chunk_number}.wav"
                self._save_wav(temp_file, frames)
                
                # Analyze with API
                try:
                    with open(temp_file, 'rb') as f:
                        files = {'file': (temp_file, f, 'audio/wav')}
                        response = requests.post(
                            f"{self.api_url}/api/analyze/audio",
                            files=files,
                            timeout=30
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            self._handle_result(result, chunk_number)
                        
                except Exception as e:
                    logger.error(f"Analysis error: {e}")
                
                # Clean up temp file
                Path(temp_file).unlink(missing_ok=True)
            
            time.sleep(0.1)
    
    def _save_wav(self, filename, frames):
        """Save audio frames to WAV file"""
        wf = wave.open(filename, 'wb')
        wf.setnchannels(self.CHANNELS)
        wf.setsampwidth(self.audio.get_sample_size(self.FORMAT))
        wf.setframerate(self.RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
    
    def _handle_result(self, result, chunk_number):
        """Handle analysis result"""
        analysis = result.get('analysis', {})
        fraud_prob = analysis.get('fraud_probability', 0)
        
        logger.info(f"Chunk {chunk_number}: Fraud Probability = {fraud_prob:.2%}")
        
        # Alert if high risk
        if fraud_prob > 0.7:
            logger.warning(f"🚨 HIGH RISK DETECTED in chunk {chunk_number}!")
            logger.warning(f"Transcription: {result.get('transcription', '')}")
            # Here you can trigger alerts, notifications, etc.


# CLI for testing
if __name__ == "__main__":
    print("="*60)
    print("🎙️ Real-Time Call Monitoring System")
    print("="*60)
    print("\nPress Ctrl+C to stop monitoring\n")
    
    monitor = RealTimeMonitor()
    
    try:
        monitor.start_monitoring()
        
        # Keep running until interrupted
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n\nStopping monitor...")
        monitor.stop_monitoring()
        print("✅ Monitoring stopped!")