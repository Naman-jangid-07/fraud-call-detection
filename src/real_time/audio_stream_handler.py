import numpy as np

class AudioStreamHandler:
    def __init__(self, chunk_size=1024):
        self.chunk_size = chunk_size
        self.buffer = []

    def receive_chunk(self, audio_data):
        """Processes a small byte-chunk of audio from the stream."""
        # Convert bytes to numpy array for processing
        audio_chunk = np.frombuffer(audio_data, dtype=np.int16)
        self.buffer.append(audio_chunk)
        
    def get_full_segment(self):
        """Combines chunks for transcription once a window (e.g., 5 seconds) is reached."""
        if not self.buffer:
            return None
        return np.concatenate(self.buffer)