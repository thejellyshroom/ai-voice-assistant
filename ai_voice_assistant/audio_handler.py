import pyaudio
import wave
import numpy as np
import speech_recognition as sr
import threading
import queue
import time

class AudioHandler:
    def __init__(self, sample_rate=44100, channels=1, chunk=1024):
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk = chunk
        self.format = pyaudio.paInt16
        self.pyaudio = pyaudio.PyAudio()
        self.recognizer = sr.Recognizer()
        self.recognizer.pause_threshold = 1.0  # Seconds of silence before considering the phrase complete
        self.recognizer.phrase_threshold = 0.3  # Minimum seconds of speaking audio before we consider the phrase started
        self.recognizer.non_speaking_duration = 0.5  # Seconds of non-speaking audio to keep on both sides of the recording

    def record_audio(self, filename="prompt.wav", duration=3):
        """Record audio from microphone for specified duration.
        
        Args:
            filename (str): Output filename (will be saved as WAV)
            duration (int): Recording duration in seconds
            
        Returns:
            str: Path to the saved audio file
        """
        stream = self.pyaudio.open(
            format=self.format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk
        )

        print("Recording...")
        frames = []

        for _ in range(0, int(self.sample_rate / self.chunk * duration)):
            data = stream.read(self.chunk)
            frames.append(data)

        print("Recording finished.")

        stream.stop_stream()
        stream.close()

        # Save the recorded data as a WAV file
        wav_filename = filename if filename.endswith('.wav') else f"{filename}.wav"
        with wave.open(wav_filename, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.pyaudio.get_sample_size(self.format))
            wf.setframerate(self.sample_rate)
            wf.writeframes(b''.join(frames))

        print(f"Audio saved as {wav_filename}")
        return wav_filename
        
    def listen_for_speech(self, filename="prompt.wav", timeout=None, phrase_time_limit=None):
        """Record audio from microphone until silence is detected.
        
        Args:
            filename (str): Output filename (will be saved as WAV)
            timeout (int, optional): Maximum number of seconds to wait before giving up
            phrase_time_limit (int, optional): Maximum number of seconds for a phrase
            
        Returns:
            str: Path to the saved audio file
        """
        print("Listening...")
        
        with sr.Microphone() as source:
            # Adjust for ambient noise
            self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
            
            # Listen for speech
            try:
                audio_data = self.recognizer.listen(
                    source, 
                    timeout=timeout, 
                    phrase_time_limit=phrase_time_limit
                )
                
                # Save the audio data to a WAV file
                wav_filename = filename if filename.endswith('.wav') else f"{filename}.wav"
                with open(wav_filename, "wb") as f:
                    f.write(audio_data.get_wav_data())
                
                print("Listening finished.")
                print(f"Audio saved as {wav_filename}")
                return wav_filename
                
            except sr.WaitTimeoutError:
                print("No speech detected within timeout period.")
                return None

    def play_audio(self, audio_data, sample_rate=22050):
        """Play audio data through speakers.
        
        Args:
            audio_data (numpy.ndarray or torch.Tensor): Audio data as numpy array or PyTorch tensor
            sample_rate (int): Sample rate of the audio data
        """
        # Convert PyTorch tensor to numpy array if needed
        if hasattr(audio_data, 'detach') and hasattr(audio_data, 'cpu') and hasattr(audio_data, 'numpy'):
            # This is likely a PyTorch tensor
            audio_data = audio_data.detach().cpu().numpy()
        
        # Ensure audio data is float32 and normalized
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        
        # Normalize if needed
        if np.max(np.abs(audio_data)) > 1.0:
            audio_data = audio_data / np.max(np.abs(audio_data))
        
        # Open stream for playback
        stream = self.pyaudio.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=sample_rate,
            output=True
        )
        
        # Play audio
        stream.write(audio_data.tobytes())
        
        # Close stream
        stream.stop_stream()
        stream.close()

    def __del__(self):
        """Cleanup PyAudio resources."""
        self.pyaudio.terminate() 