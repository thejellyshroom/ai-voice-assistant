import pyaudio
import wave
import numpy as np
import speech_recognition as sr
import threading
import queue
import time
import sounddevice as sd
import soundfile as sf
import io

class AudioHandler:
    def __init__(self, sample_rate=44100, channels=1, chunk=1024):
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk = chunk
        self.format = pyaudio.paInt16
        self.pyaudio = pyaudio.PyAudio()
        self.recognizer = sr.Recognizer()
        self.recognizer.pause_threshold = 2.0  # Seconds of silence before considering the phrase complete (increased from 1.5)
        self.recognizer.phrase_threshold = 0.3  # Minimum seconds of speaking audio before we consider the phrase started
        self.recognizer.non_speaking_duration = 0.5  # Seconds of non-speaking audio to keep on both sides of the recording
        self.recognizer.energy_threshold = 20  # Minimum audio energy to consider for recording (lowered based on observed values)
        self.recognizer.dynamic_energy_threshold = True  # Automatically adjust for ambient noise
        self.recognizer.dynamic_energy_adjustment_damping = 0.15
        self.recognizer.dynamic_energy_ratio = 1.5
        
        # Audio playback queue and thread
        self.audio_queue = queue.Queue(maxsize=100)
        self.playback_thread = None
        self.is_playing = False
        self.should_stop_playback = threading.Event()
        
    def listen_for_speech(self, filename="prompt.wav", timeout=None, stop_playback=False):
        """Record audio from microphone until silence is detected.
        
        Args:
            filename (str): Output filename (will be saved as WAV)
            timeout (int, optional): Maximum number of seconds to wait before giving up
            stop_playback (bool): Whether to stop any ongoing playback before listening
                                 Default is False to allow listening without stopping playback
            
        Returns:
            str: Path to the saved audio file
        """
        # Stop any ongoing playback if requested
        if stop_playback:
            self.stop_playback()
            print("Audio playback stopped.")
        
        print("Listening...")
        
        with sr.Microphone() as source:
            # Adjust for ambient noise with a longer duration
            self.recognizer.adjust_for_ambient_noise(source, duration=1.0)
            
            # Listen for speech
            try:
                audio_data = self.recognizer.listen(
                    source, 
                    timeout=timeout, 
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
        
        # Start the playback thread if it's not already running
        self.start_playback_thread(sample_rate)
        
        # Add audio to the queue
        self.audio_queue.put((audio_data, sample_rate))
        self.is_playing = True

    def start_playback_thread(self, sample_rate):
        """Start the background playback thread if it's not already running."""
        if self.playback_thread is None or not self.playback_thread.is_alive():
            self.should_stop_playback.clear()
            self.playback_thread = threading.Thread(
                target=self._audio_playback_thread,
                daemon=True
            )
            self.playback_thread.start()
    
    def stop_playback(self):
        """Improved playback stopping with buffer clearing"""
        if self.is_playing:
            print("Initiating graceful playback stop...")
            self.should_stop_playback.set()
            
            # Clear queue but allow current chunk to finish
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                    self.audio_queue.task_done()
                except queue.Empty:
                    break
                    
            # Add 100ms silence to prevent audio cutoff
            if self.playback_thread and self.playback_thread.is_alive():
                self.audio_queue.put((np.zeros(int(0.1*24000)), 24000))
                
            self.is_playing = False
    
    def _audio_playback_thread(self):
        """Background thread that plays audio fragments as they become available."""
        stream = None
        try:
            while not self.should_stop_playback.is_set():
                try:
                    # Get audio from queue with a timeout
                    audio_data, sample_rate = self.audio_queue.get(timeout=0.5)
                    
                    # Check if we should stop
                    if self.should_stop_playback.is_set():
                        self.audio_queue.task_done()
                        break
                    
                    # Create or recreate stream if needed
                    if stream is None:
                        stream = self.pyaudio.open(
                            format=pyaudio.paFloat32,
                            channels=1,
                            rate=sample_rate,
                            output=True
                        )
                    
                    # Play audio
                    stream.write(audio_data.tobytes())
                    self.audio_queue.task_done()
                    
                except queue.Empty:
                    # No audio in queue, continue waiting
                    continue
                except Exception as e:
                    print(f"Error in audio playback thread: {e}")
                    if self.audio_queue.unfinished_tasks > 0:
                        self.audio_queue.task_done()
        finally:
            # Clean up
            if stream is not None:
                stream.stop_stream()
                stream.close()
            self.is_playing = False
    
    def listen_while_speaking(self, timeout=10, phrase_limit=60, debounce_time=2.0, min_energy=25):
        """Listen for speech while the AI might be speaking.
        
        Args:
            timeout (int): Maximum seconds to wait before giving up
            phrase_limit (int): Maximum seconds for a single phrase
            debounce_time (float): Minimum time between interruption detections
            min_energy (int): Minimum energy level to consider as speech
            
        Returns:
            str: Path to the saved audio file or None if no speech detected
        """
        # We don't stop playback here, as we want to listen WHILE speaking
        
        print("Listening for interruption...")

        last_interrupt_time = 0
        with sr.Microphone() as source:
                # Adjust for ambient noise with a longer duration
                self.recognizer.adjust_for_ambient_noise(source, duration=1.0)
                
                try:
                    self.recognizer.pause_threshold = 0.8
                    self.recognizer.phrase_threshold = 0.3
                    self.recognizer.non_speaking_duration = 0.2
                    while True:
                        try:
                            audio_data = self.recognizer.listen(
                                source, 
                                timeout=timeout,
                                phrase_time_limit=phrase_limit,
                                snowboy_configuration=(
                                    self.recognizer.pause_threshold,
                                    self.recognizer.phrase_threshold,
                                    0.5  # Pre-roll buffer (new)
                                )
                            )

                             # New: Audio validation pipeline
                            validation_passed = True
                            audio_as_numpy = np.frombuffer(audio_data.frame_data, dtype=np.int16)

                            audio_duration = len(audio_data.frame_data) / (2 * 16000) 
                            if audio_duration < 1.0:  # Increased minimum duration
                                print(f"Rejected: Duration too short ({audio_duration:.2f}s)")
                                validation_passed = False
                            
                            # Check if the audio has enough energy to be considered speech
                            audio_as_numpy = np.frombuffer(audio_data.frame_data, dtype=np.int16)
                            
                            # Calculate RMS energy
                            audio_energy = np.sqrt(np.mean(np.square(audio_as_numpy)))
                            if audio_energy < 25:  # Very low energy is likely just background noise
                                print(f"Rejected: Low energy ({audio_energy:.2f} dB)")
                                validation_passed = False
                            
                            # Calculate zero crossing rate (helps detect speech vs noise)
                            zero_crossings = np.sum(np.abs(np.diff(np.signbit(audio_as_numpy))))/len(audio_as_numpy)
                            if zero_crossings > 0.35:  # More strict threshold
                                print(f"Rejected: High noise ({zero_crossings:.4f} crossings)")
                                validation_passed = False

                            # 4. Spectral Flatness Check (new)
                            fft = np.fft.rfft(audio_as_numpy.astype(float))
                            fft_mag = np.abs(fft)
                            spectral_flatness = np.exp(np.mean(np.log(fft_mag + 1e-10))) / np.mean(fft_mag)
                            if spectral_flatness > 0.8:
                                print(f"Rejected: Flat spectrum ({spectral_flatness:.2f})")
                                validation_passed = False

                            if not validation_passed:
                                continue
                             # Save and process valid audio
                            wav_filename = "interrupt.wav"
                            with open(wav_filename, "wb") as f:
                                f.write(audio_data.get_wav_data())
                                
                            print(f"VALID INTERRUPTION: Energy={audio_energy:.2f}dB, Duration={audio_duration:.2f}s")
                            self.stop_playback()
                            return wav_filename
                            
                        except sr.WaitTimeoutError:
                            continue
                except Exception as e:
                    print(f"Error in listen_while_speaking: {e}")
                finally:
                    # Make sure we restore the original parameters
                    self.recognizer.pause_threshold = 2.0
                    self.recognizer.phrase_threshold = 0.3

    def __del__(self):
        """Cleanup PyAudio resources."""
        self.stop_playback()
        if self.playback_thread and self.playback_thread.is_alive():
            self.playback_thread.join(timeout=1.0)
        self.pyaudio.terminate() 