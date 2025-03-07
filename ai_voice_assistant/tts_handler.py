import torch
import numpy as np
from kokoro import KPipeline

# Add RealtimeTTS imports
from RealtimeTTS import TextToAudioStream, KokoroEngine

import logging
import os
import re
import random
import time
import io
import soundfile as sf


class TTSHandler:
    def __init__(self, config=None):
        config = config or {}
        kokoro_config = config.get("kokoro", {})
        
        # Ensure we have default values if config is incomplete
        self.voice = kokoro_config.get("voice") 
        self.speed = kokoro_config.get('speed')
        self.sample_rate = kokoro_config.get('sample_rate')
        self.device = kokoro_config.get('device')
        self.sentence_silence = kokoro_config.get('sentence_silence')
        
            
        self.speech_characteristics = {
            "expressiveness": kokoro_config.get('expressiveness'), 
            "variability": kokoro_config.get('variability'),  
            "character": self.voice  
        }
        
        print(f"Initializing Kokoro TTS with voice: {self.voice}")
        print(f"Base speech speed set to: {self.speed}x")
        print(f"Sample rate set to: {self.sample_rate}")
        print(f"Speech characteristics: {self.speech_characteristics}")
        
        try:
            # Initialize RealtimeTTS KokoroEngine with valid voice
            print(f"Creating KokoroEngine with voice: {self.voice}")
            self.kokoro_engine = KokoroEngine(default_voice=self.voice)
            
            # Set speed explicitly after creation
            print(f"Setting KokoroEngine speed to: {self.speed}")
            self.kokoro_engine.speed = self.speed
            
            # Verify the speed was set correctly
            print(f"KokoroEngine speed is now: {self.kokoro_engine.speed}")
            
            # Determine language code from voice prefix for the fallback pipeline
            lang_code = self.voice[0]  # First letter of voice ID determines language
            self.kokoro_pipeline = KPipeline(lang_code=lang_code)
        except Exception as e:
            print(f"Error initializing Kokoro engines: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    
    def synthesize(self, text, **kwargs):
        # Apply any temporary characteristic overrides
        temp_characteristics = None
        if kwargs:
            temp_characteristics = self.speech_characteristics.copy()
            for key, value in kwargs.items():
                if key in self.speech_characteristics:
                    self.speech_characteristics[key] = value
        
        try:
            if not text:
                return np.zeros(0, dtype=np.float32), self.sample_rate
            
            # Process short text directly with the original method
            audio = self._synthesize_single(text)
            if audio is None:
                print(f"Warning: Got None audio for text: {text}")
                return np.zeros(0, dtype=np.float32), self.sample_rate
            return audio, self.sample_rate
        except Exception as e:
            # Catch-all for any unexpected errors
            print(f"Unexpected error in speech synthesis: {str(e)}")
            import traceback
            traceback.print_exc()
            return np.zeros(0, dtype=np.float32), self.sample_rate
        finally:
            # Restore original characteristics if they were temporarily overridden
            if temp_characteristics:
                self.speech_characteristics = temp_characteristics
    
    def _synthesize_single(self, text):
        """Synthesize a single piece of text.
        
        Args:
            text (str): Text to convert to speech
            
        Returns:
            numpy.ndarray: Audio array
        """
        try:
            print(f"Synthesizing with speed: {self.speed:.2f}x, voice: {self.voice}")
            
            generator = self.kokoro_pipeline(
                text,
                voice=self.voice,
                speed=self.speed,
                split_pattern=r'\n+'
            )
            
            audio_segments = []
            for _, _, audio in generator:
                # Check if audio is None or not a tensor
                if audio is None:
                    print("Warning: Received None audio from Kokoro pipeline")
                    continue
                
                # Handle different types of audio objects
                if hasattr(audio, 'numpy'):
                    # PyTorch tensor
                    audio_segments.append(audio.numpy())
                elif isinstance(audio, np.ndarray):
                    # Already a numpy array
                    audio_segments.append(audio)
                else:
                    # Try to convert to numpy array
                    try:
                        audio_segments.append(np.array(audio, dtype=np.float32))
                    except Exception as e:
                        print(f"Error converting audio to numpy array: {str(e)}")
                        continue
            
            if audio_segments:
                combined_audio = np.concatenate(audio_segments)
                return combined_audio
            else:
                return np.zeros(0, dtype=np.float32)
        except Exception as e:
            print(f"Error in Kokoro speech synthesis: {str(e)}")
            import traceback
            traceback.print_exc()
            return np.zeros(0, dtype=np.float32)