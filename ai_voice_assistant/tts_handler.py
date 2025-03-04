import sys
import os

# Add MeloTTS to Python path (for both runtime and IDE)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'MeloTTS'))

import torch
import numpy as np
from transformers import pipeline
from kokoro import KPipeline
from melo.api import TTS
import nltk
nltk.download('averaged_perceptron_tagger_eng')


import logging
import os
import re
import random
import time


class TTSHandler:
    def __init__(self, config):
        config = config.get('melo', {})
        self.model_id = config.get('model_id')
        self.voice = config.get('voice')
        
        # Map voice to language code for MeloTTS
        language_map = {
            "EN-US": "EN", "EN-BR": "EN", "EN_INDIA": "EN", "EN-AU": "EN", "EN-Default": "EN",
            "ES": "ES", "FR": "FR", "ZH": "ZH", "JP": "JP", "KR": "KR"
        }
        
        self.base_speech_speed = max(0.5, min(2.0, config.get('speed', 1.0)))  # Clamp between 0.5 and 2.0
        self.speed = self.base_speech_speed
        self.sample_rate = config.get('sample_rate', 24000)
        self.device = config.get('device', 'cpu')
        
        # Initialize language code for the TTS model
        language_code = language_map.get(self.voice, "EN")
        
        try:
            print(f"Initializing MeloTTS with language '{language_code}' on device '{self.device}'")
            self.model = TTS(language=language_code, device=self.device)
            print("MeloTTS model loaded successfully")
        except Exception as e:
            print(f"Error initializing MeloTTS model: {e}")
            raise
        
        # Voice characteristics
        self.speech_characteristics = {
            "expressiveness": config.get('expressiveness', 1.0),  # 0.0-2.0, how expressive the voice is
            "variability": config.get('variability', 0.2),     # 0.0-1.0, how much the speech speed varies
            "character": self.voice      # Voice character/persona
        }
        
        # Map voice names to speaker IDs - use 0 as default
        self.speaker_id_map = {
            "EN-US": 0, "EN-BR": 1, "EN_INDIA": 2, "EN-AU": 3, "EN-Default": 4,
            "ZH": 0,
            "ES": 0, "FR": 0, "JP": 0, "KR": 0
        }
        
        print(f"TTS with voice: {self.voice}")
        print(f"Base speech speed set to: {self.base_speech_speed}x")
        print(f"Sample rate set to: {self.sample_rate}")
        print(f"Speech characteristics: {self.speech_characteristics}")
        
        # Determine language code from voice prefix
        lang_code = self.voice[0]  # First letter of voice ID determines language
        # self.kokoro_pipeline = KPipeline(lang_code=lang_code)
        
    def set_characteristics(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.speech_characteristics:
                if key == "expressiveness":
                    value = max(0.0, min(2.0, value))
                elif key == "variability":
                    value = max(0.0, min(1.0, value))
                elif key == "character" and value not in self.available_voices:
                    print(f"Warning: Invalid voice '{value}'. Using '{self.voice}'.")
                    value = self.voice
                
                self.speech_characteristics[key] = value
                
                if key == "character" and value != self.voice:
                    self.voice = value
                    self.model = TTS(language=self.voice[:2], device=self.device)
        
        print(f"Updated speech characteristics: {self.speech_characteristics}")
        

        
    def synthesize(self, text, **kwargs):
        if kwargs:
            temp_characteristics = self.speech_characteristics.copy()
            self.set_characteristics(**kwargs)
        
        try:
            if not text:
                return np.zeros(0, dtype=np.float32), self.sample_rate
            
            if len(text) > 200:
                sentences = self._split_into_sentences(text)
                audio_segments = []
                silence_duration = kwargs.get('sentence_silence', 0.2)
                
                for sentence in sentences:
                    if not sentence.strip():
                        continue
                    
                    try:
                        audio_segment = self._synthesize_single(sentence)
                        if audio_segment is not None and len(audio_segment) > 0:
                            audio_segments.append(audio_segment)
                            silence = np.zeros(int(silence_duration * self.sample_rate), dtype=np.float32)
                            audio_segments.append(silence)
                    except Exception as e:
                        print(f"Error synthesizing sentence: {str(e)}")
                        continue
                
                if audio_segments:
                    combined_audio = np.concatenate(audio_segments)
                    return combined_audio, self.sample_rate
                else:
                    print("Warning: No audio segments were generated")
                    return np.zeros(0, dtype=np.float32), self.sample_rate
            else:
                audio = self._synthesize_single(text)
                return audio, self.sample_rate
        except Exception as e:
            print(f"Unexpected error in speech synthesis: {str(e)}")
            return np.zeros(0, dtype=np.float32), self.sample_rate
        finally:
            if kwargs:
                self.speech_characteristics = temp_characteristics

    
    def _synthesize_single(self, text):
        try:
            print(f"Synthesizing with speed: {self.speed:.2f}x, voice: {self.voice}")
            
            # Get numeric speaker_id from the voice name
            speaker_id = self.speaker_id_map.get(self.voice, 0)
            print(f"Using speaker_id: {speaker_id} for voice: {self.voice}")
            
            # For MeloTTS, try different parameter combinations based on what works
            try:
                # First try with speaker_id and speed
                audio = self.model.tts_to_file(text=text, speaker_id=speaker_id, speed=self.speed)
            except TypeError as e:
                print(f"First TTS attempt failed: {e}, trying alternative API")
                try:
                    # Then try with just speaker_id
                    audio = self.model.tts_to_file(text=text, speaker_id=speaker_id)
                except Exception as e2:
                    print(f"Second TTS attempt failed: {e2}, trying with minimal parameters")
                    # Finally try with just text
                    audio = self.model.tts_to_file(text=text, speaker_id=0)
            
            return audio
        except Exception as e:
            print(f"Error in MeloTTS speech synthesis: {str(e)}")
            return np.zeros(0, dtype=np.float32)
    
    def _split_into_sentences(self, text):
        """Split text into sentences for better processing.
        
        Args:
            text (str): Text to split
            
        Returns:
            list: List of sentences
        """
        # Simple sentence splitting based on common punctuation
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return sentences 