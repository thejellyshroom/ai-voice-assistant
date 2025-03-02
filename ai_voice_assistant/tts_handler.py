import torch
import numpy as np
from transformers import pipeline
from kokoro import KPipeline

import logging
import os
import re


class TTSHandler:
    def __init__(self, model_id="hexgrad/Kokoro-82M", voice="af_nova", speech_speed=1.3):
        self.model_id = model_id
        self.voice = voice
        self.speech_speed = max(0.5, min(2.0, speech_speed))  # Clamp between 0.5 and 2.0
        
        print(f"Initializing Kokoro TTS with voice: {voice}")
        print(f"Speech speed set to: {self.speech_speed}x")
        
        # Determine language code from voice prefix
        lang_code = voice[0]  # First letter of voice ID determines language
        self.kokoro_pipeline = KPipeline(lang_code=lang_code)
        
    def synthesize(self, text):
        """Convert text to speech.
        
        Args:
            text (str): Text to convert to speech
            
        Returns:
            tuple: (audio_array, sample_rate)
        """
        try:
            if not text:
                return np.zeros(0, dtype=np.float32), 24000
            
            # For very long text, split into sentences and process separately
            if len(text) > 200:
                sentences = self._split_into_sentences(text)
                audio_segments = []
                sample_rate = 24000
                
                for sentence in sentences:
                    if not sentence.strip():
                        continue
                    
                    try:
                        # Generate speech for each sentence
                        audio_segment = self._synthesize_single(sentence)
                        
                        if len(audio_segment) > 0:
                            audio_segments.append(audio_segment)
                            # Add a small silence between sentences (0.2 seconds)
                            silence = np.zeros(int(0.2 * sample_rate), dtype=np.float32)
                            audio_segments.append(silence)
                    except Exception as e:
                        print(f"Error synthesizing sentence: {str(e)}")
                        import traceback
                        traceback.print_exc()
                        continue
                
                # Combine all audio segments
                if audio_segments:
                    try:
                        combined_audio = np.concatenate(audio_segments)
                        return combined_audio, sample_rate
                    except Exception as e:
                        print(f"Error combining audio segments: {str(e)}")
                        import traceback
                        traceback.print_exc()
                        return np.zeros(0, dtype=np.float32), sample_rate
                else:
                    return np.zeros(0, dtype=np.float32), sample_rate
            else:
                # Process short text directly
                return self._synthesize_single(text), 24000
        except Exception as e:
            # Catch-all for any unexpected errors
            print(f"Unexpected error in speech synthesis: {str(e)}")
            import traceback
            traceback.print_exc()
            return np.zeros(0, dtype=np.float32), 24000
    
    def _synthesize_single(self, text):
        """Synthesize a single piece of text.
        
        Args:
            text (str): Text to convert to speech
            
        Returns:
            numpy.ndarray: Audio array
        """
        try:
            generator = self.kokoro_pipeline(
                text,
                voice=self.voice,
                speed=self.speech_speed,
                split_pattern=r'\n+'
            )
            
            audio_segments = []
            for _, _, audio in generator:
                audio_segments.append(audio.numpy())
            
            if audio_segments:
                combined_audio = np.concatenate(audio_segments)
                return combined_audio
            else:
                return np.zeros(0, dtype=np.float32)
        except Exception as e:
            print(f"Error in Kokoro speech synthesis: {str(e)}")
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