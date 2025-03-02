import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import os
from faster_whisper import WhisperModel
import numpy as np
import time

class Transcriber:
    def __init__(self, model_id="Systran/faster-whisper-small"):
        """Initialize the transcriber with Whisper model.
        
        Args:
            model_id (str): Model identifier (default: "Systran/faster-whisper-small")
        """
        self.model_id = model_id
        
        # Check if we're using faster-whisper or transformers
        self.use_faster_whisper = "Systran/faster-whisper" in model_id
        
        if self.use_faster_whisper:
            # Set compute type based on available hardware
            compute_type = "float16" if torch.cuda.is_available() else "int8"
            
            # Handle the small model specifically
            if model_id == "Systran/faster-whisper-small":
                # For the small model, we use "small" directly
                print("Using faster-whisper small model...")
                model_name = "small"
            else:
                model_name = model_id
            
            # Initialize faster-whisper model
            try:
                self.model = WhisperModel(
                    model_name,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    compute_type=compute_type
                )
                print(f"Successfully loaded model: {model_name}")
            except Exception as e:
                print(f"Error loading model: {e}")
                raise
            
            self.pipe = None  # Not used with faster-whisper
        else:
            # Set device (MPS for MacOS, CUDA for NVIDIA, or CPU)
            self.device = "mps" if torch.backends.mps.is_available() else \
                         "cuda:0" if torch.cuda.is_available() else "cpu"
            self.torch_dtype = torch.float16

            # Load model and processor
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_id,
                torch_dtype=self.torch_dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True
            )
            self.model.to(self.device)

            self.processor = AutoProcessor.from_pretrained(model_id)

            # Create pipeline
            self.pipe = pipeline(
                "automatic-speech-recognition",
                model=self.model,
                tokenizer=self.processor.tokenizer,
                feature_extractor=self.processor.feature_extractor,
                torch_dtype=self.torch_dtype,
                device=self.device,
            )

    def transcribe(self, audio_file):
        """Transcribe audio file to text.
        
        Args:
            audio_file (str): Path to audio file
            
        Returns:
            str: Transcribed text
        """
        try:
            # Validate audio file
            if not os.path.exists(audio_file) or os.path.getsize(audio_file) == 0:
                print(f"Warning: Audio file {audio_file} is empty or does not exist")
                return ""
            
            print(f"🎤 Starting transcription with model: {self.model_id.split('/')[-1]}...")
            
            start_time = time.time()
            
            if self.use_faster_whisper:
                # Use faster-whisper for transcription
                segments, _ = self.model.transcribe(audio_file, beam_size=5)
                text = " ".join([segment.text for segment in segments])
                
                end_time = time.time()
                duration = end_time - start_time
                print(f"✓ Transcription complete in {duration:.2f} seconds: \"{text[:50]}{'...' if len(text) > 50 else ''}\"")
                
                return text
            else:
                # Use transformers pipeline for transcription
                result = self.pipe(audio_file)
                
                end_time = time.time()
                duration = end_time - start_time
                text = result["text"]
                print(f"✓ Transcription complete in {duration:.2f} seconds: \"{text[:50]}{'...' if len(text) > 50 else ''}\"")
                
                return text
                
        except TypeError as e:
            if "unsupported operand type(s) for *: 'NoneType'" in str(e):
                print(f"Caught NoneType error in Whisper model. This may indicate an issue with the audio input.")
                return ""
            raise
        except Exception as e:
            print(f"Error in transcription: {str(e)}")
            return "" 