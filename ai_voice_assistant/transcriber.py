import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import os
from faster_whisper import WhisperModel
import numpy as np
import time

class Transcriber:
    def __init__(self, model_id="Systran/faster-whisper-small", **kwargs):
        self.model_id = model_id
        
        # Extract parameters from kwargs with defaults
        beam_size = kwargs.get('beam_size', 5)
        compute_type = kwargs.get('compute_type', "float16" if torch.cuda.is_available() else "int8")
        device = kwargs.get('device', "cuda" if torch.cuda.is_available() else "cpu")
        use_safetensors = kwargs.get('use_safetensors', True)
        low_cpu_mem_usage = kwargs.get('low_cpu_mem_usage', True)
        
        # Check if we're using faster-whisper or transformers
        self.use_faster_whisper = "faster-whisper" in model_id
        
        if self.use_faster_whisper:
            # Initialize faster-whisper model
            try:
                print(f"Initializing faster-whisper with model={model_id}, device={device}, compute_type={compute_type}, beam_size={beam_size}")
                
                # For the small model, we use "small" directly
                if model_id == "Systran/faster-whisper-small":
                    model_name = "small"
                elif model_id == "Systran/faster-whisper-medium":
                    model_name = "medium"
                elif model_id == "h2oai/faster-whisper-large-v3-turbo":
                    model_name = "large-v3-turbo"
                else:
                    model_name = model_id
                
                self.model = WhisperModel(
                    model_name,
                    device=device,
                    compute_type=compute_type,
                    download_root=kwargs.get('download_root', 'models/faster-whisper')
                )
                print(f"Successfully loaded model: {model_name}")
                
                # Store beam size for transcription
                self.beam_size = beam_size
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
            print(f"Initializing transformers with model={model_id}, device={self.device}")
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_id,
                torch_dtype=self.torch_dtype,
                low_cpu_mem_usage=low_cpu_mem_usage,
                use_safetensors=use_safetensors
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
                segments, _ = self.model.transcribe(audio_file, beam_size=self.beam_size)
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