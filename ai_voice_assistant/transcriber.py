import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import os

class Transcriber:
    def __init__(self, model_id="openai/whisper-large-v3-turbo"):
        """Initialize the transcriber with Whisper model.
        
        Args:
            model_id (str): Hugging Face model identifier
        """
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
                
            result = self.pipe(audio_file)
            return result["text"]
        except TypeError as e:
            if "unsupported operand type(s) for *: 'NoneType'" in str(e):
                print(f"Caught NoneType error in Whisper model. This may indicate an issue with the audio input.")
                return ""
            raise
        except Exception as e:
            print(f"Error in transcription: {str(e)}")
            return "" 