import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

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
        result = self.pipe(audio_file)
        return result["text"] 