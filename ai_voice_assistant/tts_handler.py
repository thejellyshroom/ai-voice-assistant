import torch
import numpy as np
from transformers import pipeline
import logging
import os
import requests
import json
import tempfile
import soundfile as sf
import sys
import subprocess
import onnxruntime as ort
from huggingface_hub import hf_hub_download
import re



class TTSHandler:
    def __init__(self, model_id="NeuML/kokoro-int8-onnx", voice="af_nova", speech_speed=1.3, quantization="fp32"):
        """Initialize the TTS handler.
        
        Args:
            model_id (str): Model ID to use for TTS
            voice (str): Voice to use for TTS
            speech_speed (float): Speed factor for speech (0.5 to 2.0)
            quantization (str): ONNX model quantization to use (default: "fp32", options: "fp32", "fp16", "q8", "q4", "q4f16")
        """
        # Set device (MPS for MacOS, CUDA for NVIDIA, or CPU)
        self.device = "mps" if torch.backends.mps.is_available() else \
                     "cuda:0" if torch.cuda.is_available() else "cpu"
        
        print(f"TTS using device: {self.device}")
        
        self.model_id = model_id
        self.voice = voice
        self.speech_speed = max(0.5, min(2.0, speech_speed))  # Clamp between 0.5 and 2.0
        self.quantization = quantization
        print(f"Speech speed set to: {self.speech_speed}x")
        print(f"ONNX quantization set to: {self.quantization}")
        
        self.using_kokoro = False
        self.using_kokoro_onnx = False
        self.using_kokoro_pytorch = False
        self.tts_pipeline = None
        self.kokoro_pipeline = None
        
        # Check if we're using Kokoro model
        if "kokoro" in model_id.lower():
            try:
                # Try to load Kokoro ONNX model first
                self._load_kokoro_onnx()
                self.using_kokoro = True
                self.using_kokoro_onnx = True
                print(f"Using Kokoro ONNX TTS with voice: {voice}")
            except Exception as e:
                print(f"Failed to load Kokoro ONNX model: {str(e)}")
                try:
                    # Check if espeak-ng is installed (required for phoneme generation)
                    import subprocess
                    try:
                        subprocess.run(["espeak-ng", "--version"], capture_output=True, check=True)
                    except (subprocess.SubprocessError, FileNotFoundError):
                        print("Warning: espeak-ng is not installed, which is required for Kokoro TTS.")
                        print("Please install it with:")
                        print("  - Ubuntu/Debian: sudo apt-get install espeak-ng")
                        print("  - macOS: brew install espeak-ng")
                        print("  - Windows: Download from https://github.com/espeak-ng/espeak-ng/releases")
                    
                    # Try to load Kokoro PyTorch model as fallback
                    print("Attempting to load Kokoro PyTorch model as fallback...")
                    try:
                        import kokoro
                        from kokoro import KPipeline
                        
                        # Determine language code from voice prefix
                        lang_code = voice[0]  # First letter of voice ID determines language
                        self.kokoro_pipeline = KPipeline(lang_code=lang_code)
                        self.using_kokoro = True
                        self.using_kokoro_pytorch = True
                        print(f"Using Kokoro PyTorch TTS with voice: {voice}")
                    except Exception as e2:
                        print(f"Failed to load Kokoro PyTorch model: {str(e2)}")
                        # Try with transformers pipeline
                        try:
                            from transformers import pipeline
                            self.kokoro_pipeline = pipeline(
                                "text-to-speech", 
                                model=self.model_id,
                                device=self.device
                            )
                            self.using_kokoro = True
                            self.using_kokoro_pytorch = True
                            print(f"Using Kokoro PyTorch TTS with transformers pipeline")
                        except Exception as e3:
                            print(f"Failed to load Kokoro with transformers: {str(e3)}")
                            # If both Kokoro models fail, try to load a fallback model
                            self._load_fallback_model()
                except Exception as e4:
                    print(f"Failed to initialize any Kokoro model: {str(e4)}")
                    self._load_fallback_model()
        else:
            # Try to load the TTS pipeline with the specified model
            try:
                from transformers import pipeline
                self.tts_pipeline = pipeline(
                    "text-to-speech", 
                    model=model_id,
                    device=self.device
                )
                print(f"Successfully loaded TTS model: {model_id}")
            except Exception as e:
                print(f"Error loading {model_id}: {str(e)}")
                print("Falling back to a simpler TTS model...")
                
                # Try a simpler fallback model
                self._load_fallback_model()
    
    def _load_fallback_model(self):
        """Load a fallback TTS model when Kokoro is not available."""
        try:
            fallback_model = "facebook/mms-tts-eng"
            self.tts_pipeline = pipeline(
                "text-to-speech", 
                model=fallback_model,
                device=self.device
            )
            print(f"Successfully loaded fallback TTS model: {fallback_model}")
        except Exception as e:
            print(f"Error loading fallback model: {str(e)}")
            try:
                # Try another fallback model
                fallback_model = "suno/bark-small"
                self.tts_pipeline = pipeline(
                    "text-to-speech", 
                    model=fallback_model,
                    device=self.device
                )
                print(f"Successfully loaded alternative fallback TTS model: {fallback_model}")
            except Exception as e2:
                print(f"Error loading alternative fallback model: {str(e2)}")
                print("TTS functionality will be disabled.")
                self.tts_pipeline = None
    
    def _load_kokoro_onnx(self):
        """Load the Kokoro ONNX model and voice.
        
        Args:
            voice (str): Voice ID for Kokoro
        """
        # Create directories if they don't exist
        os.makedirs('onnx', exist_ok=True)
        os.makedirs('voices', exist_ok=True)
        
        # Determine model filename based on repository and quantization
        model_filename = "model.onnx"
        if self.model_id == "NeuML/kokoro-int8-onnx":
            # NeuML repo uses a specific naming convention
            model_filename = "model.onnx"  # This is already int8 quantized
        elif "onnx-community" in self.model_id:
            # onnx-community repo has different quantization options
            if self.quantization == "fp32":
                model_filename = "model.onnx"
            elif self.quantization == "fp16":
                model_filename = "model_fp16.onnx"
            elif self.quantization == "q8":
                model_filename = "model_quantized.onnx"
            elif self.quantization == "q4":
                model_filename = "model_q4.onnx"
            elif self.quantization == "q4f16":
                model_filename = "model_q4f16.onnx"
        
        # Check if we already have the model locally
        local_model_path = os.path.join('onnx', model_filename)
        
        if not os.path.exists(local_model_path):
            print(f"ONNX model {model_filename} not found locally. Attempting to download...")
            try:
                # Try to download from Hugging Face
                try:
                    model_path = hf_hub_download(repo_id=self.model_id, filename=model_filename)
                    import shutil
                    shutil.copy(model_path, local_model_path)
                    print(f"Successfully downloaded {model_filename} from {self.model_id}")
                except Exception as e:
                    print(f"Error downloading from {self.model_id}: {str(e)}")
                    
                    # Try alternative repositories
                    alt_repos = ["NeuML/kokoro-int8-onnx", "hexgrad/Kokoro-82M", "onnx-community/Kokoro-82M-v1.0-ONNX"]
                    for repo in alt_repos:
                        if repo != self.model_id:  # Skip the one we already tried
                            try:
                                print(f"Trying alternative repository: {repo}")
                                model_path = hf_hub_download(repo_id=repo, filename=model_filename)
                                import shutil
                                shutil.copy(model_path, local_model_path)
                                print(f"Successfully downloaded {model_filename} from {repo}")
                                break
                            except Exception as e2:
                                print(f"Error downloading from {repo}: {str(e2)}")
                    
                    # If we still don't have the model, try direct URL as last resort
                    if not os.path.exists(local_model_path):
                        raise RuntimeError("Failed to download model from any repository")
            except Exception as e:
                print(f"Error downloading ONNX model: {str(e)}")
                print("Attempting to download from alternative sources...")
                
                # Try alternative sources or create a dummy model
                try:
                    # Try to download from GitHub releases or other sources
                    import urllib.request
                    model_url = "https://github.com/hexgrad/Kokoro/releases/download/v0.1.0/model.onnx"
                    print(f"Downloading from {model_url}...")
                    urllib.request.urlretrieve(model_url, local_model_path)
                except Exception as e2:
                    print(f"Error downloading from alternative source: {str(e2)}")
                    print("Using PyTorch model and converting to ONNX...")
                    
                    # As a last resort, try to download the PyTorch model and convert to ONNX
                    try:
                        # This is a simplified version - in a real implementation, you would
                        # download the PyTorch model and convert it to ONNX
                        print("Unable to obtain ONNX model. Falling back to non-ONNX implementation.")
                        raise RuntimeError("ONNX model not available")
                    except:
                        raise RuntimeError("Failed to obtain ONNX model from any source")
        
        # Set up ONNX runtime session
        providers = []
        if self.device == "cuda:0" and "CUDAExecutionProvider" in ort.get_available_providers():
            providers.append("CUDAExecutionProvider")
        elif self.device == "mps":
            # MPS is not directly supported by ONNX Runtime, fallback to CPU
            providers.append("CPUExecutionProvider")
        else:
            providers.append("CPUExecutionProvider")
            
        self.onnx_session = ort.InferenceSession(local_model_path, providers=providers)
        
        # Get the actual input names from the model
        self.onnx_input_names = [input.name for input in self.onnx_session.get_inputs()]
        print(f"ONNX model input names: {self.onnx_input_names}")
        
        # Store the input names for later use
        self.tokens_input_name = None
        self.style_input_name = None
        self.speed_input_name = None
        
        # Map common input name patterns
        token_names = ['tokens', 'input_ids', 'inputs']
        style_names = ['style', 'style_vector', 'speaker_embedding']
        speed_names = ['speed', 'speed_factor', 'speaking_rate']
        
        # Find the actual input names used by this model
        for name in self.onnx_input_names:
            if name in token_names:
                self.tokens_input_name = name
            elif name in style_names:
                self.style_input_name = name
            elif name in speed_names:
                self.speed_input_name = name
        
        print(f"Using input names: tokens='{self.tokens_input_name}', style='{self.style_input_name}', speed='{self.speed_input_name}'")
        
        # Inspect the model structure
        self._inspect_onnx_model()
        
        # Download the voice file
        try:
            # First try to download the .bin file for the voice
            voice_code = self.voice.split('_')[0]  # Extract the voice code (e.g., 'af' from 'af_sky')
            voice_bin_path = os.path.join('voices', f'{voice_code}.bin')
            
            if not os.path.exists(voice_bin_path):
                try:
                    # Try to download from Hugging Face
                    hf_voice_path = hf_hub_download(repo_id="onnx-community/Kokoro-82M-v1.0-ONNX", filename=f"voices/{voice_code}.bin")
                    import shutil
                    shutil.copy(hf_voice_path, voice_bin_path)
                except Exception as e:
                    print(f"Error downloading voice file: {str(e)}")
                    print("Creating a dummy voice file for testing...")
                    # Create a dummy voice file with random values
                    dummy_voice = np.random.randn(512, 1, 256).astype(np.float32)
                    dummy_voice.tofile(voice_bin_path)
            
            # Load the voice style vectors
            self.voice_style = np.fromfile(voice_bin_path, dtype=np.float32).reshape(-1, 1, 256)
            print(f"Loaded voice style vectors: {voice_code}.bin with shape {self.voice_style.shape}")
            
            # Also download the .pt file for compatibility
            try:
                voice_path = hf_hub_download(repo_id="onnx-community/Kokoro-82M-v1.0-ONNX", filename=f"{self.voice}.pt")
                self.voice_data = torch.load(voice_path, map_location=torch.device(self.device))
                print(f"Loaded voice file: {self.voice}.pt")
            except Exception as e:
                print(f"Error downloading voice PT file: {str(e)}")
                # Create dummy voice data
                self.voice_data = torch.randn(256, device=self.device)
                print("Created dummy voice data for testing")
        except Exception as e:
            print(f"Error loading voice files: {str(e)}")
            # Try to find an available voice
            available_voices = ["af_sky", "af_bella", "af_sarah", "af_nicole", "am_adam", "am_michael", 
                               "bf_emma", "bf_isabella", "bm_george", "bm_lewis"]
            for v in available_voices:
                try:
                    voice_path = hf_hub_download(repo_id="onnx-community/Kokoro-82M-v1.0-ONNX", filename=f"{v}.pt")
                    self.voice_data = torch.load(voice_path, map_location=torch.device(self.device))
                    
                    # Also try to get the bin file
                    voice_code = v.split('_')[0]
                    voice_bin_path = os.path.join('voices', f'{voice_code}.bin')
                    if not os.path.exists(voice_bin_path):
                        try:
                            hf_voice_path = hf_hub_download(repo_id="hexgrad/Kokoro-82M", filename=f"voices/{voice_code}.bin")
                            import shutil
                            shutil.copy(hf_voice_path, voice_bin_path)
                        except:
                            # Create a dummy voice file
                            dummy_voice = np.random.randn(512, 1, 256).astype(np.float32)
                            dummy_voice.tofile(voice_bin_path)
                    
                    self.voice_style = np.fromfile(voice_bin_path, dtype=np.float32).reshape(-1, 1, 256)
                    print(f"Falling back to voice: {v}")
                    self.voice = v
                    break
                except:
                    continue
        
        # Load phoneme mapping
        try:
            # Download the config.json file which contains the phoneme mapping
            config_path = hf_hub_download(repo_id="hexgrad/Kokoro-82M", filename="config.json")
            with open(config_path, 'r') as f:
                import json
                config = json.load(f)
                self.phoneme_to_id = config.get('phoneme_to_id', {})
                print(f"Loaded phoneme mapping with {len(self.phoneme_to_id)} entries")
        except Exception as e:
            print(f"Error loading phoneme mapping: {str(e)}")
            # Create a basic phoneme mapping as fallback
            self.phoneme_to_id = {}
            for i in range(256):
                self.phoneme_to_id[chr(i)] = i
        
        # Load phonemizer
        try:
            import espeak_phonemizer
            try:
                # Use the correct Phonemizer class
                self.phonemizer = espeak_phonemizer.Phonemizer(language='en-us')
            except (AttributeError, Exception) as e:
                # Handle the case where the module doesn't have EspeakPhonemizer attribute
                print(f"Warning: Error initializing espeak_phonemizer.Phonemizer: {str(e)}")
                print("This might be due to a version mismatch or installation issue.")
                
                # Try to use the phonemize function directly if available
                if hasattr(espeak_phonemizer, 'phonemize'):
                    self.phonemizer = espeak_phonemizer
                    # Monkey patch to make the interface consistent
                    self.phonemizer.phonemize = lambda self, text: espeak_phonemizer.phonemize(text, language='en-us')
                else:
                    # Create a dummy phonemizer that just returns the input text
                    print("Creating a dummy phonemizer that returns the input text.")
                    self.phonemizer = type('DummyPhonemizer', (), {'phonemize': lambda self, text: text})()
        except ImportError:
            print("Installing espeak_phonemizer...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "espeak-phonemizer"])
                import espeak_phonemizer
                try:
                    # Use the correct Phonemizer class
                    self.phonemizer = espeak_phonemizer.Phonemizer(language='en-us')
                except (AttributeError, Exception) as e:
                    # Create a dummy phonemizer if Phonemizer is not available
                    print(f"Warning: Error initializing espeak_phonemizer.Phonemizer: {str(e)}")
                    print("Creating a dummy phonemizer that returns the input text.")
                    self.phonemizer = type('DummyPhonemizer', (), {'phonemize': lambda self, text: text})()
            except Exception as e:
                print(f"Error installing espeak_phonemizer: {str(e)}")
                print("Creating a dummy phonemizer that returns the input text.")
                self.phonemizer = type('DummyPhonemizer', (), {'phonemize': lambda self, text: text})()
    
    def _phonemize_text(self, text):
        """Convert text to phonemes using espeak-ng.
        
        Args:
            text (str): Text to convert to phonemes
            
        Returns:
            str: Phonemes
        """
        # Clean the text
        text = text.strip()
        if not text:
            return ""
        
        # Convert to phonemes
        phonemes = self.phonemizer.phonemize(text)
        return phonemes
    
    def synthesize_with_kokoro_onnx(self, text):
        """Synthesize speech using the Kokoro ONNX model.
        
        Args:
            text (str): Text to convert to speech
            
        Returns:
            tuple: (audio_array, sample_rate)
        """
        try:
            # Convert text to phonemes
            phonemes = self._phonemize_text(text)
            if not phonemes:
                return np.zeros(0, dtype=np.float32), 24000
            
            # Convert phonemes to token IDs
            token_ids = []
            for phoneme in phonemes:
                # Get the ID from the mapping, or use a default value
                phoneme_id = self.phoneme_to_id.get(phoneme, ord(phoneme) % 256)
                token_ids.append(phoneme_id)
            
            # Ensure token length is within limits (512 - 2 for padding)
            if len(token_ids) > 510:
                print(f"Warning: Text too long ({len(token_ids)} tokens), truncating to 510 tokens")
                token_ids = token_ids[:510]
            
            # Get the style vector based on token length
            style_vector = self.voice_style[len(token_ids)]
            
            # Add padding tokens (0 at start and end)
            padded_tokens = [[0] + token_ids + [0]]
            
            # Convert to numpy arrays
            tokens_np = np.array(padded_tokens, dtype=np.int64)
            style_np = style_vector.reshape(1, 1, 256)  # Reshape to match expected input
            speed_np = np.array([self.speech_speed], dtype=np.float32)
            
            # Use the stored input names if available
            if hasattr(self, 'tokens_input_name') and self.tokens_input_name and \
               hasattr(self, 'style_input_name') and self.style_input_name and \
               hasattr(self, 'speed_input_name') and self.speed_input_name:
                
                inputs = {
                    self.tokens_input_name: tokens_np,
                    self.style_input_name: style_np,
                    self.speed_input_name: speed_np
                }
                print(f"Using stored input names: {list(inputs.keys())}")
            else:
                # Default to the standard names
                inputs = {
                    'tokens': tokens_np,
                    'style': style_np,
                    'speed': speed_np
                }
                print(f"Using default input names: {list(inputs.keys())}")
            
            # Run inference
            try:
                outputs = self.onnx_session.run(None, inputs)
            except Exception as input_error:
                print(f"Error with input names: {str(input_error)}")
                
                # If the error is about input names, try with the exact names from the error message
                if "Required inputs" in str(input_error):
                    try:
                        # Extract the required input names from the error message
                        error_msg = str(input_error)
                        required_inputs_str = error_msg.split("Required inputs (")[1].split(")")[0]
                        required_inputs = eval(required_inputs_str)
                        
                        print(f"Trying with exact input names from error message: {required_inputs}")
                        
                        # Create inputs with the exact required names
                        exact_inputs = {}
                        for i, name in enumerate(required_inputs):
                            if i == 0:  # First input is tokens
                                exact_inputs[name] = tokens_np
                            elif i == 1:  # Second input is style
                                exact_inputs[name] = style_np
                            elif i == 2:  # Third input is speed
                                exact_inputs[name] = speed_np
                        
                        # Run inference with the exact input names
                        outputs = self.onnx_session.run(None, exact_inputs)
                        
                        # Store these names for future use
                        if len(required_inputs) >= 3:
                            self.tokens_input_name = required_inputs[0]
                            self.style_input_name = required_inputs[1]
                            self.speed_input_name = required_inputs[2]
                            print(f"Updated input names: tokens='{self.tokens_input_name}', style='{self.style_input_name}', speed='{self.speed_input_name}'")
                    except Exception as e2:
                        print(f"Error trying with exact input names: {str(e2)}")
                        raise
                else:
                    raise
            
            # Process output
            audio = outputs[0].squeeze()
            
            # Convert to float32 and normalize
            audio = audio.astype(np.float32)
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio))
            
            return audio, 24000
        except Exception as e:
            print(f"Error in Kokoro ONNX speech synthesis: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Fall back to PyTorch synthesis if available
            if hasattr(self, 'using_kokoro_pytorch') and self.using_kokoro_pytorch:
                print("Falling back to PyTorch synthesis...")
                return self.synthesize_with_kokoro_pytorch(text)
            
            return np.zeros(0, dtype=np.float32), 24000
    
    def synthesize_with_kokoro_pytorch(self, text):
        """Synthesize speech using the Kokoro PyTorch model.
        
        Args:
            text (str): Text to convert to speech
            
        Returns:
            tuple: (audio_array, sample_rate)
        """
        try:
            # Generate speech using Kokoro pipeline
            generator = self.kokoro_pipeline(
                text, 
                voice=self.voice,
                speed=self.speech_speed,  # Apply speech speed
                split_pattern=r'\n+'
            )
            
            # Collect all audio segments
            audio_segments = []
            for _, _, audio in generator:
                # Convert PyTorch tensor to numpy array if needed
                if hasattr(audio, 'detach') and hasattr(audio, 'cpu') and hasattr(audio, 'numpy'):
                    # This is likely a PyTorch tensor
                    audio = audio.detach().cpu().numpy()
                audio_segments.append(audio)
            
            # If we have multiple segments, combine them
            if len(audio_segments) > 1:
                # Combine all segments
                combined_audio = np.concatenate(audio_segments)
                return combined_audio, 24000
            elif len(audio_segments) == 1:
                # Return the single audio segment
                return audio_segments[0], 24000
            else:
                return np.zeros(0, dtype=np.float32), 24000
        except Exception as e:
            print(f"Error in Kokoro PyTorch speech synthesis: {str(e)}")
            import traceback
            traceback.print_exc()
            return np.zeros(0, dtype=np.float32), 24000
    
    def synthesize(self, text):
        """Convert text to speech.
        
        Args:
            text (str): Text to convert to speech
            
        Returns:
            tuple: (audio_array, sample_rate)
        """
        try:
            if not text:
                return np.zeros(0, dtype=np.float32), 24000 if self.using_kokoro else 22050
            
            # Use Kokoro model if available
            if self.using_kokoro:
                # For very long text, split into sentences and process separately
                if len(text) > 200:
                    sentences = self._split_into_sentences(text)
                    audio_segments = []
                    sample_rate = 24000
                    
                    for sentence in sentences:
                        if not sentence.strip():
                            continue
                        
                        try:
                            # Generate speech for each sentence using the appropriate method
                            if hasattr(self, 'using_kokoro_onnx') and self.using_kokoro_onnx:
                                audio, _ = self.synthesize_with_kokoro_onnx(sentence)
                            elif hasattr(self, 'using_kokoro_pytorch') and self.using_kokoro_pytorch:
                                audio, _ = self.synthesize_with_kokoro_pytorch(sentence)
                            else:
                                # Fallback to ONNX method for backward compatibility
                                audio, _ = self.synthesize_with_kokoro_onnx(sentence)
                            
                            if len(audio) > 0:
                                audio_segments.append(audio)
                                # Add a small silence between sentences (0.2 seconds)
                                silence = np.zeros(int(0.2 * sample_rate), dtype=np.float32)
                                audio_segments.append(silence)
                        except Exception as e:
                            print(f"Error synthesizing sentence with Kokoro: {str(e)}")
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
                    # Process short text directly using the appropriate method
                    try:
                        if hasattr(self, 'using_kokoro_onnx') and self.using_kokoro_onnx:
                            return self.synthesize_with_kokoro_onnx(text)
                        elif hasattr(self, 'using_kokoro_pytorch') and self.using_kokoro_pytorch:
                            return self.synthesize_with_kokoro_pytorch(text)
                        else:
                            # Fallback to ONNX method for backward compatibility
                            return self.synthesize_with_kokoro_onnx(text)
                    except Exception as e:
                        print(f"Error in direct synthesis with Kokoro: {str(e)}")
                        import traceback
                        traceback.print_exc()
                        return np.zeros(0, dtype=np.float32), 24000
            
            # Fall back to pipeline if Kokoro is not available
            if self.tts_pipeline is None:
                return np.zeros(0, dtype=np.float32), 22050
            
            # Use the transformers pipeline
            try:
                result = self.tts_pipeline(text)
                if isinstance(result, dict):
                    return result["audio"], result["sampling_rate"]
                else:
                    return result[0]["audio"], result[0]["sampling_rate"]
            except Exception as e:
                print(f"Error in transformers pipeline synthesis: {str(e)}")
                import traceback
                traceback.print_exc()
                return np.zeros(0, dtype=np.float32), 22050
        except Exception as e:
            # Catch-all for any unexpected errors
            print(f"Unexpected error in speech synthesis: {str(e)}")
            import traceback
            traceback.print_exc()
            return np.zeros(0, dtype=np.float32), 24000
    
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