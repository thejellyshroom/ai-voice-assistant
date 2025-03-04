from .audio_handler import AudioHandler
from .transcriber import Transcriber
from .llm_handler import LLMHandler
from .tts_handler import TTSHandler
import os
import torch
import threading
import time
import gc
import numpy as np
import queue


class VoiceAssistant:
    def __init__(self, 
                 # Configuration dictionaries
                 asr_config=None,
                 tts_config=None,
                 llm_config=None,
                 # Legacy parameters - will be used if configs not provided
                 # TTS parameters
                 tts_model="hexgrad/Kokoro-82M", 
                 tts_voice="af_heart", 
                 speech_speed=1.3,
                 expressiveness=1.0,
                 variability=0.3,
                 # ASR parameters 
                 transcription_model="Systran/faster-whisper-small",
                 timeout=5,
                 # LLM parameters
                 temperature=0.7,
                 top_p=0.9,
                 top_k=40,
                 creativity="high"):
        # Store configuration dictionaries
        self.asr_config = asr_config or {}
        self.tts_config = tts_config or {}
        self.llm_config = llm_config or {}
        
        # TTS configuration parameters (prioritize config dict over legacy params)
        if 'model_id' in self.tts_config:
            self.tts_model = self.tts_config['model_id']
        else:
            self.tts_model = tts_model
            
        if 'kokoro' in self.tts_config:
            kokoro_conf = self.tts_config['kokoro']
            self.tts_voice = kokoro_conf.get('voice', tts_voice)
            self.speech_speed = kokoro_conf.get('speech_speed', speech_speed)
            self.expressiveness = kokoro_conf.get('expressiveness', expressiveness)
            self.variability = kokoro_conf.get('variability', variability)
        else:
            self.tts_voice = tts_voice
            self.speech_speed = speech_speed
            self.expressiveness = expressiveness
            self.variability = variability
        
        # ASR configuration parameters
        if 'model_id' in self.asr_config:
            self.transcription_model = self.asr_config['model_id']
        else:
            self.transcription_model = transcription_model
        
        # LLM configuration parameters
        if 'local' in self.llm_config:
            local_conf = self.llm_config['local']
            self.temperature = local_conf.get('temperature', temperature)
            self.top_p = local_conf.get('top_p', top_p)
            self.top_k = local_conf.get('top_k', top_k)
            self.creativity = local_conf.get('creativity', creativity)
        else:
            self.temperature = temperature
            self.top_p = top_p
            self.top_k = top_k
            self.creativity = creativity
        
        # Components will be initialized on demand
        self.audio_handler = None
        self.transcriber = None
        self.llm_handler = None
        self.tts_handler = None
        self.tts_enabled = False
        
        # Store conversation history
        self.conversation_history = [
            {"role": "system", "content": "You are a helpful AI voice assistant."}
        ]
        
        # Initialize all components
        self.load_all_components()
        
    def load_all_components(self):
        """Load all components of the voice assistant."""
        self.load_audio_handler()
        self.load_transcriber()
        self.load_llm_handler()
        self.load_tts_handler()
        
        # Print initialization information
        print("\nAI Voice Assistant initialized!")
        print(f"Transcription model: {self.transcription_model}")
        
        # Determine device display
        if self.transcriber and hasattr(self.transcriber, 'use_faster_whisper'):
            device = "CUDA" if torch.cuda.is_available() else "CPU"
        else:
            device = getattr(self.transcriber, 'device', 'unknown')
        print(f"Device: {device}")
        
        print(f"TTS Model: {self.tts_model}")
        print(f"TTS Voice: {self.tts_voice}")
        print(f"Speech Speed: {self.speech_speed}x")
        print("Press Ctrl+C to exit")
    
    def _unload_component(self, component_name):
        """Unload a component to free up memory.
        
        Args:
            component_name (str): Name of the component to unload
        """
        component = getattr(self, component_name)
        if component:
            print(f"Unloading existing {component_name}...")
            delattr(self, component_name)
            setattr(self, component_name, None)
            gc.collect()
            print(f"{component_name} unloaded successfully.")
        else:
            print(f"No existing {component_name} found")
    
    def load_audio_handler(self):
        """Load the audio handler component."""
        self._unload_component("audio_handler")
        try:
            # Pass the ASR config to the AudioHandler for audio validation parameters
            self.audio_handler = AudioHandler(config=self.asr_config)
            print("Audio handler initialized successfully.")
        except Exception as e:
            print(f"Error initializing audio handler: {str(e)}")
    
    def load_transcriber(self):
        """Load the transcription component."""
        self._unload_component("transcriber")
        try:
            # Extract models parameters if specified in config
            params = {}
            
            # Use model_id from config or the class attribute
            model_id = self.transcription_model
            
            # Add model-specific parameters if available in config
            if self.asr_config:
                # Extract parameters for faster-whisper if it's in use
                if 'faster-whisper' in self.asr_config and 'faster-whisper' in model_id:
                    fw_config = self.asr_config['faster-whisper']
                    params.update({
                        'beam_size': fw_config.get('beam_size', 5),
                        'compute_type': fw_config.get('compute_type', 'float16'),
                        'device': fw_config.get('device', 'cpu')
                    })
                    
                # Additional params could be added here for other ASR types
            
            print(f"Initializing transcriber with model: {model_id} and params: {params}")
            self.transcriber = Transcriber(model_id=model_id, **params)
            print("Transcriber initialized successfully.")
        except Exception as e:
            print(f"Error initializing transcriber: {str(e)}")
            import traceback
            traceback.print_exc()
            self.transcriber = None  # Ensure it's None if initialization failed
    
    def load_llm_handler(self):
        """Load the LLM handler."""
        print("Loading LLM handler...")
        if self.llm_handler is None:
            # Build parameters dict for LLM
            params = {}
            
            if 'local' in self.llm_config:
                # Use values from config
                local_conf = self.llm_config['local']
                params = {
                    'temperature': local_conf.get('temperature', self.temperature),
                    'top_p': local_conf.get('top_p', self.top_p),
                    'top_k': local_conf.get('top_k', self.top_k),
                    'frequency_penalty': local_conf.get('frequency_penalty', 0.0),
                    'presence_penalty': local_conf.get('presence_penalty', 0.0)
                }
                
                # Set model name if available
                if 'model_name' in self.llm_config:
                    params['model_name'] = self.llm_config['model_name']
                
                # Add creativity preset if specified
                if 'creativity' in local_conf:
                    params['creativity'] = local_conf['creativity']
            else:
                # Use instance variables
                params = {
                    'temperature': self.temperature,
                    'top_p': self.top_p,
                    'top_k': self.top_k
                }
                
                # Add creativity preset if specified
                if self.creativity:
                    params['creativity'] = self.creativity
                
            self.llm_handler = LLMHandler(**params)
        
    def load_tts_handler(self):
        """Load the TTS handler."""
        print("Loading TTS handler...")
        self._unload_component("tts_handler")
        
        try:
            # Prepare TTS parameters from config or legacy params
            tts_params = {
                'model_id': self.tts_model,
                'voice': self.tts_voice,
                'speech_speed': self.speech_speed
            }
            
            # Extract additional parameters from config if available
            if 'kokoro' in self.tts_config:
                kokoro_conf = self.tts_config['kokoro']
                if 'sample_rate' in kokoro_conf:
                    tts_params['sample_rate'] = kokoro_conf['sample_rate']
                
            print(f"Initializing TTS with: {tts_params}")
            self.tts_handler = TTSHandler(**tts_params)
            
            # Set voice characteristics
            char_params = {
                'expressiveness': self.expressiveness,
                'variability': self.variability,
            }
            
            if 'kokoro' in self.tts_config:
                kokoro_conf = self.tts_config['kokoro']
                if 'available_voices' in kokoro_conf:
                    # Update available voices if provided in config
                    self.tts_handler.available_voices = kokoro_conf['available_voices']
            
            self.tts_handler.set_characteristics(**char_params)
            
            self.tts_enabled = True
            print("TTS handler loaded successfully.")
        except Exception as e:
            print(f"Error loading TTS handler: {e}")
            self.tts_enabled = False
        
    def listen(self, duration=None, timeout=None):
        """Record audio and transcribe it.
        
        Args:
            duration (int, optional): Fixed recording duration in seconds (legacy mode)
            timeout (int, optional): Maximum seconds to wait before giving up
            
        Returns:
            str: Transcribed text or error code
        """
        try:
            # Make sure any ongoing playback is completely finished
            if self.audio_handler:
                if self.audio_handler.is_playing:
                    print("Stopping any ongoing audio playback before listening...")
                    # Stop playback and wait for it to complete
                    self.audio_handler.stop_playback()
                    self.audio_handler.wait_for_playback_complete()
                
                print("Starting new listening session...")
                audio_file = self.audio_handler.listen_for_speech(
                    timeout=timeout,
                    stop_playback=True  # Always stop playback before listening
                )
                
            # Immediate early return for timeout errors - skip all transcription logic
            if audio_file == "low_energy":
                print("Timeout error detected: speech volume too low")
                return "low_energy"
            elif audio_file == "TIMEOUT_ERROR":
                print("Timeout error detected: no speech detected within 5 seconds")
                return "TIMEOUT_ERROR"
            
            # For other types of recording failures
            if audio_file is None:
                print("No audio detected or recording failed for reasons other than timeout")
                return ""
            
            # Additional validation for the audio file
            try:
                if not os.path.exists(audio_file):
                    print(f"Warning: Audio file {audio_file} does not exist")
                    return ""
                    
                file_size = os.path.getsize(audio_file)
                if file_size == 0:
                    print(f"Warning: Audio file {audio_file} is empty (0 bytes)")
                    return ""
                    
                # Get min_file_size from config or use default
                min_file_size = 1000
                if self.asr_config and 'audio_validation' in self.asr_config:
                    min_file_size = self.asr_config['audio_validation'].get('min_file_size', 1000)
                
                if file_size < min_file_size:  # File is suspiciously small
                    print(f"Warning: Audio file {audio_file} is suspiciously small ({file_size} bytes < {min_file_size} minimum)")
                    # Don't attempt to transcribe too small files
                    print("Audio file is too small to contain meaningful speech. Skipping transcription.")
                    return ""
            except Exception as e:
                print(f"Error validating audio file: {str(e)}")
                return ""
                
            # Try to transcribe with additional error handling
            try:
                # Check if transcriber exists, reload it if not
                if self.transcriber is None:
                    print("Transcriber not initialized. Attempting to reload...")
                    self.load_transcriber()
                    
                # Double-check transcriber exists after attempted reload
                if self.transcriber is None:
                    print("Failed to initialize transcriber. Cannot transcribe audio.")
                    return ""
                    
                transcribed_text = self.transcriber.transcribe(audio_file)
                print(f"Transcription successful: {len(transcribed_text)} characters")
                return transcribed_text
            except Exception as e:
                print(f"Error during transcription: {str(e)}")
                import traceback
                traceback.print_exc()
                return ""
        except Exception as e:
            print(f"Unexpected error in listen method: {str(e)}")
            import traceback
            traceback.print_exc()
            return ""
    
    def process_and_respond(self, text):
        """Process text with LLM and get response.
        
        Args:
            text (str): Input text to process
            
        Returns:
            str: LLM's response
        """
        # Add user message to conversation history
        self.conversation_history.append({"role": "user", "content": text})
        
        # Get response using conversation history
        response = self.llm_handler.get_streaming_response_with_context(self.conversation_history)
        
        # Add assistant response to conversation history
        self.conversation_history.append({"role": "assistant", "content": response})
        
        return response
    
    def speak(self, text):
        """Convert text to speech and play it.
        
        Args:
            text (str): Text to speak
            
        Returns:
            bool: True if speech was successfully played, False otherwise
        """
        if not self.tts_enabled:
            print("TTS is disabled. Cannot speak the response.")
            return False
            
        try:
            # Ensure text is not None
            if text is None:
                print("Warning: Received None text to speak")
                return False

            # Split text into sentences for more natural pauses
            sentences = self._split_into_sentences(text)
            
            for sentence in sentences:
                # Synthesize and play the sentence
                audio_array, sample_rate = self.tts_handler.synthesize(sentence)
                
                # Ensure audio_array is not None
                if audio_array is None:
                    print("Warning: Received None audio array from TTS handler")
                    continue
                    
                if len(audio_array) > 0:
                    self.audio_handler.play_audio(audio_array, sample_rate)
                    
                    # Small pause between sentences for more natural speech
                    time.sleep(0.3)
                else:
                    print("Generated audio is empty. Cannot play.")
            
            # Wait for all audio to finish playing
            if self.audio_handler and self.audio_handler.is_playing:
                # Let the AudioHandler calculate the appropriate timeout based on the audio duration
                self.audio_handler.wait_for_playback_complete(timeout=None)
            
            return True
            
        except Exception as e:
            print(f"Error in speech synthesis: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def speak_streaming(self, text):
        """Convert text to speech with streaming output.
        
        This method splits text into sentences and streams them as they're synthesized,
        similar to how neurosama.py handles TTS output.
        
        Args:
            text (str): Text to speak
            
        Returns:
            bool: True if speech was successfully played, False otherwise
        """
        initial_buffer = "..."  # Helps prevent first word cutoff
        sentences = self._split_into_sentences(initial_buffer + text)
    
        for i, sentence in enumerate(sentences):
            if i == 0 and sentence == "...":  # Skip our buffer
                continue

            if not self.tts_enabled:
                print("TTS is disabled. Cannot speak the response.")
                return False
                
        try:
            # Ensure text is not None
            if text is None:
                print("Warning: Received None text to speak")
                return False
            
            # Split text into sentences for more natural streaming
            sentences = self._split_into_sentences(text)
            
            for sentence in sentences:
                
                # Synthesize and play the sentence
                audio_array, sample_rate = self.tts_handler.synthesize(sentence)
                
                # Ensure audio_array is not None
                if audio_array is None:
                    print("Warning: Received None audio array from TTS handler")
                    continue
                    
                if len(audio_array) > 0:
                    self.audio_handler.play_audio(audio_array, sample_rate)
                    
                    # Small pause between sentences for more natural speech
                    time.sleep(0.3)
                else:
                    print("Generated audio is empty. Cannot play.")
            
            # Wait for all audio to finish playing
            if self.audio_handler and self.audio_handler.is_playing:
                # Let the AudioHandler calculate the appropriate timeout based on the audio duration
                self.audio_handler.wait_for_playback_complete(timeout=None)
            
            return True
            
        except Exception as e:
            print(f"Error in speech synthesis: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def _split_into_sentences(self, text):
        """Split text into sentences for more natural speech with pauses.
        
        Args:
            text (str): Text to split
            
        Returns:
            list: List of sentences
        """
        import re
        # Split on sentence endings (., !, ?) followed by space or end of string
        sentences = re.split(r'(?<=[.!?])\s+', text)
        # Remove empty sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    

    def interact_streaming(self, duration=None, timeout=10, phrase_limit=60):
        """Record audio, transcribe, process with streaming response.
        
        This method is similar to the neurosama.py approach, where the LLM response
        is streamed in chunks and TTS is generated for each chunk.
        
        Args:
            duration (int, optional): Fixed recording duration in seconds
            timeout (int, optional): Maximum seconds to wait before giving up
            phrase_limit (int, optional): Maximum seconds for a single phrase
            
        Returns:
            tuple: (transcribed_text, ai_response)
        """
        try:
            # ===== PHASE 1: PREPARATION =====
            # Ensure all audio playback is completely stopped
            if self.audio_handler:
                self.audio_handler.stop_playback()
            
            # ===== PHASE 2: LISTENING FOR USER INPUT =====
            print("\nListening for your voice...")
            transcribed_text = self.listen(duration=duration, timeout=timeout)
            
            # Handle the case where no speech was detected
            if not transcribed_text or len(transcribed_text.strip()) < 2:
                ai_response = "Seems like you didn't say anything."
                print("\nAssistant:", ai_response)
                
                # Only speak the response if TTS is enabled
                if self.tts_enabled:
                    self.speak(ai_response)
                    
                    # Wait for all audio playback to complete
                    if self.audio_handler:
                        self.audio_handler.wait_for_playback_complete()
                
                return "", ai_response
            
            if transcribed_text == "TIMEOUT_ERROR":
                ai_response = "Time out error occurred"
                print("\nAssistant:", ai_response)
                
                # Only speak the response if TTS is enabled
                if self.tts_enabled:
                    self.speak(ai_response)
            
            # ===== PHASE 3: PROCESSING USER INPUT =====    
            print("\nYou said:", transcribed_text)
            
            # Add user message to conversation history
            self.conversation_history.append({"role": "user", "content": transcribed_text})
            
            # Prepare for streaming response
            partial_buffer = ""
            char_count = 0
            waiting_for_punctuation = False
            assistant_buffer = ""  # Store the complete response
            
            print("Assistant: ", end="", flush=True)
            
            # ===== PHASE 4: GENERATING AND SPEAKING RESPONSE =====
            try:
                # Get streaming response from LLM
                for token in self.llm_handler.get_streaming_response_with_context(self.conversation_history):
                    print(token, end="", flush=True)
                    partial_buffer += token
                    assistant_buffer += token
                    char_count += len(token)
                    
                    # Once we've accumulated ~100 characters, start waiting for punctuation
                    if not waiting_for_punctuation and char_count >= 100:
                        waiting_for_punctuation = True
                    
                    if waiting_for_punctuation:
                        # If we see punctuation, treat that as a sentence boundary
                        if any(punct in token for punct in [".", "!", "?"]):
                            # Synthesize and play this sentence
                            if self.tts_enabled:
                                audio_array, sample_rate = self.tts_handler.synthesize(partial_buffer)
                                if audio_array is not None and len(audio_array) > 0:
                                    self.audio_handler.play_audio(audio_array, sample_rate)
                            
                            # Reset partial buffer
                            partial_buffer = ""
                            char_count = 0
                            waiting_for_punctuation = False
            except Exception as e:
                print(f"\nError during LLM response generation: {e}")
                
                # Add a fallback response if needed
                if not assistant_buffer:
                    assistant_buffer = "I'm sorry, I encountered an error while generating a response."
                    print("\nFallback response:", assistant_buffer)
            
            # Process any remaining text in the buffer
            if partial_buffer.strip():
                if self.tts_enabled:
                    try:
                        audio_array, sample_rate = self.tts_handler.synthesize(partial_buffer)
                        if audio_array is not None and len(audio_array) > 0:
                            self.audio_handler.play_audio(audio_array, sample_rate)
                    except Exception as e:
                        print(f"Error synthesizing final text segment: {e}")
            
            # Add the complete response to conversation history
            self.conversation_history.append({"role": "assistant", "content": assistant_buffer})
            
            # ===== PHASE 5: ENSURE ALL AUDIO COMPLETES BEFORE NEXT ITERATION =====
            # Wait for all playback to complete before starting next cycle
            if self.audio_handler and self.tts_enabled:
                # Let the AudioHandler calculate the appropriate timeout based on the audio duration
                self.audio_handler.wait_for_playback_complete(timeout=None)
            
            # Return the transcribed text and assistant response
            return transcribed_text, assistant_buffer
            
        except Exception as e:
            print(f"Unexpected error in streaming interaction: {str(e)}")
            import traceback
            traceback.print_exc()
            return "", "I'm sorry, I encountered an unexpected error."
            