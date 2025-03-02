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
    def __init__(self, tts_model="hexgrad/Kokoro-82M", tts_voice="af_heart", speech_speed=1.3, transcription_model="h2oai/faster-whisper-large-v3-turbo"):
        """Initialize the voice assistant with all components.
        
        Args:
            tts_model (str): The TTS model to use (default: "hexgrad/Kokoro-82M")
            tts_voice (str): The voice to use for Kokoro TTS (default: "af_heart" - American female Nova)
            speech_speed (float): Speed factor for speech (default: 1.3, range: 0.5-2.0)
            transcription_model (str): The transcription model to use (default: "h2oai/faster-whisper-large-v3-turbo")
        """
        # Store configuration parameters
        self.tts_model = tts_model
        self.tts_voice = tts_voice
        self.speech_speed = speech_speed
        self.transcription_model = transcription_model
        
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
        
        # Interruption handling
        self.listening_thread = None
        self.interrupted = False
        self.allow_interruptions = True
        
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
        if self.transcriber.use_faster_whisper:
            device = "CUDA" if torch.cuda.is_available() else "CPU"
        else:
            device = self.transcriber.device
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
            self.audio_handler = AudioHandler()
            print("Audio handler initialized successfully.")
        except Exception as e:
            print(f"Error initializing audio handler: {str(e)}")
    
    def load_transcriber(self):
        """Load the transcription component."""
        self._unload_component("transcriber")
        try:
            self.transcriber = Transcriber(model_id=self.transcription_model)
            print("Transcriber initialized successfully.")
        except Exception as e:
            print(f"Error initializing transcriber: {str(e)}")
    
    def load_llm_handler(self):
        """Load the LLM component."""
        self._unload_component("llm_handler")
        try:
            self.llm_handler = LLMHandler()
            print("LLM handler initialized successfully.")
        except Exception as e:
            print(f"Error initializing LLM handler: {str(e)}")
    
    def load_tts_handler(self):
        """Load the TTS component."""
        self._unload_component("tts_handler")
        try:
            self.tts_handler = TTSHandler(
                model_id=self.tts_model, 
                voice=self.tts_voice, 
                speech_speed=self.speech_speed,
            )
            self.tts_enabled = True
            print("TTS handler initialized successfully.")
        except Exception as e:
            print(f"Error initializing TTS: {str(e)}")
            print("Voice output will be disabled.")
            self.tts_enabled = False
        
    def listen(self, duration=None, timeout=None):
        """Record audio and transcribe it.
        
        Args:
            duration (int, optional): Fixed recording duration in seconds (legacy mode)
            timeout (int, optional): Maximum seconds to wait before giving up
            
        Returns:
            str: Transcribed text
        """
        if duration is not None:
            # Legacy fixed-duration recording
            audio_file = self.audio_handler.record_audio(duration=duration)
        else:
            # Dynamic listening that stops when silence is detected
            audio_file = self.audio_handler.listen_for_speech(
                timeout=timeout,
                stop_playback=False  # Changed from default to explicitly set to False
            )
            
        if audio_file is None:
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
                
            if file_size < 1000:  # Less than 1KB is suspiciously small
                print(f"Warning: Audio file {audio_file} is suspiciously small ({file_size} bytes)")
                # We'll still try to transcribe it, but log the warning
        except Exception as e:
            print(f"Error validating audio file: {str(e)}")
            return ""
            
        # Try to transcribe with additional error handling
        try:
            return self.transcriber.transcribe(audio_file)
        except Exception as e:
            print(f"Error during transcription: {str(e)}")
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
        response = self.llm_handler.get_response_with_context(self.conversation_history)
        
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
            
            # Start listening for interruptions if enabled
            if self.allow_interruptions:
                self.start_listening_for_interruptions()
            
            # Split text into sentences for more natural pauses
            sentences = self._split_into_sentences(text)
            
            for sentence in sentences:
                # Check if we've been interrupted
                if self.interrupted:
                    print("Speech interrupted by user.")
                    self.interrupted = False
                    return False
                
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
            
            # Stop listening for interruptions
            self.stop_listening_for_interruptions()
            return True
            
        except Exception as e:
            print(f"Error in speech synthesis: {str(e)}")
            import traceback
            traceback.print_exc()
            self.stop_listening_for_interruptions()
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
                
            if self.interrupted:
                # Finish current sentence before stopping
                audio_array, sr = self.tts_handler.synthesize(sentence)
                self.audio_handler.play_audio(audio_array, sr)
                break
            if not self.tts_enabled:
                print("TTS is disabled. Cannot speak the response.")
                return False
                
        try:
            # Ensure text is not None
            if text is None:
                print("Warning: Received None text to speak")
                return False
            
            # Start listening for interruptions if enabled
            if self.allow_interruptions:
                self.start_listening_for_interruptions()
            
            # Split text into sentences for more natural streaming
            sentences = self._split_into_sentences(text)
            
            for sentence in sentences:
                # Check if we've been interrupted
                if self.interrupted:
                    print("Speech interrupted by user.")
                    self.interrupted = False
                    return False
                
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
            
            # Stop listening for interruptions
            self.stop_listening_for_interruptions()
            return True
            
        except Exception as e:
            print(f"Error in speech synthesis: {str(e)}")
            import traceback
            traceback.print_exc()
            self.stop_listening_for_interruptions()
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
    
    def start_listening_for_interruptions(self):
        """Start a background thread to listen for user interruptions."""
        self.interrupted = False
        if self.listening_thread is None or not self.listening_thread.is_alive():
            self.listening_thread = threading.Thread(
                target=self._listen_for_interruptions_thread,
                daemon=True
            )
            self.listening_thread.start()
    
    def stop_listening_for_interruptions(self):
        """Stop the background thread listening for interruptions."""
        if self.listening_thread and self.listening_thread.is_alive():
            # The thread will terminate on its own due to timeout
            self.listening_thread = None
    
    def _listen_for_interruptions_thread(self):
        """Modified to handle response cutoff"""
        try:
            audio_file = self.audio_handler.listen_while_speaking(timeout=5, phrase_limit=10)
            if audio_file:
                # New: Add 0.5s delay before processing to capture full audio
                time.sleep(0.5)
                
                interruption_text = self.transcriber.transcribe(audio_file)
                if len(interruption_text.strip()) > 3:
                    print(f"\n🚨 INTERRUPTION DETECTED: {interruption_text}")
                    
                    # New: Clear any queued audio but let current sentence finish
                    self.audio_handler.audio_queue.queue.clear()
                    
                    # Add to conversation history immediately
                    self.conversation_history.append({"role": "user", "content": interruption_text})
                    
                    # New: Wait for current audio chunk to finish
                    while self.audio_handler.is_playing:
                        time.sleep(0.1)
                    
                    self.interrupted = True
        except Exception as e:
            print(f"Interruption thread error: {e}")


                
                
        except Exception as e:
            print(f"Error in interruption listening thread: {e}")
    
    def interact(self, duration=None, timeout=10):
        """Record audio, transcribe, process, and respond.
        
        Args:
            duration (int, optional): Fixed recording duration in seconds
            timeout (int, optional): Maximum seconds to wait before giving up
            
        Returns:
            tuple: (transcribed_text, ai_response)
        """
        try:
            print("\nListening for your voice...")
            # Stop any ongoing playback before listening for a new command
            if self.audio_handler.is_playing:
                self.audio_handler.stop_playback()
                print("Audio playback stopped for new interaction.")
                
            transcribed_text = self.listen(duration=duration, timeout=timeout)
            
            if not transcribed_text:
                ai_response = "I didn't hear anything. Could you please speak again?"
                # Try to speak the response
                try:
                    speech_success = self.speak(ai_response)
                    if not speech_success:
                        print("Note: The response was not spoken aloud due to TTS issues.")
                except Exception as e:
                    print(f"Error speaking response: {str(e)}")
                return "", ai_response
                
            print("\nYou said:", transcribed_text)
            
            # Process and get response
            try:
                ai_response = self.process_and_respond(transcribed_text)
            except Exception as e:
                print(f"Error processing response: {str(e)}")
                ai_response = "I'm sorry, I encountered an error while processing your request."
            
            print("Assistant:", ai_response)
            
            # Speak the response
            try:
                speech_success = self.speak(ai_response)
                if not speech_success and self.interrupted:
                    # If we were interrupted, handle the interruption
                    if len(self.conversation_history) >= 2 and self.conversation_history[-1]["role"] == "user":
                        # Process the interruption that was added to conversation history
                        interruption_text = self.conversation_history[-1]["content"]
                        print("\nProcessing interruption...")
                        
                        # Get response to the interruption
                        interrupt_response = self.process_and_respond(interruption_text)
                        print("Assistant:", interrupt_response)
                        
                        # Speak the response to the interruption
                        self.speak(interrupt_response)
                        
                        # Return the interruption and response
                        return interruption_text, interrupt_response
                elif not speech_success:
                    print("Note: The response was not spoken aloud due to TTS issues.")
            except Exception as e:
                print(f"Error speaking response: {str(e)}")
            
            return transcribed_text, ai_response
        except Exception as e:
            print(f"Unexpected error in interaction: {str(e)}")
            import traceback
            traceback.print_exc()
            return "", "I'm sorry, I encountered an unexpected error."
    
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
            print("\nListening for your voice...")
            # Continue streaming output until the user speaks again
                
            transcribed_text = self.listen(duration=duration, timeout=timeout)
            
            if not transcribed_text:
                ai_response = "I didn't hear anything. Could you please speak again?"
                self.speak(ai_response)
                return "", ai_response
                
            print("\nYou said:", transcribed_text)
            
            # Add user message to conversation history
            self.conversation_history.append({"role": "user", "content": transcribed_text})
            
            # We'll accumulate partial tokens in a buffer and watch for punctuation to split sentences
            partial_buffer = ""
            char_count = 0
            waiting_for_punctuation = False
            assistant_buffer = ""  # Store the complete response
            
            print("Assistant: ", end="", flush=True)
            
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
            
            # Process any remaining text in the buffer
            if partial_buffer.strip():
                if self.tts_enabled:
                    audio_array, sample_rate = self.tts_handler.synthesize(partial_buffer)
                    if audio_array is not None and len(audio_array) > 0:
                        self.audio_handler.play_audio(audio_array, sample_rate)
            
            # Add the complete response to conversation history
            self.conversation_history.append({"role": "assistant", "content": assistant_buffer})
            
            return transcribed_text, assistant_buffer
        except Exception as e:
            print(f"Unexpected error in streaming interaction: {str(e)}")
            import traceback
            traceback.print_exc()
            return "", "I'm sorry, I encountered an unexpected error."
            
    def toggle_interruptions(self, allow=None):
        """Toggle whether interruptions are allowed.
        
        Args:
            allow (bool, optional): If provided, set interruption state to this value
                                   If not provided, toggle the current state
        
        Returns:
            bool: New interruption state
        """
        if allow is not None:
            self.allow_interruptions = allow
        else:
            self.allow_interruptions = not self.allow_interruptions
            
        print(f"Interruptions are now {'enabled' if self.allow_interruptions else 'disabled'}")
        return self.allow_interruptions 