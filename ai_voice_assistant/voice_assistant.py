from .audio_handler import AudioHandler
from .transcriber import Transcriber
from .llm_handler import LLMHandler
from .tts_handler import TTSHandler
import os


class VoiceAssistant:
    def __init__(self, tts_model="hexgrad/Kokoro-82M", tts_voice="af_heart", speech_speed=1.3):
        """Initialize the voice assistant with all components.
        
        Args:
            tts_model (str): The TTS model to use (default: "NeuML/kokoro-int8-onnx")
            tts_voice (str): The voice to use for Kokoro TTS (default: "af_heart" - American female Nova)
            speech_speed (float): Speed factor for speech (default: 1.3, range: 0.5-2.0)
        """
        self.audio_handler = AudioHandler()
        self.transcriber = Transcriber()
        self.llm_handler = LLMHandler()
        
        # Initialize TTS with error handling
        try:
            self.tts_handler = TTSHandler(
                model_id=tts_model, 
                voice=tts_voice, 
                speech_speed=speech_speed,
            )
            self.tts_enabled = True
        except Exception as e:
            print(f"Error initializing TTS: {str(e)}")
            print("Voice output will be disabled.")
            self.tts_enabled = False
        
        # Store conversation history
        self.conversation_history = [
            {"role": "system", "content": "You are a helpful AI voice assistant."}
        ]
        
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
                
            audio_array, sample_rate = self.tts_handler.synthesize(text)
            
            # Ensure audio_array is not None
            if audio_array is None:
                print("Warning: Received None audio array from TTS handler")
                return False
                
            if len(audio_array) > 0:
                self.audio_handler.play_audio(audio_array, sample_rate)
                return True
            else:
                print("Generated audio is empty. Cannot play.")
                return False
        except Exception as e:
            print(f"Error in speech synthesis: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def interact(self, duration=None, timeout=10):
        """Complete interaction cycle: listen, process, respond, and speak.
        
        Args:
            duration (int, optional): Fixed recording duration in seconds (legacy mode)
            timeout (int, optional): Maximum seconds to wait before giving up
            
        Returns:
            tuple: (transcribed_text, ai_response)
        """
        try:
            # Listen and transcribe
            transcribed_text = self.listen(
                duration=duration, 
                timeout=timeout,
            )
            
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
            
            # Process and get response
            try:
                ai_response = self.process_and_respond(transcribed_text)
            except Exception as e:
                print(f"Error processing response: {str(e)}")
                ai_response = "I'm sorry, I encountered an error while processing your request."
            
            # Speak the response
            try:
                speech_success = self.speak(ai_response)
                if not speech_success:
                    print("Note: The response was not spoken aloud due to TTS issues.")
            except Exception as e:
                print(f"Error speaking response: {str(e)}")
            
            return transcribed_text, ai_response
        except Exception as e:
            print(f"Unexpected error in interaction: {str(e)}")
            import traceback
            traceback.print_exc()
            return "", "I'm sorry, I encountered an unexpected error." 