from .audio_handler import AudioHandler
from .transcriber import Transcriber
from .llm_handler import LLMHandler
from .tts_handler import TTSHandler

class VoiceAssistant:
    def __init__(self, tts_model="NeuML/kokoro-int8-onnx", tts_voice="af_nova", speech_speed=1.3, quantization="fp32"):
        """Initialize the voice assistant with all components.
        
        Args:
            tts_model (str): The TTS model to use (default: "NeuML/kokoro-int8-onnx")
            tts_voice (str): The voice to use for Kokoro TTS (default: "af_nova" - American female Nova)
            speech_speed (float): Speed factor for speech (default: 1.3, range: 0.5-2.0)
            quantization (str): ONNX model quantization to use (default: "fp32", options: "fp32", "fp16", "q8", "q4", "q4f16")
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
                quantization=quantization
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
        
    def listen(self, duration=None, timeout=None, phrase_time_limit=None):
        """Record audio and transcribe it.
        
        Args:
            duration (int, optional): Fixed recording duration in seconds (legacy mode)
            timeout (int, optional): Maximum seconds to wait before giving up
            phrase_time_limit (int, optional): Maximum seconds for a phrase
            
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
                phrase_time_limit=phrase_time_limit
            )
            
        if audio_file is None:
            return ""
            
        return self.transcriber.transcribe(audio_file)
    
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
            audio_array, sample_rate = self.tts_handler.synthesize(text)
            if len(audio_array) > 0:
                self.audio_handler.play_audio(audio_array, sample_rate)
                return True
            else:
                print("Generated audio is empty. Cannot play.")
                return False
        except Exception as e:
            print(f"Error in speech synthesis: {str(e)}")
            return False
    
    def interact(self, duration=None, timeout=10, phrase_time_limit=60):
        """Complete interaction cycle: listen, process, respond, and speak.
        
        Args:
            duration (int, optional): Fixed recording duration in seconds (legacy mode)
            timeout (int, optional): Maximum seconds to wait before giving up
            phrase_time_limit (int, optional): Maximum seconds for a phrase
            
        Returns:
            tuple: (transcribed_text, ai_response)
        """
        # Listen and transcribe
        transcribed_text = self.listen(
            duration=duration, 
            timeout=timeout, 
            phrase_time_limit=phrase_time_limit
        )
        
        if not transcribed_text:
            return "", "I didn't hear anything. Could you please speak again?"
        
        # Process and get response
        ai_response = self.process_and_respond(transcribed_text)
        
        # Speak the response
        speech_success = self.speak(ai_response)
        if not speech_success:
            print("Note: The response was not spoken aloud due to TTS issues.")
        
        return transcribed_text, ai_response 