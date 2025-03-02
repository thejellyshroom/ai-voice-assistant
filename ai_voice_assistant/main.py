from ai_voice_assistant.voice_assistant import VoiceAssistant
import argparse


# Define available voices
KOKORO_VOICES = {
    "American Female": ["af_heart", "af_alloy", "af_aoede", "af_bella", "af_jessica", "af_kore", "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky"],
    "American Male": ["am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam", "am_michael", "am_onyx", "am_puck", "am_santa"],
    "British Female": ["bf_alice", "bf_emma", "bf_isabella", "bf_lily"],
    "British Male": ["bm_daniel", "bm_fable", "bm_george", "bm_lewis"]
}

def list_available_voices():
    """Print a formatted list of available voices."""
    print("\nAvailable Kokoro voices:")
    for accent, voices in KOKORO_VOICES.items():
        print(f"  {accent}:")
        for voice in voices:
            print(f"    - {voice}")
    print()

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='AI Voice Assistant')
    parser.add_argument('--fixed-duration', type=int, help='Use fixed duration recording instead of dynamic listening')
    parser.add_argument('--timeout', type=int, default=10, help='Maximum seconds to wait for speech before giving up')
    parser.add_argument('--phrase-limit', type=int, default=60, help='Maximum seconds for a single phrase')

    parser.add_argument('--tts-model', type=str, default="hexgrad/Kokoro-82M", 
                        help='TTS model to use (default: hexgrad/Kokoro-82M)')
    parser.add_argument('--tts-voice', type=str, default="af_heart", 
                        help='Voice to use for Kokoro TTS (default: af_heart)')
    parser.add_argument('--speech-speed', type=float, default=1.3, 
                        help='Speed factor for speech (default: 1.3, range: 0.5-2.0)')
    parser.add_argument('--list-voices', action='store_true', help='List all available Kokoro voices and exit')
    args = parser.parse_args()
    
    # If --list-voices is specified, print available voices and exit
    if args.list_voices:
        list_available_voices()
        return
    
    # Check if the specified voice is valid
    all_voices = [voice for voices in KOKORO_VOICES.values() for voice in voices]
    if args.tts_voice not in all_voices:
        print(f"Warning: '{args.tts_voice}' is not a recognized Kokoro voice.")
        print("Using default voice 'af_heart' instead.")
        print("Use --list-voices to see all available voices.")
        args.tts_voice = "af_heart"
    
    # Initialize the voice assistant
    try:
        assistant = VoiceAssistant(
            tts_model=args.tts_model, 
            tts_voice=args.tts_voice,
            speech_speed=args.speech_speed
        )
        
        print("\nAI Voice Assistant initialized!")
        print("Device set to use:", assistant.transcriber.device)
        print("TTS Model:", args.tts_model)
        print("TTS Voice:", args.tts_voice)
        print("Speech Speed:", f"{args.speech_speed}x")
        # Remove the ONNX Quantization print
        print("Press Ctrl+C to exit")
        
        try:
            while True:
                print("\nListening for your voice...")
                
                # Use either fixed duration or dynamic listening based on arguments
                if args.fixed_duration:
                    transcribed_text, ai_response = assistant.interact(duration=args.fixed_duration)
                else:
                    transcribed_text, ai_response = assistant.interact(
                        timeout=args.timeout,
                        phrase_time_limit=args.phrase_limit
                    )
                
                print("\nYou said:", transcribed_text)
                print("Assistant:", ai_response)
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
    except Exception as e:
        print(f"Error initializing voice assistant: {str(e)}")
        print("\nTroubleshooting tips:")
        print("1. Make sure espeak-ng is installed on your system")
        print("   - macOS: brew install espeak-ng")
        print("   - Linux: sudo apt-get install espeak-ng")
        print("2. Check that all Python dependencies are installed")
        print("   - Run: pip install -r requirements.txt")
        print("3. Try a different voice with --tts-voice")
        print("   - Run: python -m ai_voice_assistant.main --list-voices")
        print("4. Try a different quantization level with --quantization")
        print("   - Options: fp32, fp16, q8, q4, q4f16")

if __name__ == "__main__":
    main() 