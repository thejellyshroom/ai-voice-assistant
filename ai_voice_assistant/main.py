from ai_voice_assistant.voice_assistant import VoiceAssistant
import argparse
import json
import os


# Define available voices
KOKORO_VOICES = {
    "American Female": ["af_heart", "af_alloy", "af_aoede", "af_bella", "af_jessica", "af_kore", "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky"],
}

# Define available transcription models
TRANSCRIPTION_MODELS = {
    "faster-whisper": "h2oai/faster-whisper-large-v3-turbo",
    "faster-whisper-small": "Systran/faster-whisper-small",
    "transformers-whisper": "openai/whisper-large-v3-turbo"
}

# Define LLM creativity presets
LLM_CREATIVITY_PRESETS = ["low", "medium", "high", "random"]

def list_available_voices():
    """Print a formatted list of available voices."""
    print("\nAvailable Kokoro voices:")
    for accent, voices in KOKORO_VOICES.items():
        print(f"  {accent}:")
        for voice in voices:
            print(f"    - {voice}")
    print()

def list_available_transcription_models():
    """Print a formatted list of available transcription models."""
    print("\nAvailable transcription models:")
    for name, model_id in TRANSCRIPTION_MODELS.items():
        print(f"  - {name}: {model_id}")
    print()


def list_llm_creativity_presets():
    """Print a formatted list of available LLM creativity presets."""
    print("\nAvailable LLM creativity presets:")
    for preset in LLM_CREATIVITY_PRESETS:
        print(f"  - {preset}")
    print()

def load_config(config_file):
    """Load configuration from a JSON file.
    
    Args:
        config_file (str): Path to the configuration file
        
    Returns:
        dict: Configuration dictionary
    """
    try:
        with open(config_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading configuration file {config_file}: {str(e)}")
        return {}

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='AI Voice Assistant')
    parser.add_argument('--fixed-duration', type=int, help='Use fixed duration recording instead of dynamic listening')
    parser.add_argument('--timeout', type=int, default=10, help='Maximum seconds to wait for speech before giving up')
    parser.add_argument('--phrase-limit', type=int, default=60, help='Maximum seconds for a single phrase')

    # TTS parameters
    parser.add_argument('--tts-model', type=str, default="hexgrad/Kokoro-82M", 
                        help='TTS model to use (default: hexgrad/Kokoro-82M)')
    parser.add_argument('--tts-voice', type=str, default="af_heart", 
                        help='Voice to use for Kokoro TTS (default: af_heart)')
    parser.add_argument('--speech-speed', type=float, default=1.3, 
                        help='Speed factor for speech (default: 1.3, range: 0.5-2.0)')
    parser.add_argument('--expressiveness', type=float, default=1.0,
                        help='Voice expressiveness (default: 1.0, range: 0.0-2.0)')
    parser.add_argument('--variability', type=float, default=0.3,
                        help='Speech variability (default: 0.3, range: 0.0-1.0)')
                        
    # LLM parameters
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='LLM temperature (default: 0.7, range: 0.0-2.0)')
    parser.add_argument('--top-p', type=float, default=0.9,
                        help='LLM top-p sampling (default: 0.9, range: 0.0-1.0)')
    parser.add_argument('--top-k', type=int, default=40,
                        help='LLM top-k sampling (default: 40)')
    parser.add_argument('--creativity', type=str, default="high", choices=LLM_CREATIVITY_PRESETS,
                        help='LLM creativity preset (default: High)')
                        
    # List options
    parser.add_argument('--list-voices', action='store_true', help='List all available Kokoro voices and exit')
    parser.add_argument('--list-creativity-presets', action='store_true', help='List all available LLM creativity presets and exit')
    
    # Add transcription model options
    parser.add_argument('--transcription-model', type=str, default="faster-whisper-small", 
                        choices=list(TRANSCRIPTION_MODELS.keys()),
                        help='Transcription model to use (default: faster-whisper-small)')
    parser.add_argument('--list-transcription-models', action='store_true', 
                        help='List all available transcription models and exit')
    
    
    # Add configuration file options
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--asr-config', type=str, help='Path to ASR configuration file')
    parser.add_argument('--tts-config', type=str, help='Path to TTS configuration file')
    parser.add_argument('--llm-config', type=str, help='Path to LLM configuration file')
    
    args = parser.parse_args()
    
    # If --list-voices is specified, print available voices and exit
    if args.list_voices:
        list_available_voices()
        return
    
    # If --list-transcription-models is specified, print available models and exit
    if args.list_transcription_models:
        list_available_transcription_models()
        return
        
    # If --list-creativity-presets is specified, print available presets and exit
    if args.list_creativity_presets:
        list_llm_creativity_presets()
        return
    
    # Load configuration from files if specified
    config = {}
    if args.config and os.path.exists(args.config):
        config = load_config(args.config)
    
    # Load component-specific configurations
    asr_config = {}
    if args.asr_config and os.path.exists(args.asr_config):
        asr_config = load_config(args.asr_config)
    
    tts_config = {}
    if args.tts_config and os.path.exists(args.tts_config):
        tts_config = load_config(args.tts_config)
    
    llm_config = {}
    if args.llm_config and os.path.exists(args.llm_config):
        llm_config = load_config(args.llm_config)
    
    # Merge configurations with command line arguments taking precedence
    tts_model = args.tts_model
    tts_voice = args.tts_voice
    speech_speed = args.speech_speed
    expressiveness = args.expressiveness
    variability = args.variability
    
    # LLM parameters
    temperature = args.temperature
    top_p = args.top_p
    top_k = args.top_k
    creativity = args.creativity
    
    # Override with config values if available
    if 'tts_model' in config:
        tts_model = config.get('tts_model', tts_model)
    if 'tts_voice' in config:
        tts_voice = config.get('tts_voice', tts_voice)
    if 'speech_speed' in config:
        speech_speed = config.get('speech_speed', speech_speed)
    if 'expressiveness' in config:
        expressiveness = config.get('expressiveness', expressiveness)
    if 'variability' in config:
        variability = config.get('variability', variability)
    
    # Override LLM parameters from config if available
    if 'temperature' in config:
        temperature = config.get('temperature', temperature)
    if 'top_p' in config:
        top_p = config.get('top_p', top_p)
    if 'top_k' in config:
        top_k = config.get('top_k', top_k)
    if 'creativity' in config:
        creativity = config.get('creativity', creativity)
    
    # Check if the specified voice is valid
    all_voices = [voice for voices in KOKORO_VOICES.values() for voice in voices]
    if tts_voice not in all_voices:
        print(f"Warning: '{tts_voice}' is not a recognized Kokoro voice.")
        print("Using default voice 'af_heart' instead.")
        print("Use --list-voices to see all available voices.")
        tts_voice = "af_heart"
    
    # Check if the specified creativity preset is valid
    if creativity and creativity not in LLM_CREATIVITY_PRESETS:
        print(f"Warning: '{creativity}' is not a recognized creativity preset.")
        print("Not using any preset.")
        print("Use --list-creativity-presets to see all available presets.")
        creativity = None
    
    # Get the transcription model ID
    transcription_model_name = args.transcription_model
    if 'transcription_model' in config:
        transcription_model_name = config.get('transcription_model', transcription_model_name)
    
    transcription_model = TRANSCRIPTION_MODELS[transcription_model_name]
    print(f"Using transcription model: {transcription_model_name} ({transcription_model})")
    
    # Initialize the voice assistant
    assistant = VoiceAssistant(
        # TTS parameters
        tts_model=tts_model,
        tts_voice=tts_voice,
        speech_speed=speech_speed,
        expressiveness=expressiveness,
        variability=variability,
        # ASR parameters
        transcription_model=transcription_model,
        # LLM parameters
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        creativity=creativity
    )

    # Start the interaction loop
    try:
        while True:
            assistant.interact_streaming(
                duration=args.fixed_duration, 
                timeout=args.timeout,
                phrase_limit=args.phrase_limit,
            )
    except KeyboardInterrupt:
        print("\nExiting...")

if __name__ == "__main__":
    main() 