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

def list_config_presets(config_file, label="presets"):
    """Print available configuration presets from a config file.
    
    Args:
        config_file (str): Path to configuration file
        label (str): Label for the list output
    """
    try:
        config = load_config(config_file)
        if config:
            print(f"\nAvailable {label}:")
            for preset in config.keys():
                print(f"  - {preset}")
            print()
        else:
            print(f"No {label} found in {config_file}")
    except Exception as e:
        print(f"Error listing presets from {config_file}: {str(e)}")

def get_default_config_paths():
    """Get default paths for config files."""
    # Get the directory of the current file
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    return {
        'asr': os.path.join(base_dir, 'conf_asr.json'),
        'tts': os.path.join(base_dir, 'conf_tts.json'),
        'llm': os.path.join(base_dir, 'conf_llm.json')
    }

def main():
    # Get default config paths
    default_config_paths = get_default_config_paths()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='AI Voice Assistant')
    parser.add_argument('--fixed-duration', type=int, help='Use fixed duration recording instead of dynamic listening')
    parser.add_argument('--timeout', type=int, default=10, help='Maximum seconds to wait for speech before giving up')
    parser.add_argument('--phrase-limit', type=int, default=60, help='Maximum seconds for a single phrase')

    # Configuration file options
    parser.add_argument('--config', type=str, help='Path to global configuration file')
    parser.add_argument('--asr-config', type=str, default=default_config_paths['asr'], help='Path to ASR configuration file')
    parser.add_argument('--tts-config', type=str, default=default_config_paths['tts'], help='Path to TTS configuration file')
    parser.add_argument('--llm-config', type=str, default=default_config_paths['llm'], help='Path to LLM configuration file')
    
    # Preset selection options
    parser.add_argument('--asr-preset', type=str, default='default', help='ASR preset to use')
    parser.add_argument('--tts-preset', type=str, default='default', help='TTS preset to use')
    parser.add_argument('--llm-preset', type=str, default='default', help='LLM preset to use')
    
    # List available presets
    parser.add_argument('--list-asr-presets', action='store_true', help='List available ASR presets')
    parser.add_argument('--list-tts-presets', action='store_true', help='List available TTS presets')
    parser.add_argument('--list-llm-presets', action='store_true', help='List available LLM presets')
    
    # Override options for specific parameters
    parser.add_argument('--tts-voice', type=str, help='Override voice for TTS')
    parser.add_argument('--speech-speed', type=float, help='Override speed factor for speech')
    parser.add_argument('--creativity', type=str, help='Override LLM creativity preset')
    parser.add_argument('--transcription-model', type=str, help='Override transcription model')
    
    args = parser.parse_args()
    
    # List presets if requested
    if args.list_asr_presets:
        list_config_presets(args.asr_config, "ASR presets")
        return
    
    if args.list_tts_presets:
        list_config_presets(args.tts_config, "TTS presets")
        return
    
    if args.list_llm_presets:
        list_config_presets(args.llm_config, "LLM presets")
        return
    
    # Load configurations
    # Start with empty configs
    asr_config = {}
    tts_config = {}
    llm_config = {}
    
    # Load from config files
    if os.path.exists(args.asr_config):
        asr_conf_all = load_config(args.asr_config)
        if args.asr_preset in asr_conf_all:
            asr_config = asr_conf_all[args.asr_preset]
        else:
            print(f"Warning: ASR preset '{args.asr_preset}' not found. Using 'default' preset.")
            asr_config = asr_conf_all.get('default', {})
    
    if os.path.exists(args.tts_config):
        tts_conf_all = load_config(args.tts_config)
        if args.tts_preset in tts_conf_all:
            tts_config = tts_conf_all[args.tts_preset]
        else:
            print(f"Warning: TTS preset '{args.tts_preset}' not found. Using 'default' preset.")
            tts_config = tts_conf_all.get('default', {})
    
    if os.path.exists(args.llm_config):
        llm_conf_all = load_config(args.llm_config)
        if args.llm_preset in llm_conf_all:
            llm_config = llm_conf_all[args.llm_preset]
        else:
            print(f"Warning: LLM preset '{args.llm_preset}' not found. Using 'default' preset.")
            llm_config = llm_conf_all.get('default', {})
    
    # Merge command line arguments with configs
    # Command line args override config values
    
    # Override TTS config with command line args
    if args.tts_voice:
        if 'kokoro' in tts_config:
            tts_config['kokoro']['voice'] = args.tts_voice
        else:
            tts_config['voice'] = args.tts_voice
    
    if args.speech_speed:
        if 'kokoro' in tts_config:
            tts_config['kokoro']['speech_speed'] = args.speech_speed
        else:
            tts_config['speech_speed'] = args.speech_speed
    
    # Override ASR config with command line args
    if args.transcription_model:
        asr_config['model_id'] = args.transcription_model
    
    # Override LLM config with command line args
    if args.creativity:
        if 'local' in llm_config:
            llm_config['local']['creativity'] = args.creativity
        else:
            llm_config['creativity'] = args.creativity
    
    # Prepare parameters for VoiceAssistant
    assistant_params = {}
    
    # Add TTS parameters
    if 'model_id' in tts_config:
        assistant_params['tts_model'] = tts_config['model_id']
    
    if 'kokoro' in tts_config:
        kokoro_conf = tts_config['kokoro']
        assistant_params['tts_voice'] = kokoro_conf.get('voice', 'af_heart')
        assistant_params['speech_speed'] = kokoro_conf.get('speech_speed', 1.3)
        assistant_params['expressiveness'] = kokoro_conf.get('expressiveness', 1.0)
        assistant_params['variability'] = kokoro_conf.get('variability', 0.2)
    
    # Add ASR parameters
    if 'model_id' in asr_config:
        assistant_params['transcription_model'] = asr_config['model_id']
    
    # Add LLM parameters
    if 'local' in llm_config:
        local_conf = llm_config['local']
        assistant_params['temperature'] = local_conf.get('temperature', 0.7)
        assistant_params['top_p'] = local_conf.get('top_p', 0.9)
        assistant_params['top_k'] = local_conf.get('top_k', 40)
        if 'creativity' in local_conf:
            assistant_params['creativity'] = local_conf['creativity']
    
    # Provide configs to VoiceAssistant
    assistant_params['asr_config'] = asr_config
    assistant_params['tts_config'] = tts_config
    assistant_params['llm_config'] = llm_config
    
    # Initialize the voice assistant
    assistant = VoiceAssistant(**assistant_params)

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