import json
import os
import argparse
from ai_voice_assistant.voice_assistant import VoiceAssistant
from ai_voice_assistant.transcriber import Transcriber
from ai_voice_assistant.llm_handler import LLMHandler
from ai_voice_assistant.tts_handler import TTSHandler


def load_configuration(config_file, config_key):
    """Load configuration from a JSON file.
    
    Args:
        config_file (str): Path to the configuration file
        config_key (str): Key to use in the configuration file
        
    Returns:
        dict: Configuration parameters
    """
    config_path = os.path.join(os.path.dirname(__file__), config_file)
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if config_key not in data:
        raise ValueError(f"Configuration key '{config_key}' not found in {config_file}")
    
    return data[config_key]


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='AI Voice Assistant with Configuration Files')
    parser.add_argument('--asr-config', type=str, default="1", help='ASR configuration key (default: 1)')
    parser.add_argument('--llm-config', type=str, default="1", help='LLM configuration key (default: 1)')
    parser.add_argument('--tts-config', type=str, default="1", help='TTS configuration key (default: 1)')
    parser.add_argument('--timeout', type=int, default=10, help='Maximum seconds to wait for speech before giving up')
    parser.add_argument('--phrase-limit', type=int, default=60, help='Maximum seconds for a single phrase')
    parser.add_argument('--list-configs', action='store_true', help='List all available configurations and exit')
    
    args = parser.parse_args()
    
    # If --list-configs is specified, print available configurations and exit
    if args.list_configs:
        print("\nAvailable ASR configurations:")
        with open(os.path.join(os.path.dirname(__file__), "conf_asr.json"), "r") as f:
            asr_configs = json.load(f)
        for key in asr_configs.keys():
            if key != "EXAMPLE":
                print(f"  - {key}")
        
        print("\nAvailable LLM configurations:")
        with open(os.path.join(os.path.dirname(__file__), "conf_llm.json"), "r") as f:
            llm_configs = json.load(f)
        for key in llm_configs.keys():
            if key != "EXAMPLE":
                print(f"  - {key}")
        
        print("\nAvailable TTS configurations:")
        with open(os.path.join(os.path.dirname(__file__), "conf_tts.json"), "r") as f:
            tts_configs = json.load(f)
        for key in tts_configs.keys():
            if key != "EXAMPLE":
                print(f"  - {key}")
        
        return
    
    # Load configurations
    try:
        asr_config = load_configuration("conf_asr.json", args.asr_config)
        llm_config = load_configuration("conf_llm.json", args.llm_config)
        tts_config = load_configuration("conf_tts.json", args.tts_config)
    except Exception as e:
        print(f"Error loading configurations: {e}")
        return
    
    # Print configuration information
    print("\nAI Voice Assistant with Configuration Files")
    print(f"ASR Configuration: {args.asr_config}")
    print(f"LLM Configuration: {args.llm_config}")
    print(f"TTS Configuration: {args.tts_config}")
    
    # Initialize the voice assistant with the configurations
    # For now, we'll use the existing VoiceAssistant class with our configurations
    
    # Get transcription model from ASR config
    if asr_config["asr_name"] == "faster-whisper":
        transcription_model = asr_config["faster-whisper"]["model_name"]
    else:
        transcription_model = "h2oai/faster-whisper-large-v3-turbo"  # Default
    
    # Get TTS parameters from TTS config
    if tts_config["tts_name"] == "kokoro":
        tts_model = tts_config["kokoro"]["model_name"]
        tts_voice = tts_config["kokoro"]["voice"]
        speech_speed = tts_config["kokoro"]["speech_speed"]
    else:
        tts_model = "hexgrad/Kokoro-82M"  # Default
        tts_voice = "af_heart"  # Default
        speech_speed = 1.3  # Default
    
    # Initialize the voice assistant
    assistant = VoiceAssistant(
        tts_model=tts_model,
        tts_voice=tts_voice,
        speech_speed=speech_speed,
        transcription_model=transcription_model
    )
    
    # Start the interaction loop
    try:
        while True:
            assistant.interact_streaming(timeout=args.timeout)
    except KeyboardInterrupt:
        print("\nExiting...")


if __name__ == "__main__":
    main() 