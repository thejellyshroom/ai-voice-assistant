# AI Voice Assistant Configuration

This document explains how to use the configuration files for the AI Voice Assistant.

## Configuration Files

The AI Voice Assistant uses three configuration files:

1. `conf_asr.json` - Configuration for Automatic Speech Recognition (ASR)
2. `conf_llm.json` - Configuration for Large Language Model (LLM)
3. `conf_tts.json` - Configuration for Text-to-Speech (TTS)

## Running with Configurations

You can run the assistant with specific configurations using:

```bash
python -m ai_voice_assistant.config_assistant --asr-config 1 --llm-config 1 --tts-config 1
```

To see all available configurations:

```bash
python -m ai_voice_assistant.config_assistant --list-configs
```

## Current Configurations

### ASR Configuration (conf_asr.json)

- **1**: Faster Whisper (h2oai/faster-whisper-large-v3-turbo)
  - A high-quality speech recognition model that runs on CPU

### LLM Configuration (conf_llm.json)

- **1**: Llama 3.2 1B Instruct (Local)
  - The smallest version of Llama 3.2, optimized for instruction following
  - Runs locally on your machine
- **2**: OpenAI API (Remote)
  - Uses OpenAI's API for language model capabilities
  - Requires API key configuration

### TTS Configuration (conf_tts.json)

- **1**: Kokoro TTS
  - A high-quality text-to-speech model
  - Uses the "af_heart" voice by default
  - Speech speed set to 1.3x

## Customizing Configurations

You can customize the configurations by editing the JSON files directly. Each configuration has an "EXAMPLE" entry that shows all available options.

### Example: Adding a New TTS Configuration

To add a new TTS configuration with a different voice:

```json
"2": {
  "tts_name": "kokoro",
  "kokoro": {
    "model_name": "hexgrad/Kokoro-82M",
    "voice": "am_adam",
    "device": "cpu",
    "speech_speed": 1.0,
    "sample_rate": 24000
  }
}
```

Then run with:

```bash
python -m ai_voice_assistant.config_assistant --tts-config 2
```

## Troubleshooting

If you encounter issues with the configurations:

1. Make sure the model paths are correct
2. Check that you have the required dependencies installed
3. Verify that your hardware meets the minimum requirements for the models
4. For API-based models, ensure your API keys are properly configured
