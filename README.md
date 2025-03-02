# AI Voice Assistant

An interactive voice assistant that uses speech recognition, large language models, and text-to-speech to create a natural conversation experience.

## Features

- **Speech Recognition**: Uses faster-whisper or transformers-based Whisper models for accurate transcription
- **Natural Language Processing**: Powered by Ollama for local LLM inference
- **Text-to-Speech**: Uses Kokoro TTS for high-quality voice synthesis
- **Streaming Mode**: Real-time response generation and speech synthesis for more natural conversations
- **Interruption Handling**: Allows interrupting the assistant while it's speaking
- **Configuration Files**: Supports JSON configuration files for easy customization

## Installation

1. Clone this repository:

   ```
   git clone https://github.com/yourusername/ai-voice-assistant.git
   cd ai-voice-assistant
   ```

2. Install the required dependencies:

   ```
   pip install -r requirements.txt
   ```

3. Make sure you have Ollama installed and running for LLM inference:

   ```
   # Install Ollama from https://ollama.ai/
   # Pull the llama3.2 model
   ollama pull llama3.2
   ```

## Usage

### Basic Usage

Run the assistant with default settings:

```
python -m ai_voice_assistant.main
```

### Command Line Options

The assistant supports various command line options:

```
python -m ai_voice_assistant.main --tts-voice af_nova --speech-speed 1.5 --streaming
```

Key options:

- `--streaming`: Enable streaming mode for more interactive responses
- `--tts-voice`: Select a voice for TTS (use `--list-voices` to see available options)
- `--speech-speed`: Adjust the speech speed (0.5-2.0)
- `--transcription-model`: Choose the transcription model (faster-whisper or transformers-whisper)
- `--timeout`: Maximum seconds to wait for speech before giving up
- `--phrase-limit`: Maximum seconds for a single phrase
- `--no-interruptions`: Disable the ability to interrupt the assistant while speaking

If you're experiencing issues with audio capture or false interruption detections, try these options:

```
python -m ai_voice_assistant.main --no-interruptions
```

### Configuration Files

You can use JSON configuration files to customize the assistant:

```
python -m ai_voice_assistant.main --config config.json --asr-config conf_asr.json --tts-config conf_tts.json --llm-config conf_llm.json
```

Example configuration files are provided in the repository:

- `conf_asr.json`: Configuration for speech recognition
- `conf_tts.json`: Configuration for text-to-speech
- `conf_llm.json`: Configuration for the language model

## Available Voices

The assistant uses Kokoro TTS with various voice options. To list all available voices:

```
python -m ai_voice_assistant.main --list-voices
```

## Transcription Models

To list available transcription models:

```
python -m ai_voice_assistant.main --list-transcription-models
```

## Interaction

Once the assistant is running:

1. Speak when prompted
2. The assistant will transcribe your speech, process it, and respond
3. In streaming mode, the assistant will start speaking as it generates the response
4. You can interrupt the assistant by speaking while it's responding

Press Ctrl+C to exit the assistant.

## Architecture

The assistant consists of several components:

- `VoiceAssistant`: Main class that coordinates all components
- `AudioHandler`: Handles audio recording and playback
- `Transcriber`: Converts speech to text
- `LLMHandler`: Processes text with a language model
- `TTSHandler`: Converts text to speech

## License

This project is licensed under the MIT License - see the LICENSE file for details.

### Troubleshooting

#### Short Audio Capture

If the assistant is only capturing a few seconds of your speech:

- Use the latest version which has improved audio capture parameters
- Speak clearly and at a consistent volume
- Try adjusting your microphone settings
- Make sure there isn't too much background noise
- If issues persist, try increasing the pause threshold with a custom configuration file

#### False Interruption Detections

If the assistant is detecting interruptions when there aren't any:

- Use the `--no-interruptions` flag to disable interruption detection completely
- The latest version has improved interruption detection with higher thresholds
- Try speaking in a quieter environment
- Make sure your microphone isn't picking up the assistant's audio output

#### Audio Playback Stopping Prematurely

If the assistant stops audio playback when it shouldn't:

- The latest version fixes this issue by only stopping playback when necessary
- Use the `--no-interruptions` flag if you don't want any interruptions
- For streaming mode, make sure you're using the latest version which handles audio playback correctly

## Recent Improvements

- **Improved Audio Capture**: Increased phrase time limit and adjusted pause thresholds to allow for longer speech input
- **Better Interruption Detection**: Raised energy thresholds and added duration checks to avoid false interruptions
- **Fixed Premature Audio Stopping**: Audio playback now only stops when necessary
- **Added No-Interruptions Mode**: Use `--no-interruptions` flag to disable interruption detection completely
