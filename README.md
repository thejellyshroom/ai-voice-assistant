# AI Voice Assistant

A voice-based AI assistant using Whisper for speech recognition, Ollama for LLM interaction, and Kokoro v1 for text-to-speech.

## Requirements

- Python 3.10 (specifically required for compatibility)
- Conda or Miniconda (recommended for environment management)
- FFmpeg (for audio processing)
- PortAudio (for PyAudio)
- espeak-ng (for Kokoro TTS phoneme generation)

## Setup

### Option 1: Automatic Setup (Recommended)

1. Clone this repository:

   ```
   git clone <repository-url>
   cd ai-voice-assistant
   ```

2. Run the setup script to create a Python 3.10 environment and install all dependencies:

   ```
   chmod +x setup_py310_env.sh
   ./setup_py310_env.sh
   ```

### Option 2: Manual Setup

1. Create a new conda environment with Python 3.10:

   ```
   conda create -n ai-voice-py310 python=3.10
   conda activate ai-voice-py310
   ```

2. Install system dependencies:

   - On macOS:
     ```
     brew install portaudio ffmpeg espeak-ng
     ```
   - On Ubuntu/Debian:
     ```
     sudo apt-get install portaudio19-dev ffmpeg espeak-ng
     ```

3. Install Python dependencies:

   ```
   pip install torch==2.1.2 torchaudio==2.1.2
   pip install transformers==4.37.2
   pip install accelerate==0.26.1 sentencepiece==0.1.99 einops==0.7.0
   pip install soundfile==0.12.1 sounddevice==0.4.6
   pip install numpy==1.24.4 scipy==1.11.4 librosa==0.10.1
   pip install ollama pyaudio SpeechRecognition ffmpeg-python
   pip install kokoro>=0.3.4
   ```

4. Install the package in development mode:
   ```
   pip install -e .
   ```

## Testing

Test the Kokoro TTS model:

```
python test_kokoro_tts.py
```

You can also test different voices:

```
python test_kokoro_tts.py --voice af_bella
```

Available voices include:

- American English: af (default), af_bella, af_sarah, af_sky, af_nicole, am_adam, am_michael
- British English: bf_emma, bf_isabella, bm_george, bm_lewis

## Usage

1. Activate the environment:

   ```
   conda activate ai-voice-py310
   ```

2. Run the voice assistant:

   ```
   python -m ai_voice_assistant.main
   ```

   Optional arguments:

   - `--fixed-duration N`: Use fixed duration recording of N seconds instead of dynamic listening
   - `--timeout N`: Maximum seconds to wait for speech before giving up (default: 10)
   - `--phrase-limit N`: Maximum seconds for a single phrase (default: 60)
   - `--tts-model MODEL`: TTS model to use (default: hexgrad/Kokoro-82M)
   - `--tts-voice VOICE`: Voice to use for Kokoro TTS (default: af_sky)

## Features

- **Speech Recognition**: Uses OpenAI's Whisper model for accurate transcription
- **Natural Language Processing**: Processes queries with Ollama's LLM models
- **Text-to-Speech**: Generates natural-sounding responses using Kokoro v1 TTS model
- **Dynamic Listening**: Automatically detects when you start and stop speaking
- **Multiple Voices**: Choose from a variety of American and British English voices

## Troubleshooting

- **PyAudio Installation Issues**: If you encounter problems installing PyAudio, make sure you have PortAudio installed on your system.
- **TTS Model Loading Issues**: Kokoro requires espeak-ng for phoneme generation. Make sure it's installed on your system.
- **Audio Playback Issues**: If you encounter audio playback issues, check your system's audio settings and ensure the correct output device is selected.

## License

MIT
