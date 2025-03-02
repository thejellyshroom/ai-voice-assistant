#!/bin/bash

# Exit on error
set -e

echo "Creating a new conda environment with Python 3.10 for AI Voice Assistant..."
conda create -n ai-voice-py310 python=3.10 -y

echo "Activating the new environment..."
eval "$(conda shell.bash hook)"
conda activate ai-voice-py310

echo "Installing PyAudio and other system dependencies..."
# For macOS
if [[ "$OSTYPE" == "darwin"* ]]; then
    # Check if Homebrew is installed
    if ! command -v brew &> /dev/null; then
        echo "Homebrew not found. Please install Homebrew first."
        exit 1
    fi
    
    # Install portaudio for PyAudio
    brew install portaudio
    
    # Install other dependencies
    brew install ffmpeg
    
    # Install espeak-ng for Kokoro TTS
    brew install espeak-ng
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # For Linux
    sudo apt-get update
    sudo apt-get install -y portaudio19-dev ffmpeg espeak-ng
fi

echo "Installing required Python packages..."
pip install --upgrade pip
pip install torch torchaudio

# Install packages from requirements.txt
pip install -r requirements.txt

# Install additional packages needed for Kokoro TTS
pip install kokoro>=0.3.4 sentencepiece accelerate einops soundfile sounddevice SpeechRecognition

# Install the package in development mode
pip install -e .

echo "Environment setup complete! You can activate it with: conda activate ai-voice-py310"
echo "Run your voice assistant with: python -m ai_voice_assistant.main"
echo "To specify a different voice, use: python -m ai_voice_assistant.main --tts-voice af_bella" 