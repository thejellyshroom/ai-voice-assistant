#!/bin/bash

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate

# Install system dependencies
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    echo "Installing system dependencies for macOS..."
    brew install portaudio ffmpeg espeak-ng
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    echo "Installing system dependencies for Linux..."
    sudo apt-get update
    sudo apt-get install -y portaudio19-dev ffmpeg espeak-ng
else
    echo "Unsupported OS. Please install portaudio, ffmpeg, and espeak-ng manually."
fi

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Install the package in development mode
echo "Installing package in development mode..."
pip install -e .

# Install development dependencies
pip install pytest black isort mypy

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "Ollama is not installed. Please install it from https://ollama.com/download"
    echo "After installation, run: ollama pull llama3.2"
fi

echo "Development environment setup complete!"
echo "Activate the virtual environment with: source venv/bin/activate" 