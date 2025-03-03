import ollama
from typing import Dict, Any, Generator
import random

class LLMHandler:
    def __init__(self, model_name='llama3.2', **kwargs):
        """Initialize the LLM handler with Ollama.
        
        Args:
            model_name (str): Ollama model name (default: 'llama3.2')
            **kwargs: Additional parameters
                - temperature (float): Temperature setting (default: 0.7)
                - top_p (float): Top-p sampling parameter (default: 0.9)
                - top_k (int): Top-k sampling parameter (default: 40)
                - frequency_penalty (float): Frequency penalty (default: 0.0)
                - presence_penalty (float): Presence penalty (default: 0.0)
                - seed (int): Random seed (default: random)
                - creativity (str): Preset for creativity ('low', 'medium', 'high', 'random')
                - creativity_presets (dict): Custom creativity presets
        """
        self.model_name = model_name
        
        # Default parameters for text generation
        self.default_params = {
            'temperature': 0.7,
            'top_p': 0.9,
            'top_k': 40,
            'frequency_penalty': 0.0,
            'presence_penalty': 0.0,
            'seed': kwargs.get('seed', random.randint(1, 999999))
        }
        
        # Define creativity presets
        self.creativity_presets = {
            'low': {
                'temperature': 0.3,
                'top_p': 0.8,
                'top_k': 20,
                'frequency_penalty': 0.0,
                'presence_penalty': 0.0
            },
            'medium': {
                'temperature': 0.7,
                'top_p': 0.9,
                'top_k': 40,
                'frequency_penalty': 0.1,
                'presence_penalty': 0.1
            },
            'high': {
                'temperature': 1.2,
                'top_p': 0.95,
                'top_k': 60,
                'frequency_penalty': 0.3,
                'presence_penalty': 0.3
            },
            'random': {
                'temperature': random.uniform(0.5, 1.5),
                'top_p': random.uniform(0.8, 0.98),
                'top_k': random.randint(20, 80),
                'frequency_penalty': random.uniform(0.0, 0.5),
                'presence_penalty': random.uniform(0.0, 0.5),
                'seed': random.randint(1, 999999)
            }
        }
        
        # Override default presets with custom ones if provided
        if 'creativity_presets' in kwargs:
            custom_presets = kwargs.pop('creativity_presets')
            if isinstance(custom_presets, dict):
                for preset_name, preset_values in custom_presets.items():
                    if isinstance(preset_values, dict):
                        self.creativity_presets[preset_name] = preset_values
        
        # Update default parameters with any provided values
        for param in ['temperature', 'top_p', 'top_k', 'frequency_penalty', 'presence_penalty', 'seed']:
            if param in kwargs:
                self.default_params[param] = kwargs[param]
                
        # Apply creativity preset if specified
        if 'creativity' in kwargs:
            self._apply_creativity_preset(kwargs['creativity'])
    
    def _apply_creativity_preset(self, preset):
        """Apply a creativity preset to the default parameters.
        
        Args:
            preset (str): Name of the preset to apply
        """
        if preset in self.creativity_presets:
            self.default_params.update(self.creativity_presets[preset])
            print(f"Applied '{preset}' creativity preset")
    
    def get_streaming_response_with_context(self, messages: list[Dict[str, Any]], **kwargs) -> Generator[str, None, None]:
        """Get a streaming response from the LLM using conversation history.
        
        Args:
            messages (list): List of message dictionaries with 'role' and 'content'
            **kwargs: Optional generation parameters to override defaults
                - temperature (float): Controls randomness (0.0-2.0, default 0.7)
                - top_p (float): Nucleus sampling parameter (0.0-1.0, default 0.9)
                - top_k (int): Limits vocabulary to top k tokens (default 40)
                - frequency_penalty (float): Penalize repeated tokens (-2.0-2.0, default 0.0)
                - presence_penalty (float): Penalize repeated topics (-2.0-2.0, default 0.0)
                - seed (int): Random seed for deterministic output (default random)
                - creativity (str): Preset for creativity level ('low', 'medium', 'high', 'random')
                
        Yields:
            str: Chunks of the LLM's response as they are generated
        """
        # Apply preset "creativity" levels if specified
        if 'creativity' in kwargs:
            params = self._get_creativity_preset(kwargs.pop('creativity'))
        else:
            params = self.default_params.copy()
        
        # Override with any explicitly provided parameters
        params.update(kwargs)
        
        # Log the generation parameters being used
        print(f"Using LLM parameters: temperature={params['temperature']}, top_p={params['top_p']}, "
              f"top_k={params['top_k']}, seed={params['seed']}")
        
        # Call Ollama with our parameters
        response = ollama.chat(
            model=self.model_name, 
            messages=messages,
            stream=True,
            options=params
        )
        
        for chunk in response:
            if 'message' in chunk and 'content' in chunk['message']:
                yield chunk['message']['content']
    
    def _get_creativity_preset(self, level: str) -> Dict[str, Any]:
        """Get parameter presets for different creativity levels.
        
        Args:
            level (str): Preset name ('low', 'medium', 'high', 'random', or custom preset)
            
        Returns:
            Dict[str, Any]: Parameter dictionary
        """
        params = self.default_params.copy()
        
        if level in self.creativity_presets:
            params.update(self.creativity_presets[level])
            
            # For random, regenerate values each time
            if level == 'random':
                random_preset = {
                    'temperature': random.uniform(0.5, 1.5),
                    'top_p': random.uniform(0.8, 0.98),
                    'top_k': random.randint(20, 80),
                    'frequency_penalty': random.uniform(0.0, 0.5),
                    'presence_penalty': random.uniform(0.0, 0.5),
                    'seed': random.randint(1, 999999)
                }
                params.update(random_preset)
        else:
            print(f"Warning: Unknown creativity preset '{level}'. Using default parameters.")
        
        return params