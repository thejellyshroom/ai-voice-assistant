import ollama
from typing import Dict, Any, Generator
import random

class LLMHandler:
    def __init__(self, model_name='llama3.2', config=None):
        self.model_name = model_name
        
        # Default parameters for text generation running locally
        local_config = config.get('local', {})
        # create parameters to send to ollama
        self.params = {
            'temperature': local_config.get('temperature'),
            'top_p': local_config.get('top_p'),
            'top_k': local_config.get('top_k'),
            'max_tokens': local_config.get('max_tokens'),
            'n_ctx': local_config.get('n_ctx'),
            'repeat_penalty': local_config.get('repeat_penalty')
        }

    def get_streaming_response_with_context(self, messages: list[Dict[str, Any]]) -> Generator[str, None, None]:
        """Get a streaming response from the LLM using conversation history.
        Yields:
            str: Chunks of the LLM's response as they are generated
        """
        # Log the generation parameters being used
        print(f"Using LLM parameters: {self.params}")
        
        # Call Ollama with our parameters
        response = ollama.chat(
            model=self.model_name, 
            messages=messages,
            stream=True,
            options=self.params
        )
        
        for chunk in response:
            if 'message' in chunk and 'content' in chunk['message']:
                yield chunk['message']['content']