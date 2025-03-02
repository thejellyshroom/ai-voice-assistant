import ollama
from typing import Dict, Any

class LLMHandler:
    def __init__(self, model_name='llama3.2'):
        """Initialize LLM handler.
        
        Args:
            model_name (str): Name of the Ollama model to use
        """
        self.model_name = model_name

    def get_response(self, prompt: str) -> str:
        """Get response from the LLM.
        
        Args:
            prompt (str): Input text prompt
            
        Returns:
            str: Model's response
        """
        response = ollama.chat(
            model=self.model_name,
            messages=[{
                'role': 'user',
                'content': prompt,
            }]
        )
        return response['message']['content']

    def get_response_with_context(self, messages: list[Dict[str, Any]]) -> str:
        """Get response from the LLM with conversation context.
        
        Args:
            messages (list): List of message dictionaries with 'role' and 'content'
            
        Returns:
            str: Model's response
        """
        response = ollama.chat(model=self.model_name, messages=messages)
        return response['message']['content'] 