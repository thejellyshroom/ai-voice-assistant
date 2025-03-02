import ollama
from typing import Dict, Any, Generator

class LLMHandler:
    def __init__(self, model_name='llama3.2'):
        self.model_name = model_name

    def get_response(self, prompt: str) -> str:
        """Get a response from the LLM for a single prompt.
        
        Args:
            prompt (str): The prompt to send to the LLM
            
        Returns:
            str: The LLM's response
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
        """Get a response from the LLM using conversation history.
        
        Args:
            messages (list): List of message dictionaries with 'role' and 'content'
            
        Returns:
            str: The LLM's response
        """
        response = ollama.chat(model=self.model_name, messages=messages)
        return response['message']['content']
    
    def get_streaming_response(self, prompt: str) -> Generator[str, None, None]:
        """Get a streaming response from the LLM for a single prompt.
        
        Args:
            prompt (str): The prompt to send to the LLM
            
        Yields:
            str: Chunks of the LLM's response as they are generated
        """
        response = ollama.chat(
            model=self.model_name,
            messages=[{
                'role': 'user',
                'content': prompt,
            }],
            stream=True
        )
        
        for chunk in response:
            if 'message' in chunk and 'content' in chunk['message']:
                yield chunk['message']['content']
    
    def get_streaming_response_with_context(self, messages: list[Dict[str, Any]]) -> Generator[str, None, None]:
        """Get a streaming response from the LLM using conversation history.
        
        Args:
            messages (list): List of message dictionaries with 'role' and 'content'
            
        Yields:
            str: Chunks of the LLM's response as they are generated
        """
        response = ollama.chat(
            model=self.model_name, 
            messages=messages,
            stream=True
        )
        
        for chunk in response:
            if 'message' in chunk and 'content' in chunk['message']:
                yield chunk['message']['content'] 