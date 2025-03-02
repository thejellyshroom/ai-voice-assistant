import ollama
from typing import Dict, Any, Generator

class LLMHandler:
    def __init__(self, model_name='llama3.2'):
        self.model_name = model_name
    
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