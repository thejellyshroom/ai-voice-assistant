import ollama
from typing import Dict, Any

class LLMHandler:
    def __init__(self, model_name='llama3.2'):
        self.model_name = model_name

    def get_response(self, prompt: str) -> str:
        response = ollama.chat(
            model=self.model_name,
            messages=[{
                'role': 'user',
                'content': prompt,
            }]
        )
        return response['message']['content']

    def get_response_with_context(self, messages: list[Dict[str, Any]]) -> str:
        response = ollama.chat(model=self.model_name, messages=messages)
        return response['message']['content'] 