import openai
from typing import List, Optional
from .base import BaseModel
import os


class OpenAIModel(BaseModel):
    def __init__(self, model_name: str, api_key: Optional[str] = None):
        self.client = openai.OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model_name = model_name

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=1,
        )
        return response.choices[0].message.content.strip()

    def get_logits(self, prompt: str, choices: List[str]) -> Optional[List[float]]:
        return None
