from typing import List, Optional
import anthropic


class AnthropicModel(BaseModel):
    def __init__(self, model_name: str, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model_name = model_name

    def generate(self, prompt: str) -> str:
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=1,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()

    def get_logits(self, prompt: str, choices: List[str]) -> Optional[List[float]]:
        return None
