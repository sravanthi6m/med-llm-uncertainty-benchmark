from abc import ABC, abstractmethod
from typing import List, Optional


class BaseModel(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass

    @abstractmethod
    def get_logits(self, prompt: str, choices: List[str]) -> Optional[List[float]]:
        pass
