from typing import List
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from .base import BaseModel
from ..utils import softmax


class OpenSourceHFModel(BaseModel):
    def __init__(self, model_name: str, max_length=2048):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.tokenizer.truncation_side = "left"
        self.tokenizer.model_max_length = min(
            self.tokenizer.model_max_length, max_length
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()

    def _prepare_inputs(self, prompt: str):
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        return {k: v.to("cuda") for k, v in inputs.items()}

    def generate(self, prompt: str) -> str:
        inputs = self._prepare_inputs(prompt)
        with torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=1)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    def get_logits(self, prompt: str, choices: List[str]) -> List[float]:
        inputs = self._prepare_inputs(prompt)
        with torch.no_grad():
            output = self.model(**inputs)
        logits = output.logits[:, -1, :].squeeze(0)  # TODO: this needs to be evaluated

        option_tokens = [
            self.tokenizer.encode(f"Answer: {k}", add_special_tokens=False)[-1]
            for k in choices
        ]
        return logits[option_tokens].float().cpu().numpy().tolist()
