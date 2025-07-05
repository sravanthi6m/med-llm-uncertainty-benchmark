from typing import List
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from .base import BaseModel
from ..utils import softmax
import torch.nn.functional as F

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

    # def generate(self, prompt: str, choices: List[str]) -> str:
    #     inputs = self._prepare_inputs(prompt)
    #     with torch.no_grad():
    #         output = self.model.generate(**inputs, max_new_tokens=1) #TODO: this would need to change for later.
    #     return self.tokenizer.decode(output[0], skip_special_tokens=True)

    def generate(self, prompt: str, choices: List[str]):
        inputs = self._prepare_inputs(prompt)
        with torch.no_grad():
            output_tokens = self.model.generate(
                **inputs,
                max_new_tokens=1,
                return_dict_in_generate=True,
                output_scores=True
            )
        # print(f"output_tokens is {output_tokens}")
        # print(self.tokenizer.decode(output_tokens.sequences[0]))
        # print(self.tokenizer.convert_ids_to_tokens(output_tokens.sequences[0]))

        # Decode the generated text
        output_text = self.tokenizer.decode(output_tokens.sequences[0], skip_special_tokens=True)

        # Extract logits of the generated token
        logits = output_tokens.scores[-1].squeeze(0)  # [vocab_size]
        logprobs = F.log_softmax(logits, dim=-1)
        # print(f"logits are: {logits}")
        # print(f"logprobs are: {logprobs}")

        # final_tokens_test = output_tokens.sequences[0][-1].item()
        # predicted_token_str = self.tokenizer.decode([final_tokens_test])

        # print(f"Predicted token ID: {final_tokens_test}")
        # print(f"Predicted token string: '{predicted_token_str}'")

        # Convert to logprobs for the choices
        choice_token_map = {
            c: self.tokenizer.encode(c, add_special_tokens=False)
            for c in choices
        }

        for c, token_ids in choice_token_map.items():
            if len(token_ids) > 1:
                print(f"⚠️ Warning: Choice '{c}' tokenized into multiple tokens: {token_ids}")

        # If you still want the first token only for logprob lookup:
        choice_tokens = {c: token_ids[0] for c, token_ids in choice_token_map.items()}
        print(f"choice_token_map: {choice_token_map}")

        # Use the first token ID for logprob lookup
        selected_logprobs = {
            c: float(logprobs[token_ids[0]].item())
            for c, token_ids in choice_token_map.items()
        }
        print(f"selected_logprobs are: {selected_logprobs}")
        return {
            "output": output_text.strip(),
            "raw_logprobs": selected_logprobs, #TODO: update this
            "option_keys_for_logits": choices
        }

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
