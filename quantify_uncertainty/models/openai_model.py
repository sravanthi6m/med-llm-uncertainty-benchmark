from openai import OpenAI, OpenAIError, RateLimitError, APIStatusError, APITimeoutError
from typing import List, Optional
from .base import BaseModel
import os
import time
import traceback

import numpy as np


class OpenAIModel(BaseModel):
    def __init__(self, model_name: str, api_key: Optional[str] = None):
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model_name = model_name

    def generate(
        self,
        prompt: str,
        choices: List[str],
        max_retries: int = 5,
        backoff_base: float = 2.0,
    ):
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=1,
                    logprobs=True,
                    top_logprobs=len(choices),
                )
                output = response.choices[0].message.content.strip()

                try:
                    top_logprobs = response.choices[0].logprobs.content
                    raw_logprobs = self.extract_raw_logprobs(top_logprobs, choices)
                except Exception as ex:
                    print("⚠️ Warning: failed to extract logprobs:", ex)
                    traceback.print_exc()
                    raw_logprobs = {c: None for c in choices}

                return {"output": output, "raw_logprobs": raw_logprobs}

            except (RateLimitError, OpenAIError, APITimeoutError) as e:
                wait_time = backoff_base**attempt
                print(
                    f"[Retry {attempt + 1}/{max_retries}] ⚠️ Retriable error occurred:"
                )
                print(f"  └─ Type: {type(e).__name__}")
                print(f"  └─ Message: {str(e)}")
                traceback.print_exc()
                time.sleep(wait_time)
            except Exception as e:
                print("🚨 Non-retryable error:")
                print(f"  └─ Type: {type(e).__name__}")
                print(f"  └─ Message: {str(e)}")
                traceback.print_exc()
                break

        raise RuntimeError(
            f"Failed to get response from OpenAI after {max_retries} attempts"
        )

    # TODO: consider moving this to utils.
    def extract_raw_logprobs(self, logprobs_list, valid_choices):
        if not logprobs_list:
            raise ValueError("logprobs_list is empty or None")

        top_logprobs = logprobs_list[0].top_logprobs

        # Get logprobs for valid options (e.g., A–E)
        raw_scores = {
            entry.token.strip(): entry.logprob
            for entry in top_logprobs
            if entry.token.strip() in valid_choices
        }

        # Pad missing options with low logprob
        min_logprob = min(raw_scores.values()) - 5 if raw_scores else -100
        for c in valid_choices:
            raw_scores.setdefault(c, min_logprob)

        return {c: float(raw_scores[c]) for c in valid_choices}

    def get_logits(self, prompt: str, choices: List[str]) -> Optional[List[float]]:
        return None
