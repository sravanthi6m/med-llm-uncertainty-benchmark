import torch
import torch.nn.functional as F
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LlamaForCausalLM,
    GenerationConfig,
    BitsAndBytesConfig
)
from tqdm import tqdm

def load_model(model_name: str):

    if "Qwen" in model_name or "internlm" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side='left')
    elif "SUS" in model_name or "Yi-34B-Chat" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, padding_side='left')
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side='left')

    tokenizer.truncation_side = "left"
    tokenizer.model_max_length = min(tokenizer.model_max_length, 2048)
    
    if any(k in model_name.lower() for k in ("gemma", "medgemma")):
        model_kwargs = {
            "attn_implementation": "eager"
        }
    else:
        model_kwargs = {
            "attn_implementation": "sdpa"
        }

    if any(sz in model_name.lower() for sz in ("27b", "32b")):
        print(f"Applying 4-bit quantization for large model {model_name}")
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=False
        )
        model_kwargs["quantization_config"] = quant_config
        model_kwargs["device_map"] = "auto"
    elif '70b' in model_name.lower() or '72b' in model_name.lower():
        print(f"Large model ({model_name}) detected. Applying 4-bit quantization.")

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model_kwargs["quantization_config"] = quantization_config
        
        model_kwargs["max_memory"] = {
            0: "38GiB",
            1: "38GiB",
            2: "38GiB",
            3: "38GiB",
            4: "38GiB",
            "cpu": "0GiB" 
        }    
    else:
        model_kwargs["device_map"] = "auto"

    if "falcon" in model_name or "deepseek" in model_name:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            **model_kwargs
        )
        if "deepseek" in model_name:
            model.generation_config = GenerationConfig.from_pretrained(model_name)
            model.generation_config.pad_token_id = model.generation_config.eos_token_id
    elif "llama-3" in model_name.lower() or "llama3" in model_name.lower():
        model = LlamaForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            trust_remote_code=True,
            **model_kwargs
        )
    elif "Llama" in model_name:
        model = LlamaForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            **model_kwargs
        )
    elif any(k in model_name for k in ("Mistral", "mpt", "Mixtral")):
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )
    elif any(k in model_name for k in ("Yi", "SUS")):
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype="auto",
            **model_kwargs
        )
    elif any(k in model_name.lower() for k in ("gemma", "medgemma")):
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            trust_remote_code=True,
            **model_kwargs
        )
        model.config.use_cache = False
    elif "phi" in model_name.lower():
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            trust_remote_code=True,
            **model_kwargs
        )
    elif "Qwen" in model_name:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            **model_kwargs
        )
    elif "internlm" in model_name:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            **model_kwargs
        )
    elif "COKAL" in model_name:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            return_dict=True,
            **model_kwargs
        )
    else:
        raise NotImplementedError(f"Model type {model_name} not supported.")
    model.eval()
    return tokenizer, model


def prepare_inputs(tokenizer, prompt: str, device: str="cuda"):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    for k in inputs:
        if torch.is_tensor(inputs[k]):
            inputs[k] = inputs[k].to(device)
    return inputs


def get_logits_for_options(model, tokenizer, prompt: str,
                           option_tokens: list[int]) -> np.ndarray:
    with torch.no_grad():
        out = model(**prepare_inputs(tokenizer, prompt))
    logits = out.logits[:, -1, :].squeeze(0)
    return logits[option_tokens].float().cpu().numpy()


def batched_logits(model, tokenizer, formatted_examples):
    outputs = []
    for ex in tqdm(formatted_examples):
        opt_keys = list(ex["choices"].keys())
        option_tokens = [tokenizer.encode(f"Answer: {k}", add_special_tokens=False)[-1]
                         for k in opt_keys]
        logits_opt = get_logits_for_options(model, tokenizer,
                                            ex["prompt"], option_tokens)
        outputs.append({
            "id": ex["id"],
            "logits_options": logits_opt,
            "option_keys_for_logits": opt_keys
        })
    return outputs

