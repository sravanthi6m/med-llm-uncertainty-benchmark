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
    quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
    
    if "Qwen" in model_name or "internlm" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side='left')
    elif "SUS" in model_name or "Yi-34B-Chat" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, padding_side='left')
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side='left')

    tokenizer.truncation_side = "left"
    tokenizer.model_max_length = min(tokenizer.model_max_length, 2048)

    if "falcon" in model_name or "deepseek" in model_name:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="sdpa"
            )
        if "deepseek" in model_name:
            model.generation_config = GenerationConfig.from_pretrained(model_name)
            model.generation_config.pad_token_id = model.generation_config.eos_token_id
    elif "Llama" in model_name:
        model = LlamaForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="sdpa",
            quantization_config=quantization_config
        )
    elif any(k in model_name for k in ("Mistral", "mpt", "Mixtral")):
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            attn_implementation="sdpa"
        )
    elif any(k in model_name for k in ("Yi", "SUS")):
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            device_map="auto", 
            torch_dtype="auto",
            attn_implementation="sdpa"
        )
    elif any(k in model_name.lower() for k in ("gemma", "medgemma")):
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="sdpa"
        )
    elif "phi" in model_name.lower():
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype="auto",
            trust_remote_code=True,
            attn_implementation="sdpa"
        )
    elif "Qwen" in model_name:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            attn_implementation="sdpa",
            quantization_config=quantization_config
        )
    elif "internlm" in model_name:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            attn_implementation="sdpa"
        )
    elif "COKAL" in model_name:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            return_dict=True,
            attn_implementation="sdpa"
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

