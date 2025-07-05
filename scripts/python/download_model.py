from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "deepseek-ai/DeepSeek-V3"
save_path = "/project/pi_hongyu_umass_edu/zonghai/abstention/sravanthi/benchmarking/models"
# Load model and tokenizer from Hugging Face
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

# Save locally
tokenizer.save_pretrained(save_path)
model.save_pretrained(save_path)
