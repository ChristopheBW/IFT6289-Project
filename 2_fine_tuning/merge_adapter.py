import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

# Configuration
base_model_id = "meta-llama/Llama-3.2-3B-Instruct"
adapter_path = "adapter/llama3.2-culture-adapters"
merged_model_path = "2_fine_tuning/merged_llama32_culture"

# Load Base Model
print(f"Loading base model: {base_model_id}")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

print(f"Loading tokenizer: {base_model_id}")
tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True, legacy=False)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load PEFT Model (Apply Adapters)
print(f"Loading PEFT adapters from: {adapter_path}")
model = PeftModel.from_pretrained(base_model, adapter_path)
print("Adapters loaded.")

# Merge Adapters
print("Merging adapters...")
model = model.merge_and_unload()
print("Merging complete.")

# Save Merged Model
print(f"Saving merged model to: {merged_model_path}")
os.makedirs(merged_model_path, exist_ok=True)
model.save_pretrained(merged_model_path)
tokenizer.save_pretrained(merged_model_path)
print("Merged model and tokenizer saved.")

# Clean up (Optional)
del model
del base_model
torch.cuda.empty_cache()
print("Cleanup complete.")

