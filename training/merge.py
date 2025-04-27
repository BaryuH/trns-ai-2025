# merge_lora_and_save.py

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import os

# ==== Cấu hình đường dẫn ====

base_model_path = "microsoft/Phi-3-mini-4k-instruct"  # Base model từ Huggingface
lora_adapter_path = "../models/local-phi3-mini-yoda-adapter"  # Folder chứa LoRA adapter
merged_output_path = "../models/phi3-mini-yoda-merged"        # Nơi lưu model đã merge

# ==== Load base model + tokenizer ====

print("[INFO] Loading base model...")
model = AutoModelForCausalLM.from_pretrained(base_model_path, trust_remote_code=True)

print("[INFO] Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

# ==== Load LoRA adapter và merge ====

print("[INFO] Loading LoRA adapter...")
model = PeftModel.from_pretrained(model, lora_adapter_path)

print("[INFO] Merging LoRA adapter into base model...")
model = model.merge_and_unload()

# ==== Save model đã merge ====

print(f"[INFO] Saving merged model to {merged_output_path}...")
os.makedirs(merged_output_path, exist_ok=True)
model.save_pretrained(merged_output_path)
tokenizer.save_pretrained(merged_output_path)

print("[SUCCESS] Model merged and saved successfully!")
