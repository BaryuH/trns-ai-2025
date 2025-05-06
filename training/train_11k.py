import os
import torch
import random
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

# Load model
repo_id = 'microsoft/Phi-3-mini-4k-instruct'
model = AutoModelForCausalLM.from_pretrained(repo_id, device_map="cuda")
print(model.get_memory_footprint() / 1e6)

# LoRA config
model = prepare_model_for_kbit_training(model)
config = LoraConfig(
    r=16,
    lora_alpha=32,
    bias="none",
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
    target_modules=['o_proj', 'qkv_proj', 'gate_up_proj', 'down_proj'],
)
model = get_peft_model(model, config)

# Print model size info
trainable_parms, tot_parms = model.get_nb_trainable_parameters()
print(f'Trainable parameters:             {trainable_parms / 1e6:.2f}M')
print(f'Total parameters:                 {tot_parms / 1e6:.2f}M')
print(f'Fraction of trainable parameters: {100 * trainable_parms / tot_parms:.2f}%')

# Load and preprocess dataset
raw_dataset = load_dataset("json", data_files={"train": "../data/train.json"}, split="train")
raw_dataset = raw_dataset.rename_column("input", "prompt")
raw_dataset = raw_dataset.rename_column("output", "completion")

def format_dataset(example):
    return {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that translates natural language sentences into First-Order Logic."},
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": example["completion"]},
        ]
    }

formatted_dataset = raw_dataset.map(format_dataset).remove_columns(['prompt', 'completion'])

# Add prompt variation
def add_variants(example):
    original_prompt = example["messages"][1]["content"]

    # Remove the fixed instruction prefix if it exists
    prefix = "Translate into first-order logic: "
    if original_prompt.startswith(prefix):
        core_sentence = original_prompt[len(prefix):]
    else:
        core_sentence = original_prompt

    # Add prompt variations
    variants = [
        f"Translate to FOL: {core_sentence}",
        f"Convert this sentence to formal logic: {core_sentence}",
        f"What is the first-order logic form of: {core_sentence}",
        f"Give me the FOL expression of: {core_sentence}",
        f"How would you write in FOL: {core_sentence}",
        f"Please express the following in first-order logic: {core_sentence}",
        f"Formalize this statement using FOL: {core_sentence}"
    ]

    example["messages"][1]["content"] = random.choice(variants)
    return example

# Train/validation split
split_dataset = formatted_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split_dataset["train"].map(add_variants)
eval_dataset = split_dataset["test"].map(add_variants)

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(repo_id)
tokenizer.pad_token = tokenizer.unk_token
tokenizer.pad_token_id = tokenizer.unk_token_id

# Optional: preview one chat template
messages = train_dataset[0]["messages"]
print(tokenizer.apply_chat_template(messages, tokenize=False))

# Training config
sft_config = SFTConfig(
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={'use_reentrant': False},
    gradient_accumulation_steps=1,
    per_device_train_batch_size=16,
    auto_find_batch_size=False,
    max_seq_length=256,
    packing=False,
    num_train_epochs=5,
    learning_rate=2e-4,
    lr_scheduler_type='cosine',
    warmup_steps=100,
    optim='adamw_torch_fused',
    logging_steps=10,
    logging_dir='../logs',
    output_dir='../models/phi3-mini-11k-adapter',
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    save_total_limit=2,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to='none'
)

# Trainer
model.train()
trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    args=sft_config,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# Optional: preview one batch
dl = trainer.get_train_dataloader()
batch = next(iter(dl))

# Start training
trainer.train()

# Save model
trainer.save_model('../models/local-phi3-11k-adapter')
tokenizer.save_pretrained('../models/local-phi3-11k-adapter')
print("✅ Model and tokenizer saved.")

# Generate helper
def gen_prompt(tokenizer, sentence):
    converted_sample = [
        {"role": "system", "content": "You are a helpful assistant that translates natural language sentences into First-Order Logic."},
        {"role": "user", "content": sentence}
    ]
    return tokenizer.apply_chat_template(converted_sample, tokenize=False, add_generation_prompt=True)

def generate(model, tokenizer, prompt, max_new_tokens=128, skip_special_tokens=False):
    tokenized_input = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(model.device)
    model.eval()
    generation_output = model.generate(**tokenized_input,
                                       eos_token_id=tokenizer.eos_token_id,
                                       max_new_tokens=max_new_tokens)
    output = tokenizer.batch_decode(generation_output, skip_special_tokens=skip_special_tokens)
    print(output)
    return output[0]

# Example test
sentence = "Translate into first-order logic: If every student has access to study materials, then there exists at least one student who is enrolled in a course."
prompt = gen_prompt(tokenizer, sentence)
generate(model, tokenizer, prompt)

# Check saved files
print(os.listdir('../models/local-phi3-11k-adapter'))
