import os
import torch
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

repo_id = 'microsoft/Phi-3-mini-4k-instruct'
model = AutoModelForCausalLM.from_pretrained(repo_id,
                                             device_map="auto",
)

print(model.get_memory_footprint()/1e6)

model = prepare_model_for_kbit_training(model)

config = LoraConfig(
    r=16,                   # the rank of the adapter, the lower the fewer parameters you'll need to train
    lora_alpha=32,         # multiplier, usually 2*r
    bias="none",           # BEWARE: training biases *modifies* base model's behavior
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
    # Newer models, such as Phi-3 at time of writing, may require
    # manually setting target modules
    target_modules=['o_proj', 'qkv_proj', 'gate_up_proj', 'down_proj'],
)

model = get_peft_model(model, config)

trainable_parms, tot_parms = model.get_nb_trainable_parameters()
print(f'Trainable parameters:             {trainable_parms/1e6:.2f}M')
print(f'Total parameters:                 {tot_parms/1e6:.2f}M')
print(f'Fraction of trainable parameters: {100*trainable_parms/tot_parms:.2f}%')

dataset = load_dataset("json", data_files="../data/train.json", split="train")

dataset = dataset.rename_column("input", "prompt")
dataset = dataset.rename_column("output", "completion")

messages = [
    {"role": "system", "content": "You are a helpful assistant that translates natural language sentences into First-Order Logic."},
    {"role": "user", "content": dataset[0]['prompt']},
    {"role": "assistant", "content": dataset[0]['completion']}
]

def format_dataset(examples):
    if isinstance(examples["prompt"], list):
        output_texts = []
        for i in range(len(examples["prompt"])):
            converted_sample = [
                {"role": "user", "content": examples["prompt"][i]},
                {"role": "assistant", "content": examples["completion"][i]},
            ]
            output_texts.append(converted_sample)
        return {'messages': output_texts}
    else:
        converted_sample = [
            {"role": "user", "content": examples["prompt"]},
            {"role": "assistant", "content": examples["completion"]},
        ]
        return {'messages': converted_sample}

dataset = dataset.map(format_dataset).remove_columns(['prompt', 'completion'])

tokenizer = AutoTokenizer.from_pretrained(repo_id)
tokenizer.chat_template

print(tokenizer.apply_chat_template(messages, tokenize=False))

tokenizer.pad_token = tokenizer.unk_token
tokenizer.pad_token_id = tokenizer.unk_token_id

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
    output_dir='../models/phi3-mini-yoda-adapter',
    report_to='none'
)

model.train()

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    args=sft_config,
    train_dataset=dataset,
)

dl = trainer.get_train_dataloader()
batch = next(iter(dl))

trainer.train()

def gen_prompt(tokenizer, sentence):
    converted_sample = [
        {"role": "user", "content": sentence},
    ]
    prompt = tokenizer.apply_chat_template(converted_sample,
                                           tokenize=False,
                                           add_generation_prompt=True)
    return prompt

sentence ="Translate into first-order logic: If every student has access to study materials, then there exists at least one student who is enrolled in a course."
prompt = gen_prompt(tokenizer, sentence)
print(prompt)

def generate(model, tokenizer, prompt, max_new_tokens=128, skip_special_tokens=False):
    tokenized_input = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(model.device)

    model.eval()
    generation_output = model.generate(**tokenized_input,
                                       eos_token_id=tokenizer.eos_token_id,
                                       max_new_tokens=max_new_tokens)

    output = tokenizer.batch_decode(generation_output,
                                    skip_special_tokens=skip_special_tokens)
    print(output)
    return output[0]

generate(model, tokenizer, prompt)

trainer.save_model('../models/local-phi3-mini-yoda-adapter')
tokenizer.save_pretrained('../models/local-phi3-mini-yoda-adapter')

os.listdir('../models/local-phi3-mini-yoda-adapter')