# routers/predict.py
from typing import List
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# === MODEL LOADING (Load once only) ===

model_path = "./models/phi3-mini-yoda-merged"

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True,
)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# === NL to FOL Translation ===

def nl_to_fol(sentence: str, max_new_tokens: int = 256) -> str:
    """Convert natural language sentence into First-Order Logic (FOL)"""
    inputs = tokenizer(
        sentence,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(model.device)

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=False,
            use_cache=False,
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded.strip()

# === TRANSLATION HELPERS ===

def translate_premises_to_fol(premises_nl: List[str]) -> List[str]:
    return [nl_to_fol(premise) for premise in premises_nl]

def translate_questions_to_fol(questions: List[str]) -> List[str]:
    return [nl_to_fol(question) for question in questions]

# === Z3 VERIFICATION ===

def verify_with_z3(fol_expression):
    # TODO: Z3 Solver
    return 

def generate_answer(z3_result):
    if z3_result == "SATISFIABLE":
        return "Yes"
    elif z3_result == "UNSATISFIABLE":
        return "No"
    else:
        return "Uncertain"

def get_premise_indices(premises, question):
    return [1]

def generate_explanation(premises, idx_list):
    return f"Premises {', '.join(str(i) for i in idx_list)} support the reasoning."
