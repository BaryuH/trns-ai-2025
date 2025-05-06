# routers/predict.py
from typing import List, Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from z3 import *

# === MODEL LOADING (Load once only) ===

from routers.ParserAndSolver import (
    FOLLarkParser, FOLTransformer, EnhancedFOLChecker
)
model_path = "./models/phi3-mini-11k-merged"


model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="cuda",
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
    decoded.strip()

    input_len = len(sentence)
    fol_expression = decoded[input_len:].strip()

    return fol_expression


# === TRANSLATION HELPERS ===


def translate_ques_to_fol(questions: List[str]) -> List[str]:
    fol_expressions = []
    for premise in questions:
        prompt = f"Translate into first-order logic: {premise}"
        fol = nl_to_fol(prompt)
        i = 0
        while i < len(fol) and (fol[i] == ' ' or fol[i] == '.'):
            i += 1
        fol = fol[i:]
        fol_expressions.append(fol)
    return fol_expressions

# This function is used to extract options from the question string.


def get_options(questions: List[str]) -> List[List[str]]:
    result = []
    for question in questions:
        if "\n" in question:
            lines = question.split("\n")
            options = []
            for i in range(1, len(lines)):
                options.append(lines[i][3:])
                # print(lines[i][2:])
            result.append(options)
        else:
            options = []
            options.append(question)
            result.append(options)
    return result


def translate_questions_to_fol(questions: List[str]) -> List[List[str]]:
    options = get_options(questions)
    new_options = []
    for option in options:
        temp = translate_ques_to_fol(option)
        new_options.append(temp)
    return new_options


def translate_premises_to_fol(premises_nl: List[str]) -> List[str]:
    """Translate natural language premises into clean FOL expressions."""
    fol_expressions = []
    for premise in premises_nl:
        prompt = f"Translate into first-order logic: {premise}"
        fol = nl_to_fol(prompt)
        fol_expressions.append(fol)
    return fol_expressions


# === Z3 VERIFICATION ===

def verify_with_z3(premises_fol: List[str], question_fol: List[List[str]]):
    parser = FOLLarkParser()
    answers = []
    idx_list = []
    explanations = []

    # Parsed premises to Z3
    for i, premise in enumerate(premises_fol):
        parser.add_assertion(premise)

    for i in range(len(question_fol)):
        options = question_fol[i]
        if (len(options) > 1):
            entailed_options = []
            sat_options = []
            core_options = []
            option_labels = ["A", "B", "C", "D",
                             "E", "F", "G", "H"][:len(options)]
            for j, option in enumerate(options):
                is_entailed, model, explanation = parser.check_entailment(
                    option)
                if is_entailed:
                    entailed_options.append(option_labels[j])
                    _, core_indices, core_explanation = parser.get_unsat_core(
                        option)
                    core_options.append(sorted([x + 1 for x in core_indices]))
                    sat_options.append(option_labels[j])
            maxx = 0
            idx_max = 0
            for i in range(len(core_options)):
                if len(core_options[i]) > maxx:
                    idx_max = i
                    maxx = len(core_options[i])
            if maxx == 0:
                answers.append("A")
                idx_list.append([i + 1 for i in range(len(premises_fol))])
            answers.append(sat_options[idx_max])
            idx_list.append(core_options[idx_max])
        else:
            # Yes/No question
            option = options[0]
            is_entailed, model, explanation = parser.check_entailment(option)
            if is_entailed:
                answers.append("Yes")
                _, core_indices, core_explanation = parser.get_unsat_core(
                    option)
                idx_list.append(sorted([x + 1 for x in core_indices]))
            else:
                answers.append("No")
                temp_options = "¬(" + option + ")"
                is_entailed2, model, explanation = parser.check_entailment(
                    temp_options)
                if is_entailed2:
                    _, core_indices, core_explanation = parser.get_unsat_core(
                        temp_options)
                    idx_list.append(sorted([x + 1 for x in core_indices]))
                else:
                    idx_list.append([i + 1 for i in range(len(premises_fol))])
    result = []
    result.append(answers)
    result.append(idx_list)
    return result


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