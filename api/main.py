from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from typing import List

# Import functions từ predict.py
from routers.predict import (
    translate_premises_to_fol,
    translate_questions_to_fol,
    verify_with_z3,
    generate_answer,
    get_premise_indices,
    generate_explanation
)

# === CONFIG ===
API_KEY = "uitsophosmind2025"

app = FastAPI()

api_key_header = APIKeyHeader(name="Authorization", auto_error=False)

# === Request and Response Models ===
class InputRequest(BaseModel):
    premises_NL: List[str]
    questions: List[str]

class OutputResponse(BaseModel):
    answers: List[str]
    idx: List[List[int]]
    explanation: List[str]

# === API Endpoint ===
@app.post("/query", response_model=OutputResponse)
def query(input_request: InputRequest, authorization: str = Depends(api_key_header)):
    # Step 0: Check API Key
    if authorization is None:
        raise HTTPException(status_code=401, detail="Authorization header missing.")
    if authorization != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="Invalid API Key.")

    # Step 1: Translate NL to FOL
    premises_fol = translate_premises_to_fol(input_request.premises_NL)
    questions_fol = translate_questions_to_fol(input_request.questions)

    answers = []
    idx_list = []
    explanations = []

    # Step 2: Solve each question
    for i, question_fol in enumerate(questions_fol):
        z3_result = verify_with_z3(premises_fol, question_fol)
        answer = generate_answer(z3_result)
        idx = get_premise_indices(input_request.premises_NL, input_request.questions[i])
        explanation = generate_explanation(input_request.premises_NL, idx)

        answers.append(answer)
        idx_list.append(idx)
        explanations.append(explanation)

    # Step 3: Return result
    return OutputResponse(
        answers=answers,
        idx=idx_list,
        explanation=explanations
    )
