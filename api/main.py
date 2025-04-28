from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from typing import List

# Import functions từ predict.py
from routers.predict import (
    translate_premises_to_fol,
    translate_questions_to_fol,
    verify_with_z3
)

# === CONFIG ===
API_KEY = "uitsophosmind2025"

app = FastAPI()

api_key_header = APIKeyHeader(name="Authorization", auto_error=False)

# === Request and Response Models ===
class InputRequest(BaseModel):
    premises: List[str]
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

    # Step 1: Validate input
    if not input_request.premises or not input_request.questions:
        raise HTTPException(status_code=400, detail="Both 'premises' and 'questions' must be provided and non-empty.")

    # Step 2: Logging
    print("Incoming request:")
    print(f"Premises: {input_request.premises}")
    print(f"Questions: {input_request.questions}")
    print(f"Authorization: {authorization}")

    # Step 3: Translate NL to FOL
    premises_fol = translate_premises_to_fol(input_request.premises)
    questions_fol = translate_questions_to_fol(input_request.questions)

    result = verify_with_z3(premises_fol, questions_fol)
    answers = result[0]
    idx_list = result[1]

    explanations = ["Explanation not available yet." for _ in input_request.questions]

    # Step 4: Return result
    return OutputResponse(
        answers=answers,
        idx=idx_list,
        explanation=explanations
    )
