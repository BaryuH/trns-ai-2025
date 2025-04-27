# TRNS-AI 2025 Challenge - Model Pipeline Documentation

## 🧠 Overview

This document outlines the end-to-end pipeline of our educational question-answering system for the TRNS-AI 2025 Challenge. The system is designed to reason over official rules and regulations, transforming natural language queries into formal logic for symbolic reasoning and explainable outputs.

---

## 📊 1. Data Pipeline

### 1.1 Input Format

- **Expected**: Natural language questions related to university rules/regulations.
- **Dataset**: May come in `.csv` or `.json`; format is yet to be confirmed.
- **Source**: Provided by challenge organizers; content includes official institutional rules and policy documents.

### 1.2 Preprocessing

Before being fed into the model:
- Text tokenization
- Named Entity Recognition (NER) (e.g., course names, professor titles, scores)
- Parsing for logical cues
- Sentence segmentation (for long queries with multiple conditions)
- Coreference resolution (e.g., "this course", "he", "they")
- (Optional) Categorization of question type (if applicable in future versions)

---

## 🔧 2. Model Architecture

### 2.1 Base Model

- **Model**: [`HuggingFaceH4/zephyr-7b-beta`](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta) (from HuggingFace)
- **Tuning**: LoRA on regulation-related queries and logic generation
- **Framework**: HuggingFace `transformers`

### 2.2 Fine-Tuning Details

- **Supervision**: TBD (dataset might be unsupervised; synthetic Q&A pairs may be generated)
- **Objective**: Teach the model to map natural language → formal logic expressions (primarily First-Order Logic)
- **Training Tools**: Transformers + Accelerate/Deepspeed (for speed and memory efficiency)

### 2.3. Logic Format conditions

To ensure consistency between the model and the symbolic engine:
- **Variables**: lowercase (`x`, `y`)
- **Predicates**: camelCase (`hasLabScore(x)`)
- **Constants**: PascalCase or quoted string (`"DSA101"`, `FallSemester`)
- **Sample Logic**:
  - Input: "If a student scores below 5, they must retake the course."
  - Logic: `∀x (student(x) ∧ score(x) < 5 → mustRetake(x))`

---

## ⚙️ 3. Symbolic Reasoning Module

### 3.1 Role of Z3

- The model outputs logical expressions from natural language input
- These expressions are passed into the **Z3 solver** to:
  - Validate rules
  - Derive answers (yes/no/condition-based)
  - Identify contradictions or satisfiability

### 3.2 Example Flow
- Natural Language → Logic Form (via LLM) → Z3 Execution → Output Interpretation → Final Answer

---

## 💡 4. Explainable AI (xAI)

### 4.1 Explanation Strategy

- Use Z3 output (yes/no) to generate rationale
- Include citations from the rules and regulations

### 4.2 Example Output

#### 🔹 Query:
This semester, I scored 8 points on the final exam for the DSA course. However, I was absent for the lab exam. Can I still get a B in this course?

#### 🔹 Answer:
**No.** Because you missed the lab exam, you received a score of 0 for lab work. According to **Regulation #13 of X University**, a student with 0 lab points cannot pass the course.

---

## 🧬 5. Pipeline Flow

Based on [Logic-LM architecture](https://www.researchgate.net/publication/370948908/figure/fig1/AS:11431281160644218@1684812333897/Overview-of-our-LOGIC-LM-framework.png):
![alt](https://www.researchgate.net/publication/370948908/figure/fig1/AS:11431281160644218@1684812333897/Overview-of-our-LOGIC-LM-framework.png)

---

## 🚀 6. Deployment

### 6.1 API

- Framework: **FastAPI** (planned)
- Route: `/query`
- Input: JSON with `{"question": "<your question>"}` format
- Output: JSON with `{"answer": "...", "explanation": "...", "rule_cited": "..."}`

### 6.2 Inference Strategy (TBD)

- GPU inference using `transformers`
- Options: batching vs real-time
- Consider memory + latency tradeoffs

---

## 📏 7. Evaluation Plan

### 7.1 Benchmark Strategy

- **Compare**:
  - Base LLM (no logic)
  - Fine-tuned LLM + Z3 pipeline

### 7.2 Metrics

- Accuracy on binary question-answering (yes/no)
- Rule citation accuracy
- Explainability quality (TBD — possibly human review)

---

## 🧰 8. Tools & Frameworks

- HuggingFace Transformers
- Z3 Solver
- FastAPI (for serving)
- PyTorch
- Optional: LangChain / OpenLLM (if needed for chaining components)

---

## 📌 Notes & TODOs

- [ ] Confirm dataset format and structure
- [ ] Prepare a synthetic Q&A set for fine-tuning
- [ ] Decide on logic formats beyond FOL (e.g. modal logic?)
- [ ] Design evaluation rubric for explanation quality
- [ ] Optimize inference for speed + calculation cost

---