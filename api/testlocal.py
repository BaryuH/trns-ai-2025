from routers.predict import (
    translate_premises_to_fol,
    translate_questions_to_fol,
    verify_with_z3,
)

def main():
    print("=== Local FOL Verification Test ===\n")

    # Bước 1: Nhập premises
    premises = [
        "Students who have completed the core curriculum and passed the science assessment are qualified for advanced courses.",
        "Students who are qualified for advanced courses and have completed research methodology are eligible for the international program.",
        "Students who have passed the language proficiency exam are eligible for the international program.",
        "Students who are eligible for the international program and have completed a capstone project are awarded an honors diploma.",
        "Students who have been awarded an honors diploma and have completed community service qualify for the university scholarship.",
        "Students who have been awarded an honors diploma and have received a faculty recommendation qualify for the university scholarship.",
        "Sophia has completed the core curriculum.",
        "Sophia has passed the science assessment.",
        "Sophia has completed the research methodology course.",
        "Sophia has completed her capstone project.",
        "Sophia has completed the required community service hours."
    ]

    questions = [
        "Based on the above premises, which is the strongest conclusion?\nA. Sophia qualifies for the university scholarship\nB. Sophia needs a faculty recommendation to qualify for the scholarship\nC. Sophia is eligible for the international program\nD. Sophia needs to pass the language proficiency exam to get an honors diploma",
            "Does Sophia qualify for the university scholarship, according to the premises?"
    ]
    print("\nProcessing...\n")

    # Bước 3: Xử lý
    premises_fol = translate_premises_to_fol(premises)
    questions_fol = translate_questions_to_fol(questions)
    result = verify_with_z3(premises_fol, questions_fol)

    # Bước 4: In kết quả
    answers, idx_list = result

    print("=== RESULTS ===")
    for i, ans in enumerate(answers):
        print(f"Question {i+1}: Answer = {ans}")
        print(f"   Supported by Premises idx: {idx_list[i]}")
    print("================\n")


if __name__ == "__main__":
    main()