"""
M2 dataset — Adaptive Question Generator. Reuses M1's tagged question
bank, reshaped into (prompt, target) pairs for FLAN-T5 fine-tuning, per
§6.2. No new sourcing needed — this is why M1's dataset work happens
before M2's.

Run: python3 generate_m2_from_m1.py
Output: ../processed/m2/flan_t5_training_data.csv
"""
import json
import pandas as pd
from pathlib import Path

M1_DIR = Path(__file__).parent.parent / "processed" / "m1"
OUT_DIR = Path(__file__).parent.parent / "processed" / "m2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

questions = pd.read_csv(M1_DIR / "question_bank.csv")


def to_flan_t5_example(row):
    subtopic = f" ({row['subtopic']})" if pd.notna(row.get("subtopic")) else ""
    prompt = f"Generate a {row['difficulty']} difficulty multiple-choice question about {row['topic']}{subtopic}."
    target = json.dumps({
        "question": row["question_text"],
        "options": {
            "A": row["option_a"], "B": row["option_b"],
            "C": row["option_c"], "D": row["option_d"],
        },
        "correct_answer": row["correct_answer"],
    })
    # sanity check from §6.4 — every target must be valid JSON
    json.loads(target)
    return pd.Series({"input_text": prompt, "target_text": target})


m2_data = questions.apply(to_flan_t5_example, axis=1)
m2_data.to_csv(OUT_DIR / "flan_t5_training_data.csv", index=False)

print(f"{len(m2_data)} prompt/target pairs -> {OUT_DIR / 'flan_t5_training_data.csv'}")
print("Target is 500+ per §6.4 — currently short because M1's starter bank is a")
print("starter set (70 questions). Scaling up M1 with real past-paper questions")
print("scales this file automatically on the next run.")

print("\nDifficulty spread:")
print(questions["difficulty"].value_counts())
