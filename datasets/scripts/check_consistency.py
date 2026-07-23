"""
Cross-dataset consistency check — Phase 3 §7. Confirms subject names
match exactly, character-for-character, between M1's question bank and
the frontend's hardcoded subject chips (exam-prep page + ERD's
TOPIC_SCORES.subject column).

Run: python3 check_consistency.py
"""
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "processed"

FRONTEND_SUBJECTS = {"Mathematics", "English", "Physics", "Chemistry"}
dataset_subjects = set(pd.read_csv(DATA_DIR / "m1" / "question_bank.csv")["subject"].unique())

mismatch = dataset_subjects - FRONTEND_SUBJECTS
assert not mismatch, f"Mismatch: {mismatch}"
print(f"OK — M1 subjects {sorted(dataset_subjects)} all match the frontend's subject chips.")

# also confirm every M2 target is valid JSON (redundant with generate_m2's own
# check, but cheap and worth re-verifying independently here)
import json
m2 = pd.read_csv(DATA_DIR / "m2" / "flan_t5_training_data.csv")
bad = 0
for t in m2["target_text"]:
    try:
        json.loads(t)
    except json.JSONDecodeError:
        bad += 1
assert bad == 0, f"{bad} malformed target_text rows in M2 data"
print(f"OK — all {len(m2)} M2 target_text rows are valid JSON.")

print("\nAll cross-dataset consistency checks passed.")
