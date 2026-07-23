"""
M1 dataset — synthetic (student, question, correct/incorrect) history,
built on top of the REAL question bank from generate_m1_starter_bank.py.
Per §4.5: the questions are real, the sessions are simulated — a
legitimate, common approach for cold-start classifiers with no real
user history yet.

Run: python3 generate_m1_synthetic_sessions.py
Output: ../processed/m1/synthetic_answer_sessions.csv
        ../processed/m1/weak_topic_labels.csv (aggregated multi-label targets)
"""
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "processed" / "m1"
questions = pd.read_csv(DATA_DIR / "question_bank.csv")
topics = questions["topic"].unique()

rng = np.random.default_rng(11)
sessions = []
for student_id in range(1, 500):
    topic_strength = {t: rng.uniform(0, 1) for t in topics}
    n_sample = min(rng.integers(15, 30), len(questions))
    sampled_qs = questions.sample(n=n_sample, random_state=student_id)
    for _, q in sampled_qs.iterrows():
        p_correct = topic_strength[q["topic"]]
        is_correct = bool(rng.uniform() < p_correct)
        sessions.append({
            "student_id": student_id,
            "question_id": q["question_id"],
            "topic": q["topic"],
            "is_correct": is_correct,
        })

sessions_df = pd.DataFrame(sessions)
sessions_df.to_csv(DATA_DIR / "synthetic_answer_sessions.csv", index=False)

# Aggregate to multi-label "weak_in_<topic>" targets (<50% correct = weak),
# per §4.5 — this is what actually feeds AutoModelForSequenceClassification.
pct_correct = (
    sessions_df.groupby(["student_id", "topic"])["is_correct"]
    .mean()
    .unstack()
)
weak_labels = (pct_correct < 0.5).astype(int)
weak_labels.columns = [f"weak_in_{t.lower().replace(' ', '_').replace('&', 'and')}" for t in weak_labels.columns]
weak_labels = weak_labels.reset_index()
weak_labels.to_csv(DATA_DIR / "weak_topic_labels.csv", index=False)

print(f"Wrote {len(sessions_df)} synthetic answer rows across {sessions_df['student_id'].nunique()} students")
print(f"-> {DATA_DIR / 'synthetic_answer_sessions.csv'}")
print("\nClass balance per weak-topic column (mean = fraction of students flagged weak):")
print(weak_labels.drop(columns=["student_id"]).mean())
print(f"\n-> {DATA_DIR / 'weak_topic_labels.csv'}")
