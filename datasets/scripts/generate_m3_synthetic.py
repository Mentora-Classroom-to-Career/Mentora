"""
M3 dataset — Performance Trajectory Predictor. Fully synthetic by design
(per MENTORA_Phase3_Datasets.md §2) since Mentora has no real users yet.
Models 4 trajectory archetypes so the LSTM sees real variety, not just
noise around a flat line.

Run: python3 generate_m3_synthetic.py
Output: ../processed/m3/score_histories.csv (labeled, for your own plots)
        ../processed/m3/score_histories_no_label.csv (unlabeled, for training)
"""
import numpy as np
import pandas as pd
from pathlib import Path

OUT_DIR = Path(__file__).parent.parent / "processed" / "m3"
OUT_DIR.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(42)
N_STUDENTS = 800
SESSIONS_PER_STUDENT = (8, 20)

rows = []
for student_id in range(1, N_STUDENTS + 1):
    n_sessions = rng.integers(*SESSIONS_PER_STUDENT)
    archetype = rng.choice(["improving", "declining", "plateau", "volatile"])
    base = rng.uniform(40, 70)

    scores = []
    for s in range(n_sessions):
        if archetype == "improving":
            trend = base + s * rng.uniform(1.0, 2.5)
        elif archetype == "declining":
            trend = base - s * rng.uniform(0.5, 1.5)
        elif archetype == "plateau":
            trend = base + min(s, 5) * rng.uniform(1.0, 2.0)
        else:  # volatile
            trend = base + rng.uniform(-15, 15)
        noise = rng.normal(0, 4)
        score = float(np.clip(trend + noise, 0, 100))
        scores.append(score)

    for s, score in enumerate(scores):
        rows.append({
            "student_id": student_id,
            "session_number": s + 1,
            "score": round(score, 1),
            "archetype": archetype,
        })

df = pd.DataFrame(rows)

# Validation checklist from §2.3
counts = df.groupby("archetype")["student_id"].nunique()
print("Students per archetype:\n", counts)
assert counts.min() >= counts.max() / 2, "Archetype imbalance too large"
assert df["score"].between(0, 100).all(), "Scores out of [0,100] range"

df = df.groupby("student_id").filter(lambda g: len(g) >= 5)

df.to_csv(OUT_DIR / "score_histories.csv", index=False)
df.drop(columns=["archetype"]).to_csv(OUT_DIR / "score_histories_no_label.csv", index=False)

print(f"\nWrote {len(df)} rows across {df['student_id'].nunique()} students")
print(f"-> {OUT_DIR / 'score_histories.csv'}")
print(f"-> {OUT_DIR / 'score_histories_no_label.csv'}")
