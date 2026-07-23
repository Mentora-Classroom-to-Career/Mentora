# Dataset Sources — MENTORA Phase 3

This document exists for two reasons: the thesis's data section, and being
able to answer "where did this data come from?" at the defense without
hesitating. Keep it updated every time a dataset changes.

## Status summary

| Model | Dataset | Status | Real source | Rows/count |
|---|---|---|---|---|
| M3 | `processed/m3/score_histories*.csv` | **Final** | Fully synthetic by design (per the master plan — no real users yet) | 10,639 rows / 800 students |
| M1 | `processed/m1/question_bank.csv` | **Starter — needs scaling up** | 100% original, hand-authored for this project (not copied from any past paper or Kaggle set) | 70 questions |
| M1 | `processed/m1/synthetic_answer_sessions.csv` | **Final structure, scales with the bank** | Synthetic sessions layered on the real question bank above (standard cold-start approach) | 10,935 rows / 499 students |
| M2 | `processed/m2/flan_t5_training_data.csv` | **Starter — scales automatically with M1** | Derived entirely from M1's question bank | 70 pairs (target: 500+) |
| M4 | `processed/m4/career_profiles.csv` | **Starter placeholder — replace with real O\*NET join** | Hand-authored, not sourced from O\*NET (O\*NET's download site wasn't reachable from the build environment used) | 50 careers |
| M4 | `processed/m4/training_pairs.csv` | **Starter placeholder** | Derived from the career_profiles.csv above, same limitation | 100 pairs |
| M5 | `labeled/m5/gold_val_test.jsonl` | **Starter — needs scaling up with real resumes** | 20 hand-authored resumes (not from Kaggle), auto-tagged with spaCy PhraseMatcher + regex, used directly as gold since entities were written unambiguously | 20 resumes |

## What "starter" means and what to do about it before the defense

Several of these datasets are placeholders that make the full pipeline
(dataset -> Phase 4 training -> Phase 5 export -> Phase 6 inference) runnable
end-to-end today, but fall short of the master plan's target volumes:

- **M1** target is 150-300 questions **per subject** (600-1200 total); this
  starter set has ~70 total. Before Phase 4 training, add real past-paper
  questions per §4.2-4.3 of the Phase 3 doc (Sindh Board, FPSC, NTS official
  sites) using the same `question_bank.csv` schema — the pipeline scripts in
  `datasets/scripts/` don't need to change, just re-run `generate_m2_from_m1.py`
  and `generate_m1_synthetic_sessions.py` afterward to regenerate the
  downstream files.
- **M4** career_profiles.csv should ideally come from a real O\*NET join
  (Occupation Data + Skills flat files, script in the Phase 3 doc §3.2) for
  broader coverage and official data provenance. The hand-authored version
  here is defensible as "domain knowledge encoded for a starter set" but real
  O\*NET data is stronger for the thesis.
- **M5** needs real resume text from a Kaggle resume dataset (§5.2) to reach
  a usable gold set size (target: 100+ hand-corrected resumes). The 20 here
  are enough to validate the tagging pipeline works, not enough to train on
  alone.

## Fully final datasets (no further sourcing needed)

- **M3** — synthetic by explicit design per the master plan; nothing to
  replace here, ever (well before real student data exists, this stays the
  dataset; after real users exist, swap for a real `EXAM_SESSIONS` export
  using the same schema).

## Reproducing everything

All scripts live in `datasets/scripts/` and are idempotent — re-run any of
them any time the upstream data changes:

```
generate_m3_synthetic.py         # standalone
generate_m1_starter_bank.py      # standalone
generate_m1_synthetic_sessions.py  # depends on m1 question_bank.csv
generate_m2_from_m1.py           # depends on m1 question_bank.csv
generate_m4_starter_careers.py   # standalone
generate_m5_starter_resumes.py   # depends on m4 career_profiles.csv (skill vocab)
check_consistency.py             # run last, validates everything above
```

## Storage note

Per the Phase 3 doc's storage rule: everything above should also be mirrored
into `mentora_data/` on Google Drive before Colab training starts (Phase 4),
since Colab's local disk is wiped on every session disconnect. This
`datasets/` folder in the repo is the source of truth; copy it to Drive
verbatim.
