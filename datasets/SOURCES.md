# Dataset Sources — MENTORA Phase 3

This document exists for two reasons: the thesis's data section, and being
able to answer "where did this data come from?" at the defense without
hesitating. Keep it updated every time a dataset changes.

## Status summary

| Model | Dataset | Status | Real source | Rows/count |
|---|---|---|---|---|
| M3 | `processed/m3/score_histories*.csv` | **Final** | Fully synthetic by design (per the master plan — no real users yet) | 10,639 rows / 800 students |
| M1 | `processed/m1/question_bank.csv` | **Real, web-sourced** | See "M1 real sources" below | 1,724 questions |
| M1 | `processed/m1/synthetic_answer_sessions.csv` | **Final structure, scales with the bank** | Synthetic sessions layered on the real question bank above (standard cold-start approach) | 10,935 rows / 499 students |
| M2 | `processed/m2/flan_t5_training_data.csv` | **Real — target met** | Derived entirely from M1's question bank | 1,724 pairs (target: 500+, met) |
| M4 | `processed/m4/career_profiles.csv` | Starter placeholder — replace with real O*NET join | Hand-authored, not sourced from O*NET (O*NET's download site wasn't reachable from the build environment used) | 50 careers |
| M4 | `processed/m4/training_pairs.csv` | Starter placeholder | Derived from the career_profiles.csv above, same limitation | 100 pairs |
| M5 | `labeled/m5/gold_val_test.jsonl` | Starter — needs scaling up with real resumes | 20 hand-authored resumes (not from Kaggle), auto-tagged with spaCy PhraseMatcher + regex, used directly as gold since entities were written unambiguously | 20 resumes |

## M1 real sources

M1's question bank (1,724 questions) was rebuilt from four real, licensed,
web-sourced datasets plus the original 70 hand-authored starter questions
(kept for continuity, tagged `source: hand_authored_starter`). Raw files
live in `datasets/raw/<name>/` exactly as downloaded; processing script is
`datasets/scripts/build_real_question_bank.py`.

### Mathematics — AQuA-RAT (594 questions)
- **Source:** [google-deepmind/AQuA](https://github.com/google-deepmind/AQuA) — ~100K crowdsourced
  algebraic word problems with 5-way multiple choice (A-E) and natural-language rationales.
- **License:** Apache License 2.0 (Copyright 2017 Google Inc.) — free to use, including commercially.
- **How it's used:** sampled 600 questions from `train.json`, classified into
  Algebra/Geometry/Trigonometry/Statistics/Calculus via keyword heuristics
  (defaults to Algebra, matching AQuA's actual content skew), difficulty
  approximated from rationale length (a proxy, not an authoritative
  difficulty label — AQuA doesn't provide one).
- **Note:** the M1 `Question` schema gained an `option_e` column
  specifically to fit AQuA's 5-way format; 4-option sources leave it null.

### Physics & Chemistry — science-questions CSV, ARC-derived (463 questions)
- **Source:** [joelgrus/science-questions](https://github.com/joelgrus/science-questions) —
  a CSV of real US state department of education released science exam items
  (elementary/middle school), the same underlying data as AI2's ARC dataset.
- **License:** ARC itself is CC BY-SA (Allen Institute for AI) — **requires
  attribution and share-alike** if redistributed. This file is a reformatting
  of publicly released state exam items; attribute as: "Questions derived
  from the AI2 Reasoning Challenge (ARC) dataset / state department of
  education released items, CC BY-SA."
- **How it's used:** the source CSV only tags coarse subjects (Science/
  Biology/Science and Technology), not Physics/Chemistry specifically, so
  `classify_science()` keyword-matches each question stem into our
  Physics (Mechanics, Electricity & Magnetism, Waves & Optics, Modern
  Physics) or Chemistry (Organic, Inorganic, Physical Chemistry) topics,
  **discarding** anything that doesn't clearly match (including all Biology
  — not in our taxonomy). This is why Chemistry skews heavily toward
  "Inorganic" (208 of 224) — the keyword list for Organic is narrow and the
  source pool has few organic-chemistry items at this grade level. Consider
  widening `CHEMISTRY_TOPIC_KEYWORDS["Organic"]` or sourcing a
  chemistry-specific dataset if that imbalance matters for training.

### English Reading Comprehension — RACE-C (247 questions)
- **Source:** [mrcdata/race-c](https://github.com/mrcdata/race-c) — real
  college-English-exam reading comprehension passages + MCQs from China,
  a supplement to the original RACE dataset (Lai et al., 2017).
- **License:** **"This dataset is intended for non-commercial research
  purpose only"** (per the repo's `license.txt`) — fine for this FYP/thesis,
  but flag clearly if MENTORA is ever commercialized: this subset would need
  to be replaced or a commercial-use license would need to be obtained.
- **How it's used:** sampled one question per passage (250 passages sampled,
  247 survived dedup) to keep passage repetition low; full passage stored in
  the new `passage` column, difficulty fixed at "hard" (all RACE-C content
  is college-level).

### English Grammar/Vocabulary/Writing — CLOTH (350 questions)
- **Source:** [zhaowei8188127/Large-scale-Cloze-Test-Dataset-Designed-by-Teachers](https://github.com/zhaowei8188127/Large-scale-Cloze-Test-Dataset-Designed-by-Teachers) —
  a GitHub mirror of CLOTH (Xie et al., 2018), 7,131 real cloze-test passages
  from Chinese middle/high school English exams, written by teachers
  specifically to test grammar, vocabulary, and reasoning.
- **License:** MIT (per the mirror repo) — free to use, including commercially.
- **How it's used:** one blank sampled per passage (350 sampled, all
  survived dedup). Each blank's correct-answer word is classified into
  Grammar (function-word POS tags: ADP/CCONJ/SCONJ/DET/PRON/AUX/PART, via
  spaCy) vs. Vocabulary (content words) vs. Writing (a small curated list of
  discourse connectives like "however", "therefore", "in addition") — this
  is a heuristic, not a ground-truth label, and it shows: Writing only
  caught 4 questions since discourse connectives are rare as cloze answers.
  If Writing coverage matters, consider a different heuristic or a
  dedicated source.

### Known imbalances to revisit before final training
- Chemistry: Organic (7) vs. Inorganic (208) — keyword list too narrow for Organic.
- Mathematics: Calculus (4) — AQuA is algebra-word-problem-heavy; barely any
  calculus content exists in the source to keyword-match.
- Physics: Modern Physics (8) — same issue, source skews toward Mechanics.
- English: Writing (4) — heuristic-classified from CLOTH, discourse
  connectives are rare among cloze answers.
- None of these are bugs — they're the real shape of the sourced data
  reported honestly. Widening keyword lists will over-classify (false
  positives); a dedicated source per thin topic is the more defensible fix
  before the thesis defense if asked about these gaps.

## M1 remaining considerations

- Difficulty labels are approximate/heuristic for every source except the
  hand-authored starter set (which was authored with difficulty in mind).
  Don't over-index on the exact easy/medium/hard split for the sourced data.
- RACE-C's non-commercial license is the one legal constraint to actually
  track — everything else here (Apache 2.0, MIT, CC BY-SA) permits
  commercial use with attribution.

## Fully final datasets (no further sourcing needed)

- **M3** — synthetic by explicit design per the master plan; nothing to
  replace here, ever (well before real student data exists, this stays the
  dataset; after real users exist, swap for a real `EXAM_SESSIONS` export
  using the same schema).
- **M2** — fully derived from M1, already past its 500+ target with M1's
  current size; will keep scaling automatically if M1 grows further.

## What's still starter/placeholder (M4, M5)

- **M4** career_profiles.csv should ideally come from a real O*NET join
  (Occupation Data + Skills flat files, script in the Phase 3 doc §3.2) for
  broader coverage and official data provenance. O*NET's own download site
  wasn't reachable from the build environment used for this pass — same
  constraint that originally applied to M1 before the web search above
  found workable GitHub-hosted alternatives. Worth trying the same
  approach (search for a GitHub-hosted O*NET mirror) before accepting this
  as permanently a placeholder.
- **M5** needs real resume text from a Kaggle resume dataset (§5.2) to reach
  a usable gold set size (target: 100+ hand-corrected resumes). Kaggle
  itself wasn't reachable; worth checking for a GitHub-hosted mirror the
  same way M1's sources were found.

## Reproducing everything

All scripts live in `datasets/scripts/` and are idempotent — re-run any of
them any time the upstream data changes:

```
generate_m3_synthetic.py           # standalone
build_real_question_bank.py        # standalone (re-downloads nothing; reads datasets/raw/*)
generate_m1_synthetic_sessions.py  # depends on m1 question_bank.csv
generate_m2_from_m1.py             # depends on m1 question_bank.csv
generate_m4_starter_careers.py     # standalone
generate_m5_starter_resumes.py     # depends on m4 career_profiles.csv (skill vocab)
check_consistency.py               # run last, validates everything above
```

Raw downloaded files live in `datasets/raw/<name>/` — `aqua/`, `science/`,
`race_c/`, `cloth/` — kept as-is (not re-processed in place) so
`build_real_question_bank.py` can be re-run any time without re-downloading.

## Storage note

Per the Phase 3 doc's storage rule: everything above should also be mirrored
into `mentora_data/` on Google Drive before Colab training starts (Phase 4),
since Colab's local disk is wiped on every session disconnect. This
`datasets/` folder in the repo is the source of truth; copy it to Drive
verbatim. Note: `datasets/raw/cloth/data.tar.gz` (~47MB) and its extracted
`data/` folder are the biggest items — fine for Drive, just don't expect a
fast `git clone` if this repo is cloned fresh from GitHub.
