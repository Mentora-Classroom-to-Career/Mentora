---
noteId: "7bad98d086d011f18cf0e7d5b31b454b"
tags: []

---

# Dataset Sources — MENTORA Phase 3

This document exists for two reasons: the thesis's data section, and being
able to answer "where did this data come from?" at the defense without
hesitating. Keep it updated every time a dataset changes.

## Status summary

| Model | Dataset | Status | Real source | Rows/count |
|---|---|---|---|---|
| M3 | `processed/m3/score_histories*.csv` | **Final** | Fully synthetic by design (per the master plan — no real users yet) | 10,639 rows / 800 students |
| M1 | `processed/m1/question_bank.csv` | **Real, web-sourced** | See "M1 real sources" below | 6,942 questions |
| M1 | `processed/m1/synthetic_answer_sessions.csv` | **Final structure, scales with the bank** | Synthetic sessions layered on the real question bank above (standard cold-start approach) | 10,935 rows / 499 students |
| M2 | `processed/m2/flan_t5_training_data.csv` | **Real — target met** | Derived entirely from M1's question bank | 6,942 pairs (target: 500+, met) |
| M4 | `processed/m4/career_profiles.csv` | Starter placeholder — replace with real O*NET join | Hand-authored, not sourced from O*NET (O*NET's download site wasn't reachable from the build environment used) | 50 careers |
| M4 | `processed/m4/training_pairs.csv` | Starter placeholder | Derived from the career_profiles.csv above, same limitation | 100 pairs |
| M5 | `labeled/m5/gold_val_test.jsonl` | Starter — needs scaling up with real resumes | 20 hand-authored resumes (not from Kaggle), auto-tagged with spaCy PhraseMatcher + regex, used directly as gold since entities were written unambiguously | 20 resumes |

## M1 real sources

M1's question bank (6,942 questions) was rebuilt from four real, licensed,
web-sourced datasets plus the original 70 hand-authored starter questions
(kept for continuity, tagged `source: hand_authored_starter`). Raw files
live in `datasets/raw/<name>/` exactly as downloaded; processing script is
`datasets/scripts/build_real_question_bank.py`.

This went through two passes: an initial pass sampled 600-350 questions per
source, which trained cleanly but left 10 topics with under-12-example
support and exactly 0.0 F1 on each (see a full real training run's
diagnosis in `training/README.md`). Since the per-topic breakdown showed
those topics needed volume, not a different training approach, the second
pass increased sampling (AQuA 600->4000, CLOTH 350->2000, RACE-C 250->500)
and widened the keyword classification lists — both pull MORE of what's
already in these real, already-vetted sources, rather than introducing new
ones. All the counts below are post-second-pass.

### Mathematics — AQuA-RAT (3,926 questions)
- **Source:** [google-deepmind/AQuA](https://github.com/google-deepmind/AQuA) — ~100K crowdsourced
  algebraic word problems with 5-way multiple choice (A-E) and natural-language rationales.
- **License:** Apache License 2.0 (Copyright 2017 Google Inc.) — free to use, including commercially.
- **How it's used:** sampled 4,000 questions from `train.json` (up from 600),
  classified into Algebra/Geometry/Trigonometry/Statistics/Calculus via
  keyword heuristics (defaults to Algebra), difficulty approximated from
  rationale length (a proxy, not an authoritative difficulty label — AQuA
  doesn't provide one).
- **Per-topic counts:** Algebra 2,690, Statistics 903, Geometry 264,
  Trigonometry 63, **Calculus 6**. Calculus stays thin even at 4,000
  samples — AQuA is genuinely an algebra-word-problem dataset with almost
  no calculus content to keyword-match, not a sampling shortfall. A
  calculus-specific source would be needed to fix this one.
- **Note:** the M1 `Question` schema gained an `option_e` column
  specifically to fit AQuA's 5-way format; 4-option sources leave it null.

### Physics & Chemistry — science-questions CSV, ARC-derived (473 questions)
- **Source:** [joelgrus/science-questions](https://github.com/joelgrus/science-questions) —
  a CSV of real US state department of education released science exam items
  (elementary/middle school), the same underlying data as AI2's ARC dataset.
- **License:** ARC itself is CC BY-SA (Allen Institute for AI) — **requires
  attribution and share-alike** if redistributed. Attribute as: "Questions
  derived from the AI2 Reasoning Challenge (ARC) dataset / state department
  of education released items, CC BY-SA."
- **How it's used:** the source CSV only tags coarse subjects (Science/
  Biology/Science and Technology), not Physics/Chemistry specifically, so
  `classify_science()` keyword-matches each question stem into our topics,
  **discarding** anything that doesn't clearly match (including all Biology
  — not in our taxonomy). Keyword lists were widened in the second pass
  (added isotope/half-life/fission/fusion for Modern Physics; polymer/
  combustion/protein/fat etc. for Organic).
- **Per-topic counts:** Physics — Mechanics 132, Waves & Optics 69,
  Electricity & Magnetism 49, **Modern Physics 8**. Chemistry — Inorganic
  205, **Organic 17**, Physical Chemistry 23. Modern Physics and Organic
  stay thin even after widening keywords — this grade-school-level source
  simply has almost no nuclear-physics or organic-chemistry content. A
  dedicated source would be needed to fix these two specifically.

### English Reading Comprehension — RACE-C (496 questions)
- **Source:** [mrcdata/race-c](https://github.com/mrcdata/race-c) — real
  college-English-exam reading comprehension passages + MCQs from China,
  a supplement to the original RACE dataset (Lai et al., 2017).
- **License:** **"This dataset is intended for non-commercial research
  purpose only"** (per the repo's `license.txt`) — fine for this FYP/thesis,
  but flag clearly if MENTORA is ever commercialized.
- **How it's used:** sampled one question per passage (500 passages sampled,
  up from 250); full passage stored in the `passage` column, difficulty
  fixed at "hard" (all RACE-C content is college-level).

### English Grammar/Vocabulary/Writing — CLOTH (2,000 questions)
- **Source:** [zhaowei8188127/Large-scale-Cloze-Test-Dataset-Designed-by-Teachers](https://github.com/zhaowei8188127/Large-scale-Cloze-Test-Dataset-Designed-by-Teachers) —
  a GitHub mirror of CLOTH (Xie et al., 2018), 7,131 real cloze-test passages
  from Chinese middle/high school English exams, written by teachers
  specifically to test grammar, vocabulary, and reasoning.
- **License:** MIT (per the mirror repo) — free to use, including commercially.
- **How it's used:** one blank sampled per passage (2,000 sampled, up from
  350 — still under a third of the 7,131 available passages). Each blank's
  correct-answer word is classified into Grammar (function-word POS tags,
  via spaCy) vs. Vocabulary (content words) vs. Writing (a curated list of
  discourse connectives, widened in the second pass to include sequencing
  words like "first"/"next"/"finally").
- **Per-topic counts:** Vocabulary 1,794 (now the dominant English topic —
  a new imbalance in the OTHER direction, worth watching), Grammar 187,
  Writing 32, Reading Comprehension 500 (from RACE-C above).

### Remaining genuinely thin topics
Two passes of real sourcing (initial + wider sampling/keywords) converged
on the same three topics staying thin no matter how hard the existing
sources are mined:
- **Mathematics: Calculus (6)** — AQuA has almost no calculus word problems.
- **Physics: Modern Physics (8)** — grade-school ARC-derived source has
  almost no nuclear/particle physics content.
- **Chemistry: Organic (17)** — same source, thin on organic chemistry.

These are structural gaps in the sourced datasets, not sampling or keyword
issues — confirmed by widening both and seeing counts barely move. Closing
them requires a dedicated source per topic (e.g. an AP Calculus or AP
Chemistry MCQ set), which is a reasonable next step if these three specific
topics matter for the final model, but isn't a quick fix like the other
seven were.

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
