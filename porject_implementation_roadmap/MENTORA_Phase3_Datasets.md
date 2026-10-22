# MENTORA — Phase 3 Deep-Dive: Datasets

*This phase can run fully in parallel with Phase 1–2 (backend/frontend wiring) —
different skillset, no dependency between them. It must finish before Phase 4
(Colab training) can start for a given model, so treat it as the long pole: start it
early, even before Phase 2 is finished.*

---

## 0. Goal, scope, and the storage rule that saves you later

**Goal:** by the end of this phase, you have five clean, versioned, correctly-shaped
datasets sitting in Google Drive — one per model — each ready to be loaded into a
Colab notebook with a single `pd.read_csv()` or `datasets.load_dataset()` call, with
no further cleaning needed inside the training notebook.

**The one rule that matters most:** **every raw file and every processed file goes
into Google Drive, never Colab's local/temp disk.** Colab's local disk is wiped on
every session disconnect. This has caused real, avoidable data loss in FYPs before —
don't let it happen to you. Set this up in the very first Colab cell you ever run:

```python
from google.colab import drive
drive.mount('/content/drive')

import os
BASE = '/content/drive/MyDrive/mentora_data'
for sub in ['raw', 'processed', 'labeled']:
    os.makedirs(f'{BASE}/{sub}', exist_ok=True)
```

**Folder convention used throughout this document:**
```
mentora_data/
├── raw/            # exactly as downloaded, untouched, never edited in place
├── processed/       # cleaned CSV/JSON, one subfolder per model (m1/, m2/, m3/, m4/, m5/)
└── labeled/          # only for datasets requiring manual/semi-manual labeling (m5)
```

---

## 1. Overview — what data feeds which model

| Model | Needs | Format it must end in |
|---|---|---|
| M1 — Knowledge Gap Classifier | MCQ questions labeled by topic, plus (student, question, correct/incorrect) history | CSV: `question_id, question_text, topic, subtopic, difficulty` + a separate answers table |
| M2 — Adaptive Question Generator | MCQ bank with topic + difficulty tags, used as generation targets for fine-tuning FLAN-T5 | CSV/JSONL: `topic, difficulty, question_text, options[4], correct_answer` |
| M3 — Performance Trajectory Predictor | Per-student sequences of session scores over time | CSV: `student_id, session_number, date, score` (synthetic is acceptable) |
| M4 — Career-Skill Semantic Matcher | Career profiles (title, required skills) + a way to pair them with matching/non-matching skill sets | CSV: `career_id, title, required_skills[], description` + a labeled pairs file for fine-tuning |
| M5 — CV/Transcript NER Extractor | Resume text with BIO-tagged entities (SKILL, CERT, GPA, DEGREE, ORG) | CoNLL/JSONL: token-level BIO tags |

Sections 2–6 below go through each of these one at a time, in the same priority
order as Phase 4's training order (M3 → M4 → M1 → M5 → M2), since it's motivating to
have the easiest dataset ready first and get a model training while the harder
datasets are still being assembled.

---

## 2. M3 dataset — Performance Trajectory Predictor (do this first, easiest)

### 2.1 What "real" data would look like, and why synthetic is fine here
Ideally: real per-student score histories across many exam sessions. You don't have
this yet — Mentora hasn't had real users. **Synthetic data is explicitly acceptable
for M3** (per the master plan) because the model just needs to learn the general
*shape* of score trajectories (rising, falling, plateauing, noisy-but-flat) — it
doesn't need to memorize real students.

### 2.2 Step-by-step: generate a realistic synthetic dataset
```python
# generate_m3_synthetic.py — run locally or in a Colab cell, save straight to Drive
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N_STUDENTS = 800
SESSIONS_PER_STUDENT = (8, 20)  # min, max — varies per student, like real usage

rows = []
for student_id in range(1, N_STUDENTS + 1):
    n_sessions = rng.integers(*SESSIONS_PER_STUDENT)
    # Pick one of 4 trajectory archetypes so the model sees real variety,
    # not just noise around a flat line
    archetype = rng.choice(["improving", "declining", "plateau", "volatile"])
    base = rng.uniform(40, 70)

    scores = []
    for s in range(n_sessions):
        if archetype == "improving":
            trend = base + s * rng.uniform(1.0, 2.5)
        elif archetype == "declining":
            trend = base - s * rng.uniform(0.5, 1.5)
        elif archetype == "plateau":
            trend = base + min(s, 5) * rng.uniform(1.0, 2.0)  # rises then flattens
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
            "archetype": archetype,  # keep this column for your own validation,
                                       # drop it before feeding the model if it
                                       # shouldn't see the "answer key"
        })

df = pd.DataFrame(rows)
df.to_csv("/content/drive/MyDrive/mentora_data/processed/m3/score_histories.csv", index=False)
print(df.groupby("archetype")["student_id"].nunique())  # sanity check: ~200 per archetype
```

### 2.3 Validation checklist before training
- [ ] Roughly equal representation of all 4 archetypes (print the `groupby` above —
      no archetype should be less than half the size of the largest)
- [ ] Scores stay within `[0, 100]` (the `np.clip` above guarantees this)
- [ ] No student has fewer than 5 sessions (LSTM needs a minimum sequence length to
      learn anything — filter out anyone under that with `df.groupby("student_id").filter(lambda g: len(g) >= 5)`)
- [ ] Save a second copy with the `archetype` column dropped
      (`processed/m3/score_histories_no_label.csv`) — use *this* file for training
      the actual LSTM sequence input, keep the labeled version only for your own
      eyeballing/plots in the thesis.

### 2.4 A note for later (not blocking, but worth planning)
Once the platform has real users (post-defense, or late in the FYP if time allows),
swap this file for a real export from `EXAM_SESSIONS` — the schema is designed to
match, so no code changes needed beyond pointing at a different CSV.

---

## 3. M4 dataset — Career-Skill Semantic Matcher

### 3.1 What you need, in two parts
1. **Career profiles**: ~50+ career titles, each with a required-skills list and a
   short description. This is what students get matched *against*.
2. **Training pairs**: (student-skill-set, career) pairs labeled as
   matching/non-matching, so the Sentence-Transformer can be fine-tuned with
   `CosineSimilarityLoss` (per the tech doc's training script).

### 3.2 Step-by-step: build the career profiles table
**Source: O*NET** (the U.S. Department of Labor's free, public occupational
database — no API key, no cost, no scraping-policy concerns since it's a
government open-data resource intended for exactly this kind of reuse).

1. Go to O*NET's database download page and get the **"Skills"** and
   **"Occupation Data"** flat files (tab-delimited `.txt`, no signup required).
2. Load and join them:
```python
import pandas as pd

occ = pd.read_csv("raw/onet/Occupation Data.txt", sep="\t")
skills = pd.read_csv("raw/onet/Skills.txt", sep="\t")

# O*NET rates every skill 0–7 for every occupation — keep only the skills that
# actually matter for each job (importance score, "IM" scale, >= 3.5 is "relevant")
skills_relevant = skills[(skills["Scale ID"] == "IM") & (skills["Data Value"] >= 3.5)]

career_skills = (
    skills_relevant.merge(occ, on="O*NET-SOC Code")
    .groupby(["O*NET-SOC Code", "Title"])["Element Name"]
    .apply(list)
    .reset_index()
    .rename(columns={"Element Name": "required_skills", "Title": "title"})
)
career_skills["career_id"] = range(1, len(career_skills) + 1)
career_skills.to_csv("processed/m4/career_profiles.csv", index=False)
print(f"{len(career_skills)} career profiles built")  # should be in the hundreds — trim to your top 50–80 relevant to CS/tech/business students
```
3. **Filter down to ~50–80 careers actually relevant to your target students**
   (Data Scientist, ML Engineer, Business Analyst, Software Engineer, Product
   Manager, and the surrounding ~45 in that same neighborhood — not "Pipefitter" or
   "Marine Biologist"). O*NET's skill vocabulary (34 generic skill names like
   "Programming", "Critical Thinking") is broader than resume-style skills
   ("Python", "SQL") — you'll bridge that gap in the next step.

### 3.3 Step-by-step: build training pairs from a resume/job dataset
**Source: a Kaggle "Resume Dataset" or "Job Postings" dataset** (several free,
commonly-used ones exist — search Kaggle for "resume dataset" or "LinkedIn job
postings"; pick one with a `skills` or `required_skills` free-text field and a
`category`/`job_title` field).

```python
import pandas as pd

jobs = pd.read_csv("raw/kaggle_job_postings/job_postings.csv")

# Positive pairs: a job posting's actual listed skills <-> its own title
positive_pairs = jobs[["required_skills", "job_title"]].dropna()
positive_pairs["label"] = 1.0

# Negative pairs: shuffle skills against a DIFFERENT random job title
# so the model also learns what a mismatch looks like
import numpy as np
rng = np.random.default_rng(7)
shuffled_titles = jobs["job_title"].sample(frac=1, random_state=7).reset_index(drop=True)
negative_pairs = pd.DataFrame({
    "required_skills": jobs["required_skills"],
    "job_title": shuffled_titles,
    "label": 0.0,
})

pairs = pd.concat([positive_pairs, negative_pairs]).sample(frac=1, random_state=7)
pairs.to_csv("processed/m4/training_pairs.csv", index=False)
print(pairs["label"].value_counts())  # confirm roughly 50/50 split
```
This mirrors the exact input shape the tech doc's M4 training script expects
(`InputExample` pairs with a similarity label, fed into
`losses.CosineSimilarityLoss`).

### 3.4 Validation checklist
- [ ] `career_profiles.csv` has 50–80 rows, each with a non-empty `required_skills`
      list
- [ ] `training_pairs.csv` label distribution is close to 50/50 (an imbalanced
      dataset here will bias the model toward always predicting "match")
- [ ] Spot-check 10 random positive pairs by eye — do the skills actually plausibly
      belong to that job title? (Kaggle datasets vary in quality — this 10-minute
      check catches a badly-matched source dataset before you waste a training run
      on it)

---

## 4. M1 dataset — Knowledge Gap Classifier

### 4.1 What you need
A bank of MCQ questions, each tagged with a `topic` (and ideally `subtopic`), plus
enough of them per topic that a multi-label classifier can learn "wrong answers on
these question IDs → weak in this topic" patterns. Target: **at least 150–300
tagged questions per major subject** (Math, English, Physics, Chemistry — matching
your `EXAM_SESSIONS.subject` values), covering the exam types your platform targets
(Sindh University entry test, MDCAT, CSS/PMS, ISSB — per your register page's
`exam_goal` options).

### 4.2 Step-by-step: sourcing
1. **Past papers (primary source):** Sindh University entry test, MDCAT (Sindh/
   Punjab boards publish these), CSS/PMS/NTS past papers — these are widely
   circulated as free PDFs on official board websites and public educational
   portals (Sindh Board's own site, FPSC's official past-papers section, NTS's
   official site). Stick to official board/government sources and well-established
   public educational archives — this keeps you on solid ground for an academic
   project.
2. **Structured MCQ datasets (secondary/supplementary source):** search Kaggle and
   HuggingFace Datasets for existing MCQ banks in your subjects (e.g. general
   science/math MCQ collections, SAT/ACT-style question banks) to pad out topics
   that are thin from past papers alone. **Note in your thesis exactly which parts
   came from where** — this is good academic practice and useful for your defense
   if asked about data provenance.

### 4.3 Step-by-step: extracting questions from PDF past papers
This is the most manual part of the whole dataset phase — budget real time for it.
```python
# extract_pdf_questions.py — pulls raw text out of past-paper PDFs for manual tagging
import pdfplumber
import re

def extract_questions_from_pdf(pdf_path):
    raw_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            raw_text += (page.extract_text() or "") + "\n"

    # Most past papers number questions like "1." or "Q1." — adjust this regex
    # per-source, since formatting varies a lot between boards
    pattern = re.compile(r"(?:Q\.?\s*)?(\d{1,3})[.)]\s*(.+?)(?=(?:Q\.?\s*)?\d{1,3}[.)]|\Z)", re.DOTALL)
    matches = pattern.findall(raw_text)
    return [{"question_number": num, "raw_text": text.strip()} for num, text in matches]

# Run over every PDF in raw/past_papers/, dump to one big CSV for manual review —
# this step WILL need manual cleanup afterward; don't expect the regex to be perfect
# on the first try across different board formats.
```
**Realistic expectation:** this regex-based extraction gets you 70–90% of the way;
plan for a manual pass in a spreadsheet to fix questions that got split wrong,
options that got mixed into the question text, or garbled OCR text from
scanned-image PDFs (run scanned PDFs through OCR first — Tesseract via `pytesseract`
— before the regex step above).

### 4.4 Step-by-step: tagging topic labels
Once questions are cleanly extracted into a spreadsheet
(`question_id, question_text, options, correct_answer`), add a `topic` and
`subtopic` column by hand, using a **fixed, pre-agreed topic taxonomy** so labels
stay consistent — don't invent topic names ad hoc while tagging, or the label space
balloons and the classifier can't learn it well. Example taxonomy for Mathematics
(mirrors what's already shown in the exam-prep UI):
```
Mathematics: Algebra, Geometry, Trigonometry, Calculus, Statistics
English: Grammar, Reading Comprehension, Vocabulary, Writing
Physics: Mechanics, Electricity & Magnetism, Waves & Optics, Modern Physics
Chemistry: Organic, Inorganic, Physical Chemistry
```
If two people are tagging (you + your teammate, Muhammad Haroon Abbas), **tag an
overlapping sample of ~30 questions independently first, compare, and resolve
disagreements before splitting up the rest** — this is a cheap way to catch
"we secretly disagree about what counts as Algebra vs. Calculus" before it's spread
across 300 questions.

### 4.5 Step-by-step: building the (student, question, correct/incorrect) history for training
M1 needs labeled examples of "this pattern of wrong answers implies weak in topic
X." Since you don't have real student answer history yet, generate synthetic
answer sessions **on top of the real, correctly-tagged question bank** from §4.4
(the questions are real; the *sessions* are simulated) — this is a legitimate and
common approach for cold-start classifiers:
```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(11)
questions = pd.read_csv("processed/m1/question_bank.csv")  # from §4.4
topics = questions["topic"].unique()

sessions = []
for student_id in range(1, 500):
    # Each synthetic student has a hidden "true weakness" per topic (0=weak..1=strong)
    topic_strength = {t: rng.uniform(0, 1) for t in topics}
    sampled_qs = questions.sample(n=rng.integers(15, 30), random_state=student_id)
    for _, q in sampled_qs.iterrows():
        p_correct = topic_strength[q["topic"]]
        is_correct = rng.uniform() < p_correct
        sessions.append({
            "student_id": student_id,
            "question_id": q["question_id"],
            "topic": q["topic"],
            "is_correct": is_correct,
        })

pd.DataFrame(sessions).to_csv("processed/m1/synthetic_answer_sessions.csv", index=False)
```
Then aggregate to multi-label targets per student (`weak_in_algebra: 0/1`, etc.) at
some threshold (e.g. <50% correct on a topic = weak) — this final aggregated table
is what actually feeds `AutoModelForSequenceClassification` fine-tuning.

### 4.6 Validation checklist
- [ ] Every topic in your taxonomy has ≥30 real, correctly-tagged questions (below
      this, the model can't learn that topic reliably)
- [ ] No duplicate questions (common when pulling from multiple past-paper years —
      `df.drop_duplicates(subset="question_text")`)
- [ ] Multi-label target table has a reasonable class balance per topic (print
      `.mean()` per weak-topic column — if one topic is 95%+ "not weak" across all
      synthetic students, loosen the synthetic `topic_strength` distribution)

---

## 5. M5 dataset — CV/Transcript NER Extractor

### 5.1 What you need
Resume/CV text with **token-level BIO tags** for the entity types you care about:
`SKILL`, `CERT` (certification), `GPA`, `DEGREE`, `ORG` (university/company). BIO
tagging means each token is labeled `B-SKILL` (beginning of a skill span),
`I-SKILL` (inside/continuation), or `O` (outside any entity).

### 5.2 Step-by-step: sourcing raw resumes
**Source: a Kaggle "Resume Dataset"** (the same family of dataset used for M4's job
postings, or a dedicated resume-text one — search Kaggle for "resume dataset NLP" —
several free ones with plain resume text exist). You want **raw resume text**, not
just structured fields, since M5's whole job is to extract structure *from*
unstructured text — a dataset that's already neatly structured defeats the purpose.

### 5.3 Step-by-step: BIO labeling with a semi-automated bootstrap
Full manual BIO tagging of hundreds of resumes is slow. Bootstrap it instead:

1. **Auto-tag with pattern matching first**, using spaCy's `PhraseMatcher` against
   known skill/degree/cert vocabularies (you already have a skill list from the
   O*NET-derived M4 data in §3 — reuse it here):
```python
import spacy
from spacy.matcher import PhraseMatcher

nlp = spacy.load("en_core_web_sm")
matcher = PhraseMatcher(nlp.vocab, attr="LOWER")

skill_vocab = pd.read_csv("processed/m4/career_profiles.csv")["required_skills"].explode().unique()
matcher.add("SKILL", [nlp.make_doc(s) for s in skill_vocab])

cert_vocab = ["AWS Certified", "PMP", "CFA", "Google Analytics", "Scrum Master"]  # expand as needed
matcher.add("CERT", [nlp.make_doc(c) for c in cert_vocab])

def auto_tag(text):
    doc = nlp(text)
    matches = matcher(doc)
    tags = ["O"] * len(doc)
    for match_id, start, end in matches:
        label = nlp.vocab.strings[match_id]
        tags[start] = f"B-{label}"
        for i in range(start + 1, end):
            tags[i] = f"I-{label}"
    return list(zip([t.text for t in doc], tags))
```
2. **GPA and DEGREE** are regex-friendly and don't need a vocabulary list — tag
   these with patterns instead: GPA like `\b[0-4]\.\d{1,2}\s*/\s*4\.0\b`, DEGREE
   with a fixed list (`BS`, `BSc`, `MSc`, `MBA`, `Bachelor`, `Master`, `PhD`, ...).
3. **Manually review and correct a sample** — don't trust the auto-tagger blindly.
   Export ~100–150 auto-tagged resumes to a labeling-friendly format and hand-correct
   them. A spreadsheet with one row per token is workable at this scale; if it feels
   too tedious, a free tool like **doccano** (self-hosted, open source, runs locally
   or on Colab) gives a proper visual tagging UI and exports directly to the
   CoNLL/JSONL format HuggingFace's token-classification training expects — worth
   the 30-minute setup for a dataset this size.
4. **Leave the rest auto-tagged, uncorrected** — for a multi-hundred-resume
   dataset, hand-correcting 100–150 and leaving the remainder as auto-tagged
   "silver" labels is a standard, accepted trade-off (manually-corrected = "gold"
   labels, used for your validation/test split; auto-tagged = "silver," used only
   for training).

### 5.4 Step-by-step: exporting to the training format
```python
import json

def to_bio_jsonl(tagged_examples, out_path):
    with open(out_path, "w") as f:
        for tokens_tags in tagged_examples:
            tokens, tags = zip(*tokens_tags)
            f.write(json.dumps({"tokens": list(tokens), "ner_tags": list(tags)}) + "\n")

# gold = hand-corrected subset -> becomes your validation/test split
# silver = auto-tagged remainder -> becomes your training split (larger, noisier)
to_bio_jsonl(gold_examples, "labeled/m5/gold_val_test.jsonl")
to_bio_jsonl(silver_examples, "labeled/m5/silver_train.jsonl")
```

### 5.5 Validation checklist
- [ ] Gold (hand-corrected) set has at least 100 resumes, split ~50/50 into
      validation/test
- [ ] Every entity type (`SKILL`, `CERT`, `GPA`, `DEGREE`, `ORG`) appears at least
      20 times in the gold set — an entity type with near-zero examples will just
      never be predicted, and you won't know why until training results look odd
- [ ] Spot-check 15–20 auto-tagged ("silver") examples by eye for obviously wrong
      tags (e.g. a company name mis-tagged as `SKILL`) — if the error rate looks high
      (>30%), tighten the vocabulary lists in §5.3 before generating the full silver
      set

---

## 6. M2 dataset — Adaptive Question Generator (do this last — heaviest lift)

### 6.1 What you need
This model doesn't need a *new* dataset from scratch — it reuses the **same
correctly-tagged question bank built for M1 (§4.4)**, reshaped as
generation-training data: `(topic, difficulty) → generated question + 4 options +
correct answer`. This is why M1's dataset work should genuinely happen before M2's —
you get it for free once M1's bank exists.

### 6.2 Step-by-step: reshaping M1's data into M2's training format
```python
import pandas as pd
import json

questions = pd.read_csv("processed/m1/question_bank.csv")
# expected columns from §4.4: question_id, question_text, topic, subtopic,
# difficulty, option_a, option_b, option_c, option_d, correct_answer

def to_flan_t5_example(row):
    prompt = f"Generate a {row['difficulty']} difficulty multiple-choice question about {row['topic']} ({row['subtopic']})."
    target = json.dumps({
        "question": row["question_text"],
        "options": {
            "A": row["option_a"], "B": row["option_b"],
            "C": row["option_c"], "D": row["option_d"],
        },
        "correct_answer": row["correct_answer"],
    })
    return {"input_text": prompt, "target_text": target}

m2_data = questions.apply(to_flan_t5_example, axis=1, result_type="expand")
m2_data.to_csv("processed/m2/flan_t5_training_data.csv", index=False)
print(f"{len(m2_data)} prompt/target pairs — target is 500+; if short, this is a direct signal to add more to the M1 bank first")
```

### 6.3 Difficulty labeling — the one thing M1's bank might be missing
If §4.4's tagging pass didn't include a `difficulty` column (easy/medium/hard),
add it now — either by hand (fastest for a bank of a few hundred questions) or by a
simple heuristic proxy if hand-tagging is too slow: shorter questions with fewer
clauses and single-step calculations ≈ easy; multi-step word problems or
questions requiring combining two concepts ≈ hard. A heuristic first pass, spot
checked by hand, is a reasonable trade-off here given the time budget.

### 6.4 Validation checklist
- [ ] At least 500 prompt/target pairs total (BLEU/ROUGE fine-tuning on much less
      than this tends to overfit visibly — if you're short, that's a sign to enlarge
      the M1 question bank first, which benefits both models at once)
- [ ] Roughly even spread across difficulty levels (skewed-easy is the most common
      failure mode, since easy questions are fastest to source/tag)
- [ ] JSON in `target_text` is valid JSON for every row (`json.loads` each one in a
      loop before saving — a single malformed row will silently produce garbage
      generations for that topic/difficulty combination during training)

---

## 7. Cross-dataset consistency pass (do this once, after all five are built)

Before moving to Phase 4, run one linking check: **topic and subject names must
match exactly, character-for-character, across M1's question bank, M3's (if you
add real subject columns later), and the frontend's hardcoded subject values**
(`Mathematics`, `English`, `Physics`, `Chemistry` — from `exam-prep.html`'s original
subject chips and the ERD's `TOPIC_SCORES.subject` column). A mismatch here (e.g.
`"Math"` in your dataset vs. `"Mathematics"` in the frontend) is an easy, silent bug
that only surfaces much later when a real API response comes back with a weak-topic
name the frontend doesn't recognize and can't display properly.

```python
FRONTEND_SUBJECTS = {"Mathematics", "English", "Physics", "Chemistry"}
dataset_subjects = set(pd.read_csv("processed/m1/question_bank.csv")["subject"].unique())
assert dataset_subjects.issubset(FRONTEND_SUBJECTS), f"Mismatch: {dataset_subjects - FRONTEND_SUBJECTS}"
```

---

## 8. Final Phase 3 deliverables checklist

- [ ] `mentora_data/processed/m3/score_histories_no_label.csv` — synthetic, validated
- [ ] `mentora_data/processed/m4/career_profiles.csv` +
      `mentora_data/processed/m4/training_pairs.csv` — O*NET-derived, validated
- [ ] `mentora_data/processed/m1/question_bank.csv` +
      `mentora_data/processed/m1/synthetic_answer_sessions.csv` — real questions,
      synthetic sessions, validated
- [ ] `mentora_data/labeled/m5/gold_val_test.jsonl` +
      `mentora_data/labeled/m5/silver_train.jsonl` — BIO-tagged, validated
- [ ] `mentora_data/processed/m2/flan_t5_training_data.csv` — reshaped from M1,
      validated
- [ ] Cross-dataset subject-name consistency check passed (§7)
- [ ] Every raw source file's origin documented (which board/Kaggle dataset/O*NET
      table it came from) in a short `mentora_data/SOURCES.md` — you'll want this
      for the thesis's data section and for your defense if asked "where did this
      data come from"
- [ ] Everything lives under `mentora_data/` in Google Drive, nothing left only on
      local disk or only in a Colab session's temp storage

**Once this checklist is complete, Phase 4 (Colab training) can start immediately
for all five models in the priority order from the master plan: M3 → M4 → M1 → M5 →
M2.**
