"""
M1 REAL dataset builder — replaces the hand-authored starter bank with
real, licensed, ML-research-grade MCQ data pulled from the web (per the
user's explicit request for real data), reshaped into our schema.

Sources (see datasets/SOURCES.md for full provenance + licenses):
  - AQuA-RAT (google-deepmind/AQuA, Apache 2.0)          -> Mathematics
  - science-questions CSV (joelgrus, ARC-derived)         -> Physics, Chemistry
  - RACE-C (mrcdata/race-c, non-commercial research use)  -> English: Reading Comprehension
  - CLOTH (zhaowei8188127 mirror, MIT)                    -> English: Grammar, Vocabulary, Writing

Also appends the Phase 3 hand-authored starter bank (70 questions) at
the end, tagged with its original source, so nothing already reviewed
is lost.

Run: python3 build_real_question_bank.py
Output: ../processed/m1/question_bank.csv (OVERWRITES the starter-only version)
"""
import ast
import json
import random
import re
from pathlib import Path

import pandas as pd
import spacy

RAW_DIR = Path(__file__).parent.parent / "raw"
OUT_DIR = Path(__file__).parent.parent / "processed" / "m1"
random.seed(42)

nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])

rows = []  # each: dict matching question_bank.csv columns


def add_row(subject, topic, difficulty, question_text, options, correct_answer, source, passage=None):
    opts = list(options) + [None] * (5 - len(options))
    rows.append({
        "question_id": None,  # assigned at the end
        "question_text": question_text,
        "subject": subject,
        "topic": topic,
        "subtopic": None,
        "difficulty": difficulty,
        "option_a": opts[0], "option_b": opts[1], "option_c": opts[2],
        "option_d": opts[3], "option_e": opts[4],
        "passage": passage,
        "correct_answer": correct_answer,
        "source": source,
    })


# ============================================================ AQuA (Math) ==
MATH_KEYWORDS = {
    "Geometry": ["triangle", "circle", "square", "rectangle", "cylinder", "sphere", "angle", "perimeter",
                 "diameter", "radius", "polygon", "cube", "volume of", "area of", "trapezoid", "parallelogram",
                 "hexagon", "pentagon", "vertex", "vertices", "circumference"],
    "Trigonometry": ["sin(", "cos(", "tan(", "sine", "cosine", "tangent", "degrees"],
    "Statistics": ["average", "mean", "median", "probability", "standard deviation", "percent", "ratio of",
                   "combination", "permutation", "odds of", "likelihood"],
    "Calculus": ["derivative", "integral", "rate of change", "limit as"],
}


def classify_math_topic(question_text: str) -> str:
    q = question_text.lower()
    for topic, keywords in MATH_KEYWORDS.items():
        if any(kw in q for kw in keywords):
            return topic
    return "Algebra"  # AQuA's default: algebraic word problems


def load_aqua(n_sample=600):
    path = RAW_DIR / "aqua" / "train.json"
    with open(path) as f:
        lines = f.readlines()
    random.shuffle(lines)
    count = 0
    for line in lines:
        if count >= n_sample:
            break
        obj = json.loads(line)
        options = [opt.split(")", 1)[1].strip() if ")" in opt else opt for opt in obj["options"]]
        if len(options) != 5:
            continue
        topic = classify_math_topic(obj["question"])
        # AQuA doesn't label difficulty; approximate via rationale length as a proxy
        difficulty = "hard" if len(obj["rationale"]) > 300 else ("medium" if len(obj["rationale"]) > 120 else "easy")
        add_row("Mathematics", topic, difficulty, obj["question"], options, obj["correct"], "aqua_rat")
        count += 1
    print(f"AQuA: added {count} Mathematics questions")


# ================================================ science-questions (Sci) ==
PHYSICS_TOPIC_KEYWORDS = {
    "Electricity & Magnetism": ["circuit", "electric", "current", "voltage", "battery", "magnet", "wire",
                                 "resistor", "conductor", "insulator", "charge"],
    "Waves & Optics": ["light", "sound", "wave", "reflect", "lens", "mirror", "frequency", "echo",
                        "refraction", "prism", "pitch", "vibration"],
    "Modern Physics": ["radioactiv", "nuclear", "atom", "particle", "isotope", "half-life", "fission", "fusion"],
    "Mechanics": ["force", "motion", "energy", "friction", "gravity", "speed", "velocity", "mass", "weight",
                  "machine", "pulley", "lever", "acceleration", "momentum", "inertia", "pressure"],
}
CHEMISTRY_TOPIC_KEYWORDS = {
    "Organic": ["carbon compound", "hydrocarbon", "organic", "polymer", "fuel", "petroleum", "combustion",
                "photosynthesis", "cellular respiration", "carbohydrate", "protein", "fat"],
    "Inorganic": ["element", "compound", "mixture", "metal", "periodic table", "molecule", "atom", "ion"],
    "Physical Chemistry": ["reaction", "acid", "base", "ph ", "solution", "solubility", "state of matter",
                            "boiling", "melting", "evaporat", "condens", "chemical change", "physical change"],
}


def classify_science(question_text: str):
    """Returns (subject, topic) or (None, None) if it doesn't clearly match Physics/Chemistry."""
    q = question_text.lower()
    for topic, keywords in PHYSICS_TOPIC_KEYWORDS.items():
        if any(kw in q for kw in keywords):
            return "Physics", topic
    for topic, keywords in CHEMISTRY_TOPIC_KEYWORDS.items():
        if any(kw in q for kw in keywords):
            return "Chemistry", topic
    return None, None


_OPTION_RE = re.compile(r"\(A\)(.*?)\(B\)(.*?)\(C\)(.*?)\(D\)(.*)", re.DOTALL)


def load_science_questions():
    path = RAW_DIR / "science" / "questions.csv"
    df = pd.read_csv(path)
    added = 0
    for _, row in df.iterrows():
        text = str(row["question"])
        m = _OPTION_RE.search(text)
        if not m:
            continue
        stem = text[: m.start()].strip()
        options = [g.strip() for g in m.groups()]
        subject, topic = classify_science(stem)
        if subject is None:
            continue  # not confidently Physics or Chemistry -- skip (includes Biology)
        answer_key = str(row["AnswerKey"]).strip().upper()
        if answer_key not in {"A", "B", "C", "D"}:
            continue
        grade = row.get("schoolGrade")
        difficulty = "easy" if pd.notna(grade) and grade <= 4 else ("hard" if pd.notna(grade) and grade >= 7 else "medium")
        add_row(subject, topic, difficulty, stem, options, answer_key, "science_questions_arc_derived")
        added += 1
    print(f"science-questions: added {added} Physics/Chemistry questions")


# ================================================== RACE-C (English: RC) ==
def load_race_c(n_sample=250):
    base = RAW_DIR / "race_c" / "data"
    files = []
    for split in ["train", "dev", "test"]:
        for f in (base / split).glob("*.txt"):
            files.append((split, f))
    random.shuffle(files)

    added = 0
    for split, f in files:
        if added >= n_sample:
            break
        try:
            obj = json.loads(f.read_text())
        except Exception:
            continue
        article = obj["article"]
        # pick just ONE question per passage to keep passage repetition low across the sample
        idx = random.randrange(len(obj["questions"]))
        question = obj["questions"][idx]
        options = obj["options"][idx]
        answer = obj["answers"][idx]
        if len(options) != 4 or answer not in {"A", "B", "C", "D"}:
            continue
        difficulty = "hard"  # RACE-C is entirely college-level English exams
        add_row("English", "Reading Comprehension", difficulty, question, options, answer,
                "race_c", passage=article)
        added += 1
    print(f"RACE-C: added {added} English/Reading Comprehension questions")


# ====================================================== CLOTH (Eng: G/V) ==
DISCOURSE_MARKERS = {
    "however", "therefore", "moreover", "furthermore", "meanwhile", "although", "besides",
    "nevertheless", "otherwise", "instead", "thus", "hence", "consequently", "similarly",
    "in addition", "as a result", "for example", "in fact", "in other words",
    "in conclusion", "on the other hand", "in contrast", "for instance", "in summary",
    "first", "second", "finally", "next", "then", "after that", "in short",
}
FUNCTION_POS = {"ADP", "CCONJ", "SCONJ", "DET", "PRON", "AUX", "PART"}


def classify_cloth_blank(answer_word: str, context_doc) -> str:
    if answer_word.lower() in DISCOURSE_MARKERS:
        return "Writing"
    for tok in context_doc:
        if tok.text.lower() == answer_word.lower():
            return "Grammar" if tok.pos_ in FUNCTION_POS else "Vocabulary"
    return "Vocabulary"


def load_cloth(n_sample=350):
    base = RAW_DIR / "cloth" / "data" / "CLOTH"
    files = []
    for split in ["train", "valid", "test"]:
        for level in ["middle", "high"]:
            d = base / split / level
            if d.exists():
                files.extend([(level, f) for f in d.glob("*.json")])
    random.shuffle(files)

    added = 0
    for level, f in files:
        if added >= n_sample:
            break
        try:
            obj = json.loads(f.read_text())
        except Exception:
            continue
        article = obj["article"]
        segments = article.split("_")
        if len(segments) - 1 != len(obj["answers"]):
            continue  # malformed / mismatched blank count -- skip defensively

        # pick one blank from this article to keep passage variety high
        blank_idx = random.randrange(len(obj["answers"]))
        answer_letter = obj["answers"][blank_idx]
        options = obj["options"][blank_idx]
        if len(options) != 4 or answer_letter not in {"A", "B", "C", "D"}:
            continue
        answer_word = options["ABCD".index(answer_letter)]

        before = " ".join(segments[blank_idx].split()[-12:])
        after = " ".join(segments[blank_idx + 1].split()[:12])
        snippet = f"{before} _____ {after}"

        context_doc = nlp(article[:2000])  # cap for speed
        topic = classify_cloth_blank(answer_word, context_doc)
        difficulty = "hard" if level == "high" else "medium"

        add_row("English", topic, difficulty, f"Choose the word that best fits the blank: {snippet}",
                options, answer_letter, "cloth")
        added += 1
    print(f"CLOTH: added {added} English Grammar/Vocabulary/Writing questions")


# ================================================== hand-authored starter ==
def load_hand_authored():
    path = OUT_DIR / "question_bank.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    df = df[df["source"] == "hand_authored_starter"]
    added = 0
    for _, r in df.iterrows():
        options = [r["option_a"], r["option_b"], r["option_c"], r["option_d"]]
        add_row(r["subject"], r["topic"], r["difficulty"], r["question_text"], options,
                r["correct_answer"], "hand_authored_starter")
        added += 1
    print(f"hand-authored starter: carried forward {added} questions")


load_aqua(n_sample=4000)
load_science_questions()
load_race_c(n_sample=500)
load_cloth(n_sample=2000)
load_hand_authored()

df = pd.DataFrame(rows)
df["question_id"] = [f"Q{2000 + i}" for i in range(len(df))]
dup_count = df["question_text"].duplicated().sum()
print(f"\n(duplicate question_text rows before dedup: {dup_count})")
df = df.drop_duplicates(subset=["question_text"]).reset_index(drop=True)
df["question_id"] = [f"Q{2000 + i}" for i in range(len(df))]

OUT_DIR.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT_DIR / "question_bank.csv", index=False)

print(f"\nTOTAL: {len(df)} questions -> {OUT_DIR / 'question_bank.csv'}")
print("\nPer subject:")
print(df["subject"].value_counts())
print("\nPer subject/topic:")
print(df.groupby(["subject", "topic"]).size())
print("\nPer source:")
print(df["source"].value_counts())
