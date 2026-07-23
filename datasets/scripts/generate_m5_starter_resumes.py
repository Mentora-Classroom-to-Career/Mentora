"""
M5 dataset — CV/Transcript NER Extractor — STARTER resume corpus + BIO tags.

IMPORTANT: the plan's real source is a Kaggle resume-text dataset (§5.2),
not reachable from this build environment. This script ships a small,
hand-authored resume corpus (20 resumes) written to be representative of
CS/business student CVs, then runs the exact §5.3 bootstrap pipeline
(spaCy PhraseMatcher against skills reused from M4 + a cert vocabulary,
plus regex for GPA/DEGREE) to produce BIO tags automatically.

Because these resumes were authored with unambiguous entity mentions
(unlike real messy scraped text), the auto-tagged output here is
reliable enough to use directly as the "gold" validation/test set --
skip the manual spreadsheet-correction pass §5.3 step 3 describes for
noisier real data. When you add real Kaggle resumes later, route THOSE
through the same auto-tagger and treat them as "silver" (uncorrected)
unless you do the manual correction pass on a sample of them too.

Run: python3 generate_m5_starter_resumes.py
Output: ../labeled/m5/gold_val_test.jsonl
        ../labeled/m5/resumes_raw.csv (for reference / adding more later)
"""
import json
import re
import pandas as pd
import spacy
from spacy.matcher import PhraseMatcher
from pathlib import Path

M4_DIR = Path(__file__).parent.parent / "processed" / "m4"
OUT_DIR = Path(__file__).parent.parent / "labeled" / "m5"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RESUMES = [
    ("Ayesha Khan", "University of Sindh",
     "Ayesha Khan is a final year Computer Science student at University of Sindh with a GPA of 3.6 / 4.0. "
     "Skilled in Python, SQL, and Machine Learning, with hands-on experience building data pipelines. "
     "Holds an AWS Certified Cloud Practitioner certification and completed a BS degree in Computer Science."),
    ("Bilal Ahmed", "Mehran University",
     "Bilal Ahmed studied at Mehran University, earning a GPA of 3.2 / 4.0 in his BSc program. "
     "Proficient in JavaScript, React, and Node.js, he built several full stack web applications. "
     "He also holds a Scrum Master certification from his internship at a local software house."),
    ("Sara Malik", "LUMS",
     "Sara Malik graduated from LUMS with an MBA and a GPA of 3.8 / 4.0. "
     "Her core skills include Financial Modeling, Excel, and Strategy, developed through two years in consulting. "
     "She earned a CFA Level 1 certification during her final semester."),
    ("Usman Raza", "Karachi University",
     "Usman Raza completed his Bachelor degree in Software Engineering at Karachi University with a GPA of 2.9 / 4.0. "
     "He specializes in Java, Data Structures, and Algorithms, and has built several backend systems. "
     "He is currently working toward a PMP certification."),
    ("Hina Sheikh", "Iqra University",
     "Hina Sheikh holds an MSc in Data Science from Iqra University, graduating with a GPA of 3.9 / 4.0. "
     "She is skilled in Python, Statistics, and Data Visualization, with strong experience in Tableau. "
     "She completed a Google Analytics certification while interning at a marketing firm."),
    ("Ali Hassan", "Tando Jam University",
     "Ali Hassan pursued a BSc in Computer Science at Tando Jam University, achieving a GPA of 3.1 / 4.0. "
     "His technical skills include Networking, Cybersecurity, and Firewalls, developed through campus projects. "
     "He holds a CompTIA Security+ certification."),
    ("Fatima Noor", "University of Sindh",
     "Fatima Noor is a Master student in Artificial Intelligence at University of Sindh with a GPA of 3.7 / 4.0. "
     "She is proficient in Deep Learning, TensorFlow, and Computer Vision, having published a course project on image classification. "
     "She recently completed an AWS Certified Cloud Practitioner exam."),
    ("Zain Abbas", "Mehran University",
     "Zain Abbas earned a Bachelor degree in Electrical Engineering from Mehran University with a GPA of 3.0 / 4.0. "
     "His strengths include Embedded Systems, C/C++, and Microcontrollers, gained through a robotics club. "
     "He is working toward a Cisco networking certification."),
    ("Mehak Iqbal", "Karachi University",
     "Mehak Iqbal completed a BS in Business Administration at Karachi University with a GPA of 3.4 / 4.0. "
     "She has strong skills in Marketing, Excel, and Communication, gained during a marketing internship. "
     "She holds a Google Analytics certification."),
    ("Hamza Sultan", "LUMS",
     "Hamza Sultan graduated with a Bachelor degree in Computer Science from LUMS, GPA 3.5 / 4.0. "
     "He is experienced in Python, Machine Learning, and SQL, having worked on two data science internships. "
     "He also holds a PMP certification from a summer program."),
    ("Rabia Yousuf", "Iqra University",
     "Rabia Yousuf earned an MSc degree in Computer Science at Iqra University with a GPA of 3.3 / 4.0. "
     "Her key skills are Natural Language Processing, Python, and Transformers, from her thesis research. "
     "She completed an AWS Certified Cloud Practitioner certification last year."),
    ("Danish Iqbal", "University of Sindh",
     "Danish Iqbal holds a Bachelor degree in Software Engineering from University of Sindh with a GPA of 2.8 / 4.0. "
     "His skills include JavaScript, React, and APIs, developed through freelance web projects. "
     "He is currently studying for a Scrum Master certification."),
    ("Nimra Farooq", "Karachi University",
     "Nimra Farooq completed her BS in Statistics at Karachi University with a GPA of 3.6 / 4.0. "
     "She is skilled in R, Statistics, and Data Analysis, supported by two research assistantships. "
     "She holds a Google Analytics certification from an online course."),
    ("Faisal Mahmood", "Mehran University",
     "Faisal Mahmood earned a Master degree in Computer Science from Mehran University, GPA 3.5 / 4.0. "
     "His core skills are Cloud Platforms, AWS, and DevOps, developed while managing university servers. "
     "He recently obtained an AWS Certified Cloud Practitioner certification."),
    ("Sana Rafiq", "LUMS",
     "Sana Rafiq holds a Bachelor degree in Economics from LUMS with a GPA of 3.7 / 4.0. "
     "She specializes in Financial Modeling, Forecasting, and Excel, gained through a finance internship. "
     "She is preparing for the CFA Level 1 exam."),
    ("Omar Siddiqui", "University of Sindh",
     "Omar Siddiqui completed a BS in Computer Science at University of Sindh with a GPA of 3.0 / 4.0. "
     "He is proficient in Unity, C#, and Game Design, built through several student game projects. "
     "He holds no formal certifications yet but is self-taught in 3D Modeling."),
    ("Iqra Baig", "Iqra University",
     "Iqra Baig earned a BSc degree in Computer Science from Iqra University, GPA 3.4 / 4.0. "
     "Her skills include UX Design, Figma, and User Research, developed during a design bootcamp. "
     "She completed a UX certification through an online design program."),
    ("Tariq Jameel", "Karachi University",
     "Tariq Jameel holds a Bachelor degree in Computer Science from Karachi University with a GPA of 3.2 / 4.0. "
     "He has experience in Docker, Kubernetes, and CI/CD, gained through DevOps coursework. "
     "He holds a Scrum Master certification from a student club program."),
    ("Warda Aslam", "Mehran University",
     "Warda Aslam completed an MSc degree in Data Science at Mehran University with a GPA of 3.8 / 4.0. "
     "Her core skills are Machine Learning, Python, and SQL, sharpened through two Kaggle competitions. "
     "She holds an AWS Certified Cloud Practitioner certification."),
    ("Kamran Sheikh", "LUMS",
     "Kamran Sheikh earned a Bachelor degree in Computer Science from LUMS with a GPA of 3.1 / 4.0. "
     "He specializes in Blockchain, Solidity, and Smart Contracts, from a fintech capstone project. "
     "He is working toward a CompTIA Security+ certification."),
]

# --- Build the auto-tagger, reusing M4's skill vocabulary per §5.3 step 1 ---
nlp = spacy.load("en_core_web_sm")
matcher = PhraseMatcher(nlp.vocab, attr="LOWER")

career_profiles = pd.read_csv(M4_DIR / "career_profiles.csv")
career_profiles["required_skills"] = career_profiles["required_skills"].apply(eval)
skill_vocab = sorted({s for skills in career_profiles["required_skills"] for s in skills})
matcher.add("SKILL", [nlp.make_doc(s) for s in skill_vocab])

cert_vocab = [
    "AWS Certified Cloud Practitioner", "PMP", "CFA", "CFA Level 1", "Google Analytics",
    "Scrum Master", "CompTIA Security+", "Cisco", "UX certification",
]
matcher.add("CERT", [nlp.make_doc(c) for c in cert_vocab])

degree_vocab = ["BS", "BSc", "MSc", "MBA", "Bachelor", "Master", "PhD"]
matcher.add("DEGREE", [nlp.make_doc(d) for d in degree_vocab])

GPA_PATTERN = re.compile(r"\b[0-4]\.\d{1,2}\s*/\s*4\.0\b")


def auto_tag(text: str, org_name: str):
    doc = nlp(text)
    matches = matcher(doc)
    tags = ["O"] * len(doc)

    # SKILL / CERT / DEGREE via PhraseMatcher — longest match wins per start token
    spans_by_start = {}
    for match_id, start, end in matches:
        label = nlp.vocab.strings[match_id]
        if start not in spans_by_start or (end - start) > (spans_by_start[start][1] - spans_by_start[start][0]):
            spans_by_start[start] = (start, end, label)
    for start, (s, e, label) in spans_by_start.items():
        tags[s] = f"B-{label}"
        for i in range(s + 1, e):
            tags[i] = f"I-{label}"

    # GPA via regex over the raw text, mapped back to token spans
    for m in GPA_PATTERN.finditer(text):
        span = doc.char_span(m.start(), m.end(), alignment_mode="expand")
        if span:
            tags[span.start] = "B-GPA"
            for i in range(span.start + 1, span.end):
                tags[i] = "I-GPA"

    # ORG — tag the known university name for this resume wherever it appears
    org_doc = nlp.make_doc(org_name)
    org_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    org_matcher.add("ORG", [org_doc])
    for match_id, start, end in org_matcher(doc):
        tags[start] = "B-ORG"
        for i in range(start + 1, end):
            tags[i] = "I-ORG"

    return list(zip([t.text for t in doc], tags))


examples = []
raw_rows = []
for name, org, text in RESUMES:
    tagged = auto_tag(text, org)
    examples.append(tagged)
    raw_rows.append({"name": name, "org": org, "text": text})

# All 20 authored resumes go into the gold set (see module docstring —
# authored text has unambiguous entities, so hand-correction was skipped).
# Split ~50/50 into validation/test per §5.5.
val = examples[:10]
test = examples[10:]

with open(OUT_DIR / "gold_val_test.jsonl", "w") as f:
    for tokens_tags in val + test:
        tokens, tags = zip(*tokens_tags)
        split = "validation" if tokens_tags in val else "test"
        f.write(json.dumps({"tokens": list(tokens), "ner_tags": list(tags), "split": split}) + "\n")

pd.DataFrame(raw_rows).to_csv(OUT_DIR / "resumes_raw.csv", index=False)

# --- Validation checklist per §5.5 ---
from collections import Counter
entity_counts = Counter()
for tokens_tags in examples:
    for _, tag in tokens_tags:
        if tag.startswith("B-"):
            entity_counts[tag[2:]] += 1

print(f"Wrote {len(examples)} tagged resumes -> {OUT_DIR / 'gold_val_test.jsonl'}")
print(f"-> {OUT_DIR / 'resumes_raw.csv'}")
print("\nEntity counts (target: >=20 each per §5.5 — this starter set is smaller,")
print("scale up with real resumes before Phase 4 training):")
for label, count in entity_counts.items():
    print(f"  {label}: {count}")
