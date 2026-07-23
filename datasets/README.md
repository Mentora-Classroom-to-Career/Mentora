# MENTORA Datasets (Phase 3)

Mirrors the `mentora_data/` folder structure from
`MENTORA_Phase3_Datasets.md` so this stays a drop-in copy target for Google
Drive before Colab training (Phase 4).

```
datasets/
  raw/            # put real downloaded sources here (O*NET flat files,
                   # Kaggle CSVs, past-paper PDFs) — currently empty, see
                   # SOURCES.md for what's still needed
  processed/       # cleaned, model-ready CSVs — m1/, m2/, m3/, m4/
  labeled/         # BIO-tagged NER data — m5/
  scripts/         # generation/processing scripts, safe to re-run anytime
  SOURCES.md        # what's real vs. starter/placeholder, and why
```

**Start here:** read `SOURCES.md` first — it tells you exactly which
datasets are final and which are starter sets that need real data added
before Phase 4 training for a strong thesis/defense.

## Quick start

```bash
cd datasets/scripts
pip install -r ../requirements.txt --break-system-packages
python3 -m spacy download en_core_web_sm

python3 generate_m3_synthetic.py
python3 generate_m1_starter_bank.py
python3 generate_m1_synthetic_sessions.py
python3 generate_m2_from_m1.py
python3 generate_m4_starter_careers.py
python3 generate_m5_starter_resumes.py
python3 check_consistency.py
```

All scripts are idempotent (safe to re-run) and will just overwrite their
own output files.
