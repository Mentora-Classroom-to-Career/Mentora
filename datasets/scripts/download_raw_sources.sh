#!/bin/bash
# Re-downloads the large raw sources excluded from git (see datasets/.gitignore
# and SOURCES.md for provenance/licenses). Safe to re-run any time.
set -e
cd "$(dirname "$0")/.."

mkdir -p raw/aqua raw/cloth raw/race_c

echo "Downloading AQuA-RAT (Mathematics, Apache 2.0)..."
curl -sL -o raw/aqua/train.json https://raw.githubusercontent.com/google-deepmind/AQuA/master/train.json
curl -sL -o raw/aqua/test.json https://raw.githubusercontent.com/google-deepmind/AQuA/master/test.json
curl -sL -o raw/aqua/dev.json https://raw.githubusercontent.com/google-deepmind/AQuA/master/dev.json

echo "Downloading RACE-C (English Reading Comprehension, non-commercial research use)..."
curl -sL -o raw/race_c/data.zip https://raw.githubusercontent.com/mrcdata/race-c/master/data.zip
(cd raw/race_c && unzip -oq data.zip)

echo "Downloading CLOTH (English Grammar/Vocabulary/Writing, MIT)..."
curl -sL -o raw/cloth/data.tar.gz https://raw.githubusercontent.com/zhaowei8188127/Large-scale-Cloze-Test-Dataset-Designed-by-Teachers/master/data.tar.gz
(cd raw/cloth && tar xzf data.tar.gz)

echo "Done. Run scripts/build_real_question_bank.py to regenerate question_bank.csv."
