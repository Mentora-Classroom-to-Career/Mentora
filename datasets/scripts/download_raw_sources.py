"""
Cross-platform (Windows/Mac/Linux) re-downloader for the large raw sources
excluded from git — no bash/curl/unzip/tar required, pure Python stdlib.
Equivalent to download_raw_sources.sh; see datasets/SOURCES.md for
provenance/licenses.

Run: python build_real_question_bank.py  (from datasets/scripts/, same as before)
"""
import tarfile
import urllib.request
import zipfile
from pathlib import Path

RAW_DIR = Path(__file__).parent.parent / "raw"


def download(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)


def main():
    print("Downloading AQuA-RAT (Mathematics, Apache 2.0)...")
    aqua_dir = RAW_DIR / "aqua"
    download("https://raw.githubusercontent.com/google-deepmind/AQuA/master/train.json", aqua_dir / "train.json")
    download("https://raw.githubusercontent.com/google-deepmind/AQuA/master/test.json", aqua_dir / "test.json")
    download("https://raw.githubusercontent.com/google-deepmind/AQuA/master/dev.json", aqua_dir / "dev.json")

    print("Downloading RACE-C (English Reading Comprehension, non-commercial research use)...")
    race_dir = RAW_DIR / "race_c"
    zip_path = race_dir / "data.zip"
    download("https://raw.githubusercontent.com/mrcdata/race-c/master/data.zip", zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(race_dir)

    print("Downloading CLOTH (English Grammar/Vocabulary/Writing, MIT)...")
    cloth_dir = RAW_DIR / "cloth"
    tar_path = cloth_dir / "data.tar.gz"
    download(
        "https://raw.githubusercontent.com/zhaowei8188127/Large-scale-Cloze-Test-Dataset-Designed-by-Teachers/master/data.tar.gz",
        tar_path,
    )
    with tarfile.open(tar_path, "r:gz") as tf:
        tf.extractall(cloth_dir)

    print("\nDone. Now run: python3 build_real_question_bank.py")


if __name__ == "__main__":
    main()
