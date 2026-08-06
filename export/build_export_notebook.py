"""
Builds export_and_quantize.ipynb from MENTORA_Phase5_Export_Optimize.md,
adapted to our actual folder structure and training status.

STATUS as of this build: only M1 has actually been trained (eval_f1_micro
0.731, accepted). M3 partially ran (CPU smoke test only, didn't hit its
MAE<=5 target). M4, M5, M2 haven't been trained at all yet. This notebook
covers all 5 models per the Phase 5 spec, but sections for M3/M4/M5/M2 will
fail with a clear FileNotFoundError against Drive until those models are
actually trained -- each of those sections says so explicitly.

Run: python3 build_export_notebook.py
"""
import nbformat as nbf
from pathlib import Path

OUT_DIR = Path(__file__).parent / "notebooks"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def md(text):
    return nbf.v4.new_markdown_cell(text)


def code(text):
    return nbf.v4.new_code_cell(text)


cells = [
    md("# MENTORA — Phase 5: Export & Optimize for Local Inference\n\n"
       "**Status check before running this:** as of this notebook's last build, "
       "only **M1** has actually been trained (eval_f1_micro 0.731, accepted as "
       "good enough for now). M3 only ran as a CPU smoke test in a sandbox "
       "environment (didn't hit its MAE<=5 target — worth re-running for real in "
       "Colab). **M4, M5, and M2 haven't been trained at all yet.** Run §2 (M1) "
       "now; the M3/M4/M5/M2 sections will fail with a clear `FileNotFoundError` "
       "against Drive until those are actually trained first — that's expected, "
       "not a bug in this notebook."),
    md("## 0. Setup"),
    code(
        "!pip install -q optimum-onnx onnxruntime onnx\n"
        "# As of 2026, HuggingFace moved ONNX/ONNXRuntime integration out of core\n"
        "# `optimum` into `optimum-onnx` -- installing plain `optimum` alone will\n"
        "# NOT give you the ORTModelForXxx classes used below."
    ),
    code(
        "from google.colab import drive\n"
        "drive.mount('/content/drive')\n"
        "MODELS = '/content/drive/MyDrive/mentora_models'"
    ),

    md("## 1. M3 and M4 — no conversion needed, just confirm they exist\n\n"
       "Per the master plan: M3 (~2MB LSTM) and M4 (~420MB Sentence-Transformers) "
       "are cheap enough to stay plain FP32, always-resident — quantizing them "
       "buys little. **Run this only after M3/M4 are actually trained** (M3 needs "
       "a real Colab GPU run per training/README.md; M4 hasn't been trained yet "
       "at all)."),
    code(
        "import os\n\n"
        "m3_path = f'{MODELS}/model3_trajectory_predictor/lstm_final.pt'\n"
        "if os.path.exists(m3_path):\n"
        "    print(f'M3: {os.path.getsize(m3_path)/1e6:.1f} MB (expect ~1-3MB)')\n"
        "else:\n"
        "    print('M3 not found -- train it first (see training/train_m3.ipynb, run for real in Colab).')\n\n"
        "m4_path = f'{MODELS}/model4_career_matcher'\n"
        "if os.path.exists(m4_path):\n"
        "    print('M4 files:', os.listdir(m4_path))\n"
        "else:\n"
        "    print('M4 not found -- train it first (see training/train_m4.ipynb).')"
    ),

    md("## 2. M1 — DeBERTa Gap Classifier: export + quantize\n\n"
       "**This is the one that's actually ready to run today.**"),
    md("### 2.1 Export to ONNX"),
    code(
        "from optimum.onnxruntime import ORTModelForSequenceClassification\n"
        "from transformers import AutoTokenizer\n\n"
        "M1_DIR = f'{MODELS}/model1_gap_classifier/final'\n"
        "M1_ONNX_DIR = f'{MODELS}/model1_gap_classifier/onnx'\n\n"
        "ort_model = ORTModelForSequenceClassification.from_pretrained(M1_DIR, export=True)\n"
        "ort_model.save_pretrained(M1_ONNX_DIR)\n\n"
        "tokenizer = AutoTokenizer.from_pretrained(M1_DIR)\n"
        "tokenizer.save_pretrained(M1_ONNX_DIR)"
    ),
    md("### 2.2 Quantize (dynamic, INT8)"),
    code(
        "from onnxruntime.quantization import quantize_dynamic, QuantType\n"
        "import os\n\n"
        "for onnx_file in os.listdir(M1_ONNX_DIR):\n"
        "    if onnx_file.endswith('.onnx') and not onnx_file.endswith('_quantized.onnx'):\n"
        "        in_path = f'{M1_ONNX_DIR}/{onnx_file}'\n"
        "        out_path = f'{M1_ONNX_DIR}/{onnx_file.replace(\".onnx\", \"_quantized.onnx\")}'\n"
        "        quantize_dynamic(in_path, out_path, weight_type=QuantType.QUInt8)\n"
        "        print(f'{onnx_file}: {os.path.getsize(in_path)/1e6:.1f}MB -> {os.path.getsize(out_path)/1e6:.1f}MB')"
    ),
    md("### 2.3 Verify parity — PyTorch vs. ONNX vs. quantized ONNX\n\n"
       "Never skip this. Run on a batch of real validation examples from "
       "`datasets/processed/m1/question_bank.csv`, not just one string, and check "
       "that the **predicted multi-label set** (sigmoid + 0.5 threshold, matching "
       "the training notebook) mostly agrees between PyTorch and quantized-ONNX — "
       "a handful of borderline flips near the threshold is normal, many is not."),
    code(
        "import torch, numpy as np, pandas as pd\n"
        "from transformers import AutoModelForSequenceClassification\n"
        "import onnxruntime as ort\n\n"
        "qb = pd.read_csv(f'{MODELS.replace(\"mentora_models\", \"mentora_data\")}/processed/m1/question_bank.csv')\n"
        "sample_texts = qb['question_text'].sample(20, random_state=42).tolist()\n\n"
        "pt_model = AutoModelForSequenceClassification.from_pretrained(M1_DIR)\n"
        "pt_model.eval()\n\n"
        "onnx_files = [f for f in os.listdir(M1_ONNX_DIR) if f.endswith('.onnx') and '_quantized' not in f]\n"
        "quant_files = [f for f in os.listdir(M1_ONNX_DIR) if f.endswith('_quantized.onnx')]\n"
        "session = ort.InferenceSession(f'{M1_ONNX_DIR}/{onnx_files[0]}', providers=['CPUExecutionProvider'])\n"
        "qsession = ort.InferenceSession(f'{M1_ONNX_DIR}/{quant_files[0]}', providers=['CPUExecutionProvider'])\n\n"
        "flips = 0\n"
        "max_diff = 0.0\n"
        "for text in sample_texts:\n"
        "    inputs = tokenizer(text, return_tensors='pt')\n"
        "    with torch.no_grad():\n"
        "        pt_logits = pt_model(**inputs).logits.numpy()\n"
        "    onnx_inputs = {k: v.numpy() for k, v in inputs.items() if k in [i.name for i in session.get_inputs()]}\n"
        "    onnx_logits = session.run(None, onnx_inputs)[0]\n"
        "    q_logits = qsession.run(None, onnx_inputs)[0]\n\n"
        "    pt_pred = (1 / (1 + np.exp(-pt_logits)) >= 0.5)\n"
        "    q_pred = (1 / (1 + np.exp(-q_logits)) >= 0.5)\n"
        "    if not np.array_equal(pt_pred, q_pred):\n"
        "        flips += 1\n"
        "    max_diff = max(max_diff, np.abs(pt_logits - onnx_logits).max())\n\n"
        "print(f'PyTorch vs unquantized ONNX max logit diff: {max_diff:.6f} (expect < 1e-3)')\n"
        "print(f'Predicted label-set flips (PyTorch vs quantized ONNX): {flips}/{len(sample_texts)}')\n"
        "print('A handful of flips near the threshold is normal; many is a sign to re-check quantization.')"
    ),

    md("## 3. M5 — BERT NER: export + quantize\n\n"
       "**Not runnable yet — M5 hasn't been trained.** Same three steps as M1 "
       "once it has been (see training/train_m5.ipynb)."),
    code(
        "from optimum.onnxruntime import ORTModelForTokenClassification\n\n"
        "M5_DIR = f'{MODELS}/model5_cv_ner/final'\n"
        "M5_ONNX_DIR = f'{MODELS}/model5_cv_ner/onnx'\n\n"
        "ort_model = ORTModelForTokenClassification.from_pretrained(M5_DIR, export=True)\n"
        "ort_model.save_pretrained(M5_ONNX_DIR)\n"
        "m5_tokenizer = AutoTokenizer.from_pretrained(M5_DIR)\n"
        "m5_tokenizer.save_pretrained(M5_ONNX_DIR)\n\n"
        "for onnx_file in os.listdir(M5_ONNX_DIR):\n"
        "    if onnx_file.endswith('.onnx') and not onnx_file.endswith('_quantized.onnx'):\n"
        "        in_path = f'{M5_ONNX_DIR}/{onnx_file}'\n"
        "        out_path = f'{M5_ONNX_DIR}/{onnx_file.replace(\".onnx\", \"_quantized.onnx\")}'\n"
        "        quantize_dynamic(in_path, out_path, weight_type=QuantType.QUInt8)\n"
        "        print(f'{onnx_file}: {os.path.getsize(in_path)/1e6:.1f}MB -> {os.path.getsize(out_path)/1e6:.1f}MB')"
    ),
    md("For parity-checking a token-classification model, compare the **per-token "
       "predicted tag sequence** between PyTorch and quantized-ONNX on real resume "
       "snippets from `datasets/labeled/m5/gold_val_test.jsonl` — entity boundaries "
       "matter more than exact logit values here."),

    md("## 4. M2 — FLAN-T5-Large + LoRA: export + quantize (the involved one)\n\n"
       "**Not runnable yet — M2 hasn't been trained.** Once it has (see "
       "training/train_m2.ipynb), three steps: merge the LoRA adapter, export to "
       "ONNX (produces 3 graphs: encoder/decoder/decoder_with_past — normal for "
       "seq2seq, not a bug), quantize all 3."),
    code(
        "from peft import PeftModel\n"
        "from transformers import T5ForConditionalGeneration, T5Tokenizer\n\n"
        "base = T5ForConditionalGeneration.from_pretrained('google/flan-t5-large')\n"
        "peft_model = PeftModel.from_pretrained(base, f'{MODELS}/model2_question_generator/lora_adapter')\n"
        "merged = peft_model.merge_and_unload()\n\n"
        "MERGED_DIR = f'{MODELS}/model2_question_generator/merged'\n"
        "merged.save_pretrained(MERGED_DIR)\n"
        "t5_tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-large')\n"
        "t5_tokenizer.save_pretrained(MERGED_DIR)\n"
        "# This merged folder is ~3GB FP32 -- expected, intermediate-only, not downloaded to the laptop."
    ),
    code(
        "from optimum.onnxruntime import ORTModelForSeq2SeqLM\n\n"
        "M2_ONNX_DIR = f'{MODELS}/model2_question_generator/onnx'\n"
        "ort_model = ORTModelForSeq2SeqLM.from_pretrained(MERGED_DIR, export=True)\n"
        "ort_model.save_pretrained(M2_ONNX_DIR)\n"
        "t5_tokenizer.save_pretrained(M2_ONNX_DIR)\n\n"
        "for fname in os.listdir(M2_ONNX_DIR):\n"
        "    if fname.endswith('.onnx') and '_quantized' not in fname:\n"
        "        in_path = f'{M2_ONNX_DIR}/{fname}'\n"
        "        out_path = f'{M2_ONNX_DIR}/{fname.replace(\".onnx\", \"_quantized.onnx\")}'\n"
        "        quantize_dynamic(in_path, out_path, weight_type=QuantType.QUInt8)\n"
        "        print(f'{fname}: {os.path.getsize(in_path)/1e6:.0f}MB -> {os.path.getsize(out_path)/1e6:.0f}MB')"
    ),
    md("Verify parity via **generated text**, not raw logits — run across 10-15 "
       "varied prompts (use real prompts from `datasets/processed/m2/"
       "flan_t5_training_data.csv`'s `input_text` column) and confirm the output "
       "is still valid, parseable JSON (`json.loads(...)`) matching the "
       "question/options/correct_answer schema for the quantized version on every "
       "prompt — a few word choices differing is fine, a broken JSON shape is not."),
    code(
        "onnx_model = ORTModelForSeq2SeqLM.from_pretrained(\n"
        "    M2_ONNX_DIR,\n"
        "    encoder_file_name='encoder_model_quantized.onnx',\n"
        "    decoder_file_name='decoder_model_quantized.onnx',\n"
        "    decoder_with_past_file_name='decoder_with_past_model_quantized.onnx',\n"
        ")\n\n"
        "import json\n"
        "m2_data = pd.read_csv(f'{MODELS.replace(\"mentora_models\", \"mentora_data\")}/processed/m2/flan_t5_training_data.csv')\n"
        "sample_prompts = m2_data['input_text'].sample(12, random_state=42).tolist()\n\n"
        "valid_json_count = 0\n"
        "for prompt in sample_prompts:\n"
        "    inputs = t5_tokenizer(prompt, return_tensors='pt')\n"
        "    out = onnx_model.generate(**inputs, max_new_tokens=128)\n"
        "    text = t5_tokenizer.decode(out[0], skip_special_tokens=True)\n"
        "    try:\n"
        "        parsed = json.loads(text)\n"
        "        assert all(k in parsed for k in ['question', 'options', 'correct_answer'])\n"
        "        valid_json_count += 1\n"
        "    except Exception as e:\n"
        "        print(f'INVALID on prompt \"{prompt[:60]}...\": {e}')\n"
        "        print(f'  Got: {text[:200]}')\n\n"
        "print(f'\\nValid JSON output: {valid_json_count}/{len(sample_prompts)}')"
    ),

    md("## 5. Downloading everything from Drive to the laptop\n\n"
       "Only the final, right-sized pieces — not intermediate artifacts (e.g. NOT "
       "M2's `merged/` FP32 folder, ~3GB you don't need locally)."),
    code(
        "import shutil\n\n"
        "to_package = [\n"
        "    'model1_gap_classifier/onnx',\n"
        "    'model2_question_generator/onnx',      # only after M2 is trained + exported\n"
        "    'model3_trajectory_predictor',\n"
        "    'model4_career_matcher',                # only after M4 is trained\n"
        "    'model5_cv_ner/onnx',                    # only after M5 is trained + exported\n"
        "]\n"
        "for name in to_package:\n"
        "    src = f'{MODELS}/{name}'\n"
        "    if not os.path.exists(src):\n"
        "        print(f'Skipping {name} -- not trained/exported yet')\n"
        "        continue\n"
        "    zip_name = name.replace('/', '_')\n"
        "    shutil.make_archive(f'/content/{zip_name}', 'zip', src)\n"
        "    print(f'Packaged {name} -> /content/{zip_name}.zip')"
    ),
    md("Download each `.zip` via Colab's file browser (left sidebar → folder icon "
       "→ right-click → Download), or install the Google Drive desktop app on the "
       "laptop and let it sync `mentora_models/` directly — recommended if the "
       "connection can handle syncing a few GB, since it resumes automatically on "
       "interruption unlike a manual browser download.\n\n"
       "Place them into the backend's expected structure per the master plan §9:\n"
       "```\n"
       "mentora/ml_models/saved/\n"
       "├── model1_gap_classifier/        # unzipped onnx/ contents\n"
       "├── model2_question_generator/    # unzipped onnx/ contents (3 quantized graphs)\n"
       "├── model3_trajectory_predictor/  # lstm_final.pt\n"
       "├── model4_career_matcher/        # full Sentence-Transformers folder\n"
       "└── model5_cv_ner/                # unzipped onnx/ contents\n"
       "```"),

    md("## 6. Definition of done (per model)\n\n"
       "- [x] **M1**: exported, quantized, parity-checked (§2 above) — the one "
       "section of this notebook that's actually complete\n"
       "- [ ] M3: needs a real Colab GPU training run first (sandbox only ran a "
       "CPU smoke test)\n"
       "- [ ] M4: needs training first (never run)\n"
       "- [ ] M5: needs training first (never run)\n"
       "- [ ] M2: needs training first (never run)\n"
       "- [ ] Local smoke test (see `export/local_smoke_test.py`) passes on the "
       "actual target laptop for every exported ONNX model, once each is ready"),
]

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"name": "python3", "display_name": "Python 3"},
    "accelerator": "GPU",
}

path = OUT_DIR / "export_and_quantize.ipynb"
with open(path, "w") as f:
    nbf.write(nb, f)
print(f"wrote {path}")
