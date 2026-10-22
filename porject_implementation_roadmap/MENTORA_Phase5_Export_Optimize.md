# MENTORA — Phase 5 Deep Dive: Export & Optimize for Local Inference

*Prerequisite: Phase 4 is done — all five trained models sit in
`mentora_models/model{1..5}_*/` on Google Drive, each meeting its target metric. This
phase converts them into a form that will actually run on your 8GB / 3rd-Gen i5
laptop, downloads them locally, and proves they load and run there before Phase 6
wires them into the backend. Estimated time: 2–4 days, plus however long the Drive
downloads take on your connection.*

---

## 0. Recap — what needs converting, and why (don't skip this if you skipped the master plan)

| Model | Needs ONNX + quantization? | Why |
|---|---|---|
| M3 (LSTM) | **No** | ~2MB, plain FP32 is already cheap enough to run always-resident |
| M4 (Sentence-Transformers) | **No** | ~420MB FP32 is acceptable always-resident; quantizing buys little and Sentence-Transformers' own PyTorch path is simplest to keep |
| M1 (DeBERTa Gap Classifier) | **Yes** | Full FP32 DeBERTa-base is the heaviest of the on-demand models; INT8 shrinks both RAM and load time meaningfully |
| M5 (BERT NER) | **Yes** | Same reasoning as M1 |
| M2 (FLAN-T5-Large + LoRA) | **Yes** | The single largest model in the project — quantization matters most here |

**Why ONNX Runtime quantization instead of PyTorch's built-in `torch.quantization.
quantize_dynamic()`:** PyTorch's default x86 backend (FBGEMM) hard-requires the AVX2
CPU instruction set, confirmed directly in PyTorch's own quantization docs and in
multiple open PyTorch GitHub issues showing `Illegal instruction (core dumped)`
crashes on non-AVX2 CPUs. Your 3rd-Gen i5 (Ivy Bridge) doesn't have AVX2 — Ivy Bridge
tops out at AVX. ONNX Runtime's quantization has portable kernel paths that don't
hard-fail without AVX2, so it's the safe choice here regardless of exactly how much
slower it ends up being than a hypothetical AVX2 machine.

**If you skipped the Phase 0 AVX2 check:** do it now, before spending time on ONNX
export, since if your CPU *does* have AVX2 (i.e., it's newer than "3rd-Gen" actually
implies), you can skip straight to PyTorch's own `quantize_dynamic` and this whole
phase gets simpler. On the machine in question: `cat /proc/cpuinfo | grep flags | grep -o avx2`.
Empty output confirms AVX2 is absent and this phase's ONNX path is the correct one.

---

## 1. Environment setup for export (do this in a fresh Colab notebook — export is a one-time job, doesn't need to live in the training notebooks)

```python
!pip install -q optimum-onnx onnxruntime onnx
# Note: as of 2026, HuggingFace moved its ONNX/ONNXRuntime integration out of core
# `optimum` into a separate `optimum-onnx` package — installing plain `optimum`
# alone will NOT give you the ORTModelForXxx classes used below.
```
Mount Drive the same way as Phase 4 (§0.1–0.2 there) — this notebook reads from
`mentora_models/` and writes ONNX/quantized versions alongside the originals.

---

## 2. M3 and M4 — no conversion needed, just package for download

### 2.1 M3
Already saved as `lstm_final.pt` in Phase 4. Nothing to do here except confirm it's
there and small:
```python
import os
path = '/content/drive/MyDrive/mentora_models/model3_trajectory_predictor/lstm_final.pt'
print(os.path.getsize(path) / 1e6, "MB")  # expect ~1-3MB
```

### 2.2 M4
Already saved as a full Sentence-Transformers folder by `model.fit(..., save_best_model=True)`
in Phase 4. Confirm the folder has the expected files (`config.json`,
`model.safetensors` or `pytorch_model.bin`, `tokenizer.json`, etc.) and note its size
(~420MB) — this is the largest thing that stays as plain PyTorch, and it's why the
master plan keeps it always-resident rather than lazy-loaded (loading it fresh on
every request would be slower than just keeping it in RAM).

**That's the entirety of this step for M3/M4** — both get downloaded as-is in §6.

---

## 3. M1 — DeBERTa Gap Classifier: export + quantize

### 3.1 Export to ONNX
```python
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer

M1_DIR = '/content/drive/MyDrive/mentora_models/model1_gap_classifier/final'
M1_ONNX_DIR = '/content/drive/MyDrive/mentora_models/model1_gap_classifier/onnx'

ort_model = ORTModelForSequenceClassification.from_pretrained(M1_DIR, export=True)
ort_model.save_pretrained(M1_ONNX_DIR)

tokenizer = AutoTokenizer.from_pretrained(M1_DIR)
tokenizer.save_pretrained(M1_ONNX_DIR)
```
`export=True` handles the `torch.onnx.export(...)` call, dynamic axes (so it isn't
locked to one fixed batch size or sequence length), and opset selection internally —
this is the current recommended path over hand-writing `torch.onnx.export` yourself.

### 3.2 Quantize (dynamic, INT8)
```python
from onnxruntime.quantization import quantize_dynamic, QuantType
import os

for onnx_file in os.listdir(M1_ONNX_DIR):
    if onnx_file.endswith('.onnx') and not onnx_file.endswith('_quantized.onnx'):
        in_path = f'{M1_ONNX_DIR}/{onnx_file}'
        out_path = f'{M1_ONNX_DIR}/{onnx_file.replace(".onnx", "_quantized.onnx")}'
        quantize_dynamic(in_path, out_path, weight_type=QuantType.QUInt8)
        print(f"{onnx_file}: {os.path.getsize(in_path)/1e6:.1f}MB -> {os.path.getsize(out_path)/1e6:.1f}MB")
```

### 3.3 Verify parity — PyTorch vs. ONNX vs. quantized ONNX, on the SAME input
Never skip this step. A model that exports "successfully" but silently produces
different predictions than the PyTorch original is worse than no export at all.
```python
import torch, numpy as np
from transformers import AutoModelForSequenceClassification
import onnxruntime as ort

text = "A quadratic equation has the form ax² + bx + c = 0."
inputs = tokenizer(text, return_tensors="pt")

# original PyTorch
pt_model = AutoModelForSequenceClassification.from_pretrained(M1_DIR)
pt_model.eval()
with torch.no_grad():
    pt_logits = pt_model(**inputs).logits.numpy()

# ONNX (unquantized)
session = ort.InferenceSession(f'{M1_ONNX_DIR}/model.onnx', providers=["CPUExecutionProvider"])
onnx_inputs = {k: v.numpy() for k, v in inputs.items()}
onnx_logits = session.run(None, onnx_inputs)[0]

# ONNX quantized
qsession = ort.InferenceSession(f'{M1_ONNX_DIR}/model_quantized.onnx', providers=["CPUExecutionProvider"])
q_logits = qsession.run(None, onnx_inputs)[0]

print("PyTorch vs ONNX max abs diff:", np.abs(pt_logits - onnx_logits).max())
print("PyTorch vs quantized ONNX max abs diff:", np.abs(pt_logits - q_logits).max())
# Unquantized ONNX should be nearly identical (diff < 1e-3).
# Quantized ONNX will differ more (INT8 is lossy by design) — a diff under ~0.1-0.3
# on typical logit scales is normal; if predicted LABELS flip on a batch of known
# examples, that's the real thing to check, not raw logit distance.
```
Run this on a batch of 15–20 known validation examples, not just one string, and
specifically check that the **predicted multi-label set** (after the sigmoid +
threshold from Phase 4 §4.2) matches between PyTorch and quantized-ONNX for most of
them — a handful of borderline flips near the 0.5 threshold is expected and fine; a
lot of them is a sign to re-check the quantization config.

---

## 4. M5 — BERT NER: export + quantize

Same three steps as M1, using the token-classification variant:

```python
from optimum.onnxruntime import ORTModelForTokenClassification

M5_DIR = '/content/drive/MyDrive/mentora_models/model5_cv_ner/final'
M5_ONNX_DIR = '/content/drive/MyDrive/mentora_models/model5_cv_ner/onnx'

ort_model = ORTModelForTokenClassification.from_pretrained(M5_DIR, export=True)
ort_model.save_pretrained(M5_ONNX_DIR)
tokenizer = AutoTokenizer.from_pretrained(M5_DIR)
tokenizer.save_pretrained(M5_ONNX_DIR)

# quantize — identical loop to §3.2, pointed at M5_ONNX_DIR
```
For parity-checking a token-classification model, compare the **per-token predicted
tag sequence** (after `argmax`, mapped through `id2tag`) between PyTorch and
quantized-ONNX on a handful of real resume snippets from the Phase 3 gold set —
entity boundaries matter more than exact logit values here.

---

## 5. M2 — FLAN-T5-Large + LoRA: export + quantize (the involved one)

### 5.1 Step 1 — merge the LoRA adapter into the base model first
ONNX export works on a plain model graph — it doesn't understand PEFT's adapter
injection mechanism. Merge the adapter's weights into the base model's weights
first, producing one ordinary (non-PEFT) T5 model:
```python
from peft import PeftModel
from transformers import T5ForConditionalGeneration, T5Tokenizer

base = T5ForConditionalGeneration.from_pretrained('google/flan-t5-large')
peft_model = PeftModel.from_pretrained(base, '/content/drive/MyDrive/mentora_models/model2_question_generator/lora_adapter')
merged = peft_model.merge_and_unload()   # bakes the LoRA delta into the base weights, returns a plain T5ForConditionalGeneration

MERGED_DIR = '/content/drive/MyDrive/mentora_models/model2_question_generator/merged'
merged.save_pretrained(MERGED_DIR)
tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-large')
tokenizer.save_pretrained(MERGED_DIR)
```
This merged folder is now full-size (~3GB in FP32) — that's expected and fine, it's
an intermediate artifact for this notebook only, not something you'll download to
the laptop (§6 downloads the *quantized* ONNX version instead, which is much
smaller).

### 5.2 Step 2 — export the merged model to ONNX (seq2seq — 3 graph parts)
```python
from optimum.onnxruntime import ORTModelForSeq2SeqLM

M2_ONNX_DIR = '/content/drive/MyDrive/mentora_models/model2_question_generator/onnx'
ort_model = ORTModelForSeq2SeqLM.from_pretrained(MERGED_DIR, export=True)
ort_model.save_pretrained(M2_ONNX_DIR)
tokenizer.save_pretrained(M2_ONNX_DIR)
```
A T5 export produces **three** ONNX graphs, not one: `encoder_model.onnx`,
`decoder_model.onnx`, and `decoder_with_past_model.onnx` (the last one is used for
fast autoregressive generation after the first token, reusing cached key/value
states). This is normal and expected — don't be surprised there isn't a single
`model.onnx` file for this one, unlike M1/M5.

### 5.3 Step 3 — quantize all three graph parts
```python
import os
for fname in os.listdir(M2_ONNX_DIR):
    if fname.endswith('.onnx') and '_quantized' not in fname:
        in_path = f'{M2_ONNX_DIR}/{fname}'
        out_path = f'{M2_ONNX_DIR}/{fname.replace(".onnx", "_quantized.onnx")}'
        quantize_dynamic(in_path, out_path, weight_type=QuantType.QUInt8)
        print(f"{fname}: {os.path.getsize(in_path)/1e6:.0f}MB -> {os.path.getsize(out_path)/1e6:.0f}MB")
```

### 5.4 Verify parity — compare GENERATED TEXT, not raw logits
For a generation model, the meaningful check is end-to-end output text, not
intermediate logits:
```python
prompt = "Generate a medium difficulty multiple-choice question about Algebra (Quadratic Equations)."
inputs = tokenizer(prompt, return_tensors="pt")

# original merged PyTorch model
pt_out = merged.generate(**inputs, max_new_tokens=128)
print("PyTorch:", tokenizer.decode(pt_out[0], skip_special_tokens=True))

# ONNX (quantized) — Optimum's ORTModelForSeq2SeqLM picks up specific files by name;
# point it at the quantized filenames explicitly
onnx_model = ORTModelForSeq2SeqLM.from_pretrained(
    M2_ONNX_DIR,
    encoder_file_name="encoder_model_quantized.onnx",
    decoder_file_name="decoder_model_quantized.onnx",
    decoder_with_past_file_name="decoder_with_past_model_quantized.onnx",
)
onnx_out = onnx_model.generate(**inputs, max_new_tokens=128)
print("Quantized ONNX:", tokenizer.decode(onnx_out[0], skip_special_tokens=True))
```
Run this across 10–15 varied prompts (different topics/difficulties). Expect close
but not always byte-identical output — check that the JSON structure from Phase 3's
`target_text` format (question/options/correct_answer keys) still comes out valid and
parseable (`json.loads(...)`) for the quantized version on every prompt; a few word
choices differing is fine, a broken JSON shape is not.

---

## 6. Downloading everything from Drive to the local laptop

### 6.1 What to actually download
Not the intermediate artifacts — only the final, right-sized pieces:
- `model1_gap_classifier/onnx/*_quantized.onnx` + tokenizer files
- `model2_question_generator/onnx/*_quantized.onnx` + tokenizer files (**not** the
  `merged/` FP32 folder — that was intermediate-only, ~3GB you don't need locally)
- `model3_trajectory_predictor/lstm_final.pt`
- `model4_career_matcher/` (the full Sentence-Transformers folder, as-is)
- `model5_cv_ner/onnx/*_quantized.onnx` + tokenizer files

### 6.2 Zip each model folder in Colab, download the zips
```python
import shutil
for name in ['model1_gap_classifier/onnx', 'model2_question_generator/onnx',
             'model3_trajectory_predictor', 'model4_career_matcher', 'model5_cv_ner/onnx']:
    src = f'/content/drive/MyDrive/mentora_models/{name}'
    zip_name = name.replace('/', '_')
    shutil.make_archive(f'/content/{zip_name}', 'zip', src)
```
Then use Colab's file browser (left sidebar → folder icon) to right-click each `.zip`
in `/content/` and Download — or, simpler for larger files, just install the Google
Drive desktop app on the laptop and let it sync `mentora_models/` directly, which
avoids the manual zip/download/unzip cycle entirely and is the recommended approach
if the laptop's internet connection can handle syncing a few GB.

### 6.3 Place them into the backend's expected folder structure
Per the master plan's §9 folder layout:
```
mentora/ml_models/saved/
├── model1_gap_classifier/        # unzipped onnx/ contents — the *_quantized.onnx + tokenizer files
├── model2_question_generator/    # unzipped onnx/ contents — 3 quantized .onnx graphs + tokenizer files
├── model3_trajectory_predictor/  # lstm_final.pt
├── model4_career_matcher/        # the full Sentence-Transformers folder
└── model5_cv_ner/                # unzipped onnx/ contents
```

---

## 7. Local smoke test — the moment of truth for the AVX2 question

Before Phase 6 wires any of this into FastAPI, run one small local script per
ONNX model to confirm it loads and runs *at all* on this specific CPU — this is
exactly where an AVX2-related crash would surface, so catching it here (isolated,
easy to debug) beats catching it for the first time inside a FastAPI request handler.

```python
# local_smoke_test.py — run this ON THE LAPTOP, not in Colab
import onnxruntime as ort
import numpy as np
from transformers import AutoTokenizer

print("ONNX Runtime version:", ort.__version__)
print("Available providers:", ort.get_available_providers())
# Expect to see "CPUExecutionProvider" — that's the only one you need locally

MODEL_DIR = "ml_models/saved/model1_gap_classifier"
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
session = ort.InferenceSession(f"{MODEL_DIR}/model_quantized.onnx", providers=["CPUExecutionProvider"])

inputs = tokenizer("A quadratic equation has the form ax squared plus bx plus c equals zero.", return_tensors="np")
onnx_inputs = {k: v for k, v in inputs.items() if k in [i.name for i in session.get_inputs()]}
outputs = session.run(None, onnx_inputs)
print("Inference succeeded. Output shape:", outputs[0].shape)
```
Run the equivalent for M5's quantized ONNX file and for M2's three quantized graphs
(loaded together via `ORTModelForSeq2SeqLM.from_pretrained`, same as §5.4, but now
pointed at the local `ml_models/saved/model2_question_generator/` folder). **If any
of these crash with `Illegal instruction` or a similar low-level error, that's the
FBGEMM/AVX2 issue reappearing somewhere it shouldn't** — since ONNX Runtime's
quantization is supposed to be the portable path; if it still happens, check that
`onnxruntime` (not `onnxruntime-gpu` or a wheel built with vendor-specific extras)
is what's actually installed locally, and that the ONNX opset used during export
(§3.1's `export=True` default) isn't newer than what your installed `onnxruntime`
version supports — mismatched opset versions are the other common cause of a load
failure that looks superficially similar to a hardware incompatibility.

---

## 8. Size and latency benchmark (fill this in on the real laptop — numbers vary by machine)

| Model | FP32 size | Quantized size | Load time (cold) | One inference (CPU) |
|---|---|---|---|---|
| M1 | ~370MB | ~ (fill in) | — | — |
| M2 (3 graphs combined) | ~3GB | ~ (fill in) | — | — |
| M3 | ~2MB (no quant) | n/a | — | — |
| M4 | ~420MB (no quant) | n/a | — | — |
| M5 | ~420MB | ~ (fill in) | — | — |

Time each with a simple `time.perf_counter()` wrapper around the load call and one
`session.run(...)` call, on the actual laptop, not Colab — Colab's CPU is a different
(and much faster) chip than the one this app has to run on, so a Colab-measured
"fast" load time tells you nothing about the real deployment target.

---

## 9. Common pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| `ImportError: cannot import name 'ORTModelForSequenceClassification'` | Installed `optimum` instead of `optimum-onnx` (§1) | `pip install optimum-onnx`, not plain `optimum` |
| Quantized model's predictions look totally different, not just slightly off | Applied quantization to the wrong file, or quantized an already-quantized file | Re-run §3.2/§4/§5.3 from a clean re-export; always diff against the *original* FP32 ONNX, not a previous quantized attempt |
| M2 export fails partway with a shape/dtype error | Tried exporting directly from the `PeftModel` instead of the merged model | Always call `merge_and_unload()` first (§5.1) — ONNX export doesn't understand LoRA's adapter layers |
| Local smoke test (§7) throws `Illegal instruction` | `onnxruntime` package variant mismatch, or a genuinely unsupported opset for this ONNX Runtime version | Confirm plain `pip install onnxruntime` (CPU build) locally, not a GPU/vendor build; check the export opset against your installed `onnxruntime`'s supported range |
| Drive download is extremely slow / keeps failing | Trying to download many GB over a slow connection in one browser download | Use the Google Drive desktop sync client instead of manual zip-and-download (§6.2) — it resumes automatically on interruption |
| M2's generated JSON is malformed after quantization on some prompts | INT8 quantization degraded output quality more than expected for this specific model | Try `weight_type=QuantType.QInt8` instead of `QUInt8` as an alternative, or fall back to leaving M2's *decoder* graphs unquantized while quantizing only the encoder (partial quantization is a legitimate compromise if full quantization visibly hurts output quality) |

---

## 10. Definition of done

- [ ] M1: `*_quantized.onnx` + tokenizer files in `ml_models/saved/model1_gap_classifier/`,
      parity-checked against PyTorch on 15+ examples
- [ ] M5: same, in `model5_cv_ner/`, parity-checked on real resume snippets
- [ ] M2: LoRA adapter merged, exported as 3 ONNX graphs, all 3 quantized, in
      `model2_question_generator/`, parity-checked via generated-text comparison on
      10+ prompts with valid-JSON output confirmed
- [ ] M3: `lstm_final.pt` in `model3_trajectory_predictor/`
- [ ] M4: full Sentence-Transformers folder in `model4_career_matcher/`
- [ ] §7's local smoke test passes for every ONNX model, **on the actual target
      laptop**, with no `Illegal instruction` or opset errors
- [ ] §8's benchmark table filled in with real local numbers

Once every row is checked, Phase 6 (Model Manager + real inference wiring) can begin
— it consumes exactly this `ml_models/saved/` folder structure, swapping Phase 2's
mock JSON responses for real calls into these files one endpoint at a time.
