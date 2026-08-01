# MENTORA Phase 5 — Export & Optimize

Converts trained models into a form that actually runs on the target laptop
(3rd-Gen i5, 8GB RAM, no AVX2). Built from `MENTORA_Phase5_Export_Optimize.md`.

## Status

| Model | Trained? | Export needed? | Ready to run this notebook's section? |
|---|---|---|---|
| M1 (DeBERTa Gap Classifier) | **Yes** (eval_f1_micro 0.731) | Yes — ONNX + INT8 quantize | **Yes** |
| M3 (LSTM Trajectory Predictor) | Partial (CPU smoke test only, in `training/smoke_test/`) | No | Needs a real Colab GPU run first |
| M4 (Sentence-Transformers Career Matcher) | No | No | Needs training first |
| M5 (BERT NER) | No | Yes — ONNX + INT8 quantize | Needs training first |
| M2 (FLAN-T5 + LoRA Question Generator) | No | Yes — ONNX + INT8 quantize (3 graphs) | Needs training first |

**Only M1's section of `notebooks/export_and_quantize.ipynb` is actually
runnable right now.** The notebook covers all 5 models per the Phase 5 spec,
but the M3/M4/M5/M2 sections will fail with a `FileNotFoundError` against
Drive until those are trained — that's expected, the notebook says so
inline at each section.

## I could not execute or verify any of this myself

Unlike M3's LSTM (Phase 4), ONNX export needs `optimum-onnx`, real trained
model weights on your Drive, and (for M1/M5) Hugging Face model access —
none of which are available in the build environment used to write this.
This notebook is built faithfully from the Phase 5 spec and cross-checked
against the actual folder paths/schemas this project uses, but **you are
the first one to actually run it.** Send me the output (or any error) the
same way you did for the training notebooks, and I'll debug it the same way.

## Running it

1. Open `notebooks/export_and_quantize.ipynb` in Colab (T4 GPU not strictly
   required for export/quantization, but doesn't hurt).
2. Run §0-§2 now (M1 is ready). Skip §3-§4 until M5/M2 are trained.
3. Download the quantized ONNX files per §5, place into
   `mentora/ml_models/saved/<model_name>/` per the folder structure shown there.
4. Run `local_smoke_test.py` **on the actual target laptop**, not Colab —
   this is where an AVX2-related crash would surface, so catching it here
   (isolated, easy to debug) beats catching it inside a FastAPI request
   handler in Phase 6.

```bash
cd export
python3 local_smoke_test.py
```

Currently only tests M1 (the only model with something to test). Extend
`MODELS_TO_TEST` in that script as M5 gets trained + exported; M2 needs a
different loader pattern (see the export notebook §4's parity-check cell)
since it's 3 ONNX graphs, not 1.
