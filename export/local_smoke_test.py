"""
Local smoke test — run this ON THE LAPTOP (the 3rd-Gen i5, 8GB RAM, no
AVX2 machine), NOT in Colab. This is where an AVX2-related crash would
actually surface, so catching it here (isolated, easy to debug) beats
catching it for the first time inside a FastAPI request handler.

Only tests M1 right now, since it's the only model actually trained +
exported as of this writing. Extend the MODELS list below as M3/M4/M5/M2
get trained and exported (M3 and M4 don't need ONNX at all — see
export/notebooks/export_and_quantize.ipynb §1 — so they need a different,
simpler check, not this ONNX-specific one).

Run: python3 local_smoke_test.py
"""
import sys
import time
from pathlib import Path

try:
    import onnxruntime as ort
except ImportError:
    print("onnxruntime not installed. Run: pip install onnxruntime")
    sys.exit(1)

try:
    from transformers import AutoTokenizer
except ImportError:
    print("transformers not installed. Run: pip install transformers")
    sys.exit(1)


SAVED_DIR = Path(__file__).parent.parent / "ml_models" / "saved"

# (folder name, sample input text) — add M5 here once it's trained/exported;
# M2 needs a different loader (ORTModelForSeq2SeqLM, 3 graphs) — see the
# export notebook §4's parity-check cell for that pattern instead.
MODELS_TO_TEST = [
    ("model1_gap_classifier", "A quadratic equation has the form ax squared plus bx plus c equals zero."),
]


def test_model(name: str, sample_text: str) -> bool:
    model_dir = SAVED_DIR / name
    quantized_files = list(model_dir.glob("*_quantized.onnx"))

    if not model_dir.exists():
        print(f"[{name}] SKIP — folder not found at {model_dir}")
        return True  # not a failure, just not deployed yet
    if not quantized_files:
        print(f"[{name}] FAIL — folder exists but no *_quantized.onnx file inside")
        return False

    onnx_path = quantized_files[0]
    print(f"[{name}] loading {onnx_path.name}...")

    t0 = time.perf_counter()
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    except Exception as e:
        print(f"[{name}] FAIL — load raised: {type(e).__name__}: {e}")
        if "Illegal instruction" in str(e) or "illegal instruction" in str(e).lower():
            print("  -> This is the AVX2/FBGEMM issue reappearing. Confirm plain 'onnxruntime'")
            print("     (not onnxruntime-gpu or a vendor build) is installed.")
        return False
    load_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    try:
        inputs = tokenizer(sample_text, return_tensors="np")
        input_names = {i.name for i in session.get_inputs()}
        onnx_inputs = {k: v for k, v in inputs.items() if k in input_names}
        outputs = session.run(None, onnx_inputs)
    except Exception as e:
        print(f"[{name}] FAIL — inference raised: {type(e).__name__}: {e}")
        return False
    infer_time = time.perf_counter() - t0

    print(f"[{name}] OK — load {load_time:.2f}s, inference {infer_time:.3f}s, output shape {outputs[0].shape}")
    return True


def main():
    print("ONNX Runtime version:", ort.__version__)
    print("Available providers:", ort.get_available_providers())
    print("(Expect to see 'CPUExecutionProvider' — that's the only one needed locally)\n")

    results = [test_model(name, text) for name, text in MODELS_TO_TEST]

    print()
    if all(results):
        print("All tested models passed.")
    else:
        print("One or more models FAILED — see above. Do not proceed to Phase 6 wiring")
        print("for a failing model until this is resolved.")
        sys.exit(1)


if __name__ == "__main__":
    main()
