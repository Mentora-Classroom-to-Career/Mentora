# M3 smoke test

`run_m3.py` — the train_m3.ipynb training loop extracted to a standalone
script, run against the real Phase 3 dataset on CPU (no Colab/Drive
dependency) to verify the pipeline end-to-end before trusting the notebook.

`run_m3_hidden256_variant.py` — same script with `hidden_size=256`, run to
test the notebook's own suggested fix when the ≤5 MAE target isn't hit.

Results and diagnosis: see `../README.md`.

Re-run either with: `pip install torch pandas numpy && python3 run_m3.py`
