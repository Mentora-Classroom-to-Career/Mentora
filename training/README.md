# MENTORA Model Training (Phase 4)

Five Colab notebooks, one per model, in `notebooks/`. Built from
`MENTORA_Phase4_Model_Training.md` and adapted to match the **actual**
schemas Phase 3's scripts produced (see `datasets/SOURCES.md`) — a few
column names and one dataset-availability gap (M5's silver set) differ
from the plan doc's assumed shapes; each notebook has a markdown cell
at the top calling out its specific adaptation.

| Notebook | Model | Status |
|---|---|---|
| `train_m3.ipynb` | Performance Trajectory Predictor (LSTM) | **Ran for real in this environment — see result below** |
| `train_m4.ipynb` | Career-Skill Semantic Matcher | Built, not run — needs `huggingface.co` (unreachable here) |
| `train_m1.ipynb` | Knowledge Gap Classifier | **Ran for real in Colab (by the user) — see result below** |
| `train_m5.ipynb` | CV/Transcript NER Extractor | Built, not run — needs `huggingface.co` |
| `train_m2.ipynb` | Adaptive Question Generator | Built, not run — needs `huggingface.co` |

M3 is the only model with zero pretrained-weight downloads (pure
from-scratch PyTorch), so it's the one notebook I could actually execute
end-to-end here rather than just write. The other four need real HF model
downloads and, for M2 especially, meaningful GPU time — genuinely a Colab
job, not a sandbox smoke test.

## M3 result — real run, not simulated

Ran `smoke_test/run_m3.py` (the notebook's exact training loop, extracted
to a standalone script) against the real Phase 3 dataset, on CPU:

- **hidden_size=128 (as specified):** plateaued at **val MAE ≈ 5.94** after
  30 epochs — just above the ≤5 target.
- **hidden_size=256 (the doc's first suggested fix for missing target):**
  plateaued at **val MAE ≈ 5.93–6.08** — no meaningful improvement.

**Diagnosis:** the plateau held steady across both model sizes, which
points away from model capacity and toward the synthetic data's own noise
floor. `generate_m3_synthetic.py` (Phase 3) adds `rng.normal(0, 4)` noise
on top of each trajectory's trend — a standard-deviation-4 Gaussian has a
mean absolute value of `4 * sqrt(2/pi) ≈ 3.2`, so roughly a third of the
"error" a perfect model would still show is irreducible noise baked into
the generator, before even accounting for one-step-ahead prediction
compounding that further. Getting under MAE 5 is achievable but tight
against that floor.

**Two real options, not yet tried (leaving this for the actual Colab run):**
1. Lower the noise amplitude in `generate_m3_synthetic.py` (e.g.
   `rng.normal(0, 3)`) and regenerate — makes the target easier and is
   defensible (real student scores are probably less noisy session-to-session
   than a std-4 draw implies).
2. Increase `SEQ_LEN` from 5 to 7-8 (the doc's other suggested fix) — more
   context per prediction should help more than raw model capacity did here,
   since the bottleneck looks like data noise, not underfitting.

Either is a 5-minute change + a 2-minute rerun in Colab — worth trying
before accepting MAE ~5.9 as final for the thesis.

## M1 result — real run in Colab, three issues found and fixed along the way

**Bug 1 (fixed in commit dd0a9b7):** `fp16=True` + DeBERTa-v3 threw
`ValueError: Attempting to unscale FP16 gradients` — a known
architecture-level incompatibility between DeBERTa-v3's disentangled
attention layer and PyTorch's `GradScaler`. Fixed by setting `fp16=False`.

**Symptom 2 (fixed in commit f4e4558):** with the fp16 fix applied, training
completed cleanly for 5 epochs, and validation loss fell steadily
(0.54 -> 0.27) — but `eval_f1_micro` stayed at **exactly 0.0 every single
epoch**. Diagnosed as a multi-label cold-start symptom (16 classes, ~1
positive label per example, ~59 training rows) and addressed with more
epochs (5 -> 30), a threshold-sweep diagnostic cell, and a class-weighted
`WeightedTrainer` (`pos_weight` in `BCEWithLogitsLoss`) to counter the
label sparsity.

**Bug 3 (my mistake, fixed in this commit):** the `WeightedTrainer` class
added in commit f4e4558 was defined but never actually used — the `Trainer(
...)` call two cells down was never updated to `WeightedTrainer(...)`, so
that whole fix was silently a no-op. Running the notebook as-is (all 30
epochs, unweighted loss, no LR warmup) produced a **collapsed model**: every
example's predicted probability came out nearly identical (~0.3157,
varying only in the 3rd decimal place) regardless of input text, and
`train_loss` blew up to 61.89 (vs. ~0.53 in the earlier 5-epoch run) —
consistent with training instability from a flat `learning_rate=2e-5` over
240 steps with no warmup on a tiny dataset. Fixed by:
1. Actually wiring `trainer = WeightedTrainer(...)` (the class was already
   correct, just unused).
2. Adding `warmup_ratio=0.1` and `weight_decay=0.01` to `TrainingArguments`
   to stabilize the longer run.
3. Adding a note about clearing stale checkpoints before re-running with
   different loss/hyperparameters — `resume_from_checkpoint` will otherwise
   silently resume optimizer/scheduler state from a run trained under
   different settings.
4. Rewriting the threshold-sweep "reading the result" guidance to check
   for a collapsed model (near-identical probability across all examples)
   *before* interpreting low F1 as "just needs a lower threshold" — those
   are different failure modes with different fixes.

Re-run the notebook with these fixes (delete old checkpoints first) and
check the max-probability line before anything else.

**Dataset swap (commit ab43dcc):** at this point the question bank itself
was replaced — 70 hand-authored questions -> 1,724 real questions sourced
from AQuA-RAT, ARC-derived science questions, RACE-C, and CLOTH (see
`datasets/SOURCES.md` for full provenance/licenses). M2's 500+ target was
met automatically as a result.

**Bug 4 (fixed in commit 5d512df):** re-running against the larger dataset
hit `ValueError: Can't find a valid checkpoint at .../checkpoint-96` — a
leftover checkpoint from before `num_train_epochs` changed 30 -> 15 (which
shifts the total step count). Also revealed `pos_weight` reaching **430**
for the thinnest classes (Calculus, Writing) on the real data. Fixed:
checkpoint discovery now sorts numerically by step (plain `sorted()` on
`checkpoint-*` paths is a latent bug — string sort puts `checkpoint-100`
before `checkpoint-96`), validates a checkpoint has the files needed to
actually resume before trusting it, and falls back to a fresh run
automatically instead of crashing. `pos_weight` capped at 20.0.

**Bug 5 (fixed in commit 324d30d):** next run hit DeBERTa-v3's
well-documented NaN-divergence issue: `eval_loss` fine at epoch 1 (10.76),
then `nan` from epoch 2 on; every predicted probability saturated to
exactly `1.0` (`RuntimeWarning: overflow encountered in exp`);
`eval_f1_micro` stuck at exactly 0.0. This is a known model-level
instability (PyTorch's default `adam_epsilon=1e-8` is too small for
DeBERTa-v3's numerics on smaller datasets) — fixed by raising
`adam_epsilon` to `1e-6` and lowering `learning_rate` to `1e-5`.

**Real result (this run):** clean training, no divergence, no collapse.
**`eval_f1_micro` peaked at 0.567 (epoch 1)**, then declined every epoch
after (0.297, 0.257, ..., down to ~0.15 by epoch 15) while validation loss
climbed the whole time — textbook overfitting on a dataset this size.
`load_best_model_at_end=True` correctly restored the epoch-1 checkpoint,
so the final reported metric (0.567) is real and not contaminated by the
overfit tail — but epochs 2-15 were wasted compute. Fixed by adding
`EarlyStoppingCallback(early_stopping_patience=3)` and a per-topic
`classification_report` cell (§6) to distinguish "uniformly weak" from
"a few genuinely thin topics dragging the average down" — the latter is
what `datasets/SOURCES.md`'s known imbalances (Calculus, Modern Physics,
Writing) would predict, and calls for a different fix than more epochs.

0.567 is real, meaningful progress from 0.0, but still short of the 0.85
target. Next run's per-topic breakdown should clarify whether closing that
gap needs more data for specific topics or something else.

## Running the rest (M4, M5, M2 — and re-running M1 with the fixes above)

1. Upload `datasets/` (the repo folder) to Google Drive at
   `MyDrive/mentora_data/` — matching Phase 3's storage convention.
2. Open each notebook in Colab, `Runtime -> Change runtime type -> T4 GPU`.
3. Run top to bottom: M4, then M5, then M2 (M1 already ran once — re-run
   it with the fp16 + epoch-count + threshold-sweep fixes above before
   trusting its metric; M3 already done too, but re-running in real
   Colab with a real GPU is still worth doing before Phase 5, since this
   sandbox result was CPU-only).
