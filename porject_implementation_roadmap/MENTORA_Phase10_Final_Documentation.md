# MENTORA — Phase 10 Deep Dive: Final Documentation

*Prerequisite: Phases 1–9 are functionally complete — this phase doesn't build new
features, it captures and presents everything already built, trained, tested, and
rehearsed. Much of this can be written in parallel with Phase 9's rehearsals rather
than strictly after them. Estimated time: 5–10 days, highly variable depending on
your program's specific thesis/report requirements.*

---

## 0. Why this phase carries real weight, not just paperwork

For a Final Year Project specifically, documentation and the defense/viva are
typically a substantial share of the actual grade — a working system with weak
documentation and a shaky defense often scores worse than a slightly-less-polished
system explained clearly and confidently. This phase is where the engineering work
from Phases 0–9 gets translated into something a supervisor/examiner can evaluate
without having to read your source code line by line.

---

## 1. Step 1 — map what you've built onto your program's expected thesis structure

Most CS FYP theses follow some variant of this structure. Map your actual phases onto
it now, so you know exactly what's missing before you start writing, not partway
through:

| Typical thesis chapter | Draws from |
|---|---|
| 1. Introduction, Problem Statement, Objectives | The six Fixed Objectives (master plan §1.1) — you likely already have most of this from your proposal |
| 2. Literature Review | Separate from this technical roadmap — your own prior research |
| 3. System Design & Architecture | Master plan §3 (architecture diagram), §4 (ERD), §5 (class diagram), §6 (activity diagram) |
| 4. Implementation | Phases 1–7's actual code, organized by subsystem (frontend, backend, ML models, RAG) |
| 5. Testing & Results | Phase 8's testing methodology + results, Phase 4's model metrics table |
| 6. Deployment & Hardware Considerations | §5 below — the AVX2/hardware-adaptation story, genuinely worth its own subsection |
| 7. Conclusion & Future Work | What's genuinely out of scope / left as future work — be honest about this rather than implying everything is perfect |

---

## 2. Step 2 — the System Design & Architecture chapter

### 2.1 Reuse, don't recreate
The ERD, class diagram, activity diagram, and system architecture diagram already
exist from earlier in this project's planning documents — this chapter is primarily
about presenting them clearly with supporting prose, not designing them from scratch
at this stage. If any of them drifted from what was actually implemented (common —
plans shift during implementation), **update the diagram to match reality**, not the
other way around; a diagram that doesn't match the actual code is worse than no
diagram, since it actively misleads a reader.

### 2.2 A concrete way to check for drift
```bash
# quick sanity check: does every table in your SQLAlchemy models.py
# actually appear in the ERD, and vice versa?
grep -E "^class \w+\(Base\)" backend/database/models.py
```
Cross-reference this list against the ERD's entity list by hand — this five-minute
check catches the single most common "documentation doesn't match implementation"
gap in student projects.

### 2.3 Auto-generate the API reference instead of hand-writing it
FastAPI already generates a complete, accurate OpenAPI spec from your actual route
definitions — hand-transcribing endpoint documentation into the thesis risks it going
stale the moment an endpoint changes. Export it once, near the end:
```bash
# with the backend running:
curl http://localhost:8000/openapi.json > docs/openapi.json
```
Screenshotting the live `/docs` (Swagger UI) page for the appendix is a legitimate,
fast, always-accurate approach — it's generated directly from your code's type hints
and docstrings, so it can never drift from the real API the way hand-written
documentation can.

### 2.4 Document the Model Manager and RAG design decisions, with reasoning
This is where Phase 6 §0 and Phase 7 §0's "why" explanations (not just the "what")
belong — a reader should come away understanding *why* lazy-loading with an LRU
policy was necessary (RAM constraint), *why* inference is serialized via a semaphore
(CPU/RAM constraint), and *why* RAG was chosen over direct LLM prompting (grounding
requirement, Objective 6). Reasoning like this is what separates a strong systems
chapter from a feature list.

---

## 3. Step 3 — the model results chapter

### 3.1 Pull directly from Phase 4's experiment log
You already built this table while training — it becomes the spine of this section
almost unchanged:

| Model | Target metric | Target value | Achieved | Met target? |
|---|---|---|---|---|
| M1 — Gap Classifier | F1 (micro) | ≥ 0.85 | *(fill in)* | |
| M2 — Question Generator | BLEU-4 / ROUGE-L | ≥ 0.35 / ≥ 0.50 | *(fill in)* | |
| M3 — Trajectory Predictor | MAE | ≤ 5 | *(fill in)* | |
| M4 — Career Matcher | Precision@5 | ≥ 0.75 | *(fill in)* | |
| M5 — CV NER Extractor | Entity F1 | ≥ 0.82 | *(fill in)* | |

### 3.2 Don't just report the final number — show the training curve
If you logged with Weights & Biases (Phase 4 §0.4), export the loss/metric curves
directly from its dashboard (free tier, no extra work needed — it's already been
tracking this the whole time). If you skipped W&B, a quick retroactive plot from
saved checkpoint metrics is still worth the half hour:
```python
import matplotlib.pyplot as plt
epochs = [1, 2, 3, 4, 5]
f1_scores = [0.61, 0.74, 0.80, 0.84, 0.86]  # from your actual training logs
plt.plot(epochs, f1_scores, marker='o')
plt.axhline(y=0.85, color='r', linestyle='--', label='Target (F1 ≥ 0.85)')
plt.xlabel('Epoch'); plt.ylabel('F1 (micro)'); plt.title('M1 Gap Classifier — Validation F1 by Epoch')
plt.legend(); plt.savefig('docs/figures/m1_training_curve.png', dpi=150)
```
A curve showing the model *converging toward* the target, not just a single final
number, is considerably more convincing evidence of a properly-trained model than the
number alone — and directly answers the natural viva question "how do you know it
wasn't overfit/underfit."

### 3.3 If any model missed its target
**Report it honestly, with the mitigation.** A model that landed at F1=0.81 against
an ≥0.85 target, with a clear explanation of why (dataset size limitations from Phase
3, a specific under-represented topic category) and what you'd do with more time/data,
is a stronger thesis section than silently only reporting metrics that happened to
clear the bar. Examiners generally read confident honesty about a limitation far
better than an implied claim that everything was perfect.

### 3.4 Qualitative examples, not just numbers
For M2 (question generation) and M5 (NER extraction) especially, include a few actual
input/output examples directly in the thesis (a real CV snippet → real extracted
entities; a real topic prompt → real generated MCQ) — numbers alone don't convey what
the model actually produces, and a reader forms a much clearer picture from 2–3 real
examples than from a metrics table alone.

---

## 4. Step 4 — the testing & results chapter

Pull directly from Phase 8:
- The testing pyramid (unit → integration → end-to-end) as your testing methodology
  description
- The 11-point functional checklist results
- The Locust load-testing findings — how many concurrent users the system handles on
  this specific hardware, and what happens beyond that (queuing vs. failure)
- The real RAM idle/peak numbers, presented as a table matching (and correcting)
  the master plan's originally-estimated §12 table — explicitly show the before
  (estimated) vs. after (measured) numbers; this is another place where the honesty
  of "here's what we estimated, here's what we actually measured, here's the delta"
  reads as more rigorous than only ever showing final numbers
- The security checklist from Phase 8 §8

---

## 5. Step 5 — the hardware-adaptation story, as its own subsection (this is a genuine strength — don't bury it)

This deserves deliberate, structured treatment, not a passing mention, because it's a
concrete example of real engineering problem-solving under constraints — exactly what
a defense panel wants to see evidence of, distinct from "followed a tutorial."
Structure it like a small case study:

**5.1 Problem statement.** The target deployment hardware (3rd-Gen Intel i5, 8GB RAM,
no Docker, no GPU) is significantly more constrained than typical ML deployment
targets, and doesn't support the AVX2 instruction set that PyTorch's default x86
quantization backend (FBGEMM) requires.

**5.2 Investigation.** How this was discovered/confirmed (checking the CPU's
instruction set flags, cross-referencing against PyTorch's own documentation and
GitHub issue tracker for confirmed `Illegal instruction` crash reports on non-AVX2
hardware) — this is Phase 0 §0.2 and Phase 5 §0, condensed.

**5.3 Solution.** Exporting the three heaviest models to ONNX format and using ONNX
Runtime's own dynamic quantization instead, which doesn't share FBGEMM's AVX2
dependency — Phase 5 in full.

**5.4 Verification.** The parity-checking methodology (Phase 5 §3.3, §5.4) that
confirmed the ONNX/quantized models produce equivalent output to the original PyTorch
models before trusting them in production, plus the local smoke test (Phase 5 §7)
that confirmed the solution actually works on the real target hardware, not just in
theory.

**5.5 Outcome + broader lessons.** What this enabled (a fully local, no-internet-
required, no-Docker, low-RAM deployment of five transformer-based models) and what it
generalizes to (a reusable pattern for deploying ML models to resource-constrained
hardware more broadly — a genuinely citable, generalizable contribution, not just a
one-off fix specific to this project).

This structure — problem, investigation, solution, verification, outcome — is also
directly reusable as an answer template if a viva question probes this area, which,
given how central it was to the whole deployment design, it plausibly will.

---

## 6. Step 6 — code documentation (lighter touch, still worth doing properly)

### 6.1 Docstring pass
Not every function needs an essay, but every module's top-level purpose and every
non-obvious function (the Model Manager, the RAG pipeline classes, the BIO-decode
logic from Phase 6 §3.4 especially) should have a clear docstring explaining *why*,
not just restating the function name in prose.

### 6.2 Auto-generate a reference doc from those docstrings (free, low effort for the payoff)
```bash
pip install pdoc
pdoc backend/ml_core backend/services backend/routers -o docs/code_reference/
```
This produces a browsable static HTML reference site directly from your actual
docstrings — worth doing once near the end rather than maintaining a hand-written
code reference throughout, since it can never drift from the real code.

### 6.3 A per-subsystem README
One short `README.md` inside `backend/`, `frontend/`, `ml_models/`, and
`rag_pipelines/` each — a paragraph of "what this folder is, how it fits into the
whole system, where to look first" — genuinely useful for anyone (including future-
you, months later) opening the repo cold.

---

## 7. Step 7 — the consolidated setup/user guide

Distinct from Phase 9 §8's brief top-level README — this is the fuller "how to set
this project up from absolute zero" document, useful both as a thesis appendix and
as the actual practical reference if anyone (a supervisor, a future maintainer) needs
to reproduce your environment:
- Every install step from Phase 0 §0.4 and Phase 6–7's environment setup sections,
  consolidated into one linear sequence
- Every `.env` variable, what it does, and what a safe default/example value looks
  like
- How to (re)run Phase 3's data prep, Phase 4's training, and Phase 5's export, for
  anyone who needs to reproduce the ML pipeline rather than just run the pre-trained
  models

---

## 8. Step 8 — viva/defense preparation

Build a question bank from the genuine design decisions made across all ten phases —
these are exactly the kind of "why did you choose X over Y" questions a defense
panel tends to ask, and you already have real, specific answers for every one of
them from this project's own history:

| Likely question | Where your answer comes from |
|---|---|
| Why Next.js over plain React or static HTML? | Master plan §0.1 — httpOnly cookie auth security specifically |
| Why FastAPI over Django/Flask? | Async support relevant to the Model Manager's design (Phase 6) |
| Why ONNX Runtime instead of standard PyTorch quantization? | §5 above — the AVX2/FBGEMM story |
| Why LoRA for M2 instead of full fine-tuning? | Phase 4 §6.1 — Colab free-tier VRAM constraint |
| Why RAG instead of just prompting the LLM directly? | Phase 7 §0 — grounding requirement, Objective 6 |
| How do you prevent the app from running out of RAM? | Phase 6's Model Manager design + Phase 8's measured numbers |
| How did you validate your models actually work, not just that they loaded? | Phase 5's parity checks + Phase 4's metrics + Phase 8's integration tests |
| What would you do differently with more time/resources? | An honest, specific answer here (bigger datasets, a newer target CPU, cloud-hosted inference) reads far better than "nothing" |

Rehearse answering these out loud, not just having them written down — the gap
between "I know this" and "I can explain this clearly under mild pressure" is real
and worth closing before the actual defense.

---

## 9. Step 9 — final repository hygiene

- Confirm no secrets ever got committed: `git log -p | grep -i "SECRET\|PASSWORD\|API_KEY"`
  across the full history, not just the current state — a secret committed once and
  later removed from the current files is still sitting in git history unless
  actively scrubbed.
- A proper root `LICENSE` file if your program/institution expects one for submitted
  code.
- A final, clean commit history message pass if individual commits were messy during
  active development — not mandatory, but a clean `git log` reads as more
  professional if an examiner looks at it.
- Tag the final submitted state clearly: `git tag v1.0-final && git push --tags`.

---

## 10. Common pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| Diagrams in the thesis don't match the actual running system | Plans changed during implementation, diagrams never updated | Do the §2.2 drift-check explicitly, update diagrams to match reality, not the reverse |
| Metrics table has gaps/placeholder text at submission time | Phase 4's experiment log wasn't kept up to date during training | Go back to saved training logs/checkpoints now — the numbers are still recoverable even if the log wasn't maintained live |
| Viva answers feel rehearsed-but-shallow when a follow-up question comes | Memorized the answer without understanding the underlying reasoning | Practice explaining the *reasoning* (§5's structure), not a memorized sentence — the reasoning generalizes to follow-up questions, a memorized line doesn't |
| API reference in the appendix is already stale by submission day | Manually transcribed instead of exported from the live OpenAPI spec | Regenerate via §2.3 as one of the very last steps before submission, not early in this phase |
| Documentation describes features that quietly got cut/simplified along the way | Common in any real project — scope shifts | One honest "Limitations & Future Work" paragraph naming these directly is far better than documentation that implies a feature is fully live when it isn't |

---

## 11. Definition of done — final submission checklist

- [ ] All thesis chapters mapped and drafted (§1)
- [ ] Architecture diagrams verified against actual implementation, not just against
      the original plan (§2.2)
- [ ] API reference exported from the live, real OpenAPI spec (§2.3)
- [ ] Model results table filled in with real achieved numbers, training curves
      included, any missed targets honestly reported with mitigation discussion (§3)
- [ ] Testing & results chapter written from Phase 8's actual findings, including the
      estimated-vs-measured RAM comparison
- [ ] The hardware-adaptation story written up as its own structured subsection (§5)
- [ ] Code documentation pass done, `pdoc` reference generated (§6)
- [ ] Consolidated setup guide written and — ideally — tested by literally following
      it on a clean environment once
- [ ] Viva question bank built and rehearsed out loud, not just written (§8)
- [ ] Repository hygiene pass complete — no leaked secrets in history, license/tag in
      place (§9)

With this checked off, the project is complete end to end: designed, built, trained,
optimized for real constrained hardware, tested, rehearsed, and documented — every
phase from the original master plan closed out.
