# MENTORA — Phase 6 Deep Dive: Model Manager + Real Inference Wiring

*Prerequisite: Phase 5 is done — `ml_models/saved/` on the laptop has all five models
in their final local-inference form, and the local smoke test (Phase 5 §7) passed for
every ONNX model on this specific machine. Phase 2's backend still returns mock JSON
for exam/career/dashboard endpoints. This phase builds the piece that sits between
FastAPI and the five model files — the Model Manager — and then swaps the mock
responses for real ones, one endpoint at a time. Estimated time: 4–7 days.*

---

## 0. What this phase actually has to solve, before writing any code

Three real constraints, all from earlier phases, all converging here:

1. **RAM.** All five models loaded at once would blow past 8GB (master plan §12). So
   models can't just be `import`ed at module level and left resident — most of them
   need to load on first use and unload after a few idle minutes.
2. **The event loop.** ONNX Runtime inference and Sentence-Transformers `.encode()`
   calls are CPU-bound, blocking, synchronous Python calls. If one of these runs
   directly inside an `async def` FastAPI route, it **freezes the entire server** for
   every other request until it finishes — on this CPU, that could be several
   seconds. This has to be explicitly worked around, not left to chance.
3. **Concurrency across models, not just within one.** Nothing stops two different
   requests from triggering M1 and M2 inference at the same instant (a student
   submits an MCQ while another tab is mid-question-generation). Per the master
   plan's peak-RAM math, that's the scenario to actively prevent.

The **Model Manager** is the one piece of code responsible for all three. Everything
below builds up to it, then wires it in.

---

## 1. One decision to make before step 1: how many Uvicorn workers?

**Run exactly one.** `uvicorn main:app --port 8000` (no `--workers` flag, or
explicitly `--workers 1`). This matters more than it looks like it should: each
additional Uvicorn worker is a **separate OS process** with its own Python memory
space — a second worker doesn't share the first worker's loaded models, it loads its
own second copy of whatever's resident. On a machine where the whole point of the
Model Manager is keeping RAM under control, two workers would silently double
everything it's trying to save. Single-worker also happens to be the only sane choice
given only 4 CPU cores (Ivy Bridge i5) exist to share in the first place.

---

## 2. Step 1 — a common interface for every model

Different models need different loading code (an ONNX `InferenceSession` vs. a
`SentenceTransformer` vs. a raw `nn.Module`), but the Model Manager shouldn't need to
know those details — it should just be able to call `.load()`, `.infer(...)`, and
`.unload()` on anything it's managing.

```python
# backend/ml_core/base.py
from abc import ABC, abstractmethod
from typing import Any

class LoadableModel(ABC):
    """Common interface every model wrapper implements, so ModelManager
    can treat all five models uniformly."""

    #: True for models that stay loaded permanently (M3, M4)
    always_resident: bool = False

    def __init__(self):
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @abstractmethod
    def load(self) -> None:
        """Load weights into memory. Must be idempotent — calling load()
        on an already-loaded model should be a cheap no-op."""
        ...

    @abstractmethod
    def infer(self, *args, **kwargs) -> Any:
        """Run one inference call. Assumes load() has already happened —
        ModelManager guarantees this, this method doesn't need to check."""
        ...

    @abstractmethod
    def unload(self) -> None:
        """Free the memory. Must be safe to call even if never loaded."""
        ...
```

---

## 3. Step 2 — implement each of the five model wrappers

### 3.1 M3 — LSTM Trajectory Predictor (always resident)
```python
# backend/ml_core/models/trajectory_predictor.py
import torch
import torch.nn as nn
from ml_core.base import LoadableModel

class TrajectoryLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)

class TrajectoryPredictor(LoadableModel):
    always_resident = True

    def __init__(self, weights_path: str):
        super().__init__()
        self.weights_path = weights_path
        self.model: TrajectoryLSTM | None = None

    def load(self) -> None:
        if self._loaded:
            return
        self.model = TrajectoryLSTM()
        self.model.load_state_dict(torch.load(self.weights_path, map_location="cpu"))
        self.model.eval()
        self._loaded = True

    def infer(self, recent_scores: list[float]) -> dict:
        # recent_scores: last SEQ_LEN=5 session scores, oldest first
        x = torch.tensor(recent_scores, dtype=torch.float32).view(1, -1, 1)
        with torch.no_grad():
            predicted_next = self.model(x).item()
        trend = "declining" if predicted_next < recent_scores[-1] - 3 else \
                "improving" if predicted_next > recent_scores[-1] + 3 else "stable"
        return {"predicted_next_score": round(predicted_next, 1), "trend": trend}

    def unload(self) -> None:
        pass  # never actually called for an always_resident model — see §4.2
```

### 3.2 M4 — Career-Skill Matcher (always resident)
```python
# backend/ml_core/models/career_matcher.py
from sentence_transformers import SentenceTransformer
import numpy as np
from ml_core.base import LoadableModel

class CareerMatcher(LoadableModel):
    always_resident = True

    def __init__(self, model_path: str, career_catalogue: list[dict]):
        super().__init__()
        self.model_path = model_path
        self.career_catalogue = career_catalogue  # [{"title": ..., "text": "<skills description>"}]
        self.model: SentenceTransformer | None = None
        self.career_embeddings: np.ndarray | None = None

    def load(self) -> None:
        if self._loaded:
            return
        self.model = SentenceTransformer(self.model_path)
        # Precompute career embeddings ONCE at load time, not per request —
        # the catalogue (50+ careers) doesn't change between requests.
        self.career_embeddings = self.model.encode(
            [c["text"] for c in self.career_catalogue], convert_to_numpy=True
        )
        self._loaded = True

    def infer(self, skill_text: str, top_k: int = 5) -> list[dict]:
        skill_emb = self.model.encode(skill_text, convert_to_numpy=True)
        sims = self.career_embeddings @ skill_emb / (
            np.linalg.norm(self.career_embeddings, axis=1) * np.linalg.norm(skill_emb) + 1e-9
        )
        top_idx = np.argsort(-sims)[:top_k]
        return [
            {"career_title": self.career_catalogue[i]["title"], "match_score": round(float(sims[i]) * 100)}
            for i in top_idx
        ]

    def unload(self) -> None:
        pass  # never called — always_resident
```

### 3.3 M1 — Gap Classifier (on-demand, ONNX)
```python
# backend/ml_core/models/gap_classifier.py
import onnxruntime as ort
import numpy as np
from transformers import AutoTokenizer
from ml_core.base import LoadableModel

class GapClassifier(LoadableModel):
    always_resident = False

    def __init__(self, model_dir: str, topics: list[str], threshold: float = 0.5):
        super().__init__()
        self.model_dir = model_dir
        self.topics = topics
        self.threshold = threshold
        self.session: ort.InferenceSession | None = None
        self.tokenizer = None

    def load(self) -> None:
        if self._loaded:
            return
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.session = ort.InferenceSession(
            f"{self.model_dir}/model_quantized.onnx", providers=["CPUExecutionProvider"]
        )
        self._loaded = True

    def infer(self, question_text: str) -> list[str]:
        inputs = self.tokenizer(question_text, return_tensors="np", truncation=True,
                                 padding="max_length", max_length=128)
        onnx_inputs = {k: v for k, v in inputs.items() if k in [i.name for i in self.session.get_inputs()]}
        logits = self.session.run(None, onnx_inputs)[0][0]
        probs = 1 / (1 + np.exp(-logits))  # sigmoid — multi-label, not softmax
        return [self.topics[i] for i, p in enumerate(probs) if p >= self.threshold]

    def unload(self) -> None:
        self.session = None
        self.tokenizer = None
        self._loaded = False
```

### 3.4 M5 — CV NER Extractor (on-demand, ONNX)
```python
# backend/ml_core/models/cv_ner.py
import onnxruntime as ort
import numpy as np
from transformers import AutoTokenizer
from ml_core.base import LoadableModel

class CVNerExtractor(LoadableModel):
    always_resident = False

    def __init__(self, model_dir: str, id2tag: dict[int, str]):
        super().__init__()
        self.model_dir = model_dir
        self.id2tag = id2tag
        self.session = None
        self.tokenizer = None

    def load(self) -> None:
        if self._loaded:
            return
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.session = ort.InferenceSession(
            f"{self.model_dir}/model_quantized.onnx", providers=["CPUExecutionProvider"]
        )
        self._loaded = True

    def infer(self, cv_text: str) -> dict:
        # long CVs may exceed the model's max length — chunk on paragraph breaks
        chunks = self._chunk(cv_text, max_chars=800)
        entities = {"SKILL": set(), "CERT": set(), "GPA": set(), "PROJECT": set()}

        for chunk in chunks:
            tokens = chunk.split()
            enc = self.tokenizer(tokens, is_split_into_words=True, return_tensors="np",
                                  truncation=True, padding="max_length", max_length=256)
            onnx_inputs = {k: v for k, v in enc.items() if k in [i.name for i in self.session.get_inputs()]}
            logits = self.session.run(None, onnx_inputs)[0][0]
            preds = np.argmax(logits, axis=-1)
            self._decode_bio(tokens, preds, enc, entities)

        return {
            "skills": sorted(entities["SKILL"]),
            "certifications": sorted(entities["CERT"]),
            "gpa": next(iter(entities["GPA"]), None),
            "projects": sorted(entities["PROJECT"]),
        }

    def _chunk(self, text: str, max_chars: int) -> list[str]:
        paras = text.split("\n\n")
        chunks, current = [], ""
        for p in paras:
            if len(current) + len(p) > max_chars:
                chunks.append(current); current = p
            else:
                current += "\n\n" + p
        if current:
            chunks.append(current)
        return chunks or [text[:max_chars]]

    def _decode_bio(self, tokens, preds, enc, entities: dict[str, set]):
        word_ids = enc.word_ids(batch_index=0) if hasattr(enc, "word_ids") else None
        current_entity, current_type = [], None
        for i, pred_id in enumerate(preds):
            tag = self.id2tag[int(pred_id)]
            if tag.startswith("B-"):
                if current_entity:
                    entities.setdefault(current_type, set()).add(" ".join(current_entity))
                current_type = tag[2:]
                current_entity = [tokens[word_ids[i]]] if word_ids else []
            elif tag.startswith("I-") and current_type == tag[2:]:
                if word_ids:
                    current_entity.append(tokens[word_ids[i]])
            else:
                if current_entity:
                    entities.setdefault(current_type, set()).add(" ".join(current_entity))
                current_entity, current_type = [], None
        if current_entity:
            entities.setdefault(current_type, set()).add(" ".join(current_entity))

    def unload(self) -> None:
        self.session = None
        self.tokenizer = None
        self._loaded = False
```
This BIO-decode helper is the fiddliest bit of code in the whole backend — test it in
isolation (§8) before wiring it into an endpoint, using the exact gold-set CVs from
Phase 3/5 where you already know the right answer.

### 3.5 M2 — Question Generator (on-demand, ONNX seq2seq)
```python
# backend/ml_core/models/question_generator.py
import json
from optimum.onnxruntime import ORTModelForSeq2SeqLM
from transformers import AutoTokenizer
from ml_core.base import LoadableModel

class QuestionGenerator(LoadableModel):
    always_resident = False

    def __init__(self, model_dir: str):
        super().__init__()
        self.model_dir = model_dir
        self.model = None
        self.tokenizer = None

    def load(self) -> None:
        if self._loaded:
            return
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = ORTModelForSeq2SeqLM.from_pretrained(
            self.model_dir,
            encoder_file_name="encoder_model_quantized.onnx",
            decoder_file_name="decoder_model_quantized.onnx",
            decoder_with_past_file_name="decoder_with_past_model_quantized.onnx",
        )
        self._loaded = True

    def infer(self, topic: str, difficulty: str = "medium", max_retries: int = 2) -> dict:
        prompt = f"Generate a {difficulty} difficulty multiple-choice question about {topic}. Respond as JSON with keys: question, options (A-D), correct_answer."
        inputs = self.tokenizer(prompt, return_tensors="pt")

        for attempt in range(max_retries + 1):
            output_ids = self.model.generate(**inputs, max_new_tokens=200, do_sample=(attempt > 0), temperature=0.8)
            text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            try:
                parsed = json.loads(text)
                if all(k in parsed for k in ("question", "options", "correct_answer")):
                    return parsed
            except json.JSONDecodeError:
                continue
        # every attempt failed to produce valid JSON — surface this rather than
        # silently returning malformed data to the frontend
        raise ValueError(f"Question generator failed to produce valid JSON after {max_retries + 1} attempts")

    def unload(self) -> None:
        self.model = None
        self.tokenizer = None
        self._loaded = False
```
Note the retry loop with `do_sample=True` on retries — the first attempt is
deterministic (greedy decoding, matches what you measured in Phase 5), later attempts
add sampling so a retry isn't just guaranteed to fail the exact same way again.

---

## 4. Step 3 — the Model Manager itself

### 4.1 Design
- **Registry**: `dict[str, LoadableModel]`, built once at startup with every model's
  wrapper instance (unloaded, except the two always-resident ones which load
  immediately).
- **LRU tracking**: a `last_used: dict[str, float]` timestamp per on-demand model,
  updated on every `infer()` call.
- **Idle-unload background task**: an `asyncio` task started at app startup, checks
  every 60s whether any on-demand model has been idle > 5 minutes, unloads it if so.
- **Serialization of heavy calls**: one `asyncio.Semaphore(1)` shared by all
  *on-demand* models (M1, M2, M5) — acquiring it before load+infer, releasing after.
  M3/M4 don't need it (cheap, always resident, fine to run concurrently with anything).
- **Blocking-call safety**: every `load()`/`infer()` call runs inside
  `asyncio.to_thread(...)`, so the actual CPU-bound work happens on a worker thread,
  never on the event loop thread.

### 4.2 Implementation
```python
# backend/ml_core/model_manager.py
import asyncio
import time
import logging
from ml_core.base import LoadableModel

logger = logging.getLogger("model_manager")

IDLE_UNLOAD_SECONDS = 5 * 60
IDLE_CHECK_INTERVAL_SECONDS = 60

class ModelManager:
    _instance: "ModelManager | None" = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, models: dict[str, LoadableModel] | None = None):
        if getattr(self, "_initialized", False):
            return  # __init__ runs every time ModelManager() is called — guard against re-init
        self.models: dict[str, LoadableModel] = models or {}
        self.last_used: dict[str, float] = {}
        self.heavy_lock = asyncio.Semaphore(1)
        self._initialized = True
        self._watchdog_task: asyncio.Task | None = None

    async def startup(self):
        """Call once from FastAPI's lifespan startup event."""
        for name, model in self.models.items():
            if model.always_resident:
                logger.info(f"Loading always-resident model: {name}")
                await asyncio.to_thread(model.load)
        self._watchdog_task = asyncio.create_task(self._idle_watchdog())

    async def shutdown(self):
        if self._watchdog_task:
            self._watchdog_task.cancel()

    async def _idle_watchdog(self):
        while True:
            await asyncio.sleep(IDLE_CHECK_INTERVAL_SECONDS)
            now = time.time()
            for name, model in self.models.items():
                if model.always_resident or not model.is_loaded:
                    continue
                if now - self.last_used.get(name, now) > IDLE_UNLOAD_SECONDS:
                    logger.info(f"Unloading idle model: {name}")
                    await asyncio.to_thread(model.unload)

    async def infer(self, model_name: str, *args, **kwargs):
        model = self.models.get(model_name)
        if model is None:
            raise ValueError(f"Unknown model: {model_name}")

        if model.always_resident:
            # no lock needed — cheap, always loaded, safe to run concurrently
            return await asyncio.to_thread(model.infer, *args, **kwargs)

        async with self.heavy_lock:   # serializes M1/M2/M5 against each other
            if not model.is_loaded:
                logger.info(f"Lazily loading model: {model_name}")
                await asyncio.to_thread(model.load)
            self.last_used[model_name] = time.time()
            result = await asyncio.to_thread(model.infer, *args, **kwargs)
            self.last_used[model_name] = time.time()  # refresh again — infer() can take a while
            return result
```

### 4.3 Why the Singleton pattern, concretely, here
FastAPI can construct dependencies per-request by default; a Model Manager must NOT
be one of those — a fresh instance per request would mean a fresh (empty) registry
per request, defeating the entire point of caching loaded models between requests.
The `__new__` override above guarantees `ModelManager()` always returns the same
object, no matter how many times or where it's called. Combined with running Uvicorn
single-process (§1), this is enough — no need for anything more elaborate (like a
separate model-serving process) at this project's scale.

### 4.4 Wiring it into FastAPI's lifespan
```python
# backend/main.py
from contextlib import asynccontextmanager
from fastapi import FastAPI
from ml_core.model_manager import ModelManager
from ml_core.registry import build_model_registry  # see §4.5

@asynccontextmanager
async def lifespan(app: FastAPI):
    manager = ModelManager(models=build_model_registry())
    await manager.startup()
    app.state.model_manager = manager
    yield
    await manager.shutdown()

app = FastAPI(lifespan=lifespan)
```

### 4.5 Building the registry (one place that knows all five paths/configs)
```python
# backend/ml_core/registry.py
import json
from config import settings
from ml_core.models.trajectory_predictor import TrajectoryPredictor
from ml_core.models.career_matcher import CareerMatcher
from ml_core.models.gap_classifier import GapClassifier
from ml_core.models.cv_ner import CVNerExtractor
from ml_core.models.question_generator import QuestionGenerator

def build_model_registry() -> dict:
    with open(settings.TOPICS_JSON) as f:
        topics = json.load(f)          # the same topic list used to build M1's labels in Phase 4
    with open(settings.CAREER_CATALOGUE_JSON) as f:
        career_catalogue = json.load(f)
    with open(settings.M5_ID2TAG_JSON) as f:
        id2tag = {int(k): v for k, v in json.load(f).items()}

    return {
        "trajectory_predictor": TrajectoryPredictor(settings.MODEL3_PATH),
        "career_matcher": CareerMatcher(settings.MODEL4_PATH, career_catalogue),
        "gap_classifier": GapClassifier(settings.MODEL1_PATH, topics),
        "cv_ner": CVNerExtractor(settings.MODEL5_PATH, id2tag),
        "question_generator": QuestionGenerator(settings.MODEL2_PATH),
    }
```
Add the corresponding paths to `backend/.env` (mirrors the master plan's §8.4, now
filled in for real):
```
MODEL1_PATH=../ml_models/saved/model1_gap_classifier
MODEL2_PATH=../ml_models/saved/model2_question_generator
MODEL3_PATH=../ml_models/saved/model3_trajectory_predictor/lstm_final.pt
MODEL4_PATH=../ml_models/saved/model4_career_matcher
MODEL5_PATH=../ml_models/saved/model5_cv_ner
TOPICS_JSON=./data/topics.json
CAREER_CATALOGUE_JSON=./data/career_catalogue.json
M5_ID2TAG_JSON=./data/m5_id2tag.json
```

---

## 5. Step 4 — swap mock endpoints for real ones, one at a time

Do these in this order — cheapest/lowest-risk first, matching Phase 4/5's own
ordering, so a mistake in the trickiest one (M2) doesn't block verifying the rest.

### 5.1 M3 first — `/dashboard/score-trend` (or wherever "at risk" is flagged)
```python
# backend/routers/dashboard.py — replace the mock version
@router.get("/dashboard/at-risk-check")
async def check_at_risk(user_id: int = Depends(get_current_user)):
    recent_scores = await sip_service.get_recent_scores(user_id, n=5)  # from Postgres
    manager: ModelManager = request.app.state.model_manager
    result = await manager.infer("trajectory_predictor", recent_scores)
    if result["trend"] == "declining":
        await notification_service.create(user_id, kind="alert",
            message=f'Score trend flagged "at risk" — Trajectory Predictor')
    return result
```
**Test:** feed it a student with 5 declining scores in Postgres, confirm a
`NOTIFICATIONS` row gets created and the response shows `"trend": "declining"`.

### 5.2 M4 next — `/career/ai-suggest` and `/career/rankings`
```python
@router.get("/career/rankings")
async def get_rankings(user_id: int = Depends(get_current_user)):
    manager: ModelManager = request.app.state.model_manager
    skill_text = await sip_service.get_skill_text(user_id)  # built from CV_UPLOADS.extracted_skills etc.
    rankings = await manager.infer("career_matcher", skill_text, top_k=5)
    await career_service.save_rankings(user_id, rankings)   # persist to CAREER_RANKINGS
    return rankings
```
**Test:** confirm the returned `match_score`s are plausible (0–100, roughly sorted
descending) and that re-calling with an unchanged skill profile returns the same
ranking (embeddings are deterministic, so this should hold exactly).

### 5.3 M1 — `/exam/mcq/submit`
```python
@router.post("/exam/mcq/submit")
async def submit_mcq(payload: MCQSubmitRequest, user_id: int = Depends(get_current_user)):
    manager: ModelManager = request.app.state.model_manager
    question = await exam_service.get_question(payload.question_id)
    correct = payload.answer == question.correct_answer

    weak_topics = []
    if not correct:
        weak_topics = await manager.infer("gap_classifier", question.text)
        await sip_service.update_topic_scores(user_id, weak_topics, correct=False)

    return MCQSubmitResult(correct=correct, correct_answer=question.correct_answer, weak_topics_flagged=weak_topics)
```
**Test:** submit a wrong answer to a question you know the topic tags for (from
Phase 4/5's validation set) and confirm `weak_topics_flagged` matches.

### 5.4 M5 — `/career/cv-upload`
```python
@router.post("/career/cv-upload")
async def upload_cv(cv_file: UploadFile, user_id: int = Depends(get_current_user)):
    manager: ModelManager = request.app.state.model_manager
    raw_text = await extract_text_from_file(cv_file)  # pdfplumber/python-docx depending on filetype
    extracted = await manager.infer("cv_ner", raw_text)
    await career_service.save_cv_upload(user_id, extracted)
    return extracted
```
**Test:** re-upload the exact gold-set CVs from Phase 3/5, confirm the extracted
skills/certs/GPA match what the gold labels say (this is the same parity check as
Phase 5 §3.3, now run through the actual HTTP endpoint instead of a notebook cell).

### 5.5 M2 last — `/exam/mcq` (generation) and `/career`'s roadmap-adjacent uses if any
```python
@router.get("/exam/mcq")
async def get_mcq(topic: str, difficulty: str = "medium", user_id: int = Depends(get_current_user)):
    manager: ModelManager = request.app.state.model_manager
    try:
        generated = await manager.infer("question_generator", topic, difficulty)
    except ValueError:
        # generation failed validity checks even after retries — fall back to the
        # existing question bank rather than showing the student a broken page
        generated = await exam_service.get_fallback_question(topic)
    return generated
```
**This fallback matters more than it looks** — M2 is the model most likely to
occasionally misfire (it's a generative model, not a classifier), and a demo where
one bad generation crashes the exam page is a worse outcome than quietly serving a
question from the static bank that one time.

---

## 6. Step 5 — a debug endpoint (temporary, remove before the final demo)

Extremely useful while wiring all this up — a way to see the Model Manager's internal
state without digging through logs:
```python
# backend/routers/debug.py — only mount this router if settings.DEBUG is True
@router.get("/debug/model-status")
async def model_status(request: Request):
    manager: ModelManager = request.app.state.model_manager
    return {
        name: {"loaded": model.is_loaded, "always_resident": model.always_resident,
               "last_used": manager.last_used.get(name)}
        for name, model in manager.models.items()
    }
```
Hit this between requests while testing §5's endpoints to visually confirm: M3/M4 show
`loaded: true` immediately at startup and stay that way; M1/M2/M5 start `loaded:
false`, flip to `true` right after their first real request, and flip back to `false`
on their own ~5 minutes later if nothing else calls them.

---

## 7. Step 6 — a lightweight memory watchdog (optional but recommended given the RAM budget)

```python
# backend/ml_core/memory_watchdog.py
import psutil, logging

logger = logging.getLogger("memory_watchdog")

def log_memory_snapshot(label: str):
    mem = psutil.virtual_memory()
    logger.info(f"[{label}] RAM used: {mem.percent}% ({mem.used / 1e9:.2f}GB / {mem.total / 1e9:.2f}GB)")
```
Call `log_memory_snapshot("before M2 load")` / `"after M2 load"` around the
interesting parts of `ModelManager.infer()` while testing — this is how you actually
fill in Phase 5 §8's benchmark table with real numbers, and how you'll notice early if
the 8GB budget from the master plan's §12 is holding up in practice or not.

---

## 8. Step 7 — unit-test each model wrapper in isolation before trusting the endpoints

Don't debug "is it the model or the endpoint" simultaneously — write one small
pytest per wrapper first:
```python
# backend/tests/test_gap_classifier.py
from ml_core.models.gap_classifier import GapClassifier

def test_gap_classifier_known_example():
    clf = GapClassifier(model_dir="ml_models/saved/model1_gap_classifier", topics=TOPICS)
    clf.load()
    result = clf.infer("A quadratic equation has the form ax² + bx + c = 0.")
    assert "Algebra" in result or "Quadratic Equations" in result
```
Write the equivalent for all five wrappers, using the exact known-answer examples
from Phase 4's validation sets and Phase 5's parity checks — you already know the
right answers for these, so this is nearly free to write and catches integration bugs
(wrong file paths, wrong tokenizer, off-by-one in BIO decoding) well before they show
up as a confusing 500 error from the frontend.

---

## 9. Common pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| Whole server freezes for several seconds on every ML request, other users' requests stall too | `model.infer(...)` called directly inside an `async def` route without `asyncio.to_thread` | Route every blocking call through `ModelManager.infer()`, never call a model wrapper directly from a router |
| RAM usage roughly doubles what Phase 5's benchmarks predicted | Uvicorn started with `--workers 2+`, or `ModelManager()` isn't actually behaving as a singleton | Confirm single-worker (§1); add a log line in `__init__` and confirm it only ever prints once per process |
| M1/M2/M5 never unload, RAM creeps up over a long session | `last_used` timestamp not being refreshed correctly, or the idle watchdog task silently died | Check `/debug/model-status`'s `last_used` field is actually updating; check `_idle_watchdog`'s task didn't raise and get silently swallowed — wrap its loop body in try/except with logging while debugging |
| Two heavy requests at once cause a RAM spike anyway | Forgot to route one of M1/M2/M5's endpoints through `manager.infer()` — calling `model.load()`/`model.infer()` directly somewhere bypasses `heavy_lock` entirely | Grep the codebase for any direct `.infer(` or `.load(` call on a model wrapper outside `model_manager.py` — there should be none |
| M2 generation "works" but the frontend shows broken/missing question text | JSON parse succeeded but a required key was missing/renamed, not caught by the `all(k in parsed ...)` check | Tighten the validation in `QuestionGenerator.infer()` to also check `options` has exactly 4 entries with keys `A`–`D` before accepting it |
| CV upload works for short CVs, breaks or times out for long ones | `_chunk()`'s `max_chars` too high for the model's `max_length=256` tokens, or too many chunks making one upload very slow | Tune `max_chars` down; consider capping total chunks processed per upload (e.g. first 6) with a note that very long CVs are truncated |

---

## 10. Definition of done

- [ ] `ml_core/base.py` — the `LoadableModel` interface
- [ ] All five model wrapper classes implemented and unit-tested individually (§8)
      against known-answer examples from Phase 4/5
- [ ] `ModelManager` implemented with: singleton behavior, `asyncio.to_thread` for
      every blocking call, idle-unload watchdog for on-demand models, a shared
      semaphore serializing M1/M2/M5 against each other
- [ ] Wired into FastAPI's `lifespan` — `app.state.model_manager` accessible from
      every router
- [ ] All 5 endpoints from §5 swapped from Phase 2's mock JSON to real
      `manager.infer(...)` calls, each individually re-tested against a known example
- [ ] `/debug/model-status` confirms M3/M4 stay resident, M1/M2/M5 load-on-demand and
      idle-unload after ~5 minutes, during a manual click-through of the frontend
- [ ] Memory watchdog logs collected for at least one "worst case" moment (M2
      generating right after M1 classified something) — confirms the semaphore is
      actually preventing concurrent heavy inference, not just assumed to be

Once every row is checked, Phase 7 (RAG pipeline — ChromaDB + Ollama) is unblocked:
it follows the same "wrap it, register it, wire one endpoint at a time" pattern this
phase just established, applied to the mock-interview and topic-explainer chatbot
instead of the five trained models.
