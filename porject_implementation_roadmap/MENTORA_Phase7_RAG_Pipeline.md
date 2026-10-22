# MENTORA — Phase 7 Deep Dive: The RAG Pipeline (ChromaDB + Ollama)

*Prerequisite: Phase 6 is done — the Model Manager pattern (lazy load, async-safe,
serialized heavy inference) is established and working for the five trained models.
This phase builds the sixth and last "brain" of the system: the retrieval-augmented
chatbot behind the mock interview and topic-explainer features (Objective 6). It's
conceptually different from Phase 6 — no training happened for this one, it's built
entirely from off-the-shelf pieces (Ollama + ChromaDB + LangChain) wired together.
Estimated time: 5–8 days, most of it spent on content curation (§3), not code.*

---

## 0. What RAG is actually solving here, and why it's not "just call the LLM"

If you skip straight to calling `phi4-mini` with the student's raw question, you get
a chatbot that sounds confident and is frequently wrong — a 3.8B-parameter local
model has no reliable specific knowledge of, say, exactly what topics are on the
Sindh University entry test, or what a good answer to a specific behavioral interview
question looks like for a Data Scientist role. It'll happily make something up that
*sounds* plausible. That's the failure mode Objective 6 explicitly rules out
("grounded in curated content — not open-ended generation").

**RAG's actual mechanism, in one sentence:** before the LLM generates anything, you
search a database of content *you wrote or curated yourself* for the few passages
most relevant to the question, and paste those passages directly into the prompt with
an instruction to answer only from them. The LLM's job shrinks from "know the answer"
to "summarize/rephrase these 3 paragraphs I'm handing you" — a much easier, much more
reliable job for a small local model to do well.

This means **the quality of this whole feature is determined more by what you put in
the knowledge base (§3) than by anything about the model or the code** — keep that
weighting in mind for where to spend your time this week.

---

## 1. Step 1 — install and verify Ollama, locally, before writing any Python

### 1.1 Install
Download from `ollama.com/download` (Windows/Mac/Linux all supported). This installs
a background service that listens on `http://localhost:11434` — it starts
automatically after install and after every reboot; you generally never need to
launch it manually.

### 1.2 Pull the model
```bash
ollama pull phi4-mini
```
This downloads roughly 2–3GB (quantized weights) — do this once, on decent internet,
before you're relying on it for anything time-sensitive. If it turns out too slow on
your CPU once you test it (§1.3), the fallback swap is one more pull and a config
change, nothing structural:
```bash
ollama pull qwen2.5:1.5b   # smaller, faster, slightly less capable fallback
ollama pull gemma3:1b       # even smaller second fallback if qwen2.5:1.5b is still too slow
```

### 1.3 Test from the command line first — don't skip straight to Python
```bash
ollama run phi4-mini "Explain what a quadratic equation is, in two sentences."
```
Time how long this takes to finish, roughly. On a 3rd-Gen i5 with no AVX2, expect this
to be noticeably slower than any benchmark number you'll find online (those are
almost always measured on much newer hardware) — single-digit tokens/second is a
realistic expectation here, not a sign something's broken. This number directly
informs §13's tuning and the demo-day decisions in Phase 9. Type `/bye` to exit.

### 1.4 Understand the two settings that matter most on weak hardware
- **`num_ctx`** — the context window size (how many tokens of prompt + retrieved
  content + conversation history the model considers at once). Ollama's default is
  often larger than you need; a smaller `num_ctx` directly reduces both RAM use and
  generation time. For this project's short, focused prompts, `num_ctx=2048` is
  almost certainly enough — no need for the multi-thousand-token defaults some setups
  use.
- **`keep_alive`** — how long Ollama keeps a model loaded in RAM after a request
  before unloading it. The default (a few minutes) is usually fine and mirrors the
  Model Manager's own idle-unload philosophy from Phase 6 — but if you find yourself
  making back-to-back mock-interview calls during testing and each one is slow
  because the model reloaded, that's `keep_alive` expiring between your test calls;
  it's not a bug.

### 1.5 Confirm the REST API directly (this is what your Python code will actually call)
```bash
curl http://localhost:11434/api/generate -d '{
  "model": "phi4-mini",
  "prompt": "Say hello in one sentence.",
  "stream": false,
  "options": { "num_ctx": 2048 }
}'
```
If this returns a JSON response with a `"response"` field, Ollama's serving layer is
confirmed working end to end before any FastAPI/LangChain code touches it — the same
principle as Phase 2's "verify the mock backend from Swagger before touching the
frontend."

---

## 2. Step 2 — install the RAG-specific Python packages

```bash
pip install langchain langchain-ollama langchain-chroma chromadb
```
Two notes on current package layout (this has moved around over LangChain's various
reorganizations, so it's worth being explicit): the Ollama integration lives in the
separate **`langchain-ollama`** package (`from langchain_ollama import OllamaLLM,
ChatOllama, OllamaEmbeddings`), and the Chroma integration lives in the separate
**`langchain-chroma`** package (`from langchain_chroma import Chroma`) — not inside
LangChain's old core `langchain.llms`/`langchain.vectorstores` modules, which is
where older tutorials still point.

---

## 3. Step 3 — content curation: what actually goes in the three knowledge bases

This is the part of Phase 7 that isn't really "coding" — it's writing/curating the
actual reference material the chatbot will draw from. Budget real time here.

### 3.1 `exam_kb` — topic explainers
One short document per exam topic (matching the same topic taxonomy from Phase 3's
M1 dataset — reuse that list, don't invent a second one). Each document: a clear
2–4 paragraph explanation of the topic, written the way you'd want a tutor to explain
it, plus 1–2 worked examples. Source options:
- Write these yourself for the highest-priority topics (fastest to trust, most
  aligned with what the exam-prep timetable actually covers)
- Adapt from open textbooks / OER (OpenStax, etc. — check license terms allow reuse)
  for topics that would take too long to write from scratch
- **Don't** just dump raw scraped web content in unedited — inconsistent tone/format
  across documents makes retrieval noisier and answers less coherent

Target: one document per topic in your taxonomy, ~150–400 words each. For a
first-pass FYP scope, prioritize the topics your M1/M2 datasets already emphasize
most (Algebra, Grammar, etc. from the existing frontend's hardcoded examples) — that
overlap means the exam-prep page's "AI Topic Explainer" panel and this RAG system
reinforce each other on the same subject matter.

### 3.2 `career_kb` — career and roadmap content
- One document per career path in your M4 catalogue (the same 50+ careers) —
  role description, typical required skills, typical entry paths. The O*NET data
  from Phase 3 §4 is a legitimate direct source here, lightly rewritten into
  student-facing language rather than O*NET's more clinical occupational-analysis
  tone.
- A handful of general "how career roadmaps work" documents — e.g. "what a
  6-month upskilling plan for a Data Science career transition typically looks
  like" — since students will ask open-ended roadmap questions, not just
  "what does X role do."

### 3.3 `interview_kb` — mock interview content
- A bank of common interview questions **per career path**, each paired with what a
  strong answer covers (not a verbatim script — bullet points of the substance a
  good answer should hit, so the LLM has to synthesize a response, not just recite
  one). Mix behavioral ("tell me about a time...") and technical (role-specific)
  questions.
- A short "interview technique" reference set — STAR method, how to structure a
  technical explanation, etc. — for when a student asks for interview *advice*
  rather than a mock question.

### 3.4 Format and where these live
Plain Markdown or `.txt` files, one file per document, organized by knowledge base:
```
rag_pipelines/
├── exam_rag/content/
│   ├── algebra-quadratic-equations.md
│   ├── grammar-tenses.md
│   └── ...
├── career_rag/content/
│   ├── career-data-scientist.md
│   ├── career-ml-engineer.md
│   ├── roadmap-general-guide.md
│   └── ...
└── interview_rag/content/
    ├── interview-data-scientist.md
    ├── interview-technique-star-method.md
    └── ...
```
Each file's front matter (a small metadata header) should carry at least a `topic`
or `career` tag and a `type` — this metadata rides along into ChromaDB (§6) and lets
retrieval be filtered, not just similarity-ranked, later if needed:
```markdown
---
topic: "Algebra — Quadratic Equations"
type: "topic_explainer"
---

A quadratic equation is any equation that can be written in the form
ax² + bx + c = 0, where a ≠ 0...
```

---

## 4. Step 4 — set up ChromaDB (embedded/persistent mode)

### 4.1 The embedding-function decision
Every document chunk needs to become a vector before it can be searched. Two
reasonable options, and this project should pick the simpler one:

- **Option A — ChromaDB's built-in default embedding function** (a small ONNX
  MiniLM model, downloaded automatically the first time you use it, ~90MB). Zero
  extra integration work, small enough to not meaningfully affect the RAM budget.
- **Option B — reuse M4's `all-mpnet-base-v2` Sentence-Transformers model** (already
  loaded and always-resident per Phase 6) for embedding consistency with the career
  matcher.

**Recommendation: Option A.** M4's embedding space was fine-tuned specifically for
skill-to-career similarity (Phase 4 §3) — repurposing it for general topic-explainer
and interview-content retrieval is actually a worse semantic fit than a general
sentence-embedding model, not a better one, despite the "reuse what's already loaded"
instinct. Keep RAG's embedding function separate and general-purpose.

### 4.2 Persistent client + three collections
```python
# backend/rag_core/chroma_client.py
import chromadb

CHROMA_PATH = "./vector_store"  # matches the master plan's folder layout

client = chromadb.PersistentClient(path=CHROMA_PATH)

def get_collection(name: str):
    # get_or_create_collection is idempotent — safe to call on every startup
    return client.get_or_create_collection(name=name)

exam_kb = get_collection("exam_kb")
career_kb = get_collection("career_kb")
interview_kb = get_collection("interview_kb")
```
`PersistentClient` writes straight to local disk under `vector_store/` — no server
process, no Docker, consistent with the "embedded mode" decision from the master
plan.

---

## 5. Step 5 — chunking strategy

### 5.1 Why chunk at all, given the documents in §3 are already short
Even a 300-word topic explainer benefits from being split into 2–3 smaller chunks:
retrieval works better when each stored vector represents one coherent idea, not an
entire multi-paragraph document where a query might only be relevant to one
paragraph of it.

### 5.2 Implementation
```python
# backend/rag_core/chunking.py
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,       # characters, not tokens — keep well under phi4-mini's num_ctx budget once several chunks are retrieved together
    chunk_overlap=60,      # a little overlap avoids losing meaning right at a chunk boundary
    separators=["\n\n", "\n", ". ", " ", ""],  # prefer splitting on paragraph/sentence boundaries over mid-word
)

def chunk_document(text: str) -> list[str]:
    return splitter.split_text(text)
```

---

## 6. Step 6 — `ingest.py` for each knowledge base

```python
# rag_pipelines/exam_rag/ingest.py
import os
import frontmatter  # pip install python-frontmatter — parses the --- metadata header from §3.4
from rag_core.chroma_client import exam_kb
from rag_core.chunking import chunk_document

CONTENT_DIR = os.path.join(os.path.dirname(__file__), "content")

def ingest():
    # Wipe and rebuild rather than trying to diff — content set is small enough
    # (dozens to low hundreds of documents) that idempotent full rebuilds are
    # simpler and safer than incremental updates for this project's scale.
    existing_ids = exam_kb.get()["ids"]
    if existing_ids:
        exam_kb.delete(ids=existing_ids)

    ids, documents, metadatas = [], [], []
    for filename in os.listdir(CONTENT_DIR):
        if not filename.endswith(".md"):
            continue
        post = frontmatter.load(os.path.join(CONTENT_DIR, filename))
        chunks = chunk_document(post.content)
        for i, chunk in enumerate(chunks):
            ids.append(f"{filename}::chunk{i}")
            documents.append(chunk)
            metadatas.append({
                "source": filename,
                "topic": post.get("topic", ""),
                "type": post.get("type", ""),
                "chunk_index": i,
            })

    exam_kb.add(ids=ids, documents=documents, metadatas=metadatas)
    print(f"Ingested {len(documents)} chunks from {len(os.listdir(CONTENT_DIR))} documents into exam_kb")

if __name__ == "__main__":
    ingest()
```
Write the near-identical `rag_pipelines/career_rag/ingest.py` and
`rag_pipelines/interview_rag/ingest.py`, pointed at `career_kb`/`interview_kb` and
their own `content/` folders. Run all three manually whenever content changes:
```bash
python rag_pipelines/exam_rag/ingest.py
python rag_pipelines/career_rag/ingest.py
python rag_pipelines/interview_rag/ingest.py
```

### 6.1 Verify ingestion worked before writing any retrieval code
```python
from rag_core.chroma_client import exam_kb
print(exam_kb.count())  # should match the number of chunks printed by ingest.py
result = exam_kb.query(query_texts=["how do I solve a quadratic equation"], n_results=3)
for doc, meta in zip(result["documents"][0], result["metadatas"][0]):
    print(meta["source"], "->", doc[:80])
```
If the top 3 results for an obviously-algebra-related query aren't algebra-related
content, something's wrong upstream (bad chunking, bad metadata, or the content
itself doesn't actually cover what you think it does) — fix this before building
anything on top of it.

---

## 7. Step 7 — the grounded prompt template

This is the single most important piece of code in this phase for actually meeting
Objective 6 ("grounded... not open-ended generation"):

```python
# backend/rag_core/prompts.py
GROUNDED_PROMPT_TEMPLATE = """You are a helpful tutor for a Pakistani student exam-prep and career platform.
Answer the student's question using ONLY the information in the CONTEXT section below.

Rules:
- If the answer is fully contained in the context, answer clearly and concisely.
- If the context only partially covers the question, answer what you can from it and say what's missing.
- If the context has nothing relevant to the question, say "I don't have information on that in my materials right now" — do NOT make up an answer.
- Do not mention "the context" or "the provided text" in your answer — just answer naturally, as a tutor would.

CONTEXT:
{context}

STUDENT QUESTION:
{question}

ANSWER:"""

def build_prompt(context_chunks: list[str], question: str) -> str:
    context = "\n\n---\n\n".join(context_chunks)
    return GROUNDED_PROMPT_TEMPLATE.format(context=context, question=question)
```
The explicit "say you don't know" instruction is what prevents hallucination on
out-of-scope questions — small local models follow this kind of direct instruction
reasonably well, much better than they avoid hallucinating without being told to.

---

## 8. Step 8 — the `RAGPipeline` class hierarchy (matches the tech doc's class diagram)

### 8.1 Base class
```python
# backend/rag_core/pipeline.py
from abc import ABC, abstractmethod
from langchain_ollama import OllamaLLM
from rag_core.prompts import build_prompt

class RAGPipeline(ABC):
    def __init__(self, collection, ollama_model: str = "phi4-mini", top_k: int = 4):
        self.collection = collection
        self.top_k = top_k
        self.llm = OllamaLLM(model=ollama_model, num_ctx=2048)

    def retrieve(self, query: str) -> list[str]:
        result = self.collection.query(query_texts=[query], n_results=self.top_k)
        return result["documents"][0] if result["documents"] else []

    def generate(self, prompt: str) -> str:
        return self.llm.invoke(prompt)

    @abstractmethod
    def answer(self, query: str, **kwargs) -> str:
        """Each subclass defines its own answer() signature/behavior —
        e.g. InterviewRAG needs conversation history, ExamRAG doesn't."""
        ...
```

### 8.2 `ExamRAG` — topic explainer chatbot
```python
# rag_pipelines/exam_rag/exam_rag.py
from rag_core.pipeline import RAGPipeline
from rag_core.chroma_client import exam_kb
from rag_core.prompts import build_prompt

class ExamRAG(RAGPipeline):
    def __init__(self):
        super().__init__(collection=exam_kb)

    def answer(self, query: str) -> str:
        chunks = self.retrieve(query)
        if not chunks:
            return "I don't have information on that in my materials right now."
        prompt = build_prompt(chunks, query)
        return self.generate(prompt)
```

### 8.3 `CareerRAG` — career/roadmap Q&A
```python
# rag_pipelines/career_rag/career_rag.py
class CareerRAG(RAGPipeline):
    def __init__(self):
        super().__init__(collection=career_kb)

    def answer(self, query: str) -> str:
        chunks = self.retrieve(query)
        if not chunks:
            return "I don't have information on that in my materials right now."
        return self.generate(build_prompt(chunks, query))
```

### 8.4 `InterviewRAG` — the mock interview, which needs conversation state
```python
# rag_pipelines/interview_rag/interview_rag.py
class InterviewRAG(RAGPipeline):
    def __init__(self):
        super().__init__(collection=interview_kb)

    def start(self, career_title: str) -> str:
        chunks = self.retrieve(f"interview questions for {career_title}")
        prompt = f"""You are conducting a mock job interview for a {career_title} role.
Using the reference questions below as inspiration, ask the student ONE realistic
opening interview question. Ask only the question, nothing else.

REFERENCE MATERIAL:
{"".join(chunks)}
"""
        return self.generate(prompt)

    def answer(self, career_title: str, conversation_history: list[dict], student_answer: str) -> str:
        chunks = self.retrieve(f"{career_title} interview: {student_answer}")
        history_text = "\n".join(f"{turn['role']}: {turn['text']}" for turn in conversation_history)
        prompt = f"""You are conducting a mock job interview for a {career_title} role.
Give brief, constructive feedback on the student's last answer (2-3 sentences), then
ask ONE follow-up or new interview question, using the reference material where relevant.

REFERENCE MATERIAL:
{"".join(chunks)}

CONVERSATION SO FAR:
{history_text}

STUDENT'S LATEST ANSWER:
{student_answer}
"""
        return self.generate(prompt)
```

---

## 9. Step 9 — integrating with the Model Manager's concurrency discipline

Ollama runs as its **own persistent background service**, not a Python object your
process loads/unloads — so it doesn't need a `LoadableModel` wrapper the way M1–M5
do. But it's still CPU-heavy on this laptop and **must not run concurrently with M2
question generation** (both are the two heaviest CPU consumers in the whole system —
running them at once is close to a worst-case load scenario per the master plan's
§12 RAM/CPU budget). Reuse Phase 6's `heavy_lock` semaphore for this:

```python
# backend/services/rag_service.py
import asyncio
from ml_core.model_manager import ModelManager  # the same singleton from Phase 6

async def ask_exam_rag(query: str) -> str:
    manager = ModelManager()   # singleton — returns the same instance, no re-init
    async with manager.heavy_lock:
        return await asyncio.to_thread(exam_rag_instance.answer, query)
```
Same pattern for `CareerRAG`/`InterviewRAG` calls — every path that ends up calling
Ollama goes through `manager.heavy_lock` and `asyncio.to_thread`, for exactly the
same two reasons as Phase 6 §0: don't block the event loop, don't let two heavy
CPU-bound jobs fight for the same 4 cores at once.

---

## 10. Step 10 — FastAPI endpoints

```python
# backend/routers/interview.py
from fastapi import APIRouter, Depends
from services.auth_service import get_current_user
from services import rag_service, interview_service

router = APIRouter(prefix="/interview", tags=["interview"])

@router.post("/start")
async def start_interview(career_title: str, user_id: int = Depends(get_current_user)):
    first_question = await rag_service.start_interview(career_title)
    interview_id = await interview_service.create_session(user_id, career_title, first_question)
    return {"interview_id": interview_id, "first_question": first_question}

@router.post("/{interview_id}/ask")
async def ask_interview(interview_id: str, answer: str, user_id: int = Depends(get_current_user)):
    session = await interview_service.get_session(interview_id, user_id)
    reply = await rag_service.continue_interview(session.career_title, session.history, answer)
    await interview_service.append_turn(interview_id, student_answer=answer, ai_reply=reply)
    return {"answer": reply}
```
```python
# backend/routers/exam.py — add alongside the existing endpoints from Phase 2/6
@router.get("/topic-explainer")
async def topic_explainer(query: str, user_id: int = Depends(get_current_user)):
    answer = await rag_service.ask_exam_rag(query)
    return {"answer": answer}
```
`interview_service` persists to `INTERVIEW_SESSIONS` (one row per `start()` call) and
`INTERVIEW_QA` (one row per question/answer turn) per the ERD — this is what makes
mock-interview history available later (e.g. for a "review your past interviews"
feature, even if that's not in the current frontend).

---

## 11. Step 11 — optional upgrade: streaming responses to the frontend

Given realistic single-digit tokens/second on this hardware (§1.3), a student asking
a question and staring at a blank "loading" state for 15–30 seconds is a worse
experience than watching the answer appear word by word, even though the *total*
time is the same either way. This is optional for a working MVP but worth doing if
time allows:

```python
# backend/routers/interview.py — streaming variant
from fastapi.responses import StreamingResponse

@router.post("/{interview_id}/ask/stream")
async def ask_interview_stream(interview_id: str, answer: str, user_id: int = Depends(get_current_user)):
    session = await interview_service.get_session(interview_id, user_id)

    async def token_generator():
        manager = ModelManager()
        async with manager.heavy_lock:
            full_response = ""
            for chunk in interview_rag_instance.llm.stream(build_prompt_for(session, answer)):
                full_response += chunk
                yield chunk
            await interview_service.append_turn(interview_id, student_answer=answer, ai_reply=full_response)

    return StreamingResponse(token_generator(), media_type="text/plain")
```
On the Next.js side, this needs a Client Component reading the response body via the
`ReadableStream` API instead of a single `await res.json()` — a meaningfully bigger
frontend change than the rest of Phase 2's wiring, which is exactly why this is
flagged optional rather than required for this phase's definition of done.

---

## 12. Step 12 — testing

### 12.1 Retrieval quality (test this in isolation, before involving Ollama at all)
Write a small set of test queries with a known expected source document per KB, and
assert the right chunk shows up in the top-k:
```python
def test_exam_retrieval_finds_quadratic_content():
    result = exam_kb.query(query_texts=["how to factor x squared minus 5x plus 6"], n_results=3)
    sources = [m["source"] for m in result["metadatas"][0]]
    assert "algebra-quadratic-equations.md" in sources
```

### 12.2 Groundedness (harder to fully automate — do a manual spot-check pass)
Ask 10–15 questions per knowledge base, including a few **deliberately
out-of-scope** ones (e.g. asking `ExamRAG` a career question), and manually verify:
- In-scope questions get answers that actually match the source content (no
  invented facts, no wrong formulas)
- Out-of-scope questions get the "I don't have information on that" fallback, not a
  confidently wrong answer

### 12.3 Latency
Time `retrieve()` and `generate()` separately (retrieval should be near-instant;
generation is where the real time goes) — this is the other half of Phase 6 §7's
memory-watchdog habit, now for time instead of RAM, and it's what tells you whether
§13's tuning actually helped.

---

## 13. Step 13 — performance tuning for this specific hardware

- **Lower `num_ctx` further** if §12.3's latency is too slow for a usable demo —
  1024 is workable for these short, focused prompts if 2048 still feels slow.
- **Cap `num_predict`** (max generated tokens) explicitly — an unbounded response
  length on slow hardware risks a very long wait for a rambling answer; 150–250
  tokens is plenty for a tutor-style answer or one interview question.
- **Reduce `top_k` retrieval** from 4 to 2–3 chunks — fewer tokens of context to
  process before generation even starts.
- **Consider the smaller fallback models** (§1.2) as the *default*, not just an
  emergency fallback, if `phi4-mini` consistently feels too slow for a fluid demo —
  there's no penalty for choosing the smaller model as primary if it makes the
  actual experience better on this hardware.
- **Pre-warm the model before a demo** — send one throwaway prompt a minute or two
  before you need it, so `keep_alive` (§1.4) has it already loaded when it matters.

---

## 14. Common pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| `ollama.ResponseError: model not found` | Forgot to `ollama pull phi4-mini` on this machine, or typo in the model name string | Re-check `ollama list` shows the exact model tag used in code |
| Every RAG answer is generic/hallucinated regardless of the question | Retrieval returning irrelevant chunks (bad embeddings/chunking), or the prompt template not actually including the retrieved context | Run §6.1's manual retrieval check first, independent of generation |
| Answers ramble on far longer than expected | No `num_predict` cap set | Add `num_predict` to the Ollama call options (§13) |
| First request after a few idle minutes is much slower than subsequent ones | `keep_alive` expired and the model reloaded | Expected behavior, not a bug — see §1.4; pre-warm before a timed demo |
| Whole app briefly unresponsive during any RAG call | `generate()` called directly without `asyncio.to_thread`, same class of bug as Phase 6 | Route every Ollama call through the pattern in §9 |
| Interview conversation loses earlier turns | `conversation_history` not actually being persisted/passed on each `/ask` call | Check `interview_service.get_session()` is loading `INTERVIEW_QA` rows and rebuilding the history list correctly |
| ChromaDB collection has 0 results after ingest | `ingest.py` run from the wrong working directory, or `content/` folder path resolved incorrectly | Use `os.path.dirname(__file__)`-relative paths (as in §6), not paths assumed relative to wherever the script happens to be run from |

---

## 15. Definition of done

- [ ] `ollama run phi4-mini "..."` works from the command line, and the fallback
      models are pulled and ready even if not the default
- [ ] All three `content/` folders populated with real curated documents (§3) —
      this is the item most worth double-checking isn't skipped or rushed
- [ ] `ingest.py` run successfully for all three knowledge bases; §6.1's manual
      retrieval check confirms sensible top-k results for several test queries per KB
- [ ] `RAGPipeline` base class + `ExamRAG`/`CareerRAG`/`InterviewRAG` implemented
- [ ] The grounded prompt template's "I don't have information on that" fallback
      verified working on deliberately out-of-scope test questions (§12.2)
- [ ] `/interview/start` and `/interview/{id}/ask` wired, persisting to
      `INTERVIEW_SESSIONS`/`INTERVIEW_QA`, and reusing Phase 6's `heavy_lock` +
      `asyncio.to_thread` pattern
- [ ] `/exam/topic-explainer` wired the same way
- [ ] Latency measured and at least one round of §13's tuning applied if the raw
      numbers felt too slow for a live demo
- [ ] "Start Mock Interview" on the real `/career` page (from Phase 2's wiring)
      returns a real, context-grounded question, and follow-up answers work

With this checked off, every one of the master plan's six Fixed Objectives is now
backed by working code — Phase 8 (integration testing across the whole system) and
Phase 9 (polish/demo packaging) are what's left before this becomes a defensible,
demo-ready FYP.
