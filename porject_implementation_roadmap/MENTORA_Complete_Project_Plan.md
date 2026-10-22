# MENTORA — Complete Project Plan (Frontend → Backend → ML → RAG → Local Deployment)

*Compiled from `MENTORA_Frontend_Documentation.md` (the real, already-built frontend),
`Mentora_Master_Plan.md` (your prior planning doc), and fresh verification against the
web as of **Friday, July 17, 2026**. Where the two source documents disagreed, this
file states which one is correct and why — see §0.*

---

## 0. Verification pass — what was checked, and what got corrected

You asked for everything to be verified before any plan gets built on top of it. Here's
exactly what was checked and what changed.

### 0.1 Frontend reality check — this is the one real correction to your Master Plan
`Mentora_Master_Plan.md` (§0.1) states the frontend is *"8 static HTML pages + vanilla
JS"* and explicitly drops Next.js from the plan. **That's no longer accurate.** The
actual, currently-built frontend — confirmed by re-reading `MENTORA_Frontend_Documentation.md`,
which was generated directly from the real project files — is a **Next.js 14 (App
Router) + TypeScript** application with 8 routes, shared `SiteHeader`/`SiteFooter`/
`AuthShell` components, and a ported `globals.css` design system. It was built,
type-checked, and production-built successfully.

**This document treats the Next.js frontend as ground truth and plans the backend
around it.** This actually works *better* for you than the static-HTML plan, for one
concrete reason that matters for auth security: a static HTML site has nowhere to keep
a secret, so a JWT stored in the browser has to sit in `localStorage`, which is readable
by any injected script. Next.js has its own server (even in dev, and definitely in
`next start`), so the JWT can live in an **httpOnly cookie** set by a small Next.js
Route Handler — invisible to JavaScript entirely. §10 covers this in detail.

Everything else in your Master Plan that *isn't* about the frontend framework — the
hardware correction, the model list, the ERD, the phase breakdown — was independently
re-verified below and is carried forward because it checks out.

### 0.2 Hardware reality check — 3rd-Gen i5, re-confirmed
Your machine: **Intel Core i5, 3rd Generation (Ivy Bridge, ~2012), 8GB RAM, 128GB SSD.**
This is the single most important constraint in this whole plan, and it's a real,
documented one, not a guess:

- Ivy Bridge (3rd-Gen) supports **AVX**, but **not AVX2**. AVX2 arrived with Haswell
  (4th-Gen, 2013).
- PyTorch's default x86 quantization backend, **FBGEMM**, has a hard dependency on
  AVX2 — this is stated directly in PyTorch's own quantization documentation, and is
  the subject of multiple confirmed PyTorch GitHub issues where `import torch` or
  `quantize_dynamic(...)` crashes with `Illegal instruction (core dumped)` on non-AVX2
  CPUs. This is not a hypothetical edge case; it's a known, still-open class of issue.
- **Fix:** don't use PyTorch's built-in FBGEMM dynamic quantization for the three
  heavier models (M1, M2, M5). Export those to **ONNX** and quantize with **ONNX
  Runtime's** own `onnxruntime.quantization.quantize_dynamic()`, which has portable
  kernel paths that don't hard-require AVX2. This is confirmed current, working
  functionality in ONNX Runtime (latest verified release: **1.27.0**, June 2026).
- M3 (LSTM) and M4 (Sentence-Transformers) are small enough to just run in plain FP32
  — no quantization needed, no AVX2 dependency.
- Ollama's inference engine (`llama.cpp`) ships a generic scalar fallback for CPUs
  without AVX2, so it will run — just slower than the "12–28 tok/s" figures you'll see
  quoted for modern hardware. Budget for single-digit tokens/sec and design the demo
  around that (see §13, Phase 9).

**Action item before writing a line of backend code:** confirm this on the actual
machine. On Linux/WSL: `cat /proc/cpuinfo | grep flags | grep -o avx2` (empty output =
no AVX2, confirming the plan below). On Windows: CPU-Z → Instructions tab. If AVX2
*is* present (meaning the CPU is newer than "3rd-Gen" actually implies), you can skip
the ONNX detour and use standard PyTorch FBGEMM quantization — strictly less setup.

### 0.3 Software versions — verified today (July 17, 2026), not carried over from old docs
Both source documents contained version numbers from earlier points in the project.
Here's what's actually current right now, checked directly against PyPI/official
sources:

| Component | Verified current version (Jul 2026) | What this plan pins |
|---|---|---|
| PyTorch | **2.13.0** (stable, released Jul 8 2026) | Install CPU build via `--index-url https://download.pytorch.org/whl/cpu`, don't pin the patch version — see install instructions below |
| FastAPI | **0.139.x** (releasing roughly weekly) | `fastapi[standard]`, unpinned at first install, then frozen |
| Transformers / PEFT / Sentence-Transformers | Actively releasing | Install unpinned, freeze after a working smoke test (see §0.4) |
| ONNX Runtime | **1.27.0** | For M1/M2/M5 export + quantization |
| ChromaDB | **1.5.9** | Embedded/persistent mode still fully supported this way; note ChromaDB itself now depends on `onnxruntime` internally for its default embedding function, which is a convenient overlap with your own M1/M2/M5 runtime |
| PostgreSQL | Latest stable major is **18.4**; **16** is also fully supported and a safe, boring choice for an FYP | Recommend **PostgreSQL 16** (native install) — mature, well-documented, not bleeding-edge; SQLite fallback unchanged |
| Ollama | **v0.32.0** | `ollama pull phi4-mini` |
| Local LLM | **`phi4-mini`** (Microsoft, 3.8B params, ~2.2GB at Q4_K_M quantization) confirmed live in the Ollama library today | Primary pick; fallback `qwen2.5:1.5b` or `gemma3:1b` if too slow on your CPU |
| Next.js | **14.2.35** (patched; 14.2.5 has a known disclosed CVE — already fixed in your built project) | Already in your `package.json` |
| Node.js | 18.17+ required by Next.js 14; any current 20.x or 22.x LTS works | Use whatever LTS is installed |

### 0.4 One instruction that matters more than any table
Don't hand-copy version numbers into `requirements.txt` and stop. At install time:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers peft sentence-transformers spacy onnxruntime chromadb fastapi[standard] sqlalchemy psycopg2-binary python-jose[cryptography] passlib[bcrypt] python-multipart langchain ollama
```
Let `pip` resolve mutually compatible latest versions. Import everything, load one
smoke-test model, confirm nothing crashes (this is exactly where the AVX2 issue would
surface first, if it's going to). **Then** run `pip freeze > requirements.txt` to lock
in what actually worked on your real machine. Trust your own install over any table,
including this one, more than a few weeks old.

---

## 1. Project overview

Mentora is an AI-powered platform for students preparing for university-entry or
competitive government exams, that also guides them toward a career. Three journeys,
one shared brain:

1. **Exam Prep** — AI-generated study timetable, adaptive MCQ practice, automatic
   weak-topic detection (no self-reporting — detected from wrong answers).
2. **Career Counseling** — CV/transcript analysis → ranked matches across 50+ career
   paths → a concrete 6-month roadmap.
3. **Continuous Learning** — a personal library of curated videos, MCQ assignments,
   and course PDFs, auto-populated from the above two.

The connective tissue is the **Student Intelligence Profile (SIP)** — one record per
student holding their skill vector, topic scores, career rankings, priority queue, and
roadmap. Every feature reads from or writes to it.

### 1.1 The Six Fixed Objectives (do not change these regardless of tooling)
1. Maintain a Student Intelligence Profile (SIP) recording topic-wise performance per
   exam session.
2. Generate an adaptive AI study timetable and topic-level MCQ sessions.
3. Automatically detect weak topics from wrong answers via a knowledge-gap classifier.
4. Analyze an uploaded CV/transcript and rank the student against 50+ career paths via
   semantic matching.
5. Generate a concrete 6-month roadmap closing the gap to the chosen/recommended career.
6. Provide a retrieval-augmented (RAG) chatbot for topic explanations and mock
   interviews, grounded in curated content — not open-ended generation.

### 1.2 What's already done (Phase 0 — complete)
- **Frontend:** Next.js 14 (App Router, TypeScript) — all 8 routes (`/`, `/login`,
  `/register`, `/forgot-password`, `/exam-prep`, `/career`, `/learning-portal`,
  `/dashboard`), shared `SiteHeader`/`SiteFooter`/`AuthShell` components, `globals.css`
  design system (Fraunces + Inter, forest-green/mint palette), all forms carrying
  ERD-aligned `id`/`name` attributes, production-build-verified.
- **System design:** ERD, class diagram, activity diagram (all in §6–7 below).
- **Documentation:** the frontend doc this file supersedes/extends.

**Not built yet, and the subject of everything below:** the FastAPI backend, the
PostgreSQL database, the five trained ML models, the Model Manager, the RAG pipeline
(ChromaDB + Ollama), and the actual `fetch()` wiring inside the Next.js pages.

---

## 2. Final verified technology stack

| Layer | Technology | Notes |
|---|---|---|
| Frontend | **Next.js 14 (App Router) + TypeScript** — already built | Server Components by default; two files use `"use client"` (`SiteHeader`, `/exam-prep`) |
| Backend API | **FastAPI + Uvicorn** | REST JSON, auto Swagger docs at `/docs` |
| Auth | **python-jose (JWT) + passlib (bcrypt)**, delivered via httpOnly cookie | Set by a Next.js Route Handler, not `localStorage` |
| Database | **PostgreSQL 16 (native install)**, SQLite fallback | No Docker |
| ORM | **SQLAlchemy 2.x** | Mirrors the ERD in §6 |
| ML framework | **PyTorch 2.13 (CPU build)** | Single framework, no TensorFlow |
| Fine-tuning | **HuggingFace Transformers + PEFT (LoRA)** | Models 1, 2, 5 |
| Embeddings | **Sentence-Transformers (`all-mpnet-base-v2`)** | Model 4 |
| NLP preprocessing | **spaCy** | Supports NER tokenization for Model 5 |
| Local-inference optimization | **ONNX + ONNX Runtime dynamic quantization** | Avoids the AVX2/FBGEMM crash — see §0.2 |
| Local LLM serving | **Ollama → `phi4-mini`** (fallback `qwen2.5:1.5b`/`gemma3:1b`) | RAG generation |
| RAG orchestration | **LangChain** | Retrieve → prompt → generate |
| Vector DB | **ChromaDB 1.5.9, embedded/persistent mode** | No server, no Docker |
| Training environment | **Google Colab Free (T4 GPU)** | All 5 models trained here, never locally |
| Experiment tracking | **Weights & Biases (free tier)** | Optional |
| Version control | **GitHub** | — |

---

## 3. System architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│ FRONTEND — Next.js 14 (already built)                                │
│ App Router: / /login /register /forgot-password /exam-prep /career   │
│ /learning-portal /dashboard                                          │
│ Server Components (data fetch at render) + 2 Client Components       │
│ NEW (this plan): lib/api.ts, middleware.ts, app/api/auth/* route     │
│ handlers (set/clear the httpOnly session cookie)                     │
└───────────────────────────────┬──────────────────────────────────────┘
             Server Component fetch (server→server, same machine)
             Client Component fetch (browser→FastAPI directly)
                                 │  REST / JSON  →  http://localhost:8000
┌───────────────────────────────▼───────────────────────────────────────┐
│ BACKEND — FastAPI + Uvicorn                                            │
│ routers/  (auth, exam, career, learning, dashboard, interview)        │
│ services/ (ExamEngine, CareerEngine, SIPManager, AuthService)         │
│ middleware: CORSMiddleware + a JWT-verifying dependency on every      │
│ protected route (this is FastAPI's "middleware" — see §9.2)          │
│ ┌───────────────────────────────────────────────────────────────┐    │
│ │ MODEL MANAGER (Singleton, Lazy Load + LRU cache)                │   │
│ │ Always resident: M3 LSTM, M4 Sentence-Transformer (~500MB)      │   │
│ │ On-demand: M1 Gap Classifier, M2 Question Gen, M5 CV NER        │   │
│ │ Auto-unload after 5 min idle · serializes heavy inference       │   │
│ └───────────────────────────────────────────────────────────────┘    │
└───────────────┬─────────────────────────────┬─────────────────────────┘
                │                             │
   ┌────────────▼─────────────┐   ┌───────────▼──────────────────────┐
   │ ML MODELS                │   │ RAG + DATA LAYER                  │
   │ ONNX Runtime for M1/M2/M5│   │ ChromaDB (embedded, local files)  │
   │ plain PyTorch for M3/M4  │   │ Ollama → phi4-mini (:11434)       │
   │                          │   │ PostgreSQL 16 (native)            │
   └───────────────────────────┘   └───────────────────────────────────┘
```

**The golden request pattern (every button/form follows this):**
```
Browser click/submit → lib/api.ts fetch() → FastAPI router → JWT dependency
   → Service layer → Model Manager (load if needed, infer)
   → Service writes to Postgres/SIP → FastAPI returns JSON
   → React state updates → UI re-renders
```

---

## 4. Data model (ERD) — the contract the backend is built against

Core tables: `USERS` (1:1) `STUDENT_INTELLIGENCE_PROFILE` · `EXAM_SESSIONS` (1:*)
`SESSION_ANSWERS` (*:1) `QUESTIONS` · `TOPIC_SCORES` · `CAREER_PROFILES` (1:*)
`CAREER_RANKINGS` (*:1) `USERS` · `CAREER_PROFILES` (1:*) `ROADMAPS` · `CV_UPLOADS` ·
`INTERVIEW_SESSIONS` (1:*) `INTERVIEW_QA` · `NOTIFICATIONS`.

Build `backend/database/models.py` directly from this — every field name here should
match what the frontend already sends (see §8's field-name table, which was generated
from the actual `name=` attributes in the built Next.js forms).

---

## 5. The five ML models

| # | Model | Base | Job | Train target | Local runtime |
|---|---|---|---|---|---|
| M1 | Knowledge Gap Classifier | DeBERTa-v3-base | Multi-label: which topics is the student weak in | F1 ≥ 0.85 | ONNX + ONNX Runtime dynamic quant (on-demand) |
| M2 | Adaptive Question Generator | FLAN-T5-Large + LoRA | Generate new MCQs for a topic/difficulty | BLEU-4 ≥ 0.35, ROUGE-L ≥ 0.50 | ONNX-quantized base + small LoRA adapter (on-demand) |
| M3 | Performance Trajectory Predictor | 2-layer LSTM (128 units) | Predict if score trend is dropping/plateauing | MAE ≤ 5 | Plain PyTorch, always resident (~2MB) |
| M4 | Career-Skill Semantic Matcher | Sentence-Transformers `all-mpnet-base-v2` | Embed skills/careers, rank by cosine similarity | Precision@5 ≥ 0.75 | Plain PyTorch, always resident (~420MB) |
| M5 | CV/Transcript NER Extractor | BERT-base-uncased | Tag entities: SKILL, CERT, GPA in resume text | Entity F1 ≥ 0.82 | ONNX + ONNX Runtime dynamic quant (on-demand) |

**Train in this order** (cheapest/fastest first, so something end-to-end works early):
1. **M3 (LSTM)** — trains in minutes, no HF dependency drama. Unlocks the dashboard
   trend chart + "at risk" notification first.
2. **M4 (Sentence-Transformers)** — short fine-tune (4 epochs), unlocks the whole
   Career Counseling flow.
3. **M1 (DeBERTa Gap Classifier)** — the core "weak topic" feature; needs a labeled
   MCQ dataset ready first.
4. **M5 (BERT NER)** — needs a labeled (BIO-tagged) resume dataset; can run in
   parallel with M1.
5. **M2 (FLAN-T5-Large + LoRA)** — heaviest, do last. This is the one your Static
   Objectives explicitly require to stay FLAN-T5 — quantize/adapt around it, don't
   downgrade the base model.

---

## 6. RAG system (mock interview / topic explainer chatbot)

1. Student's question → embedded (same embedding family as M4).
2. **ChromaDB** (embedded mode) returns top 3–5 relevant chunks from `exam_kb` /
   `career_kb` / `interview_kb` collections.
3. **LangChain** assembles a "use ONLY this context" prompt.
4. **Ollama → `phi4-mini`** (or fallback) generates the grounded answer.
5. FastAPI stores the exchange in `INTERVIEW_QA` and returns it as JSON; the
   `/career` page's Client Component renders it.

This is why RAG was chosen over "just ask the model directly" — it grounds answers in
curated content instead of letting the model hallucinate, satisfying Objective 6.

---

## 7. Middleware — both meanings, fully explained

"Middleware" means two different things in this stack, on two different servers. Both
matter, and they do different jobs.

### 7.1 Next.js middleware (`middleware.ts`, frontend layer) — route protection
This runs **before a request reaches any page**, on Next.js's own server, before your
browser ever sees HTML. Its one job here: **don't render the app shell for someone
who isn't logged in.**

```ts
// middleware.ts (project root, sibling to app/)
import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";

const PROTECTED = ["/dashboard", "/exam-prep", "/career", "/learning-portal"];

export function middleware(request: NextRequest) {
  const session = request.cookies.get("mentora_session")?.value;
  const { pathname } = request.nextUrl;

  const isProtected = PROTECTED.some((p) => pathname.startsWith(p));
  const isAuthPage = pathname === "/login" || pathname === "/register";

  if (isProtected && !session) {
    return NextResponse.redirect(new URL("/login", request.url));
  }
  if (isAuthPage && session) {
    return NextResponse.redirect(new URL("/dashboard", request.url));
  }
  return NextResponse.next();
}

export const config = {
  matcher: [
    "/dashboard/:path*",
    "/exam-prep/:path*",
    "/career/:path*",
    "/learning-portal/:path*",
    "/login",
    "/register",
  ],
};
```

Rules for what belongs here: only check "does a session cookie exist and look
well-formed" — fast, no database call, since this runs on *every* request. It should
**not** call FastAPI, **not** do authorization logic ("does this student own this
record"), and **not** be treated as the real security boundary — treat it as UX
polish. The real security boundary is §7.2.

### 7.2 FastAPI middleware/dependencies (backend layer) — the real security boundary
Two separate mechanisms, both needed:

**(a) CORS middleware** — global, applies to every request, lets the browser's
`fetch()` calls from `http://localhost:3000` actually reach `http://localhost:8000`
(browsers block cross-origin requests by default):
```python
# backend/main.py
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,   # required since we're using cookies for the JWT
    allow_methods=["*"],
    allow_headers=["*"],
)
```

**(b) A JWT-verifying dependency** — not global middleware, but FastAPI's
`Depends()` system, applied per-route. This is where actual authorization happens:
```python
# backend/services/auth_service.py
from fastapi import Depends, HTTPException, Header
from jose import jwt, JWTError

def get_current_user(authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated")
    token = authorization.split(" ")[1]
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return payload["user_id"]
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

# backend/routers/dashboard.py
@router.get("/dashboard/stats")
def get_stats(user_id: int = Depends(get_current_user)):
    ...  # user_id is now guaranteed valid — this is the actual security check
```

Every protected endpoint gets `user_id: int = Depends(get_current_user)` in its
signature. This is intentionally *not* implemented as blanket ASGI middleware,
because some routes (e.g. `/auth/login`) must stay open — per-route dependency
injection is the correct FastAPI pattern here, not a global gate.

### 7.3 Why both layers exist
Next.js middleware = "don't show a logged-out user the dashboard shell" (fast, cheap,
UX). FastAPI's dependency = "actually verify this JWT before touching the database"
(the real gate — bypassable-in-theory Next.js middleware is not a substitute for it,
since anyone can call the FastAPI API directly with `curl`, skipping Next.js
entirely).

---

## 8. How the frontend and backend communicate

### 8.1 Two servers, one browser
Local setup: Next.js on `http://localhost:3000`, FastAPI on `http://localhost:8000`.
Next.js does **not** proxy to FastAPI automatically — `fetch()` calls target
`process.env.NEXT_PUBLIC_API_URL` directly.

### 8.2 Where API calls live in the frontend (new code to add)
```
lib/
  api.ts     # apiFetch(path, options) — prefixes NEXT_PUBLIC_API_URL,
              # attaches the session cookie, throws a typed error on non-2xx
  types.ts    # TypeScript types mirroring FastAPI's Pydantic response models
app/api/auth/
  login/route.ts     # Route Handler: calls FastAPI /auth/login, then sets
                        # the httpOnly cookie on the response — this is the
                        # only place the raw JWT ever touches Next.js's server
  logout/route.ts    # clears the cookie
```

### 8.3 Auth flow, end to end
1. User submits `login-form` on `/login` → a Client Component `onSubmit` calls
   `fetch("/api/auth/login", { method: "POST", body: JSON.stringify({email, password}) })`
   — note: this hits **Next.js's own** `/api/auth/login` route, not FastAPI directly.
2. That Next.js Route Handler calls FastAPI's `POST /auth/login`, gets back a JWT.
3. The Route Handler sets that JWT as an **httpOnly, `SameSite=Lax` cookie** on its
   response, then returns `{ success: true }` (never the raw token) to the browser.
4. `middleware.ts` (§7.1) reads that cookie on every subsequent navigation.
5. For calls that need to reach FastAPI directly from a Server Component (e.g. the
   dashboard fetching stats at render time), read the cookie server-side with
   `cookies()` from `next/headers` and forward it as `Authorization: Bearer <token>`
   to FastAPI.
6. For calls from a Client Component (e.g. submitting the MCQ form), also route
   through a small Next.js Route Handler rather than calling FastAPI directly from the
   browser — this keeps the JWT server-side only, at the cost of one extra hop.

### 8.4 Environment variables
```
# frontend/.env.local
NEXT_PUBLIC_API_URL=http://localhost:8000
JWT_COOKIE_NAME=mentora_session
```
```
# backend/.env
DATABASE_URL=postgresql://postgres:<password>@localhost:5432/mentora_db
JWT_SECRET=<random 32+ char string>
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=phi4-mini
```

---

## 9. Complete folder structure (monorepo)

```
mentora/
├── frontend/                     # the Next.js app — already built, this plan adds:
│   ├── middleware.ts               # NEW — §7.1
│   ├── lib/api.ts, lib/types.ts     # NEW — §8.2
│   ├── app/api/auth/login/route.ts  # NEW — §8.2
│   ├── app/api/auth/logout/route.ts # NEW
│   └── (all existing app/, components/, globals.css — unchanged)
│
├── backend/
│   ├── main.py
│   ├── requirements.txt
│   ├── config.py
│   ├── .env
│   ├── database/          # SQLAlchemy models + session, built from the ERD (§6)
│   ├── routers/             # auth, exam, career, learning, dashboard, interview
│   ├── services/            # ExamEngine, CareerEngine, SIPManager, AuthService
│   └── ml_core/
│       └── model_manager.py
│
├── ml_models/
│   └── saved/
│       ├── model1_gap_classifier/        # ONNX + quantized
│       ├── model2_question_generator/    # ONNX base + lora_adapter/
│       ├── model3_trajectory_predictor/  # plain PyTorch .pt
│       ├── model4_career_matcher/        # plain Sentence-Transformers folder
│       └── model5_cv_ner/                # ONNX + quantized
│
├── rag_pipelines/{exam,career,interview}_rag/ingest.py
├── vector_store/{exam_kb,career_kb,interview_kb}/
├── notebooks/                # Colab training notebooks (synced via Drive)
└── scripts/
    ├── start_backend.bat / .sh
    ├── start_frontend.bat / .sh
    └── start_ollama.bat / .sh
```

---

## 10. Consolidated API endpoint reference

Grouped by which Next.js page calls them (method, purpose, ERD table touched):

| Endpoint | Method | Called from | Purpose |
|---|---|---|---|
| `/auth/register` | POST | `/register` | Create USER + seed SIP, return JWT |
| `/auth/login` | POST | `/login` | Verify credentials, return JWT |
| `/auth/forgot-password` | POST | `/forgot-password` (step 1) | Send 6-digit reset code |
| `/auth/reset-password` | POST | `/forgot-password` (step 2) | Verify code, set new password |
| `/users/me` | GET / PATCH / DELETE | `/dashboard` | Profile read/edit/delete |
| `/auth/change-password` | POST | `/dashboard` (Settings) | — |
| `/exam/timetable` | GET | `/exam-prep` | Read `SIP.priority_queue`, render AI timetable |
| `/exam/today-topic` | GET | `/exam-prep` | Topic explainer + matched videos |
| `/exam/mcq` | GET | `/exam-prep` | Runs M2 (Question Generator) |
| `/exam/mcq/submit` | POST | `/exam-prep` | Runs M1 (Gap Classifier) → updates `TOPIC_SCORES` |
| `/exam/subjects` | PATCH | `/exam-prep` (subject chips) | Save preferences |
| `/career/cv-upload` | POST (multipart) | `/career` | Runs M5 (CV NER) → extracted skills/GPA/certs |
| `/career/goal` | POST | `/career` | Sets career target |
| `/career/ai-suggest` | GET | `/career` | Runs M4 (Career Matcher) |
| `/career/rankings` | GET | `/career` | Reads `CAREER_RANKINGS` |
| `/career/roadmap` | GET | `/career` | Reads `ROADMAPS` |
| `/interview/start` | POST | `/career` | Creates `INTERVIEW_SESSIONS` row |
| `/interview/{id}/ask` | POST | `/career` | RAG pipeline (ChromaDB + Ollama), writes `INTERVIEW_QA` |
| `/learning/playlist` | GET | `/learning-portal` | Video rows |
| `/learning/assignments` | GET | `/learning-portal` | MCQ assignment cards |
| `/learning/materials` | GET | `/learning-portal` | PDF cards |
| `/learning/materials/{id}/download` | GET | `/learning-portal` | Streams file |
| `/dashboard/stats` | GET | `/dashboard` | 4 stat cells |
| `/dashboard/score-trend` | GET | `/dashboard` | Feeds the score chart (swap raw SVG for Recharts) |
| `/dashboard/weekly-snapshot` | GET | `/dashboard` | 5-day grid |
| `/dashboard/weak-topics` | GET | `/dashboard` | Reads `TOPIC_SCORES` |
| `/notifications` | GET | `/dashboard` | Reads `NOTIFICATIONS` |

Every field name in these request/response bodies should match the `name=` attributes
already present in the built Next.js forms — see `MENTORA_Frontend_Documentation.md`
§8 for the exact list (`first_name`, `last_name`, `exam_goal`, `career_goal`,
`cv_file`, `answer_Q1001`-style per-question radio names, etc.).

---

## 11. Development phases — incremental, Colab-trained, then local

### Phase 0 — Frontend (✅ complete)
Next.js 14 app, all 8 routes, verified production build.

### Phase 1 — Backend skeleton & auth (1–2 weeks)
- Scaffold `backend/` per §9.
- `database/models.py` from the ERD; create tables (Postgres native or SQLite
  fallback).
- `/auth/register`, `/auth/login`, `/auth/forgot-password`, `/auth/reset-password`.
- CORS + JWT dependency (§7.2).
- **Milestone:** Register + Login work end-to-end from the real `/register`/`/login`
  pages, cookie is set, `/dashboard` redirect succeeds via `middleware.ts`.

### Phase 2 — Wire the existing frontend to the backend (few days, mostly mechanical)
- Add `lib/api.ts`, `middleware.ts`, and the `app/api/auth/*` route handlers (§7–8).
- Add `onSubmit`/`onClick` handlers to the forms/buttons listed in
  `MENTORA_Frontend_Documentation.md` §7–8 (this requires adding `"use client"` to
  pages that don't already have it).
- **At this stage, every FastAPI endpoint returns mock/hardcoded JSON** — no models
  yet. This decouples "is the wiring correct" from "are the models trained yet."
- **Milestone:** click every button on every page once; each produces a network
  request in DevTools and a UI update, even with fake data.

### Phase 3 — Datasets (parallel with Phase 1–2)
- Gather/clean: MDCAT/CSS/NTS-style MCQ banks (M1, M2), a labeled resume/CV dataset
  (M5), O*NET + a career-profile dataset (M4), per-student score histories (synthetic
  OK for M3 if real data isn't available yet).
- Save all raw + processed data to Google Drive, never Colab's temp disk.

### Phase 4 — Train models on Colab (order per §5)
- One notebook per model, checkpoint to Drive every epoch
  (`resume_from_checkpoint=True` if disconnected).
- Hit each target metric before moving on.

### Phase 5 — Export & optimize for local inference
- M3, M4: save as-is (`torch.save` / `SentenceTransformer.save`) — no quantization.
- M1, M2, M5: export to ONNX (`torch.onnx.export`), quantize with
  `onnxruntime.quantization.quantize_dynamic()` (§0.2). Download all
  `ml_models/saved/...` folders to the laptop.

### Phase 6 — Model Manager + real inference wiring
- Implement `ModelManager` (Singleton, lazy `load()`/`unload()`/`infer()`, LRU + 5-min
  idle unload, serialized heavy calls).
- Swap Phase 2's mock JSON for real `ModelManager.infer(...)` calls, one endpoint at a
  time — start with M3/M4 (always-loaded, simplest).
- **Milestone:** submit a real MCQ test → real weak-topic result on Dashboard; upload
  a real CV → real extracted skills + career ranking.

### Phase 7 — RAG pipeline
- `ollama pull phi4-mini`, confirm `ollama run phi4-mini "test prompt"` responds
  (expect it to be slow — that's expected on this CPU, not a bug).
- Build `rag_pipelines/{exam,career,interview}_rag/ingest.py` → populate 3 ChromaDB
  collections.
- Implement `RAGPipeline` → `ExamRAG`/`CareerRAG`/`InterviewRAG`; wire
  `/interview/start` and `/interview/{id}/ask`.
- **Milestone:** "Start Mock Interview" on `/career` returns a real, context-grounded
  reply.

### Phase 8 — Integration testing & performance tuning
- Run the 11-point end-to-end checklist (register → login → timetable → MCQ →
  dashboard weak topics → CV upload → career ranking → roadmap → mock interview →
  learning portal → forgot password).
- Measure real RAM idle vs. peak on *this* machine (§12) — don't trust any prior
  budget written for different hardware.
- If `phi4-mini` is too slow, swap to `qwen2.5:1.5b`/`gemma3:1b` — one `.env` line
  (`OLLAMA_MODEL=...`), nothing else changes.

### Phase 9 — Polish & demo packaging
- `.bat`/`.sh` one-click startup scripts for Ollama/backend/frontend.
- Optional: pre-cache a few "known good" demo interactions in case live inference is
  too slow on demo day — a common, acceptable FYP practice; mention it openly to your
  supervisor rather than risk a frozen demo.

### Phase 10 — Final documentation
- Document backend architecture, actual model results (F1/BLEU/MAE/Precision@5
  achieved vs. targets), and the AVX2/hardware adaptation as a "design decision under
  real-world constraints" section — this is a legitimately strong point for a defense,
  since it shows a diagnosed and solved engineering constraint, not just a template
  followed blindly.

---

## 12. RAM budget — re-estimate on your real hardware, don't trust old numbers

Rough starting expectations for an 8GB / 3rd-Gen i5 machine (re-measure once Phase 6–7
are running):

| State | Components | Approx. RAM |
|---|---|---|
| Idle | OS + FastAPI + M3/M4 always-loaded + Postgres + Next.js (`next start`) | ~3.5–4GB |
| Peak (M2 or M5 loaded on-demand) | Idle + one ONNX-quantized model (~0.3–0.8GB) | ~4.5–5GB |
| Peak (Ollama generating) | Idle + `phi4-mini` (~2.2–3GB) | ~6.5–7GB |

The Model Manager's job (§3, §11 Phase 6) is to make sure the second and third rows
never happen at the same instant — serialize heavy inference, never run Q-Gen and Mock
Interview concurrently. As a safety net, increase the swap/page file to ~12GB.

---

## 13. Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| FBGEMM/AVX2 crash on quantized models | High if unmitigated | ONNX Runtime quantization instead of PyTorch FBGEMM for M1/M2/M5 (§0.2) |
| Local LLM too slow on 3rd-Gen CPU | High | `phi4-mini`/`qwen2.5:1.5b`; accept slower tok/s; pre-cache demo answers as fallback |
| 8GB RAM overload at peak | Medium | Model Manager serializes heavy tasks; swap ~12GB |
| FLAN-T5 fine-tuning exceeds Colab free GPU quota | Medium | LoRA only; checkpoint every epoch to Drive |
| CORS / cookie misconfiguration blocking frontend↔backend | Medium (day-1 issue) | `allow_origins=["http://localhost:3000"]`, `allow_credentials=True`, `SameSite=Lax` cookie |
| Frontend/backend integration delays | Low if Phase 2's mock-JSON step is done first | Decouple wiring from model-readiness |

---

## 14. Immediate next steps (this week)

1. Run the AVX2 check (§0.2) on the real laptop — this decides whether the ONNX
   detour is actually needed, or whether standard PyTorch quantization is fine.
2. Scaffold `backend/` (Phase 1) and get Register/Login working against the real
   `/register`/`/login` pages, including the cookie-setting Route Handler (§7–8).
3. Start the M3 (LSTM) and M4 (Sentence-Transformers) Colab notebooks in parallel —
   cheapest wins first, keeps momentum while the heavier models train later.

---

*This document is the single current source of truth for the project plan. It
supersedes `Mentora_Master_Plan.md` wherever the two disagree (specifically: the
frontend is Next.js, not static HTML — everything else in that document that isn't
about the frontend framework was re-verified and carried forward unchanged). It
extends `MENTORA_Frontend_Documentation.md` rather than replacing it — keep that file
as the frontend-specific reference and this one as the whole-project reference.*
