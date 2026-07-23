# MENTORA Backend

FastAPI backend for the MENTORA FYP. Current status: **Phase 1 (auth) is
real; Phase 2 (dashboard/exam/career/learning/interview) is mock JSON**,
shaped exactly like the real future responses so the frontend never
needs to change when Phase 6+ swaps in real model inference.

## Setup

```bash
cd backend
pip install -r requirements.txt --break-system-packages   # or use a venv
cp .env.example .env                                       # then edit JWT_SECRET
uvicorn main:app --reload --port 8000
```

Visit http://localhost:8000/health to confirm it's running, or
http://localhost:8000/docs for the interactive API docs.

Defaults to a local SQLite file (`mentora.db`) with zero setup. Point
`DATABASE_URL` in `.env` at Postgres for the real target stack described
in the master plan.

## What's real vs. mock

| Router | Status |
|---|---|
| `/auth/*`, `/users/me` | Real — DB-backed, JWT auth |
| `/dashboard/*` | Mock JSON |
| `/exam/*` | Mock JSON (MCQ generation, topic explainer) |
| `/career/*` | Mock JSON (CV extraction, rankings, roadmap) |
| `/learning/*` | Mock JSON |
| `/interview/*` | Mock JSON (canned RAG responses) |
| `/notifications` | Mock JSON |

Every mock router has a comment pointing to which later phase (6 or 7)
replaces it with a real model/DB call — see the docstring at the top of
each file in `routers/`.

## Structure

```
backend/
  main.py              # FastAPI app, CORS, router registration
  config.py             # env-driven settings
  schemas.py             # Pydantic request/response models
  database/
    models.py            # SQLAlchemy models (full ERD)
    session.py            # engine/session/get_db dependency
  routers/               # one file per feature area
  services/
    auth_service.py       # password hashing, JWT, get_current_user
  ml_core/                # placeholder — Phase 6 ModelManager lands here
  tests/                  # placeholder — Phase 8
```

## Known local-environment note

`passlib`'s bcrypt backend detection breaks against `bcrypt>=4.1` (it
probes a `bcrypt.__about__` attribute recent bcrypt releases removed).
`services/auth_service.py` calls the `bcrypt` library directly instead
of going through `passlib.CryptContext` to sidestep this — no action
needed, just don't reintroduce passlib for hashing without pinning
`bcrypt<4.1` first.
