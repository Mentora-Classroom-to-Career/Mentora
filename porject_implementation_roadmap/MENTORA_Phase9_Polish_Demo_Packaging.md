# MENTORA — Phase 9 Deep Dive: Polish & Demo Packaging

*Prerequisite: Phase 8 is done — the system has been tested end to end, real RAM/
latency numbers are measured, and the final LLM choice (`phi4-mini` or a fallback)
is locked in. This phase turns "it works when I run it carefully" into "it works
reliably in front of an audience who can't see what's happening behind the scenes."
Estimated time: 3–5 days, plus rehearsal time before the actual defense/demo date.*

---

## 0. The mindset shift for this phase

Every prior phase optimized for "does this work correctly." This phase optimizes for
a different, narrower question: **"does this work correctly, in front of people, on a
specific day, without me being able to fix anything live."** That reframing changes
what's worth spending time on — a slightly-slower-but-bulletproof demo path beats a
faster one with a single point of failure, every time, for this specific phase.

---

## 1. Step 1 — one-click startup scripts

### 1.1 Why this matters more than it looks like it should
By this point, starting the app means: start Ollama (usually automatic), activate a
venv, `uvicorn main:app`, `cd` to the frontend, `npm run start` — in the right order,
in the right terminals. Under demo pressure, in front of people, this is exactly the
kind of multi-step manual process where a typo or wrong directory costs you visible,
stressful minutes. Collapse it to one double-click.

### 1.2 Windows batch scripts
```bat
:: scripts/start_ollama.bat
@echo off
echo Starting Ollama...
start "Ollama" ollama serve
timeout /t 3 /nobreak >nul
echo Ollama started.
```
```bat
:: scripts/start_backend.bat
@echo off
cd /d "%~dp0..\backend"
call venv\Scripts\activate
echo Starting FastAPI backend on port 8000...
uvicorn main:app --port 8000
```
```bat
:: scripts/start_frontend.bat
@echo off
cd /d "%~dp0..\frontend"
echo Starting Next.js frontend on port 3000...
npm run start
```
```bat
:: scripts/start_all.bat — the actual one-click entry point
@echo off
echo Launching MENTORA (backend, frontend, Ollama)...
start "Ollama" cmd /k "%~dp0start_ollama.bat"
timeout /t 5 /nobreak >nul
start "Backend" cmd /k "%~dp0start_backend.bat"
timeout /t 8 /nobreak >nul
start "Frontend" cmd /k "%~dp0start_frontend.bat"
timeout /t 5 /nobreak >nul
start http://localhost:3000
echo All services launching in separate windows. Browser will open shortly.
```
The staggered `timeout` delays matter — starting the frontend before the backend is
actually listening leads to the first page load failing with a connection error,
which looks alarming even though it's harmless and a refresh would fix it. Better to
just not show that in the first place.

### 1.3 macOS/Linux equivalents
```bash
#!/bin/bash
# scripts/start_all.sh
echo "Launching MENTORA..."
ollama serve &
sleep 3
(cd backend && source venv/bin/activate && uvicorn main:app --port 8000) &
sleep 8
(cd frontend && npm run start) &
sleep 5
open http://localhost:3000 2>/dev/null || xdg-open http://localhost:3000
echo "All services launching."
```
```bash
chmod +x scripts/*.sh   # do this once — a script without execute permission is a
                          # classic last-minute demo-morning surprise
```

### 1.4 A pre-flight check script — catch problems BEFORE double-clicking start_all
```python
# scripts/preflight_check.py
import socket, subprocess, sys, os

def port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) != 0

def check(label, condition):
    status = "OK" if condition else "FAIL"
    print(f"[{status}] {label}")
    return condition

all_ok = True
all_ok &= check("Port 8000 free (backend)", port_free(8000))
all_ok &= check("Port 3000 free (frontend)", port_free(3000))
all_ok &= check("backend/.env exists", os.path.exists("backend/.env"))
all_ok &= check("frontend/.env.local exists", os.path.exists("frontend/.env.local"))
all_ok &= check("ml_models/saved/ populated", len(os.listdir("ml_models/saved")) >= 5)
all_ok &= check("vector_store/ populated", os.path.exists("vector_store") and len(os.listdir("vector_store")) >= 3)

try:
    result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=5)
    all_ok &= check("Ollama model available", "phi4-mini" in result.stdout or "qwen2.5" in result.stdout)
except FileNotFoundError:
    all_ok &= check("Ollama installed", False)

sys.exit(0 if all_ok else 1)
```
Run this the morning of any demo, rehearsal, or defense — it catches "the ports are
already in use because yesterday's session never fully closed" and similar
low-drama-but-annoying issues in seconds, before they become live-demo issues.

---

## 2. Step 2 — demo data: replace random test data with a curated "hero" account

### 2.1 Why Phase 8's Faker-seeded data isn't right for this
Random synthetic data (Phase 8 §9) is exactly right for load/volume testing and
exactly wrong for a demo — a demo account with a nonsensical name, an oddly-timed
score history, or a career ranking that doesn't visually tell a clear story
undermines the pitch even if every number is technically correctly computed.

### 2.2 Build one deliberately-designed demo account
```python
# scripts/seed_demo_account.py
"""Creates ONE specific, hand-crafted student account for demos.
Not random — every value here is chosen to tell a clear, coherent story
on screen: a student who is clearly strong in some areas, has 1-2
believable weak topics that show off the gap-classifier, and a CV that
produces a clean, legible career ranking."""

DEMO_EMAIL = "demo@mentora.pk"
DEMO_PASSWORD = "MentoraDemo2026!"

def seed():
    user = create_user(
        first_name="Ayesha", last_name="Memon", email=DEMO_EMAIL, password=DEMO_PASSWORD,
        university="University of Sindh, Laar Campus", exam_goal="sindh_university",
    )
    # a score history that trends clearly upward except one obviously-weak topic —
    # tells a visually clean "AI caught this in time" story on the dashboard
    scores = [
        ("Algebra", 45), ("Algebra", 52), ("Algebra", 48),      # stays weak — the demo's "gap" story
        ("English", 60), ("English", 72), ("English", 81),       # improving — the demo's "trajectory" story
        ("Geometry", 70), ("Geometry", 75), ("Geometry", 78),
    ]
    for topic, score in scores:
        create_exam_session(user.id, topic=topic, score=score)

    # a real, clean CV file with clearly data-science-leaning content, so the
    # career ranking demo produces an obviously-sensible #1 match
    upload_demo_cv(user.id, "scripts/demo_assets/ayesha_cv.pdf")

if __name__ == "__main__":
    seed()
```
Spend real time picking the actual demo CV file and getting the score-history shape
right — this is the single highest-leverage few hours in this whole phase for how the
demo *feels*, independent of anything technical.

### 2.3 Keep this account stable — don't let Phase 8's automated tests touch it
Make sure `scripts/run_all_tests.sh` (Phase 8 §10.1) runs against the test DB, never
against whatever DB the demo account lives in — a regression test accidentally
deleting or corrupting your carefully-built demo account the night before a defense
is an avoidable, entirely self-inflicted disaster.

---

## 3. Step 3 — handling slow inference gracefully, and being upfront about it

### 3.1 Pre-warming — the easy, honest win
Before any demo/rehearsal, send one throwaway request to each on-demand model and to
Ollama, a few minutes ahead of time, so the Model Manager's lazy-loading (Phase 6) and
Ollama's `keep_alive` (Phase 7 §1.4) both have everything already resident when it
actually matters:
```bash
# scripts/prewarm_demo.sh
curl -s http://localhost:8000/exam/mcq?topic=quadratic-equations > /dev/null
curl -s http://localhost:8000/career/ai-suggest > /dev/null
curl -s -X POST http://localhost:8000/interview/start?career_title=Data+Scientist > /dev/null
echo "Models pre-warmed."
```
Run this 2–3 minutes before starting, not the moment before — give the loads time to
actually finish.

### 3.2 The honest, legitimate fallback: a small cache of known-good demo responses
This is explicitly called out in the master plan as acceptable practice, worth
repeating here because it's easy to feel like it's "cheating" — **it isn't, as long
as you're upfront about it.** Real production systems cache expensive computations
routinely; doing the same for a live, timed academic demo on genuinely constrained
hardware is a legitimate engineering decision, not a shortcut that misrepresents the
system's real capability (which was already measured honestly back in Phase 8).

```python
# backend/services/demo_cache_service.py
"""OPTIONAL demo-mode cache. Only active when DEMO_MODE=true in .env.
Serves pre-recorded (but genuinely model-generated, at some earlier point)
responses for the exact handful of prompts used in the rehearsed demo script,
falling back to live inference for anything else. This exists purely to
remove timing risk from a live, timed presentation — it is not used in
normal operation, and is documented as such in the thesis."""

import json, os

DEMO_CACHE_PATH = "backend/data/demo_cache.json"

def get_cached_response(cache_key: str) -> dict | None:
    if not os.getenv("DEMO_MODE") == "true":
        return None
    if not os.path.exists(DEMO_CACHE_PATH):
        return None
    with open(DEMO_CACHE_PATH) as f:
        cache = json.load(f)
    return cache.get(cache_key)
```
Populate `demo_cache.json` by actually running the exact demo script's prompts
through the real pipeline once, saving the real output — this is a cache of genuine
model output, not hand-written fake answers. Wire it as a first check in the relevant
router functions (`if cached := get_cached_response(key): return cached`), falling
through to the real `ModelManager`/RAG call otherwise. **Mention this openly to your
supervisor and in the thesis** (Phase 10 will have a natural place for this) — it's a
sign of engineering maturity, not something to hide.

### 3.3 What NOT to do
Don't disable the real pipeline entirely and demo only cached responses — that
crosses from "removing timing risk" into "not actually demoing the system." The cache
should cover the specific handful of interactions in your rehearsed script; anything
a judge asks that's off-script should still hit the real, live pipeline, and you
should be prepared for that to take the real, measured amount of time.

---

## 4. Step 4 — UI polish pass

Go through every page with this specific checklist, not just "does it look fine":

- **Loading states.** Every fetch from Phase 2's wiring should show *something*
  (skeleton, spinner, "Loading your timetable…") rather than a blank flash — a blank
  page for even half a second during a projected demo reads as broken even when it
  isn't.
- **Error states.** Force a few real errors (stop the backend mid-session, disconnect
  wifi) and confirm the frontend shows a legible message, not a raw stack trace or an
  infinitely-spinning loader.
- **Remove/gate every debug artifact.** Phase 6 §6's `/debug/model-status` endpoint,
  any `console.log` left in the frontend, any hardcoded "demo note" text still saying
  `demo@mentora.pk / demo2025` from the original static HTML (Phase 2 §3.4 already
  flagged deleting this — double check it's actually gone) — all of it either removed
  or explicitly gated behind `if settings.DEBUG`.
- **Favicon and page titles.** Small, easy to forget, and one of the first things
  anyone notices about a "finished-feeling" app vs. a prototype.
- **Test on the actual presentation hardware if at all possible** — screen
  resolution, projector color/contrast, and browser zoom level can all make a design
  that looks perfect on your laptop screen look cramped or oversized elsewhere.

---

## 5. Step 5 — graceful degradation for the failure modes most likely on demo day

| If this fails mid-demo... | ...the app should | Why |
|---|---|---|
| Ollama crashes/hangs | Mock-interview and topic-explainer show a clear "temporarily unavailable" message, rest of the app keeps working | One feature failing shouldn't look like the whole system is broken |
| A model fails to load (corrupted file, path issue) | The specific feature fails gracefully with a message; other endpoints unaffected | Same reasoning — isolate blast radius |
| Wifi drops (if anything relies on it — it shouldn't, everything here runs local) | Confirm nothing in the demo path actually needs internet | Double-check no stray `fonts.googleapis.com` runtime call, no accidental external API call snuck in anywhere |
| Backend restarts unexpectedly | Frontend shows a connection-lost state, recovers automatically once backend is back, doesn't require a hard refresh | Add a simple retry/reconnect check in `lib/api.ts` if time allows |

Deliberately trigger each of these once during rehearsal (§7) rather than hoping they
don't happen — seeing the actual failure-mode behavior once, calmly, beforehand is
much better than discovering it live.

---

## 6. Step 6 — backup and recovery plan

- **Database snapshot.** Right after seeding the demo account (§2) and confirming
  everything looks right, export a clean snapshot:
  ```bash
  pg_dump mentora_db > backups/demo_snapshot_$(date +%Y%m%d).sql
  # or, if on the SQLite fallback:
  cp mentora.db backups/demo_snapshot_$(date +%Y%m%d).db
  ```
  If anything gets corrupted or messed up during a rehearsal, restoring this snapshot
  is a two-minute fix instead of a re-seed-and-re-verify scramble.
- **Screen recording of one full, successful run.** A genuine safety net, not just a
  nice-to-have — if live inference is having a bad moment on the actual day (still
  possible even with all this preparation, given real hardware variability), a
  recorded backup of the exact same flow working correctly is a legitimate thing to
  fall back to, and is far better than an unrecoverable live failure with no plan B.
- **Git tag the exact working commit.** `git tag v1.0-demo && git push --tags` —
  gives you a precise, restorable point to return to if a last-minute change
  introduces a regression the night before.

---

## 7. Step 7 — rehearsal

- Run the **exact planned demo script** at least 3 times, start to finish, using
  `start_all` (§1) from a cold start each time — not from an already-running,
  already-warmed environment, since that's not what demo day actually looks like.
- Time it. If it's meant to fit a 10-minute slot, know exactly which parts to
  compress if running long.
- Rehearse **on the actual presentation device and network** at least once if
  logistically possible — a laptop that's fine at your desk can behave differently
  on unfamiliar wifi, a different power setting (battery vs. plugged in can affect
  CPU throttling on an older machine — genuinely relevant here), or a projector setup.
- Have one rehearsal where you **deliberately let something go wrong** (§5's table) so
  the recovery isn't the first time you've seen it happen.

---

## 8. Step 8 — packaging for submission/handoff

- `pip freeze > backend/requirements.txt` — final, frozen, exact versions that are
  known to work together on this machine (not the unpinned install-time versions from
  earlier phases — those were for finding what's current, this is for reproducibility).
- A top-level `README.md` consolidating setup instructions from every phase into one
  document a new person (or an examiner trying to run it) could follow start to
  finish — this is different from, and simpler than, the deep architectural
  documentation that Phase 10 covers.
- Confirm `.gitignore` actually excludes `node_modules/`, `venv/`, `.env`/`.env.local`
  (secrets), `ml_models/saved/` and `vector_store/` if they're too large for git
  (document how to obtain/regenerate them instead — e.g. "download from this Drive
  link" or "run Phase 4–5's notebooks").
- The screen recording from §6 as a submitted fallback artifact alongside the code,
  if your program allows/expects one.

---

## 9. Common pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| `start_all.bat` opens windows but nothing loads | Port already in use from a previous unclosed session | Run `preflight_check.py` (§1.4) first, always |
| Demo account's data looks different than what you remember seeding | An automated test run (Phase 8) wrote to the same DB | Confirm test suite config points at a separate test DB, restore from the §6 snapshot |
| Cached demo responses (§3.2) feel stale/wrong after later model changes | Cache built before a later retraining/tuning pass, never regenerated | Regenerate `demo_cache.json` any time a model or prompt template changes |
| Everything works on your desk, breaks on the presentation machine | Different Python/Node version, missing `.env`, or a hardcoded `localhost` assumption that doesn't hold on a different machine's network config | Rehearse on the actual device (§7); keep the frozen `requirements.txt`/`package-lock.json` consistent |
| Battery-powered laptop noticeably slower than when plugged in during rehearsal | Older CPUs often throttle aggressively on battery power management profiles | Always present plugged in; set the OS power plan to "High Performance" beforehand |

---

## 10. Definition of done

- [ ] One-click start scripts (Windows and/or macOS/Linux, matching your actual
      presentation machine) tested from a genuine cold start
- [ ] `preflight_check.py` passes cleanly before any rehearsal or the real demo
- [ ] A deliberately-curated demo account exists, seeded, verified, and snapshotted
- [ ] Every on-demand model and Ollama pre-warmed via `prewarm_demo.sh` as a standard
      pre-demo step
- [ ] (Optional but recommended) demo-response cache built and tested for the exact
      rehearsed script, with `DEMO_MODE` clearly documented as opt-in and disclosed
- [ ] UI polish pass complete — no debug endpoints exposed, no leftover placeholder
      text, loading/error states verified by deliberately triggering them
- [ ] Backup snapshot + git tag + screen recording of one full successful run exist
- [ ] At least 3 full rehearsals completed, including one deliberately-broken run to
      practice recovery
- [ ] `requirements.txt`/`package-lock.json` frozen, `README.md` consolidated,
      `.gitignore` verified

With this checked off, the system isn't just functionally complete — it's ready to be
shown to people under real conditions. Phase 10 (final documentation) is what's left,
and can largely be written in parallel with this phase's rehearsals rather than
strictly after them.
