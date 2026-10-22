# Mentora Frontend — Complete Technical Documentation

**Purpose of this document:** a single reference for the frontend as it exists today,
how Next.js works under the hood so the "why" of each decision is clear, and exactly
what the FastAPI backend needs to expose (routes, request/response shapes, auth
behavior) to make every page fully functional. Treat Section 8 (Forms & Fields
Inventory) and Section 10 (API Contract) as the spec to build the backend against.

---

## 1. What was built

The original 8 static HTML pages (`index.html`, `login.html`, `register.html`,
`forgot-password.html`, `exam-prep.html`, `career.html`, `learning-portal.html`,
`dashboard.html`) were converted into a **Next.js 14 (App Router) + TypeScript**
application. Nothing visual changed — same layout, same CSS, same copy — but the
markup is now componentized, routed, and typed, and it's structured so backend calls
can be dropped in without restructuring anything.

| Layer | Choice | Why |
|---|---|---|
| Framework | Next.js 14, App Router | Matches Part 2/3 of the MENTORA tech doc (Next.js 14, production build served locally) |
| Language | TypeScript | Catches typos in prop names / form field names before they become backend bugs |
| Styling | Plain CSS (`globals.css`, ported 1:1 from `styles.css`) + CSS variables | Zero build-step styling risk; keeps the exact design system (`--green-950`, `--font-display`, etc.) |
| Fonts | `next/font/google` (Fraunces, Inter) | Self-hosts fonts at build time — no runtime request to Google, faster page loads, works offline once built |
| State | React `useState` only where needed (e.g. exam-prep subject chips) | Everything else is static markup; no global state library is needed yet |
| Data fetching | **Not implemented yet** | This is the backend's job — see Section 10 |

---

## 2. How Next.js (App Router) actually works

This section explains the mental model so any change you or a teammate makes later is
predictable.

### 2.1 File-system routing
Every folder inside `app/` is a URL segment, and a `page.tsx` inside that folder is
what renders at that URL. There's no router config file — the folder *is* the route.

```
app/page.tsx              →  /
app/login/page.tsx         →  /login
app/exam-prep/page.tsx     →  /exam-prep
app/career/page.tsx        →  /career
```

A `layout.tsx` in a folder wraps every `page.tsx` inside it (and its subfolders). We
use exactly one root layout (`app/layout.tsx`) for the whole site — it loads fonts,
imports `globals.css`, and sets the default `<title>`. Two routes (`exam-prep`,
`career`) also have a tiny local `layout.tsx` whose only job is setting that route's
`<title>`, because their `page.tsx` is a Client Component and Client Components are
not allowed to export page metadata (see 2.2).

### 2.2 Server Components vs. Client Components — the single most important Next.js concept
By default, **every component in `app/` runs on the server** and is never shipped to
the browser as JavaScript. It renders to HTML once, on the server (or at build time),
and that's what the browser gets. This is why `app/page.tsx`, `dashboard/page.tsx`,
`career/page.tsx`, `learning-portal/page.tsx`, and the two auth-shell pages are plain
functions with no `"use client"` — they have no interactivity, so there's no reason to
send React to the browser for them at all.

A file becomes a **Client Component** — meaning it *does* ship JS and can use
`useState`, `useEffect`, `onClick`, `usePathname`, etc. — only when it starts with:

```tsx
"use client";
```

In this project, exactly two files need that:
- **`components/SiteHeader.tsx`** — uses `usePathname()` to know which nav link is
  "active" right now.
- **`app/exam-prep/page.tsx`** — uses `useState` to toggle the Mathematics/
  Chemistry/Urdu subject chips.

Everything else is a Server Component. When you wire up the backend, any page or
component that needs to call `fetch()` from the *browser* (e.g. in response to a
button click), hold form input as it's typed, or show a loading spinner will need
`"use client"` added to it. Pages that only need data *before* the page renders (e.g.
"show this student's dashboard stats") can instead fetch that data on the server
inside the Server Component itself with `async function Page() { const data = await
fetch(...) }` — no `"use client"` needed for that case. This distinction matters for
the backend team too: server-side fetches happen from your Next.js server's IP, not
the visitor's browser, which is relevant for CORS (see Section 10.5).

### 2.3 Rendering modes
Because none of the current pages fetch external data, `next build` renders **all 8
routes as static HTML** (confirmed — see the build output: every route is marked `○
(Static)`). Once pages start calling the FastAPI backend for per-user data (the
dashboard's real stats, the exam timetable, etc.), those specific pages will
automatically switch to being rendered per-request instead — Next.js decides this
automatically based on whether you use dynamic data APIs (like reading cookies) or
`fetch` with caching disabled. You don't need to configure this manually.

### 2.4 Metadata
`export const metadata: Metadata = { title: "..." }` in a `page.tsx` or `layout.tsx`
sets that route's `<title>` tag (and can set `description`, Open Graph tags, etc.). It
composes with the root layout's `template: "%s — Mentora"`, so `/login`'s title
renders as **"Login — Mentora"** automatically.

### 2.5 Why components were extracted
`SiteHeader`, `SiteFooter`, and `AuthShell` exist so the header/nav/footer markup
lives in exactly one place. Before, each of the 8 HTML files had its own copy of the
`<header>` block — changing one nav link meant editing 8 files. Now it's edited once.

---

## 3. Full project structure

```
mentora-nextjs/
├── package.json              # next, react, react-dom + TS types
├── next.config.js
├── tsconfig.json
├── next-env.d.ts
├── .gitignore
├── README.md
│
├── app/
│   ├── layout.tsx             # root layout — fonts, <html>/<body>, default metadata
│   ├── globals.css            # ported from styles.css (design tokens, all components)
│   │
│   ├── page.tsx                # "/"                  — Home (marketing/landing)
│   │
│   ├── login/page.tsx           # "/login"              — Login
│   ├── register/page.tsx        # "/register"           — Create account
│   ├── forgot-password/page.tsx # "/forgot-password"    — Request reset + reset form
│   │
│   ├── exam-prep/
│   │   ├── layout.tsx            # sets <title>Exam Prep</title>
│   │   └── page.tsx               # "/exam-prep"  — CLIENT component (subject chips)
│   │
│   ├── career/
│   │   ├── layout.tsx            # sets <title>Career Counseling</title>
│   │   └── page.tsx               # "/career"     — CV upload, career fit, roadmap
│   │
│   ├── learning-portal/page.tsx # "/learning-portal" — playlists, MCQ list, PDFs
│   └── dashboard/page.tsx       # "/dashboard"        — SIP profile, charts, weak topics
│
├── components/
│   ├── SiteHeader.tsx           # CLIENT — nav + active-link state via usePathname()
│   ├── SiteFooter.tsx           # server — footer links
│   └── AuthShell.tsx            # server — shared 2-column layout for the 3 auth pages
│
└── public/                    # static assets (currently empty — add images/icons here)
```

---

## 4. Page-by-page breakdown

For each page: what it does today, and what it needs from the backend to become real.

### 4.1 `/` — Home (`app/page.tsx`)
Static marketing page: hero, 4 feature cards, "How It Works" steps, an announcements
list, an ad banner placeholder. **No backend calls needed** — this page can stay 100%
static forever, which is good for load speed. The only dynamic-ish piece is the
announcements list, which *could* later be pulled from a `content`/`announcements`
table if you want it editable without a redeploy — not required by the ERD.

### 4.2 `/login` (`app/login/page.tsx`)
Form: `login-form` with `email`, `password`. Also a demo-credentials hint
(`demo@mentora.pk` / `demo2025`) that should be removed before a real deployment.
**Needs:** `POST /auth/login` → returns a session/JWT.

### 4.3 `/register` (`app/register/page.tsx`)
Form: `register-form` with `first_name`, `last_name`, `email`, `password`,
`confirm_password`, `university`, `exam_goal` (a `<select>` of 9 fixed values — see
4.3.1). **Needs:** `POST /auth/register`.

**4.3.1 — `exam_goal` fixed values** (must match whatever the backend/DB expects
exactly, since this is a plain `<select>`, not free text):
```
sindh_university | mehran_university | lums | karachi_university |
iqra_university | tando_jam_university | pms_spsc_ppsc | issb | army_navy_paf
```

### 4.4 `/forgot-password` (`app/forgot-password/page.tsx`)
Two separate forms on one page:
1. `request-reset-form` — just `email`. **Needs:** `POST /auth/forgot-password`.
2. `reset-password-form` — `reset_code` (6-digit, `pattern="[0-9]{6}"`), `new_password`,
   `confirm_password`. **Needs:** `POST /auth/reset-password`.

### 4.5 `/exam-prep` (`app/exam-prep/page.tsx`) — CLIENT component
Currently shows **hardcoded** data: "47 days to exam", a 5-row weekly timetable, a
today's-topic explainer panel, 2 hardcoded YouTube video cards, and one hardcoded MCQ
(`Q1001`, radio buttons `A`–`D`). The subject chip selector (Mathematics/Chemistry/
Urdu) already works client-side but doesn't persist anywhere yet.

**Needs (in priority order):**
- `GET /exam/timetable?user_id=` → replace the 5 hardcoded `<tr>` rows
- `GET /exam/today-topic?user_id=` → replace the AI Topic Explainer panel + video cards
- `GET /exam/mcq?topic=&count=5` → replace the single hardcoded MCQ with real
  FLAN-T5-generated questions (per `M2 – Adaptive Question Generator` in the tech doc)
- `POST /exam/mcq/submit` (on `mcq-form` submit) → body: `{ question_id, answer }`
- `PATCH /exam/subjects` (on chip toggle) or bundle into a "save preferences" call →
  persists `test_date` + `study_mode` + selected subjects to `EXAM_SESSIONS`/user prefs

### 4.6 `/career` (`app/career/page.tsx`)
Currently hardcoded: a CV drop-zone (`cv-upload-form`, file input `cv_file`, accepts
`.pdf,.docx`), an "extracted skills" panel (GPA, chips), a career-goal `<select>`
(`career-goal-form`), a 5-row Career Fit Rankings table, a 6-step roadmap timeline, and
a Mock Interview CTA.

**Needs:**
- `POST /career/cv-upload` (multipart, field `cv_file`) → runs Model 5 (CV/Transcript
  NER Extractor, BERT) → returns extracted `gpa`, `skills[]`, `certifications[]`,
  `projects[]` to populate Step 2's panel (currently hardcoded Python/SQL/etc.)
- `POST /career/goal` (`career-goal-form`, field `career_goal`) → saves to
  `CAREER_RANKINGS`/SIP
- `GET /career/ai-suggest` (button `#ai-suggest-career`) → runs Model 4
  (Sentence-Transformers `all-mpnet-base-v2`) → best-fit suggestion
- `GET /career/rankings?user_id=` → replaces the 5-row table (`match_score`,
  `key_skill_gap` per row, matches `CAREER_RANKINGS` table in the ERD)
- `GET /career/roadmap?user_id=&career_id=` → replaces the 6-step timeline (matches
  `ROADMAPS` table — `month_number`, `milestone_title`, `status`)
- `POST /interview/start` (button `#start-mock-interview`) → creates a row in
  `INTERVIEW_SESSIONS`, then streams RAG Q&A into `INTERVIEW_QA`

### 4.7 `/learning-portal` (`app/learning-portal/page.tsx`)
Currently hardcoded: 4 stat cards, a sidebar (My Playlists/MCQ Assignments/Course
PDFs/Watch History + subject filters), 3 video rows, 2 MCQ assignment cards, 3 PDF
cards.

**Needs:**
- `GET /learning/stats?user_id=` → the 4 stat-strip numbers
- `GET /learning/playlist?user_id=` → the video rows (title, subject, source, duration)
- `GET /learning/assignments?user_id=` → the MCQ assignment cards (status: New /
  Completed, score)
- `GET /learning/materials?user_id=` → the PDF cards (title, size, source)
- Sidebar subject filters (`Mathematics`, `English`, `Physics`, `Chemistry`) are `<a
  href="#">` placeholders — wire these to `?subject=` query params once the above
  endpoints exist

### 4.8 `/dashboard` (`app/dashboard/page.tsx`)
Currently hardcoded: student header ("Raimal Raja", avatar initials "RR"), 4 stat
cards, an SVG score-trend line chart (raw `<polyline>` points, not a charting library),
a 5-day weekly snapshot grid, 4 "Weak Topics" cards with progress bars, 4 notification
items, and a Settings button row (Account / Notification Prefs / Study Preferences /
Change Password / Delete Account — all inert buttons today).

**Needs:**
- `GET /users/me` → name, university, batch, avatar initials, `exam_goal` +
  days-remaining (drives the header)
- `GET /dashboard/stats?user_id=` → the 4 stat cells (exam sessions, avg score,
  videos watched, top career match)
- `GET /dashboard/score-trend?user_id=&sessions=10` → an array of `{session, score}` to
  replace the hardcoded SVG `points=` string (recommend swapping the raw `<svg>` for
  **Recharts**, which is already listed as an approved library in the tech doc, Part 2)
- `GET /dashboard/weekly-snapshot?user_id=` → the 5-day grid (day, topic, status:
  done/today/missed/pending)
- `GET /dashboard/weak-topics?user_id=` → matches `TOPIC_SCORES` table
- `GET /notifications?user_id=&unread=true` → matches `NOTIFICATIONS` table
- `PATCH /users/me` (Edit Profile button)
- `POST /auth/change-password`, `DELETE /users/me` (Settings row — currently inert
  buttons, need `"use client"` + confirm dialogs added when wired up)

---

## 5. Shared components — API reference

### `SiteHeader` (`components/SiteHeader.tsx`)
```tsx
<SiteHeader variant="public" />   // Home — shows Login / Sign Up buttons
<SiteHeader variant="app" />      // every logged-in page — shows "My Profile" / "Settings"
```
Currently `variant` is passed manually per page. Once real auth exists, this should
instead be derived automatically from whether a valid session exists (see Section 9),
so a logged-out user can never see the "app" header even if they type `/dashboard`
directly into the URL bar.

### `SiteFooter` (`components/SiteFooter.tsx`)
No props. Static site links + copyright line.

### `AuthShell` (`components/AuthShell.tsx`)
```tsx
<AuthShell
  sideHeading="MENTORA"
  sideCopy="Welcome back. Continue where you left off..."
  sideChecklist={["Your Student Intelligence Profile (SIP) is synced", "..."]}
>
  {/* the actual form goes here as children */}
</AuthShell>
```
Used by `/login`, `/register`, `/forgot-password` — same 2-column shell (green
sidebar + white form panel), different copy per page.

---

## 6. Styling system

`app/globals.css` is the original `styles.css` verbatim, plus the small `<style>`
block that used to live inline in `dashboard.html` (now under a clearly commented
"Dashboard page extras" section at the bottom of the file). Nothing in the design
system changed:

- Colors: CSS custom properties like `--green-950`, `--green-700`, `--ink-500`,
  `--red-100` etc., defined once in `:root`
- Fonts: `--font-display` (Fraunces, headings) and `--font-body` (Inter, everything
  else) — these two variables now point at `next/font`'s generated variable names
  (`var(--font-fraunces)`, `var(--font-inter)`) instead of the literal string
  `'Fraunces'`, which is the one intentional change to the CSS file
- No CSS framework, no Tailwind, no CSS Modules — plain global class names
  (`.btn-primary`, `.panel`, `.chip`, `.status-pill`, etc.), same as the static HTML

If a new page or component is added later, reuse these existing classes rather than
inventing new ones — they already have all the states (hover, disabled, warn/low
progress-bar variants, etc.) built out.

---

## 7. Interactive elements inventory (buttons/links with no handler yet)

These render correctly but do nothing on click yet — a checklist for wiring up:

| Page | Element | id | Intended action |
|---|---|---|---|
| exam-prep | Regenerate link | — | Re-run timetable generation |
| exam-prep | "Take MCQ Test →" | — | Scroll to / load `mcq-form` |
| exam-prep | "+ Add to Playlist" ×2 | — | `POST /learning/playlist` |
| career | Edit extracted info | — | Make Step 2 panel editable |
| career | AI Suggest Best Career | `ai-suggest-career` | `GET /career/ai-suggest` |
| career | View Roadmap ×5 | — | `GET /career/roadmap?career_id=` |
| career | Start Mock Interview | `start-mock-interview` | `POST /interview/start` |
| learning-portal | Add Video | — | Add-to-playlist modal |
| learning-portal | Watch / Remove ×3 rows | — | Playlist item actions |
| learning-portal | Download ×3 PDFs | — | File download / signed URL |
| dashboard | Edit Profile | — | `PATCH /users/me` |
| dashboard | Settings row ×5 buttons | — | See Section 4.8 |

---

## 8. Complete forms & fields inventory

This is the exact contract the backend's request bodies must accept — every `name`
attribute below is what will show up as a form field / JSON key.

| Form (`id`) | Page | Fields (`name`) | Notes |
|---|---|---|---|
| `login-form` | `/login` | `email`, `password` | |
| `register-form` | `/register` | `first_name`, `last_name`, `email`, `password`, `confirm_password`, `university`, `exam_goal` | `exam_goal` is a fixed 9-value enum — see 4.3.1 |
| `request-reset-form` | `/forgot-password` | `email` | Step 1 of password reset |
| `reset-password-form` | `/forgot-password` | `reset_code` (6-digit), `new_password`, `confirm_password` | Step 2 of password reset |
| `cv-upload-form` | `/career` | `cv_file` (file, `.pdf`/`.docx`, `multipart/form-data`) | Feeds Model 5 (CV NER) |
| `career-goal-form` | `/career` | `career_goal` (enum: `data_scientist`, `ml_engineer`, `business_analyst`, `software_engineer`, `product_manager`) | |
| `mcq-form` | `/exam-prep` | `answer_Q1001` (radio, values `A`–`D`) | Field name is per-question (`answer_<question_id>`) — the real version will need to render N of these dynamically once questions come from the backend |
| — (not a form yet) | `/exam-prep` | `test_date` (date input), `study_mode` (select: `self_study`/`tutor`) | Currently free-floating inputs, not wrapped in a `<form>` — wrap them before wiring up |

All `id`/`name` values were chosen to already match the ERD column-naming convention
from the tech doc (Part 4), so the backend's Pydantic models can mostly mirror these
names directly.

---

## 9. Next.js Middleware — what it is and how it fits here

**Middleware in Next.js (`middleware.ts` at the project root) is code that runs
*before* a request reaches any page** — on Next.js's own server layer (an edge
runtime), not in the browser and not in your FastAPI backend. It's the right place for
one specific job in this project: **route protection.**

### 9.1 What it should do here
Right now, `/dashboard`, `/exam-prep`, `/career`, and `/learning-portal` render for
anyone who visits them, logged in or not, because there's no auth check at all yet.
Once login issues a session (see 9.2), a `middleware.ts` should be added that:

1. Reads the session cookie on every incoming request
2. If the request is for a protected route (`/dashboard`, `/exam-prep`, `/career`,
   `/learning-portal`) and there's no valid session → redirect to `/login`
3. If the request is for `/login` or `/register` and the user *does* have a valid
   session → redirect to `/dashboard` (no reason to show the login form to someone
   already logged in)

Sketch of what that file will look like (not yet added to the project — this is the
plan):

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
  matcher: ["/dashboard/:path*", "/exam-prep/:path*", "/career/:path*", "/learning-portal/:path*", "/login", "/register"],
};
```

### 9.2 What middleware is *not* for
Middleware should only check "does a session cookie exist and look valid" (e.g. a
signed JWT it can verify without a database call, since middleware runs on every
single request and needs to be fast). It should **not**:
- Call the FastAPI backend on every request (too slow, defeats the purpose)
- Do business logic (e.g. "does this student own this exam session") — that
  authorization check belongs in FastAPI, on the specific endpoint, using the DB
- Replace FastAPI's own auth checks — treat middleware as a UX convenience (don't
  show a logged-out user the dashboard shell) and FastAPI's JWT validation as the real
  security boundary, since middleware can theoretically be bypassed by calling the API
  directly

---

## 10. How the backend (FastAPI) fits in

### 10.1 Two servers, one browser
Per the tech doc (Part 3, Part 11), this is a **two-server local setup**:
- Next.js frontend → `http://localhost:3000`
- FastAPI backend → `http://localhost:8000`

The browser talks to both. Next.js does **not** proxy API calls to FastAPI by default
— the frontend's `fetch()` calls will point directly at
`process.env.NEXT_PUBLIC_API_URL` (i.e. `http://localhost:8000/...`).

### 10.2 Where API calls should live in the frontend code
None exist yet. The recommended pattern for this codebase:

```
lib/
  api.ts          # one shared function: apiFetch(path, options) that prefixes
                    # NEXT_PUBLIC_API_URL, attaches the auth token, and throws
                    # a typed error on non-2xx responses
  types.ts         # TypeScript types mirroring the FastAPI Pydantic response models
                    # (e.g. `CareerRanking`, `ExamSession`, `WeakTopic`)
```

Then each page that needs data becomes either:
- An `async` Server Component that calls `apiFetch` directly and passes the result as
  props to a small Client Component for any interactive bits, **or**
- A full Client Component using `useEffect`/`useState` (or a data-fetching library
  like SWR/TanStack Query, not yet installed) if the data needs to refresh without a
  full page reload (e.g. after submitting the MCQ form)

### 10.3 Auth flow (JWT), end to end
This matches `User.login()` / `register()` in the tech doc's Class Diagram (Part 5):

1. User submits `login-form` → frontend `POST`s to
   `${NEXT_PUBLIC_API_URL}/auth/login` with `{ email, password }`
2. FastAPI verifies against `USERS.password_hash` (bcrypt, per `passlib[bcrypt]` in
   `requirements.txt`), returns a signed JWT (`python-jose`, per Part 9)
3. Frontend stores that JWT in an **httpOnly cookie** (set by a Next.js Route Handler
   that receives FastAPI's response and sets the cookie — don't store JWTs in
   `localStorage`, which is readable by any injected script)
4. `middleware.ts` (Section 9) reads that cookie to gate protected routes
5. Every subsequent `apiFetch` call to FastAPI attaches the JWT (as an `Authorization:
   Bearer <token>` header) so FastAPI can identify `current_user` on protected
   endpoints (`get_profile()`, `submit_answer()`, etc. in the Class Diagram)

### 10.4 Suggested endpoint list (consolidated from Section 4)
Grouped by the ERD tables they touch:

```
POST   /auth/register
POST   /auth/login
POST   /auth/forgot-password
POST   /auth/reset-password
GET    /users/me
PATCH  /users/me
DELETE /users/me

GET    /exam/timetable
GET    /exam/today-topic
GET    /exam/mcq
POST   /exam/mcq/submit
PATCH  /exam/subjects

POST   /career/cv-upload
POST   /career/goal
GET    /career/ai-suggest
GET    /career/rankings
GET    /career/roadmap
POST   /interview/start
POST   /interview/answer      (submits one Q&A turn during mock interview)

GET    /learning/stats
GET    /learning/playlist
GET    /learning/assignments
GET    /learning/materials

GET    /dashboard/stats
GET    /dashboard/score-trend
GET    /dashboard/weekly-snapshot
GET    /dashboard/weak-topics
GET    /notifications
```

This maps directly onto `ExamEngine`, `CareerEngine`, and the `ModelManager` classes
already designed in the tech doc's Class Diagram — the frontend doesn't need any
endpoints that aren't already implied by that diagram plus the ERD.

### 10.5 CORS
FastAPI needs to explicitly allow the Next.js origin, or every `fetch()` from the
browser will fail silently with a CORS error in the console (a common first bug when
connecting these two):

```python
# backend/main.py
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,   # required if using cookies for the JWT
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### 10.6 Environment variables (frontend side)
```
# .env.local (create this — it's gitignored)
NEXT_PUBLIC_API_URL=http://localhost:8000
```
Any env var prefixed `NEXT_PUBLIC_` is inlined into the browser bundle at build time —
never put secrets (API keys, DB passwords) in a `NEXT_PUBLIC_` variable. Since all
secrets (DB URL, JWT secret, Ollama host) belong to the FastAPI backend, not the
frontend, this is the only env var the frontend needs.

---

## 11. Running the full stack locally

```bash
# Terminal 1 — backend (see the MENTORA tech doc, Part 11, Steps 11–16)
cd mentora/backend
venv\Scripts\activate
uvicorn main:app --port 8000

# Terminal 2 — frontend
cd mentora-nextjs
npm install
npm run dev          # http://localhost:3000, hot reload
# or for the low-RAM daily-use mode:
npm run build && npm run start
```

---

## 12. Build-out checklist (suggested order)

1. `POST /auth/register`, `POST /auth/login` + `lib/api.ts` + cookie handling
2. `middleware.ts` (Section 9) — gate the 4 protected routes
3. `GET /users/me` → make `SiteHeader` and the dashboard header dynamic
4. Dashboard read-only endpoints (stats, score-trend, weak-topics, notifications) —
   highest visual payoff, all `GET`, no forms to wire
5. Exam-prep timetable + MCQ (`GET` + one `POST`)
6. Career CV upload + rankings + roadmap
7. Learning portal (mostly `GET`, lowest complexity)
8. Mock interview (RAG streaming — the most involved piece, do last)
9. Settings actions (change password, delete account) on the dashboard

---

*This document reflects the frontend as of the Next.js conversion delivered in this
conversation. If pages are added, renamed, or restructured later, update Sections 3,
4, and 8 to match — those three sections are what the backend spec is derived from.*
