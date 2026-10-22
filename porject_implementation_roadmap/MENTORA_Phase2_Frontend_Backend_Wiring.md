# MENTORA — Phase 2 Deep-Dive: Wire the Existing Frontend to the Backend

*Prerequisite: Phase 1 is done — `backend/` exists, `/auth/register` and `/auth/login`
work against a real database and return a real JWT. This phase does not require any
ML model to exist. Every endpoint returns realistic, hardcoded mock JSON. The entire
point of this phase is to prove the wiring is correct — every click reaches the
backend and updates the screen — before a single model is trained.*

---

## 0. Goal, scope, and the rule that makes this phase fast

**Goal:** by the end of this phase, every button, form, and page in the Next.js
frontend makes a real network request to FastAPI and renders a real (if fake) JSON
response. Nothing is hardcoded in JSX anymore — it all comes from `fetch()`.

**Scope boundary — read this twice:** Phase 2 does **not** touch `ml_core/`,
`rag_pipelines/`, or anything Colab-related. If you catch yourself thinking "let me
just quickly wire up the real Gap Classifier while I'm in here," stop — that's Phase
6. Mixing them means you can't tell whether a bug is in your wiring or in a
half-trained model. Keep mock data laughably fake on purpose (e.g. a career match
score of exactly `73%` everywhere) so you can never confuse mock output for real
output later.

**Estimated time:** 3–6 days for one person, mostly mechanical, page by page.

**Milestone that closes this phase:** open DevTools → Network tab, click every
interactive element on all 8 pages once, in order. Every click should produce exactly
one (or a small, deliberate number of) request(s) to `localhost:8000`, a `200`
response, and a visible UI change. Zero console errors.

---

## 1. Step 1 — Stand up the mock backend first (do this before touching the frontend)

You need something for the frontend to talk to before you can wire anything. Build
this in one sitting.

### 1.1 Mock router pattern
Every router file returns static Python dicts/lists shaped exactly like the real
future response — same field names as the ERD, same nesting, same types (numbers as
numbers, not strings). This matters: if you're sloppy here, the frontend code you
write against sloppy mock data will break the day real data arrives.

```python
# backend/routers/dashboard.py
from fastapi import APIRouter, Depends
from services.auth_service import get_current_user

router = APIRouter(prefix="/dashboard", tags=["dashboard"])

@router.get("/stats")
def get_stats(user_id: int = Depends(get_current_user)):
    # MOCK — replace in Phase 6 with a real query against EXAM_SESSIONS/TOPIC_SCORES
    return {
        "exam_sessions": 24,
        "avg_score": 72,
        "avg_score_delta": 5,
        "videos_watched": 38,
        "videos_watched_this_week": 14,
        "top_career_match": {"title": "Data Scientist", "fit_percent": 88},
    }

@router.get("/score-trend")
def get_score_trend(user_id: int = Depends(get_current_user), sessions: int = 10):
    # MOCK — matches the 10 points currently hardcoded in the dashboard SVG
    return {
        "points": [
            {"session": f"S{i+1}", "score": v}
            for i, v in enumerate([55, 60, 68, 65, 75, 72, 82, 80, 90, 88])
        ]
    }

@router.get("/weekly-snapshot")
def get_weekly_snapshot(user_id: int = Depends(get_current_user)):
    return {
        "days": [
            {"day": "Mon", "topic": "Algebra", "status": "done"},
            {"day": "Tue", "topic": "English", "status": "done"},
            {"day": "Wed", "topic": "Grammar", "status": "today"},
            {"day": "Thu", "topic": "Geometry", "status": "missed"},
            {"day": "Fri", "topic": "Revision", "status": "pending"},
        ]
    }

@router.get("/weak-topics")
def get_weak_topics(user_id: int = Depends(get_current_user)):
    return {
        "topics": [
            {"subject": "Algebra", "topic": "Quadratic Equations", "score": 38, "note": "Needs urgent review"},
            {"subject": "Grammar", "topic": "Tenses", "score": 44, "note": None},
            {"subject": "Geometry", "topic": "Circles", "score": 60, "note": "Medium"},
            {"subject": "English", "topic": "Reading Comprehension", "score": 62, "note": "Improving"},
        ]
    }
```

Repeat this pattern for `routers/auth.py` (already real from Phase 1),
`routers/exam.py`, `routers/career.py`, `routers/learning.py`,
`routers/notifications.py`, `routers/interview.py`. Use the full endpoint list from
§10 of the master plan — every row in that table becomes one function like the ones
above.

### 1.2 Register every mock router in `main.py`
```python
# backend/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import auth, dashboard, exam, career, learning, interview, users

app = FastAPI(title="Mentora API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(dashboard.router)
app.include_router(exam.router)
app.include_router(career.router)
app.include_router(learning.router)
app.include_router(interview.router)
app.include_router(users.router)
```

### 1.3 Verify the mock backend alone, before involving Next.js at all
```bash
uvicorn main:app --reload --port 8000
```
Open `http://localhost:8000/docs` (FastAPI's auto Swagger UI). Manually execute every
`GET` endpoint from the "Authorize" button downward — for the protected ones, first
call `/auth/login` with a real test user to get a token, click "Authorize", paste it
in as `Bearer <token>`, then try `/dashboard/stats` etc. **Confirm every single
endpoint returns 200 with the shape you expect before writing any frontend code.**
This is the cheapest point in the whole project to catch a typo in a field name.

---

## 2. Step 2 — Frontend auth plumbing (build once, every page depends on it)

### 2.1 `lib/types.ts` — mirror the mock responses exactly
```ts
// frontend/lib/types.ts
export interface DashboardStats {
  exam_sessions: number;
  avg_score: number;
  avg_score_delta: number;
  videos_watched: number;
  videos_watched_this_week: number;
  top_career_match: { title: string; fit_percent: number };
}

export interface ScoreTrendPoint {
  session: string;
  score: number;
}

export interface WeeklySnapshotDay {
  day: string;
  topic: string;
  status: "done" | "today" | "missed" | "pending";
}

export interface WeakTopic {
  subject: string;
  topic: string;
  score: number;
  note: string | null;
}

// ...continue this file with one interface per endpoint response as you build
// each router in Step 1. Keep this file the single source of truth for shapes —
// if a field's type changes in the backend, this is the one place to fix it,
// and TypeScript will then show you every frontend spot that breaks.
```

### 2.2 `lib/api.ts` — the one function everything else calls
```ts
// frontend/lib/api.ts
const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

/**
 * For calls made FROM THE BROWSER (Client Components) — routes through Next.js's
 * own /api/* Route Handlers, never straight to FastAPI, so the JWT cookie never
 * has to be readable by client-side JS. See §2.3–2.4.
 */
export async function apiFetch<T>(
  path: string,
  options: RequestInit = {}
): Promise<T> {
  const res = await fetch(`/api/proxy${path}`, {
    ...options,
    headers: { "Content-Type": "application/json", ...options.headers },
    credentials: "include",
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new ApiError(res.status, body.detail ?? "Request failed");
  }
  return res.json();
}

/**
 * For calls made FROM A SERVER COMPONENT (async function Page()) — talks to
 * FastAPI directly, forwarding the session cookie read from next/headers.
 * See §2.5 for the usage pattern.
 */
export async function apiFetchServer<T>(
  path: string,
  token: string | undefined,
  options: RequestInit = {}
): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
      ...options.headers,
    },
    cache: "no-store", // this is per-user data — never let Next.js cache it
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new ApiError(res.status, body.detail ?? "Request failed");
  }
  return res.json();
}
```

### 2.3 The proxy Route Handler — why it exists
Client Components (`"use client"`) run in the browser, where an httpOnly cookie is
*invisible* — that's the entire point of httpOnly. So a Client Component can't attach
`Authorization: Bearer <token>` itself. The fix: a Next.js Route Handler that runs on
the server, *can* read the httpOnly cookie, and re-issues the request to FastAPI with
the header attached. The browser only ever talks to `same-origin` `/api/proxy/*`.

```ts
// frontend/app/api/proxy/[...path]/route.ts
import { NextRequest, NextResponse } from "next/server";
import { cookies } from "next/headers";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
const COOKIE_NAME = "mentora_session";

async function forward(req: NextRequest, path: string[]) {
  const token = cookies().get(COOKIE_NAME)?.value;
  const url = `${API_URL}/${path.join("/")}${req.nextUrl.search}`;

  const init: RequestInit = {
    method: req.method,
    headers: {
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
      ...(req.headers.get("content-type")?.includes("application/json")
        ? { "Content-Type": "application/json" }
        : {}),
    },
  };

  if (req.method !== "GET" && req.method !== "HEAD") {
    const contentType = req.headers.get("content-type") ?? "";
    init.body = contentType.includes("multipart/form-data")
      ? await req.formData()
      : await req.text();
  }

  const res = await fetch(url, init);
  const data = await res.text();
  return new NextResponse(data, {
    status: res.status,
    headers: { "Content-Type": res.headers.get("content-type") ?? "application/json" },
  });
}

export async function GET(req: NextRequest, { params }: { params: { path: string[] } }) {
  return forward(req, params.path);
}
export async function POST(req: NextRequest, { params }: { params: { path: string[] } }) {
  return forward(req, params.path);
}
export async function PATCH(req: NextRequest, { params }: { params: { path: string[] } }) {
  return forward(req, params.path);
}
export async function DELETE(req: NextRequest, { params }: { params: { path: string[] } }) {
  return forward(req, params.path);
}
```
This one file handles every endpoint in §10 of the master plan — you never need to
write a new Route Handler per feature, only per *auth event* (login/logout, next).

### 2.4 Login/logout Route Handlers — the only place the raw JWT is visible
```ts
// frontend/app/api/auth/login/route.ts
import { NextRequest, NextResponse } from "next/server";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
const COOKIE_NAME = "mentora_session";

export async function POST(req: NextRequest) {
  const body = await req.json();
  const backendRes = await fetch(`${API_URL}/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  if (!backendRes.ok) {
    const err = await backendRes.json().catch(() => ({ detail: "Login failed" }));
    return NextResponse.json({ detail: err.detail }, { status: backendRes.status });
  }

  const { access_token } = await backendRes.json();
  const res = NextResponse.json({ success: true });
  res.cookies.set(COOKIE_NAME, access_token, {
    httpOnly: true,
    sameSite: "lax",
    secure: process.env.NODE_ENV === "production",
    path: "/",
    maxAge: 60 * 60 * 24 * 7, // 7 days — match your backend JWT expiry
  });
  return res;
}
```
```ts
// frontend/app/api/auth/logout/route.ts
import { NextRequest, NextResponse } from "next/server";

export async function POST(_req: NextRequest) {
  const res = NextResponse.json({ success: true });
  res.cookies.delete("mentora_session");
  return res;
}
```
`app/api/auth/register/route.ts` is the same shape as `login/route.ts`, pointed at
`/auth/register` — register on this platform also logs the student in immediately.

### 2.5 `middleware.ts` — drop in exactly as specified in the master plan
```ts
// frontend/middleware.ts
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

**Checkpoint before moving on:** manually test this plumbing with `curl` or the
browser before wiring any page:
1. `curl -X POST http://localhost:3000/api/auth/login -H "Content-Type: application/json" -d '{"email":"demo@mentora.pk","password":"demo2025"}' -i` — response headers should include a `Set-Cookie: mentora_session=...` line.
2. Visit `http://localhost:3000/dashboard` in an incognito window with no cookie —
   should redirect to `/login`.
3. Log in through the real form, then visit `/dashboard` — should render (even with
   nothing wired up yet, since it's a Server Component that hasn't been touched).

---

## 3. Step 3 — Wire the three auth pages

### 3.1 `/login` — convert to a Client Component with real submit handling
The current `app/login/page.tsx` is a Server Component with a plain `<form>`. It
needs `"use client"`, controlled-or-uncontrolled form handling, a submit handler, a
loading state, and an error state.

```tsx
// frontend/app/login/page.tsx  (excerpt — only the parts that change)
"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
// ...AuthShell, Link imports unchanged

export default function LoginPage() {
  const router = useRouter();
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function handleSubmit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    setError(null);
    setLoading(true);
    const formData = new FormData(e.currentTarget);
    const payload = {
      email: formData.get("email"),
      password: formData.get("password"),
    };
    try {
      const res = await fetch("/api/auth/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) {
        const body = await res.json();
        throw new Error(body.detail ?? "Login failed");
      }
      router.push("/dashboard");
      router.refresh(); // forces middleware/server components to re-read the new cookie
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong");
    } finally {
      setLoading(false);
    }
  }

  return (
    <AuthShell /* ...unchanged props... */>
      <h2>Login to Mentora</h2>
      <p className="subhead">Enter your registered email and password</p>

      {error && <div className="form-error">{error}</div>}

      <form id="login-form" name="login-form" onSubmit={handleSubmit}>
        {/* fields unchanged — email, password */}
        <button type="submit" className="btn btn-primary btn-block" disabled={loading}>
          {loading ? "Logging in…" : "Login →"}
        </button>
      </form>
      {/* ...rest unchanged... */}
    </AuthShell>
  );
}
```
Note: `.form-error` isn't in `globals.css` yet — add a small rule (`color:
var(--red-600); background: var(--red-100); padding: 10px 14px; border-radius: 8px;
margin-bottom: 16px; font-size: 0.88rem;`) rather than inventing a new pattern.

**Remove the hardcoded demo-credentials `<div className="demo-note">` block** once
real auth works — it was a placeholder for the static-mockup stage.

### 3.2 `/register` — same pattern, one extra step
Identical shape to 3.1, but the payload includes all 7 register fields, and on
success it should behave like a login (register + auto-login, matching Route Handler
from §2.4) then redirect to `/dashboard`.

### 3.3 `/forgot-password` — two independent forms, two independent handlers
Each of the two `<form>` elements on this page gets its own `handleSubmit`, since
they're separate steps a user might complete minutes apart (after checking email).
Add a small `successMessage` state for form 1 ("Reset code sent to your email") so
the user isn't left wondering if the click worked, since this page doesn't redirect
anywhere on step 1.

---

## 4. Step 4 — Wire `/exam-prep`

This page is already a Client Component (for the subject chips), so no `"use
client"` needs adding — but it needs real data instead of hardcoded JSX, via
`useEffect` + `useState`, since Client Components can't be `async function`.

```tsx
// frontend/app/exam-prep/page.tsx (excerpt — new data-fetching logic)
"use client";

import { useEffect, useState } from "react";
import { apiFetch } from "@/lib/api";

interface TimetableRow {
  day: string;
  topic: string;
  subject: string;
  duration: string;
  status: "pending" | "done" | "today" | "missed";
}

export default function ExamPrepPage() {
  const [selectedSubjects, setSelectedSubjects] = useState<Subject[]>(["mathematics"]);
  const [timetable, setTimetable] = useState<TimetableRow[] | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    apiFetch<{ rows: TimetableRow[] }>("/exam/timetable")
      .then((data) => setTimetable(data.rows))
      .catch(() => setTimetable([])) // fail soft — show an empty-state, not a crash
      .finally(() => setLoading(false));
  }, []);

  async function toggleSubject(id: Subject) {
    const next = selectedSubjects.includes(id)
      ? selectedSubjects.filter((s) => s !== id)
      : [...selectedSubjects, id];
    setSelectedSubjects(next);
    await apiFetch("/exam/subjects", {
      method: "PATCH",
      body: JSON.stringify({ subjects: next }),
    });
  }

  // ...render: replace the 5 hardcoded <tr> rows with
  // {loading ? <SkeletonRows /> : timetable!.map((row) => <tr key={row.day}>...)}
}
```

Apply the same `useEffect` pattern for the "Today's Topic" panel
(`/exam/today-topic`) and wrap the `mcq-form`'s submit in a handler that posts to
`/exam/mcq/submit` and then re-fetches `/exam/mcq` for the next question (mock
backend can just cycle through 2–3 canned questions for now).

**Add a loading skeleton, not a blank screen.** Since every page now waits on a
network round-trip, a brief "blank flash" before data arrives is normal — a simple
pulsing-gray placeholder using the existing `.panel`/`.chip` classes prevents it from
looking broken. Don't over-engineer this in Phase 2; a plain `{loading && <p
className="small-muted">Loading…</p>}` per section is enough for now.

---

## 5. Step 5 — Wire `/career`

This page currently has **no** `"use client"` (Section 4.6 of the frontend doc noted
it as static). It needs one now, since it has a file upload, a `<select>` `onChange`,
and two buttons that must call the backend.

Key pieces to wire, each independently:

1. **CV upload** (`cv-upload-form`) — use `FormData`, not JSON, since it's
   `multipart/form-data`:
   ```tsx
   async function handleCvUpload(e: React.FormEvent<HTMLFormElement>) {
     e.preventDefault();
     const formData = new FormData(e.currentTarget); // picks up cv_file automatically
     setUploading(true);
     try {
       const result = await apiFetch<ExtractedCvData>("/career/cv-upload", {
         method: "POST",
         body: formData,
         headers: {}, // IMPORTANT: don't set Content-Type — the browser sets the
                        // multipart boundary itself; setting it manually breaks the upload
       });
       setExtracted(result); // drives Step 2's panel (GPA, skill chips, etc.)
     } finally {
       setUploading(false);
     }
   }
   ```
   Note `apiFetch` in §2.2 sets a default `Content-Type: application/json` header —
   for this one call, override it back to nothing so `fetch` can set the multipart
   boundary itself. Small enough exception that it's worth calling out here rather
   than silently getting a broken upload.

2. **Career goal select** — `onChange` posts immediately (no separate submit button
   needed, matches how a `<select>` naturally behaves):
   ```tsx
   async function handleGoalChange(e: React.ChangeEvent<HTMLSelectElement>) {
     await apiFetch("/career/goal", {
       method: "POST",
       body: JSON.stringify({ career_goal: e.target.value }),
     });
     await refreshRankings(); // re-fetch /career/rankings so the table updates
   }
   ```

3. **AI Suggest Best Career** button (`#ai-suggest-career`) — `onClick` → `GET
   /career/ai-suggest` → update the `<select>`'s value + re-fetch rankings.

4. **View Roadmap** buttons (5 of them, one per table row) — `onClick` passes that
   row's `career_id`, fetches `/career/roadmap?career_id=`, and either navigates to
   an anchor or opens the existing 6-step `.timeline` block populated with real data
   (currently hardcoded to "Data Scientist" only).

5. **Start Mock Interview** (`#start-mock-interview`) — `POST /interview/start` →
   backend mock returns a fixed `interview_id` and a canned first question → render it
   in place of the current static panel copy, with a text input + send button
   appearing for the follow-up (`POST /interview/{id}/ask`, still mocked, can just
   echo back "That's a great point — tell me more" for now).

---

## 6. Step 6 — Wire `/learning-portal` and `/dashboard`

Both are currently Server Components with zero interactivity beyond navigation
links, and both only need **read** data (no forms) — the cleanest possible case.
**Prefer fetching this data on the server, not the client**, since there's no reason
to ship a loading spinner for data that's available before the page even renders:

```tsx
// frontend/app/dashboard/page.tsx (excerpt)
import { cookies } from "next/headers";
import { apiFetchServer } from "@/lib/api";
import type { DashboardStats, WeakTopic /* ... */ } from "@/lib/types";

export default async function DashboardPage() {
  const token = cookies().get("mentora_session")?.value;

  // Fire all independent requests in parallel — don't await them one at a time
  const [stats, trend, snapshot, weakTopics, notifications] = await Promise.all([
    apiFetchServer<DashboardStats>("/dashboard/stats", token),
    apiFetchServer<{ points: ScoreTrendPoint[] }>("/dashboard/score-trend", token),
    apiFetchServer<{ days: WeeklySnapshotDay[] }>("/dashboard/weekly-snapshot", token),
    apiFetchServer<{ topics: WeakTopic[] }>("/dashboard/weak-topics", token),
    apiFetchServer<{ items: Notification[] }>("/notifications", token),
  ]);

  return (
    <>
      <SiteHeader variant="app" />
      {/* ...replace every hardcoded number/row below with the fetched data... */}
    </>
  );
}
```
This keeps `dashboard/page.tsx` a Server Component — no `"use client"` needed just to
read data. The **Edit Profile** and **Settings row** buttons, however, do need
interactivity (they mutate data), so extract them into a small
`components/DashboardSettings.tsx` **Client** Component, imported and rendered inside
the still-server `DashboardPage`. This "server page, client island for the
interactive bits" split is the idiomatic Next.js pattern — don't convert the whole
page to `"use client"` just because one button on it needs a click handler.

Do the identical `Promise.all` + Server Component pattern for `/learning-portal`
(`/learning/playlist`, `/learning/assignments`, `/learning/materials`, all three in
parallel).

---

## 7. Step 7 — Error and empty states (small, but don't skip)

For every fetch added above, decide and implement, consistently:
- **Loading:** a lightweight inline message or skeleton using existing `.small-muted`
  styling — not a full-page spinner that hides the layout.
- **Error:** don't let an uncaught rejection blank the page. Server Components:
  wrap risky sections in a `try/catch` and render a `.panel` with "Couldn't load this
  right now" rather than letting Next.js's error boundary take over the whole route
  for one failed fetch. Client Components: the `error` state pattern from §3.1.
- **Empty:** mock data will always be non-empty right now, but write the empty-state
  branch anyway (`topics.length === 0 ? <EmptyState /> : ...`) — real data *will* be
  empty for a brand-new user (0 exam sessions, no CV uploaded yet), and this is much
  cheaper to handle now than to retrofit in Phase 6 under time pressure.

---

## 8. Step 8 — The full click-through test (this closes the phase)

Do this in one sitting, DevTools Network tab open, "Preserve log" checked:

1. **Logged out**, visit `/dashboard` directly by URL → redirected to `/login`. ✅ /
   ❌
2. **Register** a new test account → redirected to `/dashboard`, cookie present
   (Application tab → Cookies → `mentora_session`, `HttpOnly` column checked). ✅ / ❌
3. **Log out** (need a logout button somewhere — add one to `SiteHeader`'s "app"
   variant if it isn't there yet, calling `POST /api/auth/logout` then
   `router.push("/")`). ✅ / ❌
4. **Log in** with the same account → `/dashboard`. ✅ / ❌
5. On `/dashboard`: stats, chart, weekly snapshot, weak topics, notifications all
   show mock data, not the old hardcoded JSX (compare against git diff — if a number
   matches the *original* static value exactly, double check it's actually coming
   from the fetch and you didn't just leave the JSX in place). ✅ / ❌
6. Click every Settings button → each fires a request (even if the backend mock just
   returns `{"success": true}` for now). ✅ / ❌
7. On `/exam-prep`: timetable loads from the network; toggle a subject chip → a
   `PATCH` fires; submit the MCQ form → a `POST` fires and (per your mock) the next
   question loads. ✅ / ❌
8. On `/career`: upload any small PDF → `POST /career/cv-upload` fires, Step 2 panel
   updates from the mock response; change the goal `<select>` → `POST /career/goal`
   fires; click AI Suggest → `GET /career/ai-suggest` fires; click each "View
   Roadmap" → `GET /career/roadmap` fires with a different `career_id` each time;
   Start Mock Interview → `POST /interview/start` fires and a question appears. ✅ /
   ❌
9. On `/learning-portal`: page loads with data already present (Server Component —
   should see the fetches in the *server* terminal log, not the browser Network tab,
   since these run before the HTML is sent). ✅ / ❌
10. **Forgot password**: request a reset code (mock can just always "succeed" and
    log the fake code to the backend terminal), then submit the reset form with that
    code. ✅ / ❌

**If every row above is ✅, Phase 2 is done.** Commit, tag it (`git tag
phase-2-complete`), and move to Phase 3 (dataset work — can run fully in parallel
with Phase 4's Colab training) or Phase 4 directly if datasets are already staged.

---

## 9. Common pitfalls (things that will actually happen)

| Symptom | Cause | Fix |
|---|---|---|
| Browser console: `CORS policy: No 'Access-Control-Allow-Origin'` | Forgot `allow_credentials=True`, or `allow_origins` doesn't exactly match `http://localhost:3000` (protocol + port matter) | Fix `CORSMiddleware` config in `main.py`, restart Uvicorn |
| Cookie never appears in Application tab | `secure: true` set while running on plain `http://localhost` | `secure: process.env.NODE_ENV === "production"` as shown in §2.4 — never hardcode `true` locally |
| `/dashboard` Server Component fetch always 401s | Forgot `cache: "no-store"` — Next.js cached the *first* (logged-out or stale) response and never re-fetches | Add `cache: "no-store"` to every `apiFetchServer` call touching per-user data |
| File upload backend receives an empty body | `apiFetch`'s default `Content-Type: application/json` header overrode the multipart boundary | Explicitly pass `headers: {}` for the one `FormData` call, per §5.1 |
| Middleware redirect loop between `/login` and `/dashboard` | Cookie is being set with the wrong `path` (e.g. `path: "/login"` instead of `path: "/"`) | `path: "/"` always, as shown in §2.4 |
| "Hydration failed" error in console on `/exam-prep` | Reading `Math.random()`, `Date.now()`, or `window` directly in JSX render (works differently server vs. client) | Move any non-deterministic value into `useEffect`, set via `useState` after mount |
| A page still shows the *exact* old static numbers after "wiring it up" | The `fetch` call was added but its result was never used to replace the JSX — classic copy-paste leftover | Grep the page for the literal old numbers (e.g. `"88%"`) — if found outside mock backend files, it's still hardcoded |

---

## 10. What Phase 2 deliberately leaves for later

- No real ML anywhere — every "AI" response is a fixed string/number (Phase 6–7).
- No production security hardening (rate limiting, refresh tokens, password
  complexity rules) — fine for local FYP demo use, revisit if this ever goes
  beyond a demo (not in scope of the current phase list).
- No automated tests — acceptable for an FYP timeline, but if time allows, this is
  the ideal phase to add a handful of Playwright/Cypress smoke tests for the §8
  checklist, since the app is functionally complete (with mocks) and won't change
  shape again — only the data source underneath changes in later phases.
