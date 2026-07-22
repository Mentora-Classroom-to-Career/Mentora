# Mentora — Next.js Frontend

This is the Next.js 14 (App Router, TypeScript) conversion of the original static
`index.html` / `login.html` / `register.html` / `forgot-password.html` / `exam-prep.html` /
`career.html` / `learning-portal.html` / `dashboard.html` pages, matching the architecture
in `MENTORA — Updated Technical Documentation (v2.0)`, Part 3 (Frontend Layer: Next.js 14,
production build served locally on `http://localhost:3000`, talking to FastAPI on
`http://localhost:8000`).

## What changed vs. the static HTML

- **Routing.** Each `.html` file is now a route under `app/`:
  | Old file | New route |
  |---|---|
  | `index.html` | `/` (`app/page.tsx`) |
  | `login.html` | `/login` |
  | `register.html` | `/register` |
  | `forgot-password.html` | `/forgot-password` |
  | `exam-prep.html` | `/exam-prep` |
  | `career.html` | `/career` |
  | `learning-portal.html` | `/learning-portal` |
  | `dashboard.html` | `/dashboard` |
- **Shared layout.** The repeated header, nav, and footer markup was extracted into
  `components/SiteHeader.tsx`, `components/SiteFooter.tsx`, and `components/AuthShell.tsx`
  (used by the three auth pages). `app/layout.tsx` wraps every page.
- **Styling.** `styles.css` is now `app/globals.css`, imported once in the root layout.
  Fonts (Fraunces, Inter) are loaded with `next/font/google` instead of a `<link>` tag —
  this self-hosts the fonts and avoids an extra network request per page load.
- **Forms.** All `id`/`name`/validation attributes from the ERD-aligned audit were kept
  as-is, so the FastAPI endpoints (`/auth/register`, `/auth/login`, etc. — see Part 11,
  Step 16 of the tech doc) can be wired up by pointing each `<form>`'s submit handler at
  `NEXT_PUBLIC_API_URL`.
- **Small interactive bits.** The subject-selector chips on `/exam-prep` are now a client
  component (`"use client"`) using `useState` so the `aria-pressed`/`selected` state
  actually toggles, since that can't happen in a static HTML file without JS.

## What's intentionally *not* done yet

Forms currently just render — none of them call the backend. That's the next step once
the FastAPI routes exist. A natural place to add that: a small `lib/api.ts` with a
`fetch(`${process.env.NEXT_PUBLIC_API_URL}/...`)` helper, called from `onSubmit` handlers
in each form (which will need `"use client"` added to the pages that don't already have it).

## Running it

```bash
npm install
npm run dev      # http://localhost:3000, hot reload while building
# or, for the resource-optimized local demo setup described in the tech doc:
npm run build && npm run start
```

Create a `.env.local` with:

```
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## Folder structure

```
app/
  layout.tsx          # root layout, fonts, metadata
  globals.css          # ported from styles.css (+ the dashboard page's inline <style>)
  page.tsx              # Home
  login/page.tsx
  register/page.tsx
  forgot-password/page.tsx
  exam-prep/
    layout.tsx           # page <title> (page.tsx is a client component)
    page.tsx
  career/
    layout.tsx
    page.tsx
  learning-portal/page.tsx
  dashboard/page.tsx
components/
  SiteHeader.tsx        # nav + active-link state, driven by usePathname()
  SiteFooter.tsx
  AuthShell.tsx          # shared two-column layout for the 3 auth pages
```
