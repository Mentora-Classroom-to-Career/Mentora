// Shared fetch helpers.
//
// apiFetch()       — for Client Components. Calls Next.js's OWN
//                     /api/* route handlers, never FastAPI directly from
//                     the browser, so the JWT stays server-side only
//                     (see MENTORA_Phase1_Frontend_Documentation.md §10.3).
// apiFetchServer()  — for Server Components. Calls FastAPI directly,
//                     server-to-server, forwarding the JWT read from the
//                     httpOnly cookie.

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

/**
 * Client Component fetch — takes a FastAPI-style path (e.g. "/exam/mcq")
 * and routes it through this app's own /api/proxy/* Route Handler, which
 * attaches the JWT server-side. Never calls FastAPI directly from the
 * browser.
 */
export async function apiFetch<T>(path: string, options: RequestInit = {}): Promise<T> {
  const isFormData = options.body instanceof FormData;
  const proxiedPath = `/api/proxy${path}`;
  const res = await fetch(proxiedPath, {
    ...options,
    headers: isFormData
      ? options.headers
      : { "Content-Type": "application/json", ...options.headers },
  });

  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail ?? detail;
    } catch {
      /* response wasn't JSON — keep statusText */
    }
    throw new ApiError(res.status, detail);
  }

  if (res.status === 204) return undefined as T;
  return res.json();
}

/** Server Component fetch — hits FastAPI directly with a forwarded token. */
export async function apiFetchServer<T>(
  path: string,
  token: string | undefined,
  options: RequestInit = {}
): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, {
    ...options,
    cache: "no-store", // never cache per-user data — see Phase 2 §9 pitfalls
    headers: {
      "Content-Type": "application/json",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
      ...options.headers,
    },
  });

  if (!res.ok) {
    throw new ApiError(res.status, res.statusText);
  }
  return res.json();
}

/** Direct FastAPI call used only by Next.js's own API routes (server-side). */
export async function fastApiFetch<T>(path: string, options: RequestInit = {}): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, {
    ...options,
    headers: { "Content-Type": "application/json", ...options.headers },
  });
  const body = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new ApiError(res.status, body.detail ?? res.statusText);
  }
  return body;
}

/**
 * For the dedicated /api/auth/* routes only (login, register, logout,
 * forgot-password, reset-password) — these are NOT proxied through
 * /api/proxy since they need to set/clear the cookie themselves. Pass
 * the full "/api/auth/..." path.
 */
export async function apiFetchAuth<T>(path: string, options: RequestInit = {}): Promise<T> {
  const res = await fetch(path, {
    ...options,
    headers: { "Content-Type": "application/json", ...options.headers },
  });

  let body: unknown = undefined;
  try {
    body = await res.json();
  } catch {
    /* no body */
  }

  if (!res.ok) {
    const detail = (body as { detail?: string } | undefined)?.detail ?? res.statusText;
    throw new ApiError(res.status, detail);
  }
  return body as T;
}

export const COOKIE_NAME = "mentora_session";
