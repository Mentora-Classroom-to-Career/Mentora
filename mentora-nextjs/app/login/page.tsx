"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import AuthShell from "@/components/AuthShell";
import { apiFetchAuth, ApiError } from "@/lib/api";

export default function LoginPage() {
  const router = useRouter();
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  async function handleSubmit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    setError(null);
    setSubmitting(true);

    const formData = new FormData(e.currentTarget);
    const email = formData.get("email") as string;
    const password = formData.get("password") as string;

    try {
      await apiFetchAuth("/api/auth/login", {
        method: "POST",
        body: JSON.stringify({ email, password }),
      });
      router.push("/dashboard");
      router.refresh();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Something went wrong. Please try again.");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <AuthShell
      sideHeading="MENTORA"
      sideCopy="Welcome back. Continue where you left off — your AI timetable and career roadmap are ready."
      sideChecklist={[
        "Your Student Intelligence Profile (SIP) is synced",
        "Timetable auto-updated · Days remaining recalculated",
        "Notifications & session reminders waiting",
      ]}
    >
      <h2>Login to Mentora</h2>
      <p className="subhead">Enter your registered email and password</p>

      <form id="login-form" name="login-form" onSubmit={handleSubmit}>
        {error && (
          <div className="demo-note" style={{ borderColor: "#c0392b", marginBottom: 16 }}>
            {error}
          </div>
        )}

        <div className="field">
          <label htmlFor="email">Email Address</label>
          <input
            type="email"
            id="email"
            name="email"
            placeholder="student@example.com"
            required
            autoComplete="email"
          />
        </div>

        <div className="field">
          <label htmlFor="password">Password</label>
          <input
            type="password"
            id="password"
            name="password"
            placeholder="Your password"
            required
            minLength={8}
            autoComplete="current-password"
          />
        </div>

        <div className="flex-between" style={{ marginBottom: 20, marginTop: -10 }}>
          <span />
          <Link href="/forgot-password" className="small-muted">
            Forgot password?
          </Link>
        </div>

        <button type="submit" className="btn btn-primary btn-block" disabled={submitting}>
          {submitting ? "Logging in…" : "Login →"}
        </button>
      </form>

      <div className="divider-or">or continue with</div>
      <button type="button" className="btn btn-outline btn-block" disabled>
        Continue with Google
      </button>

      <p className="auth-legal">
        Don&apos;t have an account?{" "}
        <Link href="/register" className="link-inline">
          Register now
        </Link>
      </p>
    </AuthShell>
  );
}
