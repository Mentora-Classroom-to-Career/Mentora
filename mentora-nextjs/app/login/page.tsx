import type { Metadata } from "next";
import Link from "next/link";
import AuthShell from "@/components/AuthShell";

export const metadata: Metadata = { title: "Login" };

export default function LoginPage() {
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

      <form id="login-form" name="login-form" method="post">
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

        <button type="submit" className="btn btn-primary btn-block">
          Login →
        </button>
      </form>

      <div className="divider-or">or continue with</div>
      <button type="button" className="btn btn-outline btn-block">
        Continue with Google
      </button>

      <p className="auth-legal">
        Don&apos;t have an account?{" "}
        <Link href="/register" className="link-inline">
          Register now
        </Link>
      </p>

      <div className="demo-note">
        For the FYP demo, use student: <b>demo@mentora.pk</b> / pass: <b>demo2025</b>
      </div>
    </AuthShell>
  );
}
