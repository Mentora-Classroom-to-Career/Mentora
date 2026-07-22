import type { Metadata } from "next";
import Link from "next/link";
import AuthShell from "@/components/AuthShell";

export const metadata: Metadata = { title: "Forgot Password" };

export default function ForgotPasswordPage() {
  return (
    <AuthShell
      sideHeading="MENTORA"
      sideCopy="Password reset is quick and secure. Check your inbox for the reset link."
      sideChecklist={[
        "Reset link sent to your registered email",
        "Link expires in 30 minutes for security",
        "Your progress and data are safely stored",
      ]}
    >
      <Link href="/login" className="small-muted">
        ← Back to Login
      </Link>
      <h2 style={{ marginTop: 14 }}>Forgot your password?</h2>
      <p className="subhead">Enter your registered email address and we&apos;ll send you a reset link</p>

      <form id="request-reset-form" name="request-reset-form" method="post">
        <div className="field">
          <label htmlFor="email">Registered Email Address</label>
          <input
            type="email"
            id="email"
            name="email"
            placeholder="student@example.com"
            required
            autoComplete="email"
          />
        </div>
        <button type="submit" className="btn btn-primary btn-block">
          Send Reset Link →
        </button>
      </form>

      <div className="step-divider" />

      <div className="step-label">Step 2 — Enter Reset Code</div>

      <form id="reset-password-form" name="reset-password-form" method="post">
        <div className="field">
          <label htmlFor="reset_code">Reset Code (from email)</label>
          <input
            type="text"
            id="reset_code"
            name="reset_code"
            placeholder="6-digit code"
            required
            pattern="[0-9]{6}"
            maxLength={6}
            inputMode="numeric"
          />
        </div>
        <div className="field">
          <label htmlFor="new_password">New Password</label>
          <input
            type="password"
            id="new_password"
            name="new_password"
            placeholder="Min. 8 characters"
            required
            minLength={8}
            autoComplete="new-password"
          />
        </div>
        <div className="field">
          <label htmlFor="confirm_password">Confirm New Password</label>
          <input
            type="password"
            id="confirm_password"
            name="confirm_password"
            placeholder="Repeat password"
            required
            minLength={8}
            autoComplete="new-password"
          />
        </div>
        <button type="submit" className="btn btn-outline btn-block">
          Reset Password →
        </button>
      </form>

      <div className="demo-note">
        If you don&apos;t see the email, check your spam/junk folder · Contact: support@mentora.pk
      </div>
    </AuthShell>
  );
}
