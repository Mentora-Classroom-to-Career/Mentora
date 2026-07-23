"use client";

import { useState } from "react";
import Link from "next/link";
import AuthShell from "@/components/AuthShell";
import { apiFetchAuth, ApiError } from "@/lib/api";

export default function ForgotPasswordPage() {
  const [requestError, setRequestError] = useState<string | null>(null);
  const [requestSent, setRequestSent] = useState(false);
  const [requestSubmitting, setRequestSubmitting] = useState(false);

  const [resetError, setResetError] = useState<string | null>(null);
  const [resetDone, setResetDone] = useState(false);
  const [resetSubmitting, setResetSubmitting] = useState(false);

  async function handleRequestReset(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    setRequestError(null);
    setRequestSubmitting(true);
    const email = new FormData(e.currentTarget).get("email") as string;
    try {
      await apiFetchAuth("/api/auth/forgot-password", {
        method: "POST",
        body: JSON.stringify({ email }),
      });
      setRequestSent(true);
    } catch (err) {
      setRequestError(err instanceof ApiError ? err.message : "Something went wrong. Please try again.");
    } finally {
      setRequestSubmitting(false);
    }
  }

  async function handleResetPassword(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    setResetError(null);

    const formData = new FormData(e.currentTarget);
    const payload = {
      reset_code: formData.get("reset_code") as string,
      new_password: formData.get("new_password") as string,
      confirm_password: formData.get("confirm_password") as string,
    };

    if (payload.new_password !== payload.confirm_password) {
      setResetError("Passwords do not match");
      return;
    }

    setResetSubmitting(true);
    try {
      await apiFetchAuth("/api/auth/reset-password", {
        method: "POST",
        body: JSON.stringify(payload),
      });
      setResetDone(true);
    } catch (err) {
      setResetError(err instanceof ApiError ? err.message : "Invalid or expired reset code.");
    } finally {
      setResetSubmitting(false);
    }
  }

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
      <Link href="/login" className="small-muted">← Back to Login</Link>
      <h2 style={{ marginTop: 14 }}>Forgot your password?</h2>
      <p className="subhead">Enter your registered email address and we&apos;ll send you a reset link</p>

      <form id="request-reset-form" name="request-reset-form" onSubmit={handleRequestReset}>
        {requestError && (
          <div className="demo-note" style={{ borderColor: "#c0392b", marginBottom: 16 }}>
            {requestError}
          </div>
        )}
        {requestSent && (
          <div className="demo-note" style={{ marginBottom: 16 }}>
            If that email is registered, a reset code has been sent.
          </div>
        )}
        <div className="field">
          <label htmlFor="email">Registered Email Address</label>
          <input type="email" id="email" name="email" placeholder="student@example.com" required autoComplete="email" />
        </div>
        <button type="submit" className="btn btn-primary btn-block" disabled={requestSubmitting}>
          {requestSubmitting ? "Sending…" : "Send Reset Link →"}
        </button>
      </form>

      <div className="step-divider" />

      <div className="step-label">Step 2 — Enter Reset Code</div>

      <form id="reset-password-form" name="reset-password-form" onSubmit={handleResetPassword}>
        {resetError && (
          <div className="demo-note" style={{ borderColor: "#c0392b", marginBottom: 16 }}>
            {resetError}
          </div>
        )}
        {resetDone && (
          <div className="demo-note" style={{ marginBottom: 16 }}>
            Password reset successful. You can now <Link href="/login" className="link-inline">log in</Link>.
          </div>
        )}
        <div className="field">
          <label htmlFor="reset_code">Reset Code (from email)</label>
          <input type="text" id="reset_code" name="reset_code" placeholder="6-digit code" required pattern="[0-9]{6}" maxLength={6} inputMode="numeric" />
        </div>
        <div className="field">
          <label htmlFor="new_password">New Password</label>
          <input type="password" id="new_password" name="new_password" placeholder="Min. 8 characters" required minLength={8} autoComplete="new-password" />
        </div>
        <div className="field">
          <label htmlFor="confirm_password">Confirm New Password</label>
          <input type="password" id="confirm_password" name="confirm_password" placeholder="Repeat password" required minLength={8} autoComplete="new-password" />
        </div>
        <button type="submit" className="btn btn-outline btn-block" disabled={resetSubmitting}>
          {resetSubmitting ? "Resetting…" : "Reset Password →"}
        </button>
      </form>

      <div className="demo-note">
        If you don&apos;t see the email, check your spam/junk folder · Contact: support@mentora.pk
      </div>
    </AuthShell>
  );
}
