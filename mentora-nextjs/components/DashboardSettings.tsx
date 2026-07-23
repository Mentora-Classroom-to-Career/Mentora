"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { apiFetch, apiFetchAuth, ApiError } from "@/lib/api";
import type { UserPublic } from "@/lib/types";

export default function DashboardSettings({ user }: { user: UserPublic }) {
  const router = useRouter();
  const [busy, setBusy] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);

  async function handleEditProfile() {
    const university = prompt("University / Institution", user.university ?? "");
    if (university === null) return;
    setBusy("profile");
    setMessage(null);
    try {
      await apiFetch("/users/me", { method: "PATCH", body: JSON.stringify({ university }) });
      router.refresh();
    } catch (err) {
      setMessage(err instanceof ApiError ? err.message : "Couldn't update profile right now.");
    } finally {
      setBusy(null);
    }
  }

  async function handleChangePassword() {
    const current_password = prompt("Current password");
    if (!current_password) return;
    const new_password = prompt("New password (min. 8 characters)");
    if (!new_password) return;
    setBusy("password");
    setMessage(null);
    try {
      await apiFetch("/auth/change-password", { method: "POST", body: JSON.stringify({ current_password, new_password }) });
      setMessage("Password updated.");
    } catch (err) {
      setMessage(err instanceof ApiError ? err.message : "Couldn't change password right now.");
    } finally {
      setBusy(null);
    }
  }

  async function handleDeleteAccount() {
    if (!confirm("This permanently deletes your account. Continue?")) return;
    setBusy("delete");
    try {
      await apiFetch("/users/me", { method: "DELETE" });
      await apiFetchAuth("/api/auth/logout", { method: "POST" });
      router.push("/");
      router.refresh();
    } catch (err) {
      setMessage(err instanceof ApiError ? err.message : "Couldn't delete account right now.");
      setBusy(null);
    }
  }

  return (
    <div id="settings" style={{ marginTop: 36 }}>
      <h4 style={{ margin: "0 0 14px" }}>Settings</h4>
      {message && <p className="small-muted" style={{ marginBottom: 12 }}>{message}</p>}
      <div className="settings-grid">
        <button className="btn btn-outline btn-sm" onClick={handleEditProfile} disabled={busy === "profile"}>
          👤 Account
        </button>
        <button className="btn btn-outline btn-sm" disabled>
          🔔 Notification Prefs
        </button>
        <button className="btn btn-outline btn-sm" disabled>
          📊 Study Preferences
        </button>
        <button className="btn btn-outline btn-sm" onClick={handleChangePassword} disabled={busy === "password"}>
          🔒 Change Password
        </button>
        <button
          className="btn btn-sm"
          style={{ background: "var(--red-100)", color: "var(--red-600)" }}
          onClick={handleDeleteAccount}
          disabled={busy === "delete"}
        >
          🗑 Delete Account
        </button>
      </div>
    </div>
  );
}

export function EditProfileButton({ user }: { user: UserPublic }) {
  const router = useRouter();
  const [busy, setBusy] = useState(false);

  async function handleEditProfile() {
    const first_name = prompt("First name", user.first_name) ?? user.first_name;
    const last_name = prompt("Last name", user.last_name) ?? user.last_name;
    setBusy(true);
    try {
      await apiFetch("/users/me", { method: "PATCH", body: JSON.stringify({ first_name, last_name }) });
      router.refresh();
    } finally {
      setBusy(false);
    }
  }

  return (
    <button className="btn btn-primary" onClick={handleEditProfile} disabled={busy}>
      Edit Profile
    </button>
  );
}
