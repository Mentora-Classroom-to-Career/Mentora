import type { Metadata } from "next";
import { cookies } from "next/headers";
import SiteHeader from "@/components/SiteHeader";
import SiteFooter from "@/components/SiteFooter";
import DashboardSettings, { EditProfileButton } from "@/components/DashboardSettings";
import { apiFetchServer, COOKIE_NAME } from "@/lib/api";
import type {
  DashboardStats, ScoreTrendPoint, WeeklySnapshotDay, WeakTopic, NotificationItem, UserPublic,
} from "@/lib/types";

export const metadata: Metadata = { title: "Dashboard" };

function scoreTrendToPolyline(points: ScoreTrendPoint[]): { line: string; fill: string } {
  if (points.length === 0) return { line: "", fill: "" };
  const width = 560;
  const height = 160;
  const pad = 10;
  const max = Math.max(...points.map((p) => p.score), 100);
  const min = Math.min(...points.map((p) => p.score), 0);
  const range = Math.max(max - min, 1);
  const step = (width - pad * 2) / Math.max(points.length - 1, 1);

  const coords = points.map((p, i) => {
    const x = pad + i * step;
    const y = height - pad - ((p.score - min) / range) * (height - pad * 2);
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  });

  const line = coords.join(" ");
  const fill = `${line} ${coords[coords.length - 1].split(",")[0]},${height} ${coords[0].split(",")[0]},${height}`;
  return { line, fill };
}

function statusColor(status: WeeklySnapshotDay["status"]) {
  switch (status) {
    case "done": return { bg: "var(--mint-card)", text: "var(--green-700)", label: "Done ✓" };
    case "today": return { bg: "var(--green-100)", text: "var(--green-700)", label: "Today" };
    case "missed": return { bg: "var(--red-100)", text: "var(--red-600)", label: "Missed" };
    default: return { bg: "var(--green-50)", text: "var(--ink-500)", label: "Pending" };
  }
}

function weakTopicBarClass(score: number) {
  if (score < 50) return "low";
  if (score < 70) return "warn";
  return "";
}

export default async function DashboardPage() {
  const token = cookies().get(COOKIE_NAME)?.value;

  const [user, stats, trend, snapshot, weakTopics, notifications] = await Promise.all([
    apiFetchServer<UserPublic>("/users/me", token),
    apiFetchServer<DashboardStats>("/dashboard/stats", token),
    apiFetchServer<{ points: ScoreTrendPoint[] }>("/dashboard/score-trend", token),
    apiFetchServer<{ days: WeeklySnapshotDay[] }>("/dashboard/weekly-snapshot", token),
    apiFetchServer<{ topics: WeakTopic[] }>("/dashboard/weak-topics", token),
    apiFetchServer<{ items: NotificationItem[] }>("/notifications", token),
  ]);

  const { line, fill } = scoreTrendToPolyline(trend.points);
  const initials = `${user.first_name[0] ?? ""}${user.last_name[0] ?? ""}`.toUpperCase();

  return (
    <>
      <SiteHeader variant="app" />

      <section className="page-heading">
        <div className="container">
          <div className="profile-row">
            <div className="avatar-circle">{initials}</div>
            <div>
              <h1 style={{ marginBottom: 2 }}>{user.first_name} {user.last_name}</h1>
              <p className="mb-0">{user.university ?? "—"}</p>
              <p className="small-muted" style={{ margin: "2px 0 0" }}>
                Preparing for: {user.exam_goal ?? "—"}
              </p>
            </div>
          </div>
          <EditProfileButton user={user} />
        </div>
      </section>

      <section className="section">
        <div className="container">
          <div className="stat-strip" style={{ marginBottom: 28 }}>
            <div className="stat-cell">
              <div className="stat-label">Exam Sessions</div>
              <div className="stat-num">{stats.exam_sessions}</div>
              <div className="stat-sub" style={{ color: "var(--ink-500)" }}>Total practice sessions</div>
            </div>
            <div className="stat-cell">
              <div className="stat-label">Avg. Score</div>
              <div className="stat-num">{stats.avg_score}%</div>
              <div className="stat-sub">+{stats.avg_score_delta}% from last week</div>
            </div>
            <div className="stat-cell">
              <div className="stat-label">Videos Watched</div>
              <div className="stat-num">{stats.videos_watched}</div>
              <div className="stat-sub" style={{ color: "var(--ink-500)" }}>{stats.videos_watched_this_week} this week</div>
            </div>
            <div className="stat-cell">
              <div className="stat-label">Career Match (top)</div>
              <div className="stat-num" style={{ fontSize: "1.2rem" }}>{stats.top_career_match.title}</div>
              <div className="stat-sub">{stats.top_career_match.fit_percent}% fit</div>
            </div>
          </div>

          <div className="grid-2" style={{ gridTemplateColumns: "2fr 1fr", alignItems: "start" }}>
            <div>
              <div className="panel" style={{ marginBottom: 20 }}>
                <h4 style={{ margin: "0 0 16px" }}>Score trend (last {trend.points.length} sessions)</h4>
                <div className="chart-wrap">
                  <svg viewBox="0 0 560 160" width="100%" height="160" preserveAspectRatio="none">
                    <polyline points={line} fill="none" stroke="#1a5c42" strokeWidth={3} strokeLinecap="round" strokeLinejoin="round" />
                    <polyline points={fill} fill="#eef8f2" stroke="none" opacity={0.7} />
                  </svg>
                </div>
                <div className="chart-axis-labels">
                  {trend.points.map((p) => <span key={p.session}>{p.session}</span>)}
                </div>
              </div>

              <div className="panel">
                <h4 style={{ margin: "0 0 14px" }}>This week&apos;s timetable snapshot</h4>
                <div className="snapshot-grid">
                  {snapshot.days.map((d) => {
                    const c = statusColor(d.status);
                    return (
                      <div key={d.day} className="snapshot-day" style={{ background: c.bg }}>
                        <div className="day-name">{d.day}</div>
                        <div style={{ color: c.text }}>{c.label}<br />{d.topic}</div>
                      </div>
                    );
                  })}
                </div>
                <p className="small-muted" style={{ margin: "14px 0 0" }}>
                  ● Completion rate this week: {snapshot.days.filter((d) => d.status === "done").length}/{snapshot.days.length} sessions
                </p>
              </div>
            </div>

            <div>
              <h4 style={{ margin: "0 0 12px", display: "flex", alignItems: "center", gap: 6 }}>
                ⚠ Weak Topics (AI flagged)
              </h4>
              {weakTopics.topics.map((t) => (
                <div className="weak-item" key={`${t.subject}-${t.topic}`}>
                  <h4>{t.subject} — {t.topic}</h4>
                  <div className="score-line">Score: {t.score}%{t.note ? ` · ${t.note}` : ""}</div>
                  <span className="progress-track">
                    <span className={`progress-fill ${weakTopicBarClass(t.score)}`} style={{ width: `${t.score}%` }} />
                  </span>
                </div>
              ))}

              <h4 style={{ margin: "24px 0 12px", display: "flex", alignItems: "center", gap: 6 }}>
                🔔 Notifications
              </h4>
              {notifications.items.map((n) => (
                <div className={`notif-item${n.is_read ? "" : " alert"}`} key={n.id}>{n.message}</div>
              ))}
            </div>
          </div>

          <DashboardSettings user={user} />
        </div>
      </section>

      <SiteFooter />
    </>
  );
}
