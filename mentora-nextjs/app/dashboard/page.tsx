import type { Metadata } from "next";
import SiteHeader from "@/components/SiteHeader";
import SiteFooter from "@/components/SiteFooter";

export const metadata: Metadata = { title: "Dashboard" };

export default function DashboardPage() {
  return (
    <>
      <SiteHeader variant="app" />

      <section className="page-heading">
        <div className="container">
          <div className="profile-row">
            <div className="avatar-circle">RR</div>
            <div>
              <h1 style={{ marginBottom: 2 }}>Raimal Raja</h1>
              <p className="mb-0">BS Computer Science · University of Sindh, Laar Campus · Batch 2023</p>
              <p className="small-muted" style={{ margin: "2px 0 0" }}>
                Preparing for: Sindh University Entry Test · 47 days left
              </p>
            </div>
          </div>
          <button className="btn btn-primary">Edit Profile</button>
        </div>
      </section>

      <section className="section">
        <div className="container">
          <div className="stat-strip" style={{ marginBottom: 28 }}>
            <div className="stat-cell">
              <div className="stat-label">Exam Sessions</div>
              <div className="stat-num">24</div>
              <div className="stat-sub" style={{ color: "var(--ink-500)" }}>
                Total practice sessions
              </div>
            </div>
            <div className="stat-cell">
              <div className="stat-label">Avg. Score</div>
              <div className="stat-num">72%</div>
              <div className="stat-sub">+5% from last week</div>
            </div>
            <div className="stat-cell">
              <div className="stat-label">Videos Watched</div>
              <div className="stat-num">38</div>
              <div className="stat-sub" style={{ color: "var(--ink-500)" }}>
                14 this week
              </div>
            </div>
            <div className="stat-cell">
              <div className="stat-label">Career Match (top)</div>
              <div className="stat-num" style={{ fontSize: "1.2rem" }}>
                Data Scientist
              </div>
              <div className="stat-sub">88% fit</div>
            </div>
          </div>

          <div className="grid-2" style={{ gridTemplateColumns: "2fr 1fr", alignItems: "start" }}>
            <div>
              <div className="panel" style={{ marginBottom: 20 }}>
                <h4 style={{ margin: "0 0 16px" }}>Score trend (last 10 sessions)</h4>
                <div className="chart-wrap">
                  <svg viewBox="0 0 560 160" width="100%" height="160" preserveAspectRatio="none">
                    <polyline
                      points="10,120 65,110 120,95 175,100 230,80 285,85 340,60 395,65 450,45 505,40"
                      fill="none"
                      stroke="#1a5c42"
                      strokeWidth={3}
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    />
                    <polyline
                      points="10,120 65,110 120,95 175,100 230,80 285,85 340,60 395,65 450,45 505,40 505,160 10,160"
                      fill="#eef8f2"
                      stroke="none"
                      opacity={0.7}
                    />
                  </svg>
                </div>
                <div className="chart-axis-labels">
                  <span>S1</span>
                  <span>S2</span>
                  <span>S3</span>
                  <span>S4</span>
                  <span>S5</span>
                  <span>S6</span>
                  <span>S7</span>
                  <span>S8</span>
                  <span>S9</span>
                  <span>S10</span>
                </div>
              </div>

              <div className="panel">
                <h4 style={{ margin: "0 0 14px" }}>This week&apos;s timetable snapshot</h4>
                <div className="snapshot-grid">
                  <div className="snapshot-day" style={{ background: "var(--mint-card)" }}>
                    <div className="day-name">Mon</div>
                    <div style={{ color: "var(--green-700)" }}>
                      Done ✓
                      <br />
                      Algebra
                    </div>
                  </div>
                  <div className="snapshot-day" style={{ background: "var(--mint-card)" }}>
                    <div className="day-name">Tue</div>
                    <div style={{ color: "var(--green-700)" }}>
                      Done ✓
                      <br />
                      English
                    </div>
                  </div>
                  <div
                    className="snapshot-day"
                    style={{ background: "var(--green-100)", border: "2px solid var(--green-700)" }}
                  >
                    <div className="day-name">Wed</div>
                    <div style={{ color: "var(--green-700)" }}>
                      Today
                      <br />
                      Grammar
                    </div>
                  </div>
                  <div className="snapshot-day" style={{ background: "var(--red-100)" }}>
                    <div className="day-name">Thu</div>
                    <div style={{ color: "var(--red-600)" }}>
                      Missed
                      <br />
                      Geometry
                    </div>
                  </div>
                  <div className="snapshot-day" style={{ background: "var(--green-50)" }}>
                    <div className="day-name">Fri</div>
                    <div style={{ color: "var(--ink-500)" }}>
                      Pending
                      <br />
                      Revision
                    </div>
                  </div>
                </div>
                <p className="small-muted" style={{ margin: "14px 0 0" }}>
                  ● Completion rate this week: 2/5 sessions
                </p>
              </div>
            </div>

            <div>
              <h4 style={{ margin: "0 0 12px", display: "flex", alignItems: "center", gap: 6 }}>
                ⚠ Weak Topics (AI flagged)
              </h4>
              <div className="weak-item">
                <h4>Algebra — Quadratic Equations</h4>
                <div className="score-line">Score: 38% · Needs urgent review</div>
                <span className="progress-track">
                  <span className="progress-fill low" style={{ width: "38%" }} />
                </span>
              </div>
              <div className="weak-item">
                <h4>Grammar — Tenses</h4>
                <div className="score-line">Score: 44%</div>
                <span className="progress-track">
                  <span className="progress-fill low" style={{ width: "44%" }} />
                </span>
              </div>
              <div className="weak-item">
                <h4>Geometry — Circles</h4>
                <div className="score-line">Score: 60% · Medium</div>
                <span className="progress-track">
                  <span className="progress-fill warn" style={{ width: "60%" }} />
                </span>
              </div>
              <div className="weak-item">
                <h4>Reading Comprehension</h4>
                <div className="score-line">Score: 62% · Improving</div>
                <span className="progress-track">
                  <span className="progress-fill" style={{ width: "62%" }} />
                </span>
              </div>

              <h4 style={{ margin: "24px 0 12px", display: "flex", alignItems: "center", gap: 6 }}>
                🔔 Notifications
              </h4>
              <div className="notif-item alert">
                ⚠ Score trend flagged &quot;at risk&quot; in Algebra — Trajectory Predictor
              </div>
              <div className="notif-item ok">✅ New topic playlist ready for Algebra</div>
              <div className="notif-item time">🕒 Tomorrow: Grammar session — 7 PM</div>
              <div className="notif-item match">🎯 Career match updated: +2 points</div>
            </div>
          </div>

          <div id="settings" style={{ marginTop: 36 }}>
            <h4 style={{ margin: "0 0 14px" }}>Settings</h4>
            <div className="settings-grid">
              <button className="btn btn-outline btn-sm">👤 Account</button>
              <button className="btn btn-outline btn-sm">🔔 Notification Prefs</button>
              <button className="btn btn-outline btn-sm">📊 Study Preferences</button>
              <button className="btn btn-outline btn-sm">🔒 Change Password</button>
              <button className="btn btn-sm" style={{ background: "var(--red-100)", color: "var(--red-600)" }}>
                🗑 Delete Account
              </button>
            </div>
          </div>
        </div>
      </section>

      <SiteFooter />
    </>
  );
}
