import type { Metadata } from "next";
import Link from "next/link";
import SiteHeader from "@/components/SiteHeader";
import SiteFooter from "@/components/SiteFooter";

export const metadata: Metadata = { title: "Learning Portal" };

export default function LearningPortalPage() {
  return (
    <>
      <SiteHeader variant="app" />

      <section className="page-heading">
        <div className="container">
          <div>
            <h1>Learning Portal</h1>
            <p>Your personal playlists, MCQ assignments, and course materials — all in one place</p>
          </div>
          <button className="btn btn-primary">Upload Content</button>
        </div>
      </section>

      <section className="section" style={{ paddingTop: 32 }}>
        <div className="container">
          <div className="stat-strip mt-24" style={{ marginBottom: 28 }}>
            <div className="stat-cell">
              <div className="stat-label">Videos in Playlist</div>
              <div className="stat-num">38</div>
              <div className="stat-sub">14 watched this week</div>
            </div>
            <div className="stat-cell">
              <div className="stat-label">MCQ Assignments</div>
              <div className="stat-num">12</div>
              <div className="stat-sub">3 pending today</div>
            </div>
            <div className="stat-cell">
              <div className="stat-label">Course PDFs</div>
              <div className="stat-num">7</div>
              <div className="stat-sub">2 downloaded</div>
            </div>
            <div className="stat-cell">
              <div className="stat-label">Avg. MCQ Score</div>
              <div className="stat-num">72%</div>
              <div className="stat-sub">+5% this week</div>
            </div>
          </div>

          <div className="layout-sidebar">
            <aside className="side-nav-block">
              <div className="group-label">My Content</div>
              <a href="#" className="side-link active">
                My Playlists <span className="count-badge">38</span>
              </a>
              <a href="#" className="side-link">
                MCQ Assignments <span className="count-badge">3</span>
              </a>
              <a href="#" className="side-link">
                Course PDFs
              </a>
              <a href="#" className="side-link">
                Watch History
              </a>
              <div className="group-label">Filter by Subject</div>
              <a href="#" className="side-link">
                Mathematics
              </a>
              <a href="#" className="side-link">
                English
              </a>
              <a href="#" className="side-link">
                Physics
              </a>
              <a href="#" className="side-link">
                Chemistry
              </a>
              <div className="group-label">Tools</div>
              <a href="#" className="side-link">
                AI Topic Explainer
              </a>
              <a href="#" className="side-link">
                Download All
              </a>
            </aside>

            <div>
              <div className="flex-between" style={{ marginBottom: 14 }}>
                <h3 style={{ fontFamily: "var(--font-display)", margin: 0 }}>My Video Playlists</h3>
                <button className="btn btn-primary btn-sm">Add Video</button>
              </div>

              <div className="media-row">
                <div className="media-thumb" />
                <div className="media-info">
                  <h4>Quadratic Equations — Full Chapter</h4>
                  <span className="tag-subject">Mathematics</span>
                  <div className="meta">MathCity.org · 18 min · AI-added from today&apos;s session</div>
                </div>
                <div className="media-actions">
                  <button className="btn btn-primary btn-sm">Watch</button>
                  <button className="btn btn-outline btn-sm">Remove</button>
                </div>
              </div>

              <div className="media-row">
                <div className="media-thumb" />
                <div className="media-info">
                  <h4>Grammar Tenses — Active &amp; Passive Voice</h4>
                  <span className="tag-subject">English</span>
                  <div className="meta">Khan Academy · 12 min · AI-added</div>
                </div>
                <div className="media-actions">
                  <button className="btn btn-primary btn-sm">Watch</button>
                  <button className="btn btn-outline btn-sm">Remove</button>
                </div>
              </div>

              <div className="media-row">
                <div className="media-thumb" />
                <div className="media-info">
                  <h4>Geometry — Circles &amp; Angles Full Revision</h4>
                  <span className="tag-subject">Mathematics</span>
                  <div className="meta">MathCity.org · 24 min · AI-added</div>
                </div>
                <div className="media-actions">
                  <button className="btn btn-primary btn-sm">Watch</button>
                  <button className="btn btn-outline btn-sm">Remove</button>
                </div>
              </div>

              <h3 style={{ fontFamily: "var(--font-display)", margin: "28px 0 14px" }}>MCQ Assignments</h3>
              <div className="grid-2" style={{ marginBottom: 28 }}>
                <div className="panel">
                  <div className="flex-between" style={{ marginBottom: 10 }}>
                    <h4 style={{ margin: 0 }}>Quadratic Equations Test</h4>
                    <span className="status-pill status-today">New</span>
                  </div>
                  <p className="small-muted" style={{ margin: "0 0 14px" }}>
                    Mathematics · 5 MCQs · Generated by FLAN-T5 · Difficulty: Medium
                  </p>
                  <Link href="/exam-prep" className="btn btn-primary btn-block">
                    Start Test →
                  </Link>
                </div>
                <div className="panel">
                  <div className="flex-between" style={{ marginBottom: 10 }}>
                    <h4 style={{ margin: 0 }}>English Grammar — Tenses</h4>
                    <span className="status-pill status-done">Completed</span>
                  </div>
                  <p className="small-muted" style={{ margin: "0 0 14px" }}>
                    English · 5 MCQs · Score: 3/5
                    <br />
                    Score: 60% — Weak: Passive Voice
                  </p>
                  <Link href="/exam-prep" className="btn btn-outline btn-block">
                    Retry Test →
                  </Link>
                </div>
              </div>

              <h3 style={{ fontFamily: "var(--font-display)", margin: "28px 0 14px" }}>Course PDFs</h3>
              <div className="grid-3">
                <div className="panel">
                  <h4 style={{ margin: "0 0 6px" }}>MDCAT Biology Full Syllabus 2025</h4>
                  <p className="small-muted" style={{ margin: "0 0 14px" }}>
                    4.2 MB · 2025 · Sindh Board
                  </p>
                  <button className="btn btn-outline btn-block btn-sm">Download</button>
                </div>
                <div className="panel">
                  <h4 style={{ margin: "0 0 6px" }}>Mathematics Chapter Notes — Entry Test</h4>
                  <p className="small-muted" style={{ margin: "0 0 14px" }}>
                    1.8 MB · 2024 · Self-uploaded
                  </p>
                  <button className="btn btn-outline btn-block btn-sm">Download</button>
                </div>
                <div className="panel">
                  <h4 style={{ margin: "0 0 6px" }}>CSS Past Papers 2020–2024</h4>
                  <p className="small-muted" style={{ margin: "0 0 14px" }}>
                    6.1 MB · FPSC Official PDFs
                  </p>
                  <button className="btn btn-outline btn-block btn-sm">Download</button>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      <SiteFooter />
    </>
  );
}
