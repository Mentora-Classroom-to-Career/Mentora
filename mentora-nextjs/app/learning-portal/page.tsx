import type { Metadata } from "next";
import Link from "next/link";
import { cookies } from "next/headers";
import SiteHeader from "@/components/SiteHeader";
import SiteFooter from "@/components/SiteFooter";
import { apiFetchServer, COOKIE_NAME } from "@/lib/api";
import type { LearningStats, PlaylistVideo, Assignment, Material } from "@/lib/types";

export const metadata: Metadata = { title: "Learning Portal" };

export default async function LearningPortalPage() {
  const token = cookies().get(COOKIE_NAME)?.value;

  const [stats, playlistData, assignmentsData, materialsData] = await Promise.all([
    apiFetchServer<LearningStats>("/learning/stats", token),
    apiFetchServer<{ videos: PlaylistVideo[] }>("/learning/playlist", token),
    apiFetchServer<{ assignments: Assignment[] }>("/learning/assignments", token),
    apiFetchServer<{ materials: Material[] }>("/learning/materials", token),
  ]);

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
              <div className="stat-num">{stats.videos_in_playlist}</div>
              <div className="stat-sub">{stats.videos_watched_this_week} watched this week</div>
            </div>
            <div className="stat-cell">
              <div className="stat-label">MCQ Assignments</div>
              <div className="stat-num">{stats.mcq_assignments}</div>
              <div className="stat-sub">{stats.mcq_pending_today} pending today</div>
            </div>
            <div className="stat-cell">
              <div className="stat-label">Course PDFs</div>
              <div className="stat-num">{stats.course_pdfs}</div>
              <div className="stat-sub">{stats.course_pdfs_downloaded} downloaded</div>
            </div>
            <div className="stat-cell">
              <div className="stat-label">Avg. MCQ Score</div>
              <div className="stat-num">{stats.avg_mcq_score}%</div>
              <div className="stat-sub">+{stats.avg_mcq_score_delta}% this week</div>
            </div>
          </div>

          <div className="layout-sidebar">
            <aside className="side-nav-block">
              <div className="group-label">My Content</div>
              <a href="#" className="side-link active">My Playlists <span className="count-badge">{stats.videos_in_playlist}</span></a>
              <a href="#" className="side-link">MCQ Assignments <span className="count-badge">{stats.mcq_pending_today}</span></a>
              <a href="#" className="side-link">Course PDFs</a>
              <a href="#" className="side-link">Watch History</a>
              <div className="group-label">Filter by Subject</div>
              <a href="#" className="side-link">Mathematics</a>
              <a href="#" className="side-link">English</a>
              <a href="#" className="side-link">Physics</a>
              <a href="#" className="side-link">Chemistry</a>
              <div className="group-label">Tools</div>
              <a href="#" className="side-link">AI Topic Explainer</a>
              <a href="#" className="side-link">Download All</a>
            </aside>

            <div>
              <div className="flex-between" style={{ marginBottom: 14 }}>
                <h3 style={{ fontFamily: "var(--font-display)", margin: 0 }}>My Video Playlists</h3>
                <button className="btn btn-primary btn-sm">Add Video</button>
              </div>

              {playlistData.videos.map((v) => (
                <div className="media-row" key={v.id}>
                  <div className="media-thumb" />
                  <div className="media-info">
                    <h4>{v.title}</h4>
                    <span className="tag-subject">{v.subject}</span>
                    <div className="meta">{v.source} · {v.duration_min} min · {v.watched ? "Watched" : "Not watched yet"}</div>
                  </div>
                  <div className="media-actions">
                    <button className="btn btn-primary btn-sm">Watch</button>
                    <button className="btn btn-outline btn-sm">Remove</button>
                  </div>
                </div>
              ))}

              <h3 style={{ fontFamily: "var(--font-display)", margin: "28px 0 14px" }}>MCQ Assignments</h3>
              <div className="grid-2" style={{ marginBottom: 28 }}>
                {assignmentsData.assignments.map((a) => (
                  <div className="panel" key={a.id}>
                    <div className="flex-between" style={{ marginBottom: 10 }}>
                      <h4 style={{ margin: 0 }}>{a.title}</h4>
                      <span className={`status-pill status-${a.status === "done" ? "done" : "today"}`}>
                        {a.status === "done" ? "Completed" : "New"}
                      </span>
                    </div>
                    <p className="small-muted" style={{ margin: "0 0 14px" }}>
                      {a.subject} · Due: {a.due}
                    </p>
                    <Link href="/exam-prep" className={`btn ${a.status === "done" ? "btn-outline" : "btn-primary"} btn-block`}>
                      {a.status === "done" ? "Retry Test →" : "Start Test →"}
                    </Link>
                  </div>
                ))}
              </div>

              <h3 style={{ fontFamily: "var(--font-display)", margin: "28px 0 14px" }}>Course PDFs</h3>
              <div className="grid-3">
                {materialsData.materials.map((m) => (
                  <div className="panel" key={m.id}>
                    <h4 style={{ margin: "0 0 6px" }}>{m.title}</h4>
                    <p className="small-muted" style={{ margin: "0 0 14px" }}>{m.subject}</p>
                    <button className="btn btn-outline btn-block btn-sm">
                      {m.downloaded ? "Re-download" : "Download"}
                    </button>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </section>

      <SiteFooter />
    </>
  );
}
