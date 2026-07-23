"use client";

import { useEffect, useState } from "react";
import SiteHeader from "@/components/SiteHeader";
import SiteFooter from "@/components/SiteFooter";
import { apiFetch, ApiError } from "@/lib/api";
import type {
  CVExtracted, CareerRankingRow, Roadmap, InterviewStartResult, InterviewAskResult,
} from "@/lib/types";

export default function CareerPage() {
  const [rankings, setRankings] = useState<CareerRankingRow[]>([]);
  const [careerGoal, setCareerGoal] = useState("data_scientist");
  const [extracted, setExtracted] = useState<CVExtracted | null>(null);
  const [uploading, setUploading] = useState(false);
  const [roadmap, setRoadmap] = useState<Roadmap | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [interviewId, setInterviewId] = useState<string | null>(null);
  const [interviewQuestion, setInterviewQuestion] = useState<string | null>(null);
  const [interviewAnswer, setInterviewAnswer] = useState("");
  const [interviewBusy, setInterviewBusy] = useState(false);

  async function refreshRankings() {
    try {
      const data = await apiFetch<{ rankings: CareerRankingRow[] }>("/career/rankings");
      setRankings(data.rankings);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Couldn't load career rankings right now.");
    }
  }

  async function refreshRoadmap(careerId: number) {
    try {
      const data = await apiFetch<Roadmap>(`/career/roadmap?career_id=${careerId}`);
      setRoadmap(data);
    } catch {
      /* non-critical */
    }
  }

  useEffect(() => {
    refreshRankings();
    refreshRoadmap(1);
  }, []);

  async function handleCvUpload(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    const fileInput = e.currentTarget.elements.namedItem("cv_file") as HTMLInputElement;
    const file = fileInput.files?.[0];
    if (!file) return;

    const formData = new FormData();
    formData.append("cv_file", file);

    setUploading(true);
    setError(null);
    try {
      const result = await apiFetch<CVExtracted>("/career/cv-upload", { method: "POST", body: formData });
      setExtracted(result);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Couldn't process that CV right now.");
    } finally {
      setUploading(false);
    }
  }

  async function handleGoalChange(e: React.ChangeEvent<HTMLSelectElement>) {
    const value = e.target.value;
    setCareerGoal(value);
    await apiFetch("/career/goal", { method: "POST", body: JSON.stringify({ career_goal: value }) });
    await refreshRankings();
  }

  async function handleAiSuggest() {
    try {
      const suggestion = await apiFetch<{ suggested_career_id: number; suggested_title: string }>("/career/ai-suggest");
      await refreshRankings();
      await refreshRoadmap(suggestion.suggested_career_id);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Couldn't get an AI suggestion right now.");
    }
  }

  async function handleStartInterview() {
    setInterviewBusy(true);
    setError(null);
    try {
      const topTitle = rankings[0]?.title ?? "Data Scientist";
      const result = await apiFetch<InterviewStartResult>("/interview/start", {
        method: "POST",
        body: JSON.stringify({ career_title: topTitle }),
      });
      setInterviewId(result.interview_id);
      setInterviewQuestion(result.first_question);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Couldn't start the mock interview right now.");
    } finally {
      setInterviewBusy(false);
    }
  }

  async function handleAskInterview(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    if (!interviewId || !interviewAnswer.trim()) return;
    setInterviewBusy(true);
    try {
      const result = await apiFetch<InterviewAskResult>(`/interview/${interviewId}/ask`, {
        method: "POST",
        body: JSON.stringify({ answer: interviewAnswer }),
      });
      setInterviewQuestion(result.answer);
      setInterviewAnswer("");
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Couldn't continue the interview right now.");
    } finally {
      setInterviewBusy(false);
    }
  }

  return (
    <>
      <SiteHeader variant="app" />

      <section className="page-heading">
        <div className="container">
          <div>
            <h1>Career Counseling</h1>
            <p>AI-powered career-fit analysis using your CV · 50+ career paths ranked · 6-month roadmap builder</p>
          </div>
        </div>
      </section>

      <section className="section">
        <div className="container">
          {error && <div className="demo-note" style={{ borderColor: "#c0392b", marginBottom: 20 }}>{error}</div>}

          <div className="grid-3" style={{ marginBottom: 28 }}>
            <div className="panel">
              <div className="small-muted" style={{ marginBottom: 14 }}>Step 1 — Upload CV / Transcript</div>
              <form id="cv-upload-form" name="cv-upload-form" onSubmit={handleCvUpload}>
                <label
                  htmlFor="cv_file"
                  style={{ display: "block", border: "1px dashed var(--border-soft)", borderRadius: 8, padding: 24, textAlign: "center", cursor: "pointer" }}
                >
                  <p style={{ fontSize: "0.85rem", color: "var(--ink-500)", margin: "0 0 14px" }}>
                    Drag &amp; drop your CV or Transcript
                    <br />
                    Supports PDF, DOCX formats · NER model extracts your data
                  </p>
                  <span className="btn btn-primary btn-sm">{uploading ? "Uploading…" : "Browse File"}</span>
                  <input
                    type="file" id="cv_file" name="cv_file" accept=".pdf,.docx"
                    style={{ position: "absolute", width: 1, height: 1, opacity: 0, overflow: "hidden" }}
                    onChange={(e) => e.currentTarget.form?.requestSubmit()}
                  />
                </label>
              </form>
            </div>

            <div className="panel">
              <div className="small-muted" style={{ marginBottom: 14 }}>Step 2 — Skills Extracted (AI)</div>
              {extracted ? (
                <>
                  <div className="flex-between" style={{ marginBottom: 12 }}>
                    <span className="small-muted">GPA</span>
                    <strong>{extracted.gpa} / 4.0</strong>
                  </div>
                  <div className="small-muted" style={{ marginBottom: 6 }}>Technical Skills</div>
                  <div style={{ marginBottom: 12 }}>
                    {extracted.technical_skills.map((s) => <span className="chip" key={s}>{s}</span>)}
                  </div>
                  <div className="small-muted" style={{ marginBottom: 6 }}>Certifications</div>
                  <div style={{ marginBottom: 12 }}>
                    {extracted.certifications.map((s) => <span className="chip" key={s}>{s}</span>)}
                  </div>
                  <div className="small-muted" style={{ marginBottom: 6 }}>Projects</div>
                  <div style={{ marginBottom: 14 }}>
                    {extracted.projects.map((s) => <span className="chip" key={s}>{s}</span>)}
                  </div>
                </>
              ) : (
                <p className="small-muted">Upload a CV to see extracted skills, GPA, and certifications here.</p>
              )}
              <button className="btn btn-outline btn-block btn-sm" disabled={!extracted}>Edit extracted info</button>
            </div>

            <div className="panel">
              <div className="small-muted" style={{ marginBottom: 14 }}>Step 3 — Set Career Goal</div>
              <form id="career-goal-form" name="career-goal-form">
                <div className="field">
                  <label htmlFor="career_goal">Your selected goal</label>
                  <select id="career_goal" name="career_goal" required value={careerGoal} onChange={handleGoalChange}>
                    <option value="data_scientist">◆ Data Scientist</option>
                    <option value="ml_engineer">ML Engineer</option>
                    <option value="business_analyst">Business Analyst</option>
                    <option value="software_engineer">Software Engineer</option>
                    <option value="product_manager">Product Manager</option>
                  </select>
                </div>
              </form>
              <p className="small-muted text-center" style={{ margin: "0 0 12px" }}>Or let AI suggest best fit →</p>
              <button type="button" id="ai-suggest-career" className="btn btn-primary btn-block" onClick={handleAiSuggest}>
                AI Suggest Best Career
              </button>
            </div>
          </div>

          <div className="small-muted" style={{ marginBottom: 14 }}>
            Your Career Fit Rankings (AI — Top {rankings.length} of 50+ Paths · Model: all-mpnet-base-v2)
          </div>
          <table className="data-table" style={{ marginBottom: 32 }}>
            <thead>
              <tr><th>Career Path</th><th>Match Score</th><th>Key Skill Gap</th><th>Action</th></tr>
            </thead>
            <tbody>
              {rankings.map((r) => (
                <tr key={r.career_id}>
                  <td>{r.title}</td>
                  <td>
                    <span className="progress-track">
                      <span className={`progress-fill${r.match_score < 60 ? " warn" : ""}`} style={{ width: `${r.match_score}%` }} />
                    </span>
                    {r.match_score}%
                  </td>
                  <td>{r.skill_gap}</td>
                  <td>
                    <button className="btn btn-outline btn-sm" onClick={() => refreshRoadmap(r.career_id)}>View Roadmap</button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>

          {roadmap && (
            <>
              <div className="small-muted" style={{ marginBottom: 16 }}>
                6-Month Skill Roadmap — {roadmap.career_title}
              </div>
              <div className="timeline" style={{ marginBottom: 32 }}>
                {roadmap.steps.map((s) => (
                  <div className={`step${s.status === "completed" ? " is-complete" : s.status === "in_progress" ? " is-current" : ""}`} data-status={s.status} key={s.step}>
                    <div className="step-circle">{s.step}</div>
                    <p>{s.title}<br /><span className="small-muted">{s.subtitle}{s.status === "in_progress" ? " (Current)" : s.status === "completed" ? " ✓" : ""}</span></p>
                  </div>
                ))}
              </div>
            </>
          )}

          <div id="mock-interview" className="panel-tint" style={{ display: "block" }}>
            <div className="flex-between">
              <div>
                <h4 style={{ margin: "0 0 4px" }}>Mock Interview Simulation</h4>
                <p className="small-muted" style={{ margin: 0 }}>
                  Practice real interview questions for {rankings[0]?.title ?? "your top match"} roles · Powered by RAG + Interview Knowledge Base
                </p>
              </div>
              {!interviewId && (
                <button type="button" id="start-mock-interview" className="btn btn-primary" onClick={handleStartInterview} disabled={interviewBusy}>
                  {interviewBusy ? "Starting…" : "Start Mock Interview →"}
                </button>
              )}
            </div>

            {interviewQuestion && (
              <form onSubmit={handleAskInterview} style={{ marginTop: 16 }}>
                <p style={{ fontWeight: 600, marginBottom: 10 }}>{interviewQuestion}</p>
                <div className="field-row">
                  <input
                    type="text"
                    className="field mb-0"
                    style={{ flex: 1, padding: "10px 12px", borderRadius: 8, border: "1px solid var(--border-soft)" }}
                    placeholder="Type your answer…"
                    value={interviewAnswer}
                    onChange={(e) => setInterviewAnswer(e.target.value)}
                  />
                  <button type="submit" className="btn btn-primary" disabled={interviewBusy}>
                    {interviewBusy ? "…" : "Send"}
                  </button>
                </div>
              </form>
            )}
          </div>
        </div>
      </section>

      <SiteFooter />
    </>
  );
}
