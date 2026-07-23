"use client";

import { useEffect, useState } from "react";
import SiteHeader from "@/components/SiteHeader";
import SiteFooter from "@/components/SiteFooter";
import { apiFetch, ApiError } from "@/lib/api";
import type { Timetable, TodayTopic, MCQQuestion, MCQSubmitResult } from "@/lib/types";

type Subject = "mathematics" | "chemistry" | "urdu";

const SUBJECTS: { id: Subject; label: string }[] = [
  { id: "mathematics", label: "Mathematics" },
  { id: "chemistry", label: "Chemistry" },
  { id: "urdu", label: "Urdu" },
];

const STATUS_CLASS: Record<string, string> = {
  pending: "status-pending",
  done: "status-done",
  today: "status-today",
  missed: "status-missed",
};
const STATUS_LABEL: Record<string, string> = {
  pending: "Pending",
  done: "Done ✓",
  today: "Today",
  missed: "Missed",
};

export default function ExamPrepPage() {
  const [selectedSubjects, setSelectedSubjects] = useState<Subject[]>(["mathematics"]);
  const [timetable, setTimetable] = useState<Timetable | null>(null);
  const [todayTopic, setTodayTopic] = useState<TodayTopic | null>(null);
  const [mcq, setMcq] = useState<MCQQuestion | null>(null);
  const [mcqResult, setMcqResult] = useState<MCQSubmitResult | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    (async () => {
      try {
        const [tt, topic, question] = await Promise.all([
          apiFetch<Timetable>("/exam/timetable"),
          apiFetch<TodayTopic>("/exam/today-topic"),
          apiFetch<MCQQuestion>("/exam/mcq"),
        ]);
        setTimetable(tt);
        setTodayTopic(topic);
        setMcq(question);
      } catch (err) {
        setLoadError(err instanceof ApiError ? err.message : "Couldn't load your exam prep data right now.");
      }
    })();
  }, []);

  async function toggleSubject(id: Subject) {
    const next = selectedSubjects.includes(id)
      ? selectedSubjects.filter((s) => s !== id)
      : [...selectedSubjects, id];
    setSelectedSubjects(next);
    try {
      await apiFetch("/exam/subjects", { method: "PATCH", body: JSON.stringify({ subjects: next }) });
    } catch {
      /* non-critical — chip state already updated optimistically */
    }
  }

  async function handleMcqSubmit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    if (!mcq) return;
    const formData = new FormData(e.currentTarget);
    const answer = formData.get(`answer_${mcq.question_id}`) as string;
    if (!answer) return;

    setSubmitting(true);
    try {
      const result = await apiFetch<MCQSubmitResult>("/exam/mcq/submit", {
        method: "POST",
        body: JSON.stringify({ question_id: mcq.question_id, answer }),
      });
      setMcqResult(result);
    } catch (err) {
      setLoadError(err instanceof ApiError ? err.message : "Couldn't submit your answer right now.");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <>
      <SiteHeader variant="app" />

      <section className="page-heading">
        <div className="container">
          <div>
            <h1>Sindh University Entry Test Prep</h1>
            <p>AI-generated timetable · Auto-updated daily</p>
          </div>
          <button className="btn btn-primary">Upload Content</button>
        </div>
      </section>

      <section className="section">
        <div className="container">
          {loadError && (
            <div className="demo-note" style={{ borderColor: "#c0392b", marginBottom: 20 }}>{loadError}</div>
          )}

          <div className="grid-2" style={{ gridTemplateColumns: "200px 1fr", marginBottom: 28 }}>
            <div className="stat-block">
              <div className="stat-num">{timetable?.days_to_exam ?? "—"}</div>
              <div className="stat-label">Days to Exam · Auto-updated daily</div>
            </div>
            <div className="panel-tint">
              <div className="small-muted" style={{ marginBottom: 10 }}>Select subjects &amp; test date</div>
              <div id="subject-selector" role="group" aria-label="Select subjects">
                {SUBJECTS.map((subject) => {
                  const isSelected = selectedSubjects.includes(subject.id);
                  return (
                    <button
                      key={subject.id}
                      type="button"
                      className={`chip${isSelected ? " selected" : ""}`}
                      data-subject={subject.id}
                      aria-pressed={isSelected}
                      onClick={() => toggleSubject(subject.id)}
                    >
                      {isSelected ? "✓ " : "+ "}{subject.label}
                    </button>
                  );
                })}
              </div>
              <div className="field-row mt-24" style={{ marginTop: 16 }}>
                <div className="field mb-0">
                  <label htmlFor="test_date">Test date</label>
                  <input type="date" id="test_date" name="test_date" />
                </div>
                <div className="field mb-0">
                  <label htmlFor="study_mode">Study mode</label>
                  <select id="study_mode" name="study_mode" defaultValue="">
                    <option value="" disabled>Select study mode</option>
                    <option value="self_study">Self-study</option>
                    <option value="tutor">With Tutor</option>
                  </select>
                </div>
              </div>
            </div>
          </div>

          <div className="flex-between" style={{ marginBottom: 14 }}>
            <h3 style={{ fontFamily: "var(--font-display)", margin: 0 }}>AI-Generated Study Timetable</h3>
            <a href="#" className="link-inline">Regenerate</a>
          </div>

          <table className="data-table">
            <thead>
              <tr><th>Day</th><th>Topic</th><th>Subject</th><th>Duration</th><th>Status</th></tr>
            </thead>
            <tbody>
              {(timetable?.rows ?? []).map((row) => (
                <tr key={row.day}>
                  <td>{row.day}</td>
                  <td>{row.topic}</td>
                  <td>{row.subject}</td>
                  <td>{row.duration}</td>
                  <td><span className={`status-pill ${STATUS_CLASS[row.status]}`}>{STATUS_LABEL[row.status]}</span></td>
                </tr>
              ))}
            </tbody>
          </table>

          {todayTopic && (
            <>
              <h3 style={{ fontFamily: "var(--font-display)", margin: "28px 0 14px" }}>
                Today&apos;s Topic: {todayTopic.topic}
              </h3>
              <div className="grid-3">
                <div className="panel">
                  <div className="small-muted" style={{ marginBottom: 8 }}>AI Topic Explainer</div>
                  <p style={{ fontSize: "0.88rem", color: "var(--ink-700)" }}>{todayTopic.explainer}</p>
                  <ul style={{ margin: "0 0 16px", paddingLeft: 18, listStyle: "disc", fontSize: "0.85rem", color: "var(--ink-700)" }}>
                    {todayTopic.key_points.map((k) => <li key={k}>{k}</li>)}
                  </ul>
                  <a href="#mcq-form" className="btn btn-primary btn-block">Take MCQ Test →</a>
                </div>
                {todayTopic.videos.map((v) => (
                  <div className="panel" key={v.title}>
                    <div className="small-muted" style={{ marginBottom: 8 }}>YouTube Video</div>
                    <h4 style={{ margin: "0 0 4px" }}>{v.title}</h4>
                    <p className="small-muted" style={{ margin: "0 0 14px" }}>{v.source} · {v.duration_min} min · AI-matched</p>
                    <button className="btn btn-outline btn-block">+ Add to Playlist</button>
                  </div>
                ))}
              </div>
            </>
          )}

          {mcq && (
            <form id="mcq-form" name="mcq-form" className="panel mt-24" onSubmit={handleMcqSubmit}>
              <h4 style={{ margin: "0 0 14px" }}>MCQ Practice — {todayTopic?.topic ?? ""} (AI-Generated)</h4>
              <p style={{ fontSize: "0.92rem", margin: "0 0 14px" }} data-question-id={mcq.question_id}>{mcq.question}</p>
              {mcq.options.map((opt) => (
                <label className="mcq-option" key={opt.value}>
                  <input type="radio" name={`answer_${mcq.question_id}`} value={opt.value} required={opt.value === mcq.options[0].value} />
                  {" "}{opt.label}
                </label>
              ))}
              {mcqResult && (
                <p className="small-muted" style={{ marginTop: 12 }}>
                  {mcqResult.correct ? "✅ Correct!" : `❌ Incorrect — the correct answer was ${mcqResult.correct_answer}.`}
                  {mcqResult.weak_topics_flagged.length > 0 && ` Flagged as weak: ${mcqResult.weak_topics_flagged.join(", ")}.`}
                </p>
              )}
              <div className="flex-between mt-24">
                <button type="submit" className="btn btn-primary" disabled={submitting}>
                  {submitting ? "Submitting…" : "Submit →"}
                </button>
                <span className="small-muted">
                  {mcq.total_in_set} MCQs total · {mcq.remaining_in_set} remaining · Generated by {mcq.generated_by} model
                </span>
              </div>
            </form>
          )}
        </div>
      </section>

      <SiteFooter />
    </>
  );
}
