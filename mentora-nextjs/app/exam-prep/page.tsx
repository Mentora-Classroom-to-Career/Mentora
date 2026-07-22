"use client";

import { useState } from "react";
import SiteHeader from "@/components/SiteHeader";
import SiteFooter from "@/components/SiteFooter";

type Subject = "mathematics" | "chemistry" | "urdu";

const SUBJECTS: { id: Subject; label: string }[] = [
  { id: "mathematics", label: "Mathematics" },
  { id: "chemistry", label: "Chemistry" },
  { id: "urdu", label: "Urdu" },
];

export default function ExamPrepPage() {
  const [selectedSubjects, setSelectedSubjects] = useState<Subject[]>(["mathematics"]);

  function toggleSubject(id: Subject) {
    setSelectedSubjects((prev) => (prev.includes(id) ? prev.filter((s) => s !== id) : [...prev, id]));
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
          <div className="grid-2" style={{ gridTemplateColumns: "200px 1fr", marginBottom: 28 }}>
            <div className="stat-block">
              <div className="stat-num">47</div>
              <div className="stat-label">Days to Exam · Auto-updated daily</div>
            </div>
            <div className="panel-tint">
              <div className="small-muted" style={{ marginBottom: 10 }}>
                Select subjects &amp; test date
              </div>
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
                      {isSelected ? "✓ " : "+ "}
                      {subject.label}
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
                    <option value="" disabled>
                      Select study mode
                    </option>
                    <option value="self_study">Self-study</option>
                    <option value="tutor">With Tutor</option>
                  </select>
                </div>
              </div>
            </div>
          </div>

          <div className="flex-between" style={{ marginBottom: 14 }}>
            <h3 style={{ fontFamily: "var(--font-display)", margin: 0 }}>AI-Generated Study Timetable</h3>
            <a href="#" className="link-inline">
              Regenerate
            </a>
          </div>

          <table className="data-table">
            <thead>
              <tr>
                <th>Day</th>
                <th>Topic</th>
                <th>Subject</th>
                <th>Duration</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>Monday</td>
                <td>Algebra — Quadratic Equations</td>
                <td>Mathematics</td>
                <td>2 hrs</td>
                <td>
                  <span className="status-pill status-pending">Pending</span>
                </td>
              </tr>
              <tr>
                <td>Tuesday</td>
                <td>Reading Comprehension</td>
                <td>English</td>
                <td>1.5 hrs</td>
                <td>
                  <span className="status-pill status-done">Done ✓</span>
                </td>
              </tr>
              <tr>
                <td>Wednesday</td>
                <td>Grammar — Tenses &amp; Voice</td>
                <td>English</td>
                <td>2 hrs</td>
                <td>
                  <span className="status-pill status-today">Today</span>
                </td>
              </tr>
              <tr>
                <td>Thursday</td>
                <td>Geometry — Lines &amp; Angles</td>
                <td>Mathematics</td>
                <td>2 hrs</td>
                <td>
                  <span className="status-pill status-missed">Missed</span>
                </td>
              </tr>
              <tr>
                <td>Friday</td>
                <td>Revision + MCQ Test</td>
                <td>Mixed</td>
                <td>1 hr</td>
                <td>
                  <span className="status-pill status-pending">Pending</span>
                </td>
              </tr>
            </tbody>
          </table>

          <h3 style={{ fontFamily: "var(--font-display)", margin: "28px 0 14px" }}>
            Today&apos;s Topic: Quadratic Equations
          </h3>
          <div className="grid-3">
            <div className="panel">
              <div className="small-muted" style={{ marginBottom: 8 }}>
                AI Topic Explainer
              </div>
              <p style={{ fontSize: "0.88rem", color: "var(--ink-700)" }}>
                A quadratic equation has the form ax² + bx + c = 0. Key methods:
              </p>
              <ul
                style={{
                  margin: "0 0 16px",
                  paddingLeft: 18,
                  listStyle: "disc",
                  fontSize: "0.85rem",
                  color: "var(--ink-700)",
                }}
              >
                <li>Factorization method</li>
                <li>Quadratic formula</li>
                <li>Completing the square</li>
              </ul>
              <a href="#" className="btn btn-primary btn-block">
                Take MCQ Test →
              </a>
            </div>
            <div className="panel">
              <div className="small-muted" style={{ marginBottom: 8 }}>
                YouTube Video
              </div>
              <h4 style={{ margin: "0 0 4px" }}>Quadratic Equations — Full Chapter</h4>
              <p className="small-muted" style={{ margin: "0 0 14px" }}>
                MathCity.org · 18 min · AI-matched
              </p>
              <button className="btn btn-outline btn-block">+ Add to Playlist</button>
            </div>
            <div className="panel">
              <div className="small-muted" style={{ marginBottom: 8 }}>
                YouTube Video
              </div>
              <h4 style={{ margin: "0 0 4px" }}>Quadratic Formula — Step by Step</h4>
              <p className="small-muted" style={{ margin: "0 0 14px" }}>
                Khan Academy · 12 min · AI-matched
              </p>
              <button className="btn btn-outline btn-block">+ Add to Playlist</button>
            </div>
          </div>

          <form id="mcq-form" name="mcq-form" className="panel mt-24" method="post">
            <h4 style={{ margin: "0 0 14px" }}>MCQ Practice — Quadratic Equations (AI-Generated)</h4>
            <p style={{ fontSize: "0.92rem", margin: "0 0 14px" }} data-question-id="Q1001">
              Q1. Solve x² − 5x + 6 = 0. What are the roots?
            </p>
            <label className="mcq-option">
              <input type="radio" name="answer_Q1001" value="A" required /> x = 2, 3
            </label>
            <label className="mcq-option">
              <input type="radio" name="answer_Q1001" value="B" /> x = −2, 3
            </label>
            <label className="mcq-option">
              <input type="radio" name="answer_Q1001" value="C" /> x = 1, 6
            </label>
            <label className="mcq-option">
              <input type="radio" name="answer_Q1001" value="D" /> x = −1, −6
            </label>
            <div className="flex-between mt-24">
              <button type="submit" className="btn btn-primary">
                Submit →
              </button>
              <span className="small-muted">5 MCQs total · 4 remaining · Generated by FLAN-T5 model</span>
            </div>
          </form>
        </div>
      </section>

      <SiteFooter />
    </>
  );
}
