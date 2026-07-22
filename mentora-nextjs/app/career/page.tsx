import SiteHeader from "@/components/SiteHeader";
import SiteFooter from "@/components/SiteFooter";

export default function CareerPage() {
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
          <div className="grid-3" style={{ marginBottom: 28 }}>
            <div className="panel">
              <div className="small-muted" style={{ marginBottom: 14 }}>
                Step 1 — Upload CV / Transcript
              </div>
              <form id="cv-upload-form" name="cv-upload-form" method="post" encType="multipart/form-data">
                <label
                  htmlFor="cv_file"
                  style={{
                    display: "block",
                    border: "1px dashed var(--border-soft)",
                    borderRadius: 8,
                    padding: 24,
                    textAlign: "center",
                    cursor: "pointer",
                  }}
                >
                  <p style={{ fontSize: "0.85rem", color: "var(--ink-500)", margin: "0 0 14px" }}>
                    Drag &amp; drop your CV or Transcript
                    <br />
                    Supports PDF, DOCX formats · NER model extracts your data
                  </p>
                  <span className="btn btn-primary btn-sm">Browse File</span>
                  <input
                    type="file"
                    id="cv_file"
                    name="cv_file"
                    accept=".pdf,.docx"
                    style={{ position: "absolute", width: 1, height: 1, opacity: 0, overflow: "hidden" }}
                  />
                </label>
              </form>
            </div>

            <div className="panel">
              <div className="small-muted" style={{ marginBottom: 14 }}>
                Step 2 — Skills Extracted (AI)
              </div>
              <div className="flex-between" style={{ marginBottom: 12 }}>
                <span className="small-muted">GPA</span>
                <strong>3.6 / 4.0</strong>
              </div>
              <div className="small-muted" style={{ marginBottom: 6 }}>
                Technical Skills
              </div>
              <div style={{ marginBottom: 12 }}>
                <span className="chip">Python</span>
                <span className="chip">Data Analysis</span>
                <span className="chip">SQL</span>
              </div>
              <div className="small-muted" style={{ marginBottom: 6 }}>
                Certifications
              </div>
              <div style={{ marginBottom: 12 }}>
                <span className="chip">AWS Cloud Found.</span>
              </div>
              <div className="small-muted" style={{ marginBottom: 6 }}>
                Projects
              </div>
              <div style={{ marginBottom: 14 }}>
                <span className="chip">Sentiment Analyzer</span>
              </div>
              <button className="btn btn-outline btn-block btn-sm">Edit extracted info</button>
            </div>

            <div className="panel">
              <div className="small-muted" style={{ marginBottom: 14 }}>
                Step 3 — Set Career Goal
              </div>
              <form id="career-goal-form" name="career-goal-form" method="post">
                <div className="field">
                  <label htmlFor="career_goal">Your selected goal</label>
                  <select id="career_goal" name="career_goal" required defaultValue="data_scientist">
                    <option value="data_scientist">◆ Data Scientist</option>
                    <option value="ml_engineer">ML Engineer</option>
                    <option value="business_analyst">Business Analyst</option>
                    <option value="software_engineer">Software Engineer</option>
                    <option value="product_manager">Product Manager</option>
                  </select>
                </div>
              </form>
              <p className="small-muted text-center" style={{ margin: "0 0 12px" }}>
                Or let AI suggest best fit →
              </p>
              <button type="button" id="ai-suggest-career" className="btn btn-primary btn-block">
                AI Suggest Best Career
              </button>
            </div>
          </div>

          <div className="small-muted" style={{ marginBottom: 14 }}>
            Your Career Fit Rankings (AI — Top 5 of 50+ Paths · Model: all-mpnet-base-v2)
          </div>
          <table className="data-table" style={{ marginBottom: 32 }}>
            <thead>
              <tr>
                <th>Career Path</th>
                <th>Match Score</th>
                <th>Key Skill Gap</th>
                <th>Action</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>Data Scientist</td>
                <td>
                  <span className="progress-track">
                    <span className="progress-fill" style={{ width: "88%" }} />
                  </span>
                  88%
                </td>
                <td>Deep Learning, TensorFlow</td>
                <td>
                  <button className="btn btn-outline btn-sm">View Roadmap</button>
                </td>
              </tr>
              <tr>
                <td>ML Engineer</td>
                <td>
                  <span className="progress-track">
                    <span className="progress-fill" style={{ width: "78%" }} />
                  </span>
                  78%
                </td>
                <td>TensorFlow, MLOps</td>
                <td>
                  <button className="btn btn-outline btn-sm">View Roadmap</button>
                </td>
              </tr>
              <tr>
                <td>Business Analyst</td>
                <td>
                  <span className="progress-track">
                    <span className="progress-fill warn" style={{ width: "66%" }} />
                  </span>
                  66%
                </td>
                <td>SQL, Tableau, Excel</td>
                <td>
                  <button className="btn btn-outline btn-sm">View Roadmap</button>
                </td>
              </tr>
              <tr>
                <td>Software Engineer</td>
                <td>
                  <span className="progress-track">
                    <span className="progress-fill warn" style={{ width: "58%" }} />
                  </span>
                  58%
                </td>
                <td>System Design, APIs</td>
                <td>
                  <button className="btn btn-outline btn-sm">View Roadmap</button>
                </td>
              </tr>
              <tr>
                <td>Product Manager</td>
                <td>
                  <span className="progress-track">
                    <span className="progress-fill low" style={{ width: "42%" }} />
                  </span>
                  42%
                </td>
                <td>Leadership, Strategy, UX</td>
                <td>
                  <button className="btn btn-outline btn-sm">View Roadmap</button>
                </td>
              </tr>
            </tbody>
          </table>

          <div className="small-muted" style={{ marginBottom: 16 }}>
            6-Month Skill Roadmap — Data Scientist
          </div>
          <div className="timeline" style={{ marginBottom: 32 }}>
            <div className="step is-complete" data-status="completed">
              <div className="step-circle">1–2</div>
              <p>
                Python &amp; Stats
                <br />
                <span className="small-muted">Fundamentals ✓</span>
              </p>
            </div>
            <div className="step is-complete" data-status="completed">
              <div className="step-circle">3</div>
              <p>
                ML Algorithms
                <br />
                <span className="small-muted">Scikit-learn ✓</span>
              </p>
            </div>
            <div className="step is-current" data-status="in_progress">
              <div className="step-circle">4</div>
              <p>
                Deep Learning
                <br />
                <span className="small-muted">TensorFlow (Current)</span>
              </p>
            </div>
            <div className="step" data-status="pending">
              <div className="step-circle">5</div>
              <p>
                Projects &amp; RAG
                <br />
                <span className="small-muted">Portfolio</span>
              </p>
            </div>
            <div className="step" data-status="pending">
              <div className="step-circle">6</div>
              <p>
                Mock Interviews
                <br />
                <span className="small-muted">Job Applications</span>
              </p>
            </div>
          </div>

          <div id="mock-interview" className="panel-tint flex-between">
            <div>
              <h4 style={{ margin: "0 0 4px" }}>Mock Interview Simulation</h4>
              <p className="small-muted" style={{ margin: 0 }}>
                Practice real interview questions for Data Scientist roles · Powered by RAG + Interview Knowledge
                Base
              </p>
            </div>
            <button type="button" id="start-mock-interview" className="btn btn-primary">
              Start Mock Interview →
            </button>
          </div>
        </div>
      </section>

      <SiteFooter />
    </>
  );
}
