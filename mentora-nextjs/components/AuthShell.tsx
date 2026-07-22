import Link from "next/link";
import { ReactNode } from "react";

export default function AuthShell({
  sideHeading,
  sideCopy,
  sideChecklist,
  children,
}: {
  sideHeading: string;
  sideCopy: string;
  sideChecklist: string[];
  children: ReactNode;
}) {
  return (
    <div className="auth-shell">
      <aside className="auth-side">
        <div className="brand">MENTORA</div>
        <div className="side-nav">
          <Link href="/">Home</Link>
          <Link href="/exam-prep">Exam Prep</Link>
          <Link href="/career">Career</Link>
        </div>
        <div className="side-content">
          <h2>{sideHeading}</h2>
          <p>{sideCopy}</p>
          <ul className="check-list">
            {sideChecklist.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        </div>
        <p className="footer-copy">© 2025 Mentora · A Smart Platform, From Classroom to Career</p>
      </aside>

      <main className="auth-main">
        <div className="auth-form-wrap">{children}</div>
      </main>
    </div>
  );
}
