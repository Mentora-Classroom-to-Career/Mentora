import Link from "next/link";

export default function SiteFooter() {
  return (
    <footer className="site-footer">
      <div className="container">
        <span className="brand">MENTORA</span>
        <nav className="footer-links">
          <Link href="/">Home</Link>
          <Link href="/exam-prep">Exam Prep</Link>
          <Link href="/career">Career</Link>
          <Link href="/learning-portal">Learning Portal</Link>
          <Link href="/dashboard">Dashboard</Link>
        </nav>
        <p className="footer-copy">© 2025 Mentora · A Smart Platform, From Classroom to Career</p>
        <div className="footer-badge">🎓</div>
      </div>
    </footer>
  );
}
