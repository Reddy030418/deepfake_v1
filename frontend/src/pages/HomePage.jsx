import { Link } from "react-router-dom";

export default function HomePage() {
  return (
    <main className="page page-home">
      <header className="topbar">
        <div className="brand">DeepShield AI</div>
        <nav>
          <Link to="/login">Login</Link>
          <Link to="/signup" className="btn btn-small">Get Started</Link>
        </nav>
      </header>

      <section className="hero">
        <p className="eyebrow">Trust Every Pixel</p>
        <h1>Detect Deepfakes in Images and Videos in Seconds</h1>
        <p>
          Enterprise-grade AI verification with confidence scoring, admin governance,
          and an intuitive dashboard for rapid decisions.
        </p>
        <div className="hero-actions">
          <Link to="/signup" className="btn">Create Account</Link>
          <Link to="/login" className="btn btn-ghost">I already have an account</Link>
        </div>
      </section>

      <section className="feature-grid">
        <article className="card"><h3>Image Detection</h3><p>Single-pass inference with fast confidence output.</p></article>
        <article className="card"><h3>Video Sampling</h3><p>Frame-wise analysis for balanced speed and reliability.</p></article>
        <article className="card"><h3>Admin Console</h3><p>Manage users, pending approvals, and dataset operations.</p></article>
      </section>
    </main>
  );
}
