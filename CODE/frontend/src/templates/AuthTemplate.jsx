import { Link } from "react-router-dom";

export default function AuthTemplate({ title, subtitle, children }) {
  return (
    <div className="auth-shell">
      <section className="auth-card">
        <div className="chip">DeepShield AI</div>
        <h1>{title}</h1>
        <p>{subtitle}</p>
        {children}
        <div className="auth-links">
          <Link to="/">Home</Link>
          <Link to="/admin-ui/login">Admin</Link>
        </div>
      </section>
    </div>
  );
}
