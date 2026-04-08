import { Link } from "react-router-dom";

export default function AdminDashboardPage() {
  return (
    <main className="page">
      <header className="topbar">
        <div className="brand">Admin Console</div>
        <Link to="/">Home</Link>
      </header>
      <section className="feature-grid">
        <Link to="/admin-ui/users" className="card card-link"><h3>All Users</h3><p>Inspect all registered users.</p></Link>
        <Link to="/admin-ui/pending" className="card card-link"><h3>Pending Approvals</h3><p>Approve or reject access requests.</p></Link>
        <Link to="/admin-ui/upload-dataset" className="card card-link"><h3>Upload Dataset</h3><p>Add training/validation assets.</p></Link>
      </section>
    </main>
  );
}
