import { Link } from "react-router-dom";

const pending = [
  { id: "new_researcher", email: "new@deepshield.ai" },
  { id: "media_audit", email: "audit@deepshield.ai" }
];

export default function AdminPendingUsersPage() {
  return (
    <main className="page">
      <div className="topbar"><Link to="/admin-ui/dashboard">Back</Link></div>
      <section className="card wide">
        <h2>Pending User Approvals</h2>
        <div className="list">
          {pending.map((u) => (
            <article key={u.id} className="list-item">
              <div><strong>{u.id}</strong><p>{u.email}</p></div>
              <div className="actions"><button className="btn btn-small">Approve</button><button className="btn btn-ghost btn-small">Reject</button></div>
            </article>
          ))}
        </div>
      </section>
    </main>
  );
}
