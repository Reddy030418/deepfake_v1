import { Link } from "react-router-dom";

const users = [
  { id: "demo_user", email: "demo@deepshield.ai", status: "approved" },
  { id: "analyst_01", email: "analyst@deepshield.ai", status: "approved" }
];

export default function AdminAllUsersPage() {
  return (
    <main className="page">
      <div className="topbar"><Link to="/admin-ui/dashboard">Back</Link></div>
      <section className="card wide">
        <h2>All Users</h2>
        <div className="table-wrap">
          <table>
            <thead><tr><th>User ID</th><th>Email</th><th>Status</th></tr></thead>
            <tbody>
              {users.map((u) => <tr key={u.id}><td>{u.id}</td><td>{u.email}</td><td>{u.status}</td></tr>)}
            </tbody>
          </table>
        </div>
      </section>
    </main>
  );
}
