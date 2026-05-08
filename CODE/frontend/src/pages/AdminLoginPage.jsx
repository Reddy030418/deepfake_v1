import { useState } from "react";
import { useNavigate } from "react-router-dom";
import client from "../api/client";
import { setAdminToken } from "../api/adminAuth";
import AuthTemplate from "../templates/AuthTemplate";
import { getErrorMessage } from "../utils/apiError";

export default function AdminLoginPage() {
  const [userid, setUserid] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const navigate = useNavigate();

  const onSubmit = async (e) => {
    e.preventDefault();
    setError("");
    try {
      const { data } = await client.post("/login", { userid, password });
      setAdminToken(data.access_token || "admin-token");
      navigate("/admin-ui/dashboard");
    } catch (err) {
      setError(getErrorMessage(err, "Admin login failed"));
    }
  };

  return (
    <AuthTemplate title="Admin Sign In" subtitle="Manage approvals, users, and datasets.">
      <form onSubmit={onSubmit} className="form">
        <input value={userid} onChange={(e) => setUserid(e.target.value)} placeholder="Admin User ID" required />
        <input value={password} onChange={(e) => setPassword(e.target.value)} type="password" placeholder="Password" required />
        {error && <p className="error">{error}</p>}
        <button className="btn">Access Admin Panel</button>
      </form>
      <div className="card" style={{ marginTop: "0.8rem" }}>
        <h3>Admin Credentials</h3>
        <p><strong>User ID:</strong> admin_user</p>
        <p><strong>Password:</strong> Admin@1234</p>
      </div>
    </AuthTemplate>
  );
}
