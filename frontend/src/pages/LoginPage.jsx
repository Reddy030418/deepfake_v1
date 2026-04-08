import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import client from "../api/client";
import AuthTemplate from "../templates/AuthTemplate";
import { useAuth } from "../context/AuthContext";
import { getErrorMessage } from "../utils/apiError";

export default function LoginPage() {
  const [userid, setUserid] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();
  const { login } = useAuth();

  const onSubmit = async (e) => {
    e.preventDefault();
    setError("");
    setLoading(true);
    try {
      const { data } = await client.post("/login", { userid, password });
      login(data.access_token || "demo-token", data.user || { userid });
      navigate("/app/dashboard");
    } catch (err) {
      setError(getErrorMessage(err, "Login failed"));
    } finally {
      setLoading(false);
    }
  };

  return (
    <AuthTemplate title="Welcome Back" subtitle="Sign in to continue deepfake detection.">
      <form onSubmit={onSubmit} className="form">
        <input value={userid} onChange={(e) => setUserid(e.target.value)} placeholder="User ID" required />
        <input value={password} onChange={(e) => setPassword(e.target.value)} type="password" placeholder="Password" required />
        {error && <p className="error">{error}</p>}
        <button disabled={loading} className="btn">{loading ? "Signing in..." : "Login"}</button>
      </form>
      <div className="card" style={{ marginTop: "0.8rem" }}>
        <h3>Demo Credentials</h3>
        <p><strong>User ID:</strong> demo_user</p>
        <p><strong>Password:</strong> Demo@1234</p>
      </div>
      <p className="muted">No account? <Link to="/signup">Create one</Link></p>
    </AuthTemplate>
  );
}
