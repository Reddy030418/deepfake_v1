import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import client from "../api/client";
import AuthTemplate from "../templates/AuthTemplate";
import { getErrorMessage } from "../utils/apiError";

export default function SignupPage() {
  const [form, setForm] = useState({ username: "", email: "", password: "" });
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const onSubmit = async (e) => {
    e.preventDefault();
    setError("");
    setLoading(true);

    try {
      const response = await client.post("/signup", form);
      console.log("SIGNUP STATUS:", response.status);
      console.log("SIGNUP RESPONSE:", response.data);
      navigate("/login");
    } catch (err) {
      console.log("SIGNUP STATUS:", err?.response?.status);
      console.log("SIGNUP ERROR RESPONSE:", err?.response?.data);
      console.log("SIGNUP REQUEST PAYLOAD:", form);
      setError(getErrorMessage(err, "Signup failed"));
    } finally {
      setLoading(false);
    }
  };

  return (
    <AuthTemplate title="Create Your Account" subtitle="Join DeepShield AI and start secure media verification.">
      <form onSubmit={onSubmit} className="form">
        <input placeholder="Username" value={form.username} onChange={(e) => setForm({ ...form, username: e.target.value })} required />
        <input placeholder="Email" type="email" value={form.email} onChange={(e) => setForm({ ...form, email: e.target.value })} required />
        <input placeholder="Password" type="password" value={form.password} onChange={(e) => setForm({ ...form, password: e.target.value })} required />
        {error && <p className="error">{error}</p>}
        <button disabled={loading} className="btn">{loading ? "Creating..." : "Create Account"}</button>
      </form>
      <p className="muted">Already registered? <Link to="/login">Login</Link></p>
    </AuthTemplate>
  );
}
