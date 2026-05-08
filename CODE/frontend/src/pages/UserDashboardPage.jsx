import { Link, useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";

export default function UserDashboardPage() {
  const { user, logout } = useAuth();
  const navigate = useNavigate();

  const handleLogout = () => {
    logout();
    navigate("/login");
  };

  return (
    <main className="page">
      <header className="topbar">
        <div className="brand">DeepShield AI</div>
        <button className="btn btn-small" onClick={handleLogout}>Logout</button>
      </header>

      <section className="hero compact">
        <p className="eyebrow">Dashboard</p>
        <h1>Hello, {user?.userid || user?.username || "User"}</h1>
        <p>Choose a detection workflow or open the model analytics dashboard.</p>
      </section>

      <section className="feature-grid">
        <Link to="/app/image-detection" className="card card-link"><h3>Image Detection</h3><p>Upload JPG/PNG and get confidence score.</p></Link>
        <Link to="/app/video-detection" className="card card-link"><h3>Video Detection</h3><p>Upload MP4 and run frame-sampling detection.</p></Link>
        <Link to="/app/model-dashboard" className="card card-link"><h3>Model Dashboard</h3><p>View F1, ROC curve, confusion matrix and accuracy charts.</p></Link>
      </section>
    </main>
  );
}
