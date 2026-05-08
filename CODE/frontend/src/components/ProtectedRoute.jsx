import { Navigate, Outlet } from "react-router-dom";
import { useAuth } from "../context/AuthContext";

export default function ProtectedRoute({ adminOnly = false }) {
  const { isAuthenticated } = useAuth();
  const hasAdmin = Boolean(localStorage.getItem("ds_admin_token"));

  if (adminOnly) {
    return hasAdmin ? <Outlet /> : <Navigate to="/admin-ui/login" replace />;
  }

  return isAuthenticated ? <Outlet /> : <Navigate to="/login" replace />;
}
