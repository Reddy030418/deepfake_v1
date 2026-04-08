import { Navigate, Route, Routes } from "react-router-dom";
import ProtectedRoute from "./components/ProtectedRoute";
import HomePage from "./pages/HomePage";
import LoginPage from "./pages/LoginPage";
import SignupPage from "./pages/SignupPage";
import UserDashboardPage from "./pages/UserDashboardPage";
import ImageDetectionPage from "./pages/ImageDetectionPage";
import VideoDetectionPage from "./pages/VideoDetectionPage";
import AdminLoginPage from "./pages/AdminLoginPage";
import AdminDashboardPage from "./pages/AdminDashboardPage";
import AdminAllUsersPage from "./pages/AdminAllUsersPage";
import AdminPendingUsersPage from "./pages/AdminPendingUsersPage";
import AdminUploadDatasetPage from "./pages/AdminUploadDatasetPage";

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<HomePage />} />
      <Route path="/login" element={<LoginPage />} />
      <Route path="/signup" element={<SignupPage />} />
      <Route path="/admin-ui/login" element={<AdminLoginPage />} />

      <Route element={<ProtectedRoute />}>
        <Route path="/app/dashboard" element={<UserDashboardPage />} />
        <Route path="/app/image-detection" element={<ImageDetectionPage />} />
        <Route path="/app/video-detection" element={<VideoDetectionPage />} />
      </Route>

      <Route element={<ProtectedRoute adminOnly />}>
        <Route path="/admin-ui/dashboard" element={<AdminDashboardPage />} />
        <Route path="/admin-ui/users" element={<AdminAllUsersPage />} />
        <Route path="/admin-ui/pending" element={<AdminPendingUsersPage />} />
        <Route path="/admin-ui/upload-dataset" element={<AdminUploadDatasetPage />} />
      </Route>

      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}
