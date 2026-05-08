import { Suspense, lazy } from "react";
import { Navigate, Route, Routes } from "react-router-dom";
import ProtectedRoute from "./components/ProtectedRoute";

const HomePage = lazy(() => import("./pages/HomePage"));
const LoginPage = lazy(() => import("./pages/LoginPage"));
const SignupPage = lazy(() => import("./pages/SignupPage"));
const UserDashboardPage = lazy(() => import("./pages/UserDashboardPage"));
const ModelDashboardPage = lazy(() => import("./pages/ModelDashboardPage"));
const ImageDetectionPage = lazy(() => import("./pages/ImageDetectionPage"));
const VideoDetectionPage = lazy(() => import("./pages/VideoDetectionPage"));
const AdminLoginPage = lazy(() => import("./pages/AdminLoginPage"));
const AdminDashboardPage = lazy(() => import("./pages/AdminDashboardPage"));
const AdminAllUsersPage = lazy(() => import("./pages/AdminAllUsersPage"));
const AdminPendingUsersPage = lazy(() => import("./pages/AdminPendingUsersPage"));
const AdminUploadDatasetPage = lazy(() => import("./pages/AdminUploadDatasetPage"));

function LoadingScreen() {
  return (
    <main className="page">
      <section className="card wide">
        <h2>Loading UI...</h2>
        <p>Please wait while we load the page.</p>
      </section>
    </main>
  );
}

export default function App() {
  return (
    <Suspense fallback={<LoadingScreen />}>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/login" element={<LoginPage />} />
        <Route path="/signup" element={<SignupPage />} />
        <Route path="/admin-ui/login" element={<AdminLoginPage />} />

        <Route element={<ProtectedRoute />}>
          <Route path="/app/dashboard" element={<UserDashboardPage />} />
          <Route path="/app/model-dashboard" element={<ModelDashboardPage />} />
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
    </Suspense>
  );
}
