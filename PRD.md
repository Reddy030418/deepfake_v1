# DeepShield AI - Product Requirements Document (PRD)

## 1. Project Overview

**Project name:** DeepShield AI

**Purpose:** Build a full-stack deepfake detection platform that allows users to submit images and videos for authenticity analysis, while providing admin oversight and an automated model training pipeline.

**Scope:**
- Image and video deepfake detection via a web application.
- User authentication and session management.
- Admin user listing and pending approval views.
- TensorFlow-based training pipeline with automatic zip dataset ingestion and model selection.

## 2. Vision

DeepShield AI provides an easy-to-use review system for detecting manipulated media. It should support both casual users and administrators with a responsive React frontend backed by a FastAPI service and a TensorFlow model inference pipeline.

## 3. Key Stakeholders

- Product reviewer / owner
- End users who need image/video deepfake checking
- Admins responsible for user management and model oversight
- ML engineers training and deploying detection models
- Developers maintaining frontend/backend integration

## 4. Goals

- Provide accurate real-time deepfake detection for image and video uploads.
- Make it easy to train and deploy the best-performing model from zip datasets.
- Present a clean, route-based React UI for users and admins.
- Ensure backend reliability with documented setup and configuration.

## 5. Existing System Summary

### Repository structure

- `backend/` - FastAPI service, user/auth flow, prediction endpoints, admin endpoints.
- `frontend/` - React + Vite single-page application.
- `ml-model/` - TensorFlow training and auto-training scripts.
- `data/` and `backend/data/users.json` - local JSON user store plus model/metric artifacts.

### Technologies

- Frontend: React, Vite, Axios, React Router
- Backend: Python, FastAPI, Pydantic, CORS middleware
- ML: TensorFlow, OpenCV, NumPy
- Deployment: local development via `uvicorn` and Vite dev server

## 6. User Personas

### Persona 1: General user

- Wants to upload an image or video, get a fast deepfake prediction, and view confidence results.
- Needs a simple login/signup experience.

### Persona 2: Admin reviewer

- Wants to review registered users and pending sign-ups.
- Needs access to model metrics and system health.

### Persona 3: ML operator

- Wants to train models using zip datasets and automatically deploy the best model to the backend.
- Needs clear metrics and selection criteria.

## 7. Functional Requirements

### 7.1 Authentication

- Users can sign up with `username`, `email`, and `password`.
- Users can log in with `userid` and `password`.
- Login returns JWT access token and role info.
- Frontend stores token state and protects routes.

### 7.2 Prediction

- Users can upload an image or video file to `/predict`.
- Backend detects media type by filename extension.
- Image prediction uses frame-level inference.
- Video prediction samples up to 24 frames and averages deepfake probability.
- Response contains `success`, `media_type`, `prediction`, `confidence`, and `processing_time_ms`.

### 7.3 Admin

- Admin endpoints exist at `/admin/users` and `/admin/pending`.
- API returns user list and pending approval list.
- Current implementation does not enforce role-based server-side protection yet.

### 7.4 Model Metrics

- Endpoints exist to fetch `/model-metrics` and `/model-comparison`.
- Metrics are loaded from backend model comparison and metrics files.

### 7.5 ML Training Pipeline

- Auto-training script reads ZIP datasets from `ml-model/data/_zips/`.
- Each zip is extracted and labeled by path keywords.
- The pipeline prepares train/val/test splits and trains models per dataset.
- Selected best model is copied to `backend/models/best_model.keras`.
- Selection rule: highest `auc`, then highest `accuracy`, then lowest `loss`.
- Output report written to `ml-model/outputs/auto/auto_train_report.json`.

## 8. Non-Functional Requirements

- Fast response for predictions: target under 2 seconds per image and under 10 seconds for video samples.
- Portable backend deployment via environment variables in `.env`.
- Model inference must support legacy TensorFlow model compatibility.
- Frontend must be responsive and route-driven via React Router.
- Data persistence for users must be stable enough for small-scale testing.

## 9. System Architecture

### 9.1 Frontend

- `frontend/src/App.jsx` defines public, user, and admin routes.
- `frontend/src/context/AuthContext.jsx` manages `ds_token` and `ds_user` in localStorage.
- `frontend/src/components/ProtectedRoute.jsx` guards route access.
- API client uses Axios bearer tokens.
- Pages include:
  - `HomePage.jsx`
  - `LoginPage.jsx`
  - `SignupPage.jsx`
  - `UserDashboardPage.jsx`
  - `ImageDetectionPage.jsx`
  - `VideoDetectionPage.jsx`
  - `AdminDashboardPage.jsx`
  - `AdminAllUsersPage.jsx`
  - `AdminPendingUsersPage.jsx`
  - `AdminUploadDatasetPage.jsx`

### 9.2 Backend

- `backend/app/main.py` initializes FastAPI, CORS, and routers.
- `backend/app/core/config.py` reads env variables for JWT, CORS, user file, model path, and threshold.
- `backend/app/core/auth.py` handles password hashing and JWT creation.
- `backend/app/routes/auth.py` exposes `/signup` and `/login`.
- `backend/app/routes/predict.py` exposes `/predict`, `/model-metrics`, and `/model-comparison`.
- `backend/app/routes/admin.py` exposes `/admin/users` and `/admin/pending`.
- `backend/app/services/model_service.py` loads the model lazily, resolves threshold, and performs image inference.
- `backend/app/services/video_service.py` extracts frames, samples uniformly up to 24 frames, and aggregates predictions.
- `backend/app/services/user_service.py` manages JSON-based user persistence.
- `backend/app/utils/file_handler.py` infers media type from filename extension.
- `backend/app/utils/preprocessing.py` currently provides a pass-through preprocessing placeholder.

### 9.3 ML Pipeline

- `ml-model/src/train.py` trains with TensorFlow transfer learning (MobileNetV2 by default) and supports warmup/fine-tune stages.
- `ml-model/src/auto_train_from_zips.py` automates dataset extraction, label mapping, training, comparison, and deployment.
- `ml-model/outputs/` stores training results and JSON reports.

## 10. Data Flow

1. User authenticates via frontend.
2. User uploads image/video file.
3. Frontend sends file to backend `/predict` using bearer token.
4. Backend infers media type and preprocesses payload.
5. Backend loads TensorFlow model and computes probability.
6. Label is determined via threshold and response returned.
7. Admin pages fetch user listings and metrics endpoints.
8. ML operator places ZIP files in `ml-model/data/_zips/`, runs auto-train, and backend model is updated.

## 11. Assumptions

- Users can be represented with a JSON store for this prototype.
- Admin role is tracked in user data but not fully enforced by backend routes.
- Model artifacts live under `backend/models/` and require TensorFlow to load.
- Media type detection is based on file extension only.
- Video inference is frame-sampling based rather than full-frame processing.

## 12. Current Limitations / Risks

- Backend admin routes are not protected by JWT or role checks.
- JSON-based user storage is not suitable for production or concurrent access.
- Email and password validation is minimal.
- File validation is extension-based and may accept invalid content.
- TensorFlow model compatibility depends on environment and installed version.
- Video inference uses a temporary file and may be slower on large uploads.

## 13. Acceptance Criteria

- [ ] Users can sign up and login successfully.
- [ ] Image upload returns a prediction, confidence, and timing.
- [ ] Video upload returns a prediction with sampled frame analysis.
- [ ] Admin endpoints return user and pending lists.
- [ ] ML auto-training successfully selects and deploys a best model.
- [ ] The frontend displays authenticated pages behind protected routes.
- [ ] The backend loads the model from `MODEL_PATH` and uses threshold from metrics or `.env`.

## 14. Deployment Notes

### Backend

- Install dependencies from `backend/requirements.txt`.
- Create `.env` with:
  - `JWT_SECRET`
  - `JWT_ALGORITHM=HS256`
  - `CORS_ORIGINS=http://localhost:5173`
  - `USERS_FILE=./data/users.json`
  - `MODEL_PATH=./models/best_model.keras`
  - `DEEPFAKE_THRESHOLD=0.5`
- Run:
  ```powershell
  uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
  ```

### Frontend

- Install dependencies from `frontend/package.json`.
- Run:
  ```powershell
  npm install
  npm run dev
  ```
- Frontend served at `http://localhost:5173`.

### ML Training

- Install dependencies from `ml-model/requirements.txt` in a separate Python environment.
- Run auto-training:
  ```powershell
  python ml-model\src\auto_train_from_zips.py
  ```
- Restart backend after model update.

## 15. Recommended Next Improvements

- Add JWT guard middleware and enforce admin role on `/admin/*`.
- Replace JSON user store with a relational database.
- Add stricter request validation and file MIME checks.
- Add end-to-end tests for auth, prediction, and admin APIs.
- Add user-facing error states and retry handling in the frontend.
- Add a model monitoring dashboard for deployed inference performance.

## 16. PDF Export Instructions

To convert this PRD into a PDF:

- Use a Markdown viewer with Print to PDF.
- Or install `pandoc` and run:
  ```powershell
  pandoc PRD.md -o PRD.pdf
  ```

- Alternatively, open `PRD.md` in VS Code and use `File > Print` or a markdown PDF extension.
