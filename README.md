# DeepShield AI

AI-powered web platform to detect manipulated images and videos.

DeepShield AI provides:
- Secure user authentication (signup/login with JWT)
- Image deepfake detection
- Video deepfake detection via frame sampling
- Confidence-score based output for explainable decisions
- Admin workflows for user approvals and dataset operations

## Product Goals

- Image inference latency: `< 3s` (target)
- Video inference latency: `< 10s` (target)
- Initial model accuracy: `>= 85%`
- Target model accuracy: `>= 90%`

## System Architecture

```text
Frontend (React + Vite)
        |
        v
Backend API (FastAPI)
        |
        v
ML Services (TensorFlow + OpenCV + NumPy)
        |
        v
Storage (JSON now, DB-ready for PostgreSQL/MySQL)
```

## Monorepo Structure

```text
DeepShield-AI/
├── backend/
│   ├── .env.example
│   ├── .venv/
│   ├── Procfile
│   ├── requirements.txt
│   ├── app/
│   │   ├── main.py
│   │   ├── core/
│   │   │   ├── auth.py
│   │   │   └── config.py
│   │   ├── database/
│   │   ├── models/
│   │   │   └── user_store.py
│   │   ├── routes/
│   │   │   ├── auth.py
│   │   │   ├── predict.py
│   │   │   └── admin.py
│   │   ├── schemas/
│   │   │   ├── auth.py
│   │   │   └── prediction.py
│   │   ├── services/
│   │   │   ├── user_service.py
│   │   │   ├── model_service.py
│   │   │   └── video_service.py
│   │   └── utils/
│   │       ├── file_handler.py
│   │       └── preprocessing.py
│   ├── data/
│   │   └── users.json
│   └── models/
│
├── frontend/
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   ├── vercel.json
│   ├── public/
│   └── src/
│       ├── main.jsx
│       ├── App.jsx
│       ├── styles.css
│       ├── api/
│       │   ├── client.js
│       │   └── adminAuth.js
│       ├── components/
│       │   └── ProtectedRoute.jsx
│       ├── context/
│       │   └── AuthContext.jsx
│       ├── pages/
│       │   ├── HomePage.jsx
│       │   ├── LoginPage.jsx
│       │   ├── SignupPage.jsx
│       │   ├── UserDashboardPage.jsx
│       │   ├── ImageDetectionPage.jsx
│       │   ├── VideoDetectionPage.jsx
│       │   ├── AdminLoginPage.jsx
│       │   ├── AdminDashboardPage.jsx
│       │   ├── AdminAllUsersPage.jsx
│       │   ├── AdminPendingUsersPage.jsx
│       │   └── AdminUploadDatasetPage.jsx
│       ├── templates/
│       │   └── AuthTemplate.jsx
│       └── utils/
│           └── apiError.js
└── ml-model/
    └── (training pipeline in progress)
```

## Backend Overview (FastAPI)

### Core Responsibilities
- API surface for auth, predictions, and admin workflows
- User state management (currently file-based JSON)
- Model inference orchestration for image and video inputs
- Validation and error handling with Pydantic schemas

### Auth Design
- Password protection: SHA256 pre-hashing + bcrypt
- JWT-based access tokens for stateless sessions
- Supports unrestricted password length/character sets

### API Endpoints

#### Auth
- `POST /signup` - Register new user
- `POST /login` - Authenticate user (`userid` + `password`)

#### Prediction
- `POST /predict` - Detect deepfakes from image/video upload

#### Admin
- `GET /admin/*` and `POST /admin/*` - User approvals, user management, dataset operations

### Prediction Pipeline

#### Image
1. Receive uploaded image
2. Preprocess image tensor
3. Run TensorFlow inference
4. Return prediction + confidence score (`0.0 - 1.0`)

#### Video
1. Receive uploaded video (MP4)
2. Extract sampled frames (OpenCV)
3. Batch infer sampled frames
4. Aggregate frame predictions
5. Return final label + confidence score

## Frontend Overview (React + Vite)

### Core Responsibilities
- Route-based experience for public, user, and admin flows
- JWT token/session handling with protected routes
- Upload UX for image/video files
- Prediction result visualization with confidence scores

### Routing Model
- Public: `/`, `/login`, `/signup`
- User: `/app/*` (protected)
- Admin: `/admin-ui/*` (protected)
- Fallback: unknown routes redirect to home

### Auth Flow
1. User submits credentials
2. Frontend calls backend auth endpoint
3. JWT token returned on success
4. Token persisted (localStorage)
5. Protected routes unlocked via `AuthContext`

### Detection Flow
1. User selects file (image/video)
2. Frontend uploads file to `POST /predict`
3. Backend returns label + confidence score
4. UI displays result and confidence state

## API Contract (Example)

### Request: `POST /predict`
`multipart/form-data`
- `file`: binary image or video

### Success Response (example)
```json
{
  "success": true,
  "media_type": "image",
  "prediction": "deepfake",
  "confidence": 0.93,
  "processing_time_ms": 1240
}
```

### Error Response (example)
```json
{
  "success": false,
  "error": "Unsupported file format"
}
```

## Environment Variables

Create `backend/.env` from `.env.example`.

Common keys:
- `JWT_SECRET`
- `JWT_ALGORITHM=HS256`
- `ACCESS_TOKEN_EXPIRE_MINUTES=60`
- `CORS_ORIGINS=http://localhost:5173`
- `MODEL_PATH=./models/deepfake_model.h5`
- `USERS_FILE=./data/users.json`

## Local Development Setup

## 1) Backend

```bash
cd backend
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Backend API docs:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## 2) Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend app:
- `http://localhost:5173`

## 3) End-to-End Run

Run backend and frontend in separate terminals:
- Backend: `http://localhost:8000`
- Frontend: `http://localhost:5173`

Ensure frontend API base URL points to backend host/port.

## Demo Credentials (Development)

- User: `demo_user` / `Demo@1234`
- Admin: `admin_user` / `Admin@1234`

## Deployment

### Backend (Heroku or Docker)
- `Procfile` enables process startup for Heroku
- Docker deployment supported for containerized environments

### Frontend (Vercel or Static Hosting)
- `vercel.json` provides rewrite/build configuration
- Vite build output can be deployed to any static host

## Security Notes

- Use strong `JWT_SECRET` in production
- Move from JSON file storage to PostgreSQL/MySQL for production-scale deployments
- Add rate limiting and upload size caps at API gateway/backend layer
- Restrict CORS origins in production
- Store secrets in platform secret manager

## Performance Strategy

- Image path optimized for direct single-pass inference
- Video path uses frame sampling to bound compute time
- Batch frame inference reduces per-frame overhead
- Confidence thresholds can be tuned for precision/recall tradeoffs

## Testing Strategy

- Unit tests: auth, schemas, services
- API integration tests: `/signup`, `/login`, `/predict`
- Frontend tests: route guards, auth context, upload flows
- ML validation: confusion matrix, precision/recall/F1, ROC-AUC

## Roadmap

- In progress: training pipeline hardening (`ml-model/`)
- Planned: Grad-CAM explainability
- Planned: persistent relational DB + inference history
- Planned: model versioning + A/B evaluation
- Planned: observability (structured logs, tracing, latency dashboards)

## PRD-to-Implementation Mapping

| PRD Feature | Status | Implementation |
|---|---|---|
| User signup/login | Done | Auth routes + JWT |
| Image deepfake detection | Done | `/predict` + model service |
| Video deepfake detection | Done | Frame sampling + aggregation |
| Confidence score output | Done | Probability in JSON response |
| Performance targets | Done (MVP target) | Optimized inference paths |
| Explainability (Grad-CAM) | Planned | Future enhancement |

## License

Proprietary (update as needed).
#   d e e p f a k e _ v 1  
 