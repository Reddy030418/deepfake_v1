# DeepShield AI

DeepShield AI is a full-stack deepfake detection project:
- `frontend/`: React + Vite UI
- `backend/`: FastAPI APIs for auth, prediction, and metrics
- `ml-model/`: TensorFlow training + chart/image generation scripts

## Project Structure

```text
C:\FINAL
  backend/
  frontend/
  ml-model/
  PRD.md
  Q&A.md
```

## 1) Backend Setup (FastAPI)

```powershell
cd C:\FINAL\backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create `backend/.env` (if missing):

```env
JWT_SECRET=change-this-secret
JWT_ALGORITHM=HS256
CORS_ORIGINS=http://localhost:5173
USERS_FILE=./data/users.json
MODEL_PATH=./models/best_model.keras
MODEL_METRICS_PATH=./models/metrics.json
MODEL_COMPARISON_PATH=./models/model_comparison.json
DEEPFAKE_THRESHOLD=0.5
DEMO_DASHBOARD_MODE=false
```

Run backend:

```powershell
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Useful URLs:
- API docs: `http://localhost:8000/docs`
- Health: `http://localhost:8000/health`

## 2) Frontend Setup (React + Vite)

```powershell
cd C:\FINAL\frontend
npm install
npm run dev -- --host 0.0.0.0 --port 5173
```

Frontend URL:
- `http://localhost:5173`

## 3) ML Environment Setup

```powershell
cd C:\FINAL
python -m venv .venv-ml
.\.venv-ml\Scripts\Activate.ps1
pip install -r ml-model\requirements.txt
```

## 4) Train a Model (Basic)

Expected dataset layout:

```text
ml-model/data/
  train/
    authentic/
    deepfake/
  val/
    authentic/
    deepfake/
```

Train:

```powershell
python ml-model\src\train.py --data-dir ml-model\data --output-dir ml-model\outputs\run1 --epochs 10 --batch-size 32 --img-size 224
```

## 5) Auto Train from Zip Files

Put dataset zips in:

```text
ml-model/data/_zips/
```

Run:

```powershell
python ml-model\src\auto_train_from_zips.py
```

This pipeline:
- Extracts zips
- Auto-maps labels (`authentic` / `deepfake`)
- Creates train/val/test splits
- Trains and compares runs
- Deploys winner to `backend/models/best_model.keras`
- Writes `ml-model/outputs/auto/auto_train_report.json`

## 6) 3-Model Comparison Suite

Run:

```powershell
python ml-model\src\train_model_suite.py --source-raw-dir ml-model\data\_raw --prepared-dir ml-model\data\suite_subset --max-per-class 3000 --split-strategy group --epochs 24 --fine-tune-epochs 8 --batch-size 16 --backbones efficientnetb0,resnet50,xception --threshold-metric accuracy --monitor-metric val_auc
```

Outputs:
- `backend/models/best_model.keras`
- `backend/models/metrics.json`
- `backend/models/model_comparison.json`

## 7) Generate Review Charts (Image Models)

Generate full image metrics chart set:

```powershell
python ml-model\src\generate_review_images.py
```

Key outputs in `ml-model/outputs/review_images/`:
- `confusion_matrix.png`
- `f1_score.png`
- `precision_score.png`
- `recall_score.png`
- `accuracy_score.png`
- `accuracy_comparison_image_models.png`
- `roc_curve_resnet50.png`

## 8) Generate Review Charts (Video Models)

Generate video chart set (files include your video name):

```powershell
python ml-model\src\generate_video_review_images.py --video-name sample_video_01
```

Key outputs in:
- `ml-model/outputs/review_images/videos/sample_video_01/`

Includes:
- `..._video_confusion_matrix.png`
- `..._video_f1_score.png`
- `..._video_accuracy_score.png`
- `..._video_accuracy_comparison_video_models.png`
- `..._video_roc_curve_resnet50.png`

## 9) Generate Figure 5.9 (Image vs Video Accuracy)

```powershell
python ml-model\src\generate_figure_5_9_accuracy_comparison.py
```

Output:
- `ml-model/outputs/review_images/figure_5_9_accuracy_comparison_image_vs_video.png`

## 10) Main API Endpoints

- `POST /signup`
- `POST /login`
- `POST /predict`
- `GET /model-metrics`
- `GET /model-comparison`
- `GET /admin/users`
- `GET /admin/pending`

## 11) Troubleshooting

- If frontend shows blank page:
  - Remove Vite cache and reinstall:
    ```powershell
    cd C:\FINAL\frontend
    if (Test-Path node_modules\.vite) { Remove-Item -Recurse -Force node_modules\.vite }
    npm install
    npm run dev
    ```
- If `/predict` returns 500:
  - Check `MODEL_PATH` exists and points to a valid `.keras` file.
  - Confirm TensorFlow version is compatible with saved model format.
- If login fails with invalid credentials:
  - Check `backend/data/users.json` and password hashing compatibility in `backend/app/core/auth.py`.

