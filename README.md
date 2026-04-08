# DeepShield AI

DeepShield AI is a full-stack deepfake detection project with:
- React frontend (`frontend/`)
- FastAPI backend (`backend/`)
- TensorFlow training/inference pipeline (`ml-model/`)

## What Works Now

- Image prediction: real TensorFlow inference in backend
- Video prediction: frame-sampling + TensorFlow inference in backend
- Automated zip-based training pipeline: drop zip files, run one command, auto-pick best model, deploy to backend

## Repo Structure

```text
c:/Users/PC-4/Desktop/deepfake_v1/
  backend/
    app/
    models/
    requirements.txt
    .env
  frontend/
  ml-model/
    src/
      train.py
      auto_train_from_zips.py
    requirements.txt
```

## 1) Backend Setup

```powershell
cd C:\Users\PC-4\Desktop\deepfake_v1\backend
py -m virtualenv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create `backend/.env`:

```env
JWT_SECRET=change-this-to-a-long-random-secret
JWT_ALGORITHM=HS256
CORS_ORIGINS=http://localhost:5173
USERS_FILE=./data/users.json
MODEL_PATH=./models/best_model.keras
DEEPFAKE_THRESHOLD=0.5
```

Run backend:

```powershell
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## 2) Frontend Setup

```powershell
cd C:\Users\PC-4\Desktop\deepfake_v1\frontend
npm install
npm run dev
```

Frontend URL: `http://localhost:5173`

## 3) ML Setup

```powershell
cd C:\Users\PC-4\Desktop\deepfake_v1
py -m virtualenv .venv-ml
.\.venv-ml\Scripts\Activate.ps1
pip install -r ml-model\requirements.txt
```

## 4) Zip-Only Auto Training Flow (Recommended)

### Step A: Put zip files here

```text
ml-model/data/_zips/
  archive (1).zip
  archive (2).zip
  ...
```

### Step B: Run one command

```powershell
cd C:\Users\PC-4\Desktop\deepfake_v1
.\.venv-ml\Scripts\Activate.ps1
python ml-model\src\auto_train_from_zips.py
```

### What this command does

1. Extracts each zip to `ml-model/data/_raw/<dataset_name>/`
2. Auto-maps labels by folder/file path keywords:
   - `authentic, real, original, genuine` -> `authentic`
   - `deepfake, fake, manipulated, forged, synthetic` -> `deepfake`
3. Creates clean splits for each zip dataset:
   - `train` (70%)
   - `val` (15%)
   - `test` (15%)
4. Trains one model per zip using `ml-model/src/train.py`
5. Compares runs by:
   - highest `auc`
   - then highest `accuracy`
   - then lowest `loss`
6. Copies winning model to:
   - `backend/models/best_model.keras`
7. Writes report:
   - `ml-model/outputs/auto/auto_train_report.json`

## 5) Optional Training Args

```powershell
python ml-model\src\auto_train_from_zips.py --epochs 12 --fine-tune-epochs 5 --batch-size 32 --img-size 224
```

Other useful args:
- `--zip-dir` (default: `ml-model/data/_zips`)
- `--outputs-root` (default: `ml-model/outputs/auto`)
- `--backend-model-path` (default: `backend/models/best_model.keras`)

## 6) After Training

Restart backend so it loads latest model:

```powershell
cd C:\Users\PC-4\Desktop\deepfake_v1\backend
.\.venv\Scripts\Activate.ps1
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## 7) If Predictions Look Wrong (All Authentic / All Deepfake)

1. Check backend is using correct model path in `.env`:
   - `MODEL_PATH=./models/best_model.keras`
2. Tune threshold:
   - `DEEPFAKE_THRESHOLD=0.45` (more sensitive)
   - `DEEPFAKE_THRESHOLD=0.55` (more strict)
3. Confirm zip label folders contain both classes.
4. Inspect `auto_train_report.json` metrics for weak runs.

## 8) Current Selection Rule

The automation currently selects best run with:
- `max(auc)` -> `max(accuracy)` -> `min(loss)`

This is stored in report JSON for transparency.

## 9) Notes

- Do not commit raw datasets or model binaries.
- Keep backend and ML virtual environments separate.
- For reproducibility, avoid changing train/val/test logic between runs.
