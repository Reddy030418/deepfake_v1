# ML Training Pipeline (DeepShield AI)

## Dataset Layout

Put image files in this structure:

```text
ml-model/data/
  train/
    authentic/
      img1.jpg
      ...
    deepfake/
      img2.jpg
      ...
  val/
    authentic/
      ...
    deepfake/
      ...
```

## Setup

```powershell
cd c:\FINAL
python -m venv .venv-ml
.\.venv-ml\Scripts\Activate.ps1
pip install -r ml-model\requirements.txt
```

## Train

```powershell
python ml-model\src\train.py --data-dir ml-model\data --output-dir ml-model\outputs --epochs 10 --batch-size 32 --img-size 224
```

## Laptop-Friendly Quick Start (Recommended)

If your dataset is large, first build a small balanced subset:

```powershell
python ml-model\src\create_laptop_subset.py `
  --source-dir "C:\Final Project\Datasets\frames" `
  --output-dir ml-model\data\laptop_subset `
  --max-per-class 1200 `
  --split-strategy group `
  --train-ratio 0.7 `
  --val-ratio 0.15
```

Then train quickly:

```powershell
python ml-model\src\train.py `
  --data-dir ml-model\data\laptop_subset `
  --output-dir ml-model\outputs\laptop_run `
  --epochs 6 `
  --fine-tune-epochs 2 `
  --batch-size 16 `
  --img-size 224
```

Optional evaluation on the subset test split:

```powershell
python ml-model\src\evaluate.py `
  --model-path ml-model\outputs\laptop_run\best_model.keras `
  --test-dir ml-model\data\laptop_subset\test `
  --output-metrics-path backend\models\metrics.json
```

## 3-Model Comparison From `_raw` (Dashboard Ready)

Train 3 deep learning backbones and auto-publish:
- winner model -> `backend/models/best_model.keras`
- winner metrics -> `backend/models/metrics.json`
- top-3 comparison -> `backend/models/model_comparison.json`

```powershell
python ml-model\src\train_model_suite.py `
  --source-raw-dir ml-model\data\_raw `
  --prepared-dir ml-model\data\suite_subset `
  --max-per-class 3000 `
  --split-strategy group `
  --epochs 24 `
  --fine-tune-epochs 8 `
  --batch-size 16 `
  --backbones efficientnetb0,resnet50,xception `
  --threshold-metric accuracy `
  --monitor-metric val_auc
```

For a faster laptop run:

```powershell
python ml-model\src\train_model_suite.py `
  --source-raw-dir ml-model\data\_raw `
  --prepared-dir ml-model\data\suite_subset_quick `
  --max-per-class 600 `
  --split-strategy group `
  --epochs 8 `
  --fine-tune-epochs 2 `
  --batch-size 16
```

## Output Artifacts

- `ml-model/outputs/best_model.keras`
- `ml-model/outputs/final_model.keras`
- `ml-model/outputs/metrics.json`
- `ml-model/outputs/history.json`

## Next Backend Integration

After training, backend can load `best_model.keras` in `backend/app/services/model_service.py` for real predictions.
