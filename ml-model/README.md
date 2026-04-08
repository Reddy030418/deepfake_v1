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

## Output Artifacts

- `ml-model/outputs/best_model.keras`
- `ml-model/outputs/final_model.keras`
- `ml-model/outputs/metrics.json`
- `ml-model/outputs/history.json`

## Next Backend Integration

After training, backend can load `best_model.keras` in `backend/app/services/model_service.py` for real predictions.
