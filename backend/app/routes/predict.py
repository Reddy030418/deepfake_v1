from time import perf_counter

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.schemas.prediction import PredictionResponse
from app.services.model_service import predict_image
from app.services.video_service import predict_video
from app.utils.file_handler import infer_media_type
from app.utils.preprocessing import preprocess_image

router = APIRouter(tags=["predict"])


@router.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    try:
        media_type = infer_media_type(file.filename or "")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    started = perf_counter()
    raw = await file.read()

    if media_type == "image":
        label, confidence = predict_image(preprocess_image(raw))
    else:
        label, confidence = predict_video(raw)

    ms = int((perf_counter() - started) * 1000)
    return PredictionResponse(
        success=True,
        media_type=media_type,
        prediction=label,
        confidence=confidence,
        processing_time_ms=ms,
    )
