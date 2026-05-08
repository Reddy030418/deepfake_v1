from pydantic import BaseModel


class PredictionResponse(BaseModel):
    success: bool
    media_type: str
    prediction: str
    confidence: float
    processing_time_ms: int
