import tempfile
from pathlib import Path

import cv2
import numpy as np

from app.services.model_service import predict_deepfake_probability_from_bgr, to_label_and_confidence


def _sample_indices(frame_count: int, max_frames: int) -> list[int]:
    if frame_count <= 0:
        return []
    if frame_count <= max_frames:
        return list(range(frame_count))
    # Uniformly sample frame positions across the video.
    return [int(x) for x in np.linspace(0, frame_count - 1, num=max_frames)]


def predict_video(raw: bytes, max_frames: int = 24) -> tuple[str, float]:
    if not raw:
        raise ValueError("Empty video payload")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(raw)
        tmp_path = Path(tmp.name)

    try:
        cap = cv2.VideoCapture(str(tmp_path))
        if not cap.isOpened():
            raise ValueError("Unable to decode video")

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = _sample_indices(frame_count, max_frames)
        if not indices:
            raise ValueError("No frames found in video")

        probs: list[float] = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            probs.append(predict_deepfake_probability_from_bgr(frame))

        cap.release()

        if not probs:
            raise ValueError("Could not extract valid frames from video")

        mean_prob = float(np.mean(probs))
        return to_label_and_confidence(mean_prob)
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
