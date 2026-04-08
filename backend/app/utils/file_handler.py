from pathlib import Path


def infer_media_type(filename: str) -> str:
    ext = Path(filename.lower()).suffix
    if ext in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        return "image"
    if ext in {".mp4", ".avi", ".mov", ".mkv", ".webm"}:
        return "video"
    raise ValueError("Unsupported file format")
