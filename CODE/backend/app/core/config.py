import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


# Load backend/.env so local development picks up configured values.
load_dotenv(Path(__file__).resolve().parents[2] / ".env")


def _env_bool(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    jwt_secret: str = os.getenv("JWT_SECRET", "change-me")
    jwt_algorithm: str = os.getenv("JWT_ALGORITHM", "HS256")
    cors_origins: list[str] = field(
        default_factory=lambda: os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")
    )
    users_file: str = os.getenv("USERS_FILE", "./data/users.json")
    model_path: str = os.getenv("MODEL_PATH", "./models/best_model.keras")
    model_metrics_path: str = os.getenv("MODEL_METRICS_PATH", "./models/metrics.json")
    model_comparison_path: str = os.getenv("MODEL_COMPARISON_PATH", "./models/model_comparison.json")
    deepfake_threshold: float = float(os.getenv("DEEPFAKE_THRESHOLD", "0.5"))
    demo_dashboard_mode: bool = _env_bool("DEMO_DASHBOARD_MODE", "false")


settings = Settings()
