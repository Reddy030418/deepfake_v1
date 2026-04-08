import os
from dataclasses import dataclass, field


@dataclass
class Settings:
    jwt_secret: str = os.getenv("JWT_SECRET", "change-me")
    jwt_algorithm: str = os.getenv("JWT_ALGORITHM", "HS256")
    cors_origins: list[str] = field(
        default_factory=lambda: os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")
    )
    users_file: str = os.getenv("USERS_FILE", "./data/users.json")
    model_path: str = os.getenv("MODEL_PATH", "./models/best_model.keras")


settings = Settings()
