import json
import os
from pathlib import Path

from app.core.auth import hash_password


def _bootstrap_defaults(path: Path) -> None:
    users = [
        {
            "username": "demo_user",
            "email": "demo@deepshield.ai",
            "password_hash": hash_password("Demo@1234"),
            "approved": True,
            "role": "user"
        },
        {
            "username": "admin_user",
            "email": "admin@deepshield.ai",
            "password_hash": hash_password("Admin@1234"),
            "approved": True,
            "role": "admin"
        }
    ]
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(users, indent=2), encoding="utf-8")
        return

    try:
        current = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        current = []

    if not current:
        path.write_text(json.dumps(users, indent=2), encoding="utf-8")


def _get_users_path() -> Path:
    raw = os.getenv("USERS_FILE", "./data/users.json")
    path = Path(raw)
    if path.is_absolute():
        return path
    backend_root = Path(__file__).resolve().parents[2]
    return backend_root / path


def get_all_users() -> list[dict]:
    path = _get_users_path()
    _bootstrap_defaults(path)
    return json.loads(path.read_text(encoding="utf-8"))


def save_all_users(users: list[dict]) -> None:
    path = _get_users_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(users, indent=2), encoding="utf-8")


def create_user(username: str, email: str, password: str) -> dict:
    users = get_all_users()
    if any(u["username"] == username for u in users):
        raise ValueError("Username already exists")
    if any(u["email"] == email for u in users):
        raise ValueError("Email already exists")

    user = {
        "username": username,
        "email": email,
        "password_hash": hash_password(password),
        "approved": True,
        "role": "user"
    }
    users.append(user)
    save_all_users(users)
    return user


def find_user(userid: str) -> dict | None:
    users = get_all_users()
    return next((u for u in users if u["username"] == userid or u["email"] == userid), None)
