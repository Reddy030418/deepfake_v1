from fastapi import APIRouter

from app.services.user_service import get_all_users

router = APIRouter(prefix="/admin", tags=["admin"])


@router.get("/users")
def list_users():
    users = get_all_users()
    return [{"username": u["username"], "email": u["email"], "approved": u["approved"], "role": u.get("role", "user")} for u in users]


@router.get("/pending")
def list_pending_users():
    users = get_all_users()
    return [{"username": u["username"], "email": u["email"]} for u in users if not u["approved"]]
