from fastapi import APIRouter, HTTPException

from app.core.auth import create_access_token, verify_password
from app.schemas.auth import AuthResponse, LoginRequest, SignupRequest
from app.services.user_service import create_user, find_user

router = APIRouter(tags=["auth"])


@router.post("/signup")
def signup(payload: SignupRequest):
    try:
        user = create_user(payload.username, payload.email, payload.password)
        return {"success": True, "user": {"username": user["username"], "email": user["email"]}}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/login", response_model=AuthResponse)
def login(payload: LoginRequest):
    user = find_user(payload.userid)
    if not user or not verify_password(payload.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_access_token(user["username"])
    return AuthResponse(access_token=token, user={"userid": user["username"], "role": user.get("role", "user")})
