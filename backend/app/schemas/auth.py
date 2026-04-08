from pydantic import BaseModel


class SignupRequest(BaseModel):
    username: str
    email: str
    password: str


class LoginRequest(BaseModel):
    userid: str
    password: str


class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: dict
