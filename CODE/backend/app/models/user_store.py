from pydantic import BaseModel


class UserRecord(BaseModel):
    username: str
    email: str
    password_hash: str
    approved: bool = False
