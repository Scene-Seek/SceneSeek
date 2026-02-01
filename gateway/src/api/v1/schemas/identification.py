from pydantic import BaseModel


class IdentifyRequestScheme(BaseModel):
    nickname: str


class IdentifyResponseScheme(BaseModel):
    user_id: int
    nickname: str
    role: str