from pydantic import BaseModel

class IdentifyResponseScheme(BaseModel):
    user_id: int
    nickname: str
    role: str