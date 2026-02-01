from pydantic import BaseModel

class IdentifyRequestScheme(BaseModel):
    """
    Идентификация - Запрос
    """
    nickname: str


class IdentifyResponseScheme(BaseModel):
    """
    Идентификация - Ответ
    """
    user_id: int
    nickname: str
    role: str