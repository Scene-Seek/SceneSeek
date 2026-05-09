from pydantic import BaseModel, Field

class IdentifyRequestScheme(BaseModel):
    """
    Идентификация - Запрос
    """
    nickname: str = Field(min_length=1, max_length=50)
    password: str = Field(min_length=1, max_length=128)


class IdentifyResponseScheme(BaseModel):
    """
    Идентификация - Ответ
    """
    user_id: int
    nickname: str
    token: str
