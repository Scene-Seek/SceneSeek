from pydantic import BaseModel, Field


class AuthRequestScheme(BaseModel):
    """
    Auth - запрос
    """

    username: str = Field(min_length=1, max_length=50)
    password: str = Field(min_length=1, max_length=128)


class AuthResponseScheme(BaseModel):
    """
    Auth - ответ
    """

    user_id: int
    username: str
    token: str
    is_anonymous: bool = False


class CurrentUserResponseScheme(BaseModel):
    """
    Текущий пользователь - ответ
    """

    user_id: int
    username: str
    is_anonymous: bool = False
