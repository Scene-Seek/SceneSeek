from pydantic import BaseModel
from src.api.v1.schemas.common import StatusEnum


class SearchResultItem(BaseModel):
    """Результат поиска: временной сегмент с лучшим кадром и bbox"""

    start: float
    end: float
    best_ts: float
    score: float
    bbox: list[float]
    type: str


class UploadSearchRequestScheme(BaseModel):
    """
    Загрузка промпта - Запрос
    """

    video_id: int
    query_text: str


class UploadSearchResponseScheme(BaseModel):
    """
    Загрузка промпта - Ответ
    """

    query_id: int
    user_id: int
    video_id: int
    query_text: str
    status: StatusEnum


class GetSearchStatusResponseScheme(BaseModel):
    """
    Получение статуса промпта - Ответ
    """

    query_id: int
    user_id: int
    video_id: int
    query_text: str
    status: StatusEnum


class GetSearchResultsScheme(BaseModel):
    """
    Получение результата промпта - Ответ
    """

    query_id: int
    result: list[SearchResultItem]
