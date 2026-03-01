from typing import Optional

from pydantic import BaseModel
from src.api.v1.schemas.common import StatusEnum


class SearchResultItem(BaseModel):
    """Отдельный результат поиска — сгруппированный временной интервал"""

    start_time: float
    end_time: float
    timestamp: float  # best timestamp for seeking
    score: float
    caption: Optional[str] = None


class UploadSearchRequestScheme(BaseModel):
    """
    Загрузка промпта - Запрос
    """

    user_id: int
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
