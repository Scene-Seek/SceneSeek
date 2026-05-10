from datetime import datetime

from pydantic import BaseModel
from src.api.v1.schemas.common import StatusEnum


class UploadVideoResponseScheme(BaseModel):
    """
    Загрузка видео - Ответ
    """

    video_id: int
    status: StatusEnum


class GetVideoResponseScheme(BaseModel):
    """
    Получение видео - Ответ
    """

    video_id: int
    video_path: str
    status: StatusEnum


class VideoHistoryItemScheme(BaseModel):
    """
    Элемент истории видео - Ответ
    """

    video_id: int
    video_title: str
    video_status: StatusEnum
    created_at: datetime
    latest_query_id: int | None = None
    latest_query_text: str | None = None
    latest_search_status: StatusEnum | None = None
    latest_search_date: datetime | None = None


class GetVideoHistoryScheme(BaseModel):
    """
    История видео - Ответ
    """

    history: list[VideoHistoryItemScheme]
