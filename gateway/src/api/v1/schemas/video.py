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