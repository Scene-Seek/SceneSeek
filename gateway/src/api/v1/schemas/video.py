from pydantic import BaseModel
from src.api.v1.schemas.common import StatusEnum

class UploadVideoResponseScheme(BaseModel):
    video_id: int
    status: StatusEnum

class GetVideoResponseScheme(BaseModel):
    video_id: int
    video_path: str