from enum import Enum
from pydantic import BaseModel

class StatusEnum(str, Enum):
    pending = "pending"
    indexing = "indexing"
    ready = "ready"
    failed = "failed"

class UploadVideoResponseScheme(BaseModel):
    video_id: int
    status: StatusEnum

class GetVideoResponseScheme(BaseModel):
    video_id: int
    video_path: str