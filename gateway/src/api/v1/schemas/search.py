from pydantic import BaseModel
from src.api.v1.schemas.common import StatusEnum


class UploadSearchRequestScheme(BaseModel):
    user_id: int
    video_id: int
    query_text: str

class UploadSearchResponseScheme(BaseModel):
    query_id: int
    user_id: int
    video_id: int
    query_text: str
    status: StatusEnum

class GetSearchStatusResponseScheme(BaseModel):
    query_id: int
    user_id: int
    video_id: int
    query_text: str
    status: StatusEnum