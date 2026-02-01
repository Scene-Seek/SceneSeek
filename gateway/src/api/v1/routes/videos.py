from fastapi import APIRouter, UploadFile, HTTPException, Form
from src.services.broker_service import broker_service
from src.services.database_service import database_service
from src.services.minio_service import minio_service
from src.api.v1.schemas.video import (
    UploadVideoResponseScheme,
    GetVideoResponseScheme
)

router = APIRouter()

@router.post("/videos", response_model=UploadVideoResponseScheme)
async def post_videos(file: UploadFile, user_id: int = Form(...)):
    """
    Создать новое видео
    """

    try:
        user = await database_service.get_user_by_id(user_id=user_id)
        if not user:
            raise HTTPException(status_code=404, detail="user not found")
        # minio
        minio_service.save_obj(
            obj=file,
            bucket=minio_service.BUCKET_VIDEOS_IN
        )
        video_path = minio_service.get_video_url(object_name=file.filename)
        # db
        video = await database_service.create_video( # TODO: configure atributes
            uploaded_by_user_id=user_id,
            title=file.filename,
            path=video_path,
            duration=None,
            fps=None,
            resolution=None,
            processing_status="pending"
        )
        # broker
        await broker_service.pub(
            message={
                "video_id": video.video_id,
                "user_id": user_id,
                "object_name": file.filename,
                "bucket": minio_service.BUCKET_VIDEOS_IN,
                "video_url": video_path
            },
            queue=broker_service.QUEUE_VIDEOS
        )
        return UploadVideoResponseScheme(video_id=video.video_id, status=video.processing_status)
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/videos/{video_id}", response_model=GetVideoResponseScheme)
async def get_videos(video_id: int):
    """
    Получить ссылку на видео по id
    """
    try:
        video = await database_service.get_video_by_id(video_id=video_id)
        return GetVideoResponseScheme(video_id=video_id, video_path=video.path)
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))